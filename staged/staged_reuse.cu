// ============================================================================
// staged_reuse.cu - Tile-based staged SpMM with B-matrix reuse
//
// Tile dimensions BM=64, BK=64
// Each CUDA block processes one (row-tile, col-tile) pair
// B[col_tile_start:col_tile_end, :] loaded into shared memory
//
// Reuse analysis:
// - Shared B tile: BK x 32 x 4 = 8 KB loaded once per tile per 32-wide N-strip
// - Reuse factor per N-strip = nnz_in_tile / BK_actual
// - Break-even with global loads: only when reuse_factor > ~2
// - For GNN typical fill 0.01-0.05: reuse ~ 0.6-3.2 (marginal for single-call)
// - For warm plan reuse (plan built once, executed many times): full B-tile
//   benefit with zero plan overhead. STAGED_REUSE benefits are real primarily
//   for warm runs.
//
// Plan-run split: build_staged_reuse_plan / run_staged_reuse_plan / free_staged_reuse_plan
//
// Legacy / ablation note:
// - This path still uses atomic accumulation at tile granularity.
// - It is mainly informative for FULL-portfolio reuse ablations.
// - The benchmark stack now measures plan/run costs externally instead of
//   treating this kernel as a paper-facing MAIN candidate.
// ============================================================================
#include "../ra_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>

// ---------------------------------------------------------------------------
// Staged reuse kernel
// Grid: (num_tiles, ceil(N/32))
// Block: (128, 1)
// ---------------------------------------------------------------------------
template<int BM, int BK>
__global__ void staged_reuse_kernel(
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ tile_row,
    const int*   __restrict__ tile_col,
    const int*   __restrict__ tile_nnz_start,
    const int*   __restrict__ tile_nnz_count,
    const int*   __restrict__ tile_row_ids,
    const int*   __restrict__ tile_col_ids,
    const float* __restrict__ tile_vals,
    int M, int K, int N, int num_tiles)
{
    extern __shared__ float smem[];

    const int tile_id  = blockIdx.x;
    const int n_block  = blockIdx.y;
    const int n_start  = n_block * 32;
    const int n_end    = min(n_start + 32, N);
    const int n_width  = n_end - n_start;

    if (tile_id >= num_tiles) return;

    const int rt = tile_row[tile_id];
    const int ct = tile_col[tile_id];
    const int nnz_s = tile_nnz_start[tile_id];
    const int nnz_c = tile_nnz_count[tile_id];

    if (nnz_c == 0) return;

    const int col_start = ct * BK;
    const int col_end_  = min(col_start + BK, K);
    const int col_width = col_end_ - col_start;

    float* smem_B = smem;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Load B tile into shared memory
    for (int idx = tid; idx < col_width * n_width; idx += num_threads) {
        int lc = idx / n_width;
        int ln = idx % n_width;
        int gc = col_start + lc;
        int gn = n_start + ln;
        smem_B[lc * 32 + ln] = B[(i64)gc * N + gn];
    }
    __syncthreads();

    const int warp_id = tid / 32;
    const int lane    = tid % 32;
    const int warps   = blockDim.x / 32;

    for (int p = warp_id; p < nnz_c; p += warps) {
        int ridx = tile_row_ids[nnz_s + p];
        int cidx = tile_col_ids[nnz_s + p];
        float aval = tile_vals[nnz_s + p];

        int lc = cidx - col_start;
        if (lc < 0 || lc >= col_width) continue;

        if (lane < n_width) {
            float b_val = smem_B[lc * 32 + lane];
            int gn = n_start + lane;
            if (gn < N) {
                atomicAdd(&C[(i64)ridx * N + gn], aval * b_val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Build tile list on CPU
// ---------------------------------------------------------------------------
struct TileList {
    std::vector<int> tile_row;
    std::vector<int> tile_col;
    std::vector<int> tile_nnz_start;
    std::vector<int> tile_nnz_count;
    std::vector<int> row_ids;
    std::vector<int> col_ids;
    std::vector<float> tile_vals_sorted;
    int num_tiles;
};

static TileList build_tile_list(
    const int* rowptr, const int* colind, const float* vals,
    int M, int K, int BM, int BK)
{
    TileList tl;

    int num_row_tiles = (M + BM - 1) / BM;
    int num_col_tiles = (K + BK - 1) / BK;

    std::vector<std::vector<int>> tile_row_ids_map(num_row_tiles * num_col_tiles);
    std::vector<std::vector<int>> tile_col_ids_map(num_row_tiles * num_col_tiles);
    std::vector<std::vector<float>> tile_vals_map(num_row_tiles * num_col_tiles);

    for (int r = 0; r < M; ++r) {
        int rt = r / BM;
        for (int p = rowptr[r]; p < rowptr[r + 1]; ++p) {
            int c = colind[p];
            int ct = c / BK;
            int tile_idx = rt * num_col_tiles + ct;
            tile_row_ids_map[tile_idx].push_back(r);
            tile_col_ids_map[tile_idx].push_back(c);
            tile_vals_map[tile_idx].push_back(vals[p]);
        }
    }

    int cur_nnz_start = 0;
    for (int rt = 0; rt < num_row_tiles; ++rt) {
        for (int ct = 0; ct < num_col_tiles; ++ct) {
            int tile_idx = rt * num_col_tiles + ct;
            int cnt = (int)tile_row_ids_map[tile_idx].size();
            if (cnt == 0) continue;

            tl.tile_row.push_back(rt);
            tl.tile_col.push_back(ct);
            tl.tile_nnz_start.push_back(cur_nnz_start);
            tl.tile_nnz_count.push_back(cnt);

            for (int i = 0; i < cnt; ++i) {
                tl.row_ids.push_back(tile_row_ids_map[tile_idx][i]);
                tl.col_ids.push_back(tile_col_ids_map[tile_idx][i]);
                tl.tile_vals_sorted.push_back(tile_vals_map[tile_idx][i]);
            }

            cur_nnz_start += cnt;
        }
    }

    tl.num_tiles = (int)tl.tile_row.size();
    return tl;
}

// Helper: upload vector to device
template<typename T>
static T* upload_vec(const std::vector<T>& v) {
    if (v.empty()) return nullptr;
    T* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, v.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d_ptr;
}

// ---------------------------------------------------------------------------
// Build staged reuse plan (val-aware -- needs vals for sorted tile data)
// ---------------------------------------------------------------------------
StagedReusePlan build_staged_reuse_plan(
    const int* h_rowptr, const int* h_colind,
    const float* h_vals, int M, int K, int BM, int BK)
{
    StagedReusePlan plan;
    plan.M = M;
    plan.K = K;
    plan.BM = BM;
    plan.BK = BK;

    if (M == 0) return plan;

    TileList tl = build_tile_list(h_rowptr, h_colind, h_vals, M, K, BM, BK);
    plan.num_tiles = tl.num_tiles;

    if (tl.num_tiles == 0) return plan;

    // Compute average tile fill
    if (!tl.tile_nnz_count.empty()) {
        double sum = 0.0;
        for (int cnt : tl.tile_nnz_count) sum += cnt;
        plan.avg_tile_fill = (float)(sum / tl.num_tiles / (BM * BK));
    }

    // Upload all 7 arrays
    plan.d_tile_row       = upload_vec(tl.tile_row);
    plan.d_tile_col       = upload_vec(tl.tile_col);
    plan.d_tile_nnz_start = upload_vec(tl.tile_nnz_start);
    plan.d_tile_nnz_count = upload_vec(tl.tile_nnz_count);
    plan.d_row_ids        = upload_vec(tl.row_ids);
    plan.d_col_ids        = upload_vec(tl.col_ids);
    plan.d_tile_vals      = upload_vec(tl.tile_vals_sorted);

    return plan;
}

// ---------------------------------------------------------------------------
// Run staged reuse plan (plan contains full CSR data, NO sync)
// ---------------------------------------------------------------------------
void run_staged_reuse_plan(
    const StagedReusePlan& plan,
    const float* d_B, float* d_C, int N)
{
    int M = plan.M;
    int K = plan.K;
    if (M == 0 || N == 0 || plan.num_tiles == 0) return;

    CUDA_CHECK_NEXT(cudaMemset(d_C, 0, (i64)M * N * sizeof(float)));

    const int WARPS = 4;
    const int THREADS = WARPS * 32;
    dim3 grid(plan.num_tiles, (N + 31) / 32);
    dim3 block(THREADS);

    size_t smem_size = (size_t)plan.BK * 32 * sizeof(float);

    staged_reuse_kernel<64, 64><<<grid, block, smem_size>>>(
        d_B, d_C,
        plan.d_tile_row, plan.d_tile_col, plan.d_tile_nnz_start, plan.d_tile_nnz_count,
        plan.d_row_ids, plan.d_col_ids, plan.d_tile_vals,
        M, K, N, plan.num_tiles);

    CUDA_CHECK_KERNEL();
    // NO cudaDeviceSynchronize -- sync at Python boundary
}

// ---------------------------------------------------------------------------
// Free staged reuse plan
// ---------------------------------------------------------------------------
void free_staged_reuse_plan(StagedReusePlan& plan) {
    if (plan.d_tile_row)       { cudaFree(plan.d_tile_row);       plan.d_tile_row = nullptr; }
    if (plan.d_tile_col)       { cudaFree(plan.d_tile_col);       plan.d_tile_col = nullptr; }
    if (plan.d_tile_nnz_start) { cudaFree(plan.d_tile_nnz_start); plan.d_tile_nnz_start = nullptr; }
    if (plan.d_tile_nnz_count) { cudaFree(plan.d_tile_nnz_count); plan.d_tile_nnz_count = nullptr; }
    if (plan.d_row_ids)        { cudaFree(plan.d_row_ids);        plan.d_row_ids = nullptr; }
    if (plan.d_col_ids)        { cudaFree(plan.d_col_ids);        plan.d_col_ids = nullptr; }
    if (plan.d_tile_vals)      { cudaFree(plan.d_tile_vals);      plan.d_tile_vals = nullptr; }
}

// ---------------------------------------------------------------------------
// Backward-compatible launcher (copies CSR to CPU -> build -> run -> free, NO sync)
// ---------------------------------------------------------------------------
void staged_reuse_spmm(
    const int*   rowptr,
    const int*   colind,
    const float* vals,
    const float* B,
    float*       C,
    int M, int K, int N,
    int BM, int BK)
{
    if (M == 0 || N == 0) return;

    // Get CPU copy of CSR
    std::vector<int>   h_rowptr(M + 1);
    CUDA_CHECK_NEXT(cudaMemcpy(h_rowptr.data(), rowptr, (M + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    int nnz = h_rowptr[M];
    std::vector<int>   h_colind(nnz);
    std::vector<float> h_vals(nnz);
    if (nnz > 0) {
        CUDA_CHECK_NEXT(cudaMemcpy(h_colind.data(), colind, nnz * sizeof(int),   cudaMemcpyDeviceToHost));
        CUDA_CHECK_NEXT(cudaMemcpy(h_vals.data(),   vals,   nnz * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Build -> Run -> Free
    StagedReusePlan plan = build_staged_reuse_plan(h_rowptr.data(), h_colind.data(), h_vals.data(), M, K, BM, BK);
    run_staged_reuse_plan(plan, B, C, N);
    free_staged_reuse_plan(plan);

    // NO cudaDeviceSynchronize -- sync at Python boundary
}
