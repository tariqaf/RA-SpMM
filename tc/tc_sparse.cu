// ============================================================================
// tc_sparse.cu - Gated Tensor Core SpMM
//
// Detection phase (CPU): use TCFeatures to decide per-tile activation
// TC path (GPU): materialize 16x16 dense tile via WMMA
// Fallback path: route to csr_adaptive for residual nnz
//
// WMMA requires sm_70+. <mma.h> is included unconditionally (safe for host).
// nvcuda namespace used only inside __CUDA_ARCH__ >= 700 guard in kernel body.
//
// Redesigned kernel: one CTA per tile, multiple warps per CTA (one per N-strip).
// No atomics needed: each tile writes to exclusive row range.
//
// Plan-run split: build_tc_sparse_plan / run_tc_sparse_plan / free_tc_sparse_plan
//
// Legacy / ablation note:
// - Residual handling still falls back to CSR_ADAPTIVE and merge-by-add.
// - The path is useful as a FULL-portfolio TC ablation, not as the paper MAIN
//   Tensor Core path for this repository state.
// - Runtime timing should be measured externally; plan fields below are kept as
//   debug diagnostics rather than the primary benchmark interface.
// ============================================================================
#include "../ra_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>  // Top-level unconditional (safe for CUDA 12.x host pass)
#include <vector>
#include <unordered_map>

// Include csr_adaptive declarations
void csr_adaptive_spmm(const int*, const int*, const float*, const float*, float*, int, int, int);
CSRAdaptivePlan build_csr_adaptive_plan(const int* h_rowptr, int M, int K);
void run_csr_adaptive_plan(const CSRAdaptivePlan& plan,
    const int* d_rowptr, const int* d_colind, const float* d_vals,
    const float* d_B, float* d_C, int N);
void free_csr_adaptive_plan(CSRAdaptivePlan& plan);

// Include TC features
TCFeatures compute_tc_features(const int* rowptr, const int* colind, int M, int K);

// File-scope addition kernel (needed for TC + residual accumulation)
__global__ void element_add_kernel(float* __restrict__ a, const float* __restrict__ b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] += b[i];
}

// ---------------------------------------------------------------------------
// TC No-Atomic Correctness Invariant:
//
// 1. Each tile is unique: tile list has each (row_tile, col_tile) exactly once.
// 2. One CTA owns one row-tile: CTA t writes to rows [rt*16, (rt+1)*16).
// 3. Warps write disjoint N-strips: warp w writes to [w*16, (w+1)*16).
// 4. Residual path excludes TC nnz: partition at build time.
// 5. Merge via element_add after both phases complete.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Redesigned WMMA TC kernel: batched warps
// grid = (num_tc_tiles,)
// block = min(16, ceil(N/16)) * 32 threads (one warp per N-strip)
//
// Each CTA handles one tile. Warps share A tile via smem.
// Each warp handles its own N-strip independently.
// ---------------------------------------------------------------------------
__global__ void tc_wmma_kernel(
    const int*    __restrict__ tile_row_ids,
    const int*    __restrict__ tile_col_ids,
    const int*    __restrict__ tile_nnz_start,
    const int*    __restrict__ tile_nnz_count,
    const int*    __restrict__ nnz_local_row,
    const int*    __restrict__ nnz_local_col,
    const float*  __restrict__ nnz_vals,
    const float*  __restrict__ B,
    float*        __restrict__ C,
    int M, int K, int N, int num_tc_tiles)
{
    const int tile_id = blockIdx.x;
    if (tile_id >= num_tc_tiles) return;

    const int warp_id_in_cta = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;

    const int rt = tile_row_ids[tile_id];
    const int ct = tile_col_ids[tile_id];
    const int nnz_s = tile_nnz_start[tile_id];
    const int nnz_c = tile_nnz_count[tile_id];

    const int row_start = rt * 16;
    const int col_start = ct * 16;

    // Shared memory: A tile in half precision (16x16 = 256 half = 512 bytes)
    __shared__ half A_smem[16 * 16];
    // Shared memory: per-warp B tiles (max 16 warps × 256 half = 8KB)
    // Must be at function scope — not inside loops or arch guards
    __shared__ half B_smem_all[16 * 16 * 16];
    // Shared memory: per-warp C accumulators (max 16 warps × 256 float = 16KB)
    // wmma::store_matrix_sync requires shared or global pointer — local arrays crash
    __shared__ float C_smem_all[16 * 16 * 16];

    // Warp 0 initializes A tile
    if (warp_id_in_cta == 0) {
        for (int i = lane; i < 256; i += 32) {
            A_smem[i] = __float2half(0.f);
        }
    }
    __syncthreads();

    // Warp 0 fills A tile from sparse nnz
    if (warp_id_in_cta == 0) {
        for (int p = lane; p < nnz_c; p += 32) {
            int lr = nnz_local_row[nnz_s + p];
            int lc = nnz_local_col[nnz_s + p];
            float v = nnz_vals[nnz_s + p];
            A_smem[lr * 16 + lc] = __float2half(v);
        }
    }
    __syncthreads();

    // Each warp processes one or more N-strips
    // For N <= 16*num_warps, each warp handles exactly one strip
    // For N > 16*num_warps, warps loop over strips
    for (int strip = warp_id_in_cta; strip * 16 < N; strip += num_warps) {
        int n_start = strip * 16;
        int n_end_local = min(n_start + 16, N);
        int n_width = n_end_local - n_start;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        using namespace nvcuda;

        // Load B fragment into shared memory per-warp region would be expensive
        // Instead use register-based approach: load B into registers then do WMMA

        // Per-warp slices of the function-scope shared arrays
        half* B_smem  = B_smem_all  + warp_id_in_cta * 256;
        float* C_smem = C_smem_all  + warp_id_in_cta * 256;

        // Load B[col_start:col_start+16, n_start:n_start+16]
        for (int i = lane; i < 256; i += 32) {
            int lc = i / 16;
            int ln = i % 16;
            int gc = col_start + lc;
            int gn = n_start + ln;
            float val = 0.f;
            if (gc < K && gn < N) {
                val = B[(i64)gc * N + gn];
            }
            B_smem[i] = __float2half(val);
        }
        // No sync needed since each warp writes to its own smem region
        // But we need a warp-level sync for the B load
        __syncwarp();

        // WMMA
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

        wmma::fill_fragment(c_frag, 0.0f);
        wmma::load_matrix_sync(a_frag, A_smem, 16);
        wmma::load_matrix_sync(b_frag, B_smem, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        // Store to shared memory — wmma::store_matrix_sync requires shared/global pointer
        wmma::store_matrix_sync(C_smem, c_frag, 16, wmma::mem_row_major);
        __syncwarp();  // ensure all 32 lanes have committed before scatter

        for (int i = lane; i < 256; i += 32) {
            int lr = i / 16;
            int ln = i % 16;
            int gr = row_start + lr;
            int gn = n_start + ln;
            if (gr < M && gn < N) {
                // atomicAdd required: multiple col-tiles can contribute to same output rows
                atomicAdd(&C[(i64)gr * N + gn], C_smem[i]);
            }
        }
#endif  // __CUDA_ARCH__ >= 700
    }
}

// Stub for older architectures
#if !(defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700)
// The kernel above compiles for all architectures but the WMMA section
// is guarded. For sm < 70, it just does nothing in the loop.
#endif

// ---------------------------------------------------------------------------
// Helper: upload vector to device
// ---------------------------------------------------------------------------
template<typename T>
static T* upload_vec_tc(const std::vector<T>& v) {
    if (v.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// ---------------------------------------------------------------------------
// Build TC sparse plan
// ---------------------------------------------------------------------------
TCSparsePlan build_tc_sparse_plan(
    const int* h_rowptr, const int* h_colind,
    const float* h_vals, int M, int K,
    bool tc_eligible, bool hw_tc_supported)
{
    TCSparsePlan plan;
    plan.M = M;
    plan.K = K;
    plan.tc_eligible = tc_eligible;
    plan.hw_tc_supported = hw_tc_supported;

    if (M == 0) return plan;

    int total_nnz = h_rowptr[M];

    // If not eligible or no HW support, build full residual CSR only
    if (!tc_eligible || !hw_tc_supported) {
        // Upload full CSR as residual
        if (total_nnz > 0) {
            plan.res_nnz = total_nnz;
            plan.residual_nnz = total_nnz;
            CUDA_CHECK_NEXT(cudaMalloc(&plan.d_res_rowptr, (M + 1) * sizeof(int)));
            CUDA_CHECK_NEXT(cudaMalloc(&plan.d_res_colind, total_nnz * sizeof(int)));
            CUDA_CHECK_NEXT(cudaMalloc(&plan.d_res_vals, total_nnz * sizeof(float)));
            CUDA_CHECK_NEXT(cudaMemcpy(plan.d_res_rowptr, h_rowptr, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK_NEXT(cudaMemcpy(plan.d_res_colind, h_colind, total_nnz * sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK_NEXT(cudaMemcpy(plan.d_res_vals, h_vals, total_nnz * sizeof(float), cudaMemcpyHostToDevice));
        }
        return plan;
    }

    // Tile analysis
    const int tile_size = TC_TILE_SIZE;
    int num_col_tiles = (K + tile_size - 1) / tile_size;
    const int tile_capacity = tile_size * tile_size;

    // Build per-tile nnz lists
    std::unordered_map<int, std::vector<int>> tile_lrow_list, tile_lcol_list;
    std::unordered_map<int, std::vector<float>> tile_val_list;

    for (int r = 0; r < M; ++r) {
        int rt = r / tile_size;
        for (int p = h_rowptr[r]; p < h_rowptr[r + 1]; ++p) {
            int c = h_colind[p];
            int ct = c / tile_size;
            int key = rt * num_col_tiles + ct;
            tile_lrow_list[key].push_back(r % tile_size);
            tile_lcol_list[key].push_back(c % tile_size);
            tile_val_list[key].push_back(h_vals[p]);
        }
    }

    // Partition into TC vs residual
    std::vector<int> tc_tile_row, tc_tile_col;
    std::vector<int> tc_nnz_start, tc_nnz_count;
    std::vector<int> tc_local_row, tc_local_col;
    std::vector<float> tc_vals_sorted;
    std::unordered_map<int, bool> tile_is_tc;

    int cur_tc_nnz = 0;
    int activated = 0;
    double sum_fill = 0.0;
    std::vector<float> fills;
    int64_t candidate_nnz_sum = 0;

    for (auto& kv : tile_lrow_list) {
        int key = kv.first;
        int cnt = (int)kv.second.size();
        float fill = (float)cnt / tile_capacity;
        fills.push_back(fill);

        if (fill >= TC_FILL_THRESHOLD) {
            candidate_nnz_sum += cnt;
            int rt = key / num_col_tiles;
            int ct = key % num_col_tiles;

            tc_tile_row.push_back(rt);
            tc_tile_col.push_back(ct);
            tc_nnz_start.push_back(cur_tc_nnz);
            tc_nnz_count.push_back(cnt);

            for (int i = 0; i < cnt; ++i) {
                tc_local_row.push_back(kv.second[i]);
                tc_local_col.push_back(tile_lcol_list[key][i]);
                tc_vals_sorted.push_back(tile_val_list[key][i]);
            }
            cur_tc_nnz += cnt;

            tile_is_tc[key] = true;
            activated++;
            sum_fill += fill;
        } else {
            tile_is_tc[key] = false;
        }
    }

    plan.num_tc_tiles = activated;
    plan.candidate_tiles = activated;

    // Fill statistics
    if (!fills.empty()) {
        std::sort(fills.begin(), fills.end());
        int n = (int)fills.size();
        double s = 0.0;
        for (float f : fills) s += f;
        plan.fill_mean = (float)(s / n);
        plan.fill_median = fills[n / 2];
        plan.fill_p90 = fills[std::min((int)(0.9f * (n - 1)), n - 1)];
        plan.fill_max = fills[n - 1];
    }
    plan.candidate_nnz_coverage = (total_nnz > 0) ? (float)candidate_nnz_sum / total_nnz : 0.f;

    // Upload TC tile data
    if (activated > 0) {
        plan.d_tc_tile_row  = upload_vec_tc(tc_tile_row);
        plan.d_tc_tile_col  = upload_vec_tc(tc_tile_col);
        plan.d_tc_nnz_start = upload_vec_tc(tc_nnz_start);
        plan.d_tc_nnz_count = upload_vec_tc(tc_nnz_count);
        plan.d_tc_local_row = upload_vec_tc(tc_local_row);
        plan.d_tc_local_col = upload_vec_tc(tc_local_col);
        plan.d_tc_vals      = upload_vec_tc(tc_vals_sorted);
    }

    // Build residual CSR (nnz not in TC tiles)
    std::vector<int> res_rowptr(M + 1, 0);
    for (int r = 0; r < M; ++r) {
        int rt = r / tile_size;
        int count = 0;
        for (int p = h_rowptr[r]; p < h_rowptr[r + 1]; ++p) {
            int c = h_colind[p];
            int ct = c / tile_size;
            int key = rt * num_col_tiles + ct;
            auto it = tile_is_tc.find(key);
            if (it == tile_is_tc.end() || !it->second) {
                count++;
            }
        }
        res_rowptr[r + 1] = count;
    }
    for (int r = 0; r < M; ++r) {
        res_rowptr[r + 1] += res_rowptr[r];
    }

    int res_nnz = res_rowptr[M];
    plan.res_nnz = res_nnz;
    plan.residual_nnz = res_nnz;

    if (res_nnz > 0) {
        std::vector<int> res_colind(res_nnz);
        std::vector<float> res_vals(res_nnz);
        std::vector<int> pos(M);
        for (int r = 0; r < M; ++r) pos[r] = res_rowptr[r];

        for (int r = 0; r < M; ++r) {
            int rt = r / tile_size;
            for (int p = h_rowptr[r]; p < h_rowptr[r + 1]; ++p) {
                int c = h_colind[p];
                int ct = c / tile_size;
                int key = rt * num_col_tiles + ct;
                auto it = tile_is_tc.find(key);
                if (it == tile_is_tc.end() || !it->second) {
                    res_colind[pos[r]] = c;
                    res_vals[pos[r]] = h_vals[p];
                    pos[r]++;
                }
            }
        }

        CUDA_CHECK_NEXT(cudaMalloc(&plan.d_res_rowptr, (M + 1) * sizeof(int)));
        CUDA_CHECK_NEXT(cudaMalloc(&plan.d_res_colind, res_nnz * sizeof(int)));
        CUDA_CHECK_NEXT(cudaMalloc(&plan.d_res_vals, res_nnz * sizeof(float)));
        CUDA_CHECK_NEXT(cudaMemcpy(plan.d_res_rowptr, res_rowptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK_NEXT(cudaMemcpy(plan.d_res_colind, res_colind.data(), res_nnz * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK_NEXT(cudaMemcpy(plan.d_res_vals, res_vals.data(), res_nnz * sizeof(float), cudaMemcpyHostToDevice));
    }

    return plan;
}

// ---------------------------------------------------------------------------
// Run TC sparse plan (with timing decomposition via CUDA events)
// ---------------------------------------------------------------------------
void run_tc_sparse_plan(TCSparsePlan& plan, const float* d_B, float* d_C, int N)
{
    int M = plan.M;
    int K = plan.K;
    if (M == 0 || N == 0) return;

    CUDA_CHECK_NEXT(cudaMemset(d_C, 0, (i64)M * N * sizeof(float)));

    bool has_tc = (plan.num_tc_tiles > 0 && plan.tc_eligible && plan.hw_tc_supported);
    bool has_res = (plan.res_nnz > 0 && plan.d_res_rowptr != nullptr);

    // Create CUDA events for timing
    cudaEvent_t e_start, e_after_tc, e_after_resid, e_after_add;
    CUDA_CHECK_NEXT(cudaEventCreate(&e_start));
    CUDA_CHECK_NEXT(cudaEventCreate(&e_after_tc));
    CUDA_CHECK_NEXT(cudaEventCreate(&e_after_resid));
    CUDA_CHECK_NEXT(cudaEventCreate(&e_after_add));

    CUDA_CHECK_NEXT(cudaEventRecord(e_start));

    // Launch TC WMMA kernel
    if (has_tc) {
        int n_warps_per_cta = std::min(16, (N + 15) / 16);
        int threads_per_block = n_warps_per_cta * 32;
        dim3 grid(plan.num_tc_tiles);
        dim3 block(threads_per_block);

        tc_wmma_kernel<<<grid, block>>>(
            plan.d_tc_tile_row, plan.d_tc_tile_col,
            plan.d_tc_nnz_start, plan.d_tc_nnz_count,
            plan.d_tc_local_row, plan.d_tc_local_col, plan.d_tc_vals,
            d_B, d_C, M, K, N, plan.num_tc_tiles);
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK_NEXT(cudaEventRecord(e_after_tc));

    // Launch CSR adaptive fallback for residual
    float* d_C_temp = nullptr;
    if (has_res) {
        if (has_tc) {
            // Need temp buffer since d_C already has TC contributions
            CUDA_CHECK_NEXT(cudaMalloc(&d_C_temp, (i64)M * N * sizeof(float)));
            CUDA_CHECK_NEXT(cudaMemset(d_C_temp, 0, (i64)M * N * sizeof(float)));
            csr_adaptive_spmm(plan.d_res_rowptr, plan.d_res_colind, plan.d_res_vals,
                               d_B, d_C_temp, M, K, N);
        } else {
            // No TC, write directly
            csr_adaptive_spmm(plan.d_res_rowptr, plan.d_res_colind, plan.d_res_vals,
                               d_B, d_C, M, K, N);
        }
    }

    CUDA_CHECK_NEXT(cudaEventRecord(e_after_resid));

    // Merge TC + residual
    if (has_tc && has_res && d_C_temp) {
        int total = (int)((i64)M * N);
        int threads = 256;
        int blks = (total + threads - 1) / threads;
        element_add_kernel<<<blks, threads>>>(d_C, d_C_temp, total);
        CUDA_CHECK_KERNEL();
    }

    CUDA_CHECK_NEXT(cudaEventRecord(e_after_add));
    CUDA_CHECK_NEXT(cudaEventSynchronize(e_after_add));

    // Record timing
    cudaEventElapsedTime(&plan.t_tc_fused_ms, e_start, e_after_tc);
    cudaEventElapsedTime(&plan.t_residual_ms, e_after_tc, e_after_resid);
    cudaEventElapsedTime(&plan.t_accumulate_ms, e_after_resid, e_after_add);

    // Cleanup
    if (d_C_temp) cudaFree(d_C_temp);
    cudaEventDestroy(e_start);
    cudaEventDestroy(e_after_tc);
    cudaEventDestroy(e_after_resid);
    cudaEventDestroy(e_after_add);

    // NO additional sync needed (eventSynchronize already synced)
}

// ---------------------------------------------------------------------------
// Free TC sparse plan
// ---------------------------------------------------------------------------
void free_tc_sparse_plan(TCSparsePlan& plan) {
    if (plan.d_tc_tile_row)  { cudaFree(plan.d_tc_tile_row);  plan.d_tc_tile_row = nullptr; }
    if (plan.d_tc_tile_col)  { cudaFree(plan.d_tc_tile_col);  plan.d_tc_tile_col = nullptr; }
    if (plan.d_tc_nnz_start) { cudaFree(plan.d_tc_nnz_start); plan.d_tc_nnz_start = nullptr; }
    if (plan.d_tc_nnz_count) { cudaFree(plan.d_tc_nnz_count); plan.d_tc_nnz_count = nullptr; }
    if (plan.d_tc_local_row) { cudaFree(plan.d_tc_local_row); plan.d_tc_local_row = nullptr; }
    if (plan.d_tc_local_col) { cudaFree(plan.d_tc_local_col); plan.d_tc_local_col = nullptr; }
    if (plan.d_tc_vals)      { cudaFree(plan.d_tc_vals);      plan.d_tc_vals = nullptr; }
    if (plan.d_res_rowptr)   { cudaFree(plan.d_res_rowptr);   plan.d_res_rowptr = nullptr; }
    if (plan.d_res_colind)   { cudaFree(plan.d_res_colind);   plan.d_res_colind = nullptr; }
    if (plan.d_res_vals)     { cudaFree(plan.d_res_vals);     plan.d_res_vals = nullptr; }
}

// ---------------------------------------------------------------------------
// Backward-compatible launcher
// ---------------------------------------------------------------------------
void tc_sparse_spmm(
    const int*   rowptr,
    const int*   colind,
    const float* vals,
    const float* B,
    float*       C,
    int M, int K, int N,
    TCDiagnostics& diag)
{
    diag = TCDiagnostics{};

    // Check hardware support
    int device = 0;
    CUDA_CHECK_NEXT(cudaGetDevice(&device));
    cudaDeviceProp prop;
    CUDA_CHECK_NEXT(cudaGetDeviceProperties(&prop, device));
    diag.hw_tc_supported = (prop.major >= 7);

    // Get CPU copy of CSR
    std::vector<int> h_rowptr(M + 1);
    CUDA_CHECK_NEXT(cudaMemcpy(h_rowptr.data(), rowptr, (M + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    int nnz_total = h_rowptr[M];
    std::vector<int> h_colind(nnz_total);
    std::vector<float> h_vals(nnz_total);
    if (nnz_total > 0) {
        CUDA_CHECK_NEXT(cudaMemcpy(h_colind.data(), colind, nnz_total * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK_NEXT(cudaMemcpy(h_vals.data(), vals, nnz_total * sizeof(float), cudaMemcpyDeviceToHost));
    }

    // Build -> Run -> Free
    TCSparsePlan plan = build_tc_sparse_plan(
        h_rowptr.data(), h_colind.data(), h_vals.data(), M, K,
        true,  // tc_eligible (always try in backward-compat mode)
        diag.hw_tc_supported);

    run_tc_sparse_plan(plan, B, C, N);

    // Fill diagnostics
    diag.tc_candidate_tiles = plan.candidate_tiles;
    diag.tc_activated_tiles = plan.num_tc_tiles;
    diag.tc_rejected_tiles = plan.candidate_tiles > 0 ?
        (int)(plan.candidate_tiles * (1.0f - plan.candidate_nnz_coverage)) : 0;
    diag.tc_fill_avg = plan.fill_mean;
    diag.tc_path_taken = (plan.num_tc_tiles > 0);

    free_tc_sparse_plan(plan);
}
