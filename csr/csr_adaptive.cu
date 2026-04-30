// ============================================================================
// csr_adaptive.cu - Binned CSR SpMM with adaptive warp assignment
//
// Row bins: tiny (1-4), short (5-16), medium (17-64), long (65-256),
//           xlong (257+) with appropriate thread/warp strategies
//
// Long-row dispatch (Phase 8 fix):
//   row_len > 256:                -> csr_xlong_kernel (chunk-split + atomics)
//   row_len > 64 AND N <= 32:    -> csr_long_kernel_nnzpar (coop nnz, serial N)
//   row_len > 64:                -> csr_long_kernel_npar (N-parallel, 1 row/warp)
//   row_len > 16:                -> csr_medium_kernel
//   row_len > 4:                 -> csr_short_kernel (2 rows/warp)
//   row_len >= 1:                -> csr_tiny_kernel (8 rows/warp)
//
// Plan-run split: build_csr_adaptive_plan / run_csr_adaptive_plan / free_csr_adaptive_plan
//
// Legacy / ablation note:
// - This remains a useful FULL-portfolio skew baseline.
// - Very long rows still rely on atomic accumulation in the xlong path.
// - It is not promoted as the paper-facing MAIN path in this repository state.
// ============================================================================
#include "../ra_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Low-degree cooperative subwarp kernel: preserve the existing multi-row-per-
// warp mapping, but have each row's subgroup cooperate over nnz and reduce
// within that subgroup while accumulating contiguous float4 output tiles.
// ---------------------------------------------------------------------------
template<int ROWS_PER_WARP, int GROUP_SIZE, int MAX_ROW_NNZ, int WARPS_PER_BLOCK>
__global__ void csr_lowdeg_subwarp_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int warp_slot = threadIdx.x / 32;
    const int row_slot = lane / GROUP_SIZE;
    const int lane_in_group = lane % GROUP_SIZE;
    const int row_idx = warp_id * ROWS_PER_WARP + row_slot;

    __shared__ int   s_cols[WARPS_PER_BLOCK][ROWS_PER_WARP][MAX_ROW_NNZ];
    __shared__ float s_vals[WARPS_PER_BLOCK][ROWS_PER_WARP][MAX_ROW_NNZ];

    int row = -1;
    int row_len = 0;
    if (row_idx < num_rows) {
        row = row_ids[row_idx];
        const int row_start = rowptr[row];
        const int row_end   = rowptr[row + 1];
        row_len = row_end - row_start;
        if (lane_in_group < row_len && lane_in_group < MAX_ROW_NNZ) {
            const int p = row_start + lane_in_group;
            s_cols[warp_slot][row_slot][lane_in_group] = colind[p];
            s_vals[warp_slot][row_slot][lane_in_group] = vals[p];
        }
    }
    __syncwarp();

    const int N4 = N / 4;
    for (int n4 = 0; n4 < N4; ++n4) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        if (row_idx < num_rows && lane_in_group < row_len && lane_in_group < MAX_ROW_NNZ) {
            const int col = s_cols[warp_slot][row_slot][lane_in_group];
            const float a_val = s_vals[warp_slot][row_slot][lane_in_group];
            const float4* B_ptr = reinterpret_cast<const float4*>(B + (i64)col * N);
            const float4 b4 = __ldg(B_ptr + n4);
            acc.x = a_val * b4.x;
            acc.y = a_val * b4.y;
            acc.z = a_val * b4.z;
            acc.w = a_val * b4.w;
        }

        for (int offset = GROUP_SIZE / 2; offset > 0; offset >>= 1) {
            acc.x += __shfl_down_sync(0xffffffff, acc.x, offset, GROUP_SIZE);
            acc.y += __shfl_down_sync(0xffffffff, acc.y, offset, GROUP_SIZE);
            acc.z += __shfl_down_sync(0xffffffff, acc.z, offset, GROUP_SIZE);
            acc.w += __shfl_down_sync(0xffffffff, acc.w, offset, GROUP_SIZE);
        }

        if (row_idx < num_rows && lane_in_group == 0) {
            float4* C_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
            C_ptr[n4] = acc;
        }
    }
}

// Tiny rows kernel: 8 rows per warp (coarsened for short rows, 1-4 nnz)
// ---------------------------------------------------------------------------
__global__ void csr_tiny_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int ROWS_PER_WARP = 8;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int base_idx = warp_id * ROWS_PER_WARP;

    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int idx = base_idx + r;
        if (idx >= num_rows) return;
        int row = row_ids[idx];

        const int row_start = rowptr[row];
        const int row_end   = rowptr[row + 1];

        for (int n = lane; n < N; n += 32) {
            float acc = 0.f;
            for (int p = row_start; p < row_end; ++p) {
                acc += vals[p] * B[(i64)colind[p] * N + n];
            }
            C[(i64)row * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Short rows kernel: 2 rows per warp (5-16 nnz)
// ---------------------------------------------------------------------------
__global__ void csr_short_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int ROWS_PER_WARP = 2;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int base_idx = warp_id * ROWS_PER_WARP;

    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int idx = base_idx + r;
        if (idx >= num_rows) return;
        int row = row_ids[idx];

        const int row_start = rowptr[row];
        const int row_end   = rowptr[row + 1];

        for (int n = lane; n < N; n += 32) {
            float acc = 0.f;
            for (int p = row_start; p < row_end; ++p) {
                acc += vals[p] * B[(i64)colind[p] * N + n];
            }
            C[(i64)row * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Medium rows kernel: 1 row per warp (17-64 nnz)
// ---------------------------------------------------------------------------
__global__ void csr_medium_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;

    if (warp_id >= num_rows) return;
    int row = row_ids[warp_id];

    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    for (int n = lane; n < N; n += 32) {
        float acc = 0.f;
        for (int p = row_start; p < row_end; ++p) {
            acc += vals[p] * B[(i64)colind[p] * N + n];
        }
        C[(i64)row * N + n] = acc;
    }
}

__global__ void csr_medium_nnzpar_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;

    if (warp_id >= num_rows) return;
    const int row = row_ids[warp_id];
    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];
    const int N4 = N / 4;

    for (int n4 = 0; n4 < N4; ++n4) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = row_start + lane; p < row_end; p += 32) {
            const int col = colind[p];
            const float a_val = vals[p];
            const float4* B_ptr = reinterpret_cast<const float4*>(B + (i64)col * N);
            const float4 b4 = __ldg(B_ptr + n4);
            acc.x += a_val * b4.x;
            acc.y += a_val * b4.y;
            acc.z += a_val * b4.z;
            acc.w += a_val * b4.w;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
            acc.x += __shfl_down_sync(0xffffffff, acc.x, offset);
            acc.y += __shfl_down_sync(0xffffffff, acc.y, offset);
            acc.z += __shfl_down_sync(0xffffffff, acc.z, offset);
            acc.w += __shfl_down_sync(0xffffffff, acc.w, offset);
        }
        if (lane == 0) {
            float4* C_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
            C_ptr[n4] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Long rows kernel (N-parallel): 1 row per warp, N-parallel like medium
// For rows 65-256 nnz when N > 32. Main long kernel for GNN use.
// ---------------------------------------------------------------------------
__global__ void csr_long_kernel_npar(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;

    if (warp_id >= num_rows) return;
    int row = row_ids[warp_id];

    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    for (int n = lane; n < N; n += 32) {
        float acc = 0.f;
        for (int p = row_start; p < row_end; ++p) {
            acc += vals[p] * B[(i64)colind[p] * N + n];
        }
        C[(i64)row * N + n] = acc;
    }
}

// ---------------------------------------------------------------------------
// Long rows kernel (nnz-parallel): cooperative nnz reduction, serial N
// For rows 65-256 nnz when N <= 32. Original long kernel design.
// ---------------------------------------------------------------------------
__global__ void csr_long_kernel_nnzpar(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;

    if (warp_id >= num_rows) return;
    int row = row_ids[warp_id];

    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    for (int n = 0; n < N; n++) {
        float acc = 0.f;
        for (int p = row_start + lane; p < row_end; p += 32) {
            acc += vals[p] * B[(i64)colind[p] * N + n];
        }
        // Warp reduction
        for (int mask = 16; mask > 0; mask >>= 1) {
            acc += __shfl_xor_sync(0xffffffff, acc, mask);
        }
        if (lane == 0) {
            C[(i64)row * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// XLong rows kernel: row splitting into 256-nnz chunks with atomic accumulation
// For rows with 257+ nnz
// ---------------------------------------------------------------------------
__global__ void csr_xlong_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ split_info,  // [num_chunks * 3]
    int num_chunks, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;

    if (warp_id >= num_chunks) return;

    int row        = split_info[warp_id * 3 + 0];
    int chunk_s    = split_info[warp_id * 3 + 1];
    int chunk_e    = split_info[warp_id * 3 + 2];

    const int row_start = rowptr[row];
    const int abs_s = row_start + chunk_s;
    const int abs_e = row_start + chunk_e;

    for (int n = lane; n < N; n += 32) {
        float acc = 0.f;
        for (int p = abs_s; p < abs_e; ++p) {
            acc += vals[p] * B[(i64)colind[p] * N + n];
        }
        atomicAdd(&C[(i64)row * N + n], acc);
    }
}

// ---------------------------------------------------------------------------
// Build row bins (CPU)
// ---------------------------------------------------------------------------
void build_row_bins(
    const int* rowptr, int M,
    std::vector<int>& tiny_rows,
    std::vector<int>& short_rows,
    std::vector<int>& medium_rows,
    std::vector<int>& long_rows,
    std::vector<int>& xlong_rows)
{
    tiny_rows.clear();
    short_rows.clear();
    medium_rows.clear();
    long_rows.clear();
    xlong_rows.clear();

    for (int r = 0; r < M; ++r) {
        int len = rowptr[r + 1] - rowptr[r];
        if      (len == 0)    continue;
        else if (len <= 4)    tiny_rows.push_back(r);
        else if (len <= 16)   short_rows.push_back(r);
        else if (len <= 64)   medium_rows.push_back(r);
        else if (len <= 256)  long_rows.push_back(r);
        else                  xlong_rows.push_back(r);
    }
}

// Build split info for xlong rows
static std::vector<int> build_split_info(
    const int* rowptr, const std::vector<int>& xlong_rows,
    int chunk_size = 256)
{
    std::vector<int> split_info;
    for (int row : xlong_rows) {
        int len = rowptr[row + 1] - rowptr[row];
        int n_chunks = (len + chunk_size - 1) / chunk_size;
        for (int c = 0; c < n_chunks; ++c) {
            int s = c * chunk_size;
            int e = std::min(s + chunk_size, len);
            split_info.push_back(row);
            split_info.push_back(s);
            split_info.push_back(e);
        }
    }
    return split_info;
}

// Helper: upload vector to device
static int* upload_int_vec(const std::vector<int>& v) {
    if (v.empty()) return nullptr;
    int* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, v.size() * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, v.data(), v.size() * sizeof(int), cudaMemcpyHostToDevice));
    return d_ptr;
}

// ---------------------------------------------------------------------------
// Build CSR adaptive plan (structural-only, no vals needed)
// ---------------------------------------------------------------------------
CSRAdaptivePlan build_csr_adaptive_plan(const int* h_rowptr, int M, int K) {
    CSRAdaptivePlan plan;
    plan.M = M;
    plan.K = K;

    if (M == 0) return plan;

    std::vector<int> tiny_rows, short_rows, medium_rows, long_rows, xlong_rows;
    build_row_bins(h_rowptr, M, tiny_rows, short_rows, medium_rows, long_rows, xlong_rows);
    plan.n_tiny   = (int)tiny_rows.size();
    plan.n_short  = (int)short_rows.size();
    plan.n_medium = (int)medium_rows.size();
    plan.n_long   = (int)long_rows.size();

    plan.bin_histogram[0] = plan.n_tiny;
    plan.bin_histogram[1] = plan.n_short;
    plan.bin_histogram[2] = plan.n_medium;
    plan.bin_histogram[3] = plan.n_long;
    plan.bin_histogram[4] = (int)xlong_rows.size();

    // Find dominant bin
    int max_bin = 0;
    for (int i = 1; i < 5; ++i) {
        if (plan.bin_histogram[i] > plan.bin_histogram[max_bin]) max_bin = i;
    }
    plan.dominant_bin = max_bin;

    // Build split info for xlong
    std::vector<int> split_info;
    if (!xlong_rows.empty()) {
        split_info = build_split_info(h_rowptr, xlong_rows, 256);
    }
    plan.n_xlong_chunks = (int)split_info.size() / 3;
    plan.n_split_rows   = (int)xlong_rows.size();

    // Upload bin row IDs to device
    if (!tiny_rows.empty())   plan.d_tiny   = upload_int_vec(tiny_rows);
    if (!short_rows.empty())  plan.d_short  = upload_int_vec(short_rows);
    if (!medium_rows.empty()) plan.d_medium = upload_int_vec(medium_rows);
    if (!long_rows.empty())   plan.d_long   = upload_int_vec(long_rows);
    if (!split_info.empty())  plan.d_xlong  = upload_int_vec(split_info);

    return plan;
}

// ---------------------------------------------------------------------------
// Run CSR adaptive plan (launches kernels, NO sync)
// ---------------------------------------------------------------------------
void run_csr_adaptive_plan(
    const CSRAdaptivePlan& plan,
    const int* d_rowptr, const int* d_colind, const float* d_vals,
    const float* d_B, float* d_C, int N)
{
    int M = plan.M;
    if (M == 0 || N == 0) return;

    CUDA_CHECK_NEXT(cudaMemset(d_C, 0, (i64)M * N * sizeof(float)));

    const int WARPS_PER_BLOCK = 4;
    const int THREADS = WARPS_PER_BLOCK * 32;
    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
    const bool use_lowdeg_vec4 =
        (N % 4 == 0) && b_aligned && c_aligned && (N >= 64) && (M <= 65536);

    // Tiny rows (8 rows/warp)
    if (plan.n_tiny > 0 && plan.d_tiny) {
        int n_warps = (plan.n_tiny + 8 - 1) / 8;
        int n_blocks = (n_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (use_lowdeg_vec4) {
            csr_lowdeg_subwarp_vec4_kernel<8, 4, 4, WARPS_PER_BLOCK><<<n_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_tiny, plan.n_tiny, N);
        } else {
            csr_tiny_kernel<<<n_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_tiny, plan.n_tiny, N);
        }
        CUDA_CHECK_KERNEL();
    }

    // Short rows (2 rows/warp)
    if (plan.n_short > 0 && plan.d_short) {
        int n_warps = (plan.n_short + 2 - 1) / 2;
        int n_blocks = (n_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (use_lowdeg_vec4) {
            csr_lowdeg_subwarp_vec4_kernel<2, 16, 16, WARPS_PER_BLOCK><<<n_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_short, plan.n_short, N);
        } else {
            csr_short_kernel<<<n_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_short, plan.n_short, N);
        }
        CUDA_CHECK_KERNEL();
    }

    // Medium rows (1 row/warp)
    if (plan.n_medium > 0 && plan.d_medium) {
        int n_warps = plan.n_medium;
        int n_blocks = (n_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (use_lowdeg_vec4) {
            csr_medium_nnzpar_vec4_kernel<<<n_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_medium, plan.n_medium, N);
        } else {
            csr_medium_kernel<<<n_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_medium, plan.n_medium, N);
        }
        CUDA_CHECK_KERNEL();
    }

    // Long rows: N-parallel for N>32, nnz-parallel for N<=32
    if (plan.n_long > 0 && plan.d_long) {
        int n_warps = plan.n_long;
        int n_blocks = (n_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (N <= 32) {
            csr_long_kernel_nnzpar<<<n_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_long, plan.n_long, N);
        } else {
            csr_long_kernel_npar<<<n_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_long, plan.n_long, N);
        }
        CUDA_CHECK_KERNEL();
    }

    // XLong rows (split chunks)
    if (plan.n_xlong_chunks > 0 && plan.d_xlong) {
        int n_warps = plan.n_xlong_chunks;
        int n_blocks = (n_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        csr_xlong_kernel<<<n_blocks, THREADS>>>(
            d_rowptr, d_colind, d_vals, d_B, d_C, plan.d_xlong, plan.n_xlong_chunks, N);
        CUDA_CHECK_KERNEL();
    }

    // NO cudaDeviceSynchronize -- sync at Python boundary
}

// ---------------------------------------------------------------------------
// Free CSR adaptive plan
// ---------------------------------------------------------------------------
void free_csr_adaptive_plan(CSRAdaptivePlan& plan) {
    if (plan.d_tiny)   { cudaFree(plan.d_tiny);   plan.d_tiny   = nullptr; }
    if (plan.d_short)  { cudaFree(plan.d_short);  plan.d_short  = nullptr; }
    if (plan.d_medium) { cudaFree(plan.d_medium); plan.d_medium = nullptr; }
    if (plan.d_long)   { cudaFree(plan.d_long);   plan.d_long   = nullptr; }
    if (plan.d_xlong)  { cudaFree(plan.d_xlong);  plan.d_xlong  = nullptr; }
}

// ---------------------------------------------------------------------------
// Backward-compatible launcher (D2H rowptr -> build -> run -> free, NO sync)
// ---------------------------------------------------------------------------
void csr_adaptive_spmm(
    const int*   rowptr,
    const int*   colind,
    const float* vals,
    const float* B,
    float*       C,
    int M, int K, int N)
{
    if (M == 0 || N == 0) return;

    // Get CPU copy of rowptr for bin building
    std::vector<int> h_rowptr(M + 1);
    CUDA_CHECK_NEXT(cudaMemcpy(h_rowptr.data(), rowptr, (M + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    // Build -> Run -> Free
    CSRAdaptivePlan plan = build_csr_adaptive_plan(h_rowptr.data(), M, K);
    run_csr_adaptive_plan(plan, rowptr, colind, vals, B, C, N);
    free_csr_adaptive_plan(plan);

    // NO cudaDeviceSynchronize -- sync at Python boundary
}
