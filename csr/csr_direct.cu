// ============================================================================
// csr_direct.cu - Clean CSR SpMM with warp-per-row strategy
//
// CSR_DIRECT audit summary:
// - Warp-per-row mapping: one warp (32 threads) handles all nnz in one row.
// - Lanes iterate n=lane, lane+32, ... (N-parallel, coalesced when N is a power of 2).
// - float4 vectorization: when N%4==0, lanes load B in float4 chunks (4x bandwidth).
// - Coarsened kernel: when avg_nnz_per_row < 4, multiple rows share one warp.
// - This path is bandwidth-bound for GNN-typical matrices (large N, moderate nnz/row).
// - Expected behavior: best or near-best on all uniform/low-skew graphs.
// - Weakness: no adaptation for degree-skewed graphs (wastes warp time on empty slots).
// ============================================================================
#include "../ra_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <mutex>
#include <unordered_map>

// ---------------------------------------------------------------------------
// Warp-per-row CSR kernel (standard case)
// ---------------------------------------------------------------------------
__global__ void csr_direct_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;

    if (warp_id >= M) return;

    const int row_start = rowptr[warp_id];
    const int row_end   = rowptr[warp_id + 1];

    for (int n = lane; n < N; n += 32) {
        float acc = 0.0f;
        for (int p = row_start; p < row_end; ++p) {
            int col = colind[p];
            float a_val = vals[p];
            acc += a_val * __ldg(&B[(i64)col * N + n]);
        }
        C[(i64)warp_id * N + n] = acc;
    }
}

// ---------------------------------------------------------------------------
// Warp-per-row with float4 vectorized B loads (N must be multiple of 4)
// ---------------------------------------------------------------------------
__global__ void csr_direct_kernel_vec4(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;

    if (warp_id >= M) return;

    const int row_start = rowptr[warp_id];
    const int row_end   = rowptr[warp_id + 1];
    const int N4 = N / 4;

    for (int n4 = lane; n4 < N4; n4 += 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = row_start; p < row_end; ++p) {
            int col = colind[p];
            float a_val = vals[p];
            const float4* B_ptr = reinterpret_cast<const float4*>(B + (i64)col * N);
            float4 b4 = __ldg(B_ptr + n4);
            acc.x += a_val * b4.x;
            acc.y += a_val * b4.y;
            acc.z += a_val * b4.z;
            acc.w += a_val * b4.w;
        }
        float4* C_ptr = reinterpret_cast<float4*>(C + (i64)warp_id * N);
        C_ptr[n4] = acc;
    }
}

// ---------------------------------------------------------------------------
// Coarsened kernel: multiple rows per warp for very short rows
// ---------------------------------------------------------------------------
template<int ROWS_PER_WARP>
__global__ void csr_direct_kernel_coarsened(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int base_row = warp_id * ROWS_PER_WARP;
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = base_row + r;
        if (row >= M) return;

        const int row_start = rowptr[row];
        const int row_end   = rowptr[row + 1];

        for (int n = lane; n < N; n += 32) {
            float acc = 0.0f;
            for (int p = row_start; p < row_end; ++p) {
                int col = colind[p];
                float a_val = vals[p];
                acc += a_val * __ldg(&B[(i64)col * N + n]);
            }
            C[(i64)row * N + n] = acc;
        }
    }
}

template<int ROWS_PER_WARP>
__global__ void csr_direct_kernel_coarsened_vec4(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int base_row = warp_id * ROWS_PER_WARP;
    const int N4 = N / 4;
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int row = base_row + r;
        if (row >= M) return;

        const int row_start = rowptr[row];
        const int row_end   = rowptr[row + 1];

        for (int n4 = lane; n4 < N4; n4 += 32) {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int p = row_start; p < row_end; ++p) {
                const int col = colind[p];
                const float a_val = vals[p];
                const float4* B_ptr = reinterpret_cast<const float4*>(B + (i64)col * N);
                const float4 b4 = __ldg(B_ptr + n4);
                acc.x += a_val * b4.x;
                acc.y += a_val * b4.y;
                acc.z += a_val * b4.z;
                acc.w += a_val * b4.w;
            }
            float4* C_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
            C_ptr[n4] = acc;
        }
    }
}

int cached_nnz_for_rowptr(const int* rowptr, int M) {
    static std::mutex cache_mu;
    static std::unordered_map<const int*, int> nnz_cache;
    {
        std::lock_guard<std::mutex> lock(cache_mu);
        auto it = nnz_cache.find(rowptr);
        if (it != nnz_cache.end()) {
            return it->second;
        }
    }

    int nnz = 0;
    CUDA_CHECK_NEXT(cudaMemcpy(&nnz, rowptr + M, sizeof(int), cudaMemcpyDeviceToHost));

    {
        std::lock_guard<std::mutex> lock(cache_mu);
        nnz_cache[rowptr] = nnz;
    }
    return nnz;
}

// ---------------------------------------------------------------------------
// Launcher (NO cudaDeviceSynchronize -- sync at Python boundary only)
// ---------------------------------------------------------------------------
void csr_direct_spmm(
    const int*   rowptr,
    const int*   colind,
    const float* vals,
    const float* B,
    float*       C,
    int M, int K, int N)
{
    if (M == 0 || N == 0) return;

    CUDA_CHECK_NEXT(cudaMemset(C, 0, (i64)M * N * sizeof(float)));

    const int WARPS_PER_BLOCK = 4;
    const int THREADS = WARPS_PER_BLOCK * 32;

    int num_warps = M;
    int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (num_blocks == 0) return;

    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(C) % 16u) == 0u;
    const bool use_vec4 = (N % 4 == 0) && b_aligned && c_aligned;
    const int nnz = cached_nnz_for_rowptr(rowptr, M);
    const float avg_nnz_per_row =
        (M > 0) ? static_cast<float>(nnz) / static_cast<float>(M) : 0.0f;

    if (avg_nnz_per_row <= 2.5f) {
        num_warps = (M + 8 - 1) / 8;
        num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (use_vec4) {
            csr_direct_kernel_coarsened_vec4<8><<<num_blocks, THREADS>>>(
                rowptr, colind, vals, B, C, M, K, N);
        } else {
            csr_direct_kernel_coarsened<8><<<num_blocks, THREADS>>>(
                rowptr, colind, vals, B, C, M, K, N);
        }
    } else if (avg_nnz_per_row <= 6.0f) {
        num_warps = (M + 4 - 1) / 4;
        num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (use_vec4) {
            csr_direct_kernel_coarsened_vec4<4><<<num_blocks, THREADS>>>(
                rowptr, colind, vals, B, C, M, K, N);
        } else {
            csr_direct_kernel_coarsened<4><<<num_blocks, THREADS>>>(
                rowptr, colind, vals, B, C, M, K, N);
        }
    } else if (avg_nnz_per_row <= 12.0f) {
        num_warps = (M + 2 - 1) / 2;
        num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        if (use_vec4) {
            csr_direct_kernel_coarsened_vec4<2><<<num_blocks, THREADS>>>(
                rowptr, colind, vals, B, C, M, K, N);
        } else {
            csr_direct_kernel_coarsened<2><<<num_blocks, THREADS>>>(
                rowptr, colind, vals, B, C, M, K, N);
        }
    } else if (use_vec4) {
        csr_direct_kernel_vec4<<<num_blocks, THREADS>>>(
            rowptr, colind, vals, B, C, M, K, N);
    } else {
        csr_direct_kernel<<<num_blocks, THREADS>>>(
            rowptr, colind, vals, B, C, M, K, N);
    }

    CUDA_CHECK_KERNEL();
    // NO cudaDeviceSynchronize -- sync happens at Python boundary
}
