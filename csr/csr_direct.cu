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

// ---------------------------------------------------------------------------
// Subwarp + register-tile vec4 kernel (Sputnik-style A distribution).
//
// One row is owned by a W-lane subwarp (W in {8,16,32}); a warp packs 32/W
// rows in parallel, so N=64 (N4=16) runs with zero idle lanes instead of 16.
// Each lane keeps S float4 accumulators covering n4 = sl + s*W, so the A row
// is traversed once for all of N (the old kernel re-read A per 32-wide N
// stride). A colind/vals chunk is loaded coalesced (one element per lane)
// and broadcast with __shfl_sync, replacing per-lane redundant scalar loads.
// Launcher guarantees N4 == W*S exactly.
// ---------------------------------------------------------------------------
template<int W, int S>
__global__ void csr_direct_subwarp_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int K, int N)
{
    constexpr int SUBWARPS = 32 / W;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane    = threadIdx.x & 31;
    const int sub     = lane / W;
    const int sl      = lane % W;
    const unsigned mask =
        (W == 32) ? 0xffffffffu : (((1u << W) - 1u) << (sub * W));

    const int row = warp_id * SUBWARPS + sub;
    if (row >= M) return;

    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    float4 acc[S];
#pragma unroll
    for (int s = 0; s < S; ++s) acc[s] = make_float4(0.f, 0.f, 0.f, 0.f);

    for (int chunk = row_start; chunk < row_end; chunk += W) {
        const int my_p = chunk + sl;
        int   my_col = 0;
        float my_val = 0.f;
        if (my_p < row_end) {
            my_col = __ldg(colind + my_p);
            my_val = __ldg(vals + my_p);
        }
        const int limit = min(W, row_end - chunk);
        for (int j = 0; j < limit; ++j) {
            const int   col = __shfl_sync(mask, my_col, j, W);
            const float av  = __shfl_sync(mask, my_val, j, W);
            const float4* B_ptr =
                reinterpret_cast<const float4*>(B + (i64)col * N);
#pragma unroll
            for (int s = 0; s < S; ++s) {
                const float4 b4 = __ldg(B_ptr + sl + s * W);
                acc[s].x += av * b4.x;
                acc[s].y += av * b4.y;
                acc[s].z += av * b4.z;
                acc[s].w += av * b4.w;
            }
        }
    }

    float4* C_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
#pragma unroll
    for (int s = 0; s < S; ++s) C_ptr[sl + s * W] = acc[s];
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
    int M, int K, int N, int nnz)
{
    if (M == 0 || N == 0) return;

    const int WARPS_PER_BLOCK = 4;
    const int THREADS = WARPS_PER_BLOCK * 32;

    int num_warps = M;
    int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (num_blocks == 0) return;

    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(C) % 16u) == 0u;
    const bool use_vec4 = (N % 4 == 0) && b_aligned && c_aligned;
    const float avg_nnz_per_row =
        (M > 0) ? static_cast<float>(nnz) / static_cast<float>(M) : 0.0f;

    // Subwarp engine for exact-fit N4 = W*S (covers N = 32/64/128/256/512).
    if (use_vec4) {
        const int N4 = N / 4;
        int W = 0, S = 0;
        if      (N4 == 8)   { W = 8;  S = 1; }
        else if (N4 == 16)  { W = 16; S = 1; }
        else if (N4 == 32)  { W = 32; S = 1; }
        else if (N4 == 64)  { W = 32; S = 2; }
        else if (N4 == 128) { W = 32; S = 4; }
        if (W != 0) {
            const int subwarps = 32 / W;
            num_warps  = (M + subwarps - 1) / subwarps;
            num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
            switch (N4) {
                case 8:
                    csr_direct_subwarp_vec4_kernel<8, 1><<<num_blocks, THREADS>>>(
                        rowptr, colind, vals, B, C, M, K, N);
                    break;
                case 16:
                    csr_direct_subwarp_vec4_kernel<16, 1><<<num_blocks, THREADS>>>(
                        rowptr, colind, vals, B, C, M, K, N);
                    break;
                case 32:
                    csr_direct_subwarp_vec4_kernel<32, 1><<<num_blocks, THREADS>>>(
                        rowptr, colind, vals, B, C, M, K, N);
                    break;
                case 64:
                    csr_direct_subwarp_vec4_kernel<32, 2><<<num_blocks, THREADS>>>(
                        rowptr, colind, vals, B, C, M, K, N);
                    break;
                default:
                    csr_direct_subwarp_vec4_kernel<32, 4><<<num_blocks, THREADS>>>(
                        rowptr, colind, vals, B, C, M, K, N);
                    break;
            }
            CUDA_CHECK_KERNEL();
            return;
        }
    }

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
