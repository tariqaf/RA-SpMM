// ============================================================================
// ra_vectorized_coarse.cu - R2: Vectorized Coarse SpMM kernel
//
// Target regime: Ordered sparse / road-network graphs
//   - avg_nnz_per_row in [2, 6], high spatial locality, very uniform degree
//   - Sequential node IDs => adjacent rows share column indices
//   - Standard warp-per-row wastes ~90% of warp capacity on these short rows
//
// Design:
//   - Single kernel with adaptive rows_per_warp (4, 8, 16) based on avg_nnz
//   - Near-zero preprocessing: O(1) — just compute rows_per_warp from features
//   - No device memory allocations (plan_bytes == 0)
//   - GE-SpMM CRC optimization: shared memory ring buffer caches recently
//     loaded B columns, exploiting column-index locality across adjacent rows
//
// Kernel variants:
//   - Scalar: each lane handles one N-column element
//   - float4 vectorized: when N%4==0 and pointers are 16-byte aligned
//
// Plan-run split: make / run / free _ra_vectorized_coarse_plan
// Target: Ampere SM_86 (RTX 3090, RTX A6000), CUDA 12.x
// ============================================================================
#include "../ra_common.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int WARPS_PER_BLOCK = 4;
static constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;

// CRC (Coalesced Row Cache) parameters
static constexpr int CRC_LINES = 4;       // number of cache lines in ring buffer
static constexpr int CRC_LINE_WIDTH = 32;  // floats per cache line (one per lane)

// ---------------------------------------------------------------------------
// Scalar kernel: each warp processes ROWS_PER_WARP consecutive rows.
// Lanes cooperate over the N output dimension.
// Shared memory ring buffer caches B columns for cross-row reuse.
// ---------------------------------------------------------------------------
template<int ROWS_PER_WARP>
__global__ void ra_vectorized_coarse_scalar_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int warp_in_block = threadIdx.x / 32;
    const int base_row = warp_id * ROWS_PER_WARP;

    // Per-warp CRC: ring buffer of recently loaded B columns
    __shared__ int   s_cached_col[WARPS_PER_BLOCK][CRC_LINES];
    __shared__ float s_cached_B[WARPS_PER_BLOCK][CRC_LINES][CRC_LINE_WIDTH];

    // Initialize cache tags to invalid
    if (lane < CRC_LINES) {
        s_cached_col[warp_in_block][lane] = -1;
    }
    __syncwarp();

    int crc_next = 0;  // ring buffer insertion index (same across warp)

    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= M) return;

        const int start = rowptr[row];
        const int end   = rowptr[row + 1];

        for (int n = lane; n < N; n += 32) {
            float acc = 0.f;

            for (int p = start; p < end; ++p) {
                const int col = colind[p];
                const float a_val = vals[p];

                // CRC lookup: check if this column is cached
                float b_val;
                bool hit = false;
                #pragma unroll
                for (int c = 0; c < CRC_LINES; ++c) {
                    if (s_cached_col[warp_in_block][c] == col) {
                        b_val = s_cached_B[warp_in_block][c][lane];
                        hit = true;
                        break;
                    }
                }

                if (!hit) {
                    // Cache miss: load from global memory
                    b_val = __ldg(&B[(i64)col * N + n]);

                    // Update ring buffer (all lanes write cooperatively)
                    s_cached_col[warp_in_block][crc_next] = col;
                    s_cached_B[warp_in_block][crc_next][lane] = b_val;
                    crc_next = (crc_next + 1) % CRC_LINES;
                }

                acc += a_val * b_val;
            }

            C[(i64)row * N + n] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// float4 vectorized kernel: same multi-row-per-warp mapping, but processes
// 4 output columns at a time for better memory bandwidth utilization.
// Requires N%4==0 and 16-byte aligned B and C pointers.
// ---------------------------------------------------------------------------
template<int ROWS_PER_WARP>
__global__ void ra_vectorized_coarse_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int warp_in_block = threadIdx.x / 32;
    const int base_row = warp_id * ROWS_PER_WARP;
    const int N4 = N / 4;

    // Per-warp CRC ring buffer — stores float4 per lane per cache line
    __shared__ int    s_cached_col[WARPS_PER_BLOCK][CRC_LINES];
    __shared__ float4 s_cached_B4[WARPS_PER_BLOCK][CRC_LINES][CRC_LINE_WIDTH];

    // Initialize cache tags to invalid
    if (lane < CRC_LINES) {
        s_cached_col[warp_in_block][lane] = -1;
    }
    __syncwarp();

    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= M) return;

        const int start = rowptr[row];
        const int end   = rowptr[row + 1];

        for (int n4 = lane; n4 < N4; n4 += 32) {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

            int crc_next = 0;

            for (int p = start; p < end; ++p) {
                const int col = colind[p];
                const float a_val = vals[p];

                // CRC lookup
                float4 b4;
                bool hit = false;
                #pragma unroll
                for (int c = 0; c < CRC_LINES; ++c) {
                    if (s_cached_col[warp_in_block][c] == col) {
                        b4 = s_cached_B4[warp_in_block][c][lane];
                        hit = true;
                        break;
                    }
                }

                if (!hit) {
                    const float4* B_ptr = reinterpret_cast<const float4*>(
                        B + (i64)col * N);
                    b4 = __ldg(B_ptr + n4);

                    s_cached_col[warp_in_block][crc_next] = col;
                    s_cached_B4[warp_in_block][crc_next][lane] = b4;
                    crc_next = (crc_next + 1) % CRC_LINES;
                }

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
// No-CRC fallback kernels for when shared memory pressure is a concern or
// when column locality is too low for CRC to help (auto-selected at runtime).
// These are simpler and have lower register pressure.
// ---------------------------------------------------------------------------
template<int ROWS_PER_WARP>
__global__ void ra_vectorized_coarse_nocrc_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int base_row = warp_id * ROWS_PER_WARP;

    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= M) return;

        const int start = rowptr[row];
        const int end   = rowptr[row + 1];

        for (int n = lane; n < N; n += 32) {
            float acc = 0.f;
            for (int p = start; p < end; ++p) {
                acc += vals[p] * __ldg(&B[(i64)colind[p] * N + n]);
            }
            C[(i64)row * N + n] = acc;
        }
    }
}

template<int ROWS_PER_WARP>
__global__ void ra_vectorized_coarse_nocrc_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    const int base_row = warp_id * ROWS_PER_WARP;
    const int N4 = N / 4;

    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int row = base_row + r;
        if (row >= M) return;

        const int start = rowptr[row];
        const int end   = rowptr[row + 1];

        for (int n4 = lane; n4 < N4; n4 += 32) {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int p = start; p < end; ++p) {
                const int col = colind[p];
                const float a_val = vals[p];
                const float4* B_ptr = reinterpret_cast<const float4*>(
                    B + (i64)col * N);
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
// make_ra_vectorized_coarse_plan
//
// O(1) preprocessing: compute avg_nnz_per_row from rowptr endpoints,
// then select rows_per_warp. No device memory allocations.
// ---------------------------------------------------------------------------
void make_ra_vectorized_coarse_plan(
    RAVectorizedCoarsePlan& plan,
    const int* h_rowptr,
    int M, int K)
{
    plan.M = M;
    plan.K = K;
    plan.plan_bytes = 0;

    if (M == 0) {
        plan.rows_per_warp = 8;
        return;
    }

    const int total_nnz = h_rowptr[M] - h_rowptr[0];
    const float avg_nnz = static_cast<float>(total_nnz) / static_cast<float>(M);

    if (avg_nnz <= 3.0f) {
        plan.rows_per_warp = 16;
    } else if (avg_nnz <= 6.0f) {
        plan.rows_per_warp = 8;
    } else {
        plan.rows_per_warp = 4;
    }
}

// ---------------------------------------------------------------------------
// Dispatch helper: launch the correct template specialization.
// Selects CRC vs no-CRC variant based on N (CRC benefits diminish for
// very small N where the cache overhead exceeds the savings).
// ---------------------------------------------------------------------------
namespace {

template<int ROWS_PER_WARP>
void launch_vectorized_coarse(
    const int* d_rowptr, const int* d_colind, const float* d_vals,
    const float* d_B, float* d_C,
    int M, int N, bool use_vec4, bool use_crc)
{
    const int num_warps = (M + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    const int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (num_blocks == 0) return;

    if (use_vec4) {
        if (use_crc) {
            ra_vectorized_coarse_vec4_kernel<ROWS_PER_WARP>
                <<<num_blocks, THREADS_PER_BLOCK>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C, M, N);
        } else {
            ra_vectorized_coarse_nocrc_vec4_kernel<ROWS_PER_WARP>
                <<<num_blocks, THREADS_PER_BLOCK>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C, M, N);
        }
    } else {
        if (use_crc) {
            ra_vectorized_coarse_scalar_kernel<ROWS_PER_WARP>
                <<<num_blocks, THREADS_PER_BLOCK>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C, M, N);
        } else {
            ra_vectorized_coarse_nocrc_kernel<ROWS_PER_WARP>
                <<<num_blocks, THREADS_PER_BLOCK>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C, M, N);
        }
    }
    CUDA_CHECK_KERNEL();
}

}  // anonymous namespace

// ---------------------------------------------------------------------------
// run_ra_vectorized_coarse_plan
//
// Zero the output, select template specialization, launch kernel.
// No cudaDeviceSynchronize — sync is at the Python binding boundary.
// ---------------------------------------------------------------------------
void run_ra_vectorized_coarse_plan(
    const RAVectorizedCoarsePlan& plan,
    const int* d_rowptr, const int* d_colind, const float* d_vals,
    const float* d_B, float* d_C, int N)
{
    const int M = plan.M;
    if (M == 0 || N == 0) return;

    CUDA_CHECK_NEXT(cudaMemset(d_C, 0, (i64)M * N * sizeof(float)));

    // Determine whether to use float4 vectorization
    const bool b_aligned =
        (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
    const bool c_aligned =
        (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
    const bool use_vec4 = (N % 4 == 0) && b_aligned && c_aligned;

    // CRC correctness constraint: the CRC ring buffer caches B values for a
    // specific N-column position. When the N-loop iterates multiple times per
    // lane (N > 32 for scalar, N > 128 for vec4), a cache "hit" returns the
    // wrong B value from a different N position. Therefore CRC is only safe
    // when each lane handles exactly one N position:
    //   - Scalar: N <= 32  (lane handles n=lane only)
    //   - Vec4:   N <= 128 (lane handles n4=lane only, covering 4*32=128 cols)
    const int max_crc_N = use_vec4 ? 128 : 32;
    const bool use_crc = (N <= max_crc_N) && (N >= 16);

    switch (plan.rows_per_warp) {
        case 16:
            launch_vectorized_coarse<16>(
                d_rowptr, d_colind, d_vals, d_B, d_C, M, N,
                use_vec4, use_crc);
            break;
        case 8:
            launch_vectorized_coarse<8>(
                d_rowptr, d_colind, d_vals, d_B, d_C, M, N,
                use_vec4, use_crc);
            break;
        case 4:
            launch_vectorized_coarse<4>(
                d_rowptr, d_colind, d_vals, d_B, d_C, M, N,
                use_vec4, use_crc);
            break;
        default:
            // Fallback: treat as 8 rows/warp
            launch_vectorized_coarse<8>(
                d_rowptr, d_colind, d_vals, d_B, d_C, M, N,
                use_vec4, use_crc);
            break;
    }

    // NO cudaDeviceSynchronize -- sync at Python boundary
}

// ---------------------------------------------------------------------------
// free_ra_vectorized_coarse_plan
//
// No device memory to free. Just zero the fields.
// ---------------------------------------------------------------------------
void free_ra_vectorized_coarse_plan(RAVectorizedCoarsePlan& plan) {
    plan.rows_per_warp = 8;
    plan.M = 0;
    plan.K = 0;
    plan.plan_bytes = 0;
}
