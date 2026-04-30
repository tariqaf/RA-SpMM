// ============================================================================
// ra_zero_overhead.cu - R6: Zero-Overhead CSR SpMM for dense co-purchase /
//                       overhead-sensitive regime
//
// Target regime: degree_cv <= 0.30, avg_nnz 5-10, uniform degree distribution
//
// Strategy: degree-binned dispatch with near-zero preprocessing (O(M) row scan).
// Rows are classified by nnz into 4 bins, each launching a specialized kernel:
//   tiny   (1-4 nnz)  : 8 rows per warp, register-only accumulation
//   short  (5-16 nnz) : 1 warp per row, standard N-parallel
//   medium (17-64 nnz) : 1 block (128 threads = 4 warps) per row, N-chunked
//   long   (65+ nnz)  : multi-block per row with atomicAdd accumulation
//
// All bins have scalar and float4-vectorized variants (float4 when N%4==0).
//
// NO cudaDeviceSynchronize in any function -- sync happens at Python boundary.
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Upload helper (mirrors pattern from row_split.cu)
// ---------------------------------------------------------------------------
namespace {

template <typename T>
T* ra_upload(const std::vector<T>& values) {
    if (values.empty()) return nullptr;
    T* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, values.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, values.data(),
                               values.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
    return d_ptr;
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
constexpr int kTinyThreshold  = 4;    // nnz <= 4
constexpr int kShortThreshold = 16;   // nnz <= 16
constexpr int kMediumThreshold = 64;  // nnz <= 64
                                       // nnz > 64 => long

constexpr int kTinyRowsPerWarp = 8;
constexpr int kMediumThreads   = 128; // 4 warps per block for medium rows
constexpr int kLongChunkSize   = 64;  // nnz per block for long-row splitting

// ============================================================================
// Kernel 1: Tiny rows (1-4 nnz) -- 8 rows per warp, scalar
// ============================================================================
template <int ROWS_PER_WARP>
__global__ void zo_tiny_scalar_kernel(
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
    const int base_idx = warp_id * ROWS_PER_WARP;

    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int idx = base_idx + r;
        if (idx >= num_rows) return;

        const int row = row_ids[idx];
        const int row_start = rowptr[row];
        const int row_end   = rowptr[row + 1];

        for (int n = lane; n < N; n += 32) {
            float acc = 0.0f;
            for (int p = row_start; p < row_end; ++p) {
                const int col = colind[p];
                const float a_val = vals[p];
                acc += a_val * __ldg(&B[(i64)col * N + n]);
            }
            C[(i64)row * N + n] = acc;
        }
    }
}

// ============================================================================
// Kernel 1v: Tiny rows (1-4 nnz) -- 8 rows per warp, float4
// ============================================================================
template <int ROWS_PER_WARP>
__global__ void zo_tiny_vec4_kernel(
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
    const int base_idx = warp_id * ROWS_PER_WARP;
    const int N4 = N / 4;

    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        const int idx = base_idx + r;
        if (idx >= num_rows) return;

        const int row = row_ids[idx];
        const int row_start = rowptr[row];
        const int row_end   = rowptr[row + 1];

        for (int n4 = lane; n4 < N4; n4 += 32) {
            float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
            for (int p = row_start; p < row_end; ++p) {
                const int col = colind[p];
                const float a_val = vals[p];
                const float4* B_ptr =
                    reinterpret_cast<const float4*>(B + (i64)col * N);
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

// ============================================================================
// Kernel 2: Short rows (5-16 nnz) -- 1 warp per row, scalar
// ============================================================================
__global__ void zo_short_scalar_kernel(
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

    for (int n = lane; n < N; n += 32) {
        float acc = 0.0f;
        for (int p = row_start; p < row_end; ++p) {
            const int col = colind[p];
            const float a_val = vals[p];
            acc += a_val * __ldg(&B[(i64)col * N + n]);
        }
        C[(i64)row * N + n] = acc;
    }
}

// ============================================================================
// Kernel 2v: Short rows (5-16 nnz) -- 1 warp per row, float4
// ============================================================================
__global__ void zo_short_vec4_kernel(
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
    const int N4 = N / 4;

    if (warp_id >= num_rows) return;

    const int row = row_ids[warp_id];
    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    for (int n4 = lane; n4 < N4; n4 += 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = row_start; p < row_end; ++p) {
            const int col = colind[p];
            const float a_val = vals[p];
            const float4* B_ptr =
                reinterpret_cast<const float4*>(B + (i64)col * N);
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

// ============================================================================
// Kernel 3: Medium rows (17-64 nnz) -- 1 block (128 threads, 4 warps) per row
// Multiple warps handle different N-column chunks, scalar
// ============================================================================
__global__ void zo_medium_scalar_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const int row = row_ids[row_idx];
    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    // All 128 threads cooperate over N columns
    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float acc = 0.0f;
        for (int p = row_start; p < row_end; ++p) {
            const int col = colind[p];
            const float a_val = vals[p];
            acc += a_val * __ldg(&B[(i64)col * N + n]);
        }
        C[(i64)row * N + n] = acc;
    }
}

// ============================================================================
// Kernel 3v: Medium rows (17-64 nnz) -- 1 block per row, float4
// ============================================================================
__global__ void zo_medium_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const int row = row_ids[row_idx];
    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];
    const int N4 = N / 4;

    for (int n4 = threadIdx.x; n4 < N4; n4 += blockDim.x) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = row_start; p < row_end; ++p) {
            const int col = colind[p];
            const float a_val = vals[p];
            const float4* B_ptr =
                reinterpret_cast<const float4*>(B + (i64)col * N);
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

// ============================================================================
// Kernel 4: Long rows (65+ nnz) -- multi-block per row with atomicAdd
// Each block handles a 64-nnz chunk of the row's nonzeros, scalar
// Grid: (num_chunks, 1, 1) where chunks are enumerated across all long rows
// ============================================================================
__global__ void zo_long_scalar_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    // Each block handles one row. gridDim.y encodes the chunk index.
    const int row_idx   = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    if (row_idx >= num_rows) return;

    const int row = row_ids[row_idx];
    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];
    const int row_nnz   = row_end - row_start;

    // This block handles nnz in [chunk_start, chunk_end)
    const int chunk_start = row_start + chunk_idx * kLongChunkSize;
    const int chunk_end   = min(chunk_start + kLongChunkSize, row_end);
    if (chunk_start >= row_end) return;

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float acc = 0.0f;
        for (int p = chunk_start; p < chunk_end; ++p) {
            const int col = colind[p];
            const float a_val = vals[p];
            acc += a_val * __ldg(&B[(i64)col * N + n]);
        }
        // Accumulate via atomicAdd since multiple blocks write the same row
        atomicAdd(&C[(i64)row * N + n], acc);
    }
}

// ============================================================================
// Kernel 4v: Long rows (65+ nnz) -- multi-block per row, float4
// ============================================================================
__global__ void zo_long_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    const int row_idx   = blockIdx.x;
    const int chunk_idx = blockIdx.y;
    if (row_idx >= num_rows) return;

    const int row = row_ids[row_idx];
    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    const int chunk_start = row_start + chunk_idx * kLongChunkSize;
    const int chunk_end   = min(chunk_start + kLongChunkSize, row_end);
    if (chunk_start >= row_end) return;

    const int N4 = N / 4;

    for (int n4 = threadIdx.x; n4 < N4; n4 += blockDim.x) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = chunk_start; p < chunk_end; ++p) {
            const int col = colind[p];
            const float a_val = vals[p];
            const float4* B_ptr =
                reinterpret_cast<const float4*>(B + (i64)col * N);
            const float4 b4 = __ldg(B_ptr + n4);
            acc.x += a_val * b4.x;
            acc.y += a_val * b4.y;
            acc.z += a_val * b4.z;
            acc.w += a_val * b4.w;
        }
        // atomicAdd for float4 components -- C is pre-zeroed
        float* C_row = C + (i64)row * N + n4 * 4;
        atomicAdd(C_row + 0, acc.x);
        atomicAdd(C_row + 1, acc.y);
        atomicAdd(C_row + 2, acc.z);
        atomicAdd(C_row + 3, acc.w);
    }
}

} // anonymous namespace

// ============================================================================
// make_ra_zero_overhead_plan: O(M) row scan, bin rows by nnz, upload to GPU
// ============================================================================
void make_ra_zero_overhead_plan(
    RAZeroOverheadPlan& plan,
    const int*          h_rowptr,
    int                 M,
    int                 K)
{
    plan.M = M;
    plan.K = K;
    plan.plan_bytes = 0;

    // Bin rows by their nnz count
    std::vector<int> tiny_rows, short_rows, medium_rows, long_rows;
    tiny_rows.reserve(M);
    short_rows.reserve(M);
    medium_rows.reserve(M / 4);
    long_rows.reserve(M / 16);

    for (int i = 0; i < M; ++i) {
        const int nnz = h_rowptr[i + 1] - h_rowptr[i];
        if (nnz == 0) {
            // Empty rows: skip (output already zeroed)
            continue;
        } else if (nnz <= kTinyThreshold) {
            tiny_rows.push_back(i);
        } else if (nnz <= kShortThreshold) {
            short_rows.push_back(i);
        } else if (nnz <= kMediumThreshold) {
            medium_rows.push_back(i);
        } else {
            long_rows.push_back(i);
        }
    }

    plan.num_tiny   = static_cast<int>(tiny_rows.size());
    plan.num_short  = static_cast<int>(short_rows.size());
    plan.num_medium = static_cast<int>(medium_rows.size());
    plan.num_long   = static_cast<int>(long_rows.size());

    // Upload row-id lists to GPU
    plan.d_tiny_row_ids   = ra_upload(tiny_rows);
    plan.d_short_row_ids  = ra_upload(short_rows);
    plan.d_medium_row_ids = ra_upload(medium_rows);
    plan.d_long_row_ids   = ra_upload(long_rows);

    // Record total GPU memory allocated for the plan
    plan.plan_bytes = (tiny_rows.size() + short_rows.size() +
                       medium_rows.size() + long_rows.size()) * sizeof(int);
}

// ============================================================================
// run_ra_zero_overhead_plan: launch binned kernels
// ============================================================================
void run_ra_zero_overhead_plan(
    const RAZeroOverheadPlan& plan,
    const int*                d_rowptr,
    const int*                d_colind,
    const float*              d_vals,
    const float*              d_B,
    float*                    d_C,
    int                       N)
{
    const int M = plan.M;
    if (M == 0 || N == 0) return;

    // Zero output matrix (required for atomicAdd in long kernel, and for
    // empty rows that are not dispatched to any bin)
    CUDA_CHECK_NEXT(cudaMemset(d_C, 0, (i64)M * N * sizeof(float)));

    // Check alignment for float4 vectorization
    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
    const bool use_vec4  = (N % 4 == 0) && b_aligned && c_aligned;

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32; // 128

    // ----- Tiny bin: 8 rows per warp -----
    if (plan.num_tiny > 0) {
        const int num_warps  = (plan.num_tiny + kTinyRowsPerWarp - 1) /
                               kTinyRowsPerWarp;
        const int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) /
                               WARPS_PER_BLOCK;
        if (use_vec4) {
            zo_tiny_vec4_kernel<kTinyRowsPerWarp>
                <<<num_blocks, THREADS>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C,
                    plan.d_tiny_row_ids, plan.num_tiny, N);
        } else {
            zo_tiny_scalar_kernel<kTinyRowsPerWarp>
                <<<num_blocks, THREADS>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C,
                    plan.d_tiny_row_ids, plan.num_tiny, N);
        }
    }

    // ----- Short bin: 1 warp per row -----
    if (plan.num_short > 0) {
        const int num_warps  = plan.num_short;
        const int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) /
                               WARPS_PER_BLOCK;
        if (use_vec4) {
            zo_short_vec4_kernel<<<num_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C,
                plan.d_short_row_ids, plan.num_short, N);
        } else {
            zo_short_scalar_kernel<<<num_blocks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C,
                plan.d_short_row_ids, plan.num_short, N);
        }
    }

    // ----- Medium bin: 1 block (128 threads) per row -----
    if (plan.num_medium > 0) {
        if (use_vec4) {
            zo_medium_vec4_kernel
                <<<plan.num_medium, kMediumThreads>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C,
                    plan.d_medium_row_ids, plan.num_medium, N);
        } else {
            zo_medium_scalar_kernel
                <<<plan.num_medium, kMediumThreads>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C,
                    plan.d_medium_row_ids, plan.num_medium, N);
        }
    }

    // ----- Long bin: multi-block per row with atomicAdd -----
    // Grid: (num_long_rows, max_chunks_per_row)
    // Each block processes kLongChunkSize (64) nonzeros of its assigned row.
    if (plan.num_long > 0) {
        // Find max row length among long rows to determine gridDim.y.
        // We read rowptr from device for the long row IDs. To keep the plan
        // overhead minimal, we compute max chunks from the host rowptr that
        // was used during planning. Since the plan is const here, we use a
        // conservative upper bound: launch enough chunks and let the kernel
        // early-exit for out-of-range chunks.
        //
        // We upload the row_ids at plan time, so we can compute max_nnz by
        // reading d_rowptr for the long rows. But to avoid a device-to-host
        // copy at run time, we use a heuristic upper bound. For truly large
        // matrices the overhead of a few extra empty blocks is negligible.
        //
        // Practical bound: for the target regime (avg_nnz 5-10, cv<=0.30),
        // long rows are extremely rare. We cap at a generous 1024 chunks
        // (covers rows up to 65536 nnz).
        constexpr int kMaxChunksY = 1024;

        // A tighter bound: read the last long row's rowptr range from device.
        // This single cudaMemcpy is acceptable because num_long is typically
        // very small (< 0.1% of rows in the target regime).
        int max_chunks = 1;
        if (plan.num_long <= 64) {
            // For a small number of long rows, read their rowptr entries
            std::vector<int> long_row_ids_h(plan.num_long);
            CUDA_CHECK_NEXT(cudaMemcpy(long_row_ids_h.data(),
                                       plan.d_long_row_ids,
                                       plan.num_long * sizeof(int),
                                       cudaMemcpyDeviceToHost));
            int max_nnz = 0;
            for (int i = 0; i < plan.num_long; ++i) {
                int rstart = 0, rend = 0;
                CUDA_CHECK_NEXT(cudaMemcpy(&rstart,
                                           d_rowptr + long_row_ids_h[i],
                                           sizeof(int),
                                           cudaMemcpyDeviceToHost));
                CUDA_CHECK_NEXT(cudaMemcpy(&rend,
                                           d_rowptr + long_row_ids_h[i] + 1,
                                           sizeof(int),
                                           cudaMemcpyDeviceToHost));
                max_nnz = std::max(max_nnz, rend - rstart);
            }
            max_chunks = (max_nnz + kLongChunkSize - 1) / kLongChunkSize;
        } else {
            max_chunks = kMaxChunksY;
        }
        max_chunks = std::min(max_chunks, kMaxChunksY);
        max_chunks = std::max(max_chunks, 1);

        dim3 grid_long(plan.num_long, max_chunks);
        if (use_vec4) {
            zo_long_vec4_kernel<<<grid_long, kMediumThreads>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C,
                plan.d_long_row_ids, plan.num_long, N);
        } else {
            zo_long_scalar_kernel<<<grid_long, kMediumThreads>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C,
                plan.d_long_row_ids, plan.num_long, N);
        }
    }

    CUDA_CHECK_KERNEL();
    // NO cudaDeviceSynchronize -- sync happens at Python boundary
}

// ============================================================================
// free_ra_zero_overhead_plan: release GPU memory
// ============================================================================
void free_ra_zero_overhead_plan(RAZeroOverheadPlan& plan) {
    if (plan.d_tiny_row_ids)   { cudaFree(plan.d_tiny_row_ids);   plan.d_tiny_row_ids   = nullptr; }
    if (plan.d_short_row_ids)  { cudaFree(plan.d_short_row_ids);  plan.d_short_row_ids  = nullptr; }
    if (plan.d_medium_row_ids) { cudaFree(plan.d_medium_row_ids); plan.d_medium_row_ids = nullptr; }
    if (plan.d_long_row_ids)   { cudaFree(plan.d_long_row_ids);   plan.d_long_row_ids   = nullptr; }

    plan.num_tiny   = 0;
    plan.num_short  = 0;
    plan.num_medium = 0;
    plan.num_long   = 0;
    plan.M = 0;
    plan.K = 0;
    plan.plan_bytes = 0;
}
