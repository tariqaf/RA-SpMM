// ============================================================================
// ra_zero_overhead.cu - R6: Zero-Overhead CSR SpMM for dense co-purchase /
//                       overhead-sensitive regime
//
// Target regime: degree_cv <= 0.30, avg_nnz 5-10, uniform degree distribution
//
// Strategy: near-zero preprocessing (single O(M) row scan) producing
//   - one unified small-row list (1-64 nnz) stored bin-ordered (tiny, short,
//     medium) so co-scheduled warps see similar row lengths, executed by a
//     single subwarp kernel launch: a W-lane subwarp owns one row, packs
//     32/W rows per warp (no idle lanes at N=64), keeps S float4
//     accumulators (single A pass for N up to 512), and distributes
//     coalesced colind/vals chunks with __shfl_sync;
//   - a flattened long-row chunk list (256-nnz chunks): one block per real
//     chunk (no rectangular mostly-empty grid), sole chunks (row fits in
//     one chunk) store directly, only multi-chunk rows use atomicAdd;
//   - a zero list (empty rows + multi-chunk rows) cleared by a small slice
//     kernel instead of a full M x N cudaMemset.
//
// No device-wide synchronization; callers synchronize only when host
// visibility is required.
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Upload helper (mirrors pattern from row_split.cu)
// ---------------------------------------------------------------------------
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
constexpr int kTinyThreshold   = 4;    // nnz <= 4
constexpr int kShortThreshold  = 16;   // nnz <= 16
constexpr int kMediumThreshold = 64;   // nnz <= 64
                                       // nnz > 64 => long
constexpr int kLongChunkSize   = 256;  // nnz per block for long-row splitting
constexpr unsigned kSoleChunkFlag = 0x80000000u;

// ============================================================================
// Zeroing kernel: clear the C slices of listed rows (empty + multi-chunk).
// ============================================================================
__global__ void zo_zero_rows_vec4_kernel(
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    int num_rows, int N4, int N)
{
    const i64 total = (i64)num_rows * N4;
    for (i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (i64)gridDim.x * blockDim.x) {
        const int row = row_ids[idx / N4];
        const int n4  = static_cast<int>(idx % N4);
        reinterpret_cast<float4*>(C + (i64)row * N)[n4] =
            make_float4(0.f, 0.f, 0.f, 0.f);
    }
}

__global__ void zo_zero_rows_scalar_kernel(
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    int num_rows, int N)
{
    const i64 total = (i64)num_rows * N;
    for (i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (i64)gridDim.x * blockDim.x) {
        const int row = row_ids[idx / N];
        C[(i64)row * N + (idx % N)] = 0.f;
    }
}

// ============================================================================
// Unified small-row subwarp kernel (vec4). W-lane subwarp per row, S float4
// accumulators per lane; launcher guarantees N4 == W*S exactly.
// ============================================================================
template <int W, int S>
__global__ void zo_small_subwarp_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    int num_rows, int N)
{
    constexpr int SUBWARPS = 32 / W;
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane    = threadIdx.x & 31;
    const int sub     = lane / W;
    const int sl      = lane % W;
    const unsigned mask =
        (W == 32) ? 0xffffffffu : (((1u << W) - 1u) << (sub * W));

    const int idx = warp_id * SUBWARPS + sub;
    if (idx >= num_rows) return;

    const int row       = row_ids[idx];
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

// Generic vec4 fallback for N4 without an exact (W, S) fit: warp per row,
// lane-strided n4 loop.
__global__ void zo_small_warp_vec4_kernel(
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

    const int row       = row_ids[warp_id];
    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    for (int n4 = lane; n4 < N4; n4 += 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = row_start; p < row_end; ++p) {
            const int col = __ldg(colind + p);
            const float a_val = __ldg(vals + p);
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

// Scalar fallback (N % 4 != 0 or unaligned): warp per row over the list.
__global__ void zo_small_warp_scalar_kernel(
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

    const int row       = row_ids[warp_id];
    const int row_start = rowptr[row];
    const int row_end   = rowptr[row + 1];

    for (int n = lane; n < N; n += 32) {
        float acc = 0.0f;
        for (int p = row_start; p < row_end; ++p) {
            const int col = __ldg(colind + p);
            const float a_val = __ldg(vals + p);
            acc += a_val * __ldg(&B[(i64)col * N + n]);
        }
        C[(i64)row * N + n] = acc;
    }
}

// ============================================================================
// Long-row chunk kernels: one block per real chunk. Sole chunks (bit 31 of
// d_chunk_rows) cover their whole row and store directly; multi-chunk rows
// accumulate into a pre-zeroed C with atomicAdd.
// ============================================================================
__global__ void zo_chunk_vec4_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ chunk_rows,
    const int*   __restrict__ chunk_starts,
    int num_chunks, int N)
{
    const int c = blockIdx.x;
    if (c >= num_chunks) return;

    const unsigned tag = static_cast<unsigned>(chunk_rows[c]);
    const bool sole = (tag & kSoleChunkFlag) != 0u;
    const int row = static_cast<int>(tag & ~kSoleChunkFlag);

    const int chunk_start = chunk_starts[c];
    const int chunk_end   = min(chunk_start + kLongChunkSize, rowptr[row + 1]);
    const int N4 = N / 4;

    for (int n4 = threadIdx.x; n4 < N4; n4 += blockDim.x) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = chunk_start; p < chunk_end; ++p) {
            const int col = __ldg(colind + p);
            const float a_val = __ldg(vals + p);
            const float4* B_ptr =
                reinterpret_cast<const float4*>(B + (i64)col * N);
            const float4 b4 = __ldg(B_ptr + n4);
            acc.x += a_val * b4.x;
            acc.y += a_val * b4.y;
            acc.z += a_val * b4.z;
            acc.w += a_val * b4.w;
        }
        if (sole) {
            reinterpret_cast<float4*>(C + (i64)row * N)[n4] = acc;
        } else {
            float* C_row = C + (i64)row * N + n4 * 4;
            atomicAdd(C_row + 0, acc.x);
            atomicAdd(C_row + 1, acc.y);
            atomicAdd(C_row + 2, acc.z);
            atomicAdd(C_row + 3, acc.w);
        }
    }
}

__global__ void zo_chunk_scalar_kernel(
    const int*   __restrict__ rowptr,
    const int*   __restrict__ colind,
    const float* __restrict__ vals,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ chunk_rows,
    const int*   __restrict__ chunk_starts,
    int num_chunks, int N)
{
    const int c = blockIdx.x;
    if (c >= num_chunks) return;

    const unsigned tag = static_cast<unsigned>(chunk_rows[c]);
    const bool sole = (tag & kSoleChunkFlag) != 0u;
    const int row = static_cast<int>(tag & ~kSoleChunkFlag);

    const int chunk_start = chunk_starts[c];
    const int chunk_end   = min(chunk_start + kLongChunkSize, rowptr[row + 1]);

    for (int n = threadIdx.x; n < N; n += blockDim.x) {
        float acc = 0.0f;
        for (int p = chunk_start; p < chunk_end; ++p) {
            const int col = __ldg(colind + p);
            const float a_val = __ldg(vals + p);
            acc += a_val * __ldg(&B[(i64)col * N + n]);
        }
        if (sole) {
            C[(i64)row * N + n] = acc;
        } else {
            atomicAdd(&C[(i64)row * N + n], acc);
        }
    }
}

} // anonymous namespace

// ============================================================================
// make_ra_zero_overhead_plan: O(M) row scan
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

    std::vector<int> tiny_rows, short_rows, medium_rows;
    std::vector<int> chunk_rows, chunk_starts, zero_rows;
    tiny_rows.reserve(M);
    short_rows.reserve(M);
    medium_rows.reserve(M / 4);

    int max_long_nnz = 0;
    int num_empty = 0;
    int num_long = 0;
    for (int i = 0; i < M; ++i) {
        const int start = h_rowptr[i];
        const int nnz = h_rowptr[i + 1] - start;
        if (nnz == 0) {
            ++num_empty;
            zero_rows.push_back(i);
        } else if (nnz <= kTinyThreshold) {
            tiny_rows.push_back(i);
        } else if (nnz <= kShortThreshold) {
            short_rows.push_back(i);
        } else if (nnz <= kMediumThreshold) {
            medium_rows.push_back(i);
        } else {
            ++num_long;
            max_long_nnz = std::max(max_long_nnz, nnz);
            if (nnz <= kLongChunkSize) {
                chunk_rows.push_back(
                    static_cast<int>(static_cast<unsigned>(i) | kSoleChunkFlag));
                chunk_starts.push_back(start);
            } else {
                zero_rows.push_back(i);
                for (int off = 0; off < nnz; off += kLongChunkSize) {
                    chunk_rows.push_back(i);
                    chunk_starts.push_back(start + off);
                }
            }
        }
    }

    plan.num_tiny   = static_cast<int>(tiny_rows.size());
    plan.num_short  = static_cast<int>(short_rows.size());
    plan.num_medium = static_cast<int>(medium_rows.size());
    plan.num_long   = num_long;
    plan.num_empty  = num_empty;
    plan.max_long_chunks =
        (max_long_nnz + kLongChunkSize - 1) / kLongChunkSize;

    // Bin-ordered unified small-row list: tiny, then short, then medium.
    std::vector<int> small_rows;
    small_rows.reserve(tiny_rows.size() + short_rows.size() +
                       medium_rows.size());
    small_rows.insert(small_rows.end(), tiny_rows.begin(), tiny_rows.end());
    small_rows.insert(small_rows.end(), short_rows.begin(), short_rows.end());
    small_rows.insert(small_rows.end(), medium_rows.begin(), medium_rows.end());

    plan.num_small      = static_cast<int>(small_rows.size());
    plan.num_chunks     = static_cast<int>(chunk_rows.size());
    plan.num_zero_rows  = static_cast<int>(zero_rows.size());

    plan.d_small_row_ids = ra_upload(small_rows);
    plan.d_chunk_rows    = ra_upload(chunk_rows);
    plan.d_chunk_starts  = ra_upload(chunk_starts);
    plan.d_zero_row_ids  = ra_upload(zero_rows);

    plan.plan_bytes = (small_rows.size() + chunk_rows.size() +
                       chunk_starts.size() + zero_rows.size()) * sizeof(int);
}

// ============================================================================
// run_ra_zero_overhead_plan: launch zeroing + small + chunk kernels
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

    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
    const bool use_vec4  = (N % 4 == 0) && b_aligned && c_aligned;

    constexpr int WARPS_PER_BLOCK = 4;
    constexpr int THREADS = WARPS_PER_BLOCK * 32; // 128

    // ----- Pre-zero only the rows that need it -----
    if (plan.num_zero_rows > 0) {
        if (plan.num_zero_rows > M / 4) {
            CUDA_CHECK_NEXT(cudaMemsetAsync(d_C, 0,
                                            (i64)M * N * sizeof(float)));
        } else if (use_vec4) {
            const int N4 = N / 4;
            const i64 total = (i64)plan.num_zero_rows * N4;
            const int blocks = static_cast<int>(
                std::min<i64>((total + THREADS - 1) / THREADS, 65535));
            zo_zero_rows_vec4_kernel<<<blocks, THREADS>>>(
                d_C, plan.d_zero_row_ids, plan.num_zero_rows, N4, N);
        } else {
            const i64 total = (i64)plan.num_zero_rows * N;
            const int blocks = static_cast<int>(
                std::min<i64>((total + THREADS - 1) / THREADS, 65535));
            zo_zero_rows_scalar_kernel<<<blocks, THREADS>>>(
                d_C, plan.d_zero_row_ids, plan.num_zero_rows, N);
        }
    }

    // ----- Small rows (1-64 nnz): one unified launch -----
    if (plan.num_small > 0) {
        int W = 0, S = 0;
        if (use_vec4) {
            const int N4 = N / 4;
            if      (N4 == 8)   { W = 8;  S = 1; }
            else if (N4 == 16)  { W = 16; S = 1; }
            else if (N4 == 32)  { W = 32; S = 1; }
            else if (N4 == 64)  { W = 32; S = 2; }
            else if (N4 == 128) { W = 32; S = 4; }
        }
        if (W != 0) {
            const int subwarps   = 32 / W;
            const int num_warps  = (plan.num_small + subwarps - 1) / subwarps;
            const int num_blocks = (num_warps + WARPS_PER_BLOCK - 1) /
                                   WARPS_PER_BLOCK;
            const int N4 = N / 4;
            switch (N4) {
                case 8:
                    zo_small_subwarp_vec4_kernel<8, 1><<<num_blocks, THREADS>>>(
                        d_rowptr, d_colind, d_vals, d_B, d_C,
                        plan.d_small_row_ids, plan.num_small, N);
                    break;
                case 16:
                    zo_small_subwarp_vec4_kernel<16, 1><<<num_blocks, THREADS>>>(
                        d_rowptr, d_colind, d_vals, d_B, d_C,
                        plan.d_small_row_ids, plan.num_small, N);
                    break;
                case 32:
                    zo_small_subwarp_vec4_kernel<32, 1><<<num_blocks, THREADS>>>(
                        d_rowptr, d_colind, d_vals, d_B, d_C,
                        plan.d_small_row_ids, plan.num_small, N);
                    break;
                case 64:
                    zo_small_subwarp_vec4_kernel<32, 2><<<num_blocks, THREADS>>>(
                        d_rowptr, d_colind, d_vals, d_B, d_C,
                        plan.d_small_row_ids, plan.num_small, N);
                    break;
                default:
                    zo_small_subwarp_vec4_kernel<32, 4><<<num_blocks, THREADS>>>(
                        d_rowptr, d_colind, d_vals, d_B, d_C,
                        plan.d_small_row_ids, plan.num_small, N);
                    break;
            }
        } else {
            const int num_blocks = (plan.num_small + WARPS_PER_BLOCK - 1) /
                                   WARPS_PER_BLOCK;
            if (use_vec4) {
                zo_small_warp_vec4_kernel<<<num_blocks, THREADS>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C,
                    plan.d_small_row_ids, plan.num_small, N);
            } else {
                zo_small_warp_scalar_kernel<<<num_blocks, THREADS>>>(
                    d_rowptr, d_colind, d_vals, d_B, d_C,
                    plan.d_small_row_ids, plan.num_small, N);
            }
        }
    }

    // ----- Long rows: one block per real chunk -----
    if (plan.num_chunks > 0) {
        if (use_vec4) {
            zo_chunk_vec4_kernel<<<plan.num_chunks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C,
                plan.d_chunk_rows, plan.d_chunk_starts, plan.num_chunks, N);
        } else {
            zo_chunk_scalar_kernel<<<plan.num_chunks, THREADS>>>(
                d_rowptr, d_colind, d_vals, d_B, d_C,
                plan.d_chunk_rows, plan.d_chunk_starts, plan.num_chunks, N);
        }
    }

    CUDA_CHECK_KERNEL();
    // Execution remains asynchronous with respect to the host.
}

// ============================================================================
// free_ra_zero_overhead_plan: release GPU memory
// ============================================================================
void free_ra_zero_overhead_plan(RAZeroOverheadPlan& plan) {
    if (plan.d_small_row_ids) { cudaFree(plan.d_small_row_ids); plan.d_small_row_ids = nullptr; }
    if (plan.d_chunk_rows)    { cudaFree(plan.d_chunk_rows);    plan.d_chunk_rows    = nullptr; }
    if (plan.d_chunk_starts)  { cudaFree(plan.d_chunk_starts);  plan.d_chunk_starts  = nullptr; }
    if (plan.d_zero_row_ids)  { cudaFree(plan.d_zero_row_ids);  plan.d_zero_row_ids  = nullptr; }
    plan.num_small = plan.num_chunks = plan.num_zero_rows = 0;
    plan.num_tiny = plan.num_short = plan.num_medium = 0;
    plan.num_long = plan.num_empty = plan.max_long_chunks = 0;
    plan.plan_bytes = 0;
}
