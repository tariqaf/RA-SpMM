// ============================================================================
// ra_rode_enhanced.cu - R1: RoDe-Enhanced SpMM for hub-dominated power-law
//
// Regime: degree_cv > 1.2, heavy tail, top-5 rows hold >20% nnz.
//
// Design: RoDe-style block-residual decomposition with sub-block pipelining
// for long rows. Each row is split into:
//   - Regular part: largest 32-aligned nnz prefix
//   - Residual part: 0-31 nnz suffix
//
// Long rows (regular_nnz >= 128) get sub-block pipelining: the regular part
// is split into 32-nnz sub-blocks, and warps within a CTA process these
// sub-blocks in a pipeline with double-buffered shared memory.
//
// Three kernel families:
//   1. Short-row kernel  (regular_nnz < 128): one CTA per row, warps tile N
//   2. Long-row pipelined (regular_nnz >= 128): 256-thread CTA, double-buffered
//      shared memory pipeline with load/compute overlap
//   3. Residual kernel   (0-31 nnz tail): warp-per-row additive update
//
// Target: Ampere SM_86 (RTX 3090, RTX A6000), CUDA 12.x
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
namespace {

constexpr int kLongRowThreshold = 128;  // regular_nnz >= 128 => long row
constexpr int kSubBlockSize     = 32;   // nnz per sub-block in pipeline
constexpr int kLongCTAThreads   = 256;  // 8 warps per CTA for long rows
constexpr int kLongComputeWarps = 7;    // warps 1-7 compute, warp 0 loads
constexpr int kSmemPad          = 33;   // 33 instead of 32 to avoid bank conflicts

// ---------------------------------------------------------------------------
// Upload helper
// ---------------------------------------------------------------------------
template <typename T>
T* ra_upload(const std::vector<T>& v) {
    if (v.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d, v.data(), v.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
    return d;
}

// ============================================================================
// Kernel 1: Short-row scalar kernel (regular_nnz < 128)
//
// One CTA per row. Warps tile the N dimension. The regular 32-aligned prefix
// is processed in chunks of 32 with unrolled inner loops.
// ============================================================================
__global__ void rode_short_scalar_kernel(
    const int*   __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    const int*   __restrict__ starts,
    const int*   __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;

    const int row       = row_ids[row_idx];
    const int start     = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n = warp * 32 + lane; n < N; n += warps_per_block * 32) {
        float acc = 0.f;
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                acc += d_val[p] * __ldg(&B[(i64)d_col[p] * N + n]);
            }
        }
        C[(i64)row * N + n] = acc;
    }
}

// ============================================================================
// Kernel 1b: Short-row float4 kernel
// ============================================================================
__global__ void rode_short_vec4_kernel(
    const int*   __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    const int*   __restrict__ starts,
    const int*   __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int N4 = N / 4;

    const int row       = row_ids[row_idx];
    const int start     = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n4 = warp * 32 + lane; n4 < N4; n4 += warps_per_block * 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                const float4* b_ptr =
                    reinterpret_cast<const float4*>(B + (i64)d_col[p] * N);
                const float4 b4 = __ldg(b_ptr + n4);
                const float a = d_val[p];
                acc.x += a * b4.x;
                acc.y += a * b4.y;
                acc.z += a * b4.z;
                acc.w += a * b4.w;
            }
        }
        float4* c_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
        c_ptr[n4] = acc;
    }
}

// ============================================================================
// Kernel 2: Long-row pipelined scalar kernel (regular_nnz >= 128)
//
// THE KEY INNOVATION: Double-buffered shared memory pipeline.
//
// CTA layout: 256 threads = 8 warps
//   - Warp 0: dedicated loader — prefetches next sub-block into smem[buf^1]
//   - Warps 1-7: compute warps — accumulate from smem[buf] across N dimension
//
// Shared memory layout (double-buffered):
//   smem_col[2][32]     — column indices for current/next sub-block
//   smem_val[2][32]     — A values for current/next sub-block
//
// Pipeline pseudocode:
//   buf = 0
//   all_threads_load_subblock(buf=0, sb=0)      // cooperative first load
//   __syncthreads()
//   for sb = 0 .. num_sub_blocks-1:
//       if sb+1 < num_sub_blocks:
//           warp0_load_subblock(buf=1-buf, sb=sb+1)  // overlap load
//       warps1to7_compute(buf, sb)                    // compute current
//       __syncthreads()
//       buf = 1 - buf
//   write_accumulated_results()
// ============================================================================
__global__ void rode_long_pipelined_scalar_kernel(
    const int*   __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    const int*   __restrict__ starts,
    const int*   __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    const int row       = row_ids[row_idx];
    const int start     = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];
    const int num_sub_blocks = block_nnz / kSubBlockSize;

    // Double-buffered shared memory for sub-block data
    __shared__ int   smem_col[2][kSubBlockSize];
    __shared__ float smem_val[2][kSubBlockSize];

    // ---- Phase 0: Cooperative load of first sub-block into buffer 0 ----
    if (threadIdx.x < kSubBlockSize) {
        const int p = start + threadIdx.x;
        smem_col[0][threadIdx.x] = d_col[p];
        smem_val[0][threadIdx.x] = d_val[p];
    }
    __syncthreads();

    // ---- Accumulation registers for computing warps (warps 1-7) ----
    // Each computing warp covers a strided chunk of N columns.
    // We use register-based accumulation: one float per N column position.
    // The loop below iterates over N in chunks.

    int buf = 0;

    // Outer loop over N-column tiles for computing warps
    // Each computing warp handles columns: (warp_id-1)*32+lane, stepping by 7*32
    // warp_id 0 is the loader; warp_ids 1..7 are computing warps (index 0..6)
    const int compute_warp = warp_id - 1;  // -1 for warp 0 (loader)

    for (int n_base = 0; n_base < N; n_base += kLongComputeWarps * 32) {
        const int n = n_base + compute_warp * 32 + lane;
        const bool n_valid = (warp_id >= 1) && (n < N);

        float acc = 0.f;
        int cur_buf = 0;

        // Re-sync for this N-tile pass; first sub-block already loaded
        // (We reload for each N-tile to keep register pressure low)
        if (n_base > 0) {
            __syncthreads();
            if (threadIdx.x < kSubBlockSize) {
                const int p = start + threadIdx.x;
                smem_col[0][threadIdx.x] = d_col[p];
                smem_val[0][threadIdx.x] = d_val[p];
            }
            __syncthreads();
            cur_buf = 0;
        }

        // ---- Pipeline loop over sub-blocks ----
        for (int sb = 0; sb < num_sub_blocks; ++sb) {
            // Warp 0: prefetch next sub-block into alternate buffer
            if (warp_id == 0 && sb + 1 < num_sub_blocks) {
                const int next_offset = start + (sb + 1) * kSubBlockSize + lane;
                if (lane < kSubBlockSize) {
                    smem_col[1 - cur_buf][lane] = d_col[next_offset];
                    smem_val[1 - cur_buf][lane] = d_val[next_offset];
                }
            }

            // Warps 1-7: compute accumulation from current buffer
            if (n_valid) {
#pragma unroll
                for (int e = 0; e < kSubBlockSize; ++e) {
                    const int col = smem_col[cur_buf][e];
                    const float a_val = smem_val[cur_buf][e];
                    acc += a_val * __ldg(&B[(i64)col * N + n]);
                }
            }

            __syncthreads();
            cur_buf = 1 - cur_buf;
        }

        // Write accumulated result for this N-tile
        if (n_valid) {
            C[(i64)row * N + n] = acc;
        }
    }
}

// ============================================================================
// Kernel 2b: Long-row pipelined float4 kernel
// ============================================================================
__global__ void rode_long_pipelined_vec4_kernel(
    const int*   __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ row_ids,
    const int*   __restrict__ starts,
    const int*   __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) return;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int N4 = N / 4;

    const int row       = row_ids[row_idx];
    const int start     = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];
    const int num_sub_blocks = block_nnz / kSubBlockSize;

    // Double-buffered shared memory
    __shared__ int   smem_col[2][kSubBlockSize];
    __shared__ float smem_val[2][kSubBlockSize];

    // Cooperative load of first sub-block
    if (threadIdx.x < kSubBlockSize) {
        const int p = start + threadIdx.x;
        smem_col[0][threadIdx.x] = d_col[p];
        smem_val[0][threadIdx.x] = d_val[p];
    }
    __syncthreads();

    const int compute_warp = warp_id - 1;

    for (int n4_base = 0; n4_base < N4; n4_base += kLongComputeWarps * 32) {
        const int n4 = n4_base + compute_warp * 32 + lane;
        const bool n4_valid = (warp_id >= 1) && (n4 < N4);

        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        int cur_buf = 0;

        // Reload first sub-block for each N4-tile pass
        if (n4_base > 0) {
            __syncthreads();
            if (threadIdx.x < kSubBlockSize) {
                const int p = start + threadIdx.x;
                smem_col[0][threadIdx.x] = d_col[p];
                smem_val[0][threadIdx.x] = d_val[p];
            }
            __syncthreads();
            cur_buf = 0;
        }

        for (int sb = 0; sb < num_sub_blocks; ++sb) {
            // Warp 0: prefetch next sub-block
            if (warp_id == 0 && sb + 1 < num_sub_blocks) {
                const int next_offset = start + (sb + 1) * kSubBlockSize + lane;
                if (lane < kSubBlockSize) {
                    smem_col[1 - cur_buf][lane] = d_col[next_offset];
                    smem_val[1 - cur_buf][lane] = d_val[next_offset];
                }
            }

            // Warps 1-7: compute
            if (n4_valid) {
#pragma unroll
                for (int e = 0; e < kSubBlockSize; ++e) {
                    const int col = smem_col[cur_buf][e];
                    const float a_val = smem_val[cur_buf][e];
                    const float4* b_ptr =
                        reinterpret_cast<const float4*>(B + (i64)col * N);
                    const float4 b4 = __ldg(b_ptr + n4);
                    acc.x += a_val * b4.x;
                    acc.y += a_val * b4.y;
                    acc.z += a_val * b4.z;
                    acc.w += a_val * b4.w;
                }
            }

            __syncthreads();
            cur_buf = 1 - cur_buf;
        }

        if (n4_valid) {
            float4* c_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
            c_ptr[n4] = acc;
        }
    }
}

// ============================================================================
// Kernel 3: Residual scalar kernel (0-31 nnz tail per row)
//
// Lightweight warp-per-row handler. Adds into already-computed regular output.
// No atomics needed: residual kernel runs after short/long kernels, and each
// row is owned by exactly one residual descriptor.
// ============================================================================
__global__ void rode_residual_scalar_kernel(
    const int*   __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ d_res_row_ids,
    const int*   __restrict__ d_res_starts,
    const int*   __restrict__ d_res_lengths,
    int num_residual,
    int N)
{
    const int residual_idx = blockIdx.x;
    if (residual_idx >= num_residual) return;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;

    const int row   = d_res_row_ids[residual_idx];
    const int start = d_res_starts[residual_idx];
    const int len   = d_res_lengths[residual_idx];

    for (int n = warp * 32 + lane; n < N; n += warps_per_block * 32) {
        float acc = 0.f;
        for (int p = 0; p < len; ++p) {
            acc += d_val[start + p] * __ldg(&B[(i64)d_col[start + p] * N + n]);
        }
        C[(i64)row * N + n] += acc;
    }
}

// ============================================================================
// Kernel 3b: Residual float4 kernel
// ============================================================================
__global__ void rode_residual_vec4_kernel(
    const int*   __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float*       __restrict__ C,
    const int*   __restrict__ d_res_row_ids,
    const int*   __restrict__ d_res_starts,
    const int*   __restrict__ d_res_lengths,
    int num_residual,
    int N)
{
    const int residual_idx = blockIdx.x;
    if (residual_idx >= num_residual) return;

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int N4 = N / 4;

    const int row   = d_res_row_ids[residual_idx];
    const int start = d_res_starts[residual_idx];
    const int len   = d_res_lengths[residual_idx];

    for (int n4 = warp * 32 + lane; n4 < N4; n4 += warps_per_block * 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = 0; p < len; ++p) {
            const float4* b_ptr =
                reinterpret_cast<const float4*>(B + (i64)d_col[start + p] * N);
            const float4 b4 = __ldg(b_ptr + n4);
            const float a = d_val[start + p];
            acc.x += a * b4.x;
            acc.y += a * b4.y;
            acc.z += a * b4.z;
            acc.w += a * b4.w;
        }
        float4* c_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
        float4 cur = c_ptr[n4];
        cur.x += acc.x;
        cur.y += acc.y;
        cur.z += acc.z;
        cur.w += acc.w;
        c_ptr[n4] = cur;
    }
}

}  // anonymous namespace

// ============================================================================
// make_ra_rode_enhanced_plan
//
// Reads h_rowptr[0..M] and classifies each row into:
//   - Short (regular_nnz in [32, 128)): standard CTA-per-row execution
//   - Long  (regular_nnz >= 128): sub-block pipelined execution
//   - Residual (0-31 nnz suffix): lightweight additive tail
//
// For long rows, generates sub-block descriptors splitting the regular prefix
// into 32-nnz sub-blocks for the pipeline scheduler.
// ============================================================================
void make_ra_rode_enhanced_plan(RARodeEnhancedPlan& plan,
                                const int* h_rowptr, int M, int K)
{
    plan = RARodeEnhancedPlan();
    plan.M = M;
    plan.K = K;
    plan.sub_block_size = kSubBlockSize;

    if (M <= 0) return;

    // Host-side descriptor vectors
    std::vector<int> short_row_ids, short_starts, short_block_nnz;
    std::vector<int> long_row_ids,  long_starts,  long_block_nnz;
    std::vector<int> long_sub_starts, long_sub_counts, long_sub_row_map;
    std::vector<int> res_row_ids, res_starts, res_lengths;

    i64 total_nnz     = 0;
    i64 regular_total = 0;
    i64 long_nnz      = 0;

    // ---- Row classification ----
    for (int row = 0; row < M; ++row) {
        const int start = h_rowptr[row];
        const int nnz_i = h_rowptr[row + 1] - start;
        const int regular_nnz  = (nnz_i / kSubBlockSize) * kSubBlockSize;
        const int residual_nnz = nnz_i - regular_nnz;

        total_nnz += nnz_i;
        regular_total += regular_nnz;

        // Classify regular prefix
        if (regular_nnz > 0) {
            if (regular_nnz >= kLongRowThreshold) {
                // Long row: sub-block pipelined execution
                long_row_ids.push_back(row);
                long_starts.push_back(start);
                long_block_nnz.push_back(regular_nnz);
                long_nnz += regular_nnz;

                // Generate sub-block descriptors
                const int num_sub = regular_nnz / kSubBlockSize;
                for (int sb = 0; sb < num_sub; ++sb) {
                    long_sub_starts.push_back(start + sb * kSubBlockSize);
                    long_sub_counts.push_back(kSubBlockSize);
                    long_sub_row_map.push_back(row);
                }
            } else {
                // Short row: standard CTA-per-row
                short_row_ids.push_back(row);
                short_starts.push_back(start);
                short_block_nnz.push_back(regular_nnz);
            }
        }

        // Residual tail (always additive, runs after regular kernels)
        if (residual_nnz > 0) {
            res_row_ids.push_back(row);
            res_starts.push_back(start + regular_nnz);
            res_lengths.push_back(residual_nnz);
        }
    }

    // ---- Fill plan counts ----
    plan.num_short_rows     = static_cast<int>(short_row_ids.size());
    plan.num_long_rows      = static_cast<int>(long_row_ids.size());
    plan.num_long_sub_blocks = static_cast<int>(long_sub_starts.size());
    plan.num_residual       = static_cast<int>(res_row_ids.size());

    // ---- Diagnostics ----
    plan.regular_nnz_fraction = (total_nnz > 0)
        ? static_cast<float>(regular_total) / static_cast<float>(total_nnz)
        : 0.f;
    plan.long_row_nnz_fraction = (total_nnz > 0)
        ? static_cast<float>(long_nnz) / static_cast<float>(total_nnz)
        : 0.f;

    // ---- Upload to GPU ----
    plan.d_short_row_ids   = ra_upload(short_row_ids);
    plan.d_short_starts    = ra_upload(short_starts);
    plan.d_short_block_nnz = ra_upload(short_block_nnz);

    plan.d_long_row_ids    = ra_upload(long_row_ids);
    plan.d_long_starts     = ra_upload(long_starts);
    plan.d_long_block_nnz  = ra_upload(long_block_nnz);

    plan.d_long_sub_starts  = ra_upload(long_sub_starts);
    plan.d_long_sub_counts  = ra_upload(long_sub_counts);
    plan.d_long_sub_row_map = ra_upload(long_sub_row_map);

    plan.d_res_row_ids  = ra_upload(res_row_ids);
    plan.d_res_starts   = ra_upload(res_starts);
    plan.d_res_lengths  = ra_upload(res_lengths);

    // ---- Plan memory accounting ----
    plan.plan_bytes =
        short_row_ids.size()   * sizeof(int) +
        short_starts.size()    * sizeof(int) +
        short_block_nnz.size() * sizeof(int) +
        long_row_ids.size()    * sizeof(int) +
        long_starts.size()     * sizeof(int) +
        long_block_nnz.size()  * sizeof(int) +
        long_sub_starts.size() * sizeof(int) +
        long_sub_counts.size() * sizeof(int) +
        long_sub_row_map.size() * sizeof(int) +
        res_row_ids.size()     * sizeof(int) +
        res_starts.size()      * sizeof(int) +
        res_lengths.size()     * sizeof(int);
}

// ============================================================================
// run_ra_rode_enhanced_plan
//
// Execution order (all on default stream, sequentially dependent):
//   1. cudaMemset C to zero
//   2. Short-row kernel   (regular_nnz < 128): writes C[row] directly
//   3. Long-row kernel    (regular_nnz >= 128): writes C[row] directly
//   4. Residual kernel    (0-31 nnz tail):     adds into C[row]
// ============================================================================
void run_ra_rode_enhanced_plan(
    const RARodeEnhancedPlan& plan,
    const int*   d_colind,
    const float* d_vals,
    const float* d_B,
    float*       d_C,
    int N)
{
    if (plan.M <= 0 || N <= 0) return;

    // Zero the output matrix
    CUDA_CHECK_NEXT(cudaMemset(d_C, 0, (i64)plan.M * N * sizeof(float)));

    // Alignment checks for vectorized path
    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
    const bool use_vec4  = (N % 4 == 0) && b_aligned && c_aligned;

    // ---- Launch 1: Short-row kernel ----
    if (plan.num_short_rows > 0) {
        constexpr int kShortThreads = 128;  // 4 warps
        if (use_vec4) {
            rode_short_vec4_kernel<<<plan.num_short_rows, kShortThreads>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_short_row_ids,
                plan.d_short_starts,
                plan.d_short_block_nnz,
                plan.num_short_rows,
                N);
        } else {
            rode_short_scalar_kernel<<<plan.num_short_rows, kShortThreads>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_short_row_ids,
                plan.d_short_starts,
                plan.d_short_block_nnz,
                plan.num_short_rows,
                N);
        }
        CUDA_CHECK_KERNEL();
    }

    // ---- Launch 2: Long-row pipelined kernel ----
    if (plan.num_long_rows > 0) {
        if (use_vec4) {
            rode_long_pipelined_vec4_kernel<<<plan.num_long_rows, kLongCTAThreads>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_long_row_ids,
                plan.d_long_starts,
                plan.d_long_block_nnz,
                plan.num_long_rows,
                N);
        } else {
            rode_long_pipelined_scalar_kernel<<<plan.num_long_rows, kLongCTAThreads>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_long_row_ids,
                plan.d_long_starts,
                plan.d_long_block_nnz,
                plan.num_long_rows,
                N);
        }
        CUDA_CHECK_KERNEL();
    }

    // ---- Launch 3: Residual kernel ----
    if (plan.num_residual > 0) {
        constexpr int kResThreads = 128;
        if (use_vec4) {
            rode_residual_vec4_kernel<<<plan.num_residual, kResThreads>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_res_row_ids,
                plan.d_res_starts,
                plan.d_res_lengths,
                plan.num_residual,
                N);
        } else {
            rode_residual_scalar_kernel<<<plan.num_residual, kResThreads>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_res_row_ids,
                plan.d_res_starts,
                plan.d_res_lengths,
                plan.num_residual,
                N);
        }
        CUDA_CHECK_KERNEL();
    }
}

// ============================================================================
// free_ra_rode_enhanced_plan
//
// Releases all GPU allocations and resets plan to default state.
// ============================================================================
void free_ra_rode_enhanced_plan(RARodeEnhancedPlan& plan)
{
    // Short-row descriptors
    if (plan.d_short_row_ids)   { cudaFree(plan.d_short_row_ids);   plan.d_short_row_ids   = nullptr; }
    if (plan.d_short_starts)    { cudaFree(plan.d_short_starts);    plan.d_short_starts    = nullptr; }
    if (plan.d_short_block_nnz) { cudaFree(plan.d_short_block_nnz); plan.d_short_block_nnz = nullptr; }

    // Long-row descriptors
    if (plan.d_long_row_ids)    { cudaFree(plan.d_long_row_ids);    plan.d_long_row_ids    = nullptr; }
    if (plan.d_long_starts)     { cudaFree(plan.d_long_starts);     plan.d_long_starts     = nullptr; }
    if (plan.d_long_block_nnz)  { cudaFree(plan.d_long_block_nnz);  plan.d_long_block_nnz  = nullptr; }

    // Long-row sub-block descriptors
    if (plan.d_long_sub_starts)  { cudaFree(plan.d_long_sub_starts);  plan.d_long_sub_starts  = nullptr; }
    if (plan.d_long_sub_counts)  { cudaFree(plan.d_long_sub_counts);  plan.d_long_sub_counts  = nullptr; }
    if (plan.d_long_sub_row_map) { cudaFree(plan.d_long_sub_row_map); plan.d_long_sub_row_map = nullptr; }

    // Residual descriptors
    if (plan.d_res_row_ids)  { cudaFree(plan.d_res_row_ids);  plan.d_res_row_ids  = nullptr; }
    if (plan.d_res_starts)   { cudaFree(plan.d_res_starts);   plan.d_res_starts   = nullptr; }
    if (plan.d_res_lengths)  { cudaFree(plan.d_res_lengths);  plan.d_res_lengths  = nullptr; }

    // Reset counts and diagnostics
    plan.num_short_rows      = 0;
    plan.num_long_rows       = 0;
    plan.num_long_sub_blocks = 0;
    plan.num_residual        = 0;
    plan.M = 0;
    plan.K = 0;
    plan.regular_nnz_fraction  = 0.f;
    plan.long_row_nnz_fraction = 0.f;
    plan.plan_bytes = 0;
}
