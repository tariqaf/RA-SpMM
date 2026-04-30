// ============================================================================
// ra_segment_hybrid.cu - R7: SEGMENT_HYBRID SpMM for hybrid/mixed regime
//
// Regime: mixedness >= 0.55. Matrices with heterogeneous row structures that
// benefit from neither pure TC nor pure CUDA execution alone.
//
// Design: Row-level partitioning into TC and CUDA paths.
//   - TC path: rows with sufficient column-locality (compactness >= 0.15,
//     nnz >= 4, span <= 256). Uses TC_DIRECT-style WMMA execution.
//   - CUDA path: remaining rows with scattered/power-law patterns.
//     Uses RoDe-style short/long/residual decomposition.
//
// Each row is exclusively assigned to one partition. TC and CUDA paths write
// to disjoint rows, so no atomics are needed.
//
// Target: sm_70+ (Volta WMMA), optimised for sm_86 (RTX 3090).
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------

// TC partition constants (from TC_DIRECT)
constexpr int kGroupRows          = 16;
constexpr int kSignatureBuckets   = 64;
constexpr int kTileElems          = 16 * 16;   // 256 halfs per packed tile
constexpr int kMaxWarpsPerCta     = 16;

// Relaxed FP32 thresholds (same as TC_DIRECT) — TC kernels are correct
constexpr int   kFp32GroupMaxRowNnzThreshold  = 256;
constexpr int   kFp32GroupTotalNnzThreshold   = 2048;
constexpr float kFp32GroupAvgRowNnzThreshold  = 128.f;
constexpr float kMinTcGroupTileDensity        = 0.08f;

// CUDA partition constants (from RODE_ENHANCED)
constexpr int kLongRowThreshold = 128;  // regular_nnz >= 128 => long row
constexpr int kSubBlockSize     = 32;   // nnz per sub-block
constexpr int kLongCTAThreads   = 256;  // 8 warps per CTA for long rows
constexpr int kLongComputeWarps = 7;    // warps 1-7 compute, warp 0 loads

// Row classification thresholds
constexpr int   kTcMinNnz        = 4;
constexpr float kTcMinCompactness = 0.15f;
constexpr int   kTcMaxSpan       = 256;

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

// ---------------------------------------------------------------------------
// Row ordering metadata (same as TC_DIRECT)
// ---------------------------------------------------------------------------
struct RowOrderInfo {
    int row       = 0;
    int len       = 0;
    int min_col   = 0;
    int max_col   = 0;
    float centroid = 0.f;
    uint64_t signature = 0;
};

inline int popcount64(uint64_t x) {
#if defined(__GNUG__)
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) { x &= (x - 1); ++count; }
    return count;
#endif
}

inline float jaccard_u64(uint64_t a, uint64_t b) {
    const int uni = popcount64(a | b);
    if (uni == 0) return 0.f;
    return static_cast<float>(popcount64(a & b)) / static_cast<float>(uni);
}

inline uint16_t float_to_half_bits(float value) {
    const half h = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// ============================================================================
// TC kernel -- one CTA per group, up to 16 warps, single-pass tiles
// (Adapted from ra_tc_direct_kernel)
// ============================================================================
__global__ void ra_segment_hybrid_tc_kernel(
    const int* __restrict__      d_group_offsets,
    const int* __restrict__      d_group_use_fp32,
    const int* __restrict__      d_group_tile_offsets,
    const int* __restrict__      d_group_tile_k_ids,
    const uint16_t* __restrict__ d_group_tile_vals,
    const float* __restrict__    B,
    const int* __restrict__      reordered_to_original,
    float* __restrict__          C_out,
    int M, int K, int N,
    int num_groups)
{
    const int group_id = blockIdx.x;
    if (group_id >= num_groups) return;
    if (d_group_use_fp32[group_id] != 0) return;

    const int group_row_start = d_group_offsets[group_id];
    const int group_row_end   = d_group_offsets[group_id + 1];
    const int local_rows      = group_row_end - group_row_start;

    const int warp_id   = threadIdx.x / 32;
    const int lane      = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    const int num_strips = (N + 15) / 16;

    const int tile_begin = d_group_tile_offsets[group_id];
    const int tile_end   = d_group_tile_offsets[group_id + 1];
    if (tile_begin >= tile_end) return;

    const half* tile_vals_half =
        reinterpret_cast<const half*>(d_group_tile_vals);

    // Dynamic shared memory regions
    extern __shared__ char smem_raw[];
    half*  A_smem      = reinterpret_cast<half*>(smem_raw);
    half*  B_smem_base = A_smem + kTileElems;
    float* C_tmp_base  = reinterpret_cast<float*>(B_smem_base + num_warps * kTileElems);

    for (int strip_base = 0; strip_base < num_strips; strip_base += num_warps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        using namespace nvcuda;

        const int strip = strip_base + warp_id;

        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        if (strip < num_strips) {
            wmma::fill_fragment(c_frag, 0.0f);
        }

        // Single pass over ALL tiles
        for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
            const int kb      = d_group_tile_k_ids[tile_idx];
            const int k_start = kb * 16;

            // Cooperative A tile load
            const half* src = tile_vals_half + static_cast<i64>(tile_idx) * kTileElems;
            for (int i = threadIdx.x; i < kTileElems; i += blockDim.x) {
                A_smem[i] = src[i];
            }
            __syncthreads();

            if (strip < num_strips) {
                const int n_start = strip * 16;
                half* B_smem = B_smem_base + warp_id * kTileElems;

                // Load B tile: float32 -> half
                for (int i = lane; i < kTileElems; i += 32) {
                    const int lc = i / 16;
                    const int ln = i % 16;
                    const int gc = k_start + lc;
                    const int gn = n_start + ln;
                    const float val = (gc < K && gn < N)
                        ? __ldg(&B[static_cast<i64>(gc) * N + gn])
                        : 0.f;
                    B_smem[i] = __float2half(val);
                }
                __syncwarp();

                // WMMA: A(16x16) x B(16x16) -> accumulate
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, A_smem, 16);
                wmma::load_matrix_sync(b_frag, B_smem, 16);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        // Store c_frag -> scatter to C_out via perm_inv
        if (strip < num_strips) {
            const int n_start = strip * 16;
            float* C_tmp = C_tmp_base + warp_id * kTileElems;

            wmma::store_matrix_sync(C_tmp, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            for (int i = lane; i < kTileElems; i += 32) {
                const int lr = i / 16;
                const int ln = i % 16;
                const int n  = n_start + ln;
                if (lr < local_rows && n < N) {
                    const int reordered_row = group_row_start + lr;
                    const int original_row  = reordered_to_original[reordered_row];
                    if (original_row < M) {
                        C_out[static_cast<i64>(original_row) * N + n] = C_tmp[i];
                    }
                }
            }
        }
        __syncthreads();
#endif  // __CUDA_ARCH__ >= 700
    }
}

// ============================================================================
// TC FP32 fallback kernels (adapted from TC_DIRECT)
// ============================================================================

__global__ void segment_hybrid_tc_fp32_kernel(
    const int* __restrict__   d_fp32_rows,
    const int* __restrict__   d_row_ptr_r,
    const int* __restrict__   d_col_r,
    const float* __restrict__ d_val_r,
    const float* __restrict__ B,
    const int* __restrict__   reordered_to_original,
    float* __restrict__       C_out,
    int M, int N,
    int num_fp32_rows)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    if (warp_id >= num_fp32_rows) return;

    const int reordered_row = d_fp32_rows[warp_id];
    if (reordered_row >= M) return;
    const int original_row = reordered_to_original[reordered_row];

    const int start = d_row_ptr_r[reordered_row];
    const int end   = d_row_ptr_r[reordered_row + 1];

    for (int n = lane; n < N; n += 32) {
        float acc = 0.f;
        for (int p = start; p < end; ++p) {
            acc += d_val_r[p] * __ldg(&B[static_cast<i64>(d_col_r[p]) * N + n]);
        }
        C_out[static_cast<i64>(original_row) * N + n] = acc;
    }
}

__global__ void segment_hybrid_tc_fp32_kernel_vec4(
    const int* __restrict__   d_fp32_rows,
    const int* __restrict__   d_row_ptr_r,
    const int* __restrict__   d_col_r,
    const float* __restrict__ d_val_r,
    const float* __restrict__ B,
    const int* __restrict__   reordered_to_original,
    float* __restrict__       C_out,
    int M, int N,
    int num_fp32_rows)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    if (warp_id >= num_fp32_rows) return;

    const int reordered_row = d_fp32_rows[warp_id];
    if (reordered_row >= M) return;
    const int original_row = reordered_to_original[reordered_row];

    const int start = d_row_ptr_r[reordered_row];
    const int end   = d_row_ptr_r[reordered_row + 1];
    const int N4    = N / 4;

    for (int n4 = lane; n4 < N4; n4 += 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = start; p < end; ++p) {
            const int col      = d_col_r[p];
            const float a_val  = d_val_r[p];
            const float4* B_ptr =
                reinterpret_cast<const float4*>(B + static_cast<i64>(col) * N);
            const float4 b4 = B_ptr[n4];
            acc.x += a_val * b4.x;
            acc.y += a_val * b4.y;
            acc.z += a_val * b4.z;
            acc.w += a_val * b4.w;
        }
        float4* C_ptr =
            reinterpret_cast<float4*>(C_out + static_cast<i64>(original_row) * N);
        C_ptr[n4] = acc;
    }
}

// ============================================================================
// CUDA short-row scalar kernel (adapted from RODE_ENHANCED)
//
// One CTA per row. Warps tile N. Processes the 32-aligned prefix.
// Writes directly to original row in C (no reordering needed).
// ============================================================================
__global__ void segment_hybrid_cuda_short_scalar_kernel(
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
// CUDA short-row float4 kernel
// ============================================================================
__global__ void segment_hybrid_cuda_short_vec4_kernel(
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
// CUDA long-row pipelined scalar kernel (adapted from RODE_ENHANCED)
//
// Double-buffered shared memory pipeline: warp 0 loads, warps 1-7 compute.
// ============================================================================
__global__ void segment_hybrid_cuda_long_scalar_kernel(
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

    for (int n_base = 0; n_base < N; n_base += kLongComputeWarps * 32) {
        const int n = n_base + compute_warp * 32 + lane;
        const bool n_valid = (warp_id >= 1) && (n < N);

        float acc = 0.f;
        int cur_buf = 0;

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

        if (n_valid) {
            C[(i64)row * N + n] = acc;
        }
    }
}

// ============================================================================
// CUDA long-row pipelined float4 kernel
// ============================================================================
__global__ void segment_hybrid_cuda_long_vec4_kernel(
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

    __shared__ int   smem_col[2][kSubBlockSize];
    __shared__ float smem_val[2][kSubBlockSize];

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
            if (warp_id == 0 && sb + 1 < num_sub_blocks) {
                const int next_offset = start + (sb + 1) * kSubBlockSize + lane;
                if (lane < kSubBlockSize) {
                    smem_col[1 - cur_buf][lane] = d_col[next_offset];
                    smem_val[1 - cur_buf][lane] = d_val[next_offset];
                }
            }

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
// CUDA residual scalar kernel (0-31 nnz tail per row, additive)
// ============================================================================
__global__ void segment_hybrid_cuda_residual_scalar_kernel(
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
// CUDA residual float4 kernel
// ============================================================================
__global__ void segment_hybrid_cuda_residual_vec4_kernel(
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
// make_ra_segment_hybrid_plan
//
// Partitions rows into TC and CUDA paths based on per-row compactness.
// TC rows are reordered, grouped, and tile-packed (TC_DIRECT style).
// CUDA rows are decomposed into short/long/residual (RODE_ENHANCED style).
// ============================================================================
void make_ra_segment_hybrid_plan(
    RASegmentHybridPlan& plan,
    const int*   h_rowptr,
    const int*   h_col,
    const float* h_val,
    int M, int K, int N)
{
    plan = RASegmentHybridPlan{};
    plan.M = M;
    plan.K = K;

    if (M <= 0 || K <= 0 || N < 1) {
        return;
    }
    const int total_nnz = h_rowptr[M];
    if (total_nnz <= 0) {
        return;
    }

    // No force_all_fp32 guards needed — TC kernels are correct.
    const bool force_all_fp32 = false;

    // =================================================================
    // Step 1: Row classification -- TC vs CUDA
    // =================================================================
    std::vector<int> tc_candidates;
    std::vector<int> cuda_candidates;
    tc_candidates.reserve(M);
    cuda_candidates.reserve(M);

    i64 tc_nnz_total   = 0;
    i64 cuda_nnz_total = 0;

    for (int row = 0; row < M; ++row) {
        const int start = h_rowptr[row];
        const int end   = h_rowptr[row + 1];
        const int nnz_i = end - start;

        if (nnz_i < kTcMinNnz) {
            cuda_candidates.push_back(row);
            cuda_nnz_total += nnz_i;
            continue;
        }

        // Compute span and compactness
        int min_col = h_col[start];
        int max_col = h_col[start];
        for (int p = start + 1; p < end; ++p) {
            if (h_col[p] < min_col) min_col = h_col[p];
            if (h_col[p] > max_col) max_col = h_col[p];
        }
        const int span = max_col - min_col + 1;
        const float compactness = static_cast<float>(nnz_i) / static_cast<float>(span);

        if (compactness >= kTcMinCompactness && span <= kTcMaxSpan) {
            tc_candidates.push_back(row);
            tc_nnz_total += nnz_i;
        } else {
            cuda_candidates.push_back(row);
            cuda_nnz_total += nnz_i;
        }
    }

    plan.num_tc_rows   = static_cast<int>(tc_candidates.size());
    plan.tc_nnz_fraction   = (total_nnz > 0)
        ? static_cast<float>(tc_nnz_total) / static_cast<float>(total_nnz)
        : 0.f;
    plan.cuda_nnz_fraction = (total_nnz > 0)
        ? static_cast<float>(cuda_nnz_total) / static_cast<float>(total_nnz)
        : 0.f;

    // =================================================================
    // Step 2: TC partition processing (TC_DIRECT style)
    // =================================================================
    int num_tc_local = static_cast<int>(tc_candidates.size());

    // Build row metadata for TC candidates
    std::vector<RowOrderInfo> tc_info(num_tc_local);
    for (int idx = 0; idx < num_tc_local; ++idx) {
        const int row = tc_candidates[idx];
        const int start = h_rowptr[row];
        const int end   = h_rowptr[row + 1];
        const int len   = end - start;

        RowOrderInfo& ri = tc_info[idx];
        ri.row     = row;
        ri.len     = len;
        ri.min_col = (len > 0) ? h_col[start]     : 0;
        ri.max_col = (len > 0) ? h_col[end - 1]   : 0;

        double centroid_sum = 0.0;
        for (int p = start; p < end; ++p) {
            centroid_sum += h_col[p];
            const int bucket = std::min(kSignatureBuckets - 1,
                                        (h_col[p] * kSignatureBuckets) / std::max(1, K));
            ri.signature |= (uint64_t{1} << bucket);
        }
        ri.centroid = (len > 0)
            ? static_cast<float>(centroid_sum / static_cast<double>(len))
            : 0.f;
    }

    // Reorder TC rows by centroid bucket -> span -> nnz
    std::vector<int> tc_order(num_tc_local);
    std::iota(tc_order.begin(), tc_order.end(), 0);

    std::stable_sort(tc_order.begin(), tc_order.end(), [&](int a, int b) {
        const int bucket_a = static_cast<int>(
            tc_info[a].centroid * 16.f / static_cast<float>(std::max(1, K)));
        const int bucket_b = static_cast<int>(
            tc_info[b].centroid * 16.f / static_cast<float>(std::max(1, K)));
        if (bucket_a != bucket_b) return bucket_a < bucket_b;

        const int span_a = (tc_info[a].len > 0)
            ? (tc_info[a].max_col - tc_info[a].min_col + 1) : K;
        const int span_b = (tc_info[b].len > 0)
            ? (tc_info[b].max_col - tc_info[b].min_col + 1) : K;
        if (span_a != span_b) return span_a < span_b;

        return tc_info[a].len > tc_info[b].len;
    });

    // Within each 16-row group, sort by nnz descending
    for (int base = 0; base < num_tc_local; base += kGroupRows) {
        const int end = std::min(num_tc_local, base + kGroupRows);
        std::stable_sort(tc_order.begin() + base, tc_order.begin() + end,
                         [&](int a, int b) { return tc_info[a].len > tc_info[b].len; });
    }

    // Build perm_inv: tc_local_row -> original_row (for scatter in TC kernel)
    std::vector<int> tc_perm_inv(num_tc_local);
    for (int tc_row = 0; tc_row < num_tc_local; ++tc_row) {
        tc_perm_inv[tc_row] = tc_info[tc_order[tc_row]].row;
    }

    // Build reordered CSR for TC partition (needed for FP32 fallback)
    i64 tc_total_nnz = static_cast<i64>(tc_nnz_total);
    std::vector<int>   tc_rowptr(num_tc_local + 1, 0);
    std::vector<int>   tc_col;
    std::vector<float> tc_val;
    tc_col.reserve(tc_total_nnz);
    tc_val.reserve(tc_total_nnz);

    for (int tc_row = 0; tc_row < num_tc_local; ++tc_row) {
        const int original_row = tc_perm_inv[tc_row];
        std::vector<std::pair<int, float>> entries;
        entries.reserve(h_rowptr[original_row + 1] - h_rowptr[original_row]);
        for (int p = h_rowptr[original_row]; p < h_rowptr[original_row + 1]; ++p) {
            entries.push_back({h_col[p], h_val[p]});
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& entry : entries) {
            tc_col.push_back(entry.first);
            tc_val.push_back(entry.second);
        }
        tc_rowptr[tc_row + 1] = static_cast<int>(tc_col.size());
    }

    // Group construction, FP32 gating, tile packing
    std::vector<int> group_offsets;
    group_offsets.reserve((num_tc_local + kGroupRows - 1) / kGroupRows + 1);
    group_offsets.push_back(0);

    std::vector<int> group_use_fp32;
    std::vector<int> group_tile_offsets;
    group_tile_offsets.push_back(0);

    std::vector<int>      group_tile_k_ids;
    std::vector<uint16_t> group_tile_vals;
    std::vector<int>      fp32_rows;

    int fp32_groups = 0;

    for (int base = 0; base < num_tc_local; base += kGroupRows) {
        const int end = std::min(num_tc_local, base + kGroupRows);
        group_offsets.push_back(end);

        int64_t group_nnz = 0;
        int group_max_row_nnz = 0;

        for (int idx = base; idx < end; ++idx) {
            const RowOrderInfo& ri = tc_info[tc_order[idx]];
            group_nnz += ri.len;
            group_max_row_nnz = std::max(group_max_row_nnz, ri.len);
        }

        const float avg_row_nnz =
            static_cast<float>(group_nnz) / static_cast<float>(std::max(1, end - base));

        // FP32 gating: conservative thresholds + force_all_fp32 guard
        bool use_fp32_group = force_all_fp32 ||
            (group_max_row_nnz >= kFp32GroupMaxRowNnzThreshold) ||
            (group_nnz >= kFp32GroupTotalNnzThreshold) ||
            (avg_row_nnz >= kFp32GroupAvgRowNnzThreshold);

        if (!use_fp32_group) {
            // Pack nonzero entries into 16x16 tiles keyed by k-block
            std::map<int, std::array<float, kTileElems>> tile_map;
            for (int idx = base; idx < end; ++idx) {
                const int original_row = tc_perm_inv[idx];
                const int local_row    = idx - base;
                for (int p = h_rowptr[original_row]; p < h_rowptr[original_row + 1]; ++p) {
                    const int col = h_col[p];
                    const int kb  = col / 16;
                    auto it = tile_map.find(kb);
                    if (it == tile_map.end()) {
                        it = tile_map.emplace(kb, std::array<float, kTileElems>{}).first;
                    }
                    it->second[local_row * 16 + (col % 16)] = h_val[p];
                }
            }

            if (!tile_map.empty()) {
                const float tile_density = static_cast<float>(group_nnz) /
                    static_cast<float>(tile_map.size() * kTileElems);

                if (tile_density < kMinTcGroupTileDensity) {
                    use_fp32_group = true;
                } else {
                    for (const auto& entry : tile_map) {
                        group_tile_k_ids.push_back(entry.first);
                        for (float value : entry.second) {
                            group_tile_vals.push_back(float_to_half_bits(value));
                        }
                    }
                }
            } else {
                use_fp32_group = true;
            }
        }

        group_use_fp32.push_back(use_fp32_group ? 1 : 0);
        fp32_groups += use_fp32_group ? 1 : 0;
        if (use_fp32_group) {
            for (int reordered_row = base; reordered_row < end; ++reordered_row) {
                fp32_rows.push_back(reordered_row);
            }
        }
        group_tile_offsets.push_back(static_cast<int>(group_tile_k_ids.size()));
    }

    plan.num_tc_groups    = static_cast<int>(group_offsets.size()) - 1;
    plan.num_tc_tiles     = static_cast<int>(group_tile_k_ids.size());
    plan.num_tc_fp32_rows = static_cast<int>(fp32_rows.size());

    // Upload TC partition to GPU
    if (num_tc_local > 0) {
        plan.d_tc_perm_inv = ra_upload(tc_perm_inv);

        CUDA_CHECK_NEXT(cudaMalloc(&plan.d_tc_row_ptr_r, (num_tc_local + 1) * sizeof(int)));
        CUDA_CHECK_NEXT(cudaMemcpy(plan.d_tc_row_ptr_r, tc_rowptr.data(),
                                   (num_tc_local + 1) * sizeof(int), cudaMemcpyHostToDevice));

        if (!tc_col.empty()) {
            CUDA_CHECK_NEXT(cudaMalloc(&plan.d_tc_col_r, tc_col.size() * sizeof(int)));
            CUDA_CHECK_NEXT(cudaMemcpy(plan.d_tc_col_r, tc_col.data(),
                                       tc_col.size() * sizeof(int), cudaMemcpyHostToDevice));

            CUDA_CHECK_NEXT(cudaMalloc(&plan.d_tc_val_r, tc_val.size() * sizeof(float)));
            CUDA_CHECK_NEXT(cudaMemcpy(plan.d_tc_val_r, tc_val.data(),
                                       tc_val.size() * sizeof(float), cudaMemcpyHostToDevice));
        }

        plan.d_tc_group_offsets   = ra_upload(group_offsets);
        plan.d_tc_group_use_fp32  = ra_upload(group_use_fp32);
        plan.d_tc_tile_offsets    = ra_upload(group_tile_offsets);
        plan.d_tc_tile_k_ids     = ra_upload(group_tile_k_ids);
        plan.d_tc_tile_vals       = ra_upload(group_tile_vals);
        plan.d_tc_fp32_rows       = ra_upload(fp32_rows);
    }

    // =================================================================
    // Step 3: CUDA partition processing (RODE_ENHANCED style)
    // =================================================================
    std::vector<int> cuda_short_row_ids, cuda_short_starts, cuda_short_block_nnz;
    std::vector<int> cuda_long_row_ids,  cuda_long_starts,  cuda_long_block_nnz;
    std::vector<int> cuda_res_row_ids,   cuda_res_starts,   cuda_res_lengths;

    for (int row : cuda_candidates) {
        const int start = h_rowptr[row];
        const int nnz_i = h_rowptr[row + 1] - start;
        const int regular_nnz  = (nnz_i / kSubBlockSize) * kSubBlockSize;
        const int residual_nnz = nnz_i - regular_nnz;

        if (regular_nnz > 0) {
            if (regular_nnz >= kLongRowThreshold) {
                cuda_long_row_ids.push_back(row);
                cuda_long_starts.push_back(start);
                cuda_long_block_nnz.push_back(regular_nnz);
            } else {
                cuda_short_row_ids.push_back(row);
                cuda_short_starts.push_back(start);
                cuda_short_block_nnz.push_back(regular_nnz);
            }
        }

        if (residual_nnz > 0) {
            cuda_res_row_ids.push_back(row);
            cuda_res_starts.push_back(start + regular_nnz);
            cuda_res_lengths.push_back(residual_nnz);
        }
    }

    plan.num_cuda_short_rows = static_cast<int>(cuda_short_row_ids.size());
    plan.num_cuda_long_rows  = static_cast<int>(cuda_long_row_ids.size());
    plan.num_cuda_residual   = static_cast<int>(cuda_res_row_ids.size());

    // Upload CUDA partition to GPU
    plan.d_cuda_short_row_ids   = ra_upload(cuda_short_row_ids);
    plan.d_cuda_short_starts    = ra_upload(cuda_short_starts);
    plan.d_cuda_short_block_nnz = ra_upload(cuda_short_block_nnz);

    plan.d_cuda_long_row_ids    = ra_upload(cuda_long_row_ids);
    plan.d_cuda_long_starts     = ra_upload(cuda_long_starts);
    plan.d_cuda_long_block_nnz  = ra_upload(cuda_long_block_nnz);

    plan.d_cuda_res_row_ids     = ra_upload(cuda_res_row_ids);
    plan.d_cuda_res_starts      = ra_upload(cuda_res_starts);
    plan.d_cuda_res_lengths     = ra_upload(cuda_res_lengths);

    // =================================================================
    // Step 4: Plan memory accounting and activation
    // =================================================================
    plan.plan_bytes =
        // TC partition
        static_cast<size_t>(num_tc_local)             * sizeof(int)      +  // d_tc_perm_inv
        static_cast<size_t>(num_tc_local + 1)         * sizeof(int)      +  // d_tc_row_ptr_r
        static_cast<size_t>(tc_col.size())            * sizeof(int)      +  // d_tc_col_r
        static_cast<size_t>(tc_val.size())            * sizeof(float)    +  // d_tc_val_r
        static_cast<size_t>(group_offsets.size())      * sizeof(int)      +  // d_tc_group_offsets
        static_cast<size_t>(group_use_fp32.size())    * sizeof(int)      +  // d_tc_group_use_fp32
        static_cast<size_t>(group_tile_offsets.size()) * sizeof(int)      +  // d_tc_tile_offsets
        static_cast<size_t>(group_tile_k_ids.size())  * sizeof(int)      +  // d_tc_tile_k_ids
        static_cast<size_t>(group_tile_vals.size())   * sizeof(uint16_t) +  // d_tc_tile_vals
        static_cast<size_t>(fp32_rows.size())         * sizeof(int)      +  // d_tc_fp32_rows
        // CUDA partition
        static_cast<size_t>(cuda_short_row_ids.size())   * sizeof(int) +
        static_cast<size_t>(cuda_short_starts.size())    * sizeof(int) +
        static_cast<size_t>(cuda_short_block_nnz.size()) * sizeof(int) +
        static_cast<size_t>(cuda_long_row_ids.size())    * sizeof(int) +
        static_cast<size_t>(cuda_long_starts.size())     * sizeof(int) +
        static_cast<size_t>(cuda_long_block_nnz.size())  * sizeof(int) +
        static_cast<size_t>(cuda_res_row_ids.size())     * sizeof(int) +
        static_cast<size_t>(cuda_res_starts.size())      * sizeof(int) +
        static_cast<size_t>(cuda_res_lengths.size())     * sizeof(int);

    plan.active = true;
}

// ============================================================================
// run_ra_segment_hybrid_plan
//
// Execution order (all sequential, no inter-partition data dependencies):
//   1. cudaMemset C to zero (all M*N)
//   2. TC kernel on TC partition (WMMA groups)
//   3. TC FP32 fallback on gated TC groups
//   4. CUDA short-row kernel
//   5. CUDA long-row pipelined kernel
//   6. CUDA residual kernel (additive)
//
// TC and CUDA partitions write to DISJOINT rows -- no atomics needed.
// ============================================================================
void run_ra_segment_hybrid_plan(
    const RASegmentHybridPlan& plan,
    const int*   d_colind,
    const float* d_vals,
    const float* d_B,
    float*       d_C,
    int N,
    cudaStream_t stream)
{
    if (!plan.active || plan.M <= 0 || N <= 0) {
        return;
    }

    // Zero entire output matrix
    CUDA_CHECK_NEXT(cudaMemsetAsync(d_C, 0, static_cast<i64>(plan.M) * N * sizeof(float), stream));

    // Alignment checks for vectorized path
    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
    const bool use_vec4  = (N % 4 == 0) && b_aligned && c_aligned;

    // ---- Launch 1: TC kernel on TC partition ----
    if (plan.num_tc_groups > 0) {
        const int warps_per_cta = std::max(1, std::min(kMaxWarpsPerCta, (N + 15) / 16));
        const int tc_threads    = warps_per_cta * 32;

        const int smem_bytes =
            kTileElems * static_cast<int>(sizeof(half)) +
            warps_per_cta * kTileElems * static_cast<int>(sizeof(half)) +
            warps_per_cta * kTileElems * static_cast<int>(sizeof(float));

        ra_segment_hybrid_tc_kernel<<<plan.num_tc_groups, tc_threads, smem_bytes, stream>>>(
            plan.d_tc_group_offsets,
            plan.d_tc_group_use_fp32,
            plan.d_tc_tile_offsets,
            plan.d_tc_tile_k_ids,
            plan.d_tc_tile_vals,
            d_B,
            plan.d_tc_perm_inv,
            d_C,
            plan.M,
            plan.K,
            N,
            plan.num_tc_groups);
        CUDA_CHECK_KERNEL();
    }

    // ---- Launch 2: TC FP32 fallback ----
    if (plan.num_tc_fp32_rows > 0) {
        const int fp32_threads = 4 * 32;   // 4 warps per block
        const int fp32_blocks =
            (plan.num_tc_fp32_rows + (fp32_threads / 32) - 1) / (fp32_threads / 32);

        if (use_vec4) {
            segment_hybrid_tc_fp32_kernel_vec4<<<fp32_blocks, fp32_threads, 0, stream>>>(
                plan.d_tc_fp32_rows,
                plan.d_tc_row_ptr_r,
                plan.d_tc_col_r,
                plan.d_tc_val_r,
                d_B,
                plan.d_tc_perm_inv,
                d_C,
                plan.M,
                N,
                plan.num_tc_fp32_rows);
        } else {
            segment_hybrid_tc_fp32_kernel<<<fp32_blocks, fp32_threads, 0, stream>>>(
                plan.d_tc_fp32_rows,
                plan.d_tc_row_ptr_r,
                plan.d_tc_col_r,
                plan.d_tc_val_r,
                d_B,
                plan.d_tc_perm_inv,
                d_C,
                plan.M,
                N,
                plan.num_tc_fp32_rows);
        }
        CUDA_CHECK_KERNEL();
    }

    // ---- Launch 3: CUDA short-row kernel ----
    if (plan.num_cuda_short_rows > 0) {
        constexpr int kShortThreads = 128;  // 4 warps
        if (use_vec4) {
            segment_hybrid_cuda_short_vec4_kernel<<<plan.num_cuda_short_rows, kShortThreads, 0, stream>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_cuda_short_row_ids,
                plan.d_cuda_short_starts,
                plan.d_cuda_short_block_nnz,
                plan.num_cuda_short_rows,
                N);
        } else {
            segment_hybrid_cuda_short_scalar_kernel<<<plan.num_cuda_short_rows, kShortThreads, 0, stream>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_cuda_short_row_ids,
                plan.d_cuda_short_starts,
                plan.d_cuda_short_block_nnz,
                plan.num_cuda_short_rows,
                N);
        }
        CUDA_CHECK_KERNEL();
    }

    // ---- Launch 4: CUDA long-row pipelined kernel ----
    if (plan.num_cuda_long_rows > 0) {
        if (use_vec4) {
            segment_hybrid_cuda_long_vec4_kernel<<<plan.num_cuda_long_rows, kLongCTAThreads, 0, stream>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_cuda_long_row_ids,
                plan.d_cuda_long_starts,
                plan.d_cuda_long_block_nnz,
                plan.num_cuda_long_rows,
                N);
        } else {
            segment_hybrid_cuda_long_scalar_kernel<<<plan.num_cuda_long_rows, kLongCTAThreads, 0, stream>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_cuda_long_row_ids,
                plan.d_cuda_long_starts,
                plan.d_cuda_long_block_nnz,
                plan.num_cuda_long_rows,
                N);
        }
        CUDA_CHECK_KERNEL();
    }

    // ---- Launch 5: CUDA residual kernel (additive) ----
    if (plan.num_cuda_residual > 0) {
        constexpr int kResThreads = 128;
        if (use_vec4) {
            segment_hybrid_cuda_residual_vec4_kernel<<<plan.num_cuda_residual, kResThreads, 0, stream>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_cuda_res_row_ids,
                plan.d_cuda_res_starts,
                plan.d_cuda_res_lengths,
                plan.num_cuda_residual,
                N);
        } else {
            segment_hybrid_cuda_residual_scalar_kernel<<<plan.num_cuda_residual, kResThreads, 0, stream>>>(
                d_colind, d_vals, d_B, d_C,
                plan.d_cuda_res_row_ids,
                plan.d_cuda_res_starts,
                plan.d_cuda_res_lengths,
                plan.num_cuda_residual,
                N);
        }
        CUDA_CHECK_KERNEL();
    }
}

// ============================================================================
// free_ra_segment_hybrid_plan
//
// Releases all GPU allocations and resets plan to default state.
// ============================================================================
void free_ra_segment_hybrid_plan(RASegmentHybridPlan& plan)
{
    auto safe_free = [](auto*& ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    };

    // TC partition
    safe_free(plan.d_tc_perm_inv);
    safe_free(plan.d_tc_row_ptr_r);
    safe_free(plan.d_tc_col_r);
    safe_free(plan.d_tc_val_r);
    safe_free(plan.d_tc_group_offsets);
    safe_free(plan.d_tc_group_use_fp32);
    safe_free(plan.d_tc_tile_offsets);
    safe_free(plan.d_tc_tile_k_ids);
    safe_free(plan.d_tc_tile_vals);
    safe_free(plan.d_tc_fp32_rows);

    // Row classification (not separately allocated in this plan, but safe)
    safe_free(plan.d_tc_row_ids);
    safe_free(plan.d_cuda_short_ids);
    safe_free(plan.d_cuda_long_ids);

    // CUDA partition
    safe_free(plan.d_cuda_short_row_ids);
    safe_free(plan.d_cuda_short_starts);
    safe_free(plan.d_cuda_short_block_nnz);
    safe_free(plan.d_cuda_long_row_ids);
    safe_free(plan.d_cuda_long_starts);
    safe_free(plan.d_cuda_long_block_nnz);
    safe_free(plan.d_cuda_res_row_ids);
    safe_free(plan.d_cuda_res_starts);
    safe_free(plan.d_cuda_res_lengths);

    // Reset counts and diagnostics
    plan.num_tc_rows         = 0;
    plan.num_cuda_short      = 0;
    plan.num_cuda_long       = 0;
    plan.num_tc_groups       = 0;
    plan.num_tc_tiles        = 0;
    plan.num_tc_fp32_rows    = 0;
    plan.num_cuda_short_rows = 0;
    plan.num_cuda_long_rows  = 0;
    plan.num_cuda_residual   = 0;
    plan.tc_nnz_fraction     = 0.f;
    plan.cuda_nnz_fraction   = 0.f;
    plan.M = 0;
    plan.K = 0;
    plan.plan_bytes = 0;
    plan.active = false;
}
