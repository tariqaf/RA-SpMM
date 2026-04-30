// ============================================================================
// hybrid_tc_cuda.cu - HYBRID_TC_CUDA SpMM using HybridPlan
//
// This is a simple window-based hybrid path:
// - Row windows are scored with row length, local overlap, compactness, and a
//   lightweight local tile-density proxy.
// - Entire windows are assigned to either the TC partition or the CUDA partition.
// - The two partitions own disjoint output rows, so the execution path avoids
//   atomics in the common case.
// - Windows with clear FP16 accumulation risk are conservatively routed to the
//   CUDA partition because the current TC path still materializes half inputs.
//
// Still missing relative to HC / RSH / Libra-style hybrid execution:
// - 2D partition refinement
// - more adaptive selector logic
// - TC tile compression / sparse double buffering
// - finer-grained cross-window load balancing
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
#include <vector>

namespace {

constexpr int kWindowRows = 16;
constexpr int kSignatureBuckets = 64;
constexpr int kTileElems = 16 * 16;
constexpr int kMaxWarpsPerCta = 8;
constexpr int kCudaLongRowSegmentThreshold = 4;
constexpr int kCudaLongTileCols = 256;
constexpr int kCudaLongTileVec4 = kCudaLongTileCols / 4;
constexpr int kFp32WindowMaxRowNnzThreshold = 96;
constexpr int kFp32WindowTotalNnzThreshold = 512;
constexpr float kFp32WindowAvgRowNnzThreshold = 32.f;
constexpr float kLowDegreeTcTileDensityThreshold = 0.14f;
constexpr float kLowDegreeTcSimilarityThreshold = 0.55f;
constexpr float kLowDegreeTcCompactnessThreshold = 0.10f;
constexpr float kStrongTcScoreThreshold = 0.60f;
constexpr float kStrongTcSimilarityThreshold = 0.55f;
constexpr float kStrongTcCompactnessThreshold = 0.18f;
constexpr float kStrongTcTileDensityThreshold = 0.18f;
constexpr float kStrongTcAvgRowLenThreshold = 8.f;
constexpr int kStrongTcMaxRowNnzThreshold = 160;

template <typename T>
T* upload_hyb(const std::vector<T>& values) {
    if (values.empty()) {
        return nullptr;
    }
    T* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, values.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, values.data(),
                               values.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
    return d_ptr;
}

inline int popcount64(uint64_t x) {
#if defined(__GNUG__)
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        x &= (x - 1);
        ++count;
    }
    return count;
#endif
}

inline float jaccard_u64(uint64_t a, uint64_t b) {
    const int uni = popcount64(a | b);
    if (uni == 0) {
        return 0.f;
    }
    return static_cast<float>(popcount64(a & b)) / static_cast<float>(uni);
}

inline float clamp01(float x) {
    return std::max(0.f, std::min(1.f, x));
}

inline uint16_t float_to_half_bits_hybrid(float value) {
    const half h = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

__device__ __forceinline__ float load_readonly_f32(const float* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

__global__ void hybrid_tc_subkernel(
    const int* __restrict__ d_tc_group_offsets,
    const int* __restrict__ d_tc_group_tile_offsets,
    const int* __restrict__ d_tc_group_tile_k_ids,
    const uint16_t* __restrict__ d_tc_group_tile_vals,
    const int* __restrict__ d_tc_row_ids,
    const float* __restrict__ B,
    float* __restrict__ C,
    int num_tc_groups,
    int K,
    int N)
{
    const int group_id = blockIdx.x;
    if (group_id >= num_tc_groups) {
        return;
    }

    const int row_start = d_tc_group_offsets[group_id];
    const int row_end = d_tc_group_offsets[group_id + 1];
    const int local_rows = row_end - row_start;
    const int tile_begin = d_tc_group_tile_offsets[group_id];
    const int tile_end = d_tc_group_tile_offsets[group_id + 1];
    if (tile_begin >= tile_end) {
        return;
    }

    const int warp_id_in_cta = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    const int num_strips = (N + 15) / 16;

    __shared__ half A_smem[kTileElems];
    __shared__ half B_smem_all[kTileElems * kMaxWarpsPerCta];
    __shared__ float C_tile_all[kTileElems * kMaxWarpsPerCta];
    const half* d_tc_group_tile_vals_half =
        reinterpret_cast<const half*>(d_tc_group_tile_vals);

    for (int strip_base = 0; strip_base < num_strips; strip_base += num_warps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        using namespace nvcuda;

        const int strip = strip_base + warp_id_in_cta;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        if (strip < num_strips) {
            wmma::fill_fragment(c_frag, 0.0f);
        }

        for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
            const int kb = d_tc_group_tile_k_ids[tile_idx];
            const int k_start = kb * 16;
            const half* tile_ptr =
                d_tc_group_tile_vals_half + static_cast<i64>(tile_idx) * kTileElems;
            for (int i = threadIdx.x; i < kTileElems; i += blockDim.x) {
                A_smem[i] = tile_ptr[i];
            }
            __syncthreads();

            if (strip < num_strips) {
                const int n_start = strip * 16;
                half* B_smem = B_smem_all + warp_id_in_cta * kTileElems;
                for (int i = lane; i < kTileElems; i += 32) {
                    const int lc = i / 16;
                    const int ln = i % 16;
                    const int gc = k_start + lc;
                    const int gn = n_start + ln;
                    const float val = (gc < K && gn < N)
                        ? load_readonly_f32(B + (i64)gc * N + gn)
                        : 0.f;
                    B_smem[i] = __float2half(val);
                }
                __syncwarp();

                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, A_smem, 16);
                wmma::load_matrix_sync(b_frag, B_smem, 16);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        if (strip < num_strips) {
            const int n_start = strip * 16;
            float* C_tile = C_tile_all + warp_id_in_cta * kTileElems;
            wmma::store_matrix_sync(C_tile, c_frag, 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < kTileElems; i += 32) {
                const int lr = i / 16;
                const int ln = i % 16;
                if (lr < local_rows) {
                    const int original_row = d_tc_row_ids[row_start + lr];
                    const int n = n_start + ln;
                    if (n < N) {
                        C[(i64)original_row * N + n] = C_tile[i];
                    }
                }
            }
        }
        __syncthreads();
#endif
    }
}

__global__ void hybrid_cuda_subkernel(
    const int* __restrict__ d_cuda_col,
    const float* __restrict__ d_cuda_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) {
        return;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;

    const int row = row_ids[row_idx];
    const int start = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n = warp * 32 + lane; n < N; n += warps_per_block * 32) {
        float acc = 0.f;
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                acc += d_cuda_val[p] * load_readonly_f32(B + (i64)d_cuda_col[p] * N + n);
            }
        }
        C[(i64)row * N + n] = acc;
    }
}

__global__ void hybrid_cuda_short_vec4_kernel(
    const int* __restrict__ d_cuda_col,
    const float* __restrict__ d_cuda_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) {
        return;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int N4 = N / 4;

    const int row = row_ids[row_idx];
    const int start = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n4 = warp * 32 + lane; n4 < N4; n4 += warps_per_block * 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                const float4* b_ptr = reinterpret_cast<const float4*>(B + (i64)d_cuda_col[p] * N);
                const float4 b4 = b_ptr[n4];
                const float a = d_cuda_val[p];
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

__global__ void hybrid_cuda_long_scalar_kernel(
    const int* __restrict__ d_cuda_row_ptr,
    const int* __restrict__ d_cuda_col,
    const float* __restrict__ d_cuda_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) {
        return;
    }

    const int tile_start = blockIdx.y * kCudaLongTileCols;
    const int tile_end = min(tile_start + kCudaLongTileCols, N);
    const int row = row_ids[row_idx];
    const int start = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n = tile_start + threadIdx.x; n < tile_end; n += blockDim.x) {
        float acc = 0.f;
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                acc += d_cuda_val[p] * load_readonly_f32(B + (i64)d_cuda_col[p] * N + n);
            }
        }
        C[(i64)row * N + n] = acc;
    }
}

__global__ void hybrid_cuda_long_vec4_kernel(
    const int* __restrict__ d_cuda_col,
    const float* __restrict__ d_cuda_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) {
        return;
    }

    const int N4 = N / 4;
    const int tile_start4 = blockIdx.y * kCudaLongTileVec4;
    const int tile_end4 = min(tile_start4 + kCudaLongTileVec4, N4);
    const int row = row_ids[row_idx];
    const int start = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n4 = tile_start4 + threadIdx.x; n4 < tile_end4; n4 += blockDim.x) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                const float4* b_ptr = reinterpret_cast<const float4*>(B + (i64)d_cuda_col[p] * N);
                const float4 b4 = b_ptr[n4];
                const float a = d_cuda_val[p];
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

__global__ void hybrid_cuda_residual_scalar_kernel(
    const int* __restrict__ d_cuda_col,
    const float* __restrict__ d_cuda_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ lengths,
    int num_rows,
    int N)
{
    const int residual_idx = blockIdx.x;
    if (residual_idx >= num_rows) {
        return;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;

    const int row = row_ids[residual_idx];
    const int start = starts[residual_idx];
    const int len = lengths[residual_idx];

    for (int n = warp * 32 + lane; n < N; n += warps_per_block * 32) {
        float acc = 0.f;
        for (int p = 0; p < len; ++p) {
            acc += d_cuda_val[start + p] * load_readonly_f32(B + (i64)d_cuda_col[start + p] * N + n);
        }
        C[(i64)row * N + n] += acc;
    }
}

__global__ void hybrid_cuda_residual_vec4_kernel(
    const int* __restrict__ d_cuda_col,
    const float* __restrict__ d_cuda_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ lengths,
    int num_rows,
    int N)
{
    const int residual_idx = blockIdx.x;
    if (residual_idx >= num_rows) {
        return;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int N4 = N / 4;

    const int row = row_ids[residual_idx];
    const int start = starts[residual_idx];
    const int len = lengths[residual_idx];

    for (int n4 = warp * 32 + lane; n4 < N4; n4 += warps_per_block * 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = 0; p < len; ++p) {
            const float4* b_ptr = reinterpret_cast<const float4*>(B + (i64)d_cuda_col[start + p] * N);
            const float4 b4 = b_ptr[n4];
            const float a = d_cuda_val[start + p];
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

}  // namespace

HybridPlan make_hybrid_tc_cuda_plan(
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M,
    int K,
    int N,
    float partition_score_threshold)
{
    HybridPlan plan;
    plan.M = M;
    plan.K = K;
    plan.partition_score_threshold = partition_score_threshold;
    plan.window_size = kWindowRows;

    if (M <= 0 || K <= 0) {
        return plan;
    }

    struct RowInfo {
        int len = 0;
        int min_col = 0;
        int max_col = 0;
        uint64_t signature = 0;
    };

    std::vector<RowInfo> row_info(M);
    for (int row = 0; row < M; ++row) {
        const int start = h_rowptr[row];
        const int end = h_rowptr[row + 1];
        RowInfo info;
        info.len = end - start;
        info.min_col = (info.len > 0) ? h_col[start] : 0;
        info.max_col = (info.len > 0) ? h_col[end - 1] : 0;
        for (int p = start; p < end; ++p) {
            const int bucket = std::min(kSignatureBuckets - 1, (h_col[p] * kSignatureBuckets) / std::max(1, K));
            info.signature |= (uint64_t{1} << bucket);
        }
        row_info[row] = info;
    }

    std::vector<int> tc_rowptr_h(1, 0);
    std::vector<int> tc_col_h;
    std::vector<float> tc_val_h;
    std::vector<int> tc_row_ids_h;
    std::vector<int> tc_group_offsets_h(1, 0);
    std::vector<int> tc_group_tile_offsets_h(1, 0);
    std::vector<int> tc_group_tile_k_ids_h;
    std::vector<uint16_t> tc_group_tile_vals_h;

    std::vector<int> cuda_rowptr_h(1, 0);
    std::vector<int> cuda_col_h;
    std::vector<float> cuda_val_h;
    std::vector<int> cuda_row_ids_h;
    std::vector<int> cuda_short_row_ids_h;
    std::vector<int> cuda_short_starts_h;
    std::vector<int> cuda_short_block_nnz_h;
    std::vector<int> cuda_long_row_ids_h;
    std::vector<int> cuda_long_starts_h;
    std::vector<int> cuda_long_block_nnz_h;
    std::vector<int> cuda_res_row_ids_h;
    std::vector<int> cuda_res_starts_h;
    std::vector<int> cuda_res_lengths_h;

    const int total_nnz = h_rowptr[M];
    int64_t tc_nnz_total = 0;
    int64_t cuda_nnz_total = 0;
    int64_t cuda_regular_nnz_total = 0;
    int64_t cuda_residual_nnz_total = 0;
    int64_t precision_guard_nnz_total = 0;
    float score_sum = 0.f;
    float compactness_sum = 0.f;
    int window_count = 0;
    int precision_guard_rows = 0;

    for (int base = 0; base < M; base += kWindowRows) {
        const int end = std::min(M, base + kWindowRows);
        int64_t window_nnz = 0;
        int min_col = K;
        int max_col = -1;
        int window_max_row_nnz = 0;
        float sim_sum = 0.f;
        int sim_pairs = 0;

        for (int row = base; row < end; ++row) {
            window_nnz += row_info[row].len;
            window_max_row_nnz = std::max(window_max_row_nnz, row_info[row].len);
            if (row_info[row].len > 0) {
                min_col = std::min(min_col, row_info[row].min_col);
                max_col = std::max(max_col, row_info[row].max_col);
            }
            if (row > base && row_info[row - 1].len > 0 && row_info[row].len > 0) {
                sim_sum += jaccard_u64(row_info[row - 1].signature, row_info[row].signature);
                ++sim_pairs;
            }
        }

        const int span = (max_col >= min_col) ? (max_col - min_col + 1) : K;
        const float avg_row_len = static_cast<float>(window_nnz) / static_cast<float>(std::max(1, end - base));
        const float compactness =
            static_cast<float>(window_nnz) /
            static_cast<float>(std::max(1, (end - base) * std::max(1, span)));
        const float similarity = (sim_pairs > 0) ? (sim_sum / static_cast<float>(sim_pairs)) : 0.f;
        const float local_tile_density =
            static_cast<float>(window_nnz) /
            static_cast<float>(std::max(16, (end - base) * std::max(16, ((span + 15) / 16) * 16)));

        const float score =
            0.30f * clamp01(avg_row_len / 64.f) +
            0.30f * similarity +
            0.25f * clamp01(compactness * 8.f) +
            0.15f * clamp01(local_tile_density * 8.f);

        score_sum += score;
        compactness_sum += compactness;
        ++window_count;

        const bool precision_guard_window =
            (window_max_row_nnz >= kFp32WindowMaxRowNnzThreshold) ||
            (window_nnz >= kFp32WindowTotalNnzThreshold) ||
            (avg_row_len >= kFp32WindowAvgRowNnzThreshold);

        const bool low_degree_window = avg_row_len < 12.f;
        const float min_tc_similarity =
            low_degree_window ? kLowDegreeTcSimilarityThreshold : 0.45f;
        const float min_tc_compactness =
            low_degree_window ? kLowDegreeTcCompactnessThreshold : 0.08f;
        const float min_tc_tile_density =
            low_degree_window ? kLowDegreeTcTileDensityThreshold : 0.08f;

        const bool strong_tc_window =
            (N >= 128) &&
            (score >= kStrongTcScoreThreshold) &&
            (similarity >= kStrongTcSimilarityThreshold) &&
            (compactness >= kStrongTcCompactnessThreshold) &&
            (local_tile_density >= kStrongTcTileDensityThreshold) &&
            (avg_row_len >= kStrongTcAvgRowLenThreshold) &&
            (window_max_row_nnz <= kStrongTcMaxRowNnzThreshold);

        const bool use_tc_window =
            (N >= 128) &&
            (score >= std::max(partition_score_threshold, strong_tc_window ? kStrongTcScoreThreshold : 0.75f)) &&
            (similarity >= min_tc_similarity) &&
            (compactness >= min_tc_compactness) &&
            (local_tile_density >= min_tc_tile_density) &&
            (!precision_guard_window || strong_tc_window);

        if (precision_guard_window) {
            ++plan.precision_guard_windows;
            precision_guard_rows += (end - base);
            precision_guard_nnz_total += window_nnz;
        }

        for (int row = base; row < end; ++row) {
            const int start = h_rowptr[row];
            const int stop = h_rowptr[row + 1];
            if (use_tc_window) {
                tc_row_ids_h.push_back(row);
                for (int p = start; p < stop; ++p) {
                    tc_col_h.push_back(h_col[p]);
                    tc_val_h.push_back(h_val[p]);
                }
                tc_nnz_total += (stop - start);
                tc_rowptr_h.push_back(static_cast<int>(tc_nnz_total));
            } else {
                const int compact_start = static_cast<int>(cuda_nnz_total);
                const int row_nnz = stop - start;
                cuda_row_ids_h.push_back(row);
                for (int p = start; p < stop; ++p) {
                    cuda_col_h.push_back(h_col[p]);
                    cuda_val_h.push_back(h_val[p]);
                }
                cuda_nnz_total += row_nnz;
                cuda_rowptr_h.push_back(static_cast<int>(cuda_nnz_total));

                const int block_nnz = (row_nnz / 32) * 32;
                const int residual_nnz = row_nnz - block_nnz;
                const int num_segments = block_nnz / 32;
                cuda_regular_nnz_total += block_nnz;
                cuda_residual_nnz_total += residual_nnz;

                if (block_nnz > 0) {
                    if (num_segments >= kCudaLongRowSegmentThreshold) {
                        cuda_long_row_ids_h.push_back(row);
                        cuda_long_starts_h.push_back(compact_start);
                        cuda_long_block_nnz_h.push_back(block_nnz);
                    } else {
                        cuda_short_row_ids_h.push_back(row);
                        cuda_short_starts_h.push_back(compact_start);
                        cuda_short_block_nnz_h.push_back(block_nnz);
                    }
                }
                if (residual_nnz > 0) {
                    cuda_res_row_ids_h.push_back(row);
                    cuda_res_starts_h.push_back(compact_start + block_nnz);
                    cuda_res_lengths_h.push_back(residual_nnz);
                }
            }
        }
    }

    plan.num_tc_rows = static_cast<int>(tc_row_ids_h.size());
    plan.num_cuda_rows = static_cast<int>(cuda_row_ids_h.size());
    plan.tc_nnz_fraction = (total_nnz > 0) ? static_cast<float>(tc_nnz_total) / static_cast<float>(total_nnz) : 0.f;
    plan.cuda_nnz_fraction = (total_nnz > 0) ? static_cast<float>(cuda_nnz_total) / static_cast<float>(total_nnz) : 0.f;
    plan.tc_row_fraction = (M > 0) ? static_cast<float>(plan.num_tc_rows) / static_cast<float>(M) : 0.f;
    plan.cuda_row_fraction = (M > 0) ? static_cast<float>(plan.num_cuda_rows) / static_cast<float>(M) : 0.f;
    plan.num_cuda_short_rows = static_cast<int>(cuda_short_row_ids_h.size());
    plan.num_cuda_long_rows = static_cast<int>(cuda_long_row_ids_h.size());
    plan.num_cuda_residual = static_cast<int>(cuda_res_row_ids_h.size());
    plan.cuda_regular_nnz_fraction = (cuda_nnz_total > 0)
        ? static_cast<float>(cuda_regular_nnz_total) / static_cast<float>(cuda_nnz_total)
        : 0.f;
    plan.cuda_residual_nnz_fraction = (cuda_nnz_total > 0)
        ? static_cast<float>(cuda_residual_nnz_total) / static_cast<float>(cuda_nnz_total)
        : 0.f;
    plan.average_partition_score = (window_count > 0) ? (score_sum / static_cast<float>(window_count)) : 0.f;
    plan.average_window_compactness = (window_count > 0) ? (compactness_sum / static_cast<float>(window_count)) : 0.f;
    plan.precision_guard_row_fraction = (M > 0)
        ? static_cast<float>(precision_guard_rows) / static_cast<float>(M)
        : 0.f;
    plan.precision_guard_nnz_fraction = (total_nnz > 0)
        ? static_cast<float>(precision_guard_nnz_total) / static_cast<float>(total_nnz)
        : 0.f;

    // ── Sputnik-style row-similarity reordering for the CUDA partition ────────
    // Rows with similar column-bucket signatures are placed adjacent in the
    // launch order so consecutive CTAs access the same B-matrix rows, turning
    // redundant DRAM loads into L2 cache hits.  Only the CUDA partition is
    // reordered; TC partition 16-row groups stay contiguous as required by
    // hybrid_tc_subkernel.
    //
    // Reference: Gale et al., "Sparse GPU Kernels for Deep Learning",
    // NeurIPS 2020 (Sputnik), https://github.com/google-research/sputnik.
    // Specifically: Section 4.2 "Row Swizzle" — reorder rows so that
    // spatially proximate warps (consecutive CTAs) share column-index patterns,
    // maximising L2 reuse of the B matrix.  The original Sputnik implementation
    // uses a software interleave based on a 32-bucket row signature; we use the
    // already-computed 64-bucket Jaccard signature (uint64_t) from the planning
    // phase above to achieve the same effect with no extra computation.
    if (!cuda_row_ids_h.empty()) {
        struct CudaRowMeta {
            int      row_id;
            uint64_t signature;
            int      start;   // element-offset in packed cuda_col_h / cuda_val_h
            int      nnz;
        };

        const int num_cuda = static_cast<int>(cuda_row_ids_h.size());
        std::vector<CudaRowMeta> row_metas;
        row_metas.reserve(num_cuda);
        for (int i = 0; i < num_cuda; ++i) {
            const int row   = cuda_row_ids_h[i];
            const int start = cuda_rowptr_h[i];
            const int nnz   = cuda_rowptr_h[i + 1] - start;
            row_metas.push_back({row, row_info[row].signature, start, nnz});
        }

        // Stable sort by signature groups rows with overlapping column buckets.
        std::stable_sort(row_metas.begin(), row_metas.end(),
                         [](const CudaRowMeta& a, const CudaRowMeta& b) {
                             return a.signature < b.signature;
                         });

        // Rebuild packed NNZ arrays and sub-list classifications in new order.
        std::vector<int>   new_row_ids;      new_row_ids.reserve(num_cuda);
        std::vector<int>   new_col;          new_col.reserve(cuda_col_h.size());
        std::vector<float> new_val;          new_val.reserve(cuda_val_h.size());
        std::vector<int>   new_rowptr;       new_rowptr.reserve(num_cuda + 1);
        new_rowptr.push_back(0);

        std::vector<int> ns_row, ns_start, ns_bnz;   // short
        std::vector<int> nl_row, nl_start, nl_bnz;   // long
        std::vector<int> nr_row, nr_start, nr_len;   // residual
        ns_row.reserve(num_cuda);  ns_start.reserve(num_cuda);  ns_bnz.reserve(num_cuda);
        nl_row.reserve(num_cuda);  nl_start.reserve(num_cuda);  nl_bnz.reserve(num_cuda);
        nr_row.reserve(num_cuda);  nr_start.reserve(num_cuda);  nr_len.reserve(num_cuda);

        int offset = 0;
        for (const auto& m : row_metas) {
            new_row_ids.push_back(m.row_id);
            for (int j = 0; j < m.nnz; ++j) {
                new_col.push_back(cuda_col_h[m.start + j]);
                new_val.push_back(cuda_val_h[m.start + j]);
            }
            new_rowptr.push_back(new_rowptr.back() + m.nnz);

            const int block_nnz   = (m.nnz / 32) * 32;
            const int residual_nnz = m.nnz - block_nnz;
            const int num_segs    = block_nnz / 32;

            if (block_nnz > 0) {
                if (num_segs >= kCudaLongRowSegmentThreshold) {
                    nl_row.push_back(m.row_id);
                    nl_start.push_back(offset);
                    nl_bnz.push_back(block_nnz);
                } else {
                    ns_row.push_back(m.row_id);
                    ns_start.push_back(offset);
                    ns_bnz.push_back(block_nnz);
                }
            }
            if (residual_nnz > 0) {
                nr_row.push_back(m.row_id);
                nr_start.push_back(offset + block_nnz);
                nr_len.push_back(residual_nnz);
            }
            offset += m.nnz;
        }

        // Commit the reordered arrays (counts unchanged; only ordering differs).
        cuda_row_ids_h          = std::move(new_row_ids);
        cuda_col_h              = std::move(new_col);
        cuda_val_h              = std::move(new_val);
        cuda_rowptr_h           = std::move(new_rowptr);
        cuda_short_row_ids_h    = std::move(ns_row);
        cuda_short_starts_h     = std::move(ns_start);
        cuda_short_block_nnz_h  = std::move(ns_bnz);
        cuda_long_row_ids_h     = std::move(nl_row);
        cuda_long_starts_h      = std::move(nl_start);
        cuda_long_block_nnz_h   = std::move(nl_bnz);
        cuda_res_row_ids_h      = std::move(nr_row);
        cuda_res_starts_h       = std::move(nr_start);
        cuda_res_lengths_h      = std::move(nr_len);
    }
    // ─────────────────────────────────────────────────────────────────────────

    if (!tc_row_ids_h.empty()) {
        for (int base = 0; base < static_cast<int>(tc_row_ids_h.size()); base += kWindowRows) {
            const int end = std::min(static_cast<int>(tc_row_ids_h.size()), base + kWindowRows);
            tc_group_offsets_h.push_back(end);

            std::map<int, std::array<float, kTileElems>> tile_map;
            for (int compact_row = base; compact_row < end; ++compact_row) {
                const int local_row = compact_row - base;
                for (int p = tc_rowptr_h[compact_row]; p < tc_rowptr_h[compact_row + 1]; ++p) {
                    const int col = tc_col_h[p];
                    auto it = tile_map.find(col / 16);
                    if (it == tile_map.end()) {
                        it = tile_map.emplace(col / 16, std::array<float, kTileElems>{}).first;
                    }
                    it->second[local_row * 16 + (col % 16)] = tc_val_h[p];
                }
            }

            for (const auto& entry : tile_map) {
                tc_group_tile_k_ids_h.push_back(entry.first);
                for (float value : entry.second) {
                    tc_group_tile_vals_h.push_back(float_to_half_bits_hybrid(value));
                }
            }
            tc_group_tile_offsets_h.push_back(static_cast<int>(tc_group_tile_k_ids_h.size()));
        }
    }

    if (plan.num_tc_rows > 0) {
        plan.d_tc_row_ptr = upload_hyb(tc_rowptr_h);
        plan.d_tc_col = upload_hyb(tc_col_h);
        plan.d_tc_val = upload_hyb(tc_val_h);
        plan.d_tc_row_ids = upload_hyb(tc_row_ids_h);
        plan.d_tc_group_offsets = upload_hyb(tc_group_offsets_h);
        plan.d_tc_group_tile_offsets = upload_hyb(tc_group_tile_offsets_h);
        plan.d_tc_group_tile_k_ids = upload_hyb(tc_group_tile_k_ids_h);
        plan.d_tc_group_tile_vals = upload_hyb(tc_group_tile_vals_h);
        plan.num_tc_groups = static_cast<int>(tc_group_offsets_h.size()) - 1;
        plan.num_tc_tiles = static_cast<int>(tc_group_tile_k_ids_h.size());
    }

    if (plan.num_cuda_rows > 0) {
        plan.d_cuda_row_ptr = upload_hyb(cuda_rowptr_h);
        plan.d_cuda_col = upload_hyb(cuda_col_h);
        plan.d_cuda_val = upload_hyb(cuda_val_h);
        plan.d_cuda_row_ids = upload_hyb(cuda_row_ids_h);
        plan.d_cuda_short_row_ids = upload_hyb(cuda_short_row_ids_h);
        plan.d_cuda_short_starts = upload_hyb(cuda_short_starts_h);
        plan.d_cuda_short_block_nnz = upload_hyb(cuda_short_block_nnz_h);
        plan.d_cuda_long_row_ids = upload_hyb(cuda_long_row_ids_h);
        plan.d_cuda_long_starts = upload_hyb(cuda_long_starts_h);
        plan.d_cuda_long_block_nnz = upload_hyb(cuda_long_block_nnz_h);
        plan.d_cuda_res_row_ids = upload_hyb(cuda_res_row_ids_h);
        plan.d_cuda_res_starts = upload_hyb(cuda_res_starts_h);
        plan.d_cuda_res_lengths = upload_hyb(cuda_res_lengths_h);
    }

    plan.plan_bytes =
        tc_rowptr_h.size() * sizeof(int) +
        tc_col_h.size() * sizeof(int) +
        tc_val_h.size() * sizeof(float) +
        tc_row_ids_h.size() * sizeof(int) +
        tc_group_offsets_h.size() * sizeof(int) +
        tc_group_tile_offsets_h.size() * sizeof(int) +
        tc_group_tile_k_ids_h.size() * sizeof(int) +
        tc_group_tile_vals_h.size() * sizeof(uint16_t) +
        cuda_rowptr_h.size() * sizeof(int) +
        cuda_col_h.size() * sizeof(int) +
        cuda_val_h.size() * sizeof(float) +
        cuda_row_ids_h.size() * sizeof(int) +
        cuda_short_row_ids_h.size() * sizeof(int) +
        cuda_short_starts_h.size() * sizeof(int) +
        cuda_short_block_nnz_h.size() * sizeof(int) +
        cuda_long_row_ids_h.size() * sizeof(int) +
        cuda_long_starts_h.size() * sizeof(int) +
        cuda_long_block_nnz_h.size() * sizeof(int) +
        cuda_res_row_ids_h.size() * sizeof(int) +
        cuda_res_starts_h.size() * sizeof(int) +
        cuda_res_lengths_h.size() * sizeof(int);

    return plan;
}

void run_hybrid_tc_cuda_plan(
    const HybridPlan& plan,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream)
{
    if (plan.M <= 0 || N <= 0) {
        return;
    }

    if (plan.num_tc_rows > 0) {
        const int warps_per_cta = std::max(1, std::min(kMaxWarpsPerCta, (N + 15) / 16));
        const int threads = warps_per_cta * 32;
        hybrid_tc_subkernel<<<plan.num_tc_groups, threads, 0, stream>>>(
            plan.d_tc_group_offsets,
            plan.d_tc_group_tile_offsets,
            plan.d_tc_group_tile_k_ids,
            plan.d_tc_group_tile_vals,
            plan.d_tc_row_ids,
            d_B,
            d_C,
            plan.num_tc_groups,
            plan.K,
            N);
        CUDA_CHECK_KERNEL();
    }

    if (plan.num_cuda_rows > 0) {
        constexpr int kThreads = 128;
        const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
        const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
        const bool use_vec4 = (N % 4 == 0) && b_aligned && c_aligned;

        if (plan.num_cuda_short_rows > 0) {
            if (use_vec4) {
                hybrid_cuda_short_vec4_kernel<<<plan.num_cuda_short_rows, kThreads, 0, stream>>>(
                    plan.d_cuda_col,
                    plan.d_cuda_val,
                    d_B,
                    d_C,
                    plan.d_cuda_short_row_ids,
                    plan.d_cuda_short_starts,
                    plan.d_cuda_short_block_nnz,
                    plan.num_cuda_short_rows,
                    N);
            } else {
                hybrid_cuda_subkernel<<<plan.num_cuda_short_rows, kThreads, 0, stream>>>(
                    plan.d_cuda_col,
                    plan.d_cuda_val,
                    d_B,
                    d_C,
                    plan.d_cuda_short_row_ids,
                    plan.d_cuda_short_starts,
                    plan.d_cuda_short_block_nnz,
                    plan.num_cuda_short_rows,
                    N);
            }
            CUDA_CHECK_KERNEL();
        }

        if (plan.num_cuda_long_rows > 0) {
            const dim3 long_grid(plan.num_cuda_long_rows, (N + kCudaLongTileCols - 1) / kCudaLongTileCols);
            if (use_vec4) {
                hybrid_cuda_long_vec4_kernel<<<long_grid, kThreads, 0, stream>>>(
                    plan.d_cuda_col,
                    plan.d_cuda_val,
                    d_B,
                    d_C,
                    plan.d_cuda_long_row_ids,
                    plan.d_cuda_long_starts,
                    plan.d_cuda_long_block_nnz,
                    plan.num_cuda_long_rows,
                    N);
            } else {
                hybrid_cuda_long_scalar_kernel<<<long_grid, kThreads, 0, stream>>>(
                    plan.d_cuda_row_ptr,
                    plan.d_cuda_col,
                    plan.d_cuda_val,
                    d_B,
                    d_C,
                    plan.d_cuda_long_row_ids,
                    plan.d_cuda_long_starts,
                    plan.d_cuda_long_block_nnz,
                    plan.num_cuda_long_rows,
                    N);
            }
            CUDA_CHECK_KERNEL();
        }

        if (plan.num_cuda_residual > 0) {
            if (use_vec4) {
                hybrid_cuda_residual_vec4_kernel<<<plan.num_cuda_residual, kThreads, 0, stream>>>(
                    plan.d_cuda_col,
                    plan.d_cuda_val,
                    d_B,
                    d_C,
                    plan.d_cuda_res_row_ids,
                    plan.d_cuda_res_starts,
                    plan.d_cuda_res_lengths,
                    plan.num_cuda_residual,
                    N);
            } else {
                hybrid_cuda_residual_scalar_kernel<<<plan.num_cuda_residual, kThreads, 0, stream>>>(
                    plan.d_cuda_col,
                    plan.d_cuda_val,
                    d_B,
                    d_C,
                    plan.d_cuda_res_row_ids,
                    plan.d_cuda_res_starts,
                    plan.d_cuda_res_lengths,
                    plan.num_cuda_residual,
                    N);
            }
            CUDA_CHECK_KERNEL();
        }
    }
}

void free_hybrid_tc_cuda_plan(HybridPlan& plan)
{
    if (plan.d_tc_row_ptr) { cudaFree(plan.d_tc_row_ptr); plan.d_tc_row_ptr = nullptr; }
    if (plan.d_tc_col) { cudaFree(plan.d_tc_col); plan.d_tc_col = nullptr; }
    if (plan.d_tc_val) { cudaFree(plan.d_tc_val); plan.d_tc_val = nullptr; }
    if (plan.d_tc_row_ids) { cudaFree(plan.d_tc_row_ids); plan.d_tc_row_ids = nullptr; }
    if (plan.d_tc_group_offsets) { cudaFree(plan.d_tc_group_offsets); plan.d_tc_group_offsets = nullptr; }
    if (plan.d_tc_group_tile_offsets) { cudaFree(plan.d_tc_group_tile_offsets); plan.d_tc_group_tile_offsets = nullptr; }
    if (plan.d_tc_group_tile_k_ids) { cudaFree(plan.d_tc_group_tile_k_ids); plan.d_tc_group_tile_k_ids = nullptr; }
    if (plan.d_tc_group_tile_vals) { cudaFree(plan.d_tc_group_tile_vals); plan.d_tc_group_tile_vals = nullptr; }
    if (plan.d_cuda_row_ptr) { cudaFree(plan.d_cuda_row_ptr); plan.d_cuda_row_ptr = nullptr; }
    if (plan.d_cuda_col) { cudaFree(plan.d_cuda_col); plan.d_cuda_col = nullptr; }
    if (plan.d_cuda_val) { cudaFree(plan.d_cuda_val); plan.d_cuda_val = nullptr; }
    if (plan.d_cuda_row_ids) { cudaFree(plan.d_cuda_row_ids); plan.d_cuda_row_ids = nullptr; }
    if (plan.d_cuda_short_row_ids) { cudaFree(plan.d_cuda_short_row_ids); plan.d_cuda_short_row_ids = nullptr; }
    if (plan.d_cuda_short_starts) { cudaFree(plan.d_cuda_short_starts); plan.d_cuda_short_starts = nullptr; }
    if (plan.d_cuda_short_block_nnz) { cudaFree(plan.d_cuda_short_block_nnz); plan.d_cuda_short_block_nnz = nullptr; }
    if (plan.d_cuda_long_row_ids) { cudaFree(plan.d_cuda_long_row_ids); plan.d_cuda_long_row_ids = nullptr; }
    if (plan.d_cuda_long_starts) { cudaFree(plan.d_cuda_long_starts); plan.d_cuda_long_starts = nullptr; }
    if (plan.d_cuda_long_block_nnz) { cudaFree(plan.d_cuda_long_block_nnz); plan.d_cuda_long_block_nnz = nullptr; }
    if (plan.d_cuda_res_row_ids) { cudaFree(plan.d_cuda_res_row_ids); plan.d_cuda_res_row_ids = nullptr; }
    if (plan.d_cuda_res_starts) { cudaFree(plan.d_cuda_res_starts); plan.d_cuda_res_starts = nullptr; }
    if (plan.d_cuda_res_lengths) { cudaFree(plan.d_cuda_res_lengths); plan.d_cuda_res_lengths = nullptr; }

    plan.num_tc_rows = 0;
    plan.num_tc_groups = 0;
    plan.num_tc_tiles = 0;
    plan.num_cuda_rows = 0;
    plan.num_cuda_short_rows = 0;
    plan.num_cuda_long_rows = 0;
    plan.num_cuda_residual = 0;
    plan.plan_bytes = 0;
}
