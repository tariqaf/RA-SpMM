// ============================================================================
// tc_reordered.cu - TC_REORDERED SpMM using TCReorderedPlan
//
// This path is an intermediate locality-aware TC import:
// - Rows are grouped by coarse column locality and span compactness.
// - Reordered CSR and the reordered-output workspace are owned by the plan.
// - Warm runs do not allocate or free large buffers.
// - A conservative precision guard keeps long / high-depth groups out of the
//   half-input WMMA path because FP16 tile materialization is not numerically
//   robust enough yet for skewed graphs.
//
// Still missing relative to mature DTC / Acc / FlashSparse-style designs:
// - a richer compressed tile format
// - selector / policy learning for tile activation
// - sparse double buffering
// - shared-memory bypass optimizations
// - reduced sparse granularity like FlashSparse
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

constexpr int kGroupRows = 16;
constexpr int kSignatureBuckets = 64;
constexpr int kTileElems = 16 * 16;
constexpr int kMaxWarpsPerCta = 8;
constexpr int kFp32GroupMaxRowNnzThreshold = 96;
constexpr int kFp32GroupTotalNnzThreshold = 512;
constexpr float kFp32GroupAvgRowNnzThreshold = 32.f;
constexpr float kMinTcGroupTileDensity = 0.12f;

template <typename T>
T* upload_tcr(const std::vector<T>& values) {
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

struct RowOrderInfo {
    int row = 0;
    int len = 0;
    int min_col = 0;
    int max_col = 0;
    float centroid = 0.f;
    uint64_t signature = 0;
};

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


__device__ __forceinline__ float load_readonly_f32(const float* ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

inline uint16_t float_to_half_bits(float value) {
    const half h = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

std::pair<float, float> summarize_group_metrics(
    const std::vector<int>& order,
    const std::vector<RowOrderInfo>& info,
    int K)
{
    if (order.empty()) {
        return {0.f, 0.f};
    }

    double compactness_sum = 0.0;
    double similarity_sum = 0.0;
    int group_count = 0;
    for (int base = 0; base < static_cast<int>(order.size()); base += kGroupRows) {
        const int end = std::min(static_cast<int>(order.size()), base + kGroupRows);
        int min_col = K;
        int max_col = -1;
        int64_t group_nnz = 0;
        float local_similarity = 0.f;
        int similarity_pairs = 0;
        for (int idx = base; idx < end; ++idx) {
            const RowOrderInfo& row_info = info[order[idx]];
            group_nnz += row_info.len;
            if (row_info.len > 0) {
                min_col = std::min(min_col, row_info.min_col);
                max_col = std::max(max_col, row_info.max_col);
            }
            if (idx > base) {
                local_similarity += jaccard_u64(info[order[idx - 1]].signature, row_info.signature);
                ++similarity_pairs;
            }
        }
        const int span = (max_col >= min_col) ? (max_col - min_col + 1) : K;
        compactness_sum += static_cast<double>(group_nnz) /
                           static_cast<double>(std::max(1, (end - base) * std::max(1, span)));
        similarity_sum += (similarity_pairs > 0)
            ? static_cast<double>(local_similarity / static_cast<float>(similarity_pairs))
            : 0.0;
        ++group_count;
    }

    if (group_count == 0) {
        return {0.f, 0.f};
    }
    return {
        static_cast<float>(compactness_sum / static_cast<double>(group_count)),
        static_cast<float>(similarity_sum / static_cast<double>(group_count)),
    };
}

__global__ void tc_reordered_tc_kernel(
    const int* __restrict__ d_group_offsets,
    const int* __restrict__ d_group_use_fp32,
    const int* __restrict__ d_group_tile_offsets,
    const int* __restrict__ d_group_tile_k_ids,
    const uint16_t* __restrict__ d_group_tile_vals,
    const float* __restrict__ B,
    const int* __restrict__ reordered_to_original,
    float* __restrict__ C_out,
    int M,
    int K,
    int N,
    int num_groups)
{
    const int group_id = blockIdx.x;
    if (group_id >= num_groups) {
        return;
    }
    if (d_group_use_fp32[group_id] != 0) {
        return;
    }

    const int group_row_start = d_group_offsets[group_id];
    const int group_row_end = d_group_offsets[group_id + 1];
    const int local_rows = group_row_end - group_row_start;

    const int warp_id_in_cta = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    const int num_strips = (N + 15) / 16;
    const int tile_begin = d_group_tile_offsets[group_id];
    const int tile_end = d_group_tile_offsets[group_id + 1];
    if (tile_begin >= tile_end) {
        return;
    }

    __shared__ half A_smem[kTileElems];
    __shared__ half B_smem_all[kTileElems * kMaxWarpsPerCta];
    __shared__ float C_tile_all[kTileElems * kMaxWarpsPerCta];
    const half* d_group_tile_vals_half =
        reinterpret_cast<const half*>(d_group_tile_vals);

    for (int strip_base = 0; strip_base < num_strips; strip_base += num_warps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        using namespace nvcuda;

        const int strip = strip_base + warp_id_in_cta;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        if (strip < num_strips) {
            wmma::fill_fragment(c_frag, 0.0f);
        }

        for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
            const int kb = d_group_tile_k_ids[tile_idx];
            const int k_start = kb * 16;
            const half* tile_ptr = d_group_tile_vals_half + static_cast<i64>(tile_idx) * kTileElems;
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
                    const int reordered_row = group_row_start + lr;
                    const int original_row = reordered_to_original[reordered_row];
                    const int n = n_start + ln;
                    if (original_row < M && n < N) {
                        C_out[(i64)original_row * N + n] = C_tile[i];
                    }
                }
            }
        }
        __syncthreads();
#endif
    }
}

__global__ void tc_reordered_fp32_kernel(
    const int* __restrict__ d_fp32_rows,
    const int* __restrict__ d_row_ptr_r,
    const int* __restrict__ d_col_r,
    const float* __restrict__ d_val_r,
    const float* __restrict__ B,
    const int* __restrict__ reordered_to_original,
    float* __restrict__ C_out,
    int M,
    int N,
    int num_fp32_rows)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_id >= num_fp32_rows) {
        return;
    }

    const int reordered_row = d_fp32_rows[warp_id];
    if (reordered_row >= M) {
        return;
    }
    const int original_row = reordered_to_original[reordered_row];

    const int start = d_row_ptr_r[reordered_row];
    const int end = d_row_ptr_r[reordered_row + 1];
    for (int n = lane; n < N; n += 32) {
        float acc = 0.f;
        for (int p = start; p < end; ++p) {
            acc += d_val_r[p] * load_readonly_f32(B + (i64)d_col_r[p] * N + n);
        }
        C_out[(i64)original_row * N + n] = acc;
    }
}

__global__ void tc_reordered_fp32_kernel_vec4(
    const int* __restrict__ d_fp32_rows,
    const int* __restrict__ d_row_ptr_r,
    const int* __restrict__ d_col_r,
    const float* __restrict__ d_val_r,
    const float* __restrict__ B,
    const int* __restrict__ reordered_to_original,
    float* __restrict__ C_out,
    int M,
    int N,
    int num_fp32_rows)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_id >= num_fp32_rows) {
        return;
    }

    const int reordered_row = d_fp32_rows[warp_id];
    if (reordered_row >= M) {
        return;
    }
    const int original_row = reordered_to_original[reordered_row];

    const int start = d_row_ptr_r[reordered_row];
    const int end = d_row_ptr_r[reordered_row + 1];
    const int N4 = N / 4;
    for (int n4 = lane; n4 < N4; n4 += 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = start; p < end; ++p) {
            const int col = d_col_r[p];
            const float a_val = d_val_r[p];
            const float4* B_ptr =
                reinterpret_cast<const float4*>(B + static_cast<i64>(col) * N);
            const float4 b4 = B_ptr[n4];
            acc.x += a_val * b4.x;
            acc.y += a_val * b4.y;
            acc.z += a_val * b4.z;
            acc.w += a_val * b4.w;
        }
        float4* C_ptr = reinterpret_cast<float4*>(C_out + static_cast<i64>(original_row) * N);
        C_ptr[n4] = acc;
    }
}

}  // namespace

TCReorderedPlan make_tc_reordered_plan(
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M,
    int K,
    int N)
{
    TCReorderedPlan plan;
    plan.M = M;
    plan.K = K;
    plan.workspace_N = N;

    if (M <= 0 || K <= 0 || N < 64) {
        return plan;
    }

    const int total_nnz = h_rowptr[M];
    if (total_nnz <= 0) {
        return plan;
    }

    std::vector<RowOrderInfo> info(M);
    for (int row = 0; row < M; ++row) {
        const int start = h_rowptr[row];
        const int end = h_rowptr[row + 1];
        const int len = end - start;

        RowOrderInfo row_info;
        row_info.row = row;
        row_info.len = len;
        row_info.min_col = (len > 0) ? h_col[start] : 0;
        row_info.max_col = (len > 0) ? h_col[end - 1] : 0;

        double centroid_sum = 0.0;
        for (int p = start; p < end; ++p) {
            centroid_sum += h_col[p];
            const int bucket = std::min(kSignatureBuckets - 1, (h_col[p] * kSignatureBuckets) / std::max(1, K));
            row_info.signature |= (uint64_t{1} << bucket);
        }
        row_info.centroid = (len > 0) ? static_cast<float>(centroid_sum / static_cast<double>(len)) : 0.f;
        info[row] = row_info;
    }

    std::vector<int> identity_order(M);
    std::iota(identity_order.begin(), identity_order.end(), 0);

    std::vector<int> order = identity_order;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const int bucket_a = static_cast<int>(info[a].centroid * 16.f / static_cast<float>(std::max(1, K)));
        const int bucket_b = static_cast<int>(info[b].centroid * 16.f / static_cast<float>(std::max(1, K)));
        if (bucket_a != bucket_b) {
            return bucket_a < bucket_b;
        }
        const int span_a = (info[a].len > 0) ? (info[a].max_col - info[a].min_col + 1) : K;
        const int span_b = (info[b].len > 0) ? (info[b].max_col - info[b].min_col + 1) : K;
        if (span_a != span_b) {
            return span_a < span_b;
        }
        return info[a].len > info[b].len;
    });

    for (int base = 0; base < M; base += kGroupRows) {
        const int end = std::min(M, base + kGroupRows);
        std::stable_sort(order.begin() + base, order.begin() + end, [&](int a, int b) {
            return info[a].len > info[b].len;
        });
    }

    const auto [identity_compactness, identity_similarity] =
        summarize_group_metrics(identity_order, info, K);
    const auto [reordered_compactness, reordered_similarity] =
        summarize_group_metrics(order, info, K);
    const float identity_score = identity_compactness + identity_similarity;
    const float reordered_score = reordered_compactness + reordered_similarity;
    if (reordered_score <= identity_score + 0.01f) {
        order = identity_order;
    }

    plan.h_row_perm = new int[M];
    plan.h_row_perm_inv = new int[M];
    for (int reordered_row = 0; reordered_row < M; ++reordered_row) {
        const int original_row = order[reordered_row];
        plan.h_row_perm[reordered_row] = original_row;
        plan.h_row_perm_inv[original_row] = reordered_row;
    }

    std::vector<int> group_offsets;
    group_offsets.reserve((M + kGroupRows - 1) / kGroupRows + 1);
    group_offsets.push_back(0);
    std::vector<int> group_use_fp32;
    group_use_fp32.reserve((M + kGroupRows - 1) / kGroupRows);
    std::vector<int> group_tile_offsets;
    group_tile_offsets.reserve((M + kGroupRows - 1) / kGroupRows + 1);
    group_tile_offsets.push_back(0);
    std::vector<int> group_tile_k_ids;
    std::vector<uint16_t> group_tile_vals;
    std::vector<int> fp32_rows;

    double compactness_sum = 0.0;
    double similarity_sum = 0.0;
    double tc_tile_density_sum = 0.0;
    int group_count = 0;
    int fp32_groups = 0;
    int tc_groups = 0;
    for (int base = 0; base < M; base += kGroupRows) {
        const int end = std::min(M, base + kGroupRows);
        group_offsets.push_back(end);

        int min_col = K;
        int max_col = -1;
        int64_t group_nnz = 0;
        int group_max_row_nnz = 0;
        float local_similarity = 0.f;
        int similarity_pairs = 0;
        for (int idx = base; idx < end; ++idx) {
            const RowOrderInfo& row_info = info[order[idx]];
            group_nnz += row_info.len;
            group_max_row_nnz = std::max(group_max_row_nnz, row_info.len);
            if (row_info.len > 0) {
                min_col = std::min(min_col, row_info.min_col);
                max_col = std::max(max_col, row_info.max_col);
            }
            if (idx > base) {
                local_similarity += jaccard_u64(info[order[idx - 1]].signature, row_info.signature);
                ++similarity_pairs;
            }
        }
        const int span = (max_col >= min_col) ? (max_col - min_col + 1) : K;
        const float avg_row_nnz =
            static_cast<float>(group_nnz) / static_cast<float>(std::max(1, end - base));
        compactness_sum += static_cast<double>(group_nnz) /
                           static_cast<double>(std::max(1, (end - base) * std::max(1, span)));
        similarity_sum += (similarity_pairs > 0)
            ? static_cast<double>(local_similarity / static_cast<float>(similarity_pairs))
            : 0.0;

        bool use_fp32_group =
            (group_max_row_nnz >= kFp32GroupMaxRowNnzThreshold) ||
            (group_nnz >= kFp32GroupTotalNnzThreshold) ||
            (avg_row_nnz >= kFp32GroupAvgRowNnzThreshold);
        if (!use_fp32_group) {
            std::map<int, std::array<float, kTileElems>> tile_map;
            for (int idx = base; idx < end; ++idx) {
                const int original_row = order[idx];
                const int local_row = idx - base;
                for (int p = h_rowptr[original_row]; p < h_rowptr[original_row + 1]; ++p) {
                    const int col = h_col[p];
                    auto it = tile_map.find(col / 16);
                    if (it == tile_map.end()) {
                        it = tile_map.emplace(col / 16, std::array<float, kTileElems>{}).first;
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
                    tc_tile_density_sum += tile_density;
                    ++tc_groups;
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
        ++group_count;
    }
    plan.num_groups = static_cast<int>(group_offsets.size()) - 1;
    plan.num_fp32_groups = fp32_groups;
    plan.num_fp32_rows = static_cast<int>(fp32_rows.size());
    plan.num_tc_tiles = static_cast<int>(group_tile_k_ids.size());
    plan.avg_group_compactness = (group_count > 0)
        ? static_cast<float>(compactness_sum / static_cast<double>(group_count))
        : 0.f;
    plan.avg_group_similarity = (group_count > 0)
        ? static_cast<float>(similarity_sum / static_cast<double>(group_count))
        : 0.f;
    plan.fp32_group_fraction = (plan.num_groups > 0)
        ? static_cast<float>(plan.num_fp32_groups) / static_cast<float>(plan.num_groups)
        : 0.f;
    plan.avg_tc_tile_density = (tc_groups > 0)
        ? static_cast<float>(tc_tile_density_sum / static_cast<double>(tc_groups))
        : 0.f;

    std::vector<int> r_rowptr(M + 1, 0);
    std::vector<int> r_col(total_nnz);
    std::vector<float> r_val(total_nnz);
    int write_ptr = 0;
    for (int reordered_row = 0; reordered_row < M; ++reordered_row) {
        const int original_row = plan.h_row_perm[reordered_row];
        std::vector<std::pair<int, float>> entries;
        entries.reserve(h_rowptr[original_row + 1] - h_rowptr[original_row]);
        for (int p = h_rowptr[original_row]; p < h_rowptr[original_row + 1]; ++p) {
            entries.push_back({h_col[p], h_val[p]});
        }
        std::sort(entries.begin(), entries.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (const auto& entry : entries) {
            r_col[write_ptr] = entry.first;
            r_val[write_ptr] = entry.second;
            ++write_ptr;
        }
        r_rowptr[reordered_row + 1] = write_ptr;
    }

    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_row_ptr_r, (M + 1) * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_col_r, total_nnz * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_val_r, total_nnz * sizeof(float)));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_row_ptr_r, r_rowptr.data(), (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_col_r, r_col.data(), total_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_val_r, r_val.data(), total_nnz * sizeof(float), cudaMemcpyHostToDevice));

    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_perm_inv, M * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_perm_inv, plan.h_row_perm, M * sizeof(int), cudaMemcpyHostToDevice));

    plan.d_group_offsets = upload_tcr(group_offsets);
    plan.d_group_use_fp32 = upload_tcr(group_use_fp32);
    plan.d_fp32_rows = upload_tcr(fp32_rows);
    plan.d_group_tile_offsets = upload_tcr(group_tile_offsets);
    plan.d_group_tile_k_ids = upload_tcr(group_tile_k_ids);
    plan.d_group_tile_vals = upload_tcr(group_tile_vals);

    plan.plan_bytes =
        (size_t)(M + 1) * sizeof(int) +
        (size_t)total_nnz * sizeof(int) +
        (size_t)total_nnz * sizeof(float) +
        (size_t)M * sizeof(int) +
        (size_t)group_offsets.size() * sizeof(int) +
        (size_t)group_use_fp32.size() * sizeof(int) +
        (size_t)fp32_rows.size() * sizeof(int) +
        (size_t)group_tile_offsets.size() * sizeof(int) +
        (size_t)group_tile_k_ids.size() * sizeof(int) +
        (size_t)group_tile_vals.size() * sizeof(uint16_t);
    plan.active = true;
    return plan;
}

void run_tc_reordered_plan(
    const TCReorderedPlan& plan,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream)
{
    if (!plan.active || plan.M <= 0 || N <= 0) {
        return;
    }

    const int warps_per_cta = std::max(1, std::min(kMaxWarpsPerCta, (N + 15) / 16));
    const int tc_threads = warps_per_cta * 32;
    tc_reordered_tc_kernel<<<plan.num_groups, tc_threads, 0, stream>>>(
        plan.d_group_offsets,
        plan.d_group_use_fp32,
        plan.d_group_tile_offsets,
        plan.d_group_tile_k_ids,
        plan.d_group_tile_vals,
        d_B,
        plan.d_perm_inv,
        d_C,
        plan.M,
        plan.K,
        N,
        plan.num_groups);
    CUDA_CHECK_KERNEL();

    if (plan.num_fp32_rows > 0) {
        const int fp32_threads = 4 * 32;
        const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
        const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
        const bool use_vec4 = (N % 4 == 0) && b_aligned && c_aligned;
        const int fp32_blocks =
            (plan.num_fp32_rows + (fp32_threads / 32) - 1) / (fp32_threads / 32);
        if (use_vec4) {
            tc_reordered_fp32_kernel_vec4<<<fp32_blocks, fp32_threads, 0, stream>>>(
                plan.d_fp32_rows,
                plan.d_row_ptr_r,
                plan.d_col_r,
                plan.d_val_r,
                d_B,
                plan.d_perm_inv,
                d_C,
                plan.M,
                N,
                plan.num_fp32_rows);
        } else {
            tc_reordered_fp32_kernel<<<fp32_blocks, fp32_threads, 0, stream>>>(
                plan.d_fp32_rows,
                plan.d_row_ptr_r,
                plan.d_col_r,
                plan.d_val_r,
                d_B,
                plan.d_perm_inv,
                d_C,
                plan.M,
                N,
                plan.num_fp32_rows);
        }
        CUDA_CHECK_KERNEL();
    }
}

void free_tc_reordered_plan(TCReorderedPlan& plan)
{
    delete[] plan.h_row_perm;
    delete[] plan.h_row_perm_inv;
    plan.h_row_perm = nullptr;
    plan.h_row_perm_inv = nullptr;

    if (plan.d_row_ptr_r) { cudaFree(plan.d_row_ptr_r); plan.d_row_ptr_r = nullptr; }
    if (plan.d_col_r) { cudaFree(plan.d_col_r); plan.d_col_r = nullptr; }
    if (plan.d_val_r) { cudaFree(plan.d_val_r); plan.d_val_r = nullptr; }
    if (plan.d_perm_inv) { cudaFree(plan.d_perm_inv); plan.d_perm_inv = nullptr; }
    if (plan.d_group_offsets) { cudaFree(plan.d_group_offsets); plan.d_group_offsets = nullptr; }
    if (plan.d_group_use_fp32) { cudaFree(plan.d_group_use_fp32); plan.d_group_use_fp32 = nullptr; }
    if (plan.d_fp32_rows) { cudaFree(plan.d_fp32_rows); plan.d_fp32_rows = nullptr; }
    if (plan.d_group_tile_offsets) { cudaFree(plan.d_group_tile_offsets); plan.d_group_tile_offsets = nullptr; }
    if (plan.d_group_tile_k_ids) { cudaFree(plan.d_group_tile_k_ids); plan.d_group_tile_k_ids = nullptr; }
    if (plan.d_group_tile_vals) { cudaFree(plan.d_group_tile_vals); plan.d_group_tile_vals = nullptr; }
    if (plan.d_workspace_C) { cudaFree(plan.d_workspace_C); plan.d_workspace_C = nullptr; }

    plan.active = false;
    plan.plan_bytes = 0;
}
