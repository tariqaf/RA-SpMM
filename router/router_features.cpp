// ============================================================================
// router_features.cpp - CPU-side RouterFeatures extraction
//
// All features here are structural, reproducible, and O(nnz) or close to it.
// They are intended to capture:
// - long-row skew and hub concentration for RODE_ENHANCED
// - degree-binned dispatch overhead sensitivity for ZERO_OVERHEAD_CSR
// - locality / block compactness for TC_DIRECT
// - community / modular structure for COMMUNITY_TC
// - mixed TC + irregular structure for SEGMENT_HYBRID
// ============================================================================
#include "router.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <unordered_map>
#include <vector>

namespace {

constexpr int kTileSize = 16;
constexpr int kSignatureBuckets = 64;
constexpr int kWindowRows = 16;

struct RowProxy {
    int row = 0;
    int len = 0;
    int min_col = 0;
    int max_col = 0;
    float centroid = 0.f;
    uint64_t signature = 0;
};

inline float clamp01(float x) {
    return std::max(0.f, std::min(1.f, x));
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

template <typename T>
float safe_mean(const std::vector<T>& values) {
    if (values.empty()) {
        return 0.f;
    }
    double sum = 0.0;
    for (T value : values) {
        sum += static_cast<double>(value);
    }
    return static_cast<float>(sum / static_cast<double>(values.size()));
}

inline float safe_div(float num, float den) {
    return (std::abs(den) > 1e-6f) ? (num / den) : 0.f;
}

}  // namespace

RouterFeatures compute_router_features(
    const int* rowptr,
    const int* colind,
    int M, int K, int N)
{
    RouterFeatures f{};
    if (M <= 0 || K <= 0) {
        return f;
    }

    const int nnz = rowptr[M];
    f.matrix_M = M;
    f.matrix_K = K;
    f.output_dim_N = N;
    f.total_nnz = nnz;
    if (nnz <= 0) {
        return f;
    }

    std::vector<int> row_lens(M, 0);
    std::vector<RowProxy> row_proxies(M);
    std::vector<int> sorted_row_lens;
    sorted_row_lens.reserve(M);

    double sum_lens = 0.0;
    int max_len = 0;
    for (int r = 0; r < M; ++r) {
        const int start = rowptr[r];
        const int end = rowptr[r + 1];
        const int len = end - start;
        row_lens[r] = len;
        sorted_row_lens.push_back(len);
        sum_lens += static_cast<double>(len);
        max_len = std::max(max_len, len);

        RowProxy proxy;
        proxy.row = r;
        proxy.len = len;
        proxy.min_col = (len > 0) ? colind[start] : 0;
        proxy.max_col = (len > 0) ? colind[end - 1] : 0;

        double weighted_col_sum = 0.0;
        uint64_t signature = 0;
        for (int p = start; p < end; ++p) {
            const int c = colind[p];
            weighted_col_sum += static_cast<double>(c);
            const int bucket = (K > 0) ? std::min(kSignatureBuckets - 1, (c * kSignatureBuckets) / std::max(1, K)) : 0;
            signature |= (uint64_t{1} << bucket);
        }
        proxy.signature = signature;
        proxy.centroid = (len > 0) ? static_cast<float>(weighted_col_sum / static_cast<double>(len)) : 0.f;
        row_proxies[r] = proxy;
    }

    const double avg = sum_lens / static_cast<double>(M);
    double var = 0.0;
    int dense_rows = 0;
    int long_rows = 0;
    int64_t long_row_nnz = 0;
    const double long_row_threshold = std::max(32.0, 2.0 * avg);
    for (int len : row_lens) {
        const double delta = static_cast<double>(len) - avg;
        var += delta * delta;
        if (len > 2.0 * avg) {
            ++dense_rows;
        }
        if (static_cast<double>(len) >= long_row_threshold) {
            ++long_rows;
            long_row_nnz += len;
        }
    }
    const double std_dev = std::sqrt(var / static_cast<double>(std::max(1, M)));

    std::sort(sorted_row_lens.begin(), sorted_row_lens.end(), std::greater<int>());
    int64_t top1_nnz = sorted_row_lens.empty() ? 0 : sorted_row_lens.front();
    int64_t top5_nnz = 0;
    for (int i = 0; i < std::min<int>(5, sorted_row_lens.size()); ++i) {
        top5_nnz += sorted_row_lens[i];
    }

    f.avg_nnz_per_row = static_cast<float>(avg);
    f.std_nnz_per_row = static_cast<float>(std_dev);
    f.degree_cv = (avg > 1e-6) ? static_cast<float>(std_dev / avg) : 0.f;
    f.max_to_mean_ratio = (avg > 1e-6) ? static_cast<float>(static_cast<double>(max_len) / avg) : 0.f;
    f.frac_dense_rows = static_cast<float>(dense_rows) / static_cast<float>(std::max(1, M));
    f.skew_ratio = f.max_to_mean_ratio;
    f.long_row_fraction = static_cast<float>(long_rows) / static_cast<float>(std::max(1, M));
    f.long_row_nnz_fraction = static_cast<float>(long_row_nnz) / static_cast<float>(std::max(1, nnz));
    f.top_1_row_nnz_fraction = static_cast<float>(top1_nnz) / static_cast<float>(std::max(1, nnz));
    f.top_5_row_nnz_fraction = static_cast<float>(top5_nnz) / static_cast<float>(std::max(1, nnz));

    std::unordered_map<int64_t, int> tile_counts;
    tile_counts.reserve(static_cast<size_t>(nnz));
    const int tile_rows = (M + kTileSize - 1) / kTileSize;
    const int tile_cols = (K + kTileSize - 1) / kTileSize;
    for (int r = 0; r < M; ++r) {
        const int tile_r = r / kTileSize;
        for (int p = rowptr[r]; p < rowptr[r + 1]; ++p) {
            const int tile_c = colind[p] / kTileSize;
            const int64_t key =
                static_cast<int64_t>(tile_r) * static_cast<int64_t>(tile_cols) +
                static_cast<int64_t>(tile_c);
            ++tile_counts[key];
        }
    }

    std::vector<float> tile_fills;
    tile_fills.reserve(tile_counts.size());
    int64_t tc_candidate_nnz = 0;
    for (const auto& kv : tile_counts) {
        const float fill = static_cast<float>(kv.second) / static_cast<float>(kTileSize * kTileSize);
        tile_fills.push_back(fill);
        if (fill >= TC_FILL_THRESHOLD) {
            tc_candidate_nnz += kv.second;
        }
    }
    std::sort(tile_fills.begin(), tile_fills.end());

    const float fill_mean = safe_mean(tile_fills);
    double fill_var = 0.0;
    for (float fill : tile_fills) {
        const double delta = static_cast<double>(fill) - static_cast<double>(fill_mean);
        fill_var += delta * delta;
    }
    fill_var = tile_fills.empty() ? 0.0 : (fill_var / static_cast<double>(tile_fills.size()));

    f.tile_fill_mean = fill_mean;
    f.tile_fill_median = tile_fills.empty() ? 0.f : tile_fills[tile_fills.size() / 2];
    f.tile_fill_p90 = tile_fills.empty() ? 0.f : tile_fills[std::min(tile_fills.size() - 1, (tile_fills.size() * 9) / 10)];
    f.tile_fill_max = tile_fills.empty() ? 0.f : tile_fills.back();
    f.tile_fill_variance = static_cast<float>(fill_var);
    f.tile_occupancy =
        static_cast<float>(tile_counts.size()) /
        static_cast<float>(std::max<int64_t>(
            1,
            static_cast<int64_t>(tile_rows) * static_cast<int64_t>(tile_cols)));
    f.actual_nnz_coverage = static_cast<float>(tc_candidate_nnz) / static_cast<float>(std::max(1, nnz));
    f.avg_nnz_per_tile =
        tile_counts.empty() ? 0.f :
        static_cast<float>(nnz) / static_cast<float>(tile_counts.size());
    f.tc_candidate_tiles = static_cast<int>(std::count_if(
        tile_fills.begin(), tile_fills.end(),
        [](float fill) { return fill >= TC_FILL_THRESHOLD; }));
    f.tc_candidate_ratio =
        tile_fills.empty() ? 0.f :
        static_cast<float>(f.tc_candidate_tiles) / static_cast<float>(tile_fills.size());

    std::vector<float> adjacent_similarity;
    adjacent_similarity.reserve(M);
    for (int r = 1; r < M; ++r) {
        if (row_proxies[r - 1].len == 0 || row_proxies[r].len == 0) {
            continue;
        }
        adjacent_similarity.push_back(
            jaccard_u64(row_proxies[r - 1].signature, row_proxies[r].signature));
    }
    f.local_row_similarity_proxy = safe_mean(adjacent_similarity);

    std::vector<int> reorder_ids(M);
    std::iota(reorder_ids.begin(), reorder_ids.end(), 0);
    std::stable_sort(reorder_ids.begin(), reorder_ids.end(), [&](int a, int b) {
        const int bucket_a = (K > 0) ? static_cast<int>(row_proxies[a].centroid * 16.f / static_cast<float>(K)) : 0;
        const int bucket_b = (K > 0) ? static_cast<int>(row_proxies[b].centroid * 16.f / static_cast<float>(K)) : 0;
        if (bucket_a != bucket_b) {
            return bucket_a < bucket_b;
        }
        const int span_a = (row_proxies[a].len > 0) ? (row_proxies[a].max_col - row_proxies[a].min_col + 1) : K;
        const int span_b = (row_proxies[b].len > 0) ? (row_proxies[b].max_col - row_proxies[b].min_col + 1) : K;
        if (span_a != span_b) {
            return span_a < span_b;
        }
        return row_proxies[a].len > row_proxies[b].len;
    });

    std::vector<float> reordered_adjacent_similarity;
    reordered_adjacent_similarity.reserve(M);
    for (int i = 1; i < M; ++i) {
        const RowProxy& prev = row_proxies[reorder_ids[i - 1]];
        const RowProxy& curr = row_proxies[reorder_ids[i]];
        if (prev.len == 0 || curr.len == 0) {
            continue;
        }
        reordered_adjacent_similarity.push_back(jaccard_u64(prev.signature, curr.signature));
    }
    f.reordered_locality_proxy = safe_mean(reordered_adjacent_similarity);

    std::vector<float> window_compactness;
    std::vector<float> window_similarity;
    std::vector<float> window_tc_scores;
    std::vector<float> window_irregular_scores;
    int64_t tc_window_nnz = 0;
    int64_t cuda_window_nnz = 0;
    int irregular_windows = 0;
    for (int base = 0; base < M; base += kWindowRows) {
        const int end = std::min(M, base + kWindowRows);
        int64_t window_nnz = 0;
        int min_col = K;
        int max_col = -1;
        int max_row_len = 0;
        int64_t long_window_nnz = 0;
        float sim_sum = 0.f;
        int sim_count = 0;
        for (int r = base; r < end; ++r) {
            const RowProxy& proxy = row_proxies[r];
            window_nnz += proxy.len;
            max_row_len = std::max(max_row_len, proxy.len);
            if (proxy.len > 0) {
                min_col = std::min(min_col, proxy.min_col);
                max_col = std::max(max_col, proxy.max_col);
            }
            if (r > base && row_proxies[r - 1].len > 0 && proxy.len > 0) {
                sim_sum += jaccard_u64(row_proxies[r - 1].signature, proxy.signature);
                ++sim_count;
            }
        }
        const float avg_row_len =
            static_cast<float>(window_nnz) / static_cast<float>(std::max(1, end - base));
        const float window_long_threshold = std::max(32.f, 2.5f * avg_row_len);
        for (int r = base; r < end; ++r) {
            if (static_cast<float>(row_proxies[r].len) >= window_long_threshold) {
                long_window_nnz += row_proxies[r].len;
            }
        }
        const int span = (max_col >= min_col) ? (max_col - min_col + 1) : K;
        const float compactness =
            static_cast<float>(window_nnz) /
            static_cast<float>(std::max(1, (end - base) * std::max(1, span)));
        const float similarity = (sim_count > 0) ? (sim_sum / static_cast<float>(sim_count)) : 0.f;
        const int aligned_span = std::max(16, ((span + 15) / 16) * 16);
        const float tile_density =
            static_cast<float>(window_nnz) /
            static_cast<float>(std::max(16, (end - base) * aligned_span));
        const float compactness_norm = clamp01(compactness * 6.f);
        const float density_norm = clamp01(tile_density * 10.f);
        const float uniformity_norm = clamp01(1.f - safe_div(static_cast<float>(max_row_len), 4.f * avg_row_len + 1.f));
        const float long_mass = safe_div(static_cast<float>(long_window_nnz), static_cast<float>(std::max<int64_t>(1, window_nnz)));
        window_compactness.push_back(compactness);
        window_similarity.push_back(similarity);

        const float tc_score =
            0.35f * compactness_norm +
            0.30f * similarity +
            0.20f * density_norm +
            0.15f * uniformity_norm;
        const float irregular_score =
            0.40f * long_mass +
            0.25f * clamp01(safe_div(static_cast<float>(max_row_len), avg_row_len + 1.f) / 6.f) +
            0.20f * (1.f - compactness_norm) +
            0.15f * (1.f - similarity);
        window_tc_scores.push_back(tc_score);
        window_irregular_scores.push_back(irregular_score);

        const bool tc_window =
            (avg_row_len >= 8.f) &&
            (tc_score >= 0.55f) &&
            (irregular_score < 0.45f);
        const bool cuda_window =
            (irregular_score >= 0.50f) ||
            (long_mass >= 0.20f);
        if (tc_window) {
            tc_window_nnz += window_nnz;
        }
        if (cuda_window) {
            cuda_window_nnz += window_nnz;
            ++irregular_windows;
        }
    }

    f.row_window_colspan_compactness = safe_mean(window_compactness);
    const float window_similarity_mean = safe_mean(window_similarity);
    const float window_irregular_mean = safe_mean(window_irregular_scores);
    f.tc_synergy_proxy =
        0.35f * clamp01(f.tile_fill_mean / 0.20f) +
        0.25f * f.tc_candidate_ratio +
        0.20f * f.reordered_locality_proxy +
        0.20f * f.row_window_colspan_compactness;
    f.locality_gain_proxy = clamp01(f.reordered_locality_proxy - f.local_row_similarity_proxy);
    const float ordered_locality_proxy =
        clamp01(f.local_row_similarity_proxy * clamp01(f.row_window_colspan_compactness * 2.f));
    f.locality_selectivity_proxy =
        clamp01(f.locality_gain_proxy * (1.f - ordered_locality_proxy));
    f.road_likeness_proxy =
        clamp01(f.local_row_similarity_proxy *
                clamp01(1.f - f.degree_cv / 0.35f) *
                clamp01(1.f - f.avg_nnz_per_row / 24.f) *
                clamp01(f.row_window_colspan_compactness / 0.25f));
    const float skew_branch =
        0.45f * clamp01((f.skew_ratio - 2.0f) / 6.0f) +
        0.30f * f.long_row_nnz_fraction +
        0.15f * clamp01(f.top_1_row_nnz_fraction / 0.0020f) +
        0.10f * clamp01(f.top_5_row_nnz_fraction / 0.0100f);
    const float n_scale = clamp01((static_cast<float>(N) - 128.f) / 384.f);
    const float dense_regular_branch =
        clamp01(f.avg_nnz_per_row / 192.f) *
        (0.35f + 0.65f * n_scale);
    f.row_split_affinity_proxy = clamp01(std::max(skew_branch, dense_regular_branch));
    f.estimated_tc_partition_ratio =
        clamp01(static_cast<float>(tc_window_nnz) / static_cast<float>(std::max(1, nnz)));
    f.estimated_cuda_partition_ratio =
        clamp01(static_cast<float>(cuda_window_nnz) / static_cast<float>(std::max(1, nnz)));
    f.irregular_window_fraction =
        static_cast<float>(irregular_windows) /
        static_cast<float>(std::max(1, (M + kWindowRows - 1) / kWindowRows));
    const float partition_balance =
        safe_div(2.f * std::min(f.estimated_tc_partition_ratio, f.estimated_cuda_partition_ratio),
                 f.estimated_tc_partition_ratio + f.estimated_cuda_partition_ratio + 1e-6f);
    f.mixedness_proxy =
        clamp01(partition_balance *
                clamp01((f.estimated_tc_partition_ratio + f.estimated_cuda_partition_ratio) / 0.60f));
    f.tc_granularity_proxy =
        0.55f * clamp01(f.avg_nnz_per_row / 64.f) +
        0.45f * clamp01((1.f - f.long_row_fraction) + 0.5f * f.locality_selectivity_proxy);
    f.redundancy_risk_proxy =
        clamp01(0.45f * f.long_row_nnz_fraction +
                0.30f * (1.f - f.row_window_colspan_compactness) +
                0.15f * (1.f - window_similarity_mean) +
                0.10f * window_irregular_mean);

    return f;
}
