// ============================================================================
// tc_features.cu - TC feature detection from CSR matrix
//
// All computations are O(nnz) on CPU.
// Computes local fill density per 16x16 block of the sparse matrix.
// ============================================================================
#include "../ra_common.h"
#include <unordered_map>
#include <cmath>
#include <limits>

// ---------------------------------------------------------------------------
// TC feature computation (CPU, O(nnz))
// ---------------------------------------------------------------------------
TCFeatures compute_tc_features(
    const int* rowptr,
    const int* colind,
    int M, int K)
{
    TCFeatures tf{};
    if (M == 0 || K == 0) return tf;

    const int tile_size = TC_TILE_SIZE;  // 16
    int num_row_tiles = (M + tile_size - 1) / tile_size;
    int num_col_tiles = (K + tile_size - 1) / tile_size;
    const int64_t total_possible_tiles_i64 =
        static_cast<int64_t>(num_row_tiles) * static_cast<int64_t>(num_col_tiles);

    tf.total_possible_tiles = (total_possible_tiles_i64 > static_cast<int64_t>(std::numeric_limits<int>::max()))
        ? std::numeric_limits<int>::max()
        : static_cast<int>(total_possible_tiles_i64);

    // Count nnz per 16x16 tile using hash map
    // Key: row_tile * num_col_tiles + col_tile
    std::unordered_map<int64_t, int> tile_nnz;
    tile_nnz.reserve(M);  // reasonable estimate

    int total_nnz = (M > 0) ? rowptr[M] : 0;

    for (int r = 0; r < M; ++r) {
        int rt = r / tile_size;
        for (int p = rowptr[r]; p < rowptr[r + 1]; ++p) {
            int c = colind[p];
            int ct = c / tile_size;
            const int64_t key =
                static_cast<int64_t>(rt) * static_cast<int64_t>(num_col_tiles) +
                static_cast<int64_t>(ct);
            tile_nnz[key]++;
        }
    }

    const int tile_capacity = tile_size * tile_size;  // 256
    const float tc_threshold = TC_FILL_THRESHOLD;     // 0.25 => 64 nnz per 256

    int total_tiles = (int)tile_nnz.size();
    tf.total_tiles_checked = total_tiles;

    // Collect fills into sorted vector for percentile computation
    std::vector<float> fills;
    fills.reserve(total_tiles);
    int64_t candidate_nnz_sum = 0;
    int candidate_tiles = 0;

    for (auto& kv : tile_nnz) {
        float fill = (float)kv.second / tile_capacity;
        fills.push_back(fill);
        if (fill >= tc_threshold) {
            candidate_tiles++;
            candidate_nnz_sum += kv.second;
        }
    }

    std::sort(fills.begin(), fills.end());
    int n = (int)fills.size();

    tf.tc_candidate_tiles = candidate_tiles;

    if (n > 0) {
        // Mean
        double sum_fill = 0.0;
        for (float f : fills) sum_fill += f;
        tf.tile_density_proxy = (float)(sum_fill / n);  // tile_fill_mean

        // Max
        tf.tile_fill_max = fills[n - 1];

        // Median = fills[n/2]
        tf.tile_fill_median = fills[n / 2];

        // p90 = fills[int(0.9*(n-1))]
        tf.tile_fill_p90 = fills[std::min((int)(0.9f * (n - 1)), n - 1)];

        tf.tc_candidate_ratio = (float)candidate_tiles / n;
    } else {
        tf.tile_density_proxy = 0.f;
        tf.tile_fill_max = 0.f;
        tf.tile_fill_median = 0.f;
        tf.tile_fill_p90 = 0.f;
        tf.tc_candidate_ratio = 0.f;
    }

    // actual_nnz_coverage: fraction of total matrix nnz that belong to tiles satisfying
    // the candidate condition (fill >= TC_FILL_THRESHOLD) for the TC path.
    // = (sum of nnz in tiles with fill >= 0.25) / total_nnz
    // Range [0, 1]. High values indicate most of the matrix work is in dense-enough tiles.
    // Used as: STAGED_REUSE coverage gate (actual_nnz_coverage >= THRESH_STAGED_COVERAGE)
    //          TC_SPARSE coverage gate    (actual_nnz_coverage >= THRESH_TC_COVERAGE)
    tf.actual_nnz_coverage = (total_nnz > 0) ? (float)candidate_nnz_sum / total_nnz : 0.f;

    tf.avg_nnz_per_tile = (n > 0) ? (float)total_nnz / n : 0.f;

    tf.tile_occupancy = (total_possible_tiles_i64 > 0)
        ? static_cast<float>(static_cast<double>(n) / static_cast<double>(total_possible_tiles_i64))
        : 0.f;

    // tc_synergy_proxy is diagnostic only -- not used as a routing gate
    tf.tc_synergy_proxy = 0.5f * tf.tc_candidate_ratio
                         + 0.3f * tf.tile_density_proxy
                         + 0.2f * tf.actual_nnz_coverage;

    // Estimated tile overhead: fraction of total_nnz that would need to be
    // materialized into dense tiles (including zero padding)
    float overhead_ratio = (total_nnz > 0)
        ? (float)(candidate_tiles * tile_capacity) / total_nnz
        : 0.f;
    tf.estimated_tile_overhead = overhead_ratio * 0.1f;  // scale to ms estimate

    return tf;
}

// ---------------------------------------------------------------------------
// Also expose a simplified version that works with device pointers
// (copies to CPU first)
// ---------------------------------------------------------------------------
TCFeatures compute_tc_features_device(
    const int* d_rowptr,
    const int* d_colind,
    int M, int K)
{
    if (M == 0) return TCFeatures{};

    int nnz = 0;
    CUDA_CHECK_NEXT(cudaMemcpy(&nnz, d_rowptr + M, sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_rowptr(M + 1);
    CUDA_CHECK_NEXT(cudaMemcpy(h_rowptr.data(), d_rowptr, (M + 1) * sizeof(int), cudaMemcpyDeviceToHost));

    std::vector<int> h_colind(nnz);
    if (nnz > 0) {
        CUDA_CHECK_NEXT(cudaMemcpy(h_colind.data(), d_colind, nnz * sizeof(int), cudaMemcpyDeviceToHost));
    }

    return compute_tc_features(h_rowptr.data(), h_colind.data(), M, K);
}
