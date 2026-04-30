// ============================================================================
// router_scores.cpp - Interpretable diagnostic scores
//
// These scores are for diagnostics, logging, and paper tables only.
// Routing decisions remain rule-based in router_dispatch.cpp.
// ============================================================================
#include "router.h"

#include <algorithm>

namespace {

inline float clamp01(float x) {
    return std::max(0.f, std::min(1.f, x));
}

inline float balance_score(float a, float b) {
    const float sum = a + b;
    if (sum <= 1e-6f) {
        return 0.f;
    }
    return 1.f - std::abs(a - b) / sum;
}

}  // namespace

RouterScores compute_router_scores(const RouterFeatures& f)
{
    RouterScores s{};

    const float uniformity = clamp01(1.f - f.degree_cv / 2.0f);
    const float skew_signal = clamp01(0.45f * f.long_row_nnz_fraction +
                                      0.30f * f.top_5_row_nnz_fraction +
                                      0.25f * clamp01(f.skew_ratio / 8.0f));
    const float row_split_signal = clamp01(0.55f * f.row_split_affinity_proxy +
                                           0.25f * skew_signal +
                                           0.20f * clamp01(static_cast<float>(f.output_dim_N) / 512.f));
    const float locality_signal = clamp01(0.55f * f.locality_selectivity_proxy +
                                          0.20f * f.locality_gain_proxy +
                                          0.15f * uniformity +
                                          0.10f * clamp01(static_cast<float>(f.output_dim_N) / 512.f));
    const float tc_signal = clamp01(0.40f * f.tc_candidate_ratio +
                                    0.35f * f.tc_synergy_proxy +
                                    0.25f * f.estimated_tc_partition_ratio);
    const float mixed_signal = clamp01(0.50f * f.mixedness_proxy +
                                       0.25f * f.estimated_tc_partition_ratio +
                                       0.25f * f.estimated_cuda_partition_ratio);
    const float direct_signal = clamp01(0.40f * uniformity +
                                        0.25f * (1.f - row_split_signal) +
                                        0.20f * (1.f - locality_signal) +
                                        0.15f * (1.f - mixed_signal));

    s.csr_direct_score = direct_signal;

    s.csr_adaptive_score = 0.60f * clamp01(f.degree_cv / 2.0f) +
                           0.25f * f.frac_dense_rows +
                           0.15f * f.long_row_fraction;

    s.staged_reuse_score = 0.35f * f.tile_occupancy +
                           0.35f * clamp01(f.tile_fill_mean / 0.08f) +
                           0.30f * f.actual_nnz_coverage;

    s.tc_sparse_score = 0.45f * tc_signal +
                        0.35f * clamp01(f.tile_fill_mean / 0.18f) +
                        0.20f * clamp01(1.f - f.degree_cv / 1.8f);

    s.row_split_cuda_score = row_split_signal;

    s.tc_reordered_score = clamp01(0.65f * locality_signal +
                                   0.20f * tc_signal +
                                   0.15f * uniformity);

    s.hybrid_tc_cuda_score = clamp01(0.55f * mixed_signal +
                                     0.20f * tc_signal +
                                     0.15f * skew_signal +
                                     0.10f * locality_signal);

    // cuSPARSE diagnostic score: strong general-purpose, weak on block-local
    const float density_signal = clamp01(f.avg_nnz_per_row / 25.f);
    s.cusparse_score = clamp01(0.40f * density_signal +
                               0.25f * uniformity +
                               0.20f * (1.f - locality_signal) +
                               0.15f * clamp01(static_cast<float>(f.output_dim_N) / 512.f));

    return s;
}
