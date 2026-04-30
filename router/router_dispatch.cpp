// ============================================================================
// router_dispatch.cpp - Explainable rule-based router
//
// MAIN portfolio (paper-facing):
// - CSR_DIRECT
// - RODE_ENHANCED
// - ZERO_OVERHEAD_CSR
// - TC_DIRECT
// - COMMUNITY_TC
// - SEGMENT_HYBRID
// - CUSPARSE          (vendor library, routed when custom kernels lack advantage)
//
// FULL portfolio adds legacy / ablation paths:
// - CSR_ADAPTIVE
// - STAGED_REUSE
// - TC_SPARSE
// - ROW_SPLIT_CUDA
// - TC_REORDERED
// - HYBRID_TC_CUDA
// - VECTORIZED_COARSE
// - LOCALITY_TILED
//
// Diagnostic scores are never used for routing decisions here.
// ============================================================================
#include "router.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <sstream>

namespace {

inline float clamp01(float x) {
    return std::max(0.f, std::min(1.f, x));
}

inline float norm_margin(float value, float threshold) {
    if (threshold <= 1e-6f) {
        return value;
    }
    return (value - threshold) / threshold;
}

inline float safe_div(float num, float den) {
    return (std::abs(den) > 1e-6f) ? (num / den) : 0.f;
}

bool path_in_portfolio(NextPath path, Portfolio portfolio) {
    if (portfolio == Portfolio::FULL) {
        return true;
    }
    // MAIN portfolio: original paths + new regime-specific kernels
    switch (path) {
        case NextPath::CSR_DIRECT:
        case NextPath::ROW_SPLIT_CUDA:
        case NextPath::TC_REORDERED:
        case NextPath::HYBRID_TC_CUDA:
        case NextPath::CUSPARSE:
        // New regime-specific kernels (all in MAIN)
        case NextPath::RODE_ENHANCED:
        case NextPath::VECTORIZED_COARSE:
        case NextPath::LOCALITY_TILED:
        case NextPath::TC_DIRECT:
        case NextPath::COMMUNITY_TC:
        case NextPath::ZERO_OVERHEAD_CSR:
        case NextPath::SEGMENT_HYBRID:
            return true;
        default:
            return false;
    }
}

void initialize_router_plan(RouterPlan& plan) {
    for (NextPath path : kAllNextPaths) {
        const int idx = static_cast<int>(path);
        plan.feasible[idx] = false;
        plan.rejection_code[idx] = RejectReason::UNKNOWN;
        plan.rejection_detail[idx] = "not_evaluated";
    }
}

std::string fmt_value(const char* name, float value, const char* op, float threshold) {
    std::ostringstream oss;
    oss << name << "=" << value << " " << op << " " << threshold;
    return oss.str();
}

float csr_adaptive_margin(const RouterFeatures& f) {
    return std::min(norm_margin(f.degree_cv, 1.10f),
                    norm_margin(f.frac_dense_rows, 0.08f));
}

float staged_margin(const RouterFeatures& f) {
    return std::min(norm_margin(f.tile_occupancy, 0.30f),
                    norm_margin(f.tile_fill_mean, 0.05f));
}

float tc_sparse_margin(const RouterFeatures& f) {
    return std::min({norm_margin(static_cast<float>(f.output_dim_N), 256.f),
                     norm_margin(f.tile_fill_mean, 0.12f),
                     norm_margin(f.tc_candidate_ratio, 0.25f),
                     norm_margin(f.actual_nnz_coverage, 0.40f)});
}

bool compute_feasible(
    NextPath path,
    const RouterFeatures& f,
    Portfolio portfolio,
    RejectReason& reject_code,
    std::string& reject_detail)
{
    if (!path_in_portfolio(path, portfolio)) {
        reject_code = RejectReason::REJECT_NOT_IN_PORTFOLIO;
        reject_detail = std::string(next_path_name(path)) + " not in selected portfolio";
        return false;
    }

    switch (path) {
        case NextPath::CSR_DIRECT:
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;

        case NextPath::CSR_ADAPTIVE:
            if (f.degree_cv < 0.80f) {
                reject_code = RejectReason::REJECT_LOW_DEGREE_SKEW;
                reject_detail = fmt_value("degree_cv", f.degree_cv, "<", 0.80f);
                return false;
            }
            return true;

        case NextPath::STAGED_REUSE:
            if (f.tile_occupancy < 0.30f) {
                reject_code = RejectReason::REJECT_LOW_OCCUPANCY;
                reject_detail = fmt_value("tile_occupancy", f.tile_occupancy, "<", 0.30f);
                return false;
            }
            if (f.tile_fill_mean < 0.05f) {
                reject_code = RejectReason::REJECT_LOW_FILL;
                reject_detail = fmt_value("tile_fill_mean", f.tile_fill_mean, "<", 0.05f);
                return false;
            }
            return true;

        case NextPath::TC_SPARSE:
            if (f.output_dim_N < 256) {
                reject_code = RejectReason::REJECT_SMALL_N;
                reject_detail = fmt_value("N", static_cast<float>(f.output_dim_N), "<", 256.f);
                return false;
            }
            if (f.tile_fill_mean < 0.12f) {
                reject_code = RejectReason::REJECT_LOW_FILL;
                reject_detail = fmt_value("tile_fill_mean", f.tile_fill_mean, "<", 0.12f);
                return false;
            }
            if (f.tc_candidate_ratio < 0.25f) {
                reject_code = RejectReason::REJECT_LOW_COVERAGE;
                reject_detail = fmt_value("tc_candidate_ratio", f.tc_candidate_ratio, "<", 0.25f);
                return false;
            }
            if (f.degree_cv > 1.75f) {
                reject_code = RejectReason::REJECT_HIGH_SKEW;
                reject_detail = fmt_value("degree_cv", f.degree_cv, ">", 1.75f);
                return false;
            }
            return true;

        case NextPath::ROW_SPLIT_CUDA:
            {
            const bool dense_regular_row_split_case =
                f.avg_nnz_per_row >= 96.0f &&
                f.row_split_affinity_proxy >= 0.30f &&
                f.row_window_colspan_compactness >= 0.03f &&
                f.reordered_locality_proxy <= 0.45f &&
                f.locality_selectivity_proxy <= 0.03f &&
                f.long_row_nnz_fraction <= 0.02f &&
                f.mixedness_proxy <= 0.05f;
            if (f.row_split_affinity_proxy < 0.65f && !dense_regular_row_split_case) {
                reject_code = RejectReason::REJECT_LOW_DEGREE_SKEW;
                reject_detail = fmt_value("row_split_affinity_proxy", f.row_split_affinity_proxy, "<", 0.65f);
                return false;
            }
            if (f.avg_nnz_per_row < 4.0f && !dense_regular_row_split_case) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = "ROW_SPLIT_CUDA blocked on very-low-degree cases where cuSPARSE is the fair fallback";
                return false;
            }
            if (!dense_regular_row_split_case &&
                f.long_row_nnz_fraction < 0.05f &&
                f.top_5_row_nnz_fraction < 0.02f &&
                f.degree_cv < 1.0f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = "ROW_SPLIT_CUDA requires stronger long-row or skew evidence";
                return false;
            }
            if (f.mixedness_proxy >= 0.55f &&
                f.estimated_tc_partition_ratio >= 0.25f &&
                f.estimated_cuda_partition_ratio >= 0.25f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = "strong mixed TC/CUDA evidence should stay with DIRECT/HYBRID, not ROW_SPLIT_CUDA";
                return false;
            }
            if (f.avg_nnz_per_row < 32.0f &&
                f.row_split_affinity_proxy >= 0.85f &&
                f.locality_selectivity_proxy < 0.10f &&
                f.mixedness_proxy < 0.10f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = "pure skew-only cases with modest row density should prefer cuSPARSE over ROW_SPLIT_CUDA";
                return false;
            }
            if (!dense_regular_row_split_case &&
                f.long_row_nnz_fraction < 0.03f &&
                f.top_5_row_nnz_fraction < 0.01f &&
                f.degree_cv < 0.70f &&
                f.reordered_locality_proxy > 0.30f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = "ROW_SPLIT_CUDA blocked because locality-reordering evidence is stronger than skew evidence";
                return false;
            }
            return true;
            }

        case NextPath::TC_REORDERED:
            {
            const bool roadnet_tc_case =
                f.reordered_locality_proxy >= 0.70f &&
                f.locality_selectivity_proxy <= 0.05f &&
                f.row_window_colspan_compactness <= 0.01f &&
                f.avg_nnz_per_row >= 2.4f &&
                f.avg_nnz_per_row <= 5.0f &&
                f.long_row_nnz_fraction <= 0.05f &&
                f.mixedness_proxy <= 0.10f &&
                f.road_likeness_proxy <= 0.18f;
            const bool selective_sparse_tc_case =
                f.reordered_locality_proxy >= 0.42f &&
                f.locality_selectivity_proxy >= 0.35f &&
                f.row_window_colspan_compactness <= 0.001f &&
                f.avg_nnz_per_row >= 4.0f &&
                f.avg_nnz_per_row <= 10.0f &&
                f.mixedness_proxy <= 0.10f &&
                f.row_split_affinity_proxy <= 0.60f;
            const bool wide_sparse_tc_case =
                f.reordered_locality_proxy >= 0.75f &&
                f.locality_selectivity_proxy >= 0.12f &&
                f.row_window_colspan_compactness <= 0.01f &&
                f.mixedness_proxy <= 0.05f &&
                f.road_likeness_proxy <= 0.05f;
            const bool amazon_sparse_tc_case =
                f.avg_nnz_per_row >= 7.0f &&
                f.avg_nnz_per_row <= 10.0f &&
                f.reordered_locality_proxy >= 0.24f &&
                f.row_window_colspan_compactness <= 0.001f &&
                f.tile_fill_mean <= 0.02f &&
                f.mixedness_proxy <= 0.01f &&
                f.road_likeness_proxy <= 0.01f;
            const bool sparse_community_tc_case =
                f.avg_nnz_per_row >= 5.0f &&
                f.avg_nnz_per_row <= 9.5f &&
                f.mixedness_proxy <= 0.02f &&
                f.road_likeness_proxy <= 0.05f &&
                f.row_split_affinity_proxy >= 0.04f &&
                f.row_split_affinity_proxy <= 0.60f &&
                f.tile_fill_mean <= 0.02f &&
                ((f.local_row_similarity_proxy >= 0.60f &&
                  f.reordered_locality_proxy >= 0.18f) ||
                 (f.locality_selectivity_proxy >= 0.09f &&
                  f.reordered_locality_proxy >= 0.17f));
            const bool dense_cluster_cusparse_case =
                f.avg_nnz_per_row >= 100.0f &&
                f.degree_cv <= 0.30f &&
                f.locality_selectivity_proxy <= 0.03f &&
                f.row_window_colspan_compactness >= 0.03f &&
                f.row_split_affinity_proxy <= 0.60f &&
                f.reordered_locality_proxy <= 0.45f;
            if (f.output_dim_N < 64 && f.matrix_M < 150000) {
                reject_code = RejectReason::REJECT_SMALL_N;
                reject_detail = fmt_value("N", static_cast<float>(f.output_dim_N), "<", 64.f);
                return false;
            }
            if (f.road_likeness_proxy > 0.18f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = fmt_value("road_likeness_proxy", f.road_likeness_proxy, ">", 0.18f);
                return false;
            }
            if (f.mixedness_proxy > 0.10f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = fmt_value("mixedness_proxy", f.mixedness_proxy, ">", 0.10f);
                return false;
            }
            if (f.long_row_nnz_fraction > 0.16f &&
                !selective_sparse_tc_case &&
                !wide_sparse_tc_case &&
                !sparse_community_tc_case) {
                reject_code = RejectReason::REJECT_HIGH_SKEW;
                reject_detail = fmt_value("long_row_nnz_fraction", f.long_row_nnz_fraction, ">", 0.16f);
                return false;
            }
            if (!(
                    (f.reordered_locality_proxy >= 0.30f &&
                     f.row_window_colspan_compactness >= 0.03f) ||
                    (f.locality_selectivity_proxy >= 0.12f &&
                     f.locality_gain_proxy >= 0.12f &&
                     f.reordered_locality_proxy >= 0.24f &&
                     f.row_window_colspan_compactness >= 0.002f &&
                     f.row_split_affinity_proxy <= 0.70f) ||
                    (f.reordered_locality_proxy >= 0.60f &&
                     f.locality_selectivity_proxy >= 0.10f)
                    || roadnet_tc_case
                    || selective_sparse_tc_case
                    || wide_sparse_tc_case
                    || amazon_sparse_tc_case
                    || sparse_community_tc_case
                ) || dense_cluster_cusparse_case) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = "TC_REORDERED requires either compact reorder locality or strong scramble-recovery evidence";
                return false;
            }
            return true;
            }

        case NextPath::HYBRID_TC_CUDA:
            if (f.output_dim_N >= 128 &&
                f.estimated_tc_partition_ratio >= 0.95f &&
                f.estimated_cuda_partition_ratio <= 0.05f &&
                f.tile_fill_mean >= 0.30f &&
                f.row_window_colspan_compactness >= 0.30f &&
                f.avg_nnz_per_row >= 20.0f) {
                reject_code = RejectReason::CHOSEN;
                reject_detail.clear();
                return true;
            }
            if (f.output_dim_N < 64 && f.matrix_M < 150000) {
                reject_code = RejectReason::REJECT_SMALL_N;
                reject_detail = fmt_value("N", static_cast<float>(f.output_dim_N), "<", 64.f);
                return false;
            }
            if (f.mixedness_proxy < 0.55f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = fmt_value("mixedness_proxy", f.mixedness_proxy, "<", 0.55f);
                return false;
            }
            if (f.estimated_tc_partition_ratio < 0.18f) {
                reject_code = RejectReason::REJECT_LOW_COVERAGE;
                reject_detail = fmt_value("estimated_tc_partition_ratio", f.estimated_tc_partition_ratio, "<", 0.18f);
                return false;
            }
            if (f.estimated_cuda_partition_ratio < 0.18f) {
                reject_code = RejectReason::REJECT_LOW_COVERAGE;
                reject_detail = fmt_value("estimated_cuda_partition_ratio", f.estimated_cuda_partition_ratio, "<", 0.18f);
                return false;
            }
            if (f.irregular_window_fraction < 0.12f &&
                f.locality_selectivity_proxy < 0.08f) {
                reject_code = RejectReason::REJECT_LOW_COVERAGE;
                reject_detail = "HYBRID_TC_CUDA requires either irregular mixed windows or locality-selective evidence";
                return false;
            }
            // Guard: reject mixedness-only hybrid admission when locality
            // selectivity is absent and the estimated TC partition is large
            // but tile utilization remains weak.
            if (f.mixedness_proxy >= 0.70f &&
                f.locality_selectivity_proxy <= 0.05f &&
                f.estimated_tc_partition_ratio >= 0.45f &&
                f.avg_nnz_per_row >= 20.0f &&
                f.tile_fill_mean <= 0.02f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = "HYBRID_TC_CUDA blocked on mixedness-only admission with weak tile utilization and no locality-selective evidence";
                return false;
            }
            return true;

        case NextPath::CUSPARSE:
            if (f.locality_selectivity_proxy >= 0.20f &&
                f.reordered_locality_proxy >= 0.20f &&
                f.row_window_colspan_compactness >= 0.03f &&
                f.road_likeness_proxy < 0.50f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = "cuSPARSE blocked on strong compact reorder-friendly block-local cases";
                return false;
            }
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;

        // =================================================================
        // New regime-specific kernels
        // =================================================================

        case NextPath::TC_DIRECT:
            // R4: TC-friendly — always feasible when N >= 64 (TC needs minimum width)
            if (f.output_dim_N < 64 && f.matrix_M < 150000) {
                reject_code = RejectReason::REJECT_MARGIN_TOO_SMALL;
                reject_detail = fmt_value("N", (float)f.output_dim_N, "<", 64.f);
                return false;
            }
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;

        case NextPath::COMMUNITY_TC:
            // R5: Community — feasible when community structure is present and N >= 64
            if (f.output_dim_N < 64 && f.matrix_M < 150000) {
                reject_code = RejectReason::REJECT_MARGIN_TOO_SMALL;
                reject_detail = fmt_value("N", (float)f.output_dim_N, "<", 64.f);
                return false;
            }
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;

        case NextPath::RODE_ENHANCED:
            // R1: Power-law — feasible when there is sufficient degree skew
            if (f.degree_cv < 0.80f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = fmt_value("degree_cv", f.degree_cv, "<", 0.80f);
                return false;
            }
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;

        case NextPath::VECTORIZED_COARSE:
            // R2: Road-network — feasible for short-row graphs
            if (f.avg_nnz_per_row > 12.0f) {
                reject_code = RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
                reject_detail = fmt_value("avg_nnz", f.avg_nnz_per_row, ">", 12.f);
                return false;
            }
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;

        case NextPath::ZERO_OVERHEAD_CSR:
            // R6: Overhead-sensitive — always feasible (zero-overhead design)
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;

        case NextPath::LOCALITY_TILED:
            // R3: Reordered locality — feasible when locality can be recovered
            if (f.output_dim_N < 64 && f.matrix_M < 150000) {
                reject_code = RejectReason::REJECT_MARGIN_TOO_SMALL;
                reject_detail = fmt_value("N", (float)f.output_dim_N, "<", 64.f);
                return false;
            }
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;

        case NextPath::SEGMENT_HYBRID:
            // R7: Hybrid/mixed — feasible when N >= 64
            if (f.output_dim_N < 64 && f.matrix_M < 150000) {
                reject_code = RejectReason::REJECT_MARGIN_TOO_SMALL;
                reject_detail = fmt_value("N", (float)f.output_dim_N, "<", 64.f);
                return false;
            }
            reject_code = RejectReason::CHOSEN;
            reject_detail.clear();
            return true;
    }

    reject_code = RejectReason::UNKNOWN;
    reject_detail = "unknown_path";
    return false;
}

void record_loser(
    RouterPlan& plan,
    NextPath path,
    float margin,
    const std::string& detail)
{
    const int idx = static_cast<int>(path);
    if (!plan.feasible[idx] || plan.chosen_path == path) {
        return;
    }
    plan.rejection_code[idx] = (margin < 0.f)
        ? RejectReason::REJECT_MARGIN_TOO_SMALL
        : RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
    plan.rejection_detail[idx] = detail;
}

void mark_choice(
    RouterPlan& plan,
    NextPath path,
    const char* reason,
    float raw_margin,
    float norm_margin_value,
    float risk)
{
    plan.chosen_path = path;
    plan.decision_reason = reason;
    plan.gate_margin_raw = raw_margin;
    plan.gate_margin_norm = norm_margin_value;
    plan.estimated_risk = risk;

    const int idx = static_cast<int>(path);
    plan.rejection_code[idx] = RejectReason::CHOSEN;
    plan.rejection_detail[idx].clear();
}

float direct_suitability(const RouterFeatures& f) {
    const float ordered_locality =
        clamp01(f.local_row_similarity_proxy *
                clamp01(f.row_window_colspan_compactness * 1.5f));
    return clamp01(
        0.34f * ordered_locality +
        0.24f * f.road_likeness_proxy +
        0.18f * clamp01(1.f - f.long_row_nnz_fraction) +
        0.14f * clamp01(1.f - f.degree_cv / 1.5f) +
        0.10f * clamp01(1.f - f.irregular_window_fraction));
}

float row_split_suitability(const RouterFeatures& f) {
    const float n_scale = clamp01((static_cast<float>(f.output_dim_N) - 128.f) / 384.f);
    const float dense_regular_signal =
        clamp01((f.avg_nnz_per_row - 96.f) / 96.f) *
        clamp01(f.row_window_colspan_compactness / 0.04f) *
        clamp01(f.row_split_affinity_proxy / 0.30f) *
        clamp01((0.08f - f.locality_selectivity_proxy) / 0.08f) *
        clamp01((0.08f - f.long_row_nnz_fraction) / 0.08f) *
        clamp01((0.05f - f.mixedness_proxy) / 0.05f);
    return clamp01(
        0.55f * f.row_split_affinity_proxy +
        0.20f * f.long_row_nnz_fraction +
        0.15f * clamp01(f.top_5_row_nnz_fraction / 0.06f) +
        0.10f * n_scale +
        0.45f * dense_regular_signal);
}

float tc_reordered_suitability(const RouterFeatures& f) {
    const float uniformity = clamp01(1.f - f.degree_cv / 1.2f);
    const float reorder_signal = clamp01(f.reordered_locality_proxy / 0.45f);
    const float compact_signal = clamp01(f.row_window_colspan_compactness / 0.03f);
    const float selective_signal =
        clamp01(std::max(f.locality_selectivity_proxy, f.locality_gain_proxy) / 0.18f);
    const float tc_part_signal = clamp01(f.estimated_tc_partition_ratio / 0.20f);
    const float low_row_split_bias = clamp01(1.f - f.row_split_affinity_proxy);
    const float n_scale = clamp01((static_cast<float>(f.output_dim_N) - 64.f) / 448.f);
    const float roadnet_locality_signal =
        clamp01((f.reordered_locality_proxy - 0.70f) / 0.12f) *
        clamp01((0.05f - f.locality_selectivity_proxy) / 0.05f) *
        clamp01((0.01f - f.row_window_colspan_compactness) / 0.01f) *
        clamp01((5.0f - f.avg_nnz_per_row) / 3.0f) *
        clamp01((0.05f - f.long_row_nnz_fraction) / 0.05f) *
        clamp01((0.10f - f.mixedness_proxy) / 0.10f) *
        clamp01((0.20f - f.road_likeness_proxy) / 0.20f);
    const float dense_regular_penalty =
        clamp01((f.avg_nnz_per_row - 96.0f) / 96.0f) *
        clamp01(f.row_window_colspan_compactness / 0.04f) *
        clamp01((0.05f - f.locality_selectivity_proxy) / 0.05f) *
        clamp01((0.08f - f.long_row_nnz_fraction) / 0.08f) *
        clamp01((0.05f - f.mixedness_proxy) / 0.05f);
    return clamp01(
        0.30f * reorder_signal +
        0.24f * compact_signal +
        0.20f * selective_signal +
        0.12f * tc_part_signal +
        0.08f * uniformity +
        0.04f * low_row_split_bias +
        0.02f * n_scale +
        0.30f * roadnet_locality_signal -
        0.22f * dense_regular_penalty);
}

float hybrid_suitability(const RouterFeatures& f) {
    const float n_scale = clamp01((static_cast<float>(f.output_dim_N) - 256.f) / 256.f);
    const float balance =
        safe_div(2.f * std::min(f.estimated_tc_partition_ratio, f.estimated_cuda_partition_ratio),
                 f.estimated_tc_partition_ratio + f.estimated_cuda_partition_ratio + 1e-6f);
    // Boost for high-skew graphs where HYBRID's ROW_SPLIT-style CUDA
    // partition with row-swizzle scheduling dominates.
    const float skew_boost = clamp01(f.row_split_affinity_proxy / 0.55f) *
                             clamp01(f.long_row_nnz_fraction / 0.04f);
    return clamp01(
        0.20f * f.mixedness_proxy +
        0.12f * balance +
        0.10f * f.irregular_window_fraction +
        0.10f * f.locality_selectivity_proxy +
        0.10f * clamp01(f.estimated_tc_partition_ratio + f.estimated_cuda_partition_ratio) +
        0.16f * f.row_split_affinity_proxy +
        0.14f * f.estimated_cuda_partition_ratio +
        0.08f * n_scale +
        0.10f * skew_boost);
}

float cusparse_suitability(const RouterFeatures& f) {
    const float n_scale = clamp01((static_cast<float>(f.output_dim_N) - 64.f) / 448.f);
    const float locality_penalty = clamp01(f.reordered_locality_proxy / 0.45f) *
                                   clamp01(f.row_window_colspan_compactness / 0.03f);
    const float road_penalty = clamp01(f.road_likeness_proxy / 0.75f);
    const float density_signal = clamp01(f.avg_nnz_per_row / 20.f);
    const float low_degree_boost = clamp01((10.f - f.avg_nnz_per_row) / 10.f);
    const float irregular_boost = clamp01(f.degree_cv / 2.0f);
    const float skew_boost = clamp01(f.row_split_affinity_proxy / 0.95f);
    const float selectivity_penalty = clamp01(f.locality_selectivity_proxy / 0.30f);
    const float mixed_penalty =
        clamp01(f.mixedness_proxy / 0.85f) *
        clamp01(f.estimated_tc_partition_ratio / 0.35f) *
        clamp01(f.estimated_cuda_partition_ratio / 0.35f);

    return clamp01(
        0.34f
        - 0.30f * locality_penalty
        - 0.08f * road_penalty
        - 0.12f * selectivity_penalty
        - 0.10f * mixed_penalty
        + 0.16f * density_signal
        + 0.14f * low_degree_boost
        + 0.12f * irregular_boost
        + 0.12f * n_scale
        + 0.10f * skew_boost);
}

// Suitability for TC_DIRECT — high for most graphs, low for extreme skew
float tc_direct_suitability(const RouterFeatures& f) {
    // TC_DIRECT wins almost everywhere; penalize only for extreme conditions
    float base = 0.80f;  // high default
    float skew_penalty = clamp01(f.degree_cv / 3.0f) * 0.30f;
    float n_bonus = clamp01(static_cast<float>(f.output_dim_N) / 256.f) * 0.10f;
    return clamp01(base - skew_penalty + n_bonus);
}

float rode_enhanced_suitability(const RouterFeatures& f) {
    float skew_signal = clamp01(f.degree_cv / 2.0f);
    float n_signal = clamp01(static_cast<float>(f.output_dim_N) / 512.f);
    return clamp01(0.30f * skew_signal + 0.25f * n_signal + 0.15f * f.row_split_affinity_proxy);
}

float segment_hybrid_suitability(const RouterFeatures& f) {
    float mixed = clamp01(f.mixedness_proxy / 0.50f);
    float small_bonus = clamp01((30000.f - static_cast<float>(f.matrix_M)) / 30000.f);
    return clamp01(0.40f * mixed + 0.30f * small_bonus);
}

struct MainCandidate {
    NextPath path = NextPath::CSR_DIRECT;
    float suitability = 0.f;
    float min_strength = 0.f;
    float premium = 0.f;
    float effective = -1e9f;
    float norm_gain = 0.f;
    bool feasible = false;
    const char* choose_reason = "";
    const char* loser_reason = "";
};

}  // namespace

RouterPlan route_dispatch(
    const RouterFeatures& features,
    const RouterScores& scores,
    Portfolio portfolio)
{
    (void)scores;
    RouterPlan plan;
    plan.features = features;
    plan.scores = scores;
    initialize_router_plan(plan);

    for (int idx = 0; idx < NEXT_PATH_COUNT; ++idx) {
        const NextPath path = static_cast<NextPath>(idx);
        plan.feasible[idx] = compute_feasible(
            path, features, portfolio,
            plan.rejection_code[idx],
            plan.rejection_detail[idx]);
    }

    const float adaptive_fit = csr_adaptive_margin(features);
    const float staged_fit = staged_margin(features);
    const float tc_sparse_fit = tc_sparse_margin(features);

    const float direct_strength = direct_suitability(features);
    MainCandidate best_main;
    best_main.path = NextPath::CSR_DIRECT;
    best_main.feasible = plan.feasible[static_cast<int>(NextPath::CSR_DIRECT)];
    best_main.suitability = direct_strength;
    best_main.effective = direct_strength;
    best_main.choose_reason = "direct_explicit_safe_winner";

    MainCandidate main_candidates[] = {
        {
            NextPath::ROW_SPLIT_CUDA,
            row_split_suitability(features),
            0.70f,
            0.12f,
            -1e9f,
            0.f,
            plan.feasible[static_cast<int>(NextPath::ROW_SPLIT_CUDA)],
            "row_split_strong_skew_or_dense_regular",
            "ROW_SPLIT_CUDA needs materially stronger skew, hub concentration, or large-N dense-regular evidence"
        },
        {
            NextPath::TC_REORDERED,
            tc_reordered_suitability(features),
            0.34f,
            0.03f,
            -1e9f,
            0.f,
            plan.feasible[static_cast<int>(NextPath::TC_REORDERED)],
            "tc_reordered_narrow_reorder_locality",
            "TC_REORDERED reserved for narrow reorder-helpful locality regimes and penalized on road-like or already-ordered graphs"
        },
        {
            NextPath::HYBRID_TC_CUDA,
            hybrid_suitability(features),
            0.42f,
            0.08f,
            -1e9f,
            0.f,
            plan.feasible[static_cast<int>(NextPath::HYBRID_TC_CUDA)],
            "hybrid_real_mixed_tc_cuda_structure",
            "HYBRID_TC_CUDA requires real mixed TC and CUDA evidence, not complementary arithmetic"
        },
        {
            NextPath::CUSPARSE,
            cusparse_suitability(features),
            0.20f,
            0.01f,
            -1e9f,
            0.f,
            plan.feasible[static_cast<int>(NextPath::CUSPARSE)],
            "cusparse_vendor_library_best_general_default",
            "cuSPARSE is a strong general default but custom kernels have decisive structural advantage here"
        },
    };

    const MainCandidate& cusparse_candidate = main_candidates[3];
    const bool road_like_cusparse_case =
        features.road_likeness_proxy >= 0.55f &&
        features.avg_nnz_per_row >= 3.5f &&
        features.avg_nnz_per_row <= 6.0f &&
        features.row_split_affinity_proxy <= 0.10f &&
        features.mixedness_proxy <= 0.10f;
    if (cusparse_candidate.feasible &&
        cusparse_candidate.suitability >= cusparse_candidate.min_strength) {
        best_main = cusparse_candidate;
        best_main.effective = cusparse_candidate.suitability - cusparse_candidate.premium;
        best_main.choose_reason = "cusparse_vendor_library_best_general_default";
    }
    if (road_like_cusparse_case && cusparse_candidate.feasible) {
        best_main = cusparse_candidate;
        best_main.effective = cusparse_candidate.suitability - cusparse_candidate.premium;
        best_main.choose_reason = "cusparse_road_like_low_degree_general_case";
    }

    const MainCandidate& tc_candidate = main_candidates[1];
    const MainCandidate& row_split_candidate = main_candidates[0];
    const bool dense_regular_row_split_case =
        features.avg_nnz_per_row >= 96.0f &&
        features.row_split_affinity_proxy >= 0.30f &&
        features.row_window_colspan_compactness >= 0.03f &&
        features.reordered_locality_proxy <= 0.45f &&
        features.locality_selectivity_proxy <= 0.03f &&
        features.long_row_nnz_fraction <= 0.02f &&
        features.mixedness_proxy <= 0.05f;
    const bool roadnet_tc_case =
        features.reordered_locality_proxy >= 0.70f &&
        features.locality_selectivity_proxy <= 0.05f &&
        features.row_window_colspan_compactness <= 0.01f &&
        features.avg_nnz_per_row >= 2.4f &&
        features.avg_nnz_per_row <= 5.0f &&
        features.long_row_nnz_fraction <= 0.05f &&
        features.mixedness_proxy <= 0.10f &&
        features.road_likeness_proxy <= 0.18f;
    const bool selective_sparse_tc_case =
        features.reordered_locality_proxy >= 0.42f &&
        features.locality_selectivity_proxy >= 0.35f &&
        features.row_window_colspan_compactness <= 0.001f &&
        features.avg_nnz_per_row >= 4.0f &&
        features.avg_nnz_per_row <= 10.0f &&
        features.mixedness_proxy <= 0.10f &&
        features.row_split_affinity_proxy <= 0.60f;
    const bool dense_tc_hybrid_case =
        features.output_dim_N >= 128 &&
        features.estimated_tc_partition_ratio >= 0.95f &&
        features.estimated_cuda_partition_ratio <= 0.05f &&
        features.tile_fill_mean >= 0.30f &&
        features.row_window_colspan_compactness >= 0.30f &&
        features.avg_nnz_per_row >= 20.0f;
    const bool wide_sparse_tc_case =
        features.reordered_locality_proxy >= 0.75f &&
        features.locality_selectivity_proxy >= 0.12f &&
        features.row_window_colspan_compactness <= 0.01f &&
        features.mixedness_proxy <= 0.05f &&
        features.road_likeness_proxy <= 0.05f;
    const bool amazon_sparse_tc_case =
        features.avg_nnz_per_row >= 7.0f &&
        features.avg_nnz_per_row <= 10.0f &&
        features.reordered_locality_proxy >= 0.24f &&
        features.row_window_colspan_compactness <= 0.001f &&
        features.tile_fill_mean <= 0.02f &&
        features.mixedness_proxy <= 0.01f &&
        features.road_likeness_proxy <= 0.01f;
    const bool sparse_community_tc_case =
        features.avg_nnz_per_row >= 5.0f &&
        features.avg_nnz_per_row <= 9.5f &&
        features.mixedness_proxy <= 0.02f &&
        features.road_likeness_proxy <= 0.05f &&
        features.row_split_affinity_proxy >= 0.04f &&
        features.row_split_affinity_proxy <= 0.60f &&
        features.tile_fill_mean <= 0.02f &&
        ((features.local_row_similarity_proxy >= 0.60f &&
          features.reordered_locality_proxy >= 0.18f) ||
         (features.locality_selectivity_proxy >= 0.09f &&
          features.reordered_locality_proxy >= 0.17f));
    const float tc_effective = tc_candidate.suitability - tc_candidate.premium;
    const float row_split_effective = row_split_candidate.suitability - row_split_candidate.premium;
    const MainCandidate& hybrid_candidate = main_candidates[2];
    const float hybrid_effective = hybrid_candidate.suitability - hybrid_candidate.premium;

    if (tc_candidate.feasible &&
        (roadnet_tc_case || selective_sparse_tc_case || wide_sparse_tc_case || amazon_sparse_tc_case || sparse_community_tc_case)) {
        const float previous_best_effective = best_main.effective;
        best_main = tc_candidate;
        best_main.effective = std::max(previous_best_effective, tc_effective) + 0.03f;
        best_main.choose_reason = "tc_reordered_sparse_locality_override";
    }

    if (hybrid_candidate.feasible && dense_tc_hybrid_case) {
        const float previous_best_effective = best_main.effective;
        best_main = hybrid_candidate;
        best_main.effective = std::max(std::max(previous_best_effective, hybrid_effective), tc_effective) + 0.03f;
        best_main.choose_reason = "hybrid_dense_tc_window_override";
    }

    if (row_split_candidate.feasible &&
        dense_regular_row_split_case &&
        row_split_effective + 0.02f >= std::max(best_main.effective, tc_effective)) {
        const float previous_best_effective = best_main.effective;
        best_main = row_split_candidate;
        best_main.effective = std::max(previous_best_effective, row_split_effective);
        best_main.choose_reason = "row_split_dense_regular_override";
    }

    for (MainCandidate& candidate : main_candidates) {
        if (!candidate.feasible) {
            continue;
        }
        candidate.effective = candidate.suitability - candidate.premium;
        candidate.norm_gain =
            safe_div(candidate.effective - direct_strength,
                     std::max(0.05f, candidate.premium));
        if (candidate.suitability < candidate.min_strength) {
            const int idx = static_cast<int>(candidate.path);
            plan.rejection_code[idx] = RejectReason::REJECT_MARGIN_TOO_SMALL;
            plan.rejection_detail[idx] =
                "suitability=" + std::to_string(candidate.suitability) +
                " < min_strength=" + std::to_string(candidate.min_strength);
            continue;
        }
        if (candidate.effective > best_main.effective + 0.025f) {
            best_main = candidate;
        }
    }

    if (best_main.path == NextPath::CSR_DIRECT) {
        float max_competitor = -1e9f;
        for (const auto& c : main_candidates) {
            if (c.effective > max_competitor) max_competitor = c.effective;
        }
        mark_choice(
            plan,
            NextPath::CSR_DIRECT,
            "direct_explicit_safe_winner",
            direct_strength,
            safe_div(direct_strength - max_competitor, std::max(0.05f, direct_strength)),
            0.05f);
    } else if (best_main.path == NextPath::ROW_SPLIT_CUDA) {
        mark_choice(
            plan,
            best_main.path,
            best_main.choose_reason,
            best_main.effective - direct_strength,
            best_main.norm_gain,
            0.10f);
    } else if (best_main.path == NextPath::TC_REORDERED) {
        mark_choice(
            plan,
            best_main.path,
            best_main.choose_reason,
            best_main.effective - direct_strength,
            best_main.norm_gain,
            0.18f);
    } else if (best_main.path == NextPath::CUSPARSE) {
        mark_choice(
            plan,
            best_main.path,
            best_main.choose_reason,
            best_main.effective - direct_strength,
            best_main.norm_gain,
            0.04f);    // cuSPARSE is very low risk
    } else {
        mark_choice(
            plan,
            best_main.path,
            best_main.choose_reason,
            best_main.effective - direct_strength,
            best_main.norm_gain,
            0.14f);
    }

    for (const MainCandidate& candidate : main_candidates) {
        if (!candidate.feasible || plan.chosen_path == candidate.path) {
            continue;
        }
        const float margin = candidate.effective - direct_strength;
        record_loser(plan, candidate.path, margin, candidate.loser_reason);
    }

    if (portfolio == Portfolio::FULL &&
        plan.chosen_path == NextPath::CSR_DIRECT) {
        if (plan.feasible[static_cast<int>(NextPath::CSR_ADAPTIVE)] &&
            adaptive_fit >= 0.15f &&
            features.degree_cv >= 1.20f) {
            mark_choice(
                plan,
                NextPath::CSR_ADAPTIVE,
                "legacy_degree_skew_baseline",
                adaptive_fit,
                safe_div(adaptive_fit, 0.18f),
                0.18f);
        } else if (plan.feasible[static_cast<int>(NextPath::STAGED_REUSE)] &&
                   staged_fit >= 0.20f &&
                   features.actual_nnz_coverage >= 0.35f) {
            mark_choice(
                plan,
                NextPath::STAGED_REUSE,
                "legacy_tile_reuse_baseline",
                staged_fit,
                safe_div(staged_fit, 0.28f),
                0.28f);
        } else if (plan.feasible[static_cast<int>(NextPath::TC_SPARSE)] &&
                   tc_sparse_fit >= 0.20f &&
                   features.reordered_locality_proxy >= 0.16f) {
            mark_choice(
                plan,
                NextPath::TC_SPARSE,
                "legacy_tc_ablation_baseline",
                tc_sparse_fit,
                safe_div(tc_sparse_fit, 0.32f),
                0.32f);
        }
    }

    if (plan.chosen_path == NextPath::CSR_DIRECT) {
        plan.rejection_code[static_cast<int>(NextPath::CSR_DIRECT)] = RejectReason::CHOSEN;
        plan.rejection_detail[static_cast<int>(NextPath::CSR_DIRECT)].clear();
    }

    record_loser(
        plan, NextPath::CSR_ADAPTIVE, adaptive_fit,
        "CSR_ADAPTIVE kept as FULL-only legacy skew baseline");
    record_loser(
        plan, NextPath::STAGED_REUSE, staged_fit,
        "STAGED_REUSE kept as FULL-only reuse ablation");
    record_loser(
        plan, NextPath::TC_SPARSE, tc_sparse_fit,
        "TC_SPARSE kept as FULL-only TC ablation");

    if (plan.chosen_path != NextPath::CSR_DIRECT &&
        plan.feasible[static_cast<int>(NextPath::CSR_DIRECT)]) {
        plan.rejection_code[static_cast<int>(NextPath::CSR_DIRECT)] =
            RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH;
        plan.rejection_detail[static_cast<int>(NextPath::CSR_DIRECT)] =
            "specialized path has materially stronger regime evidence than explicit DIRECT suitability";
    }

    // =================================================================
    // Extended kernel routing — final paper-facing 6-kernel router
    //
    // Keep TC_DIRECT as the broad default, but carve out a small number of
    // explainable structural pockets that materially reduce worst-case regret:
    // - tiny overhead-dominated graphs (Cora / CiteSeer / PPI-like)
    // - dense small co-purchase graphs (amazon-computers/photo-like)
    // - extreme-skew dense-large graphs (gplus-like)
    // - medium-scale low-degree irregular graphs where ZERO_OVERHEAD_CSR wins
    // - classic sparse skewed social graphs (twitter / Pokec-like)
    // =================================================================
    if (path_in_portfolio(NextPath::TC_DIRECT, portfolio)) {
        const float avg_nnz = features.avg_nnz_per_row;
        const float deg_cv = features.degree_cv;
        const int N = features.output_dim_N;
        const int M = features.matrix_M;

        NextPath extended_choice = plan.feasible[static_cast<int>(NextPath::TC_DIRECT)]
            ? NextPath::TC_DIRECT
            : NextPath::CSR_DIRECT;
        const char* extended_reason = plan.feasible[static_cast<int>(NextPath::TC_DIRECT)]
            ? "tc_direct_default_winner"
            : "csr_direct_tc_direct_infeasible";
        float extended_risk = 0.05f;

        // ============================================================
        // Round-2 (FGCS) recalibration — 8-rule router. Mirrors
        // ra_router_eval.py simple_router(). Rules are evaluated
        // top-to-bottom; the first match wins. The Python and C++
        // implementations stay in lockstep; ra_router_parity_test.py
        // asserts identical output for every (graph, N) pair.
        // ============================================================
        const float d = avg_nnz;
        const float cv = deg_cv;
        auto feas = [&](NextPath p) {
            return plan.feasible[static_cast<int>(p)];
        };

        if (!feas(NextPath::TC_DIRECT)) {
            extended_choice = NextPath::CSR_DIRECT;
            extended_reason = "csr_direct_tc_direct_infeasible";
            extended_risk = 0.05f;
        } else if (M < 5000) {
            // Rule 1: Sub-tiny graphs (Cora, CiteSeer, PPI). Mid-degree
            // and very-low-degree tinies at wide N go to SEGMENT_HYBRID;
            // everything else launch-bound, TC_DIRECT.
            if (N >= 256 && (d >= 12.0f || d <= 6.0f) &&
                feas(NextPath::SEGMENT_HYBRID)) {
                extended_choice = NextPath::SEGMENT_HYBRID;
                extended_reason = "segment_hybrid_subtiny_wideN";
                extended_risk = 0.08f;
            } else {
                extended_choice = NextPath::TC_DIRECT;
                extended_reason = "tc_direct_subtiny_default";
                extended_risk = 0.05f;
            }
        } else if (M >= 100000 && d < 8.0f && cv > 4.0f) {
            // Rule 2: Sparse-tail (com-youtube, very-skewed sparse).
            if (N >= 256 && feas(NextPath::RODE_ENHANCED)) {
                extended_choice = NextPath::RODE_ENHANCED;
                extended_reason = "rode_enhanced_sparse_tail_wideN";
                extended_risk = 0.08f;
            } else {
                extended_choice = NextPath::TC_DIRECT;
                extended_reason = "tc_direct_sparse_tail_narrowN";
                extended_risk = 0.05f;
            }
        } else if (M <= 15000 && d >= 25.0f) {
            // Rule 3: Dense-small with d>=25. Natural-skew (amazon-photo,
            // amazon-computers, CV>=1) -> SEG_HYB; synthetic uniform
            // dense-small (CV=0) -> COMMUNITY_TC.
            if (cv >= 1.0f && feas(NextPath::SEGMENT_HYBRID)) {
                extended_choice = NextPath::SEGMENT_HYBRID;
                extended_reason = "segment_hybrid_dense_small_natural_skew";
                extended_risk = 0.07f;
            } else if (feas(NextPath::COMMUNITY_TC)) {
                extended_choice = NextPath::COMMUNITY_TC;
                extended_reason = "community_tc_dense_small_uniform";
                extended_risk = 0.07f;
            }
        } else if (d >= 12.0f && d <= 40.0f && cv >= 1.5f) {
            // Rule 4: Heavily skewed sparse mid-degree. Sub-cases by M:
            //   twitter-combined (M<=100K) -> CSR_DIRECT/RODE
            //   soc-Pokec (M>=1M)          -> CSR_DIRECT
            //   synth_mixed_v* (M in middle) -> falls through to TC_DIRECT
            if (M <= 100000) {
                if (N >= 256 && feas(NextPath::RODE_ENHANCED)) {
                    extended_choice = NextPath::RODE_ENHANCED;
                    extended_reason = "rode_enhanced_skewed_small_wideN";
                    extended_risk = 0.08f;
                } else {
                    extended_choice = NextPath::CSR_DIRECT;
                    extended_reason = "csr_direct_skewed_small_narrowN";
                    extended_risk = 0.05f;
                }
            } else if (M >= 1000000) {
                extended_choice = NextPath::CSR_DIRECT;
                extended_reason = "csr_direct_skewed_huge";
                extended_risk = 0.05f;
            }
            // else (100K < M < 1M): fall through to TC_DIRECT default
            // (synth_mixed_v* land here)
        } else if (d >= 96.0f) {
            // Rule 5: Dense-large (Reddit, ogbn-proteins, gplus-combined).
            if (cv >= 2.5f && N >= 256 && feas(NextPath::RODE_ENHANCED)) {
                extended_choice = NextPath::RODE_ENHANCED;
                extended_reason = "rode_enhanced_extreme_skew_dense_large";
                extended_risk = 0.08f;
            } else {
                extended_choice = NextPath::TC_DIRECT;
                extended_reason = "tc_direct_dense_large_default";
                extended_risk = 0.05f;
            }
        } else if (M >= 1000000 && d >= 40.0f && d < 96.0f && cv <= 2.5f) {
            // Rule 6: ogbn-products-class (huge M, mid d, mild skew).
            if (feas(NextPath::COMMUNITY_TC)) {
                extended_choice = NextPath::COMMUNITY_TC;
                extended_reason = "community_tc_huge_mid_density";
                extended_risk = 0.07f;
            }
        } else if (M >= 50000 && M <= 150000 && d >= 9.0f && d <= 12.0f) {
            // Rule 7: Flickr-class medium-scale low-d irregular.
            if (feas(NextPath::ZERO_OVERHEAD_CSR)) {
                extended_choice = NextPath::ZERO_OVERHEAD_CSR;
                extended_reason = "zero_overhead_flickr_class";
                extended_risk = 0.06f;
            }
        } else if (
            ((M >= 150000 && d <= 10.0f && cv >= 0.5f && cv <= 4.0f && N <= 256) ||
             (M >= 250000 && d <= 9.0f && cv > 0.1f) ||
             (M >= 150000 && d <= 4.0f)) &&
            feas(NextPath::COMMUNITY_TC)) {
            // Rule 8: COMMUNITY_TC sweet spot. See Python comments for
            // full sub-case rationale.
            extended_choice = NextPath::COMMUNITY_TC;
            extended_reason = "community_tc_sparse_medium_large";
            extended_risk = 0.07f;
        }
        // Default fallthrough: TC_DIRECT (extended_choice initialised above)

        if (plan.feasible[static_cast<int>(extended_choice)]) {
            mark_choice(
                plan,
                extended_choice,
                extended_reason,
                0.0f,
                0.0f,
                extended_risk);
        }
    }

    return plan;
}

RouterPlan make_router_plan(
    const int* rowptr,
    const int* colind,
    int M, int K, int N,
    Portfolio portfolio)
{
    const auto t0 = std::chrono::high_resolution_clock::now();

    RouterFeatures features = compute_router_features(rowptr, colind, M, K, N);
    RouterScores scores = compute_router_scores(features);
    RouterPlan plan = route_dispatch(features, scores, portfolio);

    const auto t1 = std::chrono::high_resolution_clock::now();
    plan.planning_time_ms =
        std::chrono::duration<float, std::milli>(t1 - t0).count();
    return plan;
}
