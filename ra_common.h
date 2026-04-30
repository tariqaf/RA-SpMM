// ============================================================================
// ra_common.h - Shared types for regime-aware SpMM routing
//
// Paper-facing 6-kernel portfolio (cited in the FGCS paper):
//   - CSR_DIRECT          (csr/csr_direct.cu)        warp-per-row baseline
//   - RODE_ENHANCED       (csr/ra_rode_enhanced.cu)  block-residual decomposition
//   - ZERO_OVERHEAD_CSR   (csr/ra_zero_overhead.cu)  degree-binned dispatch
//   - TC_DIRECT           (tc/ra_tc_direct.cu)       single-pass Tensor Core
//   - COMMUNITY_TC        (tc/ra_community_tc.cu)    label-prop clustering + TC
//   - SEGMENT_HYBRID      (tc/ra_segment_hybrid.cu)  row-level TC/CUDA partition
// Plus: CUSPARSE (vendor baseline, dispatched when no custom kernel dominates)
//
// Legacy / ablation kernels (kept in tree for reproducibility, not in paper):
//   - csr/csr_adaptive.cu, csr/ra_vectorized_coarse.cu, csr/row_split.cu
//   - tc/hybrid_tc_cuda.cu, tc/ra_locality_tiled.cu, tc/tc_reordered.cu,
//     tc/tc_sparse.cu
//
// Target: NVIDIA Ampere (sm_86; RTX 3090 / A6000), CUDA 12.x, cuSPARSE 12.x
// ============================================================================
#pragma once

#include <cuda_runtime.h>
#include <array>
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <functional>
#include <numeric>
#include <stdexcept>

// ---------------------------------------------------------------------------
// Type aliases
// ---------------------------------------------------------------------------
using i32 = int32_t;
using i64 = int64_t;
using u32 = uint32_t;

// ---------------------------------------------------------------------------
// Error checking macros
// ---------------------------------------------------------------------------
#define CUDA_CHECK_NEXT(err) do {                                         \
    cudaError_t _e = (err);                                               \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA error [%s:%d]: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(_e));              \
        throw std::runtime_error(std::string("CUDA error: ") +            \
                                 cudaGetErrorString(_e));                  \
    }                                                                     \
} while(0)

#define CUDA_CHECK_KERNEL() do {                                          \
    cudaError_t _e = cudaGetLastError();                                  \
    if (_e != cudaSuccess) {                                              \
        fprintf(stderr, "CUDA kernel error [%s:%d]: %s\n",               \
                __FILE__, __LINE__, cudaGetErrorString(_e));              \
        throw std::runtime_error(std::string("CUDA kernel error: ") +    \
                                 cudaGetErrorString(_e));                  \
    }                                                                     \
} while(0)

// ---------------------------------------------------------------------------
// RAPath: routing decision enum for regime-aware kernel selection
// ---------------------------------------------------------------------------
enum class NextPath : int {
    // --- Legacy kernels (kept for ablation) ---
    CSR_DIRECT      = 0,
    CSR_ADAPTIVE    = 1,
    STAGED_REUSE    = 2,
    TC_SPARSE       = 3,
    ROW_SPLIT_CUDA  = 4,
    TC_REORDERED    = 5,
    HYBRID_TC_CUDA  = 6,
    CUSPARSE        = 7,

    // --- New regime-specific kernels ---
    RODE_ENHANCED     = 8,   // R1: Hub-dominated power-law
    VECTORIZED_COARSE = 9,   // R2: Ordered sparse / road-network
    LOCALITY_TILED    = 10,  // R3: Reordered locality
    TC_DIRECT          = 11,  // R4: Dense block-local / TC-friendly
    COMMUNITY_TC      = 12,  // R5: Sparse modular community
    ZERO_OVERHEAD_CSR = 13,  // R6: Dense co-purchase / overhead-sensitive
    SEGMENT_HYBRID    = 14,  // R7: Hybrid/mixed
};

static constexpr int NEXT_PATH_COUNT = 15;

inline constexpr std::array<NextPath, NEXT_PATH_COUNT> kAllNextPaths = {
    NextPath::CSR_DIRECT,
    NextPath::CSR_ADAPTIVE,
    NextPath::STAGED_REUSE,
    NextPath::TC_SPARSE,
    NextPath::ROW_SPLIT_CUDA,
    NextPath::TC_REORDERED,
    NextPath::HYBRID_TC_CUDA,
    NextPath::CUSPARSE,
    NextPath::RODE_ENHANCED,
    NextPath::VECTORIZED_COARSE,
    NextPath::LOCALITY_TILED,
    NextPath::TC_DIRECT,
    NextPath::COMMUNITY_TC,
    NextPath::ZERO_OVERHEAD_CSR,
    NextPath::SEGMENT_HYBRID,
};

inline const char* next_path_name(NextPath p) {
    switch (p) {
        case NextPath::CSR_DIRECT:        return "CSR_DIRECT";
        case NextPath::CSR_ADAPTIVE:      return "CSR_ADAPTIVE";
        case NextPath::STAGED_REUSE:      return "STAGED_REUSE";
        case NextPath::TC_SPARSE:         return "TC_SPARSE";
        case NextPath::ROW_SPLIT_CUDA:    return "ROW_SPLIT_CUDA";
        case NextPath::TC_REORDERED:      return "TC_REORDERED";
        case NextPath::HYBRID_TC_CUDA:    return "HYBRID_TC_CUDA";
        case NextPath::CUSPARSE:          return "CUSPARSE";
        case NextPath::RODE_ENHANCED:     return "RODE_ENHANCED";
        case NextPath::VECTORIZED_COARSE: return "VECTORIZED_COARSE";
        case NextPath::LOCALITY_TILED:    return "LOCALITY_TILED";
        case NextPath::TC_DIRECT:          return "TC_DIRECT";
        case NextPath::COMMUNITY_TC:      return "COMMUNITY_TC";
        case NextPath::ZERO_OVERHEAD_CSR: return "ZERO_OVERHEAD_CSR";
        case NextPath::SEGMENT_HYBRID:    return "SEGMENT_HYBRID";
        default:                          return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// Portfolio: main vs full (ablation) path selection
// ---------------------------------------------------------------------------
enum class Portfolio { MAIN, FULL };

// ---------------------------------------------------------------------------
// RejectReason: structured rejection codes for routing diagnostics
// ---------------------------------------------------------------------------
enum class RejectReason : int {
    CHOSEN = 0,
    REJECT_NOT_FEASIBLE,
    REJECT_LOW_FILL,
    REJECT_LOW_COVERAGE,
    REJECT_HIGH_SKEW,
    REJECT_SMALL_N,
    REJECT_INSUFFICIENT_NNZ,
    REJECT_LOW_OCCUPANCY,
    REJECT_LOW_DEGREE_SKEW,
    REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH,
    REJECT_MARGIN_TOO_SMALL,
    REJECT_NOT_IN_PORTFOLIO,
    UNKNOWN
};

inline const char* reject_reason_str(RejectReason r) {
    switch (r) {
        case RejectReason::CHOSEN:                              return "CHOSEN";
        case RejectReason::REJECT_NOT_FEASIBLE:                 return "NOT_FEASIBLE";
        case RejectReason::REJECT_LOW_FILL:                     return "LOW_FILL";
        case RejectReason::REJECT_LOW_COVERAGE:                 return "LOW_COVERAGE";
        case RejectReason::REJECT_HIGH_SKEW:                    return "HIGH_SKEW";
        case RejectReason::REJECT_SMALL_N:                      return "SMALL_N";
        case RejectReason::REJECT_INSUFFICIENT_NNZ:             return "INSUFFICIENT_NNZ";
        case RejectReason::REJECT_LOW_OCCUPANCY:                return "LOW_OCCUPANCY";
        case RejectReason::REJECT_LOW_DEGREE_SKEW:              return "LOW_DEGREE_SKEW";
        case RejectReason::REJECT_SPECIALIZATION_NOT_STRONG_ENOUGH: return "NOT_STRONG_ENOUGH";
        case RejectReason::REJECT_MARGIN_TOO_SMALL:             return "MARGIN_TOO_SMALL";
        case RejectReason::REJECT_NOT_IN_PORTFOLIO:             return "NOT_IN_PORTFOLIO";
        case RejectReason::UNKNOWN:                             return "UNKNOWN";
        default:                                                return "UNKNOWN";
    }
}

// ---------------------------------------------------------------------------
// RouterFeatures: extracted matrix features used for routing decisions
// Four labeled groups.
// ---------------------------------------------------------------------------
struct RouterFeatures {
    // === Group 1: Generic CSR statistics ===
    float avg_nnz_per_row;
    float std_nnz_per_row;
    float degree_cv;
    float max_to_mean_ratio;
    float frac_dense_rows;
    float skew_ratio;         // max_row_len / avg_nnz_per_row
    float long_row_fraction;  // fraction of rows with len > 2*avg
    float long_row_nnz_fraction;
    float top_1_row_nnz_fraction;
    float top_5_row_nnz_fraction;

    // === Group 2: Tile / reuse statistics ===
    float tile_fill_mean;        // avg fill of nonempty 16x16 tiles
    float tile_fill_median;      // p50 fill
    float tile_fill_p90;         // p90 fill
    float tile_fill_max;         // max fill
    float tile_fill_variance;    // variance of nonempty tile fill
    float tile_occupancy;        // nonempty_tiles / total_possible_tiles
    float actual_nnz_coverage;   // nnz in candidate tiles / total_nnz
    float avg_nnz_per_tile;      // total_nnz / nonempty_tiles
    float row_window_colspan_compactness;
    float local_row_similarity_proxy;
    float reordered_locality_proxy;
    float locality_gain_proxy;
    float locality_selectivity_proxy;
    float road_likeness_proxy;
    float row_split_affinity_proxy;
    float mixedness_proxy;

    // === Group 3: TC-specific (diagnostic) ===
    float tc_candidate_ratio;    // fraction of nonempty tiles with fill >= TC_FILL_THRESHOLD
    float tc_synergy_proxy;      // weighted signal -- diagnostic only, not a gate
    float estimated_tc_partition_ratio;
    float estimated_cuda_partition_ratio;
    float irregular_window_fraction;
    float tc_granularity_proxy;
    float redundancy_risk_proxy;
    int   tc_candidate_tiles;

    // === Group 4: Context ===
    int output_dim_N;
    int matrix_M;
    int matrix_K;
    int total_nnz;
};

// ---------------------------------------------------------------------------
// RouterScores: per-path scores from interpretable formulas
// ---------------------------------------------------------------------------
struct RouterScores {
    float csr_direct_score;
    float csr_adaptive_score;
    float staged_reuse_score;
    float tc_sparse_score;
    float row_split_cuda_score;
    float tc_reordered_score;
    float hybrid_tc_cuda_score;
    float cusparse_score;
};

// ---------------------------------------------------------------------------
// RouterPlan: full routing plan with all metadata
// ---------------------------------------------------------------------------
struct RouterPlan {
    NextPath    chosen_path;
    std::string decision_reason;
    RouterFeatures features;
    RouterScores   scores;       // NOTE: RouterScores are diagnostic ONLY -- see note below
    float       estimated_risk;
    float       planning_time_ms;

    // NOTE: RouterScores are diagnostic. Routing decisions are made solely
    // via the rule-based router in router_dispatch.cpp.

    bool         feasible[NEXT_PATH_COUNT]         = {};  // indexed by int(NextPath)
    RejectReason rejection_code[NEXT_PATH_COUNT]   = {};
    std::string  rejection_detail[NEXT_PATH_COUNT] = {};
    float        gate_margin_raw     = 0.f; // diagnostic: chosen path's key feature - Stage2 threshold (raw units)
    float        gate_margin_norm    = 0.f; // diagnostic: (feature - threshold) / threshold (normalized, 0=just passing)
};

// ---------------------------------------------------------------------------
// TCFeatures: TC-specific analysis results (matches Group 2+3 of RouterFeatures)
// ---------------------------------------------------------------------------
struct TCFeatures {
    float tc_candidate_ratio;      // fraction of 16x16 blocks with fill >= threshold
    float tc_synergy_proxy;        // weighted estimate of TC benefit -- diagnostic only
    float tile_density_proxy;      // avg fill of non-empty 16x16 blocks (same as tile_fill_mean)
    float estimated_tile_overhead; // cost of materializing sparse->dense tiles
    int   total_tiles_checked;
    int   tc_candidate_tiles;

    // Extended tile statistics (Phase 2 additions)
    float tile_fill_median;
    float tile_fill_p90;
    float tile_fill_max;
    float actual_nnz_coverage;     // fraction of total matrix nnz in tiles with fill >= TC_FILL_THRESHOLD
    float avg_nnz_per_tile;        // total_nnz / nonempty_tiles
    float tile_occupancy;          // nonempty_tiles / total_possible_tiles
    int   total_possible_tiles;
};

// ---------------------------------------------------------------------------
// TCDiagnostics: runtime TC execution diagnostics
// ---------------------------------------------------------------------------
struct TCDiagnostics {
    int tc_candidate_tiles;   // tiles considered for TC
    int tc_activated_tiles;   // tiles actually processed via TC
    int tc_rejected_tiles;    // tiles rejected (insufficient fill)
    float tc_fill_avg;        // average fill of TC-activated tiles
    bool hw_tc_supported;     // whether hardware supports WMMA
    bool tc_path_taken;       // whether TC path was used at all
};

// ---------------------------------------------------------------------------
// CSRAdaptiveDiagnostics
// ---------------------------------------------------------------------------
struct CSRAdaptiveDiagnostics {
    int bin_histogram[5] = {};  // tiny, short, medium, long, xlong
    int n_split_rows = 0;
    int dominant_bin = 0;
    float frac_long_xlong = 0.f;
};

// ---------------------------------------------------------------------------
// Plan structs (zero-initialized pointers; bindings own them via wrapper classes)
// Memory ownership: each plan has a matching free_*_plan() calling cudaFree
// on each non-null pointer then zeroing it. Binding layer wraps plans in
// non-copyable C++ objects whose destructors call free_*_plan.
// ---------------------------------------------------------------------------

struct CSRAdaptivePlan {
    int* d_tiny    = nullptr; int n_tiny          = 0;
    int* d_short   = nullptr; int n_short         = 0;
    int* d_medium  = nullptr; int n_medium        = 0;
    int* d_long    = nullptr; int n_long          = 0;
    int* d_xlong   = nullptr; int n_xlong_chunks  = 0;
    int  M = 0, K = 0;
    int  bin_histogram[5] = {};
    int  n_split_rows = 0;
    int  dominant_bin = 0;
};


struct StagedReusePlan {
    int*   d_tile_row = nullptr;
    int*   d_tile_col = nullptr;
    int*   d_tile_nnz_start = nullptr;
    int*   d_tile_nnz_count = nullptr;
    int*   d_row_ids  = nullptr;
    int*   d_col_ids  = nullptr;
    float* d_tile_vals = nullptr;
    int    num_tiles = 0, M = 0, K = 0, BM = 64, BK = 64;
    float  avg_tile_fill = 0.f;
};

struct TCSparsePlan {
    // TC tile data
    int*   d_tc_tile_row = nullptr;   int* d_tc_tile_col = nullptr;
    int*   d_tc_nnz_start = nullptr;  int* d_tc_nnz_count = nullptr;
    int*   d_tc_local_row = nullptr;  int* d_tc_local_col = nullptr;
    float* d_tc_vals = nullptr;
    int    num_tc_tiles = 0;
    // Residual CSR
    int*   d_res_rowptr = nullptr; int* d_res_colind = nullptr;
    float* d_res_vals   = nullptr;
    int    res_nnz = 0, M = 0, K = 0;
    bool   tc_eligible = false, hw_tc_supported = false;
    // diagnostics (filled at build time)
    int    candidate_tiles = 0;
    float  fill_mean = 0.f, fill_median = 0.f, fill_p90 = 0.f, fill_max = 0.f;
    int    residual_nnz = 0;
    float  candidate_nnz_coverage = 0.f;
    // timing diagnostics (filled at run time via CUDA events)
    float  t_tc_fused_ms  = 0.f;    // PRODUCTION: pack+mma fused (not separated)
    float  t_residual_ms  = 0.f;    // csr_adaptive fallback
    float  t_accumulate_ms = 0.f;   // element_add merge
    // DEBUG/TIMED MODE ONLY (populated only by spmm_tc_sparse_timed):
    float  t_pack_ms = 0.f;         // A-tile scatter to global (debug mode)
    float  t_mma_ms  = 0.f;         // WMMA mma_sync only (debug mode)
};

struct RowSplitPlan {
    // Regular block part: each descriptor owns the 32-aligned prefix of one row.
    // The common path is atomic-free because one CTA owns one row's regular prefix.
    int* d_regular_row_ids    = nullptr;
    int* d_regular_starts     = nullptr;
    int* d_regular_block_nnz  = nullptr;
    int  num_regular_rows     = 0;

    // Short regular rows: one CTA owns one row and computes the entire regular
    // prefix directly. This remains the common path for ordinary sparse rows.
    int* d_short_row_ids      = nullptr;
    int* d_short_starts       = nullptr;
    int* d_short_block_nnz    = nullptr;
    int  num_short_rows       = 0;

    // Long regular rows: one CTA owns one row-column tile so large-N warm runs
    // can extract more parallelism without resorting to global atomics.
    int* d_long_row_ids       = nullptr;
    int* d_long_starts        = nullptr;
    int* d_long_block_nnz     = nullptr;
    int* d_long_num_segments  = nullptr;
    int  num_long_rows        = 0;

    // Long-row segment descriptors are precomputed for diagnostics and future
    // sub-CTA scheduling. The optimized runtime still keeps these as structural
    // metadata; execution uses row-column tiling for long rows instead.
    int* d_long_seg_row_ids   = nullptr;
    int* d_long_seg_starts    = nullptr;
    int  num_long_segments    = 0;

    // Residual tail descriptors: the <32 nnz suffix of each row.
    int* d_res_row_ids        = nullptr;
    int* d_res_starts         = nullptr;
    int* d_res_lengths        = nullptr;
    int  num_residual         = 0;

    // Per-row diagnostics.
    int* d_row_block_nnz      = nullptr;
    int* d_row_residual_nnz   = nullptr;
    int  num_split_long_rows  = 0;
    float regular_nnz_fraction = 0.f;
    float residual_nnz_fraction = 0.f;
    float avg_segments_per_long_row = 0.f;
    int  M = 0, K = 0, T = 32;
    size_t plan_bytes   = 0;
};

struct TCReorderedPlan {
    int* h_row_perm         = nullptr;  // host: reordered_row -> original_row
    int* h_row_perm_inv     = nullptr;  // host: original_row -> reordered_row
    int* d_row_ptr_r        = nullptr;  // device: reordered CSR rowptr
    int* d_col_r            = nullptr;  // device: reordered CSR colind
    float* d_val_r          = nullptr;  // device: reordered CSR vals
    int* d_perm_inv         = nullptr;  // device: reordered_row -> original_row
    int* d_group_offsets    = nullptr;  // device: 16-row group boundaries
    int* d_group_use_fp32   = nullptr;  // device: per-group precision guard (1 => FP32 fallback)
    int* d_fp32_rows        = nullptr;  // device: reordered rows handled by FP32 fallback
    int* d_group_tile_offsets = nullptr; // device: per-group active TC tile offsets
    int* d_group_tile_k_ids   = nullptr; // device: active TC tile k-block ids
    uint16_t* d_group_tile_vals = nullptr; // device: packed 16x16 half tiles (opaque bits)
    float* d_workspace_C    = nullptr;  // reusable reordered output workspace
    bool active             = false;
    bool placeholder_quality = true;    // honest marker for intermediate TC path
    int  num_groups         = 0;
    int  num_fp32_groups    = 0;
    int  num_fp32_rows      = 0;
    int  num_tc_tiles       = 0;
    int  workspace_N        = 0;
    int  M = 0, K = 0;
    float avg_group_compactness = 0.f;
    float avg_group_similarity  = 0.f;
    float fp32_group_fraction   = 0.f;
    float avg_tc_tile_density   = 0.f;
    size_t plan_bytes       = 0;
};

struct HybridPlan {
    // TC partition (window/group-selected rows with stronger locality signal)
    int* d_tc_row_ptr   = nullptr;
    int* d_tc_col       = nullptr;
    float* d_tc_val     = nullptr;
    int  num_tc_rows    = 0;
    int* d_tc_row_ids   = nullptr;  // original row indices for TC partition
    int* d_tc_group_offsets = nullptr;      // per-16-row TC group boundaries
    int* d_tc_group_tile_offsets = nullptr; // active tile offsets per TC group
    int* d_tc_group_tile_k_ids = nullptr;   // active k-block ids per TC group
    uint16_t* d_tc_group_tile_vals = nullptr; // packed 16x16 half tiles
    int  num_tc_groups = 0;
    int  num_tc_tiles = 0;

    // CUDA partition (residual windows / rows)
    int* d_cuda_row_ptr = nullptr;
    int* d_cuda_col     = nullptr;
    float* d_cuda_val   = nullptr;
    int  num_cuda_rows  = 0;
    int* d_cuda_row_ids = nullptr;  // original row indices for CUDA partition

    // CUDA partition descriptors using a ROW_SPLIT-style execution policy.
    int* d_cuda_short_row_ids   = nullptr;
    int* d_cuda_short_starts    = nullptr;
    int* d_cuda_short_block_nnz = nullptr;
    int  num_cuda_short_rows    = 0;

    int* d_cuda_long_row_ids    = nullptr;
    int* d_cuda_long_starts     = nullptr;
    int* d_cuda_long_block_nnz  = nullptr;
    int  num_cuda_long_rows     = 0;

    int* d_cuda_res_row_ids     = nullptr;
    int* d_cuda_res_starts      = nullptr;
    int* d_cuda_res_lengths     = nullptr;
    int  num_cuda_residual      = 0;

    // Diagnostics
    float tc_nnz_fraction   = 0.f;
    float cuda_nnz_fraction = 0.f;
    float tc_row_fraction   = 0.f;
    float cuda_row_fraction = 0.f;
    float cuda_regular_nnz_fraction = 0.f;
    float cuda_residual_nnz_fraction = 0.f;
    float average_partition_score = 0.f;
    float average_window_compactness = 0.f;
    float partition_score_threshold = 0.45f;
    int   precision_guard_windows = 0;
    float precision_guard_row_fraction = 0.f;
    float precision_guard_nnz_fraction = 0.f;
    int   window_size = 16;
    int   M = 0, K = 0;
    size_t plan_bytes       = 0;
};

// ===========================================================================
// Plan structs for new regime-specific kernels
// ===========================================================================

// R6: Zero-overhead CSR for dense co-purchase / overhead-sensitive regime
struct RAZeroOverheadPlan {
    // Degree-binned row lists
    int* d_tiny_row_ids   = nullptr;  // rows with 1-4 nnz
    int* d_short_row_ids  = nullptr;  // rows with 5-16 nnz
    int* d_medium_row_ids = nullptr;  // rows with 17-64 nnz
    int* d_long_row_ids   = nullptr;  // rows with 65+ nnz
    int  num_tiny   = 0;
    int  num_short  = 0;
    int  num_medium = 0;
    int  num_long   = 0;
    int  M = 0, K = 0;
    size_t plan_bytes = 0;
};

// R2: Vectorized coarsened kernel for ordered sparse / road-network regime
struct RAVectorizedCoarsePlan {
    int rows_per_warp = 8;   // 16 if avg_nnz<=3, 8 if <=6, 4 otherwise
    int M = 0, K = 0;
    size_t plan_bytes = 0;   // zero device allocations
};

// R1: RoDe-enhanced kernel for hub-dominated power-law regime
struct RARodeEnhancedPlan {
    // Short rows (regular_nnz < 128): standard warp-per-row
    int* d_short_row_ids    = nullptr;
    int* d_short_starts     = nullptr;
    int* d_short_block_nnz  = nullptr;
    int  num_short_rows     = 0;

    // Long rows (regular_nnz >= 128): sub-block pipelined execution
    int* d_long_row_ids     = nullptr;
    int* d_long_starts      = nullptr;
    int* d_long_block_nnz   = nullptr;
    int  num_long_rows      = 0;

    // Sub-block descriptors for long-row pipelining
    int* d_long_sub_starts  = nullptr;  // sub-block start offsets
    int* d_long_sub_counts  = nullptr;  // nnz per sub-block
    int* d_long_sub_row_map = nullptr;  // row ownership per sub-block
    int  num_long_sub_blocks = 0;
    int  sub_block_size     = 32;       // nnz per sub-block

    // Residual tail (0-31 nnz remainder per row)
    int* d_res_row_ids      = nullptr;
    int* d_res_starts       = nullptr;
    int* d_res_lengths      = nullptr;
    int  num_residual       = 0;

    // Diagnostics
    int  M = 0, K = 0;
    float regular_nnz_fraction = 0.f;
    float long_row_nnz_fraction = 0.f;
    size_t plan_bytes = 0;
};

// R4: Flash TC plan — fixed TC_REORDERED with single-pass tile iteration
struct RATcDirectPlan {
    // Row permutation (from reordering)
    int* h_row_perm       = nullptr;  // host: reordered_row -> original_row
    int* h_row_perm_inv   = nullptr;  // host: original_row -> reordered_row
    int* d_perm_inv       = nullptr;  // device: reordered_row -> original_row

    // Reordered CSR (for FP32 fallback rows)
    int*   d_row_ptr_r    = nullptr;
    int*   d_col_r        = nullptr;
    float* d_val_r        = nullptr;

    // Group descriptors (16-row groups)
    int* d_group_offsets      = nullptr;
    int* d_group_use_fp32     = nullptr;
    int  num_groups           = 0;

    // TC tile data (packed FP16 16×16 tiles)
    int*      d_group_tile_offsets = nullptr;
    int*      d_group_tile_k_ids  = nullptr;
    uint16_t* d_group_tile_vals   = nullptr;
    int       num_tc_tiles        = 0;

    // FP32 fallback rows
    int* d_fp32_rows      = nullptr;
    int  num_fp32_rows    = 0;
    int  num_fp32_groups  = 0;

    // Diagnostics
    float avg_group_compactness = 0.f;
    float avg_group_similarity  = 0.f;
    float fp32_group_fraction   = 0.f;
    float avg_tc_tile_density   = 0.f;

    bool   active         = false;
    int    M = 0, K = 0;
    size_t plan_bytes     = 0;
};

// R3: Locality-tiled plan — reordering + shared-memory B caching
struct RALocalityTiledPlan {
    // Row permutation
    int* h_row_perm       = nullptr;
    int* h_row_perm_inv   = nullptr;
    int* d_perm_inv       = nullptr;

    // Reordered CSR
    int*   d_row_ptr_r    = nullptr;
    int*   d_col_r        = nullptr;
    float* d_val_r        = nullptr;

    // Panel descriptors (32-row panels with dominant column range)
    int* d_panel_k_start  = nullptr;  // dominant column start per panel
    int  num_panels       = 0;
    int  panel_rows       = 32;
    int  cache_k          = 64;       // columns cached in shared memory

    // Diagnostics
    float avg_cache_hit_rate = 0.f;
    float reorder_gain       = 0.f;

    bool   active         = false;
    int    M = 0, K = 0;
    size_t plan_bytes     = 0;
};

// R5: Community TC plan — community-aware permutation + TC execution
struct RACommunityTCPlan {
    // Row permutation (community-aware reordering)
    int* h_row_perm       = nullptr;
    int* h_row_perm_inv   = nullptr;
    int* d_perm_inv       = nullptr;

    // Reordered CSR (for residual inter-community edges)
    int*   d_row_ptr_r    = nullptr;
    int*   d_col_r        = nullptr;
    float* d_val_r        = nullptr;

    // Community descriptors
    int* d_comm_offsets   = nullptr;
    int  num_communities  = 0;

    // TC tile data (intra-community tiles)
    int* d_group_offsets       = nullptr;
    int* d_group_use_fp32      = nullptr;
    int* d_group_tile_offsets  = nullptr;
    int* d_group_tile_k_ids   = nullptr;
    uint16_t* d_group_tile_vals = nullptr;
    int  num_groups            = 0;
    int  num_tc_tiles          = 0;

    // FP32 fallback rows
    int* d_fp32_rows      = nullptr;
    int  num_fp32_rows    = 0;

    // Diagnostics
    float intra_community_nnz_fraction = 0.f;
    float avg_tc_tile_density          = 0.f;

    bool   active         = false;
    int    M = 0, K = 0;
    size_t plan_bytes     = 0;
};

// R7: Segment hybrid plan — per-row-segment TC/CUDA partitioning
struct RASegmentHybridPlan {
    // Row-level classification
    int* d_tc_row_ids       = nullptr;
    int* d_cuda_short_ids   = nullptr;
    int* d_cuda_long_ids    = nullptr;
    int  num_tc_rows = 0, num_cuda_short = 0, num_cuda_long = 0;

    // TC partition (grouped tiles for WMMA)
    int* d_tc_group_offsets     = nullptr;
    int* d_tc_group_use_fp32   = nullptr;
    int* d_tc_tile_offsets     = nullptr;
    int* d_tc_tile_k_ids       = nullptr;
    uint16_t* d_tc_tile_vals   = nullptr;
    int  num_tc_groups = 0, num_tc_tiles = 0;

    // TC reordering
    int* d_tc_perm_inv     = nullptr;
    int* d_tc_row_ptr_r    = nullptr;
    int* d_tc_col_r        = nullptr;
    float* d_tc_val_r      = nullptr;

    // TC FP32 fallback
    int* d_tc_fp32_rows    = nullptr;
    int  num_tc_fp32_rows  = 0;

    // CUDA partition (RoDe-style decomposition)
    int* d_cuda_short_row_ids   = nullptr;
    int* d_cuda_short_starts    = nullptr;
    int* d_cuda_short_block_nnz = nullptr;
    int  num_cuda_short_rows    = 0;

    int* d_cuda_long_row_ids    = nullptr;
    int* d_cuda_long_starts     = nullptr;
    int* d_cuda_long_block_nnz  = nullptr;
    int  num_cuda_long_rows     = 0;

    int* d_cuda_res_row_ids     = nullptr;
    int* d_cuda_res_starts      = nullptr;
    int* d_cuda_res_lengths     = nullptr;
    int  num_cuda_residual      = 0;

    // Diagnostics
    float tc_nnz_fraction       = 0.f;
    float cuda_nnz_fraction     = 0.f;

    bool   active         = false;
    int    M = 0, K = 0;
    size_t plan_bytes     = 0;
};

struct SparseMatrix {
    std::vector<int>   rowptr;
    std::vector<int>   colind;
    std::vector<float> vals;
    int M = 0, K = 0;
    float avg_nnz_per_row = 0.f;
    float std_nnz_per_row = 0.f;
    float density = 0.f;
    float skew_coeff = 0.f;
    float clustering_proxy = 0.f;
};

// ---------------------------------------------------------------------------
// OracleResult: legacy timing summary retained for compatibility
// ---------------------------------------------------------------------------
struct OracleResult {
    float csr_direct_ms;
    float csr_adaptive_ms;
    float staged_reuse_ms;
    float tc_sparse_ms;
    NextPath oracle_path;      // fastest path
    float    oracle_time_ms;   // time of fastest path
    // Router evaluation
    NextPath router_path;
    float    router_time_ms;
    float    router_slowdown;  // router_time_ms / oracle_time_ms
};

struct TimingBreakdown {
    float plan_ms  = 0.f;
    float exec_ms  = 0.f;
    float total_ms = 0.f;
    float gflops   = 0.f;
};

TimingBreakdown make_timing_breakdown(float plan_ms, float exec_ms, int nnz, int N);
float measure_cuda_exec_ms(const std::function<void()>& fn, int warmup_iters, int timed_iters);

// ---------------------------------------------------------------------------
// TC fill threshold constant
// ---------------------------------------------------------------------------
static constexpr float TC_FILL_THRESHOLD = 0.25f;
static constexpr int   TC_TILE_SIZE      = 16;
