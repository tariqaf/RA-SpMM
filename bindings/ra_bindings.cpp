// ============================================================================
// bindings_next.cpp - PyBind11 module ra_spmm
//
// Exposes all kernels, router, oracle, plan-run API, and graph generators.
// Sync policy: cudaDeviceSynchronize before returning any tensor to Python.
// ============================================================================
#include <torch/extension.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <cusparse.h>

#include "../ra_common.h"
#include "../router/router.h"

#include <vector>
#include <string>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <cmath>
#include <limits>
#include <memory>

#define CUSPARSE_CHECK_NEXT(err) do {                                      \
    cusparseStatus_t _e = (err);                                           \
    if (_e != CUSPARSE_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuSPARSE error [%s:%d]: %s\n",                    \
                __FILE__, __LINE__, cusparseGetErrorString(_e));           \
        throw std::runtime_error(std::string("cuSPARSE error: ") +         \
                                 cusparseGetErrorString(_e));              \
    }                                                                      \
} while(0)

// ---------------------------------------------------------------------------
// Forward declarations of kernel launchers
// ---------------------------------------------------------------------------
void csr_direct_spmm(const int*, const int*, const float*, const float*, float*, int, int, int);
void csr_adaptive_spmm(const int*, const int*, const float*, const float*, float*, int, int, int);
void staged_reuse_spmm(const int*, const int*, const float*, const float*, float*, int, int, int, int, int);
void tc_sparse_spmm(const int*, const int*, const float*, const float*, float*, int, int, int, TCDiagnostics&);

// Plan-run API
CSRAdaptivePlan build_csr_adaptive_plan(const int* h_rowptr, int M, int K);
void run_csr_adaptive_plan(const CSRAdaptivePlan& plan,
    const int* d_rowptr, const int* d_colind, const float* d_vals,
    const float* d_B, float* d_C, int N);
void free_csr_adaptive_plan(CSRAdaptivePlan& plan);

StagedReusePlan build_staged_reuse_plan(const int* h_rowptr, const int* h_colind,
    const float* h_vals, int M, int K, int BM, int BK);
void run_staged_reuse_plan(const StagedReusePlan& plan,
    const float* d_B, float* d_C, int N);
void free_staged_reuse_plan(StagedReusePlan& plan);

TCSparsePlan build_tc_sparse_plan(const int* h_rowptr, const int* h_colind,
    const float* h_vals, int M, int K, bool tc_eligible, bool hw_tc_supported);
void run_tc_sparse_plan(TCSparsePlan& plan, const float* d_B, float* d_C, int N);
void free_tc_sparse_plan(TCSparsePlan& plan);

// Row split
RowSplitPlan make_row_split_plan(const int* h_rowptr, int M, int K);
void run_row_split_plan(const RowSplitPlan& plan, const int* d_col, const float* d_val,
    const float* d_B, float* d_C, int N, cudaStream_t stream);
void free_row_split_plan(RowSplitPlan& plan);
// TC reordered
TCReorderedPlan make_tc_reordered_plan(const int* h_rowptr, const int* h_col,
    const float* h_val, int M, int K, int N);
void run_tc_reordered_plan(const TCReorderedPlan& plan, const float* d_B, float* d_C,
    int N, cudaStream_t stream);
void free_tc_reordered_plan(TCReorderedPlan& plan);
// Hybrid
HybridPlan make_hybrid_tc_cuda_plan(const int* h_rowptr, const int* h_col,
    const float* h_val, int M, int K, int N, float density_threshold);
void run_hybrid_tc_cuda_plan(const HybridPlan& plan, const float* d_B, float* d_C,
    int N, cudaStream_t stream);
void free_hybrid_tc_cuda_plan(HybridPlan& plan);

// ---------------------------------------------------------------------------
// Forward declarations: New regime-specific kernels (Wave 1)
// ---------------------------------------------------------------------------

// R6: Zero-overhead CSR
void make_ra_zero_overhead_plan(RAZeroOverheadPlan& plan, const int* h_rowptr, int M, int K);
void run_ra_zero_overhead_plan(const RAZeroOverheadPlan& plan, const int* d_rowptr,
    const int* d_colind, const float* d_vals, const float* d_B, float* d_C, int N);
void free_ra_zero_overhead_plan(RAZeroOverheadPlan& plan);

// R2: Vectorized coarse
void make_ra_vectorized_coarse_plan(RAVectorizedCoarsePlan& plan, const int* h_rowptr, int M, int K);
void run_ra_vectorized_coarse_plan(const RAVectorizedCoarsePlan& plan, const int* d_rowptr,
    const int* d_colind, const float* d_vals, const float* d_B, float* d_C, int N);
void free_ra_vectorized_coarse_plan(RAVectorizedCoarsePlan& plan);

// R1: RoDe-enhanced
void make_ra_rode_enhanced_plan(RARodeEnhancedPlan& plan, const int* h_rowptr, int M, int K);
void run_ra_rode_enhanced_plan(const RARodeEnhancedPlan& plan, const int* d_colind,
    const float* d_vals, const float* d_B, float* d_C, int N);
void free_ra_rode_enhanced_plan(RARodeEnhancedPlan& plan);

// R4: Flash TC
void make_ra_tc_direct_plan(RATcDirectPlan& plan, const int* h_rowptr, const int* h_col,
    const float* h_val, int M, int K, int N);
void run_ra_tc_direct_plan(const RATcDirectPlan& plan, const float* d_B, float* d_C,
    int N, cudaStream_t stream);
void free_ra_tc_direct_plan(RATcDirectPlan& plan);

// R3: Locality-tiled
void make_ra_locality_tiled_plan(RALocalityTiledPlan& plan, const int* h_rowptr,
    const int* h_col, const float* h_val, int M, int K, int N);
void run_ra_locality_tiled_plan(const RALocalityTiledPlan& plan, const float* d_B,
    float* d_C, int N, cudaStream_t stream);
void free_ra_locality_tiled_plan(RALocalityTiledPlan& plan);

// R5: Community TC
void make_ra_community_tc_plan(RACommunityTCPlan& plan, const int* h_rowptr,
    const int* h_col, const float* h_val, int M, int K, int N);
void run_ra_community_tc_plan(const RACommunityTCPlan& plan, const float* d_B,
    float* d_C, int N, cudaStream_t stream);
void free_ra_community_tc_plan(RACommunityTCPlan& plan);

// R7: Segment hybrid
void make_ra_segment_hybrid_plan(RASegmentHybridPlan& plan, const int* h_rowptr,
    const int* h_col, const float* h_val, int M, int K, int N);
void run_ra_segment_hybrid_plan(const RASegmentHybridPlan& plan, const int* d_colind,
    const float* d_vals, const float* d_B, float* d_C, int N, cudaStream_t stream);
void free_ra_segment_hybrid_plan(RASegmentHybridPlan& plan);

// Graph generators
SparseMatrix random_sparse(int M, int K, int nnz_per_row, unsigned seed);
SparseMatrix skewed_powerlaw(int M, int K, float alpha, int min_nnz, int max_nnz, unsigned seed);
SparseMatrix community_clustered(int M, int K, int n_comm, float within_density, float between_density, unsigned seed);
SparseMatrix bipartite_rectangular(int M, int K, int nnz_per_row, unsigned seed);
SparseMatrix road_like(int M, int K, int avg_degree, unsigned seed);
SparseMatrix block_locality(int M, int K, int block_size, float fill, unsigned seed);
SparseMatrix hub_heavy(int M, int K, float hub_fraction, int hub_degree, int base_degree, unsigned seed);
SparseMatrix mixed_skew(int M, int K, float frac_tiny, float frac_medium, float frac_giant,
                        int tiny_degree, int medium_degree, int giant_degree, unsigned seed);
SparseMatrix clustered_window(int M, int K, int window_rows, int window_span, float intra_window_density, unsigned seed);
SparseMatrix scrambled_locality(int M, int K, int window_rows, int window_span, float intra_window_density, unsigned seed);
SparseMatrix mixed_block_skew(int M, int K, int window_rows, float frac_block_windows, float frac_skew_windows,
                              float block_fill, int skew_base_degree, int skew_hub_degree, unsigned seed);
SparseMatrix cluster_plus_hubs(int M, int K, int num_clusters, float within_density, float between_density,
                               float hub_fraction, int hub_degree, unsigned seed);
SparseMatrix heterogeneous_windows(int M, int K, int window_rows, float frac_block_dense,
                                   float frac_clustered_sparse, float frac_random_sparse,
                                   float frac_skew_heavy, unsigned seed);
SparseMatrix reordered_variant(const SparseMatrix& mat, unsigned seed);
SparseMatrix powerlaw_realistic(int M, int m_attach, unsigned seed);
SparseMatrix community_sbm(int M, int n_comm, float within_density, float between_density, unsigned seed);

// ---------------------------------------------------------------------------
// Plan wrapper classes (non-copyable, destructor calls free)
// ---------------------------------------------------------------------------
struct CSRAdaptivePlanWrapper {
    CSRAdaptivePlan plan{};
    bool valid = false;
    ~CSRAdaptivePlanWrapper() { if (valid) free_csr_adaptive_plan(plan); }
    CSRAdaptivePlanWrapper(const CSRAdaptivePlanWrapper&) = delete;
    CSRAdaptivePlanWrapper& operator=(const CSRAdaptivePlanWrapper&) = delete;
    CSRAdaptivePlanWrapper() = default;
};


struct StagedReusePlanWrapper {
    StagedReusePlan plan{};
    bool valid = false;
    ~StagedReusePlanWrapper() { if (valid) free_staged_reuse_plan(plan); }
    StagedReusePlanWrapper(const StagedReusePlanWrapper&) = delete;
    StagedReusePlanWrapper& operator=(const StagedReusePlanWrapper&) = delete;
    StagedReusePlanWrapper() = default;
};

struct TCSparsePlanWrapper {
    TCSparsePlan plan{};
    bool valid = false;
    ~TCSparsePlanWrapper() { if (valid) free_tc_sparse_plan(plan); }
    TCSparsePlanWrapper(const TCSparsePlanWrapper&) = delete;
    TCSparsePlanWrapper& operator=(const TCSparsePlanWrapper&) = delete;
    TCSparsePlanWrapper() = default;
};

struct RowSplitPlanWrapper {
    RowSplitPlan plan{};
    bool valid = false;
    ~RowSplitPlanWrapper() { if (valid) free_row_split_plan(plan); }
    RowSplitPlanWrapper(const RowSplitPlanWrapper&) = delete;
    RowSplitPlanWrapper& operator=(const RowSplitPlanWrapper&) = delete;
    RowSplitPlanWrapper() = default;
};

struct TCReorderedPlanWrapper {
    TCReorderedPlan plan{};
    bool valid = false;
    ~TCReorderedPlanWrapper() { if (valid) free_tc_reordered_plan(plan); }
    TCReorderedPlanWrapper(const TCReorderedPlanWrapper&) = delete;
    TCReorderedPlanWrapper& operator=(const TCReorderedPlanWrapper&) = delete;
    TCReorderedPlanWrapper() = default;
};

struct HybridPlanWrapper {
    HybridPlan plan{};
    bool valid = false;
    ~HybridPlanWrapper() { if (valid) free_hybrid_tc_cuda_plan(plan); }
    HybridPlanWrapper(const HybridPlanWrapper&) = delete;
    HybridPlanWrapper& operator=(const HybridPlanWrapper&) = delete;
    HybridPlanWrapper() = default;
};

// --- New regime-specific plan wrappers ---

struct RAZeroOverheadPlanWrapper {
    RAZeroOverheadPlan plan{};
    bool valid = false;
    ~RAZeroOverheadPlanWrapper() { if (valid) free_ra_zero_overhead_plan(plan); }
    RAZeroOverheadPlanWrapper(const RAZeroOverheadPlanWrapper&) = delete;
    RAZeroOverheadPlanWrapper& operator=(const RAZeroOverheadPlanWrapper&) = delete;
    RAZeroOverheadPlanWrapper() = default;
};

struct RAVectorizedCoarsePlanWrapper {
    RAVectorizedCoarsePlan plan{};
    bool valid = false;
    ~RAVectorizedCoarsePlanWrapper() { if (valid) free_ra_vectorized_coarse_plan(plan); }
    RAVectorizedCoarsePlanWrapper(const RAVectorizedCoarsePlanWrapper&) = delete;
    RAVectorizedCoarsePlanWrapper& operator=(const RAVectorizedCoarsePlanWrapper&) = delete;
    RAVectorizedCoarsePlanWrapper() = default;
};

struct RARodeEnhancedPlanWrapper {
    RARodeEnhancedPlan plan{};
    bool valid = false;
    ~RARodeEnhancedPlanWrapper() { if (valid) free_ra_rode_enhanced_plan(plan); }
    RARodeEnhancedPlanWrapper(const RARodeEnhancedPlanWrapper&) = delete;
    RARodeEnhancedPlanWrapper& operator=(const RARodeEnhancedPlanWrapper&) = delete;
    RARodeEnhancedPlanWrapper() = default;
};

struct RATcDirectPlanWrapper {
    RATcDirectPlan plan{};
    bool valid = false;
    ~RATcDirectPlanWrapper() { if (valid) free_ra_tc_direct_plan(plan); }
    RATcDirectPlanWrapper(const RATcDirectPlanWrapper&) = delete;
    RATcDirectPlanWrapper& operator=(const RATcDirectPlanWrapper&) = delete;
    RATcDirectPlanWrapper() = default;
};

struct RALocalityTiledPlanWrapper {
    RALocalityTiledPlan plan{};
    bool valid = false;
    ~RALocalityTiledPlanWrapper() { if (valid) free_ra_locality_tiled_plan(plan); }
    RALocalityTiledPlanWrapper(const RALocalityTiledPlanWrapper&) = delete;
    RALocalityTiledPlanWrapper& operator=(const RALocalityTiledPlanWrapper&) = delete;
    RALocalityTiledPlanWrapper() = default;
};

struct RACommunityTCPlanWrapper {
    RACommunityTCPlan plan{};
    bool valid = false;
    ~RACommunityTCPlanWrapper() { if (valid) free_ra_community_tc_plan(plan); }
    RACommunityTCPlanWrapper(const RACommunityTCPlanWrapper&) = delete;
    RACommunityTCPlanWrapper& operator=(const RACommunityTCPlanWrapper&) = delete;
    RACommunityTCPlanWrapper() = default;
};

struct RASegmentHybridPlanWrapper {
    RASegmentHybridPlan plan{};
    bool valid = false;
    ~RASegmentHybridPlanWrapper() { if (valid) free_ra_segment_hybrid_plan(plan); }
    RASegmentHybridPlanWrapper(const RASegmentHybridPlanWrapper&) = delete;
    RASegmentHybridPlanWrapper& operator=(const RASegmentHybridPlanWrapper&) = delete;
    RASegmentHybridPlanWrapper() = default;
};

// ---------------------------------------------------------------------------
// Helper: validate GPU tensor with dtype enforcement
// ---------------------------------------------------------------------------
static void check_gpu_tensor(const torch::Tensor& t, const char* name) {
    TORCH_CHECK(t.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
}

static void check_dense_float_tensor(const torch::Tensor& t, const char* name) {
    check_gpu_tensor(t, name);
    TORCH_CHECK(t.dtype() == torch::kFloat32, name, " must be float32");
}

static void check_csr_tensors(const torch::Tensor& rowptr, const torch::Tensor& colind,
                               const torch::Tensor& vals) {
    check_gpu_tensor(rowptr, "rowptr");
    check_gpu_tensor(colind, "colind");
    check_gpu_tensor(vals, "vals");
    TORCH_CHECK(rowptr.dtype() == torch::kInt32, "rowptr must be int32");
    TORCH_CHECK(colind.dtype() == torch::kInt32, "colind must be int32");
    TORCH_CHECK(vals.dtype() == torch::kFloat32, "vals must be float32");
}

// ---------------------------------------------------------------------------
// Helper: convert SparseMatrix to Python dict
// ---------------------------------------------------------------------------
static pybind11::dict mat_to_dict(const SparseMatrix& mat) {
    pybind11::dict d;

    auto rowptr_t = torch::empty({static_cast<i64>(mat.rowptr.size())},
                                 torch::TensorOptions().dtype(torch::kInt32));
    auto colind_t = torch::empty({static_cast<i64>(mat.colind.size())},
                                 torch::TensorOptions().dtype(torch::kInt32));
    auto vals_t = torch::empty({static_cast<i64>(mat.vals.size())},
                               torch::TensorOptions().dtype(torch::kFloat32));
    if (!mat.rowptr.empty()) {
        std::memcpy(rowptr_t.data_ptr<int>(), mat.rowptr.data(), mat.rowptr.size() * sizeof(int));
    }
    if (!mat.colind.empty()) {
        std::memcpy(colind_t.data_ptr<int>(), mat.colind.data(), mat.colind.size() * sizeof(int));
    }
    if (!mat.vals.empty()) {
        std::memcpy(vals_t.data_ptr<float>(), mat.vals.data(), mat.vals.size() * sizeof(float));
    }

    d["rowptr"]           = rowptr_t;
    d["colind"]           = colind_t;
    d["vals"]             = vals_t;
    d["M"]                = mat.M;
    d["K"]                = mat.K;
    d["nnz"]              = (int)mat.colind.size();
    d["avg_nnz_per_row"]  = mat.avg_nnz_per_row;
    d["std_nnz_per_row"]  = mat.std_nnz_per_row;
    d["density"]          = mat.density;
    d["skew_coeff"]       = mat.skew_coeff;
    d["clustering_proxy"] = mat.clustering_proxy;
    return d;
}

struct HostCSR {
    std::vector<int> rowptr;
    std::vector<int> colind;
    std::vector<float> vals;
};

static HostCSR copy_csr_to_host(
    const torch::Tensor& rowptr,
    const torch::Tensor& colind,
    const torch::Tensor& vals)
{
    HostCSR host;

    auto rp = rowptr.cpu().contiguous().to(torch::kInt32);
    auto ci = colind.cpu().contiguous().to(torch::kInt32);
    auto vl = vals.cpu().contiguous().to(torch::kFloat32);

    host.rowptr.assign(rp.data_ptr<int>(), rp.data_ptr<int>() + rp.numel());
    host.colind.assign(ci.data_ptr<int>(), ci.data_ptr<int>() + ci.numel());
    host.vals.assign(vl.data_ptr<float>(), vl.data_ptr<float>() + vl.numel());
    return host;
}

static pybind11::dict timing_to_dict(const TimingBreakdown& timing) {
    pybind11::dict d;
    d["plan_ms"] = timing.plan_ms;
    d["exec_ms"] = timing.exec_ms;
    d["total_ms"] = timing.total_ms;
    d["gflops"] = timing.gflops;
    return d;
}

static pybind11::dict path_timing_to_dict(
    const std::string& path_name,
    const TimingBreakdown& timing,
    bool in_main,
    bool is_legacy)
{
    pybind11::dict d = timing_to_dict(timing);
    d["path"] = path_name;
    d["in_main_portfolio"] = in_main;
    d["legacy_baseline"] = is_legacy;
    return d;
}

static cusparseSpMMAlg_t default_cusparse_spmm_alg() {
#if defined(CUSPARSE_SPMM_CSR_ALG2)
    return CUSPARSE_SPMM_CSR_ALG2;
#else
    return CUSPARSE_SPMM_ALG_DEFAULT;
#endif
}

static const char* cusparse_spmm_alg_name(cusparseSpMMAlg_t alg) {
    switch (alg) {
        case CUSPARSE_SPMM_ALG_DEFAULT: return "CUSPARSE_SPMM_ALG_DEFAULT";
#if defined(CUSPARSE_SPMM_CSR_ALG1)
        case CUSPARSE_SPMM_CSR_ALG1: return "CUSPARSE_SPMM_CSR_ALG1";
#endif
#if defined(CUSPARSE_SPMM_CSR_ALG2)
        case CUSPARSE_SPMM_CSR_ALG2: return "CUSPARSE_SPMM_CSR_ALG2";
#endif
#if defined(CUSPARSE_SPMM_CSR_ALG3)
        case CUSPARSE_SPMM_CSR_ALG3: return "CUSPARSE_SPMM_CSR_ALG3";
#endif
        default: return "CUSPARSE_SPMM_UNKNOWN";
    }
}

class CuSparseSpMMState {
public:
    CuSparseSpMMState(
        const torch::Tensor& rowptr,
        const torch::Tensor& colind,
        const torch::Tensor& vals,
        const torch::Tensor& B)
    {
        init(rowptr, colind, vals, B);
    }

    ~CuSparseSpMMState() {
        cleanup();
    }

    CuSparseSpMMState(const CuSparseSpMMState&) = delete;
    CuSparseSpMMState& operator=(const CuSparseSpMMState&) = delete;

    void run() {
        CUSPARSE_CHECK_NEXT(cusparseSpMM(
            handle_,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha_,
            matA_,
            matB_,
            &beta_,
            matC_,
            CUDA_R_32F,
            alg_,
            buffer_.data_ptr()));
    }

    const torch::Tensor& output() const {
        return C_;
    }

    const char* algorithm_name() const {
        return cusparse_spmm_alg_name(alg_);
    }

private:
    void init(
        const torch::Tensor& rowptr,
        const torch::Tensor& colind,
        const torch::Tensor& vals,
        const torch::Tensor& B)
    {
        try {
            const int64_t M = rowptr.size(0) - 1;
            const int64_t K = B.size(0);
            const int64_t N = B.size(1);
            const int64_t nnz = colind.numel();

            C_ = torch::zeros({M, N}, B.options().dtype(torch::kFloat32));

            CUSPARSE_CHECK_NEXT(cusparseCreate(&handle_));
            CUSPARSE_CHECK_NEXT(cusparseCreateCsr(
                &matA_,
                M,
                K,
                nnz,
                rowptr.data_ptr<int>(),
                colind.data_ptr<int>(),
                vals.data_ptr<float>(),
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F));
            CUSPARSE_CHECK_NEXT(cusparseCreateDnMat(
                &matB_,
                K,
                N,
                N,
                const_cast<float*>(B.data_ptr<float>()),
                CUDA_R_32F,
                CUSPARSE_ORDER_ROW));
            CUSPARSE_CHECK_NEXT(cusparseCreateDnMat(
                &matC_,
                M,
                N,
                N,
                C_.data_ptr<float>(),
                CUDA_R_32F,
                CUSPARSE_ORDER_ROW));

            size_t buffer_size = 0;
            CUSPARSE_CHECK_NEXT(cusparseSpMM_bufferSize(
                handle_,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_,
                matA_,
                matB_,
                &beta_,
                matC_,
                CUDA_R_32F,
                alg_,
                &buffer_size));

            const int64_t alloc_bytes = static_cast<int64_t>(std::max<size_t>(buffer_size, 1));
            buffer_ = torch::empty({alloc_bytes}, B.options().dtype(torch::kUInt8));
            CUSPARSE_CHECK_NEXT(cusparseSpMM_preprocess(
                handle_,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                CUSPARSE_OPERATION_NON_TRANSPOSE,
                &alpha_,
                matA_,
                matB_,
                &beta_,
                matC_,
                CUDA_R_32F,
                alg_,
                buffer_.data_ptr()));
        } catch (...) {
            cleanup();
            throw;
        }
    }

    void cleanup() noexcept {
        if (matC_ != nullptr) {
            cusparseDestroyDnMat(matC_);
            matC_ = nullptr;
        }
        if (matB_ != nullptr) {
            cusparseDestroyDnMat(matB_);
            matB_ = nullptr;
        }
        if (matA_ != nullptr) {
            cusparseDestroySpMat(matA_);
            matA_ = nullptr;
        }
        if (handle_ != nullptr) {
            cusparseDestroy(handle_);
            handle_ = nullptr;
        }
    }

    cusparseHandle_t handle_ = nullptr;
    cusparseSpMatDescr_t matA_ = nullptr;
    cusparseDnMatDescr_t matB_ = nullptr;
    cusparseDnMatDescr_t matC_ = nullptr;
    torch::Tensor buffer_;
    torch::Tensor C_;
    float alpha_ = 1.0f;
    float beta_ = 0.0f;
    cusparseSpMMAlg_t alg_ = default_cusparse_spmm_alg();
};

// ---------------------------------------------------------------------------
// Helper: convert RouterPlan to Python dict
// ---------------------------------------------------------------------------
static pybind11::dict plan_to_dict(const RouterPlan& plan) {
    pybind11::dict d;
    d["chosen_path"]      = std::string(next_path_name(plan.chosen_path));
    d["chosen_path_int"]  = (int)plan.chosen_path;
    d["decision_reason"]  = plan.decision_reason;
    d["estimated_risk"]   = plan.estimated_risk;
    d["planning_time_ms"] = plan.planning_time_ms;
    d["gate_margin_raw"]  = plan.gate_margin_raw;
    d["gate_margin_norm"] = plan.gate_margin_norm;

    d["path_count"] = NEXT_PATH_COUNT;

    pybind11::list feasible_list;
    pybind11::list path_names;
    pybind11::dict feasible_by_path;
    pybind11::dict reject_codes_by_path;
    pybind11::dict reject_details_by_path;
    for (NextPath path : kAllNextPaths) {
        const int idx = (int)path;
        const std::string name = next_path_name(path);
        path_names.append(name);
        feasible_list.append(plan.feasible[idx]);
        feasible_by_path[name.c_str()] = plan.feasible[idx];
        reject_codes_by_path[name.c_str()] = std::string(reject_reason_str(plan.rejection_code[idx]));
        reject_details_by_path[name.c_str()] = plan.rejection_detail[idx];
    }
    d["path_names"] = path_names;
    d["feasible"] = feasible_list;
    d["feasible_by_path"] = feasible_by_path;
    d["rejection_codes"] = reject_codes_by_path;
    d["rejection_details"] = reject_details_by_path;

    // Feature values
    pybind11::dict feat;
    feat["avg_nnz_per_row"]      = plan.features.avg_nnz_per_row;
    feat["std_nnz_per_row"]      = plan.features.std_nnz_per_row;
    feat["degree_cv"]            = plan.features.degree_cv;
    feat["max_to_mean_ratio"]    = plan.features.max_to_mean_ratio;
    feat["frac_dense_rows"]      = plan.features.frac_dense_rows;
    feat["skew_ratio"]           = plan.features.skew_ratio;
    feat["long_row_fraction"]    = plan.features.long_row_fraction;
    feat["long_row_nnz_fraction"] = plan.features.long_row_nnz_fraction;
    feat["top_1_row_nnz_fraction"] = plan.features.top_1_row_nnz_fraction;
    feat["top_5_row_nnz_fraction"] = plan.features.top_5_row_nnz_fraction;
    feat["tile_fill_mean"]       = plan.features.tile_fill_mean;
    feat["tile_fill_median"]     = plan.features.tile_fill_median;
    feat["tile_fill_p90"]        = plan.features.tile_fill_p90;
    feat["tile_fill_max"]        = plan.features.tile_fill_max;
    feat["tile_fill_variance"]   = plan.features.tile_fill_variance;
    feat["tile_occupancy"]       = plan.features.tile_occupancy;
    feat["actual_nnz_coverage"]  = plan.features.actual_nnz_coverage;
    feat["avg_nnz_per_tile"]     = plan.features.avg_nnz_per_tile;
    feat["row_window_colspan_compactness"] = plan.features.row_window_colspan_compactness;
    feat["local_row_similarity_proxy"] = plan.features.local_row_similarity_proxy;
    feat["reordered_locality_proxy"] = plan.features.reordered_locality_proxy;
    feat["locality_gain_proxy"] = plan.features.locality_gain_proxy;
    feat["locality_selectivity_proxy"] = plan.features.locality_selectivity_proxy;
    feat["road_likeness_proxy"] = plan.features.road_likeness_proxy;
    feat["row_split_affinity_proxy"] = plan.features.row_split_affinity_proxy;
    feat["mixedness_proxy"] = plan.features.mixedness_proxy;
    feat["tc_candidate_ratio"]   = plan.features.tc_candidate_ratio;
    feat["tc_synergy_proxy"]     = plan.features.tc_synergy_proxy;
    feat["estimated_tc_partition_ratio"] = plan.features.estimated_tc_partition_ratio;
    feat["estimated_cuda_partition_ratio"] = plan.features.estimated_cuda_partition_ratio;
    feat["irregular_window_fraction"] = plan.features.irregular_window_fraction;
    feat["tc_granularity_proxy"] = plan.features.tc_granularity_proxy;
    feat["redundancy_risk_proxy"] = plan.features.redundancy_risk_proxy;
    feat["tc_candidate_tiles"]   = plan.features.tc_candidate_tiles;
    d["feature_values"] = feat;

    // Scores (diagnostic only)
    pybind11::dict scores;
    scores["csr_direct"]    = plan.scores.csr_direct_score;
    scores["csr_adaptive"]  = plan.scores.csr_adaptive_score;
    scores["staged_reuse"]  = plan.scores.staged_reuse_score;
    scores["tc_sparse"]     = plan.scores.tc_sparse_score;
    scores["row_split_cuda"] = plan.scores.row_split_cuda_score;
    scores["tc_reordered"] = plan.scores.tc_reordered_score;
    scores["hybrid_tc_cuda"] = plan.scores.hybrid_tc_cuda_score;
    scores["cusparse"]       = plan.scores.cusparse_score;
    d["scores"] = scores;

    return d;
}

// ---------------------------------------------------------------------------
// Core kernel wrappers (with sync before returning tensor to Python)
// ---------------------------------------------------------------------------

torch::Tensor spmm_csr_direct_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B)
{
    check_csr_tensors(rowptr, colind, vals);
    check_gpu_tensor(B, "B");

    int M = (int)rowptr.size(0) - 1;
    int K = (int)B.size(0);
    int N = (int)B.size(1);

    auto C = torch::zeros({M, N}, B.options());

    csr_direct_spmm(
        rowptr.data_ptr<int>(), colind.data_ptr<int>(),
        vals.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, K, N);

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

torch::Tensor spmm_csr_adaptive_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B)
{
    check_csr_tensors(rowptr, colind, vals);
    check_gpu_tensor(B, "B");

    int M = (int)rowptr.size(0) - 1;
    int K = (int)B.size(0);
    int N = (int)B.size(1);

    auto C = torch::zeros({M, N}, B.options());

    csr_adaptive_spmm(
        rowptr.data_ptr<int>(), colind.data_ptr<int>(),
        vals.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, K, N);

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}


torch::Tensor spmm_staged_reuse_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B)
{
    check_csr_tensors(rowptr, colind, vals);
    check_gpu_tensor(B, "B");

    int M = (int)rowptr.size(0) - 1;
    int K = (int)B.size(0);
    int N = (int)B.size(1);

    auto C = torch::zeros({M, N}, B.options());

    staged_reuse_spmm(
        rowptr.data_ptr<int>(), colind.data_ptr<int>(),
        vals.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, K, N, 64, 64);

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

std::pair<torch::Tensor, pybind11::dict> spmm_tc_sparse_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B)
{
    check_csr_tensors(rowptr, colind, vals);
    check_gpu_tensor(B, "B");

    int M = (int)rowptr.size(0) - 1;
    int K = (int)B.size(0);
    int N = (int)B.size(1);

    auto C = torch::zeros({M, N}, B.options());

    TCDiagnostics diag;
    tc_sparse_spmm(
        rowptr.data_ptr<int>(), colind.data_ptr<int>(),
        vals.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), M, K, N, diag);

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());

    pybind11::dict diag_dict;
    diag_dict["tc_candidate_tiles"] = diag.tc_candidate_tiles;
    diag_dict["tc_activated_tiles"] = diag.tc_activated_tiles;
    diag_dict["tc_rejected_tiles"]  = diag.tc_rejected_tiles;
    diag_dict["tc_fill_avg"]        = diag.tc_fill_avg;
    diag_dict["hw_tc_supported"]    = diag.hw_tc_supported;
    diag_dict["tc_path_taken"]      = diag.tc_path_taken;

    return {C, diag_dict};
}

torch::Tensor spmm_cusparse_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B)
{
    check_csr_tensors(rowptr, colind, vals);
    check_dense_float_tensor(B, "B");

    CuSparseSpMMState state(rowptr, colind, vals, B);
    state.run();

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return state.output();
}

pybind11::dict benchmark_cusparse_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B,
    int warmup, int iters)
{
    check_csr_tensors(rowptr, colind, vals);
    check_dense_float_tensor(B, "B");

    const int nnz = static_cast<int>(colind.numel());
    const int N = static_cast<int>(B.size(1));
    CuSparseSpMMState state(rowptr, colind, vals, B);
    const float exec_ms = measure_cuda_exec_ms(
        [&]() { state.run(); },
        warmup,
        std::max(1, iters));

    pybind11::dict d = timing_to_dict(make_timing_breakdown(0.f, exec_ms, nnz, N));
    d["path"] = "CUSPARSE";
    d["in_main_portfolio"] = false;
    d["legacy_baseline"] = false;
    d["external_baseline"] = true;
    d["cusparse_algorithm"] = std::string(state.algorithm_name());
    d["preprocess_ms"] = 0.0f;
    return d;
}

pybind11::dict benchmark_cusparse_cold_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B,
    int iters)
{
    check_csr_tensors(rowptr, colind, vals);
    check_dense_float_tensor(B, "B");

    const int nnz = static_cast<int>(colind.numel());
    const int N = static_cast<int>(B.size(1));
    const int actual_iters = std::max(1, iters);
    double plan_sum = 0.0;
    double exec_sum = 0.0;
    std::string algorithm_name;
    for (int i = 0; i < actual_iters; ++i) {
        CUDA_CHECK_NEXT(cudaDeviceSynchronize());
        const auto t0 = std::chrono::high_resolution_clock::now();
        CuSparseSpMMState state(rowptr, colind, vals, B);
        CUDA_CHECK_NEXT(cudaDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();
        const float exec_ms = measure_cuda_exec_ms([&]() { state.run(); }, 0, 1);
        if (algorithm_name.empty()) {
            algorithm_name = state.algorithm_name();
        }
        plan_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
        exec_sum += exec_ms;
    }

    const TimingBreakdown timing = make_timing_breakdown(
        static_cast<float>(plan_sum / actual_iters),
        static_cast<float>(exec_sum / actual_iters),
        nnz,
        N);
    pybind11::dict d = timing_to_dict(timing);
    d["path"] = "CUSPARSE";
    d["in_main_portfolio"] = true;
    d["legacy_baseline"] = false;
    d["external_baseline"] = false;
    d["cusparse_algorithm"] = algorithm_name;
    d["preprocess_ms"] = timing.plan_ms;
    return d;
}

// ---------------------------------------------------------------------------
// Plan-run bindings
// ---------------------------------------------------------------------------

std::shared_ptr<CSRAdaptivePlanWrapper> make_csr_adaptive_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu, int M, int K)
{
    (void)colind_cpu;
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto w = std::make_shared<CSRAdaptivePlanWrapper>();
    (void)colind_cpu;
    w->plan = build_csr_adaptive_plan(rp.data_ptr<int>(), M, K);
    w->valid = true;
    return w;
}


torch::Tensor run_csr_adaptive_plan_fn(
    std::shared_ptr<CSRAdaptivePlanWrapper> wrapper,
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B)
{
    check_csr_tensors(rowptr, colind, vals);
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid CSR adaptive plan");

    int M = wrapper->plan.M;
    int N = (int)B.size(1);

    auto C = torch::zeros({M, N}, B.options());

    run_csr_adaptive_plan(wrapper->plan,
        rowptr.data_ptr<int>(), colind.data_ptr<int>(),
        vals.data_ptr<float>(), B.data_ptr<float>(),
        C.data_ptr<float>(), N);

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}


std::shared_ptr<StagedReusePlanWrapper> make_staged_reuse_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto ci = colind_cpu.cpu().contiguous().to(torch::kInt32);
    auto vl = vals_cpu.cpu().contiguous().to(torch::kFloat32);

    auto w = std::make_shared<StagedReusePlanWrapper>();
    w->plan = build_staged_reuse_plan(rp.data_ptr<int>(), ci.data_ptr<int>(),
                                       vl.data_ptr<float>(), M, K, 64, 64);
    w->valid = true;
    return w;
}

torch::Tensor run_staged_reuse_plan_fn(
    std::shared_ptr<StagedReusePlanWrapper> wrapper,
    torch::Tensor B)
{
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid staged reuse plan");

    int M = wrapper->plan.M;
    int N = (int)B.size(1);

    auto C = torch::zeros({M, N}, B.options());

    run_staged_reuse_plan(wrapper->plan, B.data_ptr<float>(), C.data_ptr<float>(), N);

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

std::shared_ptr<TCSparsePlanWrapper> make_tc_sparse_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K, bool tc_eligible)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto ci = colind_cpu.cpu().contiguous().to(torch::kInt32);
    auto vl = vals_cpu.cpu().contiguous().to(torch::kFloat32);

    // Check hw support
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    bool hw_tc = (prop.major >= 7);

    auto w = std::make_shared<TCSparsePlanWrapper>();
    w->plan = build_tc_sparse_plan(rp.data_ptr<int>(), ci.data_ptr<int>(),
                                    vl.data_ptr<float>(), M, K, tc_eligible, hw_tc);
    w->valid = true;
    return w;
}

torch::Tensor run_tc_sparse_plan_fn(
    std::shared_ptr<TCSparsePlanWrapper> wrapper,
    torch::Tensor B)
{
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid TC sparse plan");

    int M = wrapper->plan.M;
    int N = (int)B.size(1);

    auto C = torch::zeros({M, N}, B.options());

    run_tc_sparse_plan(wrapper->plan, B.data_ptr<float>(), C.data_ptr<float>(), N);

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

std::shared_ptr<RowSplitPlanWrapper> make_row_split_plan_fn(
    torch::Tensor rowptr_cpu, int M, int K)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto w = std::make_shared<RowSplitPlanWrapper>();
    w->plan = make_row_split_plan(rp.data_ptr<int>(), M, K);
    w->valid = true;
    return w;
}

static std::vector<int> copy_device_ints(const int* d_ptr, int count) {
    std::vector<int> host(std::max(0, count), 0);
    if (d_ptr != nullptr && count > 0) {
        CUDA_CHECK_NEXT(cudaMemcpy(host.data(), d_ptr, count * sizeof(int), cudaMemcpyDeviceToHost));
    }
    return host;
}

static int* copy_host_ints_to_device(const std::vector<int>& host) {
    if (host.empty()) {
        return nullptr;
    }
    int* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, host.size() * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, host.data(), host.size() * sizeof(int), cudaMemcpyHostToDevice));
    return d_ptr;
}

std::shared_ptr<RowSplitPlanWrapper> make_row_split_plan_no_long_rows_fn(
    torch::Tensor rowptr_cpu, int M, int K)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto w = std::make_shared<RowSplitPlanWrapper>();
    w->plan = make_row_split_plan(rp.data_ptr<int>(), M, K);
    w->valid = true;

    RowSplitPlan& plan = w->plan;
    const std::vector<int> short_rows = copy_device_ints(plan.d_short_row_ids, plan.num_short_rows);
    const std::vector<int> short_starts = copy_device_ints(plan.d_short_starts, plan.num_short_rows);
    const std::vector<int> short_blocks = copy_device_ints(plan.d_short_block_nnz, plan.num_short_rows);
    const std::vector<int> long_rows = copy_device_ints(plan.d_long_row_ids, plan.num_long_rows);
    const std::vector<int> long_starts = copy_device_ints(plan.d_long_starts, plan.num_long_rows);
    const std::vector<int> long_blocks = copy_device_ints(plan.d_long_block_nnz, plan.num_long_rows);

    std::vector<int> merged_rows = short_rows;
    std::vector<int> merged_starts = short_starts;
    std::vector<int> merged_blocks = short_blocks;
    merged_rows.insert(merged_rows.end(), long_rows.begin(), long_rows.end());
    merged_starts.insert(merged_starts.end(), long_starts.begin(), long_starts.end());
    merged_blocks.insert(merged_blocks.end(), long_blocks.begin(), long_blocks.end());

    if (plan.d_short_row_ids != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_short_row_ids));
    if (plan.d_short_starts != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_short_starts));
    if (plan.d_short_block_nnz != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_short_block_nnz));
    if (plan.d_long_row_ids != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_long_row_ids));
    if (plan.d_long_starts != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_long_starts));
    if (plan.d_long_block_nnz != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_long_block_nnz));
    if (plan.d_long_num_segments != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_long_num_segments));
    if (plan.d_long_seg_row_ids != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_long_seg_row_ids));
    if (plan.d_long_seg_starts != nullptr) CUDA_CHECK_NEXT(cudaFree(plan.d_long_seg_starts));

    plan.d_short_row_ids = copy_host_ints_to_device(merged_rows);
    plan.d_short_starts = copy_host_ints_to_device(merged_starts);
    plan.d_short_block_nnz = copy_host_ints_to_device(merged_blocks);
    plan.num_short_rows = static_cast<int>(merged_rows.size());

    plan.d_long_row_ids = nullptr;
    plan.d_long_starts = nullptr;
    plan.d_long_block_nnz = nullptr;
    plan.d_long_num_segments = nullptr;
    plan.d_long_seg_row_ids = nullptr;
    plan.d_long_seg_starts = nullptr;
    plan.num_long_rows = 0;
    plan.num_long_segments = 0;
    plan.num_split_long_rows = 0;
    plan.avg_segments_per_long_row = 0.f;
    return w;
}

torch::Tensor run_row_split_plan_fn(
    std::shared_ptr<RowSplitPlanWrapper> wrapper,
    torch::Tensor colind, torch::Tensor vals, torch::Tensor B)
{
    check_gpu_tensor(colind, "colind"); check_gpu_tensor(vals, "vals");
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid RowSplit plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_row_split_plan(wrapper->plan,
        colind.data_ptr<int>(), vals.data_ptr<float>(),
        B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

std::shared_ptr<TCReorderedPlanWrapper> make_tc_reordered_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K, int N)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto ci = colind_cpu.cpu().contiguous().to(torch::kInt32);
    auto vl = vals_cpu.cpu().contiguous().to(torch::kFloat32);
    auto w = std::make_shared<TCReorderedPlanWrapper>();
    w->plan = make_tc_reordered_plan(rp.data_ptr<int>(), ci.data_ptr<int>(),
                                      vl.data_ptr<float>(), M, K, N);
    w->valid = true;
    return w;
}

torch::Tensor run_tc_reordered_plan_fn(
    std::shared_ptr<TCReorderedPlanWrapper> wrapper, torch::Tensor B)
{
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid TCReordered plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_tc_reordered_plan(wrapper->plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

std::shared_ptr<HybridPlanWrapper> make_hybrid_tc_cuda_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K, int N, float threshold)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto ci = colind_cpu.cpu().contiguous().to(torch::kInt32);
    auto vl = vals_cpu.cpu().contiguous().to(torch::kFloat32);
    auto w = std::make_shared<HybridPlanWrapper>();
    w->plan = make_hybrid_tc_cuda_plan(rp.data_ptr<int>(), ci.data_ptr<int>(),
                                        vl.data_ptr<float>(), M, K, N, threshold);
    w->valid = true;
    return w;
}

torch::Tensor run_hybrid_tc_cuda_plan_fn(
    std::shared_ptr<HybridPlanWrapper> wrapper, torch::Tensor B)
{
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid Hybrid plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_hybrid_tc_cuda_plan(wrapper->plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

std::shared_ptr<HybridPlanWrapper> make_hybrid_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K, int N, float threshold)
{
    return make_hybrid_tc_cuda_plan_fn(rowptr_cpu, colind_cpu, vals_cpu, M, K, N, threshold);
}

torch::Tensor run_hybrid_plan_fn(
    std::shared_ptr<HybridPlanWrapper> wrapper, torch::Tensor B)
{
    return run_hybrid_tc_cuda_plan_fn(wrapper, B);
}

// ---------------------------------------------------------------------------
// Router / oracle timing helpers
// ---------------------------------------------------------------------------

static bool is_main_path(NextPath path) {
    return path == NextPath::CSR_DIRECT ||
           path == NextPath::ROW_SPLIT_CUDA ||
           path == NextPath::TC_REORDERED ||
           path == NextPath::HYBRID_TC_CUDA ||
           path == NextPath::CUSPARSE;
}

static bool is_legacy_path(NextPath path) {
    return !is_main_path(path);
}

static TimingBreakdown infinite_timing() {
    TimingBreakdown timing;
    timing.plan_ms = std::numeric_limits<float>::infinity();
    timing.exec_ms = std::numeric_limits<float>::infinity();
    timing.total_ms = std::numeric_limits<float>::infinity();
    timing.gflops = 0.f;
    return timing;
}

template <typename BuildFn, typename RunFn, typename FreeFn, typename ValidFn>
static TimingBreakdown measure_reusable_cold(
    BuildFn build_fn,
    RunFn run_fn,
    FreeFn free_fn,
    ValidFn valid_fn,
    int iters,
    int nnz,
    int N)
{
    if (iters <= 0) {
        return TimingBreakdown{};
    }

    double plan_sum = 0.0;
    double exec_sum = 0.0;
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK_NEXT(cudaDeviceSynchronize());
        const auto t0 = std::chrono::high_resolution_clock::now();
        auto plan = build_fn();
        CUDA_CHECK_NEXT(cudaDeviceSynchronize());
        const auto t1 = std::chrono::high_resolution_clock::now();
        if (!valid_fn(plan)) {
            free_fn(plan);
            return infinite_timing();
        }
        const float exec_ms = measure_cuda_exec_ms([&]() { run_fn(plan); }, 0, 1);
        free_fn(plan);
        plan_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
        exec_sum += exec_ms;
    }

    return make_timing_breakdown(
        static_cast<float>(plan_sum / static_cast<double>(iters)),
        static_cast<float>(exec_sum / static_cast<double>(iters)),
        nnz, N);
}

template <typename BuildFn, typename RunFn, typename FreeFn, typename ValidFn>
static TimingBreakdown measure_reusable_warm(
    BuildFn build_fn,
    RunFn run_fn,
    FreeFn free_fn,
    ValidFn valid_fn,
    int warmup,
    int iters,
    int nnz,
    int N)
{
    auto plan = build_fn();
    if (!valid_fn(plan)) {
        free_fn(plan);
        return infinite_timing();
    }
    const float exec_ms = measure_cuda_exec_ms([&]() { run_fn(plan); }, warmup, iters);
    free_fn(plan);
    return make_timing_breakdown(0.f, exec_ms, nnz, N);
}

static TimingBreakdown measure_path_timing(
    NextPath path,
    const HostCSR& host,
    torch::Tensor rowptr,
    torch::Tensor colind,
    torch::Tensor vals,
    torch::Tensor B,
    bool cold_mode,
    int warmup,
    int iters)
{
    const int M = static_cast<int>(rowptr.size(0)) - 1;
    const int K = static_cast<int>(B.size(0));
    const int N = static_cast<int>(B.size(1));
    const int nnz = static_cast<int>(colind.numel());

    if (path == NextPath::CSR_DIRECT) {
        const float exec_ms = measure_cuda_exec_ms([&]() {
            auto C = torch::zeros({M, N}, B.options());
            csr_direct_spmm(rowptr.data_ptr<int>(), colind.data_ptr<int>(),
                            vals.data_ptr<float>(), B.data_ptr<float>(),
                            C.data_ptr<float>(), M, K, N);
        }, warmup, std::max(1, iters));
        return make_timing_breakdown(0.f, exec_ms, nnz, N);
    }

    if (path == NextPath::CSR_ADAPTIVE) {
        auto build_fn = [&]() { return build_csr_adaptive_plan(host.rowptr.data(), M, K); };
        auto run_fn = [&](const CSRAdaptivePlan& plan) {
            auto C = torch::zeros({M, N}, B.options());
            run_csr_adaptive_plan(plan, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
                                  vals.data_ptr<float>(), B.data_ptr<float>(),
                                  C.data_ptr<float>(), N);
        };
        auto free_fn = [&](CSRAdaptivePlan& plan) { free_csr_adaptive_plan(plan); };
        auto valid_fn = [&](const CSRAdaptivePlan&) { return true; };
        return cold_mode
            ? measure_reusable_cold(build_fn, run_fn, free_fn, valid_fn, std::max(1, iters), nnz, N)
            : measure_reusable_warm(build_fn, run_fn, free_fn, valid_fn, warmup, std::max(1, iters), nnz, N);
    }

    if (path == NextPath::STAGED_REUSE) {
        auto build_fn = [&]() {
            return build_staged_reuse_plan(host.rowptr.data(), host.colind.data(), host.vals.data(), M, K, 64, 64);
        };
        auto run_fn = [&](const StagedReusePlan& plan) {
            auto C = torch::zeros({M, N}, B.options());
            run_staged_reuse_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N);
        };
        auto free_fn = [&](StagedReusePlan& plan) { free_staged_reuse_plan(plan); };
        auto valid_fn = [&](const StagedReusePlan&) { return true; };
        return cold_mode
            ? measure_reusable_cold(build_fn, run_fn, free_fn, valid_fn, std::max(1, iters), nnz, N)
            : measure_reusable_warm(build_fn, run_fn, free_fn, valid_fn, warmup, std::max(1, iters), nnz, N);
    }

    if (path == NextPath::TC_SPARSE) {
        int device = 0;
        CUDA_CHECK_NEXT(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK_NEXT(cudaGetDeviceProperties(&prop, device));
        const bool hw_tc = (prop.major >= 7);

        auto build_fn = [&]() {
            return build_tc_sparse_plan(host.rowptr.data(), host.colind.data(), host.vals.data(), M, K, true, hw_tc);
        };
        auto run_fn = [&](TCSparsePlan& plan) {
            auto C = torch::zeros({M, N}, B.options());
            run_tc_sparse_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N);
        };
        auto free_fn = [&](TCSparsePlan& plan) { free_tc_sparse_plan(plan); };
        auto valid_fn = [&](const TCSparsePlan&) { return true; };
        return cold_mode
            ? measure_reusable_cold(build_fn, run_fn, free_fn, valid_fn, std::max(1, iters), nnz, N)
            : measure_reusable_warm(build_fn, run_fn, free_fn, valid_fn, warmup, std::max(1, iters), nnz, N);
    }

    if (path == NextPath::ROW_SPLIT_CUDA) {
        auto build_fn = [&]() { return make_row_split_plan(host.rowptr.data(), M, K); };
        auto run_fn = [&](const RowSplitPlan& plan) {
            auto C = torch::zeros({M, N}, B.options());
            run_row_split_plan(plan, colind.data_ptr<int>(), vals.data_ptr<float>(),
                               B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        };
        auto free_fn = [&](RowSplitPlan& plan) { free_row_split_plan(plan); };
        auto valid_fn = [&](const RowSplitPlan&) { return true; };
        return cold_mode
            ? measure_reusable_cold(build_fn, run_fn, free_fn, valid_fn, std::max(1, iters), nnz, N)
            : measure_reusable_warm(build_fn, run_fn, free_fn, valid_fn, warmup, std::max(1, iters), nnz, N);
    }

    if (path == NextPath::TC_REORDERED) {
        auto build_fn = [&]() {
            return make_tc_reordered_plan(host.rowptr.data(), host.colind.data(), host.vals.data(), M, K, N);
        };
        auto run_fn = [&](const TCReorderedPlan& plan) {
            auto C = torch::zeros({M, N}, B.options());
            run_tc_reordered_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        };
        auto free_fn = [&](TCReorderedPlan& plan) { free_tc_reordered_plan(plan); };
        auto valid_fn = [&](const TCReorderedPlan& plan) { return plan.active; };
        return cold_mode
            ? measure_reusable_cold(build_fn, run_fn, free_fn, valid_fn, std::max(1, iters), nnz, N)
            : measure_reusable_warm(build_fn, run_fn, free_fn, valid_fn, warmup, std::max(1, iters), nnz, N);
    }

    if (path == NextPath::HYBRID_TC_CUDA) {
        auto build_fn = [&]() {
            return make_hybrid_tc_cuda_plan(host.rowptr.data(), host.colind.data(), host.vals.data(), M, K, N, 0.45f);
        };
        auto run_fn = [&](const HybridPlan& plan) {
            auto C = torch::zeros({M, N}, B.options());
            run_hybrid_tc_cuda_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        };
        auto free_fn = [&](HybridPlan& plan) { free_hybrid_tc_cuda_plan(plan); };
        auto valid_fn = [&](const HybridPlan&) { return true; };
        return cold_mode
            ? measure_reusable_cold(build_fn, run_fn, free_fn, valid_fn, std::max(1, iters), nnz, N)
            : measure_reusable_warm(build_fn, run_fn, free_fn, valid_fn, warmup, std::max(1, iters), nnz, N);
    }

    // CUSPARSE: vendor library path.  "Plan" = CuSparseSpMMState creation
    // (handle init, descriptor setup, buffer alloc, preprocess).
    // "Run" = cusparseSpMM execution only.
    if (path == NextPath::CUSPARSE) {
        if (cold_mode) {
            const int actual_iters = std::max(1, iters);
            double plan_sum = 0.0, exec_sum = 0.0;
            for (int i = 0; i < actual_iters; ++i) {
                CUDA_CHECK_NEXT(cudaDeviceSynchronize());
                const auto t0 = std::chrono::high_resolution_clock::now();
                CuSparseSpMMState state(rowptr, colind, vals, B);
                CUDA_CHECK_NEXT(cudaDeviceSynchronize());
                const auto t1 = std::chrono::high_resolution_clock::now();
                const float exec_ms = measure_cuda_exec_ms([&]() { state.run(); }, 0, 1);
                plan_sum += std::chrono::duration<double, std::milli>(t1 - t0).count();
                exec_sum += exec_ms;
            }
            return make_timing_breakdown(
                static_cast<float>(plan_sum / actual_iters),
                static_cast<float>(exec_sum / actual_iters),
                nnz, N);
        } else {
            CuSparseSpMMState state(rowptr, colind, vals, B);
            const float exec_ms = measure_cuda_exec_ms(
                [&]() { state.run(); }, warmup, std::max(1, iters));
            return make_timing_breakdown(0.f, exec_ms, nnz, N);
        }
    }

    // Fallback: unknown path
    return infinite_timing();
}

static pybind11::dict assemble_oracle_dict(
    const std::vector<NextPath>& paths,
    const std::vector<TimingBreakdown>& timings,
    const char* mode,
    const char* portfolio_name,
    int M,
    int K,
    int N,
    int nnz)
{
    pybind11::dict path_results;
    pybind11::dict path_times;

    int oracle_idx = 0;
    for (size_t i = 0; i < paths.size(); ++i) {
        const std::string name = next_path_name(paths[i]);
        path_results[name.c_str()] = path_timing_to_dict(
            name, timings[i], is_main_path(paths[i]), is_legacy_path(paths[i]));
        path_times[name.c_str()] = timings[i].total_ms;
        if (timings[i].total_ms < timings[oracle_idx].total_ms) {
            oracle_idx = static_cast<int>(i);
        }
    }

    pybind11::dict result;
    result["mode"] = std::string(mode);
    result["portfolio"] = std::string(portfolio_name);
    result["path_results"] = path_results;
    result["path_times"] = path_times;
    result["oracle_path"] = std::string(next_path_name(paths[oracle_idx]));
    result["oracle_time_ms"] = timings[oracle_idx].total_ms;
    result["oracle_plan_ms"] = timings[oracle_idx].plan_ms;
    result["oracle_exec_ms"] = timings[oracle_idx].exec_ms;
    result["M"] = M;
    result["K"] = K;
    result["N"] = N;
    result["nnz"] = nnz;
    return result;
}

static std::vector<NextPath> portfolio_paths(Portfolio portfolio) {
    if (portfolio == Portfolio::MAIN) {
        return {
            NextPath::CSR_DIRECT,
            NextPath::ROW_SPLIT_CUDA,
            NextPath::TC_REORDERED,
            NextPath::HYBRID_TC_CUDA,
            NextPath::CUSPARSE,
        };
    }
    return std::vector<NextPath>(kAllNextPaths.begin(), kAllNextPaths.end());
}

// ---------------------------------------------------------------------------
// Router interface
// ---------------------------------------------------------------------------

pybind11::dict make_router_plan_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, int M, int K, int N,
    std::string portfolio_str)
{
    HostCSR host = copy_csr_to_host(rowptr, colind, vals);
    Portfolio port = (portfolio_str == "FULL") ? Portfolio::FULL : Portfolio::MAIN;
    RouterPlan plan = make_router_plan(host.rowptr.data(), host.colind.data(), M, K, N, port);
    return plan_to_dict(plan);
}

torch::Tensor run_router_plan_fn(
    pybind11::dict plan_dict,
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B)
{
    check_csr_tensors(rowptr, colind, vals);
    check_gpu_tensor(B, "B");

    const int M = static_cast<int>(rowptr.size(0)) - 1;
    const int K = static_cast<int>(B.size(0));
    const int N = static_cast<int>(B.size(1));
    const std::string path_str = plan_dict["chosen_path"].cast<std::string>();
    const HostCSR host = copy_csr_to_host(rowptr, colind, vals);
    auto C = torch::zeros({M, N}, B.options());

    if (path_str == "CSR_DIRECT") {
        csr_direct_spmm(rowptr.data_ptr<int>(), colind.data_ptr<int>(),
                        vals.data_ptr<float>(), B.data_ptr<float>(),
                        C.data_ptr<float>(), M, K, N);
    } else if (path_str == "CSR_ADAPTIVE") {
        CSRAdaptivePlan plan = build_csr_adaptive_plan(host.rowptr.data(), M, K);
        run_csr_adaptive_plan(plan, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
                              vals.data_ptr<float>(), B.data_ptr<float>(),
                              C.data_ptr<float>(), N);
        free_csr_adaptive_plan(plan);
    } else if (path_str == "STAGED_REUSE") {
        StagedReusePlan plan = build_staged_reuse_plan(
            host.rowptr.data(), host.colind.data(), host.vals.data(), M, K, 64, 64);
        run_staged_reuse_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N);
        free_staged_reuse_plan(plan);
    } else if (path_str == "TC_SPARSE") {
        int device = 0;
        CUDA_CHECK_NEXT(cudaGetDevice(&device));
        cudaDeviceProp prop{};
        CUDA_CHECK_NEXT(cudaGetDeviceProperties(&prop, device));
        TCSparsePlan plan = build_tc_sparse_plan(
            host.rowptr.data(), host.colind.data(), host.vals.data(), M, K, true, prop.major >= 7);
        run_tc_sparse_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N);
        free_tc_sparse_plan(plan);
    } else if (path_str == "ROW_SPLIT_CUDA") {
        RowSplitPlan plan = make_row_split_plan(host.rowptr.data(), M, K);
        run_row_split_plan(plan, colind.data_ptr<int>(), vals.data_ptr<float>(),
                           B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        free_row_split_plan(plan);
    } else if (path_str == "TC_REORDERED") {
        TCReorderedPlan plan = make_tc_reordered_plan(
            host.rowptr.data(), host.colind.data(), host.vals.data(), M, K, N);
        if (plan.active) {
            run_tc_reordered_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        }
        free_tc_reordered_plan(plan);
    } else if (path_str == "HYBRID_TC_CUDA") {
        HybridPlan plan = make_hybrid_tc_cuda_plan(
            host.rowptr.data(), host.colind.data(), host.vals.data(), M, K, N, 0.45f);
        run_hybrid_tc_cuda_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        free_hybrid_tc_cuda_plan(plan);
    } else if (path_str == "CUSPARSE") {
        CuSparseSpMMState state(rowptr, colind, vals, B);
        state.run();
        C.copy_(state.output());
    }
    // --- New regime-specific kernels ---
    else if (path_str == "TC_DIRECT") {
        RATcDirectPlan plan{};
        make_ra_tc_direct_plan(plan, host.rowptr.data(), host.colind.data(),
                              host.vals.data(), M, K, N);
        if (plan.active) {
            run_ra_tc_direct_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        }
        free_ra_tc_direct_plan(plan);
    } else if (path_str == "COMMUNITY_TC") {
        RACommunityTCPlan plan{};
        make_ra_community_tc_plan(plan, host.rowptr.data(), host.colind.data(),
                                  host.vals.data(), M, K, N);
        if (plan.active) {
            run_ra_community_tc_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        }
        free_ra_community_tc_plan(plan);
    } else if (path_str == "RODE_ENHANCED") {
        RARodeEnhancedPlan plan{};
        make_ra_rode_enhanced_plan(plan, host.rowptr.data(), M, K);
        run_ra_rode_enhanced_plan(plan, colind.data_ptr<int>(), vals.data_ptr<float>(),
                                  B.data_ptr<float>(), C.data_ptr<float>(), N);
        free_ra_rode_enhanced_plan(plan);
    } else if (path_str == "ZERO_OVERHEAD_CSR") {
        RAZeroOverheadPlan plan{};
        make_ra_zero_overhead_plan(plan, host.rowptr.data(), M, K);
        run_ra_zero_overhead_plan(plan, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
                                  vals.data_ptr<float>(), B.data_ptr<float>(),
                                  C.data_ptr<float>(), N);
        free_ra_zero_overhead_plan(plan);
    } else if (path_str == "VECTORIZED_COARSE") {
        RAVectorizedCoarsePlan plan{};
        make_ra_vectorized_coarse_plan(plan, host.rowptr.data(), M, K);
        run_ra_vectorized_coarse_plan(plan, rowptr.data_ptr<int>(), colind.data_ptr<int>(),
                                      vals.data_ptr<float>(), B.data_ptr<float>(),
                                      C.data_ptr<float>(), N);
        free_ra_vectorized_coarse_plan(plan);
    } else if (path_str == "LOCALITY_TILED") {
        RALocalityTiledPlan plan{};
        make_ra_locality_tiled_plan(plan, host.rowptr.data(), host.colind.data(),
                                    host.vals.data(), M, K, N);
        if (plan.active) {
            run_ra_locality_tiled_plan(plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        }
        free_ra_locality_tiled_plan(plan);
    } else if (path_str == "SEGMENT_HYBRID") {
        RASegmentHybridPlan plan{};
        make_ra_segment_hybrid_plan(plan, host.rowptr.data(), host.colind.data(),
                                    host.vals.data(), M, K, N);
        if (plan.active) {
            run_ra_segment_hybrid_plan(plan, colind.data_ptr<int>(), vals.data_ptr<float>(),
                                       B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
        }
        free_ra_segment_hybrid_plan(plan);
    } else {
        TORCH_CHECK(false, "Unknown router path: ", path_str);
    }

    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

pybind11::dict run_oracle_generic_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B,
    int warmup, int iters,
    std::string portfolio_str,
    bool cold_mode)
{
    check_csr_tensors(rowptr, colind, vals);
    check_gpu_tensor(B, "B");

    const int M = static_cast<int>(rowptr.size(0)) - 1;
    const int K = static_cast<int>(B.size(0));
    const int N = static_cast<int>(B.size(1));
    const int nnz = static_cast<int>(colind.numel());
    const Portfolio portfolio = (portfolio_str == "FULL") ? Portfolio::FULL : Portfolio::MAIN;
    const HostCSR host = copy_csr_to_host(rowptr, colind, vals);

    const std::vector<NextPath> paths = portfolio_paths(portfolio);
    std::vector<TimingBreakdown> timings;
    timings.reserve(paths.size());
    for (NextPath path : paths) {
        timings.push_back(measure_path_timing(
            path, host, rowptr, colind, vals, B, cold_mode, warmup, iters));
    }

    return assemble_oracle_dict(
        paths, timings,
        cold_mode ? "cold" : "warm",
        portfolio == Portfolio::FULL ? "FULL" : "MAIN",
        M, K, N, nnz);
}

pybind11::dict run_router_generic_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    torch::Tensor vals, torch::Tensor B,
    std::string portfolio_str,
    int warmup, int iters,
    bool cold_mode)
{
    check_csr_tensors(rowptr, colind, vals);
    check_gpu_tensor(B, "B");

    const int M = static_cast<int>(rowptr.size(0)) - 1;
    const int K = static_cast<int>(B.size(0));
    const int N = static_cast<int>(B.size(1));
    const int nnz = static_cast<int>(colind.numel());
    const Portfolio portfolio = (portfolio_str == "FULL") ? Portfolio::FULL : Portfolio::MAIN;
    const HostCSR host = copy_csr_to_host(rowptr, colind, vals);

    const auto router_plan = make_router_plan(host.rowptr.data(), host.colind.data(), M, K, N, portfolio);
    const NextPath chosen = router_plan.chosen_path;
    TimingBreakdown timing{};

    if (cold_mode) {
        const auto t0 = std::chrono::high_resolution_clock::now();
        const RouterPlan plan_for_timing = make_router_plan(host.rowptr.data(), host.colind.data(), M, K, N, portfolio);
        const auto t1 = std::chrono::high_resolution_clock::now();
        const float router_plan_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        timing = measure_path_timing(chosen, host, rowptr, colind, vals, B, true, warmup, iters);
        timing.plan_ms += router_plan_ms;
        timing.total_ms = timing.plan_ms + timing.exec_ms;
        (void)plan_for_timing;
    } else {
        timing = measure_path_timing(chosen, host, rowptr, colind, vals, B, false, warmup, iters);
    }

    pybind11::dict result;
    result["mode"] = std::string(cold_mode ? "cold" : "warm");
    result["portfolio"] = std::string(portfolio == Portfolio::FULL ? "FULL" : "MAIN");
    result["router_path"] = std::string(next_path_name(chosen));
    result["timing"] = timing_to_dict(timing);
    result["plan"] = plan_to_dict(router_plan);
    result["M"] = M;
    result["K"] = K;
    result["N"] = N;
    result["nnz"] = nnz;
    return result;
}

// ---------------------------------------------------------------------------
// GPU info
// ---------------------------------------------------------------------------
std::vector<pybind11::dict> gpu_info_next_fn() {
    int n_devices = 0;
    cudaGetDeviceCount(&n_devices);
    std::vector<pybind11::dict> result;
    for (int i = 0; i < n_devices; ++i) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        pybind11::dict d;
        d["device_id"]     = i;
        d["name"]          = std::string(prop.name);
        d["compute_major"] = prop.major;
        d["compute_minor"] = prop.minor;
        d["total_memory_mb"] = (int)(prop.totalGlobalMem / (1024 * 1024));
        d["multiprocessors"] = prop.multiProcessorCount;
        d["wmma_supported"]  = (prop.major >= 7);
        result.push_back(d);
    }
    return result;
}

// ---------------------------------------------------------------------------
// Matrix analysis (with N parameter)
// ---------------------------------------------------------------------------
pybind11::dict analyze_matrix_fn(
    torch::Tensor rowptr, torch::Tensor colind,
    int M, int K, int N)
{
    std::vector<int> h_rowptr, h_colind;

    if (rowptr.is_cuda()) {
        h_rowptr.resize(rowptr.numel());
        h_colind.resize(colind.numel());
        cudaMemcpy(h_rowptr.data(), rowptr.data_ptr<int>(), rowptr.numel() * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colind.data(), colind.data_ptr<int>(), colind.numel() * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        auto rp = rowptr.contiguous().to(torch::kInt32);
        auto ci = colind.contiguous().to(torch::kInt32);
        h_rowptr.assign(rp.data_ptr<int>(), rp.data_ptr<int>() + rp.numel());
        h_colind.assign(ci.data_ptr<int>(), ci.data_ptr<int>() + ci.numel());
    }

    RouterFeatures f = compute_router_features(h_rowptr.data(), h_colind.data(), M, K, N);

    pybind11::dict d;
    d["avg_nnz_per_row"]      = f.avg_nnz_per_row;
    d["std_nnz_per_row"]      = f.std_nnz_per_row;
    d["degree_cv"]            = f.degree_cv;
    d["max_to_mean_ratio"]    = f.max_to_mean_ratio;
    d["frac_dense_rows"]      = f.frac_dense_rows;
    d["skew_ratio"]           = f.skew_ratio;
    d["long_row_fraction"]    = f.long_row_fraction;
    d["long_row_nnz_fraction"] = f.long_row_nnz_fraction;
    d["top_1_row_nnz_fraction"] = f.top_1_row_nnz_fraction;
    d["top_5_row_nnz_fraction"] = f.top_5_row_nnz_fraction;
    d["tile_fill_mean"]       = f.tile_fill_mean;
    d["tile_fill_median"]     = f.tile_fill_median;
    d["tile_fill_p90"]        = f.tile_fill_p90;
    d["tile_fill_max"]        = f.tile_fill_max;
    d["tile_fill_variance"]   = f.tile_fill_variance;
    d["tile_occupancy"]       = f.tile_occupancy;
    d["actual_nnz_coverage"]  = f.actual_nnz_coverage;
    d["avg_nnz_per_tile"]     = f.avg_nnz_per_tile;
    d["row_window_colspan_compactness"] = f.row_window_colspan_compactness;
    d["local_row_similarity_proxy"] = f.local_row_similarity_proxy;
    d["reordered_locality_proxy"] = f.reordered_locality_proxy;
    d["locality_gain_proxy"] = f.locality_gain_proxy;
    d["locality_selectivity_proxy"] = f.locality_selectivity_proxy;
    d["road_likeness_proxy"] = f.road_likeness_proxy;
    d["row_split_affinity_proxy"] = f.row_split_affinity_proxy;
    d["mixedness_proxy"] = f.mixedness_proxy;
    d["tc_candidate_ratio"]   = f.tc_candidate_ratio;
    d["tc_synergy_proxy"]     = f.tc_synergy_proxy;
    d["estimated_tc_partition_ratio"] = f.estimated_tc_partition_ratio;
    d["estimated_cuda_partition_ratio"] = f.estimated_cuda_partition_ratio;
    d["irregular_window_fraction"] = f.irregular_window_fraction;
    d["tc_granularity_proxy"] = f.tc_granularity_proxy;
    d["redundancy_risk_proxy"] = f.redundancy_risk_proxy;
    d["tc_candidate_tiles"]   = f.tc_candidate_tiles;
    d["nnz"]                  = (M > 0) ? h_rowptr[M] : 0;
    return d;
}

// ===========================================================================
// New regime-specific kernel binding functions (Wave 1)
// ===========================================================================

// --- R6: Zero-overhead CSR ---
std::shared_ptr<RAZeroOverheadPlanWrapper> make_ra_zero_overhead_plan_fn(
    torch::Tensor rowptr_cpu, int M, int K)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto w = std::make_shared<RAZeroOverheadPlanWrapper>();
    make_ra_zero_overhead_plan(w->plan, rp.data_ptr<int>(), M, K);
    w->valid = true;
    return w;
}

torch::Tensor run_ra_zero_overhead_plan_fn(
    std::shared_ptr<RAZeroOverheadPlanWrapper> wrapper,
    torch::Tensor rowptr, torch::Tensor colind, torch::Tensor vals, torch::Tensor B)
{
    check_gpu_tensor(rowptr, "rowptr"); check_gpu_tensor(colind, "colind");
    check_gpu_tensor(vals, "vals"); check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid ZeroOverhead plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_ra_zero_overhead_plan(wrapper->plan,
        rowptr.data_ptr<int>(), colind.data_ptr<int>(), vals.data_ptr<float>(),
        B.data_ptr<float>(), C.data_ptr<float>(), N);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

// --- R2: Vectorized coarse ---
std::shared_ptr<RAVectorizedCoarsePlanWrapper> make_ra_vectorized_coarse_plan_fn(
    torch::Tensor rowptr_cpu, int M, int K)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto w = std::make_shared<RAVectorizedCoarsePlanWrapper>();
    make_ra_vectorized_coarse_plan(w->plan, rp.data_ptr<int>(), M, K);
    w->valid = true;
    return w;
}

torch::Tensor run_ra_vectorized_coarse_plan_fn(
    std::shared_ptr<RAVectorizedCoarsePlanWrapper> wrapper,
    torch::Tensor rowptr, torch::Tensor colind, torch::Tensor vals, torch::Tensor B)
{
    check_gpu_tensor(rowptr, "rowptr"); check_gpu_tensor(colind, "colind");
    check_gpu_tensor(vals, "vals"); check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid VectorizedCoarse plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_ra_vectorized_coarse_plan(wrapper->plan,
        rowptr.data_ptr<int>(), colind.data_ptr<int>(), vals.data_ptr<float>(),
        B.data_ptr<float>(), C.data_ptr<float>(), N);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

// --- R1: RoDe-enhanced ---
std::shared_ptr<RARodeEnhancedPlanWrapper> make_ra_rode_enhanced_plan_fn(
    torch::Tensor rowptr_cpu, int M, int K)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto w = std::make_shared<RARodeEnhancedPlanWrapper>();
    make_ra_rode_enhanced_plan(w->plan, rp.data_ptr<int>(), M, K);
    w->valid = true;
    return w;
}

torch::Tensor run_ra_rode_enhanced_plan_fn(
    std::shared_ptr<RARodeEnhancedPlanWrapper> wrapper,
    torch::Tensor colind, torch::Tensor vals, torch::Tensor B)
{
    check_gpu_tensor(colind, "colind"); check_gpu_tensor(vals, "vals");
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid RodeEnhanced plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_ra_rode_enhanced_plan(wrapper->plan,
        colind.data_ptr<int>(), vals.data_ptr<float>(),
        B.data_ptr<float>(), C.data_ptr<float>(), N);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

// --- R4: Flash TC ---
std::shared_ptr<RATcDirectPlanWrapper> make_ra_tc_direct_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K, int N)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto ci = colind_cpu.cpu().contiguous().to(torch::kInt32);
    auto vl = vals_cpu.cpu().contiguous().to(torch::kFloat32);
    auto w = std::make_shared<RATcDirectPlanWrapper>();
    make_ra_tc_direct_plan(w->plan, rp.data_ptr<int>(), ci.data_ptr<int>(),
                          vl.data_ptr<float>(), M, K, N);
    w->valid = true;
    return w;
}

torch::Tensor run_ra_tc_direct_plan_fn(
    std::shared_ptr<RATcDirectPlanWrapper> wrapper, torch::Tensor B)
{
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid TcDirect plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_ra_tc_direct_plan(wrapper->plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

// --- R3: Locality-tiled ---
std::shared_ptr<RALocalityTiledPlanWrapper> make_ra_locality_tiled_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K, int N)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto ci = colind_cpu.cpu().contiguous().to(torch::kInt32);
    auto vl = vals_cpu.cpu().contiguous().to(torch::kFloat32);
    auto w = std::make_shared<RALocalityTiledPlanWrapper>();
    make_ra_locality_tiled_plan(w->plan, rp.data_ptr<int>(), ci.data_ptr<int>(),
                                vl.data_ptr<float>(), M, K, N);
    w->valid = true;
    return w;
}

torch::Tensor run_ra_locality_tiled_plan_fn(
    std::shared_ptr<RALocalityTiledPlanWrapper> wrapper, torch::Tensor B)
{
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid LocalityTiled plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_ra_locality_tiled_plan(wrapper->plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

// --- R5: Community TC ---
std::shared_ptr<RACommunityTCPlanWrapper> make_ra_community_tc_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K, int N)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto ci = colind_cpu.cpu().contiguous().to(torch::kInt32);
    auto vl = vals_cpu.cpu().contiguous().to(torch::kFloat32);
    auto w = std::make_shared<RACommunityTCPlanWrapper>();
    make_ra_community_tc_plan(w->plan, rp.data_ptr<int>(), ci.data_ptr<int>(),
                              vl.data_ptr<float>(), M, K, N);
    w->valid = true;
    return w;
}

torch::Tensor run_ra_community_tc_plan_fn(
    std::shared_ptr<RACommunityTCPlanWrapper> wrapper, torch::Tensor B)
{
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid CommunityTC plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_ra_community_tc_plan(wrapper->plan, B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

// --- R7: Segment hybrid ---
std::shared_ptr<RASegmentHybridPlanWrapper> make_ra_segment_hybrid_plan_fn(
    torch::Tensor rowptr_cpu, torch::Tensor colind_cpu,
    torch::Tensor vals_cpu, int M, int K, int N)
{
    auto rp = rowptr_cpu.cpu().contiguous().to(torch::kInt32);
    auto ci = colind_cpu.cpu().contiguous().to(torch::kInt32);
    auto vl = vals_cpu.cpu().contiguous().to(torch::kFloat32);
    auto w = std::make_shared<RASegmentHybridPlanWrapper>();
    make_ra_segment_hybrid_plan(w->plan, rp.data_ptr<int>(), ci.data_ptr<int>(),
                                vl.data_ptr<float>(), M, K, N);
    w->valid = true;
    return w;
}

torch::Tensor run_ra_segment_hybrid_plan_fn(
    std::shared_ptr<RASegmentHybridPlanWrapper> wrapper,
    torch::Tensor colind, torch::Tensor vals, torch::Tensor B)
{
    check_gpu_tensor(colind, "colind"); check_gpu_tensor(vals, "vals");
    check_gpu_tensor(B, "B");
    TORCH_CHECK(wrapper && wrapper->valid, "Invalid SegmentHybrid plan");
    int M = wrapper->plan.M;
    int N = (int)B.size(1);
    auto C = torch::zeros({M, N}, B.options());
    run_ra_segment_hybrid_plan(wrapper->plan,
        colind.data_ptr<int>(), vals.data_ptr<float>(),
        B.data_ptr<float>(), C.data_ptr<float>(), N, 0);
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());
    return C;
}

// ---------------------------------------------------------------------------
// pybind11 module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(ra_spmm, m) {
    m.doc() = "Regime-Aware Kernel Routing for Graph SpMM - ra_spmm module";

    // Core kernels
    m.def("spmm_csr_direct", &spmm_csr_direct_fn,
          "CSR Direct SpMM (warp-per-row)",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"));

    m.def("spmm_csr_adaptive", &spmm_csr_adaptive_fn,
          "CSR Adaptive (binned) SpMM",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"));

    m.def("spmm_staged_reuse", &spmm_staged_reuse_fn,
          "Staged Reuse SpMM (tile-based B reuse)",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"));

    m.def("spmm_tc_sparse", &spmm_tc_sparse_fn,
          "Gated TC SpMM (returns C, diag_dict)",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"));

    m.def("spmm_cusparse", &spmm_cusparse_fn,
          "cuSPARSE CSR SpMM baseline",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"));

    m.def("benchmark_cusparse", &benchmark_cusparse_fn,
          "Warm cuSPARSE CSR SpMM timing with CUDA events",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"),
          pybind11::arg("warmup") = 3,
          pybind11::arg("iters") = 10);

    m.def("benchmark_cusparse_cold", &benchmark_cusparse_cold_fn,
          "Cold cuSPARSE CSR SpMM timing with explicit state construction cost",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"),
          pybind11::arg("iters") = 10);

    // Plan wrapper classes
    pybind11::class_<CSRAdaptivePlanWrapper, std::shared_ptr<CSRAdaptivePlanWrapper>>(m, "CSRAdaptivePlan")
        .def_property_readonly("valid", [](const CSRAdaptivePlanWrapper& w) { return w.valid; })
        .def_property_readonly("bin_histogram", [](const CSRAdaptivePlanWrapper& w) {
            std::vector<int> h(w.plan.bin_histogram, w.plan.bin_histogram + 5);
            return h;
        })
        .def_property_readonly("dominant_bin", [](const CSRAdaptivePlanWrapper& w) { return w.plan.dominant_bin; })
        .def_property_readonly("n_split_rows", [](const CSRAdaptivePlanWrapper& w) { return w.plan.n_split_rows; })
        .def_property_readonly("M", [](const CSRAdaptivePlanWrapper& w) { return w.plan.M; })
        .def_property_readonly("K", [](const CSRAdaptivePlanWrapper& w) { return w.plan.K; });

    pybind11::class_<StagedReusePlanWrapper, std::shared_ptr<StagedReusePlanWrapper>>(m, "StagedReusePlan")
        .def_property_readonly("valid", [](const StagedReusePlanWrapper& w) { return w.valid; })
        .def_property_readonly("num_tiles", [](const StagedReusePlanWrapper& w) { return w.plan.num_tiles; })
        .def_property_readonly("avg_tile_fill", [](const StagedReusePlanWrapper& w) { return w.plan.avg_tile_fill; })
        .def_property_readonly("M", [](const StagedReusePlanWrapper& w) { return w.plan.M; })
        .def_property_readonly("K", [](const StagedReusePlanWrapper& w) { return w.plan.K; });

    pybind11::class_<TCSparsePlanWrapper, std::shared_ptr<TCSparsePlanWrapper>>(m, "TCSparsePlan")
        .def_property_readonly("valid", [](const TCSparsePlanWrapper& w) { return w.valid; })
        .def_property_readonly("num_tc_tiles", [](const TCSparsePlanWrapper& w) { return w.plan.num_tc_tiles; })
        .def_property_readonly("candidate_tiles", [](const TCSparsePlanWrapper& w) { return w.plan.candidate_tiles; })
        .def_property_readonly("fill_mean", [](const TCSparsePlanWrapper& w) { return w.plan.fill_mean; })
        .def_property_readonly("fill_median", [](const TCSparsePlanWrapper& w) { return w.plan.fill_median; })
        .def_property_readonly("residual_nnz", [](const TCSparsePlanWrapper& w) { return w.plan.residual_nnz; })
        .def_property_readonly("candidate_nnz_coverage", [](const TCSparsePlanWrapper& w) { return w.plan.candidate_nnz_coverage; })
        .def_property_readonly("M", [](const TCSparsePlanWrapper& w) { return w.plan.M; })
        .def_property_readonly("K", [](const TCSparsePlanWrapper& w) { return w.plan.K; });

    // Plan-run API
    m.def("make_csr_adaptive_plan", &make_csr_adaptive_plan_fn,
          "Build CSR adaptive plan (structural-only, no vals needed)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"),
          pybind11::arg("M"), pybind11::arg("K"));

    m.def("run_csr_adaptive_plan", &run_csr_adaptive_plan_fn,
          "Run CSR adaptive plan",
          pybind11::arg("plan"), pybind11::arg("rowptr"),
          pybind11::arg("colind"), pybind11::arg("vals"), pybind11::arg("B"));

    m.def("make_staged_reuse_plan", &make_staged_reuse_plan_fn,
          "Build staged reuse plan (val-aware)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"),
          pybind11::arg("vals_cpu"), pybind11::arg("M"), pybind11::arg("K"));

    m.def("run_staged_reuse_plan", &run_staged_reuse_plan_fn,
          "Run staged reuse plan (plan contains CSR data)",
          pybind11::arg("plan"), pybind11::arg("B"));

    m.def("make_tc_sparse_plan", &make_tc_sparse_plan_fn,
          "Build TC sparse plan (val-aware)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"),
          pybind11::arg("vals_cpu"), pybind11::arg("M"), pybind11::arg("K"),
          pybind11::arg("tc_eligible") = true);

    m.def("run_tc_sparse_plan", &run_tc_sparse_plan_fn,
          "Run TC sparse plan",
          pybind11::arg("plan"), pybind11::arg("B"));

    pybind11::class_<RowSplitPlanWrapper, std::shared_ptr<RowSplitPlanWrapper>>(m, "RowSplitPlan")
        .def_property_readonly("valid", [](const RowSplitPlanWrapper& w){ return w.valid; })
        .def_property_readonly("num_regular_rows", [](const RowSplitPlanWrapper& w){ return w.plan.num_regular_rows; })
        .def_property_readonly("num_short_rows", [](const RowSplitPlanWrapper& w){ return w.plan.num_short_rows; })
        .def_property_readonly("num_long_rows", [](const RowSplitPlanWrapper& w){ return w.plan.num_long_rows; })
        .def_property_readonly("num_long_segments", [](const RowSplitPlanWrapper& w){ return w.plan.num_long_segments; })
        .def_property_readonly("num_residual", [](const RowSplitPlanWrapper& w){ return w.plan.num_residual; })
        .def_property_readonly("num_split_long_rows", [](const RowSplitPlanWrapper& w){ return w.plan.num_split_long_rows; })
        .def_property_readonly("regular_nnz_fraction", [](const RowSplitPlanWrapper& w){ return w.plan.regular_nnz_fraction; })
        .def_property_readonly("residual_nnz_fraction", [](const RowSplitPlanWrapper& w){ return w.plan.residual_nnz_fraction; })
        .def_property_readonly("avg_segments_per_long_row", [](const RowSplitPlanWrapper& w){ return w.plan.avg_segments_per_long_row; });

    pybind11::class_<TCReorderedPlanWrapper, std::shared_ptr<TCReorderedPlanWrapper>>(m, "TCReorderedPlan")
        .def_property_readonly("valid", [](const TCReorderedPlanWrapper& w){ return w.valid; })
        .def_property_readonly("active", [](const TCReorderedPlanWrapper& w){ return w.plan.active; })
        .def_property_readonly("placeholder_quality", [](const TCReorderedPlanWrapper& w){ return w.plan.placeholder_quality; })
        .def_property_readonly("num_fp32_groups", [](const TCReorderedPlanWrapper& w){ return w.plan.num_fp32_groups; })
        .def_property_readonly("fp32_group_fraction", [](const TCReorderedPlanWrapper& w){ return w.plan.fp32_group_fraction; })
        .def_property_readonly("avg_group_compactness", [](const TCReorderedPlanWrapper& w){ return w.plan.avg_group_compactness; })
        .def_property_readonly("avg_group_similarity", [](const TCReorderedPlanWrapper& w){ return w.plan.avg_group_similarity; });

    pybind11::class_<HybridPlanWrapper, std::shared_ptr<HybridPlanWrapper>>(m, "HybridPlan")
        .def_property_readonly("valid", [](const HybridPlanWrapper& w){ return w.valid; })
        .def_property_readonly("tc_nnz_fraction", [](const HybridPlanWrapper& w){ return w.plan.tc_nnz_fraction; })
        .def_property_readonly("cuda_nnz_fraction", [](const HybridPlanWrapper& w){ return w.plan.cuda_nnz_fraction; })
        .def_property_readonly("tc_row_fraction", [](const HybridPlanWrapper& w){ return w.plan.tc_row_fraction; })
        .def_property_readonly("cuda_row_fraction", [](const HybridPlanWrapper& w){ return w.plan.cuda_row_fraction; })
        .def_property_readonly("average_partition_score", [](const HybridPlanWrapper& w){ return w.plan.average_partition_score; })
        .def_property_readonly("average_window_compactness", [](const HybridPlanWrapper& w){ return w.plan.average_window_compactness; })
        .def_property_readonly("precision_guard_windows", [](const HybridPlanWrapper& w){ return w.plan.precision_guard_windows; })
        .def_property_readonly("precision_guard_row_fraction", [](const HybridPlanWrapper& w){ return w.plan.precision_guard_row_fraction; })
        .def_property_readonly("precision_guard_nnz_fraction", [](const HybridPlanWrapper& w){ return w.plan.precision_guard_nnz_fraction; });

    m.def("make_row_split_plan", &make_row_split_plan_fn,
          pybind11::arg("rowptr_cpu"), pybind11::arg("M"), pybind11::arg("K"));
    m.def("make_row_split_plan_no_long_rows", &make_row_split_plan_no_long_rows_fn,
          pybind11::arg("rowptr_cpu"), pybind11::arg("M"), pybind11::arg("K"));
    m.def("run_row_split_plan", &run_row_split_plan_fn,
          pybind11::arg("plan"), pybind11::arg("colind"), pybind11::arg("vals"), pybind11::arg("B"));
    m.def("make_tc_reordered_plan", &make_tc_reordered_plan_fn,
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"), pybind11::arg("vals_cpu"),
          pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("N"));
    m.def("run_tc_reordered_plan", &run_tc_reordered_plan_fn,
          pybind11::arg("plan"), pybind11::arg("B"));
    m.def("make_hybrid_tc_cuda_plan", &make_hybrid_tc_cuda_plan_fn,
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"), pybind11::arg("vals_cpu"),
          pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("N"),
          pybind11::arg("threshold") = 0.45f);
    m.def("run_hybrid_tc_cuda_plan", &run_hybrid_tc_cuda_plan_fn,
          pybind11::arg("plan"), pybind11::arg("B"));
    m.def("make_hybrid_plan", &make_hybrid_plan_fn,
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"), pybind11::arg("vals_cpu"),
          pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("N"),
          pybind11::arg("threshold") = 0.45f);
    m.def("run_hybrid_plan", &run_hybrid_plan_fn,
          pybind11::arg("plan"), pybind11::arg("B"));

    // Router
    m.def("make_router_plan", &make_router_plan_fn,
          "Compute router plan from CSR data",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("M"),
          pybind11::arg("K"), pybind11::arg("N"),
          pybind11::arg("portfolio") = "MAIN");

    m.def("run_router_plan", &run_router_plan_fn,
          "Execute the chosen kernel from a router plan",
          pybind11::arg("plan_dict"), pybind11::arg("rowptr"),
          pybind11::arg("colind"), pybind11::arg("vals"),
          pybind11::arg("B"));

    m.def("run_oracle_cold", [](torch::Tensor rowptr, torch::Tensor colind,
                                 torch::Tensor vals, torch::Tensor B,
                                 int warmup, int iters, std::string portfolio) {
            return run_oracle_generic_fn(rowptr, colind, vals, B, warmup, iters, portfolio, true);
          },
          "Run a cold oracle benchmark with explicit plan/exec/total breakdowns",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"),
          pybind11::arg("warmup") = 3,
          pybind11::arg("iters") = 10,
          pybind11::arg("portfolio") = "MAIN");

    m.def("run_oracle_warm", [](torch::Tensor rowptr, torch::Tensor colind,
                                 torch::Tensor vals, torch::Tensor B,
                                 int warmup, int iters, std::string portfolio) {
            return run_oracle_generic_fn(rowptr, colind, vals, B, warmup, iters, portfolio, false);
          },
          "Run a warm oracle benchmark with plan reuse and execution-only timing",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"),
          pybind11::arg("warmup") = 3,
          pybind11::arg("iters") = 10,
          pybind11::arg("portfolio") = "MAIN");

    m.def("run_router_cold", [](torch::Tensor rowptr, torch::Tensor colind,
                                 torch::Tensor vals, torch::Tensor B,
                                 std::string portfolio, int warmup, int iters) {
            return run_router_generic_fn(rowptr, colind, vals, B, portfolio, warmup, iters, true);
          },
          "Run router cold timing with routing + chosen-path preprocessing included",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"),
          pybind11::arg("portfolio") = "MAIN",
          pybind11::arg("warmup") = 3,
          pybind11::arg("iters") = 10);

    m.def("run_router_warm", [](torch::Tensor rowptr, torch::Tensor colind,
                                 torch::Tensor vals, torch::Tensor B,
                                 std::string portfolio, int warmup, int iters) {
            return run_router_generic_fn(rowptr, colind, vals, B, portfolio, warmup, iters, false);
          },
          "Run router warm timing with plan reuse and execution-only timing",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"),
          pybind11::arg("portfolio") = "MAIN",
          pybind11::arg("warmup") = 3,
          pybind11::arg("iters") = 10);

    // Backward-compatible aliases. These retain the old top-level names but now
    // use the explicit cold methodology.
    m.def("run_oracle", [](torch::Tensor rowptr, torch::Tensor colind,
                            torch::Tensor vals, torch::Tensor B,
                            int warmup, int iters) {
            return run_oracle_generic_fn(rowptr, colind, vals, B, warmup, iters, "MAIN", true);
          },
          "Backward-compatible alias for run_oracle_cold(..., portfolio='MAIN')",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"),
          pybind11::arg("warmup") = 3,
          pybind11::arg("iters") = 10);

    m.def("run_oracle_full", [](torch::Tensor rowptr, torch::Tensor colind,
                                 torch::Tensor vals, torch::Tensor B,
                                 int warmup, int iters) {
            return run_oracle_generic_fn(rowptr, colind, vals, B, warmup, iters, "FULL", true);
          },
          "Backward-compatible alias for run_oracle_cold(..., portfolio='FULL')",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"),
          pybind11::arg("warmup") = 3,
          pybind11::arg("iters") = 10);

    // Graph generators
    m.def("gen_random_sparse", [](int M, int K, int nnz_per_row, unsigned seed){
        return mat_to_dict(random_sparse(M, K, nnz_per_row, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("nnz_per_row"),
       pybind11::arg("seed") = 42);

    m.def("gen_skewed_powerlaw", [](int M, int K, float alpha, int min_nnz, int max_nnz, unsigned seed){
        return mat_to_dict(skewed_powerlaw(M, K, alpha, min_nnz, max_nnz, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("alpha"),
       pybind11::arg("min_nnz"), pybind11::arg("max_nnz"),
       pybind11::arg("seed") = 42);

    m.def("gen_community_clustered", [](int M, int K, int n_comm, float within_density, float between_density, unsigned seed){
        return mat_to_dict(community_clustered(M, K, n_comm, within_density, between_density, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("n_comm"),
       pybind11::arg("within_density"), pybind11::arg("between_density"),
       pybind11::arg("seed") = 42);

    m.def("gen_bipartite_rectangular", [](int M, int K, int nnz_per_row, unsigned seed){
        return mat_to_dict(bipartite_rectangular(M, K, nnz_per_row, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("nnz_per_row"),
       pybind11::arg("seed") = 42);

    m.def("gen_road_like", [](int M, int K, int avg_degree, unsigned seed){
        return mat_to_dict(road_like(M, K, avg_degree, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("avg_degree"),
       pybind11::arg("seed") = 42);

    m.def("gen_block_locality", [](int M, int K, int block_size, float fill, unsigned seed){
        return mat_to_dict(block_locality(M, K, block_size, fill, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("block_size"),
       pybind11::arg("fill"), pybind11::arg("seed") = 42);

    m.def("gen_hub_heavy", [](int M, int K, float hub_fraction, int hub_degree, int base_degree, unsigned seed){
        return mat_to_dict(hub_heavy(M, K, hub_fraction, hub_degree, base_degree, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("hub_fraction"),
       pybind11::arg("hub_degree"), pybind11::arg("base_degree"),
       pybind11::arg("seed") = 42);

    m.def("gen_mixed_skew", [](int M, int K, float frac_tiny, float frac_medium, float frac_giant,
                                 int tiny_degree, int medium_degree, int giant_degree, unsigned seed){
        return mat_to_dict(mixed_skew(M, K, frac_tiny, frac_medium, frac_giant,
                                      tiny_degree, medium_degree, giant_degree, seed));
    }, pybind11::arg("M"), pybind11::arg("K"),
       pybind11::arg("frac_tiny"), pybind11::arg("frac_medium"), pybind11::arg("frac_giant"),
       pybind11::arg("tiny_degree"), pybind11::arg("medium_degree"), pybind11::arg("giant_degree"),
       pybind11::arg("seed") = 42);

    m.def("gen_clustered_window", [](int M, int K, int window_rows, int window_span,
                                       float intra_window_density, unsigned seed){
        return mat_to_dict(clustered_window(M, K, window_rows, window_span, intra_window_density, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("window_rows"),
       pybind11::arg("window_span"), pybind11::arg("intra_window_density"),
       pybind11::arg("seed") = 42);

    m.def("gen_scrambled_locality", [](int M, int K, int window_rows, int window_span,
                                         float intra_window_density, unsigned seed){
        return mat_to_dict(scrambled_locality(M, K, window_rows, window_span, intra_window_density, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("window_rows"),
       pybind11::arg("window_span"), pybind11::arg("intra_window_density"),
       pybind11::arg("seed") = 42);

    m.def("gen_mixed_block_skew", [](int M, int K, int window_rows, float frac_block_windows,
                                       float frac_skew_windows, float block_fill,
                                       int skew_base_degree, int skew_hub_degree, unsigned seed){
        return mat_to_dict(mixed_block_skew(M, K, window_rows, frac_block_windows, frac_skew_windows,
                                            block_fill, skew_base_degree, skew_hub_degree, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("window_rows"),
       pybind11::arg("frac_block_windows"), pybind11::arg("frac_skew_windows"),
       pybind11::arg("block_fill"), pybind11::arg("skew_base_degree"),
       pybind11::arg("skew_hub_degree"), pybind11::arg("seed") = 42);

    m.def("gen_cluster_plus_hubs", [](int M, int K, int num_clusters, float within_density,
                                        float between_density, float hub_fraction,
                                        int hub_degree, unsigned seed){
        return mat_to_dict(cluster_plus_hubs(M, K, num_clusters, within_density, between_density,
                                             hub_fraction, hub_degree, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("num_clusters"),
       pybind11::arg("within_density"), pybind11::arg("between_density"),
       pybind11::arg("hub_fraction"), pybind11::arg("hub_degree"),
       pybind11::arg("seed") = 42);

    m.def("gen_heterogeneous_windows", [](int M, int K, int window_rows,
                                           float frac_block_dense, float frac_clustered_sparse,
                                           float frac_random_sparse, float frac_skew_heavy,
                                           unsigned seed){
        return mat_to_dict(heterogeneous_windows(M, K, window_rows, frac_block_dense,
                                                 frac_clustered_sparse, frac_random_sparse,
                                                 frac_skew_heavy, seed));
    }, pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("window_rows"),
       pybind11::arg("frac_block_dense"), pybind11::arg("frac_clustered_sparse"),
       pybind11::arg("frac_random_sparse"), pybind11::arg("frac_skew_heavy"),
       pybind11::arg("seed") = 42);

    m.def("gen_powerlaw_realistic", [](int M, int m_attach, unsigned seed){
        return mat_to_dict(powerlaw_realistic(M, m_attach, seed));
    }, pybind11::arg("M"), pybind11::arg("m_attach") = 5,
       pybind11::arg("seed") = 42);

    m.def("gen_community_sbm", [](int M, int n_comm, float within_density, float between_density, unsigned seed){
        return mat_to_dict(community_sbm(M, n_comm, within_density, between_density, seed));
    }, pybind11::arg("M"), pybind11::arg("n_comm") = 32,
       pybind11::arg("within_density") = 0.06f,
       pybind11::arg("between_density") = 0.001f,
       pybind11::arg("seed") = 42);

    m.def("gen_reordered_variant", [](torch::Tensor rowptr_t, torch::Tensor colind_t, torch::Tensor vals_t,
                                       int M, int K, unsigned seed) {
        SparseMatrix mat;
        mat.M = M; mat.K = K;
        auto rp = rowptr_t.cpu();
        auto ci = colind_t.cpu();
        auto vl = vals_t.cpu();
        mat.rowptr.assign(rp.data_ptr<int>(), rp.data_ptr<int>() + rp.numel());
        mat.colind.assign(ci.data_ptr<int>(), ci.data_ptr<int>() + ci.numel());
        mat.vals.assign(vl.data_ptr<float>(), vl.data_ptr<float>() + vl.numel());
        return mat_to_dict(reordered_variant(mat, seed));
    }, pybind11::arg("rowptr"), pybind11::arg("colind"), pybind11::arg("vals"),
       pybind11::arg("M"), pybind11::arg("K"), pybind11::arg("seed") = 42);

    // Utilities
    m.def("gpu_info_next", &gpu_info_next_fn, "Get GPU information");
    m.def("analyze_matrix", &analyze_matrix_fn,
          "Analyze matrix features",
          pybind11::arg("rowptr"), pybind11::arg("colind"),
          pybind11::arg("M"), pybind11::arg("K"),
          pybind11::arg("N") = 128);

    // =======================================================================
    // New regime-specific kernels (Wave 1)
    // =======================================================================

    // --- R6: Zero-overhead CSR ---
    pybind11::class_<RAZeroOverheadPlanWrapper, std::shared_ptr<RAZeroOverheadPlanWrapper>>(
        m, "RAZeroOverheadPlan")
        .def_property_readonly("valid", [](const RAZeroOverheadPlanWrapper& w){ return w.valid; })
        .def_property_readonly("M", [](const RAZeroOverheadPlanWrapper& w){ return w.plan.M; })
        .def_property_readonly("K", [](const RAZeroOverheadPlanWrapper& w){ return w.plan.K; })
        .def_property_readonly("num_tiny", [](const RAZeroOverheadPlanWrapper& w){ return w.plan.num_tiny; })
        .def_property_readonly("num_short", [](const RAZeroOverheadPlanWrapper& w){ return w.plan.num_short; })
        .def_property_readonly("num_medium", [](const RAZeroOverheadPlanWrapper& w){ return w.plan.num_medium; })
        .def_property_readonly("num_long", [](const RAZeroOverheadPlanWrapper& w){ return w.plan.num_long; })
        .def_property_readonly("plan_bytes", [](const RAZeroOverheadPlanWrapper& w){ return w.plan.plan_bytes; });

    m.def("make_zero_overhead_plan", &make_ra_zero_overhead_plan_fn,
          "Build R6 zero-overhead CSR plan (degree-binned dispatch)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("M"), pybind11::arg("K"));

    m.def("run_zero_overhead_plan", &run_ra_zero_overhead_plan_fn,
          "Run R6 zero-overhead CSR SpMM",
          pybind11::arg("plan"), pybind11::arg("rowptr"),
          pybind11::arg("colind"), pybind11::arg("vals"), pybind11::arg("B"));

    // --- R2: Vectorized coarse ---
    pybind11::class_<RAVectorizedCoarsePlanWrapper, std::shared_ptr<RAVectorizedCoarsePlanWrapper>>(
        m, "RAVectorizedCoarsePlan")
        .def_property_readonly("valid", [](const RAVectorizedCoarsePlanWrapper& w){ return w.valid; })
        .def_property_readonly("M", [](const RAVectorizedCoarsePlanWrapper& w){ return w.plan.M; })
        .def_property_readonly("K", [](const RAVectorizedCoarsePlanWrapper& w){ return w.plan.K; })
        .def_property_readonly("rows_per_warp", [](const RAVectorizedCoarsePlanWrapper& w){ return w.plan.rows_per_warp; })
        .def_property_readonly("plan_bytes", [](const RAVectorizedCoarsePlanWrapper& w){ return w.plan.plan_bytes; });

    m.def("make_vectorized_coarse_plan", &make_ra_vectorized_coarse_plan_fn,
          "Build R2 vectorized coarse plan (adaptive multi-row warp assignment)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("M"), pybind11::arg("K"));

    m.def("run_vectorized_coarse_plan", &run_ra_vectorized_coarse_plan_fn,
          "Run R2 vectorized coarse SpMM",
          pybind11::arg("plan"), pybind11::arg("rowptr"),
          pybind11::arg("colind"), pybind11::arg("vals"), pybind11::arg("B"));

    // --- R1: RoDe-enhanced ---
    pybind11::class_<RARodeEnhancedPlanWrapper, std::shared_ptr<RARodeEnhancedPlanWrapper>>(
        m, "RARodeEnhancedPlan")
        .def_property_readonly("valid", [](const RARodeEnhancedPlanWrapper& w){ return w.valid; })
        .def_property_readonly("M", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.M; })
        .def_property_readonly("K", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.K; })
        .def_property_readonly("num_short_rows", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.num_short_rows; })
        .def_property_readonly("num_long_rows", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.num_long_rows; })
        .def_property_readonly("num_long_sub_blocks", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.num_long_sub_blocks; })
        .def_property_readonly("num_residual", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.num_residual; })
        .def_property_readonly("regular_nnz_fraction", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.regular_nnz_fraction; })
        .def_property_readonly("long_row_nnz_fraction", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.long_row_nnz_fraction; })
        .def_property_readonly("plan_bytes", [](const RARodeEnhancedPlanWrapper& w){ return w.plan.plan_bytes; });

    m.def("make_rode_enhanced_plan", &make_ra_rode_enhanced_plan_fn,
          "Build R1 RoDe-enhanced plan (block-residual decomposition with sub-block pipelining)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("M"), pybind11::arg("K"));

    m.def("run_rode_enhanced_plan", &run_ra_rode_enhanced_plan_fn,
          "Run R1 RoDe-enhanced SpMM",
          pybind11::arg("plan"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"));

    // --- R4: Flash TC ---
    pybind11::class_<RATcDirectPlanWrapper, std::shared_ptr<RATcDirectPlanWrapper>>(
        m, "RATcDirectPlan")
        .def_property_readonly("valid", [](const RATcDirectPlanWrapper& w){ return w.valid; })
        .def_property_readonly("M", [](const RATcDirectPlanWrapper& w){ return w.plan.M; })
        .def_property_readonly("K", [](const RATcDirectPlanWrapper& w){ return w.plan.K; })
        .def_property_readonly("num_groups", [](const RATcDirectPlanWrapper& w){ return w.plan.num_groups; })
        .def_property_readonly("num_tc_tiles", [](const RATcDirectPlanWrapper& w){ return w.plan.num_tc_tiles; })
        .def_property_readonly("num_fp32_rows", [](const RATcDirectPlanWrapper& w){ return w.plan.num_fp32_rows; })
        .def_property_readonly("avg_tc_tile_density", [](const RATcDirectPlanWrapper& w){ return w.plan.avg_tc_tile_density; })
        .def_property_readonly("fp32_group_fraction", [](const RATcDirectPlanWrapper& w){ return w.plan.fp32_group_fraction; })
        .def_property_readonly("plan_bytes", [](const RATcDirectPlanWrapper& w){ return w.plan.plan_bytes; });

    m.def("make_tc_direct_plan", &make_ra_tc_direct_plan_fn,
          "Build R4 Flash TC plan (fixed TC_REORDERED with single-pass tiles)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"),
          pybind11::arg("vals_cpu"), pybind11::arg("M"), pybind11::arg("K"),
          pybind11::arg("N"));

    m.def("run_tc_direct_plan", &run_ra_tc_direct_plan_fn,
          "Run R4 Flash TC SpMM",
          pybind11::arg("plan"), pybind11::arg("B"));

    // --- R3: Locality-tiled ---
    pybind11::class_<RALocalityTiledPlanWrapper, std::shared_ptr<RALocalityTiledPlanWrapper>>(
        m, "RALocalityTiledPlan")
        .def_property_readonly("valid", [](const RALocalityTiledPlanWrapper& w){ return w.valid; })
        .def_property_readonly("M", [](const RALocalityTiledPlanWrapper& w){ return w.plan.M; })
        .def_property_readonly("K", [](const RALocalityTiledPlanWrapper& w){ return w.plan.K; })
        .def_property_readonly("num_panels", [](const RALocalityTiledPlanWrapper& w){ return w.plan.num_panels; })
        .def_property_readonly("avg_cache_hit_rate", [](const RALocalityTiledPlanWrapper& w){ return w.plan.avg_cache_hit_rate; })
        .def_property_readonly("reorder_gain", [](const RALocalityTiledPlanWrapper& w){ return w.plan.reorder_gain; })
        .def_property_readonly("plan_bytes", [](const RALocalityTiledPlanWrapper& w){ return w.plan.plan_bytes; });

    m.def("make_locality_tiled_plan", &make_ra_locality_tiled_plan_fn,
          "Build R3 locality-tiled plan (two-level reordering + B caching)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"),
          pybind11::arg("vals_cpu"), pybind11::arg("M"), pybind11::arg("K"),
          pybind11::arg("N"));

    m.def("run_locality_tiled_plan", &run_ra_locality_tiled_plan_fn,
          "Run R3 locality-tiled SpMM",
          pybind11::arg("plan"), pybind11::arg("B"));

    // --- R5: Community TC ---
    pybind11::class_<RACommunityTCPlanWrapper, std::shared_ptr<RACommunityTCPlanWrapper>>(
        m, "RACommunityTCPlan")
        .def_property_readonly("valid", [](const RACommunityTCPlanWrapper& w){ return w.valid; })
        .def_property_readonly("M", [](const RACommunityTCPlanWrapper& w){ return w.plan.M; })
        .def_property_readonly("K", [](const RACommunityTCPlanWrapper& w){ return w.plan.K; })
        .def_property_readonly("num_communities", [](const RACommunityTCPlanWrapper& w){ return w.plan.num_communities; })
        .def_property_readonly("num_tc_tiles", [](const RACommunityTCPlanWrapper& w){ return w.plan.num_tc_tiles; })
        .def_property_readonly("intra_community_nnz_fraction", [](const RACommunityTCPlanWrapper& w){ return w.plan.intra_community_nnz_fraction; })
        .def_property_readonly("plan_bytes", [](const RACommunityTCPlanWrapper& w){ return w.plan.plan_bytes; });

    m.def("make_community_tc_plan", &make_ra_community_tc_plan_fn,
          "Build R5 community TC plan (community-aware reordering + WMMA)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"),
          pybind11::arg("vals_cpu"), pybind11::arg("M"), pybind11::arg("K"),
          pybind11::arg("N"));

    m.def("run_community_tc_plan", &run_ra_community_tc_plan_fn,
          "Run R5 community TC SpMM",
          pybind11::arg("plan"), pybind11::arg("B"));

    // --- R7: Segment hybrid ---
    pybind11::class_<RASegmentHybridPlanWrapper, std::shared_ptr<RASegmentHybridPlanWrapper>>(
        m, "RASegmentHybridPlan")
        .def_property_readonly("valid", [](const RASegmentHybridPlanWrapper& w){ return w.valid; })
        .def_property_readonly("M", [](const RASegmentHybridPlanWrapper& w){ return w.plan.M; })
        .def_property_readonly("K", [](const RASegmentHybridPlanWrapper& w){ return w.plan.K; })
        .def_property_readonly("num_tc_groups", [](const RASegmentHybridPlanWrapper& w){ return w.plan.num_tc_groups; })
        .def_property_readonly("num_tc_tiles", [](const RASegmentHybridPlanWrapper& w){ return w.plan.num_tc_tiles; })
        .def_property_readonly("tc_nnz_fraction", [](const RASegmentHybridPlanWrapper& w){ return w.plan.tc_nnz_fraction; })
        .def_property_readonly("cuda_nnz_fraction", [](const RASegmentHybridPlanWrapper& w){ return w.plan.cuda_nnz_fraction; })
        .def_property_readonly("plan_bytes", [](const RASegmentHybridPlanWrapper& w){ return w.plan.plan_bytes; });

    m.def("make_segment_hybrid_plan", &make_ra_segment_hybrid_plan_fn,
          "Build R7 segment hybrid plan (TC+CUDA row-level partitioning)",
          pybind11::arg("rowptr_cpu"), pybind11::arg("colind_cpu"),
          pybind11::arg("vals_cpu"), pybind11::arg("M"), pybind11::arg("K"),
          pybind11::arg("N"));

    m.def("run_segment_hybrid_plan", &run_ra_segment_hybrid_plan_fn,
          "Run R7 segment hybrid SpMM",
          pybind11::arg("plan"), pybind11::arg("colind"),
          pybind11::arg("vals"), pybind11::arg("B"));
}
