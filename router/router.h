// ============================================================================
// router.h - Public interface for Regime-Aware Kernel Router
// ============================================================================
#pragma once

#include "../ra_common.h"
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Feature extraction
// ---------------------------------------------------------------------------
RouterFeatures compute_router_features(
    const int*   rowptr,
    const int*   colind,
    int M, int K, int N);

// Exact feature set used by the production eight-rule router. This path is
// O(M), reads rowptr only, and deliberately omits diagnostic tile/locality
// features that do not participate in MAIN routing decisions.
RouterFeatures compute_production_router_features(
    const int* rowptr,
    int M, int K, int N);

// CUDA implementation for GPU-resident CSR input. It reads rowptr on the
// current CUDA stream and transfers only reduction results to the host.
RouterFeatures compute_production_router_features_gpu(
    const int* rowptr,
    int M, int K, int N);

// ---------------------------------------------------------------------------
// Score computation (diagnostic only -- not used for routing decisions)
// ---------------------------------------------------------------------------
RouterScores compute_router_scores(const RouterFeatures& features);

// ---------------------------------------------------------------------------
// Routing dispatch (explainable rule-based hierarchy)
// ---------------------------------------------------------------------------
RouterPlan route_dispatch(
    const RouterFeatures& features,
    const RouterScores& scores,
    Portfolio portfolio = Portfolio::MAIN);

// ---------------------------------------------------------------------------
// Full router pipeline
// ---------------------------------------------------------------------------
RouterPlan make_router_plan(
    const int*   rowptr,
    const int*   colind,
    int M, int K, int N,
    Portfolio portfolio = Portfolio::MAIN);
