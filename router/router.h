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
