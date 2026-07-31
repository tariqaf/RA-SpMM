// ============================================================================
// bench/benchmark.cpp - Shared timing helpers
//
// Timing semantics used throughout the repaired benchmark path:
// - `plan_ms`: host-side build / preprocessing only
// - `exec_ms`: GPU execution only, measured with CUDA events
// - `total_ms`: `plan_ms + exec_ms`
//
// Warm timing must reuse existing plans. Cold timing must include plan build.
// ============================================================================
#include "../ra_common.h"

#include <cuda_runtime.h>

#include <functional>

namespace {

inline float compute_gflops(int nnz, int N, float exec_ms) {
    if (exec_ms <= 0.f) {
        return 0.f;
    }
    const float flops = 2.f * static_cast<float>(nnz) * static_cast<float>(N);
    return flops / (exec_ms * 1e6f);
}

}  // namespace

TimingBreakdown make_timing_breakdown(float plan_ms, float exec_ms, int nnz, int N)
{
    TimingBreakdown timing;
    timing.plan_ms = plan_ms;
    timing.exec_ms = exec_ms;
    timing.total_ms = plan_ms + exec_ms;
    timing.gflops = compute_gflops(nnz, N, exec_ms);
    return timing;
}

float measure_cuda_exec_ms(const std::function<void()>& fn, int warmup_iters, int timed_iters)
{
    if (timed_iters <= 0) {
        return 0.f;
    }

    for (int i = 0; i < warmup_iters; ++i) {
        fn();
    }
    CUDA_CHECK_NEXT(cudaDeviceSynchronize());

    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_stop = nullptr;
    CUDA_CHECK_NEXT(cudaEventCreate(&ev_start));
    CUDA_CHECK_NEXT(cudaEventCreate(&ev_stop));

    CUDA_CHECK_NEXT(cudaEventRecord(ev_start));
    for (int i = 0; i < timed_iters; ++i) {
        fn();
    }
    CUDA_CHECK_NEXT(cudaEventRecord(ev_stop));
    CUDA_CHECK_NEXT(cudaEventSynchronize(ev_stop));

    float total_ms = 0.f;
    CUDA_CHECK_NEXT(cudaEventElapsedTime(&total_ms, ev_start, ev_stop));
    CUDA_CHECK_NEXT(cudaEventDestroy(ev_start));
    CUDA_CHECK_NEXT(cudaEventDestroy(ev_stop));
    return total_ms / static_cast<float>(timed_iters);
}
