// CUDA reduction for the four exact inputs to the production rule router.
#include "router.h"

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>

namespace {

__global__ void row_degree_moments_kernel(
    const int* __restrict__ rowptr,
    int M,
    double* __restrict__ sum,
    double* __restrict__ sumsq)
{
    __shared__ double block_sum[256];
    __shared__ double block_sumsq[256];

    const int tid = threadIdx.x;
    const int start = blockIdx.x * blockDim.x + tid;
    const int stride = blockDim.x * gridDim.x;
    double local_sum = 0.0;
    double local_sumsq = 0.0;
    for (int row = start; row < M; row += stride) {
        const double degree = static_cast<double>(rowptr[row + 1] - rowptr[row]);
        local_sum += degree;
        local_sumsq += degree * degree;
    }

    block_sum[tid] = local_sum;
    block_sumsq[tid] = local_sumsq;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
        if (tid < offset) {
            block_sum[tid] += block_sum[tid + offset];
            block_sumsq[tid] += block_sumsq[tid + offset];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(sum, block_sum[0]);
        atomicAdd(sumsq, block_sumsq[0]);
    }
}

}  // namespace

RouterFeatures compute_production_router_features_gpu(
    const int* rowptr,
    int M, int K, int N)
{
    RouterFeatures f{};
    f.matrix_M = M;
    f.matrix_K = K;
    f.output_dim_N = N;
    if (rowptr == nullptr || M <= 0 || K <= 0) {
        return f;
    }

    constexpr int kThreads = 256;
    const int blocks = std::clamp((M + kThreads - 1) / kThreads, 1, 1024);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();

    int device = 0;
    CUDA_CHECK_NEXT(cudaGetDevice(&device));
    auto options = at::TensorOptions().dtype(at::kDouble).device(at::kCUDA, device);
    auto device_moments_tensor = at::empty({2}, options);
    double* device_moments = device_moments_tensor.data_ptr<double>();
    CUDA_CHECK_NEXT(cudaMemsetAsync(device_moments, 0, 2 * sizeof(double), stream));
    row_degree_moments_kernel<<<blocks, kThreads, 0, stream>>>(
        rowptr, M, device_moments, device_moments + 1);
    CUDA_CHECK_KERNEL();

    double host_moments[2] = {0.0, 0.0};
    CUDA_CHECK_NEXT(cudaMemcpyAsync(
        host_moments, device_moments, sizeof(host_moments),
        cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_NEXT(cudaStreamSynchronize(stream));

    const double avg = host_moments[0] / static_cast<double>(M);
    const double variance = std::max(
        0.0, host_moments[1] / static_cast<double>(M) - avg * avg);
    const double std_dev = std::sqrt(variance);
    f.total_nnz = static_cast<int>(host_moments[0]);
    f.avg_nnz_per_row = static_cast<float>(avg);
    f.std_nnz_per_row = static_cast<float>(std_dev);
    f.degree_cv = (avg > 1e-6) ? static_cast<float>(std_dev / avg) : 0.0f;
    return f;
}
