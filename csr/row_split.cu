// ============================================================================
// row_split.cu - ROW_SPLIT_CUDA SpMM using RowSplitPlan
//
// Regular block part:
// - Per row, the largest 32-aligned nnz prefix is treated as the regular part.
// - Short regular rows keep one-CTA-per-row ownership, which avoids global atomics.
// - Long regular rows use row-column tiling so larger N can expose more parallelism
//   while keeping deterministic row ownership.
//
// Residual part:
// - The remaining 0..31 nnz suffix is handled by a lightweight residual kernel.
// - Residual updates run after the regular path and add into already-owned rows.
//
// Long-row segment descriptors are still recorded in the plan so future work can
// add finer sub-CTA decomposition. This is an intermediate RoDe-inspired import,
// not a full RoDe reproduction.
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace {

constexpr int kLongRowSegmentThreshold = 4;   // 4 * 32 = 128 regular nnz
constexpr int kLongTileCols = 256;
constexpr int kLongTileVec4 = kLongTileCols / 4;

template <typename T>
T* upload_rs(const std::vector<T>& values) {
    if (values.empty()) {
        return nullptr;
    }
    T* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, values.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, values.data(),
                               values.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
    return d_ptr;
}

__global__ void row_split_short_scalar_kernel(
    const int* __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) {
        return;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;

    const int row = row_ids[row_idx];
    const int start = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n = warp * 32 + lane; n < N; n += warps_per_block * 32) {
        float acc = 0.f;
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                acc += d_val[p] * __ldg(&B[(i64)d_col[p] * N + n]);
            }
        }
        C[(i64)row * N + n] = acc;
    }
}

__global__ void row_split_short_vec4_kernel(
    const int* __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) {
        return;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int N4 = N / 4;

    const int row = row_ids[row_idx];
    const int start = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n4 = warp * 32 + lane; n4 < N4; n4 += warps_per_block * 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                const float4* b_ptr = reinterpret_cast<const float4*>(B + (i64)d_col[p] * N);
                const float4 b4 = __ldg(b_ptr + n4);
                const float a = d_val[p];
                acc.x += a * b4.x;
                acc.y += a * b4.y;
                acc.z += a * b4.z;
                acc.w += a * b4.w;
            }
        }
        float4* c_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
        c_ptr[n4] = acc;
    }
}

__global__ void row_split_long_scalar_kernel(
    const int* __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) {
        return;
    }

    const int tile_start = blockIdx.y * kLongTileCols;
    const int tile_end = min(tile_start + kLongTileCols, N);
    const int row = row_ids[row_idx];
    const int start = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n = tile_start + threadIdx.x; n < tile_end; n += blockDim.x) {
        float acc = 0.f;
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                acc += d_val[p] * __ldg(&B[(i64)d_col[p] * N + n]);
            }
        }
        C[(i64)row * N + n] = acc;
    }
}

__global__ void row_split_long_vec4_kernel(
    const int* __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    const int* __restrict__ starts,
    const int* __restrict__ block_nnz_list,
    int num_rows,
    int N)
{
    const int row_idx = blockIdx.x;
    if (row_idx >= num_rows) {
        return;
    }

    const int N4 = N / 4;
    const int tile_start4 = blockIdx.y * kLongTileVec4;
    const int tile_end4 = min(tile_start4 + kLongTileVec4, N4);
    const int row = row_ids[row_idx];
    const int start = starts[row_idx];
    const int block_nnz = block_nnz_list[row_idx];

    for (int n4 = tile_start4 + threadIdx.x; n4 < tile_end4; n4 += blockDim.x) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int seg = 0; seg < block_nnz; seg += 32) {
#pragma unroll
            for (int e = 0; e < 32; ++e) {
                const int p = start + seg + e;
                const float4* b_ptr = reinterpret_cast<const float4*>(B + (i64)d_col[p] * N);
                const float4 b4 = __ldg(b_ptr + n4);
                const float a = d_val[p];
                acc.x += a * b4.x;
                acc.y += a * b4.y;
                acc.z += a * b4.z;
                acc.w += a * b4.w;
            }
        }
        float4* c_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
        c_ptr[n4] = acc;
    }
}

__global__ void row_split_residual_scalar_kernel(
    const int* __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ d_res_row_ids,
    const int* __restrict__ d_res_starts,
    const int* __restrict__ d_res_lengths,
    int num_residual,
    int N)
{
    const int residual_idx = blockIdx.x;
    if (residual_idx >= num_residual) {
        return;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;

    const int row = d_res_row_ids[residual_idx];
    const int start = d_res_starts[residual_idx];
    const int len = d_res_lengths[residual_idx];

    for (int n = warp * 32 + lane; n < N; n += warps_per_block * 32) {
        float acc = 0.f;
        for (int p = 0; p < len; ++p) {
            acc += d_val[start + p] * __ldg(&B[(i64)d_col[start + p] * N + n]);
        }
        C[(i64)row * N + n] += acc;
    }
}

__global__ void row_split_residual_vec4_kernel(
    const int* __restrict__ d_col,
    const float* __restrict__ d_val,
    const float* __restrict__ B,
    float* __restrict__ C,
    const int* __restrict__ d_res_row_ids,
    const int* __restrict__ d_res_starts,
    const int* __restrict__ d_res_lengths,
    int num_residual,
    int N)
{
    const int residual_idx = blockIdx.x;
    if (residual_idx >= num_residual) {
        return;
    }

    const int warp = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int warps_per_block = blockDim.x / 32;
    const int N4 = N / 4;

    const int row = d_res_row_ids[residual_idx];
    const int start = d_res_starts[residual_idx];
    const int len = d_res_lengths[residual_idx];

    for (int n4 = warp * 32 + lane; n4 < N4; n4 += warps_per_block * 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = 0; p < len; ++p) {
            const float4* b_ptr = reinterpret_cast<const float4*>(B + (i64)d_col[start + p] * N);
            const float4 b4 = __ldg(b_ptr + n4);
            const float a = d_val[start + p];
            acc.x += a * b4.x;
            acc.y += a * b4.y;
            acc.z += a * b4.z;
            acc.w += a * b4.w;
        }
        float4* c_ptr = reinterpret_cast<float4*>(C + (i64)row * N);
        float4 cur = c_ptr[n4];
        cur.x += acc.x;
        cur.y += acc.y;
        cur.z += acc.z;
        cur.w += acc.w;
        c_ptr[n4] = cur;
    }
}

}  // namespace

RowSplitPlan make_row_split_plan(const int* h_rowptr, int M, int K)
{
    RowSplitPlan plan;
    plan.M = M;
    plan.K = K;

    if (M <= 0) {
        return plan;
    }

    std::vector<int> regular_row_ids;
    std::vector<int> regular_starts;
    std::vector<int> regular_block_nnz;
    std::vector<int> short_row_ids;
    std::vector<int> short_starts;
    std::vector<int> short_block_nnz;
    std::vector<int> long_row_ids;
    std::vector<int> long_starts;
    std::vector<int> long_block_nnz;
    std::vector<int> long_num_segments;
    std::vector<int> long_seg_row_ids;
    std::vector<int> long_seg_starts;
    std::vector<int> res_row_ids;
    std::vector<int> res_starts;
    std::vector<int> res_lengths;
    std::vector<int> row_block_nnz(M, 0);
    std::vector<int> row_residual_nnz(M, 0);
    int64_t regular_nnz_total = 0;
    int64_t residual_nnz_total = 0;
    int split_long_rows = 0;

    for (int row = 0; row < M; ++row) {
        const int start = h_rowptr[row];
        const int len = h_rowptr[row + 1] - start;
        const int block_nnz = (len / plan.T) * plan.T;
        const int residual_nnz = len - block_nnz;
        const int num_segments = block_nnz / plan.T;

        row_block_nnz[row] = block_nnz;
        row_residual_nnz[row] = residual_nnz;
        regular_nnz_total += block_nnz;
        residual_nnz_total += residual_nnz;

        if (block_nnz > 0) {
            regular_row_ids.push_back(row);
            regular_starts.push_back(start);
            regular_block_nnz.push_back(block_nnz);

            if (num_segments >= kLongRowSegmentThreshold) {
                long_row_ids.push_back(row);
                long_starts.push_back(start);
                long_block_nnz.push_back(block_nnz);
                long_num_segments.push_back(num_segments);
            } else {
                short_row_ids.push_back(row);
                short_starts.push_back(start);
                short_block_nnz.push_back(block_nnz);
            }
        }

        if (num_segments > 1) {
            ++split_long_rows;
            for (int offset = 0; offset < block_nnz; offset += plan.T) {
                long_seg_row_ids.push_back(row);
                long_seg_starts.push_back(start + offset);
            }
        }

        if (residual_nnz > 0) {
            res_row_ids.push_back(row);
            res_starts.push_back(start + block_nnz);
            res_lengths.push_back(residual_nnz);
        }
    }

    plan.num_regular_rows = static_cast<int>(regular_row_ids.size());
    plan.num_short_rows = static_cast<int>(short_row_ids.size());
    plan.num_long_rows = static_cast<int>(long_row_ids.size());
    plan.num_long_segments = static_cast<int>(long_seg_row_ids.size());
    plan.num_residual = static_cast<int>(res_row_ids.size());
    plan.num_split_long_rows = split_long_rows;
    plan.regular_nnz_fraction = (regular_nnz_total + residual_nnz_total > 0)
        ? static_cast<float>(regular_nnz_total) /
          static_cast<float>(regular_nnz_total + residual_nnz_total)
        : 0.f;
    plan.residual_nnz_fraction = (regular_nnz_total + residual_nnz_total > 0)
        ? static_cast<float>(residual_nnz_total) /
          static_cast<float>(regular_nnz_total + residual_nnz_total)
        : 0.f;
    plan.avg_segments_per_long_row = (split_long_rows > 0)
        ? static_cast<float>(plan.num_long_segments) / static_cast<float>(split_long_rows)
        : 0.f;

    plan.d_regular_row_ids = upload_rs(regular_row_ids);
    plan.d_regular_starts = upload_rs(regular_starts);
    plan.d_regular_block_nnz = upload_rs(regular_block_nnz);
    plan.d_short_row_ids = upload_rs(short_row_ids);
    plan.d_short_starts = upload_rs(short_starts);
    plan.d_short_block_nnz = upload_rs(short_block_nnz);
    plan.d_long_row_ids = upload_rs(long_row_ids);
    plan.d_long_starts = upload_rs(long_starts);
    plan.d_long_block_nnz = upload_rs(long_block_nnz);
    plan.d_long_num_segments = upload_rs(long_num_segments);
    plan.d_long_seg_row_ids = upload_rs(long_seg_row_ids);
    plan.d_long_seg_starts = upload_rs(long_seg_starts);
    plan.d_res_row_ids = upload_rs(res_row_ids);
    plan.d_res_starts = upload_rs(res_starts);
    plan.d_res_lengths = upload_rs(res_lengths);
    plan.d_row_block_nnz = upload_rs(row_block_nnz);
    plan.d_row_residual_nnz = upload_rs(row_residual_nnz);

    plan.plan_bytes =
        regular_row_ids.size() * sizeof(int) +
        regular_starts.size() * sizeof(int) +
        regular_block_nnz.size() * sizeof(int) +
        short_row_ids.size() * sizeof(int) +
        short_starts.size() * sizeof(int) +
        short_block_nnz.size() * sizeof(int) +
        long_row_ids.size() * sizeof(int) +
        long_starts.size() * sizeof(int) +
        long_block_nnz.size() * sizeof(int) +
        long_num_segments.size() * sizeof(int) +
        long_seg_row_ids.size() * sizeof(int) +
        long_seg_starts.size() * sizeof(int) +
        res_row_ids.size() * sizeof(int) +
        res_starts.size() * sizeof(int) +
        res_lengths.size() * sizeof(int) +
        row_block_nnz.size() * sizeof(int) +
        row_residual_nnz.size() * sizeof(int);

    return plan;
}

void run_row_split_plan(
    const RowSplitPlan& plan,
    const int* d_col,
    const float* d_val,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream)
{
    if (plan.M <= 0 || N <= 0) {
        return;
    }

    CUDA_CHECK_NEXT(cudaMemsetAsync(d_C, 0, (i64)plan.M * N * sizeof(float), stream));

    constexpr int kThreads = 128;
    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
    const bool use_vec4 = (N % 4 == 0) && b_aligned && c_aligned;

    if (plan.num_short_rows > 0) {
        if (use_vec4) {
            row_split_short_vec4_kernel<<<plan.num_short_rows, kThreads, 0, stream>>>(
                d_col, d_val, d_B, d_C,
                plan.d_short_row_ids,
                plan.d_short_starts,
                plan.d_short_block_nnz,
                plan.num_short_rows,
                N);
        } else {
            row_split_short_scalar_kernel<<<plan.num_short_rows, kThreads, 0, stream>>>(
                d_col, d_val, d_B, d_C,
                plan.d_short_row_ids,
                plan.d_short_starts,
                plan.d_short_block_nnz,
                plan.num_short_rows,
                N);
        }
        CUDA_CHECK_KERNEL();
    }

    if (plan.num_long_rows > 0) {
        const dim3 long_grid(plan.num_long_rows, (N + kLongTileCols - 1) / kLongTileCols);
        if (use_vec4) {
            row_split_long_vec4_kernel<<<long_grid, kThreads, 0, stream>>>(
                d_col, d_val, d_B, d_C,
                plan.d_long_row_ids,
                plan.d_long_starts,
                plan.d_long_block_nnz,
                plan.num_long_rows,
                N);
        } else {
            row_split_long_scalar_kernel<<<long_grid, kThreads, 0, stream>>>(
                d_col, d_val, d_B, d_C,
                plan.d_long_row_ids,
                plan.d_long_starts,
                plan.d_long_block_nnz,
                plan.num_long_rows,
                N);
        }
        CUDA_CHECK_KERNEL();
    }

    if (plan.num_residual > 0) {
        if (use_vec4) {
            row_split_residual_vec4_kernel<<<plan.num_residual, kThreads, 0, stream>>>(
                d_col, d_val, d_B, d_C,
                plan.d_res_row_ids,
                plan.d_res_starts,
                plan.d_res_lengths,
                plan.num_residual,
                N);
        } else {
            row_split_residual_scalar_kernel<<<plan.num_residual, kThreads, 0, stream>>>(
                d_col, d_val, d_B, d_C,
                plan.d_res_row_ids,
                plan.d_res_starts,
                plan.d_res_lengths,
                plan.num_residual,
                N);
        }
        CUDA_CHECK_KERNEL();
    }
}

void free_row_split_plan(RowSplitPlan& plan)
{
    if (plan.d_regular_row_ids) { cudaFree(plan.d_regular_row_ids); plan.d_regular_row_ids = nullptr; }
    if (plan.d_regular_starts) { cudaFree(plan.d_regular_starts); plan.d_regular_starts = nullptr; }
    if (plan.d_regular_block_nnz) { cudaFree(plan.d_regular_block_nnz); plan.d_regular_block_nnz = nullptr; }
    if (plan.d_short_row_ids) { cudaFree(plan.d_short_row_ids); plan.d_short_row_ids = nullptr; }
    if (plan.d_short_starts) { cudaFree(plan.d_short_starts); plan.d_short_starts = nullptr; }
    if (plan.d_short_block_nnz) { cudaFree(plan.d_short_block_nnz); plan.d_short_block_nnz = nullptr; }
    if (plan.d_long_row_ids) { cudaFree(plan.d_long_row_ids); plan.d_long_row_ids = nullptr; }
    if (plan.d_long_starts) { cudaFree(plan.d_long_starts); plan.d_long_starts = nullptr; }
    if (plan.d_long_block_nnz) { cudaFree(plan.d_long_block_nnz); plan.d_long_block_nnz = nullptr; }
    if (plan.d_long_num_segments) { cudaFree(plan.d_long_num_segments); plan.d_long_num_segments = nullptr; }
    if (plan.d_long_seg_row_ids) { cudaFree(plan.d_long_seg_row_ids); plan.d_long_seg_row_ids = nullptr; }
    if (plan.d_long_seg_starts) { cudaFree(plan.d_long_seg_starts); plan.d_long_seg_starts = nullptr; }
    if (plan.d_res_row_ids) { cudaFree(plan.d_res_row_ids); plan.d_res_row_ids = nullptr; }
    if (plan.d_res_starts) { cudaFree(plan.d_res_starts); plan.d_res_starts = nullptr; }
    if (plan.d_res_lengths) { cudaFree(plan.d_res_lengths); plan.d_res_lengths = nullptr; }
    if (plan.d_row_block_nnz) { cudaFree(plan.d_row_block_nnz); plan.d_row_block_nnz = nullptr; }
    if (plan.d_row_residual_nnz) { cudaFree(plan.d_row_residual_nnz); plan.d_row_residual_nnz = nullptr; }

    plan.num_regular_rows = 0;
    plan.num_short_rows = 0;
    plan.num_long_rows = 0;
    plan.num_long_segments = 0;
    plan.num_residual = 0;
    plan.plan_bytes = 0;
}
