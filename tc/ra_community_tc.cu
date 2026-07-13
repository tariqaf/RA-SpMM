// ============================================================================
// ra_community_tc.cu - R5: ME-BCRS tile SpMM over locality-ordered windows
//
// Same engine as TC_DIRECT (FlashSparse-style 8x1 vectors, swapped-operand
// mma.m16n8k8, FP16 inputs / FP32 accumulate) with one difference: rows are
// sorted by their leading neighbor column before windowing, so rows that
// share neighborhoods land in the same 8-row window. Shared neighbors then
// collapse into single 8x1 vectors, reducing gather traffic on community
// graphs. Ordering is a single O(M log M) parallel sort — no label
// propagation, no CSC, no CSR rebuild.
//
// The kernel scatters each window slot to its original C row through
// d_win_rows, so C needs no unpermutation.
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

namespace {

constexpr int kWindow      = 8;
constexpr int kVecPerBlock = 8;

inline uint16_t float_to_half_bits(float value) {
    const half h = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

template <typename T>
T* upload_vec_ct(const std::vector<T>& values) {
    if (values.empty()) return nullptr;
    T* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, values.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, values.data(),
                               values.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
    return d_ptr;
}

// ---------------------------------------------------------------------------
// Deterministic parallel stable sort (byte-identical to serial stable_sort).
// ---------------------------------------------------------------------------
template <typename It, typename Cmp>
void ra_parallel_stable_sort(It first, It last, Cmp cmp) {
    const std::ptrdiff_t n = last - first;
    if (n < (1 << 15)) {
        std::stable_sort(first, last, cmp);
        return;
    }
    constexpr int kChunks = 16;
    std::ptrdiff_t bounds[kChunks + 1];
    for (int c = 0; c <= kChunks; ++c) bounds[c] = n * c / kChunks;
    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < kChunks; ++c) {
        std::stable_sort(first + bounds[c], first + bounds[c + 1], cmp);
    }
    for (int width = 1; width < kChunks; width *= 2) {
        #pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < kChunks; c += 2 * width) {
            const int mid = c + width;
            const int endc = std::min(c + 2 * width, kChunks);
            if (mid < endc) {
                std::inplace_merge(first + bounds[c], first + bounds[mid],
                                   first + bounds[endc], cmp);
            }
        }
    }
}

__global__ void ct_convert_b_to_half_kernel(
    const float* __restrict__ B,
    __half2* __restrict__ Bh2,
    i64 total_pairs)
{
    for (i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_pairs; idx += (i64)gridDim.x * blockDim.x) {
        const float2 f2 = reinterpret_cast<const float2*>(B)[idx];
        Bh2[idx] = __floats2half2_rn(f2.x, f2.y);
    }
}

__global__ void ct_convert_b_to_half_tail_kernel(
    const float* __restrict__ B,
    __half* __restrict__ Bh,
    i64 begin, i64 total)
{
    const i64 idx = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) Bh[idx] = __float2half_rn(B[idx]);
}

// Permuted-window variant of the swapped-operand mma kernel: output rows go
// through d_win_rows (original row per window slot, -1 for padding).
__global__ void ct_fs_tile_spmm_perm_kernel(
    const int*      __restrict__ block_offsets,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const int*      __restrict__ win_rows,
    const __half*   __restrict__ Bh,
    float*          __restrict__ C,
    int N, int num_windows)
{
    const int w    = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0   = (blockIdx.y * 4 + warp) * 16;
    if (w >= num_windows || n0 >= N) return;

    const int g   = lane >> 2;
    const int tid = lane & 3;

    const bool p_lo = (n0 + g)     < N;
    const bool p_hi = (n0 + g + 8) < N;

    const int b_begin = block_offsets[w];
    const int b_end   = block_offsets[w + 1];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (int b = b_begin; b < b_end; ++b) {
        const uint32_t bfrag = __ldg(reinterpret_cast<const uint32_t*>(
            vals + (i64)b * 64) + lane);
        const int vbase = b * kVecPerBlock + tid * 2;
        const int c0 = __ldg(atox + vbase);
        const int c1 = __ldg(atox + vbase + 1);

        __half a0 = __float2half(0.f), a1 = a0, a2 = a0, a3 = a0;
        if (p_lo) {
            a0 = __ldg(Bh + (i64)c0 * N + n0 + g);
            a1 = __ldg(Bh + (i64)c1 * N + n0 + g);
        }
        if (p_hi) {
            a2 = __ldg(Bh + (i64)c0 * N + n0 + g + 8);
            a3 = __ldg(Bh + (i64)c1 * N + n0 + g + 8);
        }
        const uint32_t afrag0 =
            (static_cast<uint32_t>(__half_as_ushort(a1)) << 16) |
            __half_as_ushort(a0);
        const uint32_t afrag1 =
            (static_cast<uint32_t>(__half_as_ushort(a3)) << 16) |
            __half_as_ushort(a2);

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(afrag0), "r"(afrag1), "r"(bfrag));
    }

    const int slot0 = w * kWindow + tid * 2;
    const int r0 = win_rows[slot0];
    const int r1 = win_rows[slot0 + 1];
    if (p_lo) {
        if (r0 >= 0) C[(i64)r0 * N + n0 + g] = d0;
        if (r1 >= 0) C[(i64)r1 * N + n0 + g] = d1;
    }
    if (p_hi) {
        if (r0 >= 0) C[(i64)r0 * N + n0 + g + 8] = d2;
        if (r1 >= 0) C[(i64)r1 * N + n0 + g + 8] = d3;
    }
}

}  // anonymous namespace

// ============================================================================
// make_ra_community_tc_plan
// ============================================================================
void make_ra_community_tc_plan(
    RACommunityTCPlan& plan,
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M, int K, int N)
{
    plan = RACommunityTCPlan{};
    plan.M = M;
    plan.K = K;

    if (M <= 0 || K <= 0 || N < 16) return;
    const i64 total_nnz = h_rowptr[M];
    if (total_nnz <= 0) return;

    // ---- Locality ordering: sort rows by (first neighbor col, row id) ----
    std::vector<int> order(M);
    std::iota(order.begin(), order.end(), 0);
    ra_parallel_stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const int ka = (h_rowptr[a] < h_rowptr[a + 1]) ? h_col[h_rowptr[a]] : INT_MAX;
        const int kb = (h_rowptr[b] < h_rowptr[b + 1]) ? h_col[h_rowptr[b]] : INT_MAX;
        if (ka != kb) return ka < kb;
        return a < b;
    });

    const int num_windows = (M + kWindow - 1) / kWindow;
    std::vector<int> win_rows(static_cast<size_t>(num_windows) * kWindow, -1);
    for (int i = 0; i < M; ++i) win_rows[i] = order[i];

    std::vector<int> win_blocks(num_windows, 0);
    std::vector<std::vector<int>>      w_atox(num_windows);
    std::vector<std::vector<uint16_t>> w_vals(num_windows);

    #pragma omp parallel
    {
        std::vector<std::pair<int, std::pair<int, float>>> entries;
        #pragma omp for schedule(dynamic, 64)
        for (int w = 0; w < num_windows; ++w) {
            entries.clear();
            for (int slot = 0; slot < kWindow; ++slot) {
                const int row = win_rows[static_cast<size_t>(w) * kWindow + slot];
                if (row < 0) continue;
                for (int p = h_rowptr[row]; p < h_rowptr[row + 1]; ++p) {
                    entries.push_back({h_col[p], {slot, h_val[p]}});
                }
            }
            if (entries.empty()) continue;
            std::sort(entries.begin(), entries.end(),
                      [](const auto& a, const auto& b) {
                          if (a.first != b.first) return a.first < b.first;
                          return a.second.first < b.second.first;
                      });

            int nvec = 1;
            for (size_t i = 1; i < entries.size(); ++i) {
                if (entries[i].first != entries[i - 1].first) ++nvec;
            }
            const int blocks = (nvec + kVecPerBlock - 1) / kVecPerBlock;
            win_blocks[w] = blocks;

            auto& atox = w_atox[w];
            auto& vals = w_vals[w];
            atox.assign(static_cast<size_t>(blocks) * kVecPerBlock, 0);
            vals.assign(static_cast<size_t>(blocks) * 64, 0);

            int slot_v = -1;
            int prev_col = -1;
            for (const auto& e : entries) {
                if (e.first != prev_col) {
                    ++slot_v;
                    prev_col = e.first;
                    atox[slot_v] = e.first;
                }
                const int blk = slot_v / kVecPerBlock;
                const int k   = slot_v % kVecPerBlock;
                const int r   = e.second.first;
                // PTX B-fragment order (see ra_tc_direct.cu).
                const size_t idx = static_cast<size_t>(blk) * 64 +
                    (static_cast<size_t>(r) * 4 + k / 2) * 2 + (k & 1);
                vals[idx] = float_to_half_bits(e.second.second);
            }
        }
    }

    std::vector<int> block_offsets(num_windows + 1, 0);
    for (int w = 0; w < num_windows; ++w) {
        block_offsets[w + 1] = block_offsets[w] + win_blocks[w];
    }
    const i64 total_blocks = block_offsets[num_windows];

    std::vector<int>      atox_all(static_cast<size_t>(total_blocks) * kVecPerBlock);
    std::vector<uint16_t> vals_all(static_cast<size_t>(total_blocks) * 64);
    #pragma omp parallel for schedule(dynamic, 256)
    for (int w = 0; w < num_windows; ++w) {
        std::copy(w_atox[w].begin(), w_atox[w].end(),
                  atox_all.begin() + static_cast<size_t>(block_offsets[w]) * kVecPerBlock);
        std::copy(w_vals[w].begin(), w_vals[w].end(),
                  vals_all.begin() + static_cast<size_t>(block_offsets[w]) * 64);
    }
    const i64 total_vectors = total_blocks * kVecPerBlock;

    plan.d_block_offsets = upload_vec_ct(block_offsets);
    plan.d_atox          = upload_vec_ct(atox_all);
    plan.d_vals_f16      = upload_vec_ct(vals_all);
    plan.d_win_rows      = upload_vec_ct(win_rows);

    plan.num_windows     = num_windows;
    plan.num_groups      = num_windows;
    plan.num_communities = num_windows;
    plan.num_tc_tiles    = static_cast<int>(total_blocks);
    plan.intra_community_nnz_fraction = (total_vectors > 0)
        ? static_cast<float>(static_cast<double>(total_nnz) /
                             (static_cast<double>(total_vectors) * kWindow))
        : 0.f;
    plan.avg_tc_tile_density = plan.intra_community_nnz_fraction;
    plan.plan_bytes = block_offsets.size() * sizeof(int) +
                      atox_all.size() * sizeof(int) +
                      win_rows.size() * sizeof(int) +
                      vals_all.size() * sizeof(uint16_t);
    plan.active = true;
}

// ============================================================================
// run_ra_community_tc_plan
// ============================================================================
void run_ra_community_tc_plan(
    const RACommunityTCPlan& plan,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream)
{
    if (!plan.active || plan.M <= 0 || N <= 0) return;

    const i64 total = (i64)plan.K * N;
    const size_t need = static_cast<size_t>(total) * sizeof(uint16_t);
    if (plan.bhalf_capacity < need) {
        if (plan.d_bhalf) cudaFree(plan.d_bhalf);
        CUDA_CHECK_NEXT(cudaMalloc(&plan.d_bhalf, need));
        plan.bhalf_capacity = need;
    }

    const i64 pairs = total / 2;
    if (pairs > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>(
            std::min<i64>((pairs + threads - 1) / threads, 65535));
        ct_convert_b_to_half_kernel<<<blocks, threads, 0, stream>>>(
            d_B, reinterpret_cast<__half2*>(plan.d_bhalf), pairs);
    }
    if (total % 2 != 0) {
        ct_convert_b_to_half_tail_kernel<<<1, 32, 0, stream>>>(
            d_B, reinterpret_cast<__half*>(plan.d_bhalf), pairs * 2, total);
    }

    dim3 grid(plan.num_windows, (N + 63) / 64);
    ct_fs_tile_spmm_perm_kernel<<<grid, 128, 0, stream>>>(
        plan.d_block_offsets, plan.d_atox, plan.d_vals_f16, plan.d_win_rows,
        reinterpret_cast<const __half*>(plan.d_bhalf), d_C,
        N, plan.num_windows);

    CUDA_CHECK_KERNEL();
}

// ============================================================================
// free_ra_community_tc_plan
// ============================================================================
void free_ra_community_tc_plan(RACommunityTCPlan& plan) {
    if (plan.d_block_offsets) { cudaFree(plan.d_block_offsets); plan.d_block_offsets = nullptr; }
    if (plan.d_atox)          { cudaFree(plan.d_atox);          plan.d_atox          = nullptr; }
    if (plan.d_vals_f16)      { cudaFree(plan.d_vals_f16);      plan.d_vals_f16      = nullptr; }
    if (plan.d_win_rows)      { cudaFree(plan.d_win_rows);      plan.d_win_rows      = nullptr; }
    if (plan.d_bhalf)         { cudaFree(plan.d_bhalf);         plan.d_bhalf         = nullptr; }
    plan.bhalf_capacity = 0;
    plan.num_windows = plan.num_groups = plan.num_communities = plan.num_tc_tiles = 0;
    plan.active = false;
    plan.plan_bytes = 0;
}
