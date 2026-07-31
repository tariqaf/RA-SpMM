// ============================================================================
// ra_locality_tiled.cu - R3: LOCALITY_TILED SpMM for reordered locality regime
//
// Regime: reordered_locality_proxy 0.24-0.70, recoverable locality via row
//         reordering. Matrices with moderate column-pattern similarity that
//         becomes exploitable after sorting rows by centroid/span.
//
// Design: Two-level reordering (centroid bucket + within-panel length sort)
// followed by tiled execution with shared-memory B caching. After reordering,
// adjacent rows share many column indices, enabling high B-matrix reuse within
// 32-row panels.
//
// Unlike TC_DIRECT (which targets TC-friendly matrices with high tile fill),
// LOCALITY_TILED targets matrices with MODERATE locality that becomes
// exploitable after reordering -- but the tiles are too sparse for TC to help.
// Uses CUDA-core scalar execution with shared-memory B caching, not WMMA.
//
// Kernel:
//   One CTA per panel (32 reordered rows). 256 threads (8 warps).
//   For each N-strip of width 32:
//     Phase 1: Cooperatively load B columns in [k_start, k_start+64) into
//              shared memory B_cache[64][33] (~8.4 KB, padded for banks).
//     Phase 2: Each warp processes ~4 rows. For each nnz, if the column falls
//              within the cached range, read from B_cache (fast); otherwise
//              fall back to __ldg from global memory.
//     Phase 3: Write results to C via perm_inv scatter.
//
// Target: Ampere SM_86 (RTX 3090, RTX A6000), CUDA 12.x
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <utility>
#include <vector>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
namespace {

constexpr int kPanelRows       = 32;   // rows per panel
constexpr int kCacheK          = 64;   // columns cached in shared memory
constexpr int kNStrip          = 32;   // N columns processed per pass
constexpr int kCTAThreads      = 256;  // 8 warps per CTA
constexpr int kNumWarps        = kCTAThreads / 32;
constexpr int kSmemPad         = 33;   // 33 instead of 32 to avoid bank conflicts
constexpr int kSignatureBuckets = 64;  // 64-bit Jaccard signature buckets
constexpr int kCentroidBuckets  = 32;  // K/32 centroid buckets for sorting

// ---------------------------------------------------------------------------
// Upload helper
// ---------------------------------------------------------------------------
template <typename T>
T* lt_upload(const std::vector<T>& v) {
    if (v.empty()) return nullptr;
    T* d = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d, v.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d, v.data(), v.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
    return d;
}

// ---------------------------------------------------------------------------
// Row ordering info for centroid-based reordering
// ---------------------------------------------------------------------------
struct RowOrderInfo {
    int row      = 0;
    int len      = 0;
    int min_col  = 0;
    int max_col  = 0;
    float centroid = 0.f;
    uint64_t signature = 0;
};

// ---------------------------------------------------------------------------
// Portable popcount for 64-bit
// ---------------------------------------------------------------------------
inline int popcount64(uint64_t x) {
#if defined(__GNUG__)
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) {
        x &= (x - 1);
        ++count;
    }
    return count;
#endif
}

// ---------------------------------------------------------------------------
// Jaccard similarity from 64-bit signatures
// ---------------------------------------------------------------------------
inline float jaccard_u64(uint64_t a, uint64_t b) {
    const int uni = popcount64(a | b);
    if (uni == 0) return 0.f;
    return static_cast<float>(popcount64(a & b)) / static_cast<float>(uni);
}

// ---------------------------------------------------------------------------
// Compute panel compactness: average (nnz / (rows * span)) across panels
// ---------------------------------------------------------------------------
float compute_panel_compactness(
    const std::vector<int>& order,
    const std::vector<RowOrderInfo>& info,
    int K)
{
    if (order.empty()) return 0.f;
    double compactness_sum = 0.0;
    int panel_count = 0;
    for (int base = 0; base < static_cast<int>(order.size());
         base += kPanelRows) {
        const int end = std::min(static_cast<int>(order.size()),
                                 base + kPanelRows);
        int min_col = K;
        int max_col = -1;
        int64_t panel_nnz = 0;
        for (int idx = base; idx < end; ++idx) {
            const RowOrderInfo& ri = info[order[idx]];
            panel_nnz += ri.len;
            if (ri.len > 0) {
                min_col = std::min(min_col, ri.min_col);
                max_col = std::max(max_col, ri.max_col);
            }
        }
        const int span = (max_col >= min_col) ? (max_col - min_col + 1)
                                              : std::max(1, K);
        compactness_sum += static_cast<double>(panel_nnz) /
                           static_cast<double>(std::max(1, (end - base) *
                                              std::max(1, span)));
        ++panel_count;
    }
    return (panel_count > 0)
        ? static_cast<float>(compactness_sum / static_cast<double>(panel_count))
        : 0.f;
}

// ---------------------------------------------------------------------------
// Find optimal k_start for a panel: the 64-wide window covering the most nnz.
// Uses a sliding window over the sorted unique columns of all rows in the panel.
// ---------------------------------------------------------------------------
int find_best_cache_window(
    const int* h_rowptr,
    const int* h_col,
    const std::vector<int>& order,
    int panel_start,
    int panel_end,
    int K)
{
    // Collect all column indices in this panel
    std::vector<int> cols;
    for (int idx = panel_start; idx < panel_end; ++idx) {
        const int orig = order[idx];
        for (int p = h_rowptr[orig]; p < h_rowptr[orig + 1]; ++p) {
            cols.push_back(h_col[p]);
        }
    }
    if (cols.empty()) return 0;

    std::sort(cols.begin(), cols.end());

    // Sliding window: find the position where a window of width kCacheK
    // covers the most entries
    int best_count = 0;
    int best_start = cols[0];
    int right = 0;
    const int n = static_cast<int>(cols.size());
    for (int left = 0; left < n; ++left) {
        while (right < n && cols[right] < cols[left] + kCacheK) {
            ++right;
        }
        const int count = right - left;
        if (count > best_count) {
            best_count = count;
            best_start = cols[left];
        }
    }
    return best_start;
}

}  // namespace

// ===========================================================================
// CUDA Kernel: locality_tiled_panel_kernel
//
// One CTA per panel of kPanelRows reordered rows. Processes N in strips of
// kNStrip. For each strip, loads a kCacheK x kNStrip tile of B into shared
// memory, then each warp processes its assigned rows, using the cache for
// columns in [k_start, k_start+kCacheK) and global __ldg for outliers.
// ===========================================================================
__global__ void locality_tiled_panel_kernel(
    const int* __restrict__   d_row_ptr_r,
    const int* __restrict__   d_col_r,
    const float* __restrict__ d_val_r,
    const float* __restrict__ B,
    float* __restrict__       C_out,
    const int* __restrict__   d_perm_inv,
    const int* __restrict__   d_panel_k_start,
    int M,
    int K,
    int N,
    int num_panels)
{
    const int panel_id = blockIdx.x;
    if (panel_id >= num_panels) return;

    const int panel_start = panel_id * kPanelRows;
    const int panel_end   = min(M, panel_start + kPanelRows);
    const int panel_rows  = panel_end - panel_start;
    if (panel_rows <= 0) return;

    const int k_start = d_panel_k_start[panel_id];
    const int k_end   = min(K, k_start + kCacheK);
    const int cache_cols = k_end - k_start;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    // Shared memory: B_cache[kCacheK][kSmemPad] -- padded column to avoid
    // bank conflicts. 64 * 33 * 4 = 8448 bytes.
    __shared__ float B_cache[kCacheK][kSmemPad];

    // Distribute rows across warps
    const int rows_per_warp = (panel_rows + kNumWarps - 1) / kNumWarps;

    // Process N in strips of kNStrip
    for (int n_base = 0; n_base < N; n_base += kNStrip) {
        const int n_end = min(N, n_base + kNStrip);
        const int n_count = n_end - n_base;

        // Phase 1: Cooperatively load B[k_start..k_end)[n_base..n_end) into
        //          B_cache. All threads participate.
        const int total_loads = cache_cols * n_count;
        for (int i = threadIdx.x; i < total_loads; i += kCTAThreads) {
            const int ki = i / n_count;
            const int ni = i % n_count;
            const int g_col = k_start + ki;
            const int g_n   = n_base + ni;
            B_cache[ki][ni] = __ldg(&B[(i64)g_col * N + g_n]);
        }
        __syncthreads();

        // Phase 2: Each warp processes its assigned rows
        for (int r = 0; r < rows_per_warp; ++r) {
            const int row_r = panel_start + warp_id * rows_per_warp + r;
            if (row_r >= panel_end) break;

            const int start = d_row_ptr_r[row_r];
            const int end   = d_row_ptr_r[row_r + 1];
            const int original_row = d_perm_inv[row_r];

            // Each lane handles one or more N-columns within the strip
            for (int n_off = lane; n_off < n_count; n_off += 32) {
                float acc = 0.f;
                for (int p = start; p < end; ++p) {
                    const int col   = d_col_r[p];
                    const float a_val = d_val_r[p];
                    const int k_local = col - k_start;

                    if (k_local >= 0 && k_local < cache_cols) {
                        // Cache hit: read from shared memory
                        acc += a_val * B_cache[k_local][n_off];
                    } else {
                        // Cache miss: fall back to global memory
                        acc += a_val * __ldg(&B[(i64)col * N + (n_base + n_off)]);
                    }
                }
                C_out[(i64)original_row * N + (n_base + n_off)] = acc;
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// Vectorized kernel variant for N%4==0 with float4 loads/stores
// ===========================================================================
__global__ void locality_tiled_panel_kernel_vec4(
    const int* __restrict__   d_row_ptr_r,
    const int* __restrict__   d_col_r,
    const float* __restrict__ d_val_r,
    const float* __restrict__ B,
    float* __restrict__       C_out,
    const int* __restrict__   d_perm_inv,
    const int* __restrict__   d_panel_k_start,
    int M,
    int K,
    int N,
    int num_panels)
{
    const int panel_id = blockIdx.x;
    if (panel_id >= num_panels) return;

    const int panel_start = panel_id * kPanelRows;
    const int panel_end   = min(M, panel_start + kPanelRows);
    const int panel_rows  = panel_end - panel_start;
    if (panel_rows <= 0) return;

    const int k_start = d_panel_k_start[panel_id];
    const int k_end   = min(K, k_start + kCacheK);
    const int cache_cols = k_end - k_start;

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;

    __shared__ float B_cache[kCacheK][kSmemPad];

    const int rows_per_warp = (panel_rows + kNumWarps - 1) / kNumWarps;
    const int N4 = N / 4;

    // Process N in strips of kNStrip (must be multiple of 4)
    for (int n_base = 0; n_base < N; n_base += kNStrip) {
        const int n_end = min(N, n_base + kNStrip);
        const int n_count = n_end - n_base;

        // Phase 1: Load B tile into shared memory
        const int total_loads = cache_cols * n_count;
        for (int i = threadIdx.x; i < total_loads; i += kCTAThreads) {
            const int ki = i / n_count;
            const int ni = i % n_count;
            const int g_col = k_start + ki;
            const int g_n   = n_base + ni;
            B_cache[ki][ni] = __ldg(&B[(i64)g_col * N + g_n]);
        }
        __syncthreads();

        // Phase 2: Compute with vec4 accumulation where possible
        for (int r = 0; r < rows_per_warp; ++r) {
            const int row_r = panel_start + warp_id * rows_per_warp + r;
            if (row_r >= panel_end) break;

            const int start = d_row_ptr_r[row_r];
            const int end   = d_row_ptr_r[row_r + 1];
            const int original_row = d_perm_inv[row_r];

            const int n_count4 = n_count / 4;
            for (int n4 = lane; n4 < n_count4; n4 += 32) {
                const int n_off = n4 * 4;
                float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);

                for (int p = start; p < end; ++p) {
                    const int col   = d_col_r[p];
                    const float a_val = d_val_r[p];
                    const int k_local = col - k_start;

                    if (k_local >= 0 && k_local < cache_cols) {
                        acc.x += a_val * B_cache[k_local][n_off];
                        acc.y += a_val * B_cache[k_local][n_off + 1];
                        acc.z += a_val * B_cache[k_local][n_off + 2];
                        acc.w += a_val * B_cache[k_local][n_off + 3];
                    } else {
                        const float4* B_row = reinterpret_cast<const float4*>(
                            B + (i64)col * N + n_base);
                        const float4 b4 = B_row[n4];
                        acc.x += a_val * b4.x;
                        acc.y += a_val * b4.y;
                        acc.z += a_val * b4.z;
                        acc.w += a_val * b4.w;
                    }
                }

                float4* C_row = reinterpret_cast<float4*>(
                    C_out + (i64)original_row * N + n_base);
                C_row[n4] = acc;
            }
        }
        __syncthreads();
    }
}

// ===========================================================================
// make_ra_locality_tiled_plan
//
// 1. Compute row ordering info (centroid, span, signature)
// 2. Sort rows by centroid bucket, then span, then within-panel by length
// 3. Evaluate reorder gain; fall back to identity if no improvement
// 4. Build reordered CSR
// 5. Find dominant column window per panel (sliding-window search)
// 6. Upload everything to GPU
// ===========================================================================
void make_ra_locality_tiled_plan(
    RALocalityTiledPlan& plan,
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M,
    int K,
    int N)
{
    plan = RALocalityTiledPlan{};
    plan.M = M;
    plan.K = K;

    if (M <= 0 || K <= 0) return;

    const int total_nnz = h_rowptr[M];
    if (total_nnz <= 0) return;

    // -----------------------------------------------------------------
    // Step 1: Compute row ordering info
    // -----------------------------------------------------------------
    std::vector<RowOrderInfo> info(M);
    for (int row = 0; row < M; ++row) {
        const int start = h_rowptr[row];
        const int end   = h_rowptr[row + 1];
        const int len   = end - start;

        RowOrderInfo ri;
        ri.row     = row;
        ri.len     = len;
        ri.min_col = (len > 0) ? h_col[start]   : 0;
        ri.max_col = (len > 0) ? h_col[end - 1]  : 0;

        double centroid_sum = 0.0;
        for (int p = start; p < end; ++p) {
            centroid_sum += h_col[p];
            const int bucket = std::min(
                kSignatureBuckets - 1,
                (h_col[p] * kSignatureBuckets) / std::max(1, K));
            ri.signature |= (uint64_t{1} << bucket);
        }
        ri.centroid = (len > 0)
            ? static_cast<float>(centroid_sum / static_cast<double>(len))
            : 0.f;
        info[row] = ri;
    }

    // -----------------------------------------------------------------
    // Step 2: Sort rows by centroid bucket -> span -> length desc
    // -----------------------------------------------------------------
    std::vector<int> identity_order(M);
    std::iota(identity_order.begin(), identity_order.end(), 0);

    std::vector<int> order = identity_order;
    const float bucket_scale = static_cast<float>(kCentroidBuckets) /
                               static_cast<float>(std::max(1, K));
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const int bucket_a = static_cast<int>(info[a].centroid * bucket_scale);
        const int bucket_b = static_cast<int>(info[b].centroid * bucket_scale);
        if (bucket_a != bucket_b) return bucket_a < bucket_b;
        const int span_a = (info[a].len > 0)
            ? (info[a].max_col - info[a].min_col + 1) : K;
        const int span_b = (info[b].len > 0)
            ? (info[b].max_col - info[b].min_col + 1) : K;
        if (span_a != span_b) return span_a < span_b;
        return info[a].len > info[b].len;
    });

    // Within each panel, sort by row length descending
    for (int base = 0; base < M; base += kPanelRows) {
        const int end = std::min(M, base + kPanelRows);
        std::stable_sort(order.begin() + base, order.begin() + end,
                         [&](int a, int b) {
            return info[a].len > info[b].len;
        });
    }

    // -----------------------------------------------------------------
    // Step 3: Evaluate reorder gain; fall back to identity if no improvement
    // -----------------------------------------------------------------
    const float identity_compactness =
        compute_panel_compactness(identity_order, info, K);
    const float reordered_compactness =
        compute_panel_compactness(order, info, K);

    if (reordered_compactness <= identity_compactness + 0.005f) {
        order = identity_order;
        plan.reorder_gain = 0.f;
    } else {
        plan.reorder_gain = reordered_compactness - identity_compactness;
    }

    // -----------------------------------------------------------------
    // Step 4: Build row permutation and reordered CSR
    // -----------------------------------------------------------------
    plan.h_row_perm     = new int[M];
    plan.h_row_perm_inv = new int[M];
    for (int reordered_row = 0; reordered_row < M; ++reordered_row) {
        const int original_row = order[reordered_row];
        plan.h_row_perm[reordered_row]     = original_row;
        plan.h_row_perm_inv[original_row]  = reordered_row;
    }

    std::vector<int>   r_rowptr(M + 1, 0);
    std::vector<int>   r_col(total_nnz);
    std::vector<float> r_val(total_nnz);
    int write_ptr = 0;
    for (int reordered_row = 0; reordered_row < M; ++reordered_row) {
        const int original_row = plan.h_row_perm[reordered_row];
        // Collect entries and sort by column (usually already sorted in CSR,
        // but defensive re-sort ensures correctness after reordering)
        std::vector<std::pair<int, float>> entries;
        entries.reserve(h_rowptr[original_row + 1] - h_rowptr[original_row]);
        for (int p = h_rowptr[original_row];
             p < h_rowptr[original_row + 1]; ++p) {
            entries.push_back({h_col[p], h_val[p]});
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) {
            return a.first < b.first;
        });
        for (const auto& entry : entries) {
            r_col[write_ptr] = entry.first;
            r_val[write_ptr] = entry.second;
            ++write_ptr;
        }
        r_rowptr[reordered_row + 1] = write_ptr;
    }

    // -----------------------------------------------------------------
    // Step 5: Find dominant column window per panel
    // -----------------------------------------------------------------
    plan.num_panels = (M + kPanelRows - 1) / kPanelRows;
    std::vector<int> panel_k_start(plan.num_panels);

    int64_t total_hits = 0;
    int64_t total_nnz_panels = 0;
    for (int panel = 0; panel < plan.num_panels; ++panel) {
        const int pstart = panel * kPanelRows;
        const int pend   = std::min(M, pstart + kPanelRows);

        panel_k_start[panel] = find_best_cache_window(
            h_rowptr, h_col, order, pstart, pend, K);

        // Count cache hits for diagnostics
        const int ks = panel_k_start[panel];
        const int ke = std::min(K, ks + kCacheK);
        for (int idx = pstart; idx < pend; ++idx) {
            const int orig = order[idx];
            for (int p = h_rowptr[orig]; p < h_rowptr[orig + 1]; ++p) {
                ++total_nnz_panels;
                if (h_col[p] >= ks && h_col[p] < ke) {
                    ++total_hits;
                }
            }
        }
    }
    plan.avg_cache_hit_rate = (total_nnz_panels > 0)
        ? static_cast<float>(static_cast<double>(total_hits) /
                             static_cast<double>(total_nnz_panels))
        : 0.f;
    plan.panel_rows = kPanelRows;
    plan.cache_k    = kCacheK;

    // -----------------------------------------------------------------
    // Step 6: Upload to GPU
    // -----------------------------------------------------------------
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_row_ptr_r,
                               (M + 1) * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_col_r,
                               total_nnz * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_val_r,
                               total_nnz * sizeof(float)));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_row_ptr_r, r_rowptr.data(),
                               (M + 1) * sizeof(int),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_col_r, r_col.data(),
                               total_nnz * sizeof(int),
                               cudaMemcpyHostToDevice));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_val_r, r_val.data(),
                               total_nnz * sizeof(float),
                               cudaMemcpyHostToDevice));

    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_perm_inv, M * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_perm_inv, plan.h_row_perm,
                               M * sizeof(int),
                               cudaMemcpyHostToDevice));

    plan.d_panel_k_start = lt_upload(panel_k_start);

    plan.plan_bytes =
        (size_t)(M + 1) * sizeof(int) +            // d_row_ptr_r
        (size_t)total_nnz * sizeof(int) +           // d_col_r
        (size_t)total_nnz * sizeof(float) +         // d_val_r
        (size_t)M * sizeof(int) +                   // d_perm_inv
        (size_t)plan.num_panels * sizeof(int);       // d_panel_k_start

    plan.active = true;
}

// ===========================================================================
// run_ra_locality_tiled_plan
//
// 1. Zero-initialize output C
// 2. Launch one CTA per panel; select vec4 kernel when alignment permits
// ===========================================================================
void run_ra_locality_tiled_plan(
    const RALocalityTiledPlan& plan,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream)
{
    if (!plan.active || plan.M <= 0 || N <= 0) return;

    // Zero C -- each (row, n) pair is written exactly once across all
    // n_base passes, so we need C initialized to zero for the direct writes.
    CUDA_CHECK_NEXT(cudaMemsetAsync(
        d_C, 0, (size_t)plan.M * N * sizeof(float), stream));

    const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
    const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
    const bool use_vec4  = (N % 4 == 0) && (N >= 4) && b_aligned && c_aligned;

    // Shared memory: B_cache[kCacheK][kSmemPad] = 64 * 33 * 4 = 8448 bytes
    // This fits comfortably within the default 48 KB shared memory limit.

    if (use_vec4) {
        locality_tiled_panel_kernel_vec4<<<plan.num_panels, kCTAThreads,
                                          0, stream>>>(
            plan.d_row_ptr_r,
            plan.d_col_r,
            plan.d_val_r,
            d_B,
            d_C,
            plan.d_perm_inv,
            plan.d_panel_k_start,
            plan.M,
            plan.K,
            N,
            plan.num_panels);
    } else {
        locality_tiled_panel_kernel<<<plan.num_panels, kCTAThreads,
                                     0, stream>>>(
            plan.d_row_ptr_r,
            plan.d_col_r,
            plan.d_val_r,
            d_B,
            d_C,
            plan.d_perm_inv,
            plan.d_panel_k_start,
            plan.M,
            plan.K,
            N,
            plan.num_panels);
    }
    CUDA_CHECK_KERNEL();
}

// ===========================================================================
// free_ra_locality_tiled_plan
//
// Free all device and host allocations, zero all counts.
// ===========================================================================
void free_ra_locality_tiled_plan(RALocalityTiledPlan& plan)
{
    delete[] plan.h_row_perm;
    delete[] plan.h_row_perm_inv;
    plan.h_row_perm     = nullptr;
    plan.h_row_perm_inv = nullptr;

    if (plan.d_perm_inv)       { cudaFree(plan.d_perm_inv);       plan.d_perm_inv       = nullptr; }
    if (plan.d_row_ptr_r)      { cudaFree(plan.d_row_ptr_r);      plan.d_row_ptr_r      = nullptr; }
    if (plan.d_col_r)          { cudaFree(plan.d_col_r);           plan.d_col_r          = nullptr; }
    if (plan.d_val_r)          { cudaFree(plan.d_val_r);           plan.d_val_r          = nullptr; }
    if (plan.d_panel_k_start)  { cudaFree(plan.d_panel_k_start);  plan.d_panel_k_start  = nullptr; }

    plan.active     = false;
    plan.num_panels = 0;
    plan.M          = 0;
    plan.K          = 0;
    plan.plan_bytes = 0;
    plan.avg_cache_hit_rate = 0.f;
    plan.reorder_gain       = 0.f;
}
