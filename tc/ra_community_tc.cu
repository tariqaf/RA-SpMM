// ============================================================================
// ra_community_tc.cu - R5: COMMUNITY_TC SpMM for sparse modular community regime
//
// Community-aware row permutation + TC execution for matrices with:
//   - mixedness <= 0.02, local_row_similarity >= 0.60
//
// Key idea: cluster rows by community (rows that share column neighbors via
// Jaccard signature similarity), then pack intra-community tiles for WMMA.
// Inter-community spillover handled by FP32 CSR fallback.
//
// Reuses the proven TC_DIRECT kernel design (single-pass 16-warp TC kernel)
// with community-aware reordering instead of centroid-based reordering.
//
// Target: sm_70+ (Volta WMMA), optimised for sm_86 (RTX 3090).
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <numeric>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

// ---------------------------------------------------------------------------
// Compile-time constants
// ---------------------------------------------------------------------------
constexpr int kGroupRows          = 16;
constexpr int kSignatureBuckets   = 64;
constexpr int kTileElems          = 16 * 16;   // 256 halfs per packed tile
constexpr int kMaxWarpsPerCta     = 16;

// Community detection (column-stream 1-pass label propagation)
//   kLpMaxRowDegreeForVote: rows with degree above this skip the plurality
//     vote and anchor a fresh label (very dense rows otherwise dominate the
//     vote and over-merge unrelated communities).
//   kLpRefinementSweeps: number of label-propagation sweeps (1 is the
//     "1-pass label propagation" variant; raising it tightens cohesion).
constexpr int kLpMaxRowDegreeForVote = 4096;
constexpr int kLpRefinementSweeps    = 2;

// Relaxed FP32 thresholds (same as TC_DIRECT) — TC kernels are correct
constexpr int   kFp32GroupMaxRowNnzThreshold  = 256;
constexpr int   kFp32GroupTotalNnzThreshold   = 2048;
constexpr float kFp32GroupAvgRowNnzThreshold  = 128.f;

// Tile density gate
constexpr float kMinTcGroupTileDensity        = 0.08f;

// ---------------------------------------------------------------------------
// Helper: upload host vector to device
// ---------------------------------------------------------------------------
template <typename T>
T* upload_community(const std::vector<T>& values) {
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

// ---------------------------------------------------------------------------
// Row ordering metadata
// ---------------------------------------------------------------------------
struct RowOrderInfo {
    int row       = 0;
    int len       = 0;
    int min_col   = 0;
    int max_col   = 0;
    float centroid = 0.f;
    uint64_t signature = 0;
};

inline int popcount64(uint64_t x) {
#if defined(__GNUG__)
    return __builtin_popcountll(x);
#else
    int count = 0;
    while (x) { x &= (x - 1); ++count; }
    return count;
#endif
}

inline float jaccard_u64(uint64_t a, uint64_t b) {
    const int uni = popcount64(a | b);
    if (uni == 0) return 0.f;
    return static_cast<float>(popcount64(a & b)) / static_cast<float>(uni);
}

__device__ __forceinline__ float load_readonly_f32(const float* __restrict__ ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
    return __ldg(ptr);
#else
    return *ptr;
#endif
}

inline uint16_t float_to_half_bits(float value) {
    const half h = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

// =========================================================================
// COMMUNITY TC kernel -- identical to TC_DIRECT kernel
// One CTA per group, up to 16 warps, single-pass tiles
// =========================================================================
//
// Shared memory layout (dynamic):
//   half  A_smem[256]                -- one packed 16x16 tile, shared by all warps
//   half  B_smem[num_warps][256]     -- per-warp B tile (float32->half convert)
//   float C_tmp[num_warps][256]      -- per-warp store buffer for c_frag scatter
//
// Total smem = 256*2 + num_warps*256*2 + num_warps*256*4 bytes
//            = 512 + num_warps * 1536 bytes
// At 16 warps: 512 + 24576 = 25088 bytes (~24.5 KB) -- well within 48 KB
// =========================================================================

__global__ void ra_community_tc_kernel(
    const int* __restrict__      d_group_offsets,
    const int* __restrict__      d_group_use_fp32,
    const int* __restrict__      d_group_tile_offsets,
    const int* __restrict__      d_group_tile_k_ids,
    const uint16_t* __restrict__ d_group_tile_vals,
    const float* __restrict__    B,
    const int* __restrict__      reordered_to_original,
    float* __restrict__          C_out,
    int M, int K, int N,
    int num_groups)
{
    const int group_id = blockIdx.x;
    if (group_id >= num_groups) return;
    if (d_group_use_fp32[group_id] != 0) return;

    const int group_row_start = d_group_offsets[group_id];
    const int group_row_end   = d_group_offsets[group_id + 1];
    const int local_rows      = group_row_end - group_row_start;

    const int warp_id   = threadIdx.x / 32;
    const int lane      = threadIdx.x % 32;
    const int num_warps = blockDim.x / 32;
    const int num_strips = (N + 15) / 16;

    const int tile_begin = d_group_tile_offsets[group_id];
    const int tile_end   = d_group_tile_offsets[group_id + 1];
    if (tile_begin >= tile_end) return;

    const half* tile_vals_half =
        reinterpret_cast<const half*>(d_group_tile_vals);

    // Dynamic shared memory regions
    extern __shared__ char smem_raw[];
    half*  A_smem      = reinterpret_cast<half*>(smem_raw);
    half*  B_smem_base = A_smem + kTileElems;
    float* C_tmp_base  = reinterpret_cast<float*>(B_smem_base + num_warps * kTileElems);

    // Process N-strips in batches of num_warps
    for (int strip_base = 0; strip_base < num_strips; strip_base += num_warps) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
        using namespace nvcuda;

        const int strip = strip_base + warp_id;

        // Initialize accumulator fragment (per-warp, persists across all tiles)
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
        if (strip < num_strips) {
            wmma::fill_fragment(c_frag, 0.0f);
        }

        // Single pass over ALL tiles
        for (int tile_idx = tile_begin; tile_idx < tile_end; ++tile_idx) {
            const int kb      = d_group_tile_k_ids[tile_idx];
            const int k_start = kb * 16;

            // ALL threads cooperatively load A tile into shared memory
            const half* src = tile_vals_half + static_cast<i64>(tile_idx) * kTileElems;
            for (int i = threadIdx.x; i < kTileElems; i += blockDim.x) {
                A_smem[i] = src[i];
            }
            __syncthreads();

            if (strip < num_strips) {
                const int n_start = strip * 16;
                half* B_smem = B_smem_base + warp_id * kTileElems;

                // Load B tile: convert float32 -> half into per-warp smem
                for (int i = lane; i < kTileElems; i += 32) {
                    const int lc = i / 16;   // row within 16x16 tile (K dim)
                    const int ln = i % 16;   // col within 16x16 tile (N dim)
                    const int gc = k_start + lc;
                    const int gn = n_start + ln;
                    const float val = (gc < K && gn < N)
                        ? __ldg(&B[static_cast<i64>(gc) * N + gn])
                        : 0.f;
                    B_smem[i] = __float2half(val);
                }
                __syncwarp();

                // WMMA: A(16x16) x B(16x16) -> accumulate in c_frag
                wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(a_frag, A_smem, 16);
                wmma::load_matrix_sync(b_frag, B_smem, 16);
                wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            }
            __syncthreads();
        }

        // Store accumulated c_frag -> per-warp C_tmp, then scatter to C_out
        if (strip < num_strips) {
            const int n_start = strip * 16;
            float* C_tmp = C_tmp_base + warp_id * kTileElems;

            wmma::store_matrix_sync(C_tmp, c_frag, 16, wmma::mem_row_major);
            __syncwarp();

            // Scatter to global C_out using reordered->original mapping
            for (int i = lane; i < kTileElems; i += 32) {
                const int lr = i / 16;  // local row in group [0..15]
                const int ln = i % 16;  // local N column [0..15]
                const int n  = n_start + ln;
                if (lr < local_rows && n < N) {
                    const int reordered_row = group_row_start + lr;
                    const int original_row  = reordered_to_original[reordered_row];
                    if (original_row < M) {
                        C_out[static_cast<i64>(original_row) * N + n] = C_tmp[i];
                    }
                }
            }
        }
        __syncthreads();
#endif  // __CUDA_ARCH__ >= 700
    }
}

// =========================================================================
// FP32 fallback kernels (identical to TC_DIRECT)
// =========================================================================

__global__ void ra_community_tc_fp32_kernel(
    const int* __restrict__   d_fp32_rows,
    const int* __restrict__   d_row_ptr_r,
    const int* __restrict__   d_col_r,
    const float* __restrict__ d_val_r,
    const float* __restrict__ B,
    const int* __restrict__   reordered_to_original,
    float* __restrict__       C_out,
    int M, int N,
    int num_fp32_rows)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    if (warp_id >= num_fp32_rows) return;

    const int reordered_row = d_fp32_rows[warp_id];
    if (reordered_row >= M) return;
    const int original_row = reordered_to_original[reordered_row];

    const int start = d_row_ptr_r[reordered_row];
    const int end   = d_row_ptr_r[reordered_row + 1];

    for (int n = lane; n < N; n += 32) {
        float acc = 0.f;
        for (int p = start; p < end; ++p) {
            acc += d_val_r[p] * __ldg(&B[static_cast<i64>(d_col_r[p]) * N + n]);
        }
        C_out[static_cast<i64>(original_row) * N + n] = acc;
    }
}

__global__ void ra_community_tc_fp32_kernel_vec4(
    const int* __restrict__   d_fp32_rows,
    const int* __restrict__   d_row_ptr_r,
    const int* __restrict__   d_col_r,
    const float* __restrict__ d_val_r,
    const float* __restrict__ B,
    const int* __restrict__   reordered_to_original,
    float* __restrict__       C_out,
    int M, int N,
    int num_fp32_rows)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane    = threadIdx.x % 32;
    if (warp_id >= num_fp32_rows) return;

    const int reordered_row = d_fp32_rows[warp_id];
    if (reordered_row >= M) return;
    const int original_row = reordered_to_original[reordered_row];

    const int start = d_row_ptr_r[reordered_row];
    const int end   = d_row_ptr_r[reordered_row + 1];
    const int N4    = N / 4;

    for (int n4 = lane; n4 < N4; n4 += 32) {
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
        for (int p = start; p < end; ++p) {
            const int col      = d_col_r[p];
            const float a_val  = d_val_r[p];
            const float4* B_ptr =
                reinterpret_cast<const float4*>(B + static_cast<i64>(col) * N);
            const float4 b4 = B_ptr[n4];
            acc.x += a_val * b4.x;
            acc.y += a_val * b4.y;
            acc.z += a_val * b4.z;
            acc.w += a_val * b4.w;
        }
        float4* C_ptr =
            reinterpret_cast<float4*>(C_out + static_cast<i64>(original_row) * N);
        C_ptr[n4] = acc;
    }
}

}  // namespace

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

    if (M <= 0 || K <= 0 || N < 16) {
        return;
    }
    const int total_nnz = h_rowptr[M];
    if (total_nnz <= 0) {
        return;
    }

    // No force_all_fp32 guards needed — TC kernels are correct.
    const bool force_all_fp32 = false;

    // -----------------------------------------------------------------
    // Step 1: Build row metadata (centroid, signature, span)
    // -----------------------------------------------------------------
    std::vector<RowOrderInfo> info(M);
    for (int row = 0; row < M; ++row) {
        const int start = h_rowptr[row];
        const int end   = h_rowptr[row + 1];
        const int len   = end - start;

        RowOrderInfo& ri = info[row];
        ri.row     = row;
        ri.len     = len;
        ri.min_col = (len > 0) ? h_col[start]   : 0;
        ri.max_col = (len > 0) ? h_col[end - 1]  : 0;

        double centroid_sum = 0.0;
        for (int p = start; p < end; ++p) {
            centroid_sum += h_col[p];
            const int bucket = std::min(kSignatureBuckets - 1,
                                        (h_col[p] * kSignatureBuckets) / std::max(1, K));
            ri.signature |= (uint64_t{1} << bucket);
        }
        ri.centroid = (len > 0)
            ? static_cast<float>(centroid_sum / static_cast<double>(len))
            : 0.f;
    }

    // -----------------------------------------------------------------
    // Step 2: Community detection via column-stream 1-pass label propagation
    //
    // The prior Jaccard scheme only compared each row to the previous 64
    // rows in file order, so it silently degraded to "every row in its own
    // community" whenever communities were not already clustered in file
    // order (a common case after CSR ingest). Result: COMMUNITY_TC under-
    // performed TC_DIRECT on the Community regime.
    //
    // The new algorithm exploits the row<->column bipartite graph directly:
    //   1. Build the CSC (column -> rows that touch it) once: O(nnz).
    //   2. Process rows in degree-descending order so dense rows anchor
    //      labels first.
    //   3. For each row, gather labels currently held by its columns,
    //      take the plurality (ties broken by smallest label id), and
    //      assign that as the row's community. Then write the chosen
    //      label back to col_label[c] for every column c the row touches
    //      so subsequent rows that share columns inherit it.
    //   4. Re-run step 3 once more (kLpRefinementSweeps) to tighten cohesion.
    //
    // Total work: O(nnz * kLpRefinementSweeps), independent of file order.
    // -----------------------------------------------------------------

    // Step 2a: build CSC index (column -> rows)
    std::vector<int> csc_col_ptr(K + 1, 0);
    for (int row = 0; row < M; ++row) {
        for (int p = h_rowptr[row]; p < h_rowptr[row + 1]; ++p) {
            ++csc_col_ptr[h_col[p] + 1];
        }
    }
    for (int c = 0; c < K; ++c) {
        csc_col_ptr[c + 1] += csc_col_ptr[c];
    }
    std::vector<int> csc_row_ind(static_cast<size_t>(total_nnz));
    {
        std::vector<int> cursor(csc_col_ptr.begin(), csc_col_ptr.end());
        for (int row = 0; row < M; ++row) {
            for (int p = h_rowptr[row]; p < h_rowptr[row + 1]; ++p) {
                csc_row_ind[cursor[h_col[p]]++] = row;
            }
        }
    }

    // Step 2b: initialize column labels with the first row that touches each
    // column, giving every column a deterministic starting label.
    std::vector<int> col_label(K, -1);
    for (int c = 0; c < K; ++c) {
        const int beg = csc_col_ptr[c];
        const int end = csc_col_ptr[c + 1];
        if (beg < end) col_label[c] = csc_row_ind[beg];
    }

    // Step 2c: row processing order — degree descending, with empty rows
    // pushed to the back. Stable_sort keeps original-row order on ties so
    // identical-degree rows fall back to file order (mild locality bonus).
    std::vector<int> row_order_lp(M);
    std::iota(row_order_lp.begin(), row_order_lp.end(), 0);
    std::stable_sort(row_order_lp.begin(), row_order_lp.end(),
                     [&](int a, int b) {
                         if (info[a].len != info[b].len)
                             return info[a].len > info[b].len;
                         return a < b;
                     });

    std::vector<int> community_id(M);
    std::iota(community_id.begin(), community_id.end(), 0);

    // Step 2d: kLpRefinementSweeps passes of plurality-vote label propagation
    std::unordered_map<int, int> label_counts;
    label_counts.reserve(64);
    for (int sweep = 0; sweep < kLpRefinementSweeps; ++sweep) {
        for (int r : row_order_lp) {
            const int start = h_rowptr[r];
            const int end   = h_rowptr[r + 1];
            const int len   = end - start;
            if (len == 0) continue;

            // Very wide rows would dominate the vote; let them anchor
            // their own label instead so they don't merge unrelated
            // communities together.
            if (len >= kLpMaxRowDegreeForVote) {
                community_id[r] = r;
                for (int p = start; p < end; ++p) {
                    col_label[h_col[p]] = r;
                }
                continue;
            }

            label_counts.clear();
            for (int p = start; p < end; ++p) {
                const int lbl = col_label[h_col[p]];
                if (lbl >= 0) ++label_counts[lbl];
            }

            int best_label = community_id[r];
            int best_count = -1;
            for (const auto& kv : label_counts) {
                if (kv.second > best_count ||
                    (kv.second == best_count && kv.first < best_label)) {
                    best_count = kv.second;
                    best_label = kv.first;
                }
            }
            community_id[r] = best_label;
            for (int p = start; p < end; ++p) {
                col_label[h_col[p]] = best_label;
            }
        }
    }

    // Renumber communities to compact IDs [0..num_communities)
    std::map<int, int> community_remap;
    int next_community = 0;
    for (int row = 0; row < M; ++row) {
        const int cid = community_id[row];
        if (community_remap.find(cid) == community_remap.end()) {
            community_remap[cid] = next_community++;
        }
        community_id[row] = community_remap[cid];
    }
    plan.num_communities = next_community;

    // Build row ordering: sort by community, then centroid within community
    std::vector<int> order(M);
    std::iota(order.begin(), order.end(), 0);

    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (community_id[a] != community_id[b])
            return community_id[a] < community_id[b];
        return info[a].centroid < info[b].centroid;
    });

    // Within each 16-row group, sort by nnz descending (same as TC_DIRECT)
    for (int base = 0; base < M; base += kGroupRows) {
        const int end = std::min(M, base + kGroupRows);
        std::stable_sort(order.begin() + base, order.begin() + end,
                         [&](int a, int b) { return info[a].len > info[b].len; });
    }

    // Compute intra-community nnz fraction for diagnostics
    {
        // Build a set of column indices per community for quick lookup
        // would be expensive, so approximate: count nnz in reordered groups
        // that are single-community vs mixed
        int64_t intra_nnz = 0;
        for (int base = 0; base < M; base += kGroupRows) {
            const int end = std::min(M, base + kGroupRows);
            // Check if entire group is same community
            const int first_comm = community_id[order[base]];
            bool single_community = true;
            int64_t group_nnz = 0;
            for (int idx = base; idx < end; ++idx) {
                group_nnz += info[order[idx]].len;
                if (community_id[order[idx]] != first_comm) {
                    single_community = false;
                }
            }
            if (single_community) {
                intra_nnz += group_nnz;
            }
        }
        plan.intra_community_nnz_fraction = (total_nnz > 0)
            ? static_cast<float>(static_cast<double>(intra_nnz) /
                                 static_cast<double>(total_nnz))
            : 0.f;
    }

    // Build permutation arrays
    plan.h_row_perm     = new int[M];
    plan.h_row_perm_inv = new int[M];
    for (int reordered_row = 0; reordered_row < M; ++reordered_row) {
        const int original_row = order[reordered_row];
        plan.h_row_perm[reordered_row]     = original_row;
        plan.h_row_perm_inv[original_row]  = reordered_row;
    }

    // -----------------------------------------------------------------
    // Step 3: Group construction, FP32 gating, tile packing
    //         (same logic as TC_DIRECT)
    // -----------------------------------------------------------------
    std::vector<int> group_offsets;
    group_offsets.reserve((M + kGroupRows - 1) / kGroupRows + 1);
    group_offsets.push_back(0);

    std::vector<int> group_use_fp32;
    group_use_fp32.reserve((M + kGroupRows - 1) / kGroupRows);

    std::vector<int> group_tile_offsets;
    group_tile_offsets.reserve((M + kGroupRows - 1) / kGroupRows + 1);
    group_tile_offsets.push_back(0);

    std::vector<int>      group_tile_k_ids;
    std::vector<uint16_t> group_tile_vals;
    std::vector<int>      fp32_rows;

    double tc_tile_density_sum = 0.0;
    int tc_groups    = 0;
    int fp32_groups  = 0;

    for (int base = 0; base < M; base += kGroupRows) {
        const int end = std::min(M, base + kGroupRows);
        group_offsets.push_back(end);

        int64_t group_nnz = 0;
        int group_max_row_nnz = 0;

        for (int idx = base; idx < end; ++idx) {
            const RowOrderInfo& ri = info[order[idx]];
            group_nnz += ri.len;
            group_max_row_nnz = std::max(group_max_row_nnz, ri.len);
        }
        const float avg_row_nnz =
            static_cast<float>(group_nnz) / static_cast<float>(std::max(1, end - base));

        // FP32 gating: conservative thresholds + force_all_fp32 guard
        bool use_fp32_group = force_all_fp32 ||
            (group_max_row_nnz >= kFp32GroupMaxRowNnzThreshold) ||
            (group_nnz >= kFp32GroupTotalNnzThreshold) ||
            (avg_row_nnz >= kFp32GroupAvgRowNnzThreshold);

        if (!use_fp32_group) {
            // Pack nonzero entries into 16x16 tiles keyed by k-block
            std::map<int, std::array<float, kTileElems>> tile_map;
            for (int idx = base; idx < end; ++idx) {
                const int original_row = order[idx];
                const int local_row    = idx - base;
                for (int p = h_rowptr[original_row]; p < h_rowptr[original_row + 1]; ++p) {
                    const int col = h_col[p];
                    const int kb  = col / 16;
                    auto it = tile_map.find(kb);
                    if (it == tile_map.end()) {
                        it = tile_map.emplace(kb, std::array<float, kTileElems>{}).first;
                    }
                    it->second[local_row * 16 + (col % 16)] = h_val[p];
                }
            }

            if (!tile_map.empty()) {
                const float tile_density = static_cast<float>(group_nnz) /
                    static_cast<float>(tile_map.size() * kTileElems);

                if (tile_density < kMinTcGroupTileDensity) {
                    use_fp32_group = true;
                } else {
                    tc_tile_density_sum += tile_density;
                    ++tc_groups;

                    for (const auto& entry : tile_map) {
                        group_tile_k_ids.push_back(entry.first);
                        for (float value : entry.second) {
                            group_tile_vals.push_back(float_to_half_bits(value));
                        }
                    }
                }
            } else {
                use_fp32_group = true;
            }
        }

        group_use_fp32.push_back(use_fp32_group ? 1 : 0);
        fp32_groups += use_fp32_group ? 1 : 0;
        if (use_fp32_group) {
            for (int reordered_row = base; reordered_row < end; ++reordered_row) {
                fp32_rows.push_back(reordered_row);
            }
        }
        group_tile_offsets.push_back(static_cast<int>(group_tile_k_ids.size()));
    }

    // Fill plan diagnostics
    plan.num_groups      = static_cast<int>(group_offsets.size()) - 1;
    plan.num_fp32_rows   = static_cast<int>(fp32_rows.size());
    plan.num_tc_tiles    = static_cast<int>(group_tile_k_ids.size());
    plan.avg_tc_tile_density = (tc_groups > 0)
        ? static_cast<float>(tc_tile_density_sum / static_cast<double>(tc_groups))
        : 0.f;

    // -----------------------------------------------------------------
    // Step 4: Build reordered CSR (for FP32 fallback rows)
    // -----------------------------------------------------------------
    std::vector<int>   r_rowptr(M + 1, 0);
    std::vector<int>   r_col(total_nnz);
    std::vector<float> r_val(total_nnz);
    int write_ptr = 0;

    for (int reordered_row = 0; reordered_row < M; ++reordered_row) {
        const int original_row = plan.h_row_perm[reordered_row];
        std::vector<std::pair<int, float>> entries;
        entries.reserve(h_rowptr[original_row + 1] - h_rowptr[original_row]);
        for (int p = h_rowptr[original_row]; p < h_rowptr[original_row + 1]; ++p) {
            entries.push_back({h_col[p], h_val[p]});
        }
        std::sort(entries.begin(), entries.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& entry : entries) {
            r_col[write_ptr] = entry.first;
            r_val[write_ptr] = entry.second;
            ++write_ptr;
        }
        r_rowptr[reordered_row + 1] = write_ptr;
    }

    // -----------------------------------------------------------------
    // Step 5: Upload everything to GPU
    // -----------------------------------------------------------------
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_row_ptr_r, (M + 1) * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_col_r,     total_nnz * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_val_r,     total_nnz * sizeof(float)));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_row_ptr_r, r_rowptr.data(),
                               (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_col_r, r_col.data(),
                               total_nnz * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_val_r, r_val.data(),
                               total_nnz * sizeof(float), cudaMemcpyHostToDevice));

    // d_perm_inv stores reordered_row -> original_row (for scatter in kernel)
    CUDA_CHECK_NEXT(cudaMalloc(&plan.d_perm_inv, M * sizeof(int)));
    CUDA_CHECK_NEXT(cudaMemcpy(plan.d_perm_inv, plan.h_row_perm,
                               M * sizeof(int), cudaMemcpyHostToDevice));

    plan.d_group_offsets      = upload_community(group_offsets);
    plan.d_group_use_fp32     = upload_community(group_use_fp32);
    plan.d_fp32_rows          = upload_community(fp32_rows);
    plan.d_group_tile_offsets = upload_community(group_tile_offsets);
    plan.d_group_tile_k_ids   = upload_community(group_tile_k_ids);
    plan.d_group_tile_vals    = upload_community(group_tile_vals);

    // Upload community offsets for diagnostics
    // Build community boundary offsets in reordered space
    std::vector<int> comm_offsets;
    comm_offsets.push_back(0);
    if (M > 0) {
        int current_comm = community_id[order[0]];
        for (int i = 1; i < M; ++i) {
            if (community_id[order[i]] != current_comm) {
                comm_offsets.push_back(i);
                current_comm = community_id[order[i]];
            }
        }
        comm_offsets.push_back(M);
    }
    plan.d_comm_offsets = upload_community(comm_offsets);

    // -----------------------------------------------------------------
    // Step 6: Compute plan_bytes and mark active
    // -----------------------------------------------------------------
    plan.plan_bytes =
        static_cast<size_t>(M + 1)                    * sizeof(int)      +  // d_row_ptr_r
        static_cast<size_t>(total_nnz)                 * sizeof(int)      +  // d_col_r
        static_cast<size_t>(total_nnz)                 * sizeof(float)    +  // d_val_r
        static_cast<size_t>(M)                         * sizeof(int)      +  // d_perm_inv
        static_cast<size_t>(group_offsets.size())       * sizeof(int)      +  // d_group_offsets
        static_cast<size_t>(group_use_fp32.size())     * sizeof(int)      +  // d_group_use_fp32
        static_cast<size_t>(fp32_rows.size())          * sizeof(int)      +  // d_fp32_rows
        static_cast<size_t>(group_tile_offsets.size())  * sizeof(int)      +  // d_group_tile_offsets
        static_cast<size_t>(group_tile_k_ids.size())   * sizeof(int)      +  // d_group_tile_k_ids
        static_cast<size_t>(group_tile_vals.size())    * sizeof(uint16_t) +  // d_group_tile_vals
        static_cast<size_t>(comm_offsets.size())        * sizeof(int);        // d_comm_offsets

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
    if (!plan.active || plan.M <= 0 || N <= 0) {
        return;
    }

    // ---- TC kernel ----
    const int warps_per_cta = std::max(1, std::min(kMaxWarpsPerCta, (N + 15) / 16));
    const int tc_threads    = warps_per_cta * 32;

    // Dynamic shared memory:
    //   A_smem:  kTileElems * sizeof(half)
    //   B_smem:  warps_per_cta * kTileElems * sizeof(half)
    //   C_tmp:   warps_per_cta * kTileElems * sizeof(float)
    const int smem_bytes =
        kTileElems * static_cast<int>(sizeof(half)) +
        warps_per_cta * kTileElems * static_cast<int>(sizeof(half)) +
        warps_per_cta * kTileElems * static_cast<int>(sizeof(float));

    ra_community_tc_kernel<<<plan.num_groups, tc_threads, smem_bytes, stream>>>(
        plan.d_group_offsets,
        plan.d_group_use_fp32,
        plan.d_group_tile_offsets,
        plan.d_group_tile_k_ids,
        plan.d_group_tile_vals,
        d_B,
        plan.d_perm_inv,
        d_C,
        plan.M,
        plan.K,
        N,
        plan.num_groups);
    CUDA_CHECK_KERNEL();

    // ---- FP32 fallback kernel ----
    if (plan.num_fp32_rows > 0) {
        const int fp32_threads = 4 * 32;   // 4 warps per block
        const bool b_aligned = (reinterpret_cast<std::uintptr_t>(d_B) % 16u) == 0u;
        const bool c_aligned = (reinterpret_cast<std::uintptr_t>(d_C) % 16u) == 0u;
        const bool use_vec4  = (N % 4 == 0) && b_aligned && c_aligned;
        const int fp32_blocks =
            (plan.num_fp32_rows + (fp32_threads / 32) - 1) / (fp32_threads / 32);

        if (use_vec4) {
            ra_community_tc_fp32_kernel_vec4<<<fp32_blocks, fp32_threads, 0, stream>>>(
                plan.d_fp32_rows,
                plan.d_row_ptr_r,
                plan.d_col_r,
                plan.d_val_r,
                d_B,
                plan.d_perm_inv,
                d_C,
                plan.M,
                N,
                plan.num_fp32_rows);
        } else {
            ra_community_tc_fp32_kernel<<<fp32_blocks, fp32_threads, 0, stream>>>(
                plan.d_fp32_rows,
                plan.d_row_ptr_r,
                plan.d_col_r,
                plan.d_val_r,
                d_B,
                plan.d_perm_inv,
                d_C,
                plan.M,
                N,
                plan.num_fp32_rows);
        }
        CUDA_CHECK_KERNEL();
    }
}

// ============================================================================
// free_ra_community_tc_plan
// ============================================================================

void free_ra_community_tc_plan(RACommunityTCPlan& plan)
{
    delete[] plan.h_row_perm;
    delete[] plan.h_row_perm_inv;
    plan.h_row_perm     = nullptr;
    plan.h_row_perm_inv = nullptr;

    auto safe_free = [](auto*& ptr) {
        if (ptr) { cudaFree(ptr); ptr = nullptr; }
    };

    safe_free(plan.d_perm_inv);
    safe_free(plan.d_row_ptr_r);
    safe_free(plan.d_col_r);
    safe_free(plan.d_val_r);
    safe_free(plan.d_comm_offsets);
    safe_free(plan.d_group_offsets);
    safe_free(plan.d_group_use_fp32);
    safe_free(plan.d_group_tile_offsets);
    safe_free(plan.d_group_tile_k_ids);
    safe_free(plan.d_group_tile_vals);
    safe_free(plan.d_fp32_rows);

    plan.active     = false;
    plan.plan_bytes = 0;
    plan.num_communities = 0;
    plan.num_groups      = 0;
    plan.num_fp32_rows   = 0;
    plan.num_tc_tiles    = 0;
    plan.intra_community_nnz_fraction = 0.f;
    plan.avg_tc_tile_density          = 0.f;
}
