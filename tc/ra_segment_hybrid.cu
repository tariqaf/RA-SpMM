// ============================================================================
// ra_segment_hybrid.cu - R7: ME-BCRS tile SpMM with balance-split windows
//
// Same engine as TC_DIRECT (FlashSparse-style 8x1 vectors, swapped-operand
// mma.m16n8k8, FP16 inputs / FP32 accumulate) with FlashSparse's "_balance"
// load balancing: natural-order 8-row windows whose mma-block list exceeds
// kSegBlocks are split into equal segments processed by independent thread
// blocks and merged with atomicAdd, so hub windows on skewed graphs no
// longer serialize a single warp. Sole segments (window fits in one) store
// directly; only split windows' rows are pre-zeroed.
// ============================================================================
#include "../ra_common.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace {

constexpr int kWindow      = 8;
constexpr int kVecPerBlock = 8;
constexpr int kSegBlocks   = 64;   // max mma blocks (512 vectors) per segment
constexpr unsigned kSoleSegFlag = 0x80000000u;

inline uint16_t float_to_half_bits(float value) {
    const half h = __float2half_rn(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

template <typename T>
T* upload_vec_sh(const std::vector<T>& values) {
    if (values.empty()) return nullptr;
    T* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, values.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, values.data(),
                               values.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
    return d_ptr;
}

__global__ void sh_convert_b_to_half_kernel(
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

__global__ void sh_convert_b_to_half_tail_kernel(
    const float* __restrict__ B,
    __half* __restrict__ Bh,
    i64 begin, i64 total)
{
    const i64 idx = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) Bh[idx] = __float2half_rn(B[idx]);
}

__global__ void sh_zero_rows_kernel(
    float* __restrict__ C,
    const int* __restrict__ row_ids,
    int num_rows, int N)
{
    const i64 total = (i64)num_rows * N;
    for (i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (i64)gridDim.x * blockDim.x) {
        const int row = row_ids[idx / N];
        C[(i64)row * N + (idx % N)] = 0.f;
    }
}

// Segment variant of the swapped-operand mma kernel: one blockIdx.x per
// segment; sole segments store to C, split segments write their 8xN partial
// tile to a scratch slot (merged deterministically by sh_merge_split_kernel).
// kStagedTile stages each warp's 8x16 B tile in shared memory (N%64==0).
template <bool kStagedTile>
__global__ void sh_fs_tile_spmm_seg_kernel(
    const int*      __restrict__ seg_window,
    const int*      __restrict__ seg_bbegin,
    const int*      __restrict__ seg_bend,
    const int*      __restrict__ seg_scratch,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const __half*   __restrict__ Bh,
    float*          __restrict__ C,
    float*          __restrict__ scratch,
    int M, int N, int num_segments)
{
    __shared__ __half s_b[4][8][24];
    const int s    = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0   = (blockIdx.y * 4 + warp) * 16;
    if (s >= num_segments || n0 >= N) return;

    const unsigned tag = static_cast<unsigned>(seg_window[s]);
    const bool sole = (tag & kSoleSegFlag) != 0u;
    const int  w    = static_cast<int>(tag & ~kSoleSegFlag);

    const int g   = lane >> 2;
    const int tid = lane & 3;

    const bool p_lo = (n0 + g)     < N;
    const bool p_hi = (n0 + g + 8) < N;

    const int b_begin = seg_bbegin[s];
    const int b_end   = seg_bend[s];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (int b = b_begin; b < b_end; ++b) {
        const uint32_t bfrag = __ldg(reinterpret_cast<const uint32_t*>(
            vals + (i64)b * 64) + lane);
        uint32_t afrag0, afrag1;
        if (kStagedTile) {
            const int srow   = lane >> 2;
            const int schunk = lane & 3;
            const int c = __ldg(atox + b * kVecPerBlock + srow);
            const uint2 chunk = *reinterpret_cast<const uint2*>(
                Bh + (i64)c * N + n0 + schunk * 4);
            *reinterpret_cast<uint2*>(&s_b[warp][srow][schunk * 4]) = chunk;
            __syncwarp();
            afrag0 = (static_cast<uint32_t>(
                          __half_as_ushort(s_b[warp][tid * 2 + 1][g])) << 16) |
                     __half_as_ushort(s_b[warp][tid * 2][g]);
            afrag1 = (static_cast<uint32_t>(
                          __half_as_ushort(s_b[warp][tid * 2 + 1][g + 8])) << 16) |
                     __half_as_ushort(s_b[warp][tid * 2][g + 8]);
        } else {
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
            afrag0 = (static_cast<uint32_t>(__half_as_ushort(a1)) << 16) |
                     __half_as_ushort(a0);
            afrag1 = (static_cast<uint32_t>(__half_as_ushort(a3)) << 16) |
                     __half_as_ushort(a2);
        }

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(afrag0), "r"(afrag1), "r"(bfrag));
        if (kStagedTile) __syncwarp();
    }

    const int r0 = w * kWindow + tid * 2;
    const int r1 = r0 + 1;
    if (sole) {
        if (p_lo) {
            if (r0 < M) C[(i64)r0 * N + n0 + g] = d0;
            if (r1 < M) C[(i64)r1 * N + n0 + g] = d1;
        }
        if (p_hi) {
            if (r0 < M) C[(i64)r0 * N + n0 + g + 8] = d2;
            if (r1 < M) C[(i64)r1 * N + n0 + g + 8] = d3;
        }
    } else {
        // Split segment: write the partial tile to this segment's scratch
        // slot; sh_merge_split_kernel sums slots in fixed ascending order.
        float* slot = scratch + (i64)seg_scratch[s] * kWindow * N;
        const int t0 = tid * 2;
        if (p_lo) {
            slot[(i64)t0 * N + n0 + g] = d0;
            slot[(i64)(t0 + 1) * N + n0 + g] = d1;
        }
        if (p_hi) {
            slot[(i64)t0 * N + n0 + g + 8] = d2;
            slot[(i64)(t0 + 1) * N + n0 + g + 8] = d3;
        }
    }
}

// Deterministic merge of split-window partial tiles into C.
__global__ void sh_merge_split_kernel(
    const int*   __restrict__ split_wins,
    const int*   __restrict__ split_off,
    const float* __restrict__ scratch,
    float*       __restrict__ C,
    int num_split_windows, int M, int N)
{
    const i64 total = (i64)num_split_windows * kWindow * N;
    for (i64 idx = (i64)blockIdx.x * blockDim.x + threadIdx.x;
         idx < total; idx += (i64)gridDim.x * blockDim.x) {
        const int sw   = (int)(idx / ((i64)kWindow * N));
        const i64 rem  = idx - (i64)sw * kWindow * N;
        const int slot = (int)(rem / N);
        const int col  = (int)(rem - (i64)slot * N);
        const int r    = split_wins[sw] * kWindow + slot;
        if (r >= M) continue;
        float acc = 0.f;
        const int o_end = split_off[sw + 1];
        for (int o = split_off[sw]; o < o_end; ++o) {
            acc += scratch[((i64)o * kWindow + slot) * N + col];
        }
        C[(i64)r * N + col] = acc;
    }
}

// Expand ZC-packed values into the padded fragment-order array on device
// (build-time only; the CPU never materializes the padded array).
__global__ void sh_zc_expand_vals_kernel(
    const unsigned long long* __restrict__ val_masks,
    const int*                __restrict__ val_base,
    const uint16_t*           __restrict__ vals_zc,
    uint16_t*                 __restrict__ vals,
    i64 total_vectors)
{
    for (i64 v = (i64)blockIdx.x * blockDim.x + threadIdx.x;
         v < total_vectors; v += (i64)gridDim.x * blockDim.x) {
        const i64 blk = v >> 3;
        const int k   = (int)(v & 7);
        const unsigned long long m64 = val_masks[blk];
        const unsigned byte_k = static_cast<unsigned>(m64 >> (k * 8)) & 0xFFu;
        if (!byte_k) continue;
        const unsigned long long lo = k ? (~0ull >> (64 - 8 * k)) : 0ull;
        int off = val_base[blk] + __popcll(m64 & lo);
        uint16_t* out = vals + blk * 64;
        for (int r = 0; r < kWindow; ++r) {
            if ((byte_k >> r) & 1u) {
                out[(static_cast<size_t>(r) * 4 + k / 2) * 2 + (k & 1)] =
                    vals_zc[off++];
            }
        }
    }
}

// TF32 variant of the segment kernel: reads B in FP32 directly (no convert
// pass, no half scratch). Same plan; each lane remaps its two sparse half
// slots to the tf32 B-fragment layout (rows k = tid and tid+4 of column g) —
// fp16 -> tf32 is exact. Dense loads are rounded with cvt.rna. Sole/split
// epilogue identical to the f16 kernel.
template <bool kStagedTile>
__global__ void sh_fs_tile_spmm_seg_tf32_kernel(
    const int*      __restrict__ seg_window,
    const int*      __restrict__ seg_bbegin,
    const int*      __restrict__ seg_bend,
    const int*      __restrict__ seg_scratch,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const float*    __restrict__ Bf,
    float*          __restrict__ C,
    float*          __restrict__ scratch,
    int M, int N, int num_segments)
{
    __shared__ float s_bf[4][8][24];
    const int s    = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0   = (blockIdx.y * 4 + warp) * 16;
    if (s >= num_segments || n0 >= N) return;

    const unsigned tag = static_cast<unsigned>(seg_window[s]);
    const bool sole = (tag & kSoleSegFlag) != 0u;
    const int  w    = static_cast<int>(tag & ~kSoleSegFlag);

    const int g   = lane >> 2;
    const int tid = lane & 3;

    const bool p_lo = (n0 + g)     < N;
    const bool p_hi = (n0 + g + 8) < N;

    const int b_begin = seg_bbegin[s];
    const int b_end   = seg_bend[s];

    // f16-order slot of (row r = g, vec k = tid); k = tid+4 lives at +4.
    const int vslot = (g * 4 + (tid >> 1)) * 2 + (tid & 1);

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (int b = b_begin; b < b_end; ++b) {
        const __half* vblk = reinterpret_cast<const __half*>(vals + (i64)b * 64);
        uint32_t rb0 = __float_as_uint(__half2float(__ldg(vblk + vslot)));
        uint32_t rb1 = __float_as_uint(__half2float(__ldg(vblk + vslot + 4)));

        uint32_t ra0, ra1, ra2, ra3;
        if (kStagedTile) {
            const int srow   = lane >> 2;
            const int schunk = lane & 3;
            const int c = __ldg(atox + b * kVecPerBlock + srow);
            const float4 chunk = *reinterpret_cast<const float4*>(
                Bf + (i64)c * N + n0 + schunk * 4);
            *reinterpret_cast<float4*>(&s_bf[warp][srow][schunk * 4]) = chunk;
            __syncwarp();
            ra0 = __float_as_uint(s_bf[warp][tid][g]);
            ra1 = __float_as_uint(s_bf[warp][tid][g + 8]);
            ra2 = __float_as_uint(s_bf[warp][tid + 4][g]);
            ra3 = __float_as_uint(s_bf[warp][tid + 4][g + 8]);
        } else {
            const int vbase = b * kVecPerBlock + tid;
            const int c_lo = __ldg(atox + vbase);
            const int c_hi = __ldg(atox + vbase + 4);
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            if (p_lo) {
                a0 = __ldg(Bf + (i64)c_lo * N + n0 + g);
                a2 = __ldg(Bf + (i64)c_hi * N + n0 + g);
            }
            if (p_hi) {
                a1 = __ldg(Bf + (i64)c_lo * N + n0 + g + 8);
                a3 = __ldg(Bf + (i64)c_hi * N + n0 + g + 8);
            }
            ra0 = __float_as_uint(a0); ra1 = __float_as_uint(a1);
            ra2 = __float_as_uint(a2); ra3 = __float_as_uint(a3);
        }
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra0));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra1));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra2));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra3));

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));
        if (kStagedTile) __syncwarp();
    }

    const int r0 = w * kWindow + tid * 2;
    const int r1 = r0 + 1;
    if (sole) {
        if (p_lo) {
            if (r0 < M) C[(i64)r0 * N + n0 + g] = d0;
            if (r1 < M) C[(i64)r1 * N + n0 + g] = d1;
        }
        if (p_hi) {
            if (r0 < M) C[(i64)r0 * N + n0 + g + 8] = d2;
            if (r1 < M) C[(i64)r1 * N + n0 + g + 8] = d3;
        }
    } else {
        float* slot = scratch + (i64)seg_scratch[s] * kWindow * N;
        const int t0 = tid * 2;
        if (p_lo) {
            slot[(i64)t0 * N + n0 + g] = d0;
            slot[(i64)(t0 + 1) * N + n0 + g] = d1;
        }
        if (p_hi) {
            slot[(i64)t0 * N + n0 + g + 8] = d2;
            slot[(i64)(t0 + 1) * N + n0 + g + 8] = d3;
        }
    }
}

}  // anonymous namespace

// ============================================================================
// make_ra_segment_hybrid_plan
// ============================================================================
void make_ra_segment_hybrid_plan(
    RASegmentHybridPlan& plan,
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M, int K, int N)
{
    plan = RASegmentHybridPlan{};
    plan.M = M;
    plan.K = K;

    if (M <= 0 || K <= 0 || N < 16) return;
    const i64 total_nnz = h_rowptr[M];
    if (total_nnz <= 0) return;

    const int num_windows = (M + kWindow - 1) / kWindow;
    std::vector<int> win_blocks(num_windows, 0);
    std::vector<std::vector<int>>      w_atox(num_windows);
    std::vector<std::vector<unsigned long long>> w_masks(num_windows);
    std::vector<std::vector<uint16_t>> w_zc(num_windows);

    #pragma omp parallel
    {
        std::vector<std::pair<int, std::pair<int, float>>> entries;
        #pragma omp for schedule(dynamic, 64)
        for (int w = 0; w < num_windows; ++w) {
            entries.clear();
            const int rlo = w * kWindow;
            const int rhi = std::min(M, rlo + kWindow);
            for (int r = rlo; r < rhi; ++r) {
                for (int p = h_rowptr[r]; p < h_rowptr[r + 1]; ++p) {
                    entries.push_back({h_col[p], {r - rlo, h_val[p]}});
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
            auto& masks = w_masks[w];
            auto& zvals = w_zc[w];
            atox.assign(static_cast<size_t>(blocks) * kVecPerBlock, 0);
            masks.assign(blocks, 0ull);
            zvals.reserve(entries.size());

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
                // Packed (vector, row) order; the padded fragment-order array
                // is expanded on device (sh_zc_expand_vals_kernel).
                masks[blk] |= 1ull << (k * 8 + r);
                zvals.push_back(float_to_half_bits(e.second.second));
            }
        }
    }

    std::vector<int> block_offsets(num_windows + 1, 0);
    for (int w = 0; w < num_windows; ++w) {
        block_offsets[w + 1] = block_offsets[w] + win_blocks[w];
    }
    const i64 total_blocks = block_offsets[num_windows];

    std::vector<int> atox_all(static_cast<size_t>(total_blocks) * kVecPerBlock);
    std::vector<unsigned long long> masks_all(total_blocks, 0ull);
    std::vector<int> base_all(total_blocks, 0);
    std::vector<uint16_t> zc_all(static_cast<size_t>(total_nnz));
    #pragma omp parallel for schedule(dynamic, 256)
    for (int w = 0; w < num_windows; ++w) {
        std::copy(w_atox[w].begin(), w_atox[w].end(),
                  atox_all.begin() + static_cast<size_t>(block_offsets[w]) * kVecPerBlock);
        const int b0 = block_offsets[w];
        std::copy(w_masks[w].begin(), w_masks[w].end(), masks_all.begin() + b0);
        // Natural window order: window w's packed values start at rowptr[w*8].
        int base = h_rowptr[w * kWindow];
        std::copy(w_zc[w].begin(), w_zc[w].end(), zc_all.begin() + base);
        for (int j = 0; j < win_blocks[w]; ++j) {
            base_all[b0 + j] = base;
            base += __builtin_popcountll(w_masks[w][j]);
        }
    }

    // ---- Balance splitting into segments of <= kSegBlocks blocks ----
    std::vector<int> seg_window, seg_bbegin, seg_bend, zero_rows;
    std::vector<int> seg_scratch, split_wins, split_off{0};
    int scratch_slots = 0;
    seg_window.reserve(num_windows + 16);
    for (int w = 0; w < num_windows; ++w) {
        const int b0 = block_offsets[w];
        const int b1 = block_offsets[w + 1];
        const int nblk = b1 - b0;
        if (nblk <= kSegBlocks) {
            seg_window.push_back(
                static_cast<int>(static_cast<unsigned>(w) | kSoleSegFlag));
            seg_bbegin.push_back(b0);
            seg_bend.push_back(b1);
            seg_scratch.push_back(-1);
        } else {
            split_wins.push_back(w);
            for (int b = b0; b < b1; b += kSegBlocks) {
                seg_window.push_back(w);
                seg_bbegin.push_back(b);
                seg_bend.push_back(std::min(b + kSegBlocks, b1));
                seg_scratch.push_back(scratch_slots++);
            }
            split_off.push_back(scratch_slots);
            for (int r = w * kWindow; r < std::min(M, (w + 1) * kWindow); ++r) {
                zero_rows.push_back(r);
            }
        }
    }

    plan.d_block_offsets = upload_vec_sh(block_offsets);
    plan.d_atox          = upload_vec_sh(atox_all);
    {
        unsigned long long* d_masks = upload_vec_sh(masks_all);
        int*      d_base   = upload_vec_sh(base_all);
        uint16_t* d_packed = upload_vec_sh(zc_all);
        const i64 total_vectors = total_blocks * kVecPerBlock;
        const size_t vals_bytes = static_cast<size_t>(total_blocks) * 64 * sizeof(uint16_t);
        CUDA_CHECK_NEXT(cudaMalloc(&plan.d_vals_f16, vals_bytes));
        CUDA_CHECK_NEXT(cudaMemset(plan.d_vals_f16, 0, vals_bytes));
        const int threads = 256;
        const int blocks_g = static_cast<int>(
            std::min<i64>((total_vectors + threads - 1) / threads, 65535));
        sh_zc_expand_vals_kernel<<<blocks_g, threads>>>(
            d_masks, d_base, d_packed, plan.d_vals_f16, total_vectors);
        CUDA_CHECK_NEXT(cudaDeviceSynchronize());
        cudaFree(d_masks);
        cudaFree(d_base);
        cudaFree(d_packed);
    }
    plan.d_seg_window    = upload_vec_sh(seg_window);
    plan.d_seg_bbegin    = upload_vec_sh(seg_bbegin);
    plan.d_seg_bend      = upload_vec_sh(seg_bend);
    plan.d_zero_rows     = upload_vec_sh(zero_rows);
    plan.d_seg_scratch   = upload_vec_sh(seg_scratch);
    plan.d_split_wins    = upload_vec_sh(split_wins);
    plan.d_split_off     = upload_vec_sh(split_off);
    plan.num_split_segs    = scratch_slots;
    plan.num_split_windows = static_cast<int>(split_wins.size());

    plan.num_windows   = num_windows;
    plan.num_segments  = static_cast<int>(seg_window.size());
    plan.num_zero_rows = static_cast<int>(zero_rows.size());
    plan.num_tc_groups = num_windows;
    plan.num_tc_tiles  = static_cast<int>(total_blocks);
    plan.tc_nnz_fraction   = 1.f;
    plan.cuda_nnz_fraction = 0.f;
    plan.plan_bytes = block_offsets.size() * sizeof(int) +
                      atox_all.size() * sizeof(int) +
                      (seg_window.size() + seg_bbegin.size() + seg_bend.size() +
                       zero_rows.size()) * sizeof(int) +
                      static_cast<size_t>(total_blocks) * 64 * sizeof(uint16_t);
    plan.active = true;
}

// ============================================================================
// run_ra_segment_hybrid_plan (d_colind/d_vals kept for ABI compatibility)
// ============================================================================
void run_ra_segment_hybrid_plan(
    const RASegmentHybridPlan& plan,
    const int* /*d_colind*/,
    const float* /*d_vals*/,
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
        sh_convert_b_to_half_kernel<<<blocks, threads, 0, stream>>>(
            d_B, reinterpret_cast<__half2*>(plan.d_bhalf), pairs);
    }
    if (total % 2 != 0) {
        sh_convert_b_to_half_tail_kernel<<<1, 32, 0, stream>>>(
            d_B, reinterpret_cast<__half*>(plan.d_bhalf), pairs * 2, total);
    }

    if (plan.num_split_segs > 0) {
        const size_t need = (size_t)plan.num_split_segs * 8 * N * sizeof(float);
        if (plan.scratch_capacity < need) {
            if (plan.d_scratch) cudaFree(plan.d_scratch);
            CUDA_CHECK_NEXT(cudaMalloc(&plan.d_scratch, need));
            plan.scratch_capacity = need;
        }
    }

    const char* stage_env = std::getenv("RA_TC_STAGED");
    const bool staged = !(stage_env && stage_env[0] == '0') && (N % 64 == 0);

    dim3 grid(plan.num_segments, (N + 63) / 64);
    if (staged) {
        sh_fs_tile_spmm_seg_kernel<true><<<grid, 128, 0, stream>>>(
            plan.d_seg_window, plan.d_seg_bbegin, plan.d_seg_bend,
            plan.d_seg_scratch, plan.d_atox, plan.d_vals_f16,
            reinterpret_cast<const __half*>(plan.d_bhalf), d_C,
            plan.d_scratch, plan.M, N, plan.num_segments);
    } else {
        sh_fs_tile_spmm_seg_kernel<false><<<grid, 128, 0, stream>>>(
            plan.d_seg_window, plan.d_seg_bbegin, plan.d_seg_bend,
            plan.d_seg_scratch, plan.d_atox, plan.d_vals_f16,
            reinterpret_cast<const __half*>(plan.d_bhalf), d_C,
            plan.d_scratch, plan.M, N, plan.num_segments);
    }
    if (plan.num_split_windows > 0) {
        const i64 total = (i64)plan.num_split_windows * 8 * N;
        const int threads = 256;
        const int blocks = static_cast<int>(
            std::min<i64>((total + threads - 1) / threads, 65535));
        sh_merge_split_kernel<<<blocks, threads, 0, stream>>>(
            plan.d_split_wins, plan.d_split_off, plan.d_scratch, d_C,
            plan.num_split_windows, plan.M, N);
    }

    CUDA_CHECK_KERNEL();
}

// ============================================================================
// run_ra_segment_hybrid_plan_tf32: B consumed in FP32 (no convert pass)
// ============================================================================
void run_ra_segment_hybrid_plan_tf32(
    const RASegmentHybridPlan& plan,
    const int* /*d_colind*/,
    const float* /*d_vals*/,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream)
{
    if (!plan.active || plan.M <= 0 || N <= 0) return;

    if (plan.num_split_segs > 0) {
        const size_t need = (size_t)plan.num_split_segs * 8 * N * sizeof(float);
        if (plan.scratch_capacity < need) {
            if (plan.d_scratch) cudaFree(plan.d_scratch);
            CUDA_CHECK_NEXT(cudaMalloc(&plan.d_scratch, need));
            plan.scratch_capacity = need;
        }
    }

    const char* stage_env = std::getenv("RA_TC_STAGED");
    const bool staged = !(stage_env && stage_env[0] == '0') && (N % 64 == 0);

    dim3 grid(plan.num_segments, (N + 63) / 64);
    if (staged) {
        sh_fs_tile_spmm_seg_tf32_kernel<true><<<grid, 128, 0, stream>>>(
            plan.d_seg_window, plan.d_seg_bbegin, plan.d_seg_bend,
            plan.d_seg_scratch, plan.d_atox, plan.d_vals_f16, d_B, d_C,
            plan.d_scratch, plan.M, N, plan.num_segments);
    } else {
        sh_fs_tile_spmm_seg_tf32_kernel<false><<<grid, 128, 0, stream>>>(
            plan.d_seg_window, plan.d_seg_bbegin, plan.d_seg_bend,
            plan.d_seg_scratch, plan.d_atox, plan.d_vals_f16, d_B, d_C,
            plan.d_scratch, plan.M, N, plan.num_segments);
    }
    if (plan.num_split_windows > 0) {
        const i64 total = (i64)plan.num_split_windows * 8 * N;
        const int threads = 256;
        const int blocks = static_cast<int>(
            std::min<i64>((total + threads - 1) / threads, 65535));
        sh_merge_split_kernel<<<blocks, threads, 0, stream>>>(
            plan.d_split_wins, plan.d_split_off, plan.d_scratch, d_C,
            plan.num_split_windows, plan.M, N);
    }

    CUDA_CHECK_KERNEL();
}

// ============================================================================
// free_ra_segment_hybrid_plan
// ============================================================================
void free_ra_segment_hybrid_plan(RASegmentHybridPlan& plan) {
    if (plan.d_block_offsets) { cudaFree(plan.d_block_offsets); plan.d_block_offsets = nullptr; }
    if (plan.d_atox)          { cudaFree(plan.d_atox);          plan.d_atox          = nullptr; }
    if (plan.d_vals_f16)      { cudaFree(plan.d_vals_f16);      plan.d_vals_f16      = nullptr; }
    if (plan.d_seg_window)    { cudaFree(plan.d_seg_window);    plan.d_seg_window    = nullptr; }
    if (plan.d_seg_bbegin)    { cudaFree(plan.d_seg_bbegin);    plan.d_seg_bbegin    = nullptr; }
    if (plan.d_seg_bend)      { cudaFree(plan.d_seg_bend);      plan.d_seg_bend      = nullptr; }
    if (plan.d_zero_rows)     { cudaFree(plan.d_zero_rows);     plan.d_zero_rows     = nullptr; }
    if (plan.d_seg_scratch)   { cudaFree(plan.d_seg_scratch);   plan.d_seg_scratch   = nullptr; }
    if (plan.d_split_wins)    { cudaFree(plan.d_split_wins);    plan.d_split_wins    = nullptr; }
    if (plan.d_split_off)     { cudaFree(plan.d_split_off);     plan.d_split_off     = nullptr; }
    if (plan.d_scratch)       { cudaFree(plan.d_scratch);       plan.d_scratch       = nullptr; }
    plan.scratch_capacity = 0;
    plan.num_split_segs = plan.num_split_windows = 0;
    if (plan.d_bhalf)         { cudaFree(plan.d_bhalf);         plan.d_bhalf         = nullptr; }
    plan.bhalf_capacity = 0;
    plan.num_windows = plan.num_segments = plan.num_zero_rows = 0;
    plan.num_tc_groups = plan.num_tc_tiles = 0;
    plan.active = false;
    plan.plan_bytes = 0;
}
