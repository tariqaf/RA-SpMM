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
// segment; sole segments store, split segments atomicAdd into pre-zeroed C.
__global__ void sh_fs_tile_spmm_seg_kernel(
    const int*      __restrict__ seg_window,
    const int*      __restrict__ seg_bbegin,
    const int*      __restrict__ seg_bend,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const __half*   __restrict__ Bh,
    float*          __restrict__ C,
    int M, int N, int num_segments)
{
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
        if (p_lo) {
            if (r0 < M) atomicAdd(&C[(i64)r0 * N + n0 + g], d0);
            if (r1 < M) atomicAdd(&C[(i64)r1 * N + n0 + g], d1);
        }
        if (p_hi) {
            if (r0 < M) atomicAdd(&C[(i64)r0 * N + n0 + g + 8], d2);
            if (r1 < M) atomicAdd(&C[(i64)r1 * N + n0 + g + 8], d3);
        }
    }
}

// TF32 variant of the segment kernel: reads B in FP32 directly (no convert
// pass, no half scratch). Same plan; each lane remaps its two sparse half
// slots to the tf32 B-fragment layout (rows k = tid and tid+4 of column g) —
// fp16 -> tf32 is exact. Dense loads are rounded with cvt.rna. Sole/split
// epilogue identical to the f16 kernel.
__global__ void sh_fs_tile_spmm_seg_tf32_kernel(
    const int*      __restrict__ seg_window,
    const int*      __restrict__ seg_bbegin,
    const int*      __restrict__ seg_bend,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const float*    __restrict__ Bf,
    float*          __restrict__ C,
    int M, int N, int num_segments)
{
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
        uint32_t ra0 = __float_as_uint(a0), ra1 = __float_as_uint(a1);
        uint32_t ra2 = __float_as_uint(a2), ra3 = __float_as_uint(a3);
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra0));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra1));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra2));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra3));

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));
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
        if (p_lo) {
            if (r0 < M) atomicAdd(&C[(i64)r0 * N + n0 + g], d0);
            if (r1 < M) atomicAdd(&C[(i64)r1 * N + n0 + g], d1);
        }
        if (p_hi) {
            if (r0 < M) atomicAdd(&C[(i64)r0 * N + n0 + g + 8], d2);
            if (r1 < M) atomicAdd(&C[(i64)r1 * N + n0 + g + 8], d3);
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
    std::vector<std::vector<uint16_t>> w_vals(num_windows);

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

    // ---- Balance splitting into segments of <= kSegBlocks blocks ----
    std::vector<int> seg_window, seg_bbegin, seg_bend, zero_rows;
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
        } else {
            for (int b = b0; b < b1; b += kSegBlocks) {
                seg_window.push_back(w);
                seg_bbegin.push_back(b);
                seg_bend.push_back(std::min(b + kSegBlocks, b1));
            }
            for (int r = w * kWindow; r < std::min(M, (w + 1) * kWindow); ++r) {
                zero_rows.push_back(r);
            }
        }
    }

    plan.d_block_offsets = upload_vec_sh(block_offsets);
    plan.d_atox          = upload_vec_sh(atox_all);
    plan.d_vals_f16      = upload_vec_sh(vals_all);
    plan.d_seg_window    = upload_vec_sh(seg_window);
    plan.d_seg_bbegin    = upload_vec_sh(seg_bbegin);
    plan.d_seg_bend      = upload_vec_sh(seg_bend);
    plan.d_zero_rows     = upload_vec_sh(zero_rows);

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
                      vals_all.size() * sizeof(uint16_t);
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

    if (plan.num_zero_rows > 0) {
        const i64 zr_total = (i64)plan.num_zero_rows * N;
        const int threads = 256;
        const int blocks = static_cast<int>(
            std::min<i64>((zr_total + threads - 1) / threads, 65535));
        sh_zero_rows_kernel<<<blocks, threads, 0, stream>>>(
            d_C, plan.d_zero_rows, plan.num_zero_rows, N);
    }

    dim3 grid(plan.num_segments, (N + 63) / 64);
    sh_fs_tile_spmm_seg_kernel<<<grid, 128, 0, stream>>>(
        plan.d_seg_window, plan.d_seg_bbegin, plan.d_seg_bend,
        plan.d_atox, plan.d_vals_f16,
        reinterpret_cast<const __half*>(plan.d_bhalf), d_C,
        plan.M, N, plan.num_segments);

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

    if (plan.num_zero_rows > 0) {
        const i64 zr_total = (i64)plan.num_zero_rows * N;
        const int threads = 256;
        const int blocks = static_cast<int>(
            std::min<i64>((zr_total + threads - 1) / threads, 65535));
        sh_zero_rows_kernel<<<blocks, threads, 0, stream>>>(
            d_C, plan.d_zero_rows, plan.num_zero_rows, N);
    }

    dim3 grid(plan.num_segments, (N + 63) / 64);
    sh_fs_tile_spmm_seg_tf32_kernel<<<grid, 128, 0, stream>>>(
        plan.d_seg_window, plan.d_seg_bbegin, plan.d_seg_bend,
        plan.d_atox, plan.d_vals_f16, d_B, d_C,
        plan.M, N, plan.num_segments);

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
    if (plan.d_bhalf)         { cudaFree(plan.d_bhalf);         plan.d_bhalf         = nullptr; }
    plan.bhalf_capacity = 0;
    plan.num_windows = plan.num_segments = plan.num_zero_rows = 0;
    plan.num_tc_groups = plan.num_tc_tiles = 0;
    plan.active = false;
    plan.plan_bytes = 0;
}
