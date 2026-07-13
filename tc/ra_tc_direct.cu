// ============================================================================
// ra_tc_direct.cu - R4: Tensor-core SpMM in the FlashSparse ME-BCRS style
//
// Format (built in NATURAL row order — no reordering):
//   - Rows are grouped into windows of 8. Within a window every distinct
//     column becomes an 8x1 vector; vectors are grouped into blocks of 8
//     (one mma.m16n8k8 step each), the last block zero-padded.
//   - d_atox[v] holds the original column (= B row) gathered for vector v.
//   - d_vals_f16 stores each block's 8x8 values (vector x window-row)
//     directly in the PTX B-fragment ownership order for
//     mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32, so a warp loads its
//     sparse fragment with one coalesced half2 (uint32) read per lane:
//       index(block, vec k, row r) = block*64 + (r*4 + k/2)*2 + (k&1)
//
// Kernel (swap-and-transpose, FlashSparse PPoPP'25):
//   The dense gather tile (16 N-columns x 8 B-rows) is the mma A operand and
//   the sparse block is the B operand; the accumulator is the transposed
//   16x8 C tile in FP32. FP16 inputs / FP32 accumulate on sm_80+.
//   B is converted once per call to a lazily-allocated half scratch, halving
//   gather traffic versus FP32 CUDA-core kernels.
//
// Grid: (num_windows, ceil(N/64)); block 128 = 4 warps x 16 N-columns.
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

constexpr int kWindow      = 8;   // rows per window
constexpr int kVecPerBlock = 8;   // vectors per mma block (k dimension)

// ---------------------------------------------------------------------------
// Host float -> IEEE fp16 bits (round to nearest even)
// ---------------------------------------------------------------------------
inline uint16_t host_float_to_half_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = static_cast<int32_t>((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFFu;
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u);  // inf/overflow
    if (exp <= 0) {
        if (exp < -10) return static_cast<uint16_t>(sign);        // underflow -> 0
        mant |= 0x800000u;                                        // implicit bit
        const int shift = 14 - exp;
        uint32_t half_mant = mant >> shift;
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half_mant & 1u))) ++half_mant;
        return static_cast<uint16_t>(sign | half_mant);
    }
    uint32_t half_mant = mant >> 13;
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half_mant & 1u))) {
        ++half_mant;
        if (half_mant == 0x400u) { half_mant = 0; ++exp; if (exp >= 31) return static_cast<uint16_t>(sign | 0x7C00u); }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | half_mant);
}

template <typename T>
T* upload_vec(const std::vector<T>& values) {
    if (values.empty()) return nullptr;
    T* d_ptr = nullptr;
    CUDA_CHECK_NEXT(cudaMalloc(&d_ptr, values.size() * sizeof(T)));
    CUDA_CHECK_NEXT(cudaMemcpy(d_ptr, values.data(),
                               values.size() * sizeof(T),
                               cudaMemcpyHostToDevice));
    return d_ptr;
}

// ---------------------------------------------------------------------------
// B fp32 -> fp16 conversion kernel (grid-stride, half2 stores)
// ---------------------------------------------------------------------------
__global__ void convert_b_to_half_kernel(
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

__global__ void convert_b_to_half_tail_kernel(
    const float* __restrict__ B,
    __half* __restrict__ Bh,
    i64 begin, i64 total)
{
    const i64 idx = begin + blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) Bh[idx] = __float2half_rn(B[idx]);
}

// ---------------------------------------------------------------------------
// Swapped-operand mma SpMM kernel
// ---------------------------------------------------------------------------
__global__ void fs_tile_spmm_kernel(
    const int*      __restrict__ block_offsets,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const __half*   __restrict__ Bh,
    float*          __restrict__ C,
    int M, int N, int num_windows)
{
    const int w    = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0   = (blockIdx.y * 4 + warp) * 16;
    if (w >= num_windows || n0 >= N) return;

    const int g   = lane >> 2;  // 0..7: N-column within the warp's 16
    const int tid = lane & 3;   // 0..3

    const bool p_lo = (n0 + g)     < N;
    const bool p_hi = (n0 + g + 8) < N;

    const int b_begin = block_offsets[w];
    const int b_end   = block_offsets[w + 1];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (int b = b_begin; b < b_end; ++b) {
        // Sparse B-fragment: one coalesced half2 per lane (fragment order).
        const uint32_t bfrag = __ldg(reinterpret_cast<const uint32_t*>(
            vals + (i64)b * 64) + lane);

        // Gather columns for this lane's two k-slots.
        const int vbase = b * kVecPerBlock + tid * 2;
        const int c0 = __ldg(atox + vbase);
        const int c1 = __ldg(atox + vbase + 1);

        // Dense A-fragment: A[m][k] = Bh[atox[k]][n0 + m].
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

    // D[m][n] is the transposed C tile: rows = window rows, cols = N-cols.
    const int r0 = w * kWindow + tid * 2;
    const int r1 = r0 + 1;
    if (p_lo) {
        if (r0 < M) C[(i64)r0 * N + n0 + g] = d0;
        if (r1 < M) C[(i64)r1 * N + n0 + g] = d1;
    }
    if (p_hi) {
        if (r0 < M) C[(i64)r0 * N + n0 + g + 8] = d2;
        if (r1 < M) C[(i64)r1 * N + n0 + g + 8] = d3;
    }
}

}  // anonymous namespace

// ============================================================================
// make_ra_tc_direct_plan: parallel per-window column condensation, O(nnz)
// ============================================================================
void make_ra_tc_direct_plan(
    RATcDirectPlan& plan,
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M, int K, int N)
{
    plan = RATcDirectPlan{};
    plan.M = M;
    plan.K = K;

    if (M <= 0 || K <= 0 || N < 16) {
        return;
    }
    const i64 total_nnz = h_rowptr[M];
    if (total_nnz <= 0) {
        return;
    }

    const int num_windows = (M + kWindow - 1) / kWindow;
    std::vector<int> win_blocks(num_windows, 0);
    std::vector<std::vector<int>>      w_atox(num_windows);
    std::vector<std::vector<uint16_t>> w_vals(num_windows);

    #pragma omp parallel
    {
        // (col, window_row, value) entries for one window
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

            int slot = -1;
            int prev_col = -1;
            for (const auto& e : entries) {
                if (e.first != prev_col) {
                    ++slot;
                    prev_col = e.first;
                    atox[slot] = e.first;
                }
                const int blk = slot / kVecPerBlock;
                const int k   = slot % kVecPerBlock;
                const int r   = e.second.first;
                // PTX B-fragment order: lane t = r*4 + k/2 holds slots
                // {t*2, t*2+1} = vectors {k = 2*(t%4), +1} at row t/4.
                const size_t idx = static_cast<size_t>(blk) * 64 +
                    (static_cast<size_t>(r) * 4 + k / 2) * 2 + (k & 1);
                vals[idx] = host_float_to_half_bits(e.second.second);
            }
        }
    }

    // Serial prefix + concatenation (deterministic).
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

    plan.d_block_offsets = upload_vec(block_offsets);
    plan.d_atox          = upload_vec(atox_all);
    plan.d_vals_f16      = upload_vec(vals_all);

    plan.num_windows  = num_windows;
    plan.num_groups   = num_windows;
    plan.num_tc_tiles = static_cast<int>(total_blocks);
    plan.avg_tc_tile_density = (total_vectors > 0)
        ? static_cast<float>(static_cast<double>(total_nnz) /
                             (static_cast<double>(total_vectors) * kWindow))
        : 0.f;
    plan.plan_bytes = block_offsets.size() * sizeof(int) +
                      atox_all.size() * sizeof(int) +
                      vals_all.size() * sizeof(uint16_t);
    plan.active = true;
}

// ============================================================================
// run_ra_tc_direct_plan
// ============================================================================
void run_ra_tc_direct_plan(
    const RATcDirectPlan& plan,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream)
{
    if (!plan.active || plan.M <= 0 || N <= 0) return;

    // Lazily (re)allocate the half-precision B scratch.
    const i64 total = (i64)plan.K * N;
    const size_t need = static_cast<size_t>(total) * sizeof(uint16_t);
    if (plan.bhalf_capacity < need) {
        if (plan.d_bhalf) cudaFree(plan.d_bhalf);
        CUDA_CHECK_NEXT(cudaMalloc(&plan.d_bhalf, need));
        plan.bhalf_capacity = need;
    }

    // Convert B to half.
    const i64 pairs = total / 2;
    if (pairs > 0) {
        const int threads = 256;
        const int blocks = static_cast<int>(
            std::min<i64>((pairs + threads - 1) / threads, 65535));
        convert_b_to_half_kernel<<<blocks, threads, 0, stream>>>(
            d_B, reinterpret_cast<__half2*>(plan.d_bhalf), pairs);
    }
    if (total % 2 != 0) {
        convert_b_to_half_tail_kernel<<<1, 32, 0, stream>>>(
            d_B, reinterpret_cast<__half*>(plan.d_bhalf), pairs * 2, total);
    }

    dim3 grid(plan.num_windows, (N + 63) / 64);
    fs_tile_spmm_kernel<<<grid, 128, 0, stream>>>(
        plan.d_block_offsets, plan.d_atox, plan.d_vals_f16,
        reinterpret_cast<const __half*>(plan.d_bhalf), d_C,
        plan.M, N, plan.num_windows);

    CUDA_CHECK_KERNEL();
}

// ============================================================================
// free_ra_tc_direct_plan
// ============================================================================
void free_ra_tc_direct_plan(RATcDirectPlan& plan) {
    if (plan.d_block_offsets) { cudaFree(plan.d_block_offsets); plan.d_block_offsets = nullptr; }
    if (plan.d_atox)          { cudaFree(plan.d_atox);          plan.d_atox          = nullptr; }
    if (plan.d_vals_f16)      { cudaFree(plan.d_vals_f16);      plan.d_vals_f16      = nullptr; }
    if (plan.d_bhalf)         { cudaFree(plan.d_bhalf);         plan.d_bhalf         = nullptr; }
    plan.bhalf_capacity = 0;
    plan.num_windows = plan.num_groups = plan.num_tc_tiles = 0;
    plan.active = false;
    plan.plan_bytes = 0;
}
