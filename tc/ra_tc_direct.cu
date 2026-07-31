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
#include <cstdlib>
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

// ---------------------------------------------------------------------------
// TF32 variant: reads B in FP32 directly — no convert pass, no half scratch.
// Same plan as the f16 path: sparse values stay in the f16 fragment-order
// array (fp16 -> tf32 is exact, both carry 10 mantissa bits), the kernel just
// remaps the two half slots each lane owns under the tf32 B-fragment layout
// (rows k = tid and tid+4 of column g). Dense loads are rounded to tf32 with
// cvt.rna. One mma.m16n8k8.tf32 per block, same C mapping as the f16 kernel.
// ---------------------------------------------------------------------------
__global__ void fs_tile_spmm_tf32_kernel(
    const int*      __restrict__ block_offsets,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const float*    __restrict__ Bf,
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

        // Dense A-fragment: A[m][k] = Bf[atox[k]][n0 + m].
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

// ---------------------------------------------------------------------------
// Staged variant of the swapped-operand mma kernel (requires N % 64 == 0):
// each warp cooperatively stages its 8-row x 16-column B tile into shared
// memory with one aligned 8-byte load per lane (vs 4 scattered 2-byte loads
// per lane in fragment order), then feeds the mma from shared memory.
// Row stride padded to 24 halfs for conflict-free fragment reads.
// ---------------------------------------------------------------------------
__global__ void fs_tile_spmm_staged_kernel(
    const int*      __restrict__ block_offsets,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const __half*   __restrict__ Bh,
    float*          __restrict__ C,
    int M, int N, int num_windows)
{
    __shared__ __half s_b[4][8][24];

    const int w    = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0   = (blockIdx.y * 4 + warp) * 16;
    if (w >= num_windows || n0 >= N) return;

    const int g   = lane >> 2;
    const int tid = lane & 3;

    const int srow   = lane >> 2;   // staging: B row within the tile
    const int schunk = lane & 3;    // staging: 4-half chunk within 16 cols

    const int b_begin = block_offsets[w];
    const int b_end   = block_offsets[w + 1];

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (int b = b_begin; b < b_end; ++b) {
        const uint32_t bfrag = __ldg(reinterpret_cast<const uint32_t*>(
            vals + (i64)b * 64) + lane);

        const int c = __ldg(atox + b * kVecPerBlock + srow);
        const uint2 chunk = *reinterpret_cast<const uint2*>(
            Bh + (i64)c * N + n0 + schunk * 4);
        *reinterpret_cast<uint2*>(&s_b[warp][srow][schunk * 4]) = chunk;
        __syncwarp();

        const uint32_t afrag0 =
            (static_cast<uint32_t>(__half_as_ushort(s_b[warp][tid * 2 + 1][g])) << 16) |
            __half_as_ushort(s_b[warp][tid * 2][g]);
        const uint32_t afrag1 =
            (static_cast<uint32_t>(__half_as_ushort(s_b[warp][tid * 2 + 1][g + 8])) << 16) |
            __half_as_ushort(s_b[warp][tid * 2][g + 8]);

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(afrag0), "r"(afrag1), "r"(bfrag));
        __syncwarp();
    }

    const int r0 = w * kWindow + tid * 2;
    const int r1 = r0 + 1;
    if (r0 < M) C[(i64)r0 * N + n0 + g] = d0;
    if (r1 < M) C[(i64)r1 * N + n0 + g] = d1;
    if (r0 < M) C[(i64)r0 * N + n0 + g + 8] = d2;
    if (r1 < M) C[(i64)r1 * N + n0 + g + 8] = d3;
}

// Staged TF32 variant (requires N % 64 == 0): stages FP32 tiles (one float4
// per lane), converts to tf32 at fragment read.
__global__ void fs_tile_spmm_staged_tf32_kernel(
    const int*      __restrict__ block_offsets,
    const int*      __restrict__ atox,
    const uint16_t* __restrict__ vals,
    const float*    __restrict__ Bf,
    float*          __restrict__ C,
    int M, int N, int num_windows)
{
    __shared__ float s_b[4][8][24];

    const int w    = blockIdx.x;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int n0   = (blockIdx.y * 4 + warp) * 16;
    if (w >= num_windows || n0 >= N) return;

    const int g   = lane >> 2;
    const int tid = lane & 3;

    const int srow   = lane >> 2;
    const int schunk = lane & 3;

    const int b_begin = block_offsets[w];
    const int b_end   = block_offsets[w + 1];

    const int vslot = (g * 4 + (tid >> 1)) * 2 + (tid & 1);

    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;

    for (int b = b_begin; b < b_end; ++b) {
        const __half* vblk = reinterpret_cast<const __half*>(vals + (i64)b * 64);
        uint32_t rb0 = __float_as_uint(__half2float(__ldg(vblk + vslot)));
        uint32_t rb1 = __float_as_uint(__half2float(__ldg(vblk + vslot + 4)));

        const int c = __ldg(atox + b * kVecPerBlock + srow);
        const float4 chunk = *reinterpret_cast<const float4*>(
            Bf + (i64)c * N + n0 + schunk * 4);
        *reinterpret_cast<float4*>(&s_b[warp][srow][schunk * 4]) = chunk;
        __syncwarp();

        uint32_t ra0 = __float_as_uint(s_b[warp][tid][g]);
        uint32_t ra1 = __float_as_uint(s_b[warp][tid][g + 8]);
        uint32_t ra2 = __float_as_uint(s_b[warp][tid + 4][g]);
        uint32_t ra3 = __float_as_uint(s_b[warp][tid + 4][g + 8]);
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra0));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra1));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra2));
        asm volatile("cvt.rna.tf32.f32 %0, %0;" : "+r"(ra3));

        asm volatile(
            "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(ra0), "r"(ra1), "r"(ra2), "r"(ra3), "r"(rb0), "r"(rb1));
        __syncwarp();
    }

    const int r0 = w * kWindow + tid * 2;
    const int r1 = r0 + 1;
    if (r0 < M) C[(i64)r0 * N + n0 + g] = d0;
    if (r1 < M) C[(i64)r1 * N + n0 + g] = d1;
    if (r0 < M) C[(i64)r0 * N + n0 + g + 8] = d2;
    if (r1 < M) C[(i64)r1 * N + n0 + g + 8] = d3;
}

// ---------------------------------------------------------------------------
// ZC-BCRS value lookup: masks u64 = 8 per-vector occupancy bytes; the packed
// value of (vector k, row r) sits at base + popc(masks below k) + popc(bits
// below r in byte k). Absent slots are zero.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint16_t zc_val_bits(
    unsigned long long m64, int base, const uint16_t* __restrict__ vals_zc,
    int k, int r)
{
    const unsigned byte_k = static_cast<unsigned>(m64 >> (k * 8)) & 0xFFu;
    if (!((byte_k >> r) & 1u)) return 0;
    const unsigned long long lo = k ? (~0ull >> (64 - 8 * k)) : 0ull;
    const int off = base + __popcll(m64 & lo) +
                    __popc(byte_k & ((1u << r) - 1u));
    return __ldg(vals_zc + off);
}

// Expand ZC-packed values into the padded fragment-order array on device:
// one thread per vector writes its <=8 values to their fragment slots.
// Used at build time so the CPU never materializes the padded array.
__global__ void zc_expand_vals_kernel(
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

// ZC variant of the swapped-operand mma kernel (fp16 gather path).
__global__ void fs_tile_spmm_zc_kernel(
    const int*                __restrict__ block_offsets,
    const int*                __restrict__ atox,
    const unsigned long long* __restrict__ val_masks,
    const int*                __restrict__ val_base,
    const uint16_t*           __restrict__ vals_zc,
    const __half*             __restrict__ Bh,
    float*                    __restrict__ C,
    int M, int N, int num_windows)
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

    // Software-pipeline the mask/base metadata one block ahead so the
    // dependent packed-value loads can issue at the top of each iteration
    // (otherwise mask -> popc -> value forms a two-deep latency chain).
    unsigned long long m64 = 0ull;
    int vb = 0;
    if (b_begin < b_end) {
        m64 = __ldg(val_masks + b_begin);
        vb  = __ldg(val_base + b_begin);
    }

    for (int b = b_begin; b < b_end; ++b) {
        const unsigned long long m_cur = m64;
        const int vb_cur = vb;
        if (b + 1 < b_end) {
            m64 = __ldg(val_masks + b + 1);
            vb  = __ldg(val_base + b + 1);
        }
        const uint16_t h0 = zc_val_bits(m_cur, vb_cur, vals_zc, tid * 2,     g);
        const uint16_t h1 = zc_val_bits(m_cur, vb_cur, vals_zc, tid * 2 + 1, g);
        const uint32_t bfrag = (static_cast<uint32_t>(h1) << 16) | h0;

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
    if (p_lo) {
        if (r0 < M) C[(i64)r0 * N + n0 + g] = d0;
        if (r1 < M) C[(i64)r1 * N + n0 + g] = d1;
    }
    if (p_hi) {
        if (r0 < M) C[(i64)r0 * N + n0 + g + 8] = d2;
        if (r1 < M) C[(i64)r1 * N + n0 + g + 8] = d3;
    }
}

// ZC variant of the TF32 kernel (B consumed in FP32, no convert pass).
__global__ void fs_tile_spmm_zc_tf32_kernel(
    const int*                __restrict__ block_offsets,
    const int*                __restrict__ atox,
    const unsigned long long* __restrict__ val_masks,
    const int*                __restrict__ val_base,
    const uint16_t*           __restrict__ vals_zc,
    const float*              __restrict__ Bf,
    float*                    __restrict__ C,
    int M, int N, int num_windows)
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

    // Same metadata software pipeline as the fp16 ZC kernel.
    unsigned long long m64 = 0ull;
    int vb = 0;
    if (b_begin < b_end) {
        m64 = __ldg(val_masks + b_begin);
        vb  = __ldg(val_base + b_begin);
    }

    for (int b = b_begin; b < b_end; ++b) {
        const unsigned long long m_cur = m64;
        const int vb_cur = vb;
        if (b + 1 < b_end) {
            m64 = __ldg(val_masks + b + 1);
            vb  = __ldg(val_base + b + 1);
        }
        const uint16_t h0 = zc_val_bits(m_cur, vb_cur, vals_zc, tid,     g);
        const uint16_t h1 = zc_val_bits(m_cur, vb_cur, vals_zc, tid + 4, g);
        uint32_t rb0 = __float_as_uint(__half2float(__ushort_as_half(h0)));
        uint32_t rb1 = __float_as_uint(__half2float(__ushort_as_half(h1)));

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
// Plan build: parallel per-window column condensation, O(nnz). With zc set,
// values are stored zero-compressed: packed nonzero halves in (vector, row)
// order plus one 8-bit row-occupancy mask per vector (8 masks = one u64 per
// block). A window's packed values start at h_rowptr[window*8], so the build
// stays one parallel pass with no extra prefix scan.
// ============================================================================
static void build_ra_tc_direct_plan_impl(
    RATcDirectPlan& plan,
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M, int K, int N,
    bool zc)
{
    plan = RATcDirectPlan{};
    plan.M = M;
    plan.K = K;
    plan.zc = zc;

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
    std::vector<std::vector<unsigned long long>> w_masks(num_windows);
    std::vector<std::vector<uint16_t>> w_zc(num_windows);

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
            atox.assign(static_cast<size_t>(blocks) * kVecPerBlock, 0);
            auto& masks = w_masks[w];
            auto& zvals = w_zc[w];
            masks.assign(blocks, 0ull);
            zvals.reserve(entries.size());

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
                // Packed (vector, row) order = entry iteration order. The
                // padded fragment-order array, when needed, is expanded on
                // device (zc_expand_vals_kernel) — the CPU never writes it.
                masks[blk] |= 1ull << (k * 8 + r);
                zvals.push_back(host_float_to_half_bits(e.second.second));
            }
        }
    }

    // Serial prefix + concatenation (deterministic).
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
        // Window w's packed values start at h_rowptr[w*8].
        int base = h_rowptr[w * kWindow];
        std::copy(w_zc[w].begin(), w_zc[w].end(), zc_all.begin() + base);
        for (int j = 0; j < win_blocks[w]; ++j) {
            base_all[b0 + j] = base;
            base += __builtin_popcountll(w_masks[w][j]);
        }
    }
    const i64 total_vectors = total_blocks * kVecPerBlock;

    plan.d_block_offsets = upload_vec(block_offsets);
    plan.d_atox          = upload_vec(atox_all);
    if (zc) {
        plan.d_val_masks = upload_vec(masks_all);
        plan.d_val_base  = upload_vec(base_all);
        plan.d_vals_zc   = upload_vec(zc_all);
    } else {
        // Upload the compact arrays and expand into the padded fragment-order
        // array on device — the padded array never crosses PCIe or CPU RAM.
        unsigned long long* d_masks = upload_vec(masks_all);
        int*      d_base   = upload_vec(base_all);
        uint16_t* d_packed = upload_vec(zc_all);
        const size_t vals_bytes = static_cast<size_t>(total_blocks) * 64 * sizeof(uint16_t);
        CUDA_CHECK_NEXT(cudaMalloc(&plan.d_vals_f16, vals_bytes));
        CUDA_CHECK_NEXT(cudaMemset(plan.d_vals_f16, 0, vals_bytes));
        const int threads = 256;
        const int blocks_g = static_cast<int>(
            std::min<i64>((total_vectors + threads - 1) / threads, 65535));
        zc_expand_vals_kernel<<<blocks_g, threads>>>(
            d_masks, d_base, d_packed, plan.d_vals_f16, total_vectors);
        CUDA_CHECK_NEXT(cudaDeviceSynchronize());
        cudaFree(d_masks);
        cudaFree(d_base);
        cudaFree(d_packed);
    }

    plan.num_windows  = num_windows;
    plan.num_groups   = num_windows;
    plan.num_tc_tiles = static_cast<int>(total_blocks);
    plan.avg_tc_tile_density = (total_vectors > 0)
        ? static_cast<float>(static_cast<double>(total_nnz) /
                             (static_cast<double>(total_vectors) * kWindow))
        : 0.f;
    plan.plan_bytes = block_offsets.size() * sizeof(int) +
                      atox_all.size() * sizeof(int);
    if (zc) {
        plan.plan_bytes += masks_all.size() * sizeof(unsigned long long) +
                           base_all.size() * sizeof(int) +
                           zc_all.size() * sizeof(uint16_t);
    } else {
        plan.plan_bytes += static_cast<size_t>(total_blocks) * 64 * sizeof(uint16_t);
    }
    plan.active = true;
}

void make_ra_tc_direct_plan(
    RATcDirectPlan& plan,
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M, int K, int N)
{
    build_ra_tc_direct_plan_impl(plan, h_rowptr, h_col, h_val, M, K, N, false);
}

void make_ra_tc_direct_zc_plan(
    RATcDirectPlan& plan,
    const int* h_rowptr,
    const int* h_col,
    const float* h_val,
    int M, int K, int N)
{
    build_ra_tc_direct_plan_impl(plan, h_rowptr, h_col, h_val, M, K, N, true);
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

    const char* stage_env = std::getenv("RA_TC_STAGED");
    const bool kStaged = !(stage_env && stage_env[0] == '0');

    dim3 grid(plan.num_windows, (N + 63) / 64);
    if (plan.zc) {
        fs_tile_spmm_zc_kernel<<<grid, 128, 0, stream>>>(
            plan.d_block_offsets, plan.d_atox,
            plan.d_val_masks, plan.d_val_base, plan.d_vals_zc,
            reinterpret_cast<const __half*>(plan.d_bhalf), d_C,
            plan.M, N, plan.num_windows);
    } else if (kStaged && (N % 64 == 0)) {
        fs_tile_spmm_staged_kernel<<<grid, 128, 0, stream>>>(
            plan.d_block_offsets, plan.d_atox, plan.d_vals_f16,
            reinterpret_cast<const __half*>(plan.d_bhalf), d_C,
            plan.M, N, plan.num_windows);
    } else {
        fs_tile_spmm_kernel<<<grid, 128, 0, stream>>>(
            plan.d_block_offsets, plan.d_atox, plan.d_vals_f16,
            reinterpret_cast<const __half*>(plan.d_bhalf), d_C,
            plan.M, N, plan.num_windows);
    }

    CUDA_CHECK_KERNEL();
}

// ============================================================================
// run_ra_tc_direct_plan_tf32: single launch, B consumed in FP32 (no convert)
// ============================================================================
void run_ra_tc_direct_plan_tf32(
    const RATcDirectPlan& plan,
    const float* d_B,
    float* d_C,
    int N,
    cudaStream_t stream)
{
    if (!plan.active || plan.M <= 0 || N <= 0) return;

    const char* stage_env = std::getenv("RA_TC_STAGED");
    const bool kStaged = !(stage_env && stage_env[0] == '0');

    dim3 grid(plan.num_windows, (N + 63) / 64);
    if (plan.zc) {
        fs_tile_spmm_zc_tf32_kernel<<<grid, 128, 0, stream>>>(
            plan.d_block_offsets, plan.d_atox,
            plan.d_val_masks, plan.d_val_base, plan.d_vals_zc,
            d_B, d_C, plan.M, N, plan.num_windows);
    } else if (kStaged && (N % 64 == 0)) {
        fs_tile_spmm_staged_tf32_kernel<<<grid, 128, 0, stream>>>(
            plan.d_block_offsets, plan.d_atox, plan.d_vals_f16,
            d_B, d_C, plan.M, N, plan.num_windows);
    } else {
        fs_tile_spmm_tf32_kernel<<<grid, 128, 0, stream>>>(
            plan.d_block_offsets, plan.d_atox, plan.d_vals_f16,
            d_B, d_C, plan.M, N, plan.num_windows);
    }

    CUDA_CHECK_KERNEL();
}

// ============================================================================
// free_ra_tc_direct_plan
// ============================================================================
void free_ra_tc_direct_plan(RATcDirectPlan& plan) {
    if (plan.d_block_offsets) { cudaFree(plan.d_block_offsets); plan.d_block_offsets = nullptr; }
    if (plan.d_atox)          { cudaFree(plan.d_atox);          plan.d_atox          = nullptr; }
    if (plan.d_vals_f16)      { cudaFree(plan.d_vals_f16);      plan.d_vals_f16      = nullptr; }
    if (plan.d_val_masks)     { cudaFree(plan.d_val_masks);     plan.d_val_masks     = nullptr; }
    if (plan.d_val_base)      { cudaFree(plan.d_val_base);      plan.d_val_base      = nullptr; }
    if (plan.d_vals_zc)       { cudaFree(plan.d_vals_zc);       plan.d_vals_zc       = nullptr; }
    plan.zc = false;
    if (plan.d_bhalf)         { cudaFree(plan.d_bhalf);         plan.d_bhalf         = nullptr; }
    plan.bhalf_capacity = 0;
    plan.num_windows = plan.num_groups = plan.num_tc_tiles = 0;
    plan.active = false;
    plan.plan_bytes = 0;
}
