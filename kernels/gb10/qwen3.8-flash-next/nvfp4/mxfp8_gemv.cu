// SPDX-License-Identifier: AGPL-3.0-only

// MXFP8 GEMV: FP8-E4M3 weights with one power-of-two scale byte per 32 columns,
// read in the checkpoint's own layout (no load-time conversion).
//
//   C[t, n] = sum_k A[t, k] * e4m3(B[n, k]) * 2^(S[n, k / 32] - 127)
//
// Used by Qwen3.8-Flash-Next for the GDN, attention, indexer and shared-expert
// projections. K must be a multiple of 32.
//
// 4 outputs per block, 64 threads per output, one 32-column scale group per
// thread iteration.
// Grid: (ceil(N / 4), T, 1)   Block: (256, 1, 1)

#include <cuda_bf16.h>

#define MX_BLOCK 256
#define MX_N_PER_BLOCK 4
#define MX_GROUP 32

// E4M3 byte times 2^exp2. Subnormals (exponent field 0) are m * 2^-9, normals
// (8 + m) * 2^(e - 10). The NaN codes 0x7F / 0xFF read as zero, as in the
// other FP8 kernels.
__device__ __forceinline__ float mx_e4m3_scaled(unsigned int b, int exp2) {
    const unsigned int e = (b >> 3) & 0xF;
    const unsigned int m = b & 0x7;
    const int mant = ((b & 0x7F) == 0x7F) ? 0 : (int)((e ? 8u : 0u) + m);
    const float v = scalbnf((float)mant, (int)(e ? e : 1u) - 10 + exp2);
    return (b & 0x80) ? -v : v;
}

extern "C" __global__ void mxfp8_gemv(
    const __nv_bfloat16* __restrict__ A, // [T, K]
    const unsigned char* __restrict__ B, // [N, K] FP8 E4M3
    const unsigned char* __restrict__ S, // [N, K / 32] exponent bytes
    __nv_bfloat16* __restrict__ C,       // [T, N]
    const unsigned int N,
    const unsigned int K
) {
    __shared__ float red[MX_BLOCK];
    const unsigned int threads_per_out = MX_BLOCK / MX_N_PER_BLOCK; // 64
    const unsigned int local_out = threadIdx.x / threads_per_out;
    const unsigned int lane = threadIdx.x % threads_per_out;
    const unsigned int n = blockIdx.x * MX_N_PER_BLOCK + local_out;
    const unsigned int groups = K / MX_GROUP;

    float acc = 0.0f;
    if (n < N) {
        const __nv_bfloat16* a = A + (size_t)blockIdx.y * K;
        const unsigned char* b = B + (size_t)n * K;
        const unsigned char* s = S + (size_t)n * groups;
        for (unsigned int g = lane; g < groups; g += threads_per_out) {
            const int exp2 = (int)s[g] - 127;
            const unsigned int* bw = (const unsigned int*)(b + (size_t)g * MX_GROUP);
            const __nv_bfloat16* ag = a + (size_t)g * MX_GROUP;
            #pragma unroll
            for (unsigned int q = 0; q < MX_GROUP / 4; ++q) {
                const unsigned int w32 = bw[q];
                acc += __bfloat162float(ag[q * 4 + 0]) * mx_e4m3_scaled(w32 & 0xFF, exp2);
                acc += __bfloat162float(ag[q * 4 + 1]) * mx_e4m3_scaled((w32 >> 8) & 0xFF, exp2);
                acc += __bfloat162float(ag[q * 4 + 2]) * mx_e4m3_scaled((w32 >> 16) & 0xFF, exp2);
                acc += __bfloat162float(ag[q * 4 + 3]) * mx_e4m3_scaled(w32 >> 24, exp2);
            }
        }
    }
    red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int step = threads_per_out / 2; step > 0; step >>= 1) {
        if (lane < step) red[threadIdx.x] += red[threadIdx.x + step];
        __syncthreads();
    }
    if (lane == 0 && n < N) C[(size_t)blockIdx.y * N + n] = __float2bfloat16(red[threadIdx.x]);
}

// ── swiglu_mul ──
// out[i] = silu(gate[i]) * up[i], without the activation clamp the shared
// `moe_silu_mul` applies for another model family.
// Grid: (ceil(n / 256),1,1)  Block: (256,1,1).
extern "C" __global__ void swiglu_mul(
    const __nv_bfloat16* __restrict__ gate, // [n]
    const __nv_bfloat16* __restrict__ up,   // [n]
    __nv_bfloat16* __restrict__ out,        // [n]
    const unsigned int n
) {
    const unsigned int i = blockIdx.x * MX_BLOCK + threadIdx.x;
    if (i >= n) return;
    const float g = __bfloat162float(gate[i]);
    out[i] = __float2bfloat16(g / (1.0f + expf(-g)) * __bfloat162float(up[i]));
}

// ── mxfp8_expand_bf16 ──
// out[n, k] = e4m3(B[n, k]) * 2^(S[n, k / 32] - 127) as BF16. The value has
// at most four significant bits, so BF16 holds it exactly. Used in prefill:
// a projection is expanded into a scratch matrix for the duration of one
// BF16 GEMM over all prompt tokens of a chunk; the resident weight stays
// MXFP8.
// Grid: (ceil(N * K / 32 / 256),1,1)  Block: (256,1,1), one 32-column group
// per thread.
extern "C" __global__ void mxfp8_expand_bf16(
    const unsigned char* __restrict__ B, // [N, K] FP8 E4M3
    const unsigned char* __restrict__ S, // [N, K / 32] exponent bytes
    __nv_bfloat16* __restrict__ out,     // [N, K]
    const unsigned int N,
    const unsigned int K
) {
    const size_t g = (size_t)blockIdx.x * MX_BLOCK + threadIdx.x;
    const size_t groups = (size_t)N * (K / MX_GROUP);
    if (g >= groups) return;
    const int exp2 = (int)S[g] - 127;
    const unsigned char* b = B + g * MX_GROUP;
    __nv_bfloat16* o = out + g * MX_GROUP;
    #pragma unroll
    for (unsigned int i = 0; i < MX_GROUP; ++i) {
        o[i] = __float2bfloat16(mx_e4m3_scaled(b[i], exp2));
    }
}
