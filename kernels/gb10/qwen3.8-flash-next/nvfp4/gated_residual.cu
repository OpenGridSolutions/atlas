// SPDX-License-Identifier: AGPL-3.0-only

// Gated residual streams for Qwen3.8-Flash-Next (model_type qwen4_exp).
//
// The model carries `hc` parallel residual streams per token, stored FP32 as
// [T, hc, H] (stream-major per token). Around every sublayer (GDN or
// attention, then MoE):
//   n      = group_rmsnorm(streams)              one RMS per stream, weight (1 + w)
//   a      = silu(Down(n) / hc)                  [R]   low-rank bottleneck
//   g      = sigmoid(Up(a))                      [hc, H]
//   mixed  = mean_i(g[i] * n[i])                 [H]   sublayer input
//   inj    = 2 * sigmoid(Inject(n) / hc)         [hc]
//   streams[i] += sublayer(mixed) * inj[i]
// The final mixer before the LM head has no Inject and stops at `mixed`.
//
// Formulas follow the published reference model (Qwen4ExpTextGatedResidual,
// Qwen4ExpTextRMSNorm with group_size in HF transformers).
//
// gr_up_mix holds the bottleneck activations in shared memory: R <= 512 (the model uses 320).

#include <cuda_bf16.h>

#define GR_BLOCK 256
#define GR_MAX_RANK 512

// Block-wide sum reduction over red[0..GR_BLOCK).
__device__ __forceinline__ float gr_block_reduce(float* red, unsigned int tid) {
    for (unsigned int s = GR_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

__device__ __forceinline__ float gr_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ── gr_expand ──
// streams[t, i, d] = hidden[t, d].  Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void gr_expand(
    const __nv_bfloat16* __restrict__ hidden, // [T, H]
    float* __restrict__ streams,              // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* x = hidden + (size_t)blockIdx.x * H;
    float* s = streams + (size_t)blockIdx.x * hc * H;
    for (unsigned int d = tid; d < H; d += GR_BLOCK) {
        const float v = (float)x[d];
        for (unsigned int i = 0; i < hc; ++i) s[i * H + d] = v;
    }
}

// ── gr_norm_inject ──
// normed[t, i, d] = streams[t, i, d] * rsqrt(mean_d(streams[t, i]^2) + eps) * (1 + w[i, d])
// inj[t, j]       = 2 * sigmoid(dot(inject_w[j], normed[t]) / hc)
// `inject_w` and `inj` may be null (final mixer).  Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void gr_norm_inject(
    const float* __restrict__ streams,          // [T, hc, H]
    const float* __restrict__ norm_w,           // [hc * H]
    const __nv_bfloat16* __restrict__ inject_w, // [hc, hc * H] or null
    float* __restrict__ normed,                 // [T, hc, H]
    float* __restrict__ inj,                    // [T, hc] or null
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    __shared__ float red[GR_BLOCK];
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const float* x = streams + (size_t)blockIdx.x * hc * H;
    float* n = normed + (size_t)blockIdx.x * hc * H;

    for (unsigned int i = 0; i < hc; ++i) {
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += GR_BLOCK) {
            const float v = x[i * H + d];
            acc += v * v;
        }
        red[tid] = acc;
        __syncthreads();
        const float rs = rsqrtf(gr_block_reduce(red, tid) / (float)H + eps);
        __syncthreads();
        for (unsigned int d = tid; d < H; d += GR_BLOCK) {
            n[i * H + d] = x[i * H + d] * rs * (1.0f + norm_w[i * H + d]);
        }
    }
    if (inject_w == nullptr) return;
    __syncthreads();

    const unsigned int W = hc * H;
    for (unsigned int j = 0; j < hc; ++j) {
        const __nv_bfloat16* w = inject_w + (size_t)j * W;
        float acc = 0.0f;
        for (unsigned int e = tid; e < W; e += GR_BLOCK) acc += (float)w[e] * n[e];
        red[tid] = acc;
        __syncthreads();
        const float dot = gr_block_reduce(red, tid);
        if (tid == 0) inj[(size_t)blockIdx.x * hc + j] = 2.0f * gr_sigmoid(dot / (float)hc);
        __syncthreads();
    }
}

// ── gr_down_silu ──
// act[t, r] = silu(dot(down_w[r], normed[t]) / hc).  Grid: (R,T,1)  Block: (256,1,1).
extern "C" __global__ void gr_down_silu(
    const float* __restrict__ normed,         // [T, hc, H]
    const __nv_bfloat16* __restrict__ down_w, // [R, hc * H]
    float* __restrict__ act,                  // [T, R]
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank
) {
    __shared__ float red[GR_BLOCK];
    const unsigned int tid = threadIdx.x;
    const unsigned int r = blockIdx.x;
    const unsigned int W = hc * hidden_size;
    const float* n = normed + (size_t)blockIdx.y * W;
    const __nv_bfloat16* w = down_w + (size_t)r * W;
    float acc = 0.0f;
    for (unsigned int e = tid; e < W; e += GR_BLOCK) acc += (float)w[e] * n[e];
    red[tid] = acc;
    __syncthreads();
    const float v = gr_block_reduce(red, tid) / (float)hc;
    if (tid == 0) act[(size_t)blockIdx.y * rank + r] = v * gr_sigmoid(v);
}

// ── gr_up_mix ──
// mixed[t, d] = (1 / hc) * sum_i sigmoid(dot(up_w[i * H + d], act[t])) * normed[t, i, d]
// Grid: (ceil(H / 256), T, 1)  Block: (256,1,1).
extern "C" __global__ void gr_up_mix(
    const float* __restrict__ normed,       // [T, hc, H]
    const float* __restrict__ act,          // [T, R]
    const __nv_bfloat16* __restrict__ up_w, // [hc * H, R]
    __nv_bfloat16* __restrict__ mixed,      // [T, H]
    const unsigned int hidden_size,
    const unsigned int hc,
    const unsigned int rank
) {
    __shared__ float a[GR_MAX_RANK];
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const float* at = act + (size_t)blockIdx.y * rank;
    for (unsigned int r = tid; r < rank; r += GR_BLOCK) a[r] = at[r];
    __syncthreads();

    const unsigned int d = blockIdx.x * GR_BLOCK + tid;
    if (d >= H) return;
    const float* n = normed + (size_t)blockIdx.y * hc * H;
    float sum = 0.0f;
    for (unsigned int i = 0; i < hc; ++i) {
        const __nv_bfloat16* w = up_w + ((size_t)i * H + d) * rank;
        float dot = 0.0f;
        for (unsigned int r = 0; r < rank; ++r) dot += (float)w[r] * a[r];
        sum += gr_sigmoid(dot) * n[i * H + d];
    }
    mixed[(size_t)blockIdx.y * H + d] = __float2bfloat16(sum / (float)hc);
}

// ── gr_post ──
// streams[t, i, d] += block_out[t, d] * inj[t, i].  Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void gr_post(
    const __nv_bfloat16* __restrict__ block_out, // [T, H]
    const float* __restrict__ inj,               // [T, hc]
    float* __restrict__ streams,                 // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const __nv_bfloat16* y = block_out + (size_t)blockIdx.x * H;
    const float* w = inj + (size_t)blockIdx.x * hc;
    float* s = streams + (size_t)blockIdx.x * hc * H;
    for (unsigned int d = tid; d < H; d += GR_BLOCK) {
        const float v = (float)y[d];
        for (unsigned int i = 0; i < hc; ++i) s[i * H + d] += v * w[i];
    }
}

// ── Prefill variant of the mixer ──
// For many tokens the two projections run as BF16 GEMMs over the whole chunk
// (the weights are read once), with these kernels in between:
//   gr_norm_bf16    normed = group_rmsnorm(streams), stored BF16
//   [GEMM]          d = normed . Down^T  [T, R];  l = normed . Inject^T  [T, hc]
//   gr_act          act = silu(d / hc) (BF16), inj = 2 * sigmoid(l / hc)
//   [GEMM]          u = act . Up^T  [T, hc * H]
//   gr_gate_mix     mixed = mean_i(sigmoid(u[i]) * normed[i])
// Values are rounded to BF16 where a GEMM consumes or produces them.

// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void gr_norm_bf16(
    const float* __restrict__ streams,  // [T, hc, H]
    const float* __restrict__ norm_w,   // [hc * H]
    __nv_bfloat16* __restrict__ normed, // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    __shared__ float red[GR_BLOCK];
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const float* x = streams + (size_t)blockIdx.x * hc * H;
    __nv_bfloat16* n = normed + (size_t)blockIdx.x * hc * H;
    for (unsigned int i = 0; i < hc; ++i) {
        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += GR_BLOCK) {
            const float v = x[i * H + d];
            acc += v * v;
        }
        red[tid] = acc;
        __syncthreads();
        const float rs = rsqrtf(gr_block_reduce(red, tid) / (float)H + eps);
        __syncthreads();
        for (unsigned int d = tid; d < H; d += GR_BLOCK) {
            n[i * H + d] = __float2bfloat16(x[i * H + d] * rs * (1.0f + norm_w[i * H + d]));
        }
    }
}

// act[t, r] = silu(d[t, r] / hc) in place; inj[t, i] = 2 * sigmoid(l[t, i] / hc).
// `logits` and `inj` may be null (final mixer).
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void gr_act(
    __nv_bfloat16* __restrict__ d,            // [T, R] in/out
    const __nv_bfloat16* __restrict__ logits, // [T, hc] or null
    float* __restrict__ inj,                  // [T, hc] or null
    const unsigned int rank,
    const unsigned int hc
) {
    __nv_bfloat16* row = d + (size_t)blockIdx.x * rank;
    for (unsigned int r = threadIdx.x; r < rank; r += GR_BLOCK) {
        const float v = __bfloat162float(row[r]) / (float)hc;
        row[r] = __float2bfloat16(v * gr_sigmoid(v));
    }
    if (logits != nullptr && threadIdx.x < hc) {
        const size_t o = (size_t)blockIdx.x * hc + threadIdx.x;
        inj[o] = 2.0f * gr_sigmoid(__bfloat162float(logits[o]) / (float)hc);
    }
}

// Grid: (ceil(H / 256), T, 1)  Block: (256,1,1).
extern "C" __global__ void gr_gate_mix(
    const __nv_bfloat16* __restrict__ u,      // [T, hc, H] gate logits
    const __nv_bfloat16* __restrict__ normed, // [T, hc, H]
    __nv_bfloat16* __restrict__ mixed,        // [T, H]
    const unsigned int hidden_size,
    const unsigned int hc
) {
    const unsigned int H = hidden_size;
    const unsigned int d = blockIdx.x * GR_BLOCK + threadIdx.x;
    if (d >= H) return;
    const size_t base = (size_t)blockIdx.y * hc * H;
    float sum = 0.0f;
    for (unsigned int i = 0; i < hc; ++i) {
        const size_t e = base + (size_t)i * H + d;
        sum += gr_sigmoid(__bfloat162float(u[e])) * __bfloat162float(normed[e]);
    }
    mixed[(size_t)blockIdx.y * H + d] = __float2bfloat16(sum / (float)hc);
}

// ── gr_passthrough ──
// output[t, d] = input[t, d]. The model has no final norm: the last layer's
// mixer already produced the LM-head input. This kernel takes the place of
// the final RMS norm and has its launch signature.
// Grid: (T,1,1)  Block: (min(H, 1024),1,1).
extern "C" __global__ void gr_passthrough(
    const __nv_bfloat16* __restrict__ input,  // [T, H]
    const __nv_bfloat16* __restrict__ weight, // unused
    __nv_bfloat16* __restrict__ output,       // [T, H]
    const unsigned int hidden_size,
    const float eps                           // unused
) {
    const __nv_bfloat16* x = input + (size_t)blockIdx.x * hidden_size;
    __nv_bfloat16* y = output + (size_t)blockIdx.x * hidden_size;
    for (unsigned int d = threadIdx.x; d < hidden_size; d += blockDim.x) y[d] = x[d];
}
