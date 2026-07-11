// SPDX-License-Identifier: AGPL-3.0-only

// Whisper support kernels (multiclass Plan.md W.2). Everything the
// encoder-decoder needs beyond the common GEMM set, deliberately simple:
// whisper-large-v3 shapes are tiny by Atlas standards (seq <= 1500,
// hidden 1280, head_dim 64), so naive-but-correct beats clever here.
//
//   whisper_layernorm  — LayerNorm WITH bias (Whisper is LN, not RMSNorm)
//   whisper_bias_add   — broadcast row-bias add (post-GEMM)
//   whisper_bias_gelu  — fused bias add + tanh-approx GELU (conv/MLP)
//   whisper_attention  — softmax(QK^T * scale) V over dense BF16 buffers,
//                        runtime `causal` flag, one block per (query, head),
//                        two-pass softmax via a scores[] smem buffer
//                        (nk <= 1500 -> 6 KB). Serves encoder self-attn
//                        (non-causal), decoder self-attn (causal) and
//                        cross-attn (non-causal, static K/V) alike.

#include <cuda_bf16.h>

extern "C" __global__ void whisper_layernorm(
    const __nv_bfloat16* __restrict__ input,  // [tokens, hidden]
    const __nv_bfloat16* __restrict__ gamma,  // [hidden]
    const __nv_bfloat16* __restrict__ beta,   // [hidden]
    __nv_bfloat16* __restrict__ output,       // [tokens, hidden]
    unsigned int hidden,
    float eps
) {
    const unsigned int t = blockIdx.x;
    const __nv_bfloat16* x = input + (size_t)t * hidden;
    __nv_bfloat16* out = output + (size_t)t * hidden;

    __shared__ float s_sum[256];
    __shared__ float s_mean, s_rstd;
    float acc = 0.f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x)
        acc += __bfloat162float(x[i]);
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_mean = s_sum[0] / hidden;
    __syncthreads();
    const float mean = s_mean;

    acc = 0.f;
    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float d = __bfloat162float(x[i]) - mean;
        acc += d * d;
    }
    s_sum[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_sum[threadIdx.x] += s_sum[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_rstd = rsqrtf(s_sum[0] / hidden + eps);
    __syncthreads();
    const float rstd = s_rstd;

    for (unsigned int i = threadIdx.x; i < hidden; i += blockDim.x) {
        const float v = (__bfloat162float(x[i]) - mean) * rstd;
        out[i] = __float2bfloat16(v * __bfloat162float(gamma[i]) + __bfloat162float(beta[i]));
    }
}

extern "C" __global__ void whisper_bias_add(
    __nv_bfloat16* __restrict__ data,       // [tokens, n] in place
    const __nv_bfloat16* __restrict__ bias, // [n]
    unsigned int n
) {
    const unsigned int t = blockIdx.x;
    __nv_bfloat16* row = data + (size_t)t * n;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x)
        row[i] = __float2bfloat16(__bfloat162float(row[i]) + __bfloat162float(bias[i]));
}

__device__ __forceinline__ float gelu_tanh(float x) {
    // HF/PyTorch GELU (exact erf form matters little for Whisper; use the
    // tanh approximation both HF `gelu_new`-family and this module accept).
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * x * (1.0f + tanhf(c * (x + 0.044715f * x * x * x)));
}

extern "C" __global__ void whisper_bias_gelu(
    __nv_bfloat16* __restrict__ data,       // [tokens, n] in place
    const __nv_bfloat16* __restrict__ bias, // [n]
    unsigned int n
) {
    const unsigned int t = blockIdx.x;
    __nv_bfloat16* row = data + (size_t)t * n;
    for (unsigned int i = threadIdx.x; i < n; i += blockDim.x) {
        const float v = __bfloat162float(row[i]) + __bfloat162float(bias[i]);
        row[i] = __float2bfloat16(gelu_tanh(v));
    }
}

// softmax(Q K^T * scale) V — dense BF16 buffers, head-interleaved
// [tokens, heads, head_dim] layout. Grid (nq, heads), block 256.
// `causal`: query q may attend keys j <= (nk - nq + q) — correct for a
// growing decoder self-attn cache where the last nq queries align with
// the last nq keys. Non-causal (0) for encoder self-attn and cross-attn.
#define WA_MAX_KEYS 1504
extern "C" __global__ void whisper_attention(
    const __nv_bfloat16* __restrict__ Q, // [nq, H, D]
    const __nv_bfloat16* __restrict__ K, // [nk, H, D]
    const __nv_bfloat16* __restrict__ V, // [nk, H, D]
    __nv_bfloat16* __restrict__ O,       // [nq, H, D]
    unsigned int nq,
    unsigned int nk,
    unsigned int heads,
    unsigned int head_dim,
    float scale,
    unsigned int causal
) {
    const unsigned int q = blockIdx.x;
    const unsigned int h = blockIdx.y;
    if (q >= nq || h >= heads) return;

    __shared__ float scores[WA_MAX_KEYS];
    __shared__ float s_red[256];
    __shared__ float s_max, s_sum;

    const unsigned int limit = causal ? (nk - nq + q + 1) : nk;
    const __nv_bfloat16* qv = Q + ((size_t)q * heads + h) * head_dim;

    // Pass 1: scores[j] = dot(q, k_j) * scale.
    for (unsigned int j = threadIdx.x; j < limit; j += blockDim.x) {
        const __nv_bfloat16* kv = K + ((size_t)j * heads + h) * head_dim;
        float dot = 0.f;
        for (unsigned int d = 0; d < head_dim; ++d)
            dot += __bfloat162float(qv[d]) * __bfloat162float(kv[d]);
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Block max.
    float m = -1e30f;
    for (unsigned int j = threadIdx.x; j < limit; j += blockDim.x)
        m = fmaxf(m, scores[j]);
    s_red[threadIdx.x] = m;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_red[threadIdx.x] = fmaxf(s_red[threadIdx.x], s_red[threadIdx.x + s]);
        __syncthreads();
    }
    if (threadIdx.x == 0) s_max = s_red[0];
    __syncthreads();
    const float mx = s_max;

    // exp + block sum.
    float acc = 0.f;
    for (unsigned int j = threadIdx.x; j < limit; j += blockDim.x) {
        const float e = __expf(scores[j] - mx);
        scores[j] = e;
        acc += e;
    }
    s_red[threadIdx.x] = acc;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) s_red[threadIdx.x] += s_red[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_sum = s_red[0];
    __syncthreads();
    const float inv = 1.0f / s_sum;

    // O[d] = sum_j p_j * V[j][d] — threads own output dims.
    __nv_bfloat16* ov = O + ((size_t)q * heads + h) * head_dim;
    for (unsigned int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float o = 0.f;
        for (unsigned int j = 0; j < limit; ++j)
            o += scores[j] * __bfloat162float(V[((size_t)j * heads + h) * head_dim + d]);
        ov[d] = __float2bfloat16(o * inv);
    }
}
