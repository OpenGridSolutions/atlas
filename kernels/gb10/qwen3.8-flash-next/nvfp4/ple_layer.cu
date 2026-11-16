// SPDX-License-Identifier: AGPL-3.0-only

// N-gram embedding layer (PLE) of Qwen3.8-Flash-Next (model_type qwen4_exp).
//
// At one decoder layer the model adds hashed n-gram features to every
// residual stream. Per token:
//   e       = 16 table rows of 160 values, concatenated            [E = 2560]
//   key     = key_proj(e)   [hc, H]      value = value_proj(e)     [H]
//   k_n     = group_rmsnorm(key),  q_n = group_rmsnorm(streams)    weight (1 + w)
//   d_s     = sum(k_n[s] * q_n[s]) / sqrt(H)
//   g_s     = sigmoid(sign(d_s) * sqrt(max(|d_s|, 1e-6)))
//   gated_s = g_s * value
//   c_s     = group_rmsnorm(gated)_s                               weight (1 + w)
//   streams_s += gated_s + silu(conv(c)_s)
// conv is depthwise over the token axis, kernel K, dilation D; tap K-1 is the
// current token. Its history (the last (K-1)*D values of c per channel,
// oldest first) is per-sequence state and starts at zero.
//
// key_proj / value_proj run on the BF16 GEMV/GEMM kernels; this file holds
// the table-row dequantisation, the gate and the convolution.
// Formulas follow the published reference model (Qwen4ExpTextPLELayer).

#include <cuda_bf16.h>

#define PLE_BLOCK 256

__device__ __forceinline__ float ple_block_reduce(float* red, unsigned int tid) {
    for (unsigned int s = PLE_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    return red[0];
}

// FP8 E4M3 (sign 1, exponent 4 bias 7, mantissa 3); NaN codes read as zero.
__device__ __forceinline__ float ple_e4m3(unsigned int b) {
    const unsigned int e = (b >> 3) & 0xF;
    const unsigned int m = b & 0x7;
    const int mant = ((b & 0x7F) == 0x7F) ? 0 : (int)((e ? 8u : 0u) + m);
    const float v = scalbnf((float)mant, (int)(e ? e : 1u) - 10);
    return (b & 0x80) ? -v : v;
}

// FP4 E2M1: magnitudes 0, 0.5, 1, 1.5, 2, 3, 4, 6; bit 3 is the sign.
__device__ __forceinline__ float ple_e2m1(unsigned int n) {
    const unsigned int e = (n >> 1) & 0x3;
    const unsigned int m = n & 0x1;
    const float v = e ? scalbnf((float)(2u + m), (int)e - 2) : 0.5f * (float)m;
    return (n & 0x8) ? -v : v;
}

// ── ple_decode_rows ──
// NVFP4 table rows -> BF16 embedding. Per token `heads` rows; a row is
// head_dim / 2 bytes of nibbles (low nibble = even element) and head_dim / 16
// E4M3 scale bytes:  v[j] = e2m1(nibble_j) * e4m3(scale[j / 16]) * scale2
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void ple_decode_rows(
    const unsigned char* __restrict__ rows,   // [T, heads, head_dim / 2]
    const unsigned char* __restrict__ scales, // [T, heads, head_dim / 16]
    __nv_bfloat16* __restrict__ emb,          // [T, heads * head_dim]
    const unsigned int heads,
    const unsigned int head_dim,
    const float scale2
) {
    const unsigned int E = heads * head_dim;
    const unsigned char* r = rows + (size_t)blockIdx.x * (E / 2);
    const unsigned char* s = scales + (size_t)blockIdx.x * (E / 16);
    __nv_bfloat16* o = emb + (size_t)blockIdx.x * E;
    // head_dim is a multiple of 16, so element j of the token uses byte j / 2
    // and scale j / 16 of the concatenated rows.
    for (unsigned int j = threadIdx.x; j < E; j += PLE_BLOCK) {
        const unsigned int nib = (r[j / 2] >> (4 * (j & 1))) & 0xF;
        o[j] = __float2bfloat16(ple_e2m1(nib) * ple_e4m3(s[j / 16]) * scale2);
    }
}

// ── ple_gate ──
// gated[t, s, d]   = g[t, s] * value[t, d]
// conv_in[t, s, d] = gated[t, s, d] * rsqrt(mean_d(gated[t, s]^2) + eps) * (1 + w_conv[s, d])
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void ple_gate(
    const __nv_bfloat16* __restrict__ key,   // [T, hc, H]
    const __nv_bfloat16* __restrict__ value, // [T, H]
    const float* __restrict__ streams,       // [T, hc, H]
    const float* __restrict__ w_key,         // [hc * H]
    const float* __restrict__ w_query,       // [hc * H]
    const float* __restrict__ w_conv,        // [hc * H]
    float* __restrict__ gated,               // [T, hc, H]
    float* __restrict__ conv_in,             // [T, hc, H]
    const unsigned int hidden_size,
    const unsigned int hc,
    const float eps
) {
    __shared__ float red[PLE_BLOCK];
    const unsigned int tid = threadIdx.x;
    const unsigned int H = hidden_size;
    const size_t t = blockIdx.x;
    const __nv_bfloat16* v = value + t * H;

    for (unsigned int s = 0; s < hc; ++s) {
        const __nv_bfloat16* k = key + (t * hc + s) * H;
        const float* x = streams + (t * hc + s) * H;
        const float* wk = w_key + (size_t)s * H;
        const float* wq = w_query + (size_t)s * H;
        const float* wc = w_conv + (size_t)s * H;

        float acc = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            const float a = __bfloat162float(k[d]);
            acc += a * a;
        }
        red[tid] = acc;
        __syncthreads();
        const float rs_k = rsqrtf(ple_block_reduce(red, tid) / (float)H + eps);
        __syncthreads();

        acc = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) acc += x[d] * x[d];
        red[tid] = acc;
        __syncthreads();
        const float rs_q = rsqrtf(ple_block_reduce(red, tid) / (float)H + eps);
        __syncthreads();

        acc = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            acc += (__bfloat162float(k[d]) * rs_k * (1.0f + wk[d])) * (x[d] * rs_q * (1.0f + wq[d]));
        }
        red[tid] = acc;
        __syncthreads();
        const float dot = ple_block_reduce(red, tid) / sqrtf((float)H);
        __syncthreads();
        // sign(0) is 0 in the reference, which makes the gate exactly 0.5.
        const float root = sqrtf(fmaxf(fabsf(dot), 1e-6f));
        const float arg = dot > 0.0f ? root : (dot < 0.0f ? -root : 0.0f);
        const float g = 1.0f / (1.0f + expf(-arg));

        acc = 0.0f;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            const float gv = g * __bfloat162float(v[d]);
            acc += gv * gv;
        }
        red[tid] = acc;
        __syncthreads();
        const float rs_c = rsqrtf(ple_block_reduce(red, tid) / (float)H + eps);
        __syncthreads();

        float* go = gated + (t * hc + s) * H;
        float* co = conv_in + (t * hc + s) * H;
        for (unsigned int d = tid; d < H; d += PLE_BLOCK) {
            const float gv = g * __bfloat162float(v[d]);
            go[d] = gv;
            co[d] = gv * rs_c * (1.0f + wc[d]);
        }
    }
}

// ── ple_conv_add ──
// streams[q, t, c] += gated[q, t, c] + silu(sum_k w[c, k] * hist[t + D * k]),
// hist = state of the sequence (state_len values, oldest first) followed by
// this call's conv_in. The arrays hold `num_seqs` sequences of
// `tokens_per_seq` tokens each ([q, t, C]); sequence q uses the state at
// state_base + slots[q] * state_stride (bytes), laid out [C, state_len].
// positions[q] is the absolute position of the sequence's first token in
// this call: state entries that lie before position 0 read as zero, so a
// slot never has to be cleared when a new sequence takes it over.
// The state is only read here; ple_state_update advances it afterwards.
// Grid: (ceil(C / 256), num_seqs * tokens_per_seq, 1)  Block: (256,1,1).
extern "C" __global__ void ple_conv_add(
    const float* __restrict__ gated,         // [num_seqs, T, C]
    const float* __restrict__ conv_in,       // [num_seqs, T, C]
    const __nv_bfloat16* __restrict__ w,     // [C, taps]
    const float* __restrict__ state_base,
    const unsigned int* __restrict__ slots,  // [num_seqs]
    const unsigned int* __restrict__ positions, // [num_seqs]
    float* __restrict__ streams,             // [num_seqs, T, C]
    const unsigned long long state_stride,
    const unsigned int channels,
    const unsigned int tokens_per_seq,
    const unsigned int taps,
    const unsigned int dilation
) {
    const unsigned int c = blockIdx.x * PLE_BLOCK + threadIdx.x;
    if (c >= channels) return;
    const unsigned int q = blockIdx.y / tokens_per_seq;
    const unsigned int t = blockIdx.y % tokens_per_seq;
    const unsigned int state_len = (taps - 1) * dilation;
    const float* st =
        (const float*)((const char*)state_base + (size_t)slots[q] * state_stride) + (size_t)c * state_len;
    const float* seq_in = conv_in + (size_t)q * tokens_per_seq * channels;
    const unsigned int first_pos = positions[q];
    float y = 0.0f;
    for (unsigned int k = 0; k < taps; ++k) {
        const unsigned int j = t + dilation * k;
        float h;
        if (j >= state_len) {
            h = seq_in[(size_t)(j - state_len) * channels + c];
        } else {
            // state entry j holds absolute position first_pos - state_len + j
            h = (first_pos + j >= state_len) ? st[j] : 0.0f;
        }
        y += __bfloat162float(w[(size_t)c * taps + k]) * h;
    }
    const size_t o = (size_t)blockIdx.y * channels + c;
    streams[o] += gated[o] + y / (1.0f + expf(-y));
}

// ── ple_state_update ──
// state <- the last state_len values of [state ++ conv_in] per channel.
// Grid: (ceil(C / 256), num_seqs, 1)  Block: (256,1,1).
extern "C" __global__ void ple_state_update(
    const float* __restrict__ conv_in,      // [num_seqs, T, C]
    float* __restrict__ state_base,
    const unsigned int* __restrict__ slots, // [num_seqs]
    const unsigned long long state_stride,
    const unsigned int channels,
    const unsigned int tokens_per_seq,
    const unsigned int state_len
) {
    const unsigned int c = blockIdx.x * PLE_BLOCK + threadIdx.x;
    if (c >= channels) return;
    const unsigned int q = blockIdx.y;
    const unsigned int T = tokens_per_seq;
    float* st = (float*)((char*)state_base + (size_t)slots[q] * state_stride) + (size_t)c * state_len;
    const float* seq_in = conv_in + (size_t)q * T * channels;
    // Ascending j reads st[j + T] before any write reaches it.
    for (unsigned int j = 0; j < state_len; ++j) {
        const unsigned int src = j + T;
        st[j] = src < state_len ? st[src] : seq_in[(size_t)(src - state_len) * channels + c];
    }
}
