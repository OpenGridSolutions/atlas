// SPDX-License-Identifier: AGPL-3.0-only

// Atlas Spark — Qwen-Image-2.1 sidecar kernels (Todo 3.2c, module "qwen_image").
//
// Three consumers share this file:
//   * the Qwen3-VL text encoder (dense causal LLM, hidden states only),
//   * the single-stream DiT (block-causal prefix + full-attention target),
//   * the VAE decoder (NHWC activations: one row per pixel, C columns).
//
// All activations are BF16 in global memory with f32 arithmetic in-kernel.
// Projections/convolutions run on the common GEMMs (cuBLASLt or
// dense_gemm_bf16_pipelined); attention is GEMM-SDPA per head
// (QKᵀ f32 → qimg_softmax_rows → P·V), the same shape as the Qwen3-VL ViT
// path but rectangular (nq × nk) with a per-row visible-key limit, so one
// kernel serves causal text, block-causal prefix and full target rows.

#include <cuda_bf16.h>
#include <math.h>

__device__ inline float bf16_to_f32(__nv_bfloat16 v) {
    return __bfloat162float(v);
}
__device__ inline __nv_bfloat16 f32_to_bf16(float v) {
    return __float2bfloat16(v);
}

// Block-wide sum of a per-thread partial. `red` holds one slot per warp.
__device__ inline float block_sum(float v, float* red) {
    unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
    if (lane == 0) red[warp] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (unsigned int w = 0; w < (blockDim.x + 31) / 32; ++w) s += red[w];
        red[0] = s;
    }
    __syncthreads();
    float total = red[0];
    __syncthreads();
    return total;
}

// ── Norms ────────────────────────────────────────────────────────────────────

// Affine-free LayerNorm with optional AdaLN scale: out = LN(x) * (1 + scale).
// `scale` may be null (plain LayerNorm).
// grid = (tokens,1,1), block = (256,1,1)
extern "C" __global__
void qimg_layernorm_mod(
    const __nv_bfloat16* __restrict__ in,   // [tokens, hidden]
    const float* __restrict__ scale,        // [hidden] or null
    __nv_bfloat16* __restrict__ out,        // [tokens, hidden]
    unsigned int hidden, float eps)
{
    __shared__ float red[8];
    const __nv_bfloat16* x = in + (size_t)blockIdx.x * hidden;
    __nv_bfloat16* y = out + (size_t)blockIdx.x * hidden;

    float s = 0.0f;
    for (unsigned int j = threadIdx.x; j < hidden; j += blockDim.x) s += bf16_to_f32(x[j]);
    float mean = block_sum(s, red) / (float)hidden;

    float v = 0.0f;
    for (unsigned int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float d = bf16_to_f32(x[j]) - mean;
        v += d * d;
    }
    float inv = rsqrtf(block_sum(v, red) / (float)hidden + eps);

    for (unsigned int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float r = (bf16_to_f32(x[j]) - mean) * inv;
        if (scale) r *= 1.0f + scale[j];
        y[j] = f32_to_bf16(r);
    }
}

// RMSNorm: out = x * rsqrt(mean(x²) + eps) * (w + w_offset).
// w_offset = 1 for the zero-centred DiT text norm, 0 for the LLM norms.
// grid = (tokens,1,1), block = (256,1,1)
extern "C" __global__
void qimg_rms_norm(
    const __nv_bfloat16* __restrict__ in,   // [tokens, hidden]
    const __nv_bfloat16* __restrict__ w,    // [hidden]
    __nv_bfloat16* __restrict__ out,        // [tokens, hidden]
    unsigned int hidden, float eps, float w_offset)
{
    __shared__ float red[8];
    const __nv_bfloat16* x = in + (size_t)blockIdx.x * hidden;
    __nv_bfloat16* y = out + (size_t)blockIdx.x * hidden;

    float s = 0.0f;
    for (unsigned int j = threadIdx.x; j < hidden; j += blockDim.x) {
        float f = bf16_to_f32(x[j]);
        s += f * f;
    }
    float inv = rsqrtf(block_sum(s, red) / (float)hidden + eps);
    for (unsigned int j = threadIdx.x; j < hidden; j += blockDim.x)
        y[j] = f32_to_bf16(bf16_to_f32(x[j]) * inv * (bf16_to_f32(w[j]) + w_offset));
}

// ── Attention pre/post-processing ────────────────────────────────────────────

// Per-head RMSNorm + RoPE, written head-major for the per-head GEMMs:
//   out[h][row_off + tok][d],  out head stride = out_rows * D.
// X is one projection output [tokens, H*D]. cos/sin are [tokens, D/2].
// rope_mode 0: adjacent pairs (2i, 2i+1) rotate as one complex number (DiT);
// rope_mode 1: rotate-half pairs (i, i + D/2) (Qwen LLM).
// grid = (tokens, H, 1), block = (D,1,1)  (D ≤ 256, multiple of 32)
extern "C" __global__
void qimg_norm_rope_heads(
    const __nv_bfloat16* __restrict__ X,    // [tokens, H*D]
    const __nv_bfloat16* __restrict__ w,    // [D] RMSNorm weight
    const float* __restrict__ cosv,         // [tokens, D/2]
    const float* __restrict__ sinv,         // [tokens, D/2]
    __nv_bfloat16* __restrict__ out,        // [H][out_rows][D]
    unsigned int H, unsigned int D,
    unsigned int row_off, unsigned int out_rows,
    float eps, unsigned int rope_mode)
{
    __shared__ float red[8];
    unsigned int tok = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const __nv_bfloat16* x = X + (size_t)tok * H * D + (size_t)h * D;

    float f = (d < D) ? bf16_to_f32(x[d]) : 0.0f;
    float inv = rsqrtf(block_sum(f * f, red) / (float)D + eps);
    if (d >= D) return;

    unsigned int half = D / 2;
    unsigned int partner, fi;
    float sign;   // out = a*cos + sign * b*sin, a = own, b = partner
    if (rope_mode == 0) {
        partner = d ^ 1u;
        fi = d >> 1;
        sign = (d & 1u) ? 1.0f : -1.0f;
    } else {
        partner = (d < half) ? d + half : d - half;
        fi = (d < half) ? d : d - half;
        sign = (d < half) ? -1.0f : 1.0f;
    }
    float a = f * inv * bf16_to_f32(w[d]);
    float b = bf16_to_f32(x[partner]) * inv * bf16_to_f32(w[partner]);
    float c = cosv[(size_t)tok * half + fi];
    float s = sinv[(size_t)tok * half + fi];
    out[((size_t)h * out_rows + row_off + tok) * D + d] = f32_to_bf16(a * c + sign * b * s);
}

// V projection [tokens, H*D] → transposed head-major Vt[h][d][col_off + tok]
// (head stride D * out_cols), the layout GEMM2 (P · V) reads as B[N=D, K=keys].
// grid = (ceil(tokens*H*D/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_v_transpose(
    const __nv_bfloat16* __restrict__ V,    // [tokens, H*D]
    __nv_bfloat16* __restrict__ Vt,         // [H][D][out_cols]
    unsigned int tokens, unsigned int H, unsigned int D,
    unsigned int col_off, unsigned int out_cols)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t hd = (size_t)H * D;
    if (lin >= (size_t)tokens * hd) return;
    unsigned int tok = (unsigned int)(lin / hd);
    unsigned int r = (unsigned int)(lin % hd);
    unsigned int h = r / D, d = r % D;
    Vt[((size_t)h * D + d) * out_cols + col_off + tok] = V[lin];
}

// Copy a cached prefix (K rows / V columns) into the per-layer key buffers.
//   Kc[h][src_rows][D] → Kh[h][dst_rows][D]   (first n rows of every head)
//   Vc[h][D][src_rows] → Vt[h][D][dst_rows]   (first n columns of every row)
// grid = (ceil(H*n*D/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_copy_prefix(
    const __nv_bfloat16* __restrict__ Kc, const __nv_bfloat16* __restrict__ Vc,
    __nv_bfloat16* __restrict__ Kh, __nv_bfloat16* __restrict__ Vt,
    unsigned int n, unsigned int src_rows, unsigned int dst_rows,
    unsigned int H, unsigned int D)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t per_head = (size_t)n * D;
    if (lin >= per_head * H) return;
    unsigned int h = (unsigned int)(lin / per_head);
    unsigned int r = (unsigned int)(lin % per_head);
    unsigned int tok = r / D, d = r % D;
    Kh[((size_t)h * dst_rows + tok) * D + d] = Kc[((size_t)h * src_rows + tok) * D + d];
    Vt[((size_t)h * D + d) * dst_rows + tok] = Vc[((size_t)h * D + d) * src_rows + tok];
}

// Row softmax over raw scores S[nq, nk] (f32) → P[nq, nk] (bf16), `scale`
// folded in. Row r sees keys [0, limit) with limit = limits ? limits[r] :
// nk_valid; every other column gets probability 0 (masked or padding).
// grid = (nq,1,1), block = (256,1,1)
extern "C" __global__
void qimg_softmax_rows(
    const float* __restrict__ S,            // [nq, nk]
    __nv_bfloat16* __restrict__ P,          // [nq, nk]
    const unsigned int* __restrict__ limits,// [nq] or null
    unsigned int nq, unsigned int nk, unsigned int nk_valid, float scale)
{
    unsigned int row = blockIdx.x;
    if (row >= nq) return;
    const float* srow = S + (size_t)row * nk;
    __nv_bfloat16* prow = P + (size_t)row * nk;
    unsigned int lim = limits ? limits[row] : nk_valid;
    if (lim > nk) lim = nk;

    __shared__ float red[8];
    unsigned int tid = threadIdx.x, lane = tid & 31u, warp = tid >> 5;

    float m = -1e30f;
    for (unsigned int j = tid; j < lim; j += blockDim.x) m = fmaxf(m, srow[j] * scale);
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_down_sync(0xffffffff, m, o));
    if (lane == 0) red[warp] = m;
    __syncthreads();
    if (tid == 0) {
        float mm = -1e30f;
        for (unsigned int w = 0; w < blockDim.x / 32; ++w) mm = fmaxf(mm, red[w]);
        red[0] = mm;
    }
    __syncthreads();
    float row_max = red[0];
    __syncthreads();

    float s = 0.0f;
    for (unsigned int j = tid; j < lim; j += blockDim.x) s += expf(srow[j] * scale - row_max);
    float total = block_sum(s, red);
    float inv_sum = (total > 0.0f) ? (1.0f / total) : 0.0f;

    for (unsigned int j = tid; j < nk; j += blockDim.x)
        prow[j] = f32_to_bf16(j < lim ? expf(srow[j] * scale - row_max) * inv_sum : 0.0f);
}

// Scatter one head's contiguous Oh[tokens, D] into O[tokens, dst_stride].
// `O` is already offset to the head slot.
// grid = (ceil(tokens*D/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_scatter_head(
    const __nv_bfloat16* __restrict__ Oh, __nv_bfloat16* __restrict__ O,
    unsigned int tokens, unsigned int D, unsigned int dst_stride)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= (size_t)tokens * D) return;
    size_t tok = lin / D, d = lin % D;
    O[tok * dst_stride + d] = Oh[lin];
}

// ── Elementwise ──────────────────────────────────────────────────────────────

// h += gate[col] * x  (gate null → plain add). Shapes [tokens, hidden].
// grid = (ceil(tokens*hidden/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_gated_add(
    __nv_bfloat16* __restrict__ h, const __nv_bfloat16* __restrict__ x,
    const float* __restrict__ gate, unsigned long long n, unsigned int hidden)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= n) return;
    float g = gate ? gate[lin % hidden] : 1.0f;
    h[lin] = f32_to_bf16(bf16_to_f32(h[lin]) + g * bf16_to_f32(x[lin]));
}

// out = silu(gate) * up  (elementwise; in-place on `gate` is allowed)
// grid = (ceil(n/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_silu_mul(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* out, unsigned long long n)   // out may alias gate
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= n) return;
    float g = bf16_to_f32(gate[lin]);
    out[lin] = f32_to_bf16(g / (1.0f + expf(-g)) * bf16_to_f32(up[lin]));
}

// In-place tanh-approximated GELU.
// grid = (ceil(n/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_gelu_tanh(__nv_bfloat16* __restrict__ x, unsigned long long n)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= n) return;
    float v = bf16_to_f32(x[lin]);
    float t = tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v));
    x[lin] = f32_to_bf16(0.5f * v * (1.0f + t));
}

// x[row, col] += bias[col]  (f32 bias, bf16 data)
// grid = (ceil(rows*cols/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_bias_add(
    __nv_bfloat16* __restrict__ x, const float* __restrict__ bias,
    unsigned long long n, unsigned int cols)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= n) return;
    x[lin] = f32_to_bf16(bf16_to_f32(x[lin]) + bias[lin % cols]);
}

// ── VAE decoder (NHWC) ───────────────────────────────────────────────────────

// im2col for a 3×3 / stride 1 / zero-pad 1 convolution over image rows
// [row0, row0 + rows). Output row = pixel, columns ordered (ky, kx, c) to
// match the weight rearranged as [Cout, 3, 3, Cin] at load.
// grid = (ceil(rows*W*9*C/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_im2col3x3(
    const __nv_bfloat16* __restrict__ in,   // [H*W, C]
    __nv_bfloat16* __restrict__ out,        // [rows*W, 9*C]
    unsigned int H, unsigned int W, unsigned int C,
    unsigned int row0, unsigned int rows)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t k9 = (size_t)9 * C;
    if (lin >= (size_t)rows * W * k9) return;
    size_t pix = lin / k9;
    unsigned int col = (unsigned int)(lin % k9);
    unsigned int tap = col / C, c = col % C;
    int y = (int)(row0 + pix / W) + (int)(tap / 3) - 1;
    int x = (int)(pix % W) + (int)(tap % 3) - 1;
    __nv_bfloat16 v = f32_to_bf16(0.0f);
    if (y >= 0 && y < (int)H && x >= 0 && x < (int)W)
        v = in[((size_t)y * W + x) * C + c];
    out[lin] = v;
}

// Channel RMS norm (F.normalize over C, eps 1e-12) * sqrt(C) * gamma, with
// optional SiLU.
// grid = (pixels,1,1), block = (min(256, ceil32(C)),1,1)
extern "C" __global__
void qimg_vae_norm(
    const __nv_bfloat16* in,                // [pixels, C]
    const float* __restrict__ gamma,        // [C]
    __nv_bfloat16* out,                     // [pixels, C]  (may alias `in`)
    unsigned int C, unsigned int apply_silu)
{
    __shared__ float red[8];
    const __nv_bfloat16* x = in + (size_t)blockIdx.x * C;
    __nv_bfloat16* y = out + (size_t)blockIdx.x * C;

    float s = 0.0f;
    for (unsigned int j = threadIdx.x; j < C; j += blockDim.x) {
        float f = bf16_to_f32(x[j]);
        s += f * f;
    }
    float norm = fmaxf(sqrtf(block_sum(s, red)), 1e-12f);
    float k = sqrtf((float)C) / norm;
    for (unsigned int j = threadIdx.x; j < C; j += blockDim.x) {
        float r = bf16_to_f32(x[j]) * k * gamma[j];
        if (apply_silu) r = r / (1.0f + expf(-r));
        y[j] = f32_to_bf16(r);
    }
}

// Nearest-neighbour 2× upsample, NHWC.
// grid = (ceil(4*H*W*C/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_upsample2x(
    const __nv_bfloat16* __restrict__ in,   // [H*W, C]
    __nv_bfloat16* __restrict__ out,        // [2H*2W, C]
    unsigned int H, unsigned int W, unsigned int C)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= (size_t)4 * H * W * C) return;
    size_t pix = lin / C;
    unsigned int c = (unsigned int)(lin % C);
    size_t y = pix / (2 * W), x = pix % (2 * W);
    out[lin] = in[((y / 2) * W + (x / 2)) * C + c];
}

// Residual up-block shortcut (DupUp3D, single frame): the 2×-upsampled
// output pixel (2h+i, 2w+j), channel c adds source channel
//   (c * factor + off + 2*i + j) / repeats   of source pixel (h, w).
// grid = (ceil(4*H*W*Cout/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_dup_shortcut_add(
    __nv_bfloat16* __restrict__ x,          // [2H*2W, Cout]  (in place)
    const __nv_bfloat16* __restrict__ src,  // [H*W, Cin]
    unsigned int H, unsigned int W, unsigned int Cin, unsigned int Cout,
    unsigned int factor, unsigned int off, unsigned int repeats)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= (size_t)4 * H * W * Cout) return;
    size_t pix = lin / Cout;
    unsigned int c = (unsigned int)(lin % Cout);
    size_t y = pix / (2 * W), xx = pix % (2 * W);
    unsigned int sc = (c * factor + off + 2u * (unsigned int)(y & 1) + (unsigned int)(xx & 1)) / repeats;
    float v = bf16_to_f32(src[((y / 2) * W + (xx / 2)) * Cin + sc]);
    x[lin] = f32_to_bf16(bf16_to_f32(x[lin]) + v);
}

// Split a fused [pixels, 3C] QKV into Q[pixels, C], K[k_rows, C] (first
// `pixels` rows) and the transposed Vt[C, k_rows] (first `pixels` columns).
// grid = (ceil(pixels*C/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_vae_qkv_split(
    const __nv_bfloat16* __restrict__ qkv,
    __nv_bfloat16* __restrict__ Q, __nv_bfloat16* __restrict__ K,
    __nv_bfloat16* __restrict__ Vt,
    unsigned int pixels, unsigned int C, unsigned int k_rows)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (lin >= (size_t)pixels * C) return;
    size_t p = lin / C, c = lin % C;
    const __nv_bfloat16* row = qkv + p * 3 * C;
    Q[lin] = row[c];
    K[lin] = row[C + c];
    Vt[c * k_rows + p] = row[2 * (size_t)C + c];
}
