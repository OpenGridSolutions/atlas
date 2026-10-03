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

// ── VAE encoder (NHWC) — /v1/images/edits condition-image path ──────────────

// im2col for the encoder's downsampling 3×3 / stride 2 convolution
// (nn.ZeroPad2d((0, 1, 0, 1)) + Conv2d(stride=2)): output pixel (y, x)
// reads the window rows [2y, 2y+2] × cols [2x, 2x+2] with zero padding
// ONLY below/right (the left/top pad of the stride-1 kernel is absent).
// Columns ordered (ky, kx, c) like qimg_im2col3x3. H and W are the INPUT
// dims (even); the output grid is (H/2) × (W/2) rows [row0, row0+rows).
// grid = (ceil(rows*(W/2)*9*C/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_im2col3x3_s2(
    const __nv_bfloat16* __restrict__ in,   // [H*W, C]
    __nv_bfloat16* __restrict__ out,        // [rows*(W/2), 9*C]
    unsigned int H, unsigned int W, unsigned int C,
    unsigned int row0, unsigned int rows)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int Wo = W / 2;
    size_t k9 = (size_t)9 * C;
    if (lin >= (size_t)rows * Wo * k9) return;
    size_t pix = lin / k9;
    unsigned int col = (unsigned int)(lin % k9);
    unsigned int tap = col / C, c = col % C;
    unsigned int y = 2u * (row0 + (unsigned int)(pix / Wo)) + tap / 3;
    unsigned int x = 2u * (unsigned int)(pix % Wo) + tap % 3;
    __nv_bfloat16 v = f32_to_bf16(0.0f);
    if (y < H && x < W)
        v = in[((size_t)y * W + x) * C + c];
    out[lin] = v;
}

// AvgDown3D shortcut of the encoder's residual down blocks: 2×2 spatial
// pooling with channel-group averaging. The stacked channel order after
// the reference's permute is (c, t, ky, kx) with `factor = ft*fs*fs` and
// `group = Cin*factor/Cout`; out channel co averages the group entries
// idx = co*group + g. With a single frame the reference zero-pads the
// temporal axis IN FRONT, so only the LAST temporal slot (it == ft-1)
// carries the image — every other slot contributes zero (this is why the
// even output channels of the temporal blocks are exactly zero).
// Identity blocks (factor == 1) should use a plain add instead.
// grid = (ceil((H/2)*(W/2)*Cout/256),1,1), block = (256,1,1)
extern "C" __global__
void qimg_vae_avg_down(
    const __nv_bfloat16* __restrict__ in,   // [H*W, Cin]
    __nv_bfloat16* __restrict__ out,        // [(H/2)*(W/2), Cout] — ADDED to
    unsigned int H, unsigned int W, unsigned int Cin, unsigned int Cout,
    unsigned int group, unsigned int factor, unsigned int ft)
{
    size_t lin = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int Wo = W / 2;
    unsigned int Ho = H / 2;
    if (lin >= (size_t)Ho * Wo * Cout) return;
    unsigned int co = (unsigned int)(lin % Cout);
    size_t pix = lin / Cout;
    unsigned int y = (unsigned int)(pix / Wo), x = (unsigned int)(pix % Wo);
    unsigned int fs2 = factor / ft;         // fs*fs = 4 for the down blocks
    float acc = 0.0f;
    for (unsigned int g = 0; g < group; ++g) {
        unsigned int idx = co * group + g;
        unsigned int c = idx / factor;
        unsigned int r = idx % factor;
        unsigned int it = r / fs2;
        unsigned int rest = r % fs2;
        unsigned int ih = rest / 2u, iw = rest % 2u;
        if (it == ft - 1u)
            acc += bf16_to_f32(in[((size_t)(2u * y + ih) * W + (2u * x + iw)) * Cin + c]);
    }
    out[lin] = f32_to_bf16(bf16_to_f32(out[lin]) + acc / (float)group);
}

// ── Flash attention (DiT target pass) ────────────────────────────────────────
//
// Fused Tensor-Core SDPA (Flash Attention v2, mma.sync m16n8k16 BF16) for the
// DiT's full-attention target rows, adapted from
// common/inferspark_prefill_h128.cu (BR=64 variant) to the sidecar's
// head-major attention buffers:
//   Q  [H][q_len][128]      qimg_norm_rope_heads output, rows tight
//   K  [H][kv_stride][128]  kh — rows [kv_len, kv_stride) are zeroed padding
//   Vt [H][128][kv_stride]  qimg_v_transpose output, same padding guarantee
//   O  [q_len][H*128]       written at column head*128 (scatter fused)
// Every query row attends keys [0, kv_len) — no masks; the prefix pass keeps
// the GEMM-SDPA path for its per-row limits. kv_stride is a multiple of 16
// (pad_keys), so the 8-element cp.async chunks never straddle the padding
// edge and reads up to kv_stride stay in bounds.
//
// Per CTA: 64 query rows of one head, 8 warps (256 threads), ~41 KB smem
// (2 CTAs/SM on GB10's 99 KB). Differences from the template:
//   * K is single-buffered: the next tile's load is issued after the
//     post-softmax barrier — no QK^T reader left — and overlaps with PV,
//     so the prefetch pipeline is unchanged.
//   * V tiles are kept TRANSPOSED in smem ([128][40]): the gmem loads stay
//     16-byte-contiguous along k, the PV B-fragment becomes one aligned
//     32-bit read ({V[k][d], V[k+1][d]} adjacent), and the stride of 40
//     u16 (20 words) gives a conflict-free bank pattern across a warp.
//
// QK^T: warps 0-3 own 16 query rows each; PV: warp pairs (w, w+4) share
// rows and split the 128 output columns. ldmatrix is avoided throughout
// (broken on GB10) — fragments are assembled with manual smem reads.
// grid = (H, ceil(q_len/64), 1), block = (256,1,1)

#define QFA_BR 64
#define QFA_BC 32
#define QFA_D 128
#define QFA_DPAD (QFA_D + 8)        // K/Q row stride: 272 B, 16-byte aligned
#define QFA_VSTRIDE (QFA_BC + 8)    // Vt row stride: 80 B, 16-byte aligned
#define QFA_PSTRIDE (QFA_BC + 8)
#define QFA_NT 8                    // PV n-tiles per warp: (128/8)/2

extern "C" __global__
void qimg_flash_attn(
    const __nv_bfloat16* __restrict__ Q,
    const __nv_bfloat16* __restrict__ K,
    const __nv_bfloat16* __restrict__ Vt,
    __nv_bfloat16* __restrict__ O,
    const unsigned int q_len,
    const unsigned int kv_len,
    const unsigned int kv_stride,
    const unsigned int heads,
    const float inv_sqrt_d)
{
    const unsigned int head = blockIdx.x;
    const unsigned int tid = threadIdx.x;
    const unsigned int warp_id = tid / 32;
    const unsigned int lane_id = tid % 32;

    const unsigned int q_start = blockIdx.y * QFA_BR;
    if (q_start >= q_len) return;
    const unsigned int q_rows = min(QFA_BR, q_len - q_start);

    const __nv_bfloat16* Qh = Q + (size_t)head * q_len * QFA_D;
    const __nv_bfloat16* Kh = K + (size_t)head * kv_stride * QFA_D;
    const __nv_bfloat16* Vh = Vt + (size_t)head * QFA_D * kv_stride;
    __nv_bfloat16* Oh = O + (size_t)head * QFA_D;
    const unsigned int o_stride = heads * QFA_D;

    __shared__ __nv_bfloat16 smem_Q[QFA_BR][QFA_DPAD];
    __shared__ __nv_bfloat16 smem_K[QFA_BC][QFA_DPAD];
    __shared__ __nv_bfloat16 smem_Vt[QFA_D][QFA_VSTRIDE];
    __shared__ __nv_bfloat16 smem_P[QFA_BR][QFA_PSTRIDE];
    __shared__ float smem_ml[QFA_BR][2];   // [row][0]=m, [row][1]=l

    const unsigned int group_id = lane_id >> 2;
    const unsigned int tid_in_group = lane_id & 3;

    const unsigned int qk_warp_m = warp_id * 16;        // warps 0-3
    const unsigned int pv_warp_m = (warp_id & 3) * 16;  // all 8 warps
    const unsigned int pv_n_start = (warp_id >> 2) * QFA_NT;

    float acc_o[QFA_NT][4];
    #pragma unroll
    for (int i = 0; i < QFA_NT; i++) {
        acc_o[i][0] = 0.0f; acc_o[i][1] = 0.0f;
        acc_o[i][2] = 0.0f; acc_o[i][3] = 0.0f;
    }

    // Per-row online softmax state (each thread owns 2 rows)
    float m_r0 = -1e30f, m_r1 = -1e30f;
    float l_r0 = 0.0f, l_r1 = 0.0f;

    const unsigned int num_kv_blocks = (kv_len + QFA_BC - 1) / QFA_BC;

    // === Merged Q + K[0] load (single cp.async commit group) ===
    {
        const unsigned int cpr = QFA_D / 8;  // 16 chunks per row

        for (unsigned int idx = tid; idx < QFA_BR * cpr; idx += 256) {
            unsigned int row = idx / cpr;
            unsigned int col = (idx % cpr) * 8;
            unsigned int q_row = q_start + row;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_Q[row][col]);
            if (q_row < q_len) {
                const void* gmem = (const void*)&Qh[(size_t)q_row * QFA_D + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_Q[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        for (unsigned int idx = tid; idx < QFA_BC * cpr; idx += 256) {
            unsigned int row = idx / cpr;
            unsigned int col = (idx % cpr) * 8;
            unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[row][col]);
            if (row < kv_stride) {
                const void* gmem = (const void*)&Kh[(size_t)row * QFA_D + col];
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
            } else {
                *((uint4*)&smem_K[row][col]) = make_uint4(0, 0, 0, 0);
            }
        }

        asm volatile("cp.async.commit_group;");
        asm volatile("cp.async.wait_group 0;");
    }
    __syncthreads();

    for (unsigned int kv_block = 0; kv_block < num_kv_blocks; kv_block++) {
        unsigned int kv_start = kv_block * QFA_BC;
        unsigned int kv_tile = min(QFA_BC, kv_len - kv_start);

        // === Async V tile load, transposed (overlaps with QK^T below) ===
        {
            for (unsigned int idx = tid; idx < QFA_D * (QFA_BC / 8); idx += 256) {
                unsigned int d = idx / (QFA_BC / 8);
                unsigned int kcol = (idx % (QFA_BC / 8)) * 8;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_Vt[d][kcol]);
                if (kv_start + kcol < kv_stride) {
                    const void* gmem = (const void*)&Vh[(size_t)d * kv_stride + kv_start + kcol];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_Vt[d][kcol]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === QK^T (warps 0-3, each 16 M-rows) ===
        float acc_s[4][4];
        if (warp_id < 4) {
            #pragma unroll
            for (int i = 0; i < 4; i++) {
                acc_s[i][0] = 0.0f; acc_s[i][1] = 0.0f;
                acc_s[i][2] = 0.0f; acc_s[i][3] = 0.0f;
            }

            const unsigned short* sQ = (const unsigned short*)smem_Q;
            const unsigned short* sK = (const unsigned short*)smem_K;

            #pragma unroll
            for (unsigned int ks = 0; ks < (QFA_D / 16); ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = qk_warp_m + group_id;
                unsigned int ar1 = ar0 + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = ac0 + 8;

                unsigned int a0 = *(const unsigned int*)&sQ[ar0 * QFA_DPAD + ac0];
                unsigned int a1 = *(const unsigned int*)&sQ[ar1 * QFA_DPAD + ac0];
                unsigned int a2 = *(const unsigned int*)&sQ[ar0 * QFA_DPAD + ac1];
                unsigned int a3 = *(const unsigned int*)&sQ[ar1 * QFA_DPAD + ac1];

                #pragma unroll
                for (int nt = 0; nt < 4; nt++) {
                    unsigned int n_col = nt * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k0 + 8;
                    unsigned int b0 = ((unsigned int)sK[n_col * QFA_DPAD + k0 + 1] << 16) |
                                      (unsigned int)sK[n_col * QFA_DPAD + k0];
                    unsigned int b1 = ((unsigned int)sK[n_col * QFA_DPAD + k1 + 1] << 16) |
                                      (unsigned int)sK[n_col * QFA_DPAD + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_s[nt][0]), "=f"(acc_s[nt][1]),
                          "=f"(acc_s[nt][2]), "=f"(acc_s[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_s[nt][0]), "f"(acc_s[nt][1]),
                          "f"(acc_s[nt][2]), "f"(acc_s[nt][3])
                    );
                }
            }

            // === Register-based softmax ===
            unsigned int row0 = qk_warp_m + group_id;
            unsigned int row1 = row0 + 8;

            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                acc_s[nt][0] *= inv_sqrt_d;
                acc_s[nt][1] *= inv_sqrt_d;
                acc_s[nt][2] *= inv_sqrt_d;
                acc_s[nt][3] *= inv_sqrt_d;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                unsigned int col1 = col0 + 1;
                if (col0 >= kv_tile) { acc_s[nt][0] = -1e30f; acc_s[nt][2] = -1e30f; }
                if (col1 >= kv_tile) { acc_s[nt][1] = -1e30f; acc_s[nt][3] = -1e30f; }
                if (row0 >= q_rows) { acc_s[nt][0] = -1e30f; acc_s[nt][1] = -1e30f; }
                if (row1 >= q_rows) { acc_s[nt][2] = -1e30f; acc_s[nt][3] = -1e30f; }
            }

            float rmax0 = -1e30f, rmax1 = -1e30f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                rmax0 = fmaxf(rmax0, fmaxf(acc_s[nt][0], acc_s[nt][1]));
                rmax1 = fmaxf(rmax1, fmaxf(acc_s[nt][2], acc_s[nt][3]));
            }
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 1));
            rmax0 = fmaxf(rmax0, __shfl_xor_sync(0xFFFFFFFF, rmax0, 2));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 1));
            rmax1 = fmaxf(rmax1, __shfl_xor_sync(0xFFFFFFFF, rmax1, 2));

            float m_new0 = fmaxf(m_r0, rmax0);
            float exp_old0 = __expf(m_r0 - m_new0);
            l_r0 *= exp_old0;
            #pragma unroll
            for (int i = 0; i < QFA_NT; i++) {
                acc_o[i][0] *= exp_old0; acc_o[i][1] *= exp_old0;
            }
            m_r0 = m_new0;

            float m_new1 = fmaxf(m_r1, rmax1);
            float exp_old1 = __expf(m_r1 - m_new1);
            l_r1 *= exp_old1;
            #pragma unroll
            for (int i = 0; i < QFA_NT; i++) {
                acc_o[i][2] *= exp_old1; acc_o[i][3] *= exp_old1;
            }
            m_r1 = m_new1;

            float sum0 = 0.0f, sum1 = 0.0f;
            #pragma unroll
            for (int nt = 0; nt < 4; nt++) {
                float p00 = __expf(acc_s[nt][0] - m_r0);
                float p01 = __expf(acc_s[nt][1] - m_r0);
                float p10 = __expf(acc_s[nt][2] - m_r1);
                float p11 = __expf(acc_s[nt][3] - m_r1);
                sum0 += p00 + p01;
                sum1 += p10 + p11;

                unsigned int col0 = nt * 8 + tid_in_group * 2;
                smem_P[row0][col0]     = __float2bfloat16(p00);
                smem_P[row0][col0 + 1] = __float2bfloat16(p01);
                smem_P[row1][col0]     = __float2bfloat16(p10);
                smem_P[row1][col0 + 1] = __float2bfloat16(p11);
            }

            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 1);
            sum0 += __shfl_xor_sync(0xFFFFFFFF, sum0, 2);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 1);
            sum1 += __shfl_xor_sync(0xFFFFFFFF, sum1, 2);
            l_r0 += sum0;
            l_r1 += sum1;

            if (tid_in_group == 0) {
                smem_ml[row0][0] = m_r0; smem_ml[row0][1] = l_r0;
                smem_ml[row1][0] = m_r1; smem_ml[row1][1] = l_r1;
            }
        }

        // Wait for the V tile (was loading during QK^T + softmax)
        asm volatile("cp.async.wait_group 0;");
        __syncthreads();

        // Warps 4-7: rescale accumulators to the current m
        if (warp_id >= 4) {
            unsigned int row0 = pv_warp_m + group_id;
            unsigned int row1 = row0 + 8;
            float cur_m0 = smem_ml[row0][0];
            float cur_m1 = smem_ml[row1][0];
            float exp_r0 = __expf(m_r0 - cur_m0);
            float exp_r1 = __expf(m_r1 - cur_m1);
            #pragma unroll
            for (int i = 0; i < QFA_NT; i++) {
                acc_o[i][0] *= exp_r0; acc_o[i][1] *= exp_r0;
                acc_o[i][2] *= exp_r1; acc_o[i][3] *= exp_r1;
            }
            m_r0 = cur_m0; m_r1 = cur_m1;
        }

        // === Preload K[i+1] (single buffer: QK^T readers are past the
        // barrier above; the load overlaps with PV below) ===
        if (kv_block + 1 < num_kv_blocks) {
            unsigned int next = (kv_block + 1) * QFA_BC;
            const unsigned int cpr = QFA_D / 8;
            for (unsigned int idx = tid; idx < QFA_BC * cpr; idx += 256) {
                unsigned int row = idx / cpr;
                unsigned int col = (idx % cpr) * 8;
                unsigned int k_row = next + row;
                unsigned int smem_addr = __cvta_generic_to_shared(&smem_K[row][col]);
                if (k_row < kv_stride) {
                    const void* gmem = (const void*)&Kh[(size_t)k_row * QFA_D + col];
                    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr), "l"(gmem));
                } else {
                    *((uint4*)&smem_K[row][col]) = make_uint4(0, 0, 0, 0);
                }
            }
            asm volatile("cp.async.commit_group;");
        }

        // === PV MMA (all 8 warps, 8 n-tiles each) ===
        {
            const unsigned short* sP = (const unsigned short*)smem_P;
            const unsigned short* sV = (const unsigned short*)smem_Vt;

            #pragma unroll
            for (unsigned int ks = 0; ks < 2; ks++) {
                unsigned int k_off = ks * 16;
                unsigned int ar0 = pv_warp_m + group_id;
                unsigned int ar1 = ar0 + 8;
                unsigned int ac0 = k_off + tid_in_group * 2;
                unsigned int ac1 = ac0 + 8;

                unsigned int a0 = *(const unsigned int*)&sP[ar0 * QFA_PSTRIDE + ac0];
                unsigned int a1 = *(const unsigned int*)&sP[ar1 * QFA_PSTRIDE + ac0];
                unsigned int a2 = *(const unsigned int*)&sP[ar0 * QFA_PSTRIDE + ac1];
                unsigned int a3 = *(const unsigned int*)&sP[ar1 * QFA_PSTRIDE + ac1];

                #pragma unroll
                for (int nt = 0; nt < QFA_NT; nt++) {
                    unsigned int n_col = (pv_n_start + nt) * 8 + group_id;
                    unsigned int k0 = k_off + tid_in_group * 2;
                    unsigned int k1 = k0 + 8;

                    // smem_Vt[n_col][k], k contiguous: {V[k][n], V[k+1][n]}
                    // is one aligned 32-bit read.
                    unsigned int b0 = *(const unsigned int*)&sV[n_col * QFA_VSTRIDE + k0];
                    unsigned int b1 = *(const unsigned int*)&sV[n_col * QFA_VSTRIDE + k1];

                    asm volatile(
                        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                        "{%0, %1, %2, %3}, "
                        "{%4, %5, %6, %7}, "
                        "{%8, %9}, "
                        "{%10, %11, %12, %13};"
                        : "=f"(acc_o[nt][0]), "=f"(acc_o[nt][1]),
                          "=f"(acc_o[nt][2]), "=f"(acc_o[nt][3])
                        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
                          "r"(b0), "r"(b1),
                          "f"(acc_o[nt][0]), "f"(acc_o[nt][1]),
                          "f"(acc_o[nt][2]), "f"(acc_o[nt][3])
                    );
                }
            }
        }

        // Wait for the K[i+1] prefetch before the next QK^T
        if (kv_block + 1 < num_kv_blocks) {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();
    }

    // === Final normalization and store ===
    {
        unsigned int row0 = pv_warp_m + group_id;
        unsigned int row1 = row0 + 8;

        float inv_l0, inv_l1;
        if (warp_id < 4) {
            inv_l0 = (l_r0 > 0.0f) ? (1.0f / l_r0) : 0.0f;
            inv_l1 = (l_r1 > 0.0f) ? (1.0f / l_r1) : 0.0f;
        } else {
            float l0 = smem_ml[row0][1];
            float l1 = smem_ml[row1][1];
            inv_l0 = (l0 > 0.0f) ? (1.0f / l0) : 0.0f;
            inv_l1 = (l1 > 0.0f) ? (1.0f / l1) : 0.0f;
        }

        #pragma unroll
        for (int nt = 0; nt < QFA_NT; nt++) {
            unsigned int col0 = (pv_n_start + nt) * 8 + tid_in_group * 2;

            if (row0 < q_rows) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][0] * inv_l0));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][1] * inv_l0));
                *(unsigned int*)&Oh[(size_t)(q_start + row0) * o_stride + col0] = lo | (hi << 16);
            }
            if (row1 < q_rows) {
                unsigned int lo = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][2] * inv_l1));
                unsigned int hi = (unsigned int)__bfloat16_as_ushort(__float2bfloat16(acc_o[nt][3] * inv_l1));
                *(unsigned int*)&Oh[(size_t)(q_start + row1) * o_stride + col0] = lo | (hi << 16);
            }
        }
    }
}
