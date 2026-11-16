// SPDX-License-Identifier: AGPL-3.0-only

// Gated-delta-net pieces for Qwen3.8-Flash-Next (model_type qwen4_exp) that
// the shared GDN kernels do not cover:
//
//   gdn_gates_split  The checkpoint keeps `in_proj_a` and `in_proj_b` as two
//                    projections and stores A_log / dt_bias in BF16; the
//                    shared gate kernel expects one interleaved BF16 weight
//                    and FP32 parameters.
//   gdn_out_norm     The output norm gates with sigmoid (`output_gate_type`);
//                    the shared kernel gates with SiLU only.
//
// Formulas follow the published reference model (Qwen4ExpTextGatedDeltaNet,
// Qwen4ExpTextRMSNormGated).

#include <cuda_bf16.h>

// ── gdn_gates_split ──
// decay[t, h] = exp(-exp(A_log[h]) * softplus(a[t, h] + dt_bias[h]))
// beta[t, h]  = sigmoid(b[t, h])
// `decay` is exp(g), the form gated_delta_rule_decode* consumes.
// Grid: (T,1,1)  Block: (64,1,1), num_v_heads <= 64.
extern "C" __global__ void gdn_gates_split(
    const __nv_bfloat16* __restrict__ a,       // [T, num_v_heads]
    const __nv_bfloat16* __restrict__ b,       // [T, num_v_heads]
    const __nv_bfloat16* __restrict__ A_log,   // [num_v_heads]
    const __nv_bfloat16* __restrict__ dt_bias, // [num_v_heads]
    float* __restrict__ decay,                 // [T, num_v_heads]
    float* __restrict__ beta,                  // [T, num_v_heads]
    const unsigned int num_v_heads
) {
    const unsigned int h = threadIdx.x;
    if (h >= num_v_heads) return;
    const size_t o = (size_t)blockIdx.x * num_v_heads + h;
    const float x = __bfloat162float(a[o]) + __bfloat162float(dt_bias[h]);
    // softplus; for large x it equals x to FP32 precision
    const float sp = x > 20.0f ? x : log1pf(expf(x));
    decay[o] = expf(-expf(__bfloat162float(A_log[h])) * sp);
    beta[o] = 1.0f / (1.0f + expf(-__bfloat162float(b[o])));
}

// ── gdn_out_norm ──
// out[r, d] = x[r, d] * rsqrt(mean_d(x[r]^2) + eps) * w[d] * act(z[r, d])
// One row per (token, value head). act = sigmoid if `sigmoid_gate`, else SiLU.
// Grid: (rows,1,1)  Block: (128,1,1), any dim.
extern "C" __global__ void gdn_out_norm(
    const float* __restrict__ x,          // [rows, dim]
    const __nv_bfloat16* __restrict__ z,  // [rows, dim]
    const __nv_bfloat16* __restrict__ w,  // [dim]
    __nv_bfloat16* __restrict__ out,      // [rows, dim]
    const unsigned int dim,
    const float eps,
    const unsigned int sigmoid_gate
) {
    __shared__ float red[128];
    const unsigned int tid = threadIdx.x;
    const float* xr = x + (size_t)blockIdx.x * dim;
    const __nv_bfloat16* zr = z + (size_t)blockIdx.x * dim;
    __nv_bfloat16* o = out + (size_t)blockIdx.x * dim;
    float acc = 0.0f;
    for (unsigned int d = tid; d < dim; d += 128) acc += xr[d] * xr[d];
    red[tid] = acc;
    __syncthreads();
    for (unsigned int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float rs = rsqrtf(red[0] / (float)dim + eps);
    for (unsigned int d = tid; d < dim; d += 128) {
        const float g = __bfloat162float(zr[d]);
        const float sg = 1.0f / (1.0f + expf(-g));
        const float act = sigmoid_gate ? sg : g * sg;
        o[d] = __float2bfloat16(xr[d] * rs * __bfloat162float(w[d]) * act);
    }
}

// ── gdn_out_norm_bf16 ──
// As gdn_out_norm, for the BF16 output of the prefill recurrence kernels.
// Grid: (rows,1,1)  Block: (128,1,1), any dim.
extern "C" __global__ void gdn_out_norm_bf16(
    const __nv_bfloat16* __restrict__ x,  // [rows, dim]
    const __nv_bfloat16* __restrict__ z,  // [rows, dim]
    const __nv_bfloat16* __restrict__ w,  // [dim]
    __nv_bfloat16* __restrict__ out,      // [rows, dim]
    const unsigned int dim,
    const float eps,
    const unsigned int sigmoid_gate
) {
    __shared__ float red[128];
    const unsigned int tid = threadIdx.x;
    const __nv_bfloat16* xr = x + (size_t)blockIdx.x * dim;
    const __nv_bfloat16* zr = z + (size_t)blockIdx.x * dim;
    __nv_bfloat16* o = out + (size_t)blockIdx.x * dim;
    float acc = 0.0f;
    for (unsigned int d = tid; d < dim; d += 128) {
        const float v = __bfloat162float(xr[d]);
        acc += v * v;
    }
    red[tid] = acc;
    __syncthreads();
    for (unsigned int s = 64; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float rs = rsqrtf(red[0] / (float)dim + eps);
    for (unsigned int d = tid; d < dim; d += 128) {
        const float g = __bfloat162float(zr[d]);
        const float sg = 1.0f / (1.0f + expf(-g));
        const float act = sigmoid_gate ? sg : g * sg;
        o[d] = __float2bfloat16(__bfloat162float(xr[d]) * rs * __bfloat162float(w[d]) * act);
    }
}
