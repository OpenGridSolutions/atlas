// SPDX-License-Identifier: AGPL-3.0-only

// Qwen Sparse Attention (QSA) indexer for Qwen3.8-Flash-Next (qwen4_exp).
//
// An attention layer does not attend to its whole context. Tokens are grouped
// in blocks of `ratio`; every complete block has one index key (the mean of
// its tokens' raw keys, normalised and rotated to the block's first
// position). A query scores all complete blocks it can see,
//     score(b) = sum over index heads of relu(q_head . key_b),
// keeps the `topk` best and attends to their tokens plus the tokens of the
// incomplete block at the end (which includes the query's own token unless
// that token completes a block). While at most `topk` blocks are visible this
// is ordinary causal attention.
//
// Index data lives next to the KV cache, addressed by the same physical
// block ids ("aux plane", BF16):
//     [kv_block_size, D]          raw key of every token
//     [kv_block_size / ratio, D]  index key of every complete block
// kv_block_size is a multiple of `ratio`, so a block of `ratio` tokens never
// straddles two KV blocks.
//
// Decode step of one token, in this order on one stream (the first two
// also take many tokens at once for prefill):
//     qsa_index_raw     store the raw keys
//     qsa_index_pool    complete the index key of every finished block
//     qsa_score         scores of all visible blocks
//     qsa_select        the selected blocks
//     qsa_gather        copy the selected tokens' K/V into a contiguous
//                       scratch cache for the paged attention kernel
//
// Formulas follow the published reference model (Qwen4ExpTextQSAIndexer).

#include <cuda_bf16.h>

#define QSA_BLOCK 256
#define QSA_MAX_D 256
#define QSA_MAX_HEADS 8
#define QSA_SELECT_BLOCK 1024

__device__ __forceinline__ float qsa_r16(float v) {
    return __bfloat162float(__float2bfloat16(v));
}

// Rotary embedding on the first `rot` dims of x (pairs (i, i + rot/2)),
// in place, BF16-rounded like the activations it replaces.
__device__ void qsa_rope_thread(float* x, unsigned int tid, unsigned int rot, float pos, float theta) {
    if (tid >= rot / 2) return;
    // Angle in double precision, as the shared rope kernel computes it.
    const double ang = (double)pos * pow((double)theta, -2.0 * (double)tid / (double)rot);
    const float s = (float)sin(ang), c = (float)cos(ang);
    const float a = x[tid], b = x[tid + rot / 2];
    x[tid] = qsa_r16(a * c - b * s);
    x[tid + rot / 2] = qsa_r16(b * c + a * s);
}

// ── qsa_index_raw ──
// Stores the raw key of every token of the call in the aux plane.
// Grid: (T,1,1)  Block: (256,1,1).
extern "C" __global__ void qsa_index_raw(
    const __nv_bfloat16* __restrict__ qk,       // [T, (heads + 1) * D], raw key last
    __nv_bfloat16* __restrict__ aux,            // aux plane of this layer
    const int* __restrict__ block_table,        // physical KV block per logical block
    const unsigned int* __restrict__ positions, // [T] position of each token
    const unsigned long long aux_block_bytes,
    const unsigned int kv_block_size,
    const unsigned int D,
    const unsigned int heads
) {
    const unsigned int pos = positions[blockIdx.x];
    __nv_bfloat16* blk = (__nv_bfloat16*)((char*)aux
        + (size_t)block_table[pos / kv_block_size] * aux_block_bytes);
    const __nv_bfloat16* raw = qk + (size_t)blockIdx.x * (heads + 1) * D + (size_t)heads * D;
    for (unsigned int d = threadIdx.x; d < D; d += QSA_BLOCK) {
        blk[(size_t)(pos % kv_block_size) * D + d] = raw[d];
    }
}

// ── qsa_index_pool ──
// For every token of the call that completes a block of `ratio` tokens:
// the block's index key = mean of its raw keys (read from the aux plane,
// so qsa_index_raw runs first), normalised, rotated to the block's first
// position. Other tokens do nothing.
// Grid: (T,1,1)  Block: (256,1,1).  D <= 256.
extern "C" __global__ void qsa_index_pool(
    const __nv_bfloat16* __restrict__ k_norm_w, // [D], weight (1 + w)
    __nv_bfloat16* __restrict__ aux,
    const int* __restrict__ block_table,
    const unsigned int* __restrict__ positions, // [T]
    const unsigned long long aux_block_bytes,
    const unsigned int kv_block_size,
    const unsigned int ratio,
    const unsigned int D,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    __shared__ float x[QSA_MAX_D];
    __shared__ float red[QSA_BLOCK];
    const unsigned int tid = threadIdx.x;
    const unsigned int pos = positions[blockIdx.x];
    if ((pos + 1) % ratio != 0) return;
    __nv_bfloat16* blk = (__nv_bfloat16*)((char*)aux
        + (size_t)block_table[pos / kv_block_size] * aux_block_bytes);
    const unsigned int in_blk = pos % kv_block_size;

    float acc = 0.0f;
    for (unsigned int d = tid; d < D; d += QSA_BLOCK) {
        float sum = 0.0f;
        for (unsigned int j = 0; j < ratio; ++j) {
            sum += __bfloat162float(blk[(size_t)(in_blk - j) * D + d]);
        }
        const float m = qsa_r16(sum / (float)ratio);
        x[d] = m;
        acc += m * m;
    }
    red[tid] = acc;
    __syncthreads();
    for (unsigned int s = QSA_BLOCK / 2; s > 0; s >>= 1) {
        if (tid < s) red[tid] += red[tid + s];
        __syncthreads();
    }
    const float rs = rsqrtf(red[0] / (float)D + eps);
    for (unsigned int d = tid; d < D; d += QSA_BLOCK) {
        x[d] = qsa_r16(x[d] * rs * (1.0f + __bfloat162float(k_norm_w[d])));
    }
    __syncthreads();
    qsa_rope_thread(x, tid, rot, (float)(pos + 1 - ratio), theta);
    __syncthreads();
    __nv_bfloat16* out = blk + (size_t)kv_block_size * D + (size_t)(in_blk / ratio) * D;
    for (unsigned int d = tid; d < D; d += QSA_BLOCK) out[d] = __float2bfloat16(x[d]);
}

// ── qsa_score ──
// scores[b] for every visible complete block b < (pos + 1) / ratio.
// Grid: (ceil(max_blocks / 256),1,1)  Block: (256,1,1).  heads * D <= 2048.
extern "C" __global__ void qsa_score(
    const __nv_bfloat16* __restrict__ qk,       // [(heads + 1) * D], queries first
    const __nv_bfloat16* __restrict__ q_norm_w, // [D], weight (1 + w)
    const __nv_bfloat16* __restrict__ aux,
    const int* __restrict__ block_table,
    const unsigned int* __restrict__ positions,
    float* __restrict__ scores,                 // [max_blocks]
    const unsigned long long aux_block_bytes,
    const unsigned int kv_block_size,
    const unsigned int ratio,
    const unsigned int D,
    const unsigned int heads,
    const unsigned int rot,
    const float theta,
    const float eps
) {
    __shared__ float q[QSA_MAX_HEADS * QSA_MAX_D];
    __shared__ float red[QSA_BLOCK];
    const unsigned int tid = threadIdx.x;
    const unsigned int pos = positions[0];
    const unsigned int visible = (pos + 1) / ratio;
    const unsigned int first = blockIdx.x * QSA_BLOCK;
    if (first >= visible) return;

    // Queries: per-head RMS norm (1 + w), then rotation at the token's position.
    for (unsigned int h = 0; h < heads; ++h) {
        float acc = 0.0f;
        for (unsigned int d = tid; d < D; d += QSA_BLOCK) {
            const float v = __bfloat162float(qk[(size_t)h * D + d]);
            acc += v * v;
        }
        red[tid] = acc;
        __syncthreads();
        for (unsigned int s = QSA_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) red[tid] += red[tid + s];
            __syncthreads();
        }
        const float rs = rsqrtf(red[0] / (float)D + eps);
        __syncthreads();
        for (unsigned int d = tid; d < D; d += QSA_BLOCK) {
            q[h * D + d] = qsa_r16(__bfloat162float(qk[(size_t)h * D + d]) * rs
                * (1.0f + __bfloat162float(q_norm_w[d])));
        }
        __syncthreads();
        qsa_rope_thread(q + h * D, tid, rot, (float)pos, theta);
        __syncthreads();
    }

    const unsigned int b = first + tid;
    if (b >= visible) return;
    const unsigned int tok = b * ratio;
    const __nv_bfloat16* key = (const __nv_bfloat16*)((const char*)aux
        + (size_t)block_table[tok / kv_block_size] * aux_block_bytes)
        + (size_t)kv_block_size * D + (size_t)((tok % kv_block_size) / ratio) * D;
    float score = 0.0f;
    for (unsigned int h = 0; h < heads; ++h) {
        float dot = 0.0f;
        for (unsigned int d = 0; d < D; ++d) dot += q[h * D + d] * __bfloat162float(key[d]);
        score += fmaxf(dot, 0.0f);
    }
    scores[b] = score;
}

// ── qsa_select ──
// selected[0 .. count) = the blocks to attend: all visible ones if there are
// at most `topk`, else the `topk` highest scores (ties: lower block first).
// Scores are sums of relu terms, i.e. non-negative, so their bit patterns
// order like the values.
// Grid: (1,1,1)  Block: (1024,1,1).
extern "C" __global__ void qsa_select(
    const float* __restrict__ scores,           // [max_blocks]
    const unsigned int* __restrict__ positions,
    int* __restrict__ selected,                 // [topk]
    int* __restrict__ count,                    // [1]
    const unsigned int ratio,
    const unsigned int topk
) {
    __shared__ unsigned int cnt[QSA_SELECT_BLOCK];
    __shared__ unsigned int cnt_eq[QSA_SELECT_BLOCK];
    __shared__ unsigned int prefix;
    const unsigned int tid = threadIdx.x;
    const unsigned int visible = (positions[0] + 1) / ratio;
    if (visible <= topk) {
        for (unsigned int b = tid; b < visible; b += QSA_SELECT_BLOCK) selected[b] = (int)b;
        if (tid == 0) count[0] = (int)visible;
        return;
    }
    // Contiguous share of the blocks per thread keeps the output ordered.
    const unsigned int per = (visible + QSA_SELECT_BLOCK - 1) / QSA_SELECT_BLOCK;
    const unsigned int lo = tid * per;
    const unsigned int hi = min(lo + per, visible);
    const unsigned int* bits = (const unsigned int*)scores;

    // Largest threshold T with at least `topk` scores >= T: bit by bit.
    if (tid == 0) prefix = 0;
    __syncthreads();
    for (int bit = 31; bit >= 0; --bit) {
        const unsigned int cand = prefix | (1u << bit);
        unsigned int c = 0;
        for (unsigned int b = lo; b < hi; ++b) c += bits[b] >= cand;
        cnt[tid] = c;
        __syncthreads();
        for (unsigned int s = QSA_SELECT_BLOCK / 2; s > 0; s >>= 1) {
            if (tid < s) cnt[tid] += cnt[tid + s];
            __syncthreads();
        }
        if (tid == 0 && cnt[0] >= topk) prefix = cand;
        __syncthreads();
    }
    const unsigned int thr = prefix;

    // Exclusive prefix sums of "above" and "equal" counts over the threads.
    unsigned int above = 0, equal = 0;
    for (unsigned int b = lo; b < hi; ++b) {
        above += bits[b] > thr;
        equal += bits[b] == thr;
    }
    cnt[tid] = above;
    cnt_eq[tid] = equal;
    __syncthreads();
    if (tid == 0) {
        unsigned int a = 0, e = 0;
        for (unsigned int t = 0; t < QSA_SELECT_BLOCK; ++t) {
            const unsigned int ta = cnt[t], te = cnt_eq[t];
            cnt[t] = a;
            cnt_eq[t] = e;
            a += ta;
            e += te;
        }
        prefix = a; // total above the threshold (< topk)
        count[0] = (int)topk;
    }
    __syncthreads();
    const unsigned int total_above = prefix;
    unsigned int a = cnt[tid], e = cnt_eq[tid];
    for (unsigned int b = lo; b < hi; ++b) {
        if (bits[b] > thr) {
            selected[a++] = (int)b;
        } else if (bits[b] == thr) {
            if (total_above + e < topk) selected[total_above + e] = (int)b;
            ++e;
        }
    }
}

// ── qsa_gather ──
// Copies the K/V rows of the selected tokens into a contiguous scratch cache:
// rows [0, count * ratio) are the selected blocks' tokens, followed by the
// tokens of the incomplete block at the end. tokens[0] = number of rows.
// Grid: (topk + 1,1,1)  Block: (256,1,1).
extern "C" __global__ void qsa_gather(
    const int* __restrict__ selected,           // [topk]
    const int* __restrict__ count,              // [1]
    const unsigned int* __restrict__ positions,
    const int* __restrict__ block_table,
    const __nv_bfloat16* __restrict__ k_cache,  // [blocks, kv_block_size, row]
    const __nv_bfloat16* __restrict__ v_cache,
    __nv_bfloat16* __restrict__ k_out,          // [topk * ratio + ratio - 1, row]
    __nv_bfloat16* __restrict__ v_out,
    int* __restrict__ tokens,                   // [1]
    const unsigned int kv_block_size,
    const unsigned int ratio,
    const unsigned int row,                     // elements per token (kv heads * head_dim)
    const unsigned int topk
) {
    const unsigned int tid = threadIdx.x;
    const unsigned int pos = positions[0];
    const unsigned int n = (unsigned int)count[0];
    unsigned int src_tok, dst_row, num;
    if (blockIdx.x < topk) {
        if (blockIdx.x >= n) return;
        src_tok = (unsigned int)selected[blockIdx.x] * ratio;
        dst_row = blockIdx.x * ratio;
        num = ratio;
    } else {
        const unsigned int visible = (pos + 1) / ratio;
        src_tok = visible * ratio;
        dst_row = n * ratio;
        num = pos + 1 - src_tok;
        if (tid == 0) tokens[0] = (int)(dst_row + num);
    }
    for (unsigned int t = 0; t < num; ++t) {
        const unsigned int p = src_tok + t;
        const size_t src = ((size_t)block_table[p / kv_block_size] * kv_block_size + p % kv_block_size) * row;
        const size_t dst = (size_t)(dst_row + t) * row;
        for (unsigned int d = tid; d < row; d += QSA_BLOCK) {
            k_out[dst + d] = k_cache[src + d];
            v_out[dst + d] = v_cache[src + d];
        }
    }
}
