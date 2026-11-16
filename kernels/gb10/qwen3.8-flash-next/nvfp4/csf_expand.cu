// SPDX-License-Identifier: AGPL-3.0-only

// Expansion of csf-compressed NVFP4 block scales (Qwen3.8-Flash-Next routed
// experts, `weight_scale_encoding: "csf"`).
//
// The checkpoint stores each [rows, cols] matrix of E4M3 scale bytes as
//   fixed      u8  [rows / 16, 16 * (1 + cols / 2)]
//              per 16-row slab: 16 row base bytes, then per row cols / 2 bytes
//              of 4-bit codes, low nibble = even column; byte = base + code
//   exceptions u32 [n]  low 24 bits = row * cols + col, high 8 bits = the
//              exact byte, applied after base + code
// The compressed form stays resident. Before a MoE call the scales of the
// experts it uses are expanded into a scratch area with one fixed slot per
// expert: slot e = out + e * rows * cols, row-major [rows, cols] — the layout
// the NVFP4 expert kernels read.
//
// Experts are named either by `ids` (decode: the routed top-k, -1 = skip) or,
// with ids == null, by `route_offsets` (prefill: block y is expert y; the
// i32 [E + 1] prefix sums of the tokens routed to each expert, as the MoE
// sort produces them — an expert is expanded when its range is not empty).
// Run csf_expand_fixed, then csf_expand_exceptions on the same stream.

#define CSF_BLOCK 256
#define CSF_SLAB_ROWS 16

__device__ __forceinline__ int csf_expert(
    const int* ids, const int* route_offsets, unsigned int y
) {
    if (ids != nullptr) return ids[y];
    return route_offsets[y + 1] > route_offsets[y] ? (int)y : -1;
}

// Grid: (rows / 16, n, 1)  Block: (256,1,1)
extern "C" __global__ void csf_expand_fixed(
    const unsigned long long* __restrict__ fixed_ptrs, // [E] device pointers
    const int* __restrict__ ids,                       // [n] or null
    const int* __restrict__ route_offsets,             // [E + 1] or null
    unsigned char* __restrict__ out,                   // [E, rows, cols]
    const unsigned int rows,
    const unsigned int cols
) {
    const int e = csf_expert(ids, route_offsets, blockIdx.y);
    if (e < 0) return;
    const unsigned int slab_len = CSF_SLAB_ROWS * (1 + cols / 2);
    const unsigned char* slab =
        (const unsigned char*)fixed_ptrs[e] + (size_t)blockIdx.x * slab_len;
    unsigned char* dst =
        out + (size_t)e * rows * cols + (size_t)blockIdx.x * CSF_SLAB_ROWS * cols;
    for (unsigned int i = threadIdx.x; i < CSF_SLAB_ROWS * cols; i += CSF_BLOCK) {
        const unsigned int r = i / cols;
        const unsigned int c = i % cols;
        const unsigned int pair = slab[CSF_SLAB_ROWS + r * (cols / 2) + c / 2];
        dst[i] = (unsigned char)(slab[r] + ((pair >> (4 * (c & 1))) & 0xF));
    }
}

// Grid: (ceil(max exception count / 256), n, 1)  Block: (256,1,1)
extern "C" __global__ void csf_expand_exceptions(
    const unsigned long long* __restrict__ exc_ptrs, // [E] device pointers
    const unsigned int* __restrict__ exc_counts,     // [E]
    const int* __restrict__ ids,                     // [n] or null
    const int* __restrict__ route_offsets,           // [E + 1] or null
    unsigned char* __restrict__ out,                 // [E, rows, cols]
    const unsigned int rows,
    const unsigned int cols
) {
    const int e = csf_expert(ids, route_offsets, blockIdx.y);
    if (e < 0) return;
    const unsigned int i = blockIdx.x * CSF_BLOCK + threadIdx.x;
    if (i >= exc_counts[e]) return;
    const unsigned int rec = ((const unsigned int*)exc_ptrs[e])[i];
    const unsigned int pos = rec & 0xFFFFFF;
    if (pos < rows * cols) out[(size_t)e * rows * cols + pos] = (unsigned char)(rec >> 24);
}

// ── csf_validate ──
// Load-time check of one layer's compressed scales: counts row bases above
// 240 (base + code would wrap a byte) and exception records that are out of
// range or not strictly increasing. `errors` must be zeroed by the caller.
// Grid: (E,1,1)  Block: (256,1,1).
extern "C" __global__ void csf_validate(
    const unsigned long long* __restrict__ fixed_ptrs, // [E]
    const unsigned long long* __restrict__ exc_ptrs,   // [E]
    const unsigned int* __restrict__ exc_counts,       // [E]
    unsigned int* __restrict__ errors,                 // [1]
    const unsigned int rows,
    const unsigned int cols
) {
    const unsigned int e = blockIdx.x;
    const unsigned int slab_len = CSF_SLAB_ROWS * (1 + cols / 2);
    const unsigned char* fixed = (const unsigned char*)fixed_ptrs[e];
    const unsigned int* exc = (const unsigned int*)exc_ptrs[e];
    unsigned int bad = 0;
    for (unsigned int r = threadIdx.x; r < rows; r += CSF_BLOCK) {
        bad += fixed[(size_t)(r / CSF_SLAB_ROWS) * slab_len + r % CSF_SLAB_ROWS] > 240;
    }
    for (unsigned int i = threadIdx.x; i < exc_counts[e]; i += CSF_BLOCK) {
        const unsigned int pos = exc[i] & 0xFFFFFF;
        bad += pos >= rows * cols;
        if (i > 0) bad += (exc[i - 1] & 0xFFFFFF) >= pos;
    }
    if (bad) atomicAdd(errors, bad);
}
