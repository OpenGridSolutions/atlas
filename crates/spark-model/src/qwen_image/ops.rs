// SPDX-License-Identifier: AGPL-3.0-only

//! Kernel handles and launch helpers shared by the Qwen-Image text encoder,
//! DiT and VAE decoder. Launch geometry is documented next to each kernel in
//! `qwen_image.cu`; everything runs synchronously on the default stream.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

pub(crate) struct Ops {
    gemm_pipelined: KernelHandle,
    gemm_pipelined_f32: KernelHandle,
    layernorm_mod: KernelHandle,
    rms_norm: KernelHandle,
    norm_rope_heads: KernelHandle,
    v_transpose: KernelHandle,
    copy_prefix: KernelHandle,
    softmax_rows: KernelHandle,
    scatter_head: KernelHandle,
    gated_add: KernelHandle,
    silu_mul: KernelHandle,
    gelu_tanh: KernelHandle,
    bias_add: KernelHandle,
    im2col3x3: KernelHandle,
    vae_norm: KernelHandle,
    upsample2x: KernelHandle,
    dup_shortcut_add: KernelHandle,
    vae_qkv_split: KernelHandle,
    /// Route the large projection GEMMs through cuBLASLt (default) instead
    /// of the in-tree pipelined kernel (`ATLAS_IMAGE_NO_CUBLAS=1`).
    use_cublas: bool,
    /// The sidecar's own cuBLASLt workspace: it runs on its own thread next
    /// to the scheduler, so it must not share the process-wide one.
    cublas_ws: DevicePtr,
}

pub(crate) const CUBLAS_WS_BYTES: usize = 64 << 20;

fn blocks(n: usize) -> u32 {
    n.div_ceil(256) as u32
}

/// Key count padded for the GEMM2 K dimension (cp.async fast path wants a
/// multiple of 8; 16 keeps the mma K-tiles whole).
pub(crate) fn pad_keys(n: usize) -> usize {
    n.div_ceil(16) * 16
}

/// Buffers for one GEMM-SDPA attention call.
pub(crate) struct AttnBufs {
    /// [Hq][nq][D] bf16
    pub qh: DevicePtr,
    /// [Hkv][nk_pad][D] bf16
    pub kh: DevicePtr,
    /// [Hkv][D][nk_pad] bf16
    pub vt: DevicePtr,
    /// [nq, nk_pad] f32
    pub scores: DevicePtr,
    /// [nq, nk_pad] bf16
    pub probs: DevicePtr,
    /// [nq, D] bf16
    pub o_stage: DevicePtr,
}

impl Ops {
    /// `cublas_ws` is a `CUBLAS_WS_BYTES` device buffer owned by the caller.
    pub fn resolve(g: &dyn GpuBackend, cublas_ws: DevicePtr) -> Result<Self> {
        let k = |name: &str| g.kernel("qwen_image", name);
        Ok(Self {
            gemm_pipelined: g.kernel("gemm", "dense_gemm_bf16_pipelined")?,
            gemm_pipelined_f32: g.kernel("gemm", "dense_gemm_bf16_pipelined_f32out")?,
            layernorm_mod: k("qimg_layernorm_mod")?,
            rms_norm: k("qimg_rms_norm")?,
            norm_rope_heads: k("qimg_norm_rope_heads")?,
            v_transpose: k("qimg_v_transpose")?,
            copy_prefix: k("qimg_copy_prefix")?,
            softmax_rows: k("qimg_softmax_rows")?,
            scatter_head: k("qimg_scatter_head")?,
            gated_add: k("qimg_gated_add")?,
            silu_mul: k("qimg_silu_mul")?,
            gelu_tanh: k("qimg_gelu_tanh")?,
            bias_add: k("qimg_bias_add")?,
            im2col3x3: k("qimg_im2col3x3")?,
            vae_norm: k("qimg_vae_norm")?,
            upsample2x: k("qimg_upsample2x")?,
            dup_shortcut_add: k("qimg_dup_shortcut_add")?,
            vae_qkv_split: k("qimg_vae_qkv_split")?,
            use_cublas: std::env::var("ATLAS_IMAGE_NO_CUBLAS").map_or(true, |v| v != "1"),
            cublas_ws,
        })
    }

    /// C[m,n] = A[m,k] · W[n,k]ᵀ, all BF16 — tensor-core in-tree kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_pipelined(
        &self,
        g: &dyn GpuBackend,
        a: DevicePtr,
        w: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        KernelLaunch::new(g, self.gemm_pipelined)
            .grid([(n as u32).div_ceil(128), (m as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(w)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(g.default_stream())
    }

    /// Same GEMM with an FP32 result (attention scores, final projections).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm_f32out(
        &self,
        g: &dyn GpuBackend,
        a: DevicePtr,
        w: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        KernelLaunch::new(g, self.gemm_pipelined_f32)
            .grid([(n as u32).div_ceil(128), (m as u32).div_ceil(128), 1])
            .block([256, 1, 1])
            .arg_ptr(a)
            .arg_ptr(w)
            .arg_ptr(c)
            .arg_u32(m as u32)
            .arg_u32(n as u32)
            .arg_u32(k as u32)
            .launch(g.default_stream())
    }

    /// Projection GEMM: cuBLASLt when enabled, else the pipelined kernel.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self,
        g: &dyn GpuBackend,
        a: DevicePtr,
        w: DevicePtr,
        c: DevicePtr,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<()> {
        if self.use_cublas {
            spark_runtime::cublaslt::bf16_gemm_act_weight_t_ws(
                a.0,
                w.0,
                c.0,
                m as u32,
                n as u32,
                k as u32,
                g.default_stream(),
                self.cublas_ws.0,
                CUBLAS_WS_BYTES,
            )
        } else {
            self.gemm_pipelined(g, a, w, c, m, n, k)
        }
    }

    /// out = LayerNorm(x) * (1 + scale); `scale` may be NULL.
    #[allow(clippy::too_many_arguments)]
    pub fn layernorm_mod(
        &self,
        g: &dyn GpuBackend,
        x: DevicePtr,
        scale: DevicePtr,
        out: DevicePtr,
        tokens: usize,
        hidden: usize,
        eps: f32,
    ) -> Result<()> {
        KernelLaunch::new(g, self.layernorm_mod)
            .grid([tokens as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(scale)
            .arg_ptr(out)
            .arg_u32(hidden as u32)
            .arg_f32(eps)
            .launch(g.default_stream())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm(
        &self,
        g: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        out: DevicePtr,
        tokens: usize,
        hidden: usize,
        eps: f32,
        w_offset: f32,
    ) -> Result<()> {
        KernelLaunch::new(g, self.rms_norm)
            .grid([tokens as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(out)
            .arg_u32(hidden as u32)
            .arg_f32(eps)
            .arg_f32(w_offset)
            .launch(g.default_stream())
    }

    /// Per-head RMSNorm + RoPE of a projection `x[tokens, heads*d]` into
    /// `out[heads][out_rows][d]` at row offset `row_off`.
    #[allow(clippy::too_many_arguments)]
    pub fn norm_rope_heads(
        &self,
        g: &dyn GpuBackend,
        x: DevicePtr,
        w: DevicePtr,
        cos: DevicePtr,
        sin: DevicePtr,
        out: DevicePtr,
        tokens: usize,
        heads: usize,
        d: usize,
        row_off: usize,
        out_rows: usize,
        eps: f32,
        rope_mode: u32,
    ) -> Result<()> {
        debug_assert!(d <= 256 && d.is_multiple_of(32));
        KernelLaunch::new(g, self.norm_rope_heads)
            .grid([tokens as u32, heads as u32, 1])
            .block([d as u32, 1, 1])
            .arg_ptr(x)
            .arg_ptr(w)
            .arg_ptr(cos)
            .arg_ptr(sin)
            .arg_ptr(out)
            .arg_u32(heads as u32)
            .arg_u32(d as u32)
            .arg_u32(row_off as u32)
            .arg_u32(out_rows as u32)
            .arg_f32(eps)
            .arg_u32(rope_mode)
            .launch(g.default_stream())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn v_transpose(
        &self,
        g: &dyn GpuBackend,
        v: DevicePtr,
        vt: DevicePtr,
        tokens: usize,
        heads: usize,
        d: usize,
        col_off: usize,
        out_cols: usize,
    ) -> Result<()> {
        KernelLaunch::new(g, self.v_transpose)
            .grid([blocks(tokens * heads * d), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(v)
            .arg_ptr(vt)
            .arg_u32(tokens as u32)
            .arg_u32(heads as u32)
            .arg_u32(d as u32)
            .arg_u32(col_off as u32)
            .arg_u32(out_cols as u32)
            .launch(g.default_stream())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn copy_prefix(
        &self,
        g: &dyn GpuBackend,
        kc: DevicePtr,
        vc: DevicePtr,
        kh: DevicePtr,
        vt: DevicePtr,
        n: usize,
        src_rows: usize,
        dst_rows: usize,
        heads: usize,
        d: usize,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        KernelLaunch::new(g, self.copy_prefix)
            .grid([blocks(heads * n * d), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(kc)
            .arg_ptr(vc)
            .arg_ptr(kh)
            .arg_ptr(vt)
            .arg_u32(n as u32)
            .arg_u32(src_rows as u32)
            .arg_u32(dst_rows as u32)
            .arg_u32(heads as u32)
            .arg_u32(d as u32)
            .launch(g.default_stream())
    }

    /// GEMM-SDPA over prepared head-major buffers; writes `o[nq, hq*d]`.
    /// Row r attends keys `[0, limits[r])` (`limits` NULL → all `nk_valid`).
    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        g: &dyn GpuBackend,
        b: &AttnBufs,
        o: DevicePtr,
        limits: DevicePtr,
        nq: usize,
        nk_valid: usize,
        nk_pad: usize,
        hq: usize,
        hkv: usize,
        d: usize,
    ) -> Result<()> {
        let stream = g.default_stream();
        let scale = 1.0 / (d as f32).sqrt();
        let group = hq / hkv;
        for h in 0..hq {
            let q = b.qh.offset(h * nq * d * 2);
            let kv = h / group;
            let k = b.kh.offset(kv * nk_pad * d * 2);
            let vt = b.vt.offset(kv * d * nk_pad * 2);
            self.gemm_f32out(g, q, k, b.scores, nq, nk_pad, d)?;
            KernelLaunch::new(g, self.softmax_rows)
                .grid([nq as u32, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.scores)
                .arg_ptr(b.probs)
                .arg_ptr(limits)
                .arg_u32(nq as u32)
                .arg_u32(nk_pad as u32)
                .arg_u32(nk_valid as u32)
                .arg_f32(scale)
                .launch(stream)?;
            self.gemm_pipelined(g, b.probs, vt, b.o_stage, nq, d, nk_pad)?;
            KernelLaunch::new(g, self.scatter_head)
                .grid([blocks(nq * d), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(b.o_stage)
                .arg_ptr(o.offset(h * d * 2))
                .arg_u32(nq as u32)
                .arg_u32(d as u32)
                .arg_u32((hq * d) as u32)
                .launch(stream)?;
        }
        Ok(())
    }

    /// h += gate[col] * x over `[tokens, hidden]` (`gate` NULL → plain add).
    pub fn gated_add(
        &self,
        g: &dyn GpuBackend,
        h: DevicePtr,
        x: DevicePtr,
        gate: DevicePtr,
        tokens: usize,
        hidden: usize,
    ) -> Result<()> {
        let n = tokens * hidden;
        KernelLaunch::new(g, self.gated_add)
            .grid([blocks(n), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(h)
            .arg_ptr(x)
            .arg_ptr(gate)
            .arg_u64(n as u64)
            .arg_u32(hidden as u32)
            .launch(g.default_stream())
    }

    pub fn silu_mul(
        &self,
        g: &dyn GpuBackend,
        gate: DevicePtr,
        up: DevicePtr,
        out: DevicePtr,
        n: usize,
    ) -> Result<()> {
        KernelLaunch::new(g, self.silu_mul)
            .grid([blocks(n), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(gate)
            .arg_ptr(up)
            .arg_ptr(out)
            .arg_u64(n as u64)
            .launch(g.default_stream())
    }

    pub fn gelu_tanh(&self, g: &dyn GpuBackend, x: DevicePtr, n: usize) -> Result<()> {
        KernelLaunch::new(g, self.gelu_tanh)
            .grid([blocks(n), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_u64(n as u64)
            .launch(g.default_stream())
    }

    /// x[row, col] += bias[col] (f32 bias).
    pub fn bias_add(
        &self,
        g: &dyn GpuBackend,
        x: DevicePtr,
        bias: DevicePtr,
        rows: usize,
        cols: usize,
    ) -> Result<()> {
        let n = rows * cols;
        KernelLaunch::new(g, self.bias_add)
            .grid([blocks(n), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(bias)
            .arg_u64(n as u64)
            .arg_u32(cols as u32)
            .launch(g.default_stream())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn im2col3x3(
        &self,
        g: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        h: usize,
        w: usize,
        c: usize,
        row0: usize,
        rows: usize,
    ) -> Result<()> {
        KernelLaunch::new(g, self.im2col3x3)
            .grid([blocks(rows * w * 9 * c), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(out)
            .arg_u32(h as u32)
            .arg_u32(w as u32)
            .arg_u32(c as u32)
            .arg_u32(row0 as u32)
            .arg_u32(rows as u32)
            .launch(g.default_stream())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vae_norm(
        &self,
        g: &dyn GpuBackend,
        x: DevicePtr,
        gamma: DevicePtr,
        out: DevicePtr,
        pixels: usize,
        c: usize,
        silu: bool,
    ) -> Result<()> {
        let block = (c.div_ceil(32) * 32).min(256) as u32;
        KernelLaunch::new(g, self.vae_norm)
            .grid([pixels as u32, 1, 1])
            .block([block, 1, 1])
            .arg_ptr(x)
            .arg_ptr(gamma)
            .arg_ptr(out)
            .arg_u32(c as u32)
            .arg_u32(silu as u32)
            .launch(g.default_stream())
    }

    pub fn upsample2x(
        &self,
        g: &dyn GpuBackend,
        x: DevicePtr,
        out: DevicePtr,
        h: usize,
        w: usize,
        c: usize,
    ) -> Result<()> {
        KernelLaunch::new(g, self.upsample2x)
            .grid([blocks(4 * h * w * c), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(out)
            .arg_u32(h as u32)
            .arg_u32(w as u32)
            .arg_u32(c as u32)
            .launch(g.default_stream())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dup_shortcut_add(
        &self,
        g: &dyn GpuBackend,
        x: DevicePtr,
        src: DevicePtr,
        h: usize,
        w: usize,
        cin: usize,
        cout: usize,
        factor: usize,
        off: usize,
        repeats: usize,
    ) -> Result<()> {
        KernelLaunch::new(g, self.dup_shortcut_add)
            .grid([blocks(4 * h * w * cout), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(x)
            .arg_ptr(src)
            .arg_u32(h as u32)
            .arg_u32(w as u32)
            .arg_u32(cin as u32)
            .arg_u32(cout as u32)
            .arg_u32(factor as u32)
            .arg_u32(off as u32)
            .arg_u32(repeats as u32)
            .launch(g.default_stream())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn vae_qkv_split(
        &self,
        g: &dyn GpuBackend,
        qkv: DevicePtr,
        q: DevicePtr,
        k: DevicePtr,
        vt: DevicePtr,
        pixels: usize,
        c: usize,
        k_rows: usize,
    ) -> Result<()> {
        KernelLaunch::new(g, self.vae_qkv_split)
            .grid([blocks(pixels * c), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(q)
            .arg_ptr(k)
            .arg_ptr(vt)
            .arg_u32(pixels as u32)
            .arg_u32(c as u32)
            .arg_u32(k_rows as u32)
            .launch(g.default_stream())
    }

    /// Single-head softmax for the VAE attention (no limits).
    #[allow(clippy::too_many_arguments)]
    pub fn softmax_rows(
        &self,
        g: &dyn GpuBackend,
        scores: DevicePtr,
        probs: DevicePtr,
        nq: usize,
        nk_pad: usize,
        nk_valid: usize,
        scale: f32,
    ) -> Result<()> {
        KernelLaunch::new(g, self.softmax_rows)
            .grid([nq as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(scores)
            .arg_ptr(probs)
            .arg_ptr(DevicePtr::NULL)
            .arg_u32(nq as u32)
            .arg_u32(nk_pad as u32)
            .arg_u32(nk_valid as u32)
            .arg_f32(scale)
            .launch(g.default_stream())
    }
}

/// Weight uploads of one pipeline load. Dropping it before [`finish`]
/// (a failed load) frees everything uploaded so far, so a retry does not
/// stack a second copy on top of a leaked one.
///
/// [`finish`]: WeightSet::finish
pub(crate) struct WeightSet<'a> {
    g: &'a dyn GpuBackend,
    ptrs: Vec<DevicePtr>,
    pub bytes: usize,
}

impl<'a> WeightSet<'a> {
    pub fn new(g: &'a dyn GpuBackend) -> Self {
        Self {
            g,
            ptrs: Vec::new(),
            bytes: 0,
        }
    }

    pub fn upload(&mut self, raw: &[u8]) -> Result<DevicePtr> {
        let p = self.g.alloc(raw.len())?;
        self.ptrs.push(p);
        self.g.copy_h2d(raw, p)?;
        self.bytes += raw.len();
        Ok(p)
    }

    /// A non-weight allocation that shares the set's lifetime.
    pub fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        let p = self.g.alloc(bytes)?;
        self.ptrs.push(p);
        Ok(p)
    }

    /// The load succeeded: hand the allocations to their long-term owner.
    pub fn finish(mut self) -> Vec<DevicePtr> {
        std::mem::take(&mut self.ptrs)
    }
}

impl Drop for WeightSet<'_> {
    fn drop(&mut self) {
        for p in self.ptrs.drain(..) {
            let _ = self.g.free(p);
        }
    }
}

/// Device allocations owned by one forward pass; freed together.
pub(crate) struct Pool<'a> {
    g: &'a dyn GpuBackend,
    live: Vec<DevicePtr>,
}

impl<'a> Pool<'a> {
    pub fn new(g: &'a dyn GpuBackend) -> Self {
        Self {
            g,
            live: Vec::new(),
        }
    }

    pub fn alloc(&mut self, bytes: usize) -> Result<DevicePtr> {
        let p = self.g.alloc(bytes.max(16))?;
        self.live.push(p);
        Ok(p)
    }

    pub fn alloc_zeroed(&mut self, bytes: usize) -> Result<DevicePtr> {
        let p = self.alloc(bytes)?;
        self.g.memset(p, 0, bytes.max(16))?;
        Ok(p)
    }

    pub fn upload(&mut self, bytes: &[u8]) -> Result<DevicePtr> {
        let p = self.alloc(bytes.len())?;
        self.g.copy_h2d(bytes, p)?;
        Ok(p)
    }

    /// Free one buffer early (large intermediates).
    pub fn free(&mut self, p: DevicePtr) {
        if let Some(i) = self.live.iter().position(|x| *x == p) {
            self.live.swap_remove(i);
            let _ = self.g.free(p);
        }
    }
}

impl Drop for Pool<'_> {
    fn drop(&mut self) {
        let _ = self.g.synchronize(self.g.default_stream());
        for p in self.live.drain(..) {
            let _ = self.g.free(p);
        }
    }
}
