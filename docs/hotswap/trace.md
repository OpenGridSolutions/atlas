# Hot-swap MVP — Task 1 code trace

Read-only trace of the weight load path, produced for `HOTSWAP_MVP_PLAN.md` Task 1.
No behaviour was changed. Line numbers refer to the tree at the time of writing
(branch `modifications`, after the Task 0 edits).

---

## 0. Executive summary

Six findings, in descending order of impact on Task 2.

1. **`WeightTensor` owns nothing.** It is `{ ptr: DevicePtr(u64), shape, dtype }` with
   no `Drop`. `WeightStore` has no `Drop` either. The per-tensor `cuMemAlloc_v2`
   allocations made by the loader are **never freed** — they leak by design and live
   for the process lifetime. There is nothing to fight for pointer stability.

2. **Kernels receive raw `u64`.** `KernelArg::Buffer(DevicePtr)` is flattened to
   `p.0` before launch (`gpu.rs:155`). An interior pointer `arena_base + offset` is
   therefore *transparent* to every kernel launch — zero plumbing changes. The KV
   cache already does exactly this (`paged_impl.rs:208-221`).

3. **`build_model` copies `DevicePtr` values out of the store.** The model captures
   pointers by value into `DenseWeight`/`QuantizedWeight`/… at build time. This fixes
   the migration point: **the arena migration must run between `load_weight_store()`
   and `build_model()`** (`serve.rs:210` → `:351`). Re-pointing the store afterwards
   is a no-op. The good news: **no `spark-model` call site needs to change.**

4. **The blocking hazard: 36 sites call `gpu.free()` on store-owned pointers.**
   `cuMemFree_v2` requires an *allocation base*; an interior arena pointer returns
   `CUDA_ERROR_INVALID_VALUE`. One of these (`nvfp4_detect.rs:276`) propagates with
   `?` and would hard-fail the boot. None of them fire for the MVP model
   (Qwen3-Next-80B NVFP4) — verified below — but the arena cannot be considered safe
   until they are neutralised.

5. **The 3.17 GB accounting gap is the UVM fallback, not the buffered read path.**
   Both read strategies share one allocation site. The `>5000`-tensor cap chooses
   `O_DIRECT` vs buffered `pread` and *nothing else*. See §2.

6. **`build_model` makes ~178 allocations the arena does not cover** (transposes,
   dequant, predequant, re-quant). They are not in the snapshot and, because nothing
   frees them, they will leak once per cycle. **This is the main threat to Task 5's
   "MemFree drift < 1 GB over 100 cycles"**, and it is not addressed by the plan as
   written. See §5.

---

## 1. (a) Types and ownership of `WeightStore` / `WeightTensor`

`crates/spark-runtime/src/weights.rs:86-159`:

```rust
pub struct WeightTensor { pub ptr: DevicePtr, pub shape: Vec<usize>, pub dtype: WeightDtype }
pub struct WeightStore  { weights: HashMap<String, WeightTensor> }   // field is PRIVATE
```

`DevicePtr(pub u64)` — `crates/spark-runtime/src/gpu.rs:37-50`. A raw `CUdeviceptr`,
`Copy`, with `offset(bytes) -> Self(self.0 + bytes)` and `NULL` / `is_null()`.

**It is not a cudarc type.** There is no `CudaSlice`, no RAII device buffer, no
lifetime tying the pointer to an owner. `spark-model` and `spark-server` deliberately
do not depend on cudarc at all (comments at `spark-model/Cargo.toml:41,77`).

### Who allocates

| Path | Site | API |
|---|---|---|
| Fast loader (default) | `fast_weights/mod.rs:359` | `gpu.alloc(len)` → `cuMemAlloc_v2` |
| Fast loader, on alloc failure | `fast_weights/mod.rs:373` | `gpu.alloc_managed(len)` → `cuMemAllocManaged` |
| Slow loader, sharded | `weights/loader/load_fns.rs:127` / `:142` | same pair |
| Slow loader, single file | `weights/loader/load_fns.rs:206` | `gpu.alloc` — **no managed fallback** |

Backend: `crates/spark-runtime/src/cuda_backend/gpu_impl.rs:49-88`. `alloc` =
`cuMemAlloc_v2`, `alloc_managed` = `cuMemAllocManaged(CU_MEM_ATTACH_GLOBAL)`,
`free` = `cuMemFree_v2`.

One `cuMemAlloc_v2` **per tensor**. So today every `WeightTensor.ptr` is guaranteed to
be the base of its own allocation — an invariant the arena will break, and which §4
shows is load-bearing in 36 places.

### Who frees / where it is dropped

Nowhere. Evidence:

- No `impl Drop for WeightStore` and no `impl Drop for WeightTensor` (grep across
  `crates/` returns nothing). The only weight-adjacent `Drop` is `AlignedBuffer`, a
  *host* staging buffer (`fast_weights/direct_io.rs:52`).
- No loop over `store` values calling `gpu.free`.
- No `drop(store)` anywhere. The `store` local declared at
  `spark-server/src/main_modules/serve.rs:210` is only ever borrowed (`&store` at
  `:221, :231, :239, :265, :351`) and lives in the `serve()` frame for the whole
  process.

So the loader's allocations are simultaneously (i) reachable via the store, (ii)
aliased by the model structs, and (iii) never released.

**Consequence for Task 4 (`unload`).** "Drop the model index only" is already the
status quo for the loader's allocations — dropping the `HashMap` frees no GPU memory.
Task 2's "drop the original per-tensor allocations" therefore requires *adding* an
explicit free loop that does not exist today. `cuMemFree_v2` is the correct
deallocator for both `cuMemAlloc_v2` and `cuMemAllocManaged` results, so a single
`gpu.free()` loop over the pre-migration pointers is sufficient and does not need to
record allocation provenance.

---

## 2. The 3.17 GB accounting gap and the `>5000`-tensor "buffered+pipelined" path

**The buffered path allocates through exactly the same API as the O_DIRECT path.**

`fast_weights/mod.rs:302` computes `wants_direct = try_direct_io && tensors.len() <=
direct_io_tensor_cap` (cap = `DEFAULT_DIRECT_IO_TENSOR_CAP` = 5000, `:64`). That flag
selects only *how bytes reach a host staging buffer*:

- `true` → `direct_io::open_direct(shard_path)` (O_DIRECT fd)
- `false` → plain `File::open` + kernel readahead (`:326`), optional `posix_fadvise`

Both then feed the **same** reader thread → `sync_channel` → **one** copier loop at
`fast_weights/mod.rs:354-389`, whose only allocation is:

```rust
let ptr = match gpu.alloc(meta.len) {          // :359  cuMemAlloc_v2
    Ok(p)  => { gpu.copy_h2d(src, p)?; p }
    Err(_) => {                                 // :364  first failure flips offload_logged
        let p = gpu.alloc_managed(meta.len)?;   // :373  cuMemAllocManaged  → UVM
        unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), p.0 as *mut u8, meta.len) };
        p
    }
};
```

There is no second allocator, and `direct_io.rs` allocates only host memory
(`std::alloc::alloc`, `:36`). So the read strategy cannot explain a device-memory
accounting difference.

### What does explain it

`cuMemAllocManaged` pages are not counted as device-resident by `cuMemGetInfo_v2` /
`nvidia-smi memory.used` until they migrate. They *are* the `uvm_va_range_external_t`
slab objects named as the leak canary in plan §3. So the reading is:

> ~3.17 GB of tensors landed via `cuMemAlloc_v2` before the first allocation failure;
> every tensor after that fell back to `cuMemAllocManaged` and is invisible to GPU
> memory accounting.

**How to confirm on the box** — the fallback logs exactly once, naming the tensor and
size at which it flipped (`fast_weights/mod.rs:366`):

```
GPU alloc failed for <name> (<bytes> bytes) — switching to managed (UVM) memory
```

```sh
grep -m1 "switching to managed" <boot log>
```

- **Line present** → confirmed. The tensor named is the 3.17 GB watermark.
- **Line absent** → no fallback happened, all 47 GB went through `cuMemAlloc_v2`, and
  the gap is a pure `cuMemGetInfo`/`nvidia-smi` artefact on coherent UMA. In that case
  `memory.used` is not a usable metric on this box and the Task 2 acceptance criterion
  "latency and tokens/s within 5%" should be the load-bearing check, not memory
  accounting. (I could not discriminate these two from source alone; the log line is
  decisive and costs one grep.)

### A real bug found while tracing

`fast_weights/mod.rs:230`:

```rust
if !offload_logged {
    check_oom_guard(gpu, oom_reserve_bytes, ...)?;   // per-shard safety net
}
```

Once **any** tensor falls back to managed memory, `offload_logged` is `true` and the
per-shard OOM guard is **skipped for every remaining shard**. The guard is disabled
precisely in the situation it exists for.

This matters for the Task 0 change I just landed: my rationale (and the comment I added
to `serve.sh` and `weights.rs`) says "the per-shard OOM guard and the UVM fallback
remain active as the real safety nets". That is true only until the first fallback —
after which the *only* remaining net is the UVM fallback plus Linux swap. Not a
regression introduced by Task 0 (the code was always like this), but the claim as I
wrote it is too strong and should be corrected before that comment is taken as gospel.
Worth a separate small fix; out of scope for Task 1.

---

## 3. (b) Where `spark-model` consumes raw tensor pointers

All consumption is **load-time only**. The forward pass never touches `WeightStore`; it
reads `DevicePtr`s already captured in `DenseWeight` / `QuantizedWeight` / `Fp8Weight` /
`ExpertWeight` structs (`weight_map/quantized.rs:71,179,188,213,234`,
`weight_map/expert.rs:18,79,190`).

Measured counts (`crates/spark-model/src/`):

| Pattern | Count | Files |
|---|---|---|
| `store.get(...)` | 67 | 21 |
| `.ptr` field reads | 83 | ~17 |
| `ptr(store, name)` helper | 11 | — |
| `gpu.alloc(...)` (new buffers during build) | 178 | — |
| `gpu.free(...)` | 104 | — |

Distinct **logical** read sites: **~70–80** (the `get` and `.ptr` populations overlap
heavily — `store.get(name)?.ptr` is one line, two hits).

Concentrated in two directories:
- `crates/spark-model/src/weight_map/**` — `model_a.rs` (14 `get` / 18 `.ptr`),
  `ssm_qwen35.rs` (7/9), `quant_helpers.rs` (5/9), `ssm_qwen35_more.rs` (6/6)
- `crates/spark-model/src/weight_loader/**` — `deepseek_v4/assemble.rs` (5/9),
  `step3p7/load_layers.rs` (5/5), `step3p7.rs` (4/4), `qwen35.rs`, `nemotron.rs`,
  `minimax.rs`, `gemma4/loader_a.rs`

The single chokepoint accessor is `weight_map/model_a.rs:26`:

```rust
pub(crate) fn ptr(store: &WeightStore, name: &str) -> Result<DevicePtr> { Ok(store.get(name)?.ptr) }
```

Two consumption shapes:

- **View / alias** (retain the loader's pointer, no copy) — the common case for BF16
  dense weights, NVFP4 packed weight + scale, norms, embeddings:
  `model_a.rs:121-123` `dense()` → `DenseWeight { weight: w.ptr }`;
  `model_a.rs:109-118` `quantized()` → `QuantizedWeight { weight: ptr(..), weight_scale: ptr(..), .. }`.
- **Copy / rewrite** (fresh `gpu.alloc`, original left in the store):
  FP32→BF16 (`model_a.rs:156-158`), FP8-block→BF16 dequant (`model_a.rs:175`),
  NVFP4 transpose-for-prefill (`quantized.rs:103-145`), NVFP4→FP8 predequant
  (`quantized.rs:152-174`), FP8 transpose (`quantized.rs:247-294`), LM-head re-quant
  (`factory/lm_head_setup.rs`).

The loader's own pre-flight comment already names this split — *"NVFP4 (Sehyo): ~2.0x
(store **aliased** + transposed/predequant copies)"* (`weights/loader.rs:71-79`).

### Kernel launch (part of (e), answered here)

`GpuBackend::launch(..., params: &mut [*mut c_void])` — `gpu.rs:124-132`. The typed
layer `KernelArg::Buffer(DevicePtr)` exists only so Metal can distinguish buffers from
scalars; the CUDA `launch_typed` default flattens every buffer to its raw `u64`:

```rust
KernelArg::Buffer(p) => storage.push(p.0),      // gpu.rs:155
```

`KernelLaunch::arg_ptr(p)` likewise pushes `p.0` (`kernel_args.rs:96-103`). Example:
`layers/ops/gemm_quant.rs:21-42` passes the weight `DevicePtr` straight through.

**Kernels receive raw device pointers, not typed handles.** An `arena_base + offset`
interior pointer needs no changes anywhere in the launch path.

---

## 4. (c) The minimal seam, and what blocks it

### Where the seam goes

```
serve.rs:210   let store = load_weight_store(...)           // WeightStore, ptrs = cuMemAlloc bases
serve.rs:221   auto_detect_weight_prefix(&store, ...)
serve.rs:231   preflight(&store, ...)
serve.rs:239   detect_quant_format(&config, &store)
        ▲
        └───── ✦ ARENA MIGRATION GOES HERE ✦
                 (after the store exists, before anything reads .ptr)
        ▼
serve.rs:351   build_model(&store, ...)                     // copies DevicePtr BY VALUE into model structs
```

`build_model(config, store: &WeightStore, ...)` (`factory/build.rs:24-26`) borrows the
store and never retains it. Because the model **copies pointer values out**, mutating
the store after `build_model` changes nothing. Migration strictly precedes build.

### The seam itself

Add to `WeightStore` (`weights.rs`) — the `weights` field is private, and today there
is no mutable accessor at all (`get`, `contains`, `len`, `names`, `total_bytes`,
`has_fp8_weights` only):

```rust
pub(crate) fn iter_mut(&mut self) -> impl Iterator<Item = (&String, &mut WeightTensor)>
```

Then, in `weights_arena.rs`:

```rust
// deterministic order: sort by name (or by original ptr) — NOT HashMap iteration order,
// which is nondeterministic and would make snapshots non-reproducible.
for (name, t) in store.iter_mut_sorted() {
    let off = arena.alloc(t.byte_size(), 256)?;
    gpu.copy_d2d(t.ptr, arena.base().offset(off), t.byte_size())?;
    manifest.push(Entry { name, offset: off, len: t.byte_size(), dtype: t.dtype, shape: t.shape.clone() });
    old_ptrs.push(t.ptr);
    t.ptr = arena.base().offset(off);          // re-point
}
gpu.device_synchronize()?;                      // ← does not exist yet, see §6
for p in old_ptrs { gpu.free(p)?; }
```

**Determinism note.** `WeightStore` is a `HashMap`; iteration order varies per process.
The plan asks for "a deterministic order (execution order)". Sort by tensor name — it
is stable, reproducible across runs, and independent of load order, which matters
because the snapshot's `model_version_hash` covers the arena layout version.

### Blast radius

| Change | Files |
|---|---|
| `WeightStore::iter_mut` (+ sorted variant) | `weights.rs` |
| New `WeightsArena` | `weights_arena.rs` (new) |
| Migration call + `--weights-arena-gb` flag | `serve.rs`, `cli.rs` |
| `GpuBackend::device_synchronize()` | `gpu.rs`, `cuda_backend.rs`, `cuda_backend/gpu_impl.rs`, `gpu/mock.rs` |
| Free-guard for arena pointers (§4.1) | `gpu.rs` or `weights_arena.rs` + 36 call sites *audited*, few changed |

**Zero changes to the ~70–80 `spark-model` read sites.** They read `.ptr` as a value
and are agnostic to whether it is a base or an interior pointer.

### 4.1 The blocking hazard — `cuMemFree_v2` on interior pointers

`cuMemFree_v2` requires an allocation base. Passing `arena_base + offset` returns
`CUDA_ERROR_INVALID_VALUE`. **36 sites free store-owned pointers.** The dangerous ones:

| Site | Frees | Error handling | Fires for Qwen3-Next-NVFP4? |
|---|---|---|---|
| `weight_map/nvfp4_detect.rs:276` | `gpu.free(w.ptr)` where `w = store.get(..)` | **`?` → hard fail** | No — `Bf16Raw` arm only |
| `weight_map/ssm_qwen35.rs:286,289` | `gpu.free(w.ptr)` (fused MoE source) | `let _ =` → **silent** | No — fused-BF16 source only |
| `layers/moe/helpers_a.rs:208-231, 254-265` | `gpu.free(expert.*.weight)` (aliased store ptrs) | `?` | No — see below |
| `weight_loader/nemotron.rs:169-278` | `gpu.free(q.weight)` etc. | `?` | No — nemotron only |
| `weight_loader/qwen35/load_layers/linear_attn_arms.rs:230-231` | `gpu.free(ssm35.in_proj_*.weight)` | `let _ =` | No — qwen3.5 |

Verified gating for the MVP model:

- `factory/m2_setup.rs:18-20` early-returns unless `model_type` is `minimax_m2` or
  `step3p7`, so the whole MoE-transpose free path (`helpers_a.rs`) is dead for
  `qwen3_next`.
- `nvfp4_detect.rs:276` lives in the `Bf16Raw` arm of `quantized_any`; an on-disk NVFP4
  checkpoint takes the aliased arm and never frees.

So **Task 2 will pass its acceptance test on Qwen3-Next-80B-A3B-NVFP4 without touching
any of these.** It will corrupt or hard-fail on MiniMax-M2, step3p7, Nemotron, Qwen3.5
and any BF16-source MoE. Two mitigations, cheapest first:

1. **Guard the free.** Give the arena an `owns(ptr) -> bool` (base ≤ ptr < base+used)
   and make `gpu.free()` — or a new `free_weight()` shim — a no-op for arena-backed
   pointers. ~1 site if routed through `GpuBackend::free`, plus an audit of the 36.
2. **Assert, don't silence.** Add a debug assertion that any pointer handed to
   `gpu.free` is *not* inside the arena, and fix the call sites to skip explicitly.
   Louder, better long-term, more churn.

Recommend (1) for the MVP with a `tracing::debug!` on each suppressed free, and a
follow-up issue for (2).

Note the ordering luck: migration runs *before* `build_model`, so it never reads the
pointers that `nvfp4_detect.rs:276` later frees. The store does keep dangling pointers
after those frees today (nothing re-reads them) — pre-existing, not introduced here.

### 4.2 Alignment

No code asserts weight-pointer alignment, and there is no per-tensor alignment constant.
Alignment today is implicitly whatever `cuMemAlloc_v2` returns (≥ 256 B). Kernels do
vectorised loads. A misaligned arena offset would surface as a silent wrong result or a
misaligned-address fault, **not** a clean error — so the arena's `alloc(len, align=256)`
must be treated as a correctness requirement, not a performance tweak.

Nothing calls `cuPointerGetAttribute` / `cudaPointerGetAttributes` anywhere in the
workspace, so no code introspects base-ness/managed-ness. That class of breakage does
not exist.

### 4.3 Precedent to imitate

- `kv_cache/paged_impl.rs` is the in-tree proof that interior pointers work end to end:
  slabs allocated once in `new()` (`:26-27`), blocks handed out as
  `k_pool.offset(block_idx * stride)` (`:208-221`), released by pushing an index back
  onto a free list with **no `gpu.free`** (`:161-195`), "reset" = `memset_async`
  (`:89-112`). This is exactly the arena/`reset()` model.
- `buffers.rs` `BufferArena` is **not** a bump allocator — it does one `gpu.alloc` per
  field (`:116-171`). Its useful conventions are the `DevicePtr::NULL`-means-disabled
  idiom (`:147-169`) and `BufferSizes` as SSOT. Imitate its *API shape*, not its
  allocation strategy.
- `factory/m2_setup.rs:138-183` already builds device-side pointer tables as
  `scratch_packed.0 + local_idx * stride` — an existing interior-pointer pattern.

---

## 5. Unaccounted allocations — the real risk to Task 5

`build_model` performs **178 `gpu.alloc` calls** in `spark-model` (transposes,
dequant, predequant, re-quant, LM-head, vision encoder, DFlash head, SSM pools). These
buffers:

- are **not in the arena**, so they are **not in the snapshot** (Task 3), and
- are **not freed** on model drop (no `Drop`, no free loop), so they **leak once per
  load/unload cycle** (Task 4).

Task 5's acceptance criterion is *"100-cycle loop shows … `MemFree` drift < 1 GB"*. With
the plan as written, each cycle leaks the derived-buffer working set. The pre-flight
overhead multiplier (1.3× NVFP4 / 1.5× FP8, `weights/loader.rs:77-79`) is a direct
estimate of their size: for a 47 GB checkpoint that is ~14 GB of derived buffers per
cycle. The loop will OOM long before 100 iterations.

This is not addressed anywhere in `HOTSWAP_MVP_PLAN.md`. Options for Task 4:

1. Route model-build allocations through a **second arena** (the plan already requires
   arena count to be a parameter — this is what that parameter is for), reset on unload.
2. Add an **allocation registry**: `GpuBackend` records every `alloc` made after a
   watermark; `unload` walks and frees it. Simple, no call-site changes, but must not
   free arena-backed pointers (same guard as §4.1).

Recommend deciding this **before** Task 2 lands, because it determines whether
`WeightsArena` should be a general bump allocator with multiple instances (it should)
or a weights-only special case.

Related: `--dflash` loads a **second `WeightStore`** for the drafter
(`serve_phases/weights.rs:110-156`, `serve.rs:339`). Arena count must accommodate it.

---

## 6. (e) cudarc: what exists, what is missing

### The framing is off, in a helpful way

**`spark-runtime` does not use cudarc for memory, streams, or events.** All of it is a
hand-written `unsafe extern "C"` block in `cuda_backend.rs:18-69`, linking `libcuda`
directly. cudarc is pulled in for exactly one thing in this crate: `LaunchConfig`
(`gpu_impl.rs:37`). `atlas-core` is the real cudarc consumer (modules, functions,
streams: `registry.rs:21-22`, `kernel.rs:5-8`). `spark-model` / `spark-server` do not
depend on it at all.

So the question "can cudarc wrap an mmap region as a device-accessible buffer" resolves
to: **we do not need cudarc to do anything.** The arena work extends the existing extern
block in `cuda_backend.rs` — which plan §6 explicitly permits ("Vendored CUDA wrapper
(extend if APIs missing)") and which matches house style. No `vendor/cudarc` edits
required.

### Already declared in `cuda_backend.rs:18-69`

`cuMemAlloc_v2`, `cuMemFree_v2`, `cuMemAllocManaged`, `cuMemAllocHost_v2`,
`cuMemFreeHost`, `cuMemcpyHtoDAsync_v2`, `cuMemcpyDtoHAsync_v2`, `cuMemcpyDtoDAsync_v2`,
`cuMemsetD8Async`, `cuMemGetInfo_v2`, `cuStreamCreate`, `cuStreamSynchronize`,
`cuStreamWaitEvent`, `cuEventCreate`, `cuEventRecord`, `cuEventDestroy_v2`,
`cuCtxGetCurrent`, `cuCtxSetCurrent`, graph capture/launch.

### Gap list for the arena

| Symbol | Needed for | In `vendor/cudarc` | In `cuda_backend.rs` | Action |
|---|---|---|---|---|
| `cuCtxSynchronize` | Task 2 & 4 `cudaDeviceSynchronize()` | wrapped | **ABSENT** | **Add.** `GpuBackend` has no device-wide sync — only per-stream `cuStreamSynchronize`. Blocking for Task 2. |
| `cuMemHostRegister_v2` | optional `arena_host_register` flag | raw only (`driver/sys/mod.rs:11512`), no safe wrapper | **ABSENT** | Add to extern block |
| `cuMemHostUnregister` | teardown of the above | raw only (`:11517`) | **ABSENT** | Add |
| `cuMemHostGetDevicePointer_v2` | non-ATS fallback | raw only (`:11503`) | **ABSENT** | Probably unnecessary on GB10 (host VA == device VA under ATS); add only if the spike shows otherwise |
| `cuPointerGetAttribute` | validate the arena is GPU-addressable; robust `arena.owns(ptr)` | raw only (`:12199`) | **ABSENT** | Add — cheap boot-time assertion of the whole ATS premise |
| `cuMemAdvise`, `cuMemPrefetchAsync` | ATS placement hints | wrapped (`result.rs:748,771`) | **ABSENT** | Optional |
| `cuEventSynchronize`, `cuEventElapsedTime` | migration timing | wrapped | **ABSENT** | Not needed — `std::time::Instant` suffices |
| VMM: `cuMemCreate`, `cuMemMap`, `cuMemAddressReserve`, `cuMemSetAccess` | principled stable-`arena_base`-with-remappable-backing | raw only (`:11410,11523,11330,11650`) | **ABSENT** | Not for MVP. Note for dual-residency follow-up: VMM is the *correct* primitive for pointer-stable rebacking, better than mmap + reset. |

Nothing in the workspace calls `cuMemHostRegister`/`cudaHostRegister` today, and nothing
mmaps a buffer the GPU reads directly. `memmap2` is used only to mmap safetensors on the
host and then `cuMemcpyHtoDAsync` into device memory (`weights/loader/load_fns.rs:93`).
`alloc_host_pinned` (`gpu.rs:305`) is implemented with `cuMemAllocHost_v2`
(`gpu_impl.rs:445-452`) — CUDA owns that allocation; it is *not* mmap + register.

`cufile-sys` exists and is real (dlopen over `libcufile`: `cuFileDriverOpen`,
`cuFileHandleRegister`, `cuFileBufRegister`, `cuFileRead`/`Write`), but is confined to
`spark-storage`'s bench/probe path (`bench.rs:54-98`, `probe.rs:95-124`). No serving
code depends on it. It is available as the plan §6 fallback.

cudarc is built with `fallback-dynamic-loading` and `cuda-version-from-build-system`
(root `Cargo.toml:50-56`); it `dlopen`s `libcuda.so`, is arch-agnostic, and has no
aarch64/sbsa special-casing (`vendor/cudarc/build.rs`). CUDA 11.4–13.1 supported.

### The load-bearing unknown

The entire arena design rests on: **the GPU can read plain anonymous `mmap` memory
coherently via ATS, and a host VA is directly usable as a kernel argument.** Nothing in
this repo exercises that today. Combined with §3's finding that kernels take raw `u64`,
this reduces to a single testable claim.

**Do this before writing `weights_arena.rs`** (plan §6: "If a platform assumption fails
… STOP, write `docs/hotswap/findings.md`"):

```
mmap 1 GiB anonymous → write a known pattern from the host
→ pass the host VA as a kernel arg to any existing elementwise kernel
→ copy_d2h and verify, then verify the kernel's writes are visible to the host
→ repeat with cuMemHostRegister to see whether it changes anything
```

If that fails, the fallbacks in order are: `cuMemAllocManaged` for the arena (loses the
"prefer plain mmap" preference but keeps every other property), then VMM, then cuFile.

---

## 7. (d) Tokenizer, chat template, sampling defaults — ownership

Everything the HTTP handlers need is packed into one `AppState`
(`spark-server/src/main_modules/app_state.rs:17-97`), wrapped `Arc::new(...)` at
`serve.rs:607` and moved into axum via `.with_state(state)`
(`serve_router.rs:130`). The **model** is a `Box<dyn Model>` moved into a dedicated OS
thread running `scheduler::run(...)` (`serve.rs:564-590`, `scheduler/mod.rs:110-134`).
The only handle back to it is an `mpsc::Sender<InferenceRequest>` (`app_state.rs:21`).

There is **no `Arc<RwLock>`, no `Mutex`, no `ArcSwap`** around the model or the
tokenizer — `arc_swap` is not even a workspace dependency.

| Item | Where | Owner | Replaceable at runtime? |
|---|---|---|---|
| Tokenizer (`ChatTokenizer`) | `tokenizer.rs:67-79`, built `serve.rs:422-428` | **by value** in `AppState.tokenizer` (`app_state.rs:18`) | **No** — immutable field of `Arc<AppState>`, consumed by `.with_state` |
| Chat template | `tokenizer.rs:73-78` (`minijinja::Environment<'static>` inside `ChatTokenizer`); loaded `tokenizer/chat_impl.rs:32-64` | inside the tokenizer | **No** — inherits the tokenizer's immutability |
| Sampling defaults | `app_state.rs:27-35`, from `generation_config.json` via `serve_phases/runtime.rs:54-94` | plain `f32`/`u32` on `AppState` | **No** — no interior mutability |
| `sampling_presets` | `app_state.rs:57`, from build-time `ptx_set.sampling` (`serve.rs:105`) | `AppState` | **No** — and it is per-kernel-target, not per-model |
| EOS token ids | built `serve_phases/runtime.rs:15-44` | **moved into the scheduler thread** (`serve.rs:568`, `scheduler/mod.rs:113`) | **No** — lives in that thread's frame |
| Think tokens (`think_end/start_token_id`) | `app_state.rs:45,53`, derived `tokenizer_runtime.rs:75-88` | `AppState` | **No** |
| Numeric / boundary / mid-word token masks | `scheduler/helpers.rs:378-439` | **`OnceLock<Arc<[bool]>>` module globals** | **No — write-once.** Setters do `let _ = X.set(mask)`; a second call after boot is a **silent no-op** |
| Watchdog params, loop-watchdog flag | `scheduler/helpers.rs:359-370, 204-216` | `OnceLock` | **No** — same silent no-op |
| `im_start` / tool-response hard stops | `scheduler/helpers.rs:25-45` | `AtomicU32` | **Yes** — `store()` works |
| `model_ready` | `app_state.rs:63`, `serve.rs:596` | `Arc<AtomicBool>` | **Yes** — usable as the drain/not-ready gate |

**Implications for Tasks 4/5** (beyond the MVP's single-model reload, but they bite as
soon as `POST /admin/models/load {"model": ...}` targets a *different* model):

- Reloading the **same** model needs none of this to change: tokenizer, template, EOS
  and masks stay valid.
- Reloading a **different** model requires new interior-mutability seams:
  (a) `AppState.tokenizer` + template + sampling defaults → `ArcSwap`/`RwLock`;
  (b) the three `OnceLock` masks in `scheduler/helpers.rs` → `ArcSwap` or per-scheduler
  state (currently a second `set()` is silently dropped — this will produce *wrong
  sampling* rather than an error);
  (c) a channel/mechanism to hand a new `Box<dyn Model>` to the scheduler thread, which
  today owns it exclusively for process life.

The scheduler has **no unload / reload / pause API** and no "no model loaded" branch; it
exits only when the request channel closes (`scheduler/mod.rs:523-536`). The only
existing not-ready notion is `model_ready` (read at `api/misc_handlers.rs:132`).

There are **no `/admin` endpoints and no admin router** today. The full router is
`serve_router.rs:44-130`: `/v1/*`, `/tokenize`, `/detokenize`, `/health`,
`/health/live`, `/metrics`. Auth is global middleware (`serve_router.rs:124`,
`build_auth_config` at `serve.rs:663`), so `--require-auth` will cover new `/admin`
routes automatically once they are mounted on the same router.

---

## 8. Recommended order for Task 2

1. **Spike the ATS premise** (§6, load-bearing unknown). One throwaway example, ~1 h.
   If it fails, `docs/hotswap/findings.md` and stop.
2. **Confirm the 3.17 GB mechanism** with one `grep "switching to managed"` on a boot
   log (§2). Decides whether GPU memory accounting is a usable acceptance signal.
3. **Decide the derived-buffer story** (§5) — it determines `WeightsArena`'s shape.
4. Add `GpuBackend::device_synchronize()` (§6) — small, independently reviewable.
5. Add `WeightStore::iter_mut` + `weights_arena.rs` + migration at `serve.rs:~240`.
6. Add the arena-pointer free guard (§4.1) and audit the 36 sites.

Estimated touched call sites for the migration itself: **~6 files, ~0 changes in
`spark-model`.** The cost is concentrated in (3), (5) and (6), not in re-pointing.
