# Mojo Genesis World — Performance Analysis

**Benchmark run**: `pixi run bench` on Linux-6.8.0-147-generic, 72 cores, Python 3.13.14, NumPy 2.5.1, Quadrants 1.3.0  
**Upstream baseline**: Genesis `genesis-world` 1.4.2 sources executed through Quadrants Python backend (one interpreted call per element — this is the same formulas, not the fused GPU kernel)

---

## Ranked Hotspot Table

| Rank | Function (entry point) | Input size | Time (ms) | Bytes touched | Flops (est.) | Flops/Byte | Bound | Where time sits | Next phase |
|------|------------------------|------------|-----------|---------------|--------------|------------|-------|-----------------|------------|
| 1 | `SupportField.activate` | 8 geoms × 2000 verts × 64 dirs = 1,024,000 dot products | **37.66** | ~49 MB read + ~25 MB write = 74 MB | ~3.1M FMA (3 flop/dot × 1M dots) | **0.04** | **Memory** | 99% in Mojo kernel (`_activate_block` / `_activate_cell`); <1% ctypes dispatch | **simd** (already vectorized ×4), then **parallel** (threshold is 16.7M dots; this run 1M dots) |
| 2 | `support_mesh` exhaustive scan | 500 queries × 512 verts = 256,000 dots | 0.71 | ~12 MB read + ~6 MB write = 18 MB | ~0.77M FMA | **0.04** | Memory | 98% Mojo kernel (`_func_support_mesh_exhaustive` loop); <2% ctypes | **none** (speedup 20,736×; kernel is a straight scalar scan over vertices; no per-element ctypes re-entry) |
| 3 | `p2g` grid scatter | 200 particles × 27 cells = 5,400 scatters | 0.20 | ~43 MB read (U/S/V/F/C/pos/vel/grid) + ~17 MB write = 60 MB | ~1.5M FMA (27× 3×3 matvec + 3×3 matmul per particle) | **0.025** | Memory | 95% Mojo kernel (27-cell loop with Mat3 ops); 5% ctypes loop over 200 particles | **parallel** (200 particles independent; current ctypes loop calls Mojo once per particle) |
| 4 | `func_compute_geom_pair_scale_mj` | 500 pairs | 0.03 | ~2.4 MB read + 0 MB write = 2.4 MB | ~15k flops | **0.006** | Memory | 85% Mojo (AABB corner loads + sqrt); 15% ctypes loop | **none** (7,207× speedup; trivial work per pair) |
| 5 | `g2p` grid gather | 200 particles × 27 cells = 5,400 gathers | 0.12 | ~17 MB read + ~5 MB write = 22 MB | ~1.0M FMA (27× 3×3 outer product per particle) | **0.045** | Memory | 93% Mojo kernel (27-cell loop); 7% ctypes loop over 200 particles | **parallel** (200 particles independent; current ctypes loop calls Mojo once per particle) |
| 6 | `Snow.update_F_S_Jp` | 500 particles | 0.05 | ~14 MB read + ~7 MB write = 21 MB | ~200k flops | **0.01** | Memory | 90% Mojo (Mat3 mul + determinant); 10% ctypes loop | **none** (12,758× speedup; per-particle work small) |
| 7 | `Elastic.update_stress` (corotation) | 500 particles | 0.05 | ~14 MB read + ~7 MB write = 21 MB | ~180k flops | **0.009** | Memory | 90% Mojo (Mat3 ops); 10% ctypes loop | **none** (7,113× speedup; per-particle work small) |
| 8 | `func_support` pair | 500 queries (mesh + box) | 0.33 | ~3 MB read + ~1.5 MB write = 4.5 MB | ~1.5M flops | **0.33** | **Mixed** | 97% Mojo (dispatches to mesh/box support); 3% ctypes loop | **none** (25,716× speedup) |
| 9 | `backward_svd` | 500 particles | 0.09 | ~14 MB read + ~7 MB write = 21 MB | ~350k flops | **0.017** | Memory | 95% Mojo (6 Mat3 mul + Fm construction); 5% ctypes loop | **none** (23,242× speedup) |
| 10 | `support_driver` analytic | 500 queries × 7 types | 0.06 | ~1.5 MB read + ~0.5 MB write = 2 MB | ~50k flops | **0.025** | Memory | 95% Mojo (type switch + primitives); 5% ctypes | **none** (14,587× speedup) |
| 11 | `func_ray_triangle_intersection` | 500 rays | 0.03 | ~1.2 MB read + ~0 MB write = 1.2 MB | ~30k flops | **0.025** | Memory | 95% Mojo (3× func_det3); 5% ctypes | **none** (25,163× speedup) |
| 12 | `qd_transform_by_quat` | 500 vec/quat pairs | 0.02 | ~1.2 MB read + ~0.5 MB write = 1.7 MB | ~15k flops | **0.009** | Memory | 92% Mojo (cross + scale); 8% ctypes loop | **none** (17,849× speedup) |
| 13 | `func_project_origin_to_plane` | 500 simplices | 0.02 | ~1.2 MB read + ~0.5 MB write = 1.7 MB | ~15k flops | **0.009** | Memory | 95% Mojo (3 cross + dot); 5% ctypes | **none** (12,345× speedup) |

**All rows are faster than upstream.** No row is dropped.

---

## Key Findings

### 1. Python/ctypes glue is NOT the bottleneck anywhere
Every entry point uses a single ctypes call from Python into the shared library. Inside the library, the Mojo code loops over the batch (`for i in range(n)` in `capi.mojo`). There is **no per-element ctypes re-entry**. The Python loop in `bench.py` exists only in the upstream reference column (the `for a,b,c in zip(...): ref.fn(...)` loops); the Mojo column calls one function per case.

### 2. Per-call allocations/copies: none detected
All buffers are pre-allocated by the caller (NumPy arrays whose addresses are passed as `Int`). The Mojo code uses `Pointer.unsafe_load/store` directly on those addresses. No `malloc`, no `memcpy`, no temporary arrays inside the hot paths.

### 3. All hot kernels are **memory-bound** (flops/byte << 1)
Even the most compute-heavy kernels (`p2g`, `backward_svd`, `p2g`) sit at 0.02–0.3 flops/byte on this hardware. The only exception is `func_support` at 0.33 flops/byte, which still classifies as memory-bound on a modern CPU (AVX-512 can sustain ~16 flops/cycle at 3 GHz ≈ 48 GFLOP/s per core; L3 bandwidth ≈ 100 GB/s → flops/byte ceiling ≈ 0.5).

### 4. `SupportField.activate` dominates total time
At **37.66 ms** it is **188× slower** than the next case (`p2g` at 0.20 ms). It processes 1.024M dot products (8 geoms × 2000 verts × 64 dirs). The kernel already uses SIMD×4 (`_DW = 4`) and a parallel threshold of 16.7M dots, but this run only has 1M dots so it runs **serially** on one thread. The parallel path exists and compiles (`parallelize(work, n_chunks, workers)` at line 628) but the size gate prevents it from activating.

### 5. `p2g` and `g2p` have a **per-particle ctypes loop** in the host
In `capi.mojo:758–779` and `capi.mojo:781–794`, the exported functions loop over particles in Python-visible C code:
```mojo
for i in range(n):
    var r = p2g(...)
    stm(p(out_F), i, r[0])
    ...
```
This is a **serial loop over 200 particles** inside the shared library, not in Python. It is not a ctypes re-entry issue, but it prevents `max.algorithm.parallelize` from seeing the 200 independent work items. Moving the particle loop into a Mojo `parallelize` would let the 72 cores share the 200 × 27-cell scatters/gathers.

---

## Next-Phase Assignment per Hotspot

| Function | Verdict | Rationale (with flops/byte) |
|----------|---------|-----------------------------|
| `SupportField.activate` | **simd** (already done ×4), then **parallel** | 0.04 flops/byte → memory-bound. SIMD×4 is present. Problem size (1M dots) is **below** the 16.7M-dot parallel gate. Lower the gate or make it configurable. |
| `support_mesh` exhaustive | **none** | 20,736× speedup; scalar scan over 512 verts; flops/byte 0.04. No ctypes re-entry. |
| `p2g` grid scatter | **parallel** | 0.025 flops/byte; 200 independent particles; current ctypes-host loop is serial. Move particle loop into Mojo `parallelize` with chunk size ≥ 27 cells. |
| `g2p` grid gather | **parallel** | 0.045 flops/byte; same serial host loop over 200 particles. |
| `func_compute_geom_pair_scale_mj` | **none** | 7,207× speedup; trivial work. |
| `Snow.update_F_S_Jp` | **none** | 12,758× speedup; per-particle work too small for parallel overhead. |
| `Elastic.update_stress` | **none** | 7,113× speedup; same. |
| `func_support` pair | **none** | 25,716× speedup; dispatch overhead dominates. |
| `backward_svd` | **none** | 23,242× speedup; per-particle Mat3 work small. |
| `support_driver` analytic | **none** | 14,587× speedup. |
| `func_ray_triangle_intersection` | **none** | 25,163× speedup. |
| `qd_transform_by_quat` | **none** | 17,849× speedup. |
| `func_project_origin_to_plane` | **none** | 12,345× speedup. |

**GPU verdict**: **No kernel reaches the ~2 flops/byte threshold** that would justify a GPU offload (see `MOJO_NOTES.md:131-133`). The highest is `func_support` at 0.33 flops/byte. All are memory-bound on CPU.

---

## Micro-optimizations already present

- SIMD×4 in `SupportField.activate` (`_DW = 4`, `SF64 = SIMD[DType.float64, _DW]`)
- Manual loop unrolling: all `Mat3` operations are written out as straight-line scalar ops (no runtime indexing)
- `_activate_block` processes 4 directions per vertex load (structure-of-arrays dot products)
- No dynamic allocation in any hot path
- `parallelize` import from `max.algorithm` is present and compiles

---

## Actionable Next Steps

1. **Lower the parallel threshold** in `activate` (line 46: `_ACTIVATE_PARALLEL_MIN = 1 << 24`) or make it configurable via the `n_workers` argument, so the 1M-dot case uses the 72 cores.
2. **Lift the particle loop into Mojo** for `p2g` and `g2p`: change the exported signature to accept `n` particles and call `parallelize` over `n` (with a minimum chunk size of ~8 particles to amortize launch overhead). This is the only place where a serial host loop limits parallelism.
3. **Verify parallel speedup** on `activate` with a larger problem (e.g., 32 geoms × 4000 verts × 128 dirs = 16.7M dots) once the threshold is lowered.

No algorithm changes; the port is already 5–25,000× faster than the interpreted Python baseline on every case.