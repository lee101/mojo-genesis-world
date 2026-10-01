# mojo-genesis-world

The compute-bound numeric kernels of the [Genesis](https://github.com/Genesis-Embodied-AI)
physics engine (`genesis-world` on PyPI), written in
[Mojo](https://www.modular.com/mojo) and callable from Python under the engine's
own names.

```python
import numpy as np
import mojogenesisworld as mgw

d = np.array([[0.3, -0.5, 0.81]])
quat = np.array([[0.5, 0.5, 0.5, 0.5]])
print(mgw.qd_transform_by_quat(d, quat))
print(mgw.Snow().update_F_S_Jp(np.array([0.98]), np.eye(3)[None], *([np.eye(3)] * 3), np.array([1.0]))[0])
```

`mojogenesisworld.geom.qd_transform_by_quat` is `genesis.utils.geom.qd_transform_by_quat`.
`mojogenesisworld.mpm.Snow` is `genesis.engine.materials.MPM.snow.Snow`, with the
same `E` / `nu` / `yield_lower` fields and the same `update_F_S_Jp` /
`update_stress` methods. Every exported function is checked against the upstream
source itself, not against a paraphrase of it.

## Install

```bash
pixi install
pixi run build     # src/*.mojo -> dist/libmojo-genesis-world.so
pixi run test      # parity against the vendored upstream sources
pixi run bench     # the table below
```

To install the Python package instead, the compiled library is shipped inside
the wheel and no Mojo toolchain is needed on the target machine:

```bash
pixi run build
pixi run pip install .
```

In a source checkout the shared library also builds itself on first import, and
rebuilds whenever a `.mojo` file is newer than it. To force it:
`python -m mojogenesisworld._lib --force`. Point `MOJOGENESISWORLD_LIB` at a
prebuilt `libmojo-genesis-world.so` to use a specific one.

## What is covered

Ported, one module per upstream file, in upstream's source order:

| upstream file | ported functions |
| --- | --- |
| `genesis/utils/geom.py` | `qd_transform_by_quat_fast`, `qd_transform_by_trans_quat_fast`, `qd_quat_to_R`, `qd_inv_quat`, `qd_quat_mul`, `qd_transform_quat_by_quat`, `qd_transform_by_quat`, `qd_inv_transform_by_quat`, `qd_transform_by_trans_quat`, `qd_inv_transform_by_trans_quat`, `qd_normalize`, `qd_orthogonals` |
| `.../collider/utils.py` | `func_det3` |
| `.../collider/support_field.py` | `_polar_band`, `get_direction_grid`, `activate`, `_func_direction_grid_coords`, `_func_support_mesh`, `_func_support_mesh_exhaustive`, `_func_support_sphere`, `_func_support_ellipsoid`, `_func_support_capsule`, `_func_support_cylinder`, `_func_support_prism`, `_func_support_box`, `_func_support_world`, `_func_count_supports_mesh`, `_func_count_supports_world`, `_func_count_supports_box` |
| `.../collider/gjk_support.py` | `support_mesh`, `support_driver`, `func_support` |
| `.../collider/gjk_utils.py` | `func_ray_triangle_intersection`, `func_triangle_affine_coords`, `func_point_triangle_intersection`, `func_point_plane_same_side`, `func_origin_tetra_intersection`, `func_project_origin_to_plane` |
| `.../collider/contact.py` | `func_compute_geom_rbound`, `func_compute_geom_pair_scale`, `func_compute_geom_pair_scale_mj`, `func_compute_mc_tolerance`, `func_contact_orthogonals` (MuJoCo branch), `func_rotate_frame`, `func_contact_order_key` |
| `genesis/engine/materials/MPM/base.py` | `resolve_lame`, `_update_F_S_Jp_noop`, `_update_stress_default` |
| `.../MPM/elastic.py` | `_update_F_S_Jp_elastic`, `_update_stress_corotation`, `_update_stress_neohooken` |
| `.../MPM/elasto_plastic.py` | `_update_F_S_Jp_elasto_plastic` |
| `.../MPM/liquid.py` | `_update_F_S_Jp_liquid`, `_update_stress_liquid` |
| `.../MPM/sand.py` | `sand_alpha`, `_sand_projection`, `_update_F_S_Jp_sand`, `_update_stress_sand` |
| `.../MPM/snow.py` | `_update_F_S_Jp_snow`, `_update_stress_snow` |
| `.../solvers/mpm_solver.py` | `stencil_range`, `compute_F_tmp`, `p2g`, `g2p`, `backward_svd` |

Two helpers are ported as internal functions of `src/ported.mojo` rather than as
exported entry points, and are covered by the tests of the entry points that
call them rather than by a test of their own: `signmax` (`mpm_solver.py`, used
by `backward_svd`) and `impose_pos_vel` (`genesis/utils/geometry.py`'s
`CubeBoundary.impose_pos_vel`, used by `g2p`).

### What is not covered, and why

Named function by function, because a silent gap is a bug:

- **`func_contact_orthogonals`, non-MuJoCo branch.** Upstream builds the frame in
  the reference geom's local inertial frame, reading `geoms_init_AABB`, the
  link's inertial quaternion (`dyn_state.links.i_quat`) and the link index. The
  MuJoCo branch, which is the one Genesis uses under MuJoCo compatibility, is
  ported in full and tested; the other branch is not.
- **`_func_support_world` / `support_driver`, terrain path.**
  `_func_support_prism` is ported and tested on its own, but `support_driver`'s
  terrain arm is gated on a compile-time `collider_static_config.has_terrain`
  that the C ABI cannot carry, so the port takes it as a runtime flag. The
  dispatch structure is upstream's; only the flag's compile-time-ness is lost.
- **`p2g` / `g2p`, CPIC.** The `enable_CPIC` branch calls the rigid solver's SDF
  and the coupler. Not ported; the transfer, the material dispatch, the stress
  scaling, the quadratic B-spline weights and the 27-cell scatter/gather are.
- **`p2g`, sparse-reset bookkeeping.** `grid_dirty_flag` / `grid_dirty_list` /
  `grid_dirty_count` are an atomics protocol for clearing the grid, not
  arithmetic, and a sequential port has nothing to clear. Not ported.
- **`_func_count_supports_*` is a `@qd.func` on a compile-time branch.** Ported,
  with `exhaustive` as a runtime flag like upstream's `qd.static` makes it.
- **`_update_F_S_Jp_*` / `_update_stress_*`, the `actu` and `m_dir` arguments.**
  These belong to `muscle.py`, the one MPM material not ported. The parameter
  list stops at `Jp`.
- **The FEM solver (`fem_solver.py`), the PBD and SPH solvers, the constraint
  solver, the coupler, the renderers, the recorders.** Out of scope: the
  port targets the two solvers whose inner loops are dense float64 arithmetic,
  MPM (soft body) and the collider's support field (rigid body).
- **`GSV` / `SVF` claim.** Genesis runs these kernels on GPU via quadrants. This
  port is CPU only, and stays CPU only on purpose: the host API is present on
  this toolchain (it is `max.gpu`, not `std.gpu`), but every kernel here is far
  under 2 flops per byte. See "No GPU path" under Performance for the numbers.

### Divergences forced by the FFI

Three, all documented at the top of `src/ported.mojo`:

1. `qd.Vector(3)` / `qd.Matrix(3, 3)` become the `Vec3` / `Mat3` structs defined
   there, of plain `Float64`, so `v[0]`, `v.dot(w)`, `A @ B`, `A.transpose()` and
   `A.determininant()` mean upstream's thing.
2. Upstream reaches engine state through `array_class` dataclasses
   (`dyn_info.geoms.data[i_g][0]`). Neither Mojo nor a C ABI can, so every field
   a function reads is an explicit pointer argument, named after the field.
3. Upstream's per-thread kernel call is a batched loop here: one exported
   function runs the kernel `n` times over `n` rows, which is the elementwise map
   the engine's `ndrange` performs. Argument order and defaults are unchanged.

`pixi run test` runs 79 tests. Every one of them compares the Mojo library
against the **upstream source itself**, vendored verbatim from the
`genesis_world-1.4.2-py3-none-any.whl` wheel into `tests/upstream/` and executed
by `tests/upstream_ref.py` on quadrants' Python backend, where a `@qd.func` is an
ordinary function over NumPy-backed `qd.Vector` / `qd.Matrix`. Nothing in those
files is edited; `tests/upstream/LICENSE-NOTICE.md` lists the files, the three
that are verbatim excerpts of larger files, and the three shims the harness needs
(two of them fix outright bugs in quadrants 1.3.0's Python backend: `Vector.x`
raises, and `Vector.norm(eps)` returns `inf` for a nonzero `eps`).

`import genesis` does not work here - it needs a GPU, `gs-madrona`, `mujoco`,
`vtk` and the scene - so the reference is the vendored source, not an installed
package. The geometry kernels are held to `1e-12` relative. The MPM constitutive
models, which chain `exp` / `log` / `pow`, are held to `1e-8` relative, because
quadrants routes those through torch in the harness and keeps a few more bits of
slack than the compiled backends do.

## Performance

`pixi run bench`, on the machine named below. Both columns run the same problem
size on the same data; the upstream column is the vendored upstream code executing
as an interpreted `@qd.func` call per element, which is the same arithmetic
without a compiler.

| case | mojo-genesis-world | upstream (quadrants, python backend) | ratio |
| --- | ---: | ---: | ---: |
| `SupportField.activate` (8 geoms x 2000 verts, res 64) | 15.98 ms | 190.40 ms | 11.92x faster |
| `support_mesh` exhaustive scan (500 queries x 512 verts) | 0.71 ms | 13570.18 ms | 19197.69x faster |
| `support_driver` analytic primitives (500 queries, 7 geom types) | 0.08 ms | 884.33 ms | 11707.33x faster |
| `func_support` pair (500 queries, mesh + box) | 0.32 ms | 8492.23 ms | 26387.74x faster |
| `func_ray_triangle_intersection` (500 rays) | 0.05 ms | 818.85 ms | 15582.09x faster |
| `func_project_origin_to_plane` (500 simplices) | 0.02 ms | 294.17 ms | 14840.38x faster |
| `qd_transform_by_quat` (500 vec/quat pairs) | 0.01 ms | 353.07 ms | 25011.92x faster |
| `func_compute_geom_pair_scale_mj` (500 pairs) | 0.02 ms | 216.93 ms | 9050.31x faster |
| `Elastic.update_stress` corotation (500 particles) | 0.05 ms | 394.94 ms | 7702.61x faster |
| `Snow.update_F_S_Jp` (500 particles) | 0.05 ms | 716.37 ms | 13745.37x faster |
| `backward_svd` (500 particles) | 0.09 ms | 2151.80 ms | 24425.34x faster |
| `g2p` grid gather (200 particles, 32^3 grid) | 0.13 ms | 79.64 ms | 614.98x faster |
| `p2g` grid scatter (200 particles, 32^3 grid) | 0.20 ms | 435.68 ms | 2159.22x faster |

```
machine: Linux-6.8.0-142-generic-x86_64-with-glibc2.39, 72 cores
python 3.13.14, numpy 2.5.1, quadrants 1.3.0, mojo 1.2.0.dev2026092605
```

### Where the time goes, and what the code does about it

Read the table as what one `pixi run bench` printed, not as ten significant
figures: the `mojo-genesis-world` column moves by tens of percent between runs
with the machine's load. Below about a millisecond a call is dominated by the
ctypes boundary and the numpy view setup, not by the arithmetic, which is why
the two columns agree on the arithmetic and disagree on everything else.

`activate` is the one row that compares algorithms rather than dispatch, and it
is the only loop in the library shaped for the machine. Upstream's own body is a
`v1 @ this_pos.T` GEMM plus `argmax`, run as written; the port is one fused loop
over the same dot products, and it got there in three steps.

1. **SIMD.** The inner loop is an argmax over a vertex table, one direction at a
   time: three scalar loads and a branch per vertex, which is load-throughput
   bound. One SIMD lane per *direction* instead of per vertex turns that into
   three loads and three FMAs for four directions, so the vertex table is read
   once per four dot products.
   The direction table is interleaved `(num_v, 3)`, so a contiguous vector load
   does **not** give four directions - it gives four consecutive floats, i.e.
   `(x0, y0, z0, x1)`. The block therefore starts from a 3x4 register transpose
   of the direction table, which is why the block is fixed at four lanes
   (`_DW`) rather than at `simd_width_of[DType.float64]()`; the transpose is
   twelve scalar loads per block, amortised over the whole vertex scan. Each
   lane keeps its own first maximum with a strict `>`, exactly as the scalar scan
   does, so the winner is still the one `np.argmax` picks, and the last
   `num_v % _DW` directions go through the scalar path.
2. **`parallelize`.** Every geom contributes exactly `num_v` cells, so the table
   is a flat `(geom, direction)` grid and the cells are independent - the cell
   offset that upstream accumulates is just `i_g * num_v`. A work item is four
   consecutive directions of one geom, so it never straddles a geom boundary and
   never splits a block. A parallel launch from a ctypes host is not free on
   this toolchain, so it is gated behind `1 << 24` dot products
   (`_ACTIVATE_PARALLEL_MIN`); below that the serial loop wins and is used.
3. **`Mat3`.** `Mat3.__matmul__`, `transpose` and `__mul__` were triple loops
   over `at(i, j)`, and `at` is a branch tree on a runtime index. Mojo does not
   unroll `for i in range(3)`, so the matrices never reached registers. Written
   out, the same arithmetic in the same order is several times cheaper. The
   `range(3)` scans over a diagonal in the sand, snow and elasto-plastic models
   went the same way, and `_update_stress_sand` no longer computes
   `log(S[i, i])` twice per entry.

### No GPU path, and why that is the right answer

The host API is available on this toolchain - it moved from `std.gpu` to
`max.gpu`, and `DeviceContext`, `enqueue_create_buffer`, `enqueue_copy` and
`enqueue_function` all compile under the pinned `mojo` 1.2.0 with the `max`
package beside it. What rules the GPU out here is arithmetic intensity, not
availability. `activate` does six flops per vertex it reads 24 bytes of, so
0.25 flops per byte; `backward_svd` is about 1.5; `p2g` and `g2p` are lower
still. All of them are far under the roughly 2 flops per byte where a
copy-in / launch / copy-out round trip starts to pay for itself, and the batched
entry points are called once per solver step on arrays of a few thousand
elements, where a single launch is the dominant cost. There is no input size at
which these kernels would be better on the device, so shipping a GPU path would
be slower code plus a second buffer layout to keep correct.

`g2p` and `backward_svd` are embarrassingly parallel in principle and are not
threaded: at 0.13 ms and 0.09 ms a `parallelize` launch would cost more than the
work. `p2g` cannot be threaded at all without changing the arithmetic, because
its 27-cell scatter accumulates into shared grid cells.

## How it works

**FFI.** `src/capi.mojo` is the only `@export` surface, and it is one compilation
unit: `mojo build --emit shared-lib -I src src/capi.mojo -o dist/libmojo-genesis-world.so`.
Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric; the wrapper
rebuilds the pointer with an explicit `AnyOrigin[mut=True]`. Nothing in the
library allocates: the caller owns every input, every output and every scratch
buffer, so there is nothing to leak and nothing to free.

**Memory layout.** Row-major, no padding, C-contiguous everywhere:

| buffer | shape | contents |
| --- | --- | --- |
| `init_pos` | `(n_verts, 3)` float64 | `dyn_info.verts.init_pos` |
| `geoms_data` | `(n_geoms, 3)` float64 | `dyn_info.geoms.data` |
| `geoms_init_AABB` | `(n_geoms, 8, 3)` float64 | the eight rest-pose AABB corners, corner 0 lower and corner 7 upper |
| `type`, `vert_start`, `vert_end`, `active`, `free` | `(n,)` int32 | the `gs.qd_int` fields |
| `support_v`, `support_vid`, `support_cell_start` | `(n_cells, 3)`, `(n_cells,)`, `(n_geoms,)` | the support table |
| particles | `(n, 3)`, `(n, 9)`, `(n,)` | `pos`, `vel`, and `C` / `F` / `F_tmp` / `U` / `S` / `V` as row-major 3x3 |
| grid | `(nx*ny*nz, 3)` and `(nx*ny*nz,)` | `vel_in` / `vel_out` and `mass` |

Int fields are `int32` because that is `gs.qd_int`; floats are `float64` because
that is a 64-bit Genesis run's `gs.qd_float`. The ctypes signature table in
`python/mojogenesisworld/_lib.py` is written by hand, so
`test_every_export_has_a_ctypes_signature` reads the shared library's dynamic
symbol table and fails if the two ever disagree in either direction.

The support table is indexed `(geom, direction)`: cell `i_g * num_v + i_v`, so
`support_cell_start` is `arange(n_geoms) * num_v` and `n_cells` is
`n_geoms * num_v`. That is what makes the table a flat, embarrassingly parallel
grid, and it is the same layout upstream's loop produces.

**Threading.** The only `parallelize` in the library is `activate`, over
`max.algorithm.parallelize`, behind the size gate described under Performance.
Nothing else in the library spawns a thread.

**Vendored upstream.** `tests/upstream/` holds upstream's files, unmodified, so
the parity tests compare against the real thing. `tests/upstream_ref.py` gives
them a `genesis` package to import from - a stub whose `qd_float` and `qd_vec3`
are exactly what `genesis/__init__.py` sets them to for a 64-bit CPU run - and
initialises quadrants' Python backend, where `@qd.func` bodies execute as
ordinary Python over quadrants' own `qd.Vector` and `qd.Matrix`. The vendored
`collider/utils.py` has its engine imports blanked (it is the one file whose
module scope cannot load outside an engine), and the MPM material kernels are
verbatim excerpts of their methods, de-indented; both are noted in
`tests/upstream/LICENSE-NOTICE.md`.

## Layout

```
src/ported.mojo        the kernels, in upstream's source order
src/capi.mojo          the @export C ABI
build/build.sh         mojo build --emit shared-lib
python/mojogenesisworld/   ctypes loader + the public API
tests/upstream/        upstream 1.4.2 sources, vendored
tests/upstream_ref.py  runs them
tests/ref.py           one helper per ported function, calling upstream
tests/test_parity.py   the parity suite
bench/bench.py         the benchmark
MOJO_NOTES.md          Mojo dialect notes for the pinned toolchain
```

## License
MIT, (c) Lee Penkman; see `LICENSE`. The vendored files in `tests/upstream/`
are Genesis's, Apache-2.0, Copyright the Genesis contributors; see
`tests/upstream/LICENSE-NOTICE.md`.
