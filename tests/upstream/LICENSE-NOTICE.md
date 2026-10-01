# Vendored upstream sources

These files are the reference the parity tests compare against. They are copies of
the `genesis-world` package as published on PyPI, version **1.4.2**, taken
verbatim from the wheel `genesis_world-1.4.2-py3-none-any.whl` (Apache-2.0,
Copyright the Genesis contributors). Nothing in them has been edited; `mpm_solver_tail.py`
is the only exception and it is a verbatim excerpt of one file, noted at its top.

| vendored path | upstream path |
| --- | --- |
| `genesis/utils/geom.py` | `genesis/utils/geom.py` |
| `genesis/utils/simt.py` | `genesis/utils/simt.py` |
| `genesis/engine/solvers/rigid/collider/constants.py` | same |
| `genesis/engine/solvers/rigid/collider/utils.py` | same |
| `genesis/engine/solvers/rigid/collider/support_field.py` | same |
| `genesis/engine/solvers/rigid/collider/gjk_support.py` | same |
| `genesis/engine/solvers/rigid/collider/gjk_utils.py` | same |
| `genesis/engine/solvers/rigid/collider/contact.py` | same |
| `genesis/engine/materials/MPM/{base,elastic,elasto_plastic,liquid,sand,snow}.py` | same |
| `mpm_solver_tail.py` | lines 1321-1345 of `genesis/engine/solvers/mpm_solver.py` |

## Why they are vendored rather than imported

`import genesis` does not work without a GPU and the engine's full runtime
(`gs-madrona`, `mujoco`, `vtk`, `mujoco`, the coupler, the scene). What the parity
tests need is the upstream *arithmetic*, and that is here, unmodified, executed as
ordinary Python.

`tests/upstream_ref.py` makes them runnable in three small, auditable steps:

1. It installs a stub `genesis` package whose `qd_float` is `quadrants.f64` and
   whose `qd_vec3` is a NumPy array constructor, which is exactly what the engine
   sets those two globals to in `genesis/__init__.py` for a 64-bit CPU run.
2. It rewrites the module's relative imports (`from .utils import func_det3`) into
   absolute ones into this stub tree, and replaces the handful of engine-state
   dataclasses (`array_class.DynInfo`, `ColliderInfo`, `GeomsInitAABB`) with
   look-alike objects whose attributes are plain dicts. The tests then fill those
   dicts, which is how they hand a kernel its "engine state" outside an engine.
3. It strips the `@qd.func` / `@qd.kernel` decorator lines. That is the whole
   change to the executable text: quadrants refuses to call a `qd.func` outside a
   kernel, but its body is plain Python and runs as such, with `qd.Vector` and
   `qd.Matrix` (real quadrants types, over NumPy) providing `.dot`, `.cross`,
   `.normalized`, `@`, `.transpose()` and `.determinant()`.

`qd.static(range(3))` unrolled scans run as ordinary Python loops, and
`qd.select(a >= 0, 1.0, -1.0)` is replaced by the equivalent Python conditional,
because quadrants only defines `select` inside a kernel. Every other quadrants
name the ported functions use is the real one.

If any of this is wrong, the parity tests fail loudly rather than quietly, because
they compare the Mojo output against whatever these files compute.
