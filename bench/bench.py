"""mojo-genesis-world against the upstream `genesis-world` 1.4.2 sources.

    pixi run bench

Both columns run the same problem size on the same data. The upstream column is
the vendored upstream code itself, executed through `tests/upstream_ref.py`:
quadrants' Python backend, where a `@qd.func` is an ordinary function over
NumPy-backed `qd.Vector` / `qd.Matrix` values. That is upstream's arithmetic
without the compiler, i.e. one interpreted call per element, so the column
measures the same formulas rather than upstream's fused GPU kernel. Two cases
are not elementwise and are compared against the same upstream algorithm
executed as upstream writes it: `SupportField.activate` (a `v1 @ verts.T` GEMM
plus `argmax`) and the MPM 27-cell transfer.

Every number below is measured here. Nothing is estimated.
"""

from __future__ import annotations

import math
import os
import platform
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))
sys.path.insert(0, os.path.join(ROOT, "tests"))

import ref  # noqa: E402
from mojogenesisworld import contact as C  # noqa: E402
from mojogenesisworld import geom as G  # noqa: E402
from mojogenesisworld import gjk as J  # noqa: E402
from mojogenesisworld import mpm as M  # noqa: E402
from mojogenesisworld import support_field as S  # noqa: E402
from mojogenesisworld._lib import addr, lib  # noqa: E402


def timeit(fn, repeat: int = 5) -> float:
    fn()
    best = math.inf
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


CASES = []


def case(name: str, repeat: int = 5):
    def deco(fn):
        CASES.append((name, fn, repeat))
        return fn
    return deco


def scene(rng, n_verts_per_geom: int, n_geoms: int):
    init_pos = rng.normal(size=(n_verts_per_geom * n_geoms, 3))
    vert_start = np.arange(n_geoms, dtype=np.int32) * n_verts_per_geom
    return {
        "init_pos": init_pos,
        "vert_start": vert_start,
        "vert_end": vert_start + n_verts_per_geom,
        "type": np.full(n_geoms, S.GEOM_TYPE_MESH, dtype=np.int32),
        "data": np.abs(rng.normal(size=(n_geoms, 3))) + 0.1,
    }


def quats(rng, n):
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def _svd(n, rng):
    U = np.empty((n, 3, 3))
    V = np.empty((n, 3, 3))
    Sd = np.empty((n, 3, 3))
    a = rng.normal(size=(n, 3, 3))
    b = rng.normal(size=(n, 3, 3))
    for i in range(n):
        Qa, Ra = np.linalg.qr(a[i])
        Qb, Rb = np.linalg.qr(b[i])
        U[i] = Qa * np.sign(np.diag(Ra))
        V[i] = Qb * np.sign(np.diag(Rb))
        Sd[i] = np.diag(rng.uniform(0.9, 1.1, 3))
    return U, Sd, V


# ------------------------------------------------------------------ rigid body

@case("SupportField.activate (8 geoms x 2000 verts, res 64)", repeat=3)
def _():
    rng = np.random.default_rng(0)
    sc = scene(rng, 2000, 8)
    res = 64
    field = S.SupportField(support_res=res)
    field.activate(sc["init_pos"], sc["vert_start"], sc["vert_end"])
    n_v = field.v1.shape[0]

    workers = int(os.environ.get("MOJOGENESISWORLD_WORKERS", "1"))

    def ours():
        lib().mgw_activate(
            addr(sc["init_pos"]), addr(sc["vert_start"]), addr(sc["vert_end"]), 8,
            addr(field.v1), n_v, addr(field.support_v), addr(field.support_vid),
            addr(field.support_cell_start), workers,
        )

    def theirs():
        ref.activate(sc["init_pos"], sc["vert_start"], sc["vert_end"], res)

    return ours, theirs


@case("support_mesh exhaustive scan (500 queries x 512 verts)", repeat=3)
def _():
    rng = np.random.default_rng(1)
    sc = scene(rng, 512, 1)
    n = 500
    d = rng.normal(size=(n, 3))
    pos = rng.normal(size=(n, 3))
    q = quats(rng, n)
    ig = np.zeros(n, dtype=np.int32)
    prev = np.zeros(n, dtype=np.int32)

    def ours():
        J.support_mesh(ig, d, pos, q, prev, sc["init_pos"], sc["vert_start"], sc["vert_end"])

    def theirs():
        for a, b, c in zip(d, pos, q):
            ref.support_mesh(0, a, b, c, np.zeros(1, np.int32), sc["init_pos"],
                             sc["vert_start"], sc["vert_end"])

    return ours, theirs


@case("support_driver analytic primitives (500 queries, 7 geom types)", repeat=3)
def _():
    rng = np.random.default_rng(2)
    sc = scene(rng, 8, 7)
    n = 500
    d = rng.normal(size=(n, 3))
    pos = rng.normal(size=(n, 3))
    q = quats(rng, n)
    gt = np.arange(n, dtype=np.int32) % 7
    ig = gt.copy()
    prism = rng.normal(size=(6, 8, 3))
    field = S.SupportField(support_res=16)
    field.activate(sc["init_pos"], sc["vert_start"], sc["vert_end"])
    tables = (field.support_v, field.support_vid, field.support_cell_start, field.support_res)

    def ours():
        J.support_driver(ig, np.zeros(n, np.int32), d, pos, q, False, gt, sc["data"],
                         sc["vert_start"], sc["vert_end"], sc["init_pos"], field,
                         np.zeros(n, np.int32), prism, False, True)

    def theirs():
        for i, (a, b, c) in enumerate(zip(d, pos, q)):
            ref.support_driver(0, 0, a, b, c, False, gt[i], sc["data"], sc["vert_start"],
                               sc["vert_end"], sc["init_pos"], tables, np.zeros(1, np.int32),
                               prism, False)

    return ours, theirs


@case("func_support pair (500 queries, mesh + box)", repeat=3)
def _():
    rng = np.random.default_rng(3)
    sc = scene(rng, 256, 2)
    sc["type"] = np.array([S.GEOM_TYPE_MESH, S.GEOM_TYPE_BOX], dtype=np.int32)
    n = 500
    d = rng.normal(size=(n, 3))
    pa, pb = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    qa, qb = quats(rng, n), quats(rng, n)
    ga, gb = np.zeros(n, np.int32), np.ones(n, np.int32)
    field = S.SupportField(support_res=32)
    field.activate(sc["init_pos"], sc["vert_start"], sc["vert_end"])
    prism = rng.normal(size=(6, 8, 3))
    tables = (field.support_v, field.support_vid, field.support_cell_start, field.support_res)

    def ours():
        J.func_support(ga, gb, np.zeros(n, np.int32), d, pa, qa, pb, qb, False, sc["type"],
                       sc["data"], sc["vert_start"], sc["vert_end"], sc["init_pos"], field,
                       np.zeros(2 * n, np.int32), prism, False, True)

    def theirs():
        for a, b, c, f, g in zip(d, pa, qa, pb, qb):
            ref.func_support(0, 1, 0, a, b, c, f, g, False, sc["type"], sc["data"],
                             sc["vert_start"], sc["vert_end"], sc["init_pos"], tables,
                             np.zeros(2, np.int32), prism, False)

    return ours, theirs


@case("func_ray_triangle_intersection (500 rays)", repeat=3)
def _():
    rng = np.random.default_rng(4)
    n = 500
    r1, r2 = rng.normal(size=(n, 3)), rng.normal(size=(n, 3))
    t1, t2, t3 = (rng.normal(size=(n, 3)) for _ in range(3))

    def ours():
        J.func_ray_triangle_intersection(r1, r2, t1, t2, t3)

    def theirs():
        for a, b, c, d, e in zip(r1, r2, t1, t2, t3):
            ref.ray_triangle_intersection(a, b, c, d, e)

    return ours, theirs


@case("func_project_origin_to_plane (500 simplices)", repeat=3)
def _():
    rng = np.random.default_rng(5)
    n = 500
    v1, v2, v3 = (rng.normal(size=(n, 3)) for _ in range(3))

    def ours():
        J.func_project_origin_to_plane(v1, v2, v3)

    def theirs():
        for a, b, c in zip(v1, v2, v3):
            ref.project_origin_to_plane(a, b, c)

    return ours, theirs


@case("qd_transform_by_quat (500 vec/quat pairs)", repeat=3)
def _():
    rng = np.random.default_rng(6)
    n = 500
    v = rng.normal(size=(n, 3))
    q = quats(rng, n)

    def ours():
        G.qd_transform_by_quat(v, q)

    def theirs():
        for a, b in zip(v, q):
            ref.transform_by_quat(a[None], b[None])

    return ours, theirs


@case("func_compute_geom_pair_scale_mj (500 pairs)", repeat=3)
def _():
    rng = np.random.default_rng(7)
    n_geoms = 64
    gt = rng.integers(0, 7, size=n_geoms).astype(np.int32)
    gd = np.abs(rng.normal(size=(n_geoms, 3))) + 0.1
    aabb = ref.aabb_corners(rng, n_geoms).reshape(n_geoms, 8, 3)
    n = 500
    ga = rng.integers(0, n_geoms, size=n).astype(np.int32)
    gb = rng.integers(0, n_geoms, size=n).astype(np.int32)

    def ours():
        C.func_compute_geom_pair_scale_mj(ga, gb, gt, gd, aabb)

    def theirs():
        for i in range(n):
            ref.compute_geom_pair_scale_mj(int(ga[i]), int(gb[i]), gt, gd, aabb)

    return ours, theirs


# ------------------------------------------------------------------- soft body

@case("Elastic.update_stress corotation (500 particles)", repeat=3)
def _():
    rng = np.random.default_rng(8)
    n = 500
    mat = M.Elastic()
    U, S, V = _svd(n, rng)
    F = np.einsum("nij,njk,nkl->nil", U, S, V)
    J_ = np.linalg.det(F)
    Jp = np.ones(n)

    def ours():
        mat.update_stress(U, S, V, F * 1.01, F, J_, Jp)

    def theirs():
        ref.mpm_stress(mat, "corotation", U, S, V, F * 1.01, F, J_, Jp)

    return ours, theirs


@case("Snow.update_F_S_Jp (500 particles)", repeat=3)
def _():
    rng = np.random.default_rng(9)
    n = 500
    mat = M.Snow()
    U, S, V = _svd(n, rng)
    F = np.einsum("nij,njk,nkl->nil", U, S, V)
    J_ = np.linalg.det(F)
    Jp = np.ones(n)

    def ours():
        mat.update_F_S_Jp(J_, F, U, S, V, Jp)

    def theirs():
        ref.mpm_update_F_S_Jp("snow", mat, J_, F, U, S, V, Jp)

    return ours, theirs


@case("backward_svd (500 particles)", repeat=3)
def _():
    rng = np.random.default_rng(10)
    n = 500
    U, S, V = _svd(n, rng)
    gU, gS, gV = (rng.normal(size=(n, 3, 3)) * 0.1 for _ in range(3))

    def ours():
        M.backward_svd(gU, gS, gV, U, S, V)

    def theirs():
        for a, b, c, d, e, f in zip(gU, gS, gV, U, S, V):
            ref.backward_svd(a, b, c, d, e, f)

    return ours, theirs


@case("g2p grid gather (200 particles, 32^3 grid)", repeat=3)
def _():
    rng = np.random.default_rng(11)
    n = 200
    res = np.array([32, 32, 32])
    pos = rng.uniform(0.2, 0.8, size=(n, 3))
    vel_out = rng.normal(size=(res[0], res[1], res[2], 3))
    lower, upper = np.zeros(3), np.full(3, 1.0)
    dx = 1.0 / 32.0
    return (
        lambda: M.g2p(pos, vel_out, lower, upper, 1e-4, dx, np.zeros(3), res),
        lambda: ref.g2p(pos, vel_out, lower, upper, 1e-4, dx, np.zeros(3), res),
    )


@case("p2g grid scatter (200 particles, 32^3 grid)", repeat=3)
def _():
    rng = np.random.default_rng(12)
    n = 200
    res = np.array([32, 32, 32])
    dx = 1.0 / 32.0
    mat = M.Elastic()
    U, S, V = _svd(n, rng)
    F = np.einsum("nij,njk,nkl->nil", U, S, V)
    particles = {
        "pos": rng.uniform(0.2, 0.8, size=(n, 3)),
        "vel": rng.normal(size=(n, 3)),
        "C": rng.normal(size=(n, 3, 3)) * 0.01,
        "F": F,
        "F_tmp": F * 1.01,
        "U": U, "S": S, "V": V,
        "Jp": np.ones(n),
        "active": np.ones(n, np.int32),
        "free": np.ones(n, np.int32),
        "mass": np.full(n, 1e-3),
    }
    grid = {"res": res, "offset": np.zeros(3)}
    params = {"dx": dx, "substep_dt": 1e-4, "particle_volume": 1e-3, "needs_svd": True}
    p = particles

    def ours():
        vel_in = np.zeros((res[0], res[1], res[2], 3))
        grid_mass = np.zeros((res[0], res[1], res[2]))
        M.p2g(p["pos"], p["vel"], p["C"], p["F"], p["F_tmp"], p["U"], p["S"], p["V"], p["Jp"],
              p["active"], p["mass"], p["free"], mat, 1e-4, dx, 1e-3, np.zeros(3), res,
              vel_in, grid_mass, model=M.MODEL_ELASTIC, needs_svd=True)

    def theirs():
        ref.p2g(mat, M.MODEL_ELASTIC, p, grid, params)

    return ours, theirs


def main() -> None:
    cores = len(os.sched_getaffinity(0))
    print(f"machine: {platform.platform()}, {cores} cores")
    print(f"python {platform.python_version()}, numpy {np.__version__}, quadrants 1.3.0\n")
    print("| case | mojo-genesis-world | upstream (quadrants python backend) | ratio |")
    print("| --- | ---: | ---: | ---: |")
    for name, build, repeat in CASES:
        ours, theirs = build()
        a = timeit(ours, repeat)
        b = timeit(theirs, repeat)
        flag = "" if a < b else " slower"
        print(f"| {name} | {a * 1e3:.2f} ms | {b * 1e3:.2f} ms | {b / a:.2f}x{flag} |")


if __name__ == "__main__":
    main()
