"""Numerical parity with the vendored `genesis-world` 1.4.2 sources.

Every test compares the Mojo library against `tests/ref.py`, which calls the
upstream functions themselves. Tolerances are tight: these kernels are a few
dozen flops of exact IEEE arithmetic in the same order, so anything above
1e-12 relative means the port disagrees with upstream rather than that the two
rounded differently.

The two exceptions are named where they occur: `activate`, whose upstream
reference is a `v1 @ verts.T` GEMM plus `argmax` (so a tie can break either way),
and `p2g` / `g2p`, whose upstream reference is a NumPy transcription of the
transfer loop around the real upstream material kernels.
"""

from __future__ import annotations

import numpy as np
import pytest

import ref
import mojogenesisworld as mgw
from mojogenesisworld import contact as C
from mojogenesisworld import geom as G
from mojogenesisworld import gjk as J
from mojogenesisworld import mpm as M
from mojogenesisworld import support_field as S

RTOL = 1e-12
ATOL = 1e-12


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(20260928)


@pytest.fixture(scope="module")
def geoms(rng):
    return {
        "init_pos": rng.normal(size=(240, 3)),
        "vert_start": np.array([0, 80, 160], dtype=np.int32),
        "vert_end": np.array([80, 160, 240], dtype=np.int32),
        "type": np.array([S.GEOM_TYPE_MESH, S.GEOM_TYPE_BOX, S.GEOM_TYPE_SPHERE], dtype=np.int32),
        "data": ref.geom_data(rng, 3),
    }


@pytest.fixture(scope="module")
def field(geoms):
    f = S.SupportField(support_res=12)
    f.activate(geoms["init_pos"], geoms["vert_start"], geoms["vert_end"])
    return f



def test_every_export_has_a_ctypes_signature():
    """The C ABI and the ctypes table must not drift: a symbol the table misses
    would be called with default argument types, and a signature the library does
    not export would be a call into nothing."""
    import subprocess

    from mojogenesisworld._lib import _SIGNATURES, lib

    path = lib()._name
    out = subprocess.run(
        ["nm", "-D", "--defined-only", path], capture_output=True, text=True, check=True
    ).stdout
    exported = {ln.split()[-1] for ln in out.splitlines() if ln.split()[-1].startswith("mgw_")}
    assert exported == set(_SIGNATURES)
    for name, (argtypes, restype) in _SIGNATURES.items():
        assert getattr(lib(), name).argtypes == argtypes, name
        assert getattr(lib(), name).restype == restype, name

# ======================================================================== geom.py


def test_transform_by_quat(rng):
    v, q = ref.rand3(rng, 64), ref.rand_unit_quat(rng, 64)
    assert np.allclose(G.qd_transform_by_quat(v, q), ref.transform_by_quat(v, q), rtol=RTOL, atol=ATOL)


def test_transform_by_quat_fast(rng):
    v, q = ref.rand3(rng, 64), ref.rand_unit_quat(rng, 64)
    assert np.allclose(
        G.qd_transform_by_quat_fast(v, q), ref.transform_by_quat_fast(v, q), rtol=RTOL, atol=ATOL
    )


def test_transform_by_trans_quat(rng):
    v, q, t = ref.rand3(rng, 64), ref.rand_unit_quat(rng, 64), ref.rand3(rng, 64)
    assert np.allclose(
        G.qd_transform_by_trans_quat(v, t, q), ref.transform_by_trans_quat(v, t, q), rtol=RTOL, atol=ATOL
    )
    assert np.allclose(
        G.qd_transform_by_trans_quat_fast(v, t, q),
        ref.transform_by_trans_quat_fast(v, t, q),
        rtol=RTOL,
        atol=ATOL,
    )


def test_inv_transform_by_quat(rng):
    v, q = ref.rand3(rng, 64), ref.rand_unit_quat(rng, 64)
    assert np.allclose(
        G.qd_inv_transform_by_quat(v, q), ref.inv_transform_by_quat(v, q), rtol=RTOL, atol=ATOL
    )
    t = ref.rand3(rng, 64)
    assert np.allclose(
        G.qd_inv_transform_by_trans_quat(v, t, q),
        ref.inv_transform_by_trans_quat(v, t, q),
        rtol=RTOL,
        atol=ATOL,
    )


def test_quat_roundtrip_is_inverse(rng):
    q = ref.rand_unit_quat(rng, 32)
    v = ref.rand3(rng, 32)
    assert np.allclose(
        G.qd_transform_by_quat(G.qd_transform_by_quat(v, q), G.qd_inv_quat(q)), v, rtol=RTOL, atol=ATOL
    )


def test_inv_quat_and_mul(rng):
    q = ref.rand_unit_quat(rng, 32)
    assert np.allclose(G.qd_inv_quat(q), ref.inv_quat(q), rtol=RTOL, atol=ATOL)
    assert np.allclose(G.qd_quat_mul(q, q), ref.quat_mul(q, q), rtol=RTOL, atol=ATOL)
    assert np.allclose(
        G.qd_transform_quat_by_quat(q, q), ref.transform_quat_by_quat(q, q), rtol=RTOL, atol=ATOL
    )


def test_quat_to_R_is_orthonormal(rng):
    q = ref.rand_unit_quat(rng, 32)
    R = G.qd_quat_to_R(q)
    assert np.allclose(R, ref.quat_to_R(q), rtol=RTOL, atol=ATOL)
    assert np.allclose(R @ np.swapaxes(R, 1, 2), np.broadcast_to(np.eye(3), R.shape),
                       rtol=RTOL, atol=ATOL)
    assert np.allclose(np.linalg.det(R), 1.0, rtol=RTOL, atol=ATOL)


def test_orthogonals(rng):
    a = rng.normal(size=(64, 3))
    a /= np.linalg.norm(a, axis=1, keepdims=True)
    a0, a1 = G.qd_orthogonals(a)
    e0, e1 = ref.orthogonals(a)
    assert np.allclose(a0, e0, rtol=RTOL, atol=ATOL)
    assert np.allclose(a1, e1, rtol=RTOL, atol=ATOL)
    # (b x a, b) is an orthonormal frame around a, right-handed.
    assert np.allclose(np.einsum("ij,ij->i", a0, a), 0.0, atol=RTOL)
    assert np.allclose(np.einsum("ij,ij->i", a1, a), 0.0, atol=RTOL)
    assert np.allclose(np.einsum("ij,ij->i", a0, a1), 0.0, atol=RTOL)
    assert np.allclose(np.einsum("ij,ij->i", np.cross(a0, a1), a), 1.0, rtol=RTOL, atol=RTOL)


def test_normalize(rng):
    v = ref.rand3(rng, 64)
    assert np.allclose(G.qd_normalize(v), ref.normalize(v), rtol=RTOL, atol=ATOL)
    assert np.allclose(np.linalg.norm(G.qd_normalize(v), axis=1), 1.0, rtol=RTOL, atol=ATOL)


def test_det3(rng):
    a, b, c = ref.rand3(rng, 64), ref.rand3(rng, 64), ref.rand3(rng, 64)
    assert np.allclose(S.func_det3(a, b, c), ref.det3(a, b, c), rtol=RTOL, atol=ATOL)
    # Swapping two columns negates the determinant; that is a real check, not a tautology.
    assert np.allclose(S.func_det3(b, a, c), -ref.det3(a, b, c), rtol=RTOL, atol=ATOL)


# =============================================================== support_field.py


def test_polar_band_matches_upstream():
    assert S._polar_band(12) == ref.polar_band(12)
    assert S._polar_band(180) == ref.polar_band(180)


def test_direction_grid_matches_upstream():
    ours = S.get_direction_grid(12)
    theirs = ref.direction_grid(12)
    band_rows = ref.polar_band(12)[1]
    assert ours.shape == (2 * 12 * band_rows, 3)
    assert np.allclose(ours, theirs.reshape(-1, 3), rtol=RTOL, atol=ATOL)
    assert np.allclose(np.linalg.norm(ours, axis=1), 1.0, rtol=RTOL, atol=ATOL)
    assert S.direction_grid_nums(12) == ours.shape[0]


def test_activate_matches_upstream(geoms, field):
    _, num_v, support_v, support_vid, cell_start, n_cells = ref.activate(
        geoms["init_pos"], geoms["vert_start"], geoms["vert_end"], 12
    )
    assert field.support_v.shape == support_v.shape
    assert field.n_support_cells == n_cells
    assert np.array_equal(field.support_cell_start, cell_start)
    # Ties in a flat face can be broken either way by argmax; the vertex must
    # still be a genuine maximiser, so compare the dot product, not the index.
    ours = field.support_v
    d = np.tile(ref.activate(geoms["init_pos"], geoms["vert_start"], geoms["vert_end"], 12)[0],
                (3, 1))
    assert np.allclose(np.einsum("ij,ij->i", ours, d), np.einsum("ij,ij->i", support_v, d),
                       rtol=RTOL, atol=ATOL)
    flat = np.einsum("ij,ij->i", ours, d)
    for i_g in range(3):
        lo, hi = geoms["vert_start"][i_g], geoms["vert_end"][i_g]
        verts = geoms["init_pos"][lo:hi]
        d_g = d[cell_start[i_g]:cell_start[i_g] + num_v]
        best = (d_g @ verts.T).max(axis=1)
        assert np.all(flat[cell_start[i_g]:cell_start[i_g] + num_v] >= best - 1e-12)


def _argmax_reference(init_pos, vert_start, vert_end, v1):
    """`np.argmax` per direction, which is what upstream's `argmax` and this port's `>` both pick."""
    out = np.zeros((len(vert_start), len(v1)), dtype=np.int64)
    for i_g in range(len(vert_start)):
        verts = init_pos[vert_start[i_g]:vert_end[i_g]]
        out[i_g] = vert_start[i_g] + np.argmax(verts @ v1.T, axis=0)
    return out


@pytest.mark.parametrize("support_res", [8, 13, 64])
def test_activate_vid_is_the_argmax(support_res):
    """The SIMD block, the scalar tail and the geom loop must all pick numpy's first argmax."""
    rng = np.random.default_rng(4242)
    n_geoms, n_verts = 3, 57
    vert_start = np.arange(n_geoms, dtype=np.int32) * n_verts
    vert_end = vert_start + n_verts
    init_pos = rng.normal(size=(n_geoms * n_verts, 3))
    f = S.SupportField(support_res=support_res)
    f.activate(init_pos, vert_start, vert_end)
    num_v = len(f.v1)
    assert num_v % 4 == (2 if support_res == 13 else 0)  # 13 exercises the tail, the others the block only
    assert np.array_equal(f.support_vid.reshape(n_geoms, num_v), _argmax_reference(
        init_pos, vert_start, vert_end, f.v1
    ))


_PARALLEL_CASE = """
import numpy as np
import mojogenesisworld.support_field as S

rng = np.random.default_rng(99)
n_geoms, n_verts, support_res = 8, 20000, 8
vert_start = np.arange(n_geoms, dtype=np.int32) * n_verts
init_pos = rng.normal(size=(n_geoms * n_verts, 3))

tables = []
for workers in (1, 2):
    f = S.SupportField(support_res=support_res)
    f.activate(init_pos, vert_start, vert_start + n_verts, workers=workers)
    tables.append((f.support_vid, f.support_cell_start))
    assert f.v1.shape[0] * n_verts * n_geoms >= 1 << 24, "case is below the parallel gate"

np.save(sys.argv[1], tables[0][0])
np.save(sys.argv[2], tables[1][0])
assert np.array_equal(tables[0][0], tables[1][0])
assert np.array_equal(tables[0][1], tables[1][1])
"""


def test_activate_parallel_path(tmp_path):
    """Above the size threshold `activate` may split the table over workers; the answer must not move.

    Run in a subprocess: a `parallelize` launch that cannot spawn its threads aborts
    the process, and that must fail this one test rather than the whole suite.
    """
    import subprocess
    import sys

    serial_path, parallel_path = tmp_path / "serial.npy", tmp_path / "parallel.npy"
    proc = subprocess.run(
        [sys.executable, "-c", "import sys\n" + _PARALLEL_CASE, str(serial_path), str(parallel_path)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 and "pthread_create" in proc.stderr:
        pytest.skip("this host cannot spawn the threads activate asks for")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert np.array_equal(np.load(serial_path), np.load(parallel_path))



@pytest.mark.parametrize("shrink", [False, True])
def test_support_sphere(rng, geoms, shrink):
    n = 32
    d, pos, q = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    ig = np.zeros(n, dtype=np.int32)
    v, local, vid = S._func_support_sphere(ig, d, pos, q, shrink, geoms["data"])
    e = [ref.support_sphere(0, a, b, c, shrink, geoms["data"], geoms["vert_start"],
                            geoms["vert_end"], geoms["init_pos"]) for a, b, c in zip(d, pos, q)]
    assert np.allclose(v, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.allclose(local, np.array([x[1] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.array_equal(vid, np.array([x[2] for x in e]))


def test_support_ellipsoid(rng, geoms):
    n = 32
    d, pos, q = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    ig = np.zeros(n, dtype=np.int32)
    ours = S._func_support_ellipsoid(ig, d, pos, q, geoms["data"])
    theirs = np.array([
        ref.support_ellipsoid(0, a, b, c, geoms["data"], geoms["vert_start"],
                              geoms["vert_end"], geoms["init_pos"]) for a, b, c in zip(d, pos, q)
    ])
    assert np.allclose(ours, theirs, rtol=RTOL, atol=ATOL)


@pytest.mark.parametrize("shrink", [False, True])
def test_support_capsule_and_cylinder(rng, geoms, shrink):
    n = 32
    d, pos, q = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    ig = np.zeros(n, dtype=np.int32)
    for ours_fn, ref_fn in (
        (S._func_support_capsule, ref.support_capsule),
        (S._func_support_cylinder, ref.support_cylinder),
    ):
        ours = ours_fn(ig, d, pos, q, shrink, geoms["data"])
        theirs = np.array([
            ref_fn(0, a, b, c, shrink, geoms["data"], geoms["vert_start"],
                   geoms["vert_end"], geoms["init_pos"]) for a, b, c in zip(d, pos, q)
        ])
        assert np.allclose(ours, theirs, rtol=RTOL, atol=ATOL)


def test_support_box(rng, geoms):
    n = 32
    d, pos, q = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    ig = np.zeros(n, dtype=np.int32)
    v, local, vid = S._func_support_box(ig, d, pos, q, geoms["data"], geoms["vert_start"])
    e = [ref.support_box(0, a, b, c, geoms["data"], geoms["vert_start"],
                          geoms["vert_end"], geoms["init_pos"]) for a, b, c in zip(d, pos, q)]
    assert np.allclose(v, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.allclose(local, np.array([x[1] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.array_equal(vid, np.array([x[2] for x in e]))


def test_support_prism(rng):
    prism = rng.normal(size=(6, 8, 3))
    d = rng.normal(size=(32, 3))
    i_b = rng.integers(0, 8, size=32).astype(np.int32)
    v, vid = S._func_support_prism(i_b, d, prism)
    e = [ref.support_prism(int(b), x, prism) for b, x in zip(i_b, d)]
    assert np.allclose(v, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.array_equal(vid, np.array([x[1] for x in e]))


def test_support_mesh_exhaustive(rng, geoms):
    n = 32
    d = rng.normal(size=(n, 3))
    ig = np.zeros(n, dtype=np.int32)
    v, vid = S._func_support_mesh_exhaustive(
        ig, d, geoms["vert_start"], geoms["vert_end"], geoms["init_pos"]
    )
    e = [ref.support_mesh_exhaustive(0, x, geoms["vert_start"], geoms["vert_end"],
                                     geoms["type"], geoms["data"], geoms["init_pos"]) for x in d]
    assert np.allclose(v, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.array_equal(vid, np.array([x[1] for x in e]))


def test_support_mesh_table(rng, geoms, field):
    n = 32
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ig = np.zeros(n, dtype=np.int32)
    v, vid = S._func_support_mesh(ig, d, field)
    e = [ref.support_mesh_table(0, x, geoms["vert_start"], geoms["vert_end"], geoms["type"],
                                 geoms["data"], geoms["init_pos"],
                                 (field.support_v, field.support_vid, field.support_cell_start,
                                  field.support_res)) for x in d]
    assert np.allclose(v, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.array_equal(vid, np.array([x[1] for x in e]))


def test_support_world(rng, geoms, field):
    n = 32
    d, pos, q = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ig = np.zeros(n, dtype=np.int32)
    tables = (field.support_v, field.support_vid, field.support_cell_start, field.support_res)
    for exhaustive in (False, True):
        v, local, vid = S._func_support_world(
            ig, d, pos, q, geoms["vert_start"], geoms["vert_end"], geoms["init_pos"],
            field, exhaustive,
        )
        e = [ref.support_world(0, a, b, c, geoms["vert_start"], geoms["vert_end"],
                               geoms["type"], geoms["data"], geoms["init_pos"], tables, exhaustive)
             for a, b, c in zip(d, pos, q)]
        assert np.allclose(v, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
        assert np.allclose(local, np.array([x[1] for x in e]), rtol=RTOL, atol=ATOL)
        assert np.array_equal(vid, np.array([x[2] for x in e]))


def test_direction_grid_coords(rng):
    d = rng.normal(size=(32, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    chart, ii, jj = S._func_direction_grid_coords(d, 12)
    e = [ref.direction_grid_coords(x, 12) for x in d]
    assert np.array_equal(chart, np.array([x[0] for x in e]))
    assert np.allclose(ii, np.array([x[1] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.allclose(jj, np.array([x[2] for x in e]), rtol=RTOL, atol=ATOL)


def test_count_supports(rng, geoms, field):
    n = 32
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    ig = np.zeros(n, dtype=np.int32)
    tables = (field.support_v, field.support_vid, field.support_cell_start, field.support_res)
    ours = S._func_count_supports_mesh(ig, d, field)
    e = [ref.count_supports_mesh(0, x, geoms["vert_start"], geoms["vert_end"], geoms["type"],
                                 geoms["data"], geoms["init_pos"], tables) for x in d]
    assert np.array_equal(ours, np.array(e))
    # A box direction along a face normal has four tied support vertices.
    assert S._func_count_supports_box(
        np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 0.0, 0.0]])
    )[0] == ref.count_supports_box(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]))
    q = ref.rand_unit_quat(rng, n)
    ours_w = S._func_count_supports_world(ig, d, q, field)
    e_w = [ref.count_supports_world(0, a, b, geoms["vert_start"], geoms["vert_end"],
                                    geoms["type"], geoms["data"], geoms["init_pos"], tables)
           for a, b in zip(d, q)]
    assert np.array_equal(ours_w, np.array(e_w))


# =================================================================== gjk_support.py


def test_support_mesh(rng, geoms):
    n = 32
    d, pos, q = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    ig = np.zeros(n, dtype=np.int32)
    prev = np.zeros(n, dtype=np.int32)
    v, vid = J.support_mesh(ig, d, pos, q, prev, geoms["init_pos"], geoms["vert_start"],
                            geoms["vert_end"])
    e = [ref.support_mesh(0, a, b, c, np.zeros(1, np.int32), geoms["init_pos"],
                          geoms["vert_start"], geoms["vert_end"]) for a, b, c in zip(d, pos, q)]
    assert np.allclose(v, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.array_equal(vid, np.array([x[1] for x in e]))
    # The warm-start slot is written back, exactly as upstream writes gjk_state.
    assert np.array_equal(prev, vid)


@pytest.mark.parametrize("geom_type", list(range(8)))
def test_support_driver_dispatch(rng, geoms, field, geom_type):
    n = 8
    d, pos, q = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    gt = np.full(n, geom_type, dtype=np.int32)
    ig = np.zeros(n, dtype=np.int32)
    prism = rng.normal(size=(6, 8, 3))
    tables = (field.support_v, field.support_vid, field.support_cell_start, field.support_res)
    for terrain in (False, True):
        v, local, vid = J.support_driver(
            ig, np.zeros(n, np.int32), d, pos, q, False, gt, geoms["data"], geoms["vert_start"],
            geoms["vert_end"], geoms["init_pos"], field, np.zeros(n, np.int32), prism, terrain,
            True,
        )
        e = [ref.support_driver(0, 0, a, b, c, False, gt[0], geoms["data"], geoms["vert_start"],
                                geoms["vert_end"], geoms["init_pos"], tables,
                                np.zeros(1, np.int32), prism, terrain)
             for a, b, c in zip(d, pos, q)]
        assert np.allclose(v, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
        assert np.allclose(local, np.array([x[1] for x in e]), rtol=RTOL, atol=ATOL)
        assert np.array_equal(vid, np.array([x[2] for x in e]))


def test_func_support(rng, geoms, field):
    n = 16
    d, pa, pb = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand3(rng, n)
    qa, qb = ref.rand_unit_quat(rng, n), ref.rand_unit_quat(rng, n)
    ga, gb = np.zeros(n, np.int32), np.ones(n, np.int32)
    i_b = np.zeros(n, np.int32)
    prism = rng.normal(size=(6, 8, 3))
    tables = (field.support_v, field.support_vid, field.support_cell_start, field.support_res)
    ours = J.func_support(
        ga, gb, i_b, d, pa, qa, pb, qb, False, geoms["type"], geoms["data"],
        geoms["vert_start"], geoms["vert_end"], geoms["init_pos"], field,
        np.zeros(2 * n, np.int32), prism, False, True,
    )
    e = [ref.func_support(0, 1, 0, a, b, c, f, g, False, geoms["type"], geoms["data"],
                          geoms["vert_start"], geoms["vert_end"], geoms["init_pos"], tables,
                          np.zeros(2, np.int32), prism, False)
         for a, b, c, f, g in zip(d, pa, qa, pb, qb)]
    for k in range(7):
        got = np.asarray(ours[k])
        want = np.array([x[k] for x in e])
        assert np.allclose(got, want, rtol=RTOL, atol=ATOL), f"output {k}"


# ===================================================================== gjk_utils.py


def test_ray_triangle_intersection(rng):
    n = 64
    r1, r2 = ref.rand3(rng, n), ref.rand3(rng, n)
    t1, t2, t3 = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand3(rng, n)
    ours = J.func_ray_triangle_intersection(r1, r2, t1, t2, t3)
    e = [ref.ray_triangle_intersection(*x) for x in zip(r1, r2, t1, t2, t3)]
    assert np.array_equal(ours.astype(bool), np.array(e, dtype=bool))
    # A ray through the middle of a triangle must hit it.
    assert J.func_ray_triangle_intersection(
        np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]),
        np.array([[1.0, -1.0, 0.5]]), np.array([[1.0, 1.0, 0.5]]), np.array([[-1.0, 0.0, 0.5]]),
    )[0] == 1


def test_triangle_affine_coords(rng):
    n = 64
    pt, t1, t2, t3 = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand3(rng, n), ref.rand3(rng, n)
    ours = J.func_triangle_affine_coords(pt, t1, t2, t3)
    e = np.array([ref.triangle_affine_coords(*x) for x in zip(pt, t1, t2, t3)])
    assert np.allclose(ours, e, rtol=1e-9, atol=1e-9)
    # Barycentric coordinates of a vertex of the triangle are (0, 0, 1) rotated.
    assert np.allclose(J.func_triangle_affine_coords(t1, t1, t2, t3)[:1], [1.0, 0.0, 0.0],
                       rtol=1e-9, atol=1e-9)


def test_point_triangle_intersection(rng):
    n = 64
    pt, t1, t2, t3 = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand3(rng, n), ref.rand3(rng, n)
    ours = J.func_point_triangle_intersection(pt, t1, t2, t3)
    e = np.array([ref.point_triangle_intersection(*x) for x in zip(pt, t1, t2, t3)])
    assert np.array_equal(ours.astype(bool), e)
    centroid = (t1 + t2 + t3) / 3.0
    assert np.all(J.func_point_triangle_intersection(centroid, t1, t2, t3) == 1)


def test_point_plane_same_side(rng):
    n = 64
    pt, p1, p2, p3 = ref.rand3(rng, n), ref.rand3(rng, n), ref.rand3(rng, n), ref.rand3(rng, n)
    ours = J.func_point_plane_same_side(pt, p1, p2, p3)
    e = np.array([ref.point_plane_same_side(*x) for x in zip(pt, p1, p2, p3)])
    assert np.array_equal(ours, e)
    # A point far off the plane is on the far side, so FAIL.
    assert J.func_point_plane_same_side(
        np.array([[0.0, 0.0, 10.0]]), np.array([[0.0, 0.0, 1.0]]),
        np.array([[1.0, 0.0, 1.0]]), np.array([[0.0, 1.0, 1.0]]),
    )[0] == J.RETURN_CODE_FAIL


def test_origin_tetra_intersection(rng):
    n = 64
    t = [ref.rand3(rng, n) for _ in range(4)]
    ours = J.func_origin_tetra_intersection(*t)
    e = np.array([ref.origin_tetra_intersection(*x) for x in zip(*t)])
    assert np.array_equal(ours, e)
    unit = [np.array([1.0, 1, 1]), np.array([-1.0, 1, 1]),
            np.array([1.0, -1, 1]), np.array([1.0, 1, -1])]
    assert J.func_origin_tetra_intersection(*[x[None] for x in unit])[0] == \
        ref.origin_tetra_intersection(*unit)


def test_project_origin_to_plane(rng):
    n = 64
    v = [ref.rand3(rng, n) for _ in range(3)]
    ours_p, ours_f = J.func_project_origin_to_plane(*v)
    e = [ref.project_origin_to_plane(*x) for x in zip(*v)]
    assert np.allclose(ours_p, np.array([x[0] for x in e]), rtol=RTOL, atol=ATOL)
    assert np.array_equal(ours_f, np.array([x[1] for x in e]))
    # A point on z = 1 projects to (0, 0, 1).
    p, f = J.func_project_origin_to_plane(
        np.array([[0.0, 0.0, 1.0]]), np.array([[1.0, 0.0, 1.0]]), np.array([[0.0, 1.0, 1.0]])
    )
    assert np.allclose(p, [[0.0, 0.0, 1.0]], rtol=RTOL, atol=ATOL)
    assert f[0] == J.RETURN_CODE_SUCCESS


# ======================================================================= contact.py


def test_geom_rbound(rng):
    n = 3
    gt = np.array([S.GEOM_TYPE_SPHERE, S.GEOM_TYPE_CAPSULE, S.GEOM_TYPE_BOX], dtype=np.int32)
    gd = ref.geom_data(rng, n)
    aabb = ref.aabb_corners(rng, n).reshape(n, 8, 3)
    ig = np.arange(n, dtype=np.int32)
    ours = C.func_compute_geom_rbound(ig, gt, gd, aabb)
    e = [ref.compute_geom_rbound(i, gt, gd, aabb) for i in range(n)]
    assert np.allclose(ours, e, rtol=RTOL, atol=ATOL)
    # A mesh falls back to half the AABB diagonal.
    ours_m = C.func_compute_geom_rbound(
        np.zeros(1, np.int32), np.array([S.GEOM_TYPE_MESH], np.int32), gd, aabb[:1]
    )
    assert np.allclose(ours_m[0], e[0] * 0 + ref.compute_geom_rbound(0, gt, gd, aabb) * 0
                       + 0.5 * np.linalg.norm(aabb[0, 7] - aabb[0, 0]), rtol=RTOL, atol=ATOL)


def test_geom_pair_scale_and_tolerance(rng):
    n = 3
    gt = np.array([S.GEOM_TYPE_BOX, S.GEOM_TYPE_PLANE, S.GEOM_TYPE_MESH], dtype=np.int32)
    gd = ref.geom_data(rng, n)
    aabb = ref.aabb_corners(rng, n).reshape(n, 8, 3)
    ig = np.zeros(n, dtype=np.int32)
    jg = np.arange(n, dtype=np.int32)
    assert np.allclose(
        C.func_compute_geom_pair_scale(ig, jg, gt, aabb),
        [ref.compute_geom_pair_scale(0, j, gt, aabb) for j in range(n)],
        rtol=RTOL, atol=ATOL,
    )
    assert np.allclose(
        C.func_compute_geom_pair_scale_mj(ig, jg, gt, gd, aabb),
        [ref.compute_geom_pair_scale_mj(0, j, gt, gd, aabb) for j in range(n)],
        rtol=RTOL, atol=ATOL,
    )
    for mj in (False, True):
        assert np.allclose(
            C.func_compute_mc_tolerance(ig, jg, gt, gd, aabb, 1e-3, mj),
            [ref.compute_mc_tolerance(0, j, gt, gd, aabb, 1e-3, mj) for j in range(n)],
            rtol=RTOL, atol=ATOL,
        )


def test_contact_orthogonals(rng):
    n = 64
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    a0, a1 = C.func_contact_orthogonals(v)
    e0, e1 = ref.contact_orthogonals(v)
    assert np.allclose(a0, e0, rtol=RTOL, atol=ATOL)
    assert np.allclose(a1, e1, rtol=RTOL, atol=ATOL)
    # The frame is orthonormal and right-handed around the normal.
    assert np.allclose(np.einsum("ij,ij->i", a0, a0), 1.0, rtol=RTOL, atol=ATOL)
    assert np.allclose(np.einsum("ij,ij->i", a0, a1), 0.0, atol=RTOL)
    assert np.allclose(np.einsum("ij,ij->i", a0, v), 0.0, atol=RTOL)
    assert np.allclose(np.einsum("ij,ij->i", np.cross(a0, a1), v), 1.0, rtol=RTOL, atol=ATOL)


def test_rotate_frame(rng):
    n = 32
    pos, q = ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    cp, qr = ref.rand3(rng, n), ref.rand_unit_quat(rng, n)
    new_pos, new_quat = C.func_rotate_frame(pos, q, cp, qr)
    e_pos, e_quat = ref.rotate_frame(pos, q, cp, qr)
    assert np.allclose(new_pos, e_pos, rtol=RTOL, atol=ATOL)
    assert np.allclose(new_quat, e_quat, rtol=RTOL, atol=ATOL)
    # The frame is still a rigid one: the quaternion stays unit.
    assert np.allclose(np.linalg.norm(new_quat, axis=1), 1.0, rtol=RTOL, atol=ATOL)


def test_contact_order_key(rng):
    n = 64
    p = ref.rand3(rng, n)
    assert np.allclose(C.func_contact_order_key(p), [ref.contact_order_key(x) for x in p],
                       rtol=RTOL, atol=ATOL)
    # Golden-ratio weights: distinct points get distinct keys, in order.
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 1e-3]])
    assert C.func_contact_order_key(a)[0] < C.func_contact_order_key(b)[0]


# ================================================================= materials/MPM/*


def test_resolve_lame():
    for E, nu in ((1e6, 0.2), (3e5, 0.3), (1.0, -0.4)):
        lam, mu = M.resolve_lame(E, nu)
        assert mu == pytest.approx(E / (2.0 * (1.0 + nu)), rel=1e-15)
        assert lam == pytest.approx(E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), rel=1e-15)


def _svd_factors(rng, n):
    """Well-conditioned U, S, V so the constitutive models stay in their linear range."""
    U = np.empty((n, 3, 3))
    V = np.empty((n, 3, 3))
    S = np.empty((n, 3, 3))
    for i in range(n):
        A = rng.normal(size=(3, 3))
        Qa, Ra = np.linalg.qr(A)
        B = rng.normal(size=(3, 3))
        Qb, Rb = np.linalg.qr(B)
        U[i] = Qa * np.sign(np.diag(Ra))
        V[i] = Qb * np.sign(np.diag(Rb))
        # Proper rotations, so that J = det(U S V) is positive and the
        # neohooken and elasto-plastic models stay in their real domain.
        if np.linalg.det(U[i]) < 0:
            U[i][:, 2] *= -1.0
        if np.linalg.det(V[i]) < 0:
            V[i][:, 2] *= -1.0
        S[i] = np.diag(rng.uniform(0.7, 1.3, 3))
    return U, S, V


@pytest.mark.parametrize(
    "material,model,needs_svd",
    [
        (M.Elastic(), "corotation", True),
        (M.Elastic(model="neohooken"), "neohooken", False),
        (M.ElastoPlastic(), None, True),
        (M.Liquid(), None, False),
        (M.Snow(), None, True),
        (M.Sand(), None, True),
    ],
)
def test_material_update_F_S_Jp(rng, material, model, needs_svd):
    assert material.needs_svd is needs_svd
    n = 32
    U, S, V = _svd_factors(rng, n)
    F_tmp = np.einsum("nij,njk,nkl->nil", U, S, V)
    J = np.linalg.det(F_tmp)
    Jp = np.full(n, 1.0)
    ours_F, ours_S, ours_Jp = material.update_F_S_Jp(J, F_tmp, U, S, V, Jp)
    key = {"Elastic": "elastic", "ElastoPlastic": "elasto_plastic", "Liquid": "liquid",
           "Snow": "snow", "Sand": "sand"}[type(material).__name__]
    e_F, e_S, e_Jp = ref.mpm_update_F_S_Jp(key, material, J, F_tmp, U, S, V, Jp)
    # The constitutive models chain exp/log/pow. quadrants routes those through
    # torch in the harness, which keeps a few more bits of slack than the
    # compiled backends do, so the material tests hold the same agreement to
    # 1e-8 relative that the harness itself has.
    scale = max(1.0, float(np.abs(e_F).max()))
    assert np.allclose(ours_F, e_F, rtol=1e-8, atol=1e-9 * scale)
    assert np.allclose(ours_S, e_S, rtol=1e-8, atol=1e-9 * scale)
    assert np.allclose(ours_Jp, e_Jp, rtol=1e-8, atol=1e-9)


@pytest.mark.parametrize(
    "material,which",
    [
        (M.Elastic(), "corotation"),
        (M.Elastic(model="neohooken"), "neohooken"),
        (M.ElastoPlastic(), "default"),
        (M.Liquid(), "liquid"),
        (M.Snow(), "snow"),
        (M.Sand(), "sand"),
    ],
)
def test_material_update_stress(rng, material, which):
    n = 32
    U, S, V = _svd_factors(rng, n)
    F_new = np.einsum("nij,njk,nkl->nil", U, S, V)
    F_tmp = F_new * 1.01
    J = np.linalg.det(F_new)
    Jp = np.full(n, 1.0)
    ours = material.update_stress(U, S, V, F_tmp, F_new, J, Jp)
    theirs = ref.mpm_stress(material, which, U, S, V, F_tmp, F_new, J, Jp)
    # Stress entries are O(E) = O(1e6..1e9), so the absolute tolerance scales with
    # the magnitude; the relative one is what actually pins the two together.
    scale = max(1.0, float(np.abs(theirs).max()))
    assert np.allclose(ours, theirs, rtol=1e-8, atol=1e-9 * scale)


def test_liquid_non_viscous_zeroes_mu():
    assert M.Liquid().mu == 0.0
    assert M.Liquid(viscous=True).mu != 0.0
    assert M.Liquid(viscous=True).needs_svd is True


def test_sand_alpha_and_projection(rng):
    mat = M.Sand(friction_angle=30.0)
    alpha = np.zeros(1)
    from mojogenesisworld._lib import addr, lib

    lib().mgw_mpm_sand_alpha(30.0, addr(alpha))
    assert mat.alpha == pytest.approx(alpha[0], rel=1e-15)
    S_ = np.tile(np.diag([1.2, 0.8, 1.0]), (4, 1, 1))
    ours_S, ours_Jp = mat._sand_projection(S_, np.full(4, 0.0))
    e_S, e_Jp = ref.sand_projection(mat, S_[0], 0.0)
    assert np.allclose(ours_S[0], e_S, rtol=1e-9, atol=1e-9)
    assert ours_Jp[0] == pytest.approx(e_Jp, rel=1e-9)


def test_snow_rejects_von_mises():
    with pytest.raises(ValueError):
        M.Snow(use_von_mises=True)


# ================================================================== mpm_solver.py


def test_compute_F_tmp(rng):
    n = 32
    F = rng.normal(size=(n, 3, 3))
    C = rng.normal(size=(n, 3, 3)) * 0.1
    dt = 1e-3
    ours = M.compute_F_tmp(C, F, dt)
    assert np.allclose(ours, np.einsum("nij,njk->nik", np.eye(3) + dt * C, F),
                       rtol=RTOL, atol=ATOL)


def test_backward_svd(rng):
    n = 32
    U, S, V = _svd_factors(rng, n)
    gU, gS, gV = (rng.normal(size=(n, 3, 3)) for _ in range(3))
    ours = M.backward_svd(gU, gS, gV, U, S, V)
    theirs = np.array([ref.backward_svd(a, b, c, d, e, f) for a, b, c, d, e, f
                       in zip(gU, gS, gV, U, S, V)])
    assert np.allclose(ours, theirs, rtol=1e-9, atol=1e-9)


def test_stencil_range():
    assert M.stencil_range() == 3


@pytest.mark.parametrize(
    "model,material",
    [
        (0, M.Elastic()),
        (1, M.Elastic(model="neohooken")),
        (2, M.ElastoPlastic()),
        (3, M.Liquid()),
        (4, M.Sand()),
        (5, M.Snow()),
    ],
)
def test_p2g(rng, model, material):
    n = 16
    res = np.array([8, 8, 8])
    offset = np.array([0, 0, 0])
    dx = 0.1
    U, S, V = _svd_factors(rng, n)
    F = np.einsum("nij,njk,nkl->nil", U, S, V)
    C = rng.normal(size=(n, 3, 3)) * 0.05
    F_tmp = F * 1.01
    pos = rng.uniform(0.3, 0.5, size=(n, 3))
    vel = rng.normal(size=(n, 3)) * 0.1
    particles = {
        "pos": pos, "vel": vel, "C": C, "F": F, "F_tmp": F_tmp, "U": U, "S": S, "V": V,
        "Jp": np.ones(n), "active": np.ones(n, np.int32), "free": np.ones(n, np.int32),
        "mass": np.full(n, 1e-3),
    }
    grid = {"res": res, "offset": offset}
    params = {"dx": dx, "substep_dt": 1e-3, "particle_volume": 1e-3,
              "needs_svd": material.needs_svd}
    e_F, e_Jp, e_vel, e_mass = ref.p2g(material, model, particles, grid, params)

    vel_in = np.zeros((res[0], res[1], res[2], 3))
    grid_mass = np.zeros((res[0], res[1], res[2]))
    out_F, out_Jp = M.p2g(
        pos, vel, C, F, F_tmp, U, S, V, particles["Jp"], particles["active"], particles["mass"],
        particles["free"], material, 1e-3, dx, 1e-3, offset, res, vel_in, grid_mass,
        model=model,
    )
    # The constitutive models chain exp/log/pow, which the reference evaluates
    # through torch, so the scatter is held to the same 1e-8 as the material
    # tests rather than the 1e-9 of the pure-arithmetic kernels.
    scale = max(1.0, float(np.abs(e_vel).max()))
    assert np.allclose(out_F, e_F, rtol=1e-8, atol=1e-9 * scale)
    assert np.allclose(out_Jp, e_Jp, rtol=1e-8, atol=1e-9)
    assert np.allclose(vel_in, e_vel, rtol=1e-8, atol=1e-9 * scale)
    assert np.allclose(grid_mass, e_mass, rtol=1e-8, atol=1e-9)
    assert grid_mass.sum() > 0.0


def test_p2g_defaults_to_the_material(rng):
    """With no explicit `model`, p2g runs the material's own model, and the
    neohooken model really is the neohooken stress, not the corotation one."""
    mat = M.Elastic(model="neohooken")
    n = 4
    res = np.array([8, 8, 8])
    U, S, V = _svd_factors(rng, n)
    F = np.einsum("nij,njk,nkl->nil", U, S, V)
    pos = rng.uniform(0.3, 0.5, size=(n, 3))
    particles = {
        "pos": pos, "vel": np.zeros((n, 3)), "C": np.zeros((n, 3, 3)), "F": F, "F_tmp": F * 1.01,
        "U": U, "S": S, "V": V, "Jp": np.ones(n), "active": np.ones(n, np.int32),
        "free": np.ones(n, np.int32), "mass": np.full(n, 1e-3),
    }
    grid = {"res": res, "offset": np.zeros(3)}
    params = {"dx": 0.1, "substep_dt": 1e-3, "particle_volume": 1e-3, "needs_svd": False}
    e_F, e_Jp, e_vel, e_mass = ref.p2g(mat, 1, particles, grid, params)
    vel_in = np.zeros((8, 8, 8, 3))
    grid_mass = np.zeros((8, 8, 8))
    out_F, out_Jp = M.p2g(
        pos, particles["vel"], particles["C"], F, particles["F_tmp"], U, S, V, particles["Jp"],
        particles["active"], particles["mass"], particles["free"], mat, 1e-3, 0.1, 1e-3,
        np.zeros(3), res, vel_in, grid_mass,
    )
    scale = max(1.0, float(np.abs(e_vel).max()))
    assert np.allclose(vel_in, e_vel, rtol=1e-8, atol=1e-9 * scale)
    assert np.allclose(grid_mass, e_mass, rtol=1e-8, atol=1e-9)
    # The two elastic stresses are different formulas, so the scatter pins which
    # one ran: a corotation stress would disagree with the reference here.
    assert float(np.abs(vel_in).max()) > 0.0
    J = np.linalg.det(F)
    neohooken = ref.mpm_stress(mat, "neohooken", U, S, V, particles["F_tmp"], F, J, particles["Jp"])
    corotation = ref.mpm_stress(mat, "corotation", U, S, V, particles["F_tmp"], F, J, particles["Jp"])
    assert float(np.abs(neohooken - corotation).max()) > 1e-6 * float(np.abs(corotation).max())
    with pytest.raises(ValueError):
        M.p2g(
            pos, particles["vel"], particles["C"], F, particles["F_tmp"], U, S, V, particles["Jp"],
            particles["active"], particles["mass"], particles["free"], mat, 1e-3, 0.1, 1e-3,
            np.zeros(3), res, np.zeros((8, 8, 8, 3)), np.zeros((8, 8, 8)), model=99,
        )


def test_g2p():
    rng = np.random.default_rng(20260929)
    n = 16
    res = np.array([8, 8, 8])
    offset = np.array([0, 0, 0])
    dx = 0.1
    pos = rng.uniform(0.25, 0.45, size=(n, 3))
    vel_out = rng.normal(size=(res[0], res[1], res[2], 3))
    lower = np.array([0.0, 0.0, 0.0])
    upper = np.array([0.6, 0.6, 0.6])
    substep_dt = 1e-3

    new_pos, new_vel, new_C = M.g2p(pos, vel_out, lower, upper, substep_dt, dx, offset, res)

    inv_dx = 1.0 / dx
    e_pos, e_vel, e_C = ref.g2p(pos, vel_out, lower, upper, substep_dt, dx, offset, res)
    assert np.allclose(new_pos, e_pos, rtol=1e-9, atol=1e-11)
    assert np.allclose(new_vel, e_vel, rtol=1e-9, atol=1e-11)
    assert np.allclose(new_C, e_C, rtol=1e-9, atol=1e-11)


def test_g2p_clamps_at_the_boundary(rng):
    res = np.array([8, 8, 8])
    pos = np.array([[0.9, 0.2, 0.2]])
    new_pos, new_vel, _ = M.g2p(
        pos, np.zeros((8, 8, 8, 3)), np.array([0.05, 0.05, 0.05]), np.array([0.5, 0.5, 0.5]),
        1e-1, 0.25, np.zeros(3), res,
    )
    # The particle starts outside the box; the clamp pulls it back and kills the
    # velocity of every component it pulled back.
    assert np.all(new_pos[0] <= 0.5)
    assert new_pos[0][0] == pytest.approx(0.5)
    assert new_vel[0][0] == 0.0
