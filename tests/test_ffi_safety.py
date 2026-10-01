"""The batched entry points index every input at the same row.

A shorter input would be read past its end inside the Mojo loop, and an input
array that dies before the `ctypes` call returns would leave the kernel reading
freed memory. Neither is visible in a numerical comparison, so they are checked
here directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from mojogenesisworld import contact as C, geom as G, gjk as J, mpm as M, support_field as S

N = 6


@pytest.fixture
def quats():
    rng = np.random.default_rng(7)
    q = rng.normal(size=(N, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def test_short_quat_is_rejected(quats):
    with pytest.raises(ValueError, match="quat"):
        G.qd_transform_by_quat(np.zeros((N, 3)), quats[: N - 1])


def test_short_triangle_is_rejected():
    with pytest.raises(ValueError, match="tri_v3"):
        J.func_point_triangle_intersection(
            np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N, 3)), np.zeros((N - 1, 3))
        )


def test_short_geom_pair_is_rejected():
    geoms_init_AABB = np.zeros((2, 8, 3))
    geom_type = np.zeros(2, dtype=np.int32)
    with pytest.raises(ValueError, match="i_gb"):
        C.func_compute_geom_pair_scale(np.zeros(N, dtype=np.int32), np.zeros(N - 1, np.int32),
                                       geom_type, geoms_init_AABB)


def test_short_support_direction_is_rejected(quats):
    geoms_data = np.zeros((2, 3))
    with pytest.raises(ValueError, match="quat"):
        S._func_support_sphere(np.zeros(N, np.int32), np.zeros((N, 3)), np.zeros((N, 3)),
                               quats[:2], False, geoms_data)


def test_short_mpm_matrix_is_rejected():
    m = np.zeros((N, 3, 3))
    with pytest.raises(ValueError, match="S"):
        M.Elastic().update_F_S_Jp(np.ones(N), m, m, np.zeros((N - 1, 3, 3)), m, np.ones(N))


def test_short_p2g_mass_is_rejected():
    n = 4
    m = np.zeros((n, 3, 3))
    grid = np.zeros(8 * 3)
    with pytest.raises(ValueError, match="mass"):
        M.p2g(np.zeros((n, 3)), np.zeros((n, 3)), m, m, m, m, m, m, np.ones(n),
              np.ones(n, np.int32), np.ones(n - 1), np.ones(n, np.int32), M.Elastic(),
              1e-3, 0.1, 1.0, np.zeros(3), np.full(3, 8.0), grid, grid)


def test_single_row_is_broadcast(quats):
    """One row of quats is shared by many vectors, which is how the engine calls it."""
    got = G.qd_transform_by_quat(np.ones((N, 3)), quats[0])
    assert got.shape == (N, 3)
    assert np.allclose(got[0], got[-1])


def test_activate_accepts_one_worker_by_default():
    """`workers` is optional and the serial loop is the default."""
    init_pos = np.random.default_rng(1).normal(size=(60, 3))
    vert_start = np.array([0, 30], dtype=np.int32)
    default, serial = S.SupportField(support_res=8), S.SupportField(support_res=8)
    default.activate(init_pos, vert_start, vert_start + 30)
    serial.activate(init_pos, vert_start, vert_start + 30, workers=1)
    assert np.array_equal(default.support_vid, serial.support_vid)