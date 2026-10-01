"""`genesis.utils.geom` quaternion and vector helpers, in Mojo.

Every function here is named, ordered and shaped like its upstream counterpart in
`genesis/utils/geom.py`. They take and return NumPy arrays with the same meaning
the engine's `qd_vec3` / `qd.types.vector(4)` values have, and they are batched:
`qd_transform_by_quat(v, quat)` with `v` of shape `(n, 3)` runs the kernel `n`
times, which is the elementwise map the engine's per-thread invocation performs.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, align, f64, lib

EPS = 1e-15


def _v3(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).reshape(-1, 3)


def _v4(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64).reshape(-1, 4)


def _out3(n: int) -> np.ndarray:
    return np.zeros((n, 3), dtype=np.float64)


def _out4(n: int) -> np.ndarray:
    return np.zeros((n, 4), dtype=np.float64)


def _bc(a, n: int, width: int) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 1:
        arr = np.broadcast_to(arr, (n, width))
    return np.ascontiguousarray(arr)


def qd_transform_by_quat_fast(v, quat) -> np.ndarray:
    a, b = _v3(v), _v4(quat)
    a, b = align(a.shape[0], v=a, quat=b)
    dst = _out3(a.shape[0])
    lib().mgw_transform_by_quat_fast(addr(a), addr(b), addr(dst), a.shape[0])
    return dst


def qd_transform_by_trans_quat_fast(pos, trans, quat) -> np.ndarray:
    a, c = _v3(pos), _v4(quat)
    a, c = align(a.shape[0], pos=a, quat=c)
    b = _bc(trans, a.shape[0], 3)
    dst = _out3(a.shape[0])
    lib().mgw_transform_by_trans_quat_fast(addr(a), addr(b), addr(c), addr(dst), a.shape[0])
    return dst


def qd_quat_to_R(quat, eps: float = EPS) -> np.ndarray:
    a = _v4(quat)
    dst = np.zeros((a.shape[0], 3, 3), dtype=np.float64)
    lib().mgw_quat_to_R(addr(a), eps, addr(dst.reshape(-1, 9)), a.shape[0])
    return dst


def qd_inv_quat(quat) -> np.ndarray:
    a = _v4(quat)
    dst = _out4(a.shape[0])
    lib().mgw_inv_quat(addr(a), addr(dst), a.shape[0])
    return dst


def qd_quat_mul(q, u) -> np.ndarray:
    a, b = _v4(q), _v4(u)
    a, b = align(a.shape[0], q=a, u=b)
    dst = _out4(a.shape[0])
    lib().mgw_quat_mul(addr(a), addr(b), addr(dst), a.shape[0])
    return dst


def qd_transform_quat_by_quat(v, u) -> np.ndarray:
    a, b = _v4(v), _v4(u)
    a, b = align(a.shape[0], v=a, u=b)
    dst = _out4(a.shape[0])
    lib().mgw_transform_quat_by_quat(addr(a), addr(b), addr(dst), a.shape[0])
    return dst


def qd_transform_by_quat(v, quat) -> np.ndarray:
    a, b = _v3(v), _v4(quat)
    a, b = align(a.shape[0], v=a, quat=b)
    dst = _out3(a.shape[0])
    lib().mgw_transform_by_quat(addr(a), addr(b), addr(dst), a.shape[0])
    return dst


def qd_inv_transform_by_quat(v, quat) -> np.ndarray:
    a, b = _v3(v), _v4(quat)
    a, b = align(a.shape[0], v=a, quat=b)
    dst = _out3(a.shape[0])
    lib().mgw_inv_transform_by_quat(addr(a), addr(b), addr(dst), a.shape[0])
    return dst


def qd_transform_by_trans_quat(pos, trans, quat) -> np.ndarray:
    a, c = _v3(pos), _v4(quat)
    a, c = align(a.shape[0], pos=a, quat=c)
    b = _bc(trans, a.shape[0], 3)
    dst = _out3(a.shape[0])
    lib().mgw_transform_by_trans_quat(addr(a), addr(b), addr(c), addr(dst), a.shape[0])
    return dst


def qd_inv_transform_by_trans_quat(pos, trans, quat) -> np.ndarray:
    a, c = _v3(pos), _v4(quat)
    a, c = align(a.shape[0], pos=a, quat=c)
    b = _bc(trans, a.shape[0], 3)
    dst = _out3(a.shape[0])
    lib().mgw_inv_transform_by_trans_quat(addr(a), addr(b), addr(c), addr(dst), a.shape[0])
    return dst


def qd_orthogonals(a) -> tuple[np.ndarray, np.ndarray]:
    x = _v3(a)
    n = x.shape[0]
    a0, a1 = _out3(n), _out3(n)
    lib().mgw_orthogonals(addr(x), addr(a0), addr(a1), n)
    return a0, a1


def qd_normalize(v, eps: float = EPS) -> np.ndarray:
    a = _v3(v)
    dst = _out3(a.shape[0])
    lib().mgw_normalize(addr(a), eps, addr(dst), a.shape[0])
    return dst
