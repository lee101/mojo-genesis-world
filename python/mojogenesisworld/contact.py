"""`genesis.engine.solvers.rigid.collider.contact`, the numeric parts, in Mojo.

`func_contact_orthogonals` is ported in its MuJoCo-compatible branch only: the
other branch reads `geoms_init_AABB`, the link's inertial quaternion and a
compile-time rigid-config flag, all of which are engine state rather than an
argument. See the README's coverage section.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, align, f64, i32, lib
from .geom import _out3, _v3, _v4


def func_compute_geom_rbound(i_g, geom_type, geoms_data, geoms_init_AABB) -> np.ndarray:
    """Compute the bounding sphere radius for a geom, matching MuJoCo's geom_rbound."""
    i_g = i32(i_g)
    out = np.zeros(i_g.shape[0], dtype=np.float64)
    geom_type = i32(geom_type)
    geoms_data = _v3(geoms_data)
    geoms_init_AABB = f64(geoms_init_AABB)
    lib().mgw_compute_geom_rbound(
        addr(i_g), addr(geom_type), addr(geoms_data), addr(geoms_init_AABB),
        addr(out), i_g.shape[0],
    )
    return out


def func_compute_geom_pair_scale(i_ga, i_gb, geom_type, geoms_init_AABB) -> np.ndarray:
    i_ga, i_gb = i32(i_ga), i32(i_gb)
    i_ga, i_gb = align(i_ga.shape[0], i_ga=i_ga, i_gb=i_gb)
    geom_type, geoms_init_AABB = i32(geom_type), f64(geoms_init_AABB)
    out = np.zeros(i_ga.shape[0], dtype=np.float64)
    lib().mgw_compute_geom_pair_scale(
        addr(i_ga), addr(i_gb), addr(geom_type), addr(geoms_init_AABB), addr(out), i_ga.shape[0]
    )
    return out


def func_compute_geom_pair_scale_mj(i_ga, i_gb, geom_type, geoms_data, geoms_init_AABB) -> np.ndarray:
    """Geom-pair length scale matching MuJoCo's formula: min(rbound_g1, rbound_g2)."""
    i_ga, i_gb = i32(i_ga), i32(i_gb)
    i_ga, i_gb = align(i_ga.shape[0], i_ga=i_ga, i_gb=i_gb)
    geom_type = i32(geom_type)
    geoms_data, geoms_init_AABB = _v3(geoms_data), f64(geoms_init_AABB)
    out = np.zeros(i_ga.shape[0], dtype=np.float64)
    lib().mgw_compute_geom_pair_scale_mj(
        addr(i_ga), addr(i_gb), addr(geom_type), addr(geoms_data),
        addr(geoms_init_AABB), addr(out), i_ga.shape[0],
    )
    return out


def func_compute_mc_tolerance(
    i_ga, i_gb, geom_type, geoms_data, geoms_init_AABB, mc_tolerance: float,
    enable_mujoco_compatibility: bool,
) -> np.ndarray:
    """Absolute multi-contact acceptance tolerance of a geom pair."""
    i_ga, i_gb = i32(i_ga), i32(i_gb)
    i_ga, i_gb = align(i_ga.shape[0], i_ga=i_ga, i_gb=i_gb)
    geom_type = i32(geom_type)
    geoms_data, geoms_init_AABB = _v3(geoms_data), f64(geoms_init_AABB)
    out = np.zeros(i_ga.shape[0], dtype=np.float64)
    lib().mgw_compute_mc_tolerance(
        addr(i_ga), addr(i_gb), addr(geom_type), addr(geoms_data),
        addr(geoms_init_AABB), mc_tolerance, 1 if enable_mujoco_compatibility else 0,
        addr(out), i_ga.shape[0],
    )
    return out


def func_contact_orthogonals(normal) -> tuple[np.ndarray, np.ndarray]:
    """MuJoCo-compatible branch: build an orthonormal frame around a contact normal."""
    a = _v3(normal)
    n = a.shape[0]
    a0, a1 = _out3(n), _out3(n)
    lib().mgw_contact_orthogonals(addr(a), addr(a0), addr(a1), n)
    return a0, a1


def func_rotate_frame(pos, quat, contact_pos, qrot):
    """Rotate a body's frame by [qrot] about a point that stays put in the new frame."""
    pos, quat, contact_pos, qrot = _v3(pos), _v4(quat), _v3(contact_pos), _v4(qrot)
    pos, quat, contact_pos, qrot = align(
        pos.shape[0], pos=pos, quat=quat, contact_pos=contact_pos, qrot=qrot
    )
    n = pos.shape[0]
    out_pos, out_quat = _out3(n), np.zeros((n, 4), dtype=np.float64)
    lib().mgw_rotate_frame(addr(pos), addr(quat), addr(contact_pos), addr(qrot), addr(out_pos), addr(out_quat), n)
    return out_pos, out_quat


def func_contact_order_key(pos) -> np.ndarray:
    """Order a contact position along one generic direction, as a single scalar."""
    a = _v3(pos)
    out = np.zeros(a.shape[0], dtype=np.float64)
    lib().mgw_contact_order_key(addr(a), addr(out), a.shape[0])
    return out
