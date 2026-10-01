"""`genesis.engine.solvers.rigid.collider.gjk_support` and `.gjk_utils`, in Mojo.

Same names, same argument order, same return values as upstream. The batched
entry points take one row of inputs per call the engine makes per thread; the
seven values `func_support` returns upstream come back as a tuple of arrays.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, align, i32, lib
from .geom import _out3, _v3, _v4
from .support_field import SupportField, _bc

RETURN_CODE_SUCCESS = 0
RETURN_CODE_FAIL = 1


def _flag(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.int32)


def support_mesh(
    i_g, direction, pos, quat, prev_vertex_id, init_pos, vert_start, vert_end
):
    """Find the support point on a mesh in the given direction."""
    direction, pos, quat = _v3(direction), _v3(pos), _v4(quat)
    direction, pos, quat = align(direction.shape[0], direction=direction, pos=pos, quat=quat)
    n = direction.shape[0]
    out_v = _out3(n)
    out_vid = _flag(n)
    init_pos = _v3(init_pos)
    lib().mgw_support_mesh(
        addr(i32(i_g)), addr(direction), addr(pos), addr(quat), addr(prev_vertex_id),
        addr(init_pos), addr(i32(vert_start)), addr(i32(vert_end)), addr(out_v), addr(out_vid), n,
    )
    return out_v, out_vid


def support_driver(
    i_g,
    i_b,
    direction,
    pos,
    quat,
    shrink_sphere: bool,
    geom_type,
    geoms_data,
    vert_start,
    vert_end,
    init_pos,
    field: SupportField,
    prev_vertex_id,
    prism,
    has_terrain: bool,
    enable_mujoco_compatibility: bool,
):
    """@ shrink_sphere: If True, use point and line support for sphere and capsule."""
    direction, pos, quat = _v3(direction), _v3(pos), _v4(quat)
    direction, pos, quat = align(direction.shape[0], direction=direction, pos=pos, quat=quat)
    n = direction.shape[0]
    out_v, out_local, out_vid = _out3(n), _out3(n), _flag(n)
    geom_type = i32(geom_type)
    geoms_data, init_pos, prism = _v3(geoms_data), _v3(init_pos), _v3(prism)
    lib().mgw_support_driver(
        addr(i32(i_g)), addr(i32(i_b)), addr(direction), addr(pos), addr(quat),
        1 if shrink_sphere else 0, addr(geom_type), addr(geoms_data),
        addr(i32(vert_start)), addr(i32(vert_end)), addr(init_pos), field.support_res,
        addr(field.support_cell_start), addr(field.support_vid), addr(field.support_v),
        addr(prev_vertex_id), addr(prism), 1 if has_terrain else 0,
        1 if enable_mujoco_compatibility else 0, addr(out_v), addr(out_local), addr(out_vid), n,
    )
    return out_v, out_local, out_vid


def func_support(
    i_ga,
    i_gb,
    i_b,
    dir,
    pos_a,
    quat_a,
    pos_b,
    quat_b,
    shrink_sphere: bool,
    geom_type,
    geoms_data,
    vert_start,
    vert_end,
    init_pos,
    field: SupportField,
    prev_vertex_id,
    prism,
    has_terrain: bool,
    enable_mujoco_compatibility: bool,
):
    """Find support points on the two objects using [dir].

    Parameters:
    ----------
    dir: gs.qd_vec3
        The direction in which to find the support points, from [ga] (obj 1) to [gb] (obj 2).
    """
    dir, pos_a, quat_a = _v3(dir), _v3(pos_a), _v4(quat_a)
    pos_b, quat_b = _v3(pos_b), _v4(quat_b)
    dir, pos_a, quat_a, pos_b, quat_b = align(
        dir.shape[0], dir=dir, pos_a=pos_a, quat_a=quat_a, pos_b=pos_b, quat_b=quat_b
    )
    n = dir.shape[0]
    o1, o2, l1, l2, mink = (_out3(n) for _ in range(5))
    id1, id2 = _flag(n), _flag(n)
    geom_type = i32(geom_type)
    geoms_data, init_pos, prism = _v3(geoms_data), _v3(init_pos), _v3(prism)
    lib().mgw_func_support(
        addr(i32(i_ga)), addr(i32(i_gb)), addr(i32(i_b)), addr(dir), addr(pos_a), addr(quat_a),
        addr(pos_b), addr(quat_b), 1 if shrink_sphere else 0, addr(geom_type),
        addr(geoms_data), addr(i32(vert_start)), addr(i32(vert_end)), addr(init_pos),
        field.support_res, addr(field.support_cell_start), addr(field.support_vid),
        addr(field.support_v), addr(prev_vertex_id), addr(prism),
        1 if has_terrain else 0, 1 if enable_mujoco_compatibility else 0,
        addr(o1), addr(o2), addr(l1), addr(l2), addr(id1), addr(id2), addr(mink), n,
    )
    return o1, o2, l1, l2, id1, id2, mink


# ---------------------------------------------------------------- gjk_utils.py


def func_ray_triangle_intersection(ray_v1, ray_v2, tri_v1, tri_v2, tri_v3) -> np.ndarray:
    """Check if the ray intersects the triangle."""
    a, b, c, d, e = _v3(ray_v1), _v3(ray_v2), _v3(tri_v1), _v3(tri_v2), _v3(tri_v3)
    a, b, c, d, e = align(a.shape[0], ray_v1=a, ray_v2=b, tri_v1=c, tri_v2=d, tri_v3=e)
    out = _flag(a.shape[0])
    lib().mgw_ray_triangle_intersection(addr(a), addr(b), addr(c), addr(d), addr(e), addr(out), a.shape[0])
    return out


def func_triangle_affine_coords(point, tri_v1, tri_v2, tri_v3) -> np.ndarray:
    """Compute the affine coordinates of the point with respect to the triangle."""
    a, b, c, d = _v3(point), _v3(tri_v1), _v3(tri_v2), _v3(tri_v3)
    a, b, c, d = align(a.shape[0], point=a, tri_v1=b, tri_v2=c, tri_v3=d)
    out = _out3(a.shape[0])
    lib().mgw_triangle_affine_coords(addr(a), addr(b), addr(c), addr(d), addr(out), a.shape[0])
    return out


def func_point_triangle_intersection(point, tri_v1, tri_v2, tri_v3) -> np.ndarray:
    """Check if the point is inside the triangle."""
    a, b, c, d = _v3(point), _v3(tri_v1), _v3(tri_v2), _v3(tri_v3)
    a, b, c, d = align(a.shape[0], point=a, tri_v1=b, tri_v2=c, tri_v3=d)
    out = _flag(a.shape[0])
    lib().mgw_point_triangle_intersection(addr(a), addr(b), addr(c), addr(d), addr(out), a.shape[0])
    return out


def func_point_plane_same_side(point, plane_v1, plane_v2, plane_v3) -> np.ndarray:
    """Check if the point is on the same side of the plane as the origin."""
    a, b, c, d = _v3(point), _v3(plane_v1), _v3(plane_v2), _v3(plane_v3)
    out = _flag(a.shape[0])
    lib().mgw_point_plane_same_side(addr(a), addr(b), addr(c), addr(d), addr(out), a.shape[0])
    return out


def func_origin_tetra_intersection(tet_v1, tet_v2, tet_v3, tet_v4) -> np.ndarray:
    """Check if the origin is inside the tetrahedron."""
    a, b, c, d = _v3(tet_v1), _v3(tet_v2), _v3(tet_v3), _v3(tet_v4)
    a, b, c, d = align(a.shape[0], tet_v1=a, tet_v2=b, tet_v3=c, tet_v4=d)
    out = _flag(a.shape[0])
    lib().mgw_origin_tetra_intersection(addr(a), addr(b), addr(c), addr(d), addr(out), a.shape[0])
    return out


def func_project_origin_to_plane(v1, v2, v3):
    """Project the origin onto the plane defined by the simplex vertices."""
    a, b, c = _v3(v1), _v3(v2), _v3(v3)
    a, b, c = align(a.shape[0], v1=a, v2=b, v3=c)
    n = a.shape[0]
    point = _out3(n)
    flag = _flag(n)
    lib().mgw_project_origin_to_plane(addr(a), addr(b), addr(c), addr(point), addr(flag), n)
    return point, flag
