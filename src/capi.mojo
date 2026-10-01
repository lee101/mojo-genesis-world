"""The C ABI the Python bindings call through.

Every buffer crosses as an `Int` address, because `@export` refuses parametric
functions and a pointer with an inferred origin is parametric. Callers own all
memory, including scratch, so nothing here allocates and nothing here can leak.

The kernels take one argument per upstream argument, in upstream order. What the
batched entry points add on top is a leading count and one index array per
indexed argument: upstream gets those indices from a `qd.ndrange` and from
`array_class` fields, neither of which survives the C ABI.

Layout, everywhere: `(n, 3)` and `(n, 4)` are row-major with no padding,
`(n, 9)` is a row-major 3x3, `(n_geoms, 3)` geom parameters are
`dyn_info.geoms.data`, `(n_geoms, 8, 3)` AABBs are the eight rest-pose corners
in `array_class` order (corner 0 lower, corner 7 upper), and `(i, j)` int fields
are `Int32` like `gs.qd_int`.
"""

from ported import (
    IPtr,
    Mat3,
    Ptr,
    Quat,
    Vec3,
    _aabb_corner,
    _cell_index,
    _func_count_supports_box,
    _func_count_supports_mesh,
    _func_count_supports_world,
    _func_direction_grid_coords,
    _func_support_box,
    _func_support_capsule,
    _func_support_cylinder,
    _func_support_ellipsoid,
    _func_support_mesh,
    _func_support_mesh_exhaustive,
    _func_support_prism,
    _func_support_sphere,
    _func_support_world,
    _polar_band,
    _prism_corner,
    _quadratic_weights,
    _sand_projection,
    _stencil_offset,
    _update_F_S_Jp_elastic,
    _update_F_S_Jp_elasto_plastic,
    _update_F_S_Jp_liquid,
    _update_F_S_Jp_noop,
    _update_F_S_Jp_sand,
    _update_F_S_Jp_snow,
    _update_stress_corotation,
    _update_stress_default,
    _update_stress_liquid,
    _update_stress_neohooken,
    _update_stress_sand,
    _update_stress_snow,
    activate,
    backward_svd,
    compute_F_tmp,
    direction_grid_nums,
    func_compute_geom_pair_scale,
    func_compute_geom_pair_scale_mj,
    func_compute_geom_rbound,
    func_compute_mc_tolerance,
    func_contact_order_key,
    func_contact_orthogonals,
    func_det3,
    func_origin_tetra_intersection,
    func_point_plane_same_side,
    func_point_triangle_intersection,
    func_project_origin_to_plane,
    func_ray_triangle_intersection,
    func_rotate_frame,
    func_support,
    func_triangle_affine_coords,
    g2p,
    get_direction_grid,
    p2g,
    qd_inv_quat,
    qd_inv_transform_by_quat,
    qd_inv_transform_by_trans_quat,
    qd_normalize,
    qd_orthogonals,
    qd_quat_mul,
    qd_quat_to_R,
    qd_transform_by_quat,
    qd_transform_by_quat_fast,
    qd_transform_by_trans_quat,
    qd_transform_by_trans_quat_fast,
    qd_transform_quat_by_quat,
    resolve_lame,
    sand_alpha,
    stencil_range,
    support_driver,
    support_mesh,
)


def p(a: Int) -> Ptr:
    return Ptr(unsafe_from_address=a)


def ip(a: Int) -> IPtr:
    return IPtr(unsafe_from_address=a)


def ld3(q: Ptr, i: Int) -> Vec3:
    var o = 3 * i
    return Vec3(q.unsafe_load(o), q.unsafe_load(o + 1), q.unsafe_load(o + 2))


def st3(q: Ptr, i: Int, v: Vec3):
    var o = 3 * i
    q.unsafe_store(o, v[0])
    q.unsafe_store(o + 1, v[1])
    q.unsafe_store(o + 2, v[2])


def ldq(q: Ptr, i: Int) -> Quat:
    var o = 4 * i
    return Quat(q.unsafe_load(o), q.unsafe_load(o + 1), q.unsafe_load(o + 2), q.unsafe_load(o + 3))


def stq(q: Ptr, i: Int, v: Quat):
    var o = 4 * i
    q.unsafe_store(o, v[0])
    q.unsafe_store(o + 1, v[1])
    q.unsafe_store(o + 2, v[2])
    q.unsafe_store(o + 3, v[3])


def ldm(q: Ptr, i: Int) -> Mat3:
    var o = 9 * i
    return Mat3(
        Vec3(q.unsafe_load(o), q.unsafe_load(o + 1), q.unsafe_load(o + 2)),
        Vec3(q.unsafe_load(o + 3), q.unsafe_load(o + 4), q.unsafe_load(o + 5)),
        Vec3(q.unsafe_load(o + 6), q.unsafe_load(o + 7), q.unsafe_load(o + 8)),
    )


def stm(q: Ptr, i: Int, m: Mat3):
    var o = 9 * i
    for i2 in range(3):
        for j in range(3):
            q.unsafe_store(o + 3 * i2 + j, m.at(i2, j))


# ================================================================ geom.py
@export("mgw_transform_by_quat")
def mgw_transform_by_quat(v: Int, quat: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        st3(p(dst), i, qd_transform_by_quat(ld3(p(v), i), ldq(p(quat), i)))


@export("mgw_transform_by_quat_fast")
def mgw_transform_by_quat_fast(v: Int, quat: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        st3(p(dst), i, qd_transform_by_quat_fast(ld3(p(v), i), ldq(p(quat), i)))


@export("mgw_transform_by_trans_quat")
def mgw_transform_by_trans_quat(pos: Int, trans: Int, quat: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        st3(p(dst), i, qd_transform_by_trans_quat(ld3(p(pos), i), ld3(p(trans), i), ldq(p(quat), i)))


@export("mgw_transform_by_trans_quat_fast")
def mgw_transform_by_trans_quat_fast(pos: Int, trans: Int, quat: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        st3(p(dst), i, qd_transform_by_trans_quat_fast(ld3(p(pos), i), ld3(p(trans), i), ldq(p(quat), i)))


@export("mgw_inv_transform_by_quat")
def mgw_inv_transform_by_quat(v: Int, quat: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        st3(p(dst), i, qd_inv_transform_by_quat(ld3(p(v), i), ldq(p(quat), i)))


@export("mgw_inv_transform_by_trans_quat")
def mgw_inv_transform_by_trans_quat(pos: Int, trans: Int, quat: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        st3(p(dst), i, qd_inv_transform_by_trans_quat(ld3(p(pos), i), ld3(p(trans), i), ldq(p(quat), i)))


@export("mgw_inv_quat")
def mgw_inv_quat(quat: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        stq(p(dst), i, qd_inv_quat(ldq(p(quat), i)))


@export("mgw_quat_mul")
def mgw_quat_mul(a: Int, b: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        stq(p(dst), i, qd_quat_mul(ldq(p(a), i), ldq(p(b), i)))


@export("mgw_transform_quat_by_quat")
def mgw_transform_quat_by_quat(v: Int, u: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        stq(p(dst), i, qd_transform_quat_by_quat(ldq(p(v), i), ldq(p(u), i)))


@export("mgw_quat_to_R")
def mgw_quat_to_R(quat: Int, eps: Float64, dst: Int, n: Int) abi("C"):
    for i in range(n):
        stm(p(dst), i, qd_quat_to_R(ldq(p(quat), i), eps))


@export("mgw_orthogonals")
def mgw_orthogonals(a: Int, axis_0: Int, axis_1: Int, n: Int) abi("C"):
    for i in range(n):
        var r = qd_orthogonals(ld3(p(a), i))
        st3(p(axis_0), i, r[0])
        st3(p(axis_1), i, r[1])


@export("mgw_normalize")
def mgw_normalize(v: Int, eps: Float64, dst: Int, n: Int) abi("C"):
    for i in range(n):
        st3(p(dst), i, qd_normalize(ld3(p(v), i), eps))


# ================================================= collider/utils.py
@export("mgw_det3")
def mgw_det3(v1: Int, v2: Int, v3: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        p(dst).unsafe_offset(i)[] = func_det3(ld3(p(v1), i), ld3(p(v2), i), ld3(p(v3), i))


# ================================================ collider/support_field.py
@export("mgw_polar_band")
def mgw_polar_band(support_res: Int) abi("C") -> Int:
    var r = _polar_band(support_res)
    return r[0] + 1000000 * r[1]


@export("mgw_direction_grid_nums")
def mgw_direction_grid_nums(support_res: Int) abi("C") -> Int:
    return direction_grid_nums(support_res)


@export("mgw_get_direction_grid")
def mgw_get_direction_grid(dst: Int, support_res: Int) abi("C") -> Int:
    return get_direction_grid(p(dst), support_res)


@export("mgw_activate")
def mgw_activate(
    init_pos: Int, vert_start: Int, vert_end: Int, n_geoms: Int, v1: Int, num_v: Int,
    support_v: Int, support_vid: Int, support_cell_start: Int, n_workers: Int,
) abi("C"):
    activate(
        p(init_pos), ip(vert_start), ip(vert_end), n_geoms, p(v1), num_v,
        p(support_v), ip(support_vid), ip(support_cell_start), n_workers,
    )


@export("mgw_direction_grid_coords")
def mgw_direction_grid_coords(d_mesh: Int, support_res: Int, chart: Int, ii: Int, jj: Int, n: Int) abi("C"):
    for i in range(n):
        var r = _func_direction_grid_coords(ld3(p(d_mesh), i), support_res)
        ip(chart).unsafe_offset(i)[] = r[0]
        p(ii).unsafe_offset(i)[] = r[1]
        p(jj).unsafe_offset(i)[] = r[2]


@export("mgw_support_mesh_table")
def mgw_support_mesh_table(
    i_g: Int, d_mesh: Int, support_res: Int, support_cell_start: Int, support_vid: Int,
    support_v: Int, out_v: Int, out_vid: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _func_support_mesh(
            Int(ip(i_g).unsafe_load(i)), ld3(p(d_mesh), i), support_res, ip(support_cell_start), ip(support_vid), p(support_v)
        )
        st3(p(out_v), i, r[0])
        ip(out_vid).unsafe_offset(i)[] = r[1]


@export("mgw_support_mesh_exhaustive")
def mgw_support_mesh_exhaustive(
    i_g: Int, d_mesh: Int, vert_start: Int, vert_end: Int, init_pos: Int, out_v: Int, out_vid: Int, n: Int,
) abi("C"):
    for i in range(n):
        var g = Int(ip(i_g).unsafe_load(i))
        var r = _func_support_mesh_exhaustive(
            g, ld3(p(d_mesh), i), Int(ip(vert_start).unsafe_load(g)), Int(ip(vert_end).unsafe_load(g)), p(init_pos)
        )
        st3(p(out_v), i, r[0])
        ip(out_vid).unsafe_offset(i)[] = r[1]


@export("mgw_support_sphere")
def mgw_support_sphere(
    i_g: Int, d: Int, pos: Int, quat: Int, shrink: Int, geoms_data: Int, out_v: Int, out_vlocal: Int, out_vid: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _func_support_sphere(
            Int(ip(i_g).unsafe_load(i)), ld3(p(d), i), ld3(p(pos), i), ldq(p(quat), i), shrink != 0, p(geoms_data)
        )
        st3(p(out_v), i, r[0])
        st3(p(out_vlocal), i, r[1])
        ip(out_vid).unsafe_offset(i)[] = r[2]


@export("mgw_support_ellipsoid")
def mgw_support_ellipsoid(i_g: Int, d: Int, pos: Int, quat: Int, geoms_data: Int, out_v: Int, n: Int) abi("C"):
    for i in range(n):
        st3(p(out_v), i, _func_support_ellipsoid(Int(ip(i_g).unsafe_load(i)), ld3(p(d), i), ld3(p(pos), i), ldq(p(quat), i), p(geoms_data)))


@export("mgw_support_capsule")
def mgw_support_capsule(
    i_g: Int, d: Int, pos: Int, quat: Int, shrink: Int, geoms_data: Int, out_v: Int, n: Int,
) abi("C"):
    for i in range(n):
        st3(
            p(out_v), i,
            _func_support_capsule(Int(ip(i_g).unsafe_load(i)), ld3(p(d), i), ld3(p(pos), i), ldq(p(quat), i), shrink != 0, p(geoms_data)),
        )


@export("mgw_support_cylinder")
def mgw_support_cylinder(
    i_g: Int, d: Int, pos: Int, quat: Int, shrink: Int, geoms_data: Int, out_v: Int, n: Int,
) abi("C"):
    for i in range(n):
        st3(
            p(out_v), i,
            _func_support_cylinder(Int(ip(i_g).unsafe_load(i)), ld3(p(d), i), ld3(p(pos), i), ldq(p(quat), i), shrink != 0, p(geoms_data)),
        )


@export("mgw_support_prism")
def mgw_support_prism(i_b: Int, d: Int, prism: Int, out_v: Int, out_vid: Int, n: Int) abi("C"):
    for i in range(n):
        var r = _func_support_prism(Int(ip(i_b).unsafe_load(i)), ld3(p(d), i), p(prism))
        st3(p(out_v), i, r[0])
        ip(out_vid).unsafe_offset(i)[] = r[1]


@export("mgw_support_box")
def mgw_support_box(
    i_g: Int, d: Int, pos: Int, quat: Int, geoms_data: Int, vert_start: Int, out_v: Int, out_vlocal: Int,
    out_vid: Int, n: Int,
) abi("C"):
    for i in range(n):
        var g = Int(ip(i_g).unsafe_load(i))
        var r = _func_support_box(g, ld3(p(d), i), ld3(p(pos), i), ldq(p(quat), i), p(geoms_data), Int(ip(vert_start).unsafe_load(g)))
        st3(p(out_v), i, r[0])
        st3(p(out_vlocal), i, r[1])
        ip(out_vid).unsafe_offset(i)[] = r[2]


@export("mgw_support_world")
def mgw_support_world(
    i_g: Int, d: Int, pos: Int, quat: Int, vert_start: Int, vert_end: Int, init_pos: Int, support_res: Int,
    support_cell_start: Int, support_vid: Int, support_v: Int, exhaustive: Int, out_v: Int, out_vlocal: Int,
    out_vid: Int, n: Int,
) abi("C"):
    for i in range(n):
        var g = Int(ip(i_g).unsafe_load(i))
        var r = _func_support_world(
            g, ld3(p(d), i), ld3(p(pos), i), ldq(p(quat), i), Int(ip(vert_start).unsafe_load(g)), Int(ip(vert_end).unsafe_load(g)),
            p(init_pos), support_res, ip(support_cell_start), ip(support_vid), p(support_v), exhaustive != 0,
        )
        st3(p(out_v), i, r[0])
        st3(p(out_vlocal), i, r[1])
        ip(out_vid).unsafe_offset(i)[] = r[2]


@export("mgw_count_supports_mesh")
def mgw_count_supports_mesh(
    i_g: Int, d_mesh: Int, support_res: Int, support_cell_start: Int, support_v: Int, out_count: Int, n: Int,
) abi("C"):
    for i in range(n):
        ip(out_count).unsafe_offset(i)[] = _func_count_supports_mesh(
            Int(ip(i_g).unsafe_load(i)), ld3(p(d_mesh), i), support_res, ip(support_cell_start), p(support_v)
        )


@export("mgw_count_supports_world")
def mgw_count_supports_world(
    i_g: Int, d: Int, quat: Int, support_res: Int, support_cell_start: Int, support_v: Int, out_count: Int, n: Int,
) abi("C"):
    for i in range(n):
        ip(out_count).unsafe_offset(i)[] = _func_count_supports_world(
            Int(ip(i_g).unsafe_load(i)), ld3(p(d), i), ldq(p(quat), i), support_res, ip(support_cell_start), p(support_v)
        )


@export("mgw_count_supports_box")
def mgw_count_supports_box(d: Int, quat: Int, out_count: Int, n: Int) abi("C"):
    for i in range(n):
        ip(out_count).unsafe_offset(i)[] = _func_count_supports_box(ld3(p(d), i), ldq(p(quat), i))


# ================================================== collider/gjk_support.py
@export("mgw_support_mesh")
def mgw_support_mesh(
    i_g: Int, direction: Int, pos: Int, quat: Int, prev_vertex_id: Int, init_pos: Int, vert_start: Int,
    vert_end: Int, out_v: Int, out_vid: Int, n: Int,
) abi("C"):
    for i in range(n):
        var g = Int(ip(i_g).unsafe_load(i))
        var r = support_mesh(
            g, ld3(p(direction), i), ld3(p(pos), i), ldq(p(quat), i), ip(prev_vertex_id), i, p(init_pos),
            Int(ip(vert_start).unsafe_load(g)), Int(ip(vert_end).unsafe_load(g)),
        )
        st3(p(out_v), i, r[0])
        ip(out_vid).unsafe_offset(i)[] = r[1]


@export("mgw_support_driver")
def mgw_support_driver(
    i_g: Int, i_b: Int, direction: Int, pos: Int, quat: Int, shrink_sphere: Int, geom_type: Int,
    geoms_data: Int, vert_start: Int, vert_end: Int, init_pos: Int, support_res: Int, support_cell_start: Int,
    support_vid: Int, support_v: Int, prev_vertex_id: Int, prism: Int, has_terrain: Int,
    enable_mujoco_compatibility: Int, out_v: Int, out_vlocal: Int, out_vid: Int, n: Int,
) abi("C"):
    for i in range(n):
        var g = Int(ip(i_g).unsafe_load(i))
        var r = support_driver(
            g, ld3(p(direction), i), ld3(p(pos), i), ldq(p(quat), i), shrink_sphere != 0, ip(geom_type).unsafe_load(g),
            p(geoms_data), Int(ip(vert_start).unsafe_load(g)), Int(ip(vert_end).unsafe_load(g)), p(init_pos), support_res,
            ip(support_cell_start), ip(support_vid), p(support_v), p(prism), Int(ip(i_b).unsafe_load(i)),
            has_terrain != 0, enable_mujoco_compatibility != 0, ip(prev_vertex_id), i,
        )
        st3(p(out_v), i, r[0])
        st3(p(out_vlocal), i, r[1])
        ip(out_vid).unsafe_offset(i)[] = r[2]


@export("mgw_func_support")
def mgw_func_support(
    i_ga: Int, i_gb: Int, i_b: Int, dir: Int, pos_a: Int, quat_a: Int, pos_b: Int, quat_b: Int,
    shrink_sphere: Int, geom_type: Int, geoms_data: Int, vert_start: Int, vert_end: Int, init_pos: Int,
    support_res: Int, support_cell_start: Int, support_vid: Int, support_v: Int, prev_vertex_id: Int,
    prism: Int, has_terrain: Int, enable_mujoco_compatibility: Int,
    out_obj1: Int, out_obj2: Int, out_local1: Int, out_local2: Int, out_id1: Int, out_id2: Int,
    out_minkowski: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = func_support(
            Int(ip(i_ga).unsafe_load(i)), Int(ip(i_gb).unsafe_load(i)), ld3(p(dir), i), ld3(p(pos_a), i), ldq(p(quat_a), i),
            ld3(p(pos_b), i), ldq(p(quat_b), i), shrink_sphere != 0, ip(geom_type), p(geoms_data),
            ip(vert_start), ip(vert_end), p(init_pos), support_res, ip(support_cell_start), ip(support_vid),
            p(support_v), ip(prev_vertex_id), p(prism), Int(ip(i_b).unsafe_load(i)), has_terrain != 0,
            enable_mujoco_compatibility != 0,
        )
        st3(p(out_obj1), i, r[0])
        st3(p(out_obj2), i, r[1])
        st3(p(out_local1), i, r[2])
        st3(p(out_local2), i, r[3])
        ip(out_id1).unsafe_offset(i)[] = r[4]
        ip(out_id2).unsafe_offset(i)[] = r[5]
        st3(p(out_minkowski), i, r[6])


# ==================================================== collider/gjk_utils.py
@export("mgw_ray_triangle_intersection")
def mgw_ray_triangle_intersection(
    ray_v1: Int, ray_v2: Int, tri_v1: Int, tri_v2: Int, tri_v3: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        ip(dst).unsafe_offset(i)[] = func_ray_triangle_intersection(
            ld3(p(ray_v1), i), ld3(p(ray_v2), i), ld3(p(tri_v1), i), ld3(p(tri_v2), i), ld3(p(tri_v3), i)
        )


@export("mgw_triangle_affine_coords")
def mgw_triangle_affine_coords(point: Int, tri_v1: Int, tri_v2: Int, tri_v3: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        st3(
            p(dst), i,
            func_triangle_affine_coords(ld3(p(point), i), ld3(p(tri_v1), i), ld3(p(tri_v2), i), ld3(p(tri_v3), i)),
        )


@export("mgw_point_triangle_intersection")
def mgw_point_triangle_intersection(point: Int, tri_v1: Int, tri_v2: Int, tri_v3: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        ip(dst).unsafe_offset(i)[] = func_point_triangle_intersection(
            ld3(p(point), i), ld3(p(tri_v1), i), ld3(p(tri_v2), i), ld3(p(tri_v3), i)
        )


@export("mgw_point_plane_same_side")
def mgw_point_plane_same_side(point: Int, plane_v1: Int, plane_v2: Int, plane_v3: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        ip(dst).unsafe_offset(i)[] = func_point_plane_same_side(
            ld3(p(point), i), ld3(p(plane_v1), i), ld3(p(plane_v2), i), ld3(p(plane_v3), i)
        )


@export("mgw_origin_tetra_intersection")
def mgw_origin_tetra_intersection(tet_v1: Int, tet_v2: Int, tet_v3: Int, tet_v4: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        ip(dst).unsafe_offset(i)[] = func_origin_tetra_intersection(
            ld3(p(tet_v1), i), ld3(p(tet_v2), i), ld3(p(tet_v3), i), ld3(p(tet_v4), i)
        )


@export("mgw_project_origin_to_plane")
def mgw_project_origin_to_plane(v1: Int, v2: Int, v3: Int, out_point: Int, out_flag: Int, n: Int) abi("C"):
    for i in range(n):
        var r = func_project_origin_to_plane(ld3(p(v1), i), ld3(p(v2), i), ld3(p(v3), i))
        st3(p(out_point), i, r[0])
        ip(out_flag).unsafe_offset(i)[] = r[1]


# ======================================================== collider/contact.py
@export("mgw_compute_geom_rbound")
def mgw_compute_geom_rbound(
    i_g: Int, geom_type: Int, geoms_data: Int, geoms_init_AABB: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        var g = Int(ip(i_g).unsafe_load(i))
        p(dst).unsafe_offset(i)[] = func_compute_geom_rbound(g, ip(geom_type).unsafe_load(g), p(geoms_data), p(geoms_init_AABB))


@export("mgw_compute_geom_pair_scale")
def mgw_compute_geom_pair_scale(
    i_ga: Int, i_gb: Int, geom_type: Int, geoms_init_AABB: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        var ga = Int(ip(i_ga).unsafe_load(i))
        var gb = Int(ip(i_gb).unsafe_load(i))
        p(dst).unsafe_offset(i)[] = func_compute_geom_pair_scale(ga, gb, ip(geom_type).unsafe_load(ga), p(geoms_init_AABB))


@export("mgw_compute_geom_pair_scale_mj")
def mgw_compute_geom_pair_scale_mj(
    i_ga: Int, i_gb: Int, geom_type: Int, geoms_data: Int, geoms_init_AABB: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        p(dst).unsafe_offset(i)[] = func_compute_geom_pair_scale_mj(
            Int(ip(i_ga).unsafe_load(i)), Int(ip(i_gb).unsafe_load(i)), ip(geom_type), p(geoms_data), p(geoms_init_AABB)
        )


@export("mgw_compute_mc_tolerance")
def mgw_compute_mc_tolerance(
    i_ga: Int, i_gb: Int, geom_type: Int, geoms_data: Int, geoms_init_AABB: Int, mc_tolerance: Float64,
    enable_mujoco_compatibility: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        p(dst).unsafe_offset(i)[] = func_compute_mc_tolerance(
            Int(ip(i_ga).unsafe_load(i)), Int(ip(i_gb).unsafe_load(i)), ip(geom_type), p(geoms_data), p(geoms_init_AABB),
            mc_tolerance, enable_mujoco_compatibility != 0,
        )


@export("mgw_contact_orthogonals")
def mgw_contact_orthogonals(normal: Int, out_0: Int, out_1: Int, n: Int) abi("C"):
    for i in range(n):
        var r = func_contact_orthogonals(ld3(p(normal), i))
        st3(p(out_0), i, r[0])
        st3(p(out_1), i, r[1])


@export("mgw_rotate_frame")
def mgw_rotate_frame(pos: Int, quat: Int, contact_pos: Int, qrot: Int, out_pos: Int, out_quat: Int, n: Int) abi("C"):
    for i in range(n):
        var r = func_rotate_frame(ld3(p(pos), i), ldq(p(quat), i), ld3(p(contact_pos), i), ldq(p(qrot), i))
        st3(p(out_pos), i, r[0])
        stq(p(out_quat), i, r[1])


@export("mgw_contact_order_key")
def mgw_contact_order_key(pos: Int, dst: Int, n: Int) abi("C"):
    for i in range(n):
        p(dst).unsafe_offset(i)[] = func_contact_order_key(ld3(p(pos), i))


# ================================================== materials/MPM/base.py
@export("mgw_resolve_lame")
def mgw_resolve_lame(E: Float64, nu: Float64, out_lam: Int, out_mu: Int) abi("C"):
    var r = resolve_lame(E, nu)
    p(out_lam).unsafe_offset(0)[] = r[0]
    p(out_mu).unsafe_offset(0)[] = r[1]


@export("mgw_mpm_update_F_S_Jp_noop")
def mgw_mpm_update_F_S_Jp_noop(
    J: Int, F_tmp: Int, U: Int, S: Int, V: Int, Jp: Int, out_F: Int, out_S: Int, out_Jp: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _update_F_S_Jp_noop(p(J).unsafe_load(i), ldm(p(F_tmp), i), ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), p(Jp).unsafe_load(i))
        stm(p(out_F), i, r[0])
        stm(p(out_S), i, r[1])
        p(out_Jp).unsafe_offset(i)[] = r[2]


@export("mgw_mpm_update_stress_default")
def mgw_mpm_update_stress_default(
    mu: Float64, lam: Float64, U: Int, S: Int, V: Int, F_tmp: Int, F_new: Int, J: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        stm(
            p(dst), i,
            _update_stress_default(mu, lam, ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), ldm(p(F_tmp), i), ldm(p(F_new), i), p(J).unsafe_load(i)),
        )


# ================================================ materials/MPM/elastic.py
@export("mgw_mpm_update_F_S_Jp_elastic")
def mgw_mpm_update_F_S_Jp_elastic(
    J: Int, F_tmp: Int, U: Int, S: Int, V: Int, Jp: Int, out_F: Int, out_S: Int, out_Jp: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _update_F_S_Jp_elastic(p(J).unsafe_load(i), ldm(p(F_tmp), i), ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), p(Jp).unsafe_load(i))
        stm(p(out_F), i, r[0])
        stm(p(out_S), i, r[1])
        p(out_Jp).unsafe_offset(i)[] = r[2]


@export("mgw_mpm_update_stress_corotation")
def mgw_mpm_update_stress_corotation(
    mu: Float64, lam: Float64, U: Int, S: Int, V: Int, F_tmp: Int, F_new: Int, J: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        stm(
            p(dst), i,
            _update_stress_corotation(mu, lam, ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), ldm(p(F_tmp), i), ldm(p(F_new), i), p(J).unsafe_load(i)),
        )


@export("mgw_mpm_update_stress_neohooken")
def mgw_mpm_update_stress_neohooken(
    mu: Float64, lam: Float64, U: Int, S: Int, V: Int, F_tmp: Int, F_new: Int, J: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        stm(
            p(dst), i,
            _update_stress_neohooken(mu, lam, ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), ldm(p(F_tmp), i), ldm(p(F_new), i), p(J).unsafe_load(i)),
        )


# ========================================= materials/MPM/elasto_plastic.py
@export("mgw_mpm_update_F_S_Jp_elasto_plastic")
def mgw_mpm_update_F_S_Jp_elasto_plastic(
    mu: Float64, use_von_mises: Int, von_mises_yield_stress: Float64, yield_lower: Float64, yield_higher: Float64,
    J: Int, F_tmp: Int, U: Int, S: Int, V: Int, Jp: Int, out_F: Int, out_S: Int, out_Jp: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _update_F_S_Jp_elasto_plastic(
            mu, use_von_mises != 0, von_mises_yield_stress, yield_lower, yield_higher, p(J).unsafe_load(i),
            ldm(p(F_tmp), i), ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), p(Jp).unsafe_load(i),
        )
        stm(p(out_F), i, r[0])
        stm(p(out_S), i, r[1])
        p(out_Jp).unsafe_offset(i)[] = r[2]


# ================================================= materials/MPM/liquid.py
@export("mgw_mpm_update_F_S_Jp_liquid")
def mgw_mpm_update_F_S_Jp_liquid(
    J: Int, F_tmp: Int, U: Int, S: Int, V: Int, Jp: Int, out_F: Int, out_S: Int, out_Jp: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _update_F_S_Jp_liquid(p(J).unsafe_load(i), ldm(p(F_tmp), i), ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), p(Jp).unsafe_load(i))
        stm(p(out_F), i, r[0])
        stm(p(out_S), i, r[1])
        p(out_Jp).unsafe_offset(i)[] = r[2]


@export("mgw_mpm_update_stress_liquid")
def mgw_mpm_update_stress_liquid(
    mu: Float64, lam: Float64, U: Int, S: Int, V: Int, F_tmp: Int, F_new: Int, J: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        stm(
            p(dst), i,
            _update_stress_liquid(mu, lam, ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), ldm(p(F_tmp), i), ldm(p(F_new), i), p(J).unsafe_load(i)),
        )


# ==================================================== materials/MPM/sand.py
@export("mgw_mpm_sand_alpha")
def mgw_mpm_sand_alpha(friction_angle: Float64, dst: Int) abi("C"):
    p(dst).unsafe_offset(0)[] = sand_alpha(friction_angle)


@export("mgw_mpm_sand_projection")
def mgw_mpm_sand_projection(
    lam: Float64, mu: Float64, alpha: Float64, S: Int, Jp: Int, out_S: Int, out_Jp: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _sand_projection(lam, mu, alpha, ldm(p(S), i), p(Jp).unsafe_load(i))
        stm(p(out_S), i, r[0])
        p(out_Jp).unsafe_offset(i)[] = r[1]


@export("mgw_mpm_update_F_S_Jp_sand")
def mgw_mpm_update_F_S_Jp_sand(
    lam: Float64, mu: Float64, alpha: Float64, J: Int, F_tmp: Int, U: Int, S: Int, V: Int, Jp: Int,
    out_F: Int, out_S: Int, out_Jp: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _update_F_S_Jp_sand(lam, mu, alpha, p(J).unsafe_load(i), ldm(p(F_tmp), i), ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), p(Jp).unsafe_load(i))
        stm(p(out_F), i, r[0])
        stm(p(out_S), i, r[1])
        p(out_Jp).unsafe_offset(i)[] = r[2]


@export("mgw_mpm_update_stress_sand")
def mgw_mpm_update_stress_sand(
    mu: Float64, lam: Float64, U: Int, S: Int, V: Int, F_tmp: Int, F_new: Int, J: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        stm(
            p(dst), i,
            _update_stress_sand(mu, lam, ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), ldm(p(F_tmp), i), ldm(p(F_new), i), p(J).unsafe_load(i)),
        )


# ==================================================== materials/MPM/snow.py
@export("mgw_mpm_update_F_S_Jp_snow")
def mgw_mpm_update_F_S_Jp_snow(
    yield_lower: Float64, yield_higher: Float64, J: Int, F_tmp: Int, U: Int, S: Int, V: Int, Jp: Int,
    out_F: Int, out_S: Int, out_Jp: Int, n: Int,
) abi("C"):
    for i in range(n):
        var r = _update_F_S_Jp_snow(yield_lower, yield_higher, p(J).unsafe_load(i), ldm(p(F_tmp), i), ldm(p(U), i), ldm(p(S), i), ldm(p(V), i), p(Jp).unsafe_load(i))
        stm(p(out_F), i, r[0])
        stm(p(out_S), i, r[1])
        p(out_Jp).unsafe_offset(i)[] = r[2]


@export("mgw_mpm_update_stress_snow")
def mgw_mpm_update_stress_snow(
    mu: Float64, lam: Float64, yield_lower: Float64, yield_higher: Float64, U: Int, S: Int, V: Int,
    F_tmp: Int, F_new: Int, J: Int, Jp: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        stm(
            p(dst), i,
            _update_stress_snow(
                mu, lam, yield_lower, yield_higher, ldm(p(U), i), ldm(p(S), i), ldm(p(V), i),
                ldm(p(F_tmp), i), ldm(p(F_new), i), p(J).unsafe_load(i), p(Jp).unsafe_load(i),
            ),
        )


# ==================================================== solvers/mpm_solver.py
@export("mgw_mpm_stencil_range")
def mgw_mpm_stencil_range() abi("C") -> Int:
    return stencil_range()


@export("mgw_mpm_compute_F_tmp")
def mgw_mpm_compute_F_tmp(C: Int, F: Int, substep_dt: Float64, dst: Int, n: Int) abi("C"):
    for i in range(n):
        stm(p(dst), i, compute_F_tmp(ldm(p(C), i), ldm(p(F), i), substep_dt))


@export("mgw_mpm_p2g")
def mgw_mpm_p2g(
    pos: Int, vel: Int, C: Int, F: Int, F_tmp: Int, U: Int, S: Int, V: Int, Jp: Int, active: Int,
    mass: Int, free: Int, mu: Float64, lam: Float64, yield_lower: Float64, yield_higher: Float64,
    use_von_mises: Int, von_mises_yield_stress: Float64, friction_angle: Float64, model: Int32,
    needs_svd: Int,
    substep_dt: Float64, dx: Float64, inv_dx: Float64, particle_volume: Float64, grid_offset: Int,
    grid_res: Int, grid_vel_in: Int, grid_mass: Int, out_F: Int, out_Jp: Int, n: Int,
) abi("C"):
    var off = ld3(p(grid_offset), 0)
    var res = ld3(p(grid_res), 0)
    for i in range(n):
        var r = p2g(
            ld3(p(pos), i), ld3(p(vel), i), ldm(p(C), i), ldm(p(F_tmp), i), ldm(p(U), i), ldm(p(S), i),
            ldm(p(V), i), p(Jp).unsafe_load(i), ldm(p(F), i), ip(active).unsafe_load(i) != 0, p(mass).unsafe_load(i), ip(free).unsafe_load(i) != 0,
            mu, lam, yield_lower, yield_higher, use_von_mises != 0, von_mises_yield_stress,
            friction_angle, model, needs_svd != 0, substep_dt, dx, inv_dx, particle_volume, off, res,
            p(grid_vel_in), p(grid_mass),
        )
        stm(p(out_F), i, r[0])
        p(out_Jp).unsafe_offset(i)[] = r[1]


@export("mgw_mpm_g2p")
def mgw_mpm_g2p(
    pos: Int, grid_vel_out: Int, lower: Int, upper: Int, substep_dt: Float64, inv_dx: Float64,
    grid_offset: Int, grid_res: Int, out_pos: Int, out_vel: Int, out_C: Int, n: Int,
) abi("C"):
    var off = ld3(p(grid_offset), 0)
    var res = ld3(p(grid_res), 0)
    var lo = ld3(p(lower), 0)
    var up = ld3(p(upper), 0)
    for i in range(n):
        var r = g2p(ld3(p(pos), i), p(grid_vel_out), lo, up, substep_dt, inv_dx, off, res)
        st3(p(out_pos), i, r[0])
        st3(p(out_vel), i, r[1])
        stm(p(out_C), i, r[2])


@export("mgw_mpm_backward_svd")
def mgw_mpm_backward_svd(
    grad_U: Int, grad_S: Int, grad_V: Int, U: Int, S: Int, V: Int, dst: Int, n: Int,
) abi("C"):
    for i in range(n):
        stm(
            p(dst), i,
            backward_svd(
                ldm(p(grad_U), i), ldm(p(grad_S), i), ldm(p(grad_V), i), ldm(p(U), i), ldm(p(S), i), ldm(p(V), i)
            ),
        )
