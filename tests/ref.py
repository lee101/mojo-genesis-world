"""The upstream side of every parity test, in one place.

Each helper here calls the vendored `genesis-world` source, unmodified, through
the harness in `upstream_ref.py`. A test that compares against `ref` is
comparing against upstream's own code, not against a paraphrase of it.
"""

from __future__ import annotations

import numpy as np

import upstream_ref as R

NS = R.load()


def rand3(rng, n) -> np.ndarray:
    return rng.normal(size=(n, 3))


def rand_unit_quat(rng, n) -> np.ndarray:
    q = rng.normal(size=(n, 4))
    return q / np.linalg.norm(q, axis=1, keepdims=True)


def geom_data(rng, n) -> np.ndarray:
    return np.abs(rng.normal(size=(n, 3))) + 0.1


def aabb_corners(rng, n_geoms) -> np.ndarray:
    """The eight rest-pose AABB corners per geom, corner 0 lower and corner 7 upper."""
    lo = rng.normal(size=(n_geoms, 1, 3))
    hi = lo + np.abs(rng.normal(size=(n_geoms, 1, 3))) + 0.2
    out = np.zeros((n_geoms, 8, 3))
    for k in range(8):
        out[:, k, 0] = hi[:, 0, 0] if k & 1 else lo[:, 0, 0]
        out[:, k, 1] = hi[:, 0, 1] if k & 2 else lo[:, 0, 1]
        out[:, k, 2] = hi[:, 0, 2] if k & 4 else lo[:, 0, 2]
    return out


# ----------------------------------------------------------------------- geom.py


def transform_by_quat(v, quat):
    return np.array([R.to_v3(NS["geom"].qd_transform_by_quat(R.V3(a), R.V4(q))) for a, q in zip(v, quat)])


def transform_by_quat_fast(v, quat):
    return np.array(
        [R.to_v3(NS["geom"].qd_transform_by_quat_fast(R.V3(a), R.V4(q))) for a, q in zip(v, quat)]
    )


def transform_by_trans_quat(pos, trans, quat):
    return np.array(
        [R.to_v3(NS["geom"].qd_transform_by_trans_quat(R.V3(p), R.V3(t), R.V4(q)))
         for p, t, q in zip(pos, trans, quat)]
    )


def transform_by_trans_quat_fast(pos, trans, quat):
    return np.array(
        [R.to_v3(NS["geom"].qd_transform_by_trans_quat_fast(R.V3(p), R.V3(t), R.V4(q)))
         for p, t, q in zip(pos, trans, quat)]
    )


def inv_transform_by_quat(v, quat):
    return np.array(
        [R.to_v3(NS["geom"].qd_inv_transform_by_quat(R.V3(a), R.V4(q))) for a, q in zip(v, quat)]
    )


def inv_transform_by_trans_quat(pos, trans, quat):
    return np.array(
        [R.to_v3(NS["geom"].qd_inv_transform_by_trans_quat(R.V3(p), R.V3(t), R.V4(q)))
         for p, t, q in zip(pos, trans, quat)]
    )


def inv_quat(quat):
    return np.array([R.to_v4(NS["geom"].qd_inv_quat(R.V4(q))) for q in quat])


def quat_mul(a, b):
    return np.array([R.to_v4(NS["geom"].qd_quat_mul(R.V4(x), R.V4(y))) for x, y in zip(a, b)])


def transform_quat_by_quat(v, u):
    return np.array(
        [R.to_v4(NS["geom"].qd_transform_quat_by_quat(R.V4(x), R.V4(y))) for x, y in zip(v, u)]
    )


def quat_to_R(quat, eps=1e-15):
    return np.array([R.to_m3(NS["geom"].qd_quat_to_R(R.V4(q), eps)) for q in quat])


def orthogonals(a):
    out = [NS["geom"].qd_orthogonals(R.V3(x)) for x in a]
    return np.array([R.to_v3(o[0]) for o in out]), np.array([R.to_v3(o[1]) for o in out])


def normalize(v, eps=1e-15):
    return np.array([R.to_v3(NS["geom"].qd_normalize(R.V3(x), eps)) for x in v])


def det3(v1, v2, v3):
    return np.array(
        [NS["utils"].func_det3(R.V3(a), R.V3(b), R.V3(c)) for a, b, c in zip(v1, v2, v3)]
    )


# -------------------------------------------------------------- support_field.py


def direction_grid(support_res: int) -> np.ndarray:
    """`SupportField._get_direction_grid`, run as upstream wrote it."""
    obj = object.__new__(NS["support_field"].SupportField)
    obj._support_res = support_res
    return np.asarray(obj._get_direction_grid())


def polar_band(support_res: int) -> tuple[int, int]:
    obj = object.__new__(NS["support_field"].SupportField)
    obj._support_res = support_res
    return tuple(obj._polar_band())


def activate(init_pos, vert_start, vert_end, support_res: int):
    """`SupportField.activate`'s table build, run as upstream wrote it."""

    class Solver:
        pass

    solver = Solver()
    solver.n_geoms = len(vert_start)
    solver.dyn_info = R.Fields(
        verts=R.Fields(init_pos=np.asarray(init_pos, dtype=np.float64)),
        geoms=R.Fields(vert_start=np.asarray(vert_start), vert_end=np.asarray(vert_end)),
    )
    obj = object.__new__(NS["support_field"].SupportField)
    obj._support_res = support_res
    obj.solver = solver
    v1, num_v, support_v, support_vid, cell_start, n_cells = (
        NS["activate_src"].build_support_table(obj)
    )
    return v1, num_v, support_v, support_vid, cell_start, n_cells


def _state(vert_start, vert_end, geom_type, geoms_data, init_pos, support_res, tables):
    ci = R.collider_info(
        support_res=support_res,
        support_v=tables[0],
        support_vid=tables[1],
        support_cell_start=tables[2],
    )
    di = R.dyn_info(geom_type, geoms_data, vert_start, vert_end, init_pos)
    return ci, di


def support_sphere(i_g, d, pos, quat, shrink, geoms_data, vert_start, vert_end, init_pos):
    ci, di = _state(vert_start, vert_end, np.zeros(len(vert_start), int), geoms_data, init_pos, 180, (None, None, None))
    r = NS["support_field"]._func_support_sphere(i_g, R.V3(d), R.V3(pos), R.V4(quat), shrink, di)
    return R.to_v3(r[0]), R.to_v3(r[1]), int(r[2])


def support_ellipsoid(i_g, d, pos, quat, geoms_data, vert_start, vert_end, init_pos):
    ci, di = _state(vert_start, vert_end, np.zeros(len(vert_start), int), geoms_data, init_pos, 180, (None, None, None))
    return R.to_v3(NS["support_field"]._func_support_ellipsoid(i_g, R.V3(d), R.V3(pos), R.V4(quat), di))


def support_capsule(i_g, d, pos, quat, shrink, geoms_data, vert_start, vert_end, init_pos):
    ci, di = _state(vert_start, vert_end, np.zeros(len(vert_start), int), geoms_data, init_pos, 180, (None, None, None))
    return R.to_v3(NS["support_field"]._func_support_capsule(i_g, R.V3(d), R.V3(pos), R.V4(quat), shrink, di))


def support_cylinder(i_g, d, pos, quat, shrink, geoms_data, vert_start, vert_end, init_pos):
    ci, di = _state(vert_start, vert_end, np.zeros(len(vert_start), int), geoms_data, init_pos, 180, (None, None, None))
    return R.to_v3(NS["support_field"]._func_support_cylinder(i_g, R.V3(d), R.V3(pos), R.V4(quat), shrink, di))


def support_box(i_g, d, pos, quat, geoms_data, vert_start, vert_end, init_pos):
    ci, di = _state(vert_start, vert_end, np.zeros(len(vert_start), int), geoms_data, init_pos, 180, (None, None, None))
    r = NS["support_field"]._func_support_box(i_g, R.V3(d), R.V3(pos), R.V4(quat), di)
    return R.to_v3(r[0]), R.to_v3(r[1]), int(r[2])


def support_prism(i_b, d, prism):
    cs = R.Fields(prism=R.Vec3Field(prism))
    r = NS["support_field"]._func_support_prism(i_b, R.V3(d), cs)
    return R.to_v3(r[0]), int(r[1])


def support_mesh_table(i_g, d_mesh, vert_start, vert_end, geom_type, geoms_data, init_pos, tables):
    ci, di = _state(vert_start, vert_end, geom_type, geoms_data, init_pos, tables[3], tables[:3])
    r = NS["support_field"]._func_support_mesh(i_g, R.V3(d_mesh), ci)
    return R.to_v3(r[0]), int(r[1])


def support_mesh_exhaustive(i_g, d_mesh, vert_start, vert_end, geom_type, geoms_data, init_pos):
    ci, di = _state(vert_start, vert_end, geom_type, geoms_data, init_pos, 180, (None, None, None))
    r = NS["support_field"]._func_support_mesh_exhaustive(i_g, R.V3(d_mesh), di)
    return R.to_v3(r[0]), int(r[1])


def support_world(i_g, d, pos, quat, vert_start, vert_end, geom_type, geoms_data, init_pos,
                  tables, exhaustive):
    ci, di = _state(vert_start, vert_end, geom_type, geoms_data, init_pos, tables[3], tables[:3])
    r = NS["support_field"]._func_support_world(i_g, R.V3(d), R.V3(pos), R.V4(quat), di, ci, exhaustive)
    return R.to_v3(r[0]), R.to_v3(r[1]), int(r[2])


def direction_grid_coords(d_mesh, support_res):
    r = NS["support_field"]._func_direction_grid_coords(R.V3(d_mesh), support_res)
    return int(r[0]), float(r[1]), float(r[2])


def count_supports_mesh(i_g, d_mesh, vert_start, vert_end, geom_type, geoms_data, init_pos, tables):
    ci, di = _state(vert_start, vert_end, geom_type, geoms_data, init_pos, tables[3], tables[:3])
    return int(NS["support_field"]._func_count_supports_mesh(i_g, R.V3(d_mesh), ci))


def count_supports_world(i_g, d, quat, vert_start, vert_end, geom_type, geoms_data, init_pos, tables):
    ci, di = _state(vert_start, vert_end, geom_type, geoms_data, init_pos, tables[3], tables[:3])
    return int(NS["support_field"]._func_count_supports_world(i_g, R.V3(d), R.V4(quat), ci))


def count_supports_box(d, quat):
    return int(NS["support_field"]._func_count_supports_box(R.V3(d), R.V4(quat)))


# ------------------------------------------------------------------ gjk_support.py


def support_mesh(i_g, direction, pos, quat, prev_vertex_id, init_pos, vert_start, vert_end):
    di = R.dyn_info(np.zeros(len(vert_start), int), np.zeros((len(vert_start), 3)),
                    vert_start, vert_end, init_pos)
    st = R.Fields(support_mesh_prev_vertex_id=np.asarray(prev_vertex_id).reshape(1, -1))
    ci = R.collider_info()
    r = NS["gjk_support"].support_mesh(i_g, 0, 0, R.V3(direction), R.V3(pos), R.V4(quat), st, di, ci)
    return R.to_v3(r[0]), int(r[1]), np.asarray(st.support_mesh_prev_vertex_id).reshape(-1)


def support_driver(i_g, i_b, direction, pos, quat, shrink_sphere, geom_type, geoms_data,
                   vert_start, vert_end, init_pos, tables, prev_vertex_id, prism, has_terrain,
                   i_o: int = 0):
    di = R.dyn_info(np.atleast_1d(geom_type), geoms_data, vert_start, vert_end, init_pos)
    st = R.Fields(support_mesh_prev_vertex_id=np.asarray(prev_vertex_id).reshape(1, -1))
    cs = R.Fields(prism=R.Vec3Field(prism))
    ci = R.collider_info(
        support_res=tables[3], support_v=R.Vec3Field(tables[0]), support_vid=tables[1],
        support_cell_start=tables[2],
    )
    cfg = R.Fields(enable_mujoco_compatibility=True)
    scfg = R.Fields(has_terrain=has_terrain)
    r = NS["gjk_support"].support_driver(
        i_g, i_b, i_o, R.V3(direction), R.V3(pos), R.V4(quat), shrink_sphere, cs, st, di, ci, cfg, scfg
    )
    return R.to_v3(r[0]), R.to_v3(r[1]), int(r[2])


def func_support(i_ga, i_gb, i_b, direction, pos_a, quat_a, pos_b, quat_b, shrink_sphere,
                 geom_type, geoms_data, vert_start, vert_end, init_pos, tables,
                 prev_vertex_id, prism, has_terrain):
    di = R.dyn_info(np.atleast_1d(geom_type), geoms_data, vert_start, vert_end, init_pos)
    st = R.Fields(support_mesh_prev_vertex_id=np.asarray(prev_vertex_id).reshape(1, -1))
    cs = R.Fields(prism=R.Vec3Field(prism))
    ci = R.collider_info(
        support_res=tables[3], support_v=R.Vec3Field(tables[0]), support_vid=tables[1],
        support_cell_start=tables[2],
    )
    cfg = R.Fields(enable_mujoco_compatibility=True)
    scfg = R.Fields(has_terrain=has_terrain)
    r = NS["gjk_support"].func_support(
        i_ga, i_gb, i_b, R.V3(direction), R.V3(pos_a), R.V4(quat_a), R.V3(pos_b), R.V4(quat_b),
        shrink_sphere, cs, st, di, ci, cfg, scfg,
    )
    return (R.to_v3(r[0]), R.to_v3(r[1]), R.to_v3(r[2]), R.to_v3(r[3]),
            int(r[4]), int(r[5]), R.to_v3(r[6]))


# -------------------------------------------------------------------- gjk_utils.py


def ray_triangle_intersection(ray_v1, ray_v2, tri_v1, tri_v2, tri_v3):
    return NS["gjk_utils"].func_ray_triangle_intersection(
        R.V3(ray_v1), R.V3(ray_v2), R.V3(tri_v1), R.V3(tri_v2), R.V3(tri_v3)
    )


def triangle_affine_coords(point, tri_v1, tri_v2, tri_v3):
    return R.to_v3(
        NS["gjk_utils"].func_triangle_affine_coords(R.V3(point), R.V3(tri_v1), R.V3(tri_v2), R.V3(tri_v3))
    )


def point_triangle_intersection(point, tri_v1, tri_v2, tri_v3):
    ci = R.collider_info()
    return bool(NS["gjk_utils"].func_point_triangle_intersection(
        R.V3(point), R.V3(tri_v1), R.V3(tri_v2), R.V3(tri_v3), ci
    ))


def point_plane_same_side(point, plane_v1, plane_v2, plane_v3):
    return int(NS["gjk_utils"].func_point_plane_same_side(
        R.V3(point), R.V3(plane_v1), R.V3(plane_v2), R.V3(plane_v3)
    ))


def origin_tetra_intersection(tet_v1, tet_v2, tet_v3, tet_v4):
    return int(NS["gjk_utils"].func_origin_tetra_intersection(
        R.V3(tet_v1), R.V3(tet_v2), R.V3(tet_v3), R.V3(tet_v4)
    ))


def project_origin_to_plane(v1, v2, v3):
    r = NS["gjk_utils"].func_project_origin_to_plane(R.V3(v1), R.V3(v2), R.V3(v3), R.collider_info())
    return R.to_v3(r[0]), int(r[1])


# --------------------------------------------------------------------- contact.py


def compute_geom_rbound(i_g, geom_type, geoms_data, aabb):
    di = R.dyn_info(geom_type, geoms_data, np.zeros(len(geom_type), int),
                    np.zeros(len(geom_type), int), np.zeros((1, 3)))
    return float(NS["contact"].func_compute_geom_rbound(i_g, R.Vec3Field(aabb), di))


def compute_geom_pair_scale(i_ga, i_gb, geom_type, aabb):
    di = R.dyn_info(geom_type, np.zeros((len(geom_type), 3)), np.zeros(len(geom_type), int),
                    np.zeros(len(geom_type), int), np.zeros((1, 3)))
    return float(NS["contact"].func_compute_geom_pair_scale(i_ga, i_gb, R.Vec3Field(aabb), di))


def compute_geom_pair_scale_mj(i_ga, i_gb, geom_type, geoms_data, aabb):
    di = R.dyn_info(geom_type, geoms_data, np.zeros(len(geom_type), int),
                    np.zeros(len(geom_type), int), np.zeros((1, 3)))
    return float(NS["contact"].func_compute_geom_pair_scale_mj(
        i_ga, i_gb, R.Vec3Field(aabb), di))


def compute_mc_tolerance(i_ga, i_gb, geom_type, geoms_data, aabb, mc_tolerance, mj):
    di = R.dyn_info(geom_type, geoms_data, np.zeros(len(geom_type), int),
                    np.zeros(len(geom_type), int), np.zeros((1, 3)))
    cfg = R.Fields(enable_mujoco_compatibility=mj)
    ci = R.collider_info()
    ci.mc_tolerance = R.Scalar(mc_tolerance)
    return float(NS["contact"].func_compute_mc_tolerance(
        i_ga, i_gb, R.Vec3Field(aabb), di, ci, cfg))


def contact_orthogonals(normal):
    out = [NS["contact"].func_contact_orthogonals(
        0, 0, 0, R.V3(x), R.Vec3Field(np.zeros((1, 8, 3))), R.Fields(), R.Fields(),
        R.Fields(EPS=R.Scalar(R.EPS)),
        R.Fields(enable_mujoco_compatibility=True)) for x in normal]
    return np.array([R.to_v3(o[0]) for o in out]), np.array([R.to_v3(o[1]) for o in out])


def rotate_frame(pos, quat, contact_pos, qrot):
    out = [NS["contact"].func_rotate_frame(R.V3(a), R.V4(b), R.V3(c), R.V4(d))
           for a, b, c, d in zip(pos, quat, contact_pos, qrot)]
    return np.array([R.to_v3(o[0]) for o in out]), np.array([R.to_v4(o[1]) for o in out])


def contact_order_key(pos):
    return float(NS["contact"].func_contact_order_key(R.V3(pos)))


# --------------------------------------------------------------- materials/MPM/*


def mpm_elastic_update_F_S_Jp(mat, J, F_tmp, U, S, V, Jp):
    r = NS["mpm_elastic"]._update_F_S_Jp_elastic(mat, J, R.M3(F_tmp), R.M3(U), R.M3(S), R.M3(V), Jp)
    return R.to_m3(r[0]), R.to_m3(r[1]), float(r[2])


_STRESS_MODULE = {
    "corotation": "mpm_elastic",
    "neohooken": "mpm_elastic",
    "liquid": "mpm_liquid",
    "sand": "mpm_sand",
    "snow": "mpm_snow",
    "default": "mpm_base",
}


def mpm_stress(mat, which, U, S, V, F_tmp, F_new, J, Jp=None):
    fn = getattr(NS[_STRESS_MODULE[which]], "_update_stress_" + which)
    out = []
    for a, b, c, d, e, f, g, h in zip(U, S, V, F_tmp, F_new, J, Jp, Jp):
        r = fn(mat, R.M3(a), R.M3(b), R.M3(c), R.M3(d), R.M3(e), f, h, None, None)
        out.append(R.to_m3(r))
    return np.array(out)


def mpm_update_F_S_Jp(which, mat, J, F_tmp, U, S, V, Jp):
    mod = {"elastic": "mpm_elastic", "elasto_plastic": "mpm_elasto_plastic", "liquid": "mpm_liquid",
           "sand": "mpm_sand", "snow": "mpm_snow"}[which]
    fn = getattr(NS[mod], {
        "elastic": "_update_F_S_Jp_elastic",
        "elasto_plastic": "_update_F_S_Jp_elasto_plastic",
        "liquid": "_update_F_S_Jp_liquid",
        "sand": "_update_F_S_Jp_sand",
        "snow": "_update_F_S_Jp_snow",
    }[which])
    out = []
    for j, ft, u, sv, vv, jp in zip(J, F_tmp, U, S, V, Jp):
        r = fn(mat, j, R.M3(ft), R.M3(u), R.M3(sv), R.M3(vv), jp)
        out.append((R.to_m3(r[0]), R.to_m3(r[1]), float(np.ravel(r[2])[0])))
    return (np.array([x[0] for x in out]), np.array([x[1] for x in out]),
            np.array([x[2] for x in out]))


def sand_projection(mat, S, Jp):
    r = NS["mpm_sand"]._sand_projection(mat, R.M3(S), Jp)
    return R.to_m3(r[0]), float(r[1])


# ------------------------------------------------------------- mpm_solver.py tail


def signmax(a, eps):
    return float(NS["tail"].signmax(a, eps))


def backward_svd(grad_U, grad_S, grad_V, U, S, V):
    return R.to_m3(
        NS["tail"].backward_svd(
            R.M3(grad_U), R.M3(grad_S), R.M3(grad_V), R.M3(U), R.M3(S), R.M3(V)
        )
    )


# --------------------------------------------------- mpm_solver.py transfer loops
#
# Upstream's `p2g` / `g2p` are `@qd.kernel` methods on a `data_oriented` class:
# they cannot be called one particle at a time outside an engine, because their
# state is the class's own fields. The constitutive part is upstream's (above);
# the 27-cell transfer around it is transcribed here line for line from
# `mpm_solver.py` p2g lines 413-475 and g2p lines 489-525. Both the tests and
# the benchmark compare against these.

MODEL_KEY = {0: "elastic", 1: "elastic", 2: "elasto_plastic", 3: "liquid",
             4: "sand", 5: "snow"}
MODEL_STRESS = {0: "corotation", 1: "neohooken", 2: "default", 3: "liquid",
                4: "sand", 5: "snow"}


def p2g(material, model, particles, grid, params):
    dx = params["dx"]
    inv_dx = 1.0 / dx
    res = np.asarray(grid["res"], dtype=int)
    off = np.asarray(grid["offset"], dtype=int)
    vel_in = np.zeros((res[0], res[1], res[2], 3))
    grid_mass = np.zeros((res[0], res[1], res[2]))
    out_F = np.zeros_like(particles["F"])
    out_Jp = np.zeros(len(particles["F"]))
    for i in range(len(particles["F"])):
        F_tmp = particles["F_tmp"][i]
        U, S, V = particles["U"][i], particles["S"][i], particles["V"][i]
        if params["needs_svd"]:
            J = np.linalg.det(S)
        else:
            J = np.linalg.det(F_tmp)
        F_new, S_new, Jp_new = mpm_update_F_S_Jp(
            MODEL_KEY[model], material, np.array([J]), F_tmp[None], U[None], S[None], V[None],
            particles["Jp"][i:i + 1],
        )
        F_new, S_new, Jp_new = F_new[0], S_new[0], float(Jp_new[0])
        stress = mpm_stress(material, MODEL_STRESS[model], U[None], S[None], V[None],
                            F_tmp[None], F_new[None], np.array([J]),
                            particles["Jp"][i:i + 1])[0]
        stress = (-params["substep_dt"] * params["particle_volume"] * 4.0 * inv_dx * inv_dx) * stress
        affine = stress + particles["mass"][i] * particles["C"][i]
        pos = particles["pos"][i]
        base = np.floor(pos * inv_dx - 0.5).astype(int)
        fx = pos * inv_dx - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    o = np.array([a, b, c])
                    dpos = (o - fx) * dx
                    weight = w[a][0] * w[b][1] * w[c][2]
                    idx = tuple(base + o - off)
                    if particles["active"][i]:
                        vel_in[idx] += weight * (particles["mass"][i] * particles["vel"][i]
                                                 + affine @ dpos)
                        grid_mass[idx] += weight * particles["mass"][i]
                    if not particles["free"][i]:
                        vel_in[idx] = 0.0
        out_F[i] = F_new
        out_Jp[i] = Jp_new
    return out_F, out_Jp, vel_in, grid_mass


def g2p(pos, grid_vel_out, lower, upper, substep_dt, dx, grid_offset, grid_res):
    res = np.asarray(grid_res, dtype=int)
    off = np.asarray(grid_offset, dtype=int)
    inv_dx = 1.0 / dx
    out_pos = np.zeros_like(np.asarray(pos, dtype=np.float64))
    out_vel = np.zeros_like(out_pos)
    out_C = np.zeros((len(out_pos), 3, 3))
    for i in range(len(out_pos)):
        base = np.floor(pos[i] * inv_dx - 0.5).astype(int)
        fx = pos[i] * inv_dx - base
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        nv = np.zeros(3)
        nC = np.zeros((3, 3))
        for a in range(3):
            for b in range(3):
                for c in range(3):
                    o = np.array([a, b, c])
                    gv = grid_vel_out[tuple(base + o - off)]
                    weight = w[a][0] * w[b][1] * w[c][2]
                    nv += weight * gv
                    nC += 4.0 * inv_dx * weight * np.outer(gv, o - fx)
        p = pos[i] + substep_dt * nv
        v = nv
        for d in range(3):
            if p[d] < lower[d]:
                p[d] = lower[d]
                v[d] = 0.0
            elif p[d] > upper[d]:
                p[d] = upper[d]
                v[d] = 0.0
        out_pos[i] = p
        out_vel[i] = v
        out_C[i] = nC
    return out_pos, out_vel, out_C
