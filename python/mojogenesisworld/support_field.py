"""`genesis.engine.solvers.rigid.collider.support_field`, in Mojo.

`SupportField` keeps the upstream shape - it holds a `_support_res` resolution and
an `activate()` that precomputes the argmax support table over two spherical
charts - and adds `count_supports` / `support` methods that call the same
`_func_support_*` kernels the engine's GJK/EPA driver calls.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, align, i32, lib, worker_budget
from .geom import _bc, _out3, _v3, _v4

GEOM_TYPE_PLANE = 0
GEOM_TYPE_SPHERE = 1
GEOM_TYPE_ELLIPSOID = 2
GEOM_TYPE_CYLINDER = 3
GEOM_TYPE_CAPSULE = 4
GEOM_TYPE_BOX = 5
GEOM_TYPE_MESH = 6
GEOM_TYPE_TERRAIN = 7


def _polar_band(support_res: int) -> tuple[int, int]:
    packed = lib().mgw_polar_band(support_res)
    return packed % 1000000, packed // 1000000


def get_direction_grid(support_res: int) -> np.ndarray:
    """Unit direction of every cell of the two spherical charts, as `(n, 3)`."""
    num_v = lib().mgw_direction_grid_nums(support_res)
    dst = np.zeros((num_v, 3), dtype=np.float64)
    n_band = lib().mgw_get_direction_grid(addr(dst), support_res)
    return dst[:n_band]


def direction_grid_nums(support_res: int) -> int:
    return lib().mgw_direction_grid_nums(support_res)


def _func_direction_grid_coords(d_mesh, support_res: int):
    d_mesh = _v3(d_mesh)
    n = d_mesh.shape[0]
    chart = np.zeros(n, dtype=np.int32)
    ii = np.zeros(n, dtype=np.float64)
    jj = np.zeros(n, dtype=np.float64)
    lib().mgw_direction_grid_coords(
        addr(d_mesh), support_res, addr(chart), addr(ii), addr(jj), n
    )
    return chart, ii, jj


def func_det3(v1, v2, v3) -> np.ndarray:
    a, b, c = align(_v3(v1).shape[0], v1=_v3(v1), v2=_v3(v2), v3=_v3(v3))
    dst = np.zeros(a.shape[0], dtype=np.float64)
    lib().mgw_det3(addr(a), addr(b), addr(c), addr(dst), a.shape[0])
    return dst


def _func_support_mesh(i_g, d_mesh, field: "SupportField") -> tuple[np.ndarray, np.ndarray]:
    d_mesh, = align(_v3(d_mesh).shape[0], d_mesh=_v3(d_mesh))
    n = d_mesh.shape[0]
    out_v = _out3(n)
    out_vid = np.zeros(n, dtype=np.int32)
    lib().mgw_support_mesh_table(
        addr(i32(i_g)), addr(d_mesh), field.support_res, addr(field.support_cell_start),
        addr(field.support_vid), addr(field.support_v), addr(out_v), addr(out_vid), n,
    )
    return out_v, out_vid


def _func_support_mesh_exhaustive(
    i_g, d_mesh, vert_start, vert_end, init_pos
) -> tuple[np.ndarray, np.ndarray]:
    d_mesh = _v3(d_mesh)
    n = d_mesh.shape[0]
    out_v = _out3(n)
    out_vid = np.zeros(n, dtype=np.int32)
    init_pos = _v3(init_pos)
    lib().mgw_support_mesh_exhaustive(
        addr(i32(i_g)), addr(d_mesh), addr(i32(vert_start)), addr(i32(vert_end)),
        addr(init_pos), addr(out_v), addr(out_vid), n,
    )
    return out_v, out_vid


def _func_support_sphere(i_g, d, pos, quat, shrink, geoms_data):
    d, pos, quat = _v3(d), _v3(pos), _v4(quat)
    d, pos, quat = align(d.shape[0], d=d, pos=pos, quat=quat)
    n = d.shape[0]
    geoms_data = _v3(geoms_data)
    out_v, out_local, out_vid = _out3(n), _out3(n), np.zeros(n, dtype=np.int32)
    lib().mgw_support_sphere(
        addr(i32(i_g)), addr(d), addr(pos), addr(quat), 1 if shrink else 0,
        addr(geoms_data), addr(out_v), addr(out_local), addr(out_vid), n,
    )
    return out_v, out_local, out_vid


def _func_support_ellipsoid(i_g, d, pos, quat, geoms_data):
    d, pos, quat = _v3(d), _v3(pos), _v4(quat)
    d, pos, quat = align(d.shape[0], d=d, pos=pos, quat=quat)
    n = d.shape[0]
    geoms_data = _v3(geoms_data)
    out_v = _out3(n)
    lib().mgw_support_ellipsoid(
        addr(i32(i_g)), addr(d), addr(pos), addr(quat), addr(geoms_data), addr(out_v), n
    )
    return out_v


def _func_support_capsule(i_g, d, pos, quat, shrink, geoms_data):
    d, pos, quat = _v3(d), _v3(pos), _v4(quat)
    d, pos, quat = align(d.shape[0], d=d, pos=pos, quat=quat)
    n = d.shape[0]
    geoms_data = _v3(geoms_data)
    out_v = _out3(n)
    lib().mgw_support_capsule(
        addr(i32(i_g)), addr(d), addr(pos), addr(quat), 1 if shrink else 0,
        addr(geoms_data), addr(out_v), n,
    )
    return out_v


def _func_support_cylinder(i_g, d, pos, quat, shrink, geoms_data):
    d, pos, quat = _v3(d), _v3(pos), _v4(quat)
    d, pos, quat = align(d.shape[0], d=d, pos=pos, quat=quat)
    n = d.shape[0]
    geoms_data = _v3(geoms_data)
    out_v = _out3(n)
    lib().mgw_support_cylinder(
        addr(i32(i_g)), addr(d), addr(pos), addr(quat), 1 if shrink else 0,
        addr(geoms_data), addr(out_v), n,
    )
    return out_v


def _func_support_prism(i_b, d, prism):
    d = _v3(d)
    n = d.shape[0]
    prism = _v3(prism)
    out_v = _out3(n)
    out_vid = np.zeros(n, dtype=np.int32)
    lib().mgw_support_prism(addr(i32(i_b)), addr(d), addr(prism), addr(out_v), addr(out_vid), n)
    return out_v, out_vid


def _func_support_box(i_g, d, pos, quat, geoms_data, vert_start):
    d, pos, quat = _v3(d), _v3(pos), _v4(quat)
    d, pos, quat = align(d.shape[0], d=d, pos=pos, quat=quat)
    n = d.shape[0]
    geoms_data = _v3(geoms_data)
    out_v, out_local, out_vid = _out3(n), _out3(n), np.zeros(n, dtype=np.int32)
    lib().mgw_support_box(
        addr(i32(i_g)), addr(d), addr(pos), addr(quat), addr(geoms_data),
        addr(i32(vert_start)), addr(out_v), addr(out_local), addr(out_vid), n,
    )
    return out_v, out_local, out_vid


def _func_support_world(
    i_g, d, pos, quat, vert_start, vert_end, init_pos, field: "SupportField", exhaustive: bool
):
    d, pos, quat = _v3(d), _v3(pos), _v4(quat)
    d, pos, quat = align(d.shape[0], d=d, pos=pos, quat=quat)
    n = d.shape[0]
    init_pos = _v3(init_pos)
    out_v, out_local, out_vid = _out3(n), _out3(n), np.zeros(n, dtype=np.int32)
    lib().mgw_support_world(
        addr(i32(i_g)), addr(d), addr(pos), addr(quat), addr(i32(vert_start)), addr(i32(vert_end)),
        addr(init_pos), field.support_res, addr(field.support_cell_start),
        addr(field.support_vid), addr(field.support_v), 1 if exhaustive else 0,
        addr(out_v), addr(out_local), addr(out_vid), n,
    )
    return out_v, out_local, out_vid


def _func_count_supports_mesh(i_g, d_mesh, field: "SupportField") -> np.ndarray:
    d_mesh = _v3(d_mesh)
    n = d_mesh.shape[0]
    out = np.zeros(n, dtype=np.int32)
    lib().mgw_count_supports_mesh(
        addr(i32(i_g)), addr(d_mesh), field.support_res, addr(field.support_cell_start),
        addr(field.support_v), addr(out), n,
    )
    return out


def _func_count_supports_world(i_g, d, quat, field: "SupportField") -> np.ndarray:
    d, quat = _v3(d), _v4(quat)
    d, quat = align(d.shape[0], d=d, quat=quat)
    n = d.shape[0]
    out = np.zeros(n, dtype=np.int32)
    lib().mgw_count_supports_world(
        addr(i32(i_g)), addr(d), addr(quat), field.support_res, addr(field.support_cell_start),
        addr(field.support_v), addr(out), n,
    )
    return out


def _func_count_supports_box(d, quat) -> np.ndarray:
    d, quat = _v3(d), _v4(quat)
    d, quat = align(d.shape[0], d=d, quat=quat)
    n = d.shape[0]
    out = np.zeros(n, dtype=np.int32)
    lib().mgw_count_supports_box(addr(d), addr(quat), addr(out), n)
    return out


class SupportField:
    """The precomputed support table of `SupportField`, built by `activate`."""

    _support_res = 180

    def __init__(self, support_res: int | None = None):
        if support_res is not None:
            self._support_res = int(support_res)
        self._is_active = False
        self.support_res = self._support_res
        self.n_geoms = 0
        self.n_support_cells = 0
        self.support_v = np.zeros((0, 3), dtype=np.float64)
        self.support_vid = np.zeros(0, dtype=np.int32)
        self.support_cell_start = np.zeros(0, dtype=np.int32)
        self.v1 = get_direction_grid(self.support_res)

    def _get_direction_grid(self) -> np.ndarray:
        return get_direction_grid(self.support_res)

    def _polar_band(self) -> tuple[int, int]:
        return _polar_band(self.support_res)

    @property
    def is_active(self) -> bool:
        return self._is_active

    def activate(self, init_pos, geoms_vert_start, geoms_vert_end, workers: int | None = None) -> None:
        """Precompute the support table for `n_geoms` geoms over `init_pos`.

        `workers` is how many threads the table may be split over; the default is
        the host's budget, and 1 forces the serial loop. The two must agree: the
        table is the same either way.
        """
        init_pos = _v3(init_pos)
        vert_start = i32(geoms_vert_start)
        vert_end = i32(geoms_vert_end)
        n_geoms = vert_start.shape[0]
        vert_start, vert_end = align(n_geoms, vert_start=vert_start, vert_end=vert_end)
        num_v = self.v1.shape[0]
        if workers is None:
            workers = worker_budget()

        self.n_geoms = n_geoms
        self.n_support_cells = n_geoms * num_v
        self.support_v = np.zeros((self.n_support_cells, 3), dtype=np.float64)
        self.support_vid = np.zeros(self.n_support_cells, dtype=np.int32)
        self.support_cell_start = np.zeros(n_geoms, dtype=np.int32)

        if n_geoms == 0:
            self._is_active = True
            return

        lib().mgw_activate(
            addr(init_pos), addr(vert_start), addr(vert_end), n_geoms, addr(self.v1), num_v,
            addr(self.support_v), addr(self.support_vid), addr(self.support_cell_start),
            int(workers),
        )
        self._is_active = True

    def count_supports(self, i_g, d_mesh) -> np.ndarray:
        return _func_count_supports_mesh(i32(np.full(len(np.atleast_1d(d_mesh)), i_g)), d_mesh, self)

    def support(self, i_g, d, pos, quat, vert_start, vert_end, init_pos, exhaustive: bool):
        i_g = i32(np.full(np.atleast_1d(d).shape[0], i_g))
        return _func_support_world(
            i_g, d, pos, quat, i32(np.full(i_g.shape[0], vert_start)),
            i32(np.full(i_g.shape[0], vert_end)), init_pos, self, exhaustive,
        )
