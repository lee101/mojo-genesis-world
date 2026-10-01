"""`genesis.engine.materials.MPM` and `genesis.engine.solvers.mpm_solver`, in Mojo.

The constitutive models keep upstream's class names, field names and defaults, and
their `update_F_S_Jp` / `update_stress` methods are the batched Mojo kernels.
`p2g` and `g2p` are the transfer step of `MPMSolver` for one particle each; the
CPIC coupler and the sparse-reset dirty-cell bookkeeping are not ported, and the
README says so.

The material classes below mirror upstream's dataclasses: `Elastic`,
`ElastoPlastic`, `Liquid`, `Snow`, `Sand`, all deriving from `Base`, with the
same `E` / `nu` / `rho` / `lam` / `mu` fields and the same `model_post_init`
resolution of the Lame parameters.
"""

from __future__ import annotations

import numpy as np

from ._lib import addr, align, lib
from .geom import _out3, _v3, _v4

MODEL_ELASTIC = 0
MODEL_ELASTIC_NEOHOOKEN = 1
MODEL_ELASTO_PLASTIC = 2
MODEL_LIQUID = 3
MODEL_SAND = 4
MODEL_SNOW = 5

_MODEL_NAMES = {
    MODEL_ELASTIC: "elastic",
    MODEL_ELASTIC_NEOHOOKEN: "elastic-neohooken",
    MODEL_ELASTO_PLASTIC: "elasto-plastic",
    MODEL_LIQUID: "liquid",
    MODEL_SAND: "sand",
    MODEL_SNOW: "snow",
}


def _m(n: int) -> np.ndarray:
    """An `(n, 3, 3)` batch, which is how the `(n, 9)` row-major buffer reads."""
    return np.zeros((n, 3, 3), dtype=np.float64)


def _m_in(a, n: int, name: str = "matrix") -> np.ndarray:
    arr = np.ascontiguousarray(a, dtype=np.float64).reshape(-1, 3, 3)
    if arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} rows, expected {n}")
    return arr


def _f(n: int) -> np.ndarray:
    return np.zeros(n, dtype=np.float64)


def resolve_lame(E: float, nu: float) -> tuple[float, float]:
    """`Base.model_post_init`: the first and second Lame parameters from E and nu."""
    out = np.zeros(2, dtype=np.float64)
    lib().mgw_resolve_lame(E, nu, out.ctypes.data, out.ctypes.data + 8)
    return float(out[0]), float(out[1])


def _fs(a, n: int, name: str = "vector") -> np.ndarray:
    """A flat float64 input, checked against `n`; a negative `n` skips the check."""
    arr = np.ascontiguousarray(a, dtype=np.float64).reshape(-1)
    if n >= 0 and arr.shape[0] != n:
        raise ValueError(f"{name} has {arr.shape[0]} values, expected {n}")
    return arr


def _update_F_S_Jp(fn, J, F_tmp, U, S, V, Jp):
    J = _fs(J, -1, "J")
    n = J.shape[0]
    Jp = _fs(Jp, n, "Jp")
    F_tmp, U, S, V = (_m_in(a, n, name) for a, name in
                      ((F_tmp, "F_tmp"), (U, "U"), (S, "S"), (V, "V")))
    out_F, out_S, out_Jp = _m(n), _m(n), _f(n)
    F9, U9, S9, V9 = (a.reshape(-1, 9) for a in (F_tmp, U, S, V))
    oF9, oS9 = out_F.reshape(-1, 9), out_S.reshape(-1, 9)
    fn(addr(J), addr(F9), addr(U9), addr(S9), addr(V9), addr(Jp),
       addr(oF9), addr(oS9), addr(out_Jp), n)
    return out_F, out_S, out_Jp


def _update_stress(fn, prefix, U, S, V, F_tmp, F_new, J, Jp=None):
    J = _fs(J, -1, "J")
    n = J.shape[0]
    mats = [_m_in(a, n, name) for a, name in
            ((U, "U"), (S, "S"), (V, "V"), (F_tmp, "F_tmp"), (F_new, "F_new"))]
    out = _m(n)
    out9 = out.reshape(-1, 9)
    args = [addr(a.reshape(-1, 9)) for a in mats] + [addr(J)]
    if Jp is None:
        fn(*prefix, *args, addr(out9), n)
    else:
        fn(*prefix, *args, addr(_fs(Jp, n, "Jp")), addr(out9), n)
    return out


class Base:
    """The base MPM material. Lame parameters are resolved from E and nu."""

    E: float = 1e6
    nu: float = 0.2
    rho: float = 1000.0
    lam: float | None = None
    mu: float | None = None
    needs_svd: bool = True
    model = MODEL_ELASTIC
    model_id = MODEL_ELASTIC

    def __init__(self, E=None, nu=None, rho=None, lam=None, mu=None, **kw):
        for k, v in (("E", E), ("nu", nu), ("rho", rho), ("lam", lam), ("mu", mu)):
            if v is not None:
                setattr(self, k, v)
        for k, v in kw.items():
            if not hasattr(type(self), k):
                raise TypeError(f"unknown material field {k!r}")
            setattr(self, k, v)
        self.model_post_init()

    def model_post_init(self) -> None:
        if self.mu is None:
            self.mu = self.E / (2.0 * (1.0 + self.nu))
        if self.lam is None:
            self.lam = self.E * self.nu / ((1.0 + self.nu) * (1.0 - 2.0 * self.nu))

    @property
    def name(self) -> str:
        return _MODEL_NAMES[self.model]

    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        return _update_F_S_Jp(
            lib().mgw_mpm_update_F_S_Jp_noop, J, F_tmp, U, S, V, Jp
        )

    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu=None, m_dir=None):
        return _update_stress(
            lib().mgw_mpm_update_stress_default, (self.mu, self.lam), U, S, V, F_tmp, F_new, J
        )


class Elastic(Base):
    """Elastic material for MPM. `model` is 'corotation' or 'neohooken'."""

    E: float = 3e5
    model: str = "corotation"

    def __init__(self, E=None, nu=None, rho=None, lam=None, mu=None, model=None):
        self.model = model or Elastic.model
        super().__init__(E, nu, rho, lam, mu)

    def model_post_init(self) -> None:
        super().model_post_init()
        if self.model == "corotation":
            self.model_id = MODEL_ELASTIC
            # corotation stress uses U @ V.T from SVD(F_tmp).
            self.needs_svd = True
        elif self.model == "neohooken":
            self.model_id = MODEL_ELASTIC_NEOHOOKEN
            # neohooken stress only reads F_tmp and J=det(F_tmp); SVD can be skipped.
            self.needs_svd = False
        else:
            raise ValueError(f"unknown elastic model {self.model!r}")

    @property
    def name(self) -> str:
        return _MODEL_NAMES[self.model_id]

    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        return _update_F_S_Jp(
            lib().mgw_mpm_update_F_S_Jp_elastic, J, F_tmp, U, S, V, Jp
        )

    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu=None, m_dir=None):
        if self.model == "corotation":
            return _update_stress(
                lib().mgw_mpm_update_stress_corotation, (self.mu, self.lam), U, S, V, F_tmp, F_new, J
            )
        return _update_stress(
            lib().mgw_mpm_update_stress_neohooken, (self.mu, self.lam), U, S, V, F_tmp, F_new, J
        )


class ElastoPlastic(Base):
    """Elasto-plastic material for MPM."""

    yield_lower: float = 2.5e-2
    yield_higher: float = 4.5e-3
    use_von_mises: bool = True
    von_mises_yield_stress: float = 10000.0
    model = MODEL_ELASTO_PLASTIC
    model_id = MODEL_ELASTO_PLASTIC

    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        J = _fs(J, -1, "J")
        n = J.shape[0]
        Jp = _fs(Jp, n, "Jp")
        F_tmp, U, S, V = (_m_in(a, n, name) for a, name in
                          ((F_tmp, "F_tmp"), (U, "U"), (S, "S"), (V, "V")))
        out_F, out_S, out_Jp = _m(n), _m(n), _f(n)
        F9, U9, S9, V9 = (a.reshape(-1, 9) for a in (F_tmp, U, S, V))
        oF9, oS9 = out_F.reshape(-1, 9), out_S.reshape(-1, 9)
        lib().mgw_mpm_update_F_S_Jp_elasto_plastic(
            self.mu, 1 if self.use_von_mises else 0, self.von_mises_yield_stress,
            self.yield_lower, self.yield_higher, addr(J), addr(F9),
            addr(U9), addr(S9), addr(V9), addr(Jp),
            addr(oF9), addr(oS9), addr(out_Jp), n,
        )
        return out_F, out_S, out_Jp

    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu=None, m_dir=None):
        return _update_stress(
            lib().mgw_mpm_update_stress_default, (self.mu, self.lam), U, S, V, F_tmp, F_new, J
        )


class Liquid(Base):
    """Liquid material for MPM. Non-viscous liquid sets mu to zero."""

    viscous: bool = False
    E: float = 1e6
    model = MODEL_LIQUID
    model_id = MODEL_LIQUID

    def model_post_init(self) -> None:
        super().model_post_init()
        if not self.viscous:
            self.mu = 0.0
        # Viscous liquid uses U @ V.T in the 2*mu*(F_tmp - U@V.T)@F_tmp.T term; when mu==0
        # that term vanishes and SVD is pure waste.
        self.needs_svd = self.viscous

    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        return _update_F_S_Jp(
            lib().mgw_mpm_update_F_S_Jp_liquid, J, F_tmp, U, S, V, Jp
        )

    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu=None, m_dir=None):
        return _update_stress(
            lib().mgw_mpm_update_stress_liquid, (self.mu, self.lam), U, S, V, F_tmp, F_new, J
        )


class Sand(Base):
    """Sand material for MPM, with internal pressure-dependent plasticity."""

    friction_angle: float = 45.0
    alpha: float = 0.0
    model = MODEL_SAND
    model_id = MODEL_SAND

    def __init__(self, E=None, nu=None, rho=None, lam=None, mu=None, friction_angle=None):
        kw = {} if friction_angle is None else {"friction_angle": friction_angle}
        super().__init__(E, nu, rho, lam, mu, **kw)

    def model_post_init(self) -> None:
        super().model_post_init()
        self._default_Jp = 0.0
        out = np.zeros(1, dtype=np.float64)
        lib().mgw_mpm_sand_alpha(self.friction_angle, addr(out))
        self.alpha = float(out[0])

    def _sand_projection(self, S, Jp):
        Jp = _fs(Jp, -1, "Jp")
        n = Jp.shape[0]
        S9 = _m_in(S, n, "S").reshape(-1, 9)
        out_S, out_Jp = _m(n), _f(n)
        out_S9 = out_S.reshape(-1, 9)
        lib().mgw_mpm_sand_projection(
            self.lam, self.mu, self.alpha, addr(S9), addr(Jp),
            addr(out_S9), addr(out_Jp), n,
        )
        return out_S, out_Jp

    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        J = _fs(J, -1, "J")
        n = J.shape[0]
        Jp = _fs(Jp, n, "Jp")
        F_tmp, U, S, V = (_m_in(a, n, name) for a, name in
                          ((F_tmp, "F_tmp"), (U, "U"), (S, "S"), (V, "V")))
        out_F, out_S, out_Jp = _m(n), _m(n), _f(n)
        F9, U9, S9, V9 = (a.reshape(-1, 9) for a in (F_tmp, U, S, V))
        oF9, oS9 = out_F.reshape(-1, 9), out_S.reshape(-1, 9)
        lib().mgw_mpm_update_F_S_Jp_sand(
            self.lam, self.mu, self.alpha, addr(J), addr(F9),
            addr(U9), addr(S9), addr(V9), addr(Jp),
            addr(oF9), addr(oS9), addr(out_Jp), n,
        )
        return out_F, out_S, out_Jp

    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu=None, m_dir=None):
        return _update_stress(
            lib().mgw_mpm_update_stress_sand, (self.mu, self.lam), U, S, V, F_tmp, F_new, J
        )


class Snow(ElastoPlastic):
    """Snow is elasto-plastic that hardens when compressed; no von Mises yield."""

    use_von_mises: bool = False
    model = MODEL_SNOW
    model_id = MODEL_SNOW

    def __init__(self, *args, **kw):
        if kw.get("use_von_mises", False):
            raise ValueError("Snow does not support use_von_mises=True.")
        super().__init__(*args, **kw)

    def update_F_S_Jp(self, J, F_tmp, U, S, V, Jp):
        J = _fs(J, -1, "J")
        n = J.shape[0]
        Jp = _fs(Jp, n, "Jp")
        F_tmp, U, S, V = (_m_in(a, n, name) for a, name in
                          ((F_tmp, "F_tmp"), (U, "U"), (S, "S"), (V, "V")))
        out_F, out_S, out_Jp = _m(n), _m(n), _f(n)
        F9, U9, S9, V9 = (a.reshape(-1, 9) for a in (F_tmp, U, S, V))
        oF9, oS9 = out_F.reshape(-1, 9), out_S.reshape(-1, 9)
        lib().mgw_mpm_update_F_S_Jp_snow(
            self.yield_lower, self.yield_higher, addr(J), addr(F9),
            addr(U9), addr(S9), addr(V9), addr(Jp),
            addr(oF9), addr(oS9), addr(out_Jp), n,
        )
        return out_F, out_S, out_Jp

    def update_stress(self, U, S, V, F_tmp, F_new, J, Jp, actu=None, m_dir=None):
        J = _fs(J, -1, "J")
        n = J.shape[0]
        Jp = _fs(Jp, n, "Jp")
        U, S, V, F_tmp, F_new = (_m_in(a, n, name) for a, name in
                                 ((U, "U"), (S, "S"), (V, "V"), (F_tmp, "F_tmp"), (F_new, "F_new")))
        out = _m(n)
        out9 = out.reshape(-1, 9)
        lib().mgw_mpm_update_stress_snow(
            self.mu, self.lam, self.yield_lower, self.yield_higher,
            *[addr(a.reshape(-1, 9)) for a in (U, S, V, F_tmp, F_new)], addr(J), addr(Jp),
            addr(out9), n,
        )
        return out


# ------------------------------------------------------------- mpm_solver.py


def stencil_range() -> int:
    return lib().mgw_mpm_stencil_range()


def compute_F_tmp(C, F, substep_dt: float) -> np.ndarray:
    """`MPMSolver.compute_F_tmp` for a batch of particles."""
    n = np.asarray(C).reshape(-1, 3, 3).shape[0]
    C9 = _m_in(C, n, "C").reshape(-1, 9)
    F9 = _m_in(F, n, "F").reshape(-1, 9)
    out = _m(n)
    out9 = out.reshape(-1, 9)
    lib().mgw_mpm_compute_F_tmp(addr(C9), addr(F9), substep_dt, addr(out9), n)
    return out


def p2g(
    pos,
    vel,
    C,
    F,
    F_tmp,
    U,
    S,
    V,
    Jp,
    active,
    mass,
    free,
    material: Base,
    substep_dt: float,
    dx: float,
    particle_volume: float,
    grid_offset,
    grid_res,
    grid_vel_in,
    grid_mass,
    friction_angle: float | None = None,
    use_von_mises: bool | None = None,
    von_mises_yield_stress: float | None = None,
    yield_lower: float | None = None,
    yield_higher: float | None = None,
    model: int | None = None,
    needs_svd: bool | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """`MPMSolver.p2g`: material update, stress, and the 27-cell scatter.

    `model` defaults to the material's own `model_id`, and the yield and friction
    parameters default to the fields of the same material, so a call cannot
    silently run one constitutive model with another one's parameters.
    """
    if model is None:
        model = material.model_id
    if model not in _MODEL_NAMES:
        raise ValueError(f"unknown MPM model {model!r}")
    if friction_angle is None:
        friction_angle = getattr(material, "friction_angle", 45.0)
    if yield_lower is None:
        yield_lower = getattr(material, "yield_lower", 2.5e-2)
    if yield_higher is None:
        yield_higher = getattr(material, "yield_higher", 4.5e-3)
    if use_von_mises is None:
        use_von_mises = getattr(material, "use_von_mises", True)
    if von_mises_yield_stress is None:
        von_mises_yield_stress = getattr(material, "von_mises_yield_stress", 10000.0)
    if needs_svd is None:
        needs_svd = material.needs_svd
    pos, vel = _v3(pos), _v3(vel)
    n = pos.shape[0]
    pos, vel = align(n, pos=pos, vel=vel)
    inv_dx = 1.0 / dx
    out_F, out_Jp = _m(n), _f(n)
    out_F9 = out_F.reshape(-1, 9)
    vel_grid = np.ascontiguousarray(grid_vel_in, dtype=np.float64).reshape(-1)
    mass_grid = np.ascontiguousarray(grid_mass, dtype=np.float64).reshape(-1)
    off, res = _v3(grid_offset), _v3(grid_res)
    C9, F9, F_tmp9, U9, S9, V9 = (
        _m_in(a, n, name).reshape(-1, 9) for a, name in
        ((C, "C"), (F, "F"), (F_tmp, "F_tmp"), (U, "U"), (S, "S"), (V, "V"))
    )
    Jp = _fs(Jp, n, "Jp")
    mass = _fs(mass, n, "mass")
    active = np.ascontiguousarray(active, dtype=np.int32).reshape(-1)
    free = np.ascontiguousarray(free, dtype=np.int32).reshape(-1)
    active, free = align(n, active=active, free=free)
    lib().mgw_mpm_p2g(
        addr(pos), addr(vel), addr(C9), addr(F9),
        addr(F_tmp9), addr(U9), addr(S9), addr(V9),
        addr(Jp), addr(active), addr(mass), addr(free),
        material.mu, material.lam, yield_lower, yield_higher, 1 if use_von_mises else 0,
        von_mises_yield_stress, friction_angle, model, 1 if needs_svd else 0,
        substep_dt, dx, inv_dx, particle_volume,
        addr(off), addr(res), addr(vel_grid), addr(mass_grid),
        addr(out_F9), addr(out_Jp), n,
    )
    return out_F, out_Jp


def g2p(pos, grid_vel_out, lower, upper, substep_dt: float, dx: float, grid_offset, grid_res):
    """`MPMSolver.g2p`: the 27-cell gather, the C update and the boundary clamp."""
    pos = _v3(pos)
    n = pos.shape[0]
    inv_dx = 1.0 / dx
    out_pos, out_vel, out_C = _out3(n), _out3(n), _m(n)
    # Named locals: the ctypes call takes addresses, and an unnamed temporary
    # flattened view can be released before the call reads through it.
    grid = np.ascontiguousarray(grid_vel_out, dtype=np.float64).reshape(-1)
    lo, up, off, res = _v3(lower), _v3(upper), _v3(grid_offset), _v3(grid_res)
    lib().mgw_mpm_g2p(
        addr(pos), addr(grid), addr(lo), addr(up),
        substep_dt, inv_dx, addr(off), addr(res),
        addr(out_pos), addr(out_vel), addr(out_C.reshape(-1, 9)), n,
    )
    return out_pos, out_vel, out_C


def backward_svd(grad_U, grad_S, grad_V, U, S, V) -> np.ndarray:
    """Backward of a 3x3 SVD, PyTorch Functions.cpp convention."""
    n = np.asarray(U).reshape(-1, 3, 3).shape[0]
    mats = [_m_in(a, n, name) for a, name in
            ((grad_U, "grad_U"), (grad_S, "grad_S"), (grad_V, "grad_V"),
             (U, "U"), (S, "S"), (V, "V"))]
    out = _m(n)
    out9 = out.reshape(-1, 9)
    lib().mgw_mpm_backward_svd(
        *[addr(a.reshape(-1, 9)) for a in mats], addr(out9), n
    )
    return out
