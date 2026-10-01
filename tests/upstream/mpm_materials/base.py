"""The constitutive kernels of `genesis/engine/materials/MPM/base.py`, lines 98-108.

Copied verbatim from `genesis-world==1.4.2`. The only change is that each method
is de-indented by four columns so it is a module-level function here; the bodies,
the argument order and the `self.mu` / `self.lam` / yield-field reads are
upstream's. See `tests/upstream/LICENSE-NOTICE.md`.
"""

import math

import quadrants as qd

import genesis as gs


@qd.func
def _update_F_S_Jp_noop(self, J, F_tmp, U, S, V, Jp):
    raise NotImplementedError

@qd.func
def _update_stress_default(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
    stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
        gs.qd_float, 3
    ) * self.lam * J * (J - 1)
    return stress
