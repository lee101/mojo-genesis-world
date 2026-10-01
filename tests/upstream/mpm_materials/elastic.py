"""The constitutive kernels of `genesis/engine/materials/MPM/elastic.py`, lines 46-65.

Copied verbatim from `genesis-world==1.4.2`. The only change is that each method
is de-indented by four columns so it is a module-level function here; the bodies,
the argument order and the `self.mu` / `self.lam` / yield-field reads are
upstream's. See `tests/upstream/LICENSE-NOTICE.md`.
"""

import math

import quadrants as qd

import genesis as gs


@qd.func
def _update_F_S_Jp_elastic(self, J, F_tmp, U, S, V, Jp):
    F_new = F_tmp
    S_new = S
    Jp_new = Jp
    return F_new, S_new, Jp_new

@qd.func
def _update_stress_corotation(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
    stress = 2 * self.mu * (F_new - U @ V.transpose()) @ F_new.transpose() + qd.Matrix.identity(
        gs.qd_float, 3
    ) * self.lam * J * (J - 1)
    return stress

@qd.func
def _update_stress_neohooken(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
    stress = self.mu * (F_tmp @ F_tmp.transpose()) + qd.Matrix.identity(gs.qd_float, 3) * (
        self.lam * qd.log(J) - self.mu
    )
    return stress
