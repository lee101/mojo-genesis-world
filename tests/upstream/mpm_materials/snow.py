"""The constitutive kernels of `genesis/engine/materials/MPM/snow.py`, lines 56-74.

Copied verbatim from `genesis-world==1.4.2`. The only change is that each method
is de-indented by four columns so it is a module-level function here; the bodies,
the argument order and the `self.mu` / `self.lam` / yield-field reads are
upstream's. See `tests/upstream/LICENSE-NOTICE.md`.
"""

import math

import quadrants as qd

import genesis as gs


@qd.func
def _update_F_S_Jp_snow(self, J, F_tmp, U, S, V, Jp):
    S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
    Jp_new = Jp
    for d in qd.static(range(3)):
        S_new[d, d] = min(max(S[d, d], 1 - self.yield_lower), 1 + self.yield_higher)
        Jp_new *= S[d, d] / S_new[d, d]
    F_new = U @ S_new @ V.transpose()
    return F_new, S_new, Jp_new

@qd.func
def _update_stress_snow(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
    # Hardening coefficient: material harder when compressed
    h = qd.exp(10 * (1.0 - Jp))
    mu, lam = self.mu * h, self.lam * h

    r = U @ V.transpose()
    stress = 2 * mu * (F_new - r) @ F_new.transpose() + qd.Matrix.identity(gs.qd_float, 3) * lam * J * (J - 1)

    return stress
