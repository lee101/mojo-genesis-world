"""The constitutive kernels of `genesis/engine/materials/MPM/sand.py`, lines 47-83.

Copied verbatim from `genesis-world==1.4.2`. The only change is that each method
is de-indented by four columns so it is a module-level function here; the bodies,
the argument order and the `self.mu` / `self.lam` / yield-field reads are
upstream's. See `tests/upstream/LICENSE-NOTICE.md`.
"""

import math

import quadrants as qd

import genesis as gs


@qd.func
def _sand_projection(self, S, Jp):
    S_out = qd.Matrix.zero(gs.qd_float, 3, 3)
    epsilon = qd.Vector.zero(gs.qd_float, 3)
    for i in qd.static(range(3)):
        epsilon[i] = qd.log(max(abs(S[i, i]), 1e-4))
        S_out[i, i] = 1
    tr = epsilon.sum() + Jp
    epsilon_hat = epsilon - tr / 3
    epsilon_hat_norm = epsilon_hat.norm(gs.EPS)

    Jp_new = gs.qd_float(0.0)
    if tr >= 0.0:
        Jp_new = tr
    else:
        Jp_new = 0.0
        delta_gamma = epsilon_hat_norm + (3 * self.lam + 2 * self.mu) / (2 * self.mu) * tr * self.alpha
        for i in qd.static(range(3)):
            S_out[i, i] = qd.exp(epsilon[i] - max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat[i])

    return S_out, Jp_new

@qd.func
def _update_F_S_Jp_sand(self, J, F_tmp, U, S, V, Jp):
    S_new, Jp_new = self._sand_projection(S, Jp)
    F_new = U @ S_new @ V.transpose()
    return F_new, S_new, Jp_new

@qd.func
def _update_stress_sand(self, U, S, V, F_tmp, F_new, J, Jp, actu, m_dir):
    log_S_sum = gs.qd_float(0.0)
    center = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i in qd.static(range(3)):
        log_S_sum += qd.log(S[i, i])
        center[i, i] = 2.0 * self.mu * qd.log(S[i, i]) * (1 / S[i, i])
    for i in qd.static(range(3)):
        center[i, i] += self.lam * log_S_sum * (1 / S[i, i])
    stress = U @ center @ V.transpose() @ F_new.transpose()
    return stress
