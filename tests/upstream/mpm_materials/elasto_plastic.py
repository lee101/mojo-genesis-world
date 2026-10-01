"""The constitutive kernels of `genesis/engine/materials/MPM/elasto_plastic.py`, lines 45-73.

Copied verbatim from `genesis-world==1.4.2`. The only change is that each method
is de-indented by four columns so it is a module-level function here; the bodies,
the argument order and the `self.mu` / `self.lam` / yield-field reads are
upstream's. See `tests/upstream/LICENSE-NOTICE.md`.
"""

import math

import quadrants as qd

import genesis as gs


@qd.func
def _update_F_S_Jp_elasto_plastic(self, J, F_tmp, U, S, V, Jp):
    F_new = qd.Matrix.zero(gs.qd_float, 3, 3)
    S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
    if qd.static(self.use_von_mises):
        S_new = qd.max(S, 0.05)  # to prevent NaN
        epsilon = qd.Vector([qd.log(S_new[0, 0]), qd.log(S_new[1, 1]), qd.log(S_new[2, 2])])
        epsilon_hat = epsilon - (epsilon.sum() / 3)
        epsilon_hat_norm = epsilon_hat.norm(gs.EPS)
        delta_gamma = epsilon_hat_norm - self.von_mises_yield_stress / (2 * self.mu)

        if delta_gamma > 0:  # Yields
            epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
            S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
            for d in qd.static(range(3)):
                S_new[d, d] = qd.exp(epsilon[d])
            F_new = U @ S_new @ V.transpose()
        else:
            F_new = F_tmp

    else:
        S_new = qd.Matrix.zero(gs.qd_float, 3, 3)
        for d in qd.static(range(3)):
            S_new[d, d] = min(max(S[d, d], 1 - self.yield_lower), 1 + self.yield_higher)
        F_new = U @ S_new @ V.transpose()

    Jp_new = Jp
    return F_new, S_new, Jp_new
