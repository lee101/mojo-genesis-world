"""The two pure module-level functions at the tail of upstream `mpm_solver.py`.

Copied verbatim from `genesis/engine/solvers/mpm_solver.py` lines 1321-1345 of
`genesis-world==1.4.2`. The rest of that file cannot be vendored: it imports the
whole engine (`genesis.engine.entities`, `genesis.options.solvers`, the coupler,
the boundaries) at module scope, and those pull in the GPU runtime. These two
functions read nothing but their arguments, which is the only reason they are
here. See `tests/upstream/LICENSE-NOTICE.md`.
"""

import quadrants as qd

import genesis as gs


@qd.func
def signmax(a, eps):
    sign = qd.select(a >= 0, 1.0, -1.0)
    return sign * qd.max(qd.abs(a), eps)


@qd.func
def backward_svd(grad_U, grad_S, grad_V, U, S, V):
    # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
    vt = V.transpose()
    ut = U.transpose()
    S_term = U @ grad_S @ vt

    s = qd.Vector.zero(gs.qd_float, 3)
    s = qd.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
    F = qd.Matrix.zero(gs.qd_float, 3, 3)
    for i, j in qd.static(qd.ndrange(3, 3)):
        if i == j:
            F[i, j] = 0.0
        else:
            F[i, j] = 1.0 / signmax(s[j] - s[i], 1e-6)
    u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
    v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
    return u_term + v_term + S_term
