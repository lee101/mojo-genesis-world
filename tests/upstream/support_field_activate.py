"""The table-building body of `SupportField.activate`, lines 62-98 of upstream.

Copied verbatim from `genesis/engine/solvers/rigid/collider/support_field.py`
(`genesis-world==1.4.2`), with two changes: the body is de-indented by eight
columns into a function of its own, it stops before the
`array_class.get_support_field_info` / `_kernel_init_support` tail (which needs a
live engine and only copies what this function already computed), and the three
`.to_numpy()` calls are dropped because the harness hands it NumPy arrays.

See `tests/upstream/LICENSE-NOTICE.md`.
"""

import math

import numpy as np

import quadrants as qd

import genesis as gs


def build_support_table(self):
    v = self._get_direction_grid()
    v1 = v.reshape([-1, 3])
    num_v = len(v1)

    support_v = []
    support_vid = []
    support_cell_start = []
    n_support_cells = 0
    if self.solver.n_geoms > 0:
        init_pos = self.solver.dyn_info.verts.init_pos
        geoms_vert_start = self.solver.dyn_info.geoms.vert_start
        geoms_vert_end = self.solver.dyn_info.geoms.vert_end
        for i_g in range(self.solver.n_geoms):
            this_pos = init_pos[geoms_vert_start[i_g] : geoms_vert_end[i_g]]

            window_size = int(5e8 // this_pos.shape[0])
            max_indices = np.empty(num_v, dtype=np.intp)

            for i in range(0, num_v, window_size):
                end = min(i + window_size, num_v)
                dot_chunk = v1[i:end] @ this_pos.T
                max_indices[i:end] = np.argmax(dot_chunk, axis=1)

            support = this_pos[max_indices]

            support_cell_start.append(n_support_cells)
            support_v.append(support)
            support_vid.append(max_indices)
            n_support_cells += support.shape[0]

        support_v = np.concatenate(support_v)
        support_vid = np.concatenate(support_vid, dtype=gs.np_int)
        support_cell_start = np.array(support_cell_start, dtype=gs.np_int)
    else:
        support_v = np.zeros((1, 3), dtype=gs.np_float)
        support_vid = np.zeros((1,), dtype=gs.np_int)
        support_cell_start = np.zeros((1,), dtype=gs.np_int)

    return v1, num_v, support_v, support_vid, support_cell_start, n_support_cells
