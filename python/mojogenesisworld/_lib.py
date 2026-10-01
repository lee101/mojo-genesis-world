"""Loads (and if necessary builds) the compiled Mojo library."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "src")
_LIB_NAME = "libmojo-genesis-world.so"
# An installed wheel carries the compiled library next to the package; a source
# checkout keeps it in dist/ and rebuilds it there.
_PACKAGED_LIB = os.path.join(HERE, _LIB_NAME)
_DEV_LIB = os.path.join(ROOT, "dist", _LIB_NAME)
LIB = os.environ.get("MOJOGENESISWORLD_LIB") or (
    _PACKAGED_LIB if os.path.exists(_PACKAGED_LIB) else _DEV_LIB
)

I = ctypes.c_int64
F = ctypes.c_double

# name -> (argtypes, restype)
_SIGNATURES = {
    "mgw_activate": ([I, I, I, I, I, I, I, I, I, I], None),
    "mgw_compute_geom_pair_scale": ([I, I, I, I, I, I], None),
    "mgw_compute_geom_pair_scale_mj": ([I, I, I, I, I, I, I], None),
    "mgw_compute_geom_rbound": ([I, I, I, I, I, I], None),
    "mgw_compute_mc_tolerance": ([I, I, I, I, I, F, I, I, I], None),
    "mgw_contact_order_key": ([I, I, I], None),
    "mgw_contact_orthogonals": ([I, I, I, I], None),
    "mgw_count_supports_box": ([I, I, I, I], None),
    "mgw_count_supports_mesh": ([I, I, I, I, I, I, I], None),
    "mgw_count_supports_world": ([I, I, I, I, I, I, I, I], None),
    "mgw_det3": ([I, I, I, I, I], None),
    "mgw_direction_grid_coords": ([I, I, I, I, I, I], None),
    "mgw_direction_grid_nums": ([I], I),
    "mgw_func_support": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I], None),
    "mgw_get_direction_grid": ([I, I], I),
    "mgw_inv_quat": ([I, I, I], None),
    "mgw_inv_transform_by_quat": ([I, I, I, I], None),
    "mgw_inv_transform_by_trans_quat": ([I, I, I, I, I], None),
    "mgw_mpm_backward_svd": ([I, I, I, I, I, I, I, I], None),
    "mgw_mpm_compute_F_tmp": ([I, I, F, I, I], None),
    "mgw_mpm_g2p": ([I, I, I, I, F, F, I, I, I, I, I, I], None),
    "mgw_mpm_p2g": ([I, I, I, I, I, I, I, I, I, I, I, I, F, F, F, F, I, F, F, I, I, F, F, F, F, I, I, I, I, I, I, I], None),
    "mgw_mpm_sand_alpha": ([F, I], None),
    "mgw_mpm_sand_projection": ([F, F, F, I, I, I, I, I], None),
    "mgw_mpm_stencil_range": ([], I),
    "mgw_mpm_update_F_S_Jp_elastic": ([I, I, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_F_S_Jp_elasto_plastic": ([F, I, F, F, F, I, I, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_F_S_Jp_liquid": ([I, I, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_F_S_Jp_noop": ([I, I, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_F_S_Jp_sand": ([F, F, F, I, I, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_F_S_Jp_snow": ([F, F, I, I, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_stress_corotation": ([F, F, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_stress_default": ([F, F, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_stress_liquid": ([F, F, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_stress_neohooken": ([F, F, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_stress_sand": ([F, F, I, I, I, I, I, I, I, I], None),
    "mgw_mpm_update_stress_snow": ([F, F, F, F, I, I, I, I, I, I, I, I, I], None),
    "mgw_normalize": ([I, F, I, I], None),
    "mgw_origin_tetra_intersection": ([I, I, I, I, I, I], None),
    "mgw_orthogonals": ([I, I, I, I], None),
    "mgw_point_plane_same_side": ([I, I, I, I, I, I], None),
    "mgw_point_triangle_intersection": ([I, I, I, I, I, I], None),
    "mgw_polar_band": ([I], I),
    "mgw_project_origin_to_plane": ([I, I, I, I, I, I], None),
    "mgw_quat_mul": ([I, I, I, I], None),
    "mgw_quat_to_R": ([I, F, I, I], None),
    "mgw_ray_triangle_intersection": ([I, I, I, I, I, I, I], None),
    "mgw_resolve_lame": ([F, F, I, I], None),
    "mgw_rotate_frame": ([I, I, I, I, I, I, I], None),
    "mgw_support_box": ([I, I, I, I, I, I, I, I, I, I], None),
    "mgw_support_capsule": ([I, I, I, I, I, I, I, I], None),
    "mgw_support_cylinder": ([I, I, I, I, I, I, I, I], None),
    "mgw_support_driver": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I], None),
    "mgw_support_ellipsoid": ([I, I, I, I, I, I, I], None),
    "mgw_support_mesh": ([I, I, I, I, I, I, I, I, I, I, I], None),
    "mgw_support_mesh_exhaustive": ([I, I, I, I, I, I, I, I], None),
    "mgw_support_mesh_table": ([I, I, I, I, I, I, I, I, I], None),
    "mgw_support_prism": ([I, I, I, I, I, I], None),
    "mgw_support_sphere": ([I, I, I, I, I, I, I, I, I, I], None),
    "mgw_support_world": ([I, I, I, I, I, I, I, I, I, I, I, I, I, I, I, I], None),
    "mgw_transform_by_quat": ([I, I, I, I], None),
    "mgw_transform_by_quat_fast": ([I, I, I, I], None),
    "mgw_transform_by_trans_quat": ([I, I, I, I, I], None),
    "mgw_transform_by_trans_quat_fast": ([I, I, I, I, I], None),
    "mgw_transform_quat_by_quat": ([I, I, I, I], None),
    "mgw_triangle_affine_coords": ([I, I, I, I, I, I], None),
}

_MAX_WORKERS = 32


def _pids_headroom(reserve: int = 8) -> int:
    """Threads a `parallelize` launch can still spawn on this host.

    A container's `pids` limit counts threads, and `pthread_create` failing
    inside a Mojo `parallelize` aborts the calling process rather than raising,
    so the budget has to be known before the launch rather than after it.
    """
    roots = ["/sys/fs/cgroup"]
    try:
        with open("/proc/self/cgroup") as f:
            for line in f:
                parts = line.strip().split("::", 1)
                if len(parts) == 2 and parts[1]:
                    roots.insert(0, "/sys/fs/cgroup" + parts[1])
    except OSError:
        pass
    for root in roots:
        try:
            with open(os.path.join(root, "pids.max")) as f:
                limit = f.read().strip()
            if limit == "max":
                return 1 << 30
            with open(os.path.join(root, "pids.current")) as f:
                current = int(f.read().strip())
        except OSError:
            continue
        return max(1, int(limit) - current - reserve)
    return 1 << 30


def worker_budget(limit: int = _MAX_WORKERS) -> int:
    """How many workers `activate` may ask for; 1 means the serial loop.

    The default is 1. A thread that cannot be created inside a Mojo
    `parallelize` aborts the calling process, it does not raise, so the only safe
    default on a box whose thread budget is shared with other processes is the
    serial loop. Set `MOJOGENESISWORLD_WORKERS`, or pass `workers=` to
    `SupportField.activate`, to ask for threads.
    """
    override = os.environ.get("MOJOGENESISWORLD_WORKERS")
    if override is None:
        return 1
    try:
        n = int(override)
    except ValueError:
        return 1
    if n < 1:
        return 1
    try:
        n_cpu = len(os.sched_getaffinity(0))
    except AttributeError:
        n_cpu = os.cpu_count() or 1
    return max(1, min(n, limit, n_cpu, _pids_headroom()))


def align(n: int, **arrays: np.ndarray) -> list[np.ndarray]:
    """Check that every batched input has `n` rows, broadcasting a single row.

    The Mojo loops index each input at the same `i`, so a shorter input would be
    read past its end. Anything but `n` or 1 rows is a caller error, not something
    to paper over.
    """
    out = []
    for name, a in arrays.items():
        rows = a.shape[0] if a.ndim else 1
        if rows == n:
            out.append(a)
        elif rows == 1:
            out.append(np.ascontiguousarray(np.broadcast_to(a, (n,) + a.shape[1:])))
        else:
            raise ValueError(f"{name} has {rows} rows, expected {n}")
    return out


class BuildError(RuntimeError):
    pass


def mojo_command() -> list[str]:
    override = os.environ.get("MOJOGENESISWORLD_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi") or os.path.expanduser("~/.pixi/bin/pixi")
    if os.path.exists(pixi) and os.path.exists(os.path.join(ROOT, "pixi.toml")):
        return [pixi, "run", "--manifest-path", os.path.join(ROOT, "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOGENESISWORLD_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Compile `src/capi.mojo` into `dist/libmojo-genesis-world.so` if it is stale."""
    sources = [
        os.path.join(dirpath, name)
        for dirpath, _, names in os.walk(SRC)
        for name in names
        if name.endswith(".mojo")
    ]
    if not sources:
        if os.path.exists(LIB):
            return LIB
        raise BuildError(
            f"no Mojo sources under {SRC} and no prebuilt library at {LIB}; "
            "set MOJOGENESISWORLD_LIB to a compiled libmojo-genesis-world.so"
        )
    if not force and os.path.exists(LIB):
        newest = max(os.path.getmtime(s) for s in sources)
        if os.path.getmtime(LIB) >= newest:
            return LIB
    os.makedirs(os.path.dirname(LIB) or ".", exist_ok=True)
    # Same reason as build/build.sh: without an explicit count `mojo build` spawns a
    # thread per core and can abort before reading the source under a task cap.
    threads = os.environ.get("MOJO_BUILD_THREADS", "4")
    cmd = mojo_command() + [
        "build", "-j", threads, "--emit", "shared-lib", "-I", SRC,
        os.path.join(SRC, "capi.mojo"), "-o", LIB,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or not os.path.exists(LIB):
        raise BuildError((proc.stderr or proc.stdout).strip()[:4000])
    return LIB


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            fn = getattr(_lib, name)
            fn.argtypes = argtypes
            fn.restype = restype
    return _lib


def f64(a, copy: bool = False) -> np.ndarray:
    """A C-contiguous float64 view of `a`, copying only when it has to."""
    arr = np.asarray(a, dtype=np.float64)
    if not arr.flags["C_CONTIGUOUS"] and not copy:
        arr = np.ascontiguousarray(arr)
    return arr


def i32(a, copy: bool = False) -> np.ndarray:
    arr = np.asarray(a, dtype=np.int32)
    if not arr.flags["C_CONTIGUOUS"] and not copy:
        arr = np.ascontiguousarray(arr)
    return arr


def f64_copy(a) -> np.ndarray:
    return np.ascontiguousarray(a, dtype=np.float64)


def addr(a: np.ndarray) -> int:
    return a.ctypes.data


def n_of(a) -> int:
    return a.shape[0]


def main() -> int:
    """`python -m mojogenesisworld._lib` rebuilds the library."""
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
