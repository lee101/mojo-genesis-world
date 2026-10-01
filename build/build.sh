#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$repo_dir/dist"

# src/capi.mojo is the single entry point; -I src resolves the ported module that
# holds the kernels. One compilation unit: build cost is fixed, so everything goes
# into it.

# `mojo build` defaults to one compiler thread per core. On a box that caps tasks
# (a container `pids` limit counts threads, not just processes) that default can
# overshoot what is left and abort the compiler with `pthread_create failed`
# before it ever reads the source. Compile cost here is fixed rather than
# parallel, so a small pool costs nothing; raise it with MOJO_BUILD_THREADS.
threads="${MOJO_BUILD_THREADS:-4}"

mojo build -j "$threads" --emit shared-lib -I "$repo_dir/src" \
    "$repo_dir/src/capi.mojo" \
    -o "$repo_dir/dist/libmojo-genesis-world.so"
