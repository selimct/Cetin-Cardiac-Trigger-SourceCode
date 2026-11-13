#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build_native"
venv_dir="${repo_root}/.venv"
rebuild_python_extension=0
targets=()

usage() {
    cat <<'USAGE'
Usage: bash run/rebuild_native.sh [options]

Quickly rebuild changed C++ files using an existing CMake configuration.
The configured TensorRT mode is preserved.

Options:
  --build-dir DIR       Existing CMake build directory.
  --target TARGET       Build only this CMake target. Repeatable.
  --python-extension    Also rebuild the dkfz.native._live_core extension.
  --venv DIR            Python environment used with --python-extension.
  --test                Compatibility flag; development tests are not bundled.
  --help                Show this help.

Examples:
  bash run/rebuild_native.sh
  bash run/rebuild_native.sh --target native_uart_live_runtime
  bash run/rebuild_native.sh --python-extension
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)
            [[ $# -ge 2 ]] || { echo "--build-dir requires a path" >&2; exit 2; }
            build_dir="$2"
            shift 2
            ;;
        --target)
            [[ $# -ge 2 ]] || { echo "--target requires a name" >&2; exit 2; }
            targets+=("$2")
            shift 2
            ;;
        --python-extension)
            rebuild_python_extension=1
            shift
            ;;
        --venv)
            [[ $# -ge 2 ]] || { echo "--venv requires a path" >&2; exit 2; }
            venv_dir="$2"
            shift 2
            ;;
        --test)
            echo "Development tests are not bundled in the public distribution." >&2
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    echo "No configured native build found at ${build_dir}." >&2
    echo "Run setup.sh or run/build_native.sh first." >&2
    exit 2
fi

build_args=(--build "${build_dir}" --parallel)
if [[ ${#targets[@]} -gt 0 ]]; then
    build_args+=(--target "${targets[@]}")
fi
cmake "${build_args[@]}"

if [[ "${rebuild_python_extension}" -eq 1 ]]; then
    venv_python="${venv_dir}/bin/python"
    if [[ ! -x "${venv_python}" ]]; then
        echo "Python environment not found: ${venv_python}" >&2
        exit 2
    fi
    "${venv_python}" -m pip install \
        --no-deps \
        --no-build-isolation \
        --editable "${repo_root}"
    "${venv_python}" -c \
        'from dkfz.live import feature_backend_name; print("Python feature backend:", feature_backend_name())'
fi

printf 'Incremental native rebuild complete: %s\n' "${build_dir}"
