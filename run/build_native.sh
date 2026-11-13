#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build_native"
build_type="Release"
tensorrt_mode="${DKFZ_TENSORRT_MODE:-AUTO}"

usage() {
    cat <<'USAGE'
Usage: bash run/build_native.sh [options]

Configure and incrementally build the shipped runtime, parity and capture tools.

Options:
  --tensorrt MODE   TensorRT mode: auto, on, or off. Default: auto.
  --build-dir DIR   CMake build directory. Default: build_native.
  --build-type TYPE CMake build type. Default: Release.
  --test            Compatibility flag; development tests are not bundled.
  --help            Show this help.

Use --tensorrt on for every production live-inference build. AUTO and OFF may
produce a stub inference engine that cannot run live model inference.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tensorrt)
            [[ $# -ge 2 ]] || { echo "--tensorrt requires a value" >&2; exit 2; }
            tensorrt_mode="${2^^}"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || { echo "--build-dir requires a path" >&2; exit 2; }
            build_dir="$2"
            shift 2
            ;;
        --build-type)
            [[ $# -ge 2 ]] || { echo "--build-type requires a value" >&2; exit 2; }
            build_type="$2"
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

case "${tensorrt_mode}" in
    AUTO|ON|OFF) ;;
    *)
        echo "--tensorrt must be one of: auto, on, off" >&2
        exit 2
        ;;
esac

cmake \
    -S "${repo_root}/native" \
    -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE="${build_type}" \
    -DDKFZ_ENABLE_TENSORRT="${tensorrt_mode}" \
    -DBUILD_TESTING=OFF
cmake --build "${build_dir}" --parallel

printf 'TensorRT build mode: %s\n' "${tensorrt_mode}"
printf 'Built %s\n' "${build_dir}/uart/uart_receiver_stream"
printf 'Built %s\n' "${build_dir}/inference/native_live_runtime"
printf 'Built %s\n' "${build_dir}/inference/native_uart_live_runtime"
printf 'Built %s\n' "${build_dir}/inference/native_live_recovery_control"
