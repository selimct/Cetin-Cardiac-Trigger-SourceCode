#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
profile="workstation"
python_override=""
tensorrt_override=""
venv_dir="${repo_root}/.venv"
venv_override=0
bootstrap_python="python3"
build_dir="${repo_root}/build_native"

usage() {
    cat <<'USAGE'
Usage: bash setup.sh [options]

Reuse or create a Python environment, compile the native Python extension,
configure/build the standalone C++ runtime.
The command is idempotent and may be rerun. Tests are maintained in the
development repository.

Options:
  --profile PROFILE       workstation, jetson, or production.
  --with-python           Create/reuse and repair the Python environment.
  --no-python             Skip all Python environment work.
  --python EXECUTABLE     Python used only when an environment must be created.
  --venv DIR              Virtual-environment path. Default: .venv; Jetson uses
                          the environment activated by pythonbase.
  --tensorrt MODE         Override TensorRT mode: auto, on, or off.
  --build-dir DIR         Native CMake build directory.
  --skip-tests            Accepted for compatibility; tests are not bundled.
  --help                  Show this help.

Profiles:
  workstation  Python training/ONNX environment; TensorRT AUTO.
  jetson       Python training/ONNX using the pythonbase environment; TensorRT ON.
  production   No Python environment; native TensorRT runtime only.
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --profile)
            [[ $# -ge 2 ]] || { echo "--profile requires a value" >&2; exit 2; }
            profile="$2"
            shift 2
            ;;
        --with-python)
            python_override=1
            shift
            ;;
        --no-python)
            python_override=0
            shift
            ;;
        --python)
            [[ $# -ge 2 ]] || { echo "--python requires an executable" >&2; exit 2; }
            bootstrap_python="$2"
            shift 2
            ;;
        --venv)
            [[ $# -ge 2 ]] || { echo "--venv requires a path" >&2; exit 2; }
            venv_dir="$2"
            venv_override=1
            shift 2
            ;;
        --tensorrt)
            [[ $# -ge 2 ]] || { echo "--tensorrt requires a value" >&2; exit 2; }
            tensorrt_override="${2^^}"
            shift 2
            ;;
        --build-dir)
            [[ $# -ge 2 ]] || { echo "--build-dir requires a path" >&2; exit 2; }
            build_dir="$2"
            shift 2
            ;;
        --skip-tests)
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

case "${profile}" in
    workstation)
        with_python=1
        system_site_packages=0
        python_extras="train,onnx"
        tensorrt_mode="AUTO"
        ;;
    jetson)
        with_python=1
        system_site_packages=1
        python_extras="train,onnx"
        tensorrt_mode="ON"
        ;;
    production)
        with_python=0
        system_site_packages=1
        python_extras=""
        tensorrt_mode="ON"
        ;;
    *)
        echo "--profile must be one of: workstation, jetson, production" >&2
        exit 2
        ;;
esac

if [[ -n "${python_override}" ]]; then
    with_python="${python_override}"
fi
if [[ -n "${tensorrt_override}" ]]; then
    tensorrt_mode="${tensorrt_override}"
fi
case "${tensorrt_mode}" in
    AUTO|ON|OFF) ;;
    *)
        echo "--tensorrt must be one of: auto, on, off" >&2
        exit 2
        ;;
esac
if [[ "${profile}" != "workstation" && "${tensorrt_mode}" != "ON" ]]; then
    echo "The ${profile} profile requires --tensorrt on." >&2
    echo "Use the workstation profile for AUTO or stub builds." >&2
    exit 2
fi

for required_command in cmake c++; do
    if ! command -v "${required_command}" >/dev/null 2>&1; then
        echo "Required command not found: ${required_command}" >&2
        exit 1
    fi
done

if [[ "${with_python}" -eq 1 ]]; then
    if [[ "${profile}" == "jetson" && "${venv_override}" -eq 0 ]]; then
        bash_aliases="${HOME:-}/.bash_aliases"
        if [[ ! -r "${bash_aliases}" ]]; then
            echo "Jetson Python environment configuration not found: ${bash_aliases}" >&2
            echo "Define pythonbase in ~/.bash_aliases to activate the existing environment." >&2
            exit 1
        fi

        # Resolve the activation target without invoking it so a missing
        # environment can be created at the path named by pythonbase.
        shopt -s expand_aliases
        # shellcheck disable=SC1090
        source "${bash_aliases}"
        if ! alias pythonbase >/dev/null 2>&1; then
            echo "Jetson Python environment alias not found: pythonbase" >&2
            echo "Define pythonbase in ${bash_aliases} to activate the existing environment." >&2
            exit 1
        fi

        pythonbase_definition="$(alias pythonbase)"
        eval "pythonbase_command=${pythonbase_definition#*=}"
        pythonbase_parts=()
        eval "pythonbase_parts=(${pythonbase_command})"
        if [[ "${#pythonbase_parts[@]}" -ne 2 ]] || \
                [[ "${pythonbase_parts[0]}" != "source" && "${pythonbase_parts[0]}" != "." ]] || \
                [[ "${pythonbase_parts[1]}" != */bin/activate ]]; then
            echo "The pythonbase alias must source a virtual-environment bin/activate script." >&2
            exit 1
        fi
        activate_script="${pythonbase_parts[1]}"
        venv_dir="${activate_script%/bin/activate}"
    fi

    venv_python="${venv_dir}/bin/python"
    if [[ ! -x "${venv_python}" ]]; then
        if ! command -v "${bootstrap_python}" >/dev/null 2>&1; then
            echo "Python executable not found: ${bootstrap_python}" >&2
            exit 1
        fi
        venv_args=(-m venv)
        if [[ "${system_site_packages}" -eq 1 ]]; then
            venv_args+=(--system-site-packages)
        fi
        "${bootstrap_python}" "${venv_args[@]}" "${venv_dir}"
    fi
    activate_script="${venv_dir}/bin/activate"
    if [[ ! -r "${activate_script}" ]]; then
        echo "Python environment activation script not found: ${activate_script}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${activate_script}"

    "${venv_python}" -c \
        'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 12), "Python 3.10 or 3.11 is required"'
    # Let pip repair only missing/unsatisfied requirements. Do not upgrade or
    # otherwise refresh packages which already satisfy the project.
    "${venv_python}" -m pip install \
        "setuptools>=69" wheel "pybind11>=2.12"

    package_spec="${repo_root}"
    if [[ -n "${python_extras}" ]]; then
        package_spec="${repo_root}[${python_extras}]"
    fi
    "${venv_python}" -m pip install --no-build-isolation --editable "${package_spec}"
    "${venv_python}" -c \
        'from dkfz.live import feature_backend_name; assert feature_backend_name() == "native"; print("Python feature backend: native")'
fi

native_build_args=(
    --tensorrt "${tensorrt_mode}"
    --build-dir "${build_dir}"
)
bash "${repo_root}/run/build_native.sh" "${native_build_args[@]}"

if [[ "${profile}" == "jetson" ]]; then
    if ! command -v systemctl >/dev/null 2>&1; then
        echo "Required command not found for Jetson systemd reload: systemctl" >&2
        exit 1
    fi
    echo "Reloading systemd units for the Jetson profile..."
    if [[ "${EUID}" -eq 0 ]]; then
        systemctl daemon-reload
    else
        if ! command -v sudo >/dev/null 2>&1; then
            echo "Required command not found for privileged systemd reload: sudo" >&2
            exit 1
        fi
        sudo systemctl daemon-reload
    fi
fi

printf 'Setup complete: profile=%s python=%s TensorRT=%s\n' \
    "${profile}" "${with_python}" "${tensorrt_mode}"
printf 'For subsequent C++ edits run: bash run/build_native.sh\n'
