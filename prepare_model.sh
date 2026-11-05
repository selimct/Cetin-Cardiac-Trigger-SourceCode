#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -Eeuo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${MODEL_READY_PYTHON:-}"
MODEL_NAME=""
MODEL_VARIANT="best"
PHASE="all"
SKIP_ONNX_PARITY=0
SKIP_TRT_PARITY=0
DRY_RUN=0
CONFIG_PATH=""
CHECKPOINT_PATH=""
MODEL_DIR=""
ONNX_PATH=""
ENGINE_PATH=""
PREPPED_FILE=""
CURRENT_STEP="initialization"
STEP_NUMBER=0
TOTAL_STEPS=0

usage() {
    cat <<'EOF'
Prepare an existing PyTorch checkpoint for native TensorRT deployment.

Usage:
  ./prepare_model.sh MODEL_NAME [OPTIONS]

The script starts with an existing .pth checkpoint and training config. By
default it runs the complete model-artifact pipeline:
  1. export ONNX
  2. check PyTorch/ONNX parity
  3. stage, build/check TensorRT, and activate deploy/model

By default every source artifact is resolved under models/MODEL_NAME. Use one
explicit --model-dir for alternate roots such as models_noCV/MODEL_NAME; source
artifacts from different roots are rejected.

Options:
  --variant best|tuned   Artifact variant (default: best).
  --model-dir DIR        Exact source artifact directory. Defaults to
                         models/MODEL_NAME.
  --checkpoint PATH     Existing .pth checkpoint; overrides the variant default.
  --config PATH         Training config JSON; overrides the variant default.
  --onnx PATH           ONNX artifact path within --model-dir.
  --engine PATH         Existing TensorRT engine within --model-dir. When
                         omitted, build only in deployment staging.
  --workstation-only    Run ONNX export and ONNX parity only.
  --jetson-only         Build/check TensorRT and activate deploy/model only.
  --skip-onnx-parity    Do not run the PyTorch/ONNX comparison.
  --skip-trt-parity     Do not run the PyTorch/TensorRT comparison. Useful when
                         the checkpoint or PyTorch is not copied to the Jetson.
  --prepped-file PATH    Use real saved feature windows for parity checks.
                         If omitted, MODEL_DIR/prepped.parquet is used
                         when present; otherwise the checkers use synthetic data.
  --python PATH          Override the Python interpreter. By default the full
                         and Jetson phases use the environment activated by the
                         pythonbase alias from ~/.bash_aliases, as configured by
                         setup.sh --profile jetson. Workstation-only uses .venv.
                         MODEL_READY_PYTHON provides the same override via env.
  --dry-run              Print the commands without running them.
  -h, --help             Show this help.

Examples:
  ./prepare_model.sh target_r --workstation-only
  ./prepare_model.sh target_r --model-dir models_noCV/target_r --jetson-only
  ./prepare_model.sh target_r --model-dir output/model_run
  ./prepare_model.sh tuned_model --variant tuned --jetson-only
EOF
}

die() {
    printf 'prepare_model.sh: %s\n' "$*" >&2
    exit 2
}

require_option_value() {
    local option="$1"
    local value="${2:-}"
    if [[ -z "${value}" ]]; then
        die "${option} requires a value"
    fi
}

resolve_python_environment() {
    if [[ -n "${PYTHON_BIN}" ]]; then
        return
    fi
    if [[ "${PHASE}" == "workstation" ]]; then
        PYTHON_BIN=".venv/bin/python"
        return
    fi

    local bash_aliases="${HOME:-}/.bash_aliases"
    if [[ ! -r "${bash_aliases}" ]]; then
        die "Jetson Python environment configuration not found: ${bash_aliases}; run setup.sh --profile jetson or use --python"
    fi

    shopt -s expand_aliases
    # shellcheck disable=SC1090
    source "${bash_aliases}"
    if ! alias pythonbase >/dev/null 2>&1; then
        die "Jetson Python environment alias not found: pythonbase; run setup.sh --profile jetson or use --python"
    fi

    local pythonbase_definition pythonbase_command activate_script
    local -a pythonbase_parts=()
    pythonbase_definition="$(alias pythonbase)"
    eval "pythonbase_command=${pythonbase_definition#*=}"
    eval "pythonbase_parts=(${pythonbase_command})"
    if [[ "${#pythonbase_parts[@]}" -ne 2 ]] || \
            [[ "${pythonbase_parts[0]}" != "source" && "${pythonbase_parts[0]}" != "." ]] || \
            [[ "${pythonbase_parts[1]}" != */bin/activate ]]; then
        die "the pythonbase alias must source a virtual-environment bin/activate script"
    fi

    activate_script="${pythonbase_parts[1]}"
    if [[ ! -r "${activate_script}" ]]; then
        die "Jetson Python environment activation script not found: ${activate_script}"
    fi
    # shellcheck disable=SC1090
    source "${activate_script}"
    PYTHON_BIN="${activate_script%/bin/activate}/bin/python"
}

while (($#)); do
    case "$1" in
        --variant)
            require_option_value "$1" "${2:-}"
            MODEL_VARIANT="$2"
            shift 2
            ;;
        --model-dir)
            require_option_value "$1" "${2:-}"
            MODEL_DIR="$2"
            shift 2
            ;;
        --checkpoint)
            require_option_value "$1" "${2:-}"
            CHECKPOINT_PATH="$2"
            shift 2
            ;;
        --config)
            require_option_value "$1" "${2:-}"
            CONFIG_PATH="$2"
            shift 2
            ;;
        --onnx)
            require_option_value "$1" "${2:-}"
            ONNX_PATH="$2"
            shift 2
            ;;
        --engine)
            require_option_value "$1" "${2:-}"
            ENGINE_PATH="$2"
            shift 2
            ;;
        --workstation-only)
            [[ "${PHASE}" != "jetson" ]] || die \
                "--workstation-only and --jetson-only are mutually exclusive"
            PHASE="workstation"
            shift
            ;;
        --jetson-only)
            [[ "${PHASE}" != "workstation" ]] || die \
                "--workstation-only and --jetson-only are mutually exclusive"
            PHASE="jetson"
            shift
            ;;
        --skip-onnx-parity)
            SKIP_ONNX_PARITY=1
            shift
            ;;
        --skip-trt-parity)
            SKIP_TRT_PARITY=1
            shift
            ;;
        --prepped-file)
            require_option_value "$1" "${2:-}"
            PREPPED_FILE="$2"
            shift 2
            ;;
        --python)
            require_option_value "$1" "${2:-}"
            PYTHON_BIN="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [[ -n "${MODEL_NAME}" ]]; then
                die "unexpected argument: $1"
            fi
            MODEL_NAME="$1"
            shift
            ;;
    esac
done

[[ -n "${MODEL_NAME}" ]] || die "MODEL_NAME is required (see --help)"
[[ "${MODEL_NAME}" =~ ^[A-Za-z0-9._-]+$ ]] || die \
    "MODEL_NAME may contain only letters, digits, dots, underscores, and hyphens"

DEFAULT_MODEL_DIR="models/${MODEL_NAME}"
MODEL_DIR="${MODEL_DIR:-${DEFAULT_MODEL_DIR}}"
MODEL_DIR_ABS="$(realpath -m -- "${MODEL_DIR}")"

require_source_path() {
    local label="$1"
    local path="$2"
    local must_exist="${3:-1}"
    local resolved
    resolved="$(realpath -m -- "${path}")"
    case "${resolved}" in
        "${MODEL_DIR_ABS}/"*) ;;
        *)
            die "${label} must be inside --model-dir ${MODEL_DIR}; got ${path}"
            ;;
    esac
    if [[ "${DRY_RUN}" -eq 0 && "${must_exist}" -eq 1 && ! -f "${path}" ]]; then
        die "${label} not found: ${path}"
    fi
}

case "${MODEL_VARIANT}" in
    best)
        DEFAULT_CONFIG="${MODEL_DIR}/train_config.json"
        DEFAULT_CHECKPOINT="${MODEL_DIR}/best.pth"
        DEFAULT_ONNX="${MODEL_DIR}/${MODEL_NAME}.onnx"
        ;;
    tuned)
        DEFAULT_CONFIG="${MODEL_DIR}/tuning/tuned_config.json"
        DEFAULT_CHECKPOINT="${MODEL_DIR}/best_tuned.pth"
        DEFAULT_ONNX="${MODEL_DIR}/${MODEL_NAME}_best_tuned.onnx"
        ;;
    *)
        die "--variant must be best or tuned"
        ;;
esac

CONFIG_PATH="${CONFIG_PATH:-${DEFAULT_CONFIG}}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-${DEFAULT_CHECKPOINT}}"
ONNX_PATH="${ONNX_PATH:-${DEFAULT_ONNX}}"
resolve_python_environment

require_source_path "training config" "${CONFIG_PATH}"
require_source_path "checkpoint" "${CHECKPOINT_PATH}" 0
require_source_path "ONNX artifact" "${ONNX_PATH}" 0
if [[ -n "${ENGINE_PATH}" ]]; then
    require_source_path "TensorRT engine" "${ENGINE_PATH}"
fi

if [[ "${DRY_RUN}" -eq 0 ]]; then
    [[ -x "${PYTHON_BIN}" ]] || die \
        "Python interpreter is not executable: ${PYTHON_BIN} (use --python)"
    if [[ "${PHASE}" != "jetson" || "${SKIP_TRT_PARITY}" -eq 0 ]]; then
        require_source_path "checkpoint" "${CHECKPOINT_PATH}"
    fi
    CONFIGURED_MODEL_NAME="$(
        "${PYTHON_BIN}" -c \
            'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("modelname", ""))' \
            "${CONFIG_PATH}"
    )"
    [[ "${CONFIGURED_MODEL_NAME}" == "${MODEL_NAME}" ]] || die \
        "training config declares modelname=${CONFIGURED_MODEL_NAME:-<missing>}, expected ${MODEL_NAME}"
fi

if [[ "${PHASE}" != "jetson" ]]; then
    ((TOTAL_STEPS += 1))
    if [[ "${SKIP_ONNX_PARITY}" -eq 0 ]]; then
        ((TOTAL_STEPS += 1))
    fi
fi
if [[ "${PHASE}" != "workstation" ]]; then
    ((TOTAL_STEPS += 1))
fi

on_error() {
    local status="$1"
    printf '\nFAILED: %s (exit %s)\n' "${CURRENT_STEP}" "${status}" >&2
    exit "${status}"
}
trap 'on_error "$?"' ERR

run_step() {
    local description="$1"
    shift
    ((STEP_NUMBER += 1))
    CURRENT_STEP="${description}"
    printf '\n[%d/%d] %s\n' "${STEP_NUMBER}" "${TOTAL_STEPS}" "${description}"
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    if [[ "${DRY_RUN}" -eq 0 ]]; then
        "$@"
    fi
}

resolve_parity_args() {
    parity_args=()
    if [[ -n "${PREPPED_FILE}" ]]; then
        if [[ "${DRY_RUN}" -eq 0 && ! -f "${PREPPED_FILE}" ]]; then
            die "preprocessed parity file not found: ${PREPPED_FILE}"
        fi
        parity_args=(--prepped-file "${PREPPED_FILE}")
    elif [[ -f "${MODEL_DIR}/prepped.parquet" ]]; then
        parity_args=(--prepped-file "${MODEL_DIR}/prepped.parquet")
    fi
}

if [[ "${PHASE}" != "jetson" ]]; then
    run_step "Export ${MODEL_VARIANT} checkpoint to ONNX" \
        "${PYTHON_BIN}" run/jetson_export_onnx.py \
        --modelname "${MODEL_NAME}" \
        --model-variant "${MODEL_VARIANT}" \
        --model-dir "${MODEL_DIR}" \
        --config "${CONFIG_PATH}" \
        --checkpoint "${CHECKPOINT_PATH}" \
        --output "${ONNX_PATH}"

    if [[ "${SKIP_ONNX_PARITY}" -eq 0 ]]; then
        resolve_parity_args
        run_step "Check PyTorch/ONNX parity" \
            "${PYTHON_BIN}" run/jetson_check_onnx_parity.py \
            --modelname "${MODEL_NAME}" \
            --model-variant "${MODEL_VARIANT}" \
            --model-dir "${MODEL_DIR}" \
            --config "${CONFIG_PATH}" \
            --checkpoint "${CHECKPOINT_PATH}" \
            --onnx "${ONNX_PATH}" \
            "${parity_args[@]}"
    fi
fi

if [[ "${PHASE}" != "workstation" ]]; then
    deploy_args=(
        bash
        run/deploy_model.sh
        "${MODEL_NAME}"
        --variant "${MODEL_VARIANT}"
        --model-dir "${MODEL_DIR}"
        --config "${CONFIG_PATH}"
        --python "${PYTHON_BIN}"
    )
    if [[ -n "${ENGINE_PATH}" ]]; then
        deploy_args+=(--engine "${ENGINE_PATH}")
    else
        deploy_args+=(--onnx "${ONNX_PATH}")
    fi
    if [[ "${SKIP_TRT_PARITY}" -eq 0 ]]; then
        resolve_parity_args
        deploy_args+=(--checkpoint "${CHECKPOINT_PATH}")
        deploy_args+=("${parity_args[@]}")
    fi
    run_step "Build, check, and activate stable deployment slot" \
        "${deploy_args[@]}"
fi

printf '\nModel artifact pipeline completed for %s (%s).\n' \
    "${MODEL_NAME}" "${MODEL_VARIANT}"
if [[ "${PHASE}" == "workstation" ]]; then
    printf 'Copy the model config, ONNX file, checkpoint, and optional prepped data to the Jetson, then run:\n'
    printf '  ./prepare_model.sh %q --variant %q --jetson-only' \
        "${MODEL_NAME}" "${MODEL_VARIANT}"
    if [[ "${MODEL_DIR}" != "${DEFAULT_MODEL_DIR}" ]]; then
        printf ' --model-dir %q' "${MODEL_DIR}"
    fi
    if [[ "${CONFIG_PATH}" != "${DEFAULT_CONFIG}" ]]; then
        printf ' --config %q' "${CONFIG_PATH}"
    fi
    if [[ "${CHECKPOINT_PATH}" != "${DEFAULT_CHECKPOINT}" ]]; then
        printf ' --checkpoint %q' "${CHECKPOINT_PATH}"
    fi
    if [[ "${ONNX_PATH}" != "${DEFAULT_ONNX}" ]]; then
        printf ' --onnx %q' "${ONNX_PATH}"
    fi
    if [[ -n "${ENGINE_PATH}" ]]; then
        printf ' --engine %q' "${ENGINE_PATH}"
    fi
    if [[ -n "${PREPPED_FILE}" ]]; then
        printf ' --prepped-file %q' "${PREPPED_FILE}"
    fi
    printf '\n'
elif [[ "${PHASE}" == "jetson" ]]; then
    printf 'The TensorRT engine and stable deploy/model slot are ready.\n'
else
    printf 'The ONNX model, TensorRT engine, and stable deploy/model slot are ready.\n'
fi
