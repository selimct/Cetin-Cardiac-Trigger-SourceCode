#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -euo pipefail

MODELNAME=""
VARIANT="best"
MODEL_DIR=""
ONNX_ARG=""
ENGINE_ARG=""
PRECISION="fp16"
POSITIONAL_PATHS=()

usage() {
    cat <<'EOF'
Usage: bash run/jetson_build_trt_engine.sh MODEL_NAME [OPTIONS]
       bash run/jetson_build_trt_engine.sh MODEL_NAME [best|tuned] [ONNX] [ENGINE]

Options:
  --variant best|tuned  Artifact variant. Default: best.
  --model-dir DIR       One source artifact root. Default: models/MODEL_NAME.
  --onnx PATH           Explicit ONNX input.
  --engine PATH         Explicit TensorRT engine output.
  --precision MODE      Engine compute precision: fp16 or fp32. Default: fp16.
                        fp32 also disables TF32 for strict parity fallback.
  -h, --help            Show this help.

All defaults are resolved within --model-dir, preventing same-named models in
different parent directories from being mixed implicitly.
EOF
}

die() {
    printf 'jetson_build_trt_engine.sh: %s\n' "$*" >&2
    exit 2
}

require_value() {
    [[ -n "${2:-}" ]] || die "$1 requires a value"
}

while (($#)); do
    case "$1" in
        --variant)
            require_value "$1" "${2:-}"
            VARIANT="$2"
            shift 2
            ;;
        --model-dir)
            require_value "$1" "${2:-}"
            MODEL_DIR="$2"
            shift 2
            ;;
        --onnx)
            require_value "$1" "${2:-}"
            ONNX_ARG="$2"
            shift 2
            ;;
        --engine)
            require_value "$1" "${2:-}"
            ENGINE_ARG="$2"
            shift 2
            ;;
        --precision)
            require_value "$1" "${2:-}"
            PRECISION="${2,,}"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        best|tuned)
            [[ -n "${MODELNAME}" ]] || die "MODEL_NAME must precede the variant"
            VARIANT="$1"
            shift
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            if [[ -z "${MODELNAME}" ]]; then
                MODELNAME="$1"
            else
                POSITIONAL_PATHS+=("$1")
            fi
            shift
            ;;
    esac
done

[[ -n "${MODELNAME}" ]] || die "MODEL_NAME is required (see --help)"
[[ "${MODELNAME}" =~ ^[A-Za-z0-9._-]+$ ]] || die \
    "MODEL_NAME may contain only letters, digits, dots, underscores, and hyphens"
[[ "${#POSITIONAL_PATHS[@]}" -le 2 ]] || die "too many positional paths"
if [[ "${#POSITIONAL_PATHS[@]}" -ge 1 ]]; then
    [[ -z "${ONNX_ARG}" ]] || die "ONNX path was supplied twice"
    ONNX_ARG="${POSITIONAL_PATHS[0]}"
fi
if [[ "${#POSITIONAL_PATHS[@]}" -eq 2 ]]; then
    [[ -z "${ENGINE_ARG}" ]] || die "engine path was supplied twice"
    ENGINE_ARG="${POSITIONAL_PATHS[1]}"
fi

MODEL_DIR="${MODEL_DIR:-models/${MODELNAME}}"
[[ -d "${MODEL_DIR}" ]] || die "source model directory not found: ${MODEL_DIR}"

case "${PRECISION}" in
    fp16|fp32) ;;
    *) die "--precision must be fp16 or fp32" ;;
esac

case "${VARIANT}" in
    best)
        ONNX_DEFAULT="${MODEL_DIR}/${MODELNAME}.onnx"
        ENGINE_DEFAULT="${MODEL_DIR}/${MODELNAME}_${PRECISION}.engine"
        ;;
    tuned)
        ONNX_DEFAULT="${MODEL_DIR}/${MODELNAME}_best_tuned.onnx"
        ENGINE_DEFAULT="${MODEL_DIR}/${MODELNAME}_best_tuned_${PRECISION}.engine"
        ;;
    *)
        echo "Unknown model variant: ${VARIANT}. Expected best or tuned." >&2
        exit 2
        ;;
esac

ONNX_PATH="${ONNX_ARG:-${ONNX_DEFAULT}}"
ENGINE_PATH="${ENGINE_ARG:-${ENGINE_DEFAULT}}"
INPUT_SHAPE="${INPUT_SHAPE:-}"
DYNAMIC_BATCH="${DYNAMIC_BATCH:-0}"
INPUT_NAME="${INPUT_NAME:-x}"
TRTEXEC_BIN="${TRTEXEC:-trtexec}"

[[ -f "${ONNX_PATH}" ]] || die "ONNX model not found: ${ONNX_PATH}"
MODEL_DIR_ABS="$(realpath -e -- "${MODEL_DIR}")"
ONNX_PATH_ABS="$(realpath -e -- "${ONNX_PATH}")"
case "${ONNX_PATH_ABS}" in
    "${MODEL_DIR_ABS}/"*) ;;
    *)
        die "ONNX model must be inside --model-dir ${MODEL_DIR}; got ${ONNX_PATH}"
        ;;
esac

if ! command -v "${TRTEXEC_BIN}" >/dev/null 2>&1; then
    if [ -x /usr/src/tensorrt/bin/trtexec ]; then
        TRTEXEC_BIN=/usr/src/tensorrt/bin/trtexec
    else
        echo "Could not find trtexec. Set TRTEXEC=/path/to/trtexec." >&2
        exit 1
    fi
fi

mkdir -p "$(dirname "${ENGINE_PATH}")"

TRTEXEC_ARGS=(
    "--onnx=${ONNX_PATH}"
    "--saveEngine=${ENGINE_PATH}"
)
if [[ "${PRECISION}" == "fp16" ]]; then
    TRTEXEC_ARGS+=("--fp16")
else
    # TensorRT otherwise permits TF32 tactics for an ostensibly FP32 build.
    # The fallback engine exists to maximize numerical agreement with PyTorch.
    TRTEXEC_ARGS+=("--noTF32")
fi
SHAPE_DISPLAY="static shape from ONNX"

if [ "${DYNAMIC_BATCH}" = "1" ]; then
    WINDOW_SIZE="${WINDOW_SIZE:-}"
    INPUT_DIM="${INPUT_DIM:-}"
    MIN_BATCH="${MIN_BATCH:-1}"
    OPT_BATCH="${OPT_BATCH:-128}"
    MAX_BATCH="${MAX_BATCH:-512}"
    if [ -z "${WINDOW_SIZE}" ] || [ -z "${INPUT_DIM}" ]; then
        echo "DYNAMIC_BATCH=1 requires WINDOW_SIZE and INPUT_DIM." >&2
        exit 2
    fi
    TRTEXEC_ARGS+=(
        "--minShapes=${INPUT_NAME}:${MIN_BATCH}x${WINDOW_SIZE}x${INPUT_DIM}"
        "--optShapes=${INPUT_NAME}:${OPT_BATCH}x${WINDOW_SIZE}x${INPUT_DIM}"
        "--maxShapes=${INPUT_NAME}:${MAX_BATCH}x${WINDOW_SIZE}x${INPUT_DIM}"
    )
    SHAPE_DISPLAY="${INPUT_NAME}: min ${MIN_BATCH}x${WINDOW_SIZE}x${INPUT_DIM}, opt ${OPT_BATCH}x${WINDOW_SIZE}x${INPUT_DIM}, max ${MAX_BATCH}x${WINDOW_SIZE}x${INPUT_DIM}"
elif [ -n "${INPUT_SHAPE}" ]; then
    TRTEXEC_ARGS+=("--shapes=${INPUT_SHAPE}")
    SHAPE_DISPLAY="${INPUT_SHAPE}"
fi

"${TRTEXEC_BIN}" "${TRTEXEC_ARGS[@]}" ${EXTRA_TRTEXEC_ARGS:-}

ENGINE_DISPLAY="$(realpath --relative-to="$(pwd)" "${ENGINE_PATH}")"
echo "Wrote TensorRT engine to ${ENGINE_DISPLAY}"
echo "Model variant: ${VARIANT}"
echo "Compute precision: ${PRECISION}"
echo "Input shape mode: ${SHAPE_DISPLAY}"
