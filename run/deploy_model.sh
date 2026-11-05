#!/usr/bin/env bash
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${repo_root}"

model_name=""
variant="best"
model_dir=""
config_path=""
onnx_path=""
onnx_explicit=0
engine_source=""
checkpoint_path=""
prepped_file=""
precision="auto"
python_bin="${MODEL_DEPLOY_PYTHON:-/home/orin/envs/jet/bin/python}"
build_dir="${repo_root}/build_native"
deploy_dir="${MODEL_DEPLOY_DIR:-${repo_root}/deploy/model}"
service_name="dkfz-native-uart-live.service"
dry_run=0
stage_dir=""

usage() {
    cat <<'USAGE'
Usage: bash run/deploy_model.sh MODEL_NAME [options]

Build and activate one native TensorRT model in deploy/model. The service must
be stopped. Artifacts are staged and validated before model.engine and
model.conf replace the active pair.

Options:
  --variant best|tuned  Model variant. Default: best.
  --model-dir DIR       Exact source artifact directory. Default:
                        models/MODEL_NAME.
  --config PATH         Training configuration JSON.
  --onnx PATH           ONNX input used to build a Jetson TensorRT engine.
  --engine PATH         Use an existing Jetson engine instead of building one.
  --checkpoint PATH     Check staged TensorRT logits against this checkpoint.
  --prepped-file PATH   Use saved feature windows for TensorRT parity.
  --precision MODE      Engine precision: auto, fp16, or fp32. Default: auto.
                        auto retries a failed FP16 parity check in strict FP32.
  --python PATH         Manifest-export Python. Default:
                        /home/orin/envs/jet/bin/python.
  --build-dir DIR       Native build directory. Default: build_native.
  --dry-run             Print the resolved commands without changing files.
  --help                Show this help.

Every source default is resolved under the same --model-dir. Config, ONNX, and
an existing engine must remain within that directory. The config's modelname
must also match MODEL_NAME.

Examples:
  sudo systemctl stop dkfz-native-uart-live.service
  bash run/deploy_model.sh target_r_v2
  bash run/deploy_model.sh target_r --model-dir models_noCV/target_r
  sudo systemctl restart dkfz-native-uart-live.service
USAGE
}

die() {
    printf 'deploy_model.sh: %s\n' "$*" >&2
    exit 2
}

require_value() {
    local option="$1"
    local value="${2:-}"
    [[ -n "${value}" ]] || die "${option} requires a value"
}

cleanup() {
    if [[ -n "${stage_dir}" && -d "${stage_dir}" ]]; then
        case "${stage_dir}" in
            "${repo_root}/deploy/".model-stage.*)
                rm -rf -- "${stage_dir}"
                ;;
            *)
                printf 'Refusing to remove unexpected staging path: %s\n' \
                    "${stage_dir}" >&2
                ;;
        esac
    fi
}
trap cleanup EXIT

run_command() {
    printf '  '
    printf '%q ' "$@"
    printf '\n'
    if [[ "${dry_run}" -eq 0 ]]; then
        "$@"
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --variant)
            require_value "$1" "${2:-}"
            variant="$2"
            shift 2
            ;;
        --model-dir)
            require_value "$1" "${2:-}"
            model_dir="$2"
            shift 2
            ;;
        --config)
            require_value "$1" "${2:-}"
            config_path="$2"
            shift 2
            ;;
        --onnx)
            require_value "$1" "${2:-}"
            onnx_path="$2"
            onnx_explicit=1
            shift 2
            ;;
        --engine)
            require_value "$1" "${2:-}"
            engine_source="$2"
            shift 2
            ;;
        --checkpoint)
            require_value "$1" "${2:-}"
            checkpoint_path="$2"
            shift 2
            ;;
        --prepped-file)
            require_value "$1" "${2:-}"
            prepped_file="$2"
            shift 2
            ;;
        --precision)
            require_value "$1" "${2:-}"
            precision="${2,,}"
            shift 2
            ;;
        --python)
            require_value "$1" "${2:-}"
            python_bin="$2"
            shift 2
            ;;
        --build-dir)
            require_value "$1" "${2:-}"
            build_dir="$2"
            shift 2
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            [[ -z "${model_name}" ]] || die "unexpected argument: $1"
            model_name="$1"
            shift
            ;;
    esac
done

[[ -n "${model_name}" ]] || die "MODEL_NAME is required (see --help)"
[[ "${model_name}" =~ ^[A-Za-z0-9._-]+$ ]] || die \
    "MODEL_NAME may contain only letters, digits, dots, underscores, and hyphens"
case "${precision}" in
    auto|fp16|fp32) ;;
    *) die "--precision must be auto, fp16, or fp32" ;;
esac

model_dir="${model_dir:-models/${model_name}}"
[[ -d "${model_dir}" ]] || die "source model directory not found: ${model_dir}"
model_dir_abs="$(realpath -e -- "${model_dir}")"

require_source_artifact() {
    local label="$1"
    local path="$2"
    [[ -f "${path}" ]] || die "${label} not found: ${path}"
    local resolved
    resolved="$(realpath -e -- "${path}")"
    case "${resolved}" in
        "${model_dir_abs}/"*) ;;
        *)
            die "${label} must be inside --model-dir ${model_dir}; got ${path}"
            ;;
    esac
}

case "${variant}" in
    best)
        default_config="${model_dir}/train_config.json"
        default_onnx="${model_dir}/${model_name}.onnx"
        ;;
    tuned)
        default_config="${model_dir}/tuning/tuned_config.json"
        default_onnx="${model_dir}/${model_name}_best_tuned.onnx"
        ;;
    *)
        die "--variant must be best or tuned"
        ;;
esac

config_path="${config_path:-${default_config}}"
onnx_path="${onnx_path:-${default_onnx}}"
runtime="${build_dir}/inference/native_uart_live_runtime"

[[ -x "${python_bin}" ]] || die "Python is not executable: ${python_bin}"
[[ -x "${runtime}" ]] || die "native runtime not found: ${runtime}"
require_source_artifact "training config" "${config_path}"
if [[ -n "${prepped_file}" && -z "${checkpoint_path}" ]]; then
    die "--prepped-file requires --checkpoint"
fi
if [[ -n "${checkpoint_path}" ]]; then
    require_source_artifact "PyTorch checkpoint" "${checkpoint_path}"
fi
if [[ -n "${prepped_file}" && ! -f "${prepped_file}" ]]; then
    die "preprocessed parity file not found: ${prepped_file}"
fi
configured_model_name="$(
    "${python_bin}" -c \
        'import json, sys; print(json.load(open(sys.argv[1], encoding="utf-8")).get("modelname", ""))' \
        "${config_path}"
)"
[[ "${configured_model_name}" == "${model_name}" ]] || die \
    "training config declares modelname=${configured_model_name:-<missing>}, expected ${model_name}"
if [[ -n "${engine_source}" ]]; then
    [[ "${onnx_explicit}" -eq 0 ]] || die \
        "--onnx and --engine are mutually exclusive"
    [[ "${precision}" == "auto" ]] || die \
        "--precision applies only when building from --onnx"
    require_source_artifact "TensorRT engine" "${engine_source}"
else
    require_source_artifact "ONNX model" "${onnx_path}"
fi

if command -v systemctl >/dev/null 2>&1; then
    service_state="$(systemctl is-active "${service_name}" 2>/dev/null || true)"
    case "${service_state}" in
        active|activating|reloading|deactivating)
            die "${service_name} is ${service_state}; stop it before replacing the active model"
            ;;
    esac
fi

if [[ "${dry_run}" -eq 1 ]]; then
    stage_dir="${repo_root}/deploy/.model-stage.DRYRUN"
else
    mkdir -p "${repo_root}/deploy"
    stage_dir="$(mktemp -d "${repo_root}/deploy/.model-stage.XXXXXX")"
fi
stage_engine="${stage_dir}/model.engine"
stage_manifest="${stage_dir}/model.conf"
engine_was_built=0
active_precision="existing"

build_stage_engine() {
    local build_precision="$1"
    run_command bash "${repo_root}/run/jetson_build_trt_engine.sh" \
        "${model_name}" \
        --variant "${variant}" \
        --model-dir "${model_dir}" \
        --onnx "${onnx_path}" \
        --engine "${stage_engine}" \
        --precision "${build_precision}"
    if [[ "${dry_run}" -eq 0 ]]; then
        chmod 0644 "${stage_engine}"
    fi
    active_precision="${build_precision}"
}

printf 'Preparing model=%s variant=%s source=%s\n' \
    "${model_name}" "${variant}" "${model_dir}"
if [[ -n "${engine_source}" ]]; then
    printf 'Staging existing Jetson engine\n'
    run_command install -m 0644 "${engine_source}" "${stage_engine}"
else
    engine_was_built=1
    initial_precision="${precision}"
    if [[ "${initial_precision}" == "auto" ]]; then
        initial_precision="fp16"
    fi
    printf 'Building %s TensorRT engine on this Jetson\n' "${initial_precision}"
    build_stage_engine "${initial_precision}"
fi

if [[ -n "${checkpoint_path}" ]]; then
    printf 'Checking staged TensorRT engine against its checkpoint\n'
    parity_command=(
        "${python_bin}"
        "${repo_root}/run/jetson_check_trt_parity.py"
        --modelname "${model_name}"
        --model-variant "${variant}"
        --model-dir "${model_dir}"
        --allow-engine-outside-model-dir
        --config "${config_path}"
        --checkpoint "${checkpoint_path}"
        --engine "${stage_engine}"
    )
    if [[ -n "${prepped_file}" ]]; then
        parity_command+=(--prepped-file "${prepped_file}")
    fi
    if run_command "${parity_command[@]}"; then
        parity_status=0
    else
        parity_status=$?
        if [[ "${parity_status}" -eq 1 && \
              "${engine_was_built}" -eq 1 && "${precision}" == "auto" && \
              "${active_precision}" == "fp16" ]]; then
            printf 'FP16 parity failed; rebuilding in strict FP32 and retrying\n'
            build_stage_engine fp32
            run_command "${parity_command[@]}"
        else
            printf 'TensorRT validation failed for %s engine (exit %d)\n' \
                "${active_precision}" "${parity_status}" >&2
            exit "${parity_status}"
        fi
    fi
fi

# Generate the manifest only after the final engine has passed parity, since an
# automatic FP32 rebuild changes both the engine size and its fingerprint.
printf 'Generating fingerprinted native manifest\n'
run_command "${python_bin}" \
    "${repo_root}/run/jetson_export_native_manifest.py" \
    --modelname "${model_name}" \
    --model-variant "${variant}" \
    --model-dir "${model_dir}" \
    --allow-engine-outside-model-dir \
    --config "${config_path}" \
    --engine "${stage_engine}" \
    --output "${stage_manifest}"

printf 'Validating staged manifest and TensorRT engine\n'
run_command "${runtime}" \
    --config "${repo_root}/config/native_uart_live.conf" \
    --manifest "${stage_manifest}" \
    --validate-only \
    --check-engine

printf 'Activating deploy/model\n'
if [[ "${dry_run}" -eq 1 ]]; then
    run_command install -d -m 0755 "${deploy_dir}"
    run_command install -m 0644 "${stage_engine}" "${deploy_dir}/model.engine"
    run_command install -m 0644 "${stage_manifest}" "${deploy_dir}/model.conf"
else
    install -d -m 0755 "${deploy_dir}"
    install -m 0644 "${stage_engine}" "${deploy_dir}/model.engine"
    # Install the fingerprinted manifest last so an interrupted activation
    # cannot expose a new manifest with an old engine.
    install -m 0644 "${stage_manifest}" "${deploy_dir}/model.conf"
fi

printf '\nActive model: %s (%s)\n' "${model_name}" "${variant}"
printf 'Precision:    %s\n' "${active_precision}"
printf 'Manifest: deploy/model/model.conf\n'
printf 'Engine:   deploy/model/model.engine\n'
printf 'Start or restart with:\n'
printf '  sudo systemctl restart %s\n' "${service_name}"
