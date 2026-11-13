"""Check export completeness and the public GUI's backend file interfaces.

No recordings, models, training runs, inference, SSH, systemd changes, or hardware
are needed. Application behavior remains covered by the development tests.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MPLBACKEND"] = "Agg"

PYTHON_COMMANDS = (
    "run/train.py",
    "run/train_lopo.py",
    "run/offline_predict.py",
    "run/prep_window_annotate.py",
    "run/prep_rotate_ecg.py",
    "run/jetson_export_onnx.py",
    "run/jetson_check_onnx_parity.py",
    "run/jetson_export_native_manifest.py",
    "run/create_train_metrics.py",
    "run/create_prediction_plots.py",
    "run/create_prediction_metrics_overview.py",
    "run/post_reselect_thresholds.py",
    "run/post_backfill_thresholds.py",
    "run/train_finetune.py",
    "run/train_recovery_lopo.py",
    "run/train_lopo_hpo.py",
    "run/calibrate_recovery.py",
    "run/prep_read_log_to_parquet.py",
    "run/io_uart_bytes_to_parquet.py",
    "run/create_model_table.py",
    "run/compare_prediction_models.py",
    "run/offline_predict_trt.py",
    "run/check_prepared_model.py",
    "run/jetson_check_native_runtime_parity.py",
    "run/jetson_prepare_model.py",
    "run/jetson_paced_uart_check.py",
    "run/jetson_qualify_model.py",
    "run/set_live_threshold.py",
    "run/summarize_inference_latency.py",
)
SHELL_COMMANDS = (
    "setup.sh",
    "prepare_model.sh",
    "run/build_native.sh",
    "run/rebuild_native.sh",
    "run/deploy_model.sh",
    "run/jetson_build_trt_engine.sh",
)
GENERATED_MODEL_FILES = {"deploy/model/model.conf", "deploy/model/model.engine"}
GENERATED_MODEL_DIRECTORIES = ("build_native/prepared_models/",)
GUI_PATH = re.compile(
    r"(?<![\w./-])(?:run/[\w./-]+\.(?:sh|py)|prepare_model\.sh|"
    r"build_native/[\w./-]+|config/[\w./-]+\.conf|"
    r"deploy/[\w./-]+\.(?:service|conf|engine))"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def command(*arguments: str) -> str:
    result = subprocess.run(arguments, cwd=ROOT, text=True, capture_output=True, timeout=60)
    require(result.returncode == 0,
            f"Command could not start: {' '.join(arguments)}\n{result.stdout}{result.stderr}")
    return result.stdout + result.stderr


def require_file(relative: str, *, executable: bool = False) -> None:
    require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
            f"Expected a path inside the public checkout: {relative}")
    path = ROOT / relative
    require(path.is_file(), f"Missing public file: {relative}")
    if executable:
        require(os.access(path, os.X_OK), f"GUI command is not executable: {relative}")


def check_python_sources() -> None:
    """Check local imports even in TensorRT-specific modules on a CPU runner."""
    source_files = [*sorted((ROOT / "src/dkfz").rglob("*.py")),
                    *sorted((ROOT / "run").glob("*.py"))]
    for path in source_files:
        tree = ast.parse(path.read_text(), filename=str(path))
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                dependencies.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                dependencies.add(node.module)
                dependencies.update(node.module + "." + alias.name for alias in node.names
                                    if alias.name == "_live_core")
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # Lazy imports name a package. ONNX metadata also uses the
                # dkfz prefix (e.g. dkfz.prediction_mode), but is not a module.
                if re.fullmatch(r"dkfz\.(?:core|live|native)(?:\.[A-Za-z_]\w*)*", node.value):
                    dependencies.add(node.value)
        for name in sorted(dependencies):
            if name != "dkfz" and not name.startswith("dkfz."):
                continue
            if name == "dkfz.native._live_core":
                require(importlib.util.find_spec(name) is not None, "Compiled Python extension is missing")
                continue
            module = ROOT / "src" / name.replace(".", "/")
            require(module.with_suffix(".py").is_file() or (module / "__init__.py").is_file(),
                    f"{path.relative_to(ROOT)} imports missing public module {name}")
    from dkfz.live import feature_backend_name
    from dkfz.native import _live_core
    require(feature_backend_name() == "native", "Compiled feature backend cannot be loaded")
    require(Path(_live_core.__file__).resolve().is_relative_to(ROOT / "src"),
            "The compiled extension was loaded from another checkout")
    print(f"PASS: {len(source_files)} Python files parse and their local module dependencies are included", flush=True)


def check_gui_files(gui_source: Path) -> None:
    """Read paths from the actual public GUI source rather than a copied list."""
    require((gui_source / "src/diagnostic_command_builder.cpp").is_file(),
            f"Public GUI command builder was not found in {gui_source}")
    references = set()
    for source in sorted((gui_source / "src").glob("*.cpp")):
        references.update(GUI_PATH.findall(source.read_text()))
    require({"run/build_native.sh", "run/rebuild_native.sh", "prepare_model.sh",
             "config/native_uart_live.conf", "deploy/dkfz-native-uart-live.service",
             "build_native/inference/native_uart_live_runtime"}.issubset(references),
            "GUI backend layout changed; review its file interfaces before publishing")
    for relative in sorted(references - GENERATED_MODEL_FILES):
        if relative.startswith(GENERATED_MODEL_DIRECTORIES):
            continue
        require_file(relative, executable=relative.endswith(".sh") or
                     relative == "build_native/inference/native_uart_live_runtime")
    print(f"PASS: backend file paths used by the public GUI are present ({len(references)} references)", flush=True)
    print("Model engines and manifests are generated when a user deploys a model; they are not source files.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gui-source", required=True, type=Path)
    args = parser.parse_args()
    for relative in (*PYTHON_COMMANDS, *SHELL_COMMANDS, "run/jetson_check_trt_parity.py",
                     "pyproject.toml", "setup.py", "LICENSE",
                     "config/native_uart_live.conf", "deploy/dkfz-native-uart-live.service",
                     "deploy/journald@dkfz-live.conf"):
        require_file(relative, executable=relative in {
            "prepare_model.sh", "run/deploy_model.sh", "run/build_native.sh", "run/rebuild_native.sh",
        })

    check_python_sources()
    cmake = (ROOT / "native/CMakeLists.txt").read_text()
    for relative in re.findall(r"\b(?:live_core|live_runtime|uart|python)/[\w./-]+\.(?:cpp|hpp)\b", cmake):
        require_file("native/" + relative)
    require_file("native/uart_receiver_stream.cpp")
    print("PASS: native build sources are included, including the TensorRT branch", flush=True)
    for relative in PYTHON_COMMANDS:
        command(sys.executable, str(ROOT / relative), "--help")
    for relative in SHELL_COMMANDS:
        command("bash", "-n", relative)
    shell_sources = [*ROOT.glob("*.sh"), *(ROOT / "run").glob("*.sh")]
    for source in shell_sources:
        for relative in re.findall(r"(?<![\w./-])run/[\w./-]+\.(?:py|sh)\b", source.read_text()):
            require_file(relative)
    print("PASS: exported Python commands start and shell-script dependencies are included", flush=True)

    check_gui_files(args.gui_source.resolve())
    interfaces = {
        "prepare_model.sh": ("--model-dir", "--variant", "--jetson-only"),
        "run/build_native.sh": ("--tensorrt", "--test"),
        "run/rebuild_native.sh": ("--target",),
    }
    for relative, flags in interfaces.items():
        output = command("bash", relative, "--help")
        for flag in flags:
            require(flag in output, f"GUI requires missing option {flag} in {relative}")
    output = command(str(ROOT / "build_native/inference/native_uart_live_runtime"), "--help")
    for flag in ("--config", "--manifest", "--trigger-output", "--capture-byte-file",
                 "--emit-probabilities", "--validate-only", "--check-engine"):
        require(flag in output, f"GUI requires missing native runtime option: {flag}")
    # The recovery helper is a service API without a --help mode. Checking its
    # executable is sufficient here; invoking it would query the host service.
    require_file("build_native/inference/native_live_recovery_control", executable=True)
    for relative in ("inference/native_live_runtime", "uart/uart_receiver_stream"):
        command(str(ROOT / "build_native" / relative), "--help")
    config = (ROOT / "config/native_uart_live.conf").read_text()
    require(re.search(r"^manifest_path=deploy/model/model\.conf$", config, re.MULTILINE) is not None,
            "Runtime config does not use the GUI's stable model slot")
    print("PASS: build, model-preparation, and native-runtime command options required by the GUI are available", flush=True)
    print("Distribution check passed. No training, prediction, application tests, or hardware operations were run.", flush=True)


if __name__ == "__main__":
    main()
