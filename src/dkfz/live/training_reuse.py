"""Identify equivalent completed LOPO runs before reusing their artifacts."""

from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path

from dkfz.live.io import REPO_ROOT
from dkfz.live.training_config import TRAINING_DEFAULTS

REQUEST_FILE = "training_request.json"
_COMPLETED_ARTIFACTS = (
    "best.pth",
    "train_config.json",
    "summary.json",
    "outer_results.csv",
)
_OPERATIONAL_SETTINGS = {
    "inputs",
    "threshold_inputs",
    "prepped_file",
    "use_prepped",
    "save_prepped",
    "save_last",
    "prep_workers",
    "debug_timing",
    "plots",
    "test_size",
}


def _digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def training_request(kwargs, patient_groups):
    """Fingerprint effective settings, recording contents and training code."""
    settings = {}
    for key, default in TRAINING_DEFAULTS.items():
        if key in _OPERATIONAL_SETTINGS:
            continue
        value = kwargs.get(key, default)
        if value is not None:
            if isinstance(default, bool):
                value = bool(value)
            elif isinstance(default, int):
                value = int(value)
            elif isinstance(default, float):
                value = float(value)
        settings[key] = value
    inputs = {
        patient: [
            {"path": str(Path(path).resolve()), "sha256": _digest(path)}
            for path in sorted(paths)
        ]
        for patient, paths in sorted(patient_groups.items())
    }
    code_paths = set((REPO_ROOT / "src/dkfz/core").glob("*.py"))
    code_paths.update((REPO_ROOT / "src/dkfz/native").glob("*.py"))
    for name in (
        "cross_validation",
        "training_config",
        "prediction_contract",
        "training_pipeline",
        "training_preprocessing",
        "training_evaluation",
        "training_seed",
        "training_reuse",
        "feature_framing",
        "event_matching",
        "eval_metrics",
        "recovery",
        "triggers",
        "io",
    ):
        code_paths.add(REPO_ROOT / "src/dkfz/live" / f"{name}.py")
    code_paths.add(REPO_ROOT / "run/train_lopo.py")
    for folder in ("native/live_core", "native/python"):
        for suffix in ("*.cpp", "*.hpp"):
            code_paths.update((REPO_ROOT / folder).rglob(suffix))
    code = {
        str(path.relative_to(REPO_ROOT)): _digest(path)
        for path in sorted(code_paths)
        if path.is_file()
    }
    dependencies = {}
    for package in ("torch", "numpy", "pandas", "pyarrow"):
        try:
            dependencies[package] = version(package)
        except PackageNotFoundError:
            dependencies[package] = None
    return dict(
        schema_version=1,
        training=settings,
        inputs=inputs,
        implementation=code,
        dependencies=dependencies,
    )


def _lock_path(output):
    output = Path(output)
    return output.parent / f".{output.name}.lock"


def check_run_available(output):
    """Probe an existing lock without creating files during a dry run."""
    path = _lock_path(output)
    if path.exists():
        with path.open("r") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise ValueError(f"Training is already running for {output}") from error


@contextmanager
def lock_training_run(output, *, inherited_fd=None):
    path = _lock_path(output)
    if inherited_fd is not None:
        # The wrapper retains this same open-file description through publishing.
        # Closing the Python copy must not explicitly unlock the parent's lock.
        actual = os.fstat(inherited_fd)
        expected = path.stat()
        if (actual.st_dev, actual.st_ino) != (expected.st_dev, expected.st_ino):
            raise ValueError(
                "Inherited training lock does not match the output directory"
            )
        try:
            fcntl.flock(inherited_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Training is already running for {output}") from error
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"Training is already running for {output}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def existing_run_status(output, request, *, force=False):
    output = Path(output)
    if force:
        return dict(action="train", reason="forced by --overwrite")
    if not output.exists() or not any(output.iterdir()):
        return dict(action="train", reason="new run")
    try:
        recorded = json.loads((output / REQUEST_FILE).read_text())
        if not isinstance(recorded, dict):
            raise ValueError("Invalid training signature")
    except (FileNotFoundError, ValueError):
        return dict(
            action="replace", reason="existing run has no verifiable training signature"
        )
    differences = []
    for key in request:
        if request[key] == recorded.get(key):
            continue
        if key == "training":
            old = recorded.get(key, {})
            if not isinstance(old, dict):
                old = {}
            fields = [
                field
                for field, value in request[key].items()
                if value != old.get(field)
            ]
            differences.append("training (" + ", ".join(fields) + ")")
        else:
            differences.append(key)
    if differences:
        return dict(action="replace", reason="changed " + ", ".join(differences))
    try:
        checkpoint = output / "best.pth"
        config = json.loads((output / "train_config.json").read_text())
        summary = json.loads((output / "summary.json").read_text())
        complete = (
            checkpoint.is_file()
            and checkpoint.stat().st_size > 0
            and (output / "outer_results.csv").is_file()
            and config.get("modelname") == request["training"]["modelname"]
            and summary.get("num_outer_folds") == len(request["inputs"])
            and summary["best_model"]["selected_epoch"]
            == config["selected_fixed_epoch"]
            and all(
                recorded.get("artifacts", {}).get(name) == _digest(output / name)
                for name in _COMPLETED_ARTIFACTS
            )
        )
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        complete = False
    if not complete:
        return dict(
            action="replace",
            reason="existing run is incomplete or its artifacts changed",
        )
    return dict(
        action="reuse",
        reason="identical completed training configuration, inputs and implementation",
    )


def record_training_request(output, request):
    """Write the completion signature only after the full run succeeds."""
    path = Path(output) / REQUEST_FILE
    temporary = path.with_suffix(".json.tmp")
    completed = {
        **request,
        "artifacts": {
            name: _digest(Path(output) / name) for name in _COMPLETED_ARTIFACTS
        },
    }
    temporary.write_text(json.dumps(completed, indent=2) + "\n")
    os.replace(temporary, path)
