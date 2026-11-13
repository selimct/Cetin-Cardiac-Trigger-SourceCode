# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Fingerprint inference separately from training metadata and plot rendering.

This module uses only the standard library so the shell wrapper can load it
without importing Torch or initializing the live package for every cache check.
"""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

NON_PREDICTION_FILES = {
    "core/frozen_recovery.py",
    "core/network.py",
    "core/network_helpers.py",
    "live/cross_validation.py",
    "live/recovery.py",
    "live/recovery_calibration.py",
    "live/metrics_overview.py",
    "live/threshold_analysis.py",
    "live/prediction_cache.py",
    "live/prediction_fingerprint.py",
}


def prediction_dependency(path, root):
    name = path.relative_to(root / "src/dkfz").as_posix()
    return name not in NON_PREDICTION_FILES and not name.startswith(
        ("live/training_", "live/hpo_")
    )


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_config(config, script):
    """Use the config contract declared by the actual prediction entrypoint."""
    config = dict(config)
    if "not_use_puls" not in config and "include_puls" in config:
        config["not_use_puls"] = not config["include_puls"]
    declarations = {
        "PREDICTION_CONFIG_REQUIRED_KEYS",
        "PREDICTION_CONFIG_OPTIONAL_KEYS",
        "PREDICTION_RUNTIME_DEFAULTS",
    }
    fields = {
        "modelname",
        "metrics_schema_version",
        "exact_tolerance_frames",
        "outer_error_association_frames",
        "prediction_mode",
        "prediction_delay_frames",
        "signal_quality_puls_flat_window_frames",
        "signal_quality_puls_median_window_frames",
        "feature_cols",
        "manual_threshold",
        "fp_threshold",
        "der_threshold",
    }
    found = set()
    try:
        for node in ast.parse(Path(script).read_text()).body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in declarations:
                        fields.update(ast.literal_eval(node.value))
                        found.add(target.id)
    except (ValueError, TypeError, SyntaxError):
        return config  # Unknown entrypoints conservatively retain their full config.
    if found != declarations:
        return config
    return {key: config[key] for key in sorted(fields) if key in config}


def native_extension():
    try:
        spec = importlib.util.find_spec("dkfz.native._live_core")
    except ModuleNotFoundError:
        return None
    return Path(spec.origin) if spec is not None and spec.origin else None


def prediction_files(root, script):
    root = Path(root).resolve()
    files = [Path(script).resolve()]
    files.extend(
        p for p in (root / "src/dkfz").rglob("*.py") if prediction_dependency(p, root)
    )
    files.extend((root / "native").rglob("*.cpp"))
    files.extend((root / "native").rglob("*.hpp"))
    extension = native_extension()
    if extension is not None:
        files.append(extension)
    return sorted(set(files))


def prediction_cache_info(config, checkpoint, source, script, arguments, *, root=None):
    root = Path.cwd() if root is None else Path(root)
    config, checkpoint, source, script = map(Path, (config, checkpoint, source, script))
    cfg = json.loads(config.read_text())
    stat = source.stat()
    payload = {
        "version": 2,
        "config": runtime_config(cfg, script),
        "checkpoint": file_digest(checkpoint),
        "input": (str(source.resolve()), stat.st_size, stat.st_mtime_ns),
        "code": [
            (str(path.resolve()), file_digest(path))
            for path in prediction_files(root, script)
        ],
        "arguments": list(arguments),
    }
    key = (
        "v2:" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    )
    mode = "single"
    if cfg.get("recovery_head"):
        disabled = (
            "--no-recovery" in arguments
            or not cfg.get("recovery_enabled", True)
            or cfg.get("recovery_threshold") is None
        )
        mode = "recovery-disabled" if disabled else "recovery-active"
    return mode, key


def plot_source_key(script, *, root=None):
    root = Path.cwd() if root is None else Path(root)
    files = [Path(script)]
    for name in (
        "src/dkfz/live/threshold_analysis.py",
        "src/dkfz/core/network_helpers.py",
        "src/dkfz/live/recovery.py",
        "src/dkfz/core/utils.py",
        "src/dkfz/core/colors.py",
        "src/dkfz/live/prediction_cache.py",
    ):
        path = root / name
        if path.exists():
            files.append(path)
    payload = [(str(path.resolve()), file_digest(path)) for path in files]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def plot_cache_key(
    prediction_key, source_key, threshold_step, max_errors, window_frames
):
    data = b"".join(
        str(value).encode() + b"\0"
        for value in (
            prediction_key,
            source_key,
            threshold_step,
            max_errors,
            window_frames,
        )
    )
    return hashlib.sha256(data).hexdigest() + "  -"
