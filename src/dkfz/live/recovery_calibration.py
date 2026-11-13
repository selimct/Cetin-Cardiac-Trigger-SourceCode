"""Recovery-only calibration from annotated validation artifacts, never test folds."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dkfz.core.network_helpers import ThresholdSelection
from dkfz.live.event_matching import outer_association_frames
from dkfz.live.io import REPO_ROOT, repo_relative_path
from dkfz.live.recovery import calibrate_recovery


def save_recovery_validation(predictions, output, *, epoch):
    frames = []
    for index, prediction in enumerate(predictions):
        frame = prediction.copy()
        frame["sid"] = str(index) + ":" + frame["sid"].astype(str)
        frame["calibration_epoch"] = int(epoch)
        frame["calibration_role"] = "inner_validation"
        frames.append(frame)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_parquet(output, index=False)
    return repo_relative_path(output)


def _artifact_path(value, config):
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    if (REPO_ROOT / path).exists():
        return REPO_ROOT / path
    return config.parent / path


def _cv_archives(config, cfg):
    archives = [
        Path("models")
        / (str(cfg["modelname"]).removesuffix("_cv") + "_cv")
        / cfg["cv_method"]
    ]
    for key in ("prepped_file", "selection_search_file"):
        if cfg.get(key):
            archives.insert(0, _artifact_path(cfg[key], config).parent)
    return archives


def validation_logs(config, cfg):
    if cfg.get("recovery_calibration_logs"):
        paths = [
            _artifact_path(value, config) for value in cfg["recovery_calibration_logs"]
        ]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(
                "Previous calibration logs are missing; supply --validation-log"
            )
        return paths
    for key in (
        "recovery_validation_log",
        "tuned_all_events_file",
        "train_all_events_file",
    ):
        if cfg.get(key):
            path = _artifact_path(cfg[key], config)
            if path.is_file():
                return [path]
    if cfg.get("cv_method"):
        for archive in _cv_archives(config, cfg):
            paths = sorted(
                archive.glob(
                    "outer/*/inner/*/selected_epoch_validation_predictions.parquet"
                )
            )
            if not paths:
                continue
            epoch = cfg.get("selected_fixed_epoch")
            for path in paths:
                selection = json.loads(
                    (path.parents[2] / "inner_selection.json").read_text()
                )
                if epoch is None or selection["selected_fixed_epoch"] != epoch:
                    raise ValueError(
                        "Saved inner predictions use different epochs; provide same-epoch --validation-log files or regenerate the recovery CV run"
                    )
            return paths
    name = (
        "tuned_all_events.parquet"
        if config.name == "tuned_config.json"
        else "train_all_events.parquet"
    )
    if not cfg.get("cv_method") and (config.parent / name).is_file():
        return [config.parent / name]
    raise FileNotFoundError(
        "No saved recovery validation predictions found; supply --validation-log. Outer test/prediction-metrics recordings are never used automatically"
    )


def rebuild_validation(config, cfg, output, device):
    """Replay saved inner checkpoints at the deployment epoch; never retrain."""
    import torch
    from dkfz.live.cross_validation import (
        InnerFold,
        InnerTrainingResult,
        _predict_inner_at_epoch,
        load_or_build_patient_frames,
    )
    from dkfz.live.training_config import TRAINING_DEFAULTS

    archive = next(
        (
            path
            for path in _cv_archives(config, cfg)
            if (path / "cv_config.json").is_file()
        ),
        None,
    )
    if archive is None:
        raise FileNotFoundError(
            "Recovery CV archive is unavailable; supply --validation-log"
        )
    recorded = json.loads((archive / "cv_config.json").read_text())
    kwargs = {**TRAINING_DEFAULTS, **recorded["training"]}
    kwargs.update(cfg.get("hpo_parameters") or {})
    kwargs["use_prepped"] = True
    groups = {
        patient: [Path(path) for path in paths]
        for patient, paths in recorded["patient_inputs"].items()
    }
    cache = _artifact_path(recorded["prepped_file"], archive / "cv_config.json")
    frames = load_or_build_patient_frames(groups, kwargs=kwargs, prepped_file=cache)
    predictions = []
    for fold in recorded["folds"]:
        outer = archive / "outer" / fold["test_patient_id"]
        if cfg.get("cv_method") == "nested_lopo_hpo":
            outer = outer / "search" / f"trial_{int(cfg['hpo_trial_id']):03d}"
        for inner in fold["inner_folds"]:
            directory = outer / "inner" / inner["validation_patient_id"]
            info = json.loads((directory / "fold.json").read_text())
            result = InnerTrainingResult(
                InnerFold(
                    inner["validation_patient_id"], tuple(inner["training_patient_ids"])
                ),
                directory,
                (),
                info["signal_quality_stats"],
                info["seed"],
            )
            predictions.append(
                _predict_inner_at_epoch(
                    result,
                    selected_epoch=int(cfg["selected_fixed_epoch"]),
                    patient_frames=frames,
                    kwargs=kwargs,
                    feature_cols=cfg["feature_cols"],
                    input_dim=int(cfg["input_dim"]),
                    device=torch.device(device),
                    save_predictions=False,
                )
            )
    save_recovery_validation(
        predictions, output, epoch=int(cfg["selected_fixed_epoch"])
    )
    return Path(output).resolve()


def calibrate_config(
    config,
    *,
    logs=None,
    max_fp_increase=None,
    min_extra_fp_per_minute=None,
    output=None,
    apply=False,
    rebuild_cv_validation=False,
    device="cpu",
):
    from dkfz.live.cross_validation import _pack_validation_predictions

    config = Path(config)
    cfg = json.loads(config.read_text())
    if not cfg.get("recovery_head"):
        raise ValueError("The config must declare recovery_head=true")
    try:
        paths = [Path(path) for path in logs] if logs else validation_logs(config, cfg)
    except (FileNotFoundError, ValueError):
        if logs or not rebuild_cv_validation or not cfg.get("cv_method"):
            raise
        audit = (
            Path(output)
            if output
            else config.with_name("recovery_threshold_search.csv")
        )
        paths = [
            rebuild_validation(
                config, cfg, audit.with_name(audit.stem + "_validation.parquet"), device
            )
        ]
    frames = []
    required = {"sid", "prob", "recovery_prob", cfg.get("target_col", "target")}
    for path in paths:
        frame = pd.read_parquet(path)
        if not required.issubset(frame.columns):
            raise ValueError(f"Validation log {path} needs columns {sorted(required)}")
        if "split" in frame and frame["split"].eq("test").any():
            raise ValueError("Outer test rows cannot calibrate recovery")
        if (
            "calibration_role" in frame
            and not frame.calibration_role.isin(
                ["inner_validation", "validation"]
            ).all()
        ):
            raise ValueError("Only validation roles can calibrate recovery")
        if "calibration_epoch" in frame and cfg.get("selected_fixed_epoch") is not None:
            if not frame.calibration_epoch.eq(int(cfg["selected_fixed_epoch"])).all():
                raise ValueError(
                    "Validation predictions do not match the selected deployment epoch"
                )
        frames.append(frame)
    holdoff = int(cfg.get("holdoff_frames", cfg.get("min_distance", 10)))
    from dkfz.live.event_matching import event_tolerances

    policy = event_tolerances(cfg)
    association = policy["error_association_frames"]
    probabilities, targets = _pack_validation_predictions(
        frames,
        target_col=cfg.get("target_col", "target"),
        holdoff_frames=holdoff,
        error_association_frames=association,
    )
    primary = (
        cfg.get("manual_threshold")
        if cfg.get("manual_threshold") is not None
        else cfg["threshold"]
    )
    selection = ThresholdSelection(
        threshold=float(primary),
        objective="fixed",
        sensitivity_efficiency=0.95,
        best_attainable_sensitivity=0.0,
        effective_sensitivity_floor=0.0,
        metrics={},
        search_rows=(),
        fp_threshold=(
            cfg.get("fp_threshold")
            if cfg.get("fp_threshold") is not None
            else cfg["threshold"]
        ),
        der_threshold=cfg.get("der_threshold"),
    )
    allowance = (
        max_fp_increase
        if max_fp_increase is not None
        else cfg.get("recovery_max_fp_increase", 0.10)
    )
    selected, search = calibrate_recovery(
        probabilities,
        np.flatnonzero(targets == 1),
        selection,
        holdoff_frames=holdoff,
        error_association_frames=association,
        outer_error_association_frames=policy["outer_error_association_frames"],
        exact_tolerance_frames=policy["exact_tolerance_frames"],
        max_fp_increase=float(allowance),
        min_extra_fp_per_minute=float(
            min_extra_fp_per_minute
            if min_extra_fp_per_minute is not None
            else cfg.get("recovery_min_extra_fp_per_minute", 0.5)
        ),
        validation_minutes=sum(int(frame["sid"].notna().sum()) for frame in frames)
        * float(cfg.get("frame_size_raw", 8))
        / float(cfg.get("raw_hz", 400))
        / 60,
    )
    output = (
        Path(output) if output else config.with_name("recovery_threshold_search.csv")
    )
    if output.resolve() == config.resolve():
        raise ValueError("The audit output must not overwrite the config")
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(search).to_csv(output, index=False)
    previous = cfg.get("recovery_threshold")
    if apply:
        cfg.update(
            recovery_threshold=selected.recovery_threshold,
            recovery_max_fp_increase=selected.recovery_max_fp_increase,
            recovery_min_extra_fp_per_minute=selected.recovery_min_extra_fp_per_minute,
            recovery_validation_minutes=selected.recovery_validation_minutes,
            recovery_error_policy="reduce_fn_and_fp_plus_fn",
            recovery_validation_metrics=selected.recovery_validation_metrics,
            recovery_calibration_logs=[
                repo_relative_path(path.resolve()) for path in paths
            ],
        )
        temporary = config.with_suffix(config.suffix + ".tmp")
        temporary.write_text(json.dumps(cfg, indent=2) + "\n")
        temporary.replace(config)
    return dict(
        config=str(config),
        status="complete",
        previous_threshold=primary,
        previous_recovery_threshold=previous,
        recovery_threshold=selected.recovery_threshold,
        validation_logs=[str(path) for path in paths],
        audit=str(output),
        applied=apply,
    )
