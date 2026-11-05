# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Nested leave-one-patient-out training for the live ECG model."""

from __future__ import annotations

from dataclasses import dataclass
import glob
import json
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dkfz.core.dataset import SlidingWindowDataset
from dkfz.core.network import train_torch
from dkfz.core.network_helpers import (
    ThresholdSelection,
    evaluate_threshold_grid,
    select_threshold,
    select_threshold_from_search_rows,
)
from dkfz.live import default_torch_device
from dkfz.live.event_matching import (
    EVENT_COUNT_KEYS,
    event_metrics_from_counts,
    outer_association_frames,
)
from dkfz.live.eval_metrics import build_event_position_summary
from dkfz.live.feature_framing import fit_apply_signal_quality_feature
from dkfz.live.io import REPO_ROOT, repo_relative_path, resolve_input_paths
from dkfz.live.training_evaluation import (
    build_live_eval_outputs,
    predict_validation_probabilities,
    validation_window_rows,
    write_live_eval_artifacts,
)
from dkfz.live.training_pipeline import (
    base_training_metadata,
    build_ecg_rnn,
    build_training_loaders,
    training_feature_columns,
    training_frame_build_kwargs,
    write_metadata,
)
from dkfz.live.training_preprocessing import (
    SPLIT_COL,
    TRAIN_SPLIT,
    VAL_SPLIT,
    build_training_frame_dfs,
    load_prepped_training_frame_dfs,
    save_prepped_training_frame_dfs,
)
from dkfz.live.training_seed import (
    derive_training_seed,
    set_training_seed,
    validate_training_seed,
)

PATIENT_ID_COL = "patient_id"
TEST_SPLIT = "test"
_PATIENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class InnerFold:
    """One inner patient-held-out validation split."""

    validation_patient_id: str
    training_patient_ids: tuple[str, ...]


@dataclass(frozen=True)
class OuterFold:
    """One untouched outer patient and all corresponding inner folds."""

    test_patient_id: str
    final_training_patient_ids: tuple[str, ...]
    inner_folds: tuple[InnerFold, ...]


@dataclass
class InnerTrainingResult:
    """Artifacts needed after one inner training run."""

    fold: InnerFold
    fold_dir: Path
    validation_losses: tuple[float, ...]
    signal_quality_stats: dict[str, Any] | None
    seed: int

    def checkpoint_for_epoch(self, epoch: int) -> Path:
        return self.fold_dir / "epoch_checkpoints" / f"epoch_{int(epoch):04d}.pth"


@dataclass
class JointEpochThresholdResult:
    """Joint event-based selection and its inner out-of-fold artifacts."""

    selected_epoch: int
    threshold_selection: ThresholdSelection
    epoch_search: pd.DataFrame
    threshold_search: pd.DataFrame
    selected_prediction_frames: tuple[pd.DataFrame, ...]


class _NoOpTiming:
    def mark(self, _label: str) -> None:
        pass


def nested_lopo_folds(patient_ids: Iterable[str]) -> tuple[OuterFold, ...]:
    """Return deterministic nested leave-one-patient-out folds."""
    ids = tuple(sorted(str(patient_id) for patient_id in patient_ids))
    if len(ids) < 3:
        raise ValueError("Nested leave-one-patient-out requires at least 3 patients.")
    if len(set(ids)) != len(ids):
        raise ValueError("Patient IDs must be unique.")

    outer_folds: list[OuterFold] = []
    for test_id in ids:
        final_train_ids = tuple(
            patient_id for patient_id in ids if patient_id != test_id
        )
        inner_folds = tuple(
            InnerFold(
                validation_patient_id=validation_id,
                training_patient_ids=tuple(
                    patient_id
                    for patient_id in final_train_ids
                    if patient_id != validation_id
                ),
            )
            for validation_id in final_train_ids
        )
        outer_folds.append(
            OuterFold(
                test_patient_id=test_id,
                final_training_patient_ids=final_train_ids,
                inner_folds=inner_folds,
            )
        )
    return tuple(outer_folds)


def select_fixed_epoch(
    validation_losses: Sequence[Sequence[float]],
) -> tuple[int, pd.DataFrame]:
    """Select the earliest epoch with minimum mean patient validation loss."""
    if not validation_losses or any(not losses for losses in validation_losses):
        raise ValueError("Every inner fold must contain validation losses.")
    lengths = {len(losses) for losses in validation_losses}
    if len(lengths) != 1:
        raise ValueError("Inner validation histories must have equal lengths.")
    values = np.asarray(validation_losses, dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Inner validation histories contain non-finite losses.")

    rows = pd.DataFrame(
        {
            "epoch": np.arange(1, values.shape[1] + 1, dtype=np.int64),
            "mean_validation_loss": values.mean(axis=0),
            "std_validation_loss": values.std(axis=0),
            "min_validation_loss": values.min(axis=0),
            "max_validation_loss": values.max(axis=0),
        }
    )
    mean_losses = rows["mean_validation_loss"].to_numpy()
    minimum = float(np.min(mean_losses))
    selected_idx = int(np.flatnonzero(np.isclose(mean_losses, minimum))[0])
    selected_epoch = int(rows.iloc[selected_idx]["epoch"])
    rows["selected"] = rows["epoch"].eq(selected_epoch)
    return selected_epoch, rows


def _validate_patient_id(patient_id: str) -> str:
    patient_id = str(patient_id)
    if not _PATIENT_ID_RE.fullmatch(patient_id) or patient_id in {".", ".."}:
        raise ValueError(
            f"Invalid patient ID {patient_id!r}; use letters, digits, '.', '_', or '-'."
        )
    return patient_id


def _manifest_patient_patterns(path: Path) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "patients" in raw:
        raw = raw["patients"]
    if not isinstance(raw, dict) or not raw:
        raise ValueError("Patient manifest must contain a non-empty patient mapping.")

    patterns: dict[str, list[str]] = {}
    for raw_id, raw_inputs in raw.items():
        patient_id = _validate_patient_id(str(raw_id))
        values = [raw_inputs] if isinstance(raw_inputs, str) else raw_inputs
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(value, str) and value for value in values)
        ):
            raise ValueError(
                f"Patient {patient_id!r} must map to a string or non-empty string list."
            )
        resolved_patterns = []
        for value in values:
            value_path = Path(value)
            resolved_patterns.append(
                str(
                    value_path if value_path.is_absolute() else path.parent / value_path
                )
            )
        patterns[patient_id] = resolved_patterns
    return patterns


def resolve_patient_groups(
    *,
    patient_inputs: Sequence[str] = (),
    patient_root: str | Path | None = None,
    patient_manifest: str | Path | None = None,
) -> dict[str, list[Path]]:
    """Resolve one and only one patient-group source into disjoint parquet paths."""
    selected_sources = sum(
        bool(value) for value in (patient_inputs, patient_root, patient_manifest)
    )
    if selected_sources != 1:
        raise ValueError(
            "Provide exactly one of patient_inputs, patient_root, or patient_manifest."
        )

    patterns: dict[str, list[str]] = {}
    if patient_manifest is not None:
        patterns = _manifest_patient_patterns(Path(patient_manifest).resolve())
    elif patient_root is not None:
        root = Path(patient_root).resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Patient root is not a directory: {root}")
        for patient_dir in sorted(path for path in root.iterdir() if path.is_dir()):
            if any(patient_dir.glob("*.parquet")):
                patterns[_validate_patient_id(patient_dir.name)] = [str(patient_dir)]
    else:
        expanded_dirs: list[Path] = []
        for value in patient_inputs:
            matches = [Path(match) for match in glob.glob(value)]
            candidates = matches or [Path(value)]
            expanded_dirs.extend(candidates)
        for patient_dir in sorted({path.resolve() for path in expanded_dirs}):
            if not patient_dir.is_dir():
                raise ValueError(
                    "Positional patient inputs must each resolve to a patient directory; "
                    f"got {patient_dir}."
                )
            patient_id = _validate_patient_id(patient_dir.name)
            if patient_id in patterns:
                raise ValueError(f"Duplicate patient directory ID {patient_id!r}.")
            patterns[patient_id] = [str(patient_dir)]

    if len(patterns) < 3:
        raise ValueError(
            f"Nested leave-one-patient-out requires at least 3 patients, got {len(patterns)}."
        )

    groups = {
        patient_id: resolve_input_paths(values)
        for patient_id, values in sorted(patterns.items())
    }
    path_owner: dict[Path, str] = {}
    for patient_id, paths in groups.items():
        for path in paths:
            previous = path_owner.setdefault(path, patient_id)
            if previous != patient_id:
                raise ValueError(
                    f"Input {repo_relative_path(path)} belongs to both {previous!r} "
                    f"and {patient_id!r}."
                )
    return groups


def ensure_output_outside_data(path: str | Path) -> Path:
    """Reject LOPO outputs under protected repository data directories."""
    resolved = Path(path).resolve()
    for protected in (REPO_ROOT / "data", REPO_ROOT / ".data"):
        protected = protected.resolve()
        if resolved == protected or protected in resolved.parents:
            raise ValueError(
                "LOPO outputs must not be written under data/ or .data/: "
                f"{repo_relative_path(resolved)}"
            )
    return resolved


def _validate_cached_patient_sources(
    frame: pd.DataFrame,
    patient_groups: Mapping[str, Sequence[Path]],
) -> None:
    if PATIENT_ID_COL not in frame.columns:
        raise ValueError(
            f"Preprocessed LOPO cache is missing {PATIENT_ID_COL!r}; regenerate it."
        )
    expected = {
        patient_id: {repo_relative_path(path) for path in paths}
        for patient_id, paths in patient_groups.items()
    }
    actual = {
        str(patient_id): set(group["source_file"].astype(str).unique())
        for patient_id, group in frame.groupby(PATIENT_ID_COL, sort=False)
    }
    if actual != expected:
        raise ValueError(
            "Preprocessed LOPO cache does not match the configured patient inputs; "
            "regenerate it without --use-prepped."
        )


def load_or_build_patient_frames(
    patient_groups: Mapping[str, Sequence[Path]],
    *,
    kwargs: Mapping[str, Any],
    prepped_file: str | Path,
) -> dict[str, pd.DataFrame]:
    """Build causal frames once per patient, before assigning fold roles."""
    prepped_path = ensure_output_outside_data(prepped_file)
    if bool(kwargs["use_prepped"]):
        frame = pd.concat(
            load_prepped_training_frame_dfs(str(prepped_path)),
            ignore_index=True,
        )
        _validate_cached_patient_sources(frame, patient_groups)
    else:
        build_kwargs = training_frame_build_kwargs(
            dict(kwargs),
            # Fold-specific signal-quality calibration is fitted later.
            signal_quality_stats=None,
        )
        patient_frames = []
        for patient_id, paths in sorted(patient_groups.items()):
            print(f"Preparing patient {patient_id} ({len(paths)} input files)")
            frames = build_training_frame_dfs(
                input_paths=list(paths),
                validation_fraction=None,
                split=TRAIN_SPLIT,
                **build_kwargs,
            )
            patient_frame = pd.concat(frames, ignore_index=True)
            patient_frame[PATIENT_ID_COL] = patient_id
            patient_frames.append(patient_frame)
        frame = pd.concat(patient_frames, ignore_index=True)
        if bool(kwargs["save_prepped"]):
            save_prepped_training_frame_dfs([frame], str(prepped_path))

    return {
        str(patient_id): group.copy().reset_index(drop=True)
        for patient_id, group in frame.groupby(PATIENT_ID_COL, sort=False)
    }


def _role_frames(
    patient_frames: Mapping[str, pd.DataFrame],
    roles: Mapping[str, str],
) -> list[pd.DataFrame]:
    """Copy selected patients, assign roles, and make sequence IDs disjoint."""
    frames: list[pd.DataFrame] = []
    next_sid = 0
    for patient_id, split in roles.items():
        if patient_id not in patient_frames:
            raise KeyError(f"No prepared frames for patient {patient_id!r}.")
        frame = patient_frames[patient_id].copy()
        if frame.empty:
            raise ValueError(f"Patient {patient_id!r} has no prepared frames.")
        sid_codes, unique_sids = pd.factorize(frame["sid"], sort=False)
        frame["sid"] = sid_codes.astype(np.int64) + next_sid
        next_sid += len(unique_sids)
        frame[PATIENT_ID_COL] = patient_id
        frame[SPLIT_COL] = split
        frames.append(frame)
    return frames


def _prepare_fold_frames(
    patient_frames: Mapping[str, pd.DataFrame],
    *,
    training_patient_ids: Sequence[str],
    evaluation_patient_id: str,
    evaluation_split: str,
    signal_quality_mode: str,
    signal_quality_stats: dict[str, Any] | None = None,
) -> tuple[list[pd.DataFrame], dict[str, Any] | None]:
    roles = {patient_id: TRAIN_SPLIT for patient_id in training_patient_ids}
    roles[evaluation_patient_id] = evaluation_split
    frames = _role_frames(patient_frames, roles)
    return fit_apply_signal_quality_feature(
        frames,
        mode=signal_quality_mode,
        stats=signal_quality_stats,
    )


def _split_dataframe(frames: Sequence[pd.DataFrame], split: str) -> pd.DataFrame:
    frame = pd.concat(frames, ignore_index=True)
    selected = frame.loc[frame[SPLIT_COL] == split].copy().reset_index(drop=True)
    if selected.empty:
        raise ValueError(f"Fold contains no {split!r} frames.")
    return selected


def _json_default(value: Any):
    if isinstance(value, Path):
        return repo_relative_path(value)
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, default=_json_default)


def _threshold_summary(selection: ThresholdSelection) -> dict[str, Any]:
    return {
        "threshold": selection.threshold,
        **selection.policy_metadata(),
        "objective": selection.objective,
        "sensitivity_efficiency": selection.sensitivity_efficiency,
        "best_attainable_sensitivity": selection.best_attainable_sensitivity,
        "effective_sensitivity_floor": selection.effective_sensitivity_floor,
        "metrics": selection.metrics,
    }


_EVENT_COUNT_KEYS = EVENT_COUNT_KEYS


def _safe_ratio(numerator: int, denominator: int) -> float:
    return 0.0 if denominator <= 0 else float(numerator / denominator)


def _aggregate_epoch_threshold_searches(
    search_frames: Sequence[pd.DataFrame],
) -> tuple[dict[str, Any], ...]:
    """Pool event counts for matching epoch/threshold rows across CV searches."""
    if not search_frames:
        raise ValueError("At least one epoch-threshold search is required.")
    rows = pd.concat(search_frames, ignore_index=True)
    required = {"epoch", "threshold", "selectable_threshold", *_EVENT_COUNT_KEYS}
    missing = required - set(rows.columns)
    if missing:
        raise ValueError(
            f"Epoch-threshold search is missing columns {sorted(missing)}."
        )

    pooled_rows: list[dict[str, Any]] = []
    for (epoch, threshold), group in rows.groupby(
        ["epoch", "threshold"], sort=True, dropna=False
    ):
        counts = {
            key: int(pd.to_numeric(group[key], errors="raise").sum())
            for key in _EVENT_COUNT_KEYS
        }
        first = group.iloc[0]
        pooled_rows.append(
            {
                "epoch": int(epoch),
                "threshold": float(threshold),
                "selectable_threshold": bool(group["selectable_threshold"].all()),
                **event_metrics_from_counts(
                    counts,
                    int(first.get("error_association_frames", 1)),
                    int(first.get("outer_error_association_frames", 2)),
                ),
            }
        )
    return tuple(pooled_rows)


def select_global_epoch_threshold(
    search_frames: Sequence[pd.DataFrame],
    *,
    objective: str,
    sensitivity_efficiency: float,
) -> tuple[int, ThresholdSelection, pd.DataFrame]:
    """Choose deployment epoch/threshold from all nested inner-CV searches."""
    pooled_rows = _aggregate_epoch_threshold_searches(search_frames)
    selection = select_threshold_from_search_rows(
        pooled_rows,
        objective=objective,
        sensitivity_efficiency=sensitivity_efficiency,
        metadata_keys=("epoch",),
    )
    selected_rows = [row for row in selection.search_rows if row["selected"]]
    if len(selected_rows) != 1:
        raise RuntimeError("Global CV search did not select exactly one row.")
    return (
        int(selected_rows[0]["epoch"]),
        selection,
        pd.DataFrame(selection.search_rows),
    )


def _pack_validation_predictions(
    prediction_frames: Sequence[pd.DataFrame],
    *,
    target_col: str,
    holdoff_frames: int,
    error_association_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    gap_len = int(max(holdoff_frames + 1, 2 * error_association_frames + 1, 20))
    prob_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    for frame in prediction_frames:
        for _sid, group in frame.groupby("sid", sort=False):
            prob_parts.append(group["prob"].to_numpy(dtype=np.float32))
            target_parts.append(group[target_col].to_numpy(dtype=np.float32))
            prob_parts.append(np.zeros(gap_len, dtype=np.float32))
            target_parts.append(np.zeros(gap_len, dtype=np.float32))
    if not prob_parts:
        raise ValueError("Inner validation produced no prediction rows.")
    return np.concatenate(prob_parts), np.concatenate(target_parts)


def select_pooled_inner_threshold(
    prediction_frames: Sequence[pd.DataFrame],
    *,
    target_col: str,
    objective: str,
    sensitivity_efficiency: float,
    holdoff_frames: int,
    error_association_frames: int,
    outer_error_association_frames: int = 2,
) -> ThresholdSelection:
    """Select one threshold from pooled patient-held-out event counts."""
    probabilities, targets = _pack_validation_predictions(
        prediction_frames,
        target_col=target_col,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
    )
    return select_threshold(
        probabilities,
        np.where(targets == 1)[0],
        objective=objective,
        sensitivity_efficiency=sensitivity_efficiency,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
        outer_error_association_frames=outer_error_association_frames,
    )


def select_joint_inner_epoch_threshold(
    inner_results: Sequence[InnerTrainingResult],
    *,
    patient_frames: Mapping[str, pd.DataFrame],
    kwargs: Mapping[str, Any],
    feature_cols: Sequence[str],
    input_dim: int,
    device: torch.device,
    prediction_transform: (
        Callable[[Sequence[pd.DataFrame], int], Sequence[pd.DataFrame]] | None
    ) = None,
) -> JointEpochThresholdResult:
    """Select one epoch and threshold from all inner out-of-fold event counts."""
    _bce_epoch, epoch_search = select_fixed_epoch(
        [result.validation_losses for result in inner_results]
    )
    epoch_search = epoch_search.rename(columns={"selected": "bce_minimum"})
    joint_rows: list[dict[str, Any]] = []
    best_sensitivity_by_epoch: dict[int, float] = {}
    num_epochs = len(inner_results[0].validation_losses)

    for epoch in range(1, num_epochs + 1):
        prediction_frames = [
            _predict_inner_at_epoch(
                result,
                selected_epoch=epoch,
                patient_frames=patient_frames,
                kwargs=kwargs,
                feature_cols=feature_cols,
                input_dim=input_dim,
                device=device,
                save_predictions=False,
            )
            for result in inner_results
        ]
        scoring_frames = (
            list(prediction_transform(prediction_frames, epoch))
            if prediction_transform is not None
            else prediction_frames
        )
        probabilities, targets = _pack_validation_predictions(
            scoring_frames,
            target_col=str(kwargs["target_col"]),
            holdoff_frames=int(kwargs["min_distance"]),
            error_association_frames=int(kwargs["error_association_frames"]),
        )
        rows = evaluate_threshold_grid(
            probabilities,
            np.where(targets == 1)[0],
            holdoff_frames=int(kwargs["min_distance"]),
            error_association_frames=int(kwargs["error_association_frames"]),
            outer_error_association_frames=outer_association_frames(
                kwargs["raw_hz"], kwargs["frame_size_raw"]
            ),
        )
        tagged_rows = [{"epoch": epoch, **row} for row in rows]
        joint_rows.extend(tagged_rows)
        best_sensitivity_by_epoch[epoch] = max(
            float(row["exact_sensitivity"])
            for row in tagged_rows
            if row["selectable_threshold"]
        )

    threshold_selection = select_threshold_from_search_rows(
        joint_rows,
        objective=str(kwargs["threshold_objective"]),
        sensitivity_efficiency=float(kwargs["threshold_sensitivity_efficiency"]),
        metadata_keys=("epoch",),
    )
    selected_rows = [row for row in threshold_selection.search_rows if row["selected"]]
    if len(selected_rows) != 1:
        raise RuntimeError(
            "Joint epoch-threshold search did not select exactly one row."
        )
    selected_epoch = int(selected_rows[0]["epoch"])
    epoch_search["best_attainable_exact_sensitivity"] = epoch_search["epoch"].map(
        best_sensitivity_by_epoch
    )
    epoch_search["selected_threshold"] = np.where(
        epoch_search["epoch"].eq(selected_epoch),
        float(threshold_selection.threshold),
        np.nan,
    )
    epoch_search["selected"] = epoch_search["epoch"].eq(selected_epoch)

    selected_prediction_frames = tuple(
        _predict_inner_at_epoch(
            result,
            selected_epoch=selected_epoch,
            patient_frames=patient_frames,
            kwargs=kwargs,
            feature_cols=feature_cols,
            input_dim=input_dim,
            device=device,
            save_predictions=True,
        )
        for result in inner_results
    )
    return JointEpochThresholdResult(
        selected_epoch=selected_epoch,
        threshold_selection=threshold_selection,
        epoch_search=epoch_search,
        threshold_search=pd.DataFrame(threshold_selection.search_rows),
        selected_prediction_frames=selected_prediction_frames,
    )


def _build_loader(
    frame: pd.DataFrame,
    *,
    feature_cols: Sequence[str],
    target_col: str,
    window_size: int,
    batch_size: int,
    shuffle: bool,
) -> tuple[SlidingWindowDataset, DataLoader]:
    dataset = SlidingWindowDataset(
        frame,
        feature_cols=list(feature_cols),
        target_col=target_col,
        window_size=window_size,
    )
    if not len(dataset):
        raise ValueError("Fold produced no valid sliding windows.")
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _train_inner_fold(
    *,
    outer_fold: OuterFold,
    inner_fold: InnerFold,
    patient_frames: Mapping[str, pd.DataFrame],
    kwargs: Mapping[str, Any],
    feature_cols: Sequence[str],
    input_dim: int,
    device: torch.device,
    outer_dir: Path,
    base_seed: int,
) -> InnerTrainingResult:
    fold_dir = outer_dir / "inner" / inner_fold.validation_patient_id
    fold_dir.mkdir(parents=True, exist_ok=True)
    frames, signal_quality_stats = _prepare_fold_frames(
        patient_frames,
        training_patient_ids=inner_fold.training_patient_ids,
        evaluation_patient_id=inner_fold.validation_patient_id,
        evaluation_split=VAL_SPLIT,
        signal_quality_mode=str(kwargs["signal_quality_mode"]),
    )
    _train_dataset, _val_dataset, train_loader, val_loader = build_training_loaders(
        frames,
        feature_cols=list(feature_cols),
        target_col=str(kwargs["target_col"]),
        window_size=int(kwargs["window_size"]),
        batch_size=int(kwargs["batch_size"]),
        timing=_NoOpTiming(),
    )
    seed = derive_training_seed(
        base_seed,
        "inner",
        outer_fold.test_patient_id,
        inner_fold.validation_patient_id,
    )
    set_training_seed(seed)
    model = build_ecg_rnn(
        input_dim=input_dim,
        hidden_dim=int(kwargs["hidden_dim"]),
        num_layers=int(kwargs["num_layers"]),
        device=device,
    )
    _train_losses, validation_losses = train_torch(
        model,
        nn.BCEWithLogitsLoss(),
        train_loader,
        val_loader,
        lr=float(kwargs["lr"]),
        num_epochs=int(kwargs["num_epochs"]),
        early_stopping=0,
        device=device,
        modelname=fold_dir.name,
        non_blocking=True,
        use_amp=bool(kwargs["use_amp"]),
        save_best=False,
        save_last=False,
        models_dir=fold_dir.parent,
        history_filename="train_history.json",
        epoch_checkpoint_dir="epoch_checkpoints",
    )
    result = InnerTrainingResult(
        fold=inner_fold,
        fold_dir=fold_dir,
        validation_losses=tuple(validation_losses),
        signal_quality_stats=signal_quality_stats,
        seed=seed,
    )
    _write_json(
        fold_dir / "fold.json",
        {
            "outer_test_patient_id": outer_fold.test_patient_id,
            "validation_patient_id": inner_fold.validation_patient_id,
            "training_patient_ids": list(inner_fold.training_patient_ids),
            "seed": seed,
            "signal_quality_stats": signal_quality_stats,
        },
    )
    return result


def _predict_inner_at_epoch(
    result: InnerTrainingResult,
    *,
    selected_epoch: int,
    patient_frames: Mapping[str, pd.DataFrame],
    kwargs: Mapping[str, Any],
    feature_cols: Sequence[str],
    input_dim: int,
    device: torch.device,
    save_predictions: bool = True,
) -> pd.DataFrame:
    checkpoint = result.checkpoint_for_epoch(selected_epoch)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing inner epoch checkpoint: {checkpoint}")
    frames, _ = _prepare_fold_frames(
        patient_frames,
        training_patient_ids=result.fold.training_patient_ids,
        evaluation_patient_id=result.fold.validation_patient_id,
        evaluation_split=VAL_SPLIT,
        signal_quality_mode=str(kwargs["signal_quality_mode"]),
        signal_quality_stats=result.signal_quality_stats,
    )
    val_frame = _split_dataframe(frames, VAL_SPLIT)
    dataset, loader = _build_loader(
        val_frame,
        feature_cols=feature_cols,
        target_col=str(kwargs["target_col"]),
        window_size=int(kwargs["window_size"]),
        batch_size=int(kwargs["batch_size"]),
        shuffle=False,
    )
    model = build_ecg_rnn(
        input_dim=input_dim,
        hidden_dim=int(kwargs["hidden_dim"]),
        num_layers=int(kwargs["num_layers"]),
        device=device,
        checkpoint_path=checkpoint,
    )
    probabilities = predict_validation_probabilities(
        model=model,
        loader=loader,
        device=device,
        use_amp=bool(kwargs["use_amp"]),
    )
    rows = validation_window_rows(dataset)
    rows["prob"] = probabilities
    rows["outer_test_patient_id"] = None
    rows[PATIENT_ID_COL] = result.fold.validation_patient_id
    keep_cols = [
        col
        for col in (
            PATIENT_ID_COL,
            "sid",
            "frame_idx",
            "raw_idx",
            "time_ms",
            "source_file",
            str(kwargs["target_col"]),
            "prob",
        )
        if col in rows.columns
    ]
    predictions = rows[keep_cols].copy()
    if save_predictions:
        predictions.to_parquet(
            result.fold_dir / "selected_epoch_validation_predictions.parquet",
            compression="snappy",
        )
    return predictions


def _train_final_outer_model(
    *,
    outer_fold: OuterFold,
    selected_epoch: int,
    threshold_selection: ThresholdSelection,
    patient_frames: Mapping[str, pd.DataFrame],
    patient_groups: Mapping[str, Sequence[Path]],
    kwargs: Mapping[str, Any],
    feature_cols: Sequence[str],
    input_dim: int,
    device: torch.device,
    outer_dir: Path,
    prepped_file: Path,
    base_seed: int,
) -> dict[str, Any]:
    frames, signal_quality_stats = _prepare_fold_frames(
        patient_frames,
        training_patient_ids=outer_fold.final_training_patient_ids,
        evaluation_patient_id=outer_fold.test_patient_id,
        evaluation_split=TEST_SPLIT,
        signal_quality_mode=str(kwargs["signal_quality_mode"]),
    )
    train_frame = _split_dataframe(frames, TRAIN_SPLIT)
    test_frame = _split_dataframe(frames, TEST_SPLIT)
    _train_dataset, train_loader = _build_loader(
        train_frame,
        feature_cols=feature_cols,
        target_col=str(kwargs["target_col"]),
        window_size=int(kwargs["window_size"]),
        batch_size=int(kwargs["batch_size"]),
        shuffle=True,
    )
    test_dataset, test_loader = _build_loader(
        test_frame,
        feature_cols=feature_cols,
        target_col=str(kwargs["target_col"]),
        window_size=int(kwargs["window_size"]),
        batch_size=int(kwargs["batch_size"]),
        shuffle=False,
    )

    seed = derive_training_seed(base_seed, "final", outer_fold.test_patient_id)
    set_training_seed(seed)
    model = build_ecg_rnn(
        input_dim=input_dim,
        hidden_dim=int(kwargs["hidden_dim"]),
        num_layers=int(kwargs["num_layers"]),
        device=device,
    )
    train_torch(
        model,
        nn.BCEWithLogitsLoss(),
        train_loader,
        None,
        lr=float(kwargs["lr"]),
        num_epochs=selected_epoch,
        early_stopping=0,
        device=device,
        modelname=outer_dir.name,
        non_blocking=True,
        use_amp=bool(kwargs["use_amp"]),
        save_best=False,
        save_last=True,
        models_dir=outer_dir.parent,
        last_filename="final.pth",
        history_filename="final_history.json",
    )
    final_path = outer_dir / "final.pth"
    final_model = build_ecg_rnn(
        input_dim=input_dim,
        hidden_dim=int(kwargs["hidden_dim"]),
        num_layers=int(kwargs["num_layers"]),
        device=device,
        checkpoint_path=final_path,
    )
    prediction_df, events_df = build_live_eval_outputs(
        model=final_model,
        loader=test_loader,
        dataset=test_dataset,
        threshold=float(threshold_selection.threshold),
        holdoff_frames=int(kwargs["min_distance"]),
        frame_size_raw=int(kwargs["frame_size_raw"]),
        feature_cols=list(feature_cols),
        target_col=str(kwargs["target_col"]),
        device=device,
        use_amp=bool(kwargs["use_amp"]),
    )
    write_live_eval_artifacts(
        model_dir=outer_dir,
        prediction_df=prediction_df,
        events_df=events_df,
        threshold_search_df=pd.DataFrame(),
        target_col=str(kwargs["target_col"]),
        threshold=float(threshold_selection.threshold),
        holdoff_frames=int(kwargs["min_distance"]),
        error_association_frames=int(kwargs["error_association_frames"]),
        prefix="test",
        metrics_dir=outer_dir / "metrics",
        metrics_prefix="test",
        threshold_selection=None,
        outer_error_association_frames=outer_association_frames(
            kwargs["raw_hz"], kwargs["frame_size_raw"]
        ),
    )

    metrics_path = outer_dir / "metrics" / "test_metrics.json"
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    metrics.update(
        {
            **threshold_selection.policy_metadata(),
            "outer_test_patient_id": outer_fold.test_patient_id,
            "threshold_selection_source": "nested_inner_out_of_fold",
            "epoch_selection_source": "nested_inner_out_of_fold_event_counts",
            "selected_fixed_epoch": selected_epoch,
            "final_training_patient_ids": list(outer_fold.final_training_patient_ids),
        }
    )
    _write_json(metrics_path, metrics)

    final_kwargs = dict(kwargs)
    final_kwargs["num_epochs"] = selected_epoch
    final_kwargs["signal_quality_stats"] = signal_quality_stats
    final_kwargs["save_last"] = True
    input_paths = [
        path
        for patient_id in outer_fold.final_training_patient_ids
        for path in patient_groups[patient_id]
    ]
    metadata = base_training_metadata(
        kwargs=final_kwargs,
        input_dim=input_dim,
        feature_cols=list(feature_cols),
        best_threshold=float(threshold_selection.threshold),
        threshold_selection=threshold_selection,
        input_paths=input_paths,
        threshold_input_paths=[],
        prepped_file=prepped_file,
    )
    metadata.update(
        {
            "base_modelname": kwargs.get("base_modelname", kwargs["modelname"]),
            "cv_method": kwargs.get("cv_method", "nested_lopo"),
            "checkpoint": repo_relative_path(final_path),
            "checkpoint_policy": "joint_nested_cv_epoch_threshold",
            "configured_max_epochs": int(kwargs["num_epochs"]),
            "selected_fixed_epoch": selected_epoch,
            "outer_test_patient_id": outer_fold.test_patient_id,
            "final_training_patient_ids": list(outer_fold.final_training_patient_ids),
            "test_inputs": [
                repo_relative_path(path)
                for path in patient_groups[outer_fold.test_patient_id]
            ],
            "threshold_selection_source": "nested_inner_out_of_fold",
            "epoch_selection_source": "nested_inner_out_of_fold_event_counts",
            "test_metrics_file": metrics_path,
            "test_all_events_file": outer_dir / "test_all_events.parquet",
            "test_trigger_events_file": outer_dir / "test_trigger_events.parquet",
            "seed": seed,
        }
    )
    write_metadata(outer_dir / "train_config.json", metadata)
    return metrics


def train_best_full_model(
    *,
    selected_epoch: int,
    threshold_selection: ThresholdSelection,
    patient_frames: Mapping[str, pd.DataFrame],
    patient_groups: Mapping[str, Sequence[Path]],
    kwargs: Mapping[str, Any],
    feature_cols: Sequence[str],
    input_dim: int,
    device: torch.device,
    output_dir: Path,
    prepped_file: Path,
    base_seed: int,
    selection_search_file: Path,
) -> dict[str, Any]:
    """Train one deployment checkpoint on every patient after nested CV."""
    patient_ids = tuple(sorted(patient_frames))
    frames = _role_frames(
        patient_frames,
        {patient_id: TRAIN_SPLIT for patient_id in patient_ids},
    )
    frames, signal_quality_stats = fit_apply_signal_quality_feature(
        frames,
        mode=str(kwargs["signal_quality_mode"]),
    )
    train_frame = _split_dataframe(frames, TRAIN_SPLIT)
    _train_dataset, train_loader = _build_loader(
        train_frame,
        feature_cols=feature_cols,
        target_col=str(kwargs["target_col"]),
        window_size=int(kwargs["window_size"]),
        batch_size=int(kwargs["batch_size"]),
        shuffle=True,
    )

    seed = derive_training_seed(base_seed, "best", *patient_ids)
    set_training_seed(seed)
    model = build_ecg_rnn(
        input_dim=input_dim,
        hidden_dim=int(kwargs["hidden_dim"]),
        num_layers=int(kwargs["num_layers"]),
        device=device,
    )
    print(
        f"Training best.pth on all {len(patient_ids)} patients for "
        f"{selected_epoch} epochs"
    )
    train_torch(
        model,
        nn.BCEWithLogitsLoss(),
        train_loader,
        None,
        lr=float(kwargs["lr"]),
        num_epochs=selected_epoch,
        early_stopping=0,
        device=device,
        modelname=output_dir.name,
        non_blocking=True,
        use_amp=bool(kwargs["use_amp"]),
        save_best=False,
        save_last=True,
        models_dir=output_dir.parent,
        last_filename="best.pth",
        history_filename="best_history.json",
    )
    best_path = output_dir / "best.pth"

    final_kwargs = dict(kwargs)
    final_kwargs["num_epochs"] = selected_epoch
    final_kwargs["signal_quality_stats"] = signal_quality_stats
    final_kwargs["save_last"] = True
    input_paths = [
        path for patient_id in patient_ids for path in patient_groups[patient_id]
    ]
    metadata = base_training_metadata(
        kwargs=final_kwargs,
        input_dim=input_dim,
        feature_cols=list(feature_cols),
        best_threshold=float(threshold_selection.threshold),
        threshold_selection=threshold_selection,
        input_paths=input_paths,
        threshold_input_paths=[],
        prepped_file=prepped_file,
    )
    metadata.update(
        {
            "base_modelname": kwargs.get("base_modelname", kwargs["modelname"]),
            "cv_method": kwargs.get("cv_method", "nested_lopo"),
            "deployment_model": True,
            "checkpoint": repo_relative_path(best_path),
            "checkpoint_policy": "global_nested_cv_epoch_threshold_full_data_fit",
            "configured_max_epochs": int(kwargs["num_epochs"]),
            "selected_fixed_epoch": selected_epoch,
            "final_training_patient_ids": list(patient_ids),
            "threshold_selection_source": "all_nested_inner_out_of_fold",
            "epoch_selection_source": "all_nested_inner_out_of_fold_event_counts",
            "selection_search_file": selection_search_file,
            "outer_test_results_used_for_selection": False,
            "seed": seed,
        }
    )
    config_path = output_dir / "train_config.json"
    write_metadata(config_path, metadata)
    selection_summary = _threshold_summary(threshold_selection)
    selection_summary.update(
        {
            "selected_fixed_epoch": selected_epoch,
            "training_patient_ids": list(patient_ids),
            "checkpoint": best_path,
            "train_config": config_path,
            "selection_search_file": selection_search_file,
            "selection_source": "all_nested_inner_out_of_fold_event_counts",
            "outer_test_results_used_for_selection": False,
        }
    )
    _write_json(output_dir / "best_selection.json", selection_summary)
    return {
        "checkpoint": repo_relative_path(best_path),
        "train_config": repo_relative_path(config_path),
        "selection_file": repo_relative_path(output_dir / "best_selection.json"),
        "selection_search_file": repo_relative_path(selection_search_file),
        "selected_epoch": selected_epoch,
        "threshold": float(threshold_selection.threshold),
        **threshold_selection.policy_metadata(),
        "training_patient_ids": list(patient_ids),
        "selection_source": "all_nested_inner_out_of_fold_event_counts",
        "outer_test_results_used_for_selection": False,
    }


def _aggregate_outer_metrics(
    output_dir: Path,
    outer_metrics: Sequence[dict[str, Any]],
    *,
    target_col: str,
    error_association_frames: int,
) -> dict[str, Any]:
    prediction_frames = []
    rows = []
    for metrics in outer_metrics:
        patient_id = str(metrics["outer_test_patient_id"])
        prediction = pd.read_parquet(
            output_dir / "outer" / patient_id / "test_all_events.parquet"
        )
        prediction["sid"] = patient_id + ":" + prediction["sid"].astype(str)
        prediction_frames.append(prediction)
        event_metrics = metrics["event_position"]
        rows.append(
            {
                "test_patient_id": patient_id,
                "selected_epoch": int(metrics["selected_fixed_epoch"]),
                "threshold": float(metrics["threshold"]),
                **event_metrics,
                "beat_detection_error": _safe_ratio(
                    int(event_metrics["unmatched_false_positive"])
                    + int(event_metrics["unmatched_false_negative"]),
                    int(event_metrics["num_predicted_events"]),
                ),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values("test_patient_id")
    summary_df.to_csv(output_dir / "outer_results.csv", index=False)
    pooled = build_event_position_summary(
        pd.concat(prediction_frames, ignore_index=True),
        target_col=target_col,
        error_association_frames=error_association_frames,
    )
    rate_keys = (
        "exact_sensitivity",
        "exact_positive_predictive_value",
        "unique_detection_error",
        "beat_sensitivity",
        "beat_positive_predictive_value",
        "beat_detection_error",
    )
    macro = {
        key: {
            "mean": float(summary_df[key].mean()),
            "std": float(summary_df[key].std(ddof=0)),
        }
        for key in rate_keys
    }
    summary = {
        "num_patients": len(rows),
        "num_outer_folds": len(rows),
        "pooled_event_position": pooled,
        "macro_patient_metrics": macro,
        "outer_results_file": repo_relative_path(output_dir / "outer_results.csv"),
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def run_nested_lopo(
    *,
    patient_groups: Mapping[str, Sequence[Path]],
    kwargs: Mapping[str, Any],
    output_dir: str | Path,
    prepped_file: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run nested leave-one-patient-out selection and outer testing."""
    base_seed = validate_training_seed(seed)
    kwargs = dict(kwargs)
    kwargs["seed"] = base_seed
    output_path = ensure_output_outside_data(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = ensure_output_outside_data(
        prepped_file or output_path / "patient_frames.parquet"
    )
    if int(kwargs["num_epochs"]) < 1:
        raise ValueError("num_epochs must be at least 1.")
    if str(kwargs["threshold_objective"]) not in {"min_error", "min_fp"}:
        raise ValueError("threshold_objective must be 'min_error' or 'min_fp'.")
    efficiency = float(kwargs["threshold_sensitivity_efficiency"])
    if not 0.0 < efficiency <= 1.0:
        raise ValueError(
            "threshold_sensitivity_efficiency must be greater than 0 and at most 1."
        )

    patient_ids = tuple(sorted(patient_groups))
    folds = nested_lopo_folds(patient_ids)
    patient_frames = load_or_build_patient_frames(
        patient_groups,
        kwargs=kwargs,
        prepped_file=cache_path,
    )
    if set(patient_frames) != set(patient_groups):
        raise ValueError(
            "Prepared patient frames do not match configured patient groups."
        )

    feature_cols, input_dim = training_feature_columns(
        int(kwargs["frame_size_raw"]),
        bool(kwargs["not_use_puls"]),
        include_ecg3=kwargs.get("ecg3_col") is not None,
        ecg_slope_mode=str(kwargs["ecg_slope_mode"]),
        signal_quality_mode=str(kwargs["signal_quality_mode"]),
    )
    device = default_torch_device()
    print(
        f"Nested LOPO: {len(patient_ids)} outer folds, "
        f"{len(patient_ids) - 1} inner folds per outer fold, device={device}"
    )
    _write_json(
        output_path / "cv_config.json",
        {
            "patient_ids": list(patient_ids),
            "patient_inputs": {
                patient_id: [repo_relative_path(path) for path in paths]
                for patient_id, paths in patient_groups.items()
            },
            "training": dict(kwargs),
            "seed": base_seed,
            "prepped_file": cache_path,
            "checkpoint_policy": "joint_nested_cv_epoch_threshold",
            "threshold_selection": "pooled_inner_out_of_fold_event_counts",
            "epoch_selection": "pooled_inner_out_of_fold_event_counts",
            "folds": [
                {
                    "test_patient_id": fold.test_patient_id,
                    "final_training_patient_ids": list(fold.final_training_patient_ids),
                    "inner_folds": [
                        {
                            "validation_patient_id": inner.validation_patient_id,
                            "training_patient_ids": list(inner.training_patient_ids),
                        }
                        for inner in fold.inner_folds
                    ],
                }
                for fold in folds
            ],
        },
    )

    outer_metrics: list[dict[str, Any]] = []
    outer_inner_searches: list[pd.DataFrame] = []
    for outer_idx, outer_fold in enumerate(folds, start=1):
        print(
            f"Outer fold {outer_idx}/{len(folds)}: "
            f"test patient {outer_fold.test_patient_id}"
        )
        outer_dir = output_path / "outer" / outer_fold.test_patient_id
        outer_dir.mkdir(parents=True, exist_ok=True)
        inner_results = []
        for inner_idx, inner_fold in enumerate(outer_fold.inner_folds, start=1):
            print(
                f"  Inner fold {inner_idx}/{len(outer_fold.inner_folds)}: "
                f"validation patient {inner_fold.validation_patient_id}"
            )
            inner_results.append(
                _train_inner_fold(
                    outer_fold=outer_fold,
                    inner_fold=inner_fold,
                    patient_frames=patient_frames,
                    kwargs=kwargs,
                    feature_cols=feature_cols,
                    input_dim=input_dim,
                    device=device,
                    outer_dir=outer_dir,
                    base_seed=base_seed,
                )
            )

        joint_selection = select_joint_inner_epoch_threshold(
            inner_results,
            patient_frames=patient_frames,
            kwargs=kwargs,
            feature_cols=feature_cols,
            input_dim=input_dim,
            device=device,
        )
        selected_epoch = joint_selection.selected_epoch
        threshold_selection = joint_selection.threshold_selection
        joint_selection.epoch_search.to_csv(
            outer_dir / "inner_epoch_selection.csv", index=False
        )
        joint_selection.threshold_search.to_csv(
            outer_dir / "inner_epoch_threshold_search.csv", index=False
        )
        outer_inner_searches.append(joint_selection.threshold_search)
        joint_selection.threshold_search.loc[
            joint_selection.threshold_search["epoch"].eq(selected_epoch)
        ].to_csv(
            outer_dir / "inner_threshold_search.csv",
            index=False,
        )
        print(
            f"  Selected joint epoch {selected_epoch} and threshold "
            f"{threshold_selection.threshold:.6f}, "
            f"fp_threshold={threshold_selection.fp_threshold:.6f}, "
            f"der_threshold={threshold_selection.der_threshold:.6f}"
        )
        selection_summary = _threshold_summary(threshold_selection)
        selection_summary.update(
            {
                "selected_fixed_epoch": selected_epoch,
                "epoch_selection_source": "nested_inner_out_of_fold_event_counts",
                "validation_patient_ids": [
                    result.fold.validation_patient_id for result in inner_results
                ],
            }
        )
        _write_json(outer_dir / "inner_selection.json", selection_summary)
        metrics = _train_final_outer_model(
            outer_fold=outer_fold,
            selected_epoch=selected_epoch,
            threshold_selection=threshold_selection,
            patient_frames=patient_frames,
            patient_groups=patient_groups,
            kwargs=kwargs,
            feature_cols=feature_cols,
            input_dim=input_dim,
            device=device,
            outer_dir=outer_dir,
            prepped_file=cache_path,
            base_seed=base_seed,
        )
        outer_metrics.append(metrics)

    summary = _aggregate_outer_metrics(
        output_path,
        outer_metrics,
        target_col=str(kwargs["target_col"]),
        error_association_frames=int(kwargs["error_association_frames"]),
    )
    selected_epoch, threshold_selection, final_search = select_global_epoch_threshold(
        outer_inner_searches,
        objective=str(kwargs["threshold_objective"]),
        sensitivity_efficiency=float(kwargs["threshold_sensitivity_efficiency"]),
    )
    selection_search_file = output_path / "best_epoch_threshold_search.csv"
    final_search.to_csv(selection_search_file, index=False)
    best_model = train_best_full_model(
        selected_epoch=selected_epoch,
        threshold_selection=threshold_selection,
        patient_frames=patient_frames,
        patient_groups=patient_groups,
        kwargs=kwargs,
        feature_cols=feature_cols,
        input_dim=input_dim,
        device=device,
        output_dir=output_path,
        prepped_file=cache_path,
        base_seed=base_seed,
        selection_search_file=selection_search_file,
    )
    summary["best_model"] = best_model
    _write_json(output_path / "summary.json", summary)
    return summary


__all__ = [
    "InnerFold",
    "OuterFold",
    "JointEpochThresholdResult",
    "ensure_output_outside_data",
    "nested_lopo_folds",
    "resolve_patient_groups",
    "run_nested_lopo",
    "select_fixed_epoch",
    "select_global_epoch_threshold",
    "select_joint_inner_epoch_threshold",
    "select_pooled_inner_threshold",
    "train_best_full_model",
]
