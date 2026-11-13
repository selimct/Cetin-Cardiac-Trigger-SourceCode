# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Hyperparameter optimization inside each outer nested-LOPO fold."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import torch

from dkfz.live import default_torch_device
from dkfz.live.event_matching import outer_association_frames
from dkfz.live.cross_validation import (
    OuterFold,
    _aggregate_outer_metrics,
    _calibrate_recovery_frames,
    _predict_inner_at_epoch,
    _threshold_summary,
    _train_final_outer_model,
    _train_inner_fold,
    _write_json,
    ensure_output_outside_data,
    load_or_build_patient_frames,
    nested_lopo_folds,
    select_global_epoch_threshold,
    select_joint_inner_epoch_threshold,
    select_pooled_inner_threshold,
    train_best_full_model,
)
from dkfz.live.hpo_config import HPOCandidate, HPOConfig, generate_hpo_candidates
from dkfz.live.io import repo_relative_path
from dkfz.live.training_pipeline import training_feature_columns
from dkfz.live.training_seed import validate_training_seed


@dataclass
class HPOTrialResult:
    """Inner-CV selections and score for one candidate in one outer fold."""

    candidate: HPOCandidate
    kwargs: dict[str, Any]
    selected_epoch: int
    comparison_threshold_selection: Any
    threshold_selection: Any
    score: float
    trial_dir: Path
    trimmed_validation_rows: int
    epoch_threshold_search: pd.DataFrame
    inner_results: tuple = ()


@dataclass
class FinalHPOSelection:
    """One candidate's selection pooled across every nested inner search."""

    candidate: HPOCandidate
    kwargs: dict[str, Any]
    selected_epoch: int
    comparison_threshold_selection: Any
    threshold_selection: Any
    score: float
    epoch_threshold_search: pd.DataFrame
    inner_results: tuple = ()


def _common_validation_predictions(
    prediction_frames: Sequence[pd.DataFrame],
    *,
    candidate_window_size: int,
    maximum_window_size: int,
) -> tuple[list[pd.DataFrame], int]:
    """Trim candidate predictions to the validation region common to all windows."""
    trim = int(maximum_window_size) - int(candidate_window_size)
    if trim < 0:
        raise ValueError("Candidate window exceeds the configured maximum window.")
    selected: list[pd.DataFrame] = []
    trimmed_rows = 0
    for frame in prediction_frames:
        if trim == 0:
            common = frame.copy()
        else:
            ordinal = frame.groupby("sid", sort=False).cumcount()
            common = frame.loc[ordinal >= trim].copy()
            trimmed_rows += int((ordinal < trim).sum())
        if common.empty:
            raise ValueError(
                "An inner validation fold has no rows in the common HPO evaluation region."
            )
        selected.append(common)
    return selected, trimmed_rows


def _trial_sort_key(result: HPOTrialResult, direction: str) -> tuple[float, int]:
    primary = result.score if direction == "minimize" else -result.score
    return primary, result.candidate.trial_id


def _run_hpo_trial(
    *,
    outer_fold: OuterFold,
    candidate: HPOCandidate,
    base_kwargs: Mapping[str, Any],
    patient_frames: Mapping[str, pd.DataFrame],
    feature_cols: Sequence[str],
    input_dim: int,
    device: torch.device,
    outer_dir: Path,
    maximum_window_size: int,
    base_seed: int,
    hpo_config: HPOConfig,
) -> HPOTrialResult:
    trial_kwargs = dict(base_kwargs)
    trial_kwargs.update(candidate.parameters)
    trial_dir = outer_dir / "search" / f"trial_{candidate.trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    inner_results = []
    for inner_idx, inner_fold in enumerate(outer_fold.inner_folds, start=1):
        print(
            f"      Inner fold {inner_idx}/{len(outer_fold.inner_folds)}: "
            f"validation patient {inner_fold.validation_patient_id}"
        )
        inner_results.append(
            _train_inner_fold(
                outer_fold=outer_fold,
                inner_fold=inner_fold,
                patient_frames=patient_frames,
                kwargs=trial_kwargs,
                feature_cols=feature_cols,
                input_dim=input_dim,
                device=device,
                outer_dir=trial_dir,
                base_seed=base_seed,
            )
        )

    trimmed_rows_by_epoch: dict[int, int] = {}

    def common_region(
        prediction_frames: Sequence[pd.DataFrame], epoch: int
    ) -> Sequence[pd.DataFrame]:
        common_predictions, trimmed_rows = _common_validation_predictions(
            prediction_frames,
            candidate_window_size=int(trial_kwargs["window_size"]),
            maximum_window_size=maximum_window_size,
        )
        trimmed_rows_by_epoch[epoch] = trimmed_rows
        return common_predictions

    joint_selection = select_joint_inner_epoch_threshold(
        inner_results,
        patient_frames=patient_frames,
        kwargs=trial_kwargs,
        feature_cols=feature_cols,
        input_dim=input_dim,
        device=device,
        prediction_transform=common_region,
    )
    selected_epoch = joint_selection.selected_epoch
    comparison_threshold_selection = joint_selection.threshold_selection
    prediction_frames = joint_selection.selected_prediction_frames
    trimmed_rows = trimmed_rows_by_epoch[selected_epoch]
    joint_selection.epoch_search.to_csv(
        trial_dir / "inner_epoch_selection.csv", index=False
    )
    joint_selection.threshold_search.to_csv(
        trial_dir / "hpo_epoch_threshold_search.csv", index=False
    )
    joint_selection.threshold_search.loc[
        joint_selection.threshold_search["epoch"].eq(selected_epoch)
    ].to_csv(
        trial_dir / "hpo_comparison_threshold_search.csv",
        index=False,
    )
    threshold_selection = select_pooled_inner_threshold(
        prediction_frames,
        target_col=str(trial_kwargs["target_col"]),
        objective=str(trial_kwargs["threshold_objective"]),
        sensitivity_efficiency=float(trial_kwargs["threshold_sensitivity_efficiency"]),
        holdoff_frames=int(trial_kwargs["min_distance"]),
        error_association_frames=int(trial_kwargs["error_association_frames"]),
        outer_error_association_frames=outer_association_frames(
            trial_kwargs["raw_hz"], trial_kwargs["frame_size_raw"]
        ),
        recovery_max_fp_increase=float(
            trial_kwargs.get("recovery_max_fp_increase", 0.10)
        ),
        recovery_min_extra_fp_per_minute=float(
            trial_kwargs.get("recovery_min_extra_fp_per_minute", 0.5)
        ),
        frame_hz=float(trial_kwargs["raw_hz"]) / float(trial_kwargs["frame_size_raw"]),
    )
    pd.DataFrame(threshold_selection.search_rows).to_csv(
        trial_dir / "inner_threshold_search.csv", index=False
    )
    if hpo_config.selection_metric not in comparison_threshold_selection.metrics:
        raise KeyError(
            f"Threshold metrics do not contain HPO selection metric "
            f"{hpo_config.selection_metric!r}."
        )
    score = float(comparison_threshold_selection.metrics[hpo_config.selection_metric])
    if not math.isfinite(score):
        raise ValueError(
            f"HPO trial {candidate.trial_id} produced a non-finite selection score."
        )

    selection_summary = _threshold_summary(comparison_threshold_selection)
    selection_summary.update(
        {
            "trial_id": candidate.trial_id,
            "trial_name": candidate.name,
            "is_baseline": candidate.is_baseline,
            "parameters": candidate.parameters,
            "selected_fixed_epoch": selected_epoch,
            "epoch_selection_source": "joint_inner_epoch_threshold_event_counts",
            "selection_metric": hpo_config.selection_metric,
            "selection_direction": hpo_config.selection_direction,
            "selection_score": score,
            "final_full_validation_threshold": _threshold_summary(threshold_selection),
            "maximum_candidate_window_size": maximum_window_size,
            "trimmed_validation_rows": trimmed_rows,
            "validation_patient_ids": [
                result.fold.validation_patient_id for result in inner_results
            ],
        }
    )
    _write_json(trial_dir / "trial.json", selection_summary)
    return HPOTrialResult(
        candidate=candidate,
        kwargs=trial_kwargs,
        selected_epoch=selected_epoch,
        comparison_threshold_selection=comparison_threshold_selection,
        threshold_selection=threshold_selection,
        score=score,
        trial_dir=trial_dir,
        trimmed_validation_rows=trimmed_rows,
        epoch_threshold_search=joint_selection.threshold_search,
        inner_results=tuple(inner_results),
    )


def _write_outer_hpo_selection(
    outer_dir: Path,
    results: Sequence[HPOTrialResult],
    selected: HPOTrialResult,
    config: HPOConfig,
) -> None:
    rows = []
    for result in results:
        metrics = result.comparison_threshold_selection.metrics
        rows.append(
            {
                "trial_id": result.candidate.trial_id,
                "trial_name": result.candidate.name,
                "is_baseline": result.candidate.is_baseline,
                "selected": result is selected,
                "selected_epoch": result.selected_epoch,
                "comparison_threshold": (
                    result.comparison_threshold_selection.threshold
                ),
                "final_full_validation_threshold": result.threshold_selection.threshold,
                "selection_metric": config.selection_metric,
                "selection_direction": config.selection_direction,
                "selection_score": result.score,
                "trimmed_validation_rows": result.trimmed_validation_rows,
                **{
                    f"param_{name}": value
                    for name, value in result.candidate.parameters.items()
                },
                **metrics,
            }
        )
    pd.DataFrame(rows).sort_values("trial_id").to_csv(
        outer_dir / "hyperparameter_search.csv", index=False
    )
    _write_json(
        outer_dir / "selected_hyperparameters.json",
        {
            "trial_id": selected.candidate.trial_id,
            "trial_name": selected.candidate.name,
            "is_baseline": selected.candidate.is_baseline,
            "parameters": selected.candidate.parameters,
            "selected_fixed_epoch": selected.selected_epoch,
            "threshold": selected.threshold_selection.threshold,
            "comparison_threshold": (selected.comparison_threshold_selection.threshold),
            "selection_metric": config.selection_metric,
            "selection_direction": config.selection_direction,
            "selection_score": selected.score,
            "trial_file": selected.trial_dir / "trial.json",
        },
    )


def _annotate_final_hpo_config(
    outer_dir: Path,
    *,
    selected: HPOTrialResult,
    hpo_config: HPOConfig,
    hpo_config_path: Path,
    hpo_config_snapshot: Path,
) -> None:
    path = outer_dir / "train_config.json"
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata.update(
        {
            "cv_method": "nested_lopo_hpo",
            "hpo_selected": True,
            "hpo_trial_id": selected.candidate.trial_id,
            "hpo_trial_name": selected.candidate.name,
            "hpo_is_baseline": selected.candidate.is_baseline,
            "hpo_parameters": selected.candidate.parameters,
            "hpo_selection_metric": hpo_config.selection_metric,
            "hpo_selection_direction": hpo_config.selection_direction,
            "hpo_selection_score": selected.score,
            "hpo_comparison_threshold": selected.comparison_threshold_selection.threshold,
            "threshold_selection_source": "nested_inner_out_of_fold_full_region",
            "epoch_selection_source": "joint_inner_epoch_threshold_event_counts",
            "hpo_search_config": hpo_config_snapshot,
            "hpo_source_config": repo_relative_path(hpo_config_path),
            "hpo_search_file": outer_dir / "hyperparameter_search.csv",
            "hpo_selected_parameters_file": (
                outer_dir / "selected_hyperparameters.json"
            ),
        }
    )
    _write_json(path, metadata)


def _select_final_hpo_candidate(
    *,
    results_by_trial: Mapping[int, Sequence[HPOTrialResult]],
    hpo_config: HPOConfig,
    output_dir: Path,
) -> FinalHPOSelection:
    """Select one deployable candidate using every nested inner-CV search."""
    final_results: list[FinalHPOSelection] = []
    for trial_id in sorted(results_by_trial):
        results = list(results_by_trial[trial_id])
        if not results:
            raise ValueError(f"HPO trial {trial_id} has no outer-fold results.")
        selected_epoch, threshold_selection, search = select_global_epoch_threshold(
            [result.epoch_threshold_search for result in results],
            objective=str(results[0].kwargs["threshold_objective"]),
            sensitivity_efficiency=float(
                results[0].kwargs["threshold_sensitivity_efficiency"]
            ),
        )
        if hpo_config.selection_metric not in threshold_selection.metrics:
            raise KeyError(
                f"Threshold metrics do not contain HPO selection metric "
                f"{hpo_config.selection_metric!r}."
            )
        score = float(threshold_selection.metrics[hpo_config.selection_metric])
        if not math.isfinite(score):
            raise ValueError(
                f"Pooled HPO trial {trial_id} produced a non-finite selection score."
            )
        final_results.append(
            FinalHPOSelection(
                candidate=results[0].candidate,
                kwargs=results[0].kwargs,
                selected_epoch=selected_epoch,
                comparison_threshold_selection=threshold_selection,
                threshold_selection=threshold_selection,
                score=score,
                epoch_threshold_search=search,
                inner_results=tuple(
                    inner for result in results for inner in result.inner_results
                ),
            )
        )

    selected = min(
        final_results,
        key=lambda result: _trial_sort_key(result, hpo_config.selection_direction),
    )
    comparison_rows = []
    search_frames = []
    for result in final_results:
        metrics = result.threshold_selection.metrics
        comparison_rows.append(
            {
                "trial_id": result.candidate.trial_id,
                "trial_name": result.candidate.name,
                "is_baseline": result.candidate.is_baseline,
                "selected": result is selected,
                "selected_epoch": result.selected_epoch,
                "threshold": result.threshold_selection.threshold,
                "selection_metric": hpo_config.selection_metric,
                "selection_direction": hpo_config.selection_direction,
                "selection_score": result.score,
                **{
                    f"param_{name}": value
                    for name, value in result.candidate.parameters.items()
                },
                **metrics,
            }
        )
        search = result.epoch_threshold_search.copy()
        search.insert(0, "trial_id", result.candidate.trial_id)
        search.insert(1, "trial_name", result.candidate.name)
        search.insert(2, "candidate_selected", result is selected)
        search_frames.append(search)

    pd.DataFrame(comparison_rows).sort_values("trial_id").to_csv(
        output_dir / "best_hyperparameter_search.csv", index=False
    )
    pd.concat(search_frames, ignore_index=True).to_csv(
        output_dir / "best_hpo_epoch_threshold_search.csv", index=False
    )
    _write_json(
        output_dir / "best_selected_hyperparameters.json",
        {
            "trial_id": selected.candidate.trial_id,
            "trial_name": selected.candidate.name,
            "is_baseline": selected.candidate.is_baseline,
            "parameters": selected.candidate.parameters,
            "selected_fixed_epoch": selected.selected_epoch,
            "threshold": selected.threshold_selection.threshold,
            "comparison_threshold": selected.comparison_threshold_selection.threshold,
            "selection_metric": hpo_config.selection_metric,
            "selection_direction": hpo_config.selection_direction,
            "selection_score": selected.score,
            "selection_source": "all_nested_inner_out_of_fold_common_region",
            "outer_test_results_used_for_selection": False,
        },
    )
    return selected


def _calibrate_final_hpo_selection(
    selected: FinalHPOSelection,
    *,
    patient_frames: Mapping[str, pd.DataFrame],
    feature_cols: Sequence[str],
    input_dim: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[pd.DataFrame, ...]:
    """Calibrate the winning candidate on its full validation region.

    Candidate ranking and epoch selection remain based on the common-region
    search. Replay only the winner's selected epoch, without further training,
    just as the per-outer-fold second calibration uses untrimmed predictions.
    """
    kwargs = selected.kwargs
    prediction_frames = tuple(
        _predict_inner_at_epoch(
            result,
            selected_epoch=selected.selected_epoch,
            patient_frames=patient_frames,
            kwargs=kwargs,
            feature_cols=feature_cols,
            input_dim=input_dim,
            device=device,
            save_predictions=False,
        )
        for result in selected.inner_results
    )
    # Calibrate recovery below, once, while retaining its search artifact.
    primary_frames = [
        frame.drop(columns=["recovery_prob"], errors="ignore")
        for frame in prediction_frames
    ]
    selection = select_pooled_inner_threshold(
        primary_frames,
        target_col=str(kwargs["target_col"]),
        objective=str(kwargs["threshold_objective"]),
        sensitivity_efficiency=float(kwargs["threshold_sensitivity_efficiency"]),
        holdoff_frames=int(kwargs["min_distance"]),
        error_association_frames=int(kwargs["error_association_frames"]),
        outer_error_association_frames=outer_association_frames(
            kwargs["raw_hz"], kwargs["frame_size_raw"]
        ),
    )
    selected.threshold_selection = _calibrate_recovery_frames(
        prediction_frames,
        selection,
        kwargs,
        output_dir / "best_recovery_threshold_search.csv",
    )
    pd.DataFrame(selected.threshold_selection.search_rows).to_csv(
        output_dir / "best_threshold_search.csv", index=False
    )
    path = output_dir / "best_selected_hyperparameters.json"
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata.update(
        {
            "threshold": selected.threshold_selection.threshold,
            "comparison_threshold": selected.comparison_threshold_selection.threshold,
            "threshold_selection_source": "all_nested_inner_out_of_fold_full_region",
            "final_full_validation_threshold": _threshold_summary(
                selected.threshold_selection
            ),
        }
    )
    _write_json(path, metadata)
    return prediction_frames


def _annotate_best_hpo_config(
    output_dir: Path,
    *,
    selected: FinalHPOSelection,
    hpo_config: HPOConfig,
    hpo_config_path: Path,
) -> None:
    path = output_dir / "train_config.json"
    with path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    metadata.update(
        {
            "cv_method": "nested_lopo_hpo",
            "hpo_selected": True,
            "hpo_trial_id": selected.candidate.trial_id,
            "hpo_trial_name": selected.candidate.name,
            "hpo_is_baseline": selected.candidate.is_baseline,
            "hpo_parameters": selected.candidate.parameters,
            "hpo_selection_metric": hpo_config.selection_metric,
            "hpo_selection_direction": hpo_config.selection_direction,
            "hpo_selection_score": selected.score,
            "hpo_comparison_threshold": selected.comparison_threshold_selection.threshold,
            "threshold_selection_source": "all_nested_inner_out_of_fold_full_region",
            "epoch_selection_source": (
                "all_nested_inner_out_of_fold_common_region_event_counts"
            ),
            "hpo_search_config": output_dir / "hpo_config.json",
            "hpo_source_config": repo_relative_path(hpo_config_path),
            "hpo_search_file": output_dir / "best_hyperparameter_search.csv",
            "hpo_selected_parameters_file": (
                output_dir / "best_selected_hyperparameters.json"
            ),
        }
    )
    _write_json(path, metadata)


def run_nested_lopo_hpo(
    *,
    patient_groups: Mapping[str, Sequence[Path]],
    kwargs: Mapping[str, Any],
    hpo_config: HPOConfig,
    hpo_config_path: str | Path,
    output_dir: str | Path,
    prepped_file: str | Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Run per-outer-fold HPO using only inner patient-held-out predictions."""
    base_seed = validate_training_seed(seed)
    kwargs = dict(kwargs)
    if kwargs.get("recovery_only"):
        raise ValueError(
            "Frozen recovery-head selection uses run/train_recovery_lopo.py, not architecture HPO"
        )
    kwargs["seed"] = base_seed
    output_path = ensure_output_outside_data(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    cache_path = ensure_output_outside_data(
        prepped_file or output_path / "patient_frames.parquet"
    )
    if int(kwargs["num_epochs"]) < 1:
        raise ValueError("num_epochs must be at least one.")

    candidates = generate_hpo_candidates(hpo_config, kwargs)
    candidate_kwargs = [
        dict(kwargs, **candidate.parameters) for candidate in candidates
    ]
    maximum_window_size = max(int(values["window_size"]) for values in candidate_kwargs)
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
    config_path = Path(hpo_config_path).resolve()
    _write_json(output_path / "hpo_config.json", hpo_config.raw)
    _write_json(
        output_path / "cv_config.json",
        {
            "cv_method": "nested_lopo_hpo",
            "patient_ids": list(patient_ids),
            "patient_inputs": {
                patient_id: [repo_relative_path(path) for path in paths]
                for patient_id, paths in patient_groups.items()
            },
            "training": kwargs,
            "seed": base_seed,
            "hpo_seed": hpo_config.seed,
            "hpo_config": config_path,
            "selection_metric": hpo_config.selection_metric,
            "selection_direction": hpo_config.selection_direction,
            "epoch_selection": "joint_inner_epoch_threshold_event_counts",
            "maximum_candidate_window_size": maximum_window_size,
            "candidates": [
                {
                    "trial_id": candidate.trial_id,
                    "name": candidate.name,
                    "is_baseline": candidate.is_baseline,
                    "parameters": candidate.parameters,
                }
                for candidate in candidates
            ],
            "folds": [
                {
                    "test_patient_id": fold.test_patient_id,
                    "final_training_patient_ids": list(fold.final_training_patient_ids),
                    "validation_patient_ids": [
                        inner.validation_patient_id for inner in fold.inner_folds
                    ],
                }
                for fold in folds
            ],
        },
    )
    print(
        f"Nested LOPO HPO: {len(patient_ids)} outer folds, "
        f"{len(candidates)} candidates, {len(patient_ids) - 1} inner folds per "
        f"candidate, device={device}"
    )

    outer_metrics: list[dict[str, Any]] = []
    selected_by_patient: dict[str, Any] = {}
    results_by_trial: dict[int, list[HPOTrialResult]] = {
        candidate.trial_id: [] for candidate in candidates
    }
    for outer_idx, outer_fold in enumerate(folds, start=1):
        print(
            f"Outer fold {outer_idx}/{len(folds)}: test patient "
            f"{outer_fold.test_patient_id}"
        )
        outer_dir = output_path / "outer" / outer_fold.test_patient_id
        outer_dir.mkdir(parents=True, exist_ok=True)
        trial_results = []
        for candidate_idx, candidate in enumerate(candidates, start=1):
            print(
                f"  Candidate {candidate_idx}/{len(candidates)} "
                f"(trial_{candidate.trial_id:03d}): {candidate.parameters}"
            )
            trial_results.append(
                _run_hpo_trial(
                    outer_fold=outer_fold,
                    candidate=candidate,
                    base_kwargs=kwargs,
                    patient_frames=patient_frames,
                    feature_cols=feature_cols,
                    input_dim=input_dim,
                    device=device,
                    outer_dir=outer_dir,
                    maximum_window_size=maximum_window_size,
                    base_seed=base_seed,
                    hpo_config=hpo_config,
                )
            )
        for result in trial_results:
            results_by_trial[result.candidate.trial_id].append(result)

        selected = min(
            trial_results,
            key=lambda result: _trial_sort_key(result, hpo_config.selection_direction),
        )
        _write_outer_hpo_selection(outer_dir, trial_results, selected, hpo_config)
        print(
            f"  Selected trial_{selected.candidate.trial_id:03d}: "
            f"{hpo_config.selection_metric}={selected.score:.6f}, "
            f"epoch={selected.selected_epoch}, "
            f"threshold={selected.threshold_selection.threshold:.6f}"
        )
        metrics = _train_final_outer_model(
            outer_fold=outer_fold,
            selected_epoch=selected.selected_epoch,
            threshold_selection=selected.threshold_selection,
            patient_frames=patient_frames,
            patient_groups=patient_groups,
            kwargs=selected.kwargs,
            feature_cols=feature_cols,
            input_dim=input_dim,
            device=device,
            outer_dir=outer_dir,
            prepped_file=cache_path,
            base_seed=base_seed,
        )
        _annotate_final_hpo_config(
            outer_dir,
            selected=selected,
            hpo_config=hpo_config,
            hpo_config_path=config_path,
            hpo_config_snapshot=output_path / "hpo_config.json",
        )
        outer_metrics.append(metrics)
        selected_by_patient[outer_fold.test_patient_id] = {
            "trial_id": selected.candidate.trial_id,
            "parameters": selected.candidate.parameters,
            "selected_epoch": selected.selected_epoch,
            "threshold": selected.threshold_selection.threshold,
            "selection_score": selected.score,
        }

    summary = _aggregate_outer_metrics(
        output_path,
        outer_metrics,
        target_col=str(kwargs["target_col"]),
        error_association_frames=int(kwargs["error_association_frames"]),
    )
    selected_final = _select_final_hpo_candidate(
        results_by_trial=results_by_trial,
        hpo_config=hpo_config,
        output_dir=output_path,
    )
    validation_frames = _calibrate_final_hpo_selection(
        selected_final,
        patient_frames=patient_frames,
        feature_cols=feature_cols,
        input_dim=input_dim,
        device=device,
        output_dir=output_path,
    )
    selection_search_file = output_path / "best_threshold_search.csv"
    if selected_final.kwargs.get("recovery_head"):
        from dkfz.live.recovery_calibration import save_recovery_validation

        selected_final.kwargs = dict(
            selected_final.kwargs,
            _recovery_validation_log=save_recovery_validation(
                validation_frames,
                output_path / "recovery_validation_predictions.parquet",
                epoch=selected_final.selected_epoch,
            ),
        )
    best_model = train_best_full_model(
        selected_epoch=selected_final.selected_epoch,
        threshold_selection=selected_final.threshold_selection,
        patient_frames=patient_frames,
        patient_groups=patient_groups,
        kwargs=selected_final.kwargs,
        feature_cols=feature_cols,
        input_dim=input_dim,
        device=device,
        output_dir=output_path,
        prepped_file=cache_path,
        base_seed=base_seed,
        selection_search_file=selection_search_file,
    )
    _annotate_best_hpo_config(
        output_path,
        selected=selected_final,
        hpo_config=hpo_config,
        hpo_config_path=config_path,
    )
    best_model.update(
        {
            "hpo_trial_id": selected_final.candidate.trial_id,
            "hpo_trial_name": selected_final.candidate.name,
            "hpo_parameters": selected_final.candidate.parameters,
            "hpo_selection_metric": hpo_config.selection_metric,
            "hpo_selection_score": selected_final.score,
            "hpo_comparison_threshold": (
                selected_final.comparison_threshold_selection.threshold
            ),
            "threshold_selection_source": "all_nested_inner_out_of_fold_full_region",
            "selection_source": (
                "all_nested_inner_out_of_fold_common_region_event_counts"
            ),
        }
    )
    summary.update(
        {
            "cv_method": "nested_lopo_hpo",
            "num_hpo_candidates": len(candidates),
            "hpo_selection_metric": hpo_config.selection_metric,
            "hpo_selection_direction": hpo_config.selection_direction,
            "selected_hyperparameters_by_test_patient": selected_by_patient,
            "best_model": best_model,
        }
    )
    _write_json(output_path / "summary.json", summary)
    _write_json(output_path / "hpo_summary.json", summary)
    return summary


__all__ = ["FinalHPOSelection", "HPOTrialResult", "run_nested_lopo_hpo"]
