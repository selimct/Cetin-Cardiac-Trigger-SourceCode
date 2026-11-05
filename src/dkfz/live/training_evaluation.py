# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Validation artifact builders for live ECG training."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dkfz.live.eval_metrics import (
    build_prediction_metrics_report,
    format_prediction_metrics_report,
)
from dkfz.live.io import repo_relative_path
from dkfz.live.prediction_logs import order_prediction_log_columns
from dkfz.live.threshold_analysis import (
    DEFAULT_ERROR_PLOTS,
    DEFAULT_ERROR_WINDOW_FRAMES,
    classify_validation_events,
    write_error_examples,
    write_threshold_dashboard,
)
from dkfz.live.triggers import StreamingHoldoffLocalMaxTrigger, TriggerEvent
from dkfz.core.dataset import SlidingWindowDataset
from dkfz.core.network_helpers import ThresholdSelection
from dkfz.live.event_matching import EXACT_TOLERANCE_FRAMES, METRICS_SCHEMA_VERSION


def _extract_model_tensor(model_output: torch.Tensor | tuple | list) -> torch.Tensor:
    """Return the tensor output from models that may also return recurrent state."""
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    if not torch.is_tensor(model_output):
        raise TypeError("Model output must be a tensor or tuple/list of tensors.")
    return model_output


def predict_validation_probabilities(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    """Return sigmoid probabilities for validation windows in loader order."""
    model = model.to(device).eval()
    amp_enabled = bool(use_amp and device.type == "cuda")
    probs: list[np.ndarray] = []

    with torch.inference_mode():
        for batch in loader:
            x = batch[0].to(device, non_blocking=True)
            batch_size = int(x.shape[0])
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits = _extract_model_tensor(model(x)).reshape(batch_size, -1)
            if logits.shape[1] != 1:
                raise ValueError(
                    f"Expected one logit per sample, got logits.shape={tuple(logits.shape)}"
                )
            probs.append(torch.sigmoid(logits[:, 0].float()).cpu().numpy())

    if not probs:
        return np.asarray([], dtype=np.float32)
    return np.concatenate(probs).astype(np.float32)


def validation_window_rows(dataset: SlidingWindowDataset) -> pd.DataFrame:
    """Return the frame rows corresponding to each validation window output."""
    data = dataset.__getdataframe__().reset_index(drop=True)
    end_indices = [
        int(start) + int(dataset.window_size) - 1 for start in dataset.starts
    ]
    return data.iloc[end_indices].reset_index(drop=True).copy()


def _frame_end_sample_idx(row: pd.Series, frame_size_raw: int) -> int:
    if "_raw_frame_end_idx" in row:
        return int(row["_raw_frame_end_idx"])
    return int(row["raw_idx"]) + int(frame_size_raw) - 1


def _frame_time_ms(row: pd.Series) -> float | None:
    if "time_ms" not in row or pd.isna(row["time_ms"]):
        return None
    return float(row["time_ms"])


def build_live_eval_outputs(
    model: torch.nn.Module,
    loader: DataLoader,
    dataset: SlidingWindowDataset,
    threshold: float,
    holdoff_frames: int,
    frame_size_raw: int,
    feature_cols: list[str],
    target_col: str,
    device: torch.device,
    use_amp: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build prediction-log and event-log DataFrames for validation windows.

    This mirrors the online trigger path: every probability belongs to the last
    frame of a causal validation window, and events are produced by the same
    streaming holdoff/local-maximum trigger used during live prediction.
    """
    probs = predict_validation_probabilities(
        model=model,
        loader=loader,
        device=device,
        use_amp=use_amp,
    )
    log_df = validation_window_rows(dataset)
    if len(log_df) != len(probs):
        raise ValueError(
            f"Validation row/probability mismatch: {len(log_df)=}, {len(probs)=}."
        )
    if target_col not in log_df.columns:
        raise KeyError(f"Validation rows are missing target column {target_col!r}.")

    log_df["prob"] = probs
    log_df["pred"] = 0
    log_df["trigger_emitted_now"] = False
    log_df["feature_dim"] = len(feature_cols)

    events: list[TriggerEvent] = []
    for sid, group in log_df.groupby("sid", sort=False):
        trigger = StreamingHoldoffLocalMaxTrigger(
            threshold=float(threshold),
            holdoff_frames=int(holdoff_frames),
        )
        last_row = None
        for idx, row in group.iterrows():
            last_row = row
            event = trigger.update(
                probability=float(row["prob"]),
                current_frame_idx_50hz=int(row["frame_idx"]),
                current_sample_idx_400hz=_frame_end_sample_idx(row, frame_size_raw),
                current_time_ms=_frame_time_ms(row),
                sid=sid,
            )
            if event is not None:
                events.append(event)
                log_df.loc[idx, "trigger_emitted_now"] = True

        if last_row is not None:
            final_event = trigger.flush(
                current_frame_idx_50hz=int(last_row["frame_idx"]),
                current_sample_idx_400hz=_frame_end_sample_idx(
                    last_row, frame_size_raw
                ),
                current_time_ms=_frame_time_ms(last_row),
                sid=sid,
            )
            if final_event is not None:
                events.append(final_event)

    if events:
        peak_keys = {(event.sid, event.peak_frame_idx_50hz) for event in events}
        row_keys = list(zip(log_df["sid"], log_df["frame_idx"]))
        log_df.loc[[key in peak_keys for key in row_keys], "pred"] = 1

    events_df = pd.DataFrame(
        [asdict(event) for event in events],
        columns=list(TriggerEvent.__annotations__.keys()),
    )
    log_df = order_prediction_log_columns(log_df, feature_cols)
    return log_df, events_df


def write_live_eval_artifacts(
    model_dir: Path,
    prediction_df: pd.DataFrame,
    events_df: pd.DataFrame,
    threshold_search_df: pd.DataFrame,
    target_col: str,
    threshold: float,
    holdoff_frames: int,
    error_association_frames: int,
    prefix: str = "train",
    metrics_dir: Path | None = None,
    metrics_prefix: str | None = None,
    threshold_selection: ThresholdSelection | None = None,
    outer_error_association_frames: int = 2,
) -> None:
    """Write validation artifacts with the same shape as live prediction outputs."""
    prefix = prefix or "train"
    metrics_prefix = metrics_prefix or prefix
    metrics_dir = (
        Path(metrics_dir) if metrics_dir is not None else model_dir / "metrics"
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = model_dir / f"{prefix}_all_events.parquet"
    events_path = model_dir / f"{prefix}_trigger_events.parquet"
    metrics_path = metrics_dir / f"{metrics_prefix}_metrics.json"
    threshold_search_path = metrics_dir / f"{metrics_prefix}_threshold_search.csv"
    threshold_dashboard_path = metrics_dir / f"{metrics_prefix}_threshold_dashboard.pdf"
    threshold_errors_path = metrics_dir / f"{metrics_prefix}_threshold_errors.pdf"

    if threshold_selection is not None:
        prediction_df = prediction_df.assign(
            fp_threshold=threshold_selection.fp_threshold,
            der_threshold=threshold_selection.der_threshold,
        )
    prediction_df.to_parquet(prediction_path, compression="snappy")
    events_df.to_parquet(events_path, compression="snappy")

    metrics = {
        "metrics_schema_version": METRICS_SCHEMA_VERSION,
        "threshold": float(threshold),
        "holdoff_frames": int(holdoff_frames),
        "exact_tolerance_frames": EXACT_TOLERANCE_FRAMES,
        "error_association_frames": int(error_association_frames),
        "num_all_event_rows": int(len(prediction_df)),
        "num_trigger_event_rows": int(len(events_df)),
        "num_target_frames": int(np.sum(prediction_df[target_col].to_numpy() == 1)),
        "num_threshold_search_rows": int(len(threshold_search_df)),
    }
    if threshold_selection is not None:
        metrics.update(
            {
                **threshold_selection.policy_metadata(),
                "threshold_objective": threshold_selection.objective,
                "threshold_sensitivity_efficiency": (
                    threshold_selection.sensitivity_efficiency
                ),
                "threshold_best_attainable_exact_sensitivity": (
                    threshold_selection.best_attainable_sensitivity
                ),
                "threshold_effective_exact_sensitivity_floor": (
                    threshold_selection.effective_sensitivity_floor
                ),
                "num_threshold_candidates": len(threshold_selection.search_rows),
                "threshold_validation": threshold_selection.metrics,
            }
        )
        pd.DataFrame(threshold_selection.search_rows).to_csv(
            threshold_search_path,
            index=False,
        )

    prediction_metrics = build_prediction_metrics_report(
        prediction_df=prediction_df,
        target_col=target_col,
        error_association_frames=int(error_association_frames),
        outer_error_association_frames=outer_error_association_frames,
        threshold=float(threshold),
        events_df=events_df,
    )
    metrics["event_position"] = prediction_metrics["event_position"]
    metrics["emission_timing"] = prediction_metrics.get("emission_timing")
    report_text = format_prediction_metrics_report(prediction_metrics)

    if threshold_selection is not None:
        classified_events = classify_validation_events(
            prediction_df,
            threshold=float(threshold),
            target_col=target_col,
            holdoff_frames=int(holdoff_frames),
            error_association_frames=int(error_association_frames),
            outer_error_association_frames=outer_error_association_frames,
        )
        write_threshold_dashboard(
            threshold_dashboard_path,
            selection=threshold_selection,
            presentation="training",
        )
        num_error_plots = write_error_examples(
            threshold_errors_path,
            validation_df=prediction_df,
            event_df=classified_events,
            threshold=float(threshold),
            target_col=target_col,
            max_plots=DEFAULT_ERROR_PLOTS,
            window_frames=DEFAULT_ERROR_WINDOW_FRAMES,
        )
        metrics.update(
            {
                "threshold_dashboard_file": repo_relative_path(
                    threshold_dashboard_path
                ),
                "threshold_error_examples_file": repo_relative_path(
                    threshold_errors_path
                ),
                "num_threshold_error_plots": num_error_plots,
            }
        )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Wrote validation event rows to {repo_relative_path(prediction_path)}")
    print(f"Wrote validation trigger events to {repo_relative_path(events_path)}")
    print(f"Wrote validation metrics to {repo_relative_path(metrics_path)}")
    if threshold_selection is not None:
        print(
            "Wrote threshold search to " f"{repo_relative_path(threshold_search_path)}"
        )
        print(
            "Wrote threshold dashboard to "
            f"{repo_relative_path(threshold_dashboard_path)}"
        )
        print(
            "Wrote threshold error examples to "
            f"{repo_relative_path(threshold_errors_path)}"
        )
    if report_text:
        print(report_text)
