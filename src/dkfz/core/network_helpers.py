# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Helper functions for live-oriented PyTorch training and evaluation."""

from __future__ import annotations

import hashlib
import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dkfz.live.event_matching import (
    DEFAULT_ERROR_ASSOCIATION_FRAMES,
    compute_event_position_metrics,
    ERROR_WEIGHTS,
)


@dataclass
class TrainEpochMetrics:
    """Metrics captured for one training epoch."""

    epoch: int
    train_loss: float
    val_loss: float | None
    lr: float
    seconds: float
    improved: bool


THRESHOLD_OBJECTIVES = ("min_error", "min_fp")


@dataclass(frozen=True)
class ThresholdSelection:
    """Selected operating threshold and its complete validation search."""

    threshold: float
    objective: str
    sensitivity_efficiency: float
    best_attainable_sensitivity: float
    effective_sensitivity_floor: float
    metrics: dict[str, float | int | None]
    search_rows: tuple[dict[str, float | int | bool | str | None], ...]
    fp_threshold: float | None = None
    der_threshold: float | None = None
    recovery_threshold: float | None = None
    recovery_max_fp_increase: float | None = None
    recovery_min_extra_fp_per_minute: float | None = None
    recovery_validation_minutes: float | None = None
    recovery_validation_metrics: dict | None = None

    def policy_metadata(self) -> dict:
        return {
            **(
                {
                    "recovery_threshold": self.recovery_threshold,
                    "recovery_max_fp_increase": self.recovery_max_fp_increase,
                    "recovery_min_extra_fp_per_minute": self.recovery_min_extra_fp_per_minute,
                    "recovery_validation_minutes": self.recovery_validation_minutes,
                    "recovery_error_policy": "reduce_fn_and_fp_plus_fn",
                    "recovery_validation_metrics": self.recovery_validation_metrics,
                }
                if self.recovery_max_fp_increase is not None
                else {}
            ),
            "fp_threshold": self.fp_threshold,
            "der_threshold": self.der_threshold,
            "threshold_policy_version": 3,
            "threshold_error_weights": dict(ERROR_WEIGHTS),
            "threshold_calibration_early_emission": False,
        }


def _device_or_default(device: str | torch.device | None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _autocast_context(enabled: bool):
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def _extract_logits(model_output: Any) -> torch.Tensor:
    if isinstance(model_output, (tuple, list)):
        model_output = model_output[0]
    if not torch.is_tensor(model_output):
        raise TypeError("Model output must be a tensor or tuple/list of tensors.")
    return model_output.reshape(-1)


def _batch_xy(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(batch, (tuple, list)) or len(batch) < 2:
        raise TypeError("Expected each batch to contain at least x and y.")
    return batch[0], batch[1]


def _loss_for_batch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    amp_enabled: bool,
) -> torch.Tensor:
    with _autocast_context(amp_enabled):
        logits = _extract_logits(model(x))
        y = y.float().reshape_as(logits)
        return criterion(logits, y)


def _mean_epoch_loss(
    total_loss: float,
    total_items: int,
    loader_name: str,
) -> float:
    if total_items <= 0:
        raise ValueError(f"{loader_name} produced no samples.")
    return total_loss / total_items


def _current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])


def _write_history(path: Path, history: list[TrainEpochMetrics]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump([asdict(item) for item in history], f, indent=2)


def streaming_trigger_peaks(
    prob,
    threshold: float,
    holdoff_frames: int = 0,
    frame_offset: int = 0,
    sid: int | str = 0,
) -> list[int]:
    """Return peak indices emitted by the live streaming trigger logic."""
    if torch.is_tensor(prob):
        prob = prob.detach().cpu().numpy()
    prob = np.asarray(prob, dtype=np.float32)
    from dkfz.native.triggers import streaming_trigger_peaks as native_peaks

    return native_peaks(
        prob,
        threshold=threshold,
        holdoff_frames=holdoff_frames,
        frame_offset=frame_offset,
        sid=sid,
    )


def evaluate_threshold_grid(
    pred,
    target_rpeaks,
    *,
    holdoff_frames: int = 10,
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    thresholds=None,
    outer_error_association_frames: int = 2,
    exact_tolerance_frames: int = 0,
    recovery_threshold: float | None = None,
) -> tuple[dict[str, float | int | bool | str | None], ...]:
    """Evaluate the live event metrics at every supplied threshold."""
    error_association_frames = int(error_association_frames)
    if error_association_frames < 0:
        raise ValueError("error_association_frames must be zero or positive.")

    pred = np.asarray(pred, dtype=np.float32)
    dual_pred = pred if pred.ndim == 2 and pred.shape[1] == 2 else None
    if dual_pred is not None:
        # Primary threshold/epoch selection stays independent of recovery.
        pred = pred[:, 0]
    target_rpeaks = np.asarray(target_rpeaks)
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 1001, dtype=np.float32)
    else:
        thresholds = np.asarray(thresholds, dtype=np.float32)
    if thresholds.ndim != 1 or len(thresholds) == 0:
        raise ValueError("thresholds must be a non-empty one-dimensional array.")
    if np.any(~np.isfinite(thresholds)) or np.any(
        (thresholds < 0.0) | (thresholds > 1.0)
    ):
        raise ValueError("thresholds must contain finite values between 0 and 1.")

    search_rows: list[dict[str, float | int | bool | str | None]] = []

    for th in thresholds:
        pred_peaks = streaming_trigger_peaks(
            pred,
            threshold=float(th),
            holdoff_frames=holdoff_frames,
        )
        if recovery_threshold is not None:
            if dual_pred is None:
                raise ValueError(
                    "A recovery threshold requires both probability columns"
                )
            from dkfz.live.recovery import recovery_peaks

            pred_peaks = recovery_peaks(
                dual_pred,
                threshold=float(th),
                recovery_threshold=recovery_threshold,
                holdoff_frames=holdoff_frames,
            )
        stats = compute_event_position_metrics(
            pred_peaks,
            target_rpeaks,
            error_association_frames=error_association_frames,
            outer_error_association_frames=outer_error_association_frames,
            exact_tolerance_frames=exact_tolerance_frames,
        )
        search_rows.append(
            {
                "threshold": float(th),
                "predicted_event_signature": hashlib.sha256(
                    np.asarray(pred_peaks, dtype=np.int64).tobytes()
                ).hexdigest()[:16],
                **stats,
                "selectable_threshold": bool(float(th) > 0.0),
            }
        )

    return tuple(search_rows)


def select_threshold_from_search_rows(
    search_rows,
    *,
    objective: str = "min_error",
    sensitivity_efficiency: float = 0.95,
    metadata_keys=(),
) -> ThresholdSelection:
    """Apply one sensitivity floor and event policy to evaluated candidates.

    Rows may represent thresholds from one model or a joint search over models,
    epochs, or other inner-validation candidates. Caller-supplied metadata keys
    remain in the audit rows but are excluded from the selected event metrics.
    """
    objective = str(objective)
    if objective not in THRESHOLD_OBJECTIVES:
        raise ValueError(
            f"threshold objective must be one of {THRESHOLD_OBJECTIVES}, "
            f"got {objective!r}."
        )
    sensitivity_efficiency = float(sensitivity_efficiency)
    if not 0.0 < sensitivity_efficiency <= 1.0:
        raise ValueError("sensitivity_efficiency must be greater than 0 and at most 1.")

    rows = [dict(row) for row in search_rows]
    required = {
        "threshold",
        "selectable_threshold",
        "exact_sensitivity",
        "unmatched_false_positive",
        "unmatched_false_negative",
        "unique_error_count",
        "position_error",
        "outer_position_error",
    }
    if not rows:
        raise ValueError("Threshold search rows must not be empty.")
    for row in rows:
        missing = required - set(row)
        if missing:
            raise ValueError(f"Threshold search row is missing {sorted(missing)}.")

    selectable = [row for row in rows if row["selectable_threshold"]]
    if not selectable:
        raise ValueError(
            "Threshold search requires at least one candidate greater than zero."
        )
    best_sensitivity = max(float(row["exact_sensitivity"]) for row in selectable)
    if best_sensitivity <= 0.0:
        raise ValueError(
            "No positive threshold produced an exact true positive; there is no "
            "non-degenerate sensitivity region from which to select."
        )
    effective_sensitivity_floor = sensitivity_efficiency * best_sensitivity
    eligible = []
    for row in rows:
        floor_met = bool(
            row["selectable_threshold"]
            and float(row["exact_sensitivity"]) >= effective_sensitivity_floor
        )
        row["sensitivity_floor_met"] = floor_met
        if floor_met:
            eligible.append(row)

    # Recover the pre-oPE counts exactly for the historical FP-first policy.
    # Each outer pair previously contributed one FP and one FN.
    def legacy_key(row):
        ope = int(row.get("outer_position_error", 0))
        return (
            int(row["unmatched_false_positive"]) + ope,
            int(row["unique_error_count"]) + ope,
            int(row["unmatched_false_negative"]) + ope,
            int(row["position_error"]),
            -float(row["threshold"]),
        )

    def der_key(row):
        return (
            20 * int(row["unmatched_false_positive"])
            + 10 * int(row["unmatched_false_negative"])
            + int(row["position_error"])
            + 5 * int(row.get("outer_position_error", 0)),
            int(row["unmatched_false_positive"]),
            int(row["unmatched_false_negative"]),
            int(row.get("outer_position_error", 0)),
            int(row["position_error"]),
            -float(row["threshold"]),
        )

    if objective == "min_error":
        selected = min(eligible, key=der_key)
    else:
        selected = min(
            eligible,
            key=lambda row: (
                int(row["unmatched_false_positive"])
                + int(row.get("outer_position_error", 0)),
                int(row["unmatched_false_negative"])
                + int(row.get("outer_position_error", 0)),
                int(row["position_error"]),
                -float(row["threshold"]),
            ),
        )
    # Alternate thresholds must refer to the selected model/epoch, not weights
    # from a different candidate in a joint CV search.
    same_model = [
        row
        for row in eligible
        if all(row[key] == selected[key] for key in metadata_keys)
    ]
    fp_row = min(same_model, key=legacy_key)
    der_row = min(same_model, key=der_key)

    for row in rows:
        row["selected"] = row is selected
        row["selected_fp"] = row is fp_row
        row["selected_der"] = row is der_row

    metadata_keys = set(metadata_keys)
    selected_metrics = {
        key: value
        for key, value in selected.items()
        if key
        not in {
            "threshold",
            "predicted_event_signature",
            "selectable_threshold",
            "sensitivity_floor_met",
            "selected",
            "selected_fp",
            "selected_der",
            *metadata_keys,
        }
    }
    return ThresholdSelection(
        threshold=float(selected["threshold"]),
        objective=objective,
        sensitivity_efficiency=sensitivity_efficiency,
        best_attainable_sensitivity=best_sensitivity,
        effective_sensitivity_floor=effective_sensitivity_floor,
        metrics=selected_metrics,
        search_rows=tuple(rows),
        fp_threshold=float(fp_row["threshold"]),
        der_threshold=float(der_row["threshold"]),
    )


def select_threshold(
    pred,
    target_rpeaks,
    *,
    objective: str = "min_error",
    sensitivity_efficiency: float = 0.95,
    holdoff_frames: int = 10,
    error_association_frames: int = DEFAULT_ERROR_ASSOCIATION_FRAMES,
    thresholds=None,
    outer_error_association_frames: int = 2,
    exact_tolerance_frames: int = 0,
) -> ThresholdSelection:
    """Select a threshold using the configured TP/PE/oPE live-event tiers.

    Only peak frames within the configured TP radius are true positives.
    Remaining nearby events are assigned to PE and oPE tiers. The effective exact-sensitivity floor is
    ``sensitivity_efficiency`` times the best sensitivity attainable on the
    supplied threshold grid. ``min_error`` ranks eligible thresholds by
    weighted errors (20 FP + 10 FN + PE + 5 oPE), then FP, FN, oPE, PE,
    and the higher threshold. The old FP-first rule is recorded as fp_threshold. ``min_fp`` minimizes
    unmatched false positives before missed and positioned events.
    """
    rows = evaluate_threshold_grid(
        pred,
        target_rpeaks,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
        thresholds=thresholds,
        outer_error_association_frames=outer_error_association_frames,
        exact_tolerance_frames=exact_tolerance_frames,
    )
    return select_threshold_from_search_rows(
        rows,
        objective=objective,
        sensitivity_efficiency=sensitivity_efficiency,
    )


def _make_hashable_seq_key(seq_id):
    if torch.is_tensor(seq_id):
        seq_id = seq_id.detach().cpu()
        if seq_id.numel() == 1:
            return seq_id.item()
        return tuple(seq_id.flatten().tolist())

    if isinstance(seq_id, np.ndarray):
        if seq_id.size == 1:
            return seq_id.item()
        return tuple(seq_id.flatten().tolist())

    if isinstance(seq_id, (list, tuple)):
        if len(seq_id) == 1:
            return _make_hashable_seq_key(seq_id[0])
        return tuple(_make_hashable_seq_key(item) for item in seq_id)

    return seq_id


def _seq_ids_to_list(seq_id, batch_size: int) -> list:
    if torch.is_tensor(seq_id):
        seq_id = seq_id.detach().cpu()
        if seq_id.ndim == 0:
            return [_make_hashable_seq_key(seq_id)] * batch_size
        if seq_id.shape[0] != batch_size:
            raise ValueError(
                f"seq_id batch dimension mismatch: got {seq_id.shape[0]=}, "
                f"expected {batch_size=}"
            )
        return [_make_hashable_seq_key(seq_id[i]) for i in range(batch_size)]

    if isinstance(seq_id, np.ndarray):
        if seq_id.ndim == 0:
            return [_make_hashable_seq_key(seq_id)] * batch_size
        if seq_id.shape[0] != batch_size:
            raise ValueError(
                f"seq_id batch dimension mismatch: got {seq_id.shape[0]=}, "
                f"expected {batch_size=}"
            )
        return [_make_hashable_seq_key(seq_id[i]) for i in range(batch_size)]

    if isinstance(seq_id, (list, tuple)):
        if len(seq_id) == batch_size:
            return [_make_hashable_seq_key(item) for item in seq_id]
        if batch_size == 1:
            return [_make_hashable_seq_key(seq_id)]
        raise ValueError(
            f"Cannot interpret seq_id batch structure: len(seq_id)={len(seq_id)}, "
            f"{batch_size=}"
        )

    return [_make_hashable_seq_key(seq_id)] * batch_size


def plot_train_val_loss(history_json_path: str, save: bool = False):
    from dkfz.core.utils import _get_pyplot

    plt = _get_pyplot()

    with open(history_json_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    if not isinstance(history, list) or len(history) == 0:
        raise ValueError("Expected a non-empty list of per-epoch records.")

    epochs = [int(row["epoch"]) for row in history]
    train_loss = [float(row["train_loss"]) for row in history]
    raw_val_loss = [row.get("val_loss") for row in history]
    has_validation = any(value is not None for value in raw_val_loss)
    if has_validation and any(value is None for value in raw_val_loss):
        raise ValueError("History mixes epochs with and without validation loss.")
    val_loss = [float(value) for value in raw_val_loss] if has_validation else []

    plt.figure()
    plt.plot(epochs, train_loss, marker=".", label="Train Loss")
    if has_validation:
        plt.plot(epochs, val_loss, marker=".", label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    if save:
        plt.savefig(history_json_path.removesuffix(".json") + "_plot.pdf", format="pdf")
    plt.show()
