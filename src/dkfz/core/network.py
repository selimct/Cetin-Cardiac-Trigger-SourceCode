# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Public live-oriented PyTorch training and evaluation entrypoints."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from dkfz.core.network_helpers import (
    TrainEpochMetrics,
    _autocast_context,
    _batch_xy,
    _current_lr,
    _device_or_default,
    _extract_logits,
    _loss_for_batch,
    _mean_epoch_loss,
    _seq_ids_to_list,
    _write_history,
    select_threshold,
    streaming_trigger_peaks,
)


def train_torch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    train_loader,
    val_loader=None,
    lr: float = 1e-3,
    num_epochs: int = 10,
    early_stopping: int = 3,
    device: str | torch.device | None = None,
    modelname: str = "torch",
    non_blocking: bool = True,
    use_amp: bool = True,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    weight_decay: float = 0.0,
    grad_clip_norm: float | None = 1.0,
    save_best: bool = True,
    save_last: bool = False,
    save_history: bool = True,
    models_dir: str | Path = "models",
    best_filename: str | Path = "best.pth",
    last_filename: str | Path = "last.pth",
    history_filename: str | Path = "train_history.json",
    progress: bool = True,
    epoch_checkpoint_dir: str | Path | None = None,
) -> tuple[list[float], list[float]]:
    """Train a binary sequence classifier for the live ECG pipeline.

    Args:
        model (torch.nn.Module): PyTorch model.
        criterion (torch.nn.Module): BCEWithLogitsLoss.
        train_loader (DataLoader): Input training set.
        val_loader (DataLoader | None): Input validation set. When omitted,
            training runs for exactly ``num_epochs`` and cannot save a
            validation-selected best checkpoint or use early stopping.
        lr (float, optional): Learning rate. Defaults to 1e-3.
        num_epochs (int, optional): Number of epochs. Defaults to 10.
        early_stopping (int, optional): Number of negative epochs to cause early stopping. Defaults to 3.
        device (str | torch.device | None, optional): CUDA or CPU. Defaults to None.
        modelname (str, optional): Defaults to "torch".
        non_blocking (bool, optional): When non_blocking is set to True, the function attempts to perform the conversion asynchronously with respect to the host, if possible. This asynchronous behavior applies to both pinned and pageable memory. However, caution is advised when using this feature.  Defaults to True.
        use_amp (bool, optional): Automatic mixed precision for CUDA optimisation. Defaults to True.
        optimizer (torch.optim.Optimizer | None, optional): Defaults to None.
        weight_decay (float, optional): Weight Decay for optimizer. Defaults to 0.0.
        grad_clip_norm (float | None, optional): Clip the gradient norm of an iterable of parameters. The norm is computed over the norms of the individual gradients of all parameters, as if the norms of the individual gradients were concatenated into a single vector. Gradients are modified in-place. This function is equivalent to torch.nn.utils.get_total_norm followed by torch.nn.utils.clip_grads_with_norm_ with the total_norm returned by get_total_norm. Defaults to 1.0.
        save_best (bool, optional): Saves best performing model on validation data in ``models/{modelname}/best.pth``. Defaults to True.
        save_last (bool, optional): Saves model after last epoch in ``models/{modelname}/last.pth``. Defaults to False.
        save_history (bool, optional): Saves model training history in ``models/{modelname}/train_history.json``. Defaults to True.
        models_dir (str | Path, optional): Defaults to ``models/``.
        best_filename (str | Path, optional): Filename/path for the best checkpoint. Defaults to ``best.pth``.
        last_filename (str | Path, optional): Filename/path for the final checkpoint. Defaults to ``last.pth``.
        history_filename (str | Path, optional): Filename/path for the training history. Defaults to ``train_history.json``.
        progress (bool, optional): Display progress. Defaults to True.
        epoch_checkpoint_dir (str | Path | None, optional): When provided,
            save one state dict after every epoch. Relative paths are resolved
            below the model directory. Defaults to None.

    Returns:
        tuple[list[float], list[float]]: Returns Train_losses and Val_losses.
    """
    best_vloss: float = 1000000
    validation_enabled = val_loader is not None
    if not validation_enabled and save_best:
        raise ValueError("save_best requires a validation loader.")
    if not validation_enabled and int(early_stopping) > 0:
        raise ValueError("early_stopping requires a validation loader.")
    device = _device_or_default(device)
    model = model.to(device)
    criterion = criterion.to(device)

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    if optimizer is None:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(weight_decay),
        )

    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    model_dir = Path(models_dir) / str(modelname)
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = Path(best_filename)
    if not best_model_path.is_absolute():
        best_model_path = model_dir / best_model_path
    last_model_path = Path(last_filename)
    if not last_model_path.is_absolute():
        last_model_path = model_dir / last_model_path
    history_path = Path(history_filename)
    if not history_path.is_absolute():
        history_path = model_dir / history_path
    best_model_path.parent.mkdir(parents=True, exist_ok=True)
    last_model_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    epoch_checkpoint_path = None
    if epoch_checkpoint_dir is not None:
        epoch_checkpoint_path = Path(epoch_checkpoint_dir)
        if not epoch_checkpoint_path.is_absolute():
            epoch_checkpoint_path = model_dir / epoch_checkpoint_path
        epoch_checkpoint_path.mkdir(parents=True, exist_ok=True)

    patience = int(early_stopping)
    patience_enabled = patience > 0
    stale_epochs = 0
    history: list[TrainEpochMetrics] = []
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(1, int(num_epochs) + 1):
        epoch_started = time.perf_counter()

        model.train()
        train_total = 0.0
        train_items = 0
        for batch in train_loader:
            x, y = _batch_xy(batch)
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
            batch_items = int(x.shape[0])

            optimizer.zero_grad(set_to_none=True)
            loss = _loss_for_batch(model, criterion, x, y, amp_enabled)
            scaler.scale(loss).backward()
            if grad_clip_norm is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(grad_clip_norm)
                )
            scaler.step(optimizer)
            scaler.update()

            train_total += float(loss.detach()) * batch_items
            train_items += batch_items

        train_loss = _mean_epoch_loss(train_total, train_items, "train_loader")

        val_loss = None
        if validation_enabled:
            model.eval()
            val_total = 0.0
            val_items = 0
            with torch.inference_mode():
                for batch in val_loader:
                    x, y = _batch_xy(batch)
                    x = x.to(device, non_blocking=non_blocking)
                    y = y.to(device, non_blocking=non_blocking)
                    batch_items = int(x.shape[0])

                    loss = _loss_for_batch(model, criterion, x, y, amp_enabled)
                    val_total += float(loss.detach()) * batch_items
                    val_items += batch_items

            val_loss = _mean_epoch_loss(val_total, val_items, "val_loader")
        train_losses.append(float(train_loss))
        if val_loss is not None:
            val_losses.append(float(val_loss))
        improved = val_loss is not None and val_loss < float(best_vloss)
        if improved:
            best_vloss = float(val_loss)
            stale_epochs = 0
            if save_best:
                torch.save(model.state_dict(), best_model_path)
        elif validation_enabled:
            stale_epochs += 1

        if save_last:
            torch.save(model.state_dict(), last_model_path)
        if epoch_checkpoint_path is not None:
            torch.save(
                model.state_dict(),
                epoch_checkpoint_path / f"epoch_{epoch:04d}.pth",
            )

        metrics = TrainEpochMetrics(
            epoch=epoch,
            train_loss=float(train_loss),
            val_loss=None if val_loss is None else float(val_loss),
            lr=_current_lr(optimizer),
            seconds=float(time.perf_counter() - epoch_started),
            improved=bool(improved),
        )
        history.append(metrics)

        if save_history:
            _write_history(history_path, history)

        if progress:
            message = (
                f"Epoch {metrics.epoch}/{num_epochs}, "
                f"Train Loss: {metrics.train_loss:.6f}"
            )
            if metrics.val_loss is not None:
                message += f", Val Loss: {metrics.val_loss:.6f}"
            print(message)

        if patience_enabled and stale_epochs >= patience:
            if progress:
                print("Early stopping")
            break

    return train_losses, val_losses


def eval_torch(
    model,
    val_loader,
    error_association_frames=1,
    holdoff_frames=20,
    threshold_objective="min_error",
    threshold_sensitivity_efficiency=0.95,
    device=None,
    use_amp=True,
    return_threshold_selection=False,
    outer_error_association_frames=2,
):
    """Evaluate a trained model on ground truth annotated data.

    Args:
        model (torch.nn.Module): PyTorch model.
        val_loader (DataLoader): Input validation set.
        error_association_frames (int, optional): Maximum frame distance used
            only to attribute a non-exact prediction as one position error.
            Exact-match tolerance is always zero. Defaults to 1.
        holdoff_frames (int, optional): Number of minimum frames between r peak predictions. Defaults to 20.
        threshold_objective (str, optional): Use ``min_error`` or conservative
            ``min_fp`` selection. Defaults to ``min_error``.
        threshold_sensitivity_efficiency (float, optional): Fraction of the
            best attainable exact sensitivity retained as the selection floor.
            Defaults to 0.95.
        device (str | torch.device | None, optional): CUDA or CPU. Defaults to None.
        use_amp (bool, optional): Automatic mixed precision for CUDA optimisation. Defaults to True.
        return_threshold_selection (bool, optional): Return full search details
            instead of only the selected threshold. Defaults to False.

    Raises:
        TypeError: Expected each eval batch to contain x, y, and seq_id.
        ValueError: Expected one logit per sample .
        ValueError: Expected one target per sample.
        ValueError: val_loader produced no samples.

    Returns:
        Tuple containing the results DataFrame and either the selected threshold
        or its full ``ThresholdSelection`` details.
    """
    import pandas as pd

    device = _device_or_default(device)
    model = model.to(device)
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    probs_by_seq = defaultdict(list)
    targ_by_seq = defaultdict(list)

    with torch.inference_mode():
        for batch in val_loader:
            if not isinstance(batch, (tuple, list)) or len(batch) < 3:
                raise TypeError("Expected each eval batch to contain x, y, and seq_id.")

            x, y, seq_id = batch[0], batch[1], batch[2]
            x = x.to(device, non_blocking=True)
            batch_size = int(x.shape[0])

            with _autocast_context(amp_enabled):
                logits = _extract_logits(model(x)).reshape(batch_size, -1)

            y_flat = y.detach().cpu().reshape(batch_size, -1)
            if logits.shape[1] != 1:
                raise ValueError(
                    f"Expected one logit per sample, got logits.shape={tuple(logits.shape)}"
                )
            if y_flat.shape[1] != 1:
                raise ValueError(
                    f"Expected one target per sample, got y.shape={tuple(y.shape)}"
                )

            probs_np = torch.sigmoid(logits[:, 0].float()).cpu().numpy()
            y_np = y_flat[:, 0].float().numpy()
            seq_keys = _seq_ids_to_list(seq_id, batch_size)

            for i in range(batch_size):
                probs_by_seq[seq_keys[i]].append(float(probs_np[i]))
                targ_by_seq[seq_keys[i]].append(float(y_np[i]))

    gap_len = int(max(holdoff_frames + 1, 2 * error_association_frames + 1, 20))
    long_prob_parts = []
    long_targ_parts = []
    long_seq_parts = []

    for seq_key in probs_by_seq.keys():
        p = np.asarray(probs_by_seq[seq_key], dtype=np.float32)
        t = np.asarray(targ_by_seq[seq_key], dtype=np.float32)

        long_prob_parts.append(p)
        long_targ_parts.append(t)
        long_seq_parts.append(np.full(p.shape[0], seq_key, dtype=object))

        long_prob_parts.append(np.zeros(gap_len, dtype=np.float32))
        long_targ_parts.append(np.zeros(gap_len, dtype=np.float32))
        long_seq_parts.append(np.full(gap_len, None, dtype=object))

    if not long_prob_parts:
        raise ValueError("val_loader produced no samples.")

    long_prob = np.concatenate(long_prob_parts)
    long_targ = np.concatenate(long_targ_parts)
    long_seq = np.concatenate(long_seq_parts)

    target_rpeaks = np.where(long_targ == 1)[0]
    threshold_selection = select_threshold(
        long_prob,
        target_rpeaks,
        objective=threshold_objective,
        sensitivity_efficiency=threshold_sensitivity_efficiency,
        holdoff_frames=holdoff_frames,
        error_association_frames=error_association_frames,
        outer_error_association_frames=outer_error_association_frames,
    )
    best_th = threshold_selection.threshold

    pred_rpeaks = streaming_trigger_peaks(
        long_prob,
        threshold=float(best_th),
        holdoff_frames=holdoff_frames,
    )

    result = pd.DataFrame(
        {
            "sid": long_seq,
            "prob": long_prob,
            "pred": 0,
            "target": long_targ,
        }
    )
    result.loc[pred_rpeaks, "pred"] = 1
    result["best_th"] = best_th
    result["fp_threshold"] = threshold_selection.fp_threshold
    result["der_threshold"] = threshold_selection.der_threshold
    if return_threshold_selection:
        return result, threshold_selection
    return result, best_th


def predict_torch(
    model,
    loader,
    threshold,
    window_size=25,
    device=None,
    use_amp=True,
    holdoff_frames: int | None = 20,
):
    """Return frame-aligned probabilities and live-trigger predictions.

    Args:
        model (torch.nn.Module): PyTorch model.
        loader (DataLoader): Input dataset.
        threshold (float): Model probability threshold for classification.
        window_size (int, optional): Window size. Defaults to 25.
        device (str | torch.device | None, optional): CUDA or CPU. Defaults to None.
        use_amp (bool, optional): Automatic mixed precision for CUDA optimisation. Defaults to True.
        holdoff_frames (int | None, optional): Number of minimum frames between r peak predictions. Defaults to 20.

    Returns:
        List(np.darray, np.darray): Returns probs and preds.
    """
    window_size = int(window_size)
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")

    device = _device_or_default(device)
    model = model.to(device)
    model.eval()
    amp_enabled = bool(use_amp and device.type == "cuda")

    probs = []
    with torch.inference_mode():
        for batch in loader:
            x, _y = _batch_xy(batch)
            x = x.to(device, non_blocking=True)

            with _autocast_context(amp_enabled):
                logits = _extract_logits(model(x))
            batch_probs = torch.sigmoid(logits).detach().cpu().reshape(-1).numpy()
            probs.extend(batch_probs)

    probs = np.asarray(probs, dtype=np.float32)
    probs = np.pad(probs, (window_size - 1, 0), "constant", constant_values=(0, 0))

    pred_rpeaks = streaming_trigger_peaks(
        probs,
        threshold=float(threshold),
        holdoff_frames=int(holdoff_frames),
    )
    preds = np.zeros(len(probs), dtype=int)
    preds[pred_rpeaks] = 1
    return probs, preds
