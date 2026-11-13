# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Reuse deterministic frozen representations during CPU recovery-head training."""

from __future__ import annotations

from functools import partial

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset, default_collate

from dkfz.core.dataset import SlidingWindowDataset
from dkfz.core.model import ECG_RNN


class CachedRecoveryHeads(torch.nn.Module):
    """Use the original trainable head and the cached, unchanged primary logits."""

    def __init__(self, model: ECG_RNN):
        super().__init__()
        self.model = model

    def forward(self, features):
        if features.ndim == 3:
            return self.model(features)
        return torch.cat(
            (features[:, -1:], self.model.recovery_fc(features[:, :-1])), dim=-1
        )


class _CachedDataset(Dataset):
    def __init__(self, features, targets):
        self.features, self.targets = features, targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.targets[index], index


def _collate_cached(samples, *, dataset, batch_size):
    # A short batch can use different CPU kernels. Replay it at its original
    # shape instead of training on representations computed at the full size.
    if len(samples) != batch_size:
        return default_collate([dataset[index] for _, _, index in samples])
    return default_collate([(features, target) for features, target, _ in samples])


def _supported_loader(loader):
    return (
        type(loader) is DataLoader
        and type(loader.dataset) in (SlidingWindowDataset, TensorDataset)
        and loader.batch_size is not None
        and loader.num_workers == 0
        and loader.collate_fn is default_collate
    )


def _supported_head(head):
    return type(head) is torch.nn.Sequential and tuple(
        type(layer) for layer in head
    ) == (torch.nn.Linear, torch.nn.ReLU, torch.nn.Linear)


def _cache_loader(model, loader, device):
    if len(loader.dataset) < loader.batch_size:
        return loader
    # A separate generator keeps precomputation from consuming shuffle/base seeds.
    ordered = DataLoader(
        loader.dataset,
        batch_size=loader.batch_size,
        shuffle=False,
        generator=torch.Generator().manual_seed(0),
    )
    features = targets = None
    offset = 0
    with torch.no_grad():
        for x, y, *_ in ordered:
            count = len(x)
            if count < loader.batch_size:
                padding = x.new_zeros((loader.batch_size - count, *x.shape[1:]))
                x = torch.cat((x, padding))
            output, _ = model.lstm(x.to(device))
            representation = output[:, -1, :]
            frozen = torch.cat((representation, model.fc(representation)), dim=-1)
            if features is None:
                features = torch.empty(
                    (len(loader.dataset), frozen.shape[1]), dtype=frozen.dtype
                )
                targets = torch.empty(
                    (len(loader.dataset), *y.shape[1:]), dtype=y.dtype
                )
            end = offset + count
            features[offset:end].copy_(frozen[:count].cpu())
            targets[offset:end].copy_(y)
            offset = end
    if features is None:
        return loader
    return DataLoader(
        _CachedDataset(features, targets),
        # Preserve the sampler, batch size, drop_last, and RNG used by training.
        batch_sampler=loader.batch_sampler,
        generator=loader.generator,
        pin_memory=loader.pin_memory,
        collate_fn=partial(
            _collate_cached, dataset=loader.dataset, batch_size=loader.batch_size
        ),
    )


def cached_recovery_training(
    model, train_loader, val_loader, device, *, max_bytes=2048 * 1024**2
):
    """Return a head-only forward path and loaders, or the original unsupported path.

    Caches live only for this training call. Each fold/source checkpoint therefore
    gets its own representations. GPU/mixed precision, custom models/datasets,
    stochastic encoders and caches larger than the RAM budget use ordinary training.
    """
    loaders = [loader for loader in (train_loader, val_loader) if loader is not None]
    eligible = (
        device.type == "cpu"
        and type(model) is ECG_RNN
        and model.recovery_head
        and type(model.lstm) is torch.nn.LSTM
        and model.lstm.dropout == 0
        and _supported_head(model.fc)
        and _supported_head(model.recovery_fc)
        and model.lstm.weight_ih_l0.dtype == torch.float32
        and all(not p.requires_grad for p in model.lstm.parameters())
        and all(not p.requires_grad for p in model.fc.parameters())
        and all(p.requires_grad for p in model.recovery_fc.parameters())
        and all(_supported_loader(loader) for loader in loaders)
        and any(len(loader.dataset) >= loader.batch_size for loader in loaders)
    )
    if not eligible:
        return model, train_loader, val_loader
    required = 0
    for loader in loaders:
        if len(loader.dataset) >= loader.batch_size:
            target = loader.dataset[0][1]
            required += len(loader.dataset) * (
                (model.lstm.hidden_size + 1) * 4
                + target.numel() * target.element_size()
            )
    if required > max_bytes:
        print(
            f"Frozen recovery cache needs {required / 1024**2:.1f} MiB; using uncached training."
        )
        return model, train_loader, val_loader
    was_training = model.training
    try:
        model.eval()
        cached_train = _cache_loader(model, train_loader, device)
        cached_val = (
            _cache_loader(model, val_loader, device) if val_loader is not None else None
        )
    finally:
        model.train(was_training)
    print(
        f"Cached frozen recovery representations ({required / 1024**2:.1f} MiB); training only the recovery head."
    )
    return CachedRecoveryHeads(model), cached_train, cached_val
