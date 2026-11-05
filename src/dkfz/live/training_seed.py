# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Reproducible random seeding shared by training entrypoints."""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch

MAX_TORCH_SEED = 2**63 - 1


def validate_training_seed(seed: int) -> int:
    """Return a seed accepted consistently by Python, NumPy, and PyTorch."""
    seed = int(seed)
    if not 0 <= seed <= MAX_TORCH_SEED:
        raise ValueError(f"seed must be between 0 and {MAX_TORCH_SEED}.")
    return seed


def set_training_seed(seed: int) -> int:
    """Seed Python, NumPy, PyTorch, and every visible CUDA device."""
    seed = validate_training_seed(seed)
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def derive_training_seed(base_seed: int, *parts: str) -> int:
    """Derive a stable fold-specific seed from a user-controlled base seed."""
    base_seed = validate_training_seed(base_seed)
    payload = "\0".join((str(base_seed), *map(str, parts))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


__all__ = [
    "derive_training_seed",
    "set_training_seed",
    "validate_training_seed",
]
