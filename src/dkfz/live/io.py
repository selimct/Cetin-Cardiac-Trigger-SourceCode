# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared JSON and input-resolution helpers for live ECG scripts."""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import pandas as pd


def _looks_like_repo_root(path: Path) -> bool:
    return (path / "pyproject.toml").is_file() and (path / "src" / "dkfz").is_dir()


def find_repo_root(start: str | Path | None = None) -> Path:
    """Find the project root by walking upward from cwd/module locations."""
    starts = [Path(start).resolve()] if start is not None else []
    starts.extend([Path.cwd().resolve(), Path(__file__).resolve()])

    for candidate in starts:
        current = candidate if candidate.is_dir() else candidate.parent
        for parent in (current, *current.parents):
            if _looks_like_repo_root(parent):
                return parent

    return Path.cwd().resolve()


REPO_ROOT = find_repo_root()


def repo_relative_path(path: str | Path) -> str:
    """Return a repo-root-relative path for portable logs and metadata."""
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return os.path.relpath(path, REPO_ROOT)


def load_json_config(path: str | Path | None) -> dict:
    """Load a JSON config file, or return an empty config when no path is given."""
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_input_paths(inputs: list[str]) -> list[Path]:
    """Resolve parquet files, directories, or glob patterns into unique paths."""
    paths: list[Path] = []
    for item in inputs:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.glob("*.parquet")))
            continue

        matches = [Path(x) for x in glob.glob(item)]
        if matches:
            paths.extend(sorted(matches))
            continue

        if p.is_file():
            paths.append(p)

    unique = sorted({p.resolve() for p in paths})
    if not unique:
        raise FileNotFoundError(f"No parquet files found for inputs: {inputs}")
    return unique


def require_raw_signal_column(
    df: pd.DataFrame,
    column: str | None,
    signal_name: str,
) -> str:
    """Validate a raw-rate input, including explicit synthetic ECG columns."""
    if column is None:
        raise KeyError(f"No raw {signal_name} column was provided.")

    column_name = str(column)
    is_synthetic_ecg = signal_name.startswith("ECG") and column_name in (
        "ecg1_syn",
        "ecg2_syn",
        "ecg3_syn",
    )
    if (
        "_raw" not in column_name
        and not column_name.endswith("_rotated")
        and not is_synthetic_ecg
    ):
        raise ValueError(
            f"{signal_name} must use a raw or rotated raw-rate column "
            f"(or ecgN_syn for synthetic ECG), got "
            f"{column!r}. Offline-normalized ECG/pulse columns are not supported."
        )

    if column not in df.columns:
        raise KeyError(
            f"Missing raw {signal_name} column {column!r}. "
            "Regenerate the parquet with raw ECG/pulse columns."
        )

    return column


def optional_raw_signal_column(
    df: pd.DataFrame,
    column: str | None,
    signal_name: str,
) -> str | None:
    """Return a raw column when present, or None when the optional input is absent."""
    if column is None or column not in df.columns:
        return None
    return require_raw_signal_column(df, column, signal_name)


def require_raw_signal_columns(
    df: pd.DataFrame,
    *,
    ecg1_col: str | None,
    ecg2_col: str | None,
    ecg3_col: str | None = None,
    puls_col: str | None,
    include_puls: bool,
) -> tuple[str, str, str | None, str | None]:
    """Validate ECG/pulse input columns for the live raw-signal contract."""
    actual_ecg1_col = require_raw_signal_column(df, ecg1_col, "ECG1")
    actual_ecg2_col = require_raw_signal_column(df, ecg2_col, "ECG2")
    actual_ecg3_col = (
        require_raw_signal_column(df, ecg3_col, "ECG3")
        if ecg3_col is not None
        else None
    )
    actual_puls_col = (
        require_raw_signal_column(df, puls_col, "pulse") if include_puls else None
    )
    return actual_ecg1_col, actual_ecg2_col, actual_ecg3_col, actual_puls_col
