# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

import argparse
from pathlib import Path
import sys
import tempfile
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dkfz.core.data_processing import (
    from_vector,
    rotate,
    to_vector,
)


def _is_rotation_column(column: str) -> bool:
    return column == "theta" or column.endswith("_rotated")


def _assert_only_rotation_columns_changed(
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> None:
    """Verify that in-place rotation only touched theta and rotated columns."""
    if not before.index.equals(after.index):
        raise RuntimeError("Refusing to write parquet: dataframe index changed.")

    before_cols = set(before.columns)
    after_cols = set(after.columns)
    added_or_removed = before_cols.symmetric_difference(after_cols)
    disallowed_schema_changes = sorted(
        col for col in added_or_removed if not _is_rotation_column(col)
    )
    if disallowed_schema_changes:
        raise RuntimeError(
            "Refusing to write parquet: non-rotation columns were added or removed: "
            f"{disallowed_schema_changes}"
        )

    changed_columns = []
    for col in before.columns:
        if _is_rotation_column(col):
            continue
        if not before[col].equals(after[col]):
            changed_columns.append(col)

    if changed_columns:
        raise RuntimeError(
            "Refusing to write parquet: non-rotation columns changed: "
            f"{changed_columns}"
        )


def _write_rotation_update(
    path: str | Path,
    before: pd.DataFrame,
    after: pd.DataFrame,
) -> None:
    """Write a checked in-place parquet update through a verified temp file."""
    path = Path(path)
    _assert_only_rotation_columns_changed(before, after)

    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}.",
        suffix=".parquet",
        dir=path.parent,
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        after.to_parquet(tmp_path, compression="snappy")
        written = pd.read_parquet(tmp_path)
        _assert_only_rotation_columns_changed(before, written)
        tmp_path.replace(path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _check_required_columns(
    path: Path, df: pd.DataFrame, columns: Iterable[str]
) -> None:
    missing = sorted(set(columns) - set(df.columns))
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")


def _rotation_theta(df_gauge: pd.DataFrame, gauge_path: Path) -> float:
    _check_required_columns(
        gauge_path,
        df_gauge,
        ("ecg1_raw", "ecg2_raw", "ecg3_raw", "target"),
    )

    idx = np.where(df_gauge["target"] != 0)[0]
    if len(idx) == 0:
        raise ValueError(
            f"{gauge_path} has no flagged target rows for theta calibration."
        )

    vector = to_vector(df_gauge["ecg1_raw"].iloc[idx], df_gauge["ecg2_raw"].iloc[idx])

    fun = lambda theta: -np.mean(from_vector(rotate(vector, theta))[0])

    res = minimize_scalar(fun)
    return np.round(res.x, 5)


def _add_rotation_columns(df: pd.DataFrame, theta: float) -> pd.DataFrame:
    e1, e2 = from_vector(rotate(to_vector(df["ecg1_raw"], df["ecg2_raw"]), theta))
    df["theta"] = theta  # rotation angle
    df["ecg1_rotated"] = e1.ravel().astype(np.float32)
    df["ecg2_rotated"] = e2.ravel().astype(np.float32)
    df["ecg3_rotated"] = (df["ecg2_rotated"] - df["ecg1_rotated"]).astype(np.float32)
    return df


def _print_gauge_summary(
    df_gauge: pd.DataFrame, theta: float, gauge_path: Path
) -> None:
    idx = np.where(df_gauge["target"] != 0)[0]

    print(f"Rotation angle {theta = :.4f} [rad] = {np.rad2deg(theta):.2f} [deg]")

    p1 = np.mean(df_gauge["ecg1_raw"].iloc[idx])
    p1_2 = np.mean(df_gauge["ecg2_raw"].iloc[idx])
    p1_3 = np.mean(df_gauge["ecg3_raw"].iloc[idx])

    print(
        "Mean ECG R-peak amplitude for gauge signal before rotation: "
        f"\necg1 = {p1:.2f}, ecg2 = {p1_2:.2f}, ecg3 = {p1_3:.2f}"
    )

    rotated_gauge = _add_rotation_columns(df_gauge.copy(deep=True), theta)

    p2 = np.mean(rotated_gauge["ecg1_rotated"].iloc[idx])
    p2_2 = np.mean(rotated_gauge["ecg2_rotated"].iloc[idx])
    p2_3 = np.mean(rotated_gauge["ecg3_rotated"].iloc[idx])

    print(
        f"Gauge file: {gauge_path}"
        "\nMean ECG R-peak amplitude for gauge signal after rotation: "
        f"\necg1 = {p2:.2f}, ecg2 = {p2_2:.2f}, ecg3 = {p2_3:.2f}"
    )


def _update_input_file(path: Path, theta: float) -> None:
    df = pd.read_parquet(path)
    _check_required_columns(path, df, ("ecg1_raw", "ecg2_raw"))
    df_before = df.copy(deep=True)

    _add_rotation_columns(df, theta)

    _write_rotation_update(path, df_before, df)
    print(f"Updated {path} in place.")


def rotate_ecg(
    gaugeFile: str | Path = "data/test/test_flagged.parquet",
    inFile: str | Path | None = None,
):
    gauge_path = Path(gaugeFile)
    df_gauge = pd.read_parquet(gauge_path)
    theta = _rotation_theta(df_gauge, gauge_path)
    _print_gauge_summary(df_gauge, theta, gauge_path)

    if inFile is not None and inFile != "None":
        _update_input_file(Path(inFile), theta)


def _directory_gauge_file(directory: Path) -> Path:
    return directory / f"{directory.name}_out.parquet"


def _parquet_inputs_in_directory(directory: Path) -> list[Path]:
    paths = sorted(directory.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"{directory} contains no parquet files.")
    return paths


def rotate_ecg_directory(directory: str | Path) -> None:
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    gauge_path = _directory_gauge_file(directory)
    if not gauge_path.is_file():
        raise FileNotFoundError(
            f"Expected directory gauge file {gauge_path} for subject {directory.name}."
        )

    df_gauge = pd.read_parquet(gauge_path)
    theta = _rotation_theta(df_gauge, gauge_path)
    _print_gauge_summary(df_gauge, theta, gauge_path)

    for path in _parquet_inputs_in_directory(directory):
        _update_input_file(path, theta)


def _run_path(path: Path, gauge_file: Path | None) -> None:
    if path.is_dir():
        if gauge_file is not None:
            df_gauge = pd.read_parquet(gauge_file)
            theta = _rotation_theta(df_gauge, gauge_file)
            _print_gauge_summary(df_gauge, theta, gauge_file)
            for input_path in _parquet_inputs_in_directory(path):
                _update_input_file(input_path, theta)
            return

        rotate_ecg_directory(path)
        return

    if gauge_file is None:
        raise ValueError(f"File input {path} needs --gaugeFile.")
    rotate_ecg(gauge_file, path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rotate ECG parquet files. File inputs use --gaugeFile; subject "
            "directory inputs infer <subject>_out.parquet as the gauge and update "
            "all parquet files in that directory, including the gauge file."
        )
    )
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="Parquet files or subject directories to update.",
    )
    parser.add_argument(
        "-g",
        "--gaugeFile",
        type=Path,
        help="Path to the calibration file of true R peaks with data taken outside the Scanner.",
        required=False,
    )
    parser.add_argument(
        "-i",
        "--inFile",
        type=Path,
        nargs="+",
        help=(
            "Path(s) to input parquet files or subject directories. Directories infer "
            "<subject>_out.parquet as their gauge unless --gaugeFile is supplied."
        ),
        required=False,
    )

    args = parser.parse_args()
    input_paths = [*args.paths]
    if args.inFile is not None:
        input_paths.extend(args.inFile)

    if not input_paths:
        if args.gaugeFile is None:
            parser.error("provide --gaugeFile, an input path, or a subject directory")
        if args.gaugeFile.is_dir():
            rotate_ecg_directory(args.gaugeFile)
            return
        rotate_ecg(args.gaugeFile)
        return

    for path in input_paths:
        _run_path(path, args.gaugeFile)


if __name__ == "__main__":
    main()
