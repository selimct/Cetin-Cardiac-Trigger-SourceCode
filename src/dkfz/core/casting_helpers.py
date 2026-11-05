# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import pandas as pd


NormalizedDType = Literal["float16", "float32", "float64"]


def _is_scalar(values: Any) -> bool:
    return np.isscalar(values)


def _as_array(values: Any) -> np.ndarray:
    return np.asarray(values)


def _return_like_input(original: Any, values: np.ndarray, dtype: np.dtype) -> Any:
    if isinstance(original, pd.Series):
        return pd.Series(
            values.astype(dtype),
            index=original.index,
            name=original.name,
        )
    if _is_scalar(original):
        return values.astype(dtype).item()
    return values.astype(dtype)


def _check_finite(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")


def _check_integer_like(arr: np.ndarray, name: str, atol: float) -> np.ndarray:
    rounded = np.rint(arr)
    if not np.allclose(arr, rounded, rtol=0.0, atol=atol):
        raise ValueError(f"{name} contains non-integer values.")
    return rounded


def _check_dtype_range(arr: np.ndarray, dtype: np.dtype, name: str) -> None:
    info = np.iinfo(dtype)
    if arr.min() < info.min or arr.max() > info.max:
        raise ValueError(f"{name} values do not fit into {np.dtype(dtype).name}.")


def to_int16(values: Any, name: str = "values", atol: float = 1e-9) -> Any:
    """Cast integer-like values to int16.

    Accepts scalars, lists, numpy arrays, and pandas Series. Lists and arrays are
    returned as numpy arrays. Series are returned as Series with the same index.
    """
    dtype = np.dtype(np.int16)
    arr = _as_array(values)
    _check_finite(arr, name)
    rounded = _check_integer_like(arr, name, atol)
    _check_dtype_range(rounded, dtype, name)
    return _return_like_input(values, rounded, dtype)


def to_uint8(values: Any, name: str = "values", atol: float = 1e-9) -> Any:
    """Cast integer-like values to uint8, useful for binary flags/triggers."""
    dtype = np.dtype(np.uint8)
    arr = _as_array(values)
    _check_finite(arr, name)
    rounded = _check_integer_like(arr, name, atol)
    _check_dtype_range(rounded, dtype, name)
    return _return_like_input(values, rounded, dtype)


def to_normalized_float(
    values: Any,
    dtype: NormalizedDType = "float32",
    name: str = "values",
) -> Any:
    """Cast normalized values to a floating dtype.

    Use float32 as the default for normalized model/data-processing values. It is
    usually the best tradeoff between size, precision, and ML-library support.
    Use float16 only when you specifically want smaller stored parquet columns
    and the reduced precision is acceptable.
    """
    np_dtype = np.dtype(dtype)
    arr = _as_array(values)
    _check_finite(arr, name)
    return _return_like_input(values, arr, np_dtype)


def subtract_offset_float64(
    values: Any,
    offset: float,
    name: str = "values",
) -> Any:
    """Convert to float64 before subtracting an acquisition-source offset.

    The conversion is deliberately performed before subtraction so unsigned
    ADC input cannot wrap when a sample lies below its offset.
    """
    offset_value = float(offset)
    if not np.isfinite(offset_value):
        raise ValueError(f"{name} offset must be finite.")
    arr = np.asarray(values, dtype=np.float64)
    _check_finite(arr, name)
    return _return_like_input(
        values,
        arr - offset_value,
        np.dtype(np.float64),
    )


def normalized_storage_dtype(use_float16: bool = False) -> np.dtype:
    """Return the recommended dtype for normalized data.

    float32 is the safer default. float16 is smaller but less precise.
    """
    if use_float16:
        return np.dtype(np.float16)
    return np.dtype(np.float32)


def cast_columns_to_int16(
    df: pd.DataFrame,
    columns: Iterable[str],
    copy: bool = True,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """Cast selected DataFrame columns to int16."""
    out = df.copy() if copy else df
    for col in columns:
        out[col] = to_int16(out[col], name=col, atol=atol)
    return out


def cast_columns_to_uint8(
    df: pd.DataFrame,
    columns: Iterable[str],
    copy: bool = True,
    atol: float = 1e-9,
) -> pd.DataFrame:
    """Cast selected DataFrame columns to uint8."""
    out = df.copy() if copy else df
    for col in columns:
        out[col] = to_uint8(out[col], name=col, atol=atol)
    return out


def cast_columns_to_normalized_float(
    df: pd.DataFrame,
    columns: Iterable[str],
    dtype: NormalizedDType = "float32",
    copy: bool = True,
) -> pd.DataFrame:
    """Cast selected DataFrame columns to float32 or float16."""
    out = df.copy() if copy else df
    for col in columns:
        out[col] = to_normalized_float(out[col], dtype=dtype, name=col)
    return out


def cast_int16_and_uint16_columns_to_float32(
    df: pd.DataFrame,
    copy: bool = True,
) -> pd.DataFrame:
    """Cast every int16 and uint16 DataFrame column to float32.

    Other column dtypes are left unchanged. Pandas nullable ``Int16`` and
    ``UInt16`` columns are also supported; missing values become ``NaN``.
    """
    out = df.copy() if copy else df
    columns = out.select_dtypes(include=("int16", "uint16")).columns
    out[columns] = out[columns].astype("float32")
    return out
