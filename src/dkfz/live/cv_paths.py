# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Naming and output-directory safety for patient cross-validation runs."""

from __future__ import annotations

from pathlib import Path
import shutil

from dkfz.live.io import REPO_ROOT

CV_MODEL_SUFFIX = "_cv"
NESTED_LOPO_DIRNAME = "nested_lopo"
NESTED_LOPO_HPO_DIRNAME = "nested_lopo_hpo"
NESTED_RECOVERY_LOPO_DIRNAME = "nested_recovery_lopo"
_CV_METHOD_DIRNAMES = {
    NESTED_LOPO_DIRNAME,
    NESTED_LOPO_HPO_DIRNAME,
    NESTED_RECOVERY_LOPO_DIRNAME,
}


def cv_modelname(modelname: str) -> str:
    """Return a safe model-family name with exactly one trailing ``_cv``."""
    name = str(modelname).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError(
            "CV modelname must be one non-empty path component without separators."
        )
    return name if name.endswith(CV_MODEL_SUFFIX) else f"{name}{CV_MODEL_SUFFIX}"


def cv_base_modelname(modelname: str) -> str:
    """Remove one trailing CV suffix for provenance and user-facing output."""
    name = cv_modelname(modelname)
    return name[: -len(CV_MODEL_SUFFIX)]


def default_cv_output_dir(modelname: str, method_dirname: str) -> Path:
    """Return ``models/<model>_cv/<method>`` for a supported CV workflow."""
    if method_dirname not in _CV_METHOD_DIRNAMES:
        raise ValueError(f"Unsupported CV method directory: {method_dirname!r}.")
    return Path("models") / cv_modelname(modelname) / method_dirname


def prepare_cv_output_dir(
    path: str | Path,
    *,
    overwrite: bool,
    require_empty: bool,
) -> Path:
    """Optionally reset one exact CV method directory and return its absolute path."""
    requested = Path(path)
    if requested.is_symlink():
        raise ValueError(f"Refusing to use symlinked CV output: {requested}")
    resolved = requested.resolve()
    protected = {
        REPO_ROOT.resolve(),
        (REPO_ROOT / "models").resolve(),
        Path.home().resolve(),
        Path("/").resolve(),
    }
    if resolved in protected:
        raise ValueError(f"Refusing to use broad CV output directory: {resolved}")

    if overwrite and resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"CV output path is not a directory: {resolved}")
        shutil.rmtree(resolved)
    elif resolved.exists():
        if not resolved.is_dir():
            raise ValueError(f"CV output path is not a directory: {resolved}")
        if require_empty and any(resolved.iterdir()):
            raise ValueError(
                f"CV output directory is not empty: {resolved}. Pass --overwrite to "
                "replace this method's artifacts."
            )

    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


__all__ = [
    "CV_MODEL_SUFFIX",
    "NESTED_LOPO_DIRNAME",
    "NESTED_LOPO_HPO_DIRNAME",
    "NESTED_RECOVERY_LOPO_DIRNAME",
    "cv_base_modelname",
    "cv_modelname",
    "default_cv_output_dir",
    "prepare_cv_output_dir",
]
