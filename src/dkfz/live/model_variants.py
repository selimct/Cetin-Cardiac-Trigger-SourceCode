# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Shared model artifact defaults for trained live ECG model variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_MODEL_VARIANT = "best"


@dataclass(frozen=True)
class ModelVariant:
    """Default artifact names for one trained model variant."""

    name: str
    config: str
    checkpoint: str
    output_suffix: str
    onnx_suffix: str
    engine_suffix: str

    def model_dir(
        self,
        modelname: str,
        model_dir: str | Path | None = None,
    ) -> Path:
        """Return one explicit artifact root, or the conventional default."""
        if model_dir is not None:
            return Path(model_dir)
        return Path("models") / str(modelname)

    def config_path(
        self, modelname: str, model_dir: str | Path | None = None
    ) -> Path:
        return self.model_dir(modelname, model_dir) / self.config

    def checkpoint_path(
        self, modelname: str, model_dir: str | Path | None = None
    ) -> Path:
        return self.model_dir(modelname, model_dir) / self.checkpoint

    def onnx_path(
        self, modelname: str, model_dir: str | Path | None = None
    ) -> Path:
        filename = f"{modelname}{self.onnx_suffix}.onnx"
        return self.model_dir(modelname, model_dir) / filename

    def trt_engine_path(
        self, modelname: str, model_dir: str | Path | None = None
    ) -> Path:
        filename = f"{modelname}{self.engine_suffix}.engine"
        return self.model_dir(modelname, model_dir) / filename


MODEL_VARIANTS = {
    "best": ModelVariant(
        name="best",
        config="train_config.json",
        checkpoint="best.pth",
        output_suffix="",
        onnx_suffix="",
        engine_suffix="_fp16",
    ),
    "tuned": ModelVariant(
        name="tuned",
        config="tuning/tuned_config.json",
        checkpoint="best_tuned.pth",
        output_suffix="_best_tuned",
        onnx_suffix="_best_tuned",
        engine_suffix="_best_tuned_fp16",
    ),
}
MODEL_VARIANT_CHOICES = tuple(sorted(MODEL_VARIANTS))


def get_model_variant(model_variant: str | None = None) -> ModelVariant:
    """Return the model variant defaults, raising on unsupported names."""
    name = model_variant or DEFAULT_MODEL_VARIANT
    try:
        return MODEL_VARIANTS[name]
    except KeyError as exc:
        choices = ", ".join(MODEL_VARIANT_CHOICES)
        raise ValueError(f"model_variant must be one of: {choices}") from exc


def require_model_artifact_path(
    path: str | Path,
    *,
    modelname: str,
    model_dir: str | Path | None = None,
    label: str = "artifact",
) -> Path:
    """Require an artifact path to belong to one explicit model root."""
    root = (
        Path(model_dir) if model_dir is not None else Path("models") / str(modelname)
    ).resolve()
    candidate = Path(path)
    resolved = candidate.resolve()
    if resolved != root and not resolved.is_relative_to(root):
        raise ValueError(
            f"{label} must be inside model directory {root}; got {resolved}. "
            "Use --model-dir to select a different same-named model source."
        )
    return candidate


def require_config_modelname(
    cfg: dict,
    *,
    modelname: str,
    config_path: str | Path,
) -> None:
    """Require a training config to declare the requested model identity."""
    configured_name = cfg.get("modelname")
    if configured_name is None:
        raise ValueError(f"Training config {config_path} has no modelname.")
    if str(configured_name) != str(modelname):
        raise ValueError(
            f"Training config {config_path} declares modelname={configured_name!r}, "
            f"not requested modelname={modelname!r}."
        )


def default_config_path(
    modelname: str,
    model_variant: str | None = None,
    model_dir: str | Path | None = None,
) -> Path:
    return get_model_variant(model_variant).config_path(modelname, model_dir)


def default_checkpoint_path(
    modelname: str,
    model_variant: str | None = None,
    model_dir: str | Path | None = None,
) -> Path:
    return get_model_variant(model_variant).checkpoint_path(modelname, model_dir)


def default_onnx_path(
    modelname: str,
    model_variant: str | None = None,
    model_dir: str | Path | None = None,
) -> Path:
    return get_model_variant(model_variant).onnx_path(modelname, model_dir)


def default_trt_engine_path(
    modelname: str,
    model_variant: str | None = None,
    model_dir: str | Path | None = None,
) -> Path:
    return get_model_variant(model_variant).trt_engine_path(modelname, model_dir)
