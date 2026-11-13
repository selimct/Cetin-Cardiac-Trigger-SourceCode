# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Configuration and deterministic candidate generation for nested LOPO HPO."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from dkfz.live.training_config import TRAINING_DEFAULTS

HPO_PARAMETERS = {
    "lr": float,
    "hidden_dim": int,
    "num_layers": int,
    "window_size": int,
    "batch_size": int,
}
HPO_SELECTION_METRICS = {
    "weighted_detection_error",
    "weighted_error_count",
    "outer_position_error",
    "exact_sensitivity",
    "exact_positive_predictive_value",
    "unique_detection_error",
    "beat_sensitivity",
    "beat_positive_predictive_value",
    "unmatched_false_positive",
    "unmatched_false_negative",
    "unique_error_count",
    "position_error",
}
_CONTROLLED_FIXED_KEYS = {
    "config",
    "inputs",
    "threshold_inputs",
    "modelname",
    "prepped_file",
    "signal_quality_stats",
}


def _positive_parameter(name: str, value: Any) -> int | float:
    caster = HPO_PARAMETERS[name]
    try:
        converted = caster(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"HPO parameter {name!r} has invalid value {value!r}."
        ) from exc
    if isinstance(value, bool) or not math.isfinite(float(converted)):
        raise ValueError(f"HPO parameter {name!r} must be a finite positive number.")
    if float(converted) <= 0:
        raise ValueError(f"HPO parameter {name!r} must be greater than zero.")
    if caster is int and float(converted) != float(value):
        raise ValueError(f"HPO parameter {name!r} must contain integer values.")
    return converted


@dataclass(frozen=True)
class SearchDimension:
    """One categorical or sampled HPO parameter definition."""

    name: str
    kind: str
    values: tuple[int | float, ...] = ()
    low: int | float | None = None
    high: int | float | None = None
    step: int | None = None

    def sample(self, rng: np.random.Generator) -> int | float:
        if self.kind == "categorical":
            return self.values[int(rng.integers(0, len(self.values)))]
        if self.kind == "uniform":
            return float(rng.uniform(float(self.low), float(self.high)))
        if self.kind == "log_uniform":
            return float(
                np.exp(rng.uniform(np.log(float(self.low)), np.log(float(self.high))))
            )
        if self.kind == "int_uniform":
            values = self.grid_values()
            return values[int(rng.integers(0, len(values)))]
        raise AssertionError(f"Unhandled HPO dimension kind: {self.kind}")

    def grid_values(self) -> tuple[int | float, ...]:
        if self.kind == "categorical":
            return self.values
        if self.kind == "int_uniform":
            assert (
                self.low is not None and self.high is not None and self.step is not None
            )
            return tuple(range(int(self.low), int(self.high) + 1, int(self.step)))
        raise ValueError(
            f"Grid search requires categorical or int_uniform dimensions; "
            f"{self.name!r} uses {self.kind!r}."
        )


@dataclass(frozen=True)
class HPOCandidate:
    """One deterministic hyperparameter candidate."""

    trial_id: int
    name: str
    parameters: dict[str, int | float]
    is_baseline: bool = False


@dataclass(frozen=True)
class HPOConfig:
    """Validated HPO search policy."""

    method: str
    num_trials: int | None
    seed: int
    fixed: dict[str, Any]
    dimensions: tuple[SearchDimension, ...]
    include_baseline: bool
    baseline_name: str
    baseline_parameters: dict[str, int | float]
    selection_metric: str
    selection_direction: str
    raw: dict[str, Any]


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object.")
    return dict(value)


def _parse_dimension(name: str, raw: Any) -> SearchDimension:
    if name not in HPO_PARAMETERS:
        raise ValueError(
            f"Unsupported HPO parameter {name!r}; expected one of "
            f"{sorted(HPO_PARAMETERS)}."
        )
    values = _mapping(raw, f"search_space.{name}")
    kind = str(values.get("type", "categorical"))
    if kind == "categorical":
        raw_values = values.get("values")
        if (
            not isinstance(raw_values, Sequence)
            or isinstance(raw_values, str)
            or not raw_values
        ):
            raise ValueError(f"search_space.{name}.values must be a non-empty list.")
        converted = tuple(_positive_parameter(name, value) for value in raw_values)
        if len(set(converted)) != len(converted):
            raise ValueError(f"search_space.{name}.values contains duplicates.")
        return SearchDimension(name=name, kind=kind, values=converted)

    if kind not in {"uniform", "log_uniform", "int_uniform"}:
        raise ValueError(
            f"search_space.{name}.type must be categorical, uniform, "
            "log_uniform, or int_uniform."
        )
    if kind != "int_uniform" and HPO_PARAMETERS[name] is int:
        raise ValueError(f"Integer HPO parameter {name!r} must use int_uniform.")
    low = _positive_parameter(name, values.get("low"))
    high = _positive_parameter(name, values.get("high"))
    if float(low) >= float(high):
        raise ValueError(f"search_space.{name}.low must be less than high.")
    step = None
    if kind == "int_uniform":
        step = int(values.get("step", 1))
        if step < 1:
            raise ValueError(f"search_space.{name}.step must be at least one.")
    return SearchDimension(name=name, kind=kind, low=low, high=high, step=step)


def load_hpo_config(path: str | Path) -> HPOConfig:
    """Load and validate an HPO JSON file."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    raw = _mapping(raw, "HPO config")

    method = str(raw.get("method", "random"))
    if method not in {"random", "grid"}:
        raise ValueError("HPO method must be 'random' or 'grid'.")
    raw_num_trials = raw.get("num_trials", 12 if method == "random" else None)
    num_trials = None if raw_num_trials is None else int(raw_num_trials)
    if num_trials is not None and num_trials < 1:
        raise ValueError("HPO num_trials must be at least one.")
    seed = int(raw.get("seed", 42))
    if seed < 0:
        raise ValueError("HPO seed must be zero or positive.")

    fixed = _mapping(raw.get("fixed", {}), "fixed")
    unknown_fixed = set(fixed) - set(TRAINING_DEFAULTS)
    if unknown_fixed:
        raise ValueError(f"Unknown fixed training settings: {sorted(unknown_fixed)}.")
    controlled = set(fixed) & _CONTROLLED_FIXED_KEYS
    if controlled:
        raise ValueError(
            f"HPO fixed settings cannot control {sorted(controlled)}; use the CLI."
        )

    search_space = _mapping(raw.get("search_space"), "search_space")
    if not search_space:
        raise ValueError("HPO search_space must not be empty.")
    dimensions = tuple(
        _parse_dimension(name, values) for name, values in sorted(search_space.items())
    )

    raw_baseline = raw.get("baseline", True)
    include_baseline = raw_baseline is not False
    baseline_name = "current_configuration"
    baseline_parameters: dict[str, int | float] = {}
    if isinstance(raw_baseline, Mapping):
        baseline_values = dict(raw_baseline)
        baseline_name = str(baseline_values.get("name", baseline_name)).strip()
        if not baseline_name:
            raise ValueError("HPO baseline name must not be empty.")
        baseline_raw_parameters = _mapping(
            baseline_values.get("parameters", {}), "baseline.parameters"
        )
        unknown = set(baseline_raw_parameters) - {
            dimension.name for dimension in dimensions
        }
        if unknown:
            raise ValueError(
                "Baseline parameters must also appear in search_space; "
                f"unexpected {sorted(unknown)}."
            )
        baseline_parameters = {
            name: _positive_parameter(name, value)
            for name, value in baseline_raw_parameters.items()
        }
    elif raw_baseline not in {True, False}:
        raise ValueError("HPO baseline must be true, false, or a JSON object.")

    selection = _mapping(raw.get("selection", {}), "selection")
    metric = str(selection.get("metric", "weighted_detection_error"))
    if metric not in HPO_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported HPO selection metric {metric!r}; expected one of "
            f"{sorted(HPO_SELECTION_METRICS)}."
        )
    direction = str(
        selection.get(
            "direction",
            (
                "minimize"
                if "error" in metric or metric.startswith("unmatched")
                else "maximize"
            ),
        )
    )
    if direction not in {"minimize", "maximize"}:
        raise ValueError("HPO selection direction must be 'minimize' or 'maximize'.")

    if method == "grid":
        for dimension in dimensions:
            dimension.grid_values()

    return HPOConfig(
        method=method,
        num_trials=num_trials,
        seed=seed,
        fixed=fixed,
        dimensions=dimensions,
        include_baseline=include_baseline,
        baseline_name=baseline_name,
        baseline_parameters=baseline_parameters,
        selection_metric=metric,
        selection_direction=direction,
        raw=raw,
    )


def apply_hpo_fixed_settings(
    kwargs: Mapping[str, Any], config: HPOConfig
) -> dict[str, Any]:
    """Apply fixed HPO-file settings to resolved base training settings."""
    selected = dict(kwargs)
    selected.update(config.fixed)
    return selected


def _candidate_key(parameters: Mapping[str, int | float]) -> str:
    return json.dumps(dict(parameters), sort_keys=True, separators=(",", ":"))


def generate_hpo_candidates(
    config: HPOConfig,
    base_kwargs: Mapping[str, Any],
) -> tuple[HPOCandidate, ...]:
    """Generate a deterministic, de-duplicated candidate sequence."""
    parameter_names = tuple(dimension.name for dimension in config.dimensions)
    baseline_parameters = {
        name: _positive_parameter(name, base_kwargs[name]) for name in parameter_names
    }
    baseline_parameters.update(config.baseline_parameters)

    raw_candidates: list[tuple[str, dict[str, int | float], bool]] = []
    if config.include_baseline:
        raw_candidates.append((config.baseline_name, baseline_parameters, True))

    if config.method == "grid":
        value_sets = [dimension.grid_values() for dimension in config.dimensions]
        for values in product(*value_sets):
            raw_candidates.append(
                (
                    "grid",
                    dict(zip(parameter_names, values, strict=True)),
                    False,
                )
            )
    else:
        assert config.num_trials is not None
        rng = np.random.default_rng(config.seed)
        attempts = 0
        sampled_keys = {
            _candidate_key(parameters)
            for _name, parameters, _baseline in raw_candidates
        }
        while len(raw_candidates) < config.num_trials and attempts < 10000:
            attempts += 1
            sampled = {
                dimension.name: dimension.sample(rng) for dimension in config.dimensions
            }
            key = _candidate_key(sampled)
            if key in sampled_keys:
                continue
            sampled_keys.add(key)
            raw_candidates.append(("random", sampled, False))

    unique: list[tuple[str, dict[str, int | float], bool]] = []
    seen: set[str] = set()
    for name, parameters, is_baseline in raw_candidates:
        key = _candidate_key(parameters)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, parameters, is_baseline))
        if config.num_trials is not None and len(unique) >= config.num_trials:
            break

    if not unique:
        raise ValueError("HPO configuration produced no candidates.")
    if (
        config.method == "random"
        and config.num_trials is not None
        and len(unique) < config.num_trials
    ):
        raise ValueError(
            f"HPO search space produced only {len(unique)} unique candidates, "
            f"fewer than num_trials={config.num_trials}."
        )

    return tuple(
        HPOCandidate(
            trial_id=trial_id,
            name=(name if is_baseline else f"{name}_{trial_id:03d}"),
            parameters=parameters,
            is_baseline=is_baseline,
        )
        for trial_id, (name, parameters, is_baseline) in enumerate(unique)
    )


__all__ = [
    "HPOCandidate",
    "HPOConfig",
    "HPO_PARAMETERS",
    "HPO_SELECTION_METRICS",
    "SearchDimension",
    "apply_hpo_fixed_settings",
    "generate_hpo_candidates",
    "load_hpo_config",
]
