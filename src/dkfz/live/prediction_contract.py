# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Model target timing and the opt-in 100 Hz training preset.

A delayed primary uses the final recurrent state but is supervised on the
preceding frame. Inference must preserve separate target and observation axes.
"""

from __future__ import annotations
import math

PREDICTION_MODES = ("current", "delayed_100hz")
PREDICTION_CONFIG_KEYS = {
    "prediction_mode",
    "prediction_delay_frames",
    "signal_quality_puls_flat_window_frames",
    "signal_quality_puls_median_window_frames",
}


def supported_preprocessing_contract(settings: dict) -> int:
    """Preserve contract-3 50 Hz inputs without relabelling saved models.

    Contract 4 added delayed framing/reset metadata; the existing eight-sample
    current-frame feature transformations and default quality horizons match 3.
    Older caches are still rejected by the training/cache readers.
    """
    from dkfz.live.feature_framing import PREPROCESSING_CONTRACT_VERSION

    value = settings.get("preprocessing_contract_version", PREPROCESSING_CONTRACT_VERSION)
    version = int(value)
    if isinstance(value, bool) or float(value) != version:
        raise ValueError("preprocessing_contract_version must be an integer")
    legacy_50hz = (
        version == 3
        and settings.get("prediction_mode", "current") == "current"
        and settings.get("prediction_delay_frames", 0) == 0
        and float(settings.get("raw_hz", 400)) == 400
        and int(settings.get("frame_size_raw", 8)) == 8
        and int(settings.get("signal_quality_puls_flat_window_frames", 10)) == 10
        and int(settings.get("signal_quality_puls_median_window_frames", 200)) == 200
    )
    if version != PREPROCESSING_CONTRACT_VERSION and not legacy_50hz:
        raise ValueError(
            f"preprocessing_contract_version={version} is incompatible; supported "
            f"versions are {PREPROCESSING_CONTRACT_VERSION} and legacy contract 3 "
            "for current-frame 400 Hz/eight-sample models with default quality histories"
        )
    return version


def prediction_delay(settings: dict) -> int:
    mode = settings.get("prediction_mode", "current")
    if mode not in PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
    if mode == "delayed_100hz" and "prediction_delay_frames" not in settings:
        raise ValueError(
            "delayed_100hz model configs must declare prediction_delay_frames=1"
        )
    delay = settings.get("prediction_delay_frames", 0)
    if isinstance(delay, bool) or not isinstance(delay, int) or delay not in (0, 1):
        raise ValueError("prediction_delay_frames must be 0 or 1")
    if delay != (1 if mode == "delayed_100hz" else 0):
        raise ValueError("prediction_mode and prediction_delay_frames disagree")
    if delay:
        from dkfz.live.feature_framing import PREPROCESSING_CONTRACT_VERSION

        if (
            int(
                settings.get(
                    "preprocessing_contract_version", PREPROCESSING_CONTRACT_VERSION
                )
            )
            != PREPROCESSING_CONTRACT_VERSION
        ):
            raise ValueError(
                "Delayed primary requires the current preprocessing contract"
            )
        if float(settings["raw_hz"]) != 400 or int(settings["frame_size_raw"]) != 4:
            raise ValueError(
                "delayed_100hz requires 400 Hz input and four-sample frames"
            )
        if int(settings["window_size"]) < 2:
            raise ValueError("A delayed primary requires window_size >= 2")
        if (
            settings.get("recovery_head")
            or settings.get("recovery_only")
            or settings.get("recovery_threshold") is not None
        ):
            raise ValueError(
                "delayed_100hz uses one primary output; recovery is incompatible"
            )
    for key, default in (
        ("signal_quality_puls_flat_window_frames", 10),
        ("signal_quality_puls_median_window_frames", 200),
    ):
        value = settings.get(key, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 2:
            raise ValueError(f"{key} must be an integer >= 2")
    return delay


def apply_training_prediction_mode(kwargs: dict, config: dict, explicit: dict) -> dict:
    """Scale inherited 50 Hz time spans once; explicit frame counts win."""
    mode = kwargs["prediction_mode"]
    if mode not in PREDICTION_MODES:
        raise ValueError(f"prediction_mode must be one of {PREDICTION_MODES}")
    if mode == "delayed_100hz":
        source_size = int(config.get("frame_size_raw", 8))
        if explicit.get("min_distance") is None:
            # Saved runtime holdoff overrides also describe physical time in
            # the source model. Do not replace an explicit 400 ms policy with
            # the preset's 200 ms default when starting a fresh model.
            if config.get("holdoff_frames") is not None:
                kwargs["min_distance"] = int(config["holdoff_frames"])
        if config.get("prediction_mode") != "delayed_100hz":
            if source_size not in (4, 8):
                raise ValueError(
                    "The delayed_100hz preset can convert four- or eight-sample configurations"
                )
            factor = source_size // 4
            for key in (
                "window_size",
                "puls_norm_window_frames",
                "min_distance",
            ):
                if explicit.get(key) is None:
                    kwargs[key] *= factor
            if (
                config.get("signal_quality_stats") is not None
                and explicit.get("signal_quality_stats") is None
            ):
                kwargs["signal_quality_stats"] = None  # fit on the new training frames
        for key, default in (
            ("signal_quality_puls_flat_window_frames", 20),
            ("signal_quality_puls_median_window_frames", 400),
        ):
            if explicit.get(key) is None:
                kwargs[key] = (
                    int(config[key]) * (source_size // 4)
                    if key in config and config.get("prediction_mode") != mode
                    else int(config.get(key, default))
                )
        if (
            explicit.get("frame_size_raw") is not None
            and int(explicit["frame_size_raw"]) != 4
        ):
            raise ValueError("delayed_100hz requires --frame-size-raw 4")
        kwargs["frame_size_raw"] = 4
        if (
            explicit.get("min_distance") is None
            and config.get("holdoff_frames") is None
            and config.get("holdoff_ms") is not None
        ):
            kwargs["min_distance"] = math.ceil(
                float(config["holdoff_ms"]) * float(kwargs["raw_hz"]) / 4000
            )
        if isinstance(explicit.get("prediction_delay_frames"), bool) or explicit.get(
            "prediction_delay_frames"
        ) not in (None, 1):
            raise ValueError("delayed_100hz requires prediction_delay_frames=1")
        kwargs["prediction_delay_frames"] = 1
    else:
        kwargs["prediction_delay_frames"] = kwargs.get("prediction_delay_frames", 0)
    prediction_delay(kwargs)
    return kwargs


def check_training_destination(model_dir, settings):
    from pathlib import Path
    import json

    directory = Path(model_dir)
    config = directory / "train_config.json"
    if config.is_file():
        saved = json.loads(config.read_text())
        if saved.get("prediction_mode", "current") != settings.get(
            "prediction_mode", "current"
        ) or int(saved.get("prediction_delay_frames", 0)) != int(
            settings.get("prediction_delay_frames", 0)
        ):
            raise ValueError(
                "Existing model directory has a different prediction target; choose a new modelname"
            )
    elif settings.get("prediction_delay_frames") and any(directory.glob("*.pth")):
        raise ValueError(
            "Existing weights have no verified prediction contract; choose a new modelname"
        )
