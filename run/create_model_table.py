#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
"""Export paper-relevant model and training settings to an editable Word table.

Examples (from the repository root)::

    .venv/bin/python run/create_model_table.py finished_models/ecg3
    .venv/bin/python run/create_model_table.py --config path/to/train_config.json
    .venv/bin/python run/create_model_table.py --modelname ecg3

Multiple inputs produce one document with a table per model::

    .venv/bin/python run/create_model_table.py finished_models/ecg3 finished_models/slope_ecg3

Single reports: dbg/<modelname>/<modelname>_parameters.docx
Combined reports: dbg/model_tables/<model1>__<model2>_parameters.docx
Only dbg/ is written. No training is run.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
import json
import hashlib
from pathlib import Path
import re
import sys
from xml.etree import ElementTree as ET
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dkfz.live.training_config import (  # noqa: E402
    load_training_config_values,
    training_kwargs_from_config_and_values,
)
from dkfz.live.artifact_discovery import published_model_directory  # noqa: E402


@dataclass(frozen=True)
class Row:
    section: str
    parameter: str
    value: str
    source: str


def format_value(value) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.10g}"
    if isinstance(value, dict):
        return "; ".join(f"{key}: {format_value(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return ", ".join(format_value(item) for item in value) or "None"
    return str(value)


def resolve_config(value: str) -> Path:
    """Accept a config, model directory, or unambiguous model name."""
    path = Path(value).expanduser()
    if path.is_file():
        return path.resolve()
    if path.is_dir():
        candidates = [path / "train_config.json"]
        if not candidates[0].is_file():
            candidates = [
                path / method / "train_config.json"
                for method in ("nested_lopo", "nested_lopo_hpo")
            ]
    else:
        if path.name != value or value in (".", ".."):
            raise ValueError(f"Config or model directory does not exist: {value}")
        candidates = [
            ROOT / "finished_models" / value / "train_config.json",
            ROOT / "models" / value / "train_config.json",
        ]
        published = (
            published_model_directory(value, ROOT / "finished_models")
            / "train_config.json"
        )
        if published not in candidates:
            candidates.append(published)
        candidates += [
            ROOT / "models" / value / method / "train_config.json"
            for method in ("nested_lopo", "nested_lopo_hpo")
        ]
    found = [candidate.resolve() for candidate in candidates if candidate.is_file()]
    if not found:
        raise ValueError(f"No train_config.json found for {value!r}.")
    if len(found) > 1:
        raise ValueError(
            "Multiple configs found; supply an explicit path: "
            + ", ".join(str(candidate) for candidate in found)
        )
    return found[0]


def network_defaults() -> dict:
    """Read scalar training-loop defaults without importing PyTorch or models."""
    tree = ast.parse((ROOT / "src/dkfz/core/network.py").read_text(encoding="utf-8"))
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "train_torch"
    )
    return {
        arg.arg: ast.literal_eval(default)
        for arg, default in zip(function.args.kwonlyargs, function.args.kw_defaults)
        if arg.arg in ("weight_decay", "grad_clip_norm") and default is not None
    }


def gradient_update_counts(
    config: Path, raw: dict, values: dict
) -> tuple[int, int, int] | None:
    """Count final-fit windows from an existing LOPO cache; never build a cache."""
    method = raw.get("cv_method")
    patients = raw.get("final_training_patient_ids")
    if method not in ("nested_lopo", "nested_lopo_hpo") or not patients:
        return None
    candidates = []
    if raw.get("prepped_file"):
        recorded = Path(raw["prepped_file"])
        candidates.append(recorded if recorded.is_absolute() else ROOT / recorded)
    # Covers configs in the CV root and in outer/<patient>/.
    candidates.extend(
        parent / "patient_frames.parquet"
        for parent in (config.parent, *config.parent.parents)
        if parent != ROOT and ROOT in parent.parents
    )
    base = str(raw.get("base_modelname") or raw.get("modelname") or "")
    family = base if base.endswith("_cv") else base + "_cv"
    candidates.append(ROOT / "models" / family / method / "patient_frames.parquet")
    cache = next((path for path in candidates if path.is_file()), None)
    if cache is None:
        return None

    import pandas as pd
    import pyarrow.parquet as pq
    from dkfz.core.dataset import add_live_sequence_id

    columns = pq.ParquetFile(cache).schema_arrow.names
    required = ["patient_id", "sid", "frame_idx"]
    if any(column not in columns for column in required):
        raise ValueError(f"Training cache lacks sequence identifiers: {cache}")
    if "source_file" in columns:
        required.append("source_file")
    frames = pd.read_parquet(cache, columns=required)
    frames = frames.loc[frames["patient_id"].isin(patients)]
    if set(frames["patient_id"].unique()) != set(patients):
        raise ValueError(
            f"Training cache does not contain the configured patients: {cache}"
        )
    # A finished-model export may outlive its original cache. Reject changed inputs.
    inputs = raw.get("inputs")
    if (
        isinstance(inputs, list)
        and inputs
        and all(str(item).endswith(".parquet") for item in inputs)
    ):
        if "source_file" in frames:

            def normalized(value):
                path = Path(value)
                return (path if path.is_absolute() else ROOT / path).resolve()

            expected = {normalized(item) for item in inputs}
            actual = {normalized(item) for item in frames["source_file"].unique()}
            if actual != expected:
                raise ValueError(
                    f"Training cache recordings differ from the model config: {cache}"
                )

    window = int(values["window_size"])
    batch = int(values["batch_size"])
    epochs = int(raw.get("selected_fixed_epoch") or values["num_epochs"])
    if min(window, batch, epochs) <= 0:
        raise ValueError("Window size, batch size, and epochs must be positive.")
    windows = 0
    for _, patient in frames.groupby("patient_id", sort=False):
        # Training remaps sequence IDs per patient, then uses this same helper.
        sequences = add_live_sequence_id(patient.copy())
        lengths = sequences.groupby("sid", sort=False).size()
        windows += int((lengths - window + 1).clip(lower=0).sum())
    if windows == 0:
        raise ValueError(f"No valid training windows in {cache}")
    per_epoch = (windows + batch - 1) // batch  # DataLoader has drop_last=False.
    return windows, per_epoch, per_epoch * epochs


def build_rows(config: Path) -> tuple[str, list[Row]]:
    """Build a compact methods table, grouping related settings."""
    raw = json.loads(config.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("The model config must be a JSON object.")
    cfg = load_training_config_values(str(config))
    values = training_kwargs_from_config_and_values({}, config_values=cfg)
    if "ecg3_col" in raw and raw["ecg3_col"] is None:
        values["ecg3_col"] = None
    name = str(raw.get("modelname") or config.parent.name)
    rows: list[Row] = []

    def add(label, value, *keys):
        source = (
            "Default"
            if any(
                cfg.get(key) is None and not (key == "ecg3_col" and key in raw)
                for key in keys
            )
            else "Config" if keys else "Code"
        )
        rows.append(Row("", label, format_value(value), source))

    def fmt(key):
        return format_value(values[key])

    hz = float(values["raw_hz"])
    frame = int(values["frame_size_raw"])
    if hz <= 0 or frame <= 0:
        raise ValueError("raw_hz and frame_size_raw must be positive.")
    frame_hz = hz / frame

    channels = ["ECG1", "ECG2"]
    if values["ecg3_col"]:
        channels.append("ECG3")
    if not values["not_use_puls"]:
        channels.append("pulse")
    inputs = ", ".join(channels)
    dimension = raw.get("input_dim") or len(raw.get("feature_cols") or [])
    if dimension:
        inputs += f"; {dimension} input features"
    add("Input signals", inputs, "ecg3_col", "not_use_puls")
    add(
        "Sampling / framing",
        f"{hz:g} Hz raw → {frame_hz:g} Hz model input ({frame} samples/frame)",
        "raw_hz",
        "frame_size_raw",
    )
    add(
        "ECG normalization",
        f"Rolling quantile {fmt('ecg_quantile')}; "
        f"{values['ecg_norm_window_samples'] / hz:g} s window",
        "ecg_quantile",
        "ecg_norm_window_samples",
    )
    if not values["not_use_puls"]:
        add(
            "Pulse normalization",
            f"Quantiles {fmt('puls_quantile_low')}–{fmt('puls_quantile_high')}; "
            f"{values['puls_norm_window_frames'] / frame_hz:g} s window"
            + ("; clipped to [0, 1]" if values["puls_clip"] else ""),
            "puls_quantile_low",
            "puls_quantile_high",
            "puls_norm_window_frames",
            "puls_clip",
        )
    extras = []
    if values["ecg_slope_mode"] != "none":
        slope = f"{values['ecg_slope_mode']} ECG slopes"
        if values["ecg_slope_mode"] in ("slow", "both"):
            slope += f" ({values['ecg_slope_slow_window_samples'] / hz:g} s)"
        extras.append(slope)
    if values["signal_quality_mode"] != "none":
        extras.append(f"signal quality ({values['signal_quality_mode']})")
    if values["ecg_rotation_angle"]:
        extras.append(f"ECG rotation {fmt('ecg_rotation_angle')}°")
    if extras:
        add(
            "Additional features",
            "; ".join(extras),
            "ecg_slope_mode",
            "signal_quality_mode",
        )
    add(
        "Architecture",
        f"Unidirectional LSTM, {fmt('num_layers')} layers × {fmt('hidden_dim')} hidden units; "
        "dense–ReLU–dense output head",
        "num_layers",
        "hidden_dim",
    )
    add(
        "Input window",
        f"{fmt('window_size')} frames ({values['window_size'] / frame_hz:g} s); stride 1 frame",
        "window_size",
        "raw_hz",
        "frame_size_raw",
    )
    add("Optimizer", "AdamW")
    add("Learning rate", values["lr"], "lr")
    add("Batch size", values["batch_size"], "batch_size")
    epochs = format_value(raw.get("selected_fixed_epoch") or values["num_epochs"])
    if raw.get("configured_max_epochs"):
        epochs += f" selected (maximum {raw['configured_max_epochs']})"
    add("Training epochs", epochs, "num_epochs")
    updates = gradient_update_counts(config, raw, values)
    if updates is not None:
        _, per_epoch, total = updates
        add(
            "Gradient updates (final fit)",
            f"{total:,} nominal ({per_epoch:,}/epoch × {total // per_epoch} epochs)",
        )
    add("Loss", "Binary cross-entropy with logits")
    loop = network_defaults()
    add(
        "Regularization",
        f"Dropout 0; weight decay {format_value(loop['weight_decay'])}; "
        f"gradient norm limit {format_value(loop['grad_clip_norm'])}",
    )
    cv = raw.get("cv_method")
    if cv:
        design = {
            "nested_lopo": "Nested leave-one-patient-out",
            "nested_lopo_hpo": "Nested leave-one-patient-out with hyperparameter search",
        }.get(cv, str(cv).replace("_", " "))
        if (
            raw.get("checkpoint_policy")
            == "global_nested_cv_epoch_threshold_full_data_fit"
        ):
            design += "; final fit on all training patients"
        add("Validation", design)
    elif values["threshold_inputs"]:
        add("Validation", "Separate threshold-selection recordings", "threshold_inputs")
    else:
        add(
            "Validation",
            f"Within-recording split; {100 * values['test_size']:g}% validation",
            "test_size",
        )
    threshold = raw.get("manual_threshold")
    if threshold is None:
        threshold = raw.get("threshold")
    if threshold is not None:
        holdoff = raw.get("holdoff_frames", values["min_distance"])
        add(
            "Detection",
            f"Threshold {format_value(threshold)}; holdoff {1000 * holdoff / frame_hz:g} ms",
        )
    return name, rows


W = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
ET.register_namespace("w", W)


def element(parent, name, **attributes):
    return ET.SubElement(
        parent,
        f"{{{W}}}{name}",
        {f"{{{W}}}{key}": str(value) for key, value in attributes.items()},
    )


def paragraph(parent, text, *, bold=False, size=20, keep_with_next=False):
    p = element(parent, "p")
    properties = element(p, "pPr")
    element(properties, "spacing", after=60)
    if keep_with_next:
        element(properties, "keepNext")
    run = element(p, "r")
    style = element(run, "rPr")
    element(style, "rFonts", ascii="Calibri", hAnsi="Calibri")
    element(style, "sz", val=size)
    if bold:
        element(style, "b")
    element(run, "t").text = text


def append_model_table(body, name: str, rows: list[Row], number: int) -> None:
    """Append one model's compact table and its notes."""
    paragraph(
        body,
        f"Table S{number}. Model architecture and training parameters for the {name} model.",
        bold=True,
        size=22,
        keep_with_next=True,
    )
    table = element(body, "tbl")
    properties = element(table, "tblPr")
    element(properties, "tblW", w=10000, type="dxa")
    element(properties, "tblLayout", type="fixed")
    borders = element(properties, "tblBorders")
    for side in ("top", "left", "bottom", "right", "insideH", "insideV"):
        element(borders, side, val="single", sz=4, color="B7C9D6")
    widths = (2700, 7300)
    grid = element(table, "tblGrid")
    for width in widths:
        element(grid, "gridCol", w=width)
    contents = [("Parameter", "Value")]
    contents += [
        (row.parameter + (" *" if row.source == "Default" else ""), row.value)
        for row in rows
    ]
    for index, cells in enumerate(contents):
        tr = element(table, "tr")
        if index == 0:
            element(element(tr, "trPr"), "tblHeader")
        for width, text in zip(widths, cells):
            cell = element(tr, "tc")
            properties = element(cell, "tcPr")
            element(properties, "tcW", w=width, type="dxa")
            if index == 0:
                element(properties, "shd", fill="DCE6F1")
            paragraph(cell, text, bold=index == 0)
    if any(row.source == "Default" for row in rows):
        paragraph(
            body, "* Includes values filled from current training defaults.", size=18
        )
    if any(row.parameter == "Gradient updates (final fit)" for row in rows):
        paragraph(
            body,
            "Nominal updates count training batches; mixed-precision overflow skips are not recorded.",
            size=18,
        )


def write_docx(path: Path, name: str, config: Path, rows: list[Row]) -> None:
    """Write a single-model document."""
    write_reports(path, [(name, rows)])


def write_reports(path: Path, reports: list[tuple[str, list[Row]]]) -> None:
    """Write one Word document, starting each model on a new page."""
    document = ET.Element(f"{{{W}}}document")
    body = element(document, "body")
    for index, (name, rows) in enumerate(reports):
        if index:
            element(element(element(body, "p"), "r"), "br", type="page")
        append_model_table(body, name, rows, number=index + 1)
    section = element(body, "sectPr")
    element(section, "pgSz", w=11906, h=16838)
    element(section, "pgMar", top=900, right=900, bottom=900, left=900)
    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
        "</Relationships>"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", relationships)
        archive.writestr(
            "word/document.xml",
            ET.tostring(document, encoding="utf-8", xml_declaration=True),
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "models",
        nargs="*",
        help="One or more model names, directories, or JSON configs",
    )
    parser.add_argument(
        "--config",
        action="append",
        help="Explicit JSON config; repeat for multiple models",
    )
    parser.add_argument(
        "--modelname",
        help="Model to find if no inputs are given; otherwise overrides the output name",
    )
    args = parser.parse_args(argv)
    if args.models and args.config:
        parser.error("Use either positional models/configs or --config.")
    inputs = args.config or args.models or ([args.modelname] if args.modelname else [])
    if not inputs:
        parser.error("Supply models, --config, or --modelname.")
    try:
        reports = []
        configs = list(dict.fromkeys(resolve_config(value) for value in inputs))
        for config in configs:
            name, rows = build_rows(config)
            if len(configs) == 1:
                name = args.modelname or name
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
                raise ValueError(
                    "Model name must start with a letter/digit and contain only letters, digits, _, . or -."
                )
            reports.append((name, rows))
        output_name = args.modelname or "__".join(name for name, _ in reports)
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", output_name):
            raise ValueError(
                "Output name must contain only letters, digits, _, . or - and start with a letter/digit."
            )
        # Keep combined filenames within filesystem limits for larger model sets.
        if len(output_name) > 180:
            digest = hashlib.sha256(output_name.encode()).hexdigest()[:12]
            output_name = output_name[:160] + "__" + digest
        directory = reports[0][0] if len(reports) == 1 else "model_tables"
        output = ROOT / "dbg" / directory / f"{output_name}_parameters.docx"
        if any(parent.is_symlink() for parent in (ROOT / "dbg", output.parent, output)):
            raise ValueError("Report output must not be a symlink.")
        write_reports(output, reports)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(f"Wrote {output} ({len(reports)} model table(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
