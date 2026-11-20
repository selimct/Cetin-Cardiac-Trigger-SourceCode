#!/usr/bin/env python3
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).
"""Export pooled prediction CSVs and three 100 Hz model descriptions to Word.

Normal and synthetic exports replace their own section in the same document.
This command reads saved reports/configs only; it never runs inference/training.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "run"))

from create_model_table import build_rows, element, paragraph, Row, W  # noqa: E402
from create_prediction_metrics_overview import _metrics_row  # noqa: E402
from dkfz.live.artifact_discovery import (  # noqa: E402
    published_model_directory,
    resolve_model,
)

MODELS = {
    "ecg3_nopuls_delayed_100hz": "ECG-Only Model",
    "ecg3_delayed_100hz": "Base Model",
    "slope_slow_ecg3_delayed_100hz": "Slope Model",
}
SUPPLEMENT_ORDER = (
    "ecg3_delayed_100hz",
    "ecg3_nopuls_delayed_100hz",
    "slope_slow_ecg3_delayed_100hz",
)
RATE_FIELDS = (
    "exact_sensitivity",
    "exact_positive_predictive_value",
    "unique_detection_error",
    "beat_sensitivity",
    "beat_positive_predictive_value",
    "beat_detection_error",
)
COUNT_FIELDS = (
    "num_target_events",
    "exact_true_positive",
    "position_error",
    "unmatched_false_positive",
    "unmatched_false_negative",
)
TEMPLATE = ROOT / "config/word_tables/template.xml"
TAG_PREFIX = "dkfz.metrics."


def q(name):
    return f"{{{W}}}{name}"


def set_cell(cell, text):
    """Replace template text while keeping its paragraph and font formatting."""
    old_p = cell.find(q("p"))
    p = ET.Element(q("p"))
    if old_p is not None and old_p.find(q("pPr")) is not None:
        p.append(deepcopy(old_p.find(q("pPr"))))
    run = element(p, "r")
    style = cell.find(".//" + q("rPr"))
    style = deepcopy(style) if style is not None else ET.Element(q("rPr"))
    fonts = style.find(q("rFonts"))
    if fonts is None:
        fonts = element(style, "rFonts")
    fonts.attrib = {q("ascii"): "Calibri", q("hAnsi"): "Calibri"}
    if style.find(q("sz")) is None:
        element(style, "sz", val=20)
    run.append(style)
    element(run, "t").text = str(text)
    for child in list(cell):
        if child.tag != q("tcPr"):
            cell.remove(child)
    cell.append(p)


def fill_row(row, values):
    cells = row.findall(q("tc"))
    if len(cells) != len(values):
        raise ValueError("Word template has an unexpected column count.")
    for cell, value in zip(cells, values):
        set_cell(cell, value)
    return row


def table_template(kind):
    table = deepcopy(ET.parse(TEMPLATE).find(kind + "/" + q("tbl")))
    # Editing metadata and floating positioning do not belong in generated tables.
    for parent in table.iter():
        for child in list(parent):
            if child.tag in {q("bookmarkStart"), q("bookmarkEnd"), q("tblpPr")}:
                parent.remove(child)
    return table


def format_table(table, header_rows=2):
    """Keep template borders/shading and fit the table to the printable width."""
    props = table.find(q("tblPr"))
    width = props.find(q("tblW"))
    if width is None:
        width = element(props, "tblW")
    width.attrib = {q("w"): "10000", q("type"): "dxa"}
    layout = props.find(q("tblLayout"))
    if layout is None:
        layout = element(props, "tblLayout")
    layout.set(q("type"), "fixed")
    grid = table.find(q("tblGrid"))
    columns = grid.findall(q("gridCol"))
    total = sum(int(c.get(q("w"))) for c in columns)
    widths = [round(int(c.get(q("w"))) * 10000 / total) for c in columns]
    widths[-1] += 10000 - sum(widths)
    for column, size in zip(columns, widths):
        column.set(q("w"), str(size))
    for index, row in enumerate(table.findall(q("tr"))):
        properties = row.find(q("trPr"))
        if properties is None:
            properties = ET.Element(q("trPr"))
            row.insert(0, properties)
        if index < header_rows and properties.find(q("tblHeader")) is None:
            element(properties, "tblHeader")
        if properties.find(q("cantSplit")) is None:
            element(properties, "cantSplit")
        offset = 0
        for cell in row.findall(q("tc")):
            cp = cell.find(q("tcPr"))
            merge = cp.find(q("vMerge"))
            if index >= header_rows and merge is not None:
                cp.remove(merge)
            span = cp.find(q("gridSpan"))
            count = int(span.get(q("val"))) if span is not None else 1
            cw = cp.find(q("tcW"))
            cw.set(q("w"), str(sum(widths[offset : offset + count])))
            offset += count


def read_results(path, synthetic):
    rows = {}
    with path.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            system = row["system"]
            if system not in MODELS and not (not synthetic and system == "siemens_tr"):
                continue
            raw_frequency = row.get("main_frequency_hz", "")
            if bool(raw_frequency) != synthetic:
                raise ValueError(f"Wrong normal/synthetic CSV type: {path}")
            frequency = float(raw_frequency) if synthetic else None
            if frequency is not None and (
                not math.isfinite(frequency) or frequency <= 0
            ):
                raise ValueError(f"Invalid synthetic frequency in {path}")
            if frequency == 90:
                continue
            if row["scope"] != ("gradient_config" if synthetic else "all_inputs"):
                raise ValueError(f"Expected pooled results, not per-input rows: {path}")
            for field in COUNT_FIELDS:
                value = int(row[field])
                if value < 0:
                    raise ValueError(f"Negative event count in {path}")
            for field in RATE_FIELDS:
                if not math.isfinite(float(row[field])):
                    raise ValueError(f"Invalid rate in {path}")
            # The template has no oPE column: use the default FP + FN policy,
            # even if the general CSV was requested with the optional oPE tier.
            if int(row.get("outer_position_error") or 0):
                counts = {
                    **row,
                    **{
                        field: int(row[field])
                        for field in (
                            *COUNT_FIELDS,
                            "timepoints",
                            "outer_position_error",
                        )
                    },
                }
                row = {
                    **row,
                    **_metrics_row(
                        counts,
                        scope=row["scope"],
                        input_label=row["input"],
                        evaluated_input_files=int(row["evaluated_input_files"]),
                        requested_input_files=int(row["requested_input_files"]),
                    ),
                }
            key = (frequency, system)
            if key in rows:
                raise ValueError(f"Duplicate pooled result for {system}: {path}")
            rows[key] = row
    if not rows:
        raise ValueError(f"No matching 100 Hz paper-model results in {path}")
    return rows


def result_values(label, row):
    if row is None:
        return [label] + ["—"] * 11
    return (
        [label]
        + [f"{100 * float(row[f]):.2f}" for f in RATE_FIELDS]
        + [str(int(row[f])) for f in COUNT_FIELDS]
    )


def results_section(mode, path, configs):
    synthetic = mode == "synthetic"
    rows = read_results(path, synthetic) if path is not None else {}
    container = ET.Element(q("sdtContent"))
    paragraph(
        container,
        f"Table {2 if synthetic else 1}. {'Synthetic' if synthetic else 'Normal'} metrics (100 Hz).",
        bold=True,
        keep_with_next=True,
    )
    table = table_template(mode)
    prototypes = table.findall(q("tr"))
    for row in prototypes[2:]:
        table.remove(row)
    if synthetic:
        fill_row(
            prototypes[0],
            ["Trigger", "Exact Performance", "Beat Performance", "Raw Performance"],
        )
        fill_row(
            prototypes[1],
            [""] + ["Se (%)", "+P (%)", "DER (%)"] * 2 + ["GT", "TP", "PE", "FP", "FN"],
        )
        frequencies = sorted({key[0] for key in rows}, reverse=True)
        if path is None:
            frequencies = [190, 80, 40, 20, 10]
        for frequency in frequencies:
            table.append(fill_row(deepcopy(prototypes[2]), [f"{frequency:g} Hz"]))
            for index, (system, label) in enumerate(MODELS.items()):
                row = deepcopy(prototypes[3 + index])
                table.append(
                    fill_row(row, result_values(label, rows.get((frequency, system))))
                )
    else:
        fill_row(
            prototypes[0],
            ["", "Exact Performance", "Beat Performance", "Raw Performance"],
        )
        fill_row(
            prototypes[1],
            ["Trigger"]
            + ["Se (%)", "+P (%)", "DER (%)"] * 2
            + ["GT", "TP", "PE", "FP", "FN"],
        )
        for system, label in {"siemens_tr": "Vendor Solution", **MODELS}.items():
            table.append(
                fill_row(
                    deepcopy(prototypes[2 if system == "siemens_tr" else 3]),
                    result_values(label, rows.get((None, system))),
                )
            )
    format_table(table)
    container.append(table)
    paragraph(
        container,
        "Se: sensitivity; +P: positive predictive value; DER: detection error rate. "
        "Outer position errors count as one FP and one FN. Missing results: —.",
        size=18,
    )
    for system in MODELS:
        raw = configs[system]
        frame_ms = 1000 * raw["frame_size_raw"] / raw["raw_hz"]
        exact = raw["exact_tolerance_frames"] * frame_ms
        beat = raw["error_association_frames"] * frame_ms
        paragraph(
            container,
            f"{MODELS[system]}: exact tolerance ±{exact:g} ms; beat tolerance ±{beat:g} ms.",
            size=18,
        )
    coverage = sorted(
        {
            f"{r['evaluated_input_files']}/{r['requested_input_files']} recordings"
            for r in rows.values()
        }
    )
    paragraph(
        container,
        (
            f"Source: {path.as_posix()}; coverage: {', '.join(coverage)}."
            if path is not None
            else "Results have not been exported yet."
        ),
        size=18,
    )
    if synthetic:
        paragraph(container, "90 Hz omitted from this Word table.", size=18)
    return container


def model_details(config, raw):
    # Read only compact configuration files. Training caches may be large or in
    # active use, and are unnecessary for a reporting-only export.
    _, rows = build_rows(config, include_gradient_updates=False)
    frame_ms = 1000 * raw["frame_size_raw"] / raw["raw_hz"]

    def add(label, value):
        rows.append(Row("", label, value, "Config"))

    add("Model identifier", raw["modelname"])
    delay = raw.get("prediction_delay_frames", 0)
    add(
        "Prediction mode / delay",
        f"{raw.get('prediction_mode', 'delayed_100hz')}; {delay} frame(s) ({delay * frame_ms:g} ms)",
    )
    add(
        "Event matching",
        f"Exact ±{raw['exact_tolerance_frames'] * frame_ms:g} ms; "
        f"beat ±{raw['error_association_frames'] * frame_ms:g} ms; "
        f"outer association ±{raw.get('outer_error_association_frames', 0) * frame_ms:g} ms",
    )
    if raw.get("feature_cols"):
        add("Feature columns", ", ".join(raw["feature_cols"]))
    for key, label in [
        ("seed", "Training seed"),
        ("target_reduce", "Target reduction"),
        ("threshold_selection_source", "Threshold selection"),
        ("epoch_selection_source", "Epoch selection"),
        ("checkpoint_policy", "Final checkpoint policy"),
    ]:
        if key in raw:
            add(label, str(raw[key]))
    if raw.get("threshold_objective"):
        add(
            "Threshold objective",
            str(raw["threshold_objective"])
            + "; error weights: "
            + ", ".join(
                f"{k}={v}" for k, v in raw.get("threshold_error_weights", {}).items()
            ),
        )
    if raw.get("final_training_patient_ids"):
        add("Final training patients", ", ".join(raw["final_training_patient_ids"]))
    if not any(row.parameter == "Gradient updates (final fit)" for row in rows):
        updates = raw.get("gradient_updates_final_fit")
        add(
            "Gradient updates (final fit)",
            (
                str(updates)
                if updates is not None
                else "Not recorded in saved configuration"
            ),
        )
    return rows


def supplement_section(config_paths, configs):
    container = ET.Element(q("sdtContent"))
    for number, system in enumerate(SUPPLEMENT_ORDER, 1):
        if number > 1:
            element(element(element(container, "p"), "r"), "br", type="page")
        paragraph(
            container,
            f"Table S{number}. Model architecture and training parameters for the {MODELS[system]}.",
            bold=True,
            keep_with_next=True,
        )
        table = table_template("supplement")
        header, prototype = table.findall(q("tr"))
        fill_row(header, ["Parameter", "Value"])
        table.remove(prototype)
        rows = model_details(config_paths[system], configs[system])
        for row in rows:
            table.append(
                fill_row(
                    deepcopy(prototype),
                    [
                        row.parameter + (" *" if row.source == "Default" else ""),
                        row.value,
                    ],
                )
            )
        format_table(table, header_rows=1)
        container.append(table)
        paragraph(
            container,
            "Detection settings are from the saved model configuration; the prototype threshold is manually adjustable. "
            "Gradient update counts, when available, are nominal training-batch counts.",
            size=18,
        )
        if any(row.source == "Default" for row in rows):
            paragraph(
                container,
                "* Includes values filled from current training defaults.",
                size=18,
            )
        paragraph(
            container, f"Configuration: {config_paths[system].as_posix()}", size=18
        )
    return container


def section_tag(section):
    tag = section.find(q("sdtPr") + "/" + q("tag"))
    return tag.get(q("val")) if tag is not None else None


def set_section(body, mode, content):
    tag = TAG_PREFIX + mode
    for section in body.findall(q("sdt")):
        if section_tag(section) == tag:
            previous = section.find(q("sdtContent"))
            section.remove(previous)
            section.append(content)
            return
    raise ValueError(f"Word document is missing the managed {mode} section.")


def new_document():
    doc = ET.Element(q("document"))
    body = element(doc, "body")
    for mode in ("normal", "synthetic", "supplement"):
        if mode != "normal":
            element(element(element(body, "p"), "r"), "br", type="page")
        section = element(body, "sdt")
        element(element(section, "sdtPr"), "tag", val=TAG_PREFIX + mode)
        content = element(section, "sdtContent")
        paragraph(content, f"{mode.capitalize()} tables have not been exported yet.")
    section = element(body, "sectPr")
    element(section, "pgSz", w=11906, h=16838)
    element(section, "pgMar", top=900, right=900, bottom=900, left=900)
    return doc


def write_document(output, normal_csv, synthetic_csv, config_paths):
    configs = {}
    for system in MODELS:
        raw = json.loads(config_paths[system].read_text(encoding="utf-8"))
        if (
            raw.get("modelname") != system
            or not math.isclose(raw["raw_hz"] / raw["frame_size_raw"], 100.0)
            or raw.get("recovery_head")
        ):
            raise ValueError(f"Expected the non-recovery 100 Hz config for {system}.")
        configs[system] = raw
    is_new = not output.exists()
    if not is_new:
        with ZipFile(output) as archive:
            parts = {name: archive.read(name) for name in archive.namelist()}
        doc = ET.fromstring(parts["word/document.xml"])
    else:
        doc = new_document()
        parts = {
            "[Content_Types].xml": (
                '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
                '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
                '<Default Extension="xml" ContentType="application/xml"/>'
                '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
                "</Types>"
            ).encode(),
            "_rels/.rels": (
                '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
                "</Relationships>"
            ).encode(),
        }
    body = doc.find(q("body"))
    for mode, path in (("normal", normal_csv), ("synthetic", synthetic_csv)):
        if path is not None or is_new:
            set_section(body, mode, results_section(mode, path, configs))
    set_section(body, "supplement", supplement_section(config_paths, configs))
    parts["word/document.xml"] = ET.tostring(
        doc, encoding="utf-8", xml_declaration=True
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    # A failed export leaves the previous complete document intact.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output.parent, suffix=".docx", delete=False
        ) as stream:
            temp_path = Path(stream.name)
            with ZipFile(stream, "w", compression=ZIP_DEFLATED) as archive:
                for name, data in parts.items():
                    archive.writestr(name, data)
        os.replace(temp_path, output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normal-csv", type=Path)
    parser.add_argument("--synthetic-csv", type=Path)
    parser.add_argument("--models-root", type=Path, default=ROOT / "finished_models")
    parser.add_argument(
        "--model-config", action="append", default=[], metavar="MODEL=PATH"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "metrics/Tables_metrics.docx"
    )
    args = parser.parse_args(argv)
    if not (args.normal_csv or args.synthetic_csv):
        parser.error("Supply --normal-csv and/or --synthetic-csv.")
    if args.output.suffix.lower() != ".docx":
        parser.error("Word output must have a .docx extension.")
    try:
        paths = {
            system: published_model_directory(system, args.models_root)
            / "train_config.json"
            for system in MODELS
        }
        explicit_configs = set()
        for override in args.model_config:
            name, path = override.split("=", 1)
            if name in MODELS:
                paths[name] = Path(path)
                explicit_configs.add(name)
        # Public model directories may differ from their saved training names.
        # Keep explicit config paths authoritative, including missing paths.
        for system, path in paths.items():
            if system not in explicit_configs and not path.is_file():
                artifact = resolve_model(system, roots=(args.models_root,))
                if artifact is not None:
                    paths[system] = artifact.directory / "train_config.json"
        write_document(args.output, args.normal_csv, args.synthetic_csv, paths)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        ET.ParseError,
        BadZipFile,
    ) as error:
        parser.error(str(error))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
