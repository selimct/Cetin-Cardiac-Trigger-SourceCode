#!/usr/bin/env python
# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Request a native runtime threshold change without restarting inference.

Run from the runtime's working directory. This utility uses only the Python
standard library; it never rewrites the deployed manifest or model artifacts.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import re
import tempfile


def read_config(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or key in values:
            raise ValueError(f"{path}:{number}: malformed or duplicate setting")
        values[key] = value.strip()
    return values


def override_text(
    manifest: dict[str, str], threshold: float, early_emission: bool = True
) -> str:
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and within [0, 1]")
    keys = ("model_name", "model_variant", "engine_size_bytes", "engine_crc32")
    if any(not manifest.get(key) for key in keys):
        raise ValueError(
            "Manifest must contain the model identity and engine fingerprint"
        )
    size = manifest["engine_size_bytes"]
    if not size.isascii() or not size.isdecimal() or not 0 < int(size) < 2**64:
        raise ValueError("Invalid engine_size_bytes in manifest")
    if not re.fullmatch(r"[0-9a-f]{8}", manifest["engine_crc32"]):
        raise ValueError("Invalid engine_crc32 in manifest")
    identity = {key: manifest[key] for key in keys}
    identity["engine_size_bytes"] = str(int(size))
    if any(any(char in value for char in "\r\n#") for value in identity.values()):
        raise ValueError("Malformed model identity")
    text = "".join(f"{key}={value}\n" for key, value in identity.items())
    text += f"threshold={threshold!r}\n"
    text += f"early_emission={'true' if early_emission else 'false'}\n"
    if len(text.encode("utf-8")) > 1024:
        raise ValueError("Threshold override exceeds 1024 bytes")
    return text


def write_override(path: Path, text: str) -> None:
    # Same-directory replacement makes the supervisor see either complete file.
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            delete=False,
        ) as output:
            temporary = output.name
            output.write(text)
            output.flush()
            os.fsync(output.fileno())
            # A privileged operator may write this for the service user.
            os.fchmod(output.fileno(), 0o644)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("threshold", type=float, nargs="?")
    parser.add_argument(
        "--reset", action="store_true", help="Restore the manifest threshold"
    )
    parser.add_argument(
        "--config", type=Path, default=Path("config/native_uart_live.conf")
    )
    parser.add_argument(
        "--manifest", type=Path, help="Override the config's manifest_path"
    )
    parser.add_argument(
        "--override-file", type=Path, help="Override threshold_override_path"
    )
    parser.add_argument(
        "--early-emission",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Emit peaks on crossing the higher original threshold; default from runtime config",
    )
    args = parser.parse_args(argv)
    if (args.threshold is None) == (not args.reset):
        parser.error("Provide either a threshold or --reset")
    try:
        config = (
            {} if args.manifest and args.override_file else read_config(args.config)
        )
        manifest_path = args.manifest or Path(config["manifest_path"])
        override_path = args.override_file
        if override_path is None:
            if not config.get("threshold_override_path"):
                raise ValueError(
                    "Live threshold control is disabled in the runtime config"
                )
            override_path = Path(config["threshold_override_path"])
        manifest = read_config(manifest_path)
        threshold = float(manifest["threshold"]) if args.reset else args.threshold
        mode = config.get("early_emission", "true")
        if args.early_emission is None and mode not in {"true", "false", "1", "0"}:
            raise ValueError(
                "early_emission must be true, false, 1, or 0 in the runtime config"
            )
        early_emission = (
            args.early_emission
            if args.early_emission is not None
            else mode in {"true", "1"}
        )
        write_override(
            override_path, override_text(manifest, threshold, early_emission)
        )
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))
    print(
        f"Requested threshold {threshold:g} for {manifest['model_name']} via {override_path}"
    )
    print(f"Early emission: {'enabled' if early_emission else 'disabled'}")
    print(
        "Confirm threshold_changed or the active threshold in runtime_status diagnostics."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
