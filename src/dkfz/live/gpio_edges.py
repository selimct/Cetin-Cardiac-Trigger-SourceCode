# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Binary GPIO edge capture format and live file-following helpers.

The native ``gpio_edge_capture`` process writes fixed-size ``JEDGE01`` files.
This module deliberately contains no GPIO access: Python only consumes records
that have already been timestamped by the kernel and persisted by C++.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
import time
from typing import BinaryIO, Iterator


MAGIC = b"JEDGE01\x00"
FORMAT_VERSION = 1
HEADER_STRUCT = struct.Struct("<8sHHHHIIQQ24s")
RECORD_STRUCT = struct.Struct("<QIBBH")

EDGE_RISING = 1
EDGE_FALLING = 2

FLAG_KERNEL_SEQUENCE_GAP = 1 << 0
FLAG_DUPLICATE_EDGE = 1 << 1
FLAG_QUEUE_GAP = 1 << 2
LOSS_FLAGS = (
    FLAG_KERNEL_SEQUENCE_GAP | FLAG_DUPLICATE_EDGE | FLAG_QUEUE_GAP
)


@dataclass(frozen=True)
class EdgeFileHeader:
    version: int
    header_size: int
    record_size: int
    flags: int
    line_offset: int
    event_clock: int
    realtime_start_ns: int
    monotonic_start_ns: int
    gpio_chip_label: str

    def monotonic_to_realtime_ns(self, timestamp_ns: int) -> int:
        """Convert a capture-clock timestamp using the chunk's clock snapshot."""
        return int(self.realtime_start_ns) + (
            int(timestamp_ns) - int(self.monotonic_start_ns)
        )


@dataclass(frozen=True)
class EdgeRecord:
    timestamp_ns: int
    sequence: int
    edge: int
    flags: int = 0

    @property
    def is_rising(self) -> bool:
        return self.edge == EDGE_RISING

    @property
    def is_falling(self) -> bool:
        return self.edge == EDGE_FALLING

    @property
    def level_after(self) -> int:
        if self.edge == EDGE_RISING:
            return 1
        if self.edge == EDGE_FALLING:
            return 0
        raise ValueError(f"Unknown edge identifier: {self.edge}")

    @property
    def has_loss(self) -> bool:
        return bool(self.flags & LOSS_FLAGS)


def read_header(stream: BinaryIO) -> EdgeFileHeader:
    """Read and validate one fixed-size JEDGE01 header."""
    raw = stream.read(HEADER_STRUCT.size)
    if len(raw) != HEADER_STRUCT.size:
        raise ValueError(
            f"Truncated edge header: expected {HEADER_STRUCT.size} bytes, "
            f"got {len(raw)}."
        )
    (
        magic,
        version,
        header_size,
        record_size,
        flags,
        line_offset,
        event_clock,
        realtime_start_ns,
        monotonic_start_ns,
        chip_label,
    ) = HEADER_STRUCT.unpack(raw)
    if magic != MAGIC:
        raise ValueError(f"Invalid edge capture magic: {magic!r}")
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported edge capture version {version}; "
            f"expected {FORMAT_VERSION}."
        )
    if header_size != HEADER_STRUCT.size:
        raise ValueError(
            f"Unsupported edge header size {header_size}; "
            f"expected {HEADER_STRUCT.size}."
        )
    if record_size != RECORD_STRUCT.size:
        raise ValueError(
            f"Unsupported edge record size {record_size}; "
            f"expected {RECORD_STRUCT.size}."
        )
    return EdgeFileHeader(
        version=int(version),
        header_size=int(header_size),
        record_size=int(record_size),
        flags=int(flags),
        line_offset=int(line_offset),
        event_clock=int(event_clock),
        realtime_start_ns=int(realtime_start_ns),
        monotonic_start_ns=int(monotonic_start_ns),
        gpio_chip_label=chip_label.split(b"\x00", 1)[0].decode(
            "utf-8", errors="replace"
        ),
    )


def decode_record(raw: bytes) -> EdgeRecord:
    """Decode one fixed-size edge record."""
    if len(raw) != RECORD_STRUCT.size:
        raise ValueError(
            f"Expected {RECORD_STRUCT.size} edge bytes, got {len(raw)}."
        )
    timestamp_ns, sequence, edge, flags, _reserved = RECORD_STRUCT.unpack(raw)
    if edge not in {EDGE_RISING, EDGE_FALLING}:
        raise ValueError(f"Unknown edge identifier {edge}.")
    return EdgeRecord(
        timestamp_ns=int(timestamp_ns),
        sequence=int(sequence),
        edge=int(edge),
        flags=int(flags),
    )


def iter_edge_file(
    path: str | Path,
    *,
    allow_incomplete_tail: bool = False,
) -> Iterator[tuple[EdgeFileHeader, EdgeRecord]]:
    """Iterate records in one finalized or active partial capture.

    ``allow_incomplete_tail`` is useful for a snapshot of a file that the C++
    process is currently extending. Finalized files remain strict by default.
    """
    with Path(path).open("rb") as stream:
        header = read_header(stream)
        while True:
            raw = stream.read(header.record_size)
            if not raw:
                return
            if len(raw) != header.record_size:
                if allow_incomplete_tail:
                    return
                raise ValueError(
                    f"Truncated edge record at byte "
                    f"{stream.tell() - len(raw)} in {path}."
                )
            yield header, decode_record(raw)


def _capture_candidates(directory: Path) -> list[Path]:
    paths = [
        path
        for path in directory.iterdir()
        if path.is_file()
        and path.name.startswith("gpio_edges_")
        and (
            path.name.endswith(".edges")
            or path.name.endswith(".edges.partial")
        )
    ]
    return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, path.name))


def _newest_active_capture(
    directory: Path,
    *,
    newer_than_mtime_ns: int | None = None,
) -> Path | None:
    candidates = _capture_candidates(directory)
    active = [path for path in candidates if path.name.endswith(".partial")]
    ordered = active or candidates
    if newer_than_mtime_ns is not None:
        ordered = [
            path
            for path in ordered
            if path.stat().st_mtime_ns > newer_than_mtime_ns
        ]
    return ordered[-1] if ordered else None


def _seek_to_aligned_end(stream: BinaryIO, header: EdgeFileHeader) -> None:
    stream.seek(0, 2)
    size = stream.tell()
    complete_records = max(0, size - header.header_size) // header.record_size
    stream.seek(header.header_size + complete_records * header.record_size)


def follow_capture_directory(
    directory: str | Path,
    *,
    from_start: bool = False,
    poll_interval_s: float = 0.02,
) -> Iterator[tuple[EdgeFileHeader, EdgeRecord]]:
    """Follow the active capture and automatically cross hourly rotations.

    By default the first file is tailed from its current complete-record
    boundary. New files created after the follower starts are read from their
    first record, so rotation cannot create a gap.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive.")

    current_path: Path | None = None
    stream: BinaryIO | None = None
    header: EdgeFileHeader | None = None
    current_mtime_ns = -1
    first_open = True

    try:
        while True:
            if stream is None:
                candidate = _newest_active_capture(
                    directory,
                    newer_than_mtime_ns=(
                        None if current_mtime_ns < 0 else current_mtime_ns
                    ),
                )
                if candidate is None:
                    time.sleep(poll_interval_s)
                    continue

                stream = candidate.open("rb")
                try:
                    header = read_header(stream)
                except Exception:
                    stream.close()
                    stream = None
                    time.sleep(poll_interval_s)
                    continue
                current_path = candidate
                current_mtime_ns = candidate.stat().st_mtime_ns
                if first_open and not from_start:
                    _seek_to_aligned_end(stream, header)
                first_open = False

            assert stream is not None
            assert header is not None
            raw = stream.read(header.record_size)
            if len(raw) == header.record_size:
                yield header, decode_record(raw)
                continue
            if raw:
                stream.seek(-len(raw), 1)

            next_path = _newest_active_capture(
                directory,
                newer_than_mtime_ns=current_mtime_ns,
            )
            if next_path is not None and next_path != current_path:
                stream.close()
                stream = None
                header = None
                current_path = None
                continue
            time.sleep(poll_interval_s)
    finally:
        if stream is not None:
            stream.close()


def load_key_value_config(path: str | Path) -> dict[str, str]:
    """Load the shared dependency-free ``key=value`` configuration format."""
    values: dict[str, str] = {}
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, 1):
            line = raw_line.split("#", 1)[0].strip()
            if not line:
                continue
            if "=" not in line:
                raise ValueError(
                    f"{path}:{line_number}: expected a key=value setting."
                )
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values
