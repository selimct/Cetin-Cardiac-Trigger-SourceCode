# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Raw idle-delimited bit frames reconstructed from GPIO edge captures.

This module preserves physical bit strings without interpreting their protocol
or payload layout.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from dkfz.live.gpio_edges import EdgeRecord, iter_edge_file
from dkfz.live.uart_byte_stream import (
    DEFAULT_UART_SIGNAL_OFFSET,
    GROUP_BYTES,
    UartSampleExtractor,
)


@dataclass(frozen=True)
class GpioEdgeConversionStats:
    packets: int
    samples: int
    startup_samples_discarded: int
    trailing_samples_discarded: int


def _uart_level_at(
    timestamp_ns: float,
    edge_times: list[int],
    edge_levels: list[int],
) -> int:
    position = bisect_right(edge_times, timestamp_ns) - 1
    if position < 0:
        raise ValueError("Cannot sample before the first UART edge.")
    return edge_levels[position]


def _decode_uart_bit_frame(
    frame_edges: list[EdgeRecord],
    *,
    end_ns: int,
    bit_rate_hz: float,
) -> str | None:
    """Return all start, data, parity, and stop bits in one 8E1 frame."""
    bit_time_ns = 1_000_000_000.0 / bit_rate_hz
    character_bits = 11
    start_ns = int(frame_edges[0].timestamp_ns)
    edge_times = [int(edge.timestamp_ns) for edge in frame_edges]
    edge_levels = [edge.level_after for edge in frame_edges]
    output: list[str] = []
    character_index = 0

    while True:
        character_start_ns = (
            start_ns + character_index * character_bits * bit_time_ns
        )
        final_sample_ns = character_start_ns + 10.5 * bit_time_ns
        if final_sample_ns >= end_ns:
            break

        sampled = [
            _uart_level_at(
                character_start_ns + (bit_index + 0.5) * bit_time_ns,
                edge_times,
                edge_levels,
            )
            for bit_index in range(character_bits)
        ]
        if sampled[0] != 0:
            break
        if sampled[10] != 1:
            return None

        output.extend(str(bit) for bit in sampled)
        character_index += 1

    return "".join(output) or None


def _iter_uart_bit_frame_candidates(
    edges: Iterable[EdgeRecord],
    *,
    bit_rate_hz: float,
    idle_run_bits: int,
) -> Iterator[str | None]:
    """Yield decoded frames and ``None`` markers for damaged candidates."""
    bit_time_ns = 1_000_000_000.0 / bit_rate_hz
    previous: EdgeRecord | None = None
    frame_edges: list[EdgeRecord] | None = None

    for event in edges:
        if previous is None:
            if event.is_falling and not event.has_loss:
                frame_edges = [event]
            previous = event
            continue

        sequence_gap = (
            (int(event.sequence) - int(previous.sequence)) & 0xFFFFFFFF
        ) != 1
        interval_ns = int(event.timestamp_ns) - int(previous.timestamp_ns)
        if event.has_loss or sequence_gap or interval_ns <= 0:
            if frame_edges is not None:
                yield None
            frame_edges = None
            previous = event
            continue

        begins_after_idle = (
            previous.level_after == 1
            and event.is_falling
            and interval_ns >= idle_run_bits * bit_time_ns
        )
        if begins_after_idle:
            if frame_edges:
                yield _decode_uart_bit_frame(
                    frame_edges,
                    end_ns=int(event.timestamp_ns),
                    bit_rate_hz=bit_rate_hz,
                )
            frame_edges = [event]
        elif frame_edges is not None:
            frame_edges.append(event)

        previous = event


def iter_uart_bit_frames(
    edges: Iterable[EdgeRecord],
    *,
    bit_rate_hz: float = 38_600,
    idle_run_bits: int = 10,
) -> Iterator[str]:
    """Yield complete idle-delimited 8E1 UART frames as physical bit strings.

    Every UART character contributes its LOW start bit, eight LSB-first data
    bits, even-parity bit, and HIGH stop bit. Bits are sampled at their nominal
    centers, so a final parity or stop bit that has no edge of its own is still
    represented in the result.

    The final candidate in an edge stream is deliberately not emitted because
    no following start edge exists to prove where its idle interval ends.
    """
    if bit_rate_hz <= 0:
        raise ValueError("bit_rate_hz must be positive.")
    if idle_run_bits < 2:
        raise ValueError("idle_run_bits must be at least two.")

    for bits in _iter_uart_bit_frame_candidates(
        edges,
        bit_rate_hz=bit_rate_hz,
        idle_run_bits=idle_run_bits,
    ):
        if bits is not None:
            yield bits


def _uart_bit_candidates_to_packet_dataframe(
    candidates: Iterable[str | None],
    *,
    frames_per_packet: int,
    long_frame_characters: int,
    short_frame_characters: int,
) -> pd.DataFrame:
    long_frame_bits = long_frame_characters * 11
    short_frame_bits = short_frame_characters * 11
    long_frames_per_packet = frames_per_packet - 1
    pending: list[str] = []
    damaged_packet = False
    packets: list[list[str]] = []

    for bits in candidates:
        if bits is None:
            pending.clear()
            damaged_packet = True
            continue

        if len(bits) == long_frame_bits:
            pending.append(bits)
            if len(pending) > long_frames_per_packet:
                damaged_packet = True
            continue

        if len(bits) == short_frame_bits:
            if not damaged_packet and len(pending) == long_frames_per_packet:
                packets.append([*pending, bits])
            pending.clear()
            damaged_packet = False
            continue

        pending.clear()
        damaged_packet = True

    def data_bits(frame_bits: str) -> str:
        return "".join(
            frame_bits[offset + 1 : offset + 9]
            for offset in range(0, len(frame_bits), 11)
        )

    frame_columns = [f"frame_{index}_bits" for index in range(frames_per_packet)]
    rows = []
    for index, packet in enumerate(packets):
        frame_data_bits = [data_bits(frame) for frame in packet]
        rows.append(
            {
                "idx": index,
                **dict(zip(frame_columns, frame_data_bits, strict=True)),
                "dbits": "".join(frame_data_bits),
            }
        )
    columns = ["idx", *frame_columns, "dbits"]
    dataframe = pd.DataFrame(rows, columns=columns)
    dataframe["idx"] = pd.Series(range(len(dataframe)), dtype="int64")
    for column in (*frame_columns, "dbits"):
        dataframe[column] = dataframe[column].astype("string")
    return dataframe


def uart_edges_to_packet_bit_dataframe(
    edges: Iterable[EdgeRecord],
    *,
    bit_rate_hz: float = 38_600,
    idle_run_bits: int = 10,
    frames_per_packet: int = 4,
    long_frame_characters: int = 7,
    short_frame_characters: int = 5,
) -> pd.DataFrame:
    """Return complete UART packets with one data-bit column per frame.

    A packet is recognized as ``frames_per_packet - 1`` long frames followed
    by one short frame. This makes the short frame the final frame in every
    output row and discards partial packet groups at the beginning and end.
    Malformed UART frames and capture-loss boundaries also prevent incomplete
    groups from being joined together. The per-frame columns and ``dbits``
    omit UART start, parity, and stop bits.
    """
    if bit_rate_hz <= 0:
        raise ValueError("bit_rate_hz must be positive.")
    if idle_run_bits < 2:
        raise ValueError("idle_run_bits must be at least two.")
    if frames_per_packet < 2:
        raise ValueError("frames_per_packet must be at least two.")
    if long_frame_characters < 1:
        raise ValueError("long_frame_characters must be at least one.")
    if short_frame_characters < 1:
        raise ValueError("short_frame_characters must be at least one.")
    if short_frame_characters >= long_frame_characters:
        raise ValueError(
            "short_frame_characters must be less than long_frame_characters."
        )

    return _uart_bit_candidates_to_packet_dataframe(
        _iter_uart_bit_frame_candidates(
            edges,
            bit_rate_hz=bit_rate_hz,
            idle_run_bits=idle_run_bits,
        ),
        frames_per_packet=frames_per_packet,
        long_frame_characters=long_frame_characters,
        short_frame_characters=short_frame_characters,
    )


def edge_file_to_uart_packet_bit_dataframe(
    path: str | Path,
    *,
    bit_rate_hz: float = 38_600,
    idle_run_bits: int = 10,
    frames_per_packet: int = 4,
    long_frame_characters: int = 7,
    short_frame_characters: int = 5,
    allow_incomplete_tail: bool | None = None,
) -> pd.DataFrame:
    """Read a JEDGE01 file into complete rows of UART packet bits.

    The default output columns are ``idx``, ``frame_0_bits`` through
    ``frame_3_bits`` and ``dbits``. Each ``frame_*_bits`` column contains only
    that frame's eight data-bit positions per UART character, preserving
    LSB-first wire order. ``dbits`` concatenates those four frame columns.
    """
    path = Path(path)
    if allow_incomplete_tail is None:
        allow_incomplete_tail = path.name.endswith(".partial")
    records = (
        record
        for _header, record in iter_edge_file(
            path,
            allow_incomplete_tail=allow_incomplete_tail,
        )
    )
    return uart_edges_to_packet_bit_dataframe(
        records,
        bit_rate_hz=bit_rate_hz,
        idle_run_bits=idle_run_bits,
        frames_per_packet=frames_per_packet,
        long_frame_characters=long_frame_characters,
        short_frame_characters=short_frame_characters,
    )


def default_gpio_edge_parquet_path(input_path: str | Path) -> Path:
    """Return the default parquet path beside a GPIO edge capture."""
    input_path = Path(input_path)
    name = input_path.name
    if name.endswith(".partial"):
        name = name[: -len(".partial")]
    if name.endswith(".edges"):
        name = name[: -len(".edges")]
    return input_path.with_name(f"{name}.parquet")


def convert_gpio_edges_to_parquet(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    bit_rate_hz: float = 38_600,
    sample_rate_hz: float = 400.0,
    ecg1_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
    ecg2_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
    puls_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
    idle_run_bits: int = 10,
    frames_per_packet: int = 4,
    long_frame_characters: int = 7,
    short_frame_characters: int = 5,
    overwrite: bool = False,
) -> tuple[Path, GpioEdgeConversionStats]:
    """Convert a GPIO edge capture directly to sample-level parquet.

    This combines JEDGE01 file reading, 8E1 UART bit reconstruction, packet
    grouping, and the same source-calibrated ECG/pulse/respiration extraction
    used for raw UART byte captures. ECG1, ECG2, and pulse have their configured
    source offsets removed; ECG3 is derived from the centered ECG channels.
    Active ``.partial`` captures tolerate a final edge record that is still
    being written.
    """
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"GPIO edge capture not found: {input_path}")
    output_path = (
        Path(output_path)
        if output_path is not None
        else default_gpio_edge_parquet_path(input_path)
    )
    if output_path.resolve() == input_path.resolve():
        raise ValueError("The parquet output must not overwrite the edge capture.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use overwrite=True to replace it."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    if partial_path.resolve() == input_path.resolve():
        raise ValueError(
            "The temporary parquet output must not overwrite the edge capture."
        )
    if partial_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Partial output already exists: {partial_path}. "
                "Use overwrite=True to replace it."
            )
        partial_path.unlink()

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        packet_df = edge_file_to_uart_packet_bit_dataframe(
            input_path,
            bit_rate_hz=bit_rate_hz,
            idle_run_bits=idle_run_bits,
            frames_per_packet=frames_per_packet,
            long_frame_characters=long_frame_characters,
            short_frame_characters=short_frame_characters,
        )

        extractor = UartSampleExtractor(
            sample_rate_hz=sample_rate_hz,
            ecg1_offset=ecg1_offset,
            ecg2_offset=ecg2_offset,
            puls_offset=puls_offset,
        )
        rows: list[dict[str, object]] = []
        pending_pulse_block: list[dict[str, object]] = []
        output_sample_index = 0
        startup_samples_discarded = 0

        for packet_index, bits in enumerate(packet_df["dbits"]):
            group = bytes(
                int(bits[offset : offset + 8][::-1], 2)
                for offset in range(0, len(bits), 8)
            )
            group_rows = extractor.rows_for_group(
                group,
                capture_byte_offset=packet_index * GROUP_BYTES,
            )
            for row in group_rows:
                if row["puls_updated"]:
                    if pending_pulse_block:
                        startup_samples_discarded += len(pending_pulse_block)
                    pending_pulse_block = [row]
                elif pending_pulse_block:
                    pending_pulse_block.append(row)
                else:
                    startup_samples_discarded += 1
                    continue

                if len(pending_pulse_block) == 8:
                    if any(
                        block_row["puls_raw"] is None
                        or block_row["resp_raw"] is None
                        for block_row in pending_pulse_block
                    ):
                        startup_samples_discarded += len(pending_pulse_block)
                        pending_pulse_block = []
                        continue

                    for block_row in pending_pulse_block:
                        block_row["raw_idx"] = output_sample_index
                        block_row["time_ms"] = (
                            1000.0
                            * output_sample_index
                            / float(sample_rate_hz)
                        )
                        rows.append(block_row)
                        output_sample_index += 1
                    pending_pulse_block = []

        trailing_samples_discarded = len(pending_pulse_block)
        sample_df = pd.DataFrame(
            rows,
            columns=[
                "raw_idx",
                "time_ms",
                "ecg1_raw",
                "ecg2_raw",
                "ecg3_raw",
                "puls_raw",
                "resp_raw",
                "ext",
                "pr_sign",
                "puls_updated",
                "source_packet",
                "sample_in_packet",
                "capture_byte_offset",
            ],
        ).astype(
            {
                "raw_idx": "int64",
                "time_ms": "float64",
                "ecg1_raw": "float64",
                "ecg2_raw": "float64",
                "ecg3_raw": "float64",
                "puls_raw": "float64",
                "resp_raw": "uint16",
                "ext": "uint8",
                "pr_sign": "uint8",
                "puls_updated": "bool",
                "source_packet": "int64",
                "sample_in_packet": "uint8",
                "capture_byte_offset": "int64",
            }
        )
        table = pa.Table.from_pandas(sample_df, preserve_index=False)
        metadata = dict(table.schema.metadata or {})
        metadata.update(
            {
                b"source": str(input_path).encode(),
                b"sample_rate_hz": str(float(sample_rate_hz)).encode(),
                b"uart_bit_rate_hz": str(float(bit_rate_hz)).encode(),
                b"uart_ecg1_offset": str(float(ecg1_offset)).encode(),
                b"uart_ecg2_offset": str(float(ecg2_offset)).encode(),
                b"uart_puls_offset": str(float(puls_offset)).encode(),
                b"format": b"gpio-pin29-samples-v1",
            }
        )
        pq.write_table(
            table.replace_schema_metadata(metadata),
            partial_path,
            compression="snappy",
        )
        partial_path.replace(output_path)
    except ImportError as exc:
        partial_path.unlink(missing_ok=True)
        raise RuntimeError(
            "GPIO edge parquet conversion requires pyarrow in the "
            "project environment."
        ) from exc
    except Exception:
        partial_path.unlink(missing_ok=True)
        raise

    return output_path, GpioEdgeConversionStats(
        packets=len(packet_df),
        samples=len(sample_df),
        startup_samples_discarded=startup_samples_discarded,
        trailing_samples_discarded=trailing_samples_discarded,
    )


def iter_idle_delimited_bit_frames(
    edges: Iterable[EdgeRecord],
    *,
    bit_rate_hz: float = 38600,
    idle_level: int = 1,
    idle_run_bits: int = 10,
    timing_tolerance_bits: float = 0.35,
    maximum_frame_bits: int = 4096,
) -> Iterator[str]:
    """Yield raw bit strings that begin after a long idle interval.

    Edge-to-edge durations are rounded to the nearest nominal bit period.  A
    frame begins on the transition away from ``idle_level`` after at least
    ``idle_run_bits`` of idle, and is emitted when the next such transition is
    observed.  The long idle run itself is omitted because edge timestamps
    alone cannot distinguish protocol trailer bits from extra bus idle.

    A falling first edge is accepted when ``idle_level`` is HIGH, because the
    capture may have started while the bus was already idle.  Candidates that
    contain a capture-loss marker, sequence gap, non-monotonic timestamp,
    timing error, or too many bits are discarded.
    """
    if bit_rate_hz <= 0:
        raise ValueError("bit_rate_hz must be positive.")
    if idle_level not in {0, 1}:
        raise ValueError("idle_level must be 0 or 1.")
    if idle_run_bits < 1:
        raise ValueError("idle_run_bits must be at least one.")
    if not 0 < timing_tolerance_bits < 0.5:
        raise ValueError("timing_tolerance_bits must be between 0 and 0.5.")
    if maximum_frame_bits < 1:
        raise ValueError("maximum_frame_bits must be at least one.")

    bit_time_ns = 1_000_000_000.0 / float(bit_rate_hz)
    active_edge_level = 1 - idle_level
    previous: EdgeRecord | None = None
    candidate: list[str] | None = None

    for event in edges:
        if previous is None:
            if event.level_after == active_edge_level and not event.has_loss:
                candidate = []
            previous = event
            continue

        sequence_gap = (
            (int(event.sequence) - int(previous.sequence)) & 0xFFFFFFFF
        ) != 1
        if event.has_loss or sequence_gap:
            candidate = None
            previous = event
            continue

        interval_ns = int(event.timestamp_ns) - int(previous.timestamp_ns)
        if interval_ns <= 0:
            candidate = None
            previous = event
            continue

        run_bits = max(1, round(interval_ns / bit_time_ns))
        previous_level = previous.level_after
        long_idle = previous_level == idle_level and run_bits >= idle_run_bits

        if long_idle:
            if event.level_after == active_edge_level:
                if candidate:
                    yield "".join(candidate)
                candidate = []
            else:
                candidate = None
            previous = event
            continue

        residual_ns = abs(interval_ns - run_bits * bit_time_ns)
        if residual_ns > timing_tolerance_bits * bit_time_ns:
            candidate = None
            previous = event
            continue

        if candidate is not None:
            candidate.extend(str(previous_level) for _ in range(run_bits))
            if len(candidate) > maximum_frame_bits:
                candidate = None

        previous = event


def edges_to_bit_dataframe(
    edges: Iterable[EdgeRecord],
    *,
    bit_rate_hz: float = 38600,
    idle_level: int = 1,
    idle_run_bits: int = 10,
    timing_tolerance_bits: float = 0.35,
    maximum_frame_bits: int = 4096,
) -> pd.DataFrame:
    """Return idle-delimited raw frames as ``idx`` and ``bits`` columns."""
    bits = list(
        iter_idle_delimited_bit_frames(
            edges,
            bit_rate_hz=bit_rate_hz,
            idle_level=idle_level,
            idle_run_bits=idle_run_bits,
            timing_tolerance_bits=timing_tolerance_bits,
            maximum_frame_bits=maximum_frame_bits,
        )
    )
    return pd.DataFrame(
        {
            "idx": pd.Series(range(len(bits)), dtype="int64"),
            "bits": pd.Series(bits, dtype="string"),
        }
    )


def edge_file_to_bit_dataframe(
    path: str | Path,
    *,
    bit_rate_hz: float = 38600,
    idle_level: int = 1,
    idle_run_bits: int = 10,
    timing_tolerance_bits: float = 0.35,
    maximum_frame_bits: int = 4096,
    allow_incomplete_tail: bool | None = None,
) -> pd.DataFrame:
    """Parse a JEDGE01 file and return raw frames as a two-column DataFrame.

    ``bits`` is stored as a string so leading zeroes are preserved.  Active
    ``.partial`` captures automatically tolerate a record that is currently
    being written; finalized files remain strict.
    """
    path = Path(path)
    if allow_incomplete_tail is None:
        allow_incomplete_tail = path.name.endswith(".partial")
    records = (
        record
        for _header, record in iter_edge_file(
            path,
            allow_incomplete_tail=allow_incomplete_tail,
        )
    )
    return edges_to_bit_dataframe(
        records,
        bit_rate_hz=bit_rate_hz,
        idle_level=idle_level,
        idle_run_bits=idle_run_bits,
        timing_tolerance_bits=timing_tolerance_bits,
        maximum_frame_bits=maximum_frame_bits,
    )
