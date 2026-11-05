# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Decode captured pin-10 UART bytes into source-calibrated sample rows."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import BinaryIO, Iterator

from dkfz.core.casting_helpers import subtract_offset_float64


GROUP_BYTES = 26
SAMPLES_PER_GROUP = 4
INTER_PACKET_BYTES = 3
SAMPLE_OFFSETS = (0, 7, 14, 21)
EXT_FRAME_OFFSETS = (0, 14)
EXT_BYTE_OFFSET = 5
DEFAULT_UART_SIGNAL_OFFSET = 2048.0


@dataclass
class UartByteConversionStats:
    bytes_read: int = 0
    bytes_discarded: int = 0
    malformed_packets: int = 0
    packets: int = 0
    samples: int = 0
    startup_samples_discarded: int = 0
    trailing_samples_discarded: int = 0
    trailing_bytes: int = 0

    @property
    def malformed_samples(self) -> int:
        return self.malformed_packets * SAMPLES_PER_GROUP


class UartByteGroupParser:
    """Recover 26-byte logical groups from an arbitrary byte stream."""

    def __init__(self, stats: UartByteConversionStats | None = None):
        self.stats = stats or UartByteConversionStats()
        self.buffer = bytearray()
        self.stream_offset = 0
        self.synchronized = False
        self.ever_synchronized = False
        self.resync_discarded = 0

    @staticmethod
    def _has_anchors(data: bytearray, start: int) -> bool:
        return all(
            (data[start + frame_offset + byte_offset] & 0xF0) == 0xD0
            for frame_offset in SAMPLE_OFFSETS
            for byte_offset in (3, 4)
        )

    def _discard(self, count: int) -> None:
        if count <= 0:
            return
        self.stats.bytes_discarded += count
        self._consume(count)

    def _consume(self, count: int) -> None:
        del self.buffer[:count]
        self.stream_offset += count

    def _take_group(self) -> tuple[int, bytes]:
        offset = self.stream_offset
        group = bytes(self.buffer[:GROUP_BYTES])
        del self.buffer[:GROUP_BYTES]
        self.stream_offset += GROUP_BYTES
        return offset, group

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        self.stats.bytes_read += len(data)
        self.buffer.extend(data)
        groups: list[tuple[int, bytes]] = []

        while len(self.buffer) >= GROUP_BYTES:
            if self.synchronized and self._has_anchors(self.buffer, 0):
                groups.append(self._take_group())
                continue
            if self.synchronized:
                if len(self.buffer) < GROUP_BYTES + INTER_PACKET_BYTES:
                    break
                if self._has_anchors(self.buffer, INTER_PACKET_BYTES):
                    self._consume(INTER_PACKET_BYTES)
                    groups.append(self._take_group())
                    continue
                self.synchronized = False
                self.resync_discarded = 0

            last_start = len(self.buffer) - GROUP_BYTES
            group_start = next(
                (
                    start
                    for start in range(last_start + 1)
                    if self._has_anchors(self.buffer, start)
                ),
                None,
            )
            if group_start is None:
                count = len(self.buffer) - (GROUP_BYTES - 1)
                self._discard(count)
                if self.ever_synchronized:
                    self.resync_discarded += count
                break

            self._discard(group_start)
            if self.ever_synchronized:
                self.resync_discarded += group_start
                self.stats.malformed_packets += (
                    self.resync_discarded + GROUP_BYTES // 2
                ) // GROUP_BYTES
            self.resync_discarded = 0
            self.synchronized = True
            self.ever_synchronized = True
            groups.append(self._take_group())

        return groups

    def finish(self) -> None:
        self.stats.trailing_bytes = len(self.buffer)


def iter_uart_byte_groups(
    stream: BinaryIO,
    *,
    stats: UartByteConversionStats | None = None,
    chunk_bytes: int = 64 * 1024,
) -> Iterator[tuple[int, bytes]]:
    """Yield ``(capture_byte_offset, group)`` pairs from a byte capture."""
    if chunk_bytes < GROUP_BYTES:
        raise ValueError(f"chunk_bytes must be at least {GROUP_BYTES}.")

    parser = UartByteGroupParser(stats)
    while chunk := stream.read(chunk_bytes):
        yield from parser.feed(chunk)
    parser.finish()


class UartSampleExtractor:
    """Extract samples and remove the configured ECG/pulse source offsets."""

    def __init__(
        self,
        *,
        sample_rate_hz: float = 400.0,
        ecg1_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
        ecg2_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
        puls_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
        stats: UartByteConversionStats | None = None,
    ):
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive.")
        offsets = (float(ecg1_offset), float(ecg2_offset), float(puls_offset))
        if not all(math.isfinite(value) for value in offsets):
            raise ValueError("UART signal offsets must be finite.")
        self.sample_rate_hz = float(sample_rate_hz)
        self.ecg1_offset, self.ecg2_offset, self.puls_offset = offsets
        self.stats = stats or UartByteConversionStats()
        self.pending_pulses: deque[int] = deque()
        self.current_pulse: int | None = None
        self.pulse_samples_remaining = 0
        self.pending_respirations: deque[int] = deque()
        self.current_respiration: int | None = None
        self.respiration_samples_remaining = 0

    @staticmethod
    def _ecg1(group: bytes, offset: int) -> int:
        return ((group[offset] & 0x0F) << 8) | group[offset + 1]

    @staticmethod
    def _ecg2(group: bytes, offset: int) -> int:
        return (
            ((group[offset + 2] & 0x0F) << 8)
            | ((group[offset + 3] & 0x0F) << 4)
            | (group[offset + 4] & 0x0F)
        )

    @staticmethod
    def _ext(group: bytes, sample_in_packet: int) -> int:
        """Return the 200 Hz trigger from frame 0 or 2 for this sample.

        On the wire the four trigger bits follow the second ``1011`` anchor
        LSB-first. UART byte decoding has already restored their significance,
        so the flipped four-bit number is the low nibble of data byte 5.
        """
        trigger_frame_offset = EXT_FRAME_OFFSETS[sample_in_packet // 2]
        return group[trigger_frame_offset + EXT_BYTE_OFFSET] & 0x0F

    def _select_pulse(self) -> bool:
        if self.pulse_samples_remaining > 0:
            return False
        if self.pending_pulses:
            self.current_pulse = self.pending_pulses.popleft()
            self.pulse_samples_remaining = 8
            return True
        return False

    def _select_respiration(self) -> None:
        if self.respiration_samples_remaining > 0:
            return
        if self.pending_respirations:
            self.current_respiration = self.pending_respirations.popleft()
            self.respiration_samples_remaining = 8

    def rows_for_group(
        self,
        group: bytes,
        *,
        capture_byte_offset: int,
    ) -> list[dict[str, object]]:
        if len(group) != GROUP_BYTES:
            raise ValueError(
                f"UART group must contain {GROUP_BYTES} bytes, got {len(group)}."
            )

        pr_sign = (group[12] >> 4) & 0x0F
        pulse_value = ((group[12] & 0x0F) << 8) | group[13]
        if pr_sign == 1:
            self.pending_pulses.append(pulse_value)
        elif pr_sign == 2:
            self.pending_respirations.append(pulse_value)

        packet_index = self.stats.packets
        self.stats.packets += 1
        rows: list[dict[str, object]] = []
        for sample_in_packet, offset in enumerate(SAMPLE_OFFSETS):
            puls_updated = self._select_pulse()
            self._select_respiration()
            raw_idx = self.stats.samples
            ecg1_raw = subtract_offset_float64(
                self._ecg1(group, offset),
                self.ecg1_offset,
                name="UART ECG1",
            )
            ecg2_raw = subtract_offset_float64(
                self._ecg2(group, offset),
                self.ecg2_offset,
                name="UART ECG2",
            )
            ecg3_raw = ecg2_raw - ecg1_raw
            puls_raw = (
                None
                if self.current_pulse is None
                else subtract_offset_float64(
                    self.current_pulse,
                    self.puls_offset,
                    name="UART pulse",
                )
            )
            rows.append(
                {
                    "raw_idx": raw_idx,
                    "time_ms": 1000.0 * raw_idx / self.sample_rate_hz,
                    "ecg1_raw": ecg1_raw,
                    "ecg2_raw": ecg2_raw,
                    "ecg3_raw": ecg3_raw,
                    "puls_raw": puls_raw,
                    "resp_raw": self.current_respiration,
                    "ext": self._ext(group, sample_in_packet),
                    "pr_sign": pr_sign,
                    "puls_updated": puls_updated,
                    "source_packet": packet_index,
                    "sample_in_packet": sample_in_packet,
                    "capture_byte_offset": capture_byte_offset,
                }
            )
            self.stats.samples += 1
            if self.pulse_samples_remaining > 0:
                self.pulse_samples_remaining -= 1
            if self.respiration_samples_remaining > 0:
                self.respiration_samples_remaining -= 1
        return rows


def default_uart_parquet_path(input_path: str | Path) -> Path:
    """Return ``input.bin -> input.parquet`` beside the capture."""
    return Path(input_path).with_suffix(".parquet")


def convert_uart_bytes_to_parquet(
    input_path: str | Path,
    output_path: str | Path | None = None,
    *,
    sample_rate_hz: float = 400.0,
    ecg1_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
    ecg2_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
    puls_offset: float = DEFAULT_UART_SIGNAL_OFFSET,
    batch_rows: int = 100_000,
    overwrite: bool = False,
) -> tuple[Path, UartByteConversionStats]:
    """Stream a raw UART byte capture into a sample-level parquet file."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "UART parquet conversion requires pyarrow in the project environment."
        ) from exc

    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"UART byte capture not found: {input_path}")
    output_path = (
        Path(output_path)
        if output_path is not None
        else default_uart_parquet_path(input_path)
    )
    if output_path.resolve() == input_path.resolve():
        raise ValueError("The parquet output must not overwrite the byte capture.")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use overwrite=True to replace it."
        )
    if batch_rows < SAMPLES_PER_GROUP:
        raise ValueError(f"batch_rows must be at least {SAMPLES_PER_GROUP}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output_path.with_name(f"{output_path.name}.partial")
    if partial_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Partial output already exists: {partial_path}. "
                "Use overwrite=True to replace it."
            )
        partial_path.unlink()

    schema = pa.schema(
        [
            ("raw_idx", pa.int64()),
            ("time_ms", pa.float64()),
            ("ecg1_raw", pa.float64()),
            ("ecg2_raw", pa.float64()),
            ("ecg3_raw", pa.float64()),
            ("puls_raw", pa.float64()),
            ("resp_raw", pa.uint16()),
            ("ext", pa.uint8()),
            ("pr_sign", pa.uint8()),
            ("puls_updated", pa.bool_()),
            ("source_packet", pa.int64()),
            ("sample_in_packet", pa.uint8()),
            ("capture_byte_offset", pa.int64()),
        ],
        metadata={
            b"source": str(input_path).encode(),
            b"sample_rate_hz": str(float(sample_rate_hz)).encode(),
            b"uart_ecg1_offset": str(float(ecg1_offset)).encode(),
            b"uart_ecg2_offset": str(float(ecg2_offset)).encode(),
            b"uart_puls_offset": str(float(puls_offset)).encode(),
            b"format": b"uart-pin10-samples-v1",
        },
    )

    stats = UartByteConversionStats()
    extractor = UartSampleExtractor(
        sample_rate_hz=sample_rate_hz,
        ecg1_offset=ecg1_offset,
        ecg2_offset=ecg2_offset,
        puls_offset=puls_offset,
        stats=stats,
    )
    rows: list[dict[str, object]] = []
    pending_pulse_block: list[dict[str, object]] = []
    output_sample_index = 0
    writer = pq.ParquetWriter(partial_path, schema, compression="snappy")
    try:
        with input_path.open("rb") as stream:
            for byte_offset, group in iter_uart_byte_groups(stream, stats=stats):
                group_rows = extractor.rows_for_group(
                    group,
                    capture_byte_offset=byte_offset,
                )
                for row in group_rows:
                    if row["puls_updated"]:
                        if pending_pulse_block:
                            stats.startup_samples_discarded += len(
                                pending_pulse_block
                            )
                        pending_pulse_block = [row]
                    elif pending_pulse_block:
                        pending_pulse_block.append(row)
                    else:
                        stats.startup_samples_discarded += 1
                        continue

                    if len(pending_pulse_block) == 8:
                        if any(
                            block_row["puls_raw"] is None
                            or block_row["resp_raw"] is None
                            for block_row in pending_pulse_block
                        ):
                            stats.startup_samples_discarded += len(
                                pending_pulse_block
                            )
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

                if len(rows) >= batch_rows:
                    writer.write_table(pa.Table.from_pylist(rows, schema=schema))
                    rows.clear()
        stats.trailing_samples_discarded = len(pending_pulse_block)
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=schema))
        elif output_sample_index == 0:
            writer.write_table(pa.Table.from_pylist([], schema=schema))
    except Exception:
        writer.close()
        partial_path.unlink(missing_ok=True)
        raise
    else:
        writer.close()

    stats.samples = output_sample_index
    partial_path.replace(output_path)
    return output_path, stats
