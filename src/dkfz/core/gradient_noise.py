# Copyright (c) 2026 Furkan Selim Cetin <mail@furkancetin.dev>
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
# Developed at the German Cancer Research Center (DKFZ).

"""Synthetic gradient artifacts copied from notebooks/gradient_noise.ipynb.

The low-level noise functions, default peaks, and optional band-limited white
noise floor reproduce the notebook. ``apply_synthetic_gradient`` adds
independent noise to ECG1/ECG2 with seed 42 in ecg1_syn/ecg2_syn, then derives
ecg3_syn as ecg2_syn minus ecg1_syn. ECG3 receives the difference of the first
two noise realizations, with no separate noise draw. Optional main-frequency
configurations move peaks relative to the main frequency and simulate sampled
aliasing while preserving the configured amplitudes and noise floor. Raw signals
and annotations are retained; the input is copied. The default sample rate is the notebook's
400 Hz; prediction uses its config.
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd

ECG_COLUMNS = ("ecg1_raw", "ecg2_raw")
SYNTHETIC_ECG_COLUMNS = ("ecg1_syn", "ecg2_syn", "ecg3_syn")
SAMPLE_RATE_HZ = 400.0
GRADIENT_SEED = 42
# Exact main frequencies define the configurations.
GRADIENT_MAIN_FREQUENCIES_HZ = (190, 90, 80, 40, 20, 10)


def gradient_config_id(main_frequency_hz: float) -> str:
    if main_frequency_hz not in GRADIENT_MAIN_FREQUENCIES_HZ:
        raise ValueError(
            f"main_frequency_hz must be one of {GRADIENT_MAIN_FREQUENCIES_HZ}"
        )
    return f"main_{main_frequency_hz:g}hz"


# Edit this list to choose the peaks. width_hz is the cutoff half-width:
# for example, 84 Hz with width_hz=2 is nonzero only from 82 to 86 Hz.
# RMS amplitudes are in the ECG's arbitrary/ADC units and are set independently
# for ECG1 and ECG2. They start equal to preserve the previous behavior.
# coherent_fraction near 1 keeps the artifact amplitude stable over time.
GRADIENT_PEAKS = [
    # {
    #     "frequency_hz": 25.0,
    #     "width_hz": 0.1,
    #     "rms_amplitude_ecg1": 5.0,
    #     "rms_amplitude_ecg2": 3.0,
    #     "coherent_fraction": 0.98,
    # },
    {
        "frequency_hz": 42.0,
        "width_hz": 0.1,
        "rms_amplitude_ecg1": 10.0,
        "rms_amplitude_ecg2": 3.0,
        "coherent_fraction": 0.98,
    },
    {
        "frequency_hz": 63.0,
        "width_hz": 0.1,
        "rms_amplitude_ecg1": 3.0,
        "rms_amplitude_ecg2": 3.0,
        "coherent_fraction": 0.98,
    },
    {
        "frequency_hz": 82.0,
        "width_hz": 0.05,
        "rms_amplitude_ecg1": 3.0,
        "rms_amplitude_ecg2": 1.0,
        "coherent_fraction": 0.98,
    },
    {
        "frequency_hz": 86.0,
        "width_hz": 0.05,
        "rms_amplitude_ecg1": 2.5,
        "rms_amplitude_ecg2": 1.0,
        "coherent_fraction": 0.98,
    },
    {
        "frequency_hz": 84.0,
        "width_hz": 4.0,
        "rms_amplitude_ecg1": 10.0,
        "rms_amplitude_ecg2": 10.0,
        "coherent_fraction": 0.98,
    },
    {
        "frequency_hz": 84.0,
        "width_hz": 0.025,
        "rms_amplitude_ecg1": 100.0,
        "rms_amplitude_ecg2": 30.0,
        "coherent_fraction": 0.98,
    },
    {
        "frequency_hz": 168.0,
        "width_hz": 0.1,
        "rms_amplitude_ecg1": 3.0,
        "rms_amplitude_ecg2": 2.0,
        "coherent_fraction": 0.98,
    },
    {
        "frequency_hz": 194.0,
        "width_hz": 0.1,
        "rms_amplitude_ecg1": 2.0,
        "rms_amplitude_ecg2": 2.0,
        "coherent_fraction": 0.98,
    },
]

# Optional band-limited white-noise component. RMS amplitudes are measured in
# the time domain after band limiting. Zero preserves the peak-only behavior.
GRADIENT_NOISE_FLOOR = {
    "min_frequency_hz": 0.0,
    "max_frequency_hz": 200.0,
    "rms_amplitude_ecg1": 35.0,
    "rms_amplitude_ecg2": 15.0,
}


def gradient_peaks_for_frequency(main_frequency_hz: float, fs: float = SAMPLE_RATE_HZ):
    """Move peaks relative to the main frequency and alias them into the sampled band.

    Each returned peak retains its physical center in source_frequency_hz.
    Reflection preserves the width and RMS of a symmetric peak wholly inside
    one Nyquist zone, as is the case for all six configurations at 400 Hz.
    Peaks straddling a folding boundary require a different spectral treatment
    and are rejected rather than silently clipped. The template is never edited.
    """
    gradient_config_id(main_frequency_hz)
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError("fs must be finite and positive")
    centers = {
        42.0: main_frequency_hz / 2,
        82.0: main_frequency_hz - 2,
        84.0: main_frequency_hz,
        86.0: main_frequency_hz + 2,
        168.0: main_frequency_hz * 2,
    }
    peaks = []
    for template in GRADIENT_PEAKS:
        source_hz = float(
            centers.get(template["frequency_hz"], template["frequency_hz"])
        )
        # Frequencies in [0, fs/2] are unchanged; higher zones fold periodically.
        sampled_hz = abs((source_hz + fs / 2) % fs - fs / 2)
        if not template["width_hz"] <= sampled_hz <= fs / 2 - template["width_hz"]:
            raise ValueError(
                "A configured peak crosses a DC or Nyquist folding boundary"
            )
        peaks.append(
            {
                **template,
                "source_frequency_hz": source_hz,
                "frequency_hz": sampled_hz,
            }
        )
    return peaks


def generate_gaussian_frequency_noise(n_samples, fs, peaks, channel, rng=None):
    """Generate real noise with independently scaled Gaussian PSD peaks.

    ``channel`` must be ``"ecg1"`` or ``"ecg2"`` and selects the corresponding
    ``rms_amplitude_ecg1`` or ``rms_amplitude_ecg2`` value from every peak.
    ``width_hz`` is the cutoff half-width: a peak is zero outside center +/-
    width. The Gaussian reaches three standard deviations at that cutoff.
    ``coherent_fraction`` optionally sets the power fraction in a stable carrier
    (default 0.98); the rest forms a Gaussian skirt. RMS is the total contribution
    in the same arbitrary/ADC units as the ECG.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if fs <= 0:
        raise ValueError("fs must be positive")
    if channel not in {"ecg1", "ecg2"}:
        raise ValueError("channel must be 'ecg1' or 'ecg2'")

    rng = np.random.default_rng(rng)
    frequencies = np.fft.rfftfreq(n_samples, d=1 / fs)
    nyquist_hz = fs / 2
    noise = np.zeros(n_samples, dtype=float)
    amplitude_keys = ("rms_amplitude_ecg1", "rms_amplitude_ecg2")
    amplitude_key = f"rms_amplitude_{channel}"

    for peak in peaks:
        if "rms_amplitude" in peak:
            raise ValueError(
                "Scalar 'rms_amplitude' is not supported; provide "
                "'rms_amplitude_ecg1' and 'rms_amplitude_ecg2'"
            )
        missing_amplitudes = set(amplitude_keys).difference(peak)
        if missing_amplitudes:
            raise KeyError(
                f"Peak is missing channel amplitudes: {sorted(missing_amplitudes)}"
            )
        center_hz = float(peak["frequency_hz"])
        width_hz = float(peak["width_hz"])
        rms_amplitude = float(peak[amplitude_key])
        coherent_fraction = float(peak.get("coherent_fraction", 0.98))

        if not 0 <= center_hz <= nyquist_hz:
            raise ValueError(
                f"Peak frequency {center_hz:g} Hz is outside [0, {nyquist_hz:g}] Hz"
            )
        if width_hz <= 0:
            raise ValueError("Every width_hz must be positive")
        if rms_amplitude < 0:
            raise ValueError("Every rms_amplitude must be non-negative")
        if not 0 <= coherent_fraction <= 1:
            raise ValueError("Every coherent_fraction must be between 0 and 1")
        if rms_amplitude == 0:
            continue

        # width_hz defines a hard cutoff. Within it, the Gaussian power
        # envelope falls to exp(-4.5), about 1.1% of its center value.
        distance_hz = np.abs(frequencies - center_hz)
        sigma_hz = width_hz / 3
        power_envelope = np.exp(-0.5 * (distance_hz / sigma_hz) ** 2)
        power_envelope[distance_hz > width_hz] = 0.0
        active_bins = np.flatnonzero(power_envelope > 0)
        if active_bins.size == 0:
            raise ValueError(
                "A peak is too narrow for this recording's FFT resolution "
                f"({fs / n_samples:g} Hz)"
            )

        # A carrier at the closest FFT bin supplies the repeating,
        # phase-coherent part of the gradient artifact.
        carrier_bin = active_bins[np.argmax(power_envelope[active_bins])]
        carrier_spectrum = np.zeros(frequencies.size, dtype=np.complex128)
        is_self_conjugate = carrier_bin == 0 or (
            n_samples % 2 == 0 and carrier_bin == len(frequencies) - 1
        )
        if is_self_conjugate:
            carrier_spectrum[carrier_bin] = rng.choice((-1.0, 1.0))
        else:
            carrier_spectrum[carrier_bin] = np.exp(1j * rng.uniform(0, 2 * np.pi))
        carrier = np.fft.irfft(carrier_spectrum, n=n_samples)
        carrier /= np.sqrt(np.mean(carrier**2))

        # The remaining power is a Gaussian skirt around the carrier.
        skirt_spectrum = (
            rng.normal(size=frequencies.size) + 1j * rng.normal(size=frequencies.size)
        ) * np.sqrt(power_envelope / 2)
        # DC and Nyquist have no negative-frequency partner and must be real.
        skirt_spectrum[0] = rng.normal() * np.sqrt(power_envelope[0])
        if n_samples % 2 == 0:
            skirt_spectrum[-1] = rng.normal() * np.sqrt(power_envelope[-1])
        skirt_spectrum[carrier_bin] = 0
        skirt = np.fft.irfft(skirt_spectrum, n=n_samples)
        skirt_rms = np.sqrt(np.mean(skirt**2))

        if skirt_rms == 0 or coherent_fraction == 1:
            component = carrier
        else:
            skirt /= skirt_rms
            component = (
                np.sqrt(coherent_fraction) * carrier
                + np.sqrt(1 - coherent_fraction) * skirt
            )

        # Carrier and skirt occupy different FFT bins, so their powers add.
        component *= rms_amplitude / np.sqrt(np.mean(component**2))
        noise += component

    return noise


def generate_band_limited_white_noise(
    n_samples,
    fs,
    min_frequency_hz,
    max_frequency_hz,
    rms_amplitude,
    rng=None,
):
    """Generate white Gaussian noise confined to one frequency interval.

    ``rms_amplitude`` is the time-domain RMS of the returned band-limited
    signal. An amplitude of zero returns zeros without advancing the RNG.
    """
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if fs <= 0:
        raise ValueError("fs must be positive")

    min_frequency_hz = float(min_frequency_hz)
    max_frequency_hz = float(max_frequency_hz)
    rms_amplitude = float(rms_amplitude)
    nyquist_hz = fs / 2
    if not 0 <= min_frequency_hz < max_frequency_hz <= nyquist_hz:
        raise ValueError("The white-noise frequency range must lie within Nyquist")
    if rms_amplitude < 0:
        raise ValueError("White-noise RMS amplitude must be non-negative")

    frequencies = np.fft.rfftfreq(n_samples, d=1 / fs)
    in_band = (frequencies >= min_frequency_hz) & (frequencies <= max_frequency_hz)
    if not np.any(in_band):
        raise ValueError(
            "The white-noise range contains no bins at this FFT resolution"
        )
    if rms_amplitude == 0:
        return np.zeros(n_samples, dtype=float)

    rng = np.random.default_rng(rng)
    spectrum = np.fft.rfft(rng.normal(size=n_samples))
    spectrum[~in_band] = 0
    noise = np.fft.irfft(spectrum, n=n_samples)
    noise *= rms_amplitude / np.sqrt(np.mean(noise**2))
    return noise


def add_gaussian_frequency_noise(
    ecg_frame,
    columns,
    fs,
    peaks,
    seed=None,
    noise_floor=None,
):
    """Add independent noise to each selected column and return (frame, noise).

    Every peak must contain ``rms_amplitude_ecg1`` and ``rms_amplitude_ecg2``.
    Amplitudes are selected by the order of ``columns``. Frequencies, widths,
    coherence, and the RNG sequence remain shared between channel configurations.
    When supplied, ``noise_floor`` adds band-limited white noise using
    ``min_frequency_hz``, ``max_frequency_hz``, and channel-specific RMS values.

    This notebook helper leaves unselected columns unchanged. Use
    ``apply_synthetic_gradient`` to also recalculate the dependent ECG3 channel.
    """
    missing = set(columns).difference(ecg_frame.columns)
    if missing:
        raise KeyError(f"Missing ECG columns: {sorted(missing)}")

    rng = np.random.default_rng(seed)
    noisy_frame = ecg_frame.copy()
    added_noise = pd.DataFrame(index=ecg_frame.index)

    channel_names = ("ecg1", "ecg2")
    for channel_index, column in enumerate(columns):
        if channel_index >= len(channel_names):
            raise ValueError("Channel-specific peaks support exactly ECG1 and ECG2")
        channel_noise = generate_gaussian_frequency_noise(
            n_samples=len(ecg_frame),
            fs=fs,
            peaks=peaks,
            channel=channel_names[channel_index],
            rng=rng,
        )
        if noise_floor is not None:
            amplitude_key = f"rms_amplitude_{channel_names[channel_index]}"
            required_floor_keys = {
                "min_frequency_hz",
                "max_frequency_hz",
                "rms_amplitude_ecg1",
                "rms_amplitude_ecg2",
            }
            missing_floor_keys = required_floor_keys.difference(noise_floor)
            if missing_floor_keys:
                raise KeyError(
                    "Noise-floor configuration is missing: "
                    f"{sorted(missing_floor_keys)}"
                )
            channel_noise += generate_band_limited_white_noise(
                n_samples=len(ecg_frame),
                fs=fs,
                min_frequency_hz=noise_floor["min_frequency_hz"],
                max_frequency_hz=noise_floor["max_frequency_hz"],
                rms_amplitude=noise_floor[amplitude_key],
                rng=rng,
            )
        added_noise[column] = channel_noise
        noisy_frame[column] = ecg_frame[column].to_numpy(dtype=float) + channel_noise

    return noisy_frame, added_noise


def apply_synthetic_gradient(
    frame: pd.DataFrame,
    *,
    columns=ECG_COLUMNS,
    fs: float = SAMPLE_RATE_HZ,
    main_frequency_hz: float | None = None,
) -> pd.DataFrame:
    """Return original columns plus ecg1_syn, ecg2_syn, and derived ecg3_syn.

    ``columns`` supplies ECG1 and ECG2 names in that order. Noise uses the
    GRADIENT_PEAKS and seed 42, restarted for each call. main_frequency_hz
    optionally shifts peaks relative to the main frequency and simulates aliasing.
    The synthetic columns hold signal plus noise, with
    ecg3_syn = ecg2_syn - ecg1_syn.
    All three are written even for two-input models; existing synthetic
    columns are refreshed. Raw ECGs, pulse, timestamps, and annotations remain
    unchanged. Source names must differ from the synthetic output names.
    This function does not save files; use ``save_synthetic_parquet`` for that.
    """
    if len(columns) != 2 or len(set(columns)) != 2:
        raise ValueError("Provide two distinct ECG input columns")
    if set(columns).intersection(SYNTHETIC_ECG_COLUMNS):
        raise ValueError("ECG sources must differ from the synthetic output columns")
    peaks = (
        GRADIENT_PEAKS
        if main_frequency_hz is None
        else gradient_peaks_for_frequency(main_frequency_hz, fs)
    )
    noisy_frame, _ = add_gaussian_frequency_noise(
        frame,
        columns=columns,
        fs=fs,
        peaks=peaks,
        seed=GRADIENT_SEED,
        noise_floor=GRADIENT_NOISE_FLOOR,
    )
    result = frame.copy()
    result.attrs.pop("synthetic_gradient", None)
    result["ecg1_syn"] = noisy_frame[columns[0]]
    result["ecg2_syn"] = noisy_frame[columns[1]]
    result["ecg3_syn"] = result["ecg2_syn"] - result["ecg1_syn"]
    if main_frequency_hz is not None:
        result.attrs["synthetic_gradient"] = {
            "config_id": gradient_config_id(main_frequency_hz),
            "main_frequency_hz": float(main_frequency_hz),
            "sample_rate_hz": float(fs),
            "seed": GRADIENT_SEED,
            "peaks": peaks,
            "noise_floor": dict(GRADIENT_NOISE_FLOOR),
        }
    return result


def synthetic_output_path(
    input_path: str | Path,
    output_root: str | Path = "data/synthetic",
) -> Path:
    """Place pNN-prefixed inputs under output_root/pNN with a synthetic suffix."""
    input_path = Path(input_path)
    patient = re.match(r"^(p[0-9]+)(?=[_.-]|$)", input_path.stem, re.IGNORECASE)
    if patient is None:
        raise ValueError(
            f"Cannot determine patient from {input_path.name!r}; "
            "expected a filename starting with p01, p02, etc."
        )
    destination = (
        Path(output_root)
        / patient.group(1).lower()
        / f"{input_path.stem}_synthetic.parquet"
    )
    if destination.resolve() == input_path.resolve():
        raise ValueError("Synthetic output must not overwrite the input file")
    return destination


def save_synthetic_parquet(
    frame: pd.DataFrame,
    input_path: str | Path,
    output_root: str | Path = "data/synthetic",
) -> Path:
    """Save a modified frame under output_root/[config/]pNN/<stem>_synthetic.parquet.

    Call ``apply_synthetic_gradient`` first to add ecg1_syn/ecg2_syn/ecg3_syn
    alongside the original raw signals. Both versions are saved. A configured
    main frequency adds its config_id subdirectory; metadata survives in Parquet.
    The patient comes from the input filename prefix. The index is preserved,
    directories are created as needed, and an existing synthetic output at the
    same path is replaced. The input path itself is never overwritten.
    """
    config_id = frame.attrs.get("synthetic_gradient", {}).get("config_id")
    if config_id is not None:
        output_root = Path(output_root) / config_id
    destination = synthetic_output_path(input_path, output_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination)
    return destination
