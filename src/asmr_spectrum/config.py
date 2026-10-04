"""Shared constants, band definitions and small helpers."""

from __future__ import annotations

import numpy as np

SAMPLE_RATE = 48_000
"""Everything is decoded to this rate so mixed input rates can be compared."""

# High-resolution Welch pass (the spectrum).
WELCH_NFFT = 16_384
WELCH_BIN_HZ = SAMPLE_RATE / WELCH_NFFT  # 2.93 Hz

# Time-resolved pass: exactly one second, aligned to second boundaries, so a
# block can be attributed to a source file without any boundary ambiguity.
BLOCK_NFFT = SAMPLE_RATE
BLOCK_BIN_HZ = 1.0

# Exact band-limited energy pass: non-overlapping frames of this length. See
# core.band_energy_exact for why the frame length and the non-overlap both matter.
EXACT_FRAME_S = 1.0

# Time-resolved bands. The infrasonic bands are why the block size above matters:
# a 0.34 s window cannot resolve a slope this steep near 20 Hz.
TIME_BANDS = [
    (0, 5), (5, 10), (10, 20), (20, 40), (40, 60), (60, 80), (80, 120),
    (120, 200), (200, 500), (500, 1000), (1000, 2000), (2000, 5000),
    (5000, 10_000), (10_000, 16_000), (16_000, 20_000), (20_000, 24_000),
]

# Coarser decade-ish bands for reporting.
DECADE_BANDS = [
    (20, 40), (40, 60), (60, 80), (80, 120), (120, 200), (200, 500),
    (500, 1000), (1000, 2000), (2000, 5000), (5000, 10_000),
    (10_000, 16_000), (16_000, 20_000),
]

# Fine low-frequency bands for the distribution table.
LOW_BANDS = [
    (0, 5), (5, 10), (10, 20), (20, 25), (25, 30), (30, 35), (35, 40),
    (40, 45), (45, 50), (50, 60), (60, 70), (70, 80), (80, 100),
    (100, 120), (120, 150), (150, 200), (200, 250),
]

# Cutoffs used for the exact cumulative distribution.
EXACT_CUTOFFS = (5, 10, 20, 40, 60, 100, 200, 500, 1000, 2000, 5000)

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Band edges kept whole in the JSON report, mapped to their cutoff.
EXACT_KEYS = {20: "below_20hz", 40: "below_40hz", 200: "below_200hz",
              1000: "below_1khz"}


def hann(n: int) -> np.ndarray:
    """Periodic-ish Hann window (n+1 samples truncated, as NumPy's hanning)."""
    return np.hanning(n + 1)[:n]


def hz_to_note(freq: float) -> str:
    """Nearest equal-tempered note name, A4 = 440 Hz."""
    if freq <= 0:
        return "-"
    n = int(round(12 * np.log2(freq / 440.0) + 69))
    return f"{NOTE_NAMES[n % 12]}{n // 12 - 1}"


def to_db(power: float, floor: float = 1e-30) -> float:
    return 10.0 * np.log10(max(float(power), floor))


def amp_db(amplitude: float, floor: float = 1e-12) -> float:
    return 20.0 * np.log10(max(float(amplitude), floor))


def band_mask(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs < hi)


def parse_source_spec(spec: str):
    """Parse ``t0,t1,name`` (name may contain commas)."""
    parts = spec.split(",", 2)
    if len(parts) != 3:
        raise ValueError(f"--source needs t0,t1,name; got {spec!r}")
    return float(parts[0]), float(parts[1]), parts[2]
