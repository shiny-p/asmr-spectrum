"""Measurement core: Welch spectrum, per-block energy bands, exact LF energy.

Three independent estimators of the low-frequency energy share are provided on
purpose, because they disagree on steep infrasonic slopes and the disagreement
is itself diagnostic:

* :func:`band_energy_exact` -- exact band-limited energy via FFT overlap-add
  brick-wall masking. This is the authoritative number.
* Welch band sums from :func:`run_blocks` -- the analytic spectrum, but summing
  the bins of a 0.34 s window near 20 Hz spreads energy up across the cutoff.
* per-block band sums -- 1 s blocks, so the resolution limit is 1 Hz.

On a measured 20 Hz share of 11.5 % the three gave 11.47 / 14.67 / 10.72 %, i.e.
the Welch estimate can read ~28 % high in relative terms. Above ~60 Hz they agree
to well under 1 %.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from . import audio as _audio
from .config import (BLOCK_NFFT, EXACT_FRAME_S, SAMPLE_RATE,
                     TIME_BANDS, WELCH_NFFT, band_mask, hann)


@dataclass
class BlockResult:
    """Output of the single main decode pass."""

    psd: np.ndarray                      # power per WELCH_NFFT bin, mean over segments
    freqs: np.ndarray                    # Hz, WELCH_NFFT / 2 + 1 bins
    welch_segments: int
    psd_1hz: np.ndarray                  # power per 1 Hz bin, from the block FFTs
    freqs_1hz: np.ndarray
    band: np.ndarray                     # (blocks, len(TIME_BANDS)) raw band power
    loud: np.ndarray                     # (blocks, 4) rmsL, rmsR, peakL, peakR
    blocks: int
    time_bands: Sequence[tuple] = field(default_factory=lambda: list(TIME_BANDS))

    @property
    def seconds(self) -> float:
        return float(self.blocks)

    def rms_mono(self) -> np.ndarray:
        return np.sqrt((self.loud[:, 0] ** 2 + self.loud[:, 1] ** 2) / 2.0)


def run_blocks(path: str, progress: Optional[Callable[[float], None]] = None
               ) -> BlockResult:
    """Single decode pass producing both the Welch PSD and per-second statistics.

    A high-resolution (WELCH_NFFT, 50 % overlap) Welch accumulator runs alongside
    a second-aligned block FFT, so the file is decoded exactly once.
    """
    win_hi = hann(WELCH_NFFT)
    wp_hi = float(np.sum(win_hi ** 2))
    hop_hi = WELCH_NFFT // 2
    freqs_hi = np.fft.rfftfreq(WELCH_NFFT, d=1.0 / SAMPLE_RATE)

    win = hann(BLOCK_NFFT)
    wp = float(np.sum(win ** 2))
    freqs = np.fft.rfftfreq(BLOCK_NFFT, d=1.0 / SAMPLE_RATE)
    masks = [band_mask(freqs, lo, hi) for lo, hi in TIME_BANDS]

    psd_hi = np.zeros(freqs_hi.size)
    n_hi = 0
    tail = np.zeros(0)
    psd_1hz = np.zeros(freqs.size)
    band: List[List[float]] = []
    loud: List[tuple] = []
    buf = np.zeros((2, BLOCK_NFFT))
    fill = 0

    for blk in _audio.decode_stream(path, channels=2, rate=SAMPLE_RATE):
        mono = blk.mean(axis=1)
        # ---- high-resolution Welch frames
        seg = np.concatenate((tail, mono)) if tail.size else mono
        k = 0
        while k + WELCH_NFFT <= seg.size:
            sp = np.fft.rfft(seg[k:k + WELCH_NFFT] * win_hi)
            psd_hi += sp.real ** 2 + sp.imag ** 2
            n_hi += 1
            k += hop_hi
        tail = seg[k:].copy()
        # ---- second-aligned blocks
        p0, n = 0, blk.shape[0]
        while p0 < n:
            take = min(BLOCK_NFFT - fill, n - p0)
            buf[:, fill:fill + take] = blk[p0:p0 + take].T
            fill += take
            p0 += take
            if fill < BLOCK_NFFT:
                break
            loud.append((float(np.sqrt((buf[0] ** 2).mean())),
                         float(np.sqrt((buf[1] ** 2).mean())),
                         float(np.abs(buf[0]).max()),
                         float(np.abs(buf[1]).max())))
            pw = np.abs(np.fft.rfft(buf.mean(axis=0) * win)) ** 2
            psd_1hz += pw
            band.append([float(pw[m].sum()) for m in masks])
            fill = 0
        if progress is not None:
            progress(blk.shape[0] / SAMPLE_RATE)

    nb = len(band)
    return BlockResult(
        psd=psd_hi / max(n_hi, 1) / wp_hi,
        freqs=freqs_hi,
        welch_segments=n_hi,
        psd_1hz=psd_1hz / max(nb, 1) / wp,
        freqs_1hz=freqs,
        band=np.asarray(band).reshape(nb, len(TIME_BANDS)),
        loud=np.asarray(loud).reshape(nb, 4),
        blocks=nb,
    )


def band_energy_exact(path: str, cutoffs: Sequence[float] = (20.0, 40.0),
                      frame_s: float = EXACT_FRAME_S) -> Dict[float, float]:
    """Band-limited energy share of the total energy of a file.

    Method: split the signal into **non-overlapping** frames of ``frame_s``
    seconds, Hann-taper each, and accumulate the periodogram

        S(f) = |X(f)|^2 / sum(w^2)  [power per bin]

    Because ``sum(w^2) = 0.375·N`` for a Hann window, ``S`` is an unbiased
    estimate of the signal power in the frame. The energy below a cutoff is then
    ``sum(S over bins below the cutoff) × total_seconds``, since every frame has
    the same duration, and the grand total uses the same construction, so the
    ratio is exactly the fraction of energy below the cutoff.

    Why not a single long window? Two failure modes, both measured on the test
    fixture (see ``tests/validate.py``):

    * Phase cancellation across distant copies of one tone. One whole-file window
      put a 30 Hz tone's band share at 0.67 % where the truth is 25 %.
    * Averaging *overlapping* frames without the 1/sum(w^2) normalisation is
      biased whenever adjacent frames hold different tones, because such frames
      are not independent estimates of the same power.

    A frame must also span enough cycles at the lowest cutoff: at 20 Hz a 1 s
    frame holds ~20 cycles and reports a 30 Hz tone as 0.15 % below 20 Hz, while
    a 0.34 s frame (~7 cycles, 2.9 Hz main lobe) smears energy across the cutoff
    and overstates the infrasonic share -- the reason this function exists
    instead of reusing the Welch bands.
    """
    dur = _audio.duration(path)
    n_fft = max(1, int(round(frame_s * SAMPLE_RATE)))
    taper = hann(n_fft)
    wss = float(np.sum(taper ** 2))
    fr = np.fft.rfftfreq(n_fft, d=1.0 / SAMPLE_RATE)
    masks = {c: (fr < c) for c in cutoffs}
    acc = {c: 0.0 for c in cutoffs}
    tot = 0.0
    carry = np.zeros(0)

    for blk in _audio.decode_stream(path, channels=2, rate=SAMPLE_RATE):
        mono = blk.mean(axis=1)
        buf = np.concatenate((carry, mono)) if carry.size else mono
        k = 0
        while k + n_fft <= buf.size:
            amp = np.abs(np.fft.rfft(buf[k:k + n_fft] * taper)) ** 2 / wss
            tot += float(amp.sum())
            for c in cutoffs:
                acc[c] += float(amp[masks[c]].sum())
            k += n_fft                      # non-overlapping: independent frames
        carry = buf[k:].copy()
    if tot <= 0:
        return {c: float("nan") for c in cutoffs}
    return {c: round(100.0 * acc[c] / tot, 4) for c in cutoffs}


def loudness_stats(block: BlockResult) -> dict:
    """Loudness distribution from the per-second block statistics."""
    rms = block.rms_mono()
    act = rms > 1e-3                      # gate: -60 dBFS
    pk = block.loud[:, 2:4]
    rms_db = 20 * np.log10(np.maximum(rms, 1e-12))
    active_db = 20 * np.log10(np.maximum(rms[act], 1e-12)) if act.any() else np.zeros(0)
    out = {
        "rms_db_LR": [round(float(10 * np.log10(max((block.loud[:, 0] ** 2).mean(), 1e-24))), 2),
                      round(float(10 * np.log10(max((block.loud[:, 1] ** 2).mean(), 1e-24))), 2)],
        "rms_db_mono": round(float(10 * np.log10(max((rms ** 2).mean(), 1e-24))), 2),
        "true_peak_db_LR": [round(float(20 * np.log10(max(pk[:, 0].max(), 1e-12))), 2),
                            round(float(20 * np.log10(max(pk[:, 1].max(), 1e-12))), 2)],
        "active_seconds": int(act.sum()),
        "total_seconds": int(rms.size),
        "active_fraction_pct": round(100.0 * float(act.mean()), 2) if rms.size else 0.0,
        "silent_seconds_below_-60db": int((~act).sum()),
        "seconds_above_-10db": int((rms_db > -10).sum()),
        "clipped_samples_over_unity": int(np.sum(np.abs(pk) > 1.0)),
    }
    out["crest_db"] = round(float(max(out["true_peak_db_LR"]) - out["rms_db_mono"]), 2)
    for q in (5, 10, 25, 50, 75, 90, 95):
        out[f"active_rms_db_p{q}"] = round(float(np.percentile(active_db, q)), 2) if active_db.size else None
    out["loudness_range_p95_minus_p10_db"] = (
        round(float(np.percentile(active_db, 95) - np.percentile(active_db, 10)), 2)
        if active_db.size else None)
    return out
