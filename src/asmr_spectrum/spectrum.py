"""Per-file spectral analysis: the ``analyze`` pipeline."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import numpy as np

from . import audio, plotting
from .config import DECADE_BANDS, LOW_BANDS, SAMPLE_RATE, WELCH_NFFT
from .core import band_energy_exact, loudness_stats, run_blocks
from .report import (band_flatness, band_table, cumulative, energy_distribution,
                     source_breakdown, spectral_tilt, timerresolved_shares,
                     write_bands_csv, write_json, write_loudness_csv,
                     write_psd_csv, write_trend_csv)


def describe_spectrum(psd: np.ndarray, freqs: np.ndarray, csum: np.ndarray) -> dict:
    """Everything derived from the long-term spectrum, in one dict."""
    psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
    return {
        "energy_distribution_psd_pct": energy_distribution(psd, freqs, csum),
        "spectral_tilt": spectral_tilt(psd_db, freqs),
        "band_flatness": band_flatness(psd, freqs),
    }


def analyze_file(path: str, out_prefix: str, label: Optional[str] = None,
                 sources: Sequence[tuple] = (), figure: bool = True,
                 csv: bool = True, verbose: bool = True) -> dict:
    """Run the full analysis on one audio file and write ``<out_prefix>.*``.

    Writes ``<prefix>.json`` plus optional ``_bands.csv``, ``_psd.csv``,
    ``_loudness.csv``, ``_trend.csv`` and ``.png``.
    """
    label = label or os.path.splitext(os.path.basename(path))[0]
    meta = audio.probe(path)
    if verbose:
        print(f"[{label}] decoding {meta['duration_s'] / 60:.1f} min "
              f"({meta['codec']} {meta['sample_rate']} Hz {meta['channels']} ch)…",
              flush=True)

    block = run_blocks(path)
    if block.blocks == 0:
        raise RuntimeError(f"no audio decoded from {path!r}")

    psd, freqs = block.psd, block.freqs
    psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
    csum = cumulative(psd, freqs)
    described = describe_spectrum(psd, freqs, csum)

    if verbose:
        print(f"[{label}] exact band-limited energy pass (1 s frames)…", flush=True)
    exact = band_energy_exact(path, cutoffs=(5, 10, 20, 40, 60, 100, 200, 500,
                                             1000, 2000, 5000))
    tr = timerresolved_shares(block)

    names = {20: "below_20hz", 40: "below_40hz", 200: "below_200hz", 1000: "below_1khz"}
    spread = {names[c]: {"exact": exact[c],
                         "psd_welch": described["energy_distribution_psd_pct"][names[c]],
                         "time_resolved": tr[names[c]]}
              for c in names}
    max_rel = 0.0
    for c, key in names.items():
        a, b = described["energy_distribution_psd_pct"][key], tr[key]
        if a:
            max_rel = max(max_rel, abs(a - b) / a * 100.0)

    payload = {
        "label": label,
        "file": os.path.basename(path),
        "source_format": meta,
        "duration_s": block.seconds,
        "sample_rate": SAMPLE_RATE,
        "welch_nfft": WELCH_NFFT,
        "welch_bin_hz": round(float(freqs[1] - freqs[0]), 3) if freqs.size > 1 else None,
        "welch_segments": block.welch_segments,
        "blocks": block.blocks,
        "energy_distribution_exact_pct": {f"below_{c}hz": exact[c] for c in exact},
        "estimator_spread_pct": spread,
        "estimator_max_rel_diff_pct": round(max_rel, 3),
        "bands_decade": band_table(psd, freqs, DECADE_BANDS),
        "bands_low": band_table(block.psd_1hz, block.freqs_1hz, LOW_BANDS),
        "loudness": loudness_stats(block),
        "per_source": source_breakdown(block, sources) if sources else [],
    }
    payload.update(described)

    write_json(out_prefix + ".json", payload)
    if csv:
        write_bands_csv(out_prefix + "_bands.csv",
                        payload["bands_decade"] + payload["bands_low"])
        write_psd_csv(out_prefix + "_psd.csv", freqs, psd_db, csum)
        write_loudness_csv(out_prefix + "_loudness.csv", block)
        write_trend_csv(out_prefix + "_trend.csv", block)
    if figure:
        try:
            plotting.analysis_figure(out_prefix + ".png", block, label,
                                     exact=exact, sources=sources)
        except Exception as exc:                      # plotting must never lose data
            print(f"[{label}] figure failed: {exc}", flush=True)

    if verbose:
        lv = payload["loudness"]
        print(f"[{label}] {block.seconds / 60:.1f} min | RMS {lv['rms_db_mono']} dBFS | "
              f"peak {max(lv['true_peak_db_LR']):.2f} dBFS | "
              f"<20 Hz {exact[20]}% | <40 Hz {exact[40]}% | <200 Hz {exact[200]}% | "
              f"centroid {payload['energy_distribution_psd_pct']['spectral_centroid_hz']} Hz",
              flush=True)
    return payload


def scan_file(path: str) -> dict:
    """Level + low-frequency summary for one file (one decode pass, no plots)."""
    block = run_blocks(path)
    if block.blocks == 0:
        raise RuntimeError(f"no audio decoded from {path!r}")
    lv = loudness_stats(block)
    exact = band_energy_exact(path, cutoffs=(5, 10, 20, 40, 100, 200))
    return {
        "file": os.path.basename(path),
        "duration_min": round(block.seconds / 60, 2),
        "rms_db": lv["rms_db_mono"],
        "rms_db_LR": lv["rms_db_LR"],
        "true_peak_db": max(lv["true_peak_db_LR"]),
        "crest_db": lv["crest_db"],
        "samples_over_unity": lv["clipped_samples_over_unity"],
        "silent_seconds": lv["silent_seconds_below_-60db"],
        "below_5hz_pct": exact[5],
        "below_10hz_pct": exact[10],
        "below_20hz_pct": exact[20],
        "below_40hz_pct": exact[40],
        "below_100hz_pct": exact[100],
        "below_200hz_pct": exact[200],
    }
