"""Derived tables, JSON/CSV writers and source attribution."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import (DECADE_BANDS, EXACT_KEYS, LOW_BANDS, TIME_BANDS,
                     band_mask, hz_to_note, to_db)
from .core import BlockResult


def cumulative(psd: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    total = float(psd.sum())
    return np.cumsum(psd) / (total if total > 0 else 1.0) * 100.0


def share_at(freqs: np.ndarray, csum: np.ndarray, f: float) -> float:
    """Cumulative energy percentage at or below ``f``."""
    i = min(int(np.searchsorted(freqs, f)), freqs.size - 1)
    return round(float(csum[i]), 4)


def energy_distribution(psd: np.ndarray, freqs: np.ndarray, csum: np.ndarray) -> dict:
    total = float(psd.sum())
    out = {
        "below_1hz": share_at(freqs, csum, 1),
        "below_5hz": share_at(freqs, csum, 5),
        "below_10hz": share_at(freqs, csum, 10),
        "below_20hz": share_at(freqs, csum, 20),
        "below_40hz": share_at(freqs, csum, 40),
        "below_60hz": share_at(freqs, csum, 60),
        "below_100hz": share_at(freqs, csum, 100),
        "below_200hz": share_at(freqs, csum, 200),
        "below_500hz": share_at(freqs, csum, 500),
        "below_1khz": share_at(freqs, csum, 1000),
        "below_2khz": share_at(freqs, csum, 2000),
        "below_5khz": share_at(freqs, csum, 5000),
        "below_10khz": share_at(freqs, csum, 10000),
        "spectral_centroid_hz": round(float((psd * freqs).sum() / total), 1) if total > 0 else None,
    }
    for pct in (50, 80, 90, 95, 99):
        out[f"hz_for_{pct}pct"] = round(float(freqs[min(int(np.searchsorted(csum, pct)),
                                                      freqs.size - 1)]), 1)
    return out


def spectral_tilt(psd_db: np.ndarray, freqs: np.ndarray,
                  ranges=((1, 20, "1_20hz"), (20, 200, "20_200hz"),
                          (200, 2000, "200_2000hz"), (2000, 20000, "2k_20khz"),
                          (20, 2000, "20_2000hz"))) -> dict:
    """Least-squares slope of the log-PSD against log-frequency."""
    out = {}
    for lo, hi, name in ranges:
        m = (freqs >= lo) & (freqs <= hi)
        if m.sum() < 2:
            continue
        slope = float(np.polyfit(np.log10(freqs[m]), psd_db[m], 1)[0])
        out[name] = {"db_per_decade": round(slope, 2),
                     "db_per_octave": round(slope * float(np.log10(2)), 2)}
    return out


def band_flatness(psd: np.ndarray, freqs: np.ndarray,
                  bands=((20, 40), (40, 80), (80, 160), (160, 315), (315, 630),
                         (630, 1250), (1250, 2500), (2500, 5000), (5000, 10000))
                  ) -> dict:
    """Spectral flatness per band: ~1 noise-like, << 1 tonal (resonant)."""
    out = {}
    for lo, hi in bands:
        p = psd[band_mask(freqs, lo, hi)]
        if p.size == 0 or p.mean() <= 0:
            continue
        gm = float(np.exp(np.mean(np.log(np.maximum(p, 1e-30)))))
        out[f"{lo}_{hi}"] = round(gm / float(p.mean()), 4)
    return out


def band_table(psd: np.ndarray, freqs: np.ndarray, edges: Sequence[tuple]) -> List[dict]:
    total = float(psd.sum())
    rows = []
    for lo, hi in edges:
        m = band_mask(freqs, lo, hi)
        seg = psd[m]
        if seg.size == 0:
            continue
        peak = float(freqs[m][int(np.argmax(seg))])
        rows.append({
            "band": f"{lo}-{hi}", "lo_hz": lo, "hi_hz": hi,
            "share_pct": round(100.0 * float(seg.sum()) / total, 5) if total > 0 else 0.0,
            "mean_db": round(to_db(float(seg.mean())), 2),
            "peak_hz": round(peak, 2),
            "peak_note": hz_to_note(peak),
        })
    return rows


def timerresolved_shares(block: BlockResult) -> Dict[str, float]:
    """Cumulative shares from the per-second band sums (estimator #3)."""
    sub = block.band.sum(axis=0)
    tot = float(sub.sum())
    out = {}
    for thr, key in EXACT_KEYS.items():
        e = sum(sub[j] for j, (lo, hi) in enumerate(TIME_BANDS) if hi <= thr)
        out[key] = round(100.0 * float(e) / tot, 4) if tot > 0 else float("nan")
    return out


def source_breakdown(block: BlockResult, sources: Sequence[tuple]) -> List[dict]:
    """Per-source statistics using exact second-boundary slicing."""
    rows = []
    rms = block.rms_mono()
    n = block.blocks
    for t0, t1, name in sources:
        i0, i1 = max(0, int(t0)), min(n, int(t1))
        if i1 <= i0:
            continue
        sub = block.band[i0:i1].sum(axis=0)
        tot = float(sub.sum())
        r = rms[i0:i1]
        pk = block.loud[i0:i1, 2:4]
        row = {
            "name": name, "t0_s": t0, "t1_s": t1, "seconds": i1 - i0,
            "rms_db_mono": round(float(10 * np.log10(max((r ** 2).mean(), 1e-24))), 2),
            "true_peak_db": round(float(20 * np.log10(max(pk.max(), 1e-12))), 2),
            "active_seconds": int((r > 1e-3).sum()),
        }
        for thr, key in EXACT_KEYS.items():
            e = sum(sub[j] for j, (lo, hi) in enumerate(TIME_BANDS) if hi <= thr)
            row[key] = round(100.0 * float(e) / tot, 4) if tot > 0 else float("nan")
        for j, (lo, hi) in enumerate(TIME_BANDS):
            row[f"share_{lo}_{hi}_pct"] = (
                round(100.0 * float(sub[j]) / tot, 5) if tot > 0 else float("nan"))
        rows.append(row)
    return rows


def write_json(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)


def write_bands_csv(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["band", "lo_hz", "hi_hz", "share_pct", "mean_db", "peak_hz", "peak_note"])
        for r in rows:
            w.writerow([r["band"], r["lo_hz"], r["hi_hz"], r["share_pct"],
                        r["mean_db"], r["peak_hz"], r["peak_note"]])


def write_psd_csv(path: str, freqs: np.ndarray, psd_db: np.ndarray,
                  csum: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["freq_hz", "psd_db", "cumulative_pct"])
        for f, v, c in zip(freqs, psd_db, csum):
            w.writerow([f"{f:.2f}", f"{v:.4f}", f"{c:.5f}"])


def write_loudness_csv(path: str, block: BlockResult) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sec", "rms_L_db", "rms_R_db", "peak_L_db", "peak_R_db"])
        for i in range(block.blocks):
            w.writerow([i] + [f"{20 * np.log10(max(v, 1e-12)):.2f}" for v in block.loud[i]])


def write_trend_csv(path: str, block: BlockResult) -> None:
    tot = block.band.sum(axis=1)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["sec"] + [f"E_{lo}_{hi}" for lo, hi in TIME_BANDS]
                   + ["below20_pct", "below200_pct"])
        for i in range(block.blocks):
            denom = tot[i] if tot[i] > 0 else 1e-30
            b20 = 100 * sum(block.band[i, j] for j, (lo, hi) in enumerate(TIME_BANDS)
                            if hi <= 20) / denom
            b200 = 100 * sum(block.band[i, j] for j, (lo, hi) in enumerate(TIME_BANDS)
                             if hi <= 200) / denom
            w.writerow([i] + [f"{v:.6g}" for v in block.band[i]]
                       + [f"{b20:.3f}", f"{b200:.3f}"])
