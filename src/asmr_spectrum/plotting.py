"""Matplotlib figures for the analysis results."""

from __future__ import annotations

import os
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")               # headless: always safe, never blocks
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

from .config import DECADE_BANDS, EXACT_KEYS, LOW_BANDS, TIME_BANDS  # noqa: E402
from .core import BlockResult        # noqa: E402
from .report import band_table, cumulative, share_at  # noqa: E402

_BAND_LABELS = [("sub", 20, 60), ("bass", 60, 150), ("low-mid", 150, 400),
                ("mid", 400, 1000), ("upper-mid", 1000, 2500),
                ("presence", 2500, 5000), ("brilliance", 5000, 8000),
                ("air", 8000, 12000), ("ultra", 12000, 16000),
                ("top", 16000, 20000)]


def analysis_figure(path: str, block: BlockResult, title: str,
                    exact: Optional[dict] = None,
                    sources: Sequence[tuple] = ()) -> str:
    """Write the six-panel overview figure and return its path."""
    psd, freqs = block.psd, block.freqs
    psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
    csum = cumulative(psd, freqs)
    n = block.blocks
    rms = block.rms_mono()
    rms_db = 20 * np.log10(np.maximum(rms, 1e-12))
    act = rms > 1e-3
    t = (np.arange(n) + 0.5) / 60.0
    tot = block.band.sum(axis=1)
    denom = np.maximum(tot, 1e-30)
    s20 = 100 * np.array([sum(block.band[i, j] for j, (lo, hi) in enumerate(TIME_BANDS)
                              if hi <= 20) for i in range(n)]) / denom
    s200 = 100 * np.array([sum(block.band[i, j] for j, (lo, hi) in enumerate(TIME_BANDS)
                               if hi <= 200) for i in range(n)]) / denom

    fig = plt.figure(figsize=(15, 18))
    gs = fig.add_gridspec(5, 2, hspace=0.42, wspace=0.22)
    top = psd_db[freqs >= 1].max() if (freqs >= 1).any() else psd_db.max()

    ax = fig.add_subplot(gs[0, :])
    ax.semilogx(freqs[1:], psd_db[1:], lw=0.8, color="tab:blue")
    ax.set_xlim(1, 24000)
    ax.set_ylim(top - 78, top + 4)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [dB/Hz]")
    ax.set_title(f"Average spectrum ({freqs[1] - freqs[0]:.2f} Hz bins, "
                 f"{n / 60:.1f} min) — {title}", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.axvspan(1, 20, color="red", alpha=0.10)
    ax.text(3.5, top - 3, "infrasonic 1–20 Hz", fontsize=7, color="darkred", va="top")
    for nm, lo, hi in _BAND_LABELS:
        ax.text(np.sqrt(lo * hi), ax.get_ylim()[1] - 1, nm, ha="center", va="top",
                fontsize=6, rotation=90, alpha=0.6)

    ax = fig.add_subplot(gs[1, 0])
    ax.semilogx(freqs[1:], csum[1:], lw=1.2, color="tab:green")
    ax.set_xlim(1, 24000)
    ax.set_ylim(0, 100)
    for f0 in (20, 60, 200, 1000):
        v = share_at(freqs, csum, f0)
        ax.axvline(f0, ls=":", lw=0.8, alpha=0.6)
        ax.annotate(f"{f0} Hz\n{v:.1f}%", xy=(f0, v),
                    xytext=(f0 * 1.3, max(v - 18, 2)), fontsize=7)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("cumulative energy [%]")
    ax.set_title("Cumulative energy distribution", fontsize=10)
    ax.grid(True, which="both", alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    rows = band_table(psd, freqs, DECADE_BANDS)
    val = [r["share_pct"] for r in rows]
    y = np.arange(len(rows))
    ax.barh(y, val, color="tab:purple")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['lo_hz']}–{r['hi_hz']} Hz" for r in rows], fontsize=7.5)
    ax.invert_yaxis()
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("share of total energy [%] (symlog)")
    ax.set_title("Energy share by band", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    for i, v in enumerate(val):
        ax.text(v * 1.2, i, f"{v:.2f}", va="center", fontsize=6.5)

    ax = fig.add_subplot(gs[2, 0])
    lf = band_table(block.psd_1hz, block.freqs_1hz, LOW_BANDS)
    y = np.arange(len(lf))
    ax.barh(y, [r["share_pct"] for r in lf], color="tab:red")
    ax.set_yticks(y)
    ax.set_yticklabels([r["band"] for r in lf], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("share of total energy [%]")
    ax.set_title("Low frequency 0–250 Hz (1 Hz bins)", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    m = freqs <= 300
    ax.semilogx(freqs[m], psd_db[m], lw=0.9, color="tab:blue")
    ax.set_xlim(1, 300)
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 300]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.set_xticks([], minor=True)
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("PSD [dB/Hz]")
    ax.set_title("Low-frequency detail 1–300 Hz", fontsize=10)
    ax.grid(True, which="major", alpha=0.3)

    ax = fig.add_subplot(gs[3, :])
    ax.plot(t, np.clip(20 * np.log10(np.maximum(block.loud[:, 0], 1e-12)), -80, 0),
            lw=0.3, color="tab:blue", label="L RMS")
    ax.plot(t, np.clip(20 * np.log10(np.maximum(block.loud[:, 1], 1e-12)), -80, 0),
            lw=0.3, color="tab:orange", alpha=0.7, label="R RMS")
    if act.any():
        for q, ls in ((10, ":"), (50, "--"), (90, ":")):
            v = float(np.percentile(rms_db[act], q))
            ax.axhline(v, color="k", ls=ls, lw=0.8, alpha=0.6)
            ax.text(t[-1] if n else 0, v, f" p{q}", fontsize=6, va="bottom", ha="right")
    ax.set_ylim(-80, 0)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("RMS [dBFS]")
    ax.set_title("Loudness over time (floor clipped at −80 dBFS; dotted = "
                 "p10/p50/p90 of active seconds)", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower left")
    for t0, _t1, _name in sources:
        ax.axvline(t0 / 60.0, color="k", lw=0.6, alpha=0.45)

    ax = fig.add_subplot(gs[4, 0])
    ax.hist(np.clip(rms_db[act], -80, 0) if act.any() else rms_db,
            bins=np.arange(-80, 0.5, 1), alpha=0.85, color="tab:blue")
    ax.set_yscale("log")
    ax.set_xlabel("per-second RMS [dBFS]")
    ax.set_ylabel("seconds (log)")
    ax.set_title(f"Loudness distribution ({int(act.sum())} of {n} s active)", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")

    ax = fig.add_subplot(gs[4, 1])
    ax.plot(t, np.clip(s20, 0, 100), lw=0.35, color="darkred", label="< 20 Hz")
    ax.plot(t, np.clip(s200, 0, 100), lw=0.35, color="purple", alpha=0.7, label="< 200 Hz")
    ax.set_ylim(0, 100)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("share of block energy [%]")
    ax.set_title("Low-frequency share over time", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    for t0, _t1, _name in sources:
        ax.axvline(t0 / 60.0, color="k", lw=0.6, alpha=0.45)

    if exact:
        sub = "  ".join(f"<{c} Hz {exact[c]:.2f}%" for c in sorted(exact))
        fig.suptitle(f"Frequency / loudness / energy — {title}\n{sub}",
                     fontsize=12, y=0.997)
    else:
        fig.suptitle(f"Frequency / loudness / energy — {title}", fontsize=13, y=0.995)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    fig.savefig(path, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return path
