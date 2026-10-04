#!/usr/bin/env python3
"""Self-test: build audio with known content and check the measurements.

Usage:  python tests/validate.py            (any interpreter with numpy + ffmpeg)

Everything runs against ONE generated fixture so the checks all describe the
same signal. Where the answer is known in closed form it is asserted exactly;
where the Hann taper removes part of each tone, the physical bound is asserted
instead (see the note on taper loss below).

Fixture (12 s, one event per second, 0.5 amplitude sine, equal energy each):
    0-1 s 30 Hz | 1-2 s 100 Hz | 2-3 s 1 kHz | 3-4 s 6 kHz | 4-5 s 15 kHz
    5-6 s silence | 6-7 s 300 Hz | 7-8 s 30 Hz | 8-12 s silence

Taper loss: a tone occupying a whole 1 s frame loses
``sum(w^2)/n = 0.375`` of its energy to the Hann window, so a band containing N
of the 8 sounding seconds contributes at most ``0.375*N / (0.375*8) = N/8`` of
the frame-averaged total; that is what the expected shares below express.
Distinct frequencies per second matter -- repeating one tone in two separate
segments lets a long window see them as a single phase-coherent signal, which is
precisely the artefact the exact-energy estimator must resist.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from asmr_spectrum.config import SAMPLE_RATE, TIME_BANDS  # noqa: E402
from asmr_spectrum.core import band_energy_exact, loudness_stats, run_blocks  # noqa: E402
from asmr_spectrum.report import cumulative  # noqa: E402

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def make_fixture(path: str) -> None:
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    plan = [30, 100, 1000, 6000, 15000, None, 300, 30, None, None, None, None]
    segs = [(0.5 * np.sin(2 * np.pi * f * t)) if f else np.zeros(SAMPLE_RATE)
            for f in plan]
    x = np.concatenate(segs).astype("<f4")
    stereo = np.stack([x, x], axis=1).ravel()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "f32le",
                    "-ar", str(SAMPLE_RATE), "-ac", "2", "-i", "-",
                    "-c:a", "flac", path],
                   input=stereo.tobytes(), check=True)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="asmr_spectrum_test_")
    fixture = os.path.join(tmp, "tones.flac")
    print(f"fixture: {fixture}")
    make_fixture(fixture)

    block = run_blocks(fixture)
    print(f"\n1. time-resolved bands  ({block.blocks} blocks, expect 12)")
    check("block count == 12", block.blocks == 12, f"got {block.blocks}")
    expect_band = {0: (20, 40), 1: (80, 120), 2: (1000, 2000), 3: (5000, 10000),
                   4: (10000, 16000), 6: (200, 500), 7: (20, 40)}
    for sec, want in expect_band.items():
        top = TIME_BANDS[int(np.argmax(block.band[sec]))]
        check(f"sec {sec:2d} dominant band == {want}", top == want, f"got {top}")

    print("\n2. loudness")
    lv = loudness_stats(block)
    check("active seconds == 7", lv["active_seconds"] == 7, str(lv["active_seconds"]))
    check("silent seconds == 5", lv["silent_seconds_below_-60db"] == 5,
          str(lv["silent_seconds_below_-60db"]))
    rms = block.rms_mono()
    tone_db = 20 * np.log10(rms[0])
    check("tone RMS == -9.03 dBFS", abs(tone_db + 9.03) < 0.01, f"got {tone_db:.3f} dB")
    check("silence RMS < -100 dBFS", 20 * np.log10(max(rms[5], 1e-12)) < -100)
    # File-level crest: peak 0.5 over a 12 s time base of which 7 s sound.
    # 20*log10(0.5 / sqrt(0.375*7/12/2)) = 5.37 dB (a lone tone would be 3.01 dB).
    check("crest factor ≈ 5.37 dB (12 s base, 7 s sounding)",
          abs(lv["crest_db"] - 5.37) < 0.05, f"got {lv['crest_db']} dB")
    check("no samples over unity", lv["clipped_samples_over_unity"] == 0)

    print("\n3. exact band-limited energy")
    ex = band_energy_exact(fixture, cutoffs=(20, 40, 200, 2000, 8000, 16000))
    # Every sounding second carries identical energy, so with 7 sounding seconds a
    # cutoff that includes n of them must report exactly n/7. The silent second
    # contributes no energy at all, hence 7 and not 8.
    check("30 Hz tone not counted below 20 Hz", ex[20] < 0.05, f"<20 Hz = {ex[20]}%")
    for cutoff, n_tone in ((40, 2), (200, 3), (2000, 5), (8000, 6), (16000, 7)):
        expected = 100.0 * n_tone / 7.0
        check(f"<{cutoff} Hz == {expected:.4f} % ({n_tone} of 7 sounding seconds)",
              abs(ex[cutoff] - expected) < 0.01,
              f"got {ex[cutoff]}%")

    print("\n4. cumulative curve")
    csum = cumulative(block.psd, block.freqs)
    check("cumulative ends at 100 %", abs(csum[-1] - 100.0) < 1e-6, f"{csum[-1]:.6f}")
    check("cumulative is monotonic", bool(np.all(np.diff(csum) >= -1e-9)))

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES
                  else f"{len(FAILURES)} CHECK(S) FAILED"))
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
