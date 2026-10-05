#!/usr/bin/env python3
"""Self-test: build audio with known properties and check the audit's answers.

Usage:  python tests/validate_audit.py     (needs numpy + ffmpeg on PATH)

Fixtures are generated, never committed. Each check states the physically correct
answer, so a regression in the heuristics shows up as a failure:

  16-bit master           -> effective 16 bit
  24-bit master           -> effective 24 bit
  16-bit padded into 24   -> effective 16 bit (not 24: the grid is coarser)
  lossy AAC               -> no integer grid
  genuine 96 kHz/24 with a 30 kHz tone -> hi-res, HF content PRESENT
  same content lowpassed at 20 kHz and upsampled to 96 kHz -> questionable
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from asmr_spectrum.audit import audit  # noqa: E402

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(name)


def _signal(sr: int, secs: float, ultrasonic: bool, lowpass: float | None,
            seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = int(sr * secs)
    t = np.arange(n) / sr
    x = 0.35 * (0.5 * np.sin(2 * np.pi * 220 * t)
                + 0.25 * np.sin(2 * np.pi * 1170 * t)
                + 0.15 * np.sin(2 * np.pi * 5200 * t))
    if ultrasonic:
        x += 0.05 * np.sin(2 * np.pi * 30000 * t) + 0.03 * np.sin(2 * np.pi * 40000 * t)
    x += 0.0015 * rng.standard_normal(n)
    if lowpass:
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, d=1.0 / sr)
        X[f > lowpass] = 0
        x = np.fft.irfft(X, n)
    return x


def _write(path: str, x: np.ndarray, sr: int, bits: int) -> None:
    n = x.size
    if bits == 24:
        q = np.clip(np.round(x * 8388607), -8388608, 8388607).astype(np.int64)
        u = (q & 0xFFFFFF).astype(np.uint32)
        b = np.empty((n, 3), dtype=np.uint8)
        b[:, 0], b[:, 1], b[:, 2] = u & 0xFF, (u >> 8) & 0xFF, (u >> 16) & 0xFF
        data = np.repeat(b[:, None, :], 2, axis=1).tobytes()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s24le", "-ar", str(sr),
                        "-ac", "2", "-i", "-", "-c:a", "flac", "-sample_fmt", "s32", path],
                       input=data, check=True)
    else:
        q = np.clip(np.round(x * 32767), -32768, 32767).astype("<i2")
        data = np.repeat(q[:, None], 2, axis=1).tobytes()
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "s16le", "-ar", str(sr),
                        "-ac", "2", "-i", "-", "-c:a", "flac", path],
                       input=data, check=True)


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="asmr_audit_test_")
    print(f"fixtures in {tmp}\n")

    f16 = os.path.join(tmp, "m16.flac")
    f24 = os.path.join(tmp, "m24.flac")
    real = os.path.join(tmp, "real_hires.flac")
    fake = os.path.join(tmp, "fake_hires.flac")
    lossy = os.path.join(tmp, "lossy.m4a")
    _write(f16, _signal(48000, 12, False, None), 48000, 16)
    _write(f24, _signal(48000, 12, False, None), 48000, 24)
    _write(real, _signal(96000, 20, True, None), 96000, 24)
    _write(fake, _signal(96000, 20, False, 20000), 96000, 24)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f16, "-c:a", "aac",
                    "-b:a", "192k", lossy], check=True)

    print("1. effective bit depth (quantisation grid)")
    for path, want, label in ((f16, 16, "16-bit master"), (f24, 24, "24-bit master")):
        r = audit(path, seconds=10)
        got = r["bit_depth"]["grid_bits"]
        check(f"{label} -> {want} bit", got == want, f"got {got}")

    print("\n2. codec classification")
    r = audit(f16, seconds=10)
    check("lossless FLAC detected as lossless", r["verdict"]["lossless"] is True)
    r = audit(lossy, seconds=10)
    check("AAC detected as lossy", r["verdict"]["lossless"] is False)
    check("lossy audio shows no integer grid",
          r["bit_depth"]["grid_bits"] is None, str(r["bit_depth"]["grid_bits"]))
    check("AAC -> not hi-res", r["verdict"]["is_hires"] is False)

    print("\n3. sample rate and rate-vs-bandwidth")
    r = audit(real, seconds=15)
    check("96 kHz reported", r["declared"]["sample_rate"] == 96000)
    r = audit(fake, seconds=15)
    check("upsampled file still reports 96 kHz (rate alone proves nothing)",
          r["declared"]["sample_rate"] == 96000)
    check("...but its bandwidth edge sits near the 20 kHz lowpass",
          r["bandwidth"].get("edge_-45db_hz") is not None
          and abs(r["bandwidth"]["edge_-45db_hz"] - 20000) < 1500,
          str(r["bandwidth"].get("edge_-45db_hz")))
    check("...and that edge is constant over time (fixed filter)",
          r["bandwidth_stability"].get("looks_like_fixed_filter") is True)

    print("\n4. HF content in the top half of the passband")
    r = audit(real, seconds=15)
    check("genuine 96 kHz master -> content PRESENT",
          r["top_octave"].get("content_present") is True,
          f"peak {r['top_octave'].get('band_peak_rel_db')} dB")
    check("genuine 96 kHz master -> verdict verified hi-res",
          r["verdict"]["label"].startswith("hi-res (verified"), r["verdict"]["label"])
    r = audit(fake, seconds=15)
    check("upsampled 96 kHz -> content ABSENT",
          r["top_octave"].get("content_present") is False,
          f"peak {r['top_octave'].get('band_peak_rel_db')} dB")
    check("upsampled 96 kHz -> verdict questionable",
          "questionable" in r["verdict"]["label"], r["verdict"]["label"])

    print("\n5. verdict for ordinary lossless CD-rate audio")
    r = audit(f16, seconds=10)
    check("48 kHz/16 -> not hi-res", r["verdict"]["is_hires"] is False)
    check("reason mentions the 48 kHz limit",
          any("48 kHz" in s for s in r["verdict"]["reasons"]),
          "; ".join(r["verdict"]["reasons"]))

    print("\n" + ("ALL CHECKS PASSED" if not FAILURES
                  else f"{len(FAILURES)} CHECK(S) FAILED"))
    for f in FAILURES:
        print(f"  - {f}")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
