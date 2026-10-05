#!/usr/bin/env python3
"""Cross-validate the Swift `audioscope` binary against the Python reference.

Generates fixtures with known properties, runs both implementations over them, and
compares the measured numbers field by field. Any disagreement is a bug in one of
them — the Python side is the one that was validated against closed-form values in
tests/validate_audit.py, so it acts as the oracle here.

Usage:
    python tests/crosscheck_swift.py [path/to/audioscope]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

from asmr_spectrum.audit import audit  # noqa: E402

BIN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    REPO, "swift", ".build", "release", "audioscope")

# Tolerances. FFT bin alignment and float32-vs-float64 accumulation differ slightly,
# so exact equality is not expected; these bounds catch real disagreements.
TOL = {
    "sample_rate": 0,
    "bit_depth": 0,
    "peak_dbfs": 0.05,
    "noise_floor_db": 3.0,
    "edge_hz": 400.0,
    "ratio": 0.02,
    "band_rel_db": 3.0,
    "top_octave_peak": 5.0,
    "top_octave_contrast": 5.0,
}

FAILS: list[str] = []


def check(name: str, a, b, tol) -> None:
    ok = False
    detail = f"python={a!r} swift={b!r}"
    if a is None or b is None:
        ok = (a is None) == (b is None)
    elif isinstance(tol, (int, float)) and tol == 0:
        ok = a == b
    else:
        ok = abs(float(a) - float(b)) <= float(tol)
    print(f"  {'PASS' if ok else 'FAIL'}  {name:34s} {detail}")
    if not ok:
        FAILS.append(f"{name}: {detail}")


def _signal(sr, secs, ultrasonic, lowpass, seed=11, noise_floor_db=-70.0,
            hf_content_db=-18.0):
    """Synthesise audio whose spectrum resembles real music.

    Modelled on a measured lossy file (KPOP AAC @48k), where level relative to the
    2-10 kHz band runs about -8 dB at 10-13 kHz, -15 dB at 13-16 kHz, -21 dB at
    16-18 kHz, then collapses to -56 dB above the codec's 19.3 kHz cutoff. That
    32 dB cliff is what makes a bandwidth edge measurable; a fixture whose spectrum
    is empty above 10 kHz cannot show one no matter how low its noise floor is.

    `noise_floor_db` sets a shaped broadband floor (steady to 8 kHz, then falling),
    `hf_content_db` the 10-19 kHz content level.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * secs)
    t = np.arange(n) / sr
    fn = np.fft.rfftfreq(n, d=1.0 / sr)
    tone_freqs = [(220, 0.5), (440, 0.2), (1170, 0.25), (5200, 0.15),
                  (9000, 0.08), (13000, 0.06)]
    if ultrasonic:
        tone_freqs += [(30000, 0.16), (40000, 0.10)]
    tonal = np.zeros(n)
    for f0, amp in tone_freqs:
        if f0 < sr * 0.48:
            tonal += amp * np.sin(2 * np.pi * f0 * t)

    # shaped noise floor: flat below 8 kHz, then -6 dB/octave
    floor = rng.standard_normal(n)
    shape = np.ones_like(fn)
    hi = fn > 8000
    shape[hi] = (fn[hi] / 8000.0) ** -1.0
    floor = np.fft.irfft(np.fft.rfft(floor) * shape, n)
    floor /= np.abs(floor).max()

    # dense HF content in the audible top octaves, level set relative to the tonal band
    hf = rng.standard_normal(n)
    hf_shape = np.zeros_like(fn)
    hf_shape[(fn >= 10000) & (fn < 19500)] = 1.0
    hf = np.fft.irfft(np.fft.rfft(hf) * hf_shape, n)
    hf /= max(np.abs(hf).max(), 1e-12)

    tonal /= np.abs(tonal).max()
    x = tonal + hf * 10 ** (hf_content_db / 20.0) + floor * 10 ** (noise_floor_db / 20.0)
    if lowpass:
        X = np.fft.rfft(x)
        X[fn > lowpass] = 0
        x = np.fft.irfft(X, n)
    x /= np.abs(x).max()
    return x


def _write(path, x, sr, bits):
    n = x.size
    if bits == 24:
        # scale by 2**23 so a full-scale sample maps exactly to the int24 limit
        q = np.clip(np.round(x * 8388608), -8388608, 8388607).astype(np.int64)
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


def swift_audit(path: str) -> dict:
    # Foundation cannot write to /dev/stdout, so use a real temp path.
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out_path = fh.name
    try:
        out = subprocess.run([BIN, "--json", out_path, path],
                             capture_output=True, text=True)
        if out.returncode != 0:
            raise SystemExit(f"swift audit failed on {path}:\n{out.stderr}")
        with open(out_path) as fh:
            return json.load(fh)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def compare(label: str, path: str, seconds: float = 15.0) -> None:
    print(f"\n=== {label}")
    py = audit(path, seconds=seconds)
    sw = swift_audit(path)
    check("sample rate", py["declared"]["sample_rate"],
          sw["declared"]["sample_rate"], 0)
    check("codec", py["declared"]["codec"], sw["declared"]["codec"], 0)
    check("effective bit depth", py["bit_depth"].get("grid_bits"),
          sw["bit_depth"].get("grid_bits"), 0)
    check("peak dBFS", py["probe"]["peak_dbfs"], sw["peak_dbfs"], TOL["peak_dbfs"])
    check("noise floor dB", py["noise_floor_db"], sw["noise_floor_db"],
          TOL["noise_floor_db"])

    pbe = py["bandwidth"].get("edge_-45db_hz")
    sbe = sw["bandwidth"].get("edge_-45db_hz")
    if pbe is None or sbe is None:
        check("bandwidth edge -45dB (both absent)", pbe, sbe, 0)
    else:
        check("bandwidth edge -45dB", pbe, sbe, TOL["edge_hz"])
    pbr = py["bandwidth"].get("bandwidth_to_nyquist")
    sbr = sw["bandwidth"].get("bandwidth_to_nyquist")
    if pbr is not None and sbr is not None:
        check("bandwidth / Nyquist", pbr, sbr, TOL["ratio"])
    else:
        check("bandwidth / Nyquist (both absent)", pbr, sbr, 0)

    pfix = py["bandwidth_stability"].get("looks_like_fixed_filter")
    sfix = sw["bandwidth_stability"].get("looks_like_fixed_filter")
    check("fixed filter", pfix, sfix, 0)

    ptop, stop = py["top_octave"], sw["top_octave"]
    check("top octave applicable", ptop.get("applicable"), stop.get("applicable"), 0)
    if ptop.get("applicable") and stop.get("applicable"):
        check("top octave content present", ptop.get("content_present"),
              stop.get("content_present"), 0)
        check("top octave peak rel dB", ptop.get("band_peak_rel_db"),
              stop.get("band_peak_rel_db"), TOL["top_octave_peak"])
        check("top octave contrast dB", ptop.get("peak_above_local_floor_db"),
              stop.get("peak_above_local_floor_db"), TOL["top_octave_contrast"])

    # Wording differs slightly between implementations; compare the classification.
    def classify(label: str) -> str:
        if "verified" in label:
            return "verified"
        if "questionable" in label:
            return "questionable"
        if "unverified" in label:
            return "unverified"
        return "not-hires"
    check("verdict class", classify(py["verdict"]["label"]),
          classify(sw["verdict"]["label"]), 0)


def main() -> int:
    if not os.path.exists(BIN):
        raise SystemExit(f"binary not found: {BIN}\nbuild it with: "
                         f"cd swift && swift build -c release")
    print(f"reference: python  asmr_spectrum.audit")
    print(f"candidate: swift   {BIN}")
    tmp = tempfile.mkdtemp(prefix="audioscope_cross_")
    f16 = os.path.join(tmp, "m16.flac")
    f24 = os.path.join(tmp, "m24.flac")
    real = os.path.join(tmp, "real_hires.flac")
    fake = os.path.join(tmp, "fake_hires.flac")
    lossy = os.path.join(tmp, "lossy.m4a")
    _write(f16, _signal(48000, 12, False, None, noise_floor_db=-90), 48000, 16)
    _write(f24, _signal(48000, 12, False, None, noise_floor_db=-90), 48000, 24)
    _write(real, _signal(96000, 20, True, None, noise_floor_db=-90), 96000, 24)
    _write(fake, _signal(96000, 20, False, 20000, noise_floor_db=-90), 96000, 24)
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", f16, "-c:a", "aac",
                    "-b:a", "192k", lossy], check=True)

    compare("16-bit master @48k", f16)
    compare("24-bit master @48k", f24)
    compare("genuine 96k/24 master with 30 kHz tone", real)
    compare("upsampled from 20 kHz lowpass", fake)
    compare("lossy AAC @48k", lossy)

    print("\n" + ("ALL CROSSCHECKS PASSED — implementations agree" if not FAILS
                  else f"{len(FAILS)} DISAGREEMENT(S):"))
    for f in FAILS:
        print("  -", f)
    print(f"\nfixtures left in {tmp}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    raise SystemExit(main())
