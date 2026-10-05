"""
Audio parameter audit: sample rate, effective bit depth, codec, bitrate and
ultrasonic content — plus a verdict on Hi-Res claims.

Design notes, each backed by a measurement in ``tests/``:

* **Effective vs declared bit depth.** The container's claim is not evidence.
  Integer PCM sits on a *quantisation grid*: samples of a 16-bit master are exact
  multiples of 1/32768 no matter what container they are wrapped in. We decode to
  float32 (never int16 — that would truncate a 24-bit master down to 16 bits and
  hide the very thing we are testing) and find the *coarsest* grid the samples
  land on. A 16-bit master therefore reads 16 even if stored as 24-bit.
* **Bandwidth vs Nyquist.** A lossless 96 kHz file whose energy stops at 19 kHz
  was not recorded at 96 kHz. We locate the spectral edge and compare it with the
  Nyquist frequency, and we check whether that edge is *constant over time* (a
  fixed filter, i.e. a lossy codec or a resampler) or content-dependent.
* **Ultrasonic content.** Decoding above the file's own rate and finding energy
  where the source cannot have any is the signature of upsampling.
* **DRM.** Encrypted containers (Apple Music ``.movpkg``) can be described from
  their manifests and codec config boxes, but the audio cannot be measured. The
  audit says so rather than guessing.
"""

from __future__ import annotations

import json
import math
import os
import re
import struct
import subprocess
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import audio as _audio
from .config import SAMPLE_RATE, hann

# Bit-depth grid candidates. A master at N bits lands on every grid >= N, so the
# coarsest match is the answer.
GRID_BITS = (8, 12, 16, 20, 24, 32)

# Analysis constants
PROBE_SECONDS = 90.0        # audio examined for bit-depth / bandwidth
BANDWIDTH_NFFT = 16384
BANDWIDTH_REF_DB = 45.0     # edge = first frequency this far below the 2-10 kHz mean
ULTRASONIC_CONTRAST_DB = 20.0   # peak must tower over its own band's median
ULTRASONIC_MIN_REL_DB = -45.0   # ...and must not be decoder/resampling noise
TOP_OCTAVE_MIN_REL_DB = -40.0   # genuine HF content in the top half of the passband
TOP_OCTAVE_CONTRAST_DB = 15.0   # ...towering over that band's own median


class AuditError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# container-level inspection
# --------------------------------------------------------------------------- #
def probe_streams(path: str) -> List[dict]:
    _audio.require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", path],
        capture_output=True, check=True).stdout
    j = json.loads(out)
    streams = []
    for s in j.get("streams", []):
        if s.get("codec_type") != "audio":
            continue
        streams.append({
            "index": s.get("index"),
            "codec": s.get("codec_name"),
            "codec_long": s.get("codec_long_name"),
            "profile": s.get("profile"),
            "sample_rate": int(s["sample_rate"]) if s.get("sample_rate") else None,
            "channels": s.get("channels"),
            "channel_layout": s.get("channel_layout"),
            "declared_bit_depth": (int(s["bits_per_raw_sample"])
                                   if s.get("bits_per_raw_sample") not in (None, "0", 0)
                                   else None),
            "stream_bitrate": int(s["bit_rate"]) if s.get("bit_rate") else None,
        })
    fmt = j.get("format") or {}
    return streams, {
        "format_name": fmt.get("format_name"),
        "duration_s": float(fmt.get("duration", 0.0) or 0.0),
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
        "format_bitrate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
    }


def _walk_boxes(buf: bytes, start: int = 0, end: Optional[int] = None):
    """Yield (type, body_start, body_end, box_start) for top-level MP4 boxes."""
    end = len(buf) if end is None else end
    off = start
    while off + 8 <= end:
        size = struct.unpack(">I", buf[off:off + 4])[0]
        typ = buf[off + 4:off + 8].decode("latin1", "replace")
        hdr = 8
        if size == 1 and off + 16 <= end:
            size = struct.unpack(">Q", buf[off + 8:off + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - off
        if size < hdr or off + size > end:
            return
        yield typ, off + hdr, off + size, off
        off += size


def _find_box(buf: bytes, path: Sequence[str], start: int = 0, end: Optional[int] = None):
    """Descend into nested boxes, e.g. ('moov','trak','mdia','minf','stbl','stsd')."""
    if not path:
        return None
    want = path[0]
    for typ, bs, be, bstart in _walk_boxes(buf, start, end):
        if typ != want:
            continue
        if len(path) == 1:
            return buf, bs, be
        child = buf[bs:be]
        if want == "stsd":                  # skip version/flags + entry_count
            bs2 = bs + 8
        elif want in ("enca", "mp4a", "alac"):
            bs2 = bs + 28                   # AudioSampleEntry fixed fields
        else:
            bs2 = bs
        res = _find_box(buf, path[1:], bs2, be)
        if res is not None:
            return res
    return None


def inspect_mp4_codec(path: str, limit: int = 4 << 20) -> Optional[dict]:
    """Read the codec configuration box directly (works for DRM-protected files).

    Returns declared format / bit depth / sample rate from the actual bitstream
    config, which is stronger evidence than container metadata and is available
    even when the payload is encrypted.
    """
    try:
        with open(path, "rb") as fh:
            head = fh.read(limit)
    except OSError:
        return None
    if head[4:8] != b"ftyp":
        return None
    info = {"mp4_brand": head[8:12].decode("latin1", "replace")}
    ftyp = next((b for b in _walk_boxes(head) if b[0] == "ftyp"), None)
    if ftyp:
        _, bs, be, _ = ftyp
        brands = [head[bs + 4 * i:bs + 4 * i + 4].decode("latin1", "replace")
                  for i in range((be - bs - 4) // 4)]
        info["mp4_compatible_brands"] = brands
    stsd = _find_box(head, ("moov", "trak", "mdia", "minf", "stbl", "stsd"))
    if not stsd:
        return info
    buf, bs, be = stsd
    entries = []
    for typ, es, ee, _ in _walk_boxes(buf, bs, be):
        if typ not in ("enca", "mp4a", "alac", "lpcm", "fLaC", "Opus", "ec-3", "ac-3"):
            continue
        e = {"sample_entry": typ}
        # original format / protection scheme
        sinf = _find_box(buf, ("enca", "sinf"), es, ee) if typ == "enca" else None
        if sinf:
            s_buf, s_bs, s_be = sinf
            for t2, b2, e2, _ in _walk_boxes(s_buf, s_bs, s_be):
                if t2 == "frma":
                    e["original_format"] = s_buf[b2:e2].decode("latin1", "replace")
                elif t2 == "schm":
                    body = s_buf[b2:e2]
                    e["protection_scheme"] = body[4:8].decode("latin1", "replace")
                    if len(body) >= 12:
                        e["protection_version"] = struct.unpack(">I", body[8:12])[0]
        # ALAC magic cookie: 24-byte ALACSpecificConfig
        inner = _find_box(buf, (typ, "alac"), es, ee)
        if inner:
            ib, ibs, ibe = inner
            cookie = ib[ibs:ibs + 24]
            if len(cookie) == 24:
                (frame_len, ver, bd, pb, mb, kb, nch, max_run,
                 max_frame, avg_br, srate) = struct.unpack(">IBBBBBBHIII", cookie)
                e["alac"] = {"frame_length": frame_len, "bit_depth": bd,
                             "num_channels": nch, "max_run": max_run,
                             "max_frame_bytes": max_frame, "avg_bit_rate": avg_br,
                             "sample_rate": srate, "pb_mb_kb": [pb, mb, kb]}
        entries.append(e)
    if entries:
        info["sample_entries"] = entries
    return info


# --------------------------------------------------------------------------- #
# signal-level measurement
# --------------------------------------------------------------------------- #
def _decode_mono_f32(path: str, seconds: float, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode up to ``seconds`` of audio as float32 mono (no int16 truncation)."""
    cmd = ["ffmpeg", "-v", "error", "-t", str(seconds), "-i", path, "-map", "0:a:0",
           "-vn", "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "1",
           "-ar", str(rate), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<f4").astype(np.float64)


def _decode_int32_mono(path: str, seconds: float, rate: int) -> np.ndarray:
    """Decode to int32 PCM — the unambiguous representation for grid tests."""
    cmd = ["ffmpeg", "-v", "error", "-t", str(seconds), "-i", path, "-map", "0:a:0",
           "-vn", "-f", "s32le", "-acodec", "pcm_s32le", "-ac", "1",
           "-ar", str(rate), "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return np.frombuffer(raw, dtype="<i4").astype(np.float64)


def detect_bit_depth(x: np.ndarray, peak: Optional[float] = None,
                     scale: Optional[float] = None, fs_bits: int = 32) -> dict:
    """Coarsest quantisation grid the samples land on.

    ``x`` must be **integer PCM** (int32 decode). Do not pass float samples: the
    scale factor ffmpeg uses for its float output is not a clean power of two
    (measured: float = int / 2**30.5, i.e. off by sqrt(2) from the naive 2**31),
    so any grid test on float data needs calibration and is fragile.

    ``scale`` is the full-scale value for ``fs_bits`` (e.g. 2**31 for int32).
    """
    fs = float(scale if scale is not None else 2 ** (fs_bits - 1))
    out = {"grid_bits": None, "declared_by_grid": None, "max_deviation": None,
           "margin_db": None, "method": "quantisation grid (int32 decode)",
           "full_scale": int(fs)}
    if x.size == 0:
        return out
    a = np.abs(x)
    pk = float(peak if peak is not None else a.max())
    if pk <= 0:
        return out
    active = a > pk * 1e-4                     # ignore near-silence / digital zero
    if active.sum() < 500:
        active = np.ones_like(a, dtype=bool)
    xs = x[active]
    # A grid at N bits means samples are multiples of 2**(fs_bits-N) counts.
    best = None
    for bits in GRID_BITS:
        if bits >= fs_bits:
            continue
        step = float(2 ** (fs_bits - bits))
        dev_counts = float(np.max(np.abs(xs / step - np.round(xs / step))))
        dev = dev_counts * step                      # deviation in full-scale units
        if best is None or dev < best[1]:
            best = (bits, dev)
        if dev < 1e-6:
            out.update({"grid_bits": bits, "declared_by_grid": bits,
                        "max_deviation": dev,
                        "margin_db": round(20 * math.log10(fs / max(pk, 1.0))
                                           - 6.02 * bits, 1),
                        "note": "coarsest matching grid"})
            break
    if out["grid_bits"] is None:
        out["max_deviation"] = best[1] if best else None
        out["note"] = ("no integer grid found — consistent with a lossy codec, a "
                       "float master, or a resampled/dithered signal")
    return out


def noise_floor_db(x: np.ndarray, percentile: float = 5.0) -> Optional[float]:
    """Low percentile of short-frame power, in dBFS — a noise-floor indicator.

    For material with true silence this finds the floor; for continuous music it
    reports the quietest passages instead, so it must be read as "quietest 5 % of
    frames", not as an absolute noise floor. Even so it bounds the source bit
    depth usefully: 16-bit material cannot sit far below -96 dBFS.
    """
    if x.size < 4096:
        return None
    n = 1 << 14
    hop = n // 2
    powers = []
    for i in range(0, x.size - n, hop):
        powers.append(float((x[i:i + n] ** 2).mean()))
        if len(powers) >= 4000:
            break
    p = np.asarray(powers)
    p = p[p > 0]
    if p.size == 0:
        return None
    return round(float(10 * math.log10(max(float(np.percentile(p, percentile)), 1e-30))), 1)


def bandwidth_profile(x: np.ndarray, rate: int = SAMPLE_RATE) -> dict:
    """Locate the spectral edge and measure what lies beyond it."""
    out: Dict[str, object] = {}
    n_fft = BANDWIDTH_NFFT
    if x.size < n_fft * 2:
        # short input: shrink the transform so an edge can still be located
        n_fft = int(2 ** max(11, min(14, math.floor(math.log2(max(x.size // 4, 2))))))
        if x.size < n_fft * 2:
            return {"error": f"too little audio ({x.size} samples) for bandwidth analysis"}
    w = hann(n_fft)
    wss = float(np.sum(w ** 2))
    f = np.fft.rfftfreq(n_fft, d=1.0 / rate)
    acc = np.zeros(f.size)
    n = 0
    for i in range(0, x.size - n_fft, n_fft // 2):
        acc += np.abs(np.fft.rfft(x[i:i + n_fft] * w)) ** 2
        n += 1
    psd = acc / max(n, 1) / wss
    db = 10 * np.log10(np.maximum(psd, 1e-30))
    sm = np.convolve(db, np.ones(5) / 5, mode="same")
    ref = float(sm[(f >= 2000) & (f < 10000)].mean())
    nyquist = rate / 2.0

    out["reference_level_db"] = round(ref, 2)
    out["bin_hz"] = round(rate / n_fft, 2)
    for thr in (BANDWIDTH_REF_DB, 60.0):
        idx = np.where((f > 13000) & (f < nyquist * 0.999) & (sm < ref - thr))[0]
        key = f"edge_{-int(thr)}db_hz"
        out[key] = round(float(f[idx[0]]), 0) if idx.size else None
    edge = out.get(f"edge_{-int(BANDWIDTH_REF_DB)}db_hz")
    if edge:
        out["bandwidth_to_nyquist"] = round(float(edge) / nyquist, 3)
    # band levels relative to the 2-10 kHz reference
    def rel(lo, hi):
        m = (f >= lo) & (f < min(hi, nyquist))
        if not m.any():
            return None
        return round(float(sm[m].mean()) - ref, 1)
    out["band_rel_db"] = {
        "2-10khz": 0.0,
        "10-16khz": rel(10000, 16000),
        "16-19khz": rel(16000, 19000),
        "19-21khz": rel(19000, 21000),
        "21-23khz": rel(21000, 23000),
        "23-24khz": rel(23000, 24000),
        "24-30khz": rel(24000, 30000),
        "30-40khz": rel(30000, 40000),
        "40khz+": rel(40000, 96000),
    }
    return out


def top_octave_check(x: np.ndarray, declared_rate: int) -> dict:
    """Is there real content in the top half of the passband (Nyquist/2 .. Nyquist)?

    This is the decisive "genuine Hi-Res vs upsampled" test, and it needs **no
    resampling**: the file is examined at its own rate. A genuine 96 kHz master
    carries content up to ~40 kHz; a 48 kHz master repackaged as 96 kHz is silent
    up there. Fixture measurements: a true 96 kHz master with a 30 kHz tone gave a
    band peak +68 dB above the 2-10 kHz reference; the same content lowpassed at
    20 kHz and upsampled gave -90 dB.

    Decoding at a *higher* rate than the file would instead make the resampler's own
    images appear above the source Nyquist, which is the exact artefact this test
    must not mistake for content.
    """
    nyq = declared_rate / 2.0
    if nyq <= 20000:
        return {"applicable": False,
                "note": f"Nyquist {nyq:.0f} Hz leaves no room above the audible band"}
    n = 1 << 15
    if x.size < n * 2:
        return {"applicable": False, "note": "too little audio"}
    w = hann(n)
    f = np.fft.rfftfreq(n, d=1.0 / declared_rate)
    acc = np.zeros(f.size)
    c = 0
    for i in range(0, x.size - n, n // 2):
        acc += np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2
        c += 1
    db = 10 * np.log10(np.maximum(acc / max(c, 1), 1e-30))
    ref = float(db[(f >= 2000) & (f < 10000)].mean())
    lo, hi = nyq * 0.5, nyq * 0.99
    m = (f >= lo) & (f < hi)
    if not m.any():
        return {"applicable": False}
    b = db[m]
    med = float(np.median(b))
    peak = round(float(b.max()) - ref, 1)
    contrast = round(float(b.max()) - med, 1)
    ok = peak > TOP_OCTAVE_MIN_REL_DB and contrast > TOP_OCTAVE_CONTRAST_DB
    return {
        "applicable": True,
        "band_hz": [round(lo), round(hi)],
        "band_peak_rel_db": peak,
        "band_median_rel_db": round(med - ref, 1),
        "peak_above_local_floor_db": contrast,
        "criterion": (f"peak > {TOP_OCTAVE_MIN_REL_DB:.0f} dB relative to the 2-10 kHz "
                      f"reference AND > {TOP_OCTAVE_CONTRAST_DB:.0f} dB above the "
                      "band's own median"),
        "content_present": bool(ok),
    }


def ultrasonic_check(path: str, declared_rate: int, seconds: float = 60.0) -> dict:
    """Secondary cross-check: decode above the source rate and inspect that region.

    Reported for evidence, not trusted blindly: the resampler can create images
    above the source Nyquist, so a "peak" up there is not proof of content. The
    decisive test is :func:`top_octave_check`.
    """
    target = 192000 if declared_rate > 48000 else 96000
    if target <= declared_rate:
        return {"skipped": f"declared rate {declared_rate} Hz >= probe rate {target}"}

    try:
        x = _decode_mono_f32(path, seconds, rate=target)
    except subprocess.CalledProcessError:
        return {"error": "decode at probe rate failed"}
    if x.size < 1 << 16:
        return {"error": "too little audio"}
    n = 1 << 15
    w = hann(n)
    f = np.fft.rfftfreq(n, d=1.0 / target)
    acc = np.zeros(f.size)
    c = 0
    for i in range(0, x.size - n, n // 2):
        acc += np.abs(np.fft.rfft(x[i:i + n] * w)) ** 2
        c += 1
    db = 10 * np.log10(np.maximum(acc / max(c, 1), 1e-30))
    ref = float(db[(f >= 2000) & (f < 10000)].mean())

    def band_stats(lo, hi):
        m = (f >= lo) & (f < hi)
        if not m.any():
            return None, None
        return (round(float(db[m].mean()) - ref, 1),
                round(float(db[m].max()) - ref, 1))

    above_lo = declared_rate // 2 + 1000
    above_hi = target / 2 - 1000
    m = (f >= above_lo) & (f < above_hi)
    if not m.any():
        return {"probe_rate": target, "error": "no band above the source Nyquist"}
    band = db[m]
    band_med = float(np.median(band))
    band_max = float(band.max())
    peak_rel = round(band_max - ref, 1)
    contrast = round(band_max - band_med, 1)
    # Two conditions, because either one alone can be fooled:
    #  * absolute level: resampling and decoder noise sits ~90 dB down, content does not;
    #  * contrast against the band's own median: broadband noise lifts its peak and
    #    its median together, a genuine tone does not.
    present = peak_rel > ULTRASONIC_MIN_REL_DB and contrast > ULTRASONIC_CONTRAST_DB
    return {
        "probe_rate": target,
        "probe_nyquist": target // 2,
        "source_nyquist": declared_rate // 2,
        "level_rel_db": {
            "2-10khz": 0.0,
            "10-20khz": band_stats(10000, 20000)[0],
            "20-24khz": band_stats(20000, 24000)[0],
            "24-30khz": band_stats(24000, 30000)[0],
            "30-40khz": band_stats(30000, 40000)[0],
            "40khz+": band_stats(40000, target / 2 - 1000)[0],
        },
        "above_source_nyquist": {
            "band_hz": [above_lo, above_hi],
            "band_mean_rel_db": round(float(band.mean()) - ref, 1),
            "band_median_rel_db": round(band_med - ref, 1),
            "band_peak_rel_db": round(band_max - ref, 1),
            "peak_above_local_floor_db": contrast,
            "criterion": (f"peak > {ULTRASONIC_MIN_REL_DB:.0f} dB relative to the "
                          f"2-10 kHz reference AND > {ULTRASONIC_CONTRAST_DB:.0f} dB "
                          "above the band's own median"),
        },
        "content_above_source_nyquist_rel_db": peak_rel,
        "content_above_nyquist_present": bool(present),
    }


def bandwidth_stability(x: np.ndarray, rate: int = SAMPLE_RATE,
                        windows: int = 8) -> dict:
    """Is the spectral edge fixed (a filter) or content-dependent?"""
    if x.size < BANDWIDTH_NFFT * 4:
        return {"error": "too little audio"}
    win = x.size // windows
    w = hann(BANDWIDTH_NFFT)
    wss = float(np.sum(w ** 2))
    f = np.fft.rfftfreq(BANDWIDTH_NFFT, d=1.0 / rate)
    edges, levels = [], []
    for k in range(windows):
        seg = x[k * win:(k + 1) * win]
        if seg.size < BANDWIDTH_NFFT * 2:
            continue
        acc = np.zeros(f.size)
        c = 0
        for i in range(0, seg.size - BANDWIDTH_NFFT, BANDWIDTH_NFFT // 2):
            acc += np.abs(np.fft.rfft(seg[i:i + BANDWIDTH_NFFT] * w)) ** 2
            c += 1
        sm = np.convolve(10 * np.log10(np.maximum(acc / max(c, 1) / wss, 1e-30)),
                         np.ones(5) / 5, mode="same")
        ref = float(sm[(f >= 8000) & (f < 12000)].mean())
        levels.append(round(ref, 1))
        idx = np.where((f > 13000) & (f < rate / 2 * 0.999) & (sm < ref - BANDWIDTH_REF_DB))[0]
        edges.append(float(f[idx[0]]) if idx.size else None)
    ok = [e for e in edges if e is not None]
    if len(ok) < 2:
        return {"edges_hz": edges, "note": "edge not found in enough windows"}
    mean = float(np.mean(ok))
    return {
        "edges_hz": [round(e) if e else None for e in edges],
        "window_level_db": levels,
        "edge_mean_hz": round(mean),
        "edge_std_hz": round(float(np.std(ok)), 1),
        "edge_relative_spread_pct": round(float(np.std(ok)) / mean * 100, 2),
        "looks_like_fixed_filter": bool(np.std(ok) / mean < 0.02),
    }


# --------------------------------------------------------------------------- #
# verdict
# --------------------------------------------------------------------------- #
HIRES_MIN_RATE = 88200          # JAS: "96 kHz/24 bit or above" (88.2 kHz is the 2x family)
HIRES_MIN_BITS = 24
LOSSLESS_CODECS = {"flac", "alac", "pcm_s16le", "pcm_s24le", "pcm_s32le", "wavpack",
                   "ape", "tak", "truehd", "mlp"}


def verdict(codec: Optional[str], rate: Optional[int], bits: Optional[int],
            ultra_present: Optional[bool], bandwidth_ratio: Optional[float]) -> dict:
    lossless = (codec or "").lower() in LOSSLESS_CODECS or \
        (codec or "").lower().startswith("pcm_")
    is_hires = bool(rate and rate >= HIRES_MIN_RATE) and lossless
    reasons: List[str] = []
    if not rate:
        reasons.append("sample rate unknown")
    elif rate < 44100:
        reasons.append(f"sample rate {rate} Hz is below CD")
    elif rate == 44100:
        reasons.append("44.1 kHz = CD quality, not Hi-Res")
    elif rate <= 48000:
        reasons.append(f"{rate} Hz is at or below 48 kHz; Hi-Res requires >48 kHz")
    if not lossless:
        reasons.append(f"codec {codec or '?'} is lossy; Hi-Res requires lossless")
    if bits and bits < HIRES_MIN_BITS:
        reasons.append(f"effective bit depth {bits} < 24")
    genuine: Optional[bool] = None
    if is_hires:
        genuine = True
        if rate and rate > 48000 and ultra_present is False:
            genuine = False
            reasons.append("no content above the source's own Nyquist: "
                           "upsampled from a lower rate")
        if bandwidth_ratio is not None and rate and rate >= 88200:
            expected = 20000.0 / (rate / 2.0)
            if bandwidth_ratio < max(0.45, expected * 0.9):
                genuine = False
                reasons.append(f"bandwidth only {bandwidth_ratio:.2f}x Nyquist: "
                               "the top octaves are empty")
        if ultra_present is None:
            # e.g. DRM-encrypted payload: the container says Hi-Res, the content
            # could not be examined, so the claim is unverified rather than proven.
            genuine = None
    if not is_hires:
        label = "not hi-res"
    elif genuine is False:
        label = "hi-res container, questionable content"
    elif genuine is None:
        label = "hi-res by container (content unverified)"
    else:
        label = "hi-res (verified: ultrasonic content present)"
    return {"is_hires": is_hires, "lossless": lossless, "genuine": genuine,
            "label": label, "reasons": reasons}


# --------------------------------------------------------------------------- #
# top level
# --------------------------------------------------------------------------- #
def audit(path: str, seconds: float = PROBE_SECONDS, deep: bool = True) -> dict:
    """Full parameter audit for one audio file."""
    if not os.path.exists(path):
        raise AuditError(f"not found: {path}")
    if os.path.isdir(path):
        # Apple Music .movpkg and friends: describe from manifests, no audio access
        from .movpkg import describe_movpkg
        return describe_movpkg(path)
    streams, fmt = probe_streams(path)
    if not streams:
        raise AuditError(f"no audio stream in {path}")
    st = streams[0]
    declared_rate = st["sample_rate"]
    res: Dict[str, object] = {
        "file": os.path.basename(path),
        "container": fmt,
        "streams": streams,
        "declared": {"codec": st["codec"], "sample_rate": declared_rate,
                     "channels": st["channels"],
                     "bit_depth": st["declared_bit_depth"],
                     "stream_bitrate": st["stream_bitrate"]},
    }
    if fmt.get("size_bytes") and fmt.get("duration_s"):
        res["declared"]["measured_bitrate"] = int(
            fmt["size_bytes"] * 8 / fmt["duration_s"])
    # codec config from the container (also works when encrypted)
    mp4 = inspect_mp4_codec(path)
    if mp4:
        res["codec_config"] = mp4
    if not deep:
        return res

    x = _decode_mono_f32(path, seconds, rate=declared_rate or SAMPLE_RATE)
    if x.size == 0:
        res["error"] = "decoded no samples"
        return res
    peak = float(np.abs(x).max())
    res["probe"] = {"seconds_analysed": round(x.size / (declared_rate or SAMPLE_RATE), 1),
                    "peak_dbfs": round(20 * math.log10(max(peak, 1e-12)), 2)}
    # bit depth from integer PCM (float output has a non-power-of-two scale factor)
    try:
        xi = _decode_int32_mono(path, seconds, declared_rate or SAMPLE_RATE)
        res["bit_depth"] = detect_bit_depth(xi)
        if xi.size:
            res["bit_depth"]["peak_int32"] = int(np.abs(xi).max())
    except subprocess.CalledProcessError:
        res["bit_depth"] = detect_bit_depth(x, peak=peak)
        res["bit_depth"]["method"] = "quantisation grid (float decode, uncalibrated)"
    res["noise_floor_db"] = noise_floor_db(x)
    res["bandwidth"] = bandwidth_profile(x, rate=declared_rate or SAMPLE_RATE)
    res["bandwidth_stability"] = bandwidth_stability(x, rate=declared_rate or SAMPLE_RATE)
    # decisive HF test: measured at the file's own rate, no resampling involved
    top = top_octave_check(x, declared_rate or SAMPLE_RATE)
    res["top_octave"] = top
    # secondary evidence: what lies above the source Nyquist after upsampling
    res["ultrasonic"] = ultrasonic_check(path, declared_rate or SAMPLE_RATE,
                                         seconds=min(seconds, 60))
    res["high_frequency"] = {
        "content_in_top_octave": top.get("content_present") if top.get("applicable") else None,
        "applicable": bool(top.get("applicable")),
    }

    eff_bits = res["bit_depth"].get("grid_bits")
    present = top.get("content_present") if top.get("applicable") else None
    res["verdict"] = verdict(st["codec"], declared_rate, eff_bits,
                             present,
                             res["bandwidth"].get("bandwidth_to_nyquist"))
    return res
