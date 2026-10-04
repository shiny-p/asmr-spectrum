"""Audio I/O: probing and streamed decoding through ffmpeg.

Only ffmpeg/ffprobe do decoding; no Python audio library is required, and no
temporary PCM file is ever written for the analysis paths.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import subprocess
from typing import Iterator, Optional

import numpy as np

from .config import SAMPLE_RATE

CHUNK_BYTES = 1 << 22  # 4 MiB per read


class FFmpegMissing(RuntimeError):
    pass


def require_ffmpeg() -> None:
    missing = [t for t in ("ffmpeg", "ffprobe") if shutil.which(t) is None]
    if missing:
        raise FFmpegMissing(
            "ffmpeg and ffprobe must be on PATH (install with: brew install ffmpeg)"
        )


def probe(path: str) -> dict:
    """Return duration/codec/rate/channels/bitrate for the first audio stream."""
    require_ffmpeg()
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate",
         "-show_entries", "format=duration,bit_rate,size", "-of", "json", path],
        capture_output=True, check=True).stdout
    j = json.loads(out)
    st = (j.get("streams") or [{}])[0]
    fmt = j.get("format") or {}
    return {
        "codec": st.get("codec_name"),
        "sample_rate": int(st["sample_rate"]) if st.get("sample_rate") else None,
        "channels": int(st["channels"]) if st.get("channels") else None,
        "stream_bit_rate": int(st["bit_rate"]) if st.get("bit_rate") else None,
        "duration_s": float(fmt.get("duration", 0.0)),
        "format_bit_rate": int(fmt["bit_rate"]) if fmt.get("bit_rate") else None,
        "size_bytes": int(fmt["size"]) if fmt.get("size") else None,
    }


def duration(path: str) -> float:
    return probe(path)["duration_s"]


def decode_stream(path: str, channels: int = 2,
                  rate: int = SAMPLE_RATE) -> Iterator[np.ndarray]:
    """Yield float64 blocks of shape ``(frames, channels)`` from an audio file."""
    require_ffmpeg()
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-map", "0:a:0", "-vn",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", str(channels),
           "-ar", str(rate), "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while True:
            raw = proc.stdout.read(CHUNK_BYTES)
            if not raw:
                break
            a = np.frombuffer(raw, dtype="<f4")
            a = a[: (a.size // channels) * channels].reshape(-1, channels)
            if a.size:
                yield a.astype(np.float64)
    finally:
        if proc.stdout:
            proc.stdout.close()
        err = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"ffmpeg failed on {path!r} (rc={rc}): {err[:500]}")


def decode_mono(path: str, rate: int = SAMPLE_RATE) -> np.ndarray:
    """Whole file as one mono float64 array (convenience for short inputs)."""
    parts = [blk.mean(axis=1) for blk in decode_stream(path, 2, rate)]
    return np.concatenate(parts) if parts else np.zeros(0)


@contextlib.contextmanager
def stream(path: str, channels: int = 2, rate: int = SAMPLE_RATE):
    """Context manager yielding the decode generator, closing ffmpeg cleanly."""
    gen = decode_stream(path, channels, rate)
    try:
        yield gen
    finally:
        gen.close()
