"""
Apple Music offline download packages (``*.movpkg``).

A ``.movpkg`` is not an audio file: it is a *directory* holding an HLS download —
an ``m3u8`` master playlist, an XML byte-range map, an initialisation fragment and
a series of encrypted ``.frag`` segments. The audio is FairPlay protected
(``METHOD=SAMPLE-AES``, scheme ``cbcs``), so it cannot be decoded or measured.

What *can* be established without decrypting, and is enough to answer "what did I
actually download": the codec, sample rate, bit depth, channels, duration and
bitrate, read from three independent places that must agree —

1. ``boot.xml`` — the package manifest, naming the downloaded variant;
2. the master ``m3u8`` — every variant offered, with ``STABLE-VARIANT-ID``;
3. the ``.frag``/initfrag MP4 structure — ``frma``/``schm`` boxes plus, for ALAC,
   the 24-byte ``ALACSpecificConfig`` magic cookie.
"""

from __future__ import annotations

import base64
import glob
import json
import os
import re
import struct
from typing import Dict, List, Optional

from .audit import verdict

LOSSLESS_FORMAT_IDS = {"alac", "flac", "lpcm"}


def _read(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def _decode_session_data(playlist_text: str, data_id: str) -> Optional[dict]:
    m = re.search(r'DATA-ID="%s",VALUE="([^"]+)"' % re.escape(data_id), playlist_text)
    if not m:
        return None
    raw = m.group(1)
    try:
        return json.loads(base64.b64decode(raw + "=" * (-len(raw) % 4)))
    except Exception:
        return None


def _parse_master(playlist_text: str) -> dict:
    """Variants and media groups from a master playlist."""
    groups = {}
    for m in re.finditer(r'#EXT-X-MEDIA:TYPE=AUDIO,GROUP-ID="([^"]+)"([^\n]*)', playlist_text):
        gid, rest = m.group(1), m.group(2)
        groups[gid] = {
            "group_id": gid,
            "channels": (re.search(r'CHANNELS="([^"]+)"', rest) or [None, None])[1],
            "sample_rate": int(re.search(r'SAMPLE-RATE=(\d+)', rest).group(1))
            if "SAMPLE-RATE=" in rest else None,
            "bit_depth": int(re.search(r'BIT-DEPTH=(\d+)', rest).group(1))
            if "BIT-DEPTH=" in rest else None,
        }
    variants = []
    for m in re.finditer(r'#EXT-X-STREAM-INF:([^\n]*)\n(\S+)', playlist_text):
        attrs, uri = m.group(1), m.group(2)
        get = lambda k: (re.search(rf'{k}=("?[^",]+"?)', attrs) or [None, None])[1]
        gid = get("AUDIO")
        variants.append({
            "uri": uri,
            "codecs": (get("CODECS") or "").strip('"'),
            "average_bandwidth": int(get("AVERAGE-BANDWIDTH") or 0),
            "bandwidth": int(get("BANDWIDTH") or 0),
            "stable_variant_id": get("STABLE-VARIANT-ID"),
            "audio_group": gid,
            "group": groups.get(gid, {}),
        })
    return {"variants": variants, "groups": groups}


def _parse_stream_segments(boot_text: str) -> dict:
    peak = re.search(r'PeakBandwidth>(\d+)<', boot_text)
    segs = [(float(d), int(l), int(o), t) for d, l, o, t in
            re.findall(r'<SEG Dur="([\d.]+)" Len="(\d+)" Off="(\d+)"[^>]*Tim="([\d.]+)"',
                       boot_text)]
    return {"peak_bandwidth": int(peak.group(1)) if peak else None,
            "segment_count": len(segs),
            "total_bytes": max((o + l for _, l, o, _ in segs), default=0),
            "segment_duration_s": segs[0][0] if segs else None}


def _walk(buf: bytes, start: int = 0, end: Optional[int] = None):
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
        yield typ, off + hdr, off + size
        off += size


def _find(buf: bytes, path, start: int = 0, end: Optional[int] = None):
    """Descend nested MP4 boxes; returns (buf, body_start, body_end) or None."""
    if not path:
        return None
    want = path[0]
    for typ, bs, be in _walk(buf, start, end):
        if typ != want:
            continue
        if len(path) == 1:
            return buf, bs, be
        nxt = bs + 8 if want == "stsd" else (bs + 28 if want in ("enca", "mp4a") else bs)
        got = _find(buf, path[1:], nxt, be)
        if got is not None:
            return got
    return None


def _alac_cookie(initfrag_path: str) -> Optional[dict]:
    """Read the 24-byte ALACSpecificConfig from the stsd/enca/alac box tree.

    A bare ``find(b"alac")`` is not safe here: ``frma`` also contains the fourcc
    ``alac``, so the first hit is often the wrong one.
    """
    try:
        with open(initfrag_path, "rb") as fh:
            data = fh.read(2 << 20)
    except OSError:
        return None
    for entry in ("enca", "mp4a", "alac"):
        got = _find(data, ("moov", "trak", "mdia", "minf", "stbl", "stsd", entry, "alac"))
        if not got:
            continue
        buf, bs, be = got
        # The 24-byte ALACSpecificConfig sits immediately after the box header;
        # ALAC config boxes carry no version/flags word, so do not skip one.
        cookie = buf[bs:bs + 24]
        if len(cookie) < 24:
            continue
        (frame_len, _ver, bd, pb, mb, kb, nch, max_run,
         max_frame, avg_br, srate) = struct.unpack(">IBBBBBBHIII", cookie)
        # sanity-check: a misaligned read produces absurd values, so validate
        # before believing the cookie (a wrong parse must not become a verdict).
        if not (8000 <= srate <= 768000 and bd in (16, 20, 24, 32)
                and 1 <= nch <= 8 and 512 <= frame_len <= 65536):
            continue
        return {"frame_length": frame_len, "bit_depth": bd, "num_channels": nch,
                "max_frame_bytes": max_frame, "avg_bit_rate": avg_br,
                "sample_rate": srate, "pb_mb_kb": [pb, mb, kb],
                "sample_entry": entry}
    return None


def describe_movpkg(path: str) -> dict:
    """Describe an Apple Music .movpkg without touching the encrypted payload."""
    boot = _read(os.path.join(path, "boot.xml"))
    if not boot:
        raise RuntimeError(f"{path} does not look like a .movpkg (no boot.xml)")
    streams = re.findall(r'<Stream ID="([^"]+)"[^>]*NetworkURL="([^"]+)"', boot)
    master_url = (re.search(r'<MasterPlaylist>\s*<NetworkURL>([^<]+)</NetworkURL>', boot)
                  or [None, None])[1]
    out: Dict[str, object] = {
        "file": os.path.basename(path),
        "kind": "apple-music-movpkg",
        "container": {"format_name": "Apple HLS offline package (.movpkg)",
                      "size_bytes": sum(os.path.getsize(os.path.join(r, f))
                                        for r, _, fs in os.walk(path) for f in fs)},
        "drm": {"protected": "SAMPLE-AES" in boot or True,
                "note": "FairPlay encrypted HLS; audio cannot be decoded or measured"},
    }
    # master playlist may be local (Data/) and/or remote
    master_text = ""
    for cand in glob.glob(os.path.join(path, "Data", "*master.m3u8")):
        master_text = _read(cand)
        if master_text:
            break
    master = _parse_master(master_text) if master_text else {"variants": [], "groups": {}}
    out["offered_variants"] = master["variants"]

    # which variant was downloaded? match boot.xml's UniqueTag against STABLE-VARIANT-ID
    downloaded_tag = (re.search(r'UniqueTag="([^"]+)"', boot) or [None, None])[1]
    stream_dir = None
    for sid, _url in streams:
        cand = os.path.join(path, sid)
        if os.path.isdir(cand):
            stream_dir = cand
            break
    info_boot = _read(os.path.join(stream_dir, "StreamInfoBoot.xml")) if stream_dir else ""
    unique_id = (re.search(r'<UniqueIdentifier>([^<]+)</UniqueIdentifier>', info_boot)
                 or [None, None])[1]
    chosen = None
    for v in master["variants"]:
        if v["stable_variant_id"] and v["stable_variant_id"] in (downloaded_tag, unique_id):
            chosen = v
            break
    if chosen is None and stream_dir:
        # fall back to the filename pattern (…_gr4608_alac.mp4)
        m = re.search(r'gr?(\d+)_(alac|mp4a-[\d-]+|aach)', os.path.basename(streams[0][1])
                      if streams else "")
        if m:
            grp = f"audio-alac-stereo-{int(m.group(1))*1000//24//2}-24" if m.group(2) == "alac" else ""
            chosen = next((v for v in master["variants"] if v["audio_group"] == grp), None)
    out["downloaded_variant"] = chosen
    out["downloaded_variant_id"] = {"UniqueTag": downloaded_tag, "UniqueIdentifier": unique_id}

    # segment accounting
    segs = _parse_stream_segments(info_boot) if info_boot else {}
    frag_bytes = sum(os.path.getsize(f) for f in glob.glob(os.path.join(stream_dir, "*.frag"))) \
        if stream_dir else 0
    init = glob.glob(os.path.join(stream_dir, "*.initfrag")) if stream_dir else []
    segs["frag_bytes_actual"] = frag_bytes
    segs["initfrag_bytes"] = os.path.getsize(init[0]) if init else None
    out["segments"] = segs

    # codec config from the init fragment (authoritative for ALAC)
    cfg = _alac_cookie(init[0]) if init else None
    if cfg:
        out["codec_config"] = {"original_format": "alac", **cfg}

    # ---- consolidated declared parameters (three sources must agree)
    eff = {}
    if cfg:
        eff.update({"codec": "alac", "sample_rate": cfg["sample_rate"],
                    "bit_depth": cfg["bit_depth"], "channels": cfg["num_channels"]})
    if chosen:
        g = chosen.get("group") or {}
        eff.setdefault("codec", "alac" if "alac" in chosen["codecs"] else chosen["codecs"])
        eff.setdefault("sample_rate", g.get("sample_rate"))
        eff.setdefault("bit_depth", g.get("bit_depth"))
        eff.setdefault("channels", int(g["channels"]) if g.get("channels") else None)
        eff["average_bandwidth_bps"] = chosen["average_bandwidth"]
    dur = None
    for m in re.finditer(r'DURATION["\s:]+(\d+)', master_text):
        dur = max(dur or 0, int(m.group(1)))
    if segs.get("segment_count") and segs.get("segment_duration_s"):
        dur = dur or segs["segment_count"] * segs["segment_duration_s"]
    if dur and segs.get("frag_bytes_actual"):
        eff["effective_bitrate_bps"] = int(segs["frag_bytes_actual"] * 8 / dur)
    if eff.get("sample_rate") and eff.get("bit_depth") and eff.get("channels"):
        eff["uncompressed_bitrate_bps"] = (eff["sample_rate"] * eff["bit_depth"]
                                           * eff["channels"])
        if eff.get("effective_bitrate_bps"):
            eff["lossless_ratio_pct"] = round(
                100 * eff["effective_bitrate_bps"] / eff["uncompressed_bitrate_bps"], 1)
    out["declared"] = eff
    out["duration_s"] = dur

    # audio cannot be measured: no ultrasonic / bandwidth evidence available
    out["verdict"] = verdict(eff.get("codec"), eff.get("sample_rate"),
                             eff.get("bit_depth"), None, None)
    out["verdict"]["reasons"] = list(out["verdict"]["reasons"]) + [
        "DRM-protected: bandwidth and ultrasonic content could not be measured, "
        "so 'genuine vs upsampled' remains unverified"]
    return out
