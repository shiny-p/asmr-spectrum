"""Command line interface: ``asmr-spectrum <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from . import __version__
from .audit import audit
from .config import parse_source_spec
from .concat import concat_audio, verify
from .sources import read_manifest, source_boundaries, write_manifest
from .spectrum import analyze_file, scan_file


def _sources(args) -> List[tuple]:
    if getattr(args, "source", None):
        return [parse_source_spec(s) for s in args.source]
    if getattr(args, "from_files", None):
        bounds, _ = source_boundaries(args.from_files)
        return bounds
    return []


def cmd_concat(args) -> int:
    inputs = list(args.inputs)
    if args.manifest:
        inputs += read_manifest(args.manifest)
    if not inputs:
        print("error: no inputs (pass files or --manifest)", file=sys.stderr)
        return 2
    info = concat_audio(inputs, args.output, bitrate=args.bitrate, codec=args.codec)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    if args.write_manifest:
        write_manifest(inputs, args.write_manifest)
        print(f"manifest written: {args.write_manifest}")
    if abs(info["diff_s"]) > args.tolerance_s:
        print(f"warning: output differs from the sum of inputs by "
              f"{info['diff_s']:+.3f} s (> tolerance {args.tolerance_s} s)",
              file=sys.stderr)
        return 1
    return 0


def cmd_verify(args) -> int:
    inputs = list(args.inputs)
    if args.manifest:
        inputs += read_manifest(args.manifest)
    info = verify(inputs, args.merged)
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0 if abs(info["diff_s"]) <= args.tolerance_s else 1


def cmd_analyze(args) -> int:
    sources = _sources(args)
    os.makedirs(args.outdir, exist_ok=True)
    rc = 0
    summary = []
    for path in args.inputs:
        label = args.label or os.path.splitext(os.path.basename(path))[0]
        prefix = os.path.join(args.outdir, label)
        try:
            payload = analyze_file(path, prefix, label=label, sources=sources,
                                   figure=not args.no_figure, csv=not args.no_csv)
        except Exception as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            rc = 1
            continue
        ed = payload["energy_distribution_exact_pct"]
        summary.append({
            "label": label, "file": payload["file"],
            "duration_min": round(payload["duration_s"] / 60, 2),
            "rms_db": payload["loudness"]["rms_db_mono"],
            "true_peak_db": max(payload["loudness"]["true_peak_db_LR"]),
            "below_20hz_pct": ed["below_20hz"], "below_40hz_pct": ed["below_40hz"],
            "below_200hz_pct": ed["below_200hz"],
            "centroid_hz": payload["energy_distribution_psd_pct"]["spectral_centroid_hz"],
        })
    if len(summary) > 1 and args.summary:
        with open(args.summary, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)
        print(f"summary: {args.summary}")
    return rc


def fmt_audit(r: dict) -> str:
    """Human-readable audit summary."""
    if r.get("kind") == "apple-music-movpkg":
        d = r.get("declared", {})
        v = r.get("verdict", {})
        lines = [f"file            : {r['file']}",
                 "kind            : Apple Music offline package (.movpkg)",
                 f"DRM             : {r.get('drm', {}).get('note', '?')}"]
        if d:
            lines += [
                f"codec           : {d.get('codec')}",
                f"sample rate     : {d.get('sample_rate')} Hz",
                f"bit depth       : {d.get('bit_depth')} bit",
                f"channels        : {d.get('channels')}",
            ]
            if d.get("average_bandwidth_bps"):
                lines.append(f"declared bitrate: {d['average_bandwidth_bps'] / 1000:.0f} kbps "
                             f"(avg bandwidth)")
            if d.get("effective_bitrate_bps"):
                lines.append(f"effective rate  : {d['effective_bitrate_bps'] / 1e6:.3f} Mbps "
                             f"({d.get('lossless_ratio_pct')}% of uncompressed)")
        if r.get("duration_s"):
            lines.append(f"duration        : {r['duration_s'] / 60:.2f} min")
        if r.get("downloaded_variant"):
            lines.append(f"variant         : {r['downloaded_variant']['audio_group']}")
        lines += [f"bandwidth       : NOT MEASURABLE (encrypted)",
                  f"ultrasonic      : NOT MEASURABLE (encrypted)",
                  f"verdict         : {v.get('label', '?')}"]
        for reason in v.get("reasons", []):
            lines.append(f"                  - {reason}")
        return "\n".join(lines)

    d = r.get("declared", {})
    bd = r.get("bit_depth", {})
    bw = r.get("bandwidth", {})
    us = r.get("ultrasonic", {})
    top = r.get("top_octave", {})
    stab = r.get("bandwidth_stability", {})
    v = r.get("verdict", {})

    def g(dct, key, fmt="{}", dash="n/a"):
        val = dct.get(key) if isinstance(dct, dict) else None
        return fmt.format(val) if val is not None else dash

    lines = [
        f"file            : {r['file']}",
        f"container       : {g(r.get('container', {}), 'format_name')}",
        f"codec           : {d.get('codec')}"
        + (f" ({d.get('profile')})" if d.get("profile") else ""),
        f"sample rate     : {d.get('sample_rate')} Hz"
        + (f"  (Nyquist {d['sample_rate'] // 2} Hz)" if d.get("sample_rate") else ""),
        f"bit depth       : declared {d.get('bit_depth') or 'n/a'}"
        f" | effective {(str(g(bd, 'grid_bits')) + ' bit') if bd.get('grid_bits') else 'none (lossy/float)'}"
        + (f"  [{bd.get('method')}]" if bd.get("grid_bits") else ""),
        f"channels        : {d.get('channels')}",
        f"bitrate         : stream {d.get('stream_bitrate') or 'n/a'}"
        + (f" | measured {d['measured_bitrate'] / 1000:.0f} kbps"
           if d.get("measured_bitrate") else ""),
        f"duration        : {r.get('container', {}).get('duration_s', 0) / 60:.2f} min",
        f"peak            : {g(r.get('probe', {}), 'peak_dbfs')} dBFS",
        f"noise floor     : {r.get('noise_floor_db')} dBFS",
        f"bandwidth       : {g(bw, 'edge_-45db_hz')} Hz (-45 dB)"
        + (f" | {g(bw, 'edge_-60db_hz')} Hz (-60 dB)" if bw.get("edge_-60db_hz") else ""),
        f"bandwidth       : {g(bw, 'bandwidth_to_nyquist')} x Nyquist"
        + (f"   {'FIXED FILTER (lossy/lowpassed)' if stab.get('looks_like_fixed_filter') else 'varies with content'}"
           if stab.get("looks_like_fixed_filter") is not None else ""),
        f"HF content      : "
        + (f"top half of passband {top.get('band_hz')} Hz -> "
           + ("PRESENT" if top.get("content_present")
              else ("ABSENT — upsampled from a lower rate"
                    if (d.get("sample_rate") or 0) >= 88200
                    else "ABSENT — band-limited (lowpass/lossy codec)"))
           + f"  [peak {top.get('band_peak_rel_db')} dB rel. ref, "
             f"{top.get('peak_above_local_floor_db')} dB over local floor]"
           if top.get("applicable")
           else f"not applicable — {top.get('note', 'n/a')}")
        + (f"\n                  secondary (above source Nyquist): "
           f"{us.get('content_above_source_nyquist_rel_db')} dB"
           if us.get("content_above_source_nyquist_rel_db") is not None else ""),
        f"verdict         : {v.get('label', '?')}",
    ]
    for reason in v.get("reasons", []):
        lines.append(f"                  - {reason}")
    return "\n".join(lines)


def cmd_audit(args) -> int:
    results = []
    rc = 0
    for path in args.inputs:
        try:
            r = audit(path, seconds=args.seconds, deep=not args.shallow)
        except Exception as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
            rc = 1
            continue
        results.append(r)
        if not args.json:
            if len(args.inputs) > 1:
                print("=" * 72)
            print(fmt_audit(r))
    if args.json:
        payload = results[0] if len(results) == 1 else results
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")
    return rc


def cmd_scan(args) -> int:
    rows = []
    for path in args.inputs:
        try:
            rows.append(scan_file(path))
        except Exception as exc:
            print(f"error: {path}: {exc}", file=sys.stderr)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, ensure_ascii=False)
        print(f"wrote {args.json}")
    else:
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="asmr-spectrum",
        description="Concatenate audio losslessly and analyse its frequency, "
                    "low-frequency, loudness and energy distribution.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    c = sub.add_parser("concat", help="concatenate audio files into one track")
    c.add_argument("output", help="output file, e.g. work.m4a")
    c.add_argument("inputs", nargs="*", help="input files, in order")
    c.add_argument("--manifest", help="file listing one input path per line")
    c.add_argument("--write-manifest", help="write the resolved input order here")
    c.add_argument("--bitrate", default="192k")
    c.add_argument("--codec", default="aac")
    c.add_argument("--tolerance-s", type=float, default=0.05,
                   help="allowed |output - sum(inputs)| duration in seconds")
    c.set_defaults(func=cmd_concat)

    v = sub.add_parser("verify", help="check a concatenation against its inputs")
    v.add_argument("merged")
    v.add_argument("inputs", nargs="*")
    v.add_argument("--manifest")
    v.add_argument("--tolerance-s", type=float, default=0.05)
    v.set_defaults(func=cmd_verify)

    a = sub.add_parser("analyze", help="spectrum / loudness / energy analysis")
    a.add_argument("inputs", nargs="+", help="audio file(s)")
    a.add_argument("-o", "--outdir", default="analysis_out")
    a.add_argument("--label", help="output name (only for a single input)")
    a.add_argument("--source", action="append",
                   help="attribute a time range to a source: t0,t1,name")
    a.add_argument("--from-files", nargs="+", metavar="FILE",
                   help="inputs that were concatenated, in order; boundaries are "
                        "derived from their durations")
    a.add_argument("--no-figure", action="store_true")
    a.add_argument("--no-csv", action="store_true")
    a.add_argument("--summary", help="write a cross-file comparison JSON here")
    a.set_defaults(func=cmd_analyze)

    s = sub.add_parser("scan", help="quick level + low-frequency table, no plots")
    s.add_argument("inputs", nargs="+")
    s.add_argument("--json", help="write the table as JSON")
    s.set_defaults(func=cmd_scan)

    au = sub.add_parser("audit", help="sample rate / bit depth / codec / bitrate / "
                                      "ultrasonic content, with a Hi-Res verdict")
    au.add_argument("inputs", nargs="+",
                    help="audio file(s), or an Apple Music .movpkg directory")
    au.add_argument("--json", help="write full results as JSON")
    au.add_argument("--seconds", type=float, default=90.0,
                    help="audio to analyse for bit depth / bandwidth (default 90)")
    au.add_argument("--shallow", action="store_true",
                    help="container metadata only; skip signal analysis")
    au.set_defaults(func=cmd_audit)
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
