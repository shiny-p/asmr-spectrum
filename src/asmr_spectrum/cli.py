"""Command line interface: ``asmr-spectrum <command>``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List

from . import __version__
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
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
