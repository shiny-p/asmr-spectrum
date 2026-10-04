"""Utilities for splitting a concatenated work back into its source files."""

from __future__ import annotations

import json
import os
from typing import Dict, List, Sequence, Tuple

from . import audio


def source_boundaries(paths: Sequence[str]) -> Tuple[List[Tuple[float, float, str]],
                                                    float]:
    """Cumulative ``(t0, t1, name)`` boundaries for files concatenated in order."""
    out: List[Tuple[float, float, str]] = []
    t = 0.0
    for p in paths:
        d = audio.duration(p)
        out.append((round(t, 3), round(t + d, 3), os.path.basename(p)))
        t += d
    return out, round(t, 3)


def write_manifest(paths: Sequence[str], dest: str) -> str:
    """Write one absolute input path per line."""
    with open(dest, "w", encoding="utf-8") as fh:
        for p in paths:
            fh.write(os.path.abspath(p) + "\n")
    return dest


def read_manifest(src: str) -> List[str]:
    with open(src, encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip()]


def group_report(rows: Sequence[Dict], path: str) -> str:
    """Write a cross-file comparison JSON (one entry per analysed file)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(list(rows), fh, indent=2, ensure_ascii=False)
    return path
