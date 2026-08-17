"""Export classified frames as transparent PNGs matching the main app naming.

Output layout mirrors frames_placeholder/:
  {out}/{theme}/{mouth}_{eye}_{head}.png
"""

from __future__ import annotations

import cv2
import json
from pathlib import Path

import numpy as np

from clustering import sharpness_score
from matting import white_bg_matting


def export(
    workdir: Path,
    out_dir: Path,
    frames: list[np.ndarray],
    classifications: dict[str, dict],
) -> dict[str, int]:
    """Export all classified frames, one PNG per (theme, state) key."""
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}

    # Group frames by state key, keep the sharpest per state
    grouped: dict[str, list[int]] = {}
    for fid, cls in classifications.items():
        head = cls.get("head") or "center"
        eye = cls.get("eye") or "open"
        mouth = cls.get("mouth") or "closed"
        theme = cls.get("theme", "default")
        key = f"{theme}/{mouth}_{eye}_{head}"
        grouped.setdefault(key, []).append(int(fid))

    for key, fids in grouped.items():
        theme, state = key.split("/", 1)
        best_fid = max(fids, key=lambda i: sharpness_score(frames[i]))

        rgba = white_bg_matting(frames[best_fid])
        dest = out_dir / theme / f"{state}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), rgba)

        counts[theme] = counts.get(theme, 0) + 1
        print(f"  {key}  (frame {best_fid}, {len(fids)} candidates)")

    return counts
