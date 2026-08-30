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

FEATURE_ORDER = [
    "face_aspect", "eye_dist_norm", "eye_offset_x", "eye_offset_y",
    "eye_width_asym", "eye_dark_ratio", "eye_lum",
    "mouth_dark_ratio", "mouth_aspect", "char_area_ratio",
]


def _most_typical(fids: list[int], features: list) -> int:
    """Frame closest to the group's median feature vector.

    Sharpest would bias towards busy back-of-head frames (hair texture);
    the most central frame is the most typical pose for the state.
    """
    vecs = []
    for i in fids:
        f = features[i] if i < len(features) else None
        if f is None:
            continue
        vecs.append(np.array([float(f.get(k, 0.0)) for k in FEATURE_ORDER]))
    if not vecs:
        return fids[0]
    med = np.median(np.stack(vecs), axis=0)
    spread = np.median(np.abs(np.stack(vecs) - med), axis=0) + 1e-6

    def dist(i: int) -> float:
        f = features[i]
        if f is None:
            return 1e9
        v = np.array([float(f.get(k, 0.0)) for k in FEATURE_ORDER])
        return float(np.abs((v - med) / spread).mean())

    return min(fids, key=dist)


def export(
    workdir: Path,
    out_dir: Path,
    frames: list[np.ndarray],
    classifications: dict[str, dict],
    features: list | None = None,
    min_conf: float = 0.55,
) -> dict[str, int]:
    """Export all classified frames, one PNG per (theme, state) key.

    Frames below *min_conf* are auto-classification guesses — they are
    excluded so an ambiguous frame can't define a state's look.  Manual
    annotations carry conf=1.0 and always pass.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    skipped = 0

    # Group frames by state key
    grouped: dict[str, list[int]] = {}
    for fid, cls in classifications.items():
        if float(cls.get("conf", 0.0)) < min_conf:
            skipped += 1
            continue
        head = cls.get("head") or "center"
        eye = cls.get("eye") or "open"
        mouth = cls.get("mouth") or "closed"
        theme = cls.get("theme", "default")
        key = f"{theme}/{mouth}_{eye}_{head}"
        grouped.setdefault(key, []).append(int(fid))

    for key, fids in grouped.items():
        theme, state = key.split("/", 1)
        if features is not None:
            best_fid = _most_typical(fids, features)
        else:
            best_fid = max(fids, key=lambda i: sharpness_score(frames[i]))

        rgba = white_bg_matting(frames[best_fid])
        dest = out_dir / theme / f"{state}.png"
        dest.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest), rgba)

        counts[theme] = counts.get(theme, 0) + 1
        print(f"  {key}  (frame {best_fid}, {len(fids)} candidates)")

    total = sum(len(v) for v in grouped.values())
    print(f"  ({skipped} low-confidence frames excluded from export, "
          f"{total} kept)")
    return counts
