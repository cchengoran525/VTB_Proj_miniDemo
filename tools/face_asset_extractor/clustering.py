"""Frame dedup — group near-identical frames, keep the sharpest."""

from __future__ import annotations

import cv2
import numpy as np


def dedup_frames(
    features: list[dict[str, float] | None],
    quant_steps: dict[str, float] | None = None,
    max_candidates: int = 400,
) -> list[list[int]]:
    """
    Group frames by quantised feature vector.

    Frames whose quantised features are identical are considered the same
    state; only the sharpest is kept as the candidate representative.
    Returns clusters as lists of frame indices.
    """
    default_steps = {
        "face_aspect": 0.10,
        "eye_dist_norm": 0.08,
        "eye_offset_x": 0.08,
        "eye_offset_y": 0.10,
        "eye_width_asym": 0.15,
        "eye_dark_ratio": 0.10,
        "eye_lum": 0.08,
        "mouth_dark_ratio": 0.08,
        "mouth_aspect": 0.20,
        "char_area_ratio": 0.05,
    }
    steps = quant_steps or default_steps

    buckets: dict[tuple, list[int]] = {}
    for i, feat in enumerate(features):
        if feat is None:
            continue
        key = tuple(
            round(feat.get(name, 0.0) / steps.get(name, 1.0))
            for name in default_steps
        )
        buckets.setdefault(key, []).append(i)

    clusters = sorted(buckets.values(), key=len, reverse=True)

    # Cap total candidate frames while keeping diverse coverage
    if len(clusters) > max_candidates:
        clusters = clusters[:max_candidates]

    return clusters


def sharpness_score(bgr: np.ndarray) -> float:
    """Laplacian variance — higher = sharper frame."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def pick_representatives(
    clusters: list[list[int]], frames: list[np.ndarray]
) -> list[int]:
    """Pick the sharpest frame from each cluster."""
    reps = []
    for cluster in clusters:
        best = max(cluster, key=lambda i: sharpness_score(frames[i]))
        reps.append(best)
    return reps
