"""Per-frame state features used for classification and clustering.

All features are scale-normalised so the classifier works across videos.
"""

from __future__ import annotations

import cv2
import numpy as np

from face_detect import FaceDetection


FEATURE_NAMES = [
    # head orientation
    "face_aspect",        # face bbox w/h  (frontal≈0.8, profile collapses)
    "eye_dist_norm",      # inter-eye distance / face width
    "eye_offset_x",       # eye midpoint offset from face centre, / w
    "eye_offset_y",       # eye midpoint offset from face centre, / h
    "eye_width_asym",     # min/max eye blob width (profile → far eye shrinks)
    # eye state
    "eye_dark_ratio",     # dark pixel ratio in both eye regions
    "eye_lum",            # mean luminance of eye regions (open = whiter)
    # mouth state
    "mouth_dark_ratio",   # dark pixel ratio in mouth region
    "mouth_aspect",       # mouth blob h/w
    # global scale (for forward/backward later)
    "char_area_ratio",    # silhouette area / frame area
]


def extract_features(
    bgr: np.ndarray, alpha: np.ndarray, det: FaceDetection | None
) -> dict[str, float] | None:
    """Extract the classification feature vector for one frame."""
    if det is None:
        return None

    h, w = bgr.shape[:2]
    fx, fy, fw, fh = det.bbox
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    mask = (alpha > 128).astype(np.uint8)
    dark = ((gray < 110) & (mask > 0)).astype(np.uint8)

    feats: dict[str, float] = {}

    # --- head orientation ---
    feats["face_aspect"] = fw / max(fh, 1)
    if det.left_eye and det.right_eye:
        ex1, ey1 = det.left_eye
        ex2, ey2 = det.right_eye
        eye_dist = abs(ex2 - ex1)
        feats["eye_dist_norm"] = eye_dist / max(fw, 1)
        mid_x = (ex1 + ex2) / 2
        mid_y = (ey1 + ey2) / 2
        feats["eye_offset_x"] = (mid_x - (fx + fw / 2)) / max(fw, 1)
        feats["eye_offset_y"] = (mid_y - (fy + fh / 2)) / max(fh, 1)
        # eye blob widths (from dark map around each eye)
        w1 = _blob_width(dark, (int(ex1), int(ey1)), fh)
        w2 = _blob_width(dark, (int(ex2), int(ey2)), fh)
        feats["eye_width_asym"] = min(w1, w2) / (max(w1, w2) + 1e-6)
    else:
        # single eye / no eyes → profile-ish defaults
        feats["eye_dist_norm"] = 0.0
        feats["eye_offset_x"] = 0.0
        feats["eye_offset_y"] = 0.0
        feats["eye_width_asym"] = 0.0

    # --- eye state: box centred on the eye midpoint ---
    if det.left_eye and det.right_eye:
        ecx = (det.left_eye[0] + det.right_eye[0]) / 2
        ecy = (det.left_eye[1] + det.right_eye[1]) / 2
        eye_feats = _box_stats(gray, dark, ecx, ecy, fw * 0.60, fh * 0.30)
    else:
        eye_feats = _box_stats(gray, dark, fx + fw / 2, fy + fh * 0.45, fw * 0.60, fh * 0.30)
    feats["eye_dark_ratio"] = eye_feats["dark_ratio"]
    feats["eye_lum"] = eye_feats["lum"]

    # --- mouth state: box centred on mouth (fallback: below nose) ---
    if det.mouth:
        mcx, mcy = det.mouth
    elif det.nose:
        mcx, mcy = det.nose[0], det.nose[1] + fh * 0.30
    else:
        mcx, mcy = fx + fw / 2, fy + fh * 0.80
    mouth_feats = _box_stats(gray, dark, mcx, mcy, fw * 0.45, fh * 0.28)
    feats["mouth_dark_ratio"] = mouth_feats["dark_ratio"]
    feats["mouth_aspect"] = mouth_feats["aspect"]

    # --- global scale ---
    feats["char_area_ratio"] = float(mask.sum()) / (h * w)

    return feats


# ----------------------------------------------------------------------

def _blob_width(dark: np.ndarray, center: tuple[int, int], radius: int) -> float:
    x, y = center
    r = max(4, radius // 6)
    h, w = dark.shape
    x0, x1 = max(0, x - r), min(w, x + r)
    y0, y1 = max(0, y - r), min(h, y + r)
    sub = dark[y0:y1, x0:x1]
    if sub.sum() == 0:
        return 0.0
    cols = np.nonzero(sub.sum(axis=0))[0]
    if len(cols) == 0:
        return 0.0
    return float(cols[-1] - cols[0] + 1)


def _box_stats(
    gray: np.ndarray,
    dark: np.ndarray,
    cx: float, cy: float,
    bw: float, bh: float,
) -> dict[str, float]:
    """Stats of a small box centred at (cx, cy) — avoids hair pollution."""
    h, w = gray.shape
    x0 = max(0, int(cx - bw / 2))
    x1 = min(w, int(cx + bw / 2))
    y0 = max(0, int(cy - bh / 2))
    y1 = min(h, int(cy + bh / 2))
    if x1 <= x0 or y1 <= y0:
        return {"dark_ratio": 0.0, "lum": 1.0, "aspect": 0.0}

    sub_dark = dark[y0:y1, x0:x1]
    sub_gray = gray[y0:y1, x0:x1]

    total = max(1, sub_dark.size)
    dark_ratio = float(sub_dark.sum()) / total
    lum = float(sub_gray.mean()) / 255.0 if sub_gray.size else 1.0

    # aspect of the largest dark blob in this box
    n, _, stats, _ = cv2.connectedComponentsWithStats(sub_dark, 8)
    aspect = 0.0
    best_area = 0
    for i in range(1, n):
        _, _, bw2, bh2, area = stats[i]
        if area > best_area:
            best_area = area
            aspect = bh2 / max(bw2, 1)
    return {"dark_ratio": dark_ratio, "lum": lum, "aspect": aspect}
