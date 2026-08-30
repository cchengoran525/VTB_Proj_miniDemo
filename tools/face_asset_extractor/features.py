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
    # eye state (tight per-eye crops — the old one-big-box features were
    # polluted by brows/hair/highlights and could not separate open/closed)
    "eye_dark_ratio",     # dark (iris/lash) ratio inside per-eye crops
    "eye_blob_aspect",    # largest dark blob h/w — tall iris vs flat closed lid
    "eye_lum",            # mean luminance of eye crops
    # mouth state
    "mouth_dark_ratio",   # dark ratio in mouth crop
    "mouth_red_ratio",    # saturated warm ratio — open mouth interior
    "mouth_blob_aspect",  # dark blob h/w — open mouth is tall, closed is a line
    "mouth_aspect",       # mouth blob h/w (legacy)
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

    # --- eye state: tight per-eye crops (bw≈0.22 face-w, bh≈0.16 face-h) ---
    if det.left_eye and det.right_eye:
        eye_boxes = [det.left_eye, det.right_eye]
    else:
        eye_boxes = [(fx + fw / 2, fy + fh * 0.45)]
    ratios, whites, aspects, lums = [], [], [], []
    for ex, ey in eye_boxes:
        st = _crop_stats(gray, dark, mask, ex, ey, fw * 0.22, fh * 0.16)
        ratios.append(st["dark_ratio"])
        whites.append(st["white_ratio"])
        aspects.append(st["blob_aspect"])
        lums.append(st["lum"])
    feats["eye_dark_ratio"] = float(np.mean(ratios))
    feats["eye_white_ratio"] = float(np.mean(whites))
    feats["eye_blob_aspect"] = float(np.mean(aspects))
    feats["eye_lum"] = float(np.mean(lums))

    # --- mouth state: crop centred on mouth (fallback: below nose) ---
    if det.mouth:
        mcx, mcy = det.mouth
    elif det.nose:
        mcx, mcy = det.nose[0], det.nose[1] + fh * 0.30
    else:
        mcx, mcy = fx + fw / 2, fy + fh * 0.80
    mst = _crop_stats(gray, dark, mask, mcx, mcy, fw * 0.34, fh * 0.20,
                      red=True, bgr=bgr)
    feats["mouth_dark_ratio"] = mst["dark_ratio"]
    feats["mouth_red_ratio"] = mst["red_ratio"]
    feats["mouth_blob_aspect"] = mst["blob_aspect"]
    feats["mouth_aspect"] = mst["aspect"]

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


def _crop_stats(
    gray: np.ndarray,
    dark: np.ndarray,
    mask: np.ndarray,
    cx: float, cy: float,
    bw: float, bh: float,
    red: bool = False,
    bgr: np.ndarray | None = None,
) -> dict[str, float]:
    """Stats inside a small character-masked crop — avoids hair/bg pollution."""
    h, w = gray.shape
    x0 = max(0, int(cx - bw / 2))
    x1 = min(w, int(cx + bw / 2))
    y0 = max(0, int(cy - bh / 2))
    y1 = min(h, int(cy + bh / 2))
    if x1 <= x0 or y1 <= y0:
        return {"dark_ratio": 0.0, "white_ratio": 0.0, "red_ratio": 0.0,
                "lum": 1.0, "aspect": 0.0, "blob_aspect": 0.0}

    m = mask[y0:y1, x0:x1] > 0
    sub_dark = dark[y0:y1, x0:x1]
    sub_gray = gray[y0:y1, x0:x1]

    total = max(1, int(m.sum()))
    dark_ratio = float(sub_dark[m].sum()) / total
    white_ratio = float(((sub_gray > 185) & m).sum()) / total
    lum = float(sub_gray[m].mean()) / 255.0 if m.any() else 1.0

    red_ratio = 0.0
    if red and bgr is not None:
        sub_bgr = bgr[y0:y1, x0:x1].astype(np.int16)
        warm = ((sub_bgr[:, :, 2] - sub_bgr[:, :, 0]) > 45) & m
        red_ratio = float(warm.sum()) / total

    # largest dark blob geometry inside the crop
    n, _, stats, _ = cv2.connectedComponentsWithStats(sub_dark, 8)
    aspect = 0.0
    blob_aspect = 0.0
    best_area = 0
    for i in range(1, n):
        _, _, bw2, bh2, area = stats[i]
        if area > best_area:
            best_area = area
            aspect = bh2 / max(bw2, 1)
            blob_aspect = aspect
    return {"dark_ratio": dark_ratio, "white_ratio": white_ratio,
            "red_ratio": red_ratio, "lum": lum, "aspect": aspect,
            "blob_aspect": blob_aspect}
