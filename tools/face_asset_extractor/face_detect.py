"""Anime face detection — heuristic silhouette + dark-blob analysis.

Interface is intentionally simple (`detect(bgr, alpha) → FaceDetection`)
so a real ML model can replace this module later without touching the
rest of the pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


@dataclass
class FaceDetection:
    bbox: tuple[int, int, int, int]        # x, y, w, h
    left_eye: Optional[tuple[float, float]] = None
    right_eye: Optional[tuple[float, float]] = None
    nose: Optional[tuple[float, float]] = None
    mouth: Optional[tuple[float, float]] = None
    confidence: float = 0.0                # heuristic quality score


class FaceDetector:
    """Finds the head and facial landmarks of an anime character.

    Assumes: single character, roughly upright, white-ish background
    (alpha mask supplied by matting step).
    """

    def __init__(self, min_char_area_ratio: float = 0.02) -> None:
        self.min_char_area_ratio = min_char_area_ratio

    def detect(self, bgr: np.ndarray, alpha: np.ndarray) -> Optional[FaceDetection]:
        h, w = bgr.shape[:2]
        mask = (alpha > 128).astype(np.uint8)

        # --- character must occupy a reasonable area ---
        if mask.sum() < self.min_char_area_ratio * h * w:
            return None

        # --- silhouette bounding box ---
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            return None
        top, bottom = int(ys.min()), int(ys.max())
        left, right = int(xs.min()), int(xs.max())
        char_h = bottom - top + 1

        # --- width profile over the whole silhouette ---
        prof: list[tuple[int, int]] = []   # (width, y)
        for y in range(top, bottom + 1):
            row = np.nonzero(mask[y, :])[0]
            if len(row):
                prof.append((row[-1] - row[0] + 1, y))
        if not prof:
            return None

        # widest row in the upper 45% = face width reference
        upper = [p for p in prof if p[1] <= top + int(char_h * 0.45)]
        if not upper:
            upper = prof
        max_width, _max_y = max(upper, key=lambda p: p[0])

        # face bottom = first row below the widest point where width
        # drops to 75% of max (chin taper / neck).  Fallback: silhouette bottom.
        face_top = top
        face_bottom = bottom
        for w, y in prof:
            if y > _max_y and w < max_width * 0.75:
                face_bottom = y
                break

        fh = face_bottom - face_top + 1
        if fh < 10:            # head-only fallback
            face_bottom = bottom
            fh = face_bottom - face_top + 1

        # face width at the widest row
        widest_row = np.nonzero(mask[_max_y, :])[0]
        if len(widest_row):
            face_left, face_right = int(widest_row[0]), int(widest_row[-1])
        else:
            face_left, face_right = left, right

        fw = face_right - face_left + 1
        if fw < 8 or fh < 8:
            return None

        # --- dark pixel map inside the face (eyes / mouth are dark) ---
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        dark = ((gray < 110) & (mask > 0)).astype(np.uint8)
        face_dark = dark[face_top:face_bottom, face_left:face_right]

        # --- eyes: two largest dark clusters in the upper 75% ---
        eyes = self._find_blobs(face_dark, region=(0.0, 0.75))
        left_eye = right_eye = nose = mouth = None
        conf = 0.0

        if len(eyes) >= 2:
            e1, e2 = eyes[0], eyes[1]
            if e1[0] < e2[0]:
                left_eye, right_eye = e1, e2
            else:
                left_eye, right_eye = e2, e1
            # confidence grows with symmetric, horizontally-aligned eyes
            eye_dy = abs(left_eye[1] - right_eye[1]) / max(fh, 1)
            conf += max(0.0, 0.5 - eye_dy)
        elif len(eyes) == 1:
            # single eye visible → probably profile
            left_eye = eyes[0]
            conf += 0.2

        # --- nose: midpoint of eyes, shifted down 25% of face height ---
        if left_eye and right_eye:
            nose = (
                (left_eye[0] + right_eye[0]) / 2,
                (left_eye[1] + right_eye[1]) / 2 + fh * 0.25,
            )
        elif left_eye:
            nose = (left_eye[0], left_eye[1] + fh * 0.25)

        # --- mouth: largest dark blob in the lower half, below nose ---
        mouth_blobs = self._find_blobs(face_dark, region=(0.40, 1.0))
        if mouth_blobs and nose:
            nose_fx = nose[0]
            nose_fy = nose[1]
            for blob in mouth_blobs:
                bx, by, bw, bh = blob
                if by > nose_fy:     # mouth must be below the nose
                    mouth = (bx + bw / 2, by + bh / 2)
                    break

        # confidence bonus for finding all landmarks
        if left_eye and right_eye and mouth:
            conf += 0.3
        elif (left_eye or right_eye) and mouth:
            conf += 0.1

        # convert face-local coords to full-image coords
        def to_img(pt):
            if pt is None:
                return None
            return (pt[0] + face_left, pt[1] + face_top)

        return FaceDetection(
            bbox=(face_left, face_top, fw, fh),
            left_eye=to_img(left_eye),
            right_eye=to_img(right_eye),
            nose=to_img(nose),
            mouth=to_img(mouth),
            confidence=min(1.0, max(0.0, conf)),
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _find_blobs(
        dark: np.ndarray, region: tuple[float, float]
    ) -> list[tuple[float, float, float, float]]:
        """Find dark blobs in a vertical band of the face crop.

        Returns list of (cx, cy, w, h) sorted by area, largest first.
        """
        h = dark.shape[0]
        y0, y1 = int(h * region[0]), max(int(h * region[1]), 1)
        sub = dark[y0:y1, :]
        if sub.sum() == 0:
            return []

        n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, 8)
        blobs = []
        for i in range(1, n):
            x, y, w, hgt, area = stats[i]
            if area < 4:          # ignore specks
                continue
            if hgt > y1 - y0 - 2:  # ignore full-height bars (hair strands etc.)
                continue
            blobs.append((x + w / 2, y + y0 + hgt / 2, w, hgt))
        blobs.sort(key=lambda b: b[2] * b[3], reverse=True)
        return blobs
