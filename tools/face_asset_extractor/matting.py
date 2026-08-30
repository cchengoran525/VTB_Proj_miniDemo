"""White-background matting → RGBA with alpha channel."""

from __future__ import annotations

import cv2
import numpy as np


def white_bg_matting(bgr: np.ndarray) -> np.ndarray:
    """
    Extract the character from a white background.

    Steps:
      1. Sample border pixels to learn the background's brightness/saturation,
         then mark near-white pixels with an adaptive threshold.  The band is
         tight enough that light-coloured clothing (usually 10+ value units
         below the background) stays in the character mask.
      2. Flood-fill from the image borders — only background connected
         to the border is removed (keeps white costume/highlights inside).
      3. Morphological cleanup: open (kill specks) then close (fill holes).
      4. Feather the mask edge slightly for smooth alpha.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    # Learn background stats from the border ring
    border = np.concatenate([
        hsv[:5, :, :].reshape(-1, 3),
        hsv[-5:, :, :].reshape(-1, 3),
        hsv[:, :5, :].reshape(-1, 3),
        hsv[:, -5:, :].reshape(-1, 3),
    ]).astype(np.float32)
    bg_value = float(np.median(border[:, 2]))
    bg_sat = float(np.median(border[:, 1]))

    # Near-white: within the adaptive band of the learned background.
    value = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    v_lo = min(245.0, bg_value - 6.0)
    s_hi = max(12.0, bg_sat + 10.0)
    white_mask = ((value >= v_lo) & (sat <= s_hi)).astype(np.uint8)

    # Flood fill from borders (4-connected)
    fill = white_mask.copy()
    stack = []
    for x in range(w):
        if fill[0, x]:
            stack.append((0, x))
        if fill[h - 1, x]:
            stack.append((h - 1, x))
    for y in range(h):
        if fill[y, 0]:
            stack.append((y, 0))
        if fill[y, w - 1]:
            stack.append((y, w - 1))

    bg = np.zeros_like(fill)
    while stack:
        y, x = stack.pop()
        if bg[y, x] or not fill[y, x]:
            continue
        bg[y, x] = 1
        if y > 0:
            stack.append((y - 1, x))
        if y < h - 1:
            stack.append((y + 1, x))
        if x > 0:
            stack.append((y, x - 1))
        if x < w - 1:
            stack.append((y, x + 1))

    char_mask = 1 - bg  # 1 = character

    # Morphological cleanup
    kernel = np.ones((3, 3), np.uint8)
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_OPEN, kernel)
    char_mask = cv2.morphologyEx(char_mask, cv2.MORPH_CLOSE, kernel)

    # Feather edge (1px gaussian) for softer alpha
    alpha = cv2.GaussianBlur(char_mask * 255, (3, 3), 0)

    rgba = cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba
