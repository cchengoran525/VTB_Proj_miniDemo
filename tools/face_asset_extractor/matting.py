"""White-background matting → RGBA with alpha channel."""

from __future__ import annotations

import cv2
import numpy as np


def white_bg_matting(bgr: np.ndarray) -> np.ndarray:
    """
    Extract the character from a white background.

    Steps:
      1. Find near-white pixels (high luminance + low saturation).
      2. Flood-fill from the image borders — only background connected
         to the border is removed (keeps white costume/highlights inside).
      3. Morphological cleanup: open (kill specks) then close (fill holes).
      4. Feather the mask edge slightly for smooth alpha.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, w = bgr.shape[:2]

    # Near-white: bright AND desaturated
    value = hsv[:, :, 2]
    sat = hsv[:, :, 1]
    white_mask = ((value >= 235) & (sat <= 30)).astype(np.uint8)

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
