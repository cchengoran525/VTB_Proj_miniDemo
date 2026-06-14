#!/usr/bin/env python3
"""Generate text‑labelled placeholder PNGs for the full 5×5+roll+variant frame set.

Usage:  python generate_placeholders.py          # → frames_placeholder/
        python generate_placeholders.py --inner-only  # → only inner 3×3 (faster test)
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# State space
# ---------------------------------------------------------------------------

MOUTHS = ["closed", "half", "open"]
EYES = ["closed", "half", "open"]

# 5×5 yaw / pitch grid
GRID_RADIUS = 2
YAW_LABELS = {-2: "L2", -1: "L1", 0: "", 1: "R1", 2: "R2"}
PITCH_LABELS = {-2: "U2", -1: "U1", 0: "", 1: "D1", 2: "D2"}

# Roll (only on inner 3×3)
ROLL_INNER = {-1: "WL", 0: "", 1: "WR"}

# Micro‑variants per head pose
VARIANTS = range(1, 6)


def head_poses(inner_only: bool = False):
    for yi in range(-GRID_RADIUS, GRID_RADIUS + 1):
        for pi in range(-GRID_RADIUS, GRID_RADIUS + 1):
            inner = abs(yi) <= 1 and abs(pi) <= 1
            if inner_only and not inner:
                continue
            rolls = ROLL_INNER if inner else {0: ""}
            for ri, rl in rolls.items():
                parts = [p for p in [
                    YAW_LABELS[yi], PITCH_LABELS[pi], rl
                ] if p]
                yield "_".join(parts) if parts else "center"


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

SIZE = 256, 256
BG = (40, 42, 54)
FG = (200, 200, 210)
ACCENT = {
    "closed": (120, 140, 160),
    "half":   (200, 180, 80),
    "open":   (240, 100, 80),
}


def draw_frame(mouth: str, eye: str, head: str, variant: int) -> Image.Image:
    img = Image.new("RGB", SIZE, BG)
    d = ImageDraw.Draw(img)

    # Colour accent based on mouth state
    accent = ACCENT.get(mouth, FG)

    # Top bar
    d.rectangle([(0, 0), (SIZE[0] - 1, 4)], fill=accent)

    # Variant badge
    d.text((SIZE[0] - 30, 10), f"v{variant}", fill=accent)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_sm = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    except OSError:
        font = ImageFont.load_default()
        font_sm = font

    # Labels
    y = 30
    for label, colour in [
        (f"mouth: {mouth}", ACCENT[mouth]),
        (f"eye:   {eye}", (160, 200, 160) if eye == "open" else
                          (200, 180, 80) if eye == "half" else FG),
        (f"head:  {head}", FG),
    ]:
        d.text((12, y), label, fill=colour, font=font)
        y += 22

    # Separator
    y += 6
    d.line([(12, y), (SIZE[0] - 12, y)], fill=(80, 80, 90))

    # Mouth visual
    y += 10
    mouth_h = {"closed": 2, "half": 12, "open": 28}[mouth]
    mx, my = SIZE[0] // 2, y + 20
    d.ellipse(
        [(mx - 25, my - mouth_h // 2), (mx + 25, my + mouth_h // 2)],
        outline=(220, 180, 80), width=2,
    )

    # Eye visual
    eye_h = {"closed": 1, "half": 6, "open": 14}[eye]
    for ex in (mx - 35, mx + 35):
        d.ellipse(
            [(ex - 12, my - 12 - eye_h // 2), (ex + 12, my - 12 + eye_h // 2)],
            outline=(160, 200, 160), width=2,
        )

    return img


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner-only", action="store_true",
                    help="Only generate inner 3×3 poses (faster)")
    ap.add_argument("--out", default="frames_placeholder",
                    help="Output directory (default: frames_placeholder)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    heads = list(head_poses(args.inner_only))
    total = len(heads) * len(MOUTHS) * len(EYES) * len(VARIANTS)
    print(f"Head poses: {len(heads)}")
    print(f"Total frames: {total}")
    print(f"Output: {out.resolve()}")
    print()

    count = 0
    for head in heads:
        for eye in EYES:
            for mouth in MOUTHS:
                for v in VARIANTS:
                    if v == 1:
                        name = f"{mouth}_{eye}_{head}.png"
                    else:
                        name = f"{mouth}_{eye}_{head}_v{v}.png"
                    img = draw_frame(mouth, eye, head, v)
                    img.save(out / name)
                    count += 1
                    if count % 500 == 0:
                        print(f"  … {count}/{total}")

    print(f"Done — {count} frames written to {out.name}/")


if __name__ == "__main__":
    main()
