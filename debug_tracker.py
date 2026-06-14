#!/usr/bin/env python3
"""Terminal-only tracker debug tool — no pygame, just raw numbers.

Shows calibrated pitch / yaw / roll (radians + degrees), mouth openness,
eye openness, and the resulting discrete classification.  Press Ctrl‑C to exit.
"""

from __future__ import annotations

import math
import os
import sys
import time

# Suppress SDL objc noise before anything imports cv2 / pygame
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

sys.path.insert(0, os.path.dirname(__file__))
from audio_capture import AudioCapture
import config
from mapper import StateMapper
from simulator import EyeSimulator, MouthSimulator

# ---------------------------------------------------------------------------
# Minimal tracker (extracts everything, dumps raw numbers)
# ---------------------------------------------------------------------------

LEFT_EYE_LEFT, LEFT_EYE_RIGHT = 33, 133
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 159, 145
RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT = 362, 263
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 386, 374
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
NOSE_TIP, FOREHEAD, CHIN = 1, 10, 152
LEFT_CHEEK, RIGHT_CHEEK = 234, 454


def matrix_to_pose(mat) -> tuple[float, float, float]:
    """
    Extract yaw / pitch / roll from a MediaPipe face transformation matrix.

    `mat` can be:
      - a 4×4 nested list  (MediaPipe 0.10.x actual format)
      - a flat 16-element list / tuple
      - an ndarray
    """
    data = np.asarray(mat, dtype=np.float64).squeeze()
    if data.shape == (16,) or data.shape == (1, 16):
        flat = data.flatten()
        R = np.array([[flat[0], flat[1], flat[2]],
                      [flat[4], flat[5], flat[6]],
                      [flat[8], flat[9], flat[10]]])
    elif data.shape == (4, 4):
        R = data[:3, :3]
    elif data.shape == (1, 4, 4):
        R = data[0, :3, :3]
    else:
        raise ValueError(f"Unexpected matrix shape: {data.shape}")

    rvec, _ = cv2.Rodrigues(R)
    rx, ry, rz = rvec.flatten()
    return float(rx), float(ry), float(rz)


def norm(value: float, low: float, high: float) -> float:
    return float(np.clip((value - low) / (high - low + 1e-6), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Main debug loop
# ---------------------------------------------------------------------------

def main() -> None:
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return

    base = BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL))
    opts = FaceLandmarkerOptions(
        base_options=base,
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_facial_transformation_matrixes=True,
    )
    fl = FaceLandmarker.create_from_options(opts)
    mapper = StateMapper()
    audio = AudioCapture()
    eye_sim = EyeSimulator()
    mouth_sim = MouthSimulator()

    # Anomaly-detection history (shared with tracker.py logic)
    _mouth_hist: list[float] = []
    _eye_hist: list[float] = []

    def _anomaly_conf(hist: list[float], value: float) -> float:
        hist.append(value)
        if len(hist) > 90:
            hist.pop(0)
        if len(hist) < 15:
            return 1.0
        arr = np.asarray(hist, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) + 1e-6
        deviation = abs(value - median) / mad
        return float(np.clip(1.0 - (deviation - 1.5) / 6.5, 0.0, 1.0))

    # Calibration
    pitch_offset = yaw_offset = roll_offset = 0.0
    calib_samples: list[float] = []
    calibrated = False

    print("=" * 70)
    print("  Face tracker debug – press Ctrl‑C to exit")
    print(f"  Calibrating … keep your head still for {config.CALIBRATION_FRAMES} frames")
    print("=" * 70)

    try:
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = fl.detect(mp_img)

            # ---------- face not found ----------
            if not result.face_landmarks:
                print(f"[{frame_idx:04d}] NO FACE")
                frame_idx += 1
                continue

            # ---------- landmarks ----------
            lm = result.face_landmarks[0]
            coords = np.array([(pt.x, pt.y, pt.z) for pt in lm], dtype=np.float32)

            # ---------- head pose ----------
            matrix = None
            if result.facial_transformation_matrixes:
                matrix = result.facial_transformation_matrixes[0]

            if matrix is not None:
                pitch_r, yaw_r, roll_r = matrix_to_pose(matrix)
            else:
                # heuristic fallback
                fh = float(np.linalg.norm(coords[FOREHEAD] - coords[CHIN]))
                pitch_r = (coords[NOSE_TIP][1] - coords[FOREHEAD][1]) / (fh + 1e-6) - config.PITCH_CENTER
                pitch_r *= config.PITCH_SCALE
                nl = float(np.linalg.norm(coords[NOSE_TIP] - coords[LEFT_CHEEK]))
                nr = float(np.linalg.norm(coords[NOSE_TIP] - coords[RIGHT_CHEEK]))
                yaw_r = (nl - nr) / (nl + nr + 1e-6) * config.YAW_SCALE
                roll_r = float(np.arctan2(
                    coords[RIGHT_EYE_RIGHT][1] - coords[LEFT_EYE_LEFT][1],
                    coords[RIGHT_EYE_RIGHT][0] - coords[LEFT_EYE_LEFT][0] + 1e-6,
                ) * config.ROLL_SCALE)

            # ---------- calibration ----------
            if not calibrated:
                calib_samples.append((pitch_r, yaw_r, roll_r))
                if len(calib_samples) >= config.CALIBRATION_FRAMES:
                    pitches = [s[0] for s in calib_samples]
                    yaws = [s[1] for s in calib_samples]
                    rolls = [s[2] for s in calib_samples]
                    pitch_offset = float(np.mean(pitches))
                    yaw_offset = float(np.mean(yaws))
                    roll_offset = float(np.mean(rolls))
                    calibrated = True
                    print("-" * 70)
                    print(f"  CALIBRATED:  pitch={math.degrees(pitch_offset):+.1f}°  "
                          f"yaw={math.degrees(yaw_offset):+.1f}°  "
                          f"roll={math.degrees(roll_offset):+.1f}°")
                    print("-" * 70)
                else:
                    # Just print progress every few frames
                    if len(calib_samples) % 10 == 0:
                        print(f"  … calibrating {len(calib_samples)} / {config.CALIBRATION_FRAMES}")
                    frame_idx += 1
                    continue

            pitch = pitch_r - pitch_offset
            yaw = yaw_r - yaw_offset
            roll = roll_r - roll_offset

            # ---------- eye / mouth ----------
            def eye_ratio(t, b, l, r):
                return float(np.linalg.norm(coords[t] - coords[b])) / (
                    float(np.linalg.norm(coords[l] - coords[r])) + 1e-6
                )

            le = eye_ratio(LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
            re = eye_ratio(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
            mouth_h = float(np.linalg.norm(coords[MOUTH_TOP] - coords[MOUTH_BOTTOM]))
            mouth_w = float(np.linalg.norm(coords[MOUTH_LEFT] - coords[MOUTH_RIGHT]))
            mouth_r = mouth_h / (mouth_w + 1e-6)

            le_n = norm(le, *config.MOUTH_RATIO_RANGE)
            re_n = norm(re, *config.MOUTH_RATIO_RANGE)
            mouth_n = norm(mouth_r, *config.MOUTH_RATIO_RANGE)

            # ---------- matrix shape debug (once) ----------
            if frame_idx == 0 and matrix is not None:
                mat_arr = np.asarray(matrix)
                print(f"  [debug] matrix type={type(matrix).__name__}  shape={mat_arr.shape}")

            # ---------- terminal output ----------
            yaw_deg = math.degrees(yaw)
            pitch_deg = math.degrees(pitch)
            roll_deg = math.degrees(roll)

            # Build a simple ASCII yaw bar
            bar_w = 13
            yaw_pos = int((yaw_deg / 30) * (bar_w // 2))  # assume ±30° range
            yaw_pos = max(-bar_w // 2, min(bar_w // 2, yaw_pos))
            bar = [" "] * bar_w
            bar[bar_w // 2] = "|"
            idx = bar_w // 2 + yaw_pos
            if 0 <= idx < bar_w:
                bar[idx] = "·" if yaw_pos == 0 else ("<" if yaw_deg < 0 else ">")
            yaw_bar = "".join(bar)

            # Pitch bar
            bar_p = [" "] * bar_w
            bar_p[bar_w // 2] = "—"
            pitch_pos = int((-pitch_deg / 20) * (bar_w // 2))
            pitch_pos = max(-bar_w // 2, min(bar_w // 2, pitch_pos))
            idx_p = bar_w // 2 + pitch_pos
            if 0 <= idx_p < bar_w:
                bar_p[idx_p] = "·" if pitch_pos == 0 else ("v" if pitch_deg > 0 else "^")
            pitch_bar = "".join(bar_p)

            # ---------- face confidence & dual-mode ----------
            # (1) Rotation, (2) aspect-ratio, (3) eye-WIDTH symmetry,
            # (4) mouth-geometry anomaly, (5) eye-geometry anomaly
            rot_conf = max(0.0, 1.0 - math.hypot(yaw, pitch) / 0.52)
            xs = coords[:, 0]; ys = coords[:, 1]
            face_w = float(np.ptp(xs)); face_h = float(np.ptp(ys))
            ar = face_w / (face_h + 1e-6)
            ar_conf = float(np.clip((ar - 0.30) / 0.50, 0.0, 1.0))
            le_w = float(np.linalg.norm(
                coords[LEFT_EYE_LEFT, :2] - coords[LEFT_EYE_RIGHT, :2]))
            re_w = float(np.linalg.norm(
                coords[RIGHT_EYE_LEFT, :2] - coords[RIGHT_EYE_RIGHT, :2]))
            ew_min = min(le_w, re_w); ew_max = max(le_w, re_w)
            eye_width_sym = ew_min / (ew_max + 1e-6)
            eye_conf = float(np.clip((eye_width_sym - 0.45) / 0.50, 0.0, 1.0))
            if config.OCCLUSION_DETECTION_ENABLED:
                is_extreme_mouth = mouth_r > 0.30
                mouth_geo_conf = 1.0
                if is_extreme_mouth:
                    anom = _anomaly_conf(_mouth_hist, mouth_r)
                    mouth_geo_conf = 0.0 if anom < 0.3 else 0.3
                eye_geo_conf = _anomaly_conf(_eye_hist, (le_w + re_w) / (2 * face_w + 1e-6))
                confidence = (
                    0.20 * rot_conf + 0.20 * ar_conf + 0.20 * eye_conf
                    + 0.20 * mouth_geo_conf + 0.20 * eye_geo_conf
                )
            else:
                mouth_geo_conf = 1.0
                eye_geo_conf = 1.0
                confidence = (
                    0.35 * rot_conf + 0.35 * ar_conf + 0.3 * eye_conf
                )
            sim_active = confidence < config.FACE_CONFIDENCE_THRESHOLD

            if sim_active:
                # Simulated eye / mouth
                eye_sim_state = eye_sim.update(1.0 / 30.0)
                mouth_sim_state = mouth_sim.update(1.0 / 30.0, audio.amplitude)
                eye_avg = {"open": 0.95, "half": 0.55, "closed": 0.15}[eye_sim_state]
                mouth_n = {"closed": 0.05, "half": 0.30, "open": 0.80}[mouth_sim_state]
                mode_str = "[SIM]"
            else:
                # Camera-based eye / mouth
                eye_avg = (le_n + re_n) / 2.0
                mouth_n_use = mouth_n  # already computed above
                eye_sim.reset()
                mouth_sim.reset()
                mode_str = "[CAM]"

            # Discrete classification
            mouth_label = (
                "OPEN " if mouth_n >= config.MOUTH_OPEN_THRESHOLD
                else "half " if mouth_n >= config.MOUTH_HALF_THRESHOLD
                else "closed"
            )
            eye_avg_use = (le_n + re_n) / 2.0 if not sim_active else eye_avg
            if eye_avg_use >= config.EYE_OPEN_THRESHOLD:
                eye_label = "OPEN "
            elif eye_avg_use >= config.EYE_HALF_THRESHOLD:
                eye_label = "half "
            else:
                eye_label = "closed"

            # Head direction (grid or legacy, no debounce in debug mode)
            if config.HEAD_GRID_ENABLED:
                r = config.HEAD_GRID_RADIUS
                yi = int(round(yaw / config.HEAD_GRID_YAW_STEP))
                yi = max(-r, min(r, yi))
                pi = int(round(pitch / config.HEAD_GRID_PITCH_STEP))
                pi = max(-r, min(r, pi))
                if config.HEAD_ROLL_INNER_ONLY and (abs(yi) > 1 or abs(pi) > 1):
                    ri = 0
                else:
                    ri = int(round(roll / config.HEAD_GRID_ROLL_STEP))
                    ri = max(-1, min(1, ri))
                parts = []
                if yi > 0: parts.append(f'R{yi}')
                elif yi < 0: parts.append(f'L{-yi}')
                if pi > 0: parts.append(f'D{pi}')
                elif pi < 0: parts.append(f'U{-pi}')
                if ri > 0: parts.append('WR')
                elif ri < 0: parts.append('WL')
                head_label = '_'.join(parts).ljust(8) if parts else 'center '.ljust(8)
            else:
                if abs(yaw) >= config.HEAD_YAW_THRESHOLD and abs(yaw) >= abs(pitch):
                    head_label = "RIGHT" if yaw > 0 else "LEFT "
                elif abs(pitch) >= config.HEAD_PITCH_THRESHOLD:
                    head_label = "DOWN" if pitch > 0 else "UP  "
                else:
                    head_label = "center"

            print(
                f"[{frame_idx:04d}] {mode_str} c={confidence:.2f} "
                f"(r={rot_conf:.2f} a={ar_conf:.2f} e={eye_conf:.2f} m={mouth_geo_conf:.2f} ge={eye_geo_conf:.2f}) "
                f"mic={audio.amplitude:.3f} "
                f"yaw{yaw_bar} {yaw_deg:+5.1f}°  "
                f"pitch{pitch_bar} {pitch_deg:+5.1f}°  "
                f"roll={roll_deg:+5.1f}°  |  "
                f"mouth={mouth_label}({mouth_n:.2f} r={mouth_r:.3f})  "
                f"eye={eye_label}({eye_avg_use:.2f} r={le:.3f}/{re:.3f})  |  "
                f"→ {mouth_label.strip()}_{eye_label.strip()}_{head_label.strip()}"
            )

            frame_idx += 1

    except KeyboardInterrupt:
        print("\nDone.")
    finally:
        cap.release()
        fl.close()
        audio.close()


if __name__ == "__main__":
    main()
