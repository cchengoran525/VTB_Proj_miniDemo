from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

import config

LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
LEFT_EYE_TOP = 159
LEFT_EYE_BOTTOM = 145

RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
RIGHT_EYE_TOP = 386
RIGHT_EYE_BOTTOM = 374

MOUTH_LEFT = 78
MOUTH_RIGHT = 308
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

NOSE_TIP = 1
FOREHEAD = 10
CHIN = 152
LEFT_CHEEK = 234
RIGHT_CHEEK = 454


@dataclass
class TrackingState:
    pitch: float
    yaw: float
    roll: float
    mouth_open: float
    left_eye_open: float
    right_eye_open: float
    face_found: bool
    face_confidence: float = 0.0   # 0 = side profile, 1 = straight ahead

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.pitch,
                self.yaw,
                self.roll,
                self.mouth_open,
                self.left_eye_open,
                self.right_eye_open,
            ],
            dtype=np.float32,
        )


class Kalman1D:
    def __init__(self, process_noise: float, measurement_noise: float) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.value = 0.0
        self.covariance = 1.0
        self.initialized = False

    def update(self, measurement: float) -> float:
        if not self.initialized:
            self.value = measurement
            self.initialized = True
            return self.value

        self.covariance += self.process_noise
        kalman_gain = self.covariance / (self.covariance + self.measurement_noise)
        self.value = self.value + kalman_gain * (measurement - self.value)
        self.covariance = (1.0 - kalman_gain) * self.covariance
        return self.value


class FaceTracker:
    def __init__(self, camera_index: int = config.CAMERA_INDEX) -> None:
        self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {camera_index}")

        base_options = BaseOptions(model_asset_path=str(config.FACE_LANDMARKER_MODEL))
        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_facial_transformation_matrixes=True,
        )
        self.face_landmarker = FaceLandmarker.create_from_options(options)

        base_process_noise = config.KALMAN_PROCESS_NOISE
        measurement_noise = config.KALMAN_MEASUREMENT_NOISE
        self.filters: Dict[str, Kalman1D] = {}
        for key in ("pitch", "yaw", "roll", "mouth", "left_eye", "right_eye"):
            pn = base_process_noise * config.KALMAN_RESPONSIVENESS.get(key, 1.0)
            self.filters[key] = Kalman1D(pn, measurement_noise)
        self.last_state = TrackingState(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            mouth_open=0.0,
            left_eye_open=1.0,
            right_eye_open=1.0,
            face_found=False,
            face_confidence=0.0,
        )

        # --- calibration state ---
        self._calibrated = False
        self._pitch_offset = 0.0
        self._yaw_offset = 0.0
        self._roll_offset = 0.0
        self._calib_samples: Dict[str, List[float]] = {
            "pitch": [], "yaw": [], "roll": [],
        }

        # Rolling history for occlusion detection (mouth / eye geometry)
        self._mouth_geo_history: List[float] = []
        self._eye_geo_history: List[float] = []

        # 用第一帧来"热身"模型，避免首帧卡顿
        ok, frame = self.cap.read()
        if ok:
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            self.face_landmarker.detect(mp_image)

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    def close(self) -> None:
        self.cap.release()
        self.face_landmarker.close()

    def read_state(self) -> TrackingState:
        ok, frame = self.cap.read()
        if not ok:
            return self.last_state

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.face_landmarker.detect(mp_image)

        if not result.face_landmarks:
            return TrackingState(
                pitch=self.last_state.pitch,
                yaw=self.last_state.yaw,
                roll=self.last_state.roll,
                mouth_open=self.last_state.mouth_open,
                left_eye_open=self.last_state.left_eye_open,
                right_eye_open=self.last_state.right_eye_open,
                face_found=False,
                face_confidence=0.0,
            )

        # face_landmarks[0] 直接是 NormalizedLandmark 列表
        landmarks = result.face_landmarks[0]
        coords = np.array([(pt.x, pt.y, pt.z) for pt in landmarks], dtype=np.float32)

        # --- head pose: matrix-based (primary) or heuristic (fallback) ---
        if (
            config.HEAD_POSE_SOURCE == "matrix"
            and result.facial_transformation_matrixes
        ):
            pitch_raw, yaw_raw, roll_raw = self._matrix_to_pose(
                result.facial_transformation_matrixes[0]
            )
        else:
            pitch_raw, yaw_raw, roll_raw = self._head_pose(coords)

        # --- calibration: zero out neutral pose ---
        if not self._calibrated:
            pitch, yaw, roll = self._collect_calibration(pitch_raw, yaw_raw, roll_raw)
        else:
            pitch = pitch_raw - self._pitch_offset
            yaw = yaw_raw - self._yaw_offset
            roll = roll_raw - self._roll_offset

        left_eye_raw = self._eye_ratio(coords, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
        right_eye_raw = self._eye_ratio(coords, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        mouth_raw = self._mouth_ratio(coords)

        # Face confidence: 4‑signal blend — rotation, geometry, eye quality,
        # and mouth‑geometry anomaly detection (catches hand‑over‑mouth etc.).
        #
        # (1) Rotation confidence — drops as head turns from calibrated neutral.
        rot_conf = max(0.0, 1.0 - math.hypot(pitch, yaw) / 0.52)
        #
        # (2) Aspect-ratio confidence — face bounding-box width/height ratio.
        xs = coords[:, 0]
        ys = coords[:, 1]
        face_w = float(np.ptp(xs))
        face_h = float(np.ptp(ys))
        ar = face_w / (face_h + 1e-6)
        ar_conf = float(np.clip((ar - 0.30) / 0.50, 0.0, 1.0))
        #
        # (3) Eye-width symmetry — frontal: left eye ≈ right eye in 2D.
        left_eye_w = float(np.linalg.norm(
            coords[LEFT_EYE_LEFT, :2] - coords[LEFT_EYE_RIGHT, :2]
        ))
        right_eye_w = float(np.linalg.norm(
            coords[RIGHT_EYE_LEFT, :2] - coords[RIGHT_EYE_RIGHT, :2]
        ))
        ew_min = min(left_eye_w, right_eye_w)
        ew_max = max(left_eye_w, right_eye_w)
        eye_width_sym = ew_min / (ew_max + 1e-6)
        eye_conf = float(np.clip((eye_width_sym - 0.45) / 0.50, 0.0, 1.0))
        #
        # (4) Mouth-geometry anomaly — extreme + abrupt mouth-ratio jump.
        #     (disabled for now — revisit with NO FACE handling)
        # (5) Eye-geometry anomaly — sudden width change → occlusion.
        #     (disabled for now — revisit with NO FACE handling)

        if config.OCCLUSION_DETECTION_ENABLED:
            is_extreme_mouth = mouth_raw > 0.30
            mouth_geo_conf = 1.0
            if is_extreme_mouth:
                anom = self._anomaly_conf(self._mouth_geo_history, mouth_raw)
                mouth_geo_conf = 0.0 if anom < 0.3 else 0.3

            eye_geo_ratio = (left_eye_w + right_eye_w) / (2.0 * face_w + 1e-6)
            eye_geo_conf = self._anomaly_conf(
                self._eye_geo_history, eye_geo_ratio
            )

            confidence = (
                0.20 * rot_conf + 0.20 * ar_conf
                + 0.20 * eye_conf + 0.20 * mouth_geo_conf
                + 0.20 * eye_geo_conf
            )
        else:
            confidence = (
                0.35 * rot_conf + 0.35 * ar_conf + 0.30 * eye_conf
            )

        state = TrackingState(
            pitch=self.filters["pitch"].update(pitch),
            yaw=self.filters["yaw"].update(yaw),
            roll=self.filters["roll"].update(roll),
            mouth_open=self.filters["mouth"].update(
                self._normalize(mouth_raw, *config.MOUTH_RATIO_RANGE)
            ),
            left_eye_open=self.filters["left_eye"].update(
                self._normalize(left_eye_raw, *config.EYE_RATIO_RANGE)
            ),
            right_eye_open=self.filters["right_eye"].update(
                self._normalize(right_eye_raw, *config.EYE_RATIO_RANGE)
            ),
            face_found=True,
            face_confidence=float(confidence),
        )
        self.last_state = state
        return state

    # ------------------------------------------------------------------
    #  Matrix-based head pose (accurate, uses MediaPipe's internal model)
    # ------------------------------------------------------------------

    def _matrix_to_pose(self, matrix_data) -> tuple[float, float, float]:
        """
        Extract yaw / pitch / roll from the 4×4 face transformation matrix.

        MediaPipe can return either a flat 16‑element list or a nested
        4×4 structure — normalise before extracting the rotation sub‑matrix.
        """
        data = np.asarray(matrix_data, dtype=np.float64).squeeze()

        if data.shape == (16,) or data.shape == (1, 16):
            flat = data.flatten()
            R = np.array(
                [[flat[0], flat[1], flat[2]],
                 [flat[4], flat[5], flat[6]],
                 [flat[8], flat[9], flat[10]]],
                dtype=np.float64,
            )
        elif data.shape == (4, 4):
            R = data[:3, :3].astype(np.float64)
        elif data.shape == (1, 4, 4):
            R = data[0, :3, :3].astype(np.float64)
        else:
            raise ValueError(f"Unexpected face matrix shape: {data.shape}")

        rvec, _ = cv2.Rodrigues(R)
        rx, ry, rz = rvec.flatten()
        return float(rx), float(ry), float(rz)

    def _collect_calibration(
        self, pitch: float, yaw: float, roll: float
    ) -> tuple[float, float, float]:
        """Accumulate neutral-pose samples; finalise when enough collected."""
        self._calib_samples["pitch"].append(pitch)
        self._calib_samples["yaw"].append(yaw)
        self._calib_samples["roll"].append(roll)

        if len(self._calib_samples["pitch"]) >= config.CALIBRATION_FRAMES:
            self._pitch_offset = float(np.mean(self._calib_samples["pitch"]))
            self._yaw_offset = float(np.mean(self._calib_samples["yaw"]))
            self._roll_offset = float(np.mean(self._calib_samples["roll"]))
            self._calibrated = True
            print(
                f"[Calibrated] pitch_offset={math.degrees(self._pitch_offset):.1f}°  "
                f"yaw_offset={math.degrees(self._yaw_offset):.1f}°  "
                f"roll_offset={math.degrees(self._roll_offset):.1f}°"
            )
            return (
                pitch - self._pitch_offset,
                yaw - self._yaw_offset,
                roll - self._roll_offset,
            )

        # During calibration, return raw values (will drift until calibrated)
        return pitch, yaw, roll

    # ----------------------------------------------------------------
    #  Anomaly detection helper (shared by mouth / eye occlusion)
    # ----------------------------------------------------------------

    @staticmethod
    def _anomaly_conf(history: List[float], value: float) -> float:
        """Confidence drop when *value* deviates from rolling median (MAD)."""
        history.append(value)
        if len(history) > 90:
            history.pop(0)
        if len(history) < 15:
            return 1.0
        arr = np.asarray(history, dtype=np.float64)
        median = float(np.median(arr))
        mad = float(np.median(np.abs(arr - median))) + 1e-6
        deviation = abs(value - median) / mad
        # deviation ≤ 1.5 MAD → 1.0    ≥ 8 MAD → 0.0
        return float(np.clip(1.0 - (deviation - 1.5) / 6.5, 0.0, 1.0))

    # ----------------------------------------------------------------
    #  Legacy heuristic head pose (fallback when matrix unavailable)
    # ----------------------------------------------------------------

    @staticmethod
    def _dist(points: np.ndarray, idx_a: int, idx_b: int) -> float:
        return float(np.linalg.norm(points[idx_a] - points[idx_b]))

    def _eye_ratio(
        self,
        points: np.ndarray,
        top_idx: int,
        bottom_idx: int,
        left_idx: int,
        right_idx: int,
    ) -> float:
        eye_height = self._dist(points, top_idx, bottom_idx)
        eye_width = self._dist(points, left_idx, right_idx)
        return eye_height / (eye_width + 1e-6)

    def _mouth_ratio(self, points: np.ndarray) -> float:
        mouth_height = self._dist(points, MOUTH_TOP, MOUTH_BOTTOM)
        mouth_width = self._dist(points, MOUTH_LEFT, MOUTH_RIGHT)
        return mouth_height / (mouth_width + 1e-6)

    def _head_pose(self, points: np.ndarray) -> tuple[float, float, float]:
        face_height = self._dist(points, FOREHEAD, CHIN)
        pitch = ((points[NOSE_TIP][1] - points[FOREHEAD][1]) / (face_height + 1e-6) - config.PITCH_CENTER)
        pitch *= config.PITCH_SCALE

        nose_left = self._dist(points, NOSE_TIP, LEFT_CHEEK)
        nose_right = self._dist(points, NOSE_TIP, RIGHT_CHEEK)
        yaw = ((nose_left - nose_right) / (nose_left + nose_right + 1e-6)) * config.YAW_SCALE

        roll = np.arctan2(
            points[RIGHT_EYE_RIGHT][1] - points[LEFT_EYE_LEFT][1],
            points[RIGHT_EYE_RIGHT][0] - points[LEFT_EYE_LEFT][0] + 1e-6,
        )
        roll = float(roll * config.ROLL_SCALE)
        return float(pitch), float(yaw), roll

    @staticmethod
    def _normalize(value: float, low: float, high: float) -> float:
        normalized = (value - low) / (high - low + 1e-6)
        return float(np.clip(normalized, 0.0, 1.0))
