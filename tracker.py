from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import mediapipe as mp
import numpy as np

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

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        process_noise = config.KALMAN_PROCESS_NOISE
        measurement_noise = config.KALMAN_MEASUREMENT_NOISE
        self.filters: Dict[str, Kalman1D] = {
            key: Kalman1D(process_noise, measurement_noise)
            for key in ("pitch", "yaw", "roll", "mouth", "left_eye", "right_eye")
        }
        self.last_state = TrackingState(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            mouth_open=0.0,
            left_eye_open=1.0,
            right_eye_open=1.0,
            face_found=False,
        )

    def close(self) -> None:
        self.cap.release()
        self.face_mesh.close()

    def read_state(self) -> TrackingState:
        ok, frame = self.cap.read()
        if not ok:
            return self.last_state

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return TrackingState(
                pitch=self.last_state.pitch,
                yaw=self.last_state.yaw,
                roll=self.last_state.roll,
                mouth_open=self.last_state.mouth_open,
                left_eye_open=self.last_state.left_eye_open,
                right_eye_open=self.last_state.right_eye_open,
                face_found=False,
            )

        lm = result.multi_face_landmarks[0].landmark
        coords = np.array([(point.x, point.y, point.z) for point in lm], dtype=np.float32)

        left_eye_raw = self._eye_ratio(coords, LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_LEFT, LEFT_EYE_RIGHT)
        right_eye_raw = self._eye_ratio(coords, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_LEFT, RIGHT_EYE_RIGHT)
        mouth_raw = self._mouth_ratio(coords)
        pitch_raw, yaw_raw, roll_raw = self._head_pose(coords)

        state = TrackingState(
            pitch=self.filters["pitch"].update(pitch_raw),
            yaw=self.filters["yaw"].update(yaw_raw),
            roll=self.filters["roll"].update(roll_raw),
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
        )
        self.last_state = state
        return state

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
