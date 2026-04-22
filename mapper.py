from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import config
from tracker import TrackingState


MOUTH_PRIORITY_WEIGHT = 100
EYE_PRIORITY_WEIGHT = 10


@dataclass(frozen=True)
class DiscreteState:
    mouth: str
    eye: str
    head: str

    @property
    def key(self) -> str:
        return f"{self.mouth}_{self.eye}_{self.head}"


class StateMapper:
    def __init__(self, frame_dir: Union[Path, str] = config.FRAME_DB_PATH) -> None:
        self.frame_dir = Path(frame_dir)
        self.state_frames: Dict[str, Path] = {}
        self.transition_frames: Dict[tuple[str, str], Path] = {}
        self._index_frames()

    def _index_frames(self) -> None:
        if not self.frame_dir.exists():
            raise FileNotFoundError(f"Frame database not found: {self.frame_dir}")

        for path in self.frame_dir.glob("*.png"):
            stem = path.stem
            if "_to_" in stem:
                source, target = stem.split("_to_", 1)
                self.transition_frames[(source, target)] = path
                continue

            parts = stem.split("_")
            if len(parts) != 3:
                continue
            self.state_frames[stem] = path

        if not self.state_frames:
            raise RuntimeError(f"No base frames found in {self.frame_dir}")

    def classify(self, tracking_state: TrackingState) -> DiscreteState:
        mouth = "open" if tracking_state.mouth_open >= config.MOUTH_OPEN_THRESHOLD else "closed"

        eye_openness = (tracking_state.left_eye_open + tracking_state.right_eye_open) / 2.0
        if eye_openness >= config.EYE_OPEN_THRESHOLD:
            eye = "open"
        elif eye_openness >= config.EYE_HALF_THRESHOLD:
            eye = "half"
        else:
            eye = "closed"

        head = "center"
        abs_yaw = abs(tracking_state.yaw)
        abs_pitch = abs(tracking_state.pitch)
        if abs_yaw >= config.HEAD_YAW_THRESHOLD and abs_yaw >= abs_pitch:
            head = "left" if tracking_state.yaw > 0 else "right"
        elif abs_pitch >= config.HEAD_PITCH_THRESHOLD:
            head = "down" if tracking_state.pitch > 0 else "up"

        return DiscreteState(mouth=mouth, eye=eye, head=head)

    def resolve_frame(self, discrete_state: DiscreteState) -> tuple[str, Path]:
        if discrete_state.key in self.state_frames:
            return discrete_state.key, self.state_frames[discrete_state.key]

        fallback_key = min(
            self.state_frames,
            key=lambda candidate: self._distance(discrete_state, self._state_from_key(candidate)),
        )
        return fallback_key, self.state_frames[fallback_key]

    def get_transition_frame(self, old_state_key: Optional[str], new_state_key: str) -> Optional[Path]:
        if not old_state_key:
            return None
        return self.transition_frames.get((old_state_key, new_state_key))

    def random_state_key(self) -> str:
        return next(iter(self.state_frames))

    def available_state_keys(self) -> Iterable[str]:
        return self.state_frames.keys()

    @staticmethod
    def _state_from_key(key: str) -> DiscreteState:
        mouth, eye, head = key.split("_")
        return DiscreteState(mouth=mouth, eye=eye, head=head)

    @staticmethod
    def _distance(source: DiscreteState, target: DiscreteState) -> int:
        distance = 0
        if source.mouth != target.mouth:
            distance += MOUTH_PRIORITY_WEIGHT
        if source.eye != target.eye:
            distance += EYE_PRIORITY_WEIGHT
        distance += StateMapper._head_distance(source.head, target.head)
        return distance

    @staticmethod
    def _head_distance(source: str, target: str) -> int:
        if source == target:
            return 0
        if "center" in {source, target}:
            return 1
        vertical = {"up", "down"}
        horizontal = {"left", "right"}
        if {source, target} <= vertical or {source, target} <= horizontal:
            return 2
        return 3
