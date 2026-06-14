from __future__ import annotations

import time
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

        # Hysteresis state for eye and mouth (Schmitt trigger)
        self._last_eye = "open"
        self._last_mouth = "closed"

        # Head direction debounce state
        self._last_head = "center"
        self._head_locked_until = 0.0

    def classify(self, tracking_state: TrackingState) -> DiscreteState:
        # --- mouth: 2-state Schmitt trigger ---
        mouth = self._classify_mouth(tracking_state.mouth_open)

        # --- eye: 3-state Schmitt trigger ---
        eye_openness = (tracking_state.left_eye_open + tracking_state.right_eye_open) / 2.0
        eye = self._classify_eye(eye_openness)

        # --- head classification ---
        if config.HEAD_GRID_ENABLED:
            head = self._classify_head_grid(tracking_state.yaw, tracking_state.pitch, tracking_state.roll)
        else:
            head = self._classify_head_5way(tracking_state.yaw, tracking_state.pitch)

        return DiscreteState(mouth=mouth, eye=eye, head=head)

    # ------------------------------------------------------------------
    #  Head: legacy 5-direction
    # ------------------------------------------------------------------

    def _classify_head_5way(self, yaw: float, pitch: float) -> str:
        abs_yaw = abs(yaw)
        abs_pitch = abs(pitch)

        raw_head = "center"
        if abs_yaw >= config.HEAD_YAW_THRESHOLD and abs_yaw >= abs_pitch:
            raw_head = "right" if yaw > 0 else "left"
        elif abs_pitch >= config.HEAD_PITCH_THRESHOLD:
            raw_head = "down" if pitch > 0 else "up"

        now = time.time()
        if raw_head != self._last_head and now >= self._head_locked_until:
            self._last_head = raw_head
            if raw_head != "center":
                self._head_locked_until = now + config.HEAD_LOCK_DURATION

        return self._last_head

    # ------------------------------------------------------------------
    #  Head: N×N grid + roll  (yaw / pitch / roll quantised)
    # ------------------------------------------------------------------

    def _classify_head_grid(self, yaw: float, pitch: float, roll: float = 0.0) -> str:
        r = config.HEAD_GRID_RADIUS

        # Yaw
        yi = int(round(yaw / config.HEAD_GRID_YAW_STEP))
        yi = max(-r, min(r, yi))
        # Pitch
        pi = int(round(pitch / config.HEAD_GRID_PITCH_STEP))
        pi = max(-r, min(r, pi))
        # Roll — always 3 levels (WL / — / WR)
        ri = int(round(roll / config.HEAD_GRID_ROLL_STEP))
        ri = max(-1, min(1, ri))

        parts: list[str] = []
        if yi > 0:
            parts.append(f"R{yi}")
        elif yi < 0:
            parts.append(f"L{-yi}")
        if pi > 0:
            parts.append(f"D{pi}")
        elif pi < 0:
            parts.append(f"U{-pi}")
        if ri > 0:
            parts.append("WR")
        elif ri < 0:
            parts.append("WL")

        raw_head = "_".join(parts) if parts else "center"

        now = time.time()
        if raw_head != self._last_head and now >= self._head_locked_until:
            self._last_head = raw_head
            if raw_head != "center":
                self._head_locked_until = now + config.HEAD_LOCK_DURATION

        return self._last_head

    def _classify_mouth(self, mouth_open: float) -> str:
        """2-state Schmitt trigger: prevents jitter around MOUTH_OPEN_THRESHOLD."""
        h = config.MOUTH_HYSTERESIS  # 阈值需要跨越的余量
        t = config.MOUTH_OPEN_THRESHOLD

        if self._last_mouth == "open":
            if mouth_open < t - h:
                self._last_mouth = "closed"
        else:  # closed
            if mouth_open > t + h:
                self._last_mouth = "open"

        return self._last_mouth

    def _classify_eye(self, eye_openness: float) -> str:
        """3-state Schmitt trigger: prevents jitter around both eye thresholds."""
        h = config.EYE_HYSTERESIS
        t_open = config.EYE_OPEN_THRESHOLD
        t_half = config.EYE_HALF_THRESHOLD

        if self._last_eye == "open":
            if eye_openness < t_open - h:
                self._last_eye = "half" if eye_openness >= t_half else "closed"
        elif self._last_eye == "closed":
            if eye_openness > t_half + h:
                self._last_eye = "half" if eye_openness < t_open else "open"
        else:  # half
            if eye_openness > t_open + h:
                self._last_eye = "open"
            elif eye_openness < t_half - h:
                self._last_eye = "closed"

        return self._last_eye

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
        # Keys are "mouth_eye_head" — head may contain underscores for grid
        # coords (e.g. "open_open_L2_D1"). Split carefully.
        parts = key.split("_")
        if len(parts) == 3:
            mouth, eye, head = parts
        elif len(parts) == 4:
            mouth, eye, head = parts[0], parts[1], f"{parts[2]}_{parts[3]}"
        else:
            mouth, eye, head = parts[0], parts[1], "_".join(parts[2:])
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

        def _to_base_dir(h: str) -> str:
            """Map any head label (legacy or grid) to its base direction."""
            if h == "center":
                return "center"
            if "L" in h:
                return "left"
            if "R" in h:
                return "right"
            if "U" in h:
                return "up"
            if "D" in h:
                return "down"
            return h  # unknown, fall through

        s_dir = _to_base_dir(source)
        t_dir = _to_base_dir(target)

        if s_dir == t_dir:
            return 0
        if "center" in {s_dir, t_dir}:
            return 1
        vertical = {"up", "down"}
        horizontal = {"left", "right"}
        if {s_dir, t_dir} <= vertical or {s_dir, t_dir} <= horizontal:
            return 2
        return 3
