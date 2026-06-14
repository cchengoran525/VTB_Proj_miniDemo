"""Dual-mode fallback simulators for eye blinking and mouth movement.

When face confidence drops (e.g. looking far to the side), camera-based
eye / mouth tracking becomes unreliable.  These simulators kick in to keep
the character alive instead of freezing or showing garbage data.

Eye   – natural blink timer (random interval + duration, double blinks)
Mouth – placeholder (random open/close); audio‑drive hook is prepared.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
#  Eye blinking simulator
# ---------------------------------------------------------------------------

@dataclass
class EyeSimulator:
    """Stateful blinking simulator.  Call ``.update()`` every frame."""

    # Tunables
    blink_interval_range: tuple[float, float] = (2.0, 6.0)   # seconds between blinks
    blink_duration_frames: tuple[int, int] = (3, 10)          # frames per blink
    double_blink_chance: float = 0.15                         # probability of double blink

    # Internal state
    _state: str = "open"                # open | half | closed
    _timer: float = 0.0
    _blink_remaining: int = 0
    _blink_phase: int = 0               # 0=half-close, 1=closed, 2=half-open
    _in_double: bool = False

    def __post_init__(self) -> None:
        self._timer = random.uniform(*self.blink_interval_range)

    def update(self, dt: float) -> str:
        """Advance one frame and return the eye state ("open" | "half" | "closed")."""
        if self._blink_remaining > 0:
            return self._tick_blink()
        return self._tick_idle(dt)

    # ---- internal ----

    def _tick_idle(self, dt: float) -> str:
        self._timer -= dt
        if self._timer <= 0.0:
            self._start_blink()
            return self._tick_blink()
        return self._state

    def _tick_blink(self) -> str:
        self._blink_remaining -= 1

        # Blink sequence: half → closed → half → open
        total = self._blink_duration
        if self._blink_duration == 0:
            self._finish_blink()
            return self._state

        progress = 1.0 - (self._blink_remaining / max(1, total))

        if progress < 0.15:
            self._state = "half"
        elif progress < 0.6:
            self._state = "closed"
        elif progress < 0.85:
            self._state = "half"
        else:
            self._state = "open"

        if self._blink_remaining <= 0:
            self._finish_blink()
        return self._state

    def _start_blink(self) -> None:
        dur = random.randint(*self.blink_duration_frames)
        if not self._in_double and random.random() < self.double_blink_chance:
            self._in_double = True
            dur += random.randint(2, 5)  # short gap + second blink
        else:
            self._in_double = False
        self._blink_remaining = dur
        self._blink_duration = dur
        self._state = "half"

    def _finish_blink(self) -> None:
        self._state = "open"
        base = random.uniform(*self.blink_interval_range)
        if self._in_double:
            base *= 0.4  # shorter wait after double blink
        self._timer = base

    def reset(self) -> None:
        """Restart the idle timer (use when switching back from camera mode)."""
        self._state = "open"
        self._blink_remaining = 0
        self._timer = random.uniform(*self.blink_interval_range)


# ---------------------------------------------------------------------------
#  Mouth simulator — audio‑driven  (falls back to random when no audio)
# ---------------------------------------------------------------------------

@dataclass
class MouthSimulator:
    """Audio‑driven mouth movement — 3 states with hysteresis.

    Pass ``audio_amplitude`` (0‑1) to ``update()``.  Falls back to random
    idle movement when amplitude is ``None``.
    """

    # Amplitude thresholds (0‑1 RMS)
    half_threshold: float = 0.015      # above → talking (half)
    open_threshold: float = 0.040      # above → wide open
    close_threshold: float = 0.008     # below → fully closed
    hold_frames: int = 3               # minimum frames to hold after change

    _state: str = "closed"
    _hold: int = 0

    def update(self, dt: float, audio_amplitude: float | None = None) -> str:
        """Advance one frame and return ``"closed"``, ``"half"``, or ``"open"``."""
        if audio_amplitude is not None:
            return self._update_audio(audio_amplitude)
        return self._update_random(dt)

    # ---- audio-driven ---------------------------------------------------

    def _update_audio(self, amp: float) -> str:
        if self._hold > 0:
            self._hold -= 1
            return self._state

        if self._state == "closed":
            if amp >= self.open_threshold:
                self._state = "open"
                self._hold = self.hold_frames
            elif amp >= self.half_threshold:
                self._state = "half"
                self._hold = self.hold_frames
        elif self._state == "half":
            if amp >= self.open_threshold:
                self._state = "open"
                self._hold = self.hold_frames
            elif amp < self.close_threshold:
                self._state = "closed"
                self._hold = self.hold_frames
        else:  # open
            if amp < self.close_threshold:
                self._state = "closed"
                self._hold = self.hold_frames
            elif amp < self.half_threshold:
                self._state = "half"
                self._hold = self.hold_frames
        return self._state

    # ---- random (fallback / placeholder) ---------------------------------

    _random_open_chance: float = 0.03
    _random_open_frames: tuple[int, int] = (5, 15)
    _random_closed_frames: tuple[int, int] = (30, 90)
    _random_remaining: int = 0

    def _update_random(self, dt: float) -> str:
        if self._random_remaining > 0:
            self._random_remaining -= 1
            if self._random_remaining <= 0:
                self._state = (
                    "closed" if self._state == "open" else "open"
                )
                self._random_remaining = random.randint(
                    *(
                        self._random_closed_frames
                        if self._state == "closed"
                        else self._random_open_frames
                    )
                )
            return self._state
        if random.random() < self._random_open_chance:
            self._state = "open"
            self._random_remaining = random.randint(*self._random_open_frames)
        return self._state

    def reset(self) -> None:
        self._state = "closed"
        self._hold = 0
        self._random_remaining = 0
