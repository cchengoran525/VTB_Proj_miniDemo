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
#  Mouth simulator (placeholder – audio hook goes here later)
# ---------------------------------------------------------------------------

@dataclass
class MouthSimulator:
    """Simple mouth movement placeholder.

    TODO: replace ``_should_move`` with audio‑amplitude‑based trigger.
    """

    open_duration_frames: tuple[int, int] = (5, 15)
    closed_duration_frames: tuple[int, int] = (30, 90)
    move_chance: float = 0.03  # per-frame probability of starting a mouth cycle

    _state: str = "closed"
    _remaining: int = 0

    def update(self, dt: float) -> str:
        """Advance one frame and return "open" or "closed"."""
        if self._remaining > 0:
            self._remaining -= 1
            if self._remaining <= 0:
                if self._state == "open":
                    self._state = "closed"
                    self._remaining = random.randint(*self.closed_duration_frames)
                else:
                    self._state = "open"
                    self._remaining = random.randint(*self.open_duration_frames)
            return self._state

        # Idle (closed) — randomly decide to open
        if random.random() < self.move_chance:
            self._state = "open"
            self._remaining = random.randint(*self.open_duration_frames)
            return self._state

        return self._state

    def reset(self) -> None:
        self._state = "closed"
        self._remaining = 0
