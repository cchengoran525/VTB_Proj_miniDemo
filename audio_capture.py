"""Lightweight microphone capture — runs a callback stream, exposes smoothed RMS."""

from __future__ import annotations

import numpy as np
import sounddevice as sd


class AudioCapture:
    """Continuously samples the default microphone.

    Call ``.amplitude`` each frame to get a 0‑1-ish RMS value smoothed
    over the last few blocks.
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        block_size: int = 512,
        smoothing: float = 0.6,
    ) -> None:
        self._smooth = smoothing
        self._rms: float = 0.0
        self._stream = sd.InputStream(
            samplerate=sample_rate,
            blocksize=block_size,
            channels=1,
            callback=self._callback,
        )
        self._stream.start()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def amplitude(self) -> float:
        """Smoothed RMS amplitude, roughly 0–1 for normal speech."""
        return self._rms

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _callback(
        self, indata: np.ndarray, _frames: int, _time, status: int
    ) -> None:
        if status:
            return
        raw = float(np.sqrt(np.mean(indata**2)))
        self._rms = self._smooth * self._rms + (1.0 - self._smooth) * raw
