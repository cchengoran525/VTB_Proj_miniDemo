"""V1 dual‑window display — camera + debug (left), character (right)."""

from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np
import pygame

import config


class FrameDisplay:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("VTB Mini Demo – V1")
        flags = pygame.FULLSCREEN if config.FULLSCREEN else 0
        size = (0, 0) if config.FULLSCREEN else config.WINDOW_SIZE
        self.screen = pygame.display.set_mode(size, flags, vsync=1)
        self.clock = pygame.time.Clock()
        self.surface_cache: dict[Path, pygame.Surface] = {}
        self.current_frame_path: Optional[Path] = None
        self.current_state_key: Optional[str] = None
        self.pending_frame_path: Optional[Path] = None
        self.pending_state_key: Optional[str] = None
        self.next_random_cut_at = self._next_random_deadline()

        # Debug font
        self._font = pygame.font.SysFont("monospace", 12)

    # ------------------------------------------------------------------
    #  Public API
    # ------------------------------------------------------------------

    def close(self) -> None:
        pygame.quit()

    def should_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key in (
                pygame.K_ESCAPE, pygame.K_q
            ):
                return True
        return False

    def tick(self) -> None:
        self.clock.tick(config.TARGET_FPS)

    def choose_next_frame(
        self,
        target_state_key: str,
        target_frame_path: Path,
        transition_frame_path: Optional[Path],
        available_state_keys: Iterable[str],
        get_frame_path,
    ) -> tuple[Path, str, str]:
        now = time.time()

        if now >= self.next_random_cut_at:
            state_key = random.choice(list(available_state_keys))
            frame_path = get_frame_path(state_key)
            self.next_random_cut_at = self._next_random_deadline(now)
            return frame_path, state_key, "random-hard-cut"

        if (
            self.pending_frame_path is not None
            and self.pending_state_key is not None
        ):
            frame_path = self.pending_frame_path
            state_key = self.pending_state_key
            self.pending_frame_path = None
            self.pending_state_key = None
            return frame_path, state_key, "transition-final"

        if (
            self.current_state_key != target_state_key
            and transition_frame_path is not None
        ):
            self.pending_frame_path = target_frame_path
            self.pending_state_key = target_state_key
            return (
                transition_frame_path,
                self.current_state_key or target_state_key,
                "transition",
            )

        return target_frame_path, target_state_key, "direct-hard-cut"

    def render(
        self,
        frame_path: Path,
        state_key: Optional[str],
        theme: str = "default",
        debug: Optional[dict] = None,
    ) -> None:
        """Draw dual‑window: camera+debug left, character right."""
        screen_w, screen_h = self.screen.get_size()
        self.screen.fill((12, 12, 16))

        # Panel widths
        left_w = screen_w * 2 // 5   # 40 %
        right_w = screen_w - left_w   # 60 %

        # ── Left panel: camera preview + debug ──
        self._draw_left_panel(0, 0, left_w, screen_h, debug)

        # Divider
        pygame.draw.line(
            self.screen, (60, 60, 80), (left_w, 0), (left_w, screen_h), 2
        )

        # ── Right panel: character ──
        self._draw_right_panel(left_w, 0, right_w, screen_h,
                               frame_path, state_key, theme, debug)

        pygame.display.flip()
        self.current_frame_path = frame_path
        self.current_state_key = state_key

    # ------------------------------------------------------------------
    #  Left panel: camera + debug overlay
    # ------------------------------------------------------------------

    def _draw_left_panel(
        self, x: int, y: int, w: int, h: int,
        debug: Optional[dict] = None,
    ) -> None:
        # Background
        pygame.draw.rect(self.screen, (20, 22, 30), (x, y, w, h))

        # Camera frame
        cam_frame = None
        if debug:
            cam_frame = debug.get("camera_frame")

        if cam_frame is not None:
            # Convert BGR → RGB and scale to panel
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            # Draw face landmarks if available
            if debug and debug.get("landmarks") is not None:
                lm = debug["landmarks"]
                h_frame, w_frame = cam_frame.shape[:2]
                for idx in [1, 33, 133, 362, 263, 61, 291, 13, 14, 78, 308]:
                    if idx < len(lm):
                        px, py = int(lm[idx][0] * w_frame), int(lm[idx][1] * h_frame)
                        cv2.circle(rgb, (px, py), 2, (0, 255, 128), -1)

            surf = pygame.image.frombuffer(
                rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB"
            )
            # Scale to fit panel
            scale = min(w / surf.get_width(), h / surf.get_height())
            new_w = max(1, int(surf.get_width() * scale))
            new_h = max(1, int(surf.get_height() * scale))
            surf = pygame.transform.smoothscale(surf, (new_w, new_h))
            self.screen.blit(
                surf,
                (x + (w - new_w) // 2, y + (h - new_h) // 2),
            )

        # Debug text overlay
        if debug:
            lines = []
            if debug.get("state_key"):
                lines.append(f"State: {debug['state_key']}")
            if debug.get("head"):
                lines.append(f"Head:  {debug['head']}")
            if debug.get("mouth") is not None:
                lines.append(f"Mouth: {debug['mouth']}")
            if debug.get("eye") is not None:
                lines.append(f"Eye:   {debug['eye']}")
            lines.append(f"Theme: {debug.get('theme', 'default')}")
            if debug.get("mode"):
                lines.append(f"Mode:  {debug['mode']}")
            if debug.get("conf") is not None:
                lines.append(f"Conf:  {debug['conf']:.2f}")
            if debug.get("yaw_deg") is not None:
                lines.append(
                    f"Yaw: {debug['yaw_deg']:+.1f}°  "
                    f"Pitch: {debug['pitch_deg']:+.1f}°"
                )

            for i, line in enumerate(lines):
                ts = self._font.render(line, True, (180, 200, 220))
                self.screen.blit(ts, (x + 6, y + 6 + i * 15))

    # ------------------------------------------------------------------
    #  Right panel: character
    # ------------------------------------------------------------------

    def _draw_right_panel(
        self, x: int, y: int, w: int, h: int,
        frame_path: Path, state_key: Optional[str], theme: str,
        debug: Optional[dict] = None,
    ) -> None:
        # Background
        pygame.draw.rect(self.screen, (18, 20, 28), (x, y, w, h))

        surface = self._load_surface(frame_path)
        scaled = self._fit_rect(surface, w, h)
        rect = scaled.get_rect(center=(x + w // 2, y + h // 2))
        self.screen.blit(scaled, rect)

        # Overlay: theme badge
        if theme != "default":
            badge = self._font.render(f"[{theme}]", True, (255, 200, 80))
            self.screen.blit(badge, (x + 8, y + 8))

    # ------------------------------------------------------------------
    #  Helpers
    # ------------------------------------------------------------------

    def _load_surface(self, frame_path: Path) -> pygame.Surface:
        if frame_path not in self.surface_cache:
            loaded = pygame.image.load(str(frame_path)).convert_alpha()
            self.surface_cache[frame_path] = loaded
        return self.surface_cache[frame_path]

    def _fit_rect(self, surface: pygame.Surface, w: int, h: int) -> pygame.Surface:
        src_w, src_h = surface.get_size()
        scale = min(w / src_w, h / src_h)
        new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
        return pygame.transform.smoothscale(surface, new_size)

    @staticmethod
    def _next_random_deadline(base_time: Optional[float] = None) -> float:
        start = time.time() if base_time is None else base_time
        low, high = config.RANDOM_CUT_INTERVAL
        return start + random.uniform(low, high)
