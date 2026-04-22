from __future__ import annotations

import random
import time
from pathlib import Path
from typing import Iterable, Optional

import pygame

import config


class FrameDisplay:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("VTB Mini Demo")
        flags = pygame.FULLSCREEN if config.FULLSCREEN else 0
        size = (0, 0) if config.FULLSCREEN else config.WINDOW_SIZE
        self.screen = pygame.display.set_mode(size, flags)
        self.clock = pygame.time.Clock()
        self.surface_cache: dict[Path, pygame.Surface] = {}
        self.current_frame_path: Optional[Path] = None
        self.current_state_key: Optional[str] = None
        self.pending_frame_path: Optional[Path] = None
        self.pending_state_key: Optional[str] = None
        self.next_random_cut_at = self._next_random_deadline()

    def close(self) -> None:
        pygame.quit()

    def should_quit(self) -> bool:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q):
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

        if self.pending_frame_path is not None and self.pending_state_key is not None:
            frame_path = self.pending_frame_path
            state_key = self.pending_state_key
            self.pending_frame_path = None
            self.pending_state_key = None
            return frame_path, state_key, "transition-final"

        if self.current_state_key != target_state_key and transition_frame_path is not None:
            self.pending_frame_path = target_frame_path
            self.pending_state_key = target_state_key
            return transition_frame_path, self.current_state_key or target_state_key, "transition"

        return target_frame_path, target_state_key, "direct-hard-cut"

    def render(self, frame_path: Path, state_key: Optional[str]) -> None:
        surface = self._load_surface(frame_path)
        frame = self._fit_to_screen(surface)

        self.screen.fill((12, 12, 16))
        rect = frame.get_rect(center=self.screen.get_rect().center)
        self.screen.blit(frame, rect)
        pygame.display.flip()

        self.current_frame_path = frame_path
        self.current_state_key = state_key

    def _load_surface(self, frame_path: Path) -> pygame.Surface:
        if frame_path not in self.surface_cache:
            loaded = pygame.image.load(str(frame_path)).convert_alpha()
            self.surface_cache[frame_path] = loaded
        return self.surface_cache[frame_path]

    def _fit_to_screen(self, surface: pygame.Surface) -> pygame.Surface:
        screen_w, screen_h = self.screen.get_size()
        src_w, src_h = surface.get_size()
        scale = min(screen_w / src_w, screen_h / src_h)
        new_size = (max(1, int(src_w * scale)), max(1, int(src_h * scale)))
        return pygame.transform.smoothscale(surface, new_size)

    @staticmethod
    def _next_random_deadline(base_time: Optional[float] = None) -> float:
        start = time.time() if base_time is None else base_time
        low, high = config.RANDOM_CUT_INTERVAL
        return start + random.uniform(low, high)
