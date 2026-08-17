"""Interactive annotation UI (pygame) — label candidate frames.

Keyboard:
  朝向:  1=center  2=L1  3=L2  4=R1  5=R2  6=U1  7=D1
  眼睛:  A=open    S=half   D=closed
  嘴巴:  Z=closed  X=half   C=open
  导航:  ←/→ 前后帧     N 下一个未标注     P 上一个未标注
  编辑:  Enter 确认保存    Backspace 清除标注
  退出:  Q / Esc（自动保存）
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pygame

from face_detect import FaceDetection

HEAD_LABELS = ["center", "L1", "L2", "R1", "R2", "U1", "D1"]
HEAD_KEYS = {
    pygame.K_1: "center", pygame.K_2: "L1", pygame.K_3: "L2",
    pygame.K_4: "R1", pygame.K_5: "R2", pygame.K_6: "U1", pygame.K_7: "D1",
}
EYE_LABELS = ["open", "half", "closed"]
EYE_KEYS = {pygame.K_a: "open", pygame.K_s: "half", pygame.K_d: "closed"}
MOUTH_LABELS = ["closed", "half", "open"]
MOUTH_KEYS = {pygame.K_z: "closed", pygame.K_x: "half", pygame.K_c: "open"}


class Annotator:
    def __init__(
        self,
        workdir: Path,
        candidates: list[int],
        frames: list[np.ndarray],
        detections: list[FaceDetection | None],
        annotations: dict[str, dict] | None = None,
        prefills: dict[str, dict] | None = None,
        budget: int = 100,
    ) -> None:
        self.workdir = workdir
        self.candidates = candidates
        self.frames = frames
        self.detections = detections
        self.annotations = annotations or {}   # frame_id → {head,eye,mouth}
        self.prefills = prefills or {}          # auto-classification guesses
        self.budget = budget

        self.idx = 0
        self.running = True
        self.saved_count = sum(1 for k in self.annotations if k != "")

        self._init_pygame()

    # ------------------------------------------------------------------

    def _init_pygame(self) -> None:
        pygame.init()
        pygame.display.set_caption("Face Asset Annotator")
        self.screen = pygame.display.set_mode((960, 620))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        self.font_big = pygame.font.SysFont("monospace", 20)

    # ------------------------------------------------------------------

    def run(self) -> dict[str, dict]:
        while self.running and self.saved_count < self.budget:
            self._handle_events()
            self._draw()
            self.clock.tick(30)

        self._save()
        pygame.quit()
        return self.annotations

    # ------------------------------------------------------------------

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_ESCAPE, pygame.K_q):
                    self.running = False
                elif k == pygame.K_LEFT:
                    self.idx = max(0, self.idx - 1)
                elif k == pygame.K_RIGHT:
                    self.idx = min(len(self.candidates) - 1, self.idx + 1)
                elif k == pygame.K_n:
                    self._jump_unannotated(+1)
                elif k == pygame.K_p:
                    self._jump_unannotated(-1)
                elif k == pygame.K_RETURN:
                    self._save()
                elif k == pygame.K_BACKSPACE:
                    self._clear_current()
                elif k in HEAD_KEYS:
                    self._set_current("head", HEAD_KEYS[k])
                elif k in EYE_KEYS:
                    self._set_current("eye", EYE_KEYS[k])
                elif k in MOUTH_KEYS:
                    self._set_current("mouth", MOUTH_KEYS[k])

    def _current_fid(self) -> str:
        return str(self.candidates[self.idx])

    def _set_current(self, field: str, value: str) -> None:
        fid = self._current_fid()
        ann = self.annotations.setdefault(fid, {})
        ann[field] = value

    def _clear_current(self) -> None:
        self.annotations.pop(self._current_fid(), None)

    def _jump_unannotated(self, direction: int) -> None:
        n = len(self.candidates)
        for step in range(1, n):
            j = (self.idx + direction * step) % n
            if str(self.candidates[j]) not in self.annotations:
                self.idx = j
                return

    def _save(self) -> None:
        path = self.workdir / "annotations.json"
        path.write_text(json.dumps(self.annotations, indent=1))
        self.saved_count = len(self.annotations)
        print(f"[save] {self.saved_count} annotations → {path}")

    # ------------------------------------------------------------------

    def _draw(self) -> None:
        self.screen.fill((30, 32, 40))

        fid = self._current_fid()
        frame = self.frames[self.candidates[self.idx]]
        det = self.detections[self.candidates[self.idx]]

        # frame display (scale to fit 960×540 area)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        surf = pygame.image.frombuffer(rgb.tobytes(), (rgb.shape[1], rgb.shape[0]), "RGB")
        surf = pygame.transform.smoothscale(surf, (720, 540))
        self.screen.blit(surf, (10, 40))

        # face overlay
        if det:
            fx, fy, fw, fh = det.bbox
            scale = 720 / frame.shape[1]
            rect = pygame.Rect(
                10 + int(fx * scale), 40 + int(fy * scale),
                int(fw * scale), int(fh * scale),
            )
            pygame.draw.rect(self.screen, (0, 255, 128), rect, 2)
            for pt, color in [
                (det.left_eye, (255, 200, 0)),
                (det.right_eye, (255, 200, 0)),
                (det.nose, (255, 128, 0)),
                (det.mouth, (0, 200, 255)),
            ]:
                if pt:
                    pygame.draw.circle(
                        self.screen, color,
                        (10 + int(pt[0] * scale), 40 + int(pt[1] * scale)), 4,
                    )

        # info panel (right side)
        x = 750
        ann = self.annotations.get(fid, {})
        pre = self.prefills.get(fid, {})
        lines = [
            f"Frame: {fid}  ({self.idx + 1}/{len(self.candidates)})",
            f"Budget: {self.saved_count}/{self.budget}",
            "",
            "HEAD  [1-7]:",
            *[f"  {'>>' if ann.get('head') == h else '  '} {h}  "
              f"{'[' + pre.get('head','') + ']' if pre.get('head') else ''}"
              for h in HEAD_LABELS],
            "",
            "EYE  [A/S/D]:",
            *[f"  {'>>' if ann.get('eye') == e else '  '} {e}"
              for e in EYE_LABELS],
            "",
            "MOUTH [Z/X/C]:",
            *[f"  {'>>' if ann.get('mouth') == m else '  '} {m}"
              for m in MOUTH_LABELS],
            "",
            "N/P: next/prev unlabelled",
            "Enter: save   Backspace: clear",
            "Q/Esc: quit (auto-save)",
        ]
        for i, line in enumerate(lines):
            color = (220, 220, 230) if not line.startswith("  >>") else (120, 255, 160)
            ts = self.font.render(line, True, color)
            self.screen.blit(ts, (x, 40 + i * 18))

        pygame.display.flip()
