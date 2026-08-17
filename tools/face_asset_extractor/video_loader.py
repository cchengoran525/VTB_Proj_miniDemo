"""Video import & frame extraction, normalised to 480p."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class FrameInfo:
    frame_id: int                 # global unique id
    video: str                    # source video filename
    video_frame: int              # index inside source video
    timestamp: float              # seconds from video start
    path: str = ""                # PNG path in workdir/frames


class VideoLoader:
    """Extracts frames from videos at a fixed sample rate, 480p normalised."""

    TARGET_HEIGHT = 480

    def __init__(self, workdir: Path, sample_fps: float = 10.0) -> None:
        self.workdir = workdir
        self.frames_dir = workdir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.sample_fps = sample_fps

    # ------------------------------------------------------------------

    def extract(self, video_paths: list[Path]) -> list[FrameInfo]:
        """Extract frames from all videos.  Returns global frame list."""
        all_frames: list[FrameInfo] = []
        next_id = 0

        for vp in video_paths:
            if not vp.exists():
                print(f"  [skip] not found: {vp}")
                continue

            frames = self._extract_one(vp, next_id)
            if not frames:
                print(f"  [warn] no frames from {vp.name} — wrong codec?")
                continue

            print(f"  {vp.name}: {len(frames)} frames")
            all_frames.extend(frames)
            next_id += len(frames)

        print(f"Total: {len(all_frames)} frames")
        return all_frames

    def save_meta(self, frames: list[FrameInfo]) -> None:
        meta = [
            {
                "frame_id": f.frame_id,
                "video": f.video,
                "video_frame": f.video_frame,
                "timestamp": f.timestamp,
                "path": f.path,
            }
            for f in frames
        ]
        (self.workdir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=1)
        )

    # ------------------------------------------------------------------

    def _extract_one(self, vp: Path, start_id: int) -> list[FrameInfo]:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(1, int(round(fps / self.sample_fps)))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # 480p downscale factor (720p→480p; smaller videos stay untouched)
        scale = self.TARGET_HEIGHT / src_h if src_h > self.TARGET_HEIGHT else 1.0

        frames: list[FrameInfo] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue

            if scale != 1.0:
                new_w = int(round(frame.shape[1] * scale))
                frame = cv2.resize(frame, (new_w, self.TARGET_HEIGHT))

            fid = start_id + len(frames)
            path = self.frames_dir / f"{fid:06d}.png"
            cv2.imwrite(str(path), frame)

            frames.append(
                FrameInfo(
                    frame_id=fid,
                    video=vp.name,
                    video_frame=idx,
                    timestamp=idx / fps,
                    path=str(path.relative_to(self.workdir)),
                )
            )
            idx += 1

        cap.release()
        return frames

    @staticmethod
    def load_frame(workdir: Path, frame: FrameInfo) -> np.ndarray:
        return cv2.imread(str(workdir / frame.path))
