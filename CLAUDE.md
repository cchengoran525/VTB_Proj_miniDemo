# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# One-time: download the MediaPipe FaceLandmarker model
mkdir -p models && curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

# Install dependencies
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt

# Run
python3 main.py
```

There is no test suite or linter configured for this project.

## Architecture

A single-threaded 2D VTuber pipeline: **camera → CV perception → discrete state mapping → pygame frame display**.

### Data flow (`main.py:10-43`)

1. `FaceTracker.read_state()` reads a webcam frame, runs MediaPipe FaceMesh, extracts pitch/yaw/roll, mouth openness, and left/right eye openness, smooths each dimension through a 1D Kalman filter, returns a `TrackingState`.
2. `StateMapper.classify()` thresholds the continuous state into discrete labels — `mouth` (open/closed), `eye` (open/half/closed), `head` (center/left/right/up/down) — returning a `DiscreteState`.
3. `StateMapper.resolve_frame()` looks up the matching PNG by the `{mouth}_{eye}_{head}` key. If no exact match exists, it finds the nearest neighbor with a weighted distance (mouth mismatch=100, eye=10, head=1–3).
4. `Display.choose_next_frame()` picks the actual frame to render using a three-tier priority: (a) random hard-cut if the timer fired, (b) one-frame transition frame `{old_key}_to_{new_key}.png` if available, (c) direct hard-cut otherwise.
5. `Display.render()` draws the chosen PNG centered on a pygame surface at 30fps.

### Module responsibilities

- **`config.py`** — All tunable constants: thresholds, Kalman noise, feature normalization ranges, random-cut interval window.
- **`tracker.py`** — `FaceTracker` wraps `cv2.VideoCapture` + `mediapipe.FaceMesh`; `Kalman1D` is a simple per-dimension filter. MediaPipe landmark indices are module-level constants.
- **`mapper.py`** — `StateMapper` indexes the `frames/` directory at init, separating base frames from transition frames by the `_to_` substring in the filename. `classify()` uses configurable thresholds.
- **`display.py`** — `FrameDisplay` manages the pygame window (fullscreen by default), caches loaded surfaces, scales frames to fit.

### Frame asset convention

Files in `frames/` must be PNGs:
- **Base frames**: `{mouth}_{eye}_{head}.png` (e.g., `open_open_center.png`)
- **Transition frames**: `{old_mouth}_{old_eye}_{old_head}_to_{new_mouth}_{new_eye}_{new_head}.png`

The shipped `frames/` dir contains 30 base frames + 62 transition frames as placeholder art.
