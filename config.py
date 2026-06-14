from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FRAME_DB_PATH = BASE_DIR / "frames"
FACE_LANDMARKER_MODEL = BASE_DIR / "models" / "face_landmarker.task"

CAMERA_INDEX = 0
TARGET_FPS = 30
FULLSCREEN = True
WINDOW_SIZE = (1280, 720)

# Head pose source: "matrix" (MediaPipe built-in, accurate) or "heuristic" (legacy)
HEAD_POSE_SOURCE = "matrix"

# Calibration: face the camera directly for this many frames on startup.
# The tracker averages your neutral pose and subtracts it automatically.
CALIBRATION_FRAMES = 45

# Dual-mode threshold: below this confidence, eye/mouth switch from camera
# to simulator.  0.52 rad ≈ 30° head turn → confidence = 0.
FACE_CONFIDENCE_THRESHOLD = 0.25

# Raw feature normalization ranges. Tune these if your camera angle differs a lot.
MOUTH_RATIO_RANGE = (0.02, 0.20)   # narrowed: raw mouth ratio spans ~0.02-0.15
EYE_RATIO_RANGE = (0.10, 0.33)

# Pose centering offsets. These keep a neutral face closer to "center".
PITCH_CENTER = 0.44   # 提高以消除中性位"往下看"的偏置
PITCH_SCALE = 1.8
YAW_SCALE = 6.0       # 大幅增大增益让左右更明显（放大信号，不放大噪声）
ROLL_SCALE = 1.0

# Discrete mapping thresholds.
MOUTH_OPEN_THRESHOLD = 0.42
EYE_OPEN_THRESHOLD = 0.68
EYE_HALF_THRESHOLD = 0.30
HEAD_YAW_THRESHOLD = 0.05   # 降低阈值配合高增益
HEAD_PITCH_THRESHOLD = 0.10  # 配合 PITCH_CENTER 提高，防止误触发

# Head direction debounce: lock non-center direction for this many seconds.
HEAD_LOCK_DURATION = 1.5

# ---- head-direction grid ----------------------------------------------------
HEAD_GRID_ENABLED = True
HEAD_GRID_RADIUS = 1          # 1 = 3×3  (L1/0/R1 × U1/0/D1)
                               # 2 = 5×5  (L2/L1/0/R1/R2 × …)
HEAD_GRID_YAW_STEP = 0.14     # ~8° per yaw step
HEAD_GRID_PITCH_STEP = 0.10   # ~6° per pitch step
HEAD_GRID_ROLL_STEP = 0.07    # ~4° : WL / — / WR  (tilt left / neutral / tilt right)

# Hysteresis margins for eye/mouth state transitions.
# The signal must cross threshold ± margin to change state, preventing
# threshold-boundary jitter (Schmitt trigger).
EYE_HYSTERESIS = 0.06
MOUTH_HYSTERESIS = 0.04

# Random hard-cut timing range in seconds.
RANDOM_CUT_INTERVAL = (1.0, 4.0)

# Simple 1D Kalman filter parameters.
KALMAN_PROCESS_NOISE = 0.008
KALMAN_MEASUREMENT_NOISE = 0.08

# Per-dimension Kalman responsiveness multiplier (>1 = more responsive, less smoothing).
# Head pose needs slightly faster tracking but too high causes jitter.
KALMAN_RESPONSIVENESS = {
    "pitch": 1.2,
    "yaw": 1.5,
    "roll": 1.2,
    "mouth": 1.0,
    "left_eye": 1.0,
    "right_eye": 1.0,
}

