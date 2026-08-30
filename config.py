from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FRAME_DB_PATH = BASE_DIR / "frames_placeholder"  # 改为 "frames" 使用旧版素材
FACE_LANDMARKER_MODEL = BASE_DIR / "models" / "face_landmarker.task"
HAND_LANDMARKER_MODEL = BASE_DIR / "models" / "hand_landmarker.task"

CAMERA_INDEX = 0
TARGET_FPS = 30
FULLSCREEN = False          # V1: 双窗口模式用窗口
WINDOW_SIZE = (1280, 720)

# ---- V1 MVP mode ------------------------------------------------------------
# Limits head poses to the 7 stable directions; blocks roll + diagonals.
HEAD_V1_MODE = True

# Theme system
THEME_DEFAULT = "default"
THEME_HAND_ON_FACE = "hand_on_face"
THEME_DRINKING = "drinking"
THEME_LEAN_FORWARD = "lean_forward"

# Leaving / returning — both are ONE-SHOT actions: each fires exactly once,
# only after its confirmation interval gives near-certainty.
LEAVING_TIMEOUT = 2.0          # face lost continuously > this → leave action fires
RETURN_CONFIRM_DURATION = 0.7  # face must stay back this long → return action fires
RETURNING_DURATION = 2.0       # return action presentation length once fired

# ---- hand-face interaction (撑脸 / 喝水) --------------------------------------
# Hand keypoints tested against face regions (MediaPipe hand landmark ids):
#   0 wrist · 5/9/13/17 knuckles · 4/8/12/16/20 fingertips
HAND_CHECK_INDICES = (0, 4, 5, 8, 9, 12, 13, 16, 17, 20)

# "Hand on face" test — expand face bbox by these fractions of its own size,
# then count hand keypoints inside.  Catches cheek-rest / chin-rest / eye-rub.
HAND_FACE_MARGIN_X = 0.45
HAND_FACE_MARGIN_Y = 0.35
HAND_ON_FACE_MIN_POINTS = 3

# "At mouth" test (drinking) — zone centred on the mouth, sized in face units.
# Deliberately narrower in x than the on-face box so cheek-rest doesn't
# double-fire as drinking; hand/cup must be in front of the mouth.
# H is tight enough that chin-rest falls outside the zone (→ hand_on_face).
DRINK_MOUTH_HALF_W = 0.35      # × face width, each side of mouth centre
DRINK_MOUTH_HALF_H = 0.30      # × face height, up/down from mouth centre
DRINK_MIN_POINTS = 2

# Debounce (frames @ 30 fps) — enter/exit persistence per special state.
HAND_ON_FACE_ENTER_FRAMES = 12   # ~0.4 s sustained to trigger
HAND_ON_FACE_EXIT_FRAMES = 20    # ~0.7 s clear to release
DRINKING_ENTER_FRAMES = 18       # ~0.6 s
DRINKING_EXIT_FRAMES = 15        # ~0.5 s

# ---- lean forward (前倾) ------------------------------------------------------
# A STATE, not an action: held while the face stays close, released when the
# user settles back.  Face bbox height vs the neutral baseline captured during
# startup calibration, with a Schmitt band so the boundary never flickers.
LEAN_SCALE_THRESHOLD = 1.18    # enter: face ≥ 118% of baseline
LEAN_EXIT_SCALE = 1.10         # exit: drop below 110% to release
LEAN_ENTER_FRAMES = 20         # ~0.7 s sustained to enter
LEAN_EXIT_FRAMES = 30          # ~1.0 s sustained to release

# Head pose source: "matrix" (MediaPipe built-in, accurate) or "heuristic" (legacy)
HEAD_POSE_SOURCE = "matrix"

# Calibration: face the camera directly for this many frames on startup.
# The tracker averages your neutral pose and subtracts it automatically.
CALIBRATION_FRAMES = 45

# Dual-mode threshold: below this confidence, eye/mouth switch from camera
# to simulator.  0.52 rad ≈ 30° head turn → confidence = 0.
FACE_CONFIDENCE_THRESHOLD = 0.25
OCCLUSION_DETECTION_ENABLED = False  # hand-over-face detection (WIP)

# Raw feature normalization ranges.  Mouth/eye ratios now use FACE HEIGHT
# as denominator (yaw-invariant).  Recalibrate with the debug overlay's
# "raw" readout if your camera differs: note raw values for closed/half/open.
MOUTH_RATIO_RANGE = (0.010, 0.100)  # closed ~0.01-0.03 · half ~0.03-0.06 · open ~0.06+
EYE_RATIO_RANGE = (0.008, 0.045)    # closed ~0.005-0.015 · half ~0.02 · open ~0.03+

# Pose centering offsets. These keep a neutral face closer to "center".
PITCH_CENTER = 0.44   # 提高以消除中性位"往下看"的偏置
PITCH_SCALE = 1.8
YAW_SCALE = 6.0       # 大幅增大增益让左右更明显（放大信号，不放大噪声）
ROLL_SCALE = 1.0

# Discrete mapping thresholds (normalized 0-1 over the ranges above).
MOUTH_OPEN_THRESHOLD = 0.70   # half → open  (raw ≈ 0.073)
MOUTH_HALF_THRESHOLD = 0.30   # closed → half (raw ≈ 0.037; 闭嘴误判 half 就调大)
EYE_OPEN_THRESHOLD = 0.75     # half → open  (raw ≈ 0.036)
EYE_HALF_THRESHOLD = 0.30     # closed → half (raw ≈ 0.019)
HEAD_YAW_THRESHOLD = 0.05
HEAD_PITCH_THRESHOLD = 0.10

# Head direction debounce: lock non-center direction for this many seconds.
HEAD_LOCK_DURATION = 1.5

# Transition frames — disabled for now, use debounce lock only.
TRANSITION_FRAMES_ENABLED = False

# ---- head-direction grid ----------------------------------------------------
HEAD_GRID_ENABLED = True
HEAD_GRID_RADIUS = 2          # 2 = 5×5  (L2/L1/0/R1/R2  ×  U2/U1/0/D1/D2)
HEAD_GRID_YAW_STEP = 0.31     # ~18° per step (full-grid mode only)
HEAD_GRID_PITCH_STEP = 0.17   # ~10° per step
HEAD_GRID_ROLL_STEP = 0.10    # ~6° : WL / — / WR
HEAD_CENTER_YAW_MAX = 0.17    # ~10° — pitch only shows within this yaw range
HEAD_ROLL_INNER_ONLY = True   # only apply roll to inner 3×3 (extreme angles hurt)

# V1 mode head levels — user spec: 轻微偏 = 1, ~45° = 2, 完全侧面 = 3.
# Yaw edge thresholds in radians (~11.5° / ~35.5° / ~55°).
HEAD_YAW_EDGES = (0.20, 0.62, 0.96)
# Pitch levels: ~10° / ~20°
HEAD_PITCH_EDGES = (0.17, 0.34)

# Micro‑variations per head key (for visual freshness).
# Set to 1 to disable.  Frame naming:  open_open_R1_v3.png
HEAD_VARIANTS_PER_KEY = 1  # 关闭微差分（每个头姿只画一版）

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

