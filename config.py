from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
FRAME_DB_PATH = BASE_DIR / "frames"

CAMERA_INDEX = 0
TARGET_FPS = 30
FULLSCREEN = True
WINDOW_SIZE = (1280, 720)

# Raw feature normalization ranges. Tune these if your camera angle differs a lot.
MOUTH_RATIO_RANGE = (0.02, 0.30)
EYE_RATIO_RANGE = (0.10, 0.33)

# Pose centering offsets. These keep a neutral face closer to "center".
PITCH_CENTER = 0.36
PITCH_SCALE = 1.8
YAW_SCALE = 2.0
ROLL_SCALE = 1.0

# Discrete mapping thresholds.
MOUTH_OPEN_THRESHOLD = 0.42
EYE_OPEN_THRESHOLD = 0.68
EYE_HALF_THRESHOLD = 0.30
HEAD_YAW_THRESHOLD = 0.10
HEAD_PITCH_THRESHOLD = 0.08

# Random hard-cut timing range in seconds.
RANDOM_CUT_INTERVAL = (1.0, 4.0)

# Simple 1D Kalman filter parameters.
KALMAN_PROCESS_NOISE = 0.008
KALMAN_MEASUREMENT_NOISE = 0.08

