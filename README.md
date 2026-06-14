# VTB_Proj_miniDemo

2D VTuber 原型 — 摄像头人脸追踪 + 离散状态映射 + 本地帧库显示。

```
摄像头 → MediaPipe FaceLandmarker → 矩阵法头部位姿 → 自动校准
                                         ↓
                        TrackingState (pitch/yaw/roll/mouth/eye/confidence)
                                         ↓
                        StateMapper → DiscreteState (43 头姿 × 3 眼 × 3 嘴 × 5 variant)
                                         ↓
                        帧检索 (精确匹配 or 最近邻 fallback)
                                         ↓
                        Pygame 全屏 30fps 渲染
```

## 快速开始

```bash
# 1. 下载模型 & 安装依赖
mkdir -p models
curl -L -o models/face_landmarker.task \
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. 生成占位帧（或直接使用已有 frames/ 目录）
python generate_placeholders.py

# 3. 运行
python main.py          # 正式运行 (pygame 全屏)
python debug_tracker.py # 终端调试 (无 GUI，含麦克风音量)
```

## 核心特性

| 特性 | 说明 |
|---|---|
| **矩阵法头部位姿** | MediaPipe 4×4 变换矩阵 + Rodrigues 提取 yaw/pitch/roll，精度远超手写几何 |
| **自动校准** | 启动时采集 45 帧中性位姿，自动归零 |
| **5×5 + roll 网格** | 25 方向 (L2~R2 × U2~D2)，内圈 9 位置叠加歪头 (WL/WR) |
| **3 态眼/嘴** | open / half / closed，施密特迟滞防抖 |
| **双模切换** | 正脸 CAM 走摄像头，侧脸低置信 SIM 切模拟器（随机眨眼 + 音频驱动嘴） |
| **微差分 (variant)** | 每头姿 5 种随机变体，离开再进入时重抽 |
| **音频驱动嘴** | sounddevice 实时 RMS，3 级迟滞映射到 closed/half/open |
| **终端调试工具** | `debug_tracker.py` 实时显示角度条、置信度分量、麦克风电平 |

## 文件结构

```text
VTB_Proj_miniDemo/
├── main.py               # 主循环
├── debug_tracker.py       # 终端调试工具
├── config.py              # 所有可调参数
├── tracker.py             # 摄像头 + MediaPipe + Kalman
├── mapper.py              # 状态分类 + 帧检索
├── display.py             # Pygame 渲染
├── simulator.py           # 眼/嘴模拟器（CAM↔SIM 切换用）
├── audio_capture.py       # 麦克风 RMS 采集
├── generate_placeholders.py  # 占位帧生成器
├── requirements.txt
├── frames/                # 旧版 92 张 demo 帧 (legacy 5 方向)
├── frames_placeholder/    # 新版 1935 张占位帧 (5×5+roll+variant)
└── models/                # face_landmarker.task 模型文件
```

## 状态空间 & 帧命名

```
43 头姿 × 5 variant × 3 眼睛 × 3 嘴巴 = 1,935 帧
```

```
{m}_{e}_{yaw}{pitch}[_{roll}]_v{variant}.png

m  ∈ {closed, half, open}     # 闭嘴 / 说话微张 / 大张
e  ∈ {closed, half, open}     # 闭眼 / 半开 / 全开
yaw ∈ {L2, L1, 空, R1, R2}   # 5 级左右
pitch ∈ {U2, U1, 空, D1, D2}  # 5 级上下
roll ∈ {WL, 空, WR}           # 仅内圈 3×3（外圈不歪头）
variant ∈ {1..5}              # 微差分 (v1 省略后缀)

示例:
  half_open_R1_D1_WL_v3.png   # 说话、睁眼、右1下1、左歪、variant 3
  open_closed_L2_v1.png       # 大张嘴、闭眼、左2
  closed_open_center.png      # 闭嘴、睁眼、正脸、variant 1 (无后缀)
```

### 头部姿态网格

```
        U2    U1    0    D1    D2
L2     [—]   [—]   [—]  [—]   [—]    ← 外圈 16 位置：不歪头
L1     [—]  [↺↻]  [↺↻] [↺↻]  [—]    ← 内圈 9 位置：可歪头
0      [—]  [↺↻]  [↺↻] [↺↻]  [—]
R1     [—]  [↺↻]  [↺↻] [↺↻]  [—]
R2     [—]   [—]   [—]  [—]   [—]
```

### 过渡帧（当前已禁用）

```
{old_state}_to_{new_state}.png
例: closed_half_center_to_open_open_R1.png

TRANSITION_FRAMES_ENABLED = False  ← config.py 控制
```

## 主要配置项

```python
# config.py

# 网格
HEAD_GRID_ENABLED = True
HEAD_GRID_RADIUS = 2          # 2 = 5×5
HEAD_ROLL_INNER_ONLY = True   # 仅内圈歪头
HEAD_VARIANTS_PER_KEY = 5      # 每头姿变体数

# 阈值 (Schmitt 迟滞)
MOUTH_OPEN_THRESHOLD = 0.45   # half → open
MOUTH_HALF_THRESHOLD = 0.15   # closed → half
EYE_OPEN_THRESHOLD = 0.68
EYE_HALF_THRESHOLD = 0.30
MOUTH_HYSTERESIS = 0.04
EYE_HYSTERESIS = 0.06

# 双模切换
FACE_CONFIDENCE_THRESHOLD = 0.25  # 低于此值切 SIM

# 帧资产
FRAME_DB_PATH = BASE_DIR / "frames_placeholder"
TRANSITION_FRAMES_ENABLED = False
```

## 替换正式美术

1. 按上方命名规范产出 1,935 张 PNG
2. 放入 `frames_placeholder/`（或修改 `FRAME_DB_PATH` 指向你的素材目录）
3. 按需生成过渡帧 (`_to_` 命名)，然后设 `TRANSITION_FRAMES_ENABLED = True`
