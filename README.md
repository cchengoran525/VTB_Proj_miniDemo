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
python generate_placeholders.py          # 只生成新主题占位帧 (drinking/lean_forward)
python generate_placeholders.py --base   # 重新生成整套基础帧（量大，一般不需要）

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
| **3 态眼/嘴** | 嘴 closed/half/open、眼三态，施密特迟滞防抖 |
| **双模切换** | 正脸 CAM 走摄像头，侧脸低置信 SIM 切模拟器（随机眨眼 + 音频驱动嘴） |
| **特殊状态 (themes)** | 离开座位 / 回到座位 / 手撑脸 / 喝水举杯 / 前倾，详见下节 |
| **微差分 (variant)** | 每头姿 5 种随机变体，离开再进入时重抽 |
| **音频驱动嘴** | sounddevice 实时 RMS，3 级迟滞映射到 closed/half/open |
| **终端调试工具** | `debug_tracker.py` 实时显示角度条、置信度分量、麦克风电平 |

## 特殊状态 (themes)

主循环每帧按优先级解析当前主题。注意两种语义的区分——**动作 (action)** 一次性触发、确认后才播,不重复;**状态 (state)** 条件持续期间一直保持:

| 优先级 | 状态 | 语义 | 触发条件 | 帧目录 |
|---|---|---|---|---|
| 1 | **leaving** | 动作 | 丢脸持续 > 2s (LEAVING_TIMEOUT) 触发一次;缺席期间保持该画面(真素材到位后应拆成"离开动作+离席状态") | `frames_placeholder/leaving/` |
| 2 | **returning** | 动作 | 脸回来且持续 > 0.7s (RETURN_CONFIRM_DURATION) 才触发一次,播 2s (RETURNING_DURATION);闪烁一帧不会误触发 | `frames_placeholder/returning/` |
| 3 | **drinking** | 状态 | 手/杯子在嘴部区域,持续 ~0.6s 进入,离开 ~0.5s 退出 | `frames_placeholder/drinking/` |
| 4 | **hand_on_face** | 状态 | 手在扩大脸框内 ≥3 个关键点,持续 ~0.4s;`H` 键手动强制 | `frames_placeholder/hand_on_face/` |
| 5 | **lean_forward** | 状态 | 脸高 ≥ 基线 118% 进入,回落 < 110% 才退出(迟滞带),持续 ~0.7s | `frames_placeholder/lean_forward/` |

### V1 头位档位 (3-level)

按你的规格定义,yaw 三档 + pitch 两档,共 9 个方向:

```
|yaw| < 11.5°    → center (此时才显示 pitch U1/D1/U2/D2)
11.5° ~ 35.5°    → L1 / R1   (轻微偏头)
35.5° ~ 55°      → L2 / R2   (~45° 明显转头)
55°+             → L3 / R3   (接近完全侧面)
pitch: 10° / 20° → U1 D1 / U2 D2
```

边界在 `config.py` 的 `HEAD_YAW_EDGES` / `HEAD_PITCH_EDGES`。真素材还没有 L3 帧时,显示自动回退到 L2 画面。

- 手部判定基于 HandLandmarker 的 10 个关键点（手腕+指关节+指尖）与脸部区域的关系，不再依赖"手腕贴近鼻子"（撑脸时手腕根本不在脸旁）
- 进入/退出都有帧数防抖（Schmitt 式），单帧噪声不会触发
- 每个状态是一个独立帧目录，替换正式美术时按 `{mouth}_{eye}_{head}.png` 命名放入即可；`leaving`/`returning` 目前是单张占位图
- debug 窗口会画出绿色检测框（扩大脸框）和蓝色嘴部区域，方便调参

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

完整愿景(V1 开关全开时):

```
43 头姿 × 5 variant × 3 眼睛 × 3 嘴巴 = 1,935 帧
```

**当前 V1 规格**(表现层始终三态嘴;视频提取的素材可以先只有 closed/open,
half 由手绘差分补齐,缺帧时显示层最近邻回退):

```
9 头向 × 3 嘴 × 3 眼 = 81 帧/主题 (素材不足时自动回退,不阻塞)

{m}_{e}_{head}.png

m    ∈ {closed, half, open}    # 闭嘴 / 说话微张 / 大张
e    ∈ {closed, half, open}    # 闭眼 / 半开 / 全开
head ∈ {center, L1, L2, L3, R1, R2, R3, U1, D1}

示例:
  open_closed_L3.png     # 张嘴、闭眼、左侧面
  half_open_center.png   # 说话微张、睁眼、正脸
  closed_open_center.png # 闭嘴、睁眼、正脸
```

早期/未来扩展命名(5×5 网格 + roll + variant,暂存档):

```
{m}_{e}_{yaw}{pitch}[_{roll}]_v{variant}.png
yaw ∈ {L2,L1,空,R1,R2} · pitch ∈ {U2,U1,空,D1,D2} · roll ∈ {WL,空,WR}
variant ∈ {1..5}  (v1 省略后缀)
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
MOUTH_OPEN_THRESHOLD = 0.70   # half → open
MOUTH_HALF_THRESHOLD = 0.30   # closed → half
EYE_OPEN_THRESHOLD = 0.68
EYE_HALF_THRESHOLD = 0.30
MOUTH_HYSTERESIS = 0.04
EYE_HYSTERESIS = 0.06

# 双模切换
FACE_CONFIDENCE_THRESHOLD = 0.25  # 低于此值切 SIM

# 特殊状态
LEAVING_TIMEOUT = 2.0           # 丢脸超过 → leaving
HAND_ON_FACE_ENTER_FRAMES = 12  # 撑脸需持续 ~0.4s
DRINKING_ENTER_FRAMES = 18      # 喝水需持续 ~0.6s
DRINK_MOUTH_HALF_W = 0.35       # 嘴部区域半宽 (×脸宽)
DRINK_MOUTH_HALF_H = 0.30       # 嘴部区域半高 (×脸高，调小可避免托下巴误判喝水)
LEAN_SCALE_THRESHOLD = 1.18     # 脸 ≥ 基线 118% → 前倾

# 帧资产
FRAME_DB_PATH = BASE_DIR / "frames_placeholder"
TRANSITION_FRAMES_ENABLED = False
```

## 替换正式美术

1. 按上方命名规范产出 1,935 张 PNG
2. 放入 `frames_placeholder/`（或修改 `FRAME_DB_PATH` 指向你的素材目录）
3. 按需生成过渡帧 (`_to_` 命名)，然后设 `TRANSITION_FRAMES_ENABLED = True`
