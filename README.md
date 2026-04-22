# VTB_Proj_miniDemo

一个最小可用的 2D VTuber 原型系统，按「CV 感知 -> 离散状态映射 -> 帧检索显示」三层拆分实现。

项目满足以下目标：

- `MediaPipe FaceMesh` 实时读取摄像头，提取 `pitch / yaw / roll`、嘴部开合、左右眼开合。
- 使用简单 `Kalman` 平滑抖动后输出状态向量。
- 将连续状态映射成离散标签：`mouth / eye / head`。
- 通过 `{嘴}_{眼}_{头部}.png` 文件名检索本地帧库。
- 显示层支持三种切换逻辑，优先级为：
  1. 随机硬切
  2. 中间帧过渡
  3. 直接硬切

## 文件结构

```text
VTB_Proj_miniDemo/
├── config.py
├── tracker.py
├── mapper.py
├── display.py
├── main.py
├── requirements.txt
├── README.md
└── frames/
```

## 模块说明

### 1. `tracker.py`

- 打开摄像头并运行 `MediaPipe FaceMesh`
- 提取以下连续特征：
  - `pitch / yaw / roll`
  - `mouth_open`
  - `left_eye_open`
  - `right_eye_open`
- 对每个维度做 1D Kalman 平滑

状态向量结构：

```python
[pitch, yaw, roll, mouth_open, left_eye_open, right_eye_open]
```

### 2. `mapper.py`

- 把连续状态映射到离散标签：
  - 嘴：`closed / open`
  - 眼：`open / half / closed`
  - 头部：`center / left / right / up / down`
- 从 `config.FRAME_DB_PATH` 指向的本地素材库中检索帧
- 若找不到精确匹配，执行最近邻回退：
  - 嘴部不匹配惩罚最高
  - 眼部其次
  - 头部方向最后

帧命名规则：

```text
{mouth}_{eye}_{head}.png
```

例如：

```text
open_open_center.png
closed_half_left.png
```

过渡帧命名规则：

```text
{old_state}_to_{new_state}.png
```

例如：

```text
closed_open_center_to_open_open_center.png
```

### 3. `display.py`

- 使用 `pygame` 全屏显示当前帧
- 运行目标帧率：`30fps`
- 切换策略：
  - 随机硬切：每 `1~4` 秒随机跳到一张基础帧
  - 中间帧过渡：如果存在过渡帧，先显示过渡帧一帧，再切换到目标帧
  - 直接硬切：直接替换当前帧
- `Esc` 或 `Q` 退出

## 默认素材库

仓库内已经附带一套可直接运行的占位帧库：

- 30 张基础状态帧
- 62 张常用过渡帧

这些帧是为了验证系统链路而生成的简化 demo 素材。后续只需要替换 `frames/` 目录内容，并保持相同命名规则即可接入正式美术资源。

## 运行方式

### 1. 安装依赖

```bash
cd /Users/chenmingyuan/Desktop/VTB_Proj_miniDemo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 运行

```bash
python3 main.py
```

## 可调参数

主要调参入口在 `config.py`：

- `FRAME_DB_PATH`：帧数据库路径
- `MOUTH_OPEN_THRESHOLD`：嘴巴开合离散阈值
- `EYE_OPEN_THRESHOLD`
- `EYE_HALF_THRESHOLD`
- `HEAD_YAW_THRESHOLD`
- `HEAD_PITCH_THRESHOLD`
- `RANDOM_CUT_INTERVAL`

如果你后面替换成自己的 Live2D 风格贴图，通常只需要先改这些阈值，再按相同命名规则补齐素材。

## 说明

这是一个最小原型，不做多线程，也不追求精确姿态估计。目标是先把「摄像头感知 -> 离散状态 -> 本地帧驱动」整条链路跑通，方便后续继续扩展成更完整的 VTuber 系统。
