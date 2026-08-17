# Face Asset Extractor

从二次元角色视频自动提取「透明 PNG 状态帧」，输出命名直接对齐主程序
`frames_placeholder/` 结构，拖过去即可使用。

## 工作流

```
视频 → 抽帧(480p) → 白底抠图 → 启发式脸部检测 → 特征提取
     → 聚类去重 → 人工标注(≤100张) → 阈值学习 → 自动分类 → 导出PNG
```

推荐流程（先小批量试跑，再大范围导入）：

```bash
# 1. 试跑 10 个视频
python extractor.py extract  --videos /path/to/10_videos --workdir work
python extractor.py detect   --workdir work
python extractor.py cluster  --workdir work
python extractor.py annotate --workdir work --budget 100

# 2. 从标注学习你的分类标准（微转/转动/大幅度的边界由你定义）
python extractor.py fit      --workdir work
python extractor.py classify --workdir work
python extractor.py export   --workdir work --out output

# 3. 大范围导入：新 workdir 跑 extract/detect/cluster 后
#    复用旧 classifier.json 直接 classify + export
```

## 标注键盘

| 键 | 含义 |
|---|---|
| 1-7 | 朝向：center / L1 / L2 / R1 / R2 / U1 / D1 |
| A/S/D | 眼睛：open / half / closed |
| Z/X/C | 嘴巴：closed / half / open |
| ←/→ | 前后翻帧 |
| N/P | 下一个/上一个未标注 |
| Enter | 保存 |
| Backspace | 清除当前标注 |
| Q/Esc | 退出（自动保存） |

## 设计说明

- **脸部检测是启发式的**（白底剪影 + 深色斑块 = 眼睛/嘴），AI 生成视频
  的抖动会让检测偶尔出错——标注界面可以人工纠正，分类器只用你的标注
  学习，不依赖检测的完美。
- **分类器 = 最近质心**：你标注的样本直接定义「什么是 L1 / 什么是 half」。
- **抠图 = 白底 flood fill**：只移除与画面边缘相连的白色区域，角色衣服
  上的白色保留。
- **输出分辨率与视频一致**（720p 视频自动降到 480p）。

## 可扩展点

- `face_detect.py`：接口只有 `detect(bgr, alpha) → FaceDetection`。
  未来接入 anime-face-detector 等 ML 模型只需替换这个模块。
- `features.py`：新增特征只需加进 `FEATURE_NAMES`。
- 特殊动作（撑脸/喝水/前后移动）：`classifications.json` 里已有
  `theme` 字段预留，未来加分类器即可。
