# PROGRESS — 进程备忘

> 工作备忘,不属于用户文档。主文档看 [README.md](README.md)。
> 最后更新: 2026-08-30

## 当前状态一句话

主程序特殊状态系统 + 三档头位 + 感知层物理修复全部完成(合成测试通过);
素材提取工具首次实战 12 个视频,导出 23 张人工核验帧。
**卡点:两项用户实测(main.py 手感 / annotate 复查),验完即 MVP v0.2。**

## 已完成

### 主程序
- [x] 特殊状态五件套:离开 / 返回 / 撑脸 / 喝水 / 前倾
  - 撑脸:手部 10 关键点 × 扩大脸框区域判定(旧"手腕-鼻子距离"方案作废)
  - 喝水:嘴前窄区域(托下巴不误判)
  - 离开/返回:动作语义,丢脸 2s / 回归确认 0.7s,修了 returning 卡死 bug
  - 前倾:脸高 vs 校准基线,118% 进 / 110% 出迟滞带
  - 全部帧数防抖(Debounced Schmitt)
- [x] 三档头位 L1/L2/L3(11.5°/35.5°/55°)+ pitch,共 9 向;真素材缺 L3 时回退 L2 画面
- [x] 嘴/眼比例分母改脸高(根治侧脸误判 open 的几何问题)
- [x] debug 面板:脸框/嘴区叠加、Mouth raw 读数、DRINK/HOF/LEAN 信号
- [x] 占位帧:drinking / lean_forward 各 81 张(9 向 × 3 嘴 × 3 眼)

### 素材提取工具 (tools/face_asset_extractor)
- [x] 12 视频 → 1344 帧 → 检出 100% → 聚类 400 簇
- [x] 多模态预标注 100 张(按三档规格,极端帧 L3/R3)
- [x] 修复:抠图吃衬衫(自适应背景阈值)
- [x] 修复:导出选后脑勺帧 → 最典型帧(类中心最近)
- [x] 修复:置信度公式永不及格的数学 bug → 真间隔置信度,低置信不进导出
- [x] 特征重写:眼/嘴小裁剪区 + eye_blob_aspect / mouth_red_ratio(消融选定 13 特征)
- [x] LOO 基线:head 43% / eye 63% / mouth 63%(9/3/3 分类)→ 自动标签只当草稿,
      低置信永不进导出,人工复查是质量闸门
- [x] 标注键:E=L3, R=R3

## 待办

### 用户侧(卡点,MVP v0.2 前必须)
- [ ] `python main.py` 实测:五个特殊状态手感;记下 Mouth raw 三读数(闭/微张/大张)
      → 校准 `MOUTH_HALF_THRESHOLD`(0.30) / `MOUTH_OPEN_THRESHOLD`(0.70)
- [ ] `annotate --workdir work --budget 200` 复查:重点眼状态(反光眼闭/半睁难分);
      复查后 fit → classify → export,可信状态数随标注增长

### 工程侧(不阻塞,可自主推进)
- [ ] 眼状态特征继续迭代(当前 LOO 63%,二次元眼是硬问题)
- [ ] 丢脸三分级处理(<0.5s 保持 / 0.5-2s 遮脸 / >2s 离开)——见旧记忆 no-face-edge-cases
- [ ] CAM/SIM 切换阈值调优(旧记忆 dual-mode-tuning:中等角度过早切 SIM)
- [ ] 过渡帧系统预备(TRANSITION_FRAMES_ENABLED 仍为 False,等素材齐)
- [ ] 5×5 网格 / roll / variant 解封(素材到位后开 HEAD_V1_MODE=False)

## 校准速查

| 现象 | 调哪里 |
|---|---|
| 闭嘴误判 half | `MOUTH_HALF_THRESHOLD` 调大(现 0.30) |
| 大张不判 open | `MOUTH_OPEN_THRESHOLD` 调小(现 0.70) |
| 侧脸嘴乱跳 | 不用调,脸高分母已免疫;极端侧走 SIM |
| 稍偏就 L1 / L2 太远 | `HEAD_YAW_EDGES`(现 0.20/0.62/0.96 rad) |
| 前倾太灵敏 | `LEAN_SCALE_THRESHOLD`(现 1.18) |
| 撑脸太敏感 | `HAND_ON_FACE_MIN_POINTS`(现 3) |
| 喝水/托下巴混 | `DRINK_MOUTH_HALF_H` 调小(现 0.30) |

## 已知限制(诚实清单)

- 自动分类眼状态 LOO 63%:提取帧的眼标签要靠人工复查,不是 bug 是特征现实
- 视频提取素材嘴部可只标 closed/open,half 由手绘差分补(最近邻兜底)
- 摄像头真实链路只能用户验证(agent shell 无相机权限)

## 关键决策记录

| 决策 | 理由 |
|---|---|
| 自动分类标签只当草稿,低置信帧不进导出 | LOO 实测 50-63%,宁缺毋滥;人工标注永远是标准 |
| 素材左右约定:脸朝画面右 = R | 与主程序镜像视图一致(摄像头帧已水平翻转) |
| 嘴表现层三态,素材提取放宽 | 表现多态是设计意图;视频素材不严求,half 手绘补 |
| 动作(一次性) vs 状态(持续)语义分离 | 离开/返回需确认间隔;撑脸/喝水/前倾有进出场防抖 |
| exporter 选"最接近类中心"而非"最清晰" | 最清晰偏向头发纹理丰富的背面帧 |
| 占位帧不入库(frames_placeholder/ 在 .gitignore) | generate_placeholders.py 随时重生成 |

## 踩坑备忘

- **特征在 `cluster` 步骤生成**(不是 detect);改 features.py 后需重跑 cluster
- **重跑 cluster 后 candidates.json 顺序会变** — 一切引用必须用帧号,
  不要用候选位置(踩过:按位置还原标注导致 8 帧误标,靠原始帧号记录才修复)
- agent 调工具用项目根 venv: `../../.venv/bin/python`(工具目录内没有自己的 venv)
- 12 个源视频在 `tools/face_asset_extractor/input_videos/`(gitignore 不入库)
- 用户素材目录约定: 项目根 `video/`(gitignore)— `normal/` 基础状态,
  `drink/` `front/`(前倾) `handOnFace/` `leaveAndBack/` 留给特殊状态素材(待填);
  填好后走同一条提取管线,产出对应 theme 目录的帧
- 素材角色:红发蓝内层动漫少女,白底 grok 生成视频,背景 V≈251/S=0(衬衫 V≈237/S=23)
