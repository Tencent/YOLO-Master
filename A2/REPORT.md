# A2 研究报告：STAL 小目标自适应标签分配

> 任务代号 **A2**：STAL（Small-object Task-aligned Label Assignment，面积感知任务对齐标签分配）。
> 模型：YOLO-Master v0.1-N（MoE 7.5M）。数据：VisDrone2019-DET。日期：2026-09。

---

## 摘要

本报告研究面向密集小目标（VisDrone）的**面积感知任务对齐标签分配**（STAL）方法，在 Ultralytics
`TaskAlignedAssigner` 之上新增三套可选机制——候选区域扩展、面积/尺度自适应 top-k、最小正样本保障——
用于改善小目标的正样本覆盖。在 YOLO-Master v0.1-N 上、完整 VisDrone2019-DET、120 epoch 协议下，
三组对照实验（纯 TAL / fixed / adaptive）表明：**adaptive STAL 将小目标平均正样本数提升 ×1.88（vs fixed）、
×2.20（vs 纯 TAL），零正样本小目标比例从 7.05% 降至 1.50%**，但主指标 **APs 仅 +0.20pp（vs fixed）**，
未达任务书「APs ≥ +1.0pp」目标。本文给出方法细节、完整实验结果，并分析「正样本覆盖提升未充分兑现为
AP 增益」的原因。

---

## 1. 背景与问题定义

### 1.1 任务对齐标签分配（TAL）

YOLO-Master 沿用 Ultralytics 的 `TaskAlignedAssigner`（源自 TOOD 的任务对齐思想）：对每个 GT，
用对齐度量 `score^α · iou^β`（α=分类指数，β=定位指数）对候选锚点打分，取 top-k 作为正样本，
再经重叠冲突消解得到最终分配。该机制在通用目标检测上效果良好，但**未显式区分目标尺度**。

### 1.2 小目标的正样本困境

小目标（COCO 口径 < 32² px）在 TAL 下存在结构性劣势：

1. **候选稀疏**：小框经 stride 下采样后面积过小，落入框内的锚点中心数量少，正样本候选池天然不足；
2. **零正样本**：极端小框在冲突消解后可能被剥离到 0 个正样本，完全失去监督信号；
3. **IoU 噪声大**：小框的 IoU 对定位误差敏感，对齐度量不稳定。

VisDrone 是典型密集小目标场景，该困境被放大。本课题即针对此，在不改动模型架构、不调 α/β 的前提下，
通过**分配几何**层面的面积感知机制改善覆盖。

---

## 2. 方法

STAL 在 `ultralytics/utils/tal.py::TaskAlignedAssigner` 内新增三套**可选**机制，由 `stal_mode`
枚举统一开关（`none`=纯 TAL / `fixed`=仓库既有 stride clamp / `adaptive`=全部新机制）。

### 2.1 候选区域扩展（`_expand_small_candidates`）

small 档 GT 框每边外扩 `stal_expand × stride[0]` px（默认 `stal_expand=1.0`，即外扩 8 px）。
外扩只作用于「面积 < `stal_area_small`」的框，让更多锚点中心落入框内，扩大候选池。

### 2.2 面积/尺度自适应 top-k（`_adaptive_topk`）

small 档 GT 的候选数从基础 `tal_topk=10` 提升到 `stal_topk_small=13`，其余档保持 10。
实现为 per-GT 的 `k` 张量（`(b, n_boxes)`），在 `select_topk_candidates` 内对已排序的 top-k 轴按列截断，
全程向量化、无 GPU 同步。

### 2.3 最小正样本保障（`_ensure_min_positive`）

冲突消解后仍为 0 正样本的 GT，强制提升一个「框内最高 IoU 且未被占用」的锚点为正。关键约束：

- 只提升**未被占用**的框内锚点，从不抢占已归属其它 GT 的正样本，保持 one-anchor-one-GT 不变量；
- 若某 GT 的所有框内锚点都已被占用，则放弃（不可在不破坏不变量的前提下恢复）。

### 2.4 分档口径

小/中/大分档判定使用「训练期 resize 后、clamp/expand 之前」的原始框面积，与 `_adaptive_topk` 一致，
避免「先撑大再判 small」的分档漂移；评测侧则使用原始验证图 GT bbox 面积（见 §3.2）。

### 2.5 配置契约

全部参数 config-driven（`ultralytics/cfg/default.yaml`），并做类型/枚举/排序校验：

| 键 | 默认 | 说明 |
|---|---|---|
| `tal_topk` | 10 | 基础 top-k |
| `tal_alpha` | 0.5 | `score^α` 分类指数（P1 不动） |
| `tal_beta` | 6.0 | `iou^β` 定位指数（P1 不动） |
| `stal_mode` | `fixed` | `none`/`fixed`/`adaptive` |
| `stal_min_positive` | `False` | ≥1 正样本保障 |
| `stal_area_small` | 1024 | 32² 训练期面积阈值 |
| `stal_area_medium` | 9216 | 96² 训练期面积阈值 |
| `stal_topk_small` | 13 | adaptive 小目标 top-k |
| `stal_expand` | 1.0 | 候选扩张半径（stride[0] 倍数） |

默认 `stal_mode=fixed`，新机制需显式开启，默认行为与仓库既有实现一致。

---

## 3. 实验设置

### 3.1 训练协议

| 项 | 设置 |
|---|---|
| 模型 | YOLO-Master v0.1-N（MoE 7.5M，from-scratch，无预训练权重） |
| 数据 | VisDrone2019-DET（train=6471） |
| 输入 | imgsz 800，batch 6 |
| 训练 | 120 epoch，seed 0，FP32（`--amp false`），patience 0 |
| 优化器 | MuSGD（auto：iterations=102×epochs>10000 → MuSGD） |

### 3.2 评测协议

- 主指标 **APs@[.50:.95]**：COCO-style 32²/96² 分档（small<32²，medium 32²–96²，large≥96²），maxDets=500；
- 该分档为**补充口径**，非 VisDrone 官方（官方只报 AP/AP50/AP75 + AR@1/10/100/500）；
- 评测面积按原始验证图 GT bbox；训练 STAL 尺度按增强+resize 后进入 assigner 的实际尺寸；
- 每 epoch 正样本统计随训练写入 `results.csv`（`fg_sum`/`fg_{tier}`/`avg_pos_{tier}`/`zero_ratio_{tier}`）。

---

## 4. 结果与分析

### 4.1 三组对照（主结果，120ep）

| 组 | `stal_mode` | APs | APm | APl | AP50s | ARs@500 |
|---|---|---|---|---|---|---|
| 纯 TAL | `none` | 0.0875 | 0.2569 | 0.2449 | 0.1878 | 0.1947 |
| fixed-stride STAL | `fixed` | 0.0905 | 0.2569 | 0.2369 | 0.1965 | 0.2001 |
| **adaptive STAL** | `adaptive` | **0.0925** | 0.2567 | 0.2415 | **0.2003** | **0.2022** |

Δ APs：fixed vs 纯 TAL **+0.30pp**；adaptive vs 纯 TAL **+0.50pp**；adaptive vs fixed **+0.20pp**。

### 4.2 正样本覆盖（P1 核心目标，epoch 120 收敛态 small 档）

| 组 | avg_pos_small | zero_ratio_small | fg_small |
|---|---|---|---|
| 纯 TAL | 3.464 | 19.41% | 0.947M |
| fixed | 4.055 | 7.05% | 1.101M |
| **adaptive** | **7.627** | **1.50%** | 2.082M |

- adaptive vs fixed：平均正样本 ×1.88，零正样本比例降到约 1/4.7；
- adaptive vs 纯 TAL：平均正样本 ×2.20，零正样本从 19.41% 降到 1.50%（约 1/13）；
- 覆盖改善自 epoch ~30 起即稳定，说明来自分配机制本身而非训练早期噪声。

### 4.3 训练期指标

三组训练期 val mAP50-95 基本持平：纯 TAL 0.21138（ep111）/ fixed 0.21236（ep110）/
adaptive 0.21180（ep106）。印证 STAL 只重排小目标分配，对总体指标几乎无影响——其收益集中在小目标档。

### 4.4 P2 参数敏感性（30ep 筛参，2/7 完成）

| run | 变更 | Δ APs（vs 基准 0.0603） | 结论 |
|---|---|---|---|
| topk16 | `stal_topk_small` 13→16 | **-0.15pp** | 13 已接近正样本上限甜点 |
| beta8 | `tal_beta` 6→8 | -0.01pp（噪声内） | β=6 处平台期 |

---

## 5. 讨论

### 5.1 为什么覆盖提升未兑现为 AP 增益？

这是本课题最核心的发现。adaptive 将平均正样本翻了近一倍、零正样本锐减到 1/13，但 APs 只 +0.20pp。
原因在于：**扩出的候选里大量是低质量锚点**——外扩候选区与提升 top-k 引入的正样本，其对齐度量、
IoU 与定位质量参差不齐，对最终回归/分类的边际贡献递减。即「更多正样本 ≠ 成比例 AP 提升」。
topk16 的负收益（-0.15pp）进一步印证：在覆盖已饱和后继续加候选，只会引入噪声正样本。

### 5.2 口径与实现注意事项

- **面积口径双轨**：训练期用 resize 后、clamp 前的原始面积；评测期用原始验证图面积，二者不可混用；
- **优化器耦合**：`optimizer=auto` 下迭代数 `=ceil(6471/max(batch,nbs))×epochs=102×epochs`，
  30ep 会走 AdamW 而 120ep 冠军走 MuSGD，短周期筛参需显式 `--optimizer MuSGD` 才能与冠军可比；
- **FP32/AMP 一致**：单测已核查 FP32/AMP 下 mask/top-k/正样本数/loss/梯度一致有限，复现统一 FP32。

### 5.3 局限

- 主指标未达任务书「APs ≥ +1.0pp」，结论仅能表述为「覆盖显著改善 + APs 小幅正收益」；
- 参数敏感性仅完成 2/7；Mosaic on/off 交互对照未做；120ep 复现认定未完成（P2 为阶段性结项）；
- α/β 只做了 β=8 单点，未做 α×β 组合与 warmup 曲线。

---

## 6. 结论与展望

**结论**：面积感知标签分配能**显著改善小目标正样本覆盖**（×1.88 平均正样本、零正样本 1/4.7），
是「覆盖层面」的成功；但**APs 增益有限**（+0.20pp），未兑现为「精度层面」的目标，说明 TAL 的
小目标瓶颈不完全在正样本数量，更在于候选质量与后续训练信号。

**后续方向**：

1. 完成剩余扫参（topk20 / expand2 / expand05 / beta4 / alpha1）与 Mosaic 交互对照、120ep 认定；
2. **候选质量**：不盲目扩候选，改为对扩出的低质量锚点做质量门控（对齐度量阈值过滤）；
3. **α/β 联合**：小框 IoU 噪声大，可探索小档专属的 `α/β`（当前 α/β 全局生效，改它会溢出到中/大档）；
4. **warmup 曲线**：STAL 参数随 epoch 渐进放开，缓解早期强分配带来的不稳定性。

---

*本报告与 [README.md](README.md)（阶段总览）、[P0/](P0/README.md)（基线）、[P1/](P1/README.md)（对照）、
[P2/](P2/README.md)（扫参）配套，数据均来自各阶段 runs 的 `results.csv` 与 `tiered_eval.py` 输出。*
