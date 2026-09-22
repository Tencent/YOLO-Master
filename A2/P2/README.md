# P2 — 扫描 + 扩展（stretch）

> **⏸️ 已阶段性结项（2026-09-11）**：Phase 1 筛参完成 topk16（-0.15pp）、beta8（-0.01pp）；剩余 topk20/expand2/expand05/beta4/alpha1、Mosaic 交互、120ep 认定均未做。详见 [A2/README.md](../README.md)「结项结论」。

> 目标：(1) 扫描面积阈值 / 放宽幅度 / topk / α·β，出参数敏感性说明（验收 #4）；(2) Mosaic on/off 交互对照（验收 #6）；(3) 第二数据集 或 seg/pose。
> P1 结论「更多正样本 ≠ 成比例 AP 提升」，P2 用 30-epoch 短周期**筛参**找出甜点，再 120-epoch 复现协议**认定**；Mosaic 交互对照独立于参数扫描，插在「扫描」与「认定」之间。

## 前提（本轮已完成）

- 全部超参 config-driven（`default.yaml` 键），可直接扫：
  - `stal_area_small` / `stal_area_medium`（面积阈值）
  - `stal_topk_small`（small 档 top-k）、`stal_expand`（候选扩张半径）
  - **`tal_alpha` / `tal_beta`（TAL 度量 `score^α · iou^β`，本轮新增，默认 0.5 / 6.0）** ← P1 只读不动，P2 独立消融
- 入口脚本 `A2/P0/baseline_train.py` 新增 `--tal-alpha` / `--tal-beta` / `--optimizer` / `--warmup-bias-lr` 覆盖，支持 `--resume` 暂停/续跑。
- 新增 `tests/test_default_config_integrity.py::test_tal_alpha_beta_type_and_value_checked`，单测 34 项通过。

## 基准

冠军配置（同 P1 adaptive 冠军）：`stal_mode=adaptive, stal_min_positive=true, stal_topk_small=13, stal_expand=1.0, tal_topk=10, tal_alpha=0.5, tal_beta=6.0`。

| 基准 | 优化器 | epochs | APs | val mAP50-95 |
|---|---|---|---|---|
| P1 冠军 | MuSGD (auto) | 120 | **0.0925** | — |
| **P2 30ep 筛参基准** | **MuSGD（显式）** | **30** | **0.0603** | **0.15452** |

P1 冠军 APs=0.0925 是认定线（+0.20pp vs fixed、+0.50pp vs 纯 TAL）；**P2 筛参以 30ep MuSGD 的 APs=0.0603 为排序基准**，120ep 复现后才谈认定。

## Phase 1 — 30-epoch 粗筛（全量数据，FP32，seed 0，imgsz 800，batch 6，**MuSGD**）

单变量敏感性（每次只偏离冠军一个维度），30 epoch 只作**排序**、不作认定：

| run | 变更维度 | 值 | 假说 |
|---|---|---|---|
| topk16 | `stal_topk_small` | 16 | 13 是否已达正样本上限，继续加是否还有增益 |
| topk20 | `stal_topk_small` | 20 | 正样本上限探底 |
| expand2 | `stal_expand` | 2.0 | 更大候选区域（更多低质量锚点） |
| expand05 | `stal_expand` | 0.5 | 更紧区域，减少低质量锚点 |
| beta4 | `tal_beta` | 4.0 | 降低 IoU 权重——小框 IoU 噪声大，β=6 过度压制 |
| beta8 | `tal_beta` | 8.0 | 提高 IoU 权重——验证 β=6 是否已是甜点，继续加大是否更差 |
| alpha1 | `tal_alpha` | 1.0 | 提高分类权重，弱化 IoU 主导 |

### 结果（MuSGD 30ep，进行中）

| run | APs | Δ APs（vs 基准 0.0603） | val mAP50-95（训练期） |
|---|---|---|---|
| 基准（topk13 冠军） | 0.0603 | — | 0.15452 |
| topk16 | 0.0588 | **-0.15pp** | 0.15372 |
| beta8 | 0.0602 | -0.01pp | 0.15417 |

- **topk16**：small 档候选 13→16，APs 反而 **-0.15pp**（overall -0.48pp）——印证「更多正样本 ≠ 成比例 AP 提升」，
  13 已接近正样本上限甜点，继续加候选边际递减甚至负收益。
- **beta8**：IoU 权重 β 6→8，APs 几乎不变（-0.01pp，噪声内）——β=6 已处平台期，继续加大 IoU 权重对小框无增益。

> **topk16 / beta8 已按 MuSGD 口径重跑**（结果见上表；旧 AdamW 结果 APs≈0.034 作废）。其余 topk20 / expand2 / expand05 / beta4 / alpha1 未跑。

## Phase 1b — Mosaic on/off 交互对照（验收 #6，30-epoch）

Mosaic 直接改变训练时目标面积，可能放大/削弱 STAL。按验收口径「至少 baseline/STAL × Mosaic on/off」做 2×2
交互对照，**不与 Phase 1 的扫参矩阵全叉乘**。STAL 参数固定冠军默认（`tal_topk=10, tal_alpha=0.5, tal_beta=6.0`），
只让 `stal_mode` 与 `mosaic` 两个维度变化：

| run | `stal_mode` | `mosaic` | 说明 |
|---|---|---|---|
| fixed-on | `fixed` | 1.0 | baseline（既有 stride clamp）× mosaic on |
| fixed-off | `fixed` | 0 | baseline × mosaic off |
| adaptive-on | `adaptive` | 1.0 | **已跑 = 30ep 基准（APs=0.0603）** |
| adaptive-off | `adaptive` | 0 | 新增机制 × mosaic off |

- 读法：fixed-on↔fixed-off 看 mosaic 对 baseline 的独立影响；adaptive-on↔adaptive-off 看 mosaic 对 STAL 的影响；
  两组 on↔off 的差值是否一致，判断 mosaic 是否与 STAL 交互。
- 30-epoch 只作**交互诊断**、不作 APs 认定；若 on/off 差值在两组间明显不同（明显交互），再挑相应组补 120-epoch。
- **前置**：`A2/P0/baseline_train.py` 需新增 `--mosaic` / `--close-mosaic` 透传（当前脚本未暴露，跑前补上）。

排程：Phase 1 参数扫描跑完后执行（单卡与扫描串行），随后进入 Phase 2 认定。

## Phase 2 — 120-epoch 复现认定

Phase 1 排名前 1–2 名跑满 120 epoch（协议见 `A2/P0/tiered_eval.py` + `P0/README.md`），
只有 120-epoch 结果才能认定「APs 提升」。

## Phase 3 — α×β 交互 / warmup（stretch，视 Phase 1 结果）

- α×β 组合消融（如 β∈{4,6} × α∈{0.5,1.0}）。
- warmup 曲线（STAL 参数随 epoch 渐进放开）——需新增机制，非纯配置可扫。
- 第二数据集 或 seg/pose（分叉决策，延后）。

## 复现命令（Phase 1，MuSGD 口径）

```bash
# 基准（冠军配置 30ep）
python A2/P0/baseline_train.py --epochs 30 --imgsz 800 --batch 6 --device 0 --amp false --seed 0 \
  --stal-mode adaptive --stal-min-positive true --stal-topk-small 13 --stal-expand 1.0 \
  --tal-alpha 0.5 --tal-beta 6.0 --optimizer MuSGD --warmup-bias-lr 0.0 \
  --project "D:/CODE/smoke/runs/a2/p2" --name visdrone-p2-base30-musgd-v01n

# 其余按上表只改一个维度（--stal-topk-small / --stal-expand / --tal-beta / --tal-alpha），
# 其余参数与基准完全一致，尤其保留 --optimizer MuSGD --warmup-bias-lr 0.0

# Mosaic 交互对照（Phase 1b）：固定冠军默认参数，只改 --stal-mode 与 --mosaic 0
# （前置：baseline_train.py 需先加 --mosaic / --close-mosaic 透传）
python A2/P0/baseline_train.py --epochs 30 --imgsz 800 --batch 6 --device 0 --amp false --seed 0 \
  --stal-mode fixed --optimizer MuSGD --warmup-bias-lr 0.0 --mosaic 0 --close-mosaic 0 \
  --project "D:/CODE/smoke/runs/a2/p2" --name visdrone-p2-fixed-off-musgd-v01n
```

## 风险 / 口径

- 30-epoch 仅筛参；close_mosaic=10 → 仅 ~20 个 post-mosaic epoch，APs 排名有噪声，Phase 2 必须复现。
- α/β 是全局 TAL 参数，改它会同时影响 medium/large 档；主指标仍盯 APs，但会一并报 APm/APl 观察溢出。
- 训练期 / 评测期面积口径不同（见 `stal-eval-protocol`），勿混用。
- `train/gt_small` 是 mosaic 相关的计数（mosaic ON ~470k / OFF ~272k），**跨 run 对比必须取同一 mosaic 阶段**（30ep 取 ep1–20 或 ep21–30 各自对齐），否则会误判为异常。
