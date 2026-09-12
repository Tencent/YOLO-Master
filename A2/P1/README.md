# P1 — STAL 面积感知 on/off 对照（核心，已完成）

> **结论一句话**：adaptive STAL 显著改善小目标正样本覆盖（平均每小目标正样本 ×1.9、零正样本比例从 7.05% → 1.50%），
> 但主指标 APs 增益有限（相对 fixed 基线 **+0.20pp**、相对纯 TAL **+0.50pp**），未达任务书「APs ≥ +1.0pp」。
> **覆盖层面达标，AP 增益未兑现**——P2 部分扫参（topk16 -0.15pp、beta8 -0.01pp）亦无增益，项目已阶段性结项（见 [A2/README.md](../README.md)「结项结论」）。

## 三组对照（主结果）

统一协议：YOLO-Master v0.1-N（MoE，7.5M）/ imgsz 800 / 120 epoch / batch 6 / seed 0 / FP32（`--amp false`）/
patience 0。评测用 `A2/P0/tiered_eval.py`（方形 800，maxDets=500，COCO-style 32²/96²，非 VisDrone 官方分档）。

| 组 | `stal_mode` | APs (small) | APm | APl | overall AP50-95 | AP50s | ARs@500 |
|---|---|---|---|---|---|---|---|
| 纯 TAL | `none` | 0.0875 | 0.2569 | 0.2449 | 0.2174 | 0.1878 | 0.1947 |
| fixed-stride STAL | `fixed` | 0.0905 | 0.2569 | 0.2369 | 0.2199 | 0.1965 | 0.2001 |
| **adaptive STAL** | `adaptive` | **0.0925** | 0.2567 | 0.2415 | 0.2196 | **0.2003** | **0.2022** |

**主指标 APs 绝对百分点差（Δ APs）：**

| 对比 | Δ APs |
|---|---|
| fixed vs 纯 TAL | +0.30pp |
| adaptive vs 纯 TAL | **+0.50pp** |
| adaptive vs fixed | **+0.20pp** |

训练期 val mAP50-95（best epoch）：纯 TAL 0.21138（ep111）/ fixed 0.21236（ep110）/ adaptive 0.21180（ep106），三者基本持平，
进一步印证 STAL 只重排小目标分配，对总体指标几乎无影响。

## 正样本覆盖演化（P1 核心目标）

每 epoch 正样本统计随训练写入 `results.csv`（`train/fg_sum` / `train/fg_{small|medium|large}` /
`train/avg_pos_{tier}` / `train/zero_ratio_{tier}`），收敛态（epoch 120，三组均已 close_mosaic）small 档：

| 组 | avg_pos_small（平均每小目标正样本） | zero_ratio_small（零正样本小目标比例） | fg_small |
|---|---|---|---|
| 纯 TAL | 3.464 | 19.41% | 0.947M |
| fixed | 4.055 | 7.05% | 1.101M |
| **adaptive** | **7.627** | **1.50%** | 2.082M |

- **adaptive vs fixed**：平均正样本 ×1.88（7.63 vs 4.06），零正样本小目标比例降到约 1/4.7（1.50% vs 7.05%）。
- **adaptive vs 纯 TAL**：平均正样本 ×2.20，零正样本小目标从 19.41% 降到 1.50%（约 1/13）。
- 演化从 epoch ~30 起即稳定（adaptive 的 avg_pos 在 7.0 附近、zero_ratio 在 2% 附近小幅波动），
  说明改善来自分配机制本身，不是训练早期的噪声。

**关键洞察**：小目标正样本覆盖提升了近 2 倍、零正样本小目标锐减到 1/13，但 APs 只 +0.50pp。
即「更多正样本 ≠ 成比例 AP 提升」——扩出的候选里大量是低质量锚点，对最终 mAP 的边际贡献递减。
这正是 P2 需要扫参（topk/expand/α/β）找甜点的原因。

## 实现机制（三件套，均在 `ultralytics/utils/tal.py` `TaskAlignedAssigner`）

| 机制 | 函数 | 行为 |
|---|---|---|
| 候选区域扩展 | `_expand_small_candidates` | small 档 GT 框每边外扩 `stal_expand × stride[0]` px，让更多锚点中心落进框内 |
| 面积/尺度自适应 top-k | `_adaptive_topk` | small 档 GT 候选数从 `tal_topk=10` 提到 `stal_topk_small=13` |
| 最小正样本保障 | `_ensure_min_positive` | 冲突消解后仍 0 正样本的 GT，强制升一个「框内最高 IoU 且未被占用」的锚点为正 |

- 面积分档判定用**训练期 resize 后、clamp/expand 之前**的原始框面积（`original_gt_bboxes`），
  与 `_adaptive_topk` 口径一致，避免「先撑大再判 small」的分档漂移。
- α/β 与 IoU 权重（`get_box_metrics` 的 `score^α · iou^β`）**本轮不动**，留 P2 独立消融。
- `select_candidates_in_gts` 的 fixed-stride clamp 保留（`fixed` 模式 = 仓库既有行为），
  `none` 模式完全关闭全部 STAL 钩子（等价纯 TAL）。

## 配置契约（config-driven，P2 可扫描）

`ultralytics/cfg/default.yaml`（L169–175）：

| 键 | 默认 | 说明 |
|---|---|---|
| `tal_topk` | 10 | 基础 top-k |
| `stal_mode` | `fixed` | `none`/`fixed`/`adaptive`（默认保持仓库既有行为，新机制显式开启） |
| `stal_min_positive` | `False` | ≥1 正样本保障（adaptive 显式开启） |
| `stal_area_small` | 1024 | 32² 训练期面积阈值 |
| `stal_area_medium` | 9216 | 96² 训练期面积阈值 |
| `stal_topk_small` | 13 | adaptive 小目标 top-k |
| `stal_expand` | 1.0 | adaptive 候选扩张半径（stride[0] 倍数） |

校验在 `ultralytics/cfg/__init__.py` `validate_stal_config`：`stal_mode` 枚举校验、`stal_area_small < stal_area_medium`
排序校验、各键类型校验。

## 单元测试（33 项通过，映射验收要求）

`python -m pytest tests/test_tal_stal.py tests/test_stal_loss.py tests/test_default_config_integrity.py -q` → **33 passed**。

| 验收要求 | 覆盖测试 |
|---|---|
| 关闭 STAL 与纯 TAL 等价 | `test_none_is_pure_tal`、`test_stal_none_matches_pure_tal_no_rescue` |
| 面积阈值边界 | `test_adaptive_topk_area_boundary`、`test_record_tier_positives_exact_area_boundaries` |
| 空 GT | `test_empty_gt_early_return` |
| 极小框 | `test_fixed_stride_rescues_tiny_gt`、`test_select_candidates_in_gts_expands_thin_small_box` |
| 重叠 GT / 正样本冲突 | `test_select_highest_overlaps_resolves_conflict`、`test_ensure_min_positive_does_not_steal_foreground_anchor`、`test_min_positive_recovers_gt_stripped_by_conflict` |
| FP32/AMP 有限值 + 一致性 | `test_assigner_forward_finite_{fp32,mixed_precision,with_min_positive}`、`test_loss_and_grad_finite_{fp32,amp}`、`test_assignment_consistent_fp32_vs_amp`、`test_fg_mask_consistent_fp32_vs_amp` |
| 配置契约 | `test_stal_defaults_parse_with_expected_types`、`test_stal_mode_is_enum_checked`、`test_stal_area_ordering_is_validated`、`test_stal_keys_are_type_checked` 等 |

## 复现命令

```bash
# 三组（batch=6 按显存自定，三组须一致；--amp false 统一 FP32 只让分配方式单变量变化）
python A2/P0/baseline_train.py --epochs 120 --imgsz 800 --batch 6 --device 0 --amp false --seed 0 --stal-mode none     --name visdrone-pure-tal-v01n
python A2/P0/baseline_train.py --epochs 120 --imgsz 800 --batch 6 --device 0 --amp false --seed 0 --stal-mode fixed    --name visdrone-baseline-v01n   # = P0 基线
python A2/P0/baseline_train.py --epochs 120 --imgsz 800 --batch 6 --device 0 --amp false --seed 0 --stal-mode adaptive --stal-min-positive true --name visdrone-adaptive-tal-v01n

# 分档评测（每组一次，取 best.pt）
python A2/P0/tiered_eval.py --weights runs/a2/p0/<name>/weights/best.pt --imgsz 800 --device 0
```

## 风险与降级

- **Mosaic 交互**：mosaic 会改变训练目标面积，可能放大/削弱 STAL。本轮三组都用默认 `mosaic=1.0`/`close_mosaic=10`，
  未做 Mosaic on/off 交互对照（验收口径为「至少做 baseline/STAL × Mosaic on/off」）；至项目阶段性结项仍未做。
- **AMP**：已由 `test_assignment_consistent_fp32_vs_amp` / `test_fg_mask_consistent_fp32_vs_amp` 核查 FP32/AMP 下
  mask、top-k、正样本数一致且有限；复现统一跑 FP32，不牺牲分配正确性。
- **主指标未达标**：APs +0.20pp（vs fixed）远低于 ≥1.0pp，结论只能表述为「正样本覆盖显著改善 + APs 小幅正收益」，
  不能表述为「显著提升检测精度」。P2 部分扫参（topk16 -0.15pp / beta8 -0.01pp）未找到增益，进一步印证此结论。
