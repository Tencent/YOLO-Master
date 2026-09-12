# P0 — VisDrone2019-DET 基线：COCO-style 分档指标 + 每 epoch 正样本统计

> 任务：STAL 面积感知标签分配。P0（保底）= 真实 VisDrone 基线 + small/medium/large 分档 AP + 每 epoch 正样本统计，
> 作为 P1 面积感知 on/off 对照的底座。

## 交付文件

| 文件 | 作用 |
|---|---|
| `A2/P0/prepare_visdrone.py` | 下载/复用 VisDrone2019-DET → 校验 split 计数 → 写本地 `datasets/VisDrone.yaml`（绝对路径） |
| `A2/P0/baseline_train.py` | YOLO-Master v0.1-N 从零训练 + 每 epoch 正样本统计回调（含 per-tier + 零正样本比）+ `--amp`/`--resume` 开关 |
| `A2/P0/tiered_eval.py` | 统一分档评测：small/medium/large AP + AR（COCO-style 32²/96²，maxDets=500） |
| `ultralytics/utils/loss.py` | 核心改动：`v8DetectionLoss` 增加 `fg_count` / `fg_count_by_stride` / per-tier 正样本计数 |
| `ultralytics/engine/trainer.py` | resume 白名单加入 `amp`（允许续跑覆盖精度） |
| `tests/test_tal_stal.py` / `tests/test_stal_loss.py` | STAL 单元测试（含 FP32/AMP 分配一致性 + loss/梯度有限值） |

## 复现命令

```bash
# 1. 数据（幂等：已存在则跳过下载，只校验 + 写本地 yaml）
python A2/P0/prepare_visdrone.py

# 2. 冒烟 / 短周期链路验证（均只验管线，不做 APs 结论）
#    (a) 冒烟 = 固定子集 1 epoch（A2/smoke，见 smoke/README.md）
python A2/smoke/smoke_test.py --device 0 --epochs 1 --imgsz 320 --batch 4
#    (b) 短周期链路验证 = 全量数据 1 epoch，确认 loss 埋点 + 回调列 + 分档评测整条链路
python A2/P0/baseline_train.py --epochs 1 --imgsz 320 --batch 8 --device 0 --name visdrone-link-1ep
python A2/P0/tiered_eval.py --weights runs/a2/p0/visdrone-link-1ep/weights/last.pt --imgsz 320 --device 0

# 3. 全量基线（复现协议：YOLO-Master v0.1-N = ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml，
#    imgsz 800 / 120 epoch / patience 0 / 完整 val 548；batch 按显存自定，三组对照须一致）
#    注：--amp false 统一跑 FP32，见「精度模式」
python A2/P0/baseline_train.py --epochs 120 --imgsz 800 --batch 6 --device 0 --amp false
python A2/P0/tiered_eval.py --weights runs/a2/p0/visdrone-baseline-v01n/weights/best.pt --imgsz 800 --device 0

# 3b. 暂停 / 续跑：中断后从 weights/last.pt 续跑（--name 须与首次一致，epochs 仍写总目标 120，--amp false 保持一致）
python A2/P0/baseline_train.py --resume --epochs 120 --imgsz 800 --batch 6 --device 0 --name visdrone-baseline-v01n --amp false
```

## 数据与环境

- 数据落在用户 settings 指定的 `D:\yolo\datasets\VisDrone`（非仓库 `datasets/`）。
- Split：train **6471** / val **548** / test-dev **1610**，10 类（VisDrone2019-DET 标准）。
- 环境：RTX 5070 Ti Laptop 12GB，torch 2.13.0+cu132，ultralytics 8.4.101。

## 指标口径

- **分档阈值**（COCO-style 32²/96²，本课题口径，非 VisDrone 官方定义——VisDrone DET 无面积分档，只报 AP/AP50/AP75/AR@1/10/100/500）：
  - `small` < 32²，`medium` 32²–96²，`large` ≥ 96²，另有 `all`。
  - **评测面积**按原始验证图像中的 GT bbox 计算；**训练期** STAL 的尺度判断按数据增强 + resize 后进入 assigner 的实际尺寸计算。
- **主指标**：APs@[IoU=.50:.95]（small 档 mAP50-95，maxDets=500）。
- **必报指标**（`tiered_eval.py` 输出）：总体 AP / AP50 / AP75 / AR@500 + 分档 APs / APm / APl + ARs@500。
- **正样本统计**（`results.csv` 新增列，随训练自动写入）：
  - `train/fg_sum`：每 epoch 正样本（前景锚点）总数。
  - `train/fg_s8` / `train/fg_s16` / `train/fg_s32`：按 FPN 层级（stride 8/16/32）分桶的正样本数。
  - `train/fg_{small|medium|large}` / `train/gt_{...}` / `train/zero_{...}` / `train/avg_pos_{...}` / `train/zero_ratio_{...}`：
    按 GT 面积档（训练期像素面积，32²/96²）分桶的正样本数、GT 数、零正样本 GT 数、每档平均正样本数、零正样本比例。

## 基线结果（P0 完成）

> YOLO-Master v0.1-N（MoE，7,516,742 参数），FP32（`--amp false`），120 epoch，imgsz 800，batch 6，patience 0，
> seed 0。best.pt = epoch 110（按训练期 val mAP50-95 选取）。

**训练期 `results.csv`（best epoch 110）**：mAP50-95 = **0.21236**，mAP50 = 0.36750，P = 0.48872，R = 0.37996。

**分档评测 `tiered_eval.py`（方形 800 输入，maxDets=500，COCO-style 32²/96²）**：

| tier | instances | AP50 | AP75 | AP50-95 | AR@500 |
|---|---|---|---|---|---|
| small | 26586 | 0.1965 | 0.0733 | **0.0905** | 0.2001 |
| medium | 11105 | 0.4065 | 0.2773 | 0.2569 | 0.4558 |
| large | 1068 | 0.3018 | 0.2687 | 0.2369 | 0.5738 |
| overall | 38759 | 0.3674 | 0.2241 | 0.2199 | 0.3079 |

- **主指标 APs = 0.0905**（small 档 AP@[.50:.95]），P1 三组对照据此报绝对百分点差（Δ APs）。
- `tiered_eval.py` 的 overall AP50-95（0.2199）与训练期 `results.csv`（0.21236）是两套 NMS 后处理，略有差异属预期；
  对比实验统一以 `tiered_eval.py` 为准。评测用方形 800 letterbox（`rect=False`），与训练期 val 一致。
- 正样本统计（epoch 120）：`fg_sum` 1.566M（s8 1.151M / s16 0.398M / s32 0.017M），small 档零正样本比 0.0705、平均正样本 4.055。

## 关键实现说明

- **正样本计数**：`v8DetectionLoss.get_assigned_targets_and_loss` 在 assigner 返回后累加 `fg_mask.sum()`，
  并按 `stride_tensor` 分桶；`_record_tier_positives` 按 GT 面积档（32²/96²）累计正样本 / GT / 零正样本。数值损失路径零改动。
- **上抛**：`model.add_callback("on_train_epoch_end", fn)` 把计数写入 `trainer.lr`（自动进 `results.csv`），
  `on_fit_epoch_end` 在 validate 之后清零（validation 复用训练 criterion，不清零会污染下一 epoch）。
- **分档评测**：standalone 脚本，COCO 语义 —— 检测按其命中 GT 的面积分档；未命中任何 GT 的检测在各档都算 FP；
  命中非本档 GT 的检测在该档忽略。总体与分档 AP/AR 出自同一份代码，口径自洽。

## 精度模式

- **复现统一 FP32**：基线、暂停/续跑、以及 P1 三组对照（纯 TAL / fixed-stride STAL / adaptive STAL）全部显式
  `--amp false`，只让「标签分配方式」这一个变量变化。
- `baseline_train.py` 的 `--amp` 开关默认不传（沿用 default.yaml 的 `amp: True`），`--amp false` 显式关闭混合精度；
  续跑时 `amp` 在 resume 白名单中，可随命令覆盖。
- FP32/AMP 下 STAL 分配一致性（mask / top-k / 正样本数）与 loss/梯度有限值由单元测试覆盖
  （`test_assignment_consistent_fp32_vs_amp`、`test_loss_and_grad_finite_{fp32,amp}`），选 FP32 不牺牲分配正确性。

## 风险与降级

- 下载受限 → `check_det_dataset` 走官方镜像，失败可改 `scripts/download_visdrone_dataset.sh hf`（hf-mirror）。
- 训练抖动 → 默认 warmup 3 epoch；不稳再调 `warmup_epochs`。
- 分档评测与训练期 `results.csv` 的 mAP 是两套 NMS 后处理，数值可能略有差异；对比实验统一用 `tiered_eval.py`。
- **短周期 / 子集实验仅用于筛选参数，不能直接认定「APs 提升 ≥1.0」**；结论必须来自完整 VisDrone train + 固定周期（120 epoch）。
- `--batch` 是复现旋钮（不属固定协议，按显存自定）：三组对照（纯 TAL / fixed-stride STAL / adaptive STAL）须用同一 batch。
