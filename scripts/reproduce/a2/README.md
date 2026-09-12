# A2 VisDrone STAL 结项报告

本目录对应任务书中的“主题 A：STAL 式小目标自适应标签分配”。最终提交代码只保留
原生 TAL、fixed-stride STAL 对照和之前效果最好的 AT-STAL A16 实现。实验报告使用
W&B 项目 `51nd0re1-/yolo-master-a2-0831`：

- [YOLO-Master-a2-0831 W&B 结项报告](https://api.wandb.ai/links/51nd0re1-/66t5a2ex)
- [YOLO-Master-a2-0831 W&B 项目](https://wandb.ai/51nd0re1-/yolo-master-a2-0831)

建议 PR 标题：`[犀牛鸟-A2]：完成 VisDrone STAL P0 闭环并提交 A16 最佳尝试`

本项目首先完成了任务书要求的 **P0 保底目标**：跑通 VisDrone2019-DET 训练与评估流程，
输出 small/medium/large 分档指标，并按 epoch 记录正样本统计。在 P0 闭环完成后，继续围绕
任务书的 P1 目标尝试了多种面积阈值、质量门控、IoU/NWD 排序和 Dynamic STAL 方案。

由于项目进入结项阶段，综合已有实验结果和剩余时间，最终选取 **AT-STAL A16** 作为 P1
阶段的最佳尝试并保留在 PR 中。A16 的 APs 提升最接近 P1 要求，但仍未达到规定的 1.0
个百分点，因此本项目最终结论是 **P0 已完成，P1 未完成**，不能将 A16 宣称为 P1 达标方案。

## 结项结论

| 级别 | 任务书要求 | 结论 | 证据 |
|---|---|---|---|
| P0 保底 | 跑通 VisDrone2019-DET 基线，输出 small/medium/large 分档指标，记录每 epoch 正样本统计 | **已完成** | fixed-stride baseline 和 A16 均完成训练、scale-aware eval，并生成逐 epoch CSV |
| P1 预期 | 实现面积感知阈值或代价调节；small AP 提升至少 1.0 个百分点，并用重复实验或置信区间证明 | **未完成** | A16 是最佳结果，但 APs 只提升 `0.8467` 个百分点，距离目标还差 `0.1533` 个百分点 |
| P2 理想 | 扫描阈值、放宽幅度、warmup，或扩展第二数据集/任务 | 未完成 | 仅做了若干探索性尝试，没有纳入最终代码和结论 |

这里“未达到”的方案均指没有达到 **P1**，不是 P0。P0 的运行闭环和验收产物已经具备。

### P1 差距

正式 QA 口径为 `v0.1-N`、完整 VisDrone train、`imgsz=800`、`epochs=120`、`batch=8`、
`seed=42`、`patience=0`、dense evaluation、完整验证集 548 张、`maxDets=500`。面积分档
使用原始验证图像 GT bbox：

| 档位 | 原图 GT bbox 面积 |
|---|---:|
| small / APs | `< 32^2` |
| medium / APm | `32^2 <= area < 96^2` |
| large / APl | `>= 96^2` |

正式 fixed-stride STAL baseline 的 `APs=0.130182`，因此 P1 目标为：

```text
P1 target APs = 0.130182 + 0.010000 = 0.140182
```

最佳 A16 的结果为 `APs=0.138649`：

```text
A16 gain = 0.138649 - 0.130182 = 0.008467 (+0.8467 pp)
P1 gap   = 0.140182 - 0.138649 = 0.001533 (0.1533 pp)
```

因此最终项目等级是：**P0 已完成，P1 未完成，P2 未完成**。A16 没有达到 P1 门槛，
不能宣称完成 P1，也没有进行“达到门槛后”的 Bootstrap 和 seed 复验。

## P0 验收证据

P0 要求的三类产物均已纳入最终流程：

1. VisDrone baseline 训练和验证流程可以运行。
2. `evaluate_scale_aware.py` 输出 `AP`、`AP50`、`AP75`、`APs`、`APm`、`APl`、
   `AR@1`、`AR@10`、`AR@100`、`AR@500`、`ARs@500`、`ARm@500` 和 `ARl@500`。
3. 训练回调按 epoch 写入正样本统计：baseline 为 `a2_positive_stats.csv`，A16 为
   `a2_area_stal_positive_stats.csv`，并同步记录 `small/medium/large` 的 GT 数量、
   冲突前后正样本数、平均正样本数和 zero-positive 比例。

结项训练曲线、指标表和历史实验对照见
[YOLO-Master-a2-0831 W&B 结项报告](https://api.wandb.ai/links/51nd0re1-/66t5a2ex)。
公开 PR 不保存带 `accessToken` 的链接，避免泄露报告访问凭据。

训练阶段面积指传入 assigner 的 resize/增强后 bbox；评估阶段面积指原始验证图像 bbox。
两者用途不同，不能混用。

## W&B 0831 正式结果

下表是此前统一评估得到的结果，主指标为原图面积 `<32^2` 的 `APs`，不是 VisDrone 官方
单独发布的面积指标。所有评估使用 `maxDets=500`。

| 方法 | AP | AP50 | AP75 | APs | APm | APl | AR@500 | ARs@500 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| pure TAL | 0.220460 | 0.375680 | 0.223490 | 0.132490 | 0.324970 | 0.415900 | 0.391670 | 0.295190 |
| fixed-stride STAL baseline | 0.220050 | 0.380780 | 0.218510 | 0.130182 | 0.323080 | 0.419000 | 0.389870 | 0.293620 |
| AT-STAL A8 | 0.219780 | 0.381140 | 0.216450 | 0.134000 | 0.318110 | 0.429010 | 0.391090 | 0.296390 |
| AT-STAL A12 | 0.219510 | 0.381670 | 0.216830 | 0.133130 | 0.320810 | 0.435480 | 0.390080 | 0.291020 |
| **AT-STAL A16** | **0.222630** | **0.382990** | 0.220510 | **0.138649** | 0.323630 | **0.436990** | **0.392460** | 0.293530 |
| AT-STAL A20 | 0.221320 | 0.382430 | 0.220420 | 0.132900 | 0.322390 | 0.406520 | 0.392240 | 0.295280 |
| AT-STAL A32 | 0.220710 | 0.378860 | **0.221250** | 0.135950 | **0.323770** | 0.399290 | 0.388480 | 0.293470 |
| AT-STAL A64 | 0.219120 | 0.378580 | 0.217630 | 0.131280 | 0.322070 | 0.420650 | 0.389070 | 0.289280 |

相对正式 baseline，A16 的 APs 增益为 `+0.008467`。A8、A12、A20、A32、A64 的增益分别为
`+0.003818`、`+0.002948`、`+0.002718`、`+0.005768`、`+0.001098`，均未达到 P1。

## 最终保留的 A16 实现

A2 的正式算法和验收参数统一写在仓库根目录的 `ultralytics/cfg/default.yaml` 中，入口不再
通过 `--stal-area-threshold` 接收 A16 阈值。当前配置项为：

```yaml
stal_area_threshold: 16.0
stal_min_candidates: 4
a2_model: v0.1-N
a2_epochs: 120
a2_imgsz: 800
a2_batch: 8
a2_seed: 42
a2_patience: 0
a2_wandb_project: yolo-master-a2-0831
```

训练脚本会在启动时读取并校验这些配置；如果 A16 的阈值、候选数或正式模型被改成其他
值，程序会直接拒绝启动，避免验收时误跑其他历史方案。上述正式字段不注册为命令行参数，
因此不会被命令行覆盖；`data`、`device`、`workers`、输出目录和 W&B online/offline 状态
仍属于运行环境参数，可以由命令行指定。

A16 中的 `16` 是 **bbox 面积不超过 16 个像素平方**，即 `width * height <= 16`，
不是 `16 x 16` 像素图像，也不是 QA 的整个 small 范围 `<32^2=1024`。

对 one-to-many 分支中的每个有效 GT：

1. 先使用原生 STAL 的候选点掩码。
2. 只对增强后 bbox 面积 `<=16` 且原生候选不足 4 个的 GT 处理。
3. 以 GT 中心到 anchor 中心的欧氏距离排序，从同一个 GT 的候选点外补最近的缺失点，
   直到候选数达到 4 个。
4. GT 框本身不扩大；TAL score、CIoU、target score、冲突消解和 BCE/CIoU/DFL 损失
   保持原实现。
5. one-to-one 分支和面积大于 16 的 GT 不改变原生候选规则。

最终代码入口为：

- `ultralytics/utils/tal.py`：`AreaAwareTaskAlignedAssigner`
- `scripts/reproduce/a2/reproduce_esmoe_stal.py`：A16 训练与正样本统计
- `scripts/reproduce/a2/reproduce_esmoe_baseline.py`：pure TAL/fixed-stride STAL 对照
- `scripts/reproduce/a2/evaluate_scale_aware.py`：原图面积分档评估
- `scripts/reproduce/a2/bootstrap_scale_delta.py`：保留为后续达到 P1 后的统计工具

最终版本已移除 Dynamic STAL、QG、普通 IoU Top-k、NWD Top-k、warmup 和阈值扫描编排代码，
避免 PR 中出现无法作为最终方案复现的实验入口。

## 已做但未达 P1 的尝试

这些结果用于说明实验过程和失败原因，不属于最终提交代码：

| 尝试 | 结果或结论 |
|---|---|
| A8/A12/A20/A32/A64 | A16 最好，扩大或缩小面积范围都没有稳定提升；候选增加后引入的低质量监督抵消了收益 |
| tiny quality gate 25%/35%/45% | 三组均低于 A16；相对质量门控没有把新增候选转化为更高 APs |
| small quality gate | 试图覆盖 QA 的 small 范围，但结果没有超过 A16，且改变了更多候选分配因素 |
| A16 + 普通 IoU Top-k | 曾发现仅替换排序张量时通常仍是 4 个候选，排序变化实际不起作用；修正后仍未达到 P1 |
| A16 + NWD Top-k | 作为小目标定位质量排序的探索，未得到达到 P1 的正式结果；没有保留到最终入口 |
| Dynamic STAL | 原生动态 k 在训练早期出现冷启动：epoch 1 平均 `k≈1`，大量 small GT 少于 4 个正样本，最终 APs 低于 A16 |
| Dynamic STAL warmup | 尝试用初期正样本配额缓解冷启动，但在结项前没有形成超过 A16 且满足 P1 的正式结果 |

这些尝试没有改变 P0 结论：P0 关注的是可运行的基线、分档评估和逐 epoch 统计，已经满足；
它们失败的是 P1 的性能门槛。

## 复现环境

以下命令在训练服务器执行，使用历史 W&B 项目 `yolo-master-a2-0831`，不会创建 0908
实验，也不会自动 pull。

```bash
cd /home/zdyin/sfh/code/YOLO-Master-main
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export A2_DATA="$PWD/dataset/VisDrone.local.yaml"
export A2_WANDB="yolo-master-a2-0831"
mkdir -p logs runs/a2/qa_0831/{control,at_stal,scale_eval}
wandb login
```

### P0 fixed-stride baseline

```bash
nohup python -u scripts/reproduce/a2/reproduce_esmoe_baseline.py \
  --data "$A2_DATA" --assigner fixed-stal \
  --workers 8 --amp --no-sparse-eval \
  --project "$PWD/runs/a2/qa_0831/control" \
  --name "v0.1-N_fixed-STAL_800_120e_b8" \
  --wandb --wandb-mode online \
  > logs/a2_qa0831_fixed_stal_800_120e_b8.log 2>&1 &
echo $! > logs/a2_qa0831_fixed_stal_800_120e_b8.pid
```

### 最终 A16

```bash
nohup python -u scripts/reproduce/a2/reproduce_esmoe_stal.py \
  --data "$A2_DATA" --workers 8 --amp --no-sparse-eval \
  --project "$PWD/runs/a2/qa_0831/at_stal" \
  --name "v0.1-N_AT-STAL_A16_800_120e_b8" \
  --wandb --wandb-mode online \
  > logs/a2_qa0831_at_stal_a16_800_120e_b8.log 2>&1 &
echo $! > logs/a2_qa0831_at_stal_a16_800_120e_b8.pid
```

训练被中断时，只能在同一 run 目录使用原来的 `last.pt` 恢复，并保持模型、seed、面积
阈值、batch、epoch 和 W&B 配置不变：

```bash
python -u scripts/reproduce/a2/reproduce_esmoe_stal.py \
  --data "$A2_DATA" --resume --workers 8 --amp --no-sparse-eval \
  --project "$PWD/runs/a2/qa_0831/at_stal" \
  --name "v0.1-N_AT-STAL_A16_800_120e_b8" \
  --wandb --wandb-mode online
```

如果 checkpoint 中保存的是空 GradScaler，最终代码会恢复为关闭 AMP 的运行状态，避免
`source state dict is empty` 阻断恢复；不要删除或覆盖原有 `last.pt` 和统计 CSV。

### 统一 scale-aware 评估

```bash
eval_a2() {
  local WEIGHTS="$1"
  local NAME="$2"
  python -u scripts/reproduce/a2/evaluate_scale_aware.py \
    --weights "$WEIGHTS" --data "$A2_DATA" --split val \
    --imgsz 800 --batch 8 --max-det 500 --device 0 --workers 8 --no-sparse-eval \
    --project "$PWD/runs/a2/qa_0831/scale_eval" --name "$NAME" \
    --wandb --wandb-project "$A2_WANDB" --wandb-name "${NAME}_scale"
}

eval_a2 "$PWD/runs/a2/qa_0831/control/v0.1-N_fixed-STAL_800_120e_b8/weights/best.pt" \
  "v0.1-N_fixed-STAL_800_120e_b8_best"
eval_a2 "$PWD/runs/a2/qa_0831/at_stal/v0.1-N_AT-STAL_A16_800_120e_b8/weights/best.pt" \
  "v0.1-N_AT-STAL_A16_800_120e_b8_best"
```

评估输出目录中的 `scale_metrics.json` 是 P0/P1 对比使用的正式指标文件。P0 只要求产出
这些分档结果；P1 还要求 APs 达到 baseline `+0.010000`，并在达到点估计门槛后进行配对
Bootstrap 和 seed 复验。本次 A16 没有达到该前置门槛，因此不伪造 P1 的统计验收结果。

## 测试与提交边界

最终 PR 重点覆盖 A16 候选掩码、候选补足、冲突统计、scale-aware 面积分档和 baseline
行为回归。提交前执行：

```bash
ruff check ultralytics/ tests/ scripts/ agent/
ruff format --check ultralytics/ tests/ scripts/ agent/
codespell
pytest tests/test_area_stal_assigner.py -q
pytest tests/test_tal_mps_regression.py tests/test_mixture_loss_composition.py -q
```

训练生成的 `runs/`、`logs/`、权重、数据集缓存和 W&B 本地文件不提交；PR 中只提交
A16 所需代码、测试、评估脚本和本结项 README。
