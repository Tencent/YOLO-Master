# Issue #52: MoE 专家剪枝与动态超参数调度（实现指南）

> 完整实验数据、图片、Pareto/Sweet Spot 分析、动态调度对照、场景化建议和 Discussion 草稿见
> [`reports/issues-52-moe-pruning-dynamic-scheduling.md`](../reports/issues-52-moe-pruning-dynamic-scheduling.md)。
> 本页仅保留配置与运行入口，避免与正式报告重复。

本文档对应 [Tencent/YOLO-Master Issue #52](https://github.com/Tencent/YOLO-Master/issues/52)。
完整入口为 `scripts/run_issue52_full.py`；它会执行基线训练（或复用 checkpoint）、五档剪枝、
直接推理、LoRA 10-epoch 恢复、三组动态调度对照、指标汇总、Pareto 分析和技术报告生成。

## Gini 动态调度

动态组在每个训练 epoch 内累计所有 batch、所有核心 MoE 层的专家利用率。逐层计算 Gini 后取均值，
只在 epoch 未被 NaN/recovery 机制拒绝时更新：

```text
ema_t = beta * ema_(t-1) + (1 - beta) * gini_t

coeff_(t+1) = clip(
    base_coeff * exp(alpha * (ema_t - target_gini)),
    min_balance_coeff,
    max_balance_coeff
)
```

路由集中（Gini 高）时增强均衡损失，路由健康时减弱约束以允许专家分化。调度默认关闭，
`moe_dynamic_schedule=none` 与原有行为一致；启用方式为：

```bash
yolo train \
  model=ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml \
  data=VisDrone.yaml \
  moe_dynamic_schedule=gini \
  moe_dynamic_gini_target=0.25 \
  moe_dynamic_gini_alpha=1.0 \
  moe_dynamic_gini_beta=0.8 \
  moe_dynamic_balance_min=0.5 \
  moe_dynamic_balance_max=2.0
```

每个有效 epoch 会写入 `moe_dynamic_schedule.csv`，包含 mean/layer Gini、EMA、实际系数和路由观测数。
状态同步到 EMA checkpoint，resume 不会重新开始 EMA。

## 一键实验

使用已有训练 checkpoint 可避免重复训练基线：

```bash
yolo/bin/python scripts/run_issue52_full.py \
  --baseline-checkpoint weights/YOLO-Master-EsMoE-N.pt \
  --data VisDrone.yaml \
  --device 0 \
  --imgsz 1344 \
  --batch 36 \
  --thresholds 0.05 0.10 0.15 0.20 0.30 \
  --lora-epochs 10 \
  --skip-existing
```

默认 `batch=36, imgsz=1344` 来自 97GB GPU 的实际 coco128 峰值测试：分配/保留显存约为
`87.28/92.89 GiB`。`batch=128, imgsz=1600` 会触发 OOM 并自动降到 16，因此没有采用表面更大、
实际会降档的配置。其他显存规格应显式覆盖这两个参数。

不传 `--baseline-checkpoint` 时会先从模型 YAML 训练基线。三组调度实验始终从同一个保存的
`schedule/initial_state.pt` 开始，避免模型构造阶段随机初始化破坏公平性：

- `baseline`：固定 `moe_balance_loss=1.0`；
- `dynamic`：Gini EMA 指数调度；
- `ablation`：固定低系数 `0.3`。

## 记录指标与产物

每个阈值均记录 dense/direct/LoRA10 的：

- mAP50-95、mAP50；
- 全模型 GFLOPs、batch-1 forward latency、参数量；
- 每层保留专家数；
- 每层及平均专家利用率 Gini；
- checkpoint 路径和基线 SHA-256。

输出目录包括：

- `pruning/results.csv`：五档阈值 × 两种恢复策略及 dense 基线；
- `pruning/pareto.csv`：精度—延迟 Pareto 前沿；
- `pruning/recommendations.json`：Sweet Spot 和服务器/边缘场景建议；
- `pruning/plots/`：阈值—mAP/GFLOPs/Latency、3D 权衡图和 Pareto 图；
- `schedule/schedule_summary.csv`：三组 final/best 指标、95% 收敛 epoch、比例和加速比；
- `schedule/dynamic_gini_balance/moe_dynamic_schedule.csv`：逐 epoch 调度 trace；
- `issue52_report.md`：可直接整理为 GitHub Discussion 的技术总结；
- `experiment_manifest.json`：数据、模型、阈值、seed 和 checkpoint 哈希。

## Sweet Spot 与场景推荐

默认质量门槛为相对 dense baseline 的 mAP50-95 绝对下降不超过 `0.01`。只有同时满足以下条件的
Pareto 点才可标注 Sweet Spot：

1. 专家结构确实发生物理剪枝，排除 no-op 阈值；
2. 精度下降通过质量门槛；
3. 在可行点中 latency 最低，精度作为次级排序。

若没有点通过门槛，结果会明确写为 `not_observed`，服务器端和边缘端均推荐保留 dense 模型，
而不是强行给出一个阈值。

## 副作用分析

报告同时检查 final 相对 best 的后期质量坍塌、动态组相对固定基线的 final mAP 差值，以及
“baseline final 已坍塌导致 95% final 指标失去区分力”的情况。常见改进方向：

- 系数振荡：增大 `moe_dynamic_gini_beta`；
- 过度均衡、专家同质化：降低 `alpha` 或提高 target Gini；
- 单 seed 偶然性：至少追加 3 个随机种子；
- final checkpoint 坍塌：同时使用稳定 best-checkpoint 目标报告收敛速度。
