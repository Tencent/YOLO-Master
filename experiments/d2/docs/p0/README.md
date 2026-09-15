# P0｜跑通现有 foundation distill 路径

**状态：✅ 已闭环**（2026-08-25，CUDA；2026-08-29 在 RTX 3090 上复现）

P0 = 用配置驱动仓库自身的训练路径跑通一次真实训练，并核对
teacher / tap / projector / loss 与日志。

| | |
|---|---|
| 定义与实测 | [`../design.md §5`](../design.md) |
| 复现命令与证据解读 | [`../../README.md`](../../README.md)「复现 P0」 |
| 证据 | [`../../results/p0_train_ok/`](../../results/p0_train_ok/) |

> **P0 不构成任何精度主张。** 3 epoch、`pretrained=False`、mAP50-95 全程为 0。
> 它只证明链路接通且 KD 项进入了被反传的目标。

P0 暴露的三个问题（权重过小、MoE 辅助损失压倒 KD、optimizer 不一致）见
[`../design.md`](../design.md) §5.4。其中权重问题的处置构成了整个 P1 前置工作，
见 [`../p1/`](../p1/README.md)。
