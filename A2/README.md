# A2 — STAL 小目标自适应标签分配

> 任务代号 **A2**：STAL（面积感知标签分配，面向 VisDrone 密集小目标）。
> 分阶段推进：**P0**（保底基线）→ **P1**（面积感知 on/off 对照，核心）→ **P2**（扫描 / 扩展）。

## 阶段

| 阶段 | 状态 | 内容 | 目录 |
|---|---|---|---|
| P0 | ✅ 已完成 | 真实 VisDrone2019-DET 基线 + small/medium/large 分档 AP（COCO 32²/96²）+ 每 epoch 正样本统计。主指标 APs=**0.0905** | [P0/](P0/README.md) |
| P1 | ✅ 已完成 | 面积感知 on/off 三组对照（纯 TAL / fixed / adaptive）+ `default.yaml` 注入键 + 33 项单测。APs 0.0875 / 0.0905 / **0.0925** | [P1/](P1/README.md) |
| P2 | ⏸️ 阶段性结项（部分） | 参数筛参 2/7 完成；Mosaic 交互 / 120ep 认定未做（见「结项结论」） | [P2/](P2/README.md) |

## 准入检查（冒烟）

- [smoke/](smoke/README.md)：VisDrone 子集 1 epoch 冒烟 + 定位 assigner / 配置注入点（已完成）。

## 验收标准（Definition of Done）

1. ✅ STAL 模块与配置 PR —— `tal.py` 逻辑 + `default.yaml` 配置键（P1 完成）。
2. ✅ 三组对照分档 AP on/off 对比表（纯 TAL / fixed-stride STAL / adaptive STAL；small/medium/large）。
3. ✅ 每 epoch 正样本演化曲线（`fg_s8/s16/s32` + per-tier avg_pos/zero_ratio，随 `results.csv` 写入）。
4. ⚠️ 参数敏感性说明（**部分**：仅 topk16/beta8 2/7；Mosaic 交互 / 120ep 认定未做）。

- 指标口径见 [P0/README.md](P0/README.md)：主指标 APs@[.50:.95]（COCO 32²/96²，maxDets=500），必须分档报，禁止只报总 mAP。
- 训练抖动 → 渐进 warmup。

## 结项结论（阶段性结项，2026-09-11）

**核心结论**：面积感知 STAL 显著改善小目标**正样本覆盖**——adaptive vs fixed 平均正样本 ×1.88、零正样本小目标比 7.05%→1.50%（vs 纯 TAL ×2.20、约 1/13）。但主指标 **APs 增益有限**：vs fixed **+0.20pp**、vs 纯 TAL **+0.50pp**，**未达任务书「APs ≥ +1.0pp」**。

**已完成**：
- P0：真实 VisDrone2019-DET 基线 + COCO-style 32²/96² 分档（APs=0.0905，maxDets=500）。
- P1：三组对照（纯 TAL 0.0875 / fixed 0.0905 / adaptive 0.0925）+ `default.yaml` 配置契约 + 33 项单测。
- P2 部分：参数筛参 2/7（topk16 = 0.0588，-0.15pp；beta8 = 0.0602，-0.01pp——β 6→8 无增益）。

**未达标 / 未完成**：
- 主指标 APs 提升未达 ≥ +1.0pp（实测 +0.20pp vs fixed）。
- 参数敏感性说明仅 2/7（剩 topk20 / expand2 / expand05 / beta4 / alpha1 未跑）。
- Mosaic on/off 交互对照未做（验收 #6）。
- 120ep 复现认定未做（无「认定级」结论）。
- Phase 3 stretch（α×β 组合 / warmup 曲线 / 第二数据集）未做。

**遗留问题**：`TaskAlignedAssigner` 在 CUDA OOM 回退 CPU 时偶发 segfault（仅 OOM 触发，非主路径）。
