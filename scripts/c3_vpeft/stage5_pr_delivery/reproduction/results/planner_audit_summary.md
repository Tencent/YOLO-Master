# Planner 审计小结 (V-PEFT placement + LoVO 护栏)

来源: 6 个 vpeft 单元(neu×3 + pcb×3, seed 824/2024/777) 训练日志 [V-PEFT]/[LoRA] 决策行 + summary.json。

## 决策分布

| 决策 | 次数 | 说明 |
|---|---|---|
| ACCEPT | 6/6 | `ACCEPT: selected 81 targets with ranks=[8]; base rank remains 8.` |
| ADAPT | 0 | — |
| REFUSE | 0 | — |

- 两数据集(NEU/PCB 全量训练)决策**完全一致**: 81 个 LoRA 目标层、rank=8、
  base rank 不升。planner 对同架构/同预算的决策跨 seed 可复现(deterministic)。
- 决策后 `Skipping the legacy Planner because an accepted V-PEFT placement plan is active`:
  统一走 V-PEFT 后端,legacy solver(ao/dco/mipr)路径未触发(非降级,是显式接管)。
- LoRA 实际适配器参数: **116,736**(96 个 lora_A/lora_B 权重, 占全参 4.15%)。
- 类别重初始化(head 输出 80→6)触发检测头 ~348,514 参数解冻训练
  (日志 `Unfrozen ... detection head parameters due to class-mismatch re-initialization`)。
  为两类数据共同现象(均为 6 类),计入 effective trainable=465,250 (16.5%)。

## 护栏/LOVO 相关字段记录

- `lora_planner_enabled=True`, `lora_planner_backend=vpeft`, `lora_vpeft_strict=True`
- exclude 清单(`lora_exclude_modules`): `routing_network.2`, `dfl.conv`, `0.conv`
  —— 三个 param-cap<8 的已知缺陷层(见 P2 bug 素材),strict 保持 True(未关闭护栏)。
- YOLO12 safety guard 自动生效: `exclude attn.{qkv,proj,pe} & ABlock mlp`, `alpha_warmup>=3`,
  `cap lora_lr_mult=1.0`(原 2.0 → 1.0)。

## 与效果对照 (ΔmAP50 = full_sft − vpeft, per dataset)

planner 全 ACCEPT 且预算充足时, 精度差异全来自 LoRA 容量本身:
- NEU: mean Δ = +0.165 (含 outlier seed2024 后方差大; 剔除 outlier 后 Δ≈0.04)
- PCB: mean Δ = +0.108 (0.03~0.17)
- 结论待补充单元定稿后更新(robust seed 判读)。

## P2 素材挂钩

- 已知缺陷 `cap<8` 层(0.conv / routing_network.2 / dfl.conv)的 planner ValueError 路径
  见 stage5 p2/planner_bug_issue.md 草稿; 本次通过 exclude 清单规避(strict 不降级)。
