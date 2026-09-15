# [P2 素材·Issue 草稿] V-PEFT planner: 层容量 < 最小候选 rank 时硬抛 ValueError, 无法自适应降级

> 状态: 草稿(基于实验观察 + 源码引用, 未正式提 PR)。若提交请补最小复现脚本与报错堆栈截图。

## Bug 概述

V-PEFT/AO solver 在候选放置层 `rank > layer capacity (cap=min(in_c,out_c))` 时直接
`raise ValueError`（硬校验），且不自动把该层降级/过滤/标记 ADAPT，导致用户在
EsMoE-YOLO 等小容量层模型上无法仅凭参数配置跑通，必须手工维护
`lora_exclude_modules` 黑名单（子串匹配,存在 `0.conv` 连带误伤 `10.conv` 等层级）。

## 触发位置(源码)

`ultralytics/vpeft/placement_plan.py:113-118`

```python
capacity = min(...in..., ...out...)
if rank > capacity:
    raise ValueError(f"PlacementPlan rank {rank} for {target.name!r} exceeds layer capacity {capacity}")
```

候选 rank 由 solver 控制: AO/DCO rank_min=4（rank∈[4,8,…,64]）, MIP rank_set=[4,8,…]。
因此任何 cap<4 的目标一旦被选中即抛错。

## 受影响层(实测, YOLO-Master-EsMoE-N)

| 目标 | 说明 | cap | AO rank_min=4 时 |
|---|---|---|---|
| `*.routing.routing_network.2` | MoE 路由线性层(3/6/9/12 层) | ≈3 | raise |
| `25.dfl.conv` | DFL 头(语义保护常排除, 主要 legacy 路径触发) | ≈1 | raise |
| `0.conv` | stem 首层(lora_skip_stem=True 默认排除) | 较小 | raise |

> vpeft backend 下 DFL 已被 `SemanticProtectionConstraint` 保护, 报错多见于 routing 层;
> strict=True 时上层按配置 `lora_vpeft_strict` 直接异常(不 fallback), 证据链干净但用户无路可走。

## 实验表现(2026-09-07 C3 21 单元, strict=True + exclude 清单规避)

- 6/6 vpeft 单元决策一致 ACCEPT(81 targets, rank=8), 未触发此路径(因已 exclude);
- 若去除 exclude(早期验证)立即触发 ValueError, 是 stage2 中被迫维护黑名单的直接原因。

## 期望行为(修复建议, 供讨论)

1. 容量不足的层在求解器内部自动从候选移除并记录一条 ADAPT/跳过审计, 而非 raise;
   或
2. `rank` 硬校验改为 min(rank, floor(cap)) 软裁剪并把实际 rank 写入 placement 决策;
3. exclude 黑名单改为**精确模块名匹配**(当前子串匹配 `0.conv` 连带 `10.conv`), 或输出"实际受影响模块清单"便于用户核对;
4. 报错信息中列出候选目标与容量, 便于诊断。

## 复现线索

- 复现条件: `lora_r=8 lora_planner_enabled=True lora_planner_backend=vpeft lora_vpeft_strict=True`
  且目标数据/模型含 routing 层; 去掉 exclude 黑名单即可复现 ValueError。
- 完整复现脚本/堆栈待正式提交 issue 前补齐(运行 ~1 epoch 即可, A800 单卡 <5 分钟)。
