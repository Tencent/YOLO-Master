<!--
PR 标题（GitHub 上填这一行，对齐 #253 的 [犀牛鸟-Ax] 前缀风格，正文用中文）：
[犀牛鸟-C3]：修复 V-PEFT 容量约束 —— 窄于最小候选 rank 的层不再被选为适配目标
分支：lycyhrc/YOLO-Master:fix/vpeft-capacity-guard（base: Tencent:main，2 commits，tip daed306）
用途：**新建**的上游源码修复 PR（编号待创建；仓库 PR #279 已改作整体交付 PR，见 stage5 README）。
下面的内容从 "## 概述" 开始整段粘贴到 PR 描述框。
-->

## 概述

- 补上缺失的 `C_cap` 硬约束（`RankCapacityConstraint`：仅当 `rank <= min(in, out)` 时该 rank 可行），求解器不再把窄层投影成适配目标。
- 生成放置计划时二次校验已选目标：被跳过的层记录进 `metadata["capacity_excluded"]`，并打一行可 grep 的日志 `[V-PEFT] capacity-excluded: ...`。
- `PlacementPlan.validate_model` 保持严格校验，但改为一次报出所有违规目标及其容量。
- 显式目标列表或 profiling 过滤把某些层排除在外时，冻结这些未适配层，使 manual 后端与 PEFT 后端一致（可训练权重只有 adapter）。

本 PR 只修 bug：不新增配置项、不改默认值（`rank_min` 仍为 `4`），legacy planner 行为不变。

## 问题背景

七个硬约束（`C_op` / `C_sem` / `C_budget` / `C_deploy` / `C_compat` / `C_moe` / `C_div`）都没有建模层容量，而 `PlacementPlan.validate_model` 会算 `min(in, out)` 并拒绝大于它的 rank，于是投影阶段和校验阶段互相矛盾：

- 投影阶段认为 `routing.routing_network.2`（容量 3）、`25.dfl.conv`、`0.conv` 可行并选中；
- 校验阶段 `apply_lora` 恰好对这些目标抛 `ValueError`。

`lora_planner_backend=vpeft` 时该异常被 `except (ValueError, TypeError)` 吞掉，静默回退到 legacy planner（`vpeft_strict=True` 则直接失败），也就是 V-PEFT 实际没生效。规避办法是手写 `lora_exclude_modules` 清单，而它按子串匹配（`0.conv` 会连 `10.conv` 一起匹配）。

修复前的复现：

```bash
python -c "
import torch.nn as nn
from ultralytics.utils.lora.api import apply_lora
from ultralytics.utils.lora.config import LoRAConfig
m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 8, 3, padding=1))
apply_lora(m, LoRAConfig(r=2, alpha=4, backend='fallback', planner_backend='vpeft',
                         adapter_budget=100_000, vpeft_strict=True))
"
```

修复后的审计输出（EsMoE-N 形状的窄层模型）：

```
[V-PEFT] capacity-excluded: 2 layer(s) below the requested rank: 0(rank=4>capacity=3), 3(rank=4>capacity=3)
plan: ACCEPT targets=['1', '2'] capacity_excluded=[{'name': '0', 'rank': 4, 'capacity': 3}, ...]
```

## 行为变化

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 窄层模型 + `vpeft` | plan 校验抛错 → 静默回退 legacy（`vpeft_strict=True` 下直接失败） | 窄层被排除，其余目标正常放置，审计信息落盘 |
| 容量日志 | 只有异常 / 回退告警 | `[V-PEFT] capacity-excluded: N layer(s) below the requested rank: ...` |
| 手写 plan 超出容量 | 命中第一个违规目标就抛错 | 一次列出全部违规目标，并引用审计字段 |
| 显式 `target_modules` | 被跳过的层仍然在训练 | 未适配层被冻结 |
| `rank_min=4` 语义 / 既有配置键 | — | 不变（不加新键，`default.yaml` 与 `cfg/__init__.py` 未改） |

## 验证

```bash
python -m pytest tests/test_vpeft_capacity_guard.py tests/test_vpeft.py tests/test_placement_plan_schema.py \
  tests/test_vpeft_lora_e2e.py tests/test_lora_fallback_effective_config.py tests/test_model_adapter_facade.py \
  tests/test_adapter_backend_contract.py tests/test_lora_training_strategy.py tests/test_p0_system_gates.py -q
# 70 passed, 1 failed
```

- 唯一失败的是 `test_p0_system_gates.py::test_cpu_gloo_two_rank_routed_continuous_training`，它断言的是 `torchrun` 子进程 stdout，而本机这份 stdout 里夹带了 `Ultralytics settings updated to the latest schema` 警告。该用例不涉及本 PR 的任何代码路径，也没有被 skip 或放宽条件。
- 新增 `tests/test_vpeft_capacity_guard.py`（8 个用例）：约束语义、容量未知时不排除、注册表与 legacy 别名注册、求解器排除窄层、plan 审计字段、严格校验一次报全、`rank == capacity` 两侧边界。
- `tests/test_vpeft.py` 的硬约束列表断言补上 `C_cap`（有意为之：默认硬约束集合变大了）。
- 改动文件 Ruff 检查与 `git diff --check` 通过。

## 改动范围

两个 commit，8 个文件（+350 / −10），只动源码和测试 —— 不含实验脚手架、数据或报告：

1. `fix(vpeft): align solver rank projection with plan capacity validation`（6 文件，+299 / −10）
   - `ultralytics/vpeft/constraints.py`：新增 `RankCapacityConstraint`（`C_cap`：仅当 `rank <= min(in, out)` 时可行）并注册进硬约束集合。
   - `ultralytics/vpeft/__init__.py`：导出该约束。
   - `ultralytics/vpeft/placement_plan.py`：严格校验改为一次报出全部违规目标及其容量。
   - `ultralytics/utils/lora/api.py`：`apply_lora` 里的第二遍容量校验 + `capacity_excluded` 审计字段 + `[V-PEFT]` 日志。
   - `tests/test_vpeft_capacity_guard.py`（新增，8 个用例）、`tests/test_vpeft.py`（硬约束列表断言补 `C_cap`）。
2. `fix(lora): freeze layers left out of an explicit target list`（2 文件，+51）
   - `ultralytics/utils/lora/fallback.py`：新增 `_freeze_unadapted_module`，在五条跳过路径上冻结该层 —— rank 与 groups 不匹配、depthwise 被禁、head 类名字、`only_3x3` 过滤、`target_modules` 不匹配。`_replace_conv_with_manual_lora` 原先只冻结自己包裹的层，这些被跳过的层仍在训练（因为是非 adapter 参数，也不会进 `save_adapters`）。
   - `tests/test_lora_fallback_effective_config.py`：补两个用例（显式 `target_modules` 之外的层被冻结；depthwise 与 head 类层被冻结）。
   - 官方 P0 门禁用例 `test_planner_adapter_full_lifecycle` 覆盖的就是这条路径；在 commit 2 之前它只是"看起来通过"，因为 plan 校验失败把整个运行降级到了 legacy planner。

## 实验（背景信息，不作为本 PR 的验收依据）

NEU-DET（1800 张）与 DeepPCB，YOLO-Master EsMoE-N（2.8M 参数），3 策略 × 3 seed，同预算（epochs 100、batch 8、imgsz 640、amp 关闭）：

| 策略 | NEU mAP50（best） | DeepPCB mAP50（best） | 可训练参数 | 显存峰值 |
|---|---|---|---|---|
| full_sft | 0.767±0.014 | 0.989±0.000 | 2,813,626（100%） | ~5.3G |
| vpeft（planner，6/6 ACCEPT） | 0.694±0.052* | 0.881±0.072 | 116,736（4.15%） | ~4.1G |
| frozen_backbone | 0.694±0.041* | 0.769±0.151 | 1,906,822（67.8%） | ~3.7G |

\* 稳健估计（剔除 1 个偶发 seed，n=3）；若计入该 seed，vpeft 为 0.612±0.167。
上述训练都是用 `lora_exclude_modules` 规避该缺陷跑出来的；**修复后的代码没有重跑训练**。

## 局限

- 不主张精度收益。本 PR 只消除校验失败与静默回退；消融数字仍来自修复前的规避方案。
- seed 稳定性：NEU 上 vpeft 与 frozen 各出现过一个极端 seed（0.369 / 0.210），换新 seed 重跑即恢复正常；DeepPCB frozen 在 seed 824 的振荡用同一 seed 可确定性复现。该观察仅限于当前 seed 池，不能估计失败概率。
- EsMoE-N 是 2.8M 的小模型，其参数效率结论不能直接外推到更大模型；同 epoch 下三种策略的墙钟时间接近（瓶颈在数据管线）。
- `capacity_excluded` 只是审计元数据，不改 plan schema 版本；但 `C_cap` 进入了默认硬约束集合，因此会改变 plan fingerprint。

## 不做的事

- 不做静默 rank 裁剪（`ADAPT` / `REFUSE` 语义不变）。
- `exclude_modules` 的子串匹配保持原样。
- legacy planner 行为不变。
- 暂不冻结非 Conv2d 的未适配模块（Linear / BN）。
- 不向仓库加入实验脚手架、数据或报告。

## 复现

```bash
git fetch origin && git checkout af961b9
git checkout fix/vpeft-capacity-guard   # 或直接切到本 PR 分支
python -m pytest tests/test_vpeft_capacity_guard.py -q
python -c "
import torch.nn as nn
from ultralytics.utils.lora.api import apply_lora
from ultralytics.utils.lora.config import LoRAConfig
m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1),
                  nn.Conv2d(16, 4, 1), nn.Conv2d(4, 3, 1))
w = apply_lora(m, LoRAConfig(r=4, alpha=8, backend='fallback', planner_backend='vpeft',
                             adapter_budget=1_000_000, vpeft_strict=True))
print(w.lora_target_modules)                                  # ['1', '2']
print(w.lora_placement_plan['metadata']['capacity_excluded'])  # 0/3 excluded by capacity
"
```

## PPT

https://drive.google.com/file/d/1dg8tZyI11xJR1JrCCKBWqbC4_kFgmsVp/view?usp=sharing
