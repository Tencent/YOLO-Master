# fix(vpeft): 让 solver 的 rank 投影与 placement plan 的容量校验一致

- 基线：`Tencent/YOLO-Master@af961b9`（官方 main）
- 分支：`lycyhrc/YOLO-Master:fix/vpeft-capacity-guard`（2 个 commit，仅源码 + 测试，不含任何实验材料）
- 关联背景：`lora_planner_backend=vpeft` 在含窄层的模型上无法真正生效（本 PR 修复前的行为）

---

## 1. 问题（根因）

求解器与计划校验对「层容量」的判定不一致：

| 环节 | 文件 | 行为 |
|---|---|---|
| 求解阶段 | `ultralytics/vpeft/solver.py` → `ConstraintRegistry.get_hard_mask(..., candidate_ranks=rank_set)` | 7 个硬约束（`C_op/C_sem/C_budget/C_deploy/C_compat/C_moe/C_div`）**没有任何一条建模容量**，因此 `min(in_channels, out_channels) < rank` 的窄层被判为可行并被放置 |
| 校验阶段 | `ultralytics/vpeft/placement_plan.py::PlacementPlan.validate_model` | 单独计算 `capacity = min(in, out)`，`rank > capacity` 直接 `raise ValueError` |

后果：`routing.routing_network.2`(capacity 3)、`dfl.conv`、`0.conv` 这类窄层被选进 plan 后，`apply_lora` 在 `validate_model` 处抛错；该异常被 `except (ValueError, TypeError)` 捕获后 **静默降级为 legacy planner**（`vpeft_strict=True` 时直接失败）。用户只能手工维护 `lora_exclude_modules` 规避，而该匹配是子串语义，容易误伤（如 `0.conv` 连带命中 `10.conv`）。

复现（修复前会走到降级/报错分支）：

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

## 2. 改动内容

**commit 1 `fix(vpeft): align solver rank projection with plan capacity validation`**

1. **上游根治**：新增硬约束 `RankCapacityConstraint`（`C_cap`，`rank <= min(in, out)`），注册进 `__all__`、`_default_name_map()`（legacy 别名 `cap`）与 `ConstraintRegistry.default()` 的默认硬约束集。求解阶段即排除窄层，预算不再浪费在不可放置节点上。
   - 容量维度未知（`0`/缺失）→ 视为可行，避免误杀无通道元数据的合成图；
   - 非 Conv2d/Linear 家族的算子不在此约束判定，仍由 `C_op` 负责。
2. **下游保险 + 审计**：`_build_vpeft_placement_plan` 对求解结果再校验一次容量，把因容量被跳过的层写入 `PlacementPlan.metadata["capacity_excluded"]`（`{name, rank, capacity}`），并打印可 grep 的 `[V-PEFT] capacity-excluded: ...` 日志。plan 是序列化产物，不能假设「求解器一定已经过滤过」。
3. **最后防线**：`validate_model` 保留严格校验，但把报错改为**一次性列出全部越界目标与各自容量**并提示 `metadata['capacity_excluded']`，便于手写/外部 plan 排障。

**commit 2 `fix(lora): freeze layers left out of an explicit target list`**（伴随修复，必须一起）

`_replace_conv_with_manual_lora` 只冻结自己被包裹的层（`ManualLoRAConv.__init__`），被 `target_modules` 过滤、`allow_depthwise=False` 的深度卷积、`only_3x3` 过滤或 head-like 名称跳过的层**仍然可训练**。commit 1 让 V-PEFT plan 真正生效后，这一点会立刻暴露：未被 plan 选中的基础层继续参与训练，且不属于 adapter 参数、`save_adapters` 也不会保存。官方 P0 关卡 `tests/test_p0_system_gates.py::test_planner_adapter_full_lifecycle` 正好覆盖该路径（修复前它只是因为 plan 校验失败、整体降级到 legacy 才“通过”）。

因此对「未被适配的层」统一冻结，使 manual 后端与 PEFT 后端语义一致（仅 adapter 可训练）。检测头仍由既有 `_unfreeze_detection_head` 显式解冻，行为不变。

## 3. 行为对照

| 场景 | 修复前 | 修复后 |
|---|---|---|
| 含窄层的模型 + `vpeft` | plan 校验抛错 → 静默降级 legacy（strict 下报错） | 窄层被排除，其余层正常放置，`metadata["capacity_excluded"]` 可审计 |
| 窄层日志 | 只有异常/降级 warning | `[V-PEFT] capacity-excluded: N layer(s) ... name(rank=4>capacity=3)` |
| 手写 plan 越界 | 首个越界目标即报错 | 一次列全部越界目标 + 提示审计字段 |
| 显式 `target_modules` | 未入选层仍可训练 | 未适配层冻结（仅 adapter 可训练） |
| 默认 `rank_min=4` 语义 / 既有配置项 | — | 不变（未新增配置项，`default.yaml`、`cfg/__init__.py` 无需改动） |

## 4. 测试证据

```bash
python3 -m pytest tests/test_vpeft_capacity_guard.py tests/test_vpeft.py tests/test_placement_plan_schema.py \
  tests/test_vpeft_lora_e2e.py tests/test_lora_fallback_effective_config.py tests/test_model_adapter_facade.py \
  tests/test_adapter_backend_contract.py tests/test_lora_training_strategy.py tests/test_p0_system_gates.py -q
# 70 passed, 1 failed
```

- 唯一失败：`test_p0_system_gates.py::test_cpu_gloo_two_rank_routed_continuous_training`，断言的是 `torchrun` 子进程 stdout，失败原因是本机 `Ultralytics settings updated to the latest schema` 警告（环境 settings schema 迁移），与本改动无关，未修改任何相关代码路径。
- 新增 `tests/test_vpeft_capacity_guard.py`（8 例）：约束语义、未知维度不误杀、registry/legacy 别名注册、求解器排除窄层、plan 审计字段、严格校验一次报全、边界值 `rank == capacity` 两侧一致。
- 既有 `tests/test_vpeft.py` 的硬约束名单断言同步为含 `C_cap`（有意变更：默认硬约束集扩充）。
- 静态检查：`ruff check` 全绿；`git diff --check` 无空白问题。

审计输出样例（EsMoE-N 形态的窄层模型）：

```
[V-PEFT] capacity-excluded: 2 layer(s) below the requested rank: 0(rank=4>capacity=3), 3(rank=4>capacity=3)
plan: ACCEPT targets=['1', '2'] capacity_excluded=[{'name': '0', 'rank': 4, 'capacity': 3}, ...]
```

## 5. 实测背景数据（C3 小样本消融，供参考，非本 PR 的验收依据）

NEU-DET(1800 张)/DeepPCB + YOLO-Master EsMoE-N(2.8M 全参)，同预算 3 策略 × 3 seed：

| 策略 | NEU mAP50(best) | DeepPCB mAP50(best) | 可训参数 | 峰值显存 |
|---|---|---|---|---|
| full_sft | 0.767±0.014 | 0.989±0.000 | 2,813,626 (100%) | ~5.3G |
| vpeft(planner, 6/6 ACCEPT) | 0.694±0.052* | 0.881±0.072 | 116,736 (4.15%) | ~4.1G |
| frozen_backbone | 0.694±0.041* | 0.769±0.151 | 1,906,822 (67.8%) | ~3.7G |

\* 稳健口径（剔除 1 个偶发 seed 后 n=3）；含偶发时 vpeft 0.612±0.167。
这些单元运行时使用 `lora_exclude_modules` 规避了上面的窄层缺陷（即本 PR 修复前的规避姿势），本 PR 未重跑训练。

## 6. 负结果与局限（如实披露）

1. **本 PR 只修「窄层导致的 plan 校验失败 / 静默降级」，不声称任何精度提升**；消融数字仍来自 exclude 规避口径，未用修复后代码重跑。
2. **seed 稳定性**：NEU 上 vpeft/frozen 各有 1/4 seed 出现极端收敛异常（frozen 卡死 <0.06、vpeft 停滞 0.37），经补跑确认为 seed 偶发而非系统性失效；DeepPCB frozen 在 seed 824 的震荡同 seed 重跑可确定性复现。结论限定在报告 seed 池内，不外推偶发概率。
3. **规模局限**：EsMoE-N 仅 2.8M，LoRA 的参数优势被小模型基数压缩；同 epochs 预算下三策略时长无差异（数据管线瓶颈）。
4. **新增的 `capacity_excluded` 只是审计字段**，不改变 plan schema 版本；但默认硬约束集新增 `C_cap` 会进入 plan 指纹/约束清单，已同步既有断言。

## 7. 复现命令

```bash
git fetch origin && git checkout af961b9
git checkout -b fix/vpeft-capacity-guard   # 或直接 check out 本 PR 分支
python3 -m pytest tests/test_vpeft_capacity_guard.py -q
python3 -c "
import torch.nn as nn
from ultralytics.utils.lora.api import apply_lora
from ultralytics.utils.lora.config import LoRAConfig
m = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.Conv2d(8, 16, 3, padding=1),
                  nn.Conv2d(16, 4, 1), nn.Conv2d(4, 3, 1))
w = apply_lora(m, LoRAConfig(r=4, alpha=8, backend='fallback', planner_backend='vpeft',
                             adapter_budget=1_000_000, vpeft_strict=True))
print(w.lora_target_modules)                                # ['1', '2']
print(w.lora_placement_plan['metadata']['capacity_excluded'])  # 0/3 因容量排除
"
```

## 8. 非目标（刻意不做）

- 不静默裁剪 rank（保持既有 ADAPT/REFUSE 语义，只做「排除 + 审计」）；
- 不改 `exclude_modules` 的子串匹配语义（应改为精确匹配，另开 issue）；
- 不改 legacy planner 行为；
- 不冻结非 Conv2d 的未适配模块（如 Linear/BN；manual 后端目前只包裹 Conv2d），留作后续；
- 不把任何实验脚手架/数据纳入仓库。

---

附：证据与复现包（C3 结项包）：<<EVIDENCE_LINK_PLACEHOLDER>>
