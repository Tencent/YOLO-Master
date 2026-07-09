# MoE-aware PEFT 消融实验与评测脚本 — 执行计划

## 背景
基于 AAAI 2026 战略重定位方案，需要在现有 YOLO-Master MoLoRA 基础设施上补充 **MoE-aware PEFT** 的消融实验和评测脚本。

## 现有基础设施
- `ultralytics/nn/peft/molora/` — MoLoRA 核心实现（layer.py, router.py, config.py, model.py, loss.py, utils.py）
- `ultralytics/nn/modules/moe/modules.py` — MoE 模块基础设施（MOE_LOSS_REGISTRY, _registry_set, _record_moe_snapshot）
- `scripts/peft_validation/run_peft_compare.py` — PEFT 对比验证脚本模板
- `tests/test_molora.py` — MoLoRA 单元测试模板

## 需要新增的组件

### Stage 1: 核心模块
- `ultralytics/nn/peft/molora/moe_aware.py`
  - `PerExpertRankAllocator`: 基于 expert 历史激活频率分配 rank 预算
  - `RouterCalibration`: 在 frozen router logits 上叠加低秩校正项 ΔW_r = B_r @ A_r
  - `MoLoRAMoEAwareConfig`: 扩展 MoLoRAConfig 添加 MoE-aware 字段

### Stage 2: 实验脚本
- `scripts/ablation_moe_peft_e1_molora_rank.py`
  - E1: MoLoRA per-expert rank vs 统一 rank baseline（同 budget 下）
- `scripts/ablation_moe_peft_e2_router_calibration.py`
  - E2: Router calibration 消融（有 ΔW_r vs 无 ΔW_r）
- `scripts/ablation_moe_peft_e3_expert_load_viz.py`
  - E3: Expert 负载可视化（激活频率分布 + rank 分配验证）

### Stage 3: 评测与测试
- `scripts/eval_moe_peft.py` — 统一评测脚本，支持多 seed、多配置批量运行
- `tests/test_moe_aware_peft.py` — 新模块的单元测试

## 依赖关系
- 所有实验脚本依赖核心模块 `moe_aware.py`
- 但各脚本之间无运行时依赖，可以并行实现
- 单元测试依赖所有新模块

## 接口约定（所有子代理共享）

### RouterCalibration
```python
class RouterCalibration(nn.Module):
    """在 frozen router 输出上叠加低秩校正项.
    
    router_logits_new = router_logits + B_r @ A_r(x)
    其中 A_r: in_channels -> r_r, B_r: r_r -> num_experts
    """
    def __init__(self, in_channels: int, num_experts: int, r_r: int = 4):
        ...
    def forward(self, x: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        """x: [B, C, H, W] or [B, C]; router_logits: [B, E]"""
        ...
```

### PerExpertRankAllocator
```python
class PerExpertRankAllocator:
    """基于历史激活频率分配 per-expert rank.
    
    启发式: 频率高的 expert 分配更高 rank。
    提供两种模式: 'uniform'（统一rank）, 'frequency'（按频率比例分配）
    """
    def __init__(self, num_experts: int, total_budget: int, min_rank: int = 2, mode: str = "frequency"):
        ...
    def allocate(self, usage_history: torch.Tensor) -> List[int]:
        """usage_history: [num_experts] 归一化频率 -> 返回 [r_e for e in range(num_experts)]"""
        ...
```

### MoLoRAMoEAwareLayer (扩展 MoLoRALayer)
- 添加 `router_calibration: Optional[RouterCalibration]` 字段
- 添加 `expert_ranks: List[int]` 字段（per-expert rank）
- forward 中在 router_logits 后叠加 calibration 项
- 在 `_last_routing_stats` 中记录 `calibration_applied` 和 `expert_ranks`

## 质量要求
- 所有新模块需保持与现有 MoLoRA API 兼容
- 实验脚本需支持 `WANDB_MODE=disabled` 离线运行
- 单元测试覆盖率 >= 80%
- 代码风格与现有仓库一致（Google-style docstrings, type hints）
