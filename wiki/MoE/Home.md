# MoE — Mixture-of-Experts

YOLO-Master 的 MoE 模块是一套完整的稀疏专家混合架构实现，覆盖从基础概念验证到工业级高性能部署的完整演进路线。

---

## 文档索引

| 文档 | 说明 | 适合读者 |
|------|------|----------|
| [核心模块详解](MoE_Core_Modules.md) | 40+ MoE 变体的完整实现分析 | 深入理解架构设计 |
| [路由与专家模块](MoE_Routers_Experts.md) | 路由器（7种）与专家网络（7种）详解 | 模块选型与定制 |
| [训练、损失与剪枝](MoE_Training_Loss_Pruning.md) | 辅助损失系统、动态调度、专家剪枝 | 训练调参与优化 |
| [诊断与分析工具](MoE_Diagnostics_Analysis.md) | 路由崩溃检测、专家使用追踪、诊断持久化 | 训练监控与调试 |
| [Mixture of Attention](Mixture_of_Attention.md) | MoA 模块与 v0.1 → v0.15 完整演进 | 了解最新架构方向 |
| [版本演进指南](../MoE_Modules_Explanation.md) | v0.1~v0.15 演进历史与部署建议（中文） | 快速选型决策 |
| [Version Evolution (EN)](../MoE_Modules_Explanation_EN.md) | English version | English readers |

---

## 快速选型指南

| 场景 | 推荐模块 | 配置示例 |
|------|----------|----------|
| **生产稳定训练** | `HybridAdaptiveGateMoE` (v0.6) | `[512, 8, 2]` |
| **极致推理速度** | `UltraOptimizedMoE` | `[512, 4, 2]` + `expert_type='ghost'` |
| **高上限研究** | `DetailAwareLowRankHybridAdaptiveGateMoE` (v0.9) | 需配合路由坍缩防护 |
| **移动端部署** | `UltraOptimizedMoE` + `GhostExpert` | `[256, 4, 2, 'ghost']` |
| **小目标检测** | `HybridAdaptiveGateMoE` + `LocalRoutingLayer` | 保留更多局部信息 |
| **多尺度上下文** | `VisualEnhancedAdaptiveGateMoE` (v0.10) | `[512, 8, 2]` + pyramid context |
| **与 MoLoRA 协同** | `HybridAdaptiveGateMoE` | `moe: true` + `molora_num_experts: 4` |

---

## 核心概念

### 路由器 (Router)

为每个输入样本分配专家。核心实现包括：
- `EfficientSpatialRouter` — 通用推荐，先降采样再路由
- `UltraEfficientRouter` — 极致速度，FLOPs 降低 ~95%
- `LocalRoutingLayer` — 小目标场景，保留局部纹理
- `AdaptiveRoutingLayer` — 资源极度受限，仅通道信息

### 专家网络 (Expert)

处理被分配到的数据。核心实现包括：
- `SimpleExpert` — 标准 Conv-BN-SiLU-Conv-BN
- `GhostExpert` / `FusedGhostExpert` — 参数量减半
- `InvertedResidualExpert` — 移动端优化
- `OptimizedSimpleExpert` — GroupNorm，小 Batch 稳定

### 核心 MoE 模块

| 版本 | 模块名 | 关键特性 |
|------|--------|----------|
| v0.1 | `ES_MOE` | 最早概念验证，异构专家 |
| v0.2 | `OptimizedMOE` | 引入 Shared Expert |
| v0.3 | `UltraOptimizedMoE` | Batched 并行计算，极致速度 |
| v0.4 | `AdaptiveGateMoE` | 双流路由，SE split |
| v0.5 | `FusedAdaptiveGateMoE` | 全融合专家候选 |
| **v0.6** | **`HybridAdaptiveGateMoE`** | **当前稳定版推荐** |
| v0.7 | `LowRankHybridAdaptiveGateMoE` | 低秩融合 |
| v0.8 | `RefinedLowRankHybridAdaptiveGateMoE` | 轻量特征 refinement |
| v0.9 | `DetailAwareLowRankHybridAdaptiveGateMoE` | 峰值最高，波动大 |
| v0.10 | `VisualEnhancedAdaptiveGateMoE` | 复杂度最高 |

---

## YAML 配置示例

```yaml
# yolo-master.yaml
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 3, C2f, [128, True]]
  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 6, C2f, [256, True]]
  - [-1, 1, Conv, [512, 3, 2]]
  # 插入 MoE 层：8 专家选 2，输出 512 通道
  - [-1, 1, HybridAdaptiveGateMoE, [512, 8, 2]]
  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 3, C2f, [1024, True]]
```

---

## 辅助损失

MoE 训练需要辅助损失来防止路由坍缩：

```python
from ultralytics.nn.modules.moe import MoELoss

loss_fn = MoELoss(
    num_experts=8,
    top_k=2,
    balance_loss_coef=0.01,    # 负载均衡损失
    router_z_loss_coef=0.001,  # Z-Loss（数值稳定性）
    diversity_loss_coef=0.0,   # 专家多样性损失
)
```

---

## 诊断工具

```python
from ultralytics.nn.modules.moe import (
    diagnose_model,                # 模型级诊断
    ExpertUsageTracker,            # 专家使用追踪
    RoutingCollapseDetector,       # 路由崩溃检测
    MoEDiagnosticsRecorder,        # 诊断持久化
)

# 训练监控
tracker = ExpertUsageTracker(num_experts=8)
# 自动检测 Dead Expert / Low Usage / Hot Expert

# 崩溃检测与自动恢复
detector = RoutingCollapseDetector(threshold=0.9)
health, actions = detector.diagnose(usage_gini=0.95)
if health == "CRITICAL":
    detector.apply_recovery(actions, model)
```
