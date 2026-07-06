# YOLO-Master MoE Core Modules 技术文档

> 本文档基于 YOLO-Master 项目 `ultralytics/nn/modules/moe/` 目录下的实际代码实现编写，涵盖从基础到生产级的全部 MoE (Mixture-of-Experts) 核心模块。技术术语保留英文原词（如 MoE, Top-K, GroupNorm, DDP 等）。

---

## 目录

- [1. 概述与设计理念](#1-概述与设计理念)
- [2. 全局基础设施](#2-全局基础设施)
- [3. 专家模块 (experts.py)](#3-专家模块-expertspy)
- [4. 路由模块 (routers.py)](#4-路由模块-routerspy)
- [5. 核心 MoE 模块 (modules.py)](#5-核心-moe-模块-modulespy)
- [6. 辅助损失 (loss.py)](#6-辅助损失-losspy)
- [7. 工具与诊断](#7-工具与诊断)
- [8. 与 YOLO 架构的集成](#8-与-yolo-架构的集成)
- [9. 使用示例](#9-使用示例)
- [10. 版本演进与选型建议](#10-版本演进与选型建议)

---

## 1. 概述与设计理念

YOLO-Master 的 MoE 子系统采用**模块化、可插拔、生产级**的设计理念，覆盖从研究实验到边缘部署的全生命周期。整个子系统按功能划分为以下文件：

| 文件 | 职责 | 行数 |
|:---|:---|:---|
| `modules.py` | 核心 MoE 模块（20+ 种变体） | ~4384 |
| `experts.py` | 专家网络实现（6 种专家类型） | ~306 |
| `routers.py` | 路由网络实现（6 种路由器） | ~405 |
| `loss.py` | 辅助损失函数（含 DDP 感知） | ~351 |
| `utils.py` | 工具函数（FLOPs 计算、批量专家计算等） | ~189 |
| `analysis.py` | 专家使用分析与可视化 | ~648 |
| `diagnostics.py` | 轻量级路由诊断 | ~96 |
| `history.py` | 诊断持久化与历史绘图 | ~309 |
| `scheduler.py` | Gini 驱动的动态调度器 | ~108 |
| `pruning.py` | 基于使用统计的 MoE 剪枝 | ~489 |
| `__init__.py` | 公共 API 导出与兼容别名 | ~146 |

### 1.1 设计原则

1. **稀疏激活**：每个样本仅激活 `top_k` 个专家（通常 `top_k << num_experts`），推理时计算量与参数量解耦。
2. **共享专家 (Shared Expert)**：所有样本始终通过共享路径，保证训练稳定性与保底性能。
3. **负载均衡**：通过 `balance_loss` 与 `z_loss` 防止路由坍缩（Routing Collapse）。
4. **DDP 安全**：所有分布式训练路径使用 `all_reduce` 与 `float32` 精度进行跨卡同步，避免 `float16` 累积误差。
5. **ONNX/导出兼容**：稀疏路径在导出时自动切换为密集路径（`torch.gather` + `stack`），确保静态图可追溯。
6. **deepcopy 安全**：通过 `MOE_LOSS_REGISTRY`（`WeakKeyDictionary` + 线程锁）存储非叶子张量，避免 `copy.deepcopy` 失败。

---

## 2. 全局基础设施

### 2.1 辅助损失注册表

```python
MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()
_MOE_LOSS_REGISTRY_LOCK = _threading.Lock()
```

- **用途**：每个 MoE 模块在 `forward` 训练阶段将辅助损失写入注册表，由 `tasks.py` 统一收集后加到总损失。
- **为什么不用 `self.aux_loss` 属性？**：避免将带有 `grad_fn` 的非叶子张量存储在模块 `__dict__` 中，这会导致 `deepcopy` / `EMA` 失败。
- **线程安全**：`registry_set` 与 `registry_get` 均受锁保护，支持多线程 eval / hook 回调。

### 2.2 路由快照机制

```python
MOE_SNAPSHOT_INTERVAL = max(int(os.environ.get("MOE_SNAPSHOT_INTERVAL", "10")), 1)
```

- 默认每 10 个 `forward` 步记录一次 `last_routing_snapshot`，用于后续诊断分析。
- 张量保留在原设备上，消费者（如 `format_moe_diagnostics`）仅在需要时才搬移到 CPU。

### 2.3 鲁棒 deepcopy

```python
def _robust_deepcopy(obj, memo):
    ...
```

核心逻辑：遍历 `__dict__`，若发现 `grad_fn is not None` 的张量，替换为同设备同类型的零张量；对可深拷贝的属性递归处理。所有核心 MoE 模块均覆写 `__deepcopy__` 为 `_robust_deepcopy(self, memo)`。

---

## 3. 专家模块 (experts.py)

专家模块定义了被路由激活的神经网络子结构。所有专家均使用 `GroupNorm` 替代 `BatchNorm`，因为在 Top-K 路由后每个专家往往只处理 1 个样本，BN 的 running stats 与 `n=1` 方差是病态的。

### 3.1 SimpleExpert

```python
class SimpleExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2, num_groups=8):
        ...
```

- 结构：`1x1 Conv → GroupNorm → SiLU → 1x1 Conv → GroupNorm`
- 特点：最简洁的标准专家，参数与计算量适中。

### 3.2 OptimizedSimpleExpert

与 `SimpleExpert` 结构相同，但显式声明了 `hidden_dim` 与 `compute_flops` 方法，便于 FLOPs 统计。

### 3.3 SpatialExpert

```python
class SpatialExpert(nn.Module):
    def __init__(self, in_ch, out_ch, expand_ratio=2, num_groups=8):
        ...
```

- 结构：在 `SimpleExpert` 基础上增加 `3x3 DW Conv`（depthwise separable）层。
- 用途：让专家学习空间模式，适合浅层特征图。

### 3.4 GhostExpert / FusedGhostExpert

```python
class FusedGhostExpert(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, ratio=2, num_groups=8):
        ...
```

- 基于 GhostNet 思想：通过廉价操作（`cheap_operation`）从 primary 特征生成额外特征，减少一半参数量。
- `FusedGhostExpert` 与 `GhostExpert` 实现几乎相同，仅 `compute_flops` 统计方式略有差异。

### 3.5 InvertedResidualExpert

```python
class InvertedResidualExpert(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio=2, kernel_size=3, num_groups=8):
        ...
```

- MobileNetV2 风格倒残差结构：
  1. `1x1` Pointwise Expand（升维）
  2. `3x3` Depthwise Spatial（空间卷积）
  3. `1x1` Pointwise Project（降维输出）
- 速度比标准卷积快 2-3 倍，参数量更少。

### 3.6 SharedInvertedExpertGroup

```python
class SharedInvertedExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, expand_ratio=2.0,
                 kernel_size=3, top_k=2, weight_threshold=0.0):
        ...
```

- **关键优化**：昂贵的 `expand + depthwise` 共享计算一次，各专家仅通过轻量的 `1x1 projection head` 进行区分。
- 稀疏计算路径：在训练/正常推理时，仅计算被激活的专家投影；ONNX 导出时回退到密集路径（`stack` + `gather`）。
- `weight_threshold`：可跳过低权重专家的计算（推理加速）。

### 3.7 EfficientExpertGroup

```python
class EfficientExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        ...
```

- 使用 `DepthwiseSeparableConv` 的浅层封装。
- `ES_MOE` 的默认专家类型，支持不同卷积核尺寸（3x3, 5x5, 7x7...）的异构专家配置。

### 3.8 DepthwiseSeparableConv

标准 `depthwise + pointwise + BN + SiLU` 组合，被 `EfficientExpertGroup` 引用。

---

## 4. 路由模块 (routers.py)

路由器负责为每个输入样本生成专家选择概率，并输出 Top-K 专家索引与权重。

### 4.1 UltraEfficientRouter（极致效率路由）

```python
class UltraEfficientRouter(nn.Module):
    def __init__(self, in_channels, num_experts, reduction=16, top_k=2,
                 noise_std=1.0, temperature=1.0, pool_scale=8):
        ...
```

- **核心优化**：
  1. 深度可分离卷积（Depthwise + Pointwise）替代标准卷积。
  2. 激进降采样：`avg_pool2d(kernel_size=pool_scale, stride=pool_scale)`，FLOPs 降低约 95%。
  3. 通道压缩：`reduced_channels = max(in_channels // reduction, 4)`。
- 训练时注入高斯噪声（`noise_std`）到 logits，防止专家坍缩。
- Z-loss 计算在噪声注入之后、softmax 之前，确保正则化的目标与实际路由决策一致。
- 输出五元组：`(topk_vals, topk_indices, usage_frequency, importance, z_loss_metric)`

### 4.2 BaseRouter

抽象基类，封装统一的 logits 处理逻辑：

1. 训练时注入噪声（Gumbel-Softmax trick）
2. Softmax 归一化
3. Top-K 选择
4. 权重归一化
5. 收集训练所需的 `loss_dict`（`router_logits`, `router_probs`, `topk_indices`）

### 4.3 EfficientSpatialRouter

```python
class EfficientSpatialRouter(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0, pool_scale=4):
        ...
```

- 预池化（`pool_scale=4`）后接标准卷积路由。
- 输出 `(routing_weights, routing_indices, loss_dict)`，被 `OptimizedMOE` 使用。

### 4.4 AdaptiveRoutingLayer

```python
class AdaptiveRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        ...
```

- 使用 `AdaptiveAvgPool2d(1)` 将特征图压缩到 `1x1`，完全忽略空间信息，仅利用通道统计量。
- FLOPs 极低，适合计算资源受限的场景。

### 4.5 LocalRoutingLayer

```python
class LocalRoutingLayer(BaseRouter):
    def __init__(self, in_channels, num_experts, reduction=8, top_k=2, noise_std=1.0):
        ...
```

- 默认 `pool_scale=2`，保留更多局部纹理信息。
- 适合小目标检测任务。

### 4.6 AdvancedRoutingLayer

兼容旧版 checkpoint 的路由器。支持：
- 运行时动态创建 `avg_pool` / `softmax` / `router`（缺失属性修复）
- 通道不匹配时通过 `_proj` 进行动态映射（导出时禁止）
- 当 `top_k == num_experts` 时退化为全 Softmax 路由

### 4.7 DynamicRoutingLayer

```python
class DynamicRoutingLayer(nn.Module):
    def __init__(self, in_channels, num_experts=3, reduction=8, top_k=None):
        ...
```

- `ES_MOE` 的默认路由器。
- 支持**软 Top-K**（训练/导出时，可微分）与**硬 Top-K**（推理时，真稀疏）两种模式。
- ONNX 导出时统一使用软 Top-K 路径，避免动态控制流破坏追踪。

---

## 5. 核心 MoE 模块 (modules.py)

modules.py 是 MoE 子系统的核心，包含 20 余种 MoE 变体，按演进顺序排列。

### 5.1 UltraOptimizedMoE（推荐基础版）

```python
class UltraOptimizedMoE(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2,
                 expert_type='simple', router_reduction=16, router_pool_scale=8,
                 noise_std=1.0, router_temperature=1.0,
                 balance_loss_coeff=1.0, router_z_loss_coeff=1.0,
                 num_groups=8, weight_threshold=0.01):
        ...
```

- **路由**：`UltraEfficientRouter`（深度可分离 + 8x 降采样）
- **专家**：可选 `simple` / `ghost` / `inverted`
- **共享路径**：`shared_expert = Conv2d + GroupNorm + SiLU`
- **稀疏计算**：使用 `BatchedExpertComputation.compute_sparse_experts_batched` 进行批量专家并行计算
- **辅助损失**：`differentiable_balance_loss`（GShard 形式，`importance` 保持梯度流至路由器）+ Z-loss
- **DDP 感知**：`reduce_ddp=True` 对 `importance` 与 `usage` 进行跨卡平均
- **激活钳位**：`clamp_(-1e4, 1e4)` 防止路由坍缩导致下游 NaN

**forward 流程**：

1. `routing(x)` → `(weights, indices, usage_freq, importance, z_loss)`
2. `shared_expert(x)` → 共享输出
3. `BatchedExpertComputation.compute_sparse_experts_batched(...)` → 稀疏专家输出
4. `output = shared_output + expert_output`
5. 训练时计算 `aux_loss` → 写入 `MOE_LOSS_REGISTRY`

### 5.2 AdaptiveCapacityMoE

```python
class AdaptiveCapacityMoE(UltraOptimizedMoE):
    def __init__(self, *args, capacity_factor=1.5, **kwargs):
        ...
```

- **动态容量调整**：不改变离散 `top_k`，而是通过 `complexity_estimator` 学习一个复杂度因子 `scale ∈ [1/cf, cf]`，对专家输出进行加权缩放。
- `scale = torch.exp((2.0 * s - 1.0) * math.log(capacity_factor))`
- 复杂度越高 → 专家贡献越大；复杂度越低 → 专家贡献越小。
- **无 GPU→CPU 同步**：所有操作均为纯张量运算，支持 MPS/DDP。

### 5.3 ES_MOE（早期实现）

```python
class ES_MOE(nn.Module):
    def __init__(self, in_channels, out_channels=None, num_experts=3, reduction=8,
                 top_k=None, use_sparse_inference=True, dynamic_threshold=0.4):
        ...
```

- **异构专家**：默认使用不同卷积核（3x3, 5x5, 7x7...）的 `EfficientExpertGroup`
- **动态路由**：`DynamicRoutingLayer`，支持 Soft/Hard Top-K
- **训练/推理分离**：训练时 dense forward（计算所有专家），推理时 sparse forward（仅计算 Top-K）
- **动态剪枝**：`dynamic_threshold` 在推理时进一步过滤低置信度专家
- **负载均衡**：`gshard_balance_loss` 形式，DDP 感知

### 5.4 OptimizedMOE

```python
class OptimizedMOE(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2,
                 expert_expand_ratio=2, balance_loss_coeff=1.0, z_loss_coeff=1.0):
        ...
```

- **共享专家**：引入 `shared_expert` 路径，稳定梯度。
- **路由**：`EfficientSpatialRouter`（预池化 4x）
- **专家**：同构 `SimpleExpert`
- **辅助损失**：使用 `MoELoss` 类统一计算（`balance_loss + z_loss`）
- **ONNX 兼容**：导出时自动切换为密集路径（`stack` + `gather`）

### 5.5 OptimizedMOEImproved（别名 ModularRouterExpertMoE）

```python
class OptimizedMOEImproved(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2,
                 expert_type='simple', router_type='efficient', noise_std=1.0,
                 balance_loss_coeff=1.0, router_z_loss_coeff=1.0,
                 expert_expand_ratio=2.0, progressive_sparsity=True,
                 detach_routing=False, add_residual=True):
        ...
```

- **可插拔设计**：`expert_type ∈ {simple, ghost, inverted, spatial}`，`router_type ∈ {efficient, local, adaptive}`
- **渐进稀疏**：`progressive_sparsity=True` 时，前 `warmup_steps=5000` 步从 `num_experts` 线性下降到 `top_k`，避免早期训练专家未充分学习就被稀疏跳过。
- **专家 Dropout**：训练时每 100 步随机丢弃 15% 专家，防止均匀路由。DDP 安全：通过固定种子 `torch.Generator` 确保所有 rank 丢弃相同的专家。
- **路由梯度隔离**：`detach_routing=True` 时，主任务损失不回流到路由器（路由器仅通过 aux loss 学习）。
- **残差控制**：`add_residual=False` 时，由外层模块（如 `ABlockMoE`）管理残差，避免双重相加。

### 5.6 ABlockMoE / A2C2fMoE（与 YOLO 注意力块集成）

```python
class ABlockMoE(ABlock):
    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1, num_experts=4, top_k=2, expert_type='simple'):
        ...

class A2C2fMoE(A2C2f):
    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0,
                 e=0.5, g=1, shortcut=True, num_experts=4, top_k=2, expert_type='simple'):
        ...
```

- `ABlockMoE`：将 `ABlock` 中的 MLP 替换为 `OptimizedMOEImproved`，保留 Area-Attention 的残差结构。
- `A2C2fMoE`：在 `A2C2f` 的 `self.m` 中用 `ABlockMoE` 替换标准 `ABlock`。
- `aux_loss` 委托：外层模块的 `aux_loss` 属性递归收集内层 `ABlockMoE` / `OptimizedMOEImproved` 的辅助损失。

### 5.7 AdaptiveGateMoE（v0.4）

```python
class AdaptiveGateMoE(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2,
                 split_ratio=0.5, num_groups=8, initial_temperature=1.0,
                 final_temperature=0.5, balance_loss_coeff=1.0,
                 router_z_loss_coeff=1.0, entropy_loss_coeff=0.01):
        ...
```

- **SE-Gated 通道分配**：用 Squeeze-and-Excitation 模块学习静态/动态路径的最优通道分配，替代固定 `split_ratio`。
- **双流路由**：`DualStreamGateRouter` 合并全局统计流（近似零开销）与局部空间流（轻量 DW-Conv）。
- **稳定复杂度估计器**：`complexity_estimator` 输出 clamp 到 `[0.3, 1.5]`，避免 NaN 与退化。
- **温度退火**：余弦退火从 `initial_temperature` → `final_temperature`，前 2000 步完成。
- **软均衡**：`MoELoss(..., use_soft_balancing=True)` 保持梯度流至路由器。

### 5.8 DualStreamGateRouter / DualStreamGateRouterV2

```python
class DualStreamGateRouter(nn.Module):
    def __init__(self, in_channels, num_experts, top_k, temperature=1.0,
                 local_reduction=16, pool_scale=4):
        ...

class DualStreamGateRouterV2(DualStreamGateRouter):
    def __init__(self, in_channels, num_experts, top_k, temperature=1.0,
                 local_reduction=16, pool_scale=4, noise_std=0.1):
        ...
```

- **Stream A（全局）**：`mean + std` 通道统计 → `Linear` 投影到专家 logits。
- **Stream B（局部）**：轻量 `DW-Conv + PW-Conv` 提取空间线索，再平均到 logits。
- **Merge**：`alpha = sigmoid(self.alpha)` 混合两个流。
- **V2 改进**：
  1. `LayerNorm` 归一化通道统计，减少 batch/层间尺度差异。
  2. `expert_prior` 可学习参数，作为无辅助损失的先验负载均衡偏置（DDP 自动 all-reduce）。
  3. Switch-Transformer 风格噪声：训练时线性衰减至 0，前 50% 训练步有效。

### 5.9 HyperSplitMoE / HyperFusedMoE / HyperUltimateMoE

这三个模块构成“Hyper”系列，以**通道分割 + 融合专家**为核心架构。

**HyperSplitMoE**：
- 静态路径：`DW-Conv + PW-Conv`（处理固定比例的通道）
- 动态路径：`InvertedResidualExpert` 专家组 + 全局池化路由
- 输出：`concat → 1x1 Conv + BN + residual`

**HyperFusedMoE**：
- 路由：`ZeroCostRouter`（复用 `mean/std` 统计量，仅一个 `Linear`）或 `UltraEfficientRouter`
- 专家：`FusedExpertGroup`（所有专家权重融合为一个大卷积，通过 `grouped conv` 隔离）
- 自适应负载均衡：`AdaptiveBalanceController` 动态调整 `balance_loss` 系数
- 渐进稀疏：同 `OptimizedMOEImproved`

**HyperUltimateMoE**：
- 整合通道分割、融合专家、缓存路由。
- `MatMulFusedExperts`（`FusedExpertGroup` 别名）作为核心计算后端。
- 支持 `capacity_factor` 自适应容量缩放。

### 5.10 FusedExpertGroup / LowRankFusedExpertGroup

```python
class FusedExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, num_groups=8, top_k=2):
        ...
```

- **融合卷积**：所有专家的权重合并为一个 `nn.Conv2d(in_channels, num_experts * out_channels, ...)`，通过 `groups` 参数隔离。
- **Top-K 优先 gather**：先 `gather` 出 Top-K 专家输出，再对这部分应用 `GroupNorm` 与激活，避免为未激活专家做无用计算。
- **向量化归一化**：利用 `F.group_norm` 的批量处理能力，对 `B * top_k` 个实例一次性归一化。
- **Legacy 兼容**：`_load_from_state_dict` 支持从旧版 `expert_norms.{i}.weight` 映射到新的 `expert_norm_weight` 表。

```python
class LowRankFusedExpertGroup(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts, num_groups=8, top_k=2,
                 bottleneck_ratio=0.5, min_channels=16):
        ...
```

- 在 `FusedExpertGroup` 前增加 `1x1 bottleneck`，先压缩通道再进入专家卷积，降低 P3/P4 大特征图的计算量。

### 5.11 ZeroCostRouter / UltraLightRouter

```python
class ZeroCostRouter(nn.Module):
    def __init__(self, in_channels, num_experts, top_k, temperature=1.0):
        ...
```

- **零成本路由**：直接复用特征图的 `mean` 与 `std`（BN 计算中已有）作为路由信号。
- 仅需一个 `Linear(2*in_channels, num_experts)`，FLOPs 降低 95% 以上。
- `UltraLightRouter` 继承 `ZeroCostRouter`，预留缓存接口（当前未启用缓存以避免形状不匹配）。

### 5.12 AdaptiveBalanceController

```python
class AdaptiveBalanceController(nn.Module):
    def __init__(self, num_experts, initial_coeff=1.0, final_coeff=0.1,
                 decay_steps=50000, dynamic_scheduler=None, dynamic_scheduler_config=None):
        ...
```

- **动态系数衰减**：从 `initial_coeff` 线性衰减到 `final_coeff`。
- **GShard 可微均衡**：`importance = mean(router_probs)` 保持梯度流；`expert_importance` 可学习参数作为目标分布先验。
- **熵正则化**：惩罚低熵（鼓励均匀分布），`max_entropy = log(N)` 时惩罚为 0。
- **Gini 动态调度**：可选集成 `MoEDynamicScheduler`，根据 Gini 系数自动调整 balance 系数。
- **非负性保证**：输出通过 `torch.nan_to_num` 与 `clamp` 确保非负。

### 5.13 v0.11 - v0.15 演进系列

#### HybridAdaptiveGateMoE / HybridAdaptiveGateMoEv2（v0.6 / v0.11）
- `HybridAdaptiveGateMoE`：根据 `num_experts` 阈值自动选择 `fused`（专家数 ≤ 8）或 `shared_inverted`（专家数 > 8）后端。
- `HybridAdaptiveGateMoEv2`：仅升级路由器为 `DualStreamGateRouterV2`，其余结构与 v0.6 完全一致。
- **通道混洗**：`_channel_shuffle` 改善静态/动态特征交互。

#### LowRankHybridAdaptiveGateMoE（v0.7）
- 在 fused 后端前增加 `LowRankFusedExpertGroup` bottleneck，降低大特征图成本。

#### RefinedLowRankHybridAdaptiveGateMoE（v0.8）
- 在 v0.7 基础上增加轻量残差 DW 精化块（`feature_refiner + feature_gate`），通过全局 SE 门控自适应增强边界/纹理通道。

#### DetailAwareLowRankHybridAdaptiveGateMoE（v0.9）
- 在动态分支前增加 `VisualDetailGate`，增强高频细节（边界、纹理）信息，让路由器和专家感知小目标线索。

#### ContextRefinedLowRankHybridAdaptiveGateMoE（v0.10）
- 在 v0.8 基础上增加 `PyramidContextMixer`，聚合多尺度上下文（`2x, 4x` 池化 + 插值），适合检测/分割特征。

#### VisualEnhancedAdaptiveGateMoE（v0.10+）
- 同时集成 `detail_gate` 与 `context_mixer`，是最丰富的视觉 MoE 块，用于消融实验。

#### OptimalHybridGateMoE（v0.12，生产推荐）
- 综合 v0.1-v0.11 的消融结论，保留最佳组合：
  - SE-gated 通道分割
  - `DualStreamGateRouterV2`（归一化 + 先验偏置）
  - 混合专家后端（fused / shared-inverted）
  - 通道混洗 + 复杂度门控
  - 可选轻量 DW 精化（`refine=True`）
  - 层自适应 `split_ratio`（浅层 P3 动态容量多，深层 P5 静态容量多）
- 所有状态为 `nn.Parameter` 或 Python int，DDP 安全。

#### MultiHeadRouterMoE（v0.13）
- 将 `DualStreamGateRouterV2` 升级为 `MultiHeadRouterV3`：
  - 多 head 并行投影（`num_heads=4`）
  - 全局残差投影保持完整统计视图
  - 可学习温度加权融合
  - 专家 dropout（软缩放而非置零）
- 其余结构与 v0.12 相同。

#### DiversifiedExpertMoE（v0.14）
- 将 `SharedInvertedExpertGroup` 替换为 `DiversifiedExpertGroup`：
  - 共享 expand 层
  - 每个专家使用不同 `dilation` 率的 `3x3 DW`（1, 1, 2, 2...），实现不同有效感受野的功能多样性
  - 独立投影头

#### GatedFusionMoE（v0.15）
- 引入 `CrossPathGate`：基于静态/动态路径实际输出内容的学习门控融合，替代简单 concat。
- 温和 Drop-Path：仅对投影残差进行随机丢弃，保持 identity 路径存活。
- 适合深层 P5 正则化，防止小数据集过拟合。

### 5.14 UltimateOptimizedMoE（Hyper 系列最终版）

```python
class UltimateOptimizedMoE(nn.Module):
    def __init__(self, in_channels, out_channels, num_experts=4, top_k=2,
                 split_ratio=0.5, num_groups=8, use_routing_cache=True,
                 capacity_factor=1.5, initial_temperature=2.0,
                 final_temperature=0.5, entropy_coeff=0.01):
        ...
```

- 整合 `HyperUltimateMoE` 所有特性，增加：
  - 动态温度退火（与渐进稀疏同步）
  - 熵损失系数注入
  - AMP 混合精度加速（仅 CUDA）
  - `balance_loss_coeff` / `router_z_loss_coeff` 桥接属性，支持 YAML/CLI 注入

---

## 6. 辅助损失 (loss.py)

### 6.1 MoELoss

```python
class MoELoss(nn.Module):
    def __init__(self, balance_loss_coeff=1.0, z_loss_coeff=1.0,
                 entropy_loss_coeff=0.0, diversity_loss_coeff=0.0,
                 variance_loss_coeff=0.0, num_experts=8, top_k=2,
                 use_soft_balancing=False, coeff_floor=0.0,
                 dynamic_scheduler=None, dynamic_scheduler_config=None):
        ...
```

**损失组成**：

1. **Load Balancing Loss**
   - `use_soft_balancing=False`（Hard）：使用离散 Top-K 计数 `usage = counts / total`，梯度仅通过 `importance = mean(router_probs)` 回流路由器。
   - `use_soft_balancing=True`（Soft）：`importance` 与 `usage` 均保留梯度，标准 GShard 形式 `N * sum(importance * usage)`。
   - 支持 `target_usage` 加权与 DDP 跨卡平均。

2. **Z-Loss**
   - `torch.logsumexp(router_logits, dim=1).pow(2).mean()`
   - 防止 logits 数值爆炸，提升数值稳定性。

3. **Entropy Loss**（可选）
   - `-sum(probs * log(probs))`，惩罚路由器犹豫不决（低熵）。

4. **Diversity Loss**（可选）
   - 要求 `expert_outputs=[B, E, D]`，计算专家输出两两余弦相似度，惩罚非正交性。
   - 要求 `E >= 2`（`E==1` 时自动返回 0）。

5. **Variance Loss**（可选）
   - 直接惩罚 usage 与均匀分布的方差。

6. **动态调度**
   - 集成 `MoEDynamicScheduler`，根据 Gini 系数实时调整 `balance_loss_coeff`。

7. **NaN Guard**
   - 若 total_loss 非有限，自动替换为 0，并周期性警告。

### 6.2 独立损失函数

```python
def gshard_balance_loss(expert_usage, num_experts, reduce_ddp=False)
def weighted_gshard_balance_loss(expert_usage, target_usage, num_experts, reduce_ddp=False)
def differentiable_balance_loss(router_probs, expert_usage, num_experts, target_usage=None, reduce_ddp=False)
def all_reduce_mean(tensor)
```

- `all_reduce_mean`：DDP 安全的平均归约，单卡/CPU 时无操作（no-op）。
- 所有跨卡 reduce 均在 `float32` 下完成，避免 `float16` 累积误差。

---

## 7. 工具与诊断

### 7.1 BatchedExpertComputation (utils.py)

```python
class BatchedExpertComputation:
    @staticmethod
    def compute_sparse_experts_batched(x, experts, routing_weights, routing_indices, top_k, num_experts):
        ...
```

- **稀疏路径**：遍历每个专家，通过 `torch.where` 收集选中该专家的 `(batch, k)` 位置，一次性输入专家网络，结果通过 `index_add_` 累加。
- **密集路径（ONNX）**：计算所有专家输出 → `torch.stack` → `torch.gather` + 加权求和。
- **权重阈值**：训练时 `threshold=0.0`（保留所有选中专家以学习），推理时 `threshold=0.01`（跳过低权重专家）。
- **激活钳位**：`clamp_(-1e4, 1e4)` 防止溢出。

### 7.2 FlopsUtils

```python
class FlopsUtils:
    @staticmethod
    def count_conv2d(layer, input_shape):
        ...
```

- 支持 `nn.Conv2d` 与 `nn.Sequential` 的 FLOPs 统计。
- 自动推导输出空间尺寸（考虑 `padding`, `stride`, `dilation`）。
- 包含 `bias` 与 `groups` 的精确计算。

### 7.3 诊断工具

#### ExpertUsageTracker (analysis.py)
- 通过 `register_forward_hook` 自动挂载到所有路由器上。
- 支持密集权重 `[B, E, H, W]` 与稀疏 Top-K `(vals, indices)` 两种模式。
- 生成：
  - 终端报告（每专家使用百分比、状态、标准差）
  - 热力图 (`expert_usage_heatmap.png`)
  - 柱状图 (`expert_usage_bar.png`)

#### RoutingCollapseDetector (analysis.py)
- 实时检测路由坍缩（单专家占比 > 80%）与死专家（使用 < 5%）。
- 自动生成恢复动作：增加 balance loss、增加噪声、重新初始化死专家。

#### MoEDiagnosticsRecorder (history.py)
- 将诊断数据持久化为 `routing_history.jsonl` / `routing_history.csv`。
- 滚动窗口检测：`dead_window` 与 `collapse_window` 步内持续异常则触发 alert。
- 导出趋势图：`{layer}_usage.png` + `aux_loss_vs_step.png`。

#### MoELayerDiagnostic (diagnostics.py)
- 轻量 `dataclass`，结构化存储单层的：名称、专家数、top_k、aux_loss、usage、counts、主导专家占比、坍缩标志等。

### 7.4 MoEDynamicScheduler (scheduler.py)

```python
@dataclass
class MoEDynamicSchedulerConfig:
    enabled: bool = True
    target_gini: float = 0.25
    gain: float = 1.5
    min_balance_coeff: float = 0.02
    max_balance_coeff: float = 2.0
    ema_momentum: float = 0.9
```

- **Gini 系数**：`compute_gini(expert_usage)` 衡量专家使用不平等度。
- **调度公式**：`coeff = base_coeff * (1 + gain * (ema_gini - target_gini))`
- 高 Gini → 路由不平衡 → 增大 balance 系数；低 Gini → 路由健康 → 放松约束。
- 状态可序列化：`state_dict()` / `load_state_dict()`。

### 7.5 MoEPruner (pruning.py)

```python
class MoEPruner:
    def __init__(self, model_path, threshold=0.15, dataset='coco8.yaml', device=None):
        ...
```

- **五步剪枝流水线**：
  1. 加载模型
  2. 运行验证收集 usage 统计
  3. 创建剪枝计划（保留使用率 ≥ threshold 的专家）
  4. 执行手术（裁剪专家 + 裁剪路由器投影层权重）
  5. 保存并验证
- 自动设备检测：CUDA → MPS → CPU。
- 安全约束：至少保留 1 个专家；若保留数 < `top_k` 则警告。

---

## 8. 与 YOLO 架构的集成

### 8.1 DyMoEBlock (block.py)

```python
class DyMoEBlock(nn.Module):
    def __init__(self, dim, num_experts=4, top_k=2, mlp_ratio=2.0):
        ...
```

- 动态 MoE 块，将 `nn.Linear` MLP 替换为 `BatchedExpertComputation` 驱动的稀疏专家 MLP。
- 结构：`LayerNorm → MLP(MoE) → gamma 缩放` + 残差
- 使用 `differentiable_balance_loss` 计算负载均衡，并写入 `MOE_LOSS_REGISTRY`。

### 8.2 ABlockMoE / A2C2fMoE (modules.py)

已在 5.6 节详述。这些模块将 MoE 嵌入到 YOLO 的 Area-Attention 块中，通过 `aux_loss` 属性递归收集辅助损失。

### 8.3 模型配置 (YAML)

在 `yolo-master-v0_11.yaml` 等配置中，MoE 模块可通过以下方式插入：

```yaml
# 示例：在 C2f 块中插入 A2C2fMoE
- [-1, 1, A2C2fMoE, [512, 1, True, 1, 4, 2, 'simple']]  # c2, n, a2, area, num_experts, top_k, expert_type
```

各位置参数含义：
- `c2`：输出通道
- `n`：重复次数
- `a2`：是否启用 Area-Attention
- `area`：Area 大小
- `num_experts`：专家数量
- `top_k`：激活专家数
- `expert_type`：专家类型

---

## 9. 使用示例

### 9.1 基础用法

```python
import torch
from ultralytics.nn.modules.moe import OptimizedMOEImproved

# 创建 MoE 模块
moe = OptimizedMOEImproved(
    in_channels=64,
    out_channels=64,
    num_experts=4,
    top_k=2,
    expert_type='simple',
    router_type='efficient',
    balance_loss_coeff=1.0,
    router_z_loss_coeff=1.0,
)

moe.train()
x = torch.randn(2, 64, 32, 32)
out = moe(x)

# 获取辅助损失
aux_loss = moe.aux_loss
print(f"Output shape: {out.shape}, Aux loss: {aux_loss.item():.4f}")
```

### 9.2 使用 UltraOptimizedMoE 进行高效推理

```python
from ultralytics.nn.modules.moe import UltraOptimizedMoE

moe = UltraOptimizedMoE(
    in_channels=128, out_channels=128, num_experts=8, top_k=2,
    expert_type='ghost', router_pool_scale=8, num_groups=8
)
moe.eval()

with torch.no_grad():
    out = moe(torch.randn(1, 128, 64, 64))

# FLOPs 统计
flops = moe.get_gflops((1, 128, 64, 64))
print(f"Total GFLOPs: {flops['total_gflops']:.4f}")
print(f"  Router: {flops['routing']:.4f}")
print(f"  Shared: {flops['shared_expert']:.4f}")
print(f"  Sparse: {flops['sparse_experts']:.4f}")
```

### 9.3 使用专家使用分析器

```python
from ultralytics.nn.modules.moe import ExpertUsageTracker
from ultralytics import YOLO

model = YOLO("yolo-master.pt")

with ExpertUsageTracker(model.model) as tracker:
    model.val(data="coco8.yaml", split="val", batch=1, verbose=False, device="cpu")
    tracker.print_report()
# 自动生成 expert_usage_heatmap.png 与 expert_usage_bar.png
```

### 9.4 使用 MoE 剪枝

```python
from ultralytics.nn.modules.moe import prune_moe_model

success = prune_moe_model(
    model_path="yolo-master.pt",
    output_path="yolo-master-pruned.pt",
    threshold=0.15,  # 保留使用率 ≥ 15% 的专家
    dataset="coco8.yaml"
)
print("Pruning success:", success)
```

### 9.5 使用动态调度器

```python
from ultralytics.nn.modules.moe import MoELoss, MoEDynamicScheduler, MoEDynamicSchedulerConfig

scheduler_config = MoEDynamicSchedulerConfig(
    target_gini=0.25, gain=1.5,
    min_balance_coeff=0.02, max_balance_coeff=2.0
)

loss_fn = MoELoss(
    balance_loss_coeff=1.0, z_loss_coeff=1.0,
    num_experts=4, top_k=2,
    dynamic_scheduler_config=scheduler_config
)

probs = torch.softmax(torch.randn(8, 4), dim=1)
logits = torch.randn(8, 4)
indices = torch.topk(probs, 2, dim=1).indices

result = loss_fn(probs, logits, indices, return_dict=True)
print(f"Total loss: {result['loss']:.4f}")
print(f"Balance loss: {result['balance_loss']:.4f}")
print(f"Z-loss: {result['z_loss']:.4f}")
print(f"Dynamic schedule: {result['dynamic_schedule']}")
```

---

## 10. 版本演进与选型建议

### 10.1 演进时间线

| 版本 | 核心模块 | 关键改进 |
|:---|:---|:---|
| v0.1 | `ES_MOE` | 基础概念验证，异构专家 |
| v0.2 | `OptimizedMOE` | 引入 Shared Expert，稳定训练 |
| v0.3 | `OptimizedMOEImproved` | 可插拔路由/专家，Z-Loss |
| v0.4 | `AdaptiveGateMoE` | SE-Gated 分割，DualStream 路由 |
| v0.5 | `FusedAdaptiveGateMoE` | 融合专家组 |
| v0.6 | `HybridAdaptiveGateMoE` | 混合后端（fused/shared）+ 通道混洗 |
| v0.7 | `LowRankHybrid...` | 低秩 bottleneck |
| v0.8 | `RefinedLowRank...` | 残差特征精化 |
| v0.9 | `DetailAware...` | 视觉细节门控 |
| v0.10 | `ContextRefined...` | 金字塔上下文聚合 |
| v0.11 | `HybridAdaptiveGateMoEv2` | V2 归一化路由器 + 先验偏置 |
| v0.12 | `OptimalHybridGateMoE` | **生产推荐**：综合最优组合 |
| v0.13 | `MultiHeadRouterMoE` | 多 head 并行路由 |
| v0.14 | `DiversifiedExpertMoE` | 异构感受野专家 |
| v0.15 | `GatedFusionMoE` | 跨路径门控融合 + Drop-Path |

### 10.2 场景化选型建议

| 场景 | 推荐模块 | 配置建议 |
|:---|:---|:---|
| **通用生产部署** | `OptimalHybridGateMoE` (v0.12) | `num_experts=4, top_k=2, refine=True` |
| **极致推理速度** | `UltraOptimizedMoE` | `expert_type='ghost', router_pool_scale=8` |
| **移动端/边缘** | `UltraOptimizedMoE` + `GhostExpert` | `num_experts=4, top_k=2`，使用 GroupNorm |
| **小目标检测** | `DetailAwareLowRankHybridAdaptiveGateMoE` | `detail_reduction=8` |
| **大分辨率 P3/P4** | `LowRankHybridAdaptiveGateMoE` | `bottleneck_ratio=0.5` |
| **科研消融** | `OptimizedMOEImproved` | `progressive_sparsity=True`，可插拔路由/专家 |
| **旧模型兼容** | `ES_MOE` / `OptimizedMOE` | 保持 backward compatibility |

### 10.3 关键注意事项

1. **GroupNorm vs BatchNorm**：所有 MoE 专家使用 `GroupNorm`，小 batch 稳定，但推理时无法像 BN 那样融合进 Conv。
2. **ONNX 导出**：稀疏路径会自动切换为密集路径，确保 `opset_version >= 11`（推荐 13+）。
3. **DDP 训练**：确保 `reduce_ddp=True`，避免各卡优化不同局部平衡目标。
4. **渐进稀疏**：短训练计划（如 coco128）建议关闭 `progressive_sparsity` 或减小 `warmup_steps`。
5. **deepcopy/EMA**：所有核心模块已覆写 `__deepcopy__`，但避免在 `forward` 中创建新的 `nn.Module` 实例。

---

*文档基于代码版本 v260703 编写，实际 API 可能随版本迭代微调。*
