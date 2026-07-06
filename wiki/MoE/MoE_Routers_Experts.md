# MoE Routers & Experts 技术文档

> **版本**: v260703  
> **模块路径**: `ultralytics/nn/modules/moe/`  
> **最后更新**: 基于 2026-06-25 代码审计后的最新实现

---

## 目录

1. [架构概述](#1-架构概述)
2. [路由器层 (Routers)](#2-路由器层-routers)
3. [专家网络 (Experts)](#3-专家网络-experts)
4. [MoE 核心模块](#4-moe-核心模块)
5. [辅助损失系统](#5-辅助损失系统)
6. [工具与计算优化](#6-工具与计算优化)
7. [动态调度器](#7-动态调度器)
8. [诊断与监控](#8-诊断与监控)
9. [使用示例](#9-使用示例)
10. [性能参考](#10-性能参考)

---

## 1. 架构概述

YOLO-Master 的 **Mixture-of-Experts (MoE)** 子系统采用稀疏激活策略，在保持模型容量的同时显著降低推理计算量。核心设计遵循 **GShard / Switch Transformer** 范式：

- **Top-K 稀疏路由**: 每个输入 token 只激活 `top_k` 个专家（典型配置 `top_k=2`, `num_experts=4`）
- **共享专家 (Shared Expert)**: 所有输入均经过共享路径，稳定梯度流
- **负载均衡损失**: 防止路由坍塌（所有输入涌向单一专家）
- **Z-Loss 正则**: 约束 router logits 的数值幅度，提升训练稳定性
- **GroupNorm 稳定性**: 专家使用 `GroupNorm` 替代 `BatchNorm`，解决 Top-K 路由后 batch size 过小（甚至为 1）导致的统计不稳定问题

### 1.1 模块文件结构

```
ultralytics/nn/modules/moe/
├── __init__.py       # 公共 API 导出
├── modules.py        # 核心 MoE 模块（UltraOptimizedMoE, ES_MOE 等）
├── routers.py        # 路由器实现
├── experts.py        # 专家网络实现
├── loss.py           # 辅助损失函数
├── utils.py          # FLOPs 计算、批处理优化工具
├── scheduler.py      # Gini 驱动的动态系数调度
├── analysis.py       # 专家使用分析工具
├── diagnostics.py    # 逐层诊断工具
├── history.py        # 训练历史记录
└── pruning.py        # 专家剪枝工具
```

---

## 2. 路由器层 (Routers)

路由器负责为每个输入空间位置计算专家分配权重。所有路由器继承自 `nn.Module`，遵循统一的输出接口。

### 2.1 UltraEfficientRouter

**文件**: `ultralytics/nn/modules/moe/routers.py`  
**定位**: 核心推荐路由器，FLOPs 较基线降低约 **95%**

```python
class UltraEfficientRouter(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        reduction: int = 16,
        top_k: int = 2,
        noise_std: float = 1.0,
        temperature: float = 1.0,
        pool_scale: int = 8
    )
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `in_channels` | `int` | — | 输入特征通道数 |
| `num_experts` | `int` | — | 专家总数 |
| `reduction` | `int` | `16` | 通道压缩比，中间通道 = `max(in_channels // reduction, 4)` |
| `top_k` | `int` | `2` | 每个位置激活的专家数 |
| `noise_std` | `float` | `1.0` | 训练时注入的 Gumbel 噪声标准差（仅 `training=True` 时生效） |
| `temperature` | `float` | `1.0` | Softmax 温度，控制分布锐度；最小值被钳制为 `1e-3` |
| `pool_scale` | `int` | `8` | 下采样倍数，通过 `avg_pool2d` 降低路由计算量 |

#### 核心优化策略

1. **Depthwise-Separable 卷积**: 路由网络使用 `DW + PW` 结构替代标准卷积
2. **激进下采样**: `pool_scale=8` 将特征图分辨率降低 8× 后再计算路由
3. **早期通道压缩**: `reduction=16` 大幅降低中间特征维度
4. **Z-Loss 后置计算**: `z_loss` 在噪声注入和 clamp 之后计算，确保正则目标与实际路由决策一致

#### 返回值

返回 5 元组 `(topk_vals, topk_indices, usage_frequency, importance, z_loss_metric)`：

| 返回项 | 形状 | 训练时 | 推理时 |
|--------|------|--------|--------|
| `topk_vals` | `[B, top_k, 1, 1]` | ✓ | ✓ |
| `topk_indices` | `[B, top_k, 1, 1]` | ✓ | ✓ |
| `usage_frequency` | `[num_experts]` | ✓ | `None` |
| `importance` | `[num_experts]` | ✓ | `None` |
| `z_loss_metric` | `scalar Tensor` | ✓ | `None` |

#### 使用示例

```python
import torch
from ultralytics.nn.modules.moe.routers import UltraEfficientRouter

router = UltraEfficientRouter(
    in_channels=256,
    num_experts=4,
    top_k=2,
    pool_scale=8
)

x = torch.randn(2, 256, 64, 64)
topk_vals, topk_indices, usage_freq, importance, z_loss = router(x)

print(topk_vals.shape)      # torch.Size([2, 2, 1, 1])
print(topk_indices.shape)   # torch.Size([2, 2, 1, 1])
```

---

### 2.2 EfficientSpatialRouter

```python
class EfficientSpatialRouter(BaseRouter):
    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        reduction: int = 8,
        top_k: int = 2,
        noise_std: float = 1.0,
        pool_scale: int = 4
    )
```

标准空间路由器，使用 `Conv2d(3x3) + Conv2d(1x1)` 结构，`pool_scale=4` 的适度下采样。适用于对空间细节有一定要求但需控制计算量的场景。

---

### 2.3 AdaptiveRoutingLayer

```python
class AdaptiveRoutingLayer(BaseRouter):
    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        reduction: int = 8,
        top_k: int = 2,
        noise_std: float = 1.0
    )
```

全局自适应路由器。使用 `AdaptiveAvgPool2d(1)` 将输入压缩为全局向量后计算路由，FLOPs 最低，适合通道维度足够丰富、空间位置对路由决策影响较小的层。

---

### 2.4 LocalRoutingLayer

```python
class LocalRoutingLayer(BaseRouter):
    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        reduction: int = 8,
        top_k: int = 2,
        noise_std: float = 1.0
    )
```

局部空间路由器，`pool_scale=2` 的轻度下采样。保留更多空间细节，适用于需要细粒度空间路由决策的场景（如小目标检测）。

---

### 2.5 DynamicRoutingLayer

```python
class DynamicRoutingLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_experts: int = 3,
        reduction: int = 8,
        top_k: Optional[int] = None
    )
```

支持 **Soft Top-K** 与 **Hard Top-K** 的动态路由器。

- **训练 / ONNX 导出**: 使用 `_soft_top_k`，保持梯度流，图结构静态
- **推理 (eager mode)**: 使用 `_hard_top_k`，产生真正的稀疏性（未选中专家权重精确为 0）

当 `top_k=None` 时退化为标准 Softmax（所有专家参与，无稀疏性）。

---

### 2.6 AdvancedRoutingLayer

```python
class AdvancedRoutingLayer(nn.Module):
    def __init__(
        self,
        in_channels: int = 64,
        num_experts: int = 3,
        top_k: Optional[int] = None
    )
```

**兼容性路由器**，用于加载旧版 checkpoint。支持运行时通道不匹配修复（通过动态注册 `_proj` 层），但在 ONNX 导出期间遇到通道不匹配会抛出 `RuntimeError`。

---

### 2.7 BaseRouter

```python
class BaseRouter(nn.Module):
    def __init__(self, num_experts: int, top_k: int)
```

所有路由器的抽象基类，提供统一的 `_process_logits` 方法：

- 训练噪声注入 (`noise_std`)
- Softmax 概率计算
- Top-K 选择与权重归一化
- 训练时收集 `loss_dict`（包含 `router_logits`, `router_probs`, `topk_indices`）

---

## 3. 专家网络 (Experts)

专家是实际执行特征变换的子网络。所有专家均使用 **GroupNorm** 替代 BatchNorm，以兼容 Top-K 路由后 `batch_size=1` 的极端情况。

### 3.1 SimpleExpert

```python
class SimpleExpert(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 2,
        num_groups: int = 8
    )
```

最简专家结构：

```
Conv2d(1x1) → GroupNorm → SiLU → Conv2d(1x1) → GroupNorm
```

- 中间通道: `in_channels * expand_ratio`
- 参数量小、计算高效，作为默认专家类型

#### 使用示例

```python
from ultralytics.nn.modules.moe.experts import SimpleExpert

expert = SimpleExpert(in_channels=128, out_channels=128, expand_ratio=2)
x = torch.randn(4, 128, 32, 32)
out = expert(x)
assert out.shape == (4, 128, 32, 32)
```

---

### 3.2 OptimizedSimpleExpert

`SimpleExpert` 的别名，功能完全一致。

---

### 3.3 SpatialExpert

```python
class SpatialExpert(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        expand_ratio: float = 2,
        num_groups: int = 8
    )
```

引入 **Depthwise 空间卷积** 的专家：

```
Conv2d(1x1) → GN → SiLU → DW-Conv2d(3x3) → GN → SiLU → Conv2d(1x1) → GN
```

适用于需要专家学习空间位置模式（如边缘、纹理）的场景。

---

### 3.4 GhostExpert / FusedGhostExpert

```python
class GhostExpert(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        ratio: int = 2,
        num_groups: int = 8
    )
```

基于 [GhostNet](https://arxiv.org/abs/1911.11907) 的廉价操作专家：

1. **Primary conv**: 生成 `ceil(out_channels / ratio)` 个固有特征图
2. **Cheap operation**: 通过 depthwise 卷积生成剩余廉价特征图
3. **Concat + Slice**: 拼接后截断至 `out_channels`

`FusedGhostExpert` 是融合版本，减少内存访问次数。

---

### 3.5 InvertedResidualExpert

```python
class InvertedResidualExpert(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        expand_ratio: float = 2,
        kernel_size: int = 3,
        num_groups: int = 8
    )
```

**MobileNetV2 风格**的逆残差专家：

```
Pointwise Expand → GN → SiLU → DW Spatial → GN → SiLU → Pointwise Project → GN
```

相比标准卷积专家，速度提升 **2-3×**，参数量更少。

---

### 3.6 SharedInvertedExpertGroup

```python
class SharedInvertedExpertGroup(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int,
        expand_ratio: float = 2.0,
        kernel_size: int = 3,
        top_k: int = 2,
        weight_threshold: float = 0.0
    )
```

**共享特征提取 + 独立投影头**的高效专家组：

- `shared_feature`: 昂贵的 `expand + depthwise` 计算一次
- `expert_projections`: 每个专家仅含轻量 `1x1 pointwise` 投影
- 通过 `torch.unique` 识别实际激活的专家，仅计算激活分支
- ONNX 导出时自动切换为 dense path（`torch.gather`）

---

### 3.7 EfficientExpertGroup / DepthwiseSeparableConv

```python
class EfficientExpertGroup(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1
    )
```

ES_MOE 使用的标准专家，内部封装 `DepthwiseSeparableConv`。

---

## 4. MoE 核心模块

### 4.1 UltraOptimizedMoE

**文件**: `ultralytics/nn/modules/moe/modules.py`  
**定位**: 最推荐的生产级 MoE 模块

```python
class UltraOptimizedMoE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = 'simple',          # 'simple' | 'ghost' | 'inverted'
        router_reduction: int = 16,
        router_pool_scale: int = 8,
        noise_std: float = 1.0,
        router_temperature: float = 1.0,
        balance_loss_coeff: float = 1.0,
        router_z_loss_coeff: float = 1.0,
        num_groups: int = 8,
        weight_threshold: float = 0.01
    )
```

#### 前向流程

```python
def forward(self, x):
    # 1. 路由计算 → topk_vals, topk_indices, usage_freq, importance, z_loss
    routing_result = self.routing(x)
    
    # 2. 共享专家（所有输入经过）
    shared_output = self.shared_expert(x)
    
    # 3. 稀疏专家计算（仅 top_k 个专家）
    expert_output = BatchedExpertComputation.compute_sparse_experts_batched(...)
    
    # 4. 融合输出
    output = shared_output + expert_output
    
    # 5. 训练时计算辅助损失并写入 MOE_LOSS_REGISTRY
    if self.training:
        balance_loss = differentiable_balance_loss(...)
        aux_loss = balance_loss_coeff * balance_loss + router_z_loss_coeff * z_loss
        _registry_set(self, aux_loss)
```

#### 关键特性

| 特性 | 说明 |
|------|------|
| **aux_loss 属性** | 通过 `@property` 从 `MOE_LOSS_REGISTRY` 读取，避免非叶子张量 deepcopy 错误 |
| **deepcopy 安全** | 自定义 `__deepcopy__` 自动清理 `grad_fn` 张量 |
| **FLOPs 统计** | `get_gflops(input_shape)` 返回各组件 GFLOPs 占比 |
| **效率统计** | `get_efficiency_stats()` 返回参数总量、损失历史等 |

#### 使用示例

```python
import torch
from ultralytics.nn.modules.moe import UltraOptimizedMoE

moe = UltraOptimizedMoE(
    in_channels=256,
    out_channels=256,
    num_experts=4,
    top_k=2,
    expert_type='inverted',        # 使用 InvertedResidual 专家
    router_pool_scale=8,
    balance_loss_coeff=1.0,
    router_z_loss_coeff=0.01
)

x = torch.randn(2, 256, 32, 32)
moe.train()
out = moe(x)

# 获取辅助损失（训练时）
aux = moe.aux_loss
print(f"aux_loss: {aux.item():.4f}")

# 获取 FLOPs 统计
stats = moe.get_efficiency_stats((1, 256, 32, 32))
print(f"Total GFLOPs: {stats['gflops']['total_gflops']:.3f}")
print(f"Params: {stats['num_params']:.2f}M")
```

---

### 4.2 AdaptiveCapacityMoE

```python
class AdaptiveCapacityMoE(UltraOptimizedMoE):
    def __init__(self, *args, capacity_factor: float = 1.5, **kwargs)
```

**复杂度自适应 MoE**。在 `UltraOptimizedMoE` 基础上增加 `complexity_estimator`：

- 通过学习输入复杂度因子 `s ∈ (0, 1)` 动态调节专家输出贡献
- 缩放系数: `scale = exp((2s - 1) * log(capacity_factor))`，范围 `[1/cf, cf]`
- 高复杂度输入 → 更大的专家容量；低复杂度 → 更小的专家容量
- **关键修复 (rev: 2026-06-25)**: 固定 `top_k` 不变，通过可微分输出缩放实现自适应，避免 `int(score.item())` 导致的 GPU→CPU 同步和线程安全问题

---

### 4.3 OptimizedMOE

```python
class OptimizedMOE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_expand_ratio: int = 2,
        balance_loss_coeff: float = 1.0,
        z_loss_coeff: float = 1.0
    )
```

使用 `EfficientSpatialRouter` + `SimpleExpert` + `MoELoss` 的经典组合。支持完整的 dense / sparse 双路径：

- **训练**: dense 路径（所有专家计算，梯度完整）
- **推理**: sparse 路径（仅 Top-K 专家）
- **ONNX 导出**: 自动切换为 dense gather 路径

---

### 4.4 OptimizedMOEImproved

```python
class OptimizedMOEImproved(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int = 4,
        top_k: int = 2,
        expert_type: str = 'simple',
        router_type: str = 'efficient',
        progressive_sparsity: bool = True,
        detach_routing: bool = False,
        add_residual: bool = True
    )
```

可插拔路由器/专家类型的改进版 MoE：

| 参数 | 可选值 | 说明 |
|------|--------|------|
| `expert_type` | `'simple'` \| `'ghost'` \| `'inverted'` \| `'spatial'` | 专家架构 |
| `router_type` | `'efficient'` \| `'local'` \| `'adaptive'` | 路由器架构 |
| `progressive_sparsity` | `bool` | 训练前 `warmup_steps=5000` 渐进从 `top_k=num_experts` 降至目标值 |
| `detach_routing` | `bool` | `True` 隔离 router 与主任务梯度（ legacy 模式） |
| `add_residual` | `bool` | 是否添加残差连接；嵌入 `ABlockMoE` 时应设为 `False` 避免双重残差 |

---

### 4.5 ES_MOE

```python
class ES_MOE(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        num_experts: int = 3,
        reduction: int = 8,
        top_k: Optional[int] = None,
        use_sparse_inference: bool = True,
        dynamic_threshold: float = 0.4
    )
```

通用 MoE 块，使用 `DynamicRoutingLayer` 和不同卷积核尺寸的专家组（默认 `3×3`, `5×5`, `7×7`）。

- `dynamic_threshold`: 推理时动态剪枝低置信度专家（保留 Top-1 或权重 ≥ threshold 的专家）
- `set_top_k()`: 支持运行时动态调整 Top-K
- `enable_sparse_inference()`: 开关稀疏推理

---

## 5. 辅助损失系统

### 5.1 MoELoss

**文件**: `ultralytics/nn/modules/moe/loss.py`

```python
class MoELoss(nn.Module):
    def __init__(
        self,
        balance_loss_coeff: float = 1.0,
        z_loss_coeff: float = 1.0,
        entropy_loss_coeff: float = 0.0,
        diversity_loss_coeff: float = 0.0,
        variance_loss_coeff: float = 0.0,
        num_experts: int = 8,
        top_k: int = 2,
        use_soft_balancing: bool = False,
        coeff_floor: float = 0.0,
        dynamic_scheduler: Optional[MoEDynamicScheduler] = None
    )
```

#### 损失组件

| 损失项 | 公式 | 作用 |
|--------|------|------|
| **Balance Loss** | `N * Σ(importance * usage)` | 防止路由坍塌，均衡专家负载 |
| **Z-Loss** | `mean(logsumexp(logits)²)` | 约束 logits 幅度，提升数值稳定性 |
| **Entropy Loss** | `-Σ(p * log(p))` | 防止 router 犹豫不决（可选） |
| **Diversity Loss** | 专家输出余弦相似度惩罚 | 鼓励专家专业化（需 `expert_outputs`） |
| **Variance Loss** | `mean((usage - 1/N)²)` | 直接惩罚使用不均衡（可选） |

#### 硬/软平衡模式

- **Hard Balancing** (`use_soft_balancing=False`): `usage` 来自离散 Top-K 选择计数，非可微
- **Soft Balancing** (`use_soft_balancing=True`): `usage` 优先来自 `expert_indices` 的离散计数，回退到 `importance.detach()`，保持梯度流

#### DDP 兼容

所有跨进程聚合通过 `all_reduce_mean()` 实现：
- 在 `float32` 中执行 reduce，避免 `float16/bfloat16` 累加精度损失
- 单 GPU / CPU 环境下自动退化为 no-op

#### 使用示例

```python
from ultralytics.nn.modules.moe.loss import MoELoss

loss_fn = MoELoss(
    num_experts=4,
    top_k=2,
    balance_loss_coeff=1.0,
    z_loss_coeff=0.01,
    use_soft_balancing=True
)

B, E = 8, 4
router_logits = torch.randn(B, E, requires_grad=True)
router_probs = torch.softmax(router_logits, dim=1)
expert_indices = torch.topk(router_probs, k=2, dim=1).indices

total_loss = loss_fn(router_probs, router_logits, expert_indices)
total_loss.backward()

# router_logits.grad 非零，验证梯度回流
assert router_logits.grad.abs().sum() > 0
```

---

### 5.2 独立损失函数

```python
# GShard 标准平衡损失（uniform usage = 1.0）
gshard_balance_loss(expert_usage, num_experts, reduce_ddp=False)

# 带目标分布的加权平衡损失
weighted_gshard_balance_loss(expert_usage, target_usage, num_experts, reduce_ddp=False)

# 可微平衡损失（importance 保留梯度，usage detached）
differentiable_balance_loss(router_probs, expert_usage, num_experts, target_usage=None, reduce_ddp=False)
```

---

### 5.3 MOE_LOSS_REGISTRY

全局 `weakref.WeakKeyDictionary` 存储各 MoE 模块的辅助损失：

```python
from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY

# 线程安全读写（内部自动加锁）
# 训练循环中收集总辅助损失
from ultralytics.utils.loss import _collect_moe_aux_loss
total_aux = _collect_moe_aux_loss(model, device=torch.device("cuda"))
```

**设计原因**: 将非叶子张量（aux loss 含 `grad_fn`）存储在模块外部，避免 `copy.deepcopy()` 和 `torch.save()` 时触发 `RuntimeError: Only Tensors created explicitly by the user...`。

---

## 6. 工具与计算优化

### 6.1 BatchedExpertComputation

**文件**: `ultralytics/nn/modules/moe/utils.py`

```python
class BatchedExpertComputation:
    @staticmethod
    def compute_sparse_experts_batched(
        x: torch.Tensor,              # [B, C, H, W]
        experts: nn.ModuleList,       # 专家列表
        routing_weights: torch.Tensor,# [B, top_k, 1, 1]
        routing_indices: torch.Tensor,# [B, top_k, 1, 1]
        top_k: int,
        num_experts: int
    ) -> torch.Tensor               # [B, out_C, H, W]
```

**核心优化策略**:

1. **稀疏路径**: 通过 `torch.where` 收集选中每个专家的样本子集，批量前向传播
2. **index_add_ 聚合**: 避免逐样本赋值，使用 `index_add_` 累积输出
3. **ONNX 安全回退**: 导出时自动切换 dense path（计算所有专家 + `torch.gather`）
4. **训练/推理阈值差异**: 训练时 `weight_threshold=0.0`（保留所有选中路由）；推理时 `weight_threshold=0.01`（跳过极低权重专家）
5. **激活裁剪**: 输出 clamp 到 `[-1e4, 1e4]`，防止路由坍塌导致的数值爆炸

#### 使用示例

```python
from ultralytics.nn.modules.moe.utils import BatchedExpertComputation
import torch.nn as nn

experts = nn.ModuleList([nn.Conv2d(64, 64, 1) for _ in range(4)])
x = torch.randn(4, 64, 16, 16)
indices = torch.randint(0, 4, (4, 2, 1, 1))
weights = torch.rand(4, 2, 1, 1)

out = BatchedExpertComputation.compute_sparse_experts_batched(
    x, experts, weights, indices, top_k=2, num_experts=4
)
assert out.shape == (4, 64, 16, 16)
```

---

### 6.2 FlopsUtils

```python
class FlopsUtils:
    @staticmethod
    def count_conv2d(
        layer: Union[nn.Conv2d, nn.Sequential],
        input_shape: Tuple[int, int, int, int]  # (B, C, H, W)
    ) -> float
```

计算 `Conv2d` 或 `nn.Sequential` 的 FLOPs（考虑 dilation、stride、padding、groups）。

---

### 6.3 模型级工具函数

```python
from ultralytics.nn.modules.moe.utils import (
    is_core_moe_block,           # 判断模块是否为顶层 MoE block
    model_has_core_moe,          # 模型是否包含 MoE block
    iter_core_moe_expert_params  # 迭代 MoE 专家参数（排除 router/shared）
)
```

---

## 7. 动态调度器

### 7.1 MoEDynamicScheduler

**文件**: `ultralytics/nn/modules/moe/scheduler.py`

基于 **Gini 系数**的动态平衡损失系数调度器：

```python
@dataclass
class MoEDynamicSchedulerConfig:
    enabled: bool = True
    target_gini: float = 0.25       # 目标 Gini 系数
    gain: float = 1.5               # 调节增益
    min_balance_coeff: float = 0.02 # 系数下限
    max_balance_coeff: float = 2.0  # 系数上限
    ema_momentum: float = 0.9       # Gini EMA 动量
```

**调度公式**:

```
coeff_t = clamp(base_coeff * (1 + gain * (ema_gini_t - target_gini)), min, max)
```

- Gini 高（路由不均衡）→ 系数增大 → 更强惩罚
- Gini 低（路由健康）→ 系数减小 → 允许专家更专业化

```python
from ultralytics.nn.modules.moe.scheduler import MoEDynamicScheduler, MoEDynamicSchedulerConfig

scheduler = MoEDynamicScheduler(MoEDynamicSchedulerConfig(
    target_gini=0.2,
    gain=2.0
))

# 每个训练 step 调用
state = scheduler.step(expert_usage, base_balance_coeff=1.0)
print(f"Gini: {state.gini:.3f}, Adjusted coeff: {state.balance_loss_coeff:.3f}")
```

---

## 8. 诊断与监控

### 8.1 路由快照 (Routing Snapshot)

每个 MoE 模块在训练时自动记录 `last_routing_snapshot`：

```python
{
    "num_experts": 4,
    "top_k": 2,
    "expert_usage": tensor([0.25, 0.25, 0.25, 0.25]),  # 归一化使用频率
    "topk_counts": tensor([4., 4., 4., 4.]),            # 原始命中计数
    "mean_router_probs": tensor([0.25, 0.25, 0.25, 0.25]),
    "mean_topk_weight": tensor([0.6, 0.4]),
    "aux_loss": 1.0523
}
```

**采样频率**由环境变量 `MOE_SNAPSHOT_INTERVAL` 控制（默认每 10 个 forward 记录一次），避免诊断开销。

### 8.2 诊断工具

```python
from ultralytics.nn.modules.moe.diagnostics import collect_moe_diagnostics, format_moe_diagnostics
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker, diagnose_model

# 收集全模型诊断信息
diagnostics = collect_moe_diagnostics(model)
print(format_moe_diagnostics(diagnostics))

# 专家使用跟踪
tracker = ExpertUsageTracker()
tracker.update(model)
```

---

## 9. 使用示例

### 9.1 基础用法：替换标准卷积层

```python
import torch
from ultralytics.nn.modules.moe import UltraOptimizedMoE

# 假设原网络某层为 Conv2d(256, 256, 3)
# 替换为 MoE 模块
moe_layer = UltraOptimizedMoE(
    in_channels=256,
    out_channels=256,
    num_experts=4,
    top_k=2,
    expert_type='inverted',
    router_pool_scale=8,
    balance_loss_coeff=1.0,
    router_z_loss_coeff=0.01
)

x = torch.randn(2, 256, 32, 32)
moe_layer.train()

# 前向传播
out = moe_layer(x)

# 获取辅助损失（用于总损失加权）
aux_loss = moe_layer.aux_loss
print(f"Output shape: {out.shape}")       # [2, 256, 32, 32]
print(f"Aux loss: {aux_loss.item():.4f}")
```

### 9.2 在训练循环中整合

```python
from ultralytics.utils.loss import _collect_moe_aux_loss

model.train()
for batch in dataloader:
    images, targets = batch
    
    # 主任务前向
    preds = model(images)
    
    # 主损失（检测损失）
    main_loss = detection_loss(preds, targets)
    
    # 收集所有 MoE 模块的辅助损失
    moe_aux = _collect_moe_aux_loss(model, images.device)
    
    # 总损失 = 主损失 + MoE 辅助损失
    total_loss = main_loss + moe_aux
    
    total_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

### 9.3 动态调整 Top-K

```python
from ultralytics.nn.modules.moe import ES_MOE

moe = ES_MOE(in_channels=128, num_experts=6, top_k=2)

# 训练后切换到更大的 top_k 以提升精度
moe.set_top_k(3)

# 或完全关闭稀疏性（所有专家参与）
moe.set_top_k(None)

# 推理时开启/关闭稀疏加速
moe.enable_sparse_inference(True)   # 仅计算 Top-K 专家（快）
moe.enable_sparse_inference(False)  # 计算所有专家（精度略高）
```

### 9.4 FLOPs 与效率分析

```python
# 获取详细效率统计
stats = moe_layer.get_efficiency_stats((1, 256, 32, 32))

print(f"Router 占比: {stats['router_percentage']:.1f}%")
print(f"Experts 占比: {stats['experts_percentage']:.1f}%")
print(f"总参数量: {stats['num_params']:.2f}M")
print(f"总 GFLOPs: {stats['gflops']['total_gflops']:.3f}")
print(f"最近 balance_loss: {stats['last_balance_loss']:.4f}")
print(f"最近 z_loss: {stats['last_z_loss']:.4f}")
```

### 9.5 使用动态调度器

```python
from ultralytics.nn.modules.moe.loss import MoELoss
from ultralytics.nn.modules.moe.scheduler import MoEDynamicScheduler, MoEDynamicSchedulerConfig

scheduler = MoEDynamicScheduler(MoEDynamicSchedulerConfig(
    target_gini=0.2,
    gain=2.0,
    min_balance_coeff=0.01,
    max_balance_coeff=3.0
))

loss_fn = MoELoss(
    num_experts=4,
    top_k=2,
    dynamic_scheduler=scheduler
)

# 每个 step
for step, batch in enumerate(dataloader):
    # ... 前向计算得到 router_probs, router_logits, expert_indices
    loss = loss_fn(router_probs, router_logits, expert_indices, return_dict=True)
    
    # loss["dynamic_schedule"] 包含当前 Gini 和调整后的系数
    if loss["dynamic_schedule"]:
        print(f"Step {step}: Gini={loss['dynamic_schedule']['gini']:.3f}, "
              f"coeff={loss['dynamic_schedule']['balance_loss_coeff']:.3f}")
```

---

## 10. 性能参考

### 10.1 路由器 FLOPs 对比

| 路由器 | 下采样率 | 相对 FLOPs | 适用场景 |
|--------|----------|-----------|----------|
| `UltraEfficientRouter` | 8× | **~5%** 基线 | 生产首选 |
| `EfficientSpatialRouter` | 4× | ~20% 基线 | 平衡精度/速度 |
| `LocalRoutingLayer` | 2× | ~40% 基线 | 小目标检测 |
| `AdaptiveRoutingLayer` | Global Pool | ~1% 基线 | 通道丰富层 |

### 10.2 专家类型对比

| 专家类型 | 相对速度 | 参数量 | 空间感知 | 推荐场景 |
|----------|----------|--------|----------|----------|
| `SimpleExpert` | 1.0× | 低 | 否 | 通用默认 |
| `SpatialExpert` | 0.8× | 中 | **是** | 需要空间模式 |
| `GhostExpert` | 1.3× | **最低** | 否 | 移动端/边缘部署 |
| `InvertedResidualExpert` | **2.5×** | 低 | 是 | 高吞吐推理 |
| `SharedInvertedExpertGroup` | 3.0×+ | 中 | 是 | 共享特征提取 |

### 10.3 典型配置推荐

| 场景 | 推荐模块 | num_experts | top_k | expert_type | router_pool_scale |
|------|----------|-------------|-------|-------------|-------------------|
| 服务器训练 | `UltraOptimizedMoE` | 4-8 | 2 | `inverted` | 8 |
| 高精度推理 | `AdaptiveCapacityMoE` | 6 | 3 | `spatial` | 4 |
| 边缘部署 | `UltraOptimizedMoE` | 4 | 2 | `ghost` | 8 |
| 快速迭代实验 | `OptimizedMOEImproved` | 4 | 2 | `simple` | 4 |
| 大分辨率输入 | `ES_MOE` | 3-5 | 2 | — | — |

---

## 附录 A: 常见问题

**Q: 为什么使用 GroupNorm 而不是 BatchNorm？**
> Top-K 路由后，单个专家可能只接收到 batch 中的 1 个样本（甚至 0 个）。BatchNorm 在 `batch_size=1` 时方差无定义，而 GroupNorm 不受 batch size 影响。

**Q: `MOE_LOSS_REGISTRY` 会内存泄漏吗？**
> 不会。使用 `WeakKeyDictionary`，当 MoE 模块被垃圾回收时，对应的 registry 条目自动释放。每次 forward 会覆盖同模块的旧值，registry 大小等于模型中 MoE 模块数量。

**Q: 推理时 MoE 比 Dense 慢？**
> 确保 `use_sparse_inference=True` 且处于 `eval()` 模式。稀疏路径只计算 Top-K 专家，理论上可达 `num_experts / top_k` 倍加速。若使用 `ES_MOE`，确认 `torch.onnx.is_in_onnx_export()` 为 False（导出时会切换为 dense path）。

**Q: 如何检测路由坍塌？**
> 检查 `last_routing_snapshot["expert_usage"]`。若某一专家接近 1.0 而其余接近 0，说明发生坍塌。可调大 `balance_loss_coeff` 或启用 `MoEDynamicScheduler`。

---

## 附录 B: 版本变更日志

| 版本 | 日期 | 关键变更 |
|------|------|----------|
| v0.11 | 2026-02-07 | 初始 MoE 模块实现（Tencent Modification） |
| v0.12 | 2026-02-13 | 引入 `UltraOptimizedMoE`、`BatchedExpertComputation` |
| rev5 | 2026-06-24 | 统一 `gshard_balance_loss` 尺度为 O(1)；修复 aux loss 重复计数 |
| rev7 | 2026-06-24 | 修复 balance loss 梯度无法回流 router；registry 清理策略优化 |
| rev8 | 2026-06-25 | soft balancing 修复；`AdaptiveCapacityMoE` 复杂度估计改为可微分输出缩放 |

