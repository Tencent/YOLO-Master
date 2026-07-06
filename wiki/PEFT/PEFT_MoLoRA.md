# PEFT_MoLoRA 技术文档

## 目录

- [1. 概述](#1-概述)
- [2. 核心架构](#2-核心架构)
- [3. 配置模块 (`config.py`)](#3-配置模块-configpy)
- [4. 路由模块 (`router.py`)](#4-路由模块-routerpy)
- [5. 专家层模块 (`layer.py`)](#5-专家层模块-layerpy)
- [6. 损失函数模块 (`loss.py`)](#6-损失函数模块-losspy)
- [7. 模型包装模块 (`model.py`)](#7-模型包装模块-modelpy)
- [8. 工具函数模块 (`utils.py`)](#8-工具函数-modulesutilspy)
- [9. 使用示例](#9-使用示例)
- [10. 训练与调参建议](#10-训练与调参建议)
- [11. 附录：API 速查表](#11-附录api-速查表)

---

## 1. 概述

**MoLoRA (Mixture-of-LoRA)** 是 YOLO-Master 项目中针对目标检测任务设计的 PEFT (Parameter-Efficient Fine-Tuning) 变体。它将标准 LoRA 与 **Mixture-of-Experts (MoE)** 的路由机制相结合，通过稀疏激活多个 LoRA 专家来适配预训练模型，以极少的额外参数实现多域、多场景下的高效微调。

### 1.1 设计动机

标准 LoRA 对每个目标层注入单一的低秩适配矩阵，这在单域微调中表现优异，但面临以下局限：

- **单域适配瓶颈**：单一低秩矩阵难以同时建模多种域偏移（如白天/夜间/雾天）。
- **灾难性遗忘**：顺序训练新域时，旧域性能显著下降。
- **表达能力受限**：固定秩 r 的适配容量对复杂场景可能不足。

MoLoRA 通过引入多个专家级 LoRA 适配器和一个可学习的 Router，实现：

- **稀疏专家选择**：每样本仅激活 top-K 个专家，保持参数效率。
- **域隔离**：通过 `domain_experts` 将专家与域绑定，防止域间干扰。
- **渐进式 warmup**：`top_k_warmup` 逐步增加激活专家数，稳定训练早期。
- **权重合并/解合并**：推理前可将所有专家 Delta 合并回基线权重，实现零推理开销。

### 1.2 与标准 LoRA 的对比

| 维度 | 标准 LoRA | MoLoRA |
|------|----------|--------|
| 参数量 | ~0.1%–1% | ~0.3%–3%（随专家数线性增长） |
| 专家数量 | 1（每目标层） | N（可配置，默认 4） |
| 激活策略 | 全激活 | Top-K 稀疏激活（默认 K=2） |
| 域适配 | 需全模型重训练 | 支持域隔离与专家回放 |
| 辅助损失 | 无 | balance_loss + z_loss (+ diversity_loss) |
| 推理开销 | 低秩前向 | 合并后等价于基线推理 |
| 路由粒度 | — | 图像级（`linear`）、空间级（`spatial`）、混合（`hybrid`） |

---

## 2. 核心架构

MoLoRA 的完整数据流如下：

```
输入特征 x ─┬─> Base Layer (冻结) ──> base_out
            │
            ├─> Router ──> router_logits ──> softmax ──> router_probs
            │                                    │
            │                                    ▼
            │                           top_k(router_probs, K)
            │                                    │
            │                         top_k_weights, top_k_indices
            │                                    │
            │                                    ▼
            │                     ┌───────────────────────────────┐
            │                     │  Expert 0 ──> lora_B(lora_A(x)) │
            │                     │  Expert 1 ──> lora_B(lora_A(x)) │
            │                     │  ...                            │
            │                     │  Expert E-1 ──> ...             │
            │                     └───────────────────────────────┘
            │                                    │
            │                         Σ g_k · expert_k(x)  (加权聚合)
            │                                    │
            │                                    ▼
            └────────────────────>  base_out + adapted_delta
```

### 2.1 模块依赖关系

```
get_peft_molora_model()           # 入口函数
    ├── MoLoRAConfig              # 配置 dataclass
    ├── MoLoRAConfigBuilder       # 自动目标层检测（继承 LoRAConfigBuilder）
    ├── MoLoRALayer               # 核心适配层
    │   ├── base_layer (frozen)   # 原始 Conv2d / Linear
    │   ├── experts[]             # MoLoRAExpert 列表
    │   ├── router                # LinearRouter / SpatialRouter / HybridRouter
    │   └── loss_fn               # MoLoRALoss
    └── mark_only_molora_as_trainable()

MoLoRAModel                       # 便捷包装类
    ├── compute_aux_loss()        # 收集辅助损失
    ├── merge() / unmerge()       # 权重合并
    ├── save/load_checkpoint()    # 轻量检查点
    └── save/load_expert_replay_buffer()  # 持续学习
```

---

## 3. 配置模块 (`config.py`)

### 3.1 `MoLoRAConfig`

`MoLoRAConfig` 继承自 `LoRAConfig`（标准 LoRA 配置），新增 MoE 相关的所有参数。

```python
@dataclass
class MoLoRAConfig(LoRAConfig):
    """Mixture-of-LoRA 配置类。"""
```

#### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `r` | `int` | — (继承) | LoRA 秩，控制低秩适配矩阵的秩。 |
| `alpha` | `int` | — (继承) | 缩放因子，配合 `use_rslora` 决定实际缩放。 |
| `dropout` | `float` | — (继承) | 注入到低秩路径的 dropout 率。 |
| `target_modules` | `List[str]` | `None` | 目标模块名称列表；`None` 时自动检测。 |
| `num_experts` | `int` | `4` | 每个目标层的专家总数 E。 |
| `top_k` | `int` | `2` | 每样本激活的专家数 K（1 ≤ K ≤ E）。 |
| `router_type` | `str` | `"linear"` | 路由器类型：`"linear"`、`"spatial"`、`"hybrid"`。 |
| `router_hidden_dim` | `Optional[int]` | `None` | Router 隐藏层维度；`None` 时自动设为 `C // 4`。 |
| `balance_loss_coef` | `float` | `0.01` | load-balancing 损失系数 λ_balance。 |
| `z_loss_coef` | `float` | `0.001` | Router z-loss 系数 λ_z。 |
| `diversity_loss_coef` | `float` | `0.0` | 专家输出多样性损失系数 λ_diversity。 |
| `capacity_factor` | `float` | `1.0` | 动态容量限制因子；`1.0` 表示无限制。 |
| `expert_dropout` | `float` | `0.0` | 训练时随机禁用专家的概率。 |
| `top_k_warmup` | `Optional[int]` | `None` | 是否启用 top-K warmup。 |
| `warmup_steps` | `int` | `0` | warmup 总步数。 |
| `domain_experts` | `Optional[Dict[str, List[int]]]` | `None` | 域到专家索引的映射表。 |
| `freeze_experts` | `Optional[List[int]]` | `None` | 待冻结的专家索引（保留字段）。 |
| `share_moe_registry` | `bool` | `True` | 是否将 balance loss 写入共享的 `MOE_LOSS_REGISTRY`。 |
| `expert_init` | `str` | `"default"` | 专家初始化策略：`"default"`、`"orthogonal"`、`"gaussian"`。 |
| `use_rslora` | `bool` | `True` | 是否使用 rsLoRA 缩放：`alpha / sqrt(r)`。 |

#### 校验规则

在 `__post_init__` 中执行以下校验：

- `num_experts >= 1`
- `1 <= top_k <= num_experts`
- `router_type in {"linear", "spatial", "hybrid"}`
- 所有损失系数 `>= 0`
- `expert_init in {"default", "orthogonal", "gaussian"}`

#### 类方法

```python
# 从现有 LoRAConfig 升级为 MoLoRAConfig，保留所有原有设置
@classmethod
def from_lora_config(cls, lora_config: LoRAConfig, **molora_overrides) -> "MoLoRAConfig"

# 从 Ultralytics args 或 kwargs 构造配置
# 自动映射 molora_ 前缀的参数，如 molora_r, molora_num_experts 等
@classmethod
def from_args(cls, args=None, **kwargs) -> "MoLoRAConfig"
```

**参数映射表（`from_args` 内部使用）：**

| 配置字段 | 映射的 kwargs/args 键 |
|---------|---------------------|
| `r` | `molora_r` |
| `alpha` | `molora_alpha` |
| `num_experts` | `molora_num_experts` |
| `top_k` | `molora_top_k` |
| `router_type` | `molora_router_type` |
| `balance_loss_coef` | `molora_balance_loss` |
| `z_loss_coef` | `molora_router_z_loss` |
| `use_rslora` | `molora_use_rslora` |
| `expert_init` | `molora_expert_init` |
| ... | ... |

### 3.2 `MoLoRAConfigBuilder`

继承自 `LoRAConfigBuilder`，复用父类的 `auto_detect_targets` 逻辑，**不修改目标层的选择范围**，仅改变适配方式。

```python
class MoLoRAConfigBuilder(LoRAConfigBuilder):
    """MoLoRA 配置构建器，复用 LoRA 的目标层检测逻辑。"""
```

#### `create_molora_config`

```python
@staticmethod
def create_molora_config(
    model: nn.Module,
    r: int = 8,
    alpha: Optional[int] = None,
    num_experts: int = 4,
    top_k: int = 2,
    router_type: str = "linear",
    balance_loss_coef: float = 0.01,
    z_loss_coef: float = 0.001,
    use_rslora: bool = True,
    expert_init: str = "default",
    **kwargs,
) -> Optional[Dict[str, Any]]
```

**参数：**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `nn.Module` | — | 待适配的基线模型。 |
| `r` | `int` | `8` | LoRA 秩。 |
| `alpha` | `Optional[int]` | `None` | 缩放因子；`None` 时自动设为 `2 * r`。 |
| `num_experts` | `int` | `4` | 专家数。 |
| `top_k` | `int` | `2` | 激活专家数。 |
| `router_type` | `str` | `"linear"` | 路由器类型。 |
| `balance_loss_coef` | `float` | `0.01` | Balance loss 权重。 |
| `z_loss_coef` | `float` | `0.001` | Z-loss 权重。 |
| `use_rslora` | `bool` | `True` | rsLoRA 缩放。 |
| `expert_init` | `str` | `"default"` | 初始化策略。 |
| `**kwargs` | — | — | 额外过滤参数（见 `auto_detect_targets` 文档）。 |

**返回：**

- 成功时返回 `Dict[str, Any]`，可直接传递给 `get_peft_molora_model()`。
- 无可适配层时返回 `None`。

**用法示例：**

```python
from ultralytics.nn.peft.molora import MoLoRAConfigBuilder

cfg_dict = MoLoRAConfigBuilder.create_molora_config(
    model,
    r=8, alpha=16,
    num_experts=4, top_k=2,
    router_type="hybrid",
    include_moe=True,
    only_backbone=False,
    skip_stem=True,      # 跳过 backbone stem
    min_channels=32,     # 跳过窄层
)
```

### 3.3 `get_molora_preset`

预设工厂函数，返回命名配置字典。

```python
def get_molora_preset(name: str) -> Dict[str, Any]
```

**可用预设：**

| 预设名 | 专家数 | top_k | r | alpha | router_type | 适用场景 |
|--------|--------|-------|---|-------|-------------|---------|
| `preset_small` | 2 | 1 | 4 | 8 | `linear` | 移动端/最小化部署 |
| `preset_standard` | 4 | 2 | 8 | 16 | `linear` | 默认推荐 |
| `preset_large` | 8 | 2 | 16 | 32 | `hybrid` | 高容量需求 |
| `preset_continual` | 8 | 2 | 8 | 16 | `linear` | 持续学习 |

**用法示例：**

```python
from ultralytics.nn.peft.molora import get_molora_preset

cfg = MoLoRAConfig(**get_molora_preset("preset_standard"))
```

---

## 4. 路由模块 (`router.py`)

MoLoRA 的 Router 采用 **CNN-Native** 设计：与 NLP 中逐 token 路由不同，YOLO-Master 的 Router 以整图为决策单位，将特征图池化后做全局路由。支持三种变体。

### 4.1 `LinearRouter`

**图像级全局路由**：对每个输入样本计算一次路由决策，计算复杂度 O(C)。

```python
class LinearRouter(nn.Module):
    """Global Average Pool -> Linear Router. 每个图像一个决策。"""

    def __init__(
        self,
        in_channels: int,        # 输入特征通道数 C
        num_experts: int,        # 专家数 E
        hidden_dim: Optional[int] = None,  # 隐藏层维度，默认 C // 4
    )
```

**前向流程：**

```
x: [B, C, H, W]  ──> mean([2,3]) ──> [B, C]
                     │
                     ▼
               fc: [C -> hidden -> E]
                     │
                     ▼
              logits: [B, E]
```

- 输入为 2D 张量 `[B, C]` 时直接通过 MLP。
- 输出 logits 初始化为接近零的小值（`std=0.01`），保证训练初期路由接近均匀分布。

### 4.2 `SpatialRouter`

**空间感知路由**：通过 1×1 Conv 保留空间信息，再空间平均池化，复杂度 O(C·H·W)。

```python
class SpatialRouter(nn.Module):
    """1x1 Conv -> Spatial AvgPool Router. 细粒度空间感知。"""

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
    )
```

**前向流程：**

```
x: [B, C, H, W]  ──> conv1x1(C->hidden) ──> ReLU ──> conv1x1(hidden->E)
                                                        │
                                                        ▼
                                              [B, E, H, W]
                                                        │
                                               mean([2,3])
                                                        │
                                                        ▼
                                              logits: [B, E]
```

- 当输入为 2D 时，先 `unsqueeze(-1).unsqueeze(-1)` 转为 4D。
- 适合需要空间感知的场景（如前景/背景使用不同专家）。

### 4.3 `HybridRouter`

**混合路由**：融合 LinearRouter 的全局视图与 SpatialRouter 的局部细节，通过可学习权重 `α ∈ [0,1]` 动态平衡两者。

```python
class HybridRouter(nn.Module):
    """全局 + 局部融合，可学习门控 α。"""

    def __init__(
        self,
        in_channels: int,
        num_experts: int,
        hidden_dim: Optional[int] = None,
    )
```

**融合公式：**

```
α = sigmoid(self.alpha)          # 可学习参数，初始 0.5
logits = α · logits_linear + (1 - α) · logits_spatial
```

### 4.4 `build_router` 工厂函数

```python
def build_router(
    router_type: str,      # "linear" | "spatial" | "hybrid"
    in_channels: int,      # 输入特征通道数
    num_experts: int,      # 专家数
    hidden_dim: Optional[int] = None,  # 隐藏层维度
) -> nn.Module
```

**用法示例：**

```python
from ultralytics.nn.peft.molora import build_router

router = build_router("hybrid", in_channels=256, num_experts=4, hidden_dim=64)
```

---

## 5. 专家层模块 (`layer.py`)

### 5.1 `MoLoRAExpert`

单个 LoRA 专家，由一个低秩 A（下投影）和一个低秩 B（上投影）组成。

```python
class MoLoRAExpert(nn.Module):
    """单个 LoRA 专家：低秩 A + B 对。"""

    def __init__(
        self,
        base_layer: nn.Module,      # 原始 Conv2d 或 Linear 层（仅用于形状推导）
        r: int,                     # 低秩 r
        alpha: int,                 # 缩放因子
        dropout: float = 0.0,       # dropout 率
        use_rslora: bool = True,    # rsLoRA 缩放
        init_type: str = "default", # 初始化策略
    )
```

#### 层结构

**对于 Conv2d 基线层：**

| 层 | 类型 | 输入 | 输出 | 核大小 | 说明 |
|---|------|------|------|--------|------|
| `lora_A` | `nn.Conv2d` | `C_in` | `r` | `1×1` | 下投影，stride=1, padding=0 |
| `lora_B` | `nn.Conv2d` | `r` | `C_out` | `K×K` | 上投影，与基线层相同的 stride/padding/dilation/groups |

**对于 Linear 基线层：**

| 层 | 类型 | 输入 | 输出 |
|---|------|------|------|
| `lora_A` | `nn.Linear` | `in_features` | `r` |
| `lora_B` | `nn.Linear` | `r` | `out_features` |

#### 前向传播

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """计算 delta = B @ A(x)，返回适配增量。"""
```

公式：

```
ΔW = B · A(x) · scaling
scaling = alpha / sqrt(r)   (rsLoRA)
```

#### `delta_weight()`

```python
def delta_weight(self) -> torch.Tensor
```

返回该专家的等效全秩增量权重，用于：
- 权重检查与可视化
- `merge_weights()` 内部调用

对于 Conv2d：
```python
delta = torch.einsum("orkw,ri->oikw", b, a) * scaling
```

对于 Linear：
```python
delta = (B.weight @ A.weight) * scaling
```

### 5.2 `MoLoRALayer`

MoLoRA 的核心适配层，将基线 `Conv2d` / `Linear` 包装为稀疏专家混合层。

```python
class MoLoRALayer(nn.Module):
    """将 Conv2d/Linear 替换为 top-K 稀疏专家混合的包装层。"""

    def __init__(
        self,
        base_layer: nn.Module,                  # 原始层（冻结）
        r: int = 8,
        alpha: int = 16,
        num_experts: int = 4,
        top_k: int = 2,
        router_type: str = "linear",
        dropout: float = 0.0,
        use_rslora: bool = True,
        balance_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        diversity_loss_coef: float = 0.0,
        expert_init: str = "default",
        share_moe_registry: bool = True,
        router_hidden_dim: Optional[int] = None,
        capacity_factor: float = 1.0,
        expert_dropout: float = 0.0,
        top_k_warmup: Optional[int] = None,
        warmup_steps: int = 0,
        domain_experts: Optional[Dict[str, List[int]]] = None,
    )
```

#### 前向传播流程

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: [B, C, H, W] for Conv2d, [B, C] or [B, C, *] for Linear
    Returns:
        y: 与 x 同形状，经 top-k 专家适配后的输出
    """
```

**步骤详解：**

1. **基线前向**：`base_out = self.base_layer(x)`（始终冻结，不计算梯度）。
2. **步数递增**：训练时 `self._step_count += 1`。
3. **路由计算**：`router_logits = self.router(x)` → `[B, E]`。
4. **域限制**：若设置了 `domain_experts`，对非活跃专家施加 `-1e9` mask。
5. **Softmax 归一化**：`router_probs = softmax(router_logits)` → `[B, E]`。
6. **专家 Dropout**：训练时以 `expert_dropout` 概率随机禁用专家并重新归一化。
7. **动态 top-K**：根据 warmup 步数逐步从 1 增加到 `top_k`。
8. **Top-K 选择**：`torch.topk(router_probs, effective_k)` → `top_k_weights`, `top_k_indices`。
9. **容量限制**：若 `0 < capacity_factor < 1.0`，对超载专家的权重做软截断。
10. **稀疏专家计算**：`_compute_sparse_experts()` 按专家分组聚合。
11. **辅助损失**：`loss_fn(router_probs, router_logits, top_k_indices)` → `aux_loss`。
12. **注册损失**：若 `share_moe_registry=True`，写入 `MOE_LOSS_REGISTRY`。
13. **返回**：`base_out + adapted_delta`。

#### `_compute_sparse_experts`

```python
def _compute_sparse_experts(
    self,
    x: torch.Tensor,
    top_k_weights: torch.Tensor,   # [B, K]
    top_k_indices: torch.Tensor,   # [B, K]
    out_template: torch.Tensor,    # 用于输出形状模板
) -> torch.Tensor
```

**实现优化：**

- 不按样本循环，而是**按专家分组**（`for e in range(num_experts)`）。
- 同一专家处理的所有样本组成一个子批次，一次性通过该专家网络。
- 若专家被冻结（`_expert_frozen_mask[e] == True`），则在 `torch.no_grad()` 下执行。
- 输出按 `top_k_weights` 加权后累加。

#### 域相关方法

```python
def set_domain(self, domain: str) -> None
```
- 将路由限制在 `domain_experts[domain]` 指定的专家子集上。
- 非活跃专家在 logits 上被施加 `-1e9` mask，使其 softmax 概率趋近于 0。

```python
def clear_domain(self) -> None
```
- 清除域限制，恢复所有专家可用。

#### 专家冻结/解冻

```python
def freeze_experts(self, expert_indices: List[int]) -> None
def unfreeze_experts(self, expert_indices: Optional[List[int]] = None) -> None
```

- `freeze_experts`：将指定专家的 `requires_grad` 设为 `False`。
- `unfreeze_experts`：恢复指定或全部专家的梯度计算。
- 适用于持续学习中保护旧域专家的权重。

#### 权重合并/解合并

```python
def merge_weights(self) -> None
def unmerge_weights(self) -> None
```

**`merge_weights()` 机制：**

- 对每个专家计算 `delta_weight()`，并按 `scaling / num_experts` 的权重合并到基线层的 `weight.data` 中。
- 合并后设置 `self.merged = True`，后续前向将跳过 LoRA 路径。
- **适用场景**：ONNX 导出、TensorRT 推理、部署前优化。

**合并公式（Conv2d）：**

```
W_base ← W_base + Σ_e (delta_weight(expert_e) / num_experts)
```

---

## 6. 损失函数模块 (`loss.py`)

### 6.1 `MoLoRALoss`

MoLoRA 的辅助损失模块，包含三个组件：

```python
class MoLoRALoss(nn.Module):
    """MoLoRA 路由辅助损失。

    Components:
      - balance_loss: GShard 风格负载均衡
      - z_loss: 抑制过大 router logits
      - diversity_loss: 惩罚专家输出相似度
    """

    def __init__(
        self,
        num_experts: int,
        top_k: int,
        balance_loss_coef: float = 0.01,
        z_loss_coef: float = 0.001,
        diversity_loss_coef: float = 0.0,
        reduce_ddp: bool = False,
    )
```

#### `_balance_loss` — GShard 负载均衡

```python
def _balance_loss(
    self,
    router_probs: torch.Tensor,   # [B, E]
    expert_indices: torch.Tensor, # [B, K]
) -> torch.Tensor
```

**公式：**

```
importance_e = mean(router_probs[:, e])      # 专家 e 的平均重要性
importance_e = importance_e / sum(importance)  # 归一化

usage_e = count(expert_indices == e) / total   # 专家 e 的离散使用率（detach，无梯度）

balance_loss = num_experts * Σ_e (importance_e * usage_e)
```

- `importance` 保留梯度，引导 router 学习均衡分配。
- `usage`  detach，作为非可微的监督信号。
- 若 `reduce_ddp=True`，通过 `all_reduce` 聚合多卡统计。

#### `_z_loss` — 稳定性损失

```python
def _z_loss(self, router_logits: torch.Tensor) -> torch.Tensor
```

**公式：**

```
z_loss = mean( logsumexp(logits, dim=1) ^ 2 )
```

- 惩罚过大的 logits，防止 softmax 输出过于尖锐（极端路由）。
- 参考自 Switch Transformer / ST-MoE。

#### `_diversity_loss` — 专家多样性

```python
def _diversity_loss(self, expert_outputs: torch.Tensor) -> torch.Tensor
```

**公式：**

```python
expert_outputs: [B, E, D]  # 每个专家的输出向量
normed = normalize(expert_outputs, dim=-1)     # [B, E, D]
sim = normed @ normed.T                        # [B, E, E] 余弦相似度
mask = 1 - eye(E)                              # 排除自相似
diversity_loss = sum((sim * mask)^2) / (B * E * (E-1))
```

- 默认关闭（`diversity_loss_coef=0.0`），因计算所有专家输出开销较大。
- 开启后需在前向中传入 `expert_outputs`。

#### `forward`

```python
def forward(
    self,
    router_probs: torch.Tensor,
    router_logits: torch.Tensor,
    expert_indices: torch.Tensor,
    expert_outputs: Optional[torch.Tensor] = None,
    return_dict: bool = False,
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]
```

**返回：**

- `return_dict=False`：返回标量 `total_loss`。
- `return_dict=True`：返回字典，包含 `loss`、`balance_loss`、`z_loss`、`diversity_loss` 的 detached 值。

**总损失：**

```
total = λ_balance * balance_loss + λ_z * z_loss + λ_diversity * diversity_loss
```

### 6.2 `compute_expert_usage`

```python
def compute_expert_usage(expert_indices: torch.Tensor, num_experts: int) -> torch.Tensor
```

返回归一化的专家使用直方图 `[E]`，用于诊断和可视化。

---

## 7. 模型包装模块 (`model.py`)

### 7.1 `get_peft_molora_model`

**核心入口函数**，将 Ultralytics 模型（或任意 `nn.Module`）包装为 MoLoRA 版本。

```python
def get_peft_molora_model(
    model: nn.Module,
    config: Union[MoLoRAConfig, Dict[str, Any]],
) -> nn.Module
```

**参数：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `model` | `nn.Module` | 待适配的基线模型。**会被原地修改**。 |
| `config` | `MoLoRAConfig` 或 `dict` | 配置对象或字典。 |

**执行流程：**

1. **防重复包装**：检查 `model.molora_enabled`，若为 `True` 则跳过。
2. **配置解析**：将字典转为 `MoLoRAConfig`。
3. **目标层解析**：
   - 若 `config.target_modules` 为空，调用 `MoLoRAConfigBuilder.auto_detect_targets()` 自动检测。
   - 支持 `include_moe`、`only_backbone`、`skip_stem`、`min_channels` 等过滤参数。
4. **逐层包装**：遍历目标模块，通过 `_parent_child_name` 和 `_get_submodule` 定位父模块，原地替换为 `MoLoRALayer`。
5. **元数据附加**：设置 `model.molora_config` 和 `model.molora_enabled`。
6. **冻结非 MoLoRA 参数**：调用 `mark_only_molora_as_trainable(model)`，仅 `lora_A`、`lora_B`、`router`、`molora` 前缀的参数可训练。

**返回：** 原地修改后的模型（与输入为同一对象）。

**用法示例：**

```python
from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAConfig, get_peft_molora_model

yolo = YOLO("yolov8n.pt")
cfg = MoLoRAConfig(r=8, alpha=16, num_experts=4, top_k=2, router_type="linear")
model = get_peft_molora_model(yolo.model, cfg)
```

### 7.2 `MoLoRAModel`

便捷包装类，提供高层 API。

```python
class MoLoRAModel(nn.Module):
    def __init__(self, model: nn.Module, config: Union[MoLoRAConfig, Dict[str, Any]])
```

#### 辅助损失收集

```python
def compute_aux_loss(self) -> torch.Tensor
```

遍历模型所有 `MoLoRALayer`，从 `MOE_LOSS_REGISTRY` 中收集辅助损失并求和。应在每个训练步的前向后调用，并加到总损失中。

```python
# 训练循环示例
for batch in dataloader:
    out = model(batch)
    loss = criterion(out, target)
    aux = wrapper.compute_aux_loss()
    total_loss = loss + aux
    total_loss.backward()
```

#### 权重合并/解合并

```python
def merge(self) -> None      # 所有 MoLoRALayer 合并

def unmerge(self) -> None    # 所有 MoLoRALayer 解合并
```

#### 域管理

```python
def set_domain(self, domain: str) -> None
def clear_domain(self) -> None
def freeze_experts(self, expert_indices: List[int]) -> None
def unfreeze_experts(self, expert_indices: Optional[List[int]] = None) -> None
```

以上方法批量作用于模型中所有 `MoLoRALayer`。

#### 专家回放（持续学习）

```python
def save_expert_replay_buffer(
    self, domain: str, path: Optional[str] = None
) -> Dict[str, Any]
```

将当前所有 `MoLoRALayer` 的专家权重保存为回放缓冲区，用于防止灾难性遗忘。

**返回结构：**

```python
{
    "domain": str,           # 域名称
    "experts": {
        "module_name": {
            0: {"lora_A": state_dict, "lora_B": state_dict},
            1: {"lora_A": state_dict, "lora_B": state_dict},
            ...
        },
        ...
    }
}
```

```python
def load_expert_replay_buffer(
    self,
    buffer: Union[str, Dict[str, Any]],
    domain: Optional[str] = None,
) -> None
```

从文件路径或字典加载专家权重。若指定 `domain`，会验证缓冲区中的域名是否匹配。

#### 检查点管理

```python
def save_checkpoint(self, path: str) -> None
def load_checkpoint(self, path: str) -> None
```

仅保存/加载 MoLoRA 相关参数（`lora_A`、`lora_B`、`router`、`molora`），体积小、速度快。

#### 参数统计

```python
def param_stats(self) -> Dict[str, Any]
```

返回参数字典：`total`、`trainable`、`frozen`、`molora`、`trainable_pct`、`molora_pct`。

---

## 8. 工具函数模块 (`utils.py`)

### 8.1 `_molora_scales` — 缩放因子

```python
def _molora_scales(r: int, alpha: int, use_rslora: bool = True) -> float
```

**公式：**

```
rsLoRA:  scaling = alpha / sqrt(max(r, 1))
Standard: scaling = alpha / max(r, 1)
```

### 8.2 `init_lora_expert_a` / `init_lora_expert_b`

```python
def init_lora_expert_a(weight: nn.Parameter, init_type: str = "default") -> None
def init_lora_expert_b(weight: nn.Parameter, init_type: str = "default") -> None
```

**`init_lora_expert_a` 策略：**

| `init_type` | A 初始化方式 |
|-------------|-------------|
| `"default"` | Kaiming uniform（Hu et al. 2021） |
| `"orthogonal"` | QR 分解正交初始化 |
| `"gaussian"` | `N(0, 0.02)` |

**`init_lora_expert_b` 策略：**

| `init_type` | B 初始化方式 |
|-------------|-------------|
| `"default"` | 零初始化（训练从基线权重开始） |
| `"gaussian"` | `N(0, 0.02)` |
| `"orthogonal"` | 零初始化（训练从基线权重开始） |

### 8.3 `allocate_domain_experts`

```python
def allocate_domain_experts(
    num_experts: int,
    domains: List[str],
) -> Dict[str, List[int]]
```

将专家均匀分配给各个域。采用**贪心分配**：每个域至少分配 `num_experts // len(domains)` 个专家，余数按顺序分配给前几个域。

**示例：**

```python
allocate_domain_experts(8, ["day", "night", "fog"])
# 返回: {"day": [0,1,2], "night": [3,4,5], "fog": [6,7]}
```

### 8.4 `mark_only_molora_as_trainable`

```python
def mark_only_molora_as_trainable(model: nn.Module) -> None
```

遍历模型所有参数，仅保留以下名称模式的可训练性：
- `lora_A`
- `lora_B`
- `router`
- `molora`

其他所有参数设为 `requires_grad=False`。

### 8.5 `count_parameters`

```python
def count_parameters(model: nn.Module) -> Dict[str, Any]
```

返回参数统计字典：

```python
{
    "total": int,           # 总参数量
    "trainable": int,       # 可训练参数量
    "frozen": int,          # 冻结参数量
    "molora": int,          # MoLoRA 专属参数量
    "trainable_pct": float, # 可训练占比 %
    "molora_pct": float,    # MoLoRA 占比 %
}
```

### 8.6 合并/解合并辅助函数

```python
def _merge_conv_delta(base_weight, lora_a, lora_b, scale) -> None
def _merge_linear_delta(base_weight, lora_a, lora_b, scale) -> None
def _unmerge_conv_delta(base_weight, lora_a, lora_b, scale) -> None
def _unmerge_linear_delta(base_weight, lora_a, lora_b, scale) -> None
```

通过 `einsum`（Conv2d）或矩阵乘法（Linear）计算增量并应用到基线权重。仅在 `torch.no_grad()` 上下文中使用。

---

## 9. 使用示例

### 9.1 基础微调（COCO 单域）

```python
from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAConfig, get_peft_molora_model

# 1. 加载预训练模型
yolo = YOLO("yolov8n.pt")

# 2. 创建配置
cfg = MoLoRAConfig(
    r=8,
    alpha=16,
    num_experts=4,
    top_k=2,
    router_type="linear",
    balance_loss_coef=0.01,
    z_loss_coef=0.001,
    use_rslora=True,
    expert_init="default",
)

# 3. 包装模型
model = get_peft_molora_model(yolo.model, cfg)
yolo.model = model

# 4. 训练（仅 MoLoRA 参数可训练）
results = yolo.train(
    data="coco128.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    lr0=0.01,
    lrf=0.01,
    freeze=0,
    augment=True,
)
```

### 9.2 多域持续学习（白天 → 黑夜 → 雾天）

```python
from ultralytics import YOLO
from ultralytics.nn.peft.molora import (
    MoLoRAConfig, get_peft_molora_model, MoLoRAModel, allocate_domain_experts
)

# 1. 加载模型
yolo = YOLO("yolov8n.pt")

# 2. 分配专家
domains = ["day", "night", "fog"]
domain_experts = allocate_domain_experts(num_experts=8, domains=domains)
# domain_experts = {"day": [0,1,2], "night": [3,4,5], "fog": [6,7]}

# 3. 创建配置
cfg = MoLoRAConfig(
    r=8, alpha=16,
    num_experts=8, top_k=2,
    router_type="linear",
    domain_experts=domain_experts,
    balance_loss_coef=0.01,
    z_loss_coef=0.001,
)

# 4. 包装
model = get_peft_molora_model(yolo.model, cfg)
wrapper = MoLoRAModel(model, cfg)

# ---------- 阶段 1：Day 域 ----------
wrapper.set_domain("day")
# ... train on day data ...
day_buffer = wrapper.save_expert_replay_buffer("day")

# ---------- 阶段 2：Night 域（冻结 day 专家） ----------
wrapper.freeze_experts(domain_experts["day"])
wrapper.set_domain("night")
# ... train on night data ...
night_buffer = wrapper.save_expert_replay_buffer("night")

# 评估 day 域时回放 day 专家
wrapper.load_expert_replay_buffer(day_buffer, domain="day")
wrapper.set_domain("day")
# ... evaluate ...

# ---------- 阶段 3：Fog 域 ----------
wrapper.unfreeze_experts(domain_experts["day"])
wrapper.freeze_experts(domain_experts["day"] + domain_experts["night"])
wrapper.set_domain("fog")
# ... train on fog data ...
```

### 9.3 推理前权重合并（零开销部署）

```python
# 训练完成后合并权重
wrapper.merge()

# 此时 forward 等价于原始模型 + 平均增量，无 LoRA 路径开销
# 适合 ONNX 导出 / TensorRT 推理

# 如需恢复（如继续训练），可解合并
wrapper.unmerge()
```

### 9.4 使用预设快速配置

```python
from ultralytics.nn.peft.molora import MoLoRAConfig, get_molora_preset, get_peft_molora_model

# 使用高容量预设
cfg = MoLoRAConfig(**get_molora_preset("preset_large"))
model = get_peft_molora_model(yolo.model, cfg)
```

### 9.5 手动注入并对比 LoRA / MoLoRA

```python
from ultralytics.nn.peft.molora import (
    MoLoRAConfig, get_peft_molora_model, mark_only_molora_as_trainable
)
from ultralytics.nn.peft.molora.layer import MoLoRAExpert

# 标准 LoRA 注入（手动方式）
def inject_lora(model, r, alpha):
    params = []
    for name, mod in list(model.named_modules()):
        if isinstance(mod, (nn.Conv2d, nn.Linear)) and "conv" in name:
            expert = MoLoRAExpert(mod, r=r, alpha=alpha, use_rslora=True)
            for p in mod.parameters():
                p.requires_grad = False
            for p in expert.parameters():
                p.requires_grad = True
                params.append(p)
            orig = mod.forward
            mod.forward = lambda x, o=orig, e=expert: o(x) + e(x)
    return params

# MoLoRA 注入（自动方式）
cfg = MoLoRAConfig(r=4, alpha=8, num_experts=4, top_k=2,
                   target_modules=["conv1", "conv2", "fc"])
model = get_peft_molora_model(model, cfg)
mark_only_molora_as_trainable(model)
```

---

## 10. 训练与调参建议

### 10.1 学习率

MoLoRA 的参数量通常为标准 LoRA 的 E 倍（E = 专家数），但由于稀疏激活，实际梯度更新量相近。建议：

- **lr0**：可从标准 LoRA 的 `0.005` 提升至 `0.01`，因为仅适配器可训练。
- **lrf**：保持与标准 LoRA 一致（如 `0.01`）。

### 10.2 专家数与 top-K 选择

| 场景 | num_experts | top_k | 建议 |
|------|------------|-------|------|
| 单域微调，资源受限 | 2 | 1 | 最小开销，近似标准 LoRA |
| 单域微调，标准设置 | 4 | 2 | 默认推荐 |
| 多域持续学习 | 8 | 2 | 域隔离效果好 |
| 高容量需求 | 8 | 2 | 配合 hybrid router |

### 10.3 辅助损失系数

| 系数 | 推荐值 | 说明 |
|------|--------|------|
| `balance_loss_coef` | `0.001` – `0.01` | 过低导致专家坍缩，过高干扰主任务 |
| `z_loss_coef` | `0.0001` – `0.001` | 保持稳定即可，通常不需大幅调整 |
| `diversity_loss_coef` | `0.0`（默认关闭） | 仅在观察到专家输出高度相似时开启 |

### 10.4 Router 类型选择

| Router 类型 | 计算开销 | 适用场景 |
|------------|---------|---------|
| `linear` | O(C) | 默认推荐，单图全局决策 |
| `spatial` | O(C·H·W) | 需要空间感知（如前景/背景差异大） |
| `hybrid` | O(C·H·W) | 兼顾全局与局部，复杂场景 |

### 10.5 top_k_warmup 策略

训练初期只激活 1 个专家，逐步增加到 `top_k`，有助于稳定路由学习：

```python
cfg = MoLoRAConfig(
    top_k=2,
    top_k_warmup=True,    # 启用 warmup
    warmup_steps=1000,    # 前 1000 步从 K=1 渐变到 K=2
)
```

### 10.6 推理优化

训练完成后，**强烈建议合并权重**：

```python
wrapper.merge()
```

合并后：
- 推理速度 ≈ 原始模型（无额外层开销）
- 内存占用 ≈ 原始模型（不加载专家参数）
- 精度损失极小（各专家增量平均化）

如需恢复训练能力：

```python
wrapper.unmerge()
```

### 10.7 持续学习最佳实践

1. **域分配**：使用 `allocate_domain_experts()` 均匀分配专家。
2. **顺序训练**：每域训练时 `set_domain()` 限制路由范围。
3. **专家冻结**：训练新域时 `freeze_experts()` 保护旧域专家。
4. **专家回放**：定期 `save_expert_replay_buffer()` 保存旧域权重。
5. **联合评估**：最终评估时加载所有域的回放缓冲区，按域切换。

---

## 11. 附录：API 速查表

### 11.1 公共 API 导入

```python
from ultralytics.nn.peft.molora import (
    # 配置
    MoLoRAConfig,
    MoLoRAConfigBuilder,
    get_molora_preset,
    # 路由器
    build_router,
    LinearRouter,
    SpatialRouter,
    HybridRouter,
    # 层
    MoLoRAExpert,
    MoLoRALayer,
    # 损失
    MoLoRALoss,
    compute_expert_usage,
    # 模型包装
    get_peft_molora_model,
    MoLoRAModel,
    # 工具
    mark_only_molora_as_trainable,
    count_parameters,
    allocate_domain_experts,
)
```

### 11.2 核心类/函数签名速查

```python
# === 配置 ===
MoLoRAConfig(r, alpha, num_experts=4, top_k=2, router_type="linear", ...)
MoLoRAConfig.from_lora_config(lora_config, **overrides)
MoLoRAConfig.from_args(args=None, **kwargs)
MoLoRAConfigBuilder.create_molora_config(model, r=8, alpha=None, ...)
get_molora_preset(name: str) -> dict

# === 路由器 ===
build_router(router_type, in_channels, num_experts, hidden_dim=None) -> nn.Module
LinearRouter(in_channels, num_experts, hidden_dim=None)
SpatialRouter(in_channels, num_experts, hidden_dim=None)
HybridRouter(in_channels, num_experts, hidden_dim=None)

# === 层 ===
MoLoRAExpert(base_layer, r, alpha, dropout=0.0, use_rslora=True, init_type="default")
MoLoRALayer(base_layer, r=8, alpha=16, num_experts=4, top_k=2, ...)
    .set_domain(domain: str)
    .clear_domain()
    .freeze_experts(expert_indices: List[int])
    .unfreeze_experts(expert_indices=None)
    .merge_weights()
    .unmerge_weights()

# === 损失 ===
MoLoRALoss(num_experts, top_k, balance_loss_coef=0.01, z_loss_coef=0.001,
           diversity_loss_coef=0.0, reduce_ddp=False)
compute_expert_usage(expert_indices, num_experts) -> Tensor

# === 模型包装 ===
get_peft_molora_model(model, config) -> nn.Module
MoLoRAModel(model, config)
    .compute_aux_loss() -> Tensor
    .merge() / .unmerge()
    .set_domain(domain) / .clear_domain()
    .freeze_experts(indices) / .unfreeze_experts(indices)
    .save_expert_replay_buffer(domain, path=None) -> dict
    .load_expert_replay_buffer(buffer, domain=None)
    .save_checkpoint(path) / .load_checkpoint(path)
    .param_stats() -> dict

# === 工具 ===
mark_only_molora_as_trainable(model)
count_parameters(model) -> dict
allocate_domain_experts(num_experts, domains) -> dict
```

### 11.3 文件位置

```
/Users/gatilin/PycharmProjects/YOLO-Master-v260703/
├── ultralytics/nn/peft/molora/
│   ├── __init__.py       # 公共 API 导出
│   ├── config.py         # MoLoRAConfig / ConfigBuilder / Presets
│   ├── router.py         # LinearRouter / SpatialRouter / HybridRouter
│   ├── layer.py          # MoLoRAExpert / MoLoRALayer
│   ├── loss.py           # MoLoRALoss / compute_expert_usage
│   ├── model.py          # get_peft_molora_model / MoLoRAModel
│   └── utils.py          # init / merge / freeze / stats 工具
└── examples/molora/
    ├── basic_finetune.py         # 基础单域微调示例
    ├── compare_lora_molora.py    # LoRA vs MoLoRA 对比实验
    ├── continual_learning.py     # 持续学习（多域顺序训练）
    └── compare_coco128.py        # COCO128 快速对比
```

---

*文档版本：v1.0 | 基于 YOLO-Master MoLoRA 实现源码 | 技术术语保留英文原词*
