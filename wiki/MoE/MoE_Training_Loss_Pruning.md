# MoE 训练、损失与剪枝（MoE Training, Loss & Pruning）

本文档详细阐述 YOLO-Master 项目中 Mixture-of-Experts（MoE）模块的**训练辅助损失（Auxiliary Loss）**、**动态调度（Dynamic Scheduling）**与**专家剪枝（Expert Pruning）**三大核心机制。所有 API 签名、参数说明与使用示例均基于实际代码实现，技术术语保留英文原词。

---

## 目录

- [1. 概述](#1-概述)
- [2. MoE 辅助损失体系](#2-moe-辅助损失体系)
  - [2.1 核心设计原则](#21-核心设计原则)
  - [2.2 `MoELoss` 类](#22-moeloss-类)
  - [2.3 独立损失函数](#23-独立损失函数)
    - [`gshard_balance_loss`](#gshard_balance_loss)
    - [`weighted_gshard_balance_loss`](#weighted_gshard_balance_loss)
    - [`differentiable_balance_loss`](#differentiable_balance_loss)
  - [2.4 动态调度器 `MoEDynamicScheduler`](#24-动态调度器-moedynamicscheduler)
- [3. 全局辅助损失注册表](#3-全局辅助损失注册表)
  - [3.1 `MOE_LOSS_REGISTRY`](#31-moe_loss_registry)
  - [3.2 聚合辅助损失](#32-聚合辅助损失)
- [4. 路由诊断与监控](#4-路由诊断与监控)
  - [4.1 快照机制 `_record_moe_snapshot`](#41-快照机制-_record_moe_snapshot)
  - [4.2 `MoELayerDiagnostic` 数据结构](#42-moelayerdiagnostic-数据结构)
  - [4.3 `MoEDiagnosticsRecorder`](#43-moediagnosticsrecorder)
  - [4.4 `ExpertUsageTracker`](#44-expertusagetracker)
  - [4.5 `RoutingCollapseDetector`](#45-routingcollapsedetector)
- [5. 专家剪枝（Pruning）](#5-专家剪枝pruning)
  - [5.1 剪枝流程概览](#51-剪枝流程概览)
  - [5.2 `MoEPruner` 类](#52-moepruner-类)
  - [5.3 剪枝后的模型保存与验证](#53-剪枝后的模型保存与验证)
- [6. 使用示例](#6-使用示例)
  - [6.1 训练中使用 MoELoss](#61-训练中使用-moeloss)
  - [6.2 启用动态调度](#62-启用动态调度)
  - [6.3 运行专家使用诊断](#63-运行专家使用诊断)
  - [6.4 执行专家剪枝](#64-执行专家剪枝)
- [7. 参考实现路径](#7-参考实现路径)

---

## 1. 概述

YOLO-Master 的 MoE 系统采用 **GShard / Switch Transformer** 风格的负载均衡设计，通过辅助损失（auxiliary loss）引导路由器（router）将输入 token 均匀分配给各专家（expert）。核心组件包括：

| 组件 | 职责 | 关键文件 |
|------|------|----------|
| `MoELoss` | 计算 balance loss、z-loss、entropy loss、diversity loss、variance loss | `ultralytics/nn/modules/moe/loss.py` |
| `MoEDynamicScheduler` | 基于 Gini 系数动态调节 balance loss 系数 | `ultralytics/nn/modules/moe/scheduler.py` |
| `MOE_LOSS_REGISTRY` | 全局 WeakKeyDictionary，避免非 leaf tensor 导致的 deepcopy 错误 | `ultralytics/nn/modules/moe/modules.py` |
| `_record_moe_snapshot` | 每 N 步记录一次路由快照，供诊断使用 | `ultralytics/nn/modules/moe/modules.py` |
| `MoEPruner` | 基于专家使用率阈值剪除低利用率专家 | `ultralytics/nn/modules/moe/pruning.py` |

---

## 2. MoE 辅助损失体系

### 2.1 核心设计原则

1. **GShard 尺度统一**：所有 balance loss 在均匀分布时输出约 `1.0`，在完全 collapse（所有 token 路由到单一专家）时输出约 `num_experts`。这一统一尺度确保多 MoE 块叠加时不会相互淹没。
2. **DDP 感知**：所有跨 rank 的归约操作在 `float32` 下执行，避免 `float16` / `bfloat16` 精度灾难。
3. **梯度安全**：`differentiable_balance_loss` 通过 `importance = mean(router_probs)` 保留到 router logits 的梯度路径，确保 router 能真正学习负载均衡。
4. **NaN 防护**：所有损失输出均经过 `torch.isfinite` 检查，非有限值自动替换为 `0.0`。

### 2.2 `MoELoss` 类

```python
class MoELoss(nn.Module):
    def __init__(
        self,
        balance_loss_coeff: float = 1.0,       # 负载均衡损失系数
        z_loss_coeff: float = 1.0,             # Z-loss（路由器稳定性）系数
        entropy_loss_coeff: float = 0.0,       # 熵正则化系数（防止 router 犹豫不决）
        diversity_loss_coeff: float = 0.0,     # 多样性损失系数（惩罚相似的专家输出）
        variance_loss_coeff: float = 0.0,      # 方差损失系数（直接惩罚使用不均衡）
        num_experts: int = 8,
        top_k: int = 2,
        use_soft_balancing: bool = False,      # True: 软均衡（可微分）; False: 硬均衡
        coeff_floor: float = 0.0,              # 系数下限（防止过小系数被静默覆盖）
        dynamic_scheduler: Optional[MoEDynamicScheduler] = None,
        dynamic_scheduler_config: Optional[MoEDynamicSchedulerConfig] = None,
    )
```

#### `forward` 方法

```python
def forward(
    self,
    router_probs: torch.Tensor,              # [B, num_experts] 路由器概率分布
    router_logits: torch.Tensor,             # [B, num_experts] 原始 logits
    expert_indices: Optional[torch.Tensor] = None,   # [B, k] Top-K 选中的专家索引
    expert_outputs: Optional[torch.Tensor] = None,   # [B, num_experts, D] 专家输出特征（用于 diversity loss）
    return_dict: bool = False
) -> Union[torch.Tensor, Dict[str, torch.Tensor]]
```

**返回**：
- `return_dict=False`：返回标量 `total_loss`
- `return_dict=True`：返回字典，包含各分项损失值：
  ```python
  {
      "loss": total_loss,
      "balance_loss": balance_loss.detach(),
      "z_loss": z_loss.detach(),
      "entropy_loss": entropy_loss.detach(),
      "diversity_loss": diversity_loss.detach(),
      "variance_loss": variance_loss.detach(),
      "dynamic_schedule": schedule_state.to_dict() if schedule_state else None,
  }
  ```

#### 损失项详解

| 损失项 | 公式 | 作用 |
|--------|------|------|
| **Balance Loss** | `N * sum(importance * usage)` | 惩罚专家负载不均衡。`importance` 保留梯度，`usage` detached。 |
| **Z-Loss** | `mean(logsumexp(logits)^2)` | 惩罚 router logits 过大，提升数值稳定性。 |
| **Entropy Loss** | `-sum(probs * log(probs + eps))` | 鼓励 router 做出确定性决策（降低熵）。 |
| **Diversity Loss** | `sum(cosine_similarity^2) / num_pairs` | 惩罚专家输出余弦相似度过高，鼓励正交化。 |
| **Variance Loss** | `mean((usage - 1/N)^2)` | 直接惩罚使用率的方差，鼓励均匀分布。 |

#### 软均衡 vs 硬均衡

- **Hard Balancing（`use_soft_balancing=False`，默认）**：`usage` 由离散 Top-K 计数统计得到，需要传入 `expert_indices`。与 GShard / Switch Transformer 一致。
- **Soft Balancing（`use_soft_balancing=True`）**：`usage` 优先使用离散 Top-K 计数（当 `expert_indices` 可用时），否则回退到 `importance.detach()`。`importance` 保留到 router 的梯度。

### 2.3 独立损失函数

#### `gshard_balance_loss`

```python
def gshard_balance_loss(
    expert_usage: torch.Tensor,
    num_experts: int,
    reduce_ddp: bool = False
) -> torch.Tensor
```

**GShard-style balance loss**：`N * sum(usage^2)`。均匀分布时等于 `1.0`。当 `reduce_ddp=True` 时，先对 usage 做跨 rank 平均。

#### `weighted_gshard_balance_loss`

```python
def weighted_gshard_balance_loss(
    expert_usage: torch.Tensor,
    target_usage: torch.Tensor,
    num_experts: int,
    reduce_ddp: bool = False,
) -> torch.Tensor
```

**可学习目标分布的 balance loss**。当 `target_usage` 为均匀分布时退化为 `gshard_balance_loss`；最小值在 `usage == target` 时取得。

#### `differentiable_balance_loss`

```python
def differentiable_balance_loss(
    router_probs: torch.Tensor,
    expert_usage: torch.Tensor,
    num_experts: int,
    target_usage: Optional[torch.Tensor] = None,
    reduce_ddp: bool = False,
) -> torch.Tensor
```

**可微分 balance loss**，`importance = mean(router_probs)` 保留到 router logits 的梯度。这是 `UltraOptimizedMoE` 等模块内部使用的标准形式。

### 2.4 动态调度器 `MoEDynamicScheduler`

```python
@dataclass
class MoEDynamicSchedulerConfig:
    enabled: bool = True
    target_gini: float = 0.25        # 目标 Gini 系数（越低越均衡）
    gain: float = 1.5                # 调节增益
    min_balance_coeff: float = 0.02  # 系数下限
    max_balance_coeff: float = 2.0   # 系数上限
    ema_momentum: float = 0.9        # EMA 动量
```

```python
class MoEDynamicScheduler:
    def __init__(self, config: MoEDynamicSchedulerConfig | None = None)
    def step(self, expert_usage: torch.Tensor, base_balance_coeff: float) -> MoEDynamicScheduleState
```

**调度公式**：

```
coeff_t = clamp(base_coeff * (1 + gain * (ema_gini_t - target_gini)), min_coeff, max_coeff)
```

- Gini 高（路由不均衡）→ 增大 balance loss 系数，强制均衡。
- Gini 低（路由健康）→ 降低系数，让专家自由特化。

**使用方式**：

```python
scheduler = MoEDynamicScheduler(MoEDynamicSchedulerConfig(target_gini=0.2, gain=2.0))
loss_fn = MoELoss(
    balance_loss_coeff=1.0,
    dynamic_scheduler=scheduler
)
```

---

## 3. 全局辅助损失注册表

### 3.1 `MOE_LOSS_REGISTRY`

```python
MOE_LOSS_REGISTRY = weakref.WeakKeyDictionary()
```

为避免将非 leaf tensor 直接存储在模块实例中（会导致 `deepcopy` / EMA 报错），YOLO-Master 使用 **线程安全的 WeakKeyDictionary** 作为全局注册表：

- **键（Key）**：MoE 模块实例（如 `UltraOptimizedMoE` 对象）
- **值（Value）**：该模块前向传播产生的辅助损失标量张量

线程安全通过 `_threading.Lock()` 保证：

```python
def _registry_set(module: nn.Module, value: torch.Tensor) -> None:
    with _MOE_LOSS_REGISTRY_LOCK:
        MOE_LOSS_REGISTRY[module] = value

def _registry_get(module: nn.Module):
    with _MOE_LOSS_REGISTRY_LOCK:
        return MOE_LOSS_REGISTRY.get(module)
```

### 3.2 聚合辅助损失

训练时，通过 `_collect_moe_aux_loss` 遍历模型所有模块，从注册表中收集并求和：

```python
from ultralytics.utils.loss import _collect_moe_aux_loss

# model: nn.Module，device: torch.device
aux_loss = _collect_moe_aux_loss(model, device)
# aux_loss 为标量张量，可直接加入总损失

total_loss = detection_loss + aux_loss
```

**重要特性**：
- 每个模块每步只产生一个注册表条目，不会泄漏增长。
- `eval()` 模式下不写入注册表，聚合结果自动为 `0`。
- `A2C2fMoE` 等包装模块通过 `aux_loss` property 委托到内部 MoE 模块，避免重复计数。

---

## 4. 路由诊断与监控

### 4.1 快照机制 `_record_moe_snapshot`

每模块每 `MOE_SNAPSHOT_INTERVAL` 步（默认 `10`，可通过环境变量 `MOE_SNAPSHOT_INTERVAL` 调整）记录一次路由快照：

```python
def _record_moe_snapshot(
    module: nn.Module,
    *,
    expert_usage: Optional[torch.Tensor] = None,
    topk_indices: Optional[torch.Tensor] = None,
    topk_weights: Optional[torch.Tensor] = None,
    router_probs: Optional[torch.Tensor] = None,
    aux_loss: Optional[torch.Tensor] = None,
) -> None
```

快照存储在模块的 `last_routing_snapshot` 属性中：

```python
{
    "num_experts": int,
    "top_k": int,
    "expert_usage": Tensor,        # 各专家使用占比
    "topk_counts": Tensor,         # 各专家被选中次数
    "mean_router_probs": Tensor,   # 平均路由器概率
    "aux_loss": float,
    "mean_topk_weight": Tensor,    # 平均 Top-K 权重
}
```

### 4.2 `MoELayerDiagnostic` 数据结构

```python
@dataclass
class MoELayerDiagnostic:
    name: str                       # 模块完整名称
    module_type: str                # 模块类型名
    num_experts: int
    top_k: int
    aux_loss: float
    usage: list[float]              # 各专家使用占比
    counts: list[float]             # 各专家被选中次数
    dominant_expert: int            # 主导专家索引
    dominant_share: float           # 主导专家占比
    mean_router_probs: list[float] | None
    mean_topk_weight: list[float] | None
    collapse_flag: bool             # 是否发生路由崩溃
```

### 4.3 `MoEDiagnosticsRecorder`

持久化诊断记录器，支持 CSV / JSONL 格式输出与自动告警：

```python
class MoEDiagnosticsRecorder:
    def __init__(
        self,
        save_dir: str | Path,
        dead_threshold: float = 0.01,      # 死专家阈值
        dead_window: int = 5,              # 死专家判定窗口
        collapse_threshold: float = 0.8,   # 崩溃阈值
        collapse_window: int = 3,          # 崩溃判定窗口
    )
    def record(self, *, step: int, epoch: int, diagnostics: list[MoELayerDiagnostic], stage: str = "train") -> list[dict]
    def export_plots(self) -> list[Path]
```

**告警类型**：
- **dead_expert**：某专家连续 `dead_window` 步使用占比低于 `dead_threshold`
- **routing_collapse**：某专家连续 `collapse_window` 步成为主导且占比超过 `collapse_threshold`

### 4.4 `ExpertUsageTracker`

通过 forward hook 收集模型各 router 模块的输出统计：

```python
class ExpertUsageTracker:
    def __init__(self, model: torch.nn.Module)
    def print_report(self) -> None          # 打印诊断报告与可视化
    def __enter__(self) -> "ExpertUsageTracker"   # 支持上下文管理器
    def __exit__(self, *args) -> None
```

**使用示例**：

```python
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker
from ultralytics import YOLO

model = YOLO("yolo_moe_model.pt")

with ExpertUsageTracker(model.model) as tracker:
    model.val(data="coco8.yaml", split="val", batch=1, verbose=False)
    tracker.print_report()  # 输出热力图与柱状图
```

### 4.5 `RoutingCollapseDetector`

轻量级实时崩溃检测器，适用于训练循环：

```python
class RoutingCollapseDetector:
    def __init__(self, collapse_threshold: float = 0.8, dead_threshold: float = 0.05)
    def diagnose(self, model: nn.Module) -> dict    # 返回每层诊断结果
    def get_recovery_actions(self, diagnosis: dict) -> list[dict]  # 生成恢复建议
    def apply_recovery(self, model: nn.Module, diagnosis: dict) -> int  # 自动应用恢复
```

**恢复动作**：
- `increase_balance_loss`：增大 balance loss 系数
- `increase_noise`：增大 router 噪声以鼓励探索
- `reinit_dead_experts`：重新初始化死专家

---

## 5. 专家剪枝（Pruning）

### 5.1 剪枝流程概览

`MoEPruner` 实现完整的 5 阶段剪枝流水线：

| 阶段 | 方法 | 说明 |
|------|------|------|
| Phase 1 | `_load_model` | 加载 YOLO 模型 |
| Phase 2 | `_diagnose_usage` | 运行验证，收集各专家使用统计 |
| Phase 3 | `_create_pruning_plan` | 基于阈值决定保留哪些专家 |
| Phase 4 | `_perform_surgery` | 执行实际剪枝：移除专家 + 调整 router projection |
| Phase 5 | `_save_model` + `_verify_model` | 保存并验证剪枝后模型 |

### 5.2 `MoEPruner` 类

```python
class MoEPruner:
    def __init__(
        self,
        model_path: str,                     # 输入模型路径
        threshold: float = 0.15,             # 最低使用占比阈值（0.0 ~ 1.0）
        dataset: str = "coco8.yaml",         # 验证数据集配置
        device: Optional[str] = None,        # 设备（None 自动检测 CUDA/MPS/CPU）
    )
    def prune(self, output_path: str) -> bool   # 执行完整剪枝流水线
```

**关键方法详解**：

#### `_prune_experts`

```python
def _prune_experts(self, moe_module: nn.Module, keep_indices: List[int]) -> None
```

- 从 `moe_module.experts` 中仅保留 `keep_indices` 指定的专家
- 更新 `moe_module.num_experts`
- 若 `top_k` 超过剩余专家数，自动下调

#### `_prune_router_weights`

```python
def _prune_router_weights(
    self,
    router: nn.Module,
    keep_indices: List[int],
    num_old_experts: int
) -> bool
```

- 自动定位 router 的 projection 层（支持 `nn.Conv2d` 与 `nn.Linear`）
- 创建新的 projection 层，输出维度 = `len(keep_indices)`
- 复制保留专家的权重与偏置

### 5.3 剪枝后的模型保存与验证

剪枝后的模型以 checkpoint 格式保存：

```python
checkpoint = {
    "model": pruned_model,
    "updates": None,
    "pruning_info": {
        "threshold": self.threshold,
        "pruning_plan": self.pruning_plan,   # 每层保留的专家索引
    }
}
torch.save(checkpoint, output_path)
```

保存后自动执行验证：加载模型并运行 `model.val()`，确保剪枝后模型可正常推理。

---

## 6. 使用示例

### 6.1 训练中使用 MoELoss

```python
import torch
from ultralytics.nn.modules.moe.loss import MoELoss

# 初始化损失函数（软均衡 + 熵正则化 + diversity loss）
loss_fn = MoELoss(
    balance_loss_coeff=1.0,
    z_loss_coeff=0.5,
    entropy_loss_coeff=0.01,
    diversity_loss_coeff=0.1,
    num_experts=8,
    top_k=2,
    use_soft_balancing=True,
)

# 模拟 router 输出
B, E, K = 16, 8, 2
router_logits = torch.randn(B, E, requires_grad=True)
router_probs = torch.softmax(router_logits, dim=1)
expert_indices = torch.topk(router_probs, K, dim=1).indices

# 计算损失
total_loss = loss_fn(router_probs, router_logits, expert_indices)
print(f"Total aux loss: {total_loss.item():.4f}")

# 反向传播
total_loss.backward()
print(f"Router grad sum: {router_logits.grad.abs().sum().item():.4f}")
```

### 6.2 启用动态调度

```python
from ultralytics.nn.modules.moe.loss import MoELoss
from ultralytics.nn.modules.moe.scheduler import MoEDynamicScheduler, MoEDynamicSchedulerConfig

config = MoEDynamicSchedulerConfig(
    enabled=True,
    target_gini=0.25,
    gain=1.5,
    min_balance_coeff=0.02,
    max_balance_coeff=2.0,
    ema_momentum=0.9,
)

scheduler = MoEDynamicScheduler(config)
loss_fn = MoELoss(
    balance_loss_coeff=1.0,
    dynamic_scheduler=scheduler,
)

# 训练循环中
expert_usage = torch.tensor([0.5, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05])  # 不均衡
loss = loss_fn(router_probs, router_logits, expert_indices)
# scheduler 自动在 loss_fn.forward 内部被调用
# 可通过 return_dict=True 获取动态调度状态
result = loss_fn(router_probs, router_logits, expert_indices, return_dict=True)
print(result["dynamic_schedule"])
# {'gini': 0.35, 'ema_gini': 0.32, 'balance_loss_coeff': 1.21, ...}
```

### 6.3 运行专家使用诊断

```python
from ultralytics.nn.modules.moe.analysis import diagnose_model

# 一键诊断
diagnose_model(
    model_path="yolo_moe_trained.pt",
    dataset="coco8.yaml",
    batch_size=1,
    verbose=False,
)
```

输出示例：
```
================================================================================
                    🔍 EXPERT USAGE DIAGNOSIS REPORT
================================================================================

📊 Total Tokens Processed: 5,000

────────────────────────────────────────────────────────────────────────────────
📍 Layer: model.10.moe
────────────────────────────────────────────────────────────────────────────────
ID     | Usage %    | Avg Weight   | Hits         | Status
------|------------|--------------|--------------|----------
0     |    45.20%  |      0.5210  |       22,600 | 🔥 HOT
1     |    12.50%  |      0.4800  |        6,250 | ✅ OK
2     |     8.30%  |      0.4500  |        4,150 | ⚠️  LOW
3     |    34.00%  |      0.5100  |       17,000 | 🔥 HOT

📈 Summary:
   • Total Experts: 4
   • Ideal Share: 25.00%
   • Total Hits: 50,000
   • Load Balance (StdDev): 16.35%
```

### 6.4 执行专家剪枝

```python
from ultralytics.nn.modules.moe.pruning import prune_moe_model

# 一键剪枝（保留使用率 >= 15% 的专家）
success = prune_moe_model(
    model_path="yolo_moe_trained.pt",
    output_path="yolo_moe_pruned.pt",
    threshold=0.15,
    dataset="coco8.yaml",
)

if success:
    print("✅ 剪枝完成，模型已保存")
```

**CLI 使用**：

```bash
python -m ultralytics.nn.modules.moe.pruning yolo_moe_trained.pt \
    --output pruned_model.pt \
    --threshold 0.15 \
    --dataset coco8.yaml
```

---

## 7. 参考实现路径

| 模块 | 路径 | 说明 |
|------|------|------|
| `MoELoss` | `ultralytics/nn/modules/moe/loss.py` | 辅助损失实现 |
| `MoEDynamicScheduler` | `ultralytics/nn/modules/moe/scheduler.py` | 动态调度器 |
| `MoEPruner` | `ultralytics/nn/modules/moe/pruning.py` | 专家剪枝器 |
| `ExpertUsageTracker` | `ultralytics/nn/modules/moe/analysis.py` | 使用统计跟踪 |
| `RoutingCollapseDetector` | `ultralytics/nn/modules/moe/analysis.py` | 崩溃检测器 |
| `MoEDiagnosticsRecorder` | `ultralytics/nn/modules/moe/history.py` | 诊断持久化 |
| `MoELayerDiagnostic` | `ultralytics/nn/modules/moe/diagnostics.py` | 诊断数据结构 |
| `MOE_LOSS_REGISTRY` | `ultralytics/nn/modules/moe/modules.py` | 全局损失注册表 |
| `_record_moe_snapshot` | `ultralytics/nn/modules/moe/modules.py` | 快照记录 |
| `_collect_moe_aux_loss` | `ultralytics/utils/loss.py` | 辅助损失聚合 |
| 单元测试 | `tests/test_moe.py` | MoE 模块回归测试 |
| 混合测试 | `tests/test_mixture_aux_loss.py` | MoA/MoT/MoE 混合损失测试 |
