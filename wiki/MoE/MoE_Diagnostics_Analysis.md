# MoE 诊断与分析系统 (MoE Diagnostics & Analysis)

## 概述

YOLO-Master 的 **MoE (Mixture-of-Experts)** 诊断与分析系统提供了一套完整的工具链，用于监控、分析、诊断和优化模型中 MoE 模块的路由行为。该系统覆盖从训练时实时监控到事后离线分析的完整生命周期，帮助开发者识别路由崩溃 (routing collapse)、专家死锁 (dead experts)、负载不均衡等常见问题，并支持自动恢复与模型剪枝。

本文档基于 `ultralytics/nn/modules/moe/` 目录下的实际代码实现编写，涵盖以下核心子模块：

| 子模块 | 文件 | 职责 |
|--------|------|------|
| `diagnostics` | `diagnostics.py` | 轻量级路由快照收集与格式化输出 |
| `analysis` | `analysis.py` | 专家使用追踪、路由崩溃检测、诊断报告 |
| `history` | `history.py` | 诊断数据持久化、历史趋势绘图、告警系统 |
| `pruning` | `pruning.py` | 基于使用统计的专家剪枝与模型压缩 |
| `loss` | `loss.py` | MoE 辅助损失 (auxiliary loss) 计算 |
| `scheduler` | `scheduler.py` | Gini 驱动的动态平衡损失调度 |

---

## 目录

1. [轻量级诊断快照 (MoELayerDiagnostic)](#1-轻量级诊断快照-moelayerdiagnostic)
2. [专家使用追踪器 (ExpertUsageTracker)](#2-专家使用追踪器-expertusagetracker)
3. [路由崩溃检测器 (RoutingCollapseDetector)](#3-路由崩溃检测器-routingcollapsedetector)
4. [诊断持久化与告警 (MoEDiagnosticsRecorder)](#4-诊断持久化与告警-moediagnosticsrecorder)
5. [模型剪枝 (MoEPruner)](#5-模型剪枝-moepruner)
6. [训练集成](#6-训练集成)
7. [完整使用示例](#7-完整使用示例)
8. [测试覆盖](#8-测试覆盖)

---

## 1. 轻量级诊断快照 (MoELayerDiagnostic)

### 1.1 数据结构

```python
from dataclasses import dataclass
from typing import Any

@dataclass
class MoELayerDiagnostic:
    """单个 MoE 层的结构化路由摘要。"""

    name: str                          # 模块完整路径名 (如 "model.2.moe")
    module_type: str                   # 模块类名 (如 "UltraOptimizedMoE")
    num_experts: int                   # 专家总数
    top_k: int                         # 每次前向激活的专家数
    aux_loss: float                    # 辅助损失值
    usage: list[float]                 # 各专家使用占比 (归一化到 1.0)
    counts: list[float]                # 各专家被选中次数
    dominant_expert: int               # 主导专家索引 (使用占比最高)
    dominant_share: float              # 主导专家占比
    mean_router_probs: list[float] | None   # 平均路由概率分布
    mean_topk_weight: list[float] | None    # Top-K 权重均值
    collapse_flag: bool                # 是否触发路由崩溃标记
```

`MoELayerDiagnostic` 是所有诊断工具的通用数据交换格式。它由 `_record_moe_snapshot()` 在每次前向传播时自动填充到模块的 `last_routing_snapshot` 属性中，采样间隔可通过环境变量 `MOE_SNAPSHOT_INTERVAL` 控制（默认每 10 步采样一次，避免训练时过多开销）。

### 1.2 核心函数

#### `collect_moe_diagnostics`

```python
def collect_moe_diagnostics(
    model: torch.nn.Module,
    collapse_threshold: float = 0.8
) -> list[MoELayerDiagnostic]:
    """从暴露 `last_routing_snapshot` 的 MoE 层收集诊断信息。

    Args:
        model: 待诊断的 PyTorch 模型。
        collapse_threshold: 路由崩溃判定阈值。若某专家使用占比超过该值，
            则 `collapse_flag` 标记为 True。

    Returns:
        按模型遍历顺序排列的 MoELayerDiagnostic 列表。
    """
```

**实现细节：**
- 遍历 `model.named_modules()`，筛选出带有 `last_routing_snapshot` 属性的模块
- 若同时提供了 `expert_usage` 和 `topk_indices`，优先使用前者（反映 router 实际计算的使用频率）
- 自动将 `torch.Tensor` 转换为 Python `list[float]`，便于 JSON 序列化

#### `diagnostics_to_dict`

```python
def diagnostics_to_dict(
    diagnostics: list[MoELayerDiagnostic]
) -> list[dict[str, Any]]:
    """将诊断列表转换为 JSON 可序列化的字典列表。"""
```

#### `format_moe_diagnostics`

```python
def format_moe_diagnostics(
    diagnostics: list[MoELayerDiagnostic],
    title: str = "MoE Routing Diagnostics"
) -> str:
    """渲染紧凑的文本摘要，供 CLI 和训练回调统一使用。

    输出格式示例:
        [MoE] MoE Routing Diagnostics
        [MoE] model.2.moe | aux=1.234567 | top_k=2/4 | dominant=E0(0.512) | collapse=False
        [MoE] usage  E0:0.512, E1:0.245, E2:0.123, E3:0.120
        [MoE] counts E0:512, E1:245, E2:123, E3:120
        [MoE] topk   k0:0.712, k1:0.288
    """
```

### 1.3 环境变量配置

| 环境变量 | 默认值 | 说明 |
|----------|--------|------|
| `MOE_SNAPSHOT_INTERVAL` | `10` | 每个模块每 N 次前向传播记录一次快照。设为 `1` 可恢复逐次记录。 |

---

## 2. 专家使用追踪器 (ExpertUsageTracker)

`ExpertUsageTracker` 通过 **forward hook** 机制在模型推理过程中实时收集各 router 的输出，统计每个专家的命中次数 (hits) 和加权权重 (weighted sum)，最终生成包含热力图和柱状图的可视化诊断报告。

### 2.1 类签名

```python
class ExpertUsageTracker:
    """Tracker for Mixture-of-Experts (MoE) expert usage patterns."""

    SKIP_TYPES = (
        torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.SiLU,
        torch.nn.Sequential, torch.nn.AdaptiveAvgPool2d,
        torch.nn.Linear, torch.nn.GroupNorm, torch.nn.Softmax
    )
    ROUTER_KEYWORDS = ('routing', 'router')

    def __init__(self, model: torch.nn.Module):
        """
        Args:
            model: 包含 MoE 层的 PyTorch 模型。
        """
```

### 2.2 核心方法

| 方法 | 签名 | 说明 |
|------|------|------|
| `print_report` | `() -> None` | 打印综合诊断报告，包含各层专家使用统计、负载均衡标准差，并自动生成热力图和柱状图 |
| `remove_hooks` | `() -> None` | 移除所有已注册的 forward hook，防止内存泄漏 |
| `__enter__` / `__exit__` | 上下文管理器 | 支持 `with` 语句，确保退出时自动清理 hooks |

### 2.3 专家健康状态判定

Tracker 根据实际使用占比与理想均衡占比的比率，将每个专家标记为以下状态：

| 状态 | Emoji | 判定条件 |
|------|-------|----------|
| DEAD | 💀 | `share_pct < ideal_share * 0.1` |
| LOW | ⚠️ | `ideal_share * 0.1 <= share_pct < ideal_share * 0.5` |
| HOT | 🔥 | `share_pct > ideal_share * 2.0` |
| OK | ✅ | 其他 |

### 2.4 使用示例

```python
from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker
from ultralytics import YOLO

# 加载模型
model = YOLO("yolo-master-moe.pt")

# 使用上下文管理器自动管理 hooks
with ExpertUsageTracker(model.model) as tracker:
    # 运行验证集推理
    model.val(data="coco8.yaml", split="val", batch=1, verbose=False)
    
    # 打印诊断报告（含热力图 expert_usage_heatmap.png 和柱状图 expert_usage_bar.png）
    tracker.print_report()
```

### 2.5 可视化输出

`print_report()` 会自动生成两张图：

1. **expert_usage_heatmap.png** — 热力图展示各层各专家的选择百分比
2. **expert_usage_bar.png** — 柱状图展示全局平均专家使用分布，附带理想均衡参考线

---

## 3. 路由崩溃检测器 (RoutingCollapseDetector)

`RoutingCollapseDetector` 是专为**训练循环**设计的轻量级实时检测器。它定期扫描模型中的所有 MoE 模块，检测路由崩溃（单个专家占据绝大部分路由权重）和专家死锁（专家长期不被选中），并可自动执行恢复动作。

### 3.1 类签名

```python
class RoutingCollapseDetector:
    """Lightweight real-time routing collapse detector for training loops."""

    def __init__(
        self,
        collapse_threshold: float = 0.8,
        dead_threshold: float = 0.05
    ):
        """
        Args:
            collapse_threshold: 若某专家使用占比超过该阈值，标记为崩溃。
            dead_threshold: 若某专家使用占比低于该阈值，标记为死锁。
        """
```

### 3.2 核心方法

#### `diagnose`

```python
def diagnose(self, model: torch.nn.Module) -> dict:
    """扫描所有 MoE 模块，返回逐层专家使用比率。

    Returns:
        dict: layer_name -> {
            'usage': [float],      # 各专家使用占比
            'collapsed': bool,     # 是否崩溃
            'dead_experts': [int], # 死锁专家索引列表
            'max_usage': float,
            'min_usage': float,
        }
    """
```

**数据来源优先级：**
1. `last_routing_snapshot['expert_usage']`
2. `last_routing_snapshot['mean_router_probs']`
3. `expert_usage_counts` (ES_MOE 专用)

#### `get_recovery_actions`

```python
def get_recovery_actions(self, diagnosis: dict) -> list:
    """根据诊断结果生成纠正动作列表。

    Returns:
        [{'action': str, 'params': dict, 'reason': str}, ...]

    当前支持的动作:
        - 'increase_balance_loss': 增大 balance_loss 系数（默认 factor=2.0）
        - 'increase_noise': 增大 router 的 noise_std（默认 1.0）
        - 'reinit_dead_experts': 重新初始化死锁专家的权重
    """
```

#### `apply_recovery`

```python
def apply_recovery(
    self,
    model: torch.nn.Module,
    diagnosis: dict
) -> int:
    """自动将恢复动作应用到模型。

    Returns:
        实际应用的动作数量。
    """
```

---

## 4. 诊断持久化与告警 (MoEDiagnosticsRecorder)

`MoEDiagnosticsRecorder` 将 MoE 诊断数据持久化到磁盘，支持 CSV 和 JSON Lines 格式，并内建滑动窗口告警机制，可持续检测死锁专家和路由崩溃。

### 4.1 类签名

```python
class MoEDiagnosticsRecorder:
    """Persist MoE routing snapshots and detect sustained routing issues."""

    def __init__(
        self,
        save_dir: str | Path,
        dead_threshold: float = 0.01,      # 死锁判定阈值
        dead_window: int = 5,              # 死锁持续窗口（步数）
        collapse_threshold: float = 0.8,   # 崩溃判定阈值
        collapse_window: int = 3,          # 崩溃持续窗口（步数）
    ) -> None:
```

### 4.2 输出文件结构

```
save_dir/
├── routing_history.jsonl   # 每行一个 JSON 对象，完整快照记录
├── routing_history.csv     # CSV 格式，便于 Excel/ pandas 分析
├── alerts.jsonl            # 触发的告警记录
└── plots/
    ├── <layer_name>_usage.png   # 各层专家使用占比随 step 变化曲线
    └── aux_loss_vs_step.png     # 全局 aux_loss 变化曲线
```

### 4.3 CSV 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `stage` | str | 阶段，如 `"train"` / `"val"` |
| `step` | int | 全局训练步数 |
| `epoch` | int | 当前 epoch |
| `layer_name` | str | MoE 层路径 |
| `module_type` | str | 模块类名 |
| `num_experts` | int | 专家总数 |
| `top_k` | int | Top-K 值 |
| `expert_id` | int | 专家索引 |
| `usage` | float | 该专家使用占比 |
| `count` | float | 该专家被选中次数 |
| `dominant_expert` | int | 当前主导专家 |
| `dominant_share` | float | 主导专家占比 |
| `aux_loss` | float | 辅助损失值 |
| `collapse_flag` | bool | 崩溃标记 |

### 4.4 告警机制

Recorder 维护两个滑动窗口：

- **死锁告警 (dead_expert)**：某专家连续 `dead_window` 步使用率低于 `dead_threshold`
- **崩溃告警 (routing_collapse)**：某专家连续 `collapse_window` 步成为主导专家且占比超过 `collapse_threshold`

告警触发后会写入 `alerts.jsonl`，并在内部 `_active_alerts` 集合中追踪，避免重复告警直至条件解除。

### 4.5 使用示例

```python
from ultralytics.nn.modules.moe.history import MoEDiagnosticsRecorder
from ultralytics.nn.modules.moe.diagnostics import collect_moe_diagnostics

recorder = MoEDiagnosticsRecorder(save_dir="./moe_logs")

for epoch in range(num_epochs):
    for step, batch in enumerate(train_loader):
        # ... 训练前向/反向 ...
        
        # 每 10 步记录一次诊断
        if step % 10 == 0:
            diagnostics = collect_moe_diagnostics(model)
            alerts = recorder.record(
                step=global_step,
                epoch=epoch,
                diagnostics=diagnostics,
                stage="train",
            )
            if alerts:
                print(f"[告警] 触发 {len(alerts)} 条告警: {[a['alert_type'] for a in alerts]}")

# 训练结束后导出所有趋势图
plot_paths = recorder.export_plots()
print(f"已生成 {len(plot_paths)} 张图表")
```

---

## 5. 模型剪枝 (MoEPruner)

`MoEPruner` 基于 `ExpertUsageTracker` 收集的使用统计，自动识别并移除利用率低的专家，同时调整 router 的投影层权重，生成更轻量的模型。

### 5.1 类签名

```python
class MoEPruner:
    """Pruner for Mixture-of-Experts models based on usage statistics."""

    def __init__(
        self,
        model_path: str,
        threshold: float = 0.15,          # 保留专家的最低使用率阈值
        dataset: str = 'coco8.yaml',      # 用于诊断的数据集
        device: Optional[str] = None,     # 计算设备，默认自动检测 CUDA/MPS/CPU
    ):
```

### 5.2 剪枝流程

剪枝流程分为 5 个阶段：

```
[Phase 1] 加载模型
    ↓
[Phase 2] 诊断专家使用 (ExpertUsageTracker)
    ↓
[Phase 3] 制定剪枝计划 → 逐层决定保留/删除的专家
    ↓
[Phase 4] 执行手术
    ├── 修剪 experts ModuleList
    └── 修剪 router 投影层权重
    ↓
[Phase 5] 保存并验证
```

### 5.3 安全机制

- **至少保留一个专家**：若某层所有专家使用率都低于阈值，自动保留使用率最高的专家
- **Top-K 兼容性检查**：若保留专家数少于原始 `top_k`，打印警告但不阻止剪枝
- **投影层权重裁剪**：自动复制保留专家对应的 router 权重到新的投影层

### 5.4 便捷函数

```python
def prune_moe_model(
    model_path: str,
    output_path: str,
    threshold: float = 0.15,
    dataset: str = 'coco8.yaml'
) -> bool:
    """一键剪枝 MoE 模型。

    Args:
        model_path: 输入模型路径 (.pt)
        output_path: 输出模型路径
        threshold: 最低使用率阈值 (0.0-1.0)
        dataset: 验证数据集配置

    Returns:
        True 表示剪枝成功并通过验证。
    """
```

### 5.5 使用示例

```python
from ultralytics.nn.modules.moe.pruning import prune_moe_model

success = prune_moe_model(
    model_path="yolo-master-moe.pt",
    output_path="yolo-master-moe-pruned.pt",
    threshold=0.15,          # 移除使用率 < 15% 的专家
    dataset="coco8.yaml",
)

if success:
    print("剪枝完成，模型已验证通过")
```

---

## 6. 训练集成

MoE 诊断系统已深度集成到 YOLO-Master 的**训练器 (`ultralytics/engine/trainer.py`)** 中，无需手动干预即可自动工作。

### 6.1 初始化阶段

在 `BaseTrainer.__init__` 中，若检测到模型包含核心 MoE 模块 (`model_has_core_moe`)，则自动：

1. **实例化 `RoutingCollapseDetector`**，默认崩溃阈值 `0.8`
2. **注入 MoE 超参数**：从 YAML 配置读取 `moe_balance_loss`、`moe_router_z_loss`、`moe_noise_std`、`moe_temperature`、`moe_weight_threshold`，并写入各 MoE 模块
3. **参数安全检查**：若 `balance_loss_coeff < 0.1` 或 `z_loss_coeff < 0.1`，强制提升到 `1.0` 并打印警告

```python
# 伪代码（来自 trainer.py）
if has_moe:
    from ultralytics.nn.modules.moe.analysis import RoutingCollapseDetector
    self._moe_collapse_detector = RoutingCollapseDetector(collapse_threshold=0.8)
    
    # 注入配置
    for m in self.model.modules():
        if is_core_moe_block(m):
            m.balance_loss_coeff = balance_loss_coeff
            m.router_z_loss_coeff = router_z_loss_coeff
            # ...
```

### 6.2 训练循环中的崩溃检测

每 5 个 epoch（且已过 warmup），训练器自动执行崩溃诊断与恢复：

```python
# 伪代码（来自 trainer.py _do_train）
if epoch > 0 and epoch % 5 == 0 and hasattr(self, '_moe_collapse_detector'):
    diag = self._moe_collapse_detector.diagnose(self.model)
    collapsed_layers = [n for n, d in diag.items() if d['collapsed']]
    
    if collapsed_layers:
        # 自动增大 noise_std 以恢复探索性
        applied = self._moe_collapse_detector.apply_recovery(self.model, diag)
        
        # 同时提升 balance_loss 系数（若尚未很高）
        # ...
```

### 6.3 辅助损失聚合

训练时，所有 MoE 模块的 `aux_loss` 通过全局注册表 `MOE_LOSS_REGISTRY` (`weakref.WeakKeyDictionary`) 聚合，避免：

- **重复计数**：通过模块 `id` 去重，解决嵌套 wrapper（如 `A2C2fMoE` → `ABlockMoE`）导致的多次累加
- **内存泄漏**：`WeakKeyDictionary` 确保模块被删除时条目自动释放
- **deepcopy 错误**：aux loss 不存储在模块 `__dict__` 中，避免 `copy.deepcopy` 时触发 "Only Tensors created explicitly by the user..." 错误
- **Eval 污染**：`eval()` 模式下不写入注册表，防止推理张量残留

聚合入口位于 `ultralytics/utils/loss.py`：

```python
def _collect_moe_aux_loss(model, device) -> torch.Tensor:
    """收集并汇总模型中所有 MoE 模块的辅助损失。"""
```

---

## 7. 完整使用示例

### 7.1 离线诊断现有模型

```python
from ultralytics.nn.modules.moe.analysis import diagnose_model

# 一键诊断（内部使用 ExpertUsageTracker）
diagnose_model(
    model_path="yolo-master-moe.pt",
    dataset="coco8.yaml",
    batch_size=1,
    verbose=False,
)
```

### 7.2 训练时实时监控

```python
from ultralytics.nn.modules.moe.diagnostics import collect_moe_diagnostics, format_moe_diagnostics
from ultralytics.nn.modules.moe.history import MoEDiagnosticsRecorder

recorder = MoEDiagnosticsRecorder(save_dir="./runs/moe_diag")

for epoch in range(100):
    for step, batch in enumerate(dataloader):
        loss = model(batch)
        loss.backward()
        optimizer.step()
        
        # 每 50 步收集并打印诊断
        if step % 50 == 0:
            diagnostics = collect_moe_diagnostics(model, collapse_threshold=0.8)
            print(format_moe_diagnostics(diagnostics))
            
            # 持久化到磁盘
            recorder.record(
                step=epoch * len(dataloader) + step,
                epoch=epoch,
                diagnostics=diagnostics,
                stage="train",
            )
```

### 7.3 训练后分析与可视化

```python
from ultralytics.nn.modules.moe.history import export_moe_history_plots
from pathlib import Path

# 从已有日志目录生成所有图表
save_dir = Path("./runs/moe_diag")
plot_paths = export_moe_history_plots(save_dir)

for p in plot_paths:
    print(f"图表已保存: {p}")
```

### 7.4 模型剪枝

```python
from ultralytics.nn.modules.moe.pruning import MoEPruner

pruner = MoEPruner(
    model_path="yolo-master-moe.pt",
    threshold=0.10,      # 更激进的剪枝
    dataset="coco128.yaml",
)

success = pruner.prune("yolo-master-moe-light.pt")
print(f"参数减少: {pruner.pruning_plan}")
```

---

## 8. 测试覆盖

MoE 诊断与分析系统的核心功能由 `tests/test_moe.py` 覆盖，关键测试项包括：

| 测试函数 | 验证点 |
|----------|--------|
| `test_aux_aggregation_no_double_count` | `aux_loss` 通过注册表聚合，避免嵌套模块重复计数 |
| `test_registry_no_leak_across_forwards` | 重复前向不会增长 `MOE_LOSS_REGISTRY` |
| `test_moe_snapshot_tensors_remain_on_source_device` | 快照张量保留在原始设备，不强制 CPU 同步 |
| `test_deepcopy_safe_after_forward` | 训练后 `copy.deepcopy` 安全（EMA/checkpoint 加载依赖） |
| `test_routing_gradient_flows_by_default` | `detach_routing=False` 时主任务梯度可达 router |
| `test_soft_balance_loss_grad_reaches_router` | Soft balancing 的梯度真实非零地流向 router logits |
| `test_eval_does_not_write_registry` | `eval()` 模式不污染全局注册表 |
| `test_gshard_balance_loss_uniform_equals_one` | GShard balance loss 在均衡时等于 1.0 |
| `test_gshard_balance_loss_collapsed_is_large` | 崩溃时 loss 放大为 `num_experts` 倍 |
| `test_es_moe_aux_on_gshard_scale` | ES_MOE aux loss 处于 O(1) 量级而非旧版 O(1e-3) |

---

## 附录 A：相关配置参数

在训练 YAML 或命令行中，以下参数控制 MoE 诊断与训练行为：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `moe_balance_loss` | float | `1.0` | Balance loss 系数 |
| `moe_router_z_loss` | float | `1.0` | Router Z-loss 系数 |
| `moe_noise_std` | float | `0.5` | Router 探索噪声标准差 |
| `moe_temperature` | float | `1.0` | Router softmax temperature |
| `moe_weight_threshold` | float | `0.01` | 条件计算权重阈值 |
| `moe_collapse_threshold` | float | `0.8` | 路由崩溃判定阈值 |

---

## 附录 B：文件索引

```
ultralytics/nn/modules/moe/
├── __init__.py          # 公共 API 导出
├── diagnostics.py       # MoELayerDiagnostic, collect_moe_diagnostics, format_moe_diagnostics
├── analysis.py          # ExpertUsageTracker, RoutingCollapseDetector, diagnose_model
├── history.py           # MoEDiagnosticsRecorder, export_moe_history_plots
├── pruning.py           # MoEPruner, prune_moe_model
├── loss.py              # MoELoss, gshard_balance_loss, differentiable_balance_loss
├── scheduler.py         # MoEDynamicScheduler, compute_gini
├── modules.py           # UltraOptimizedMoE, ES_MOE, OptimizedMOE 等（含 _record_moe_snapshot）
├── utils.py             # FlopsUtils, BatchedExpertComputation, is_core_moe_block
├── experts.py           # 各类 Expert 实现
└── routers.py           # 各类 Router 实现
```

---

*文档版本: v1.0 | 基于 YOLO-Master MoE 模块代码（截至 2026-06-25 审计修复版本）*
