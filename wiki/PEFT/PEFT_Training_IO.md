# PEFT_Training_IO 技术文档

> **Version**: YOLO-Master v260703  
> **Scope**: PEFT / LoRA / MoLoRA 训练配置、模型适配、策略调度与 Checkpoint I/O  
> **Lang**: 中文撰写，技术术语保留英文原词（如 `MoE`, `PEFT`, `LoRA`, `TensorRT` 等）

---

## 1. 概述

`PEFT_Training_IO` 是 YOLO-Master 的参数高效微调（Parameter-Efficient Fine-Tuning, PEFT）子系统的核心入口。它负责：

- **配置解析**：将用户参数（CLI / YAML / Python API）统一转换为结构化 `LoRAConfig` / `MoLoRAConfig`。
- **模型适配**：通过 `apply_lora()` 或 `get_peft_molora_model()` 将 LoRA / MoLoRA 注入到 `DetectionModel` 中，同时保持 Ultralytics 原有的 `nn.Sequential` 语义。
- **训练策略**：提供 `LoraTrainingStrategy` 实现 Layer-wise LR Decay、Alpha Warmup、Orthogonal Regularization、Dynamic Dropout Scheduling 等高级策略。
- **I/O 与 Checkpoint**：提供 `save_lora_adapters()` / `load_lora_adapters()` / `merge_lora_weights()` 完成 adapter 权重的持久化、恢复与合并。
- **架构感知规划**：`PEFTPlanner` 基于 `ArchitectureFingerprint` 对模型进行架构指纹识别，自动决策 ACCEPT / REFUSE / ADAPT。

---

## 2. 模块架构

```text
ultralytics/utils/lora/
├── __init__.py          # 公共 API 聚合
├── api.py               # apply_lora、参数分组、运行时校验、状态字典加载
├── config.py            # LoRAConfig、LoRAConfigBuilder
├── fallback.py          # Fallback backend：ManualLoRAConv、FewShotLoRAConv、PeftProxy
├── io.py                # save/load/merge adapters
├── planner.py           # PEFTPlanner、ArchitectureFingerprint、LOVO 验证
└── training.py          # LoraTrainingStrategy、训练统计、suggest_lora_config_for_dataset

ultralytics/nn/peft/molora/
├── __init__.py
├── config.py            # MoLoRAConfig、MoLoRAConfigBuilder
├── layer.py             # MoLoRALayer、MoLoRAExpert
├── loss.py              # MoLoRA 辅助损失
├── model.py             # get_peft_molora_model、MoLoRAModel
├── router.py            # LinearRouter、SpatialRouter、HybridRouter
└── utils.py             # 参数统计、Domain Expert 分配
```

---

## 3. 配置系统

### 3.1 LoRAConfig

`LoRAConfig` 是标准的 dataclass，承载所有 LoRA 相关的超参数。字段超过 60 个，以下为训练 I/O 中最常用的核心字段。

```python
from dataclasses import dataclass
from typing import Optional, List, Union

@dataclass
class LoRAConfig:
    # ---- Core ----
    r: int = 0                          # LoRA Rank，0 表示禁用
    alpha: int = 32                     # 缩放因子 alpha
    dropout: float = 0.05               # LoRA dropout
    bias: str = "none"                  # "none" | "all" | "lora_only"
    backend: str = "auto"               # "auto" | "peft" | "fallback"
    variant: str = "lora"               # "lora" | "loha" | "dora" | ...
    peft_type: str = "lora"             # PEFT 变体类型

    # ---- Target Selection ----
    include_head: bool = False          # 是否包含检测头
    include_moe: bool = True            # 是否包含 MoE 层
    include_attention: bool = False     # 是否包含 Attention 层
    only_backbone: bool = False         # 仅适配 backbone
    exclude_modules: Optional[List[str]] = None
    target_modules: Optional[List[str]] = None

    # ---- Strategy ----
    lr_mult: float = 1.0                # LoRA 参数 LR 乘数
    layer_decay: float = 0.0            # Layer-wise LR decay 率
    alpha_warmup: int = 0               # Alpha warmup epoch 数
    ortho_weight: float = 0.0           # 正交正则化权重
    dropout_end: float = 0.15           # 动态 dropout 终点
    dropout_start_ratio: float = 0.3    # 动态 dropout 启动比例

    # ---- Advanced ----
    use_dora: bool = False              # 启用 DoRA
    use_rslora: bool = True             # Rank-Stabilized LoRA
    init_lora_weights: Union[str, bool] = True   # "gaussian" / "pissa" / "olora"
    gradient_checkpointing: bool = False
    freeze_bn: bool = False             # 训练时冻结 BatchNorm

    # ---- Few-Shot ----
    few_shot_mode: bool = False
    few_shot_adaptive_rank: bool = True
    few_shot_dropconnect: float = 0.1
    ...
```

**关键方法**

| 方法 | 签名 | 说明 |
|------|------|------|
| `from_args` | `classmethod from_args(cls, args=None, **kwargs) -> LoRAConfig` | 从 Ultralytics `args` 或 `kwargs` 自动映射 `lora_*` 前缀字段。 |
| `__post_init__` | — | 对 `kernels`、`exclude_modules`、`target_modules` 做字符串拆分与类型标准化；校验 `few_shot_*` 的合法性。 |

### 3.2 LoRAConfigBuilder

`LoRAConfigBuilder` 是配置工厂，负责**自动目标检测**与**PEFT Config 生成**。

#### `auto_detect_targets`

```python
@staticmethod
def auto_detect_targets(
    model: nn.Module,
    r: int,
    include_moe: bool = True,
    include_attention: bool = False,
    only_backbone: bool = False,
    exclude_modules: Optional[List[str]] = None,
    layer_from: Optional[int] = None,
    layer_to: Optional[int] = None,
    last_n: Optional[int] = None,
    allow_depthwise: bool = False,
    kernels: Optional[List[int]] = None,
    skip_stem: bool = False,
    min_channels: int = 0,
    planner_enabled: bool = False,
    **kwargs,
) -> List[str]:
```

**参数说明**

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model` | `nn.Module` | — | 待扫描的模型（通常为 `model.model`，即 `nn.Sequential`）。 |
| `r` | `int` | — | LoRA rank，用于校验 grouped conv 的整除性。 |
| `include_moe` | `bool` | `True` | 是否将含 `expert` / `moe` 的模块纳入目标。 |
| `include_attention` | `bool` | `False` | 是否纳入 Attention 层；YOLO12 的 `AAttn` 默认排除以防数值崩溃。 |
| `only_backbone` | `bool` | `False` | 为 `True` 时跳过 head / detect / bbox / cls 等模块。 |
| `exclude_modules` | `List[str]` | `None` | 显式排除的模块名列表。 |
| `layer_from` / `layer_to` | `int` | `None` | 按顶层索引截取适配区间。 |
| `last_n` | `int` | `None` | 仅适配最后 N 个顶层 block。 |
| `allow_depthwise` | `bool` | `False` | 是否允许对 depthwise conv 注入 LoRA。 |
| `kernels` | `List[int]` | `None` | 仅对指定 kernel size（如 `[3]`）的卷积进行适配。 |
| `skip_stem` | `bool` | `False` | 跳过前 3 个顶层 backbone 层（stem 通常不需要 LoRA）。 |
| `min_channels` | `int` | `0` | 跳过 `min(in, out)` 小于该值的层，避免在过窄层上浪费容量。 |
| `planner_enabled` | `bool` | `False` | 启用 `PEFTPlanner` 提供的 `target_modules_hint` 预过滤。 |

**过滤规则（代码级）**

1. 显式排除 (`exclude_modules`)。
2. Planner hint 过滤（若启用）。
3. 索引范围过滤 (`last_n`, `from_layer`, `to_layer`)。
4. `skip_stem` 排除前 3 层。
5. 仅保留 `nn.Conv2d` 或 `nn.Linear`。
6. `min_channels` 阈值过滤。
7. `only_backbone` 排除 head 关键词。
8. Grouped / Depthwise Conv 的 rank 整除校验：`r % groups == 0`。
9. Kernel size 过滤 (`only_3x3`, `kernels`)。
10. 语义关键词排除：`dfl`, `score_head`, `bbox_head`, `sampling_offsets`, `attention_weights`, YOLO12 `AAttn` 的 `qkv/proj/pe` 及 `ABlock` 内部 MLP。

#### `create_config`

```python
@staticmethod
def create_config(
    model: nn.Module,
    r: int = 16,
    alpha: Optional[int] = None,
    auto_r_ratio: float = 0.0,
    peft_type: str = "lora",
    **kwargs
) -> Union['LoraConfig', 'LoHaConfig', 'LoKrConfig', 'IA3Config', 'OFTConfig', 'BOFTConfig', 'HRAConfig', None]:
```

内部逻辑：

1. 调用 `auto_detect_targets` 获取 `valid_targets`。
2. 若用户传了 `target_modules`，则取交集 `_filter_target_modules(valid_targets, user_targets)`。
3. 针对 `adalora` 预过滤仅保留 `nn.Linear`（PEFT 0.18 限制）。
4. `auto_r_ratio > 0` 时调用 `calculate_auto_rank` 自动估算 rank。
5. 根据 `peft_type` 分发到具体的 PEFT Config 类（`LoraConfig`, `LoHaConfig`, `OFTConfig`, `BOFTConfig`, `HRAConfig`, `AdaLoraConfig`, `IA3Config`）。
6. 对 `init_lora_weights` 做 Conv2d 兼容性校验（`_validate_peft_init_compatibility`）。

### 3.3 MoLoRAConfig

`MoLoRAConfig` 位于 `ultralytics/nn/peft/molora/config.py`，是 MoE + LoRA 的专用配置。

```python
@dataclass
class MoLoRAConfig:
    r: int = 8
    alpha: int = 16
    num_experts: int = 4           # 专家数量 E
    top_k: int = 2                 # 每次前向选择的专家数 K
    router_type: str = "linear"    # "linear" | "spatial" | "hybrid"
    dropout: float = 0.05
    use_rslora: bool = True
    balance_loss_coef: float = 0.01
    z_loss_coef: float = 0.01
    diversity_loss_coef: float = 0.001
    expert_init: str = "default"
    capacity_factor: float = 1.0
    expert_dropout: float = 0.0
    top_k_warmup: bool = False
    warmup_steps: int = 0
    share_moe_registry: bool = True
    domain_experts: Optional[Dict[str, List[int]]] = None
```

---

## 4. 模型适配与包装

### 4.1 `apply_lora` — 统一适配入口

```python
def apply_lora(
    model: "DetectionModel",
    args=None,
    **kwargs
) -> "DetectionModel":
```

**执行流程**

1. **防重入**：若 `model.lora_enabled` 已为 `True`，则跳过。
2. **配置构建**：`LoRAConfig.from_args(args, **kwargs)`；支持 `sensitivity_data_loader` 注入（用于梯度敏感探测）。
3. **Few-Shot 模式**：自动提升 rank、降低 dropout、增大 `lr_mult`。
4. **Planner 决策**（可选）：若启用 `planner_enabled`，调用 `PEFTPlanner.plan()`，可能返回 REFUSE（回退到 Full-SFT）或 ADAPT（覆盖 rank/variant）。
5. **Backend 选择**：`select_lora_backend()` 在 `peft` 与 `fallback` 之间做决策。
6. **架构安全守卫**：
   - **RT-DETR**：强制 `alpha_warmup >= 3`、`lora_lr_mult <= 1.0`、启用 safe attention projection、禁止 DoRA（除非 `allow_rtdetr_dora=True`）。
   - **YOLO12 Area-Attention**：强制排除 `attn.{qkv,proj,pe}` 与 `ABlock` 内部 MLP，限制 `lr_mult <= 1.0`。
7. **目标模块检测与过滤**：
   - 先运行 `auto_detect_targets` 获取全部合法层。
   - 若用户指定 `target_modules`，再做交集过滤。
   - 自动排除 grouped conv 不兼容层（`r % groups != 0`）。
8. **PEFT 包装**：
   - `get_peft_model(model.model, peft_config)` 生成 `PeftModel`。
   - 将 `PeftModel` 的 `__class__` 替换为 `PeftProxy`，使其支持 `__getitem__` / `__len__` / `__iter__`，兼容 YOLO 的 Sequential 语义。
   - 顶层 `DetectionModel` 通过 `_wrap_top_level_lora_model()` 替换为 `LoRADetectionModelWrapper`，并设置 `lora_enabled`、`lora_backend`、`lora_target_modules`、`lora_target_audit`、`lora_runtime_metadata` 等标志。
9. **检测头解冻**：`_unfreeze_detection_head(model)` 确保 head 参数可训练（否则 mAP 会为 0）。
10. **BatchNorm 冻结**：若 `freeze_bn=True`，调用 `_freeze_batchnorm_layers()`。
11. **Gradient Checkpointing**：若开启，递归地在 `C3k2` / `C2f` / `Bottleneck` 等 block 上启用。
12. **参数统计打印**：`_print_param_stats()` 输出 trainable / frozen / adapter 参数数量及显存占用。

### 4.2 `get_peft_molora_model` — MoLoRA 适配入口

```python
def get_peft_molora_model(
    model: nn.Module,
    config: Union[MoLoRAConfig, Dict[str, Any]],
) -> nn.Module:
```

**执行流程**

1. 防重入检查 `molora_enabled`。
2. 若未指定 `target_modules`，自动调用 `MoLoRAConfigBuilder.auto_detect_targets()`。
3. 遍历目标模块，将 `nn.Conv2d` / `nn.Linear` 原地替换为 `MoLoRALayer`。
4. 调用 `mark_only_molora_as_trainable(model)` 冻结所有非 MoLoRA 参数。
5. 附加 `molora_config` 与 `molora_enabled` 标记。

### 4.3 包装类详解

#### `PeftProxy` (`fallback.py` + `api.py`)

```python
class PeftProxy(PeftModel):
    def _get_base(self) -> nn.Module: ...
    def forward(self, x, *args, **kwargs): ...
    def __getitem__(self, idx: Union[int, slice]): ...
    def __len__(self) -> int: ...
    def __iter__(self): ...
    def children(self): ...
    def named_children(self): ...
    def state_dict(self, *args, **kwargs): ...
    def fuse(self, verbose: bool = True): ...
```

- `_get_base()` 会缓存底层 `nn.Sequential`，避免每次前向都重新遍历 wrapper 链。
- `__getitem__` 支持 `model[0]`、`model[2:5]` 等切片操作，这是 YOLO 内部多处依赖的语义。
- `fuse()` 被拦截并阻止，防止训练/验证期间意外合并 adapter。

#### `LoRADetectionModel`

一个 Mixin，仅做两件事：

- 设置 `lora_enabled = True`。
- 重写 `fuse()` 为空操作，避免 Ultralytics 默认的 `Conv + BN` 融合破坏 LoRA 结构。

具体包装子类：

```python
class LoRADetectionModelWrapper(LoRADetectionModel, DetectionModel): pass
class LoRASegmentationModelWrapper(LoRADetectionModel, SegmentationModel): pass
class LoRAPoseModelWrapper(LoRADetectionModel, PoseModel): pass
class LoRAClassificationModelWrapper(LoRADetectionModel, ClassificationModel): pass
class LoRAOBBModelWrapper(LoRADetectionModel, OBBModel): pass
class LoRARTDETRDetectionModelWrapper(LoRADetectionModel, RTDETRDetectionModel): pass
class LoRAWorldModelWrapper(LoRADetectionModel, WorldModel): pass
```

### 4.4 Fallback Backend

当 `backend="fallback"` 或 PEFT 库不可用时，系统退化为内部实现的 manual LoRA。

#### `ManualLoRAConv`

```python
class ManualLoRAConv(nn.Module):
    def __init__(self, conv: nn.Conv2d, r: int = 8, alpha: int = 16, dropout: float = 0.0): ...
```

- 支持 dense Conv2d (`groups=1`) 与 grouped Conv2d (`groups>1`)。
- `lora_A`: `(groups, in_per_group * kH * kW, r_per_group)`
- `lora_B`: `(groups, out_per_group, r_per_group)`
- 前向时通过 `nn.functional.unfold` 提取 patch，再执行 `bmm` 计算低秩增量。
- 兼容旧版 checkpoint（无 `groups` 维度的 2D 矩阵）。

#### `FewShotLoRAConv`

继承并扩展了 `ManualLoRAConv`：

- **Scheduled DropConnect**：支持 `constant` / `linear` / `cosine` / `exponential` 四种调度。
- **Gradient-Importance Weighted DropConnect (GIW-DC)**：基于 Fisher 信息（梯度平方 EMA）动态调整每个连接的丢弃概率。
- **Variational Rank Selection**：通过 Gumbel-Softmax 学习稀疏 rank mask。
- **Knowledge Distillation**：支持传入 `teacher_features` 计算 MSE alignment loss。
- **1x1 Conv 短路优化**：对于 `1x1` 且 `padding=0` 的卷积，直接用 `view` 代替昂贵的 `unfold`。

---

## 5. 训练策略 (`LoraTrainingStrategy`)

```python
class LoraTrainingStrategy:
    def __init__(self, model, config=None, epochs=100): ...
```

所有策略均以 **静态方法** 或 **实例方法** 提供，可直接在 Trainer 的 `on_train_epoch_start` / `optimizer_step` 等 hook 中调用。

### 5.1 Layer-wise LR Decay

```python
@staticmethod
def get_layer_decay_factors(
    model, total_layers=None, decay_rate=0.85
) -> Dict[str, float]:

# 应用至 optimizer
def apply_layer_decay_to_optimizer(self, optimizer, decay_rate=0.85) -> int:
```

- **原理**：浅层（小索引）获得更高 LR，深层按指数衰减 `decay_rate ** normalized_depth`。
- **自动探测 `total_layers`**：优先从 `model.model` 或 `base_model` 的 `__len__` 获取顶层 block 数（YOLO 通常为 ~23）。
- **防错**：
  - `decay_rate <= 0` 或 `> 1` 直接跳过。
  - `decay_rate < 0.5` 发出警告（深层 LR 趋近于 0，易导致 adapter under-training）。
- **分组精度**：`round(factor, 1)`，将 ~18 个层压缩为 3-5 个 `param_group`，避免 optimizer 过度膨胀。
- **返回值**：被调整 LR 的参数数量。

### 5.2 Alpha Warmup

```python
def prepare_alpha_warmup(self): ...
def step_alpha_warmup(self, epoch, warmup_epochs=5) -> float: ...
def finalize_alpha_warmup(self): ...
```

- **原理**：在 warmup 阶段将 `lora_alpha`（或 `scaling`）从 0 逐渐增加至目标值，使用 **cosine ease-in** 曲线。
- **多版本 PEFT 兼容**：内部实现 5 条路径（Path A-E）：
  - **Path A**：PEFT >= 0.18 的 `scaling` dict（`{'default': value}`）。
  - **Path B**：旧版 PEFT 的直接数字 `lora_alpha`。
  - **Path C**：`lora_alpha` 为 property，尝试写回，失败则降级到 Path D。
  - **Path D**：直接写 `scaling` 数值。
  - **Path E**：`peft_config` dict。
- **关键安全机制**：若 `prepare_alpha_warmup` 失败且模型包含 YOLO12 `AAttn` / `A2C2f`，则报错并标记 `_warmup_required = True`（训练极易崩溃）。

### 5.3 Orthogonal Regularization

```python
@staticmethod
def compute_orthogonal_loss(model, weight=1e-4) -> torch.Tensor:
```

- **公式**：
  $$
  \mathcal{L}_{ortho} = \lambda \cdot \frac{1}{N} \sum \left( \|A^T A - I\|_F + \|B^T B - I\|_F \right)
  $$
- **实现细节**：
  - 不 `detach()` weight tensor，保证梯度可回传。
  - 支持 PEFT >= 0.18 的 `nn.ModuleDict` 形式 `lora_A` / `lora_B`。
  - 自动处理多维 weight reshape。

### 5.4 Dynamic Dropout Scheduling

```python
@staticmethod
def update_dropout_schedule(
    model, epoch, epochs_total,
    start_dropout=0.0, end_dropout=0.15,
    schedule_start_ratio=0.3
) -> int:
```

- **调度逻辑**：
  - 前 `schedule_start_ratio * epochs_total` 个 epoch 保持 `start_dropout`。
  - 之后线性插值到 `end_dropout`。
- **缓存机制**：`_last_dropout_value` 避免无意义的重复设置。
- **返回值**：实际被修改的 dropout layer 数量。

### 5.5 训练统计

```python
def get_lora_training_stats(
    model, svd_sample_ratio: float = 0.2, svd_max_layers: int = 20
) -> Dict[str, Any]:
```

返回字典包含：

| 字段 | 说明 |
|------|------|
| `lora_enabled` | 是否启用 LoRA |
| `total_params` / `trainable_params` / `frozen_params` | 参数计数 |
| `lora_params` | adapter 参数量 |
| `lora_modules` | LoRA 层数 |
| `effective_rank_avg` | 基于 SVD 的有效 rank 占比（采样，非全量） |
| `norm_A_frobenius` / `norm_B_frobenius` | A/B 矩阵的平均 Frobenius 范数 |

### 5.6 配置建议

```python
def suggest_lora_config_for_dataset(
    num_images: Optional[int] = None,
    num_classes: Optional[int] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
) -> Dict[str, Any]:
```

基于项目实验数据给出经验性推荐：

| 数据规模 | `lora_r` | `lora_alpha` | 备注 |
|----------|----------|--------------|------|
| `< 500` 张 | `32` | `64` | 小数据集 LoRA 通常弱于 Full SFT，建议对比 baseline |
| `500 ~ 5K` | `32` | `64` | 开启 Orthogonal Regularization |
| `5K ~ 20K` | `16` | `32` | LoRA 通常可持平或超越 Full SFT |
| `> 20K` | `16` | `32` | 大场景推荐 LoRA 或 DoRA |

---

## 6. 输入输出与 Checkpoint 管理

### 6.1 `save_lora_adapters`

```python
def save_lora_adapters(
    model: "DetectionModel",
    path: Union[str, Path]
) -> bool:
```

**行为**

- 自动解包 DDP（`model.module`）。
- 若 `lora_backend == "fallback"`：
  - 调用 `_collect_fallback_adapter_state()` 收集 `ManualLoRAConv` / `FewShotLoRAConv` 的 `lora_A` / `lora_B`。
  - 写入 `fallback_adapter.pt`（权重）与 `fallback_meta.json`（元数据）。
  - 为兼容旧 loader，若目录中不存在 `adapter_config.json`，则同时写入一份。
- 若 `lora_backend == "peft"`：
  - 调用 `model.model.save_pretrained(path)`（PEFT 原生，仅保存 adapter）。
  - 额外写入 `runtime_metadata.json`，记录 `backend`、`variant`、`freeze_bn`、`target_modules`、`target_audit` 等运行时信息。

### 6.2 `load_lora_adapters`

```python
def load_lora_adapters(
    model: "DetectionModel",
    path: Union[str, Path],
    merge: bool = False,
    force_replace: bool = False,
    trainable: bool = False,
) -> bool:
```

**行为**

- 先尝试读取 `fallback_meta.json`，再回退到 `adapter_config.json` 并检查 `backend == "fallback"`。
- **Fallback 路径**：调用 `_load_fallback_adapter_state()` 重建 `ManualLoRAConv` / `FewShotLoRAConv`。
- **PEFT 路径**：
  - `PeftModel.from_pretrained(model.model, path, is_trainable=trainable)` 加载 adapter。
  - 替换 `__class__ = PeftProxy`。
  - 读取 `runtime_metadata.json` 恢复 `lora_backend`、`lora_target_modules` 等标记。
  - 调用 `_validate_lora_runtime_model()` 做运行时假设校验（`len()`、`index access`、`children()`、adapter 参数非空）。
- 若 `merge=True`，加载成功后立即调用 `merge_lora_weights()`。

### 6.3 `merge_lora_weights`

```python
def merge_lora_weights(model: "DetectionModel") -> bool:
```

**Fallback 路径**

- 遍历 `ManualLoRAConv` / `FewShotLoRAConv`，调用 `_merge_manual_lora_conv()`：
  - 计算 `delta = B @ A^T`，reshape 为 Conv2d weight 形状。
  - `conv.weight += delta * scaling`。
  - 将 wrapper 替换回原始 `nn.Conv2d`。
- 恢复原始类（`model.__class__ = lora_original_class`）。
- 调用 `_clear_lora_runtime_state()` 清除所有 LoRA 标记。

**PEFT 路径**

- 调用 `model.model.merge_and_unload()` 得到干净的 `nn.Sequential`。
- 通过 MRO 探测原始类（`_find_original_model_class()`），恢复 `model.__class__`。
- 清除 LoRA 运行时状态。

### 6.4 MoLoRA Checkpoint

`MoLoRAModel` 提供独立的 checkpoint 语义：

```python
class MoLoRAModel(nn.Module):
    def save_checkpoint(self, path: str) -> None: ...   # 仅保存含 "lora_A", "lora_B", "router", "molora" 的键
    def load_checkpoint(self, path: str) -> None: ...   # strict=False 加载
    def save_expert_replay_buffer(self, domain: str, path=None) -> Dict[str, Any]: ...
    def load_expert_replay_buffer(self, buffer, domain=None) -> None: ...
```

- `save_expert_replay_buffer` / `load_expert_replay_buffer` 用于**持续学习**场景，防止 domain 切换时的灾难性遗忘。
- `merge()` / `unmerge()` 可在推理前将 MoLoRA 权重合并到基卷积，推理后恢复。

---

## 7. PEFT Planner（架构感知规划）

### 7.1 `ArchitectureFingerprint`

10 维架构指纹，用于回归模型：

```python
@dataclass
class ArchitectureFingerprint:
    phi_attn: float = 0.0      # Attention 模块占比
    phi_text: float = 0.0      # Text-fusion 模块占比
    phi_dw: float = 0.0        # Depthwise conv 占比
    phi_group: float = 0.0     # Grouped conv 占比
    phi_linear: float = 0.0    # Linear 层占比
    phi_depth: float = 0.0     # 归一化深度（top-level blocks / 30）
    phi_width: float = 0.0     # 对数平均 channel 宽度
    phi_head: float = 0.0      # Head 参数量占比
    phi_residual: float = 0.0  # 残差模块密度
    phi_norm: float = 0.0      # LayerNorm 占比
```

- 使用 `weakref.WeakKeyDictionary` 缓存指纹，自动随模型 GC 失效。
- 探测规则按优先级：`RT-DETR` > `YOLO-World` > `YOLO12` > `YOLO-Master-MoE` > `YOLO-CNN`。

### 7.2 `PEFTPlanner`

```python
class PEFTPlanner:
    def plan(self, model: nn.Module, config: LoRAConfig) -> PlacementDecision: ...
    def predict(self, fingerprint: ArchitectureFingerprint, variant: str, rank: int) -> float: ...
    def fit(self, history: List[Tuple[ArchitectureFingerprint, str, float]]) -> None: ...
```

**回归模型**（Eq. 1）：

$$
\Delta mAP \approx \beta_0 + \beta_1 \phi_{attn} + \beta_2 \phi_{text} + \beta_3 \phi_{dw} + \beta_4 \xi_p
$$

其中 $\xi_p$ 为变体系数（`PEFTVariantProfile.xi`）。

**决策状态**

| 状态 | 含义 | 下游行为 |
|------|------|----------|
| `ACCEPT` | 预测 ΔmAP 安全，继续 | 正常训练 |
| `REFUSE` | 预测灾难性下降 | 回退到 Full-SFT（`return model`，不注入 LoRA） |
| `ADAPT` | 需调整 variant/rank 或 safety override | 修改 `config` 后继续 |

### 7.3 LOVO 验证

`LOVOValidator` 提供 Leave-One-Variant-Out 交叉验证：

```python
validator = LOVOValidator(threshold=-0.05)
result = validator.cross_validate(data_points)   # 返回 LOVOValidationResult
report = validator.full_report(collector)        # 综合 catastrophe detection 与 decision boundary 指标
```

---

## 8. 安全机制与兼容性

### 8.1 RT-DETR 安全守卫 (`_apply_rtdetr_lora_safety`)

| 守卫项 | 行为 |
|--------|------|
| `alpha_warmup` | 强制 `>= 3` |
| `lora_lr_mult` | 强制上限 `1.0` |
| `include_attention` | 强制启用 safe attention projection |
| `use_dora` | 默认降级为 plain LoRA（除非 `allow_rtdetr_dora=True`） |

### 8.2 YOLO12 Area-Attention 安全守卫

- 自动排除 `.attn.(qkv|proj|pe)` 与 `ABlock` 内部 `mlp` Conv2d。
- 强制 `alpha_warmup >= 3` 与 `lora_lr_mult <= 1.0`。
- 直接修改 `args`、`config`、`kwargs` 三处，确保 Trainer 读取到一致的值。

### 8.3 状态字典兼容性 (`load_lora_compatible_state_dict`)

```python
def load_lora_compatible_state_dict(
    model: nn.Module,
    source_state: Dict[str, torch.Tensor],
    context: str = "LoRA checkpoint",
) -> Dict[str, int]:
```

- 对 base weight 做宽松匹配（跳过 shape 不匹配的键）。
- 对 **adapter weight** 做严格匹配：若出现 shape mismatch、missing、unexpected adapter key，则抛出 `RuntimeError`，防止训练 resume 时 adapter 拓扑不一致导致静默重初始化。

### 8.4 Backend 选择逻辑 (`select_lora_backend`)

```python
def select_lora_backend(
    config: LoRAConfig,
    peft_available: bool,
    supports_peft: bool,
    supports_fallback: bool,
) -> Dict[str, str]:
```

- `requested_backend == "peft"`：必须满足 `peft_available and supports_peft`，否则报错。
- `requested_backend == "fallback"`：必须 `supports_fallback`，否则报错。
- `requested_backend == "auto"`：优先 PEFT；若 PEFT 不可用则报错（**不再静默回退 fallback**）。

---

## 9. 使用示例

### 9.1 标准 LoRA 训练

```python
from ultralytics import YOLO
from ultralytics.utils.lora import apply_lora, save_lora_adapters, merge_lora_weights
from ultralytics.utils.lora.training import LoraTrainingStrategy

# 加载基模型
model = YOLO("yolo11s.pt")

# 注入 LoRA（PEFT backend）
model.model = apply_lora(
    model.model,
    lora_r=16,
    lora_alpha=32,
    lora_target_modules=None,      # 自动检测
    lora_dropout=0.05,
    lora_layer_decay=0.9,
    lora_alpha_warmup=3,
    lora_ortho_weight=1e-4,
    lora_backend="auto",
)

# 训练（Ultralytics Trainer 会自动识别 LoRA 参数）
model.train(data="coco128.yaml", epochs=50, batch=16, lr0=0.01)

# 保存 adapter（仅保存 LoRA 权重）
save_lora_adapters(model.model, path="runs/lora/coco128_adapter")

# 合并并导出
merge_lora_weights(model.model)
model.export(format="onnx")
```

### 9.2 使用 Fallback Backend

```python
model.model = apply_lora(
    model.model,
    lora_r=8,
    lora_alpha=16,
    lora_backend="fallback",       # 显式使用内部实现
    lora_freeze_bn=True,
)
```

### 9.3 高级训练策略 Hook

```python
strategy = LoraTrainingStrategy(model.model, epochs=100)

# 1) 准备 alpha warmup
strategy.prepare_alpha_warmup()

# 2) 在 Trainer 的 on_train_epoch_start 中调用
for epoch in range(100):
    scale = strategy.step_alpha_warmup(epoch, warmup_epochs=5)
    updated = strategy.update_dropout_schedule(
        model.model, epoch, epochs_total=100,
        start_dropout=0.0, end_dropout=0.15, schedule_start_ratio=0.3
    )

# 3) 在 optimizer 构建后应用 layer decay
strategy.apply_layer_decay_to_optimizer(optimizer, decay_rate=0.85)

# 4) 在 loss 计算中加入正交正则
ortho_loss = LoraTrainingStrategy.compute_orthogonal_loss(model.model, weight=1e-4)
total_loss = detection_loss + ortho_loss

# 5) 训练结束恢复 alpha
strategy.finalize_alpha_warmup()
```

### 9.4 MoLoRA 训练

```python
from ultralytics.nn.peft.molora import get_peft_molora_model, MoLoRAConfig, MoLoRAModel

config = MoLoRAConfig(
    r=8, alpha=16, num_experts=4, top_k=2,
    router_type="linear",
    balance_loss_coef=0.01,
    z_loss_coef=0.01,
)

model = YOLO("yolo11s.pt")
get_peft_molora_model(model.model, config)

# 使用 MoLoRAModel 包装以方便管理 aux_loss 与 checkpoint
wrapped = MoLoRAModel(model.model, config)

# 训练循环中
det_loss = ...
aux_loss = wrapped.compute_aux_loss()
total_loss = det_loss + aux_loss

# 保存 / 加载
wrapped.save_checkpoint("molora_ckpt.pt")
wrapped.load_checkpoint("molora_ckpt.pt")

# 持续学习：保存 domain expert replay buffer
wrapped.save_expert_replay_buffer(domain="medical", path="medical_replay.pt")
```

### 9.5 PEFT Planner 决策

```python
from ultralytics.utils.lora.planner import PEFTPlanner
from ultralytics.utils.lora import LoRAConfig

planner = PEFTPlanner()
config = LoRAConfig(planner_enabled=True, lora_r=16, lora_type="lora")
decision = planner.plan(model.model, config)

print(decision.status)            # "ACCEPT" / "REFUSE" / "ADAPT"
print(decision.predicted_delta)   # 预测的 ΔmAP
if decision.status == "ADAPT":
    config.r = decision.recommended_rank
    config.peft_type = decision.recommended_variant
```

---

## 10. API 索引

### 10.1 I/O 函数

| 函数 | 文件 | 签名 |
|------|------|------|
| `save_lora_adapters` | `io.py` | `save_lora_adapters(model, path) -> bool` |
| `load_lora_adapters` | `io.py` | `load_lora_adapters(model, path, merge=False, force_replace=False, trainable=False) -> bool` |
| `merge_lora_weights` | `io.py` | `merge_lora_weights(model) -> bool` |
| `load_lora_compatible_state_dict` | `api.py` | `load_lora_compatible_state_dict(model, source_state, context="LoRA checkpoint") -> Dict[str, int]` |

### 10.2 适配函数

| 函数 | 文件 | 签名 |
|------|------|------|
| `apply_lora` | `api.py` | `apply_lora(model, args=None, **kwargs) -> DetectionModel` |
| `get_peft_molora_model` | `molora/model.py` | `get_peft_molora_model(model, config) -> nn.Module` |
| `apply_manual_lora` | `fallback.py` | `apply_manual_lora(model, config, include_head=False) -> nn.Module` |

### 10.3 策略与工具

| 类 / 函数 | 文件 | 说明 |
|-----------|------|------|
| `LoraTrainingStrategy` | `training.py` | Layer decay, Alpha warmup, Orthogonal loss, Dropout schedule |
| `get_lora_training_stats` | `training.py` | 训练监控统计 |
| `suggest_lora_config_for_dataset` | `training.py` | 基于数据规模给出超参数建议 |
| `get_lora_param_groups` | `api.py` | 将参数拆分为 LoRA / non-LoRA 两组，支持独立 weight decay |

### 10.4 Planner 类

| 类 | 文件 | 说明 |
|----|------|------|
| `ArchitectureFingerprint` | `planner.py` | 10 维架构指纹 |
| `PEFTPlanner` | `planner.py` | ACCEPT / REFUSE / ADAPT 决策器 |
| `PlacementDecision` | `planner.py` | 规划结果封装 |
| `LOVOValidator` | `planner.py` | Leave-One-Variant-Out 交叉验证 |

### 10.5 包装类

| 类 | 文件 | 说明 |
|----|------|------|
| `PeftProxy` | `fallback.py` | 兼容 `nn.Sequential` 语义的 PEFT 包装器 |
| `LoRADetectionModelWrapper` | `fallback.py` | 顶层模型 Mixin + 原始类多重继承 |
| `ManualLoRAConv` | `fallback.py` | Fallback 的 Conv2d LoRA 实现 |
| `FewShotLoRAConv` | `fallback.py` | 支持 DropConnect、Distillation、Variational Rank 的增强版 |
| `MoLoRAModel` | `molora/model.py` | MoLoRA 的便利包装，提供 merge/unmerge/checkpoint/replay buffer |

---

## 11. 常见问题 (FAQ)

**Q: 为什么加载 adapter checkpoint 时提示 adapter topology 不匹配？**  
A: `load_lora_compatible_state_dict` 对 adapter 键执行严格校验。请确保 `lora_r`、`lora_type`、`target_modules` 与训练时完全一致。

**Q: RT-DETR 使用 LoRA 训练后 mAP 为 0？**  
A: 请检查是否启用了 `_apply_rtdetr_lora_safety` 推荐的 `alpha_warmup >= 3`，并确保 detection head 已解冻（`_unfreeze_detection_head`）。

**Q: YOLO12 训练中期 loss 突降为 0 / NaN？**  
A: 这是 Area-Attention 的已知数值不稳定问题。请确认 `attn.qkv/proj/pe` 与 `ABlock` 内部 MLP 已被安全守卫排除，且 `alpha_warmup >= 3`。

**Q: `save_lora_adapters` 与 PyTorch `torch.save` 有什么区别？**  
A: `save_lora_adapters` 仅保存 adapter 权重（PEFT 原生机制或 fallback 的 `lora_A/B`），不保存 base model，因此体积极小（通常数 MB）。

**Q: 能否在已有 LoRA 模型上继续叠加新的 LoRA？**  
A: 目前不支持叠加。若 `lora_enabled=True`，`load_lora_adapters` 默认跳过；可通过 `force_replace=True` 替换现有 adapter。

---

*文档结束*
