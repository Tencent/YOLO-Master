# PEFT LoRA Core 技术文档

> 本文档覆盖 YOLO-Master 项目中 `ultralytics/utils/lora/` 与 `ultralytics/nn/peft/molora/` 两个核心模块，详细说明 PEFT (Parameter-Efficient Fine-Tuning) 与 LoRA (Low-Rank Adaptation) 的完整实现链路。

---

## 目录

1. [模块总览](#1-模块总览)
2. [配置系统：LoRAConfig 与 LoRAConfigBuilder](#2-配置系统loraconfig-与-loraconfigbuilder)
3. [核心 API：apply_lora](#3-核心-apiapply_lora)
4. [后端架构：PEFT vs Fallback](#4-后端架构peft-vs-fallback)
5. [训练策略：LoraTrainingStrategy](#5-训练策略loratrainingstrategy)
6. [IO 与生命周期管理](#6-io-与生命周期管理)
7. [PEFT Planner：架构条件化适配决策](#7-peft-planner架构条件化适配决策)
8. [MoLoRA：Mixture-of-LoRA](#8-moloramixture-of-lora)
9. [Few-Shot LoRA](#9-few-shot-lora)
10. [架构安全保护机制](#10-架构安全保护机制)
11. [使用示例](#11-使用示例)
12. [参考与依赖](#12-参考与依赖)

---

## 1. 模块总览

YOLO-Master 的 LoRA/PEFT 子系统采用 **双后端架构** 设计：

| 后端 | 依赖 | 适用场景 | 支持变体 |
|------|------|----------|----------|
| `peft` | `peft` 库 (HuggingFace) | 生产环境，完整功能 | LoRA, DoRA, LoHa, LoKr, AdaLoRA, IA³, OFT, BOFT, HRA |
| `fallback` | 无外部依赖 | 无 PEFT 时的降级保护 | 仅 LoRA (Conv2d) |

### 1.1 文件结构

```
ultralytics/utils/lora/
├── __init__.py          # 公共 API 聚合与导出
├── api.py               # 核心入口 apply_lora、工具函数
├── config.py            # LoRAConfig / LoRAConfigBuilder
├── training.py          # LoraTrainingStrategy (4 种训练策略)
├── io.py                # save / load / merge 适配器
├── fallback.py          # ManualLoRAConv、FewShotLoRAConv、PeftProxy
└── planner.py           # PEFTPlanner、ArchitectureFingerprint

ultralytics/nn/peft/molora/
├── __init__.py          # MoLoRA 公共 API
├── config.py            # MoLoRAConfig / MoLoRAConfigBuilder
├── layer.py             # MoLoRAExpert / MoLoRALayer
├── model.py             # get_peft_molora_model
├── router.py            # LinearRouter / SpatialRouter / HybridRouter
├── loss.py              # MoLoRALoss / compute_expert_usage
└── utils.py             # 参数统计与域分配工具
```

---

## 2. 配置系统：LoRAConfig 与 LoRAConfigBuilder

### 2.1 LoRAConfig

`LoRAConfig` 是一个 `@dataclass`，承载所有 LoRA 训练参数。支持通过 `from_args()` 从 Ultralytics 命令行参数映射构建。

#### 类签名

```python
@dataclass
class LoRAConfig:
    # ── 核心参数 ──
    r: int = 0                          # LoRA Rank，0 表示禁用
    alpha: int = 32                     # 缩放因子
    dropout: float = 0.05               # Dropout 率
    bias: str = "none"                  # "none" | "all" | "lora_only"
    backend: str = "auto"               # "auto" | "peft" | "fallback"
    variant: str = "lora"               # 适配器变体名
    include_head: bool = False          # 是否包含检测头
    freeze_bn: bool = False             # 是否冻结 BatchNorm

    # ── 策略控制 ──
    lr_mult: float = 1.0
    include_moe: bool = True
    include_attention: bool = False
    only_backbone: bool = False
    exclude_modules: Optional[List[str]] = None
    target_modules: Optional[List[str]] = None

    # ── 层过滤 ──
    last_n: Optional[int] = None
    from_layer: Optional[int] = None
    to_layer: Optional[int] = None

    # ── 卷积特定 ──
    allow_depthwise: bool = False
    kernels: Optional[List[int]] = None
    only_3x3: bool = False

    # ── 容量分配 ──
    skip_stem: bool = False             # 跳过前 3 个顶层 backbone 层
    min_channels: int = 0               # 跳过窄通道层

    # ── 高级选项 ──
    gradient_checkpointing: bool = False
    auto_r_ratio: float = 0.0           # 基于参数比例自动计算 rank
    use_dora: bool = False              # 启用 DoRA
    allow_rtdetr_dora: bool = False     # RT-DETR + DoRA 实验开关
    use_rslora: bool = True             # Rank-Stabilized LoRA
    init_lora_weights: Union[str, bool] = True  # True/False/"gaussian"/"pissa"/"olora"
    peft_type: str = "lora"             # "lora"|"loha"|"lokr"|"adalora"|"ia3"|"oft"|"boft"|"hra"
    quantization: str = "none"          # "none" | "4bit" | "8bit"

    # ── 训练策略参数 ──
    layer_decay: float = 0.0            # 层-wise LR 衰减
    alpha_warmup: int = 0               # Alpha 余弦预热 epoch 数
    ortho_weight: float = 0.0           # 正交正则化权重
    ortho_frequency: int = 10           # 正交损失计算间隔
    dropout_end: float = 0.15           # 动态 dropout 最终值
    dropout_start_ratio: float = 0.3    # 开始增加 dropout 的 epoch 比例

    # ── AdaLoRA 特定 ──
    target_r: int = 8
    init_r: int = 12
    tinit: int = 0
    tfinal: int = 0
    delta_t: int = 1
    beta1: float = 0.85
    beta2: float = 0.85
    orth_reg_weight: float = 0.5
    total_step: Optional[int] = None

    # ── OFT 特定 ──
    oft_block_size: int = 0
    oft_coft: bool = False
    oft_eps: float = 6e-5
    oft_block_share: bool = False

    # ── BOFT 特定 ──
    boft_block_size: int = 2
    boft_block_num: int = 0
    boft_n_butterfly_factor: int = 2

    # ── HRA 特定 ──
    hra_apply_gs: bool = False          # Gram-Schmidt 正交化

    # ── Few-Shot 模式 ──
    few_shot_mode: bool = False
    few_shot_teacher: Optional[str] = None
    few_shot_dropconnect: float = 0.1
    few_shot_distill_weight: float = 0.5
    few_shot_adaptive_rank: bool = True
    # ... (v3 增强详见第 9 节)

    # ── Planner 集成 ──
    planner_enabled: bool = False
```

#### 关键方法

```python
@classmethod
def from_args(cls, args=None, **kwargs) -> "LoRAConfig"
```

支持从 Ultralytics `args` 对象或 `kwargs` 自动映射 `lora_` 前缀参数。例如 `lora_r` → `r`，`lora_alpha` → `alpha`。

---

### 2.2 LoRAConfigBuilder

`LoRAConfigBuilder` 提供 **模型结构感知的目标层自动检测** 与 **最优配置生成** 能力。

#### 核心静态方法

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
) -> List[str]
```

**目标层检测逻辑（执行顺序）：**

1. **显式排除**：`exclude_modules` 中指定的模块名直接跳过。
2. **Planner Hint 过滤**：若 `planner_enabled=True`，使用 `PEFTPlanner` 提供的 `target_modules_hint` 预过滤。
3. **索引过滤**：根据 `last_n` / `from_layer` / `to_layer` 筛选顶层序号范围内的层。
4. **Stem 跳过**：`skip_stem=True` 时排除前 3 个顶层 backbone 层（低层特征提取 rarely benefits from LoRA）。
5. **类型过滤**：仅保留 `nn.Conv2d` 与 `nn.Linear`。
6. **通道过滤**：`min_channels > 0` 时跳过 `min(in, out) < min_channels` 的层。
7. **Backbone 过滤**：`only_backbone=True` 时排除检测头相关层（匹配 `head|detect|box|cls|pred` 等正则）。
8. **卷积特定检查**：
   - 分组卷积：`groups > 1` 时要求 `r % groups == 0`。
   - Depthwise：`allow_depthwise=False` 时跳过。
   - 核大小：`kernels` 白名单 / `only_3x3` 过滤。
9. **语义排除**：
   - `score_head` / `bbox_head`（RT-DETR 预测头）
   - `dfl`（Distribution Focal Loss 固定卷积）
   - `attn.{qkv,proj,pe}`（YOLO12 Area-Attention，除非 `include_attention=True`）
   - `MSDeformAttn` 的 `sampling_offsets` / `attention_weights`

```python
@staticmethod
def calculate_auto_rank(model: nn.Module, targets: List[str], ratio: float) -> int
```

启发式自动计算 Rank。近似公式：

```
LoRA_Params ≈ Num_Targets × Rank × (In_Ch + Out_Ch)
Rank = Target_Param_Budget / (Num_Targets × Avg_Dim)
```

结果 clamp 到 `[4, 128]` 并取最近的 4 的倍数。

```python
@staticmethod
def create_config(
    model: nn.Module,
    r: int = 16,
    alpha: Optional[int] = None,
    auto_r_ratio: float = 0.0,
    peft_type: str = "lora",
    **kwargs
) -> Union[LoraConfig, LoHaConfig, LoKrConfig, IA3Config, OFTConfig, BOFTConfig, HRAConfig, None]
```

工厂方法，根据 `peft_type` 分派生成对应的 PEFT Config 对象。

**分派规则：**

| `peft_type` | 生成配置类 | 特殊处理 |
|-------------|-----------|----------|
| `lora` (默认) | `LoraConfig` | 支持 DoRA (`use_dora=True`)、rsLoRA、初始化模式选择 |
| `loha` | `LoHaConfig` | — |
| `lokr` | `LoKrConfig` | — |
| `adalora` | `AdaLoraConfig` | 要求 `total_step > 0`；Conv2d 层会被静默跳过，自动降级警告 |
| `ia3` | `IA3Config` | 无 rank，仅缩放向量 |
| `oft` | `OFTConfig` | 忽略 `r`，使用 `oft_block_size`（默认 32）驱动容量 |
| `boft` | `BOFTConfig` | 预过滤不可整除的层；自动降级 `boft_block_size` |
| `hra` | `HRAConfig` | 支持 `apply_GS` 增强数值稳定性 |

---

## 3. 核心 API：apply_lora

`apply_lora()` 是整个 LoRA 子系统的 **唯一主入口**，负责将 LoRA 适配器应用到 Ultralytics `DetectionModel` 上。

### 3.1 函数签名

```python
def apply_lora(
    model: DetectionModel,
    args=None,
    **kwargs
) -> DetectionModel
```

### 3.2 执行流程

```
┌─────────────────────────────────────────────────────────────┐
│  0. 防重复应用                                              │
│     若 model.lora_enabled=True，直接返回原模型              │
├─────────────────────────────────────────────────────────────┤
│  1. 配置初始化                                              │
│     LoRAConfig.from_args(args, **kwargs)                    │
│     注入 sensitivity_data_loader（梯度敏感度探测）          │
├─────────────────────────────────────────────────────────────┤
│  2. Few-Shot 模式自适应调整                                 │
│     r=max(r,32), alpha=max(alpha,64), dropout≤0.02         │
│     lr_mult≥3.0                                             │
├─────────────────────────────────────────────────────────────┤
│  3. PEFT Planner 架构条件决策（opt-in）                     │
│     planner_enabled → PEFTPlanner.plan()                    │
│     ├─ REFUSE → 回退 Full-SFT                               │
│     ├─ ADAPT  → 覆盖 variant/rank/safety_overrides          │
│     └─ ACCEPT → 继续                                        │
├─────────────────────────────────────────────────────────────┤
│  4. 后端选择                                                │
│     select_lora_backend() → "peft" 或 "fallback"           │
│     fallback → apply_manual_lora() 直接返回                 │
├─────────────────────────────────────────────────────────────┤
│  5. 架构检测与安全保护                                      │
│     ├─ Auto-Disable MoE/Attention（若模型无对应组件）       │
│     ├─ RT-DETR Safety：alpha_warmup≥3, lr_mult≤1.0         │
│     └─ YOLO12 Area-Attention Safety：排除 attn 层           │
├─────────────────────────────────────────────────────────────┤
│  6. 目标层自动检测与过滤                                    │
│     auto_detect_targets() → valid_targets                   │
│     用户 target_modules 交集过滤 → final_targets            │
│     记录 target_audit（valid/selected/excluded 统计）        │
├─────────────────────────────────────────────────────────────┤
│  7. PEFT Config 生成                                        │
│     LoRAConfigBuilder.create_config()                       │
├─────────────────────────────────────────────────────────────┤
│  8. PEFT 包装                                               │
│     get_peft_model(model.model, peft_config)                │
│     └─ 类替换为 PeftProxy（支持 Sequential 语义）           │
│     顶层类替换为 LoRADetectionModelWrapper                  │
│     附加运行时元数据（backend/variant/targets/audit）        │
├─────────────────────────────────────────────────────────────┤
│  9. 运行时校验                                              │
│     _validate_lora_runtime_model()                          │
├─────────────────────────────────────────────────────────────┤
│  10. 异常降级（P0 Fix）                                     │
│     PEFT 失败 + auto backend → fallback manual LoRA        │
└─────────────────────────────────────────────────────────────┘
```

### 3.3 核心运行时属性

LoRA 应用后，`model` 上附加以下属性：

| 属性 | 类型 | 说明 |
|------|------|------|
| `model.lora_enabled` | `bool` | 是否已启用 LoRA |
| `model.lora_backend` | `str` | `"peft"` / `"fallback"` |
| `model.lora_variant` | `str` | 实际生效的变体名 |
| `model.lora_target_modules` | `List[str]` | 最终选中的目标模块名列表 |
| `model.lora_target_audit` | `Dict` | 目标选择审计记录 |
| `model.lora_runtime_metadata` | `Dict` | 完整运行时元数据 |
| `model.lora_include_head` | `bool` | 是否包含检测头 |
| `model.lora_freeze_bn` | `bool` | 是否冻结 BN |
| `model.lora_config` | `LoRAConfig` | 配置对象引用 |

---

## 4. 后端架构：PEFT vs Fallback

### 4.1 PEFT 后端

依赖外部 `peft` 库。使用 `get_peft_model()` 包装 `model.model`（`nn.Sequential`）。

#### PeftProxy 类

`PeftProxy` 是解决 PEFT 包装器与 Ultralytics `nn.Sequential` 语义不兼容的核心组件。

```python
class PeftProxy(PeftModel):
    def _get_base(self) -> nn.Module
    def forward(self, x, *args, **kwargs)
    def __getitem__(self, idx: Union[int, slice])   # 支持 model[i]
    def __len__(self) -> int                         # 支持 len(model)
    def __iter__(self)                               # 支持 for layer in model
    def children(self)
    def named_children(self)
    def fuse(self, verbose=True)                     # 拦截融合，保护 LoRA 结构
```

**关键优化**：`_get_base()` 使用 `_cached_base_model` 缓存底层模型引用，避免每次访问时重新遍历包装链，降低 1-2% 训练开销。

#### 顶层包装

```python
class LoRADetectionModelWrapper(LoRADetectionModel, DetectionModel): pass
class LoRASegmentationModelWrapper(LoRADetectionModel, SegmentationModel): pass
class LoRAPoseModelWrapper(LoRADetectionModel, PoseModel): pass
class LoRAClassificationModelWrapper(LoRADetectionModel, ClassificationModel): pass
class LoRAOBBModelWrapper(LoRADetectionModel, OBBModel): pass
class LoRARTDETRDetectionModelWrapper(LoRADetectionModel, RTDETRDetectionModel): pass
class LoRAWorldModelWrapper(LoRADetectionModel, WorldModel): pass
```

顶层类替换确保：
- `fuse()` 被拦截（防止训练/验证期间意外合并 LoRA 权重）
- `pickle` 序列化兼容性
- 保留原始模型类的所有行为

---

### 4.2 Fallback 后端

无外部依赖，使用 `ManualLoRAConv` 手动包装 `nn.Conv2d`。

#### ManualLoRAConv

```python
class ManualLoRAConv(nn.Module):
    def __init__(
        self,
        conv: nn.Conv2d,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0
    )
```

**前向传播逻辑：**

```
out = conv(x)                    # 冻结的基础卷积
x_unfold = F.unfold(x, kernel)   # (B, C_in*kH*kW, L)
lora = x_unfold @ A @ B^T        # 低秩更新
return out + lora * scaling
```

**分组卷积支持**：当 `groups > 1` 时，每组分配独立的 `(A_g, B_g)` 参数对，形状为 `(groups, in_per_group, r_per_group)` 和 `(groups, out_per_group, r_per_group)`。要求 `r % groups == 0`。

#### apply_manual_lora

```python
def apply_manual_lora(
    model: nn.Module,
    config: LoRAConfig,
    include_head: bool = False
) -> nn.Module
```

递归遍历 `model.model` 的所有子模块，将符合条件的 `nn.Conv2d` 替换为 `ManualLoRAConv` 或 `FewShotLoRAConv`。

---

## 5. 训练策略：LoraTrainingStrategy

`LoraTrainingStrategy` 提供 **4 种互补的高级训练策略**，用于提升 LoRA 微调稳定性与最终精度。

```python
class LoraTrainingStrategy:
    def __init__(self, model, config=None, epochs=100)
```

### 5.1 策略 1：Layer-wise LR Decay（层-wise 学习率衰减）

```python
@staticmethod
def get_layer_decay_factors(
    model,
    total_layers=None,
    decay_rate=0.85
) -> Dict[str, float]

def apply_layer_decay_to_optimizer(self, optimizer, decay_rate=0.85) -> int
```

**原理**：浅层 backbone 使用较高 LR，深层使用较低 LR。基于指数衰减：

```
factor = decay_rate ^ (layer_idx / total_layers)
```

**实现细节**：
- 自动检测 YOLO 顶层块数量（`len(model.model)` 或从参数名推断）。
- 将原单一 LoRA 参数组拆分为 **按 factor 分组的多参数组**（factor 取整到 1 位小数，通常 3-5 个组）。
- 设置 `initial_lr` 供 warmup scheduler 使用。

**Guardrails**：
- `decay_rate ≤ 0` 或 `> 1`：跳过并警告。
- `decay_rate < 0.5`：警告深层 LR 接近 0，可能导致 adapter under-training。
- 仅 1 个 LR 组时：警告分层失败。

### 5.2 策略 2：Alpha Warmup（Alpha 余弦预热）

```python
def prepare_alpha_warmup(self) -> bool
def step_alpha_warmup(self, epoch, warmup_epochs=5) -> float
def finalize_alpha_warmup(self)
```

**原理**：训练初期将 LoRA 的 effective alpha 设为 0（即 `scaling = 0`），随 epoch 增加按余弦曲线恢复到目标值：

```
scale = 0.5 * (1 - cos(π * epoch / warmup_epochs))
```

**多版本 PEFT 兼容路径**：

| PEFT 版本 | 控制方式 | 路径标识 |
|-----------|----------|----------|
| ≥ 0.18 | `scaling` dict (`{'default': float}`) | `scaling_dict` |
| < 0.13 | 直接数值 `lora_alpha` / `scaling` | `direct` / `scaling` |
| Property 只读 | 尝试写入后验证，失败则 fallback 到 `scaling` | `property` → `scaling_fallback` |
| `peft_config` dict | 直接修改 dict 中的 `lora_alpha` | `config_dict` |

**YOLO12 关键保护**：若检测到 `A2C2f` / `AAttn` 架构且 warmup 准备失败，会输出 **CRITICAL 错误**：该架构必须使用 alpha warmup 否则训练会 collapse（loss→0, mAP→0）。

### 5.3 策略 3：Orthogonal Regularization（正交正则化）

```python
@staticmethod
def compute_orthogonal_loss(model, weight=1e-4) -> torch.Tensor
```

**损失函数**：

```
L_ortho = λ × (Σ||A·A^T - I||_F + Σ||B^T·B - I||_F) / N_pairs
```

**作用**：惩罚 LoRA A/B 矩阵的非正交性，防止 rank collapse（A·B 退化为低有效秩乘积）。

**关键实现**：**不**对 weight tensor 调用 `.detach()`，保持梯度图完整，使正则化在 backward 中真正生效。

### 5.4 策略 4：Dynamic Dropout Scheduling（动态 Dropout 调度）

```python
@staticmethod
def update_dropout_schedule(
    model, epoch, epochs_total,
    start_dropout=0.0, end_dropout=0.15,
    schedule_start_ratio=0.3
) -> int
```

**调度策略**：
- 前期（`epoch < epochs_total * 0.3`）：`dropout = start_dropout`，保留梯度信号。
- 后期：线性插值至 `end_dropout`，作为正则化防止过拟合。

**优化**：使用类级缓存 `_last_dropout_value`，若当前值与上次相同则跳过更新，避免遍历所有模块。

---

## 6. IO 与生命周期管理

### 6.1 保存适配器

```python
def save_lora_adapters(model: DetectionModel, path: Union[str, Path]) -> bool
```

| 后端 | 保存内容 | 元数据文件 |
|------|----------|-----------|
| `peft` | `model.model.save_pretrained(path)` (仅 adapter 权重) | `runtime_metadata.json` |
| `fallback` | `fallback_adapter.pt` (含 `modules` + `state` dict) | `fallback_meta.json` + `adapter_config.json` (兼容 symlink) |

### 6.2 加载适配器

```python
def load_lora_adapters(
    model: DetectionModel,
    path: Union[str, Path],
    merge: bool = False,
    force_replace: bool = False,
    trainable: bool = False,
) -> bool
```

**流程**：
1. 探测 `fallback_meta.json` 或 `adapter_config.json` 判断后端类型。
2. 若模型已有 LoRA：
   - `force_replace=True`：先 merge_and_unload 或 clear state，再加载新适配器。
   - `force_replace=False`：跳过并警告。
3. Fallback 路径：调用 `_load_fallback_adapter_state()` 重建 `ManualLoRAConv` / `FewShotLoRAConv` 结构并加载权重。
4. PEFT 路径：调用 `PeftModel.from_pretrained()`，替换为 `PeftProxy`。

### 6.3 合并适配器

```python
def merge_lora_weights(model: DetectionModel) -> bool
```

将 LoRA 增量权重合并回基础模型，卸载适配器，恢复原始模型类。

**Fallback 合并**：

```python
def _merge_manual_lora_conv(module) -> nn.Conv2d
```

```
delta_per_group = B @ A^T                          # (out_per_group, in_per_group*kH*kW)
weight_delta = delta_per_group.reshape(out_c, in_c/groups, kH, kW)
merged_weight = conv.weight + weight_delta * scaling
```

**PEFT 合并**：

```python
merged_base = model.model.merge_and_unload()
model.model = merged_base
model.__class__ = _find_original_model_class(model)  # MRO 检测恢复原始类
```

### 6.4 兼容的 Checkpoint 加载

```python
def load_lora_compatible_state_dict(
    model: nn.Module,
    source_state: Dict[str, torch.Tensor],
    context: str = "LoRA checkpoint"
) -> Dict[str, int]
```

**严格性设计**：
- 基础权重不匹配 → 静默跳过。
- **Adapter 拓扑不匹配**（shape mismatch / missing / unexpected adapter keys）→ **抛出 RuntimeError**，要求使用相同的 `lora_type`、`lora_r`、`lora_target_modules` 恢复，或重新开始训练。

---

## 7. PEFT Planner：架构条件化适配决策

`PEFTPlanner` 实现了论文中的回归模型（Eq. 1）：

```
ΔmAP ≈ β₀ + β₁·φ_attn + β₂·φ_text + β₃·φ_dw + β₄·ξ_p
```

### 7.1 ArchitectureFingerprint

10 维架构指纹向量：

| 维度 | 符号 | 计算方式 | 含义 |
|------|------|----------|------|
| Attention 比例 | φ_attn | attention_modules / (conv + linear) | 注意力模块占比 |
| Text-fusion 比例 | φ_text | text_modules / (conv + linear) | 文本融合模块占比 |
| Depthwise 比例 | φ_dw | depthwise_conv / total_conv | 深度可分离卷积占比 |
| Grouped 比例 | φ_group | grouped_conv / total_conv | 分组卷积占比 |
| Linear 比例 | φ_linear | linear / (conv + linear) | 线性层占比 |
| 深度 | φ_depth | top_level_blocks / 30 | 归一化模型深度 |
| 宽度 | φ_width | log2(avg_channels) / 10 | 对数尺度平均通道宽度 |
| 检测头 | φ_head | head_params / total_params | 检测头参数占比 |
| 残差密度 | φ_residual | residual_modules / total_modules | 残差连接密度 |
| 归一化分布 | φ_norm | LN / (BN + LN + GN) | LayerNorm 占比 |

**缓存机制**：使用 `weakref.WeakKeyDictionary` 缓存指纹，模型 GC 后自动失效。

**架构族检测优先级**：RT-DETR > YOLO-World > YOLO12 > YOLO-Master-MoE > YOLO-CNN

### 7.2 PEFTPlanner 决策

```python
class PEFTPlanner:
    DEFAULT_COEFFS: List[float]  # 默认回归系数 [β₀, β₁, β₂, β₃, β₄]

    def plan(
        self,
        model: nn.Module,
        config: LoRAConfig,
    ) -> PlacementDecision

    def predict(
        self,
        fingerprint: ArchitectureFingerprint,
        variant: str,
        rank: int,
    ) -> float

    def fit(
        self,
        history: List[Tuple[ArchitectureFingerprint, str, float]]
    ) -> None
```

**决策状态**：

| 状态 | 条件 | 行为 |
|------|------|------|
| `ACCEPT` | predicted ΔmAP ≥ threshold | 正常继续 |
| `ADAPT` | 0 ≤ predicted ΔmAP < threshold 但可挽救 | 应用推荐的 variant/rank/safety_overrides |
| `REFUSE` | predicted ΔmAP < threshold 且风险过高 | 回退到 Full-SFT，输出 refusal_reason |

### 7.3 LOVO 交叉验证

```python
class LOVOValidator:
    def __init__(self, threshold: float = -0.05)
    def cross_validate(self, data_points: List[LOVODataPoint]) -> LOVOValidationResult
    def evaluate_catastrophe_detection(self, collector: LOVODataCollector) -> Dict[str, Any]
    def evaluate_decision_boundary(self, collector: LOVODataCollector) -> Dict[str, Any]
    def full_report(self, collector: LOVODataCollector) -> Dict[str, Any]
```

LOVO (Leave-One-Variant-Out) 验证：
- 每次 leave out 一个 `(fingerprint, variant, ΔmAP)` 数据点。
- 在剩余数据上拟合回归模型，预测被 leave out 的点的 ΔmAP。
- 计算 MSE、MAE、R²、灾难检测混淆矩阵（Precision/Recall/F1）。

**要求至少 5 个唯一数据点。**

---

## 8. MoLoRA：Mixture-of-LoRA

MoLoRA 在标准 LoRA 之上引入 **稀疏专家路由机制**，每个目标层维护多个 LoRA 专家，由 router 动态选择 Top-K 个专家进行加权组合。

### 8.1 MoLoRAConfig

继承自 `LoRAConfig`，扩展字段：

```python
@dataclass
class MoLoRAConfig(LoRAConfig):
    num_experts: int = 4                # 专家数量
    top_k: int = 2                      # 每次激活的专家数
    router_type: str = "linear"         # "linear" | "spatial" | "hybrid"
    router_hidden_dim: Optional[int] = None  # 默认 = C // 4

    # 辅助损失系数
    balance_loss_coef: float = 0.01     # 负载均衡损失
    z_loss_coef: float = 0.001          # Router logit Z-loss
    diversity_loss_coef: float = 0.0    # 专家多样性损失

    # 路由行为
    capacity_factor: float = 1.0        # 动态容量限制
    expert_dropout: float = 0.0         # 训练时禁用专家的概率
    top_k_warmup: Optional[int] = None  # 渐进增加 K 的步数
    warmup_steps: int = 0

    # 持续学习 / 域隔离
    domain_experts: Optional[Dict[str, List[int]]] = None
    freeze_experts: Optional[List[int]] = None

    # 初始化
    expert_init: str = "default"        # "default" | "orthogonal" | "gaussian"
```

**Preset 工厂**：

```python
def get_molora_preset(name: str) -> Dict[str, Any]
```

| Preset | experts | top_k | r | alpha | router |
|--------|---------|-------|---|-------|--------|
| `preset_small` | 2 | 1 | 4 | 8 | linear |
| `preset_standard` | 4 | 2 | 8 | 16 | linear |
| `preset_large` | 8 | 2 | 16 | 32 | hybrid |
| `preset_continual` | 8 | 2 | 8 | 16 | linear |

### 8.2 Router 类型

```python
# LinearRouter: 基于全局特征向量的线性门控
class LinearRouter(nn.Module): ...

# SpatialRouter: 基于空间位置感知的门控（每个空间位置独立路由）
class SpatialRouter(nn.Module): ...

# HybridRouter: 结合全局与空间信息的混合路由
class HybridRouter(nn.Module): ...
```

### 8.3 MoLoRA 层

```python
class MoLoRALayer(nn.Module):
    def __init__(self, base_layer, config: MoLoRAConfig)
    # 包含 num_experts 个 LoRA 专家 + 1 个 Router

class MoLoRAExpert(nn.Module):
    # 单个专家的 A/B 低秩矩阵
```

### 8.4 训练流程

```python
from ultralytics.nn.peft.molora import (
    MoLoRAConfig, get_peft_molora_model, mark_only_molora_as_trainable
)

config = MoLoRAConfig.from_lora_config(lora_config, num_experts=4, top_k=2)
model = get_peft_molora_model(model, config)
mark_only_molora_as_trainable(model)

# 训练时额外加入 balance_loss + z_loss
molora_loss = MoLoRALoss(config)
aux_loss = molora_loss(router_probs, expert_indices)
total_loss = det_loss + aux_loss
```

---

## 9. Few-Shot LoRA

针对小样本场景的增强 LoRA 实现，封装在 `FewShotLoRAConv` 中。

### 9.1 核心增强

| 特性 | 说明 |
|------|------|
| **Scheduled DropConnect** | 支持 constant/linear/cosine/exponential 四种调度策略 |
| **Gradient-Importance Weighted DropConnect** | 基于 Fisher 信息的重要性 EMA，高重要连接更低 drop 概率 |
| **知识蒸馏** | 支持 teacher feature 对齐损失 (`_compute_alignment_loss`) |
| **Adaptive Rank** | 根据数据稀缺度自动提升 rank（默认 `r≥32`） |
| **Variational Rank Selection** | Gumbel-Softmax 稀疏秩选择（straight-through estimator） |
| **Layer-wise Rank** | 每层独立计算 rank：浅层/宽通道层获得更大 rank |

### 9.2 DropConnect 调度

```python
def get_scheduled_dropconnect_rate(self, progress: float = 0.0) -> float
```

- `constant`: 固定 rate
- `linear`: `max → min` 线性下降
- `cosine`: 余弦下降
- `exponential`: 指数衰减 `exp(-5·progress)`

### 9.3 知识蒸馏 v3 增强

```python
few_shot_distill_schedule: str = "cosine"       # 蒸馏权重调度
few_shot_distill_weight_max: float = 1.0
few_shot_distill_weight_min: float = 0.1
few_shot_use_ema_teacher: bool = False          # EMA 教师模型
few_shot_ema_decay: float = 0.999
few_shot_response_distill: bool = False         # 检测头响应蒸馏
few_shot_response_distill_weight: float = 0.3
few_shot_layerwise_rank: bool = False           # 每层自适应 rank
```

---

## 10. 架构安全保护机制

### 10.1 RT-DETR 保护

当检测到 RT-DETR 架构时自动应用：

| 保护项 | 强制值 | 原因 |
|--------|--------|------|
| `alpha_warmup` | `≥ 3` | 防止 deformable attention 初始化不稳定 |
| `lr_mult` | `≤ 1.0` | 降低高 LR 对 MSDeformAttn 的扰动 |
| `include_attention` | `True` | 允许安全的 attention projection |
| `use_dora` | 降级为 `False`（除非 `allow_rtdetr_dora=True`） | RT-DETR + DoRA 在本地探测中出现 early NaN collapse |

### 10.2 YOLO12 Area-Attention 保护

当检测到 `A2C2f` / `AAttn` 模块时自动应用：

| 保护项 | 行为 |
|--------|------|
| 目标排除 | 排除 `attn.{qkv,proj,pe}` 和 `ABlock-internal mlp Conv2d` |
| `alpha_warmup` | 强制 `≥ 3` |
| `lr_mult` | 强制 `≤ 1.0` |

**根本原因**：YOLO12 的 AAttn 使用 Conv2d-based softmax attention，LoRA 注入易导致数值崩溃（loss 突降至 0，mAP/P/R 归零）。ABlock 内部 MLP 位于同一残差流且无 LayerNorm，同样触发梯度爆炸。

### 10.3 分组卷积保护

自动检测 `groups > 1` 的 `nn.Conv2d`：

- 若 `r % groups != 0`：自动将该层加入 `exclude_modules`，避免 PEFT 抛出 `ValueError`。
- 添加多层前缀变体：`name`、`model.name`、`model.model.name`，确保 PEFT 能正确匹配。

---

## 11. 使用示例

### 11.1 基础 LoRA 微调

```python
from ultralytics import YOLO
from ultralytics.utils.lora import apply_lora, LoRAConfig

# 加载预训练模型
model = YOLO("yolov8n.pt")

# 方式 1：通过配置对象
config = LoRAConfig(r=16, alpha=32, dropout=0.05, use_rslora=True)
model.model = apply_lora(model.model, config)

# 方式 2：通过 kwargs
model.model = apply_lora(
    model.model,
    lora_r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    lora_use_rslora=True,
    lora_target_modules=["conv"],  # 仅适配 conv 层
)

# 训练
model.train(data="coco128.yaml", epochs=50, lr0=0.001)
```

### 11.2 使用训练策略

```python
from ultralytics.utils.lora import LoraTrainingStrategy

strategy = LoraTrainingStrategy(model.model, epochs=100)

# 1. 准备 alpha warmup
strategy.prepare_alpha_warmup()  # 初始 scaling = 0

# 2. 应用层-wise LR decay
strategy.apply_layer_decay_to_optimizer(optimizer, decay_rate=0.85)

# 训练循环
for epoch in range(100):
    # 3. 更新 alpha warmup
    scale = strategy.step_alpha_warmup(epoch, warmup_epochs=5)
    
    # 4. 更新动态 dropout
    strategy.update_dropout_schedule(
        model.model, epoch, 100,
        start_dropout=0.0, end_dropout=0.15, schedule_start_ratio=0.3
    )
    
    # 5. 计算正交正则化损失
    ortho_loss = strategy.compute_orthogonal_loss(model.model, weight=1e-4)
    total_loss = det_loss + ortho_loss
    
    # ... backward & step

# 结束 warmup
strategy.finalize_alpha_warmup()
```

### 11.3 保存与加载适配器

```python
from ultralytics.utils.lora import save_lora_adapters, load_lora_adapters, merge_lora_weights

# 保存（仅 adapter 权重）
save_lora_adapters(model.model, "runs/lora_adapter_best")

# 加载到新模型
model2 = YOLO("yolov8n.pt")
load_lora_adapters(model2.model, "runs/lora_adapter_best")

# 合并到基础权重（推理加速）
merge_lora_weights(model2.model)
```

### 11.4 启用 PEFT Planner

```python
model.model = apply_lora(
    model.model,
    lora_r=16,
    lora_planner_enabled=True,  # 启用架构条件化决策
)
```

Planner 可能的输出：
```
[Planner] ACCEPT (predicted ΔmAP=0.052).
[Planner] ADAPT — applying recommended overrides.
  variant → dora
  rank → 8
  lora_alpha_warmup: 0 → 3
[Planner] REFUSE — predicted catastrophic ΔmAP=-0.12 for adalora on Conv-heavy architecture.
  Falling back to full-model fine-tuning.
```

### 11.5 使用 MoLoRA

```python
from ultralytics.nn.peft.molora import (
    MoLoRAConfig, get_peft_molora_model,
    mark_only_molora_as_trainable, get_molora_preset
)

# 使用 preset
preset = get_molora_preset("preset_standard")
config = MoLoRAConfig(**preset, lora_r=8, lora_alpha=16)

# 应用
model = get_peft_molora_model(model, config)
mark_only_molora_as_trainable(model)

# 查看参数统计
from ultralytics.nn.peft.molora import count_parameters
count_parameters(model)  # 输出 trainable / total 比例
```

### 11.6 数据集规模自适应配置

```python
from ultralytics.utils.lora import suggest_lora_config_for_dataset

rec = suggest_lora_config_for_dataset(
    num_images=16000,    # VOC 规模
    num_classes=20,
    epochs=50,
    batch_size=128,
)
# 返回推荐的 lora_r, lora_alpha, lora_lr_mult 等参数 + notes 说明
```

---

## 12. 参考与依赖

### 12.1 外部依赖

| 包 | 版本要求 | 用途 |
|----|----------|------|
| `peft` | ≥ 0.18 (推荐) | PEFT 后端核心 |
| `bitsandbytes` | 可选 | QLoRA 4-bit/8-bit 量化 |
| `transformers` | 可选 | `BitsAndBytesConfig` |
| `numpy` | 可选 | PEFT Planner LOVO 验证 |

### 12.2 支持的 PEFT 变体矩阵

| 变体 | Conv2d | Linear | Attention | 推荐场景 |
|------|--------|--------|-----------|----------|
| LoRA | ✅ | ✅ | ✅ | 通用默认 |
| DoRA | ✅ | ✅ | ✅ | 需要更稳定权重分解 |
| LoHa | ✅ | ✅ | ✅ | 文本-融合架构 |
| LoKr | ✅ | ✅ | ✅ | 高秩需求 |
| AdaLoRA | ❌* | ✅ | ✅ | 仅 Linear 层丰富时 |
| IA³ | ✅ | ✅ | ✅ | 超轻量（无 rank） |
| OFT | ✅ | ✅ | ✅ | 正交变换场景 |
| BOFT | ✅ | ✅ | ✅ | 蝴蝶正交变换 |
| HRA | ✅ | ✅ | ✅ | 高秩 + Gram-Schmidt |

*AdaLoRA 在 PEFT 0.18 中仅支持 `nn.Linear`，Conv2d 层会被静默跳过。

### 12.3 关键设计原则

1. **Fail-fast**：Adapter 拓扑不匹配时硬错误而非静默重初始化。
2. **Graceful degradation**：PEFT 失败时自动降级到 fallback（`lora_backend=auto`）。
3. **Architecture-conditioned safety**：针对不同架构（RT-DETR/YOLO12）应用硬编码安全保护。
4. **Zero silent behavior**：所有自动排除/降级操作均有明确日志。
5. **Backward compatibility**：fallback checkpoint 支持旧格式加载；`PeftProxy` 兼容所有 YOLO 索引/切片操作。
