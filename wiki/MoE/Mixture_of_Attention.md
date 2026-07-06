# MoA (Mixture of Attention) 与版本演进

> 本文档涵盖 YOLO-Master 项目中 **Mixture of Attention (MoA)** 模块的完整技术实现，以及从 v0 到 v0.15 的架构版本演进路线。所有类/函数签名均基于实际代码，技术术语保留英文原词。

---

## 目录

1. [设计哲学与核心概念](#一设计哲学与核心概念)
2. [MoA 模块详解](#二moa-模块详解)
3. [训练集成与辅助损失](#三训练集成与辅助损失)
4. [版本演进路线](#四版本演进路线)
5. [模型配置示例](#五模型配置示例)
6. [性能评估与消融实验](#六性能评估与消融实验)
7. [API 参考](#七api-参考)

---

## 一、设计哲学与核心概念

### 1.1 MoA vs MoE

YOLO-Master 同时支持 **MoE (Mixture of Experts)** 和 **MoA (Mixture of Attention)** 两种混合路由范式：

| 维度 | MoE | MoA |
|------|-----|-----|
| 路由对象 | Token → Expert FFN | Token → Attention Head |
| 核心操作 | 专家网络选择 | 注意力头选择 |
| 计算复杂度 | O(N·E·d²) | O(N·win²) / O(N²/4) / O(N) |
| 适用场景 | 特征变换增强 | 多尺度上下文聚合 |
| 空间维度 | 通道级操作 | 空间级操作 |

### 1.2 MoA 的三头设计

MoA 将注意力头分为三个功能组，每个组捕获不同尺度的上下文信息：

- **Local Heads** (`_LocalAttnHead`)：基于 DW-3×3 偏置的 QKV，使用窗口分区自注意力（Swin-style），捕获细粒度纹理和边缘细节，复杂度为 **O(N·win²)**
- **Regional Heads** (`_RegionalAttnHead`)：对 Key/Value 进行 stride-2 池化下采样，提供中距离上下文，复杂度为 **O(N²/4)**
- **Global Heads** (`_GlobalAttnHead`)：基于 Performer 风格的随机特征近似，将 softmax 注意力的 O(N²) 降至 **O(N)**，适合大空间特征图

### 1.3 设计约束

- **Fully CNN-native**：输入/输出均为 `[B, C, H, W]`，无需序列维度 reshape
- **Flash Attention 兼容**：PyTorch ≥ 2.0 时自动使用 `F.scaled_dot_product_attention`
- **即插即用**：通过 `C2fMoA` 包装器可直接在 YAML 配置中替换 `C3k2` / `A2C2f`
- **端到端可微**：软概率路由（无硬 dispatch），无负载均衡开销

---

## 二、MoA 模块详解

### 2.1 核心模块层次结构

```
ultralytics/nn/modules/moa/
├── __init__.py          # 公共导出
└── moa.py               # 核心实现 (716 行)
```

### 2.2 辅助函数

#### `_flash_attn`

```python
def _flash_attn(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                scale: float) -> torch.Tensor:
```

| 参数 | 类型 | 说明 |
|------|------|------|
| `q` | `torch.Tensor` | Query tensor, shape `[B, nh, N, hd]` |
| `k` | `torch.Tensor` | Key tensor, shape `[B, nh, N, hd]` |
| `v` | `torch.Tensor` | Value tensor, shape `[B, nh, N, hd]` |
| `scale` | `float` | 注意力缩放因子 |

**返回**：`torch.Tensor`，shape `[B, nh, N, hd]`

**特性**：
- 优先使用 `F.scaled_dot_product_attention`（PyTorch ≥ 2.0）
- 兼容旧版 PyTorch（无 `scale` 参数时自动回退）
- 自动处理 `TypeError` 异常，保证跨版本兼容性

#### `_window_flash_attn`

```python
def _window_flash_attn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    window_size: int,
    height: int,
    width: int,
) -> torch.Tensor:
```

窗口分区 SDPA，复杂度为 **O(N·win²)**。支持非整数倍窗口大小的特征图（自动 padding）。

### 2.3 注意力头实现

#### `_LocalAttnHead`

```python
class _LocalAttnHead(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None,
                 window_size: int = 7):
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dim` | `int` | — | 输入通道维度 |
| `num_heads` | `int` | — | 注意力头数 |
| `head_dim` | `Optional[int]` | `max(dim // num_heads, 16)` | 每头维度 |
| `window_size` | `int` | `7` | 窗口大小 |

**架构**：
```
Input [B, C, H, W]
  ├── qkv_dw: Conv2d(C, C, 3, padding=1, groups=C)  # 深度可分离 3×3
  ├── qkv_pw: Conv2d(C, 3*inner, 1)                 # 点卷积投影
  ├── pe: Conv2d(inner, inner, 7, padding=3, groups=inner)  # 位置编码
  ├── _window_flash_attn(...)                       # 窗口注意力
  ├── proj: Conv2d(inner, C, 1)                     # 输出投影
  └── norm: GroupNorm                               # 归一化
```

**前向传播**：
```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, H, W]
    # output: [B, C, H, W]
```

#### `_RegionalAttnHead`

```python
class _RegionalAttnHead(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None,
                 pool_stride: int = 2):
```

通过 stride-2 平均池化对 K/V 进行下采样，使每个 Query 拥有更大的有效感受野。

**架构**：
```
Input [B, C, H, W]
  ├── q_proj: Conv2d(C, inner, 1)                   # Query 投影
  ├── kv_pool: AvgPool2d(2, 2) → Conv2d(C, 2*inner, 1)  # K/V 下采样
  ├── _flash_attn(q, k, v, scale)                   # 标准 SDPA
  ├── proj: Conv2d(inner, C, 1)
  └── norm: GroupNorm
```

#### `_GlobalAttnHead`

```python
class _GlobalAttnHead(nn.Module):
    def __init__(self, dim: int, num_heads: int, head_dim: Optional[int] = None,
                 nb_features: int = 64, rf_seed: int = 0x5F3759DF):
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `nb_features` | `int` | `64` | 随机特征数量 |
| `rf_seed` | `int` | `0x5F3759DF` | 随机特征生成种子 |

**线性注意力机制**（Performer-style）：

1. 使用 QR 分解生成正交随机特征矩阵 `[hd, hd]`
2. 通过 ReLU 核函数将 Q/K 投影到特征空间：`φ(x) = ReLU(x) + 1e-6`
3. 计算 `kv = k^T @ v` 和 `k_sum = sum(k)`
4. 输出：`output = (q @ kv) / (q @ k_sum)`

**回退机制**：当 `N ≤ 256` 时自动回退到标准 `_flash_attn`

**数值稳定性**：
- 对 kernel 特征进行 `clamp(max=1e4)` 防止 float16 溢出
- 对 kv 累加器进行 L2 归一化
- 对 numerator/denominator 进行 clamp 限制

### 2.4 路由器

#### `_MoARouter`

```python
class _MoARouter(nn.Module):
    def __init__(self, dim: int, num_groups: int, reduction: int = 8,
                 temperature: float = 1.0):
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dim` | `int` | — | 输入通道数 |
| `num_groups` | `int` | — | 头组数量（MoABlock 中固定为 3） |
| `reduction` | `int` | `8` | 中间层通道缩减比 |
| `temperature` | `float` | `1.0` | Softmax 温度 |

**架构**：
```
Conv2d(dim, hidden, 1) → GroupNorm → SiLU → Conv2d(hidden, num_groups, 1)
```

**初始化**：权重和偏置均初始化为零，使初始路由接近均匀分布。

**前向传播**：
```python
def forward(self, x: torch.Tensor, return_logits: bool = False) -> torch.Tensor:
    # x: [B, C, H, W]
    # probs: [B, num_groups, H, W], sum-to-one over num_groups
```

训练时温度系数生效，推理时固定为 `1.0`。

### 2.5 核心构建块

#### `MoABlock`

```python
class MoABlock(nn.Module):
    NUM_GROUPS: int = 3  # local / regional / global

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        attn_drop: float = 0.0,
        shortcut: bool = True,
        aux_loss_coeff: float = 0.01,
        block_index: int = 0,
        local_window_size: int = 7,
    ):
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `dim` | `int` | — | 输入/输出通道数 |
| `num_heads` | `int` | `8` | 总注意力头数，必须能被 `NUM_GROUPS` 整除 |
| `mlp_ratio` | `float` | `2.0` | FFN 扩展比例 |
| `temperature` | `float` | `1.0` | 路由器初始温度 |
| `attn_drop` | `float` | `0.0` | 注意力输出 Dropout 率 |
| `shortcut` | `bool` | `True` | 是否使用残差连接 |
| `aux_loss_coeff` | `float` | `0.01` | 辅助损失系数 |
| `block_index` | `int` | `0` | 块索引，用于生成 diverse RF seed |
| `local_window_size` | `int` | `7` | 局部注意力窗口大小 |

**前向传播流程**：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # 1. 路由权重 [B, 3, H, W]
    weights, router_logits = self.router(x, return_logits=True)
    
    # 2. 计算辅助损失（仅训练时）
    self.last_aux_loss = _moa_router_aux_loss(weights, router_logits, self.aux_loss_coeff)
    
    # 3. 三头注意力输出
    out_l = self.local_head(x)    # [B, C, H, W]
    out_r = self.region_head(x)   # [B, C, H, W]
    out_g = self.global_head(x)   # [B, C, H, W]
    
    # 4. 软混合
    mixed = w_l * out_l + w_r * out_r + w_g * out_g
    mixed = self.attn_drop(self.fusion(mixed))
    
    # 5. 残差 + LayerScale
    if self.shortcut:
        x = x + self.ls_attn * mixed
        x = x + self.ls_ffn * self.ffn(x)
    else:
        x = self.ls_attn * mixed
        x = self.ls_ffn * self.ffn(x)
    
    return x
```

**特性**：
- **LayerScale**：初始化为 `0.1`，稳定早期训练（CaiT 风格）
- **全局头多样性**：每个块使用不同的 RF seed（`block_index * 7919 + 2 * 65537`），确保不同层的随机特征基不重复

### 2.6 C2f 包装器

#### `C2fMoA`

```python
class C2fMoA(nn.Module):
    def __init__(
        self,
        c1: int,                          # 输入通道
        c2: int,                          # 输出通道
        n: int = 1,                       # MoABlock 堆叠数量
        num_heads: int = 6,               # 每块注意力头数
        mlp_ratio: float = 2.0,           # FFN 扩展比例
        temperature: float = 1.0,         # 初始温度
        shortcut: bool = True,            # 残差开关
        e: float = 0.5,                   # 内部通道扩展比
        aux_loss_coeff: float = 0.01,     # 辅助损失系数
        local_window_size: int = 7,       # 窗口大小
    ):
```

**架构**（与 C3k2 / A2C2f 兼容）：
```
Input [B, c1, H, W]
  ├── cv1: Conv(c1, 2*c, 1) ──┬── chunk(2) ── identity branch [B, c, H, W]
  │                            └── n × MoABlock(c, ...)       [B, c, H, W]
  └── cv2: Conv((n+2)*c, c2, 1)  # 融合所有分支
```

**头数自适应**：
```python
eff_heads = num_heads
while eff_heads % MoABlock.NUM_GROUPS != 0:
    eff_heads += 1
while self.c // eff_heads < 16 and eff_heads > MoABlock.NUM_GROUPS:
    eff_heads -= MoABlock.NUM_GROUPS
eff_heads = max(eff_heads, MoABlock.NUM_GROUPS)
```

### 2.7 Neck 跨尺度融合

#### `NeckMoAFusion`

```python
class NeckMoAFusion(nn.Module):
    def __init__(
        self,
        c_hi: int,                        # 高分辨率特征通道
        c_lo: int,                        # 低分辨率特征通道
        c_out: int,                       # 输出通道
        num_heads: int = 4,               # 交叉注意力头数
        shortcut: bool = True,            # 残差开关
        aux_loss_coeff: float = 0.01,     # 辅助损失系数
    ):
```

**功能**：在 FPN/PAN Neck 中替代简单的 Concat + Conv，使用双向交叉注意力融合高分辨率（细粒度）和低分辨率（语义）特征图。

**路由决策**：基于内容相似性，让模型学习哪些 Query 需要长距离上下文（→ global head）vs 局部细化（→ local head）。

**前向传播**：
```python
def forward(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor:
    # hi: [B, c_hi, H, W]    (fine-grained, e.g. P3/P4)
    # lo: [B, c_lo, H/2, W/2] (semantic, e.g. P4/P5)
    # output: [B, c_out, H, W]
```

---

## 三、训练集成与辅助损失

### 3.1 路由器辅助损失

#### `_moa_router_aux_loss`

```python
def _moa_router_aux_loss(weights: torch.Tensor, logits: torch.Tensor, coeff: float) -> torch.Tensor:
```

基于 GShard 风格的 MoA 路由器正则化，包含三项：

1. **Balance Loss**：`num_groups * Σ(importance²)`，鼓励负载均衡
2. **Z-Loss**：`0.1 * mean(logsumexp(logits)²)`，防止 logits 过大
3. **Entropy Deficit**：`0.01 * (max_entropy - entropy) / max_entropy`，避免过度均匀

```python
balance_loss = num_groups * torch.sum(importance * importance)
z_loss = torch.logsumexp(logits.float(), dim=1).pow(2).mean()
entropy_deficit = (max_entropy - entropy).clamp_min(0.0) / max_entropy
return coeff * (balance_loss + 0.1 * z_loss + 0.01 * entropy_deficit)
```

### 3.2 辅助损失收集

#### `collect_moa_aux_loss`

```python
def collect_moa_aux_loss(model: nn.Module) -> torch.Tensor:
```

智能收集 `C2fMoA` 和 `NeckMoAFusion` 中的辅助损失，**避免重复计数**：

```python
for m in model.modules():
    if isinstance(m, C2fMoA):
        # 收集 C2fMoA 级别，跳过内部 MoABlock
        total += m.last_aux_loss
        covered.update(id(child) for child in m.modules())
    elif isinstance(m, NeckMoAFusion):
        total += m.last_aux_loss
    elif isinstance(m, MoABlock) and id(m) not in covered:
        total += m.last_aux_loss
```

### 3.3 Mixture Loss EMA 归一化

在 `ultralytics/utils/loss.py` 中，`_collect_mixture_aux_loss` 实现了 MoE/MoT/MoA 三类辅助损失的 **EMA 归一化**：

```python
def _collect_mixture_aux_loss(model, device):
    moe_l = _collect_moe_aux_loss(model, device)
    mot_l = _collect_mot_aux_loss(model, device)
    moa_l = _collect_moa_aux_loss(model, device)
    
    # EMA 归一化防止大尺度损失（MoE ~1.0）淹没小尺度损失（MoA ~0.01-0.1）
    return (moe_l / moe_scale) + (mot_l / mot_scale) + (moa_l / moa_scale)
```

### 3.4 温度退火

#### `anneal_moa_temperature`

```python
def anneal_moa_temperature(model: nn.Module, factor: float = 0.99,
                           min_temp: float = 0.3) -> None:
```

在每个 epoch 结束时调用，乘法退火路由器温度：

```python
for m in model.modules():
    if isinstance(m, _MoARouter):
        m.temperature = max(m.temperature * factor, min_temp)
```

**训练器集成**（`ultralytics/engine/trainer.py`）：

```python
def _detect_moa_mot_modules(self):
    """检测模型是否包含 MoA/MoT 模块，缓存结果。"""
    self._has_moa_mot = any(isinstance(m, (C2fMoA, C2fMoT)) for m in model.modules())

def _anneal_moa_mot_temperature(self):
    """每个 epoch 结束后退火温度。"""
    if not getattr(self, "_has_moa_mot", False):
        return
    factor = float(getattr(self.args, "moa_mot_temperature_factor", 0.97))
    min_temp = float(getattr(self.args, "moa_mot_min_temperature", 0.3))
    anneal_moa_temperature(model, factor=factor, min_temp=min_temp)
```

**超参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `moa_mot_temperature_factor` | `0.97` | 温度退火因子 |
| `moa_mot_min_temperature` | `0.3` | 最小温度 |
| `moa_local_window_size` | `7` | 局部窗口大小 |

---

## 四、版本演进路线

### 4.1 版本时间线

```
v0 ──→ v0.1 ──→ v0.3 ──→ v0.4 ──→ v0.5 ──→ v0.6 ──→ v0.7 ──→ v0.8 ──→ v0.9 ──→ v0.10 ──→ v0.12 ──→ v0.13 ──→ v0.14 ──→ v0.15
       │         │         │         │         │         │         │         │          │           │           │           │
       │         │         │         │         │         │         │    MoA引入    │      Optimal    MultiHead  Diversified GatedFusion
       │         │         │         │         │         │    LowRank   Refined      Visual      Hybrid        Router       Expert       MoE
       │         │         │         │         │    A2C2f  Hybrid    LowRank     Enhanced      GateMoE       MoE          MoE
       │         │         │         │    基础改进  引入     Adaptive   Hybrid      Adaptive      v0.6核心     并行路由    异构专家池   门控融合
       │         │         │    基础改进  backbone  backbone  GateMoE    Adaptive   GateMoE       +v0.11      +专家       +1×1/3×3/    +随机深度
       │         │    Ultimate  backbone  稳定     引入      引入       GateMoE    引入          路由器升级   Dropout     dilated     正则化
       │    基础检测  Optimized  引入MoE   版本    多任务      低秩分解   混合门控   细节感知      v0.10 MoA   layer-adaptive 冗余路径   3×3混合核
   初始版本  模型配置   MoE模块   位置优化   支持    支持        专家       路由       增强
```

### 4.2 各版本核心 MoE 模块

| 版本 | 核心 MoE 模块 | 关键特性 |
|------|--------------|----------|
| v0 | 基础 Backbone | 标准 YOLO 架构 |
| v0.1 | 基础检测 | 初始多任务支持 |
| v0.3 | `UltimateOptimizedMoE` | 通道分割 + 融合专家 + 动态温度 + 自适应容量 |
| v0.4 | 改进版 Backbone | 路由优化 |
| v0.5 | 稳定版本 | 多尺度特征增强 |
| v0.6 | `A2C2f` 引入 | 注意力增强 C2f |
| v0.7 | `LowRankHybridAdaptiveGateMoE` | 低秩分解 + 混合自适应门控 |
| v0.8 | `RefinedLowRankHybridAdaptiveGateMoE` | 特征精细化低秩混合专家 |
| v0.9 | 优化版本 | 推理效率提升 |
| v0.10 | `VisualEnhancedAdaptiveGateMoE` | 细节感知 + 上下文增强视觉专家 |
| v0.12 | `OptimalHybridGateMoE` | v0.6 核心 + v0.11 路由器升级 + layer-adaptive split_ratio |
| v0.13 | `MultiHeadRouterMoE` | 多头并行路由 + 专家 Dropout |
| v0.14 | `DiversifiedExpertMoE` | 异构专家池（1×1 / 3×3 / dilated-3×3 混合核） |
| v0.15 | `GatedFusionMoE` | 跨路径内容感知门控融合 + 随机深度正则化 |

### 4.3 MoA 引入里程碑

**v0.8-MoA** 是首个引入 MoA 的版本：

```yaml
# YOLO-Master v0.8-MoA
backbone:
  # ... 标准 Backbone ...
  - [-1, 1, LowRankHybridAdaptiveGateMoE, [512, 4, 2, 0.5]]   # P3/P4
  # ...
  - [-1, 1, LowRankHybridAdaptiveGateMoE, [512, 8, 2, 0.5]]   # P4/P5
  # ...
  - [-1, 1, LowRankHybridAdaptiveGateMoE, [1024, 16, 2, 0.5]] # P5

head:
  # P5 → P4
  - [-1, 2, C2fMoA, [512, 2, 6, 2.0, 1.0, True]]   # MoA neck block P4
  # P4 → P3
  - [-1, 2, C3k2, [256, True]]                      # 标准卷积块 P3
  # P3 → P4 (bottom-up)
  - [-1, 2, C2fMoA, [512, 2, 6, 2.0, 0.8, True]]   # MoA neck block P4-up
  # P4 → P5 (bottom-up)
  - [-1, 2, C2fMoA, [512, 2, 6, 2.0, 0.8, True]]   # MoA neck block P5-up
```

设计决策：
- **P3 保持标准 C3k2**：浅层特征图尺寸大，MoA 计算成本高
- **P4/P5 使用 C2fMoA**：深层语义特征更适合多尺度注意力
- **温度递减**：深层使用更低的初始温度（0.8 vs 1.0），增强路由确定性

### 4.4 v0.10-MoA 演进

v0.10 将 Backbone 中的 MoE 升级为 `VisualEnhancedAdaptiveGateMoE`，同时保留 MoA Neck：

```yaml
# YOLO-Master v0.10-MoA
backbone:
  - [-1, 1, VisualEnhancedAdaptiveGateMoE, [512, 4, 2, 0.5]]   # P3/P4
  - [-1, 1, VisualEnhancedAdaptiveGateMoE, [512, 8, 2, 0.5]]   # P4/P5
  - [-1, 1, VisualEnhancedAdaptiveGateMoE, [1024, 16, 2, 0.5]] # P5
```

### 4.5 v0.12-v0.15 核心演进

**v0.12 - OptimalHybridGateMoE**：
- 采用 v0.6 验证成功的核心架构
- 引入 v0.11 路由器升级
- **Layer-adaptive split_ratio**：
  - P3/P4 (layer 5): `0.5` — 更多动态容量处理细粒度特征
  - P4/P5 (layer 8): `0.5` — 平衡配置
  - P5 (layer 11): `0.375` — 小特征图更多静态容量

**v0.13 - MultiHeadRouterMoE**：
- 在 v0.12 核心基础上增加多头并行路由
- 专家 Dropout 机制促进冗余路径学习

**v0.14 - DiversifiedExpertMoE**：
- 异构专家池：1×1 / 3×3 / dilated-3×3 混合卷积核
- 实现真正的功能多样性

**v0.15 - GatedFusionMoE**：
- 跨路径内容感知门控融合
- Stochastic Depth 正则化

---

## 五、模型配置示例

### 5.1 完整 MoA 检测配置 (v0.10-MoA)

```yaml
# YOLO-Master v0.10-MoA 🚀 AGPL-3.0 License
# Stage 1 MoA variant: VisualEnhancedAdaptiveGateMoE backbone + C2fMoA neck.

nc: 80
scales:
  n: [0.50, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]
  - [-1, 1, Conv, [128, 3, 2]]
  - [-1, 2, C3k2, [256, False, 0.25]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [-1, 2, C3k2, [512, False, 0.25]]

  - [-1, 1, VisualEnhancedAdaptiveGateMoE, [512, 4, 2, 0.5]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [-1, 4, A2C2f, [512, True, 4]]

  - [-1, 1, VisualEnhancedAdaptiveGateMoE, [512, 8, 2, 0.5]]

  - [-1, 1, Conv, [1024, 3, 2]]
  - [-1, 4, A2C2f, [1024, True, 1]]

  - [-1, 1, VisualEnhancedAdaptiveGateMoE, [1024, 16, 2, 0.5]]

head:
  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 8], 1, Concat, [1]]
  - [-1, 2, C2fMoA, [512, 2, 6, 2.0, 1.0, True]]

  - [-1, 1, nn.Upsample, [None, 2, "nearest"]]
  - [[-1, 5], 1, Concat, [1]]
  - [-1, 2, C3k2, [256, True]]

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 14], 1, Concat, [1]]
  - [-1, 2, C2fMoA, [512, 2, 6, 2.0, 0.8, True]]

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 11], 1, Concat, [1]]
  - [-1, 2, C2fMoA, [512, 2, 6, 2.0, 0.8, True]]

  - [[17, 20, 23], 1, Detect, [nc]]
```

### 5.2 C2fMoA 参数说明

```python
# YAML 中的参数顺序
[-1, 2, C2fMoA, [512,   # c2: 输出通道
                   2,    # n: MoABlock 数量
                   6,    # num_heads: 注意力头数
                   2.0,  # mlp_ratio: FFN 扩展比
                   1.0,  # temperature: 初始温度
                   True  # shortcut: 残差开关
                  ]]
```

### 5.3 NeckMoAFusion 配置示例

```python
from ultralytics.nn.modules.moa import NeckMoAFusion
import torch

# 初始化
fusion = NeckMoAFusion(
    c_hi=64,      # P4 高分辨率特征通道
    c_lo=128,     # P5 低分辨率特征通道
    c_out=64,     # 输出通道
    num_heads=4,  # 交叉注意力头数
    shortcut=True
)

# 前向传播
hi = torch.randn(2, 64, 32, 32)   # P4 特征
lo = torch.randn(2, 128, 16, 16)  # P5 特征
out = fusion(hi, lo)              # [2, 64, 32, 32]
```

---

## 六、性能评估与消融实验

### 6.1 消融实验脚本

项目提供 `scripts/compare_moa_ablation.py` 用于复现 MoA 消融实验：

```bash
# 构建检查：验证模型结构和参数量
python scripts/compare_moa_ablation.py --check-build

# 延迟基准测试
python scripts/compare_moa_ablation.py --benchmark --imgsz 256 --reps 5 --device cpu

# 训练对比（v0.7 baseline vs v0.8 MoA vs v0.10 baseline vs v0.10 MoA）
python scripts/compare_moa_ablation.py --train --epochs 50 --imgsz 640 --batch 8 --device 0

# 仅生成汇总
python scripts/compare_moa_ablation.py --summary-only
```

**支持的模型对比**：

| Key | 描述 |
|-----|------|
| `v07` | YOLO-Master v0.7 baseline |
| `v08_moa` | YOLO-Master v0.8 MoA |
| `v10` | YOLO-Master v0.10 baseline |
| `v10_moa` | YOLO-Master v0.10 MoA |

### 6.2 测试覆盖

`tests/test_moa.py` 提供 16 个测试用例：

| 测试 | 描述 |
|------|------|
| `test_moa_modules_forward_backward` | MoABlock / C2fMoA 前向+反向传播 |
| `test_neck_moa_fusion_forward_backward` | NeckMoAFusion 前向+反向传播 |
| `test_neck_moa_fusion_handles_non_strict_scale_ratio` | 非严格尺度比处理 |
| `test_moa_router_stays_finite_at_tiny_annealed_temperature` | 极小温度下数值稳定性 |
| `test_moa_attention_heads_handle_non_divisible_dim_and_heads` | 非整除维度/头数处理 |
| `test_c2fmoa_aux_loss_not_double_counted_for_nested_blocks` | 辅助损失防重复计数 |
| `test_flash_attn_supports_sdpa_without_scale_keyword` | 旧版 PyTorch 兼容性 |
| `test_moa_aux_loss_collected_for_c2f_and_neck` | 辅助损失收集验证 |
| `test_c2fmoa_small_channels_keep_valid_head_count` | 小通道数头数自适应 |
| `test_c2fmoa_rounds_head_count_to_expert_groups` | 头数取整验证 |
| `test_neck_moa_fusion_eval_projects_self_path_and_zero_aux_loss` | Eval 模式零辅助损失 |
| `test_collect_moa_aux_loss_handles_empty_module_and_standalone_block` | 边界条件处理 |
| `test_moa_temperature_anneal` | 温度退火功能 |
| `test_moa_global_head_per_block_seed` | 每块 RF seed 多样性 |
| `test_moa_model_configs_parse` | YAML 配置解析（v0.8/v0.10） |

---

## 七、API 参考

### 7.1 公共导出接口

```python
from ultralytics.nn.modules.moa import (
    MoABlock,              # 核心 MoA 构建块
    C2fMoA,                # C2f 风格包装器
    NeckMoAFusion,         # Neck 跨尺度融合
    anneal_moa_temperature, # 温度退火
    collect_moa_aux_loss,   # 辅助损失收集
)
```

### 7.2 完整类签名参考

#### MoABlock

```python
class MoABlock(nn.Module):
    NUM_GROUPS: int = 3

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        attn_drop: float = 0.0,
        shortcut: bool = True,
        aux_loss_coeff: float = 0.01,
        block_index: int = 0,
        local_window_size: int = 7,
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor
    # Input:  [B, dim, H, W]
    # Output: [B, dim, H, W]
```

#### C2fMoA

```python
class C2fMoA(nn.Module):
    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        temperature: float = 1.0,
        shortcut: bool = True,
        e: float = 0.5,
        aux_loss_coeff: float = 0.01,
        local_window_size: int = 7,
    )

    def forward(self, x: torch.Tensor) -> torch.Tensor
    # Input:  [B, c1, H, W]
    # Output: [B, c2, H, W]
```

#### NeckMoAFusion

```python
class NeckMoAFusion(nn.Module):
    def __init__(
        self,
        c_hi: int,
        c_lo: int,
        c_out: int,
        num_heads: int = 4,
        shortcut: bool = True,
        aux_loss_coeff: float = 0.01,
    )

    def forward(self, hi: torch.Tensor, lo: torch.Tensor) -> torch.Tensor
    # hi: [B, c_hi, H, W]     (fine-grained)
    # lo: [B, c_lo, H/2, W/2] (semantic)
    # Output: [B, c_out, H, W]
```

#### 工具函数

```python
def anneal_moa_temperature(
    model: nn.Module,
    factor: float = 0.99,
    min_temp: float = 0.3
) -> None

def collect_moa_aux_loss(model: nn.Module) -> torch.Tensor
```

### 7.3 训练超参数

| 超参数 | 默认值 | 说明 |
|--------|--------|------|
| `moa_mot_temperature_factor` | `0.97` | 温度退火因子 |
| `moa_mot_min_temperature` | `0.3` | 最小温度阈值 |
| `moa_local_window_size` | `7` | 局部注意力窗口 |
| `mot_balance_loss` | `0.01` | MoT 平衡损失系数 |
| `mot_router_z_loss` | `0.01` | MoT Z-loss 系数 |
| `moe` | `0.01` | MoE 辅助损失增益 |

---

## 八、设计决策与最佳实践

### 8.1 何时使用 MoA

| 场景 | 推荐方案 |
|------|----------|
| 需要多尺度上下文聚合 | MoA Neck (C2fMoA) |
| 大分辨率特征图 (P3) | 避免 MoA，使用标准 C3k2 |
| 中等分辨率 (P4/P5) | C2fMoA 收益最大 |
| 边缘部署 | 评估延迟成本，考虑 `shortcut=False` |
| 小通道数 (< 32) | 自动头数调整机制已处理 |

### 8.2 温度退火策略

```python
# 推荐：每个 epoch 结束时调用
anneal_moa_temperature(model, factor=0.97, min_temp=0.3)

# 激进策略（更快收敛，可能损失精度）
anneal_moa_temperature(model, factor=0.95, min_temp=0.2)

# 保守策略（更稳定，收敛慢）
anneal_moa_temperature(model, factor=0.99, min_temp=0.5)
```

### 8.3 辅助损失调参

```python
# 默认配置（推荐）
MoABlock(dim=512, aux_loss_coeff=0.01)

# 如果观察到路由崩溃（所有 token 走向同一头）
MoABlock(dim=512, aux_loss_coeff=0.05)

# 如果辅助损失过大影响主损失收敛
MoABlock(dim=512, aux_loss_coeff=0.001)
```

### 8.4 与 MoE 的协同使用

YOLO-Master 支持 MoE (Backbone) + MoA (Neck) 的混合架构：

```yaml
backbone:
  # Backbone 使用 MoE 进行特征变换
  - [-1, 1, VisualEnhancedAdaptiveGateMoE, [512, 4, 2, 0.5]]

head:
  # Neck 使用 MoA 进行多尺度上下文聚合
  - [-1, 2, C2fMoA, [512, 2, 6, 2.0, 1.0, True]]
```

损失函数自动收集并归一化 MoE + MoT + MoA 的辅助损失，无需手动干预。

---

> **文件位置**：`ultralytics/nn/modules/moa/moa.py` (716 行)  
> **测试文件**：`tests/test_moa.py` (223 行，16 个测试用例)  
> **消融脚本**：`scripts/compare_moa_ablation.py`  
> **模型配置**：`ultralytics/cfg/models/master/v0_8/det/yolo-master-moa-n.yaml`  
> **训练集成**：`ultralytics/engine/trainer.py` (`_detect_moa_mot_modules`, `_anneal_moa_mot_temperature`)  
> **损失集成**：`ultralytics/utils/loss.py` (`_collect_moa_aux_loss`, `_collect_mixture_aux_loss`)
