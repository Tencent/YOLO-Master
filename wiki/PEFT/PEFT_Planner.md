# PEFT Planner — 架构条件化适配器部署规划器

PEFT Planner 是 YOLO-Master 中用于**架构条件化地决策 PEFT 适配器部署方案**的核心组件。它基于论文公式 (Eq. 1) 的回归模型，结合硬安全护栏（hard guardrails），对给定的模型架构和 PEFT 配置做出 **ACCEPT / ADAPT / REFUSE** 三类决策。

---

## 1. 核心设计理念

### 1.1 回归模型 (Eq. 1)

PEFT Planner 的核心是一个 11 维特征的线性回归模型：

```
ΔmAP ≈ β₀ + β₁φ_attn + β₂φ_text + β₃φ_dw + β₄ξ_p
       + β₅φ_depth + β₆φ_width + β₇φ_head + β₈φ_residual
       + β₉φ_norm + β₁₀·log(r)
```

其中：
- `φ_attn`, `φ_text`, `φ_dw` 等 — 来自 `ArchitectureFingerprint` 的 10 维架构指纹
- `ξ_p` — PEFT 变体系数，来自 `PEFTVariantProfile.xi`
- `log(r)` — LoRA 秩的对数效应

**默认回归系数**（基于论文 Table 1 的 12 个标准数据点校准）：

| 系数 | 维度 | 默认值 | 说明 |
|------|------|--------|------|
| β₀ | intercept | 0.0656 | 截距 |
| β₁ | φ_attn | 0.0026 | Attention 模块占比效应 |
| β₂ | φ_text | 0.0 | Text-fusion 占比效应 |
| β₃ | φ_dw | 0.0054 | Depthwise 卷积占比效应 |
| β₄ | ξ_p | 1.0 | 变体系数权重 |
| β₅~β₉ | 扩展维度 | 0.0 | 默认不激活，LOVO 拟合后启用 |
| β₁₀ | log(r) | 0.0 | 秩的对数效应 |

### 1.2 三态决策系统

| 决策状态 | 含义 | 后续动作 |
|----------|------|----------|
| **ACCEPT** | 配置安全可行 | 直接应用请求的 PEFT 配置 |
| **ADAPT** | 需要调整 | 采用推荐的 variant / rank / 安全覆盖项 |
| **REFUSE** | 配置危险 | 回退到 Full-SFT（全参数微调） |

**REFUSE 阈值**：`ΔmAP < -0.05`，即预测 mAP 下降超过 5% 时触发拒绝。

---

## 2. ArchitectureFingerprint — 10 维架构指纹

`ArchitectureFingerprint` 是一个紧凑的 10 维向量，用于量化描述任意 PyTorch 模型的架构特征。

### 2.1 原始 5 维（v1）

| 维度 | 计算方式 | 典型值示例 |
|------|----------|-----------|
| `phi_attn` | attention 模块数 / (conv + linear) 总数 | YOLO-CNN: 0, YOLO12: 0.45, RT-DETR: 0.85 |
| `phi_text` | text-fusion 模块数 / (conv + linear) 总数 | YOLO-World: 0.5, 其他: 0 |
| `phi_dw` | depthwise conv 数 / conv 总数 | 标准 CNN: ~0.1 |
| `phi_group` | grouped conv 数 / conv 总数 | 标准 CNN: ~0.05 |
| `phi_linear` | linear 层数 / (conv + linear) 总数 | Transformer: ~0.3 |

### 2.2 扩展 5 维（v2）

| 维度 | 计算方式 | 用途 |
|------|----------|------|
| `phi_depth` | 顶层 block 数 / 30, clamp 到 [0, 1] | 区分同家族不同深度模型 |
| `phi_width` | log₂(平均通道数) / 10 | 区分 n/s/m/l/x 规模 |
| `phi_head` | head 参数量 / 总参数量 | 检测头复杂度 |
| `phi_residual` | 含残差连接模块数 / 总模块数 | 残差密度 |
| `phi_norm` | LayerNorm 数 / (BN + LN + GN) 总数 | 归一化层分布 |

### 2.3 关键 API

```python
from ultralytics.utils.lora.planner import ArchitectureFingerprint

# 计算指纹（自动 unwrap DDP / torch.compile）
fingerprint = ArchitectureFingerprint.compute(model)

# 手动使缓存失效（架构修改后调用）
ArchitectureFingerprint.invalidate_cache(model)

# 架构家族检测（辅助工具，不改变指纹值）
family = ArchitectureFingerprint._detect_architecture_family(model)
# 返回: "rtdetr" | "yolo_world" | "yolo12" | "yolo_master_moe" | "yolo_cnn"
```

### 2.4 缓存机制

使用 `weakref.WeakKeyDictionary` 实现指纹缓存：
- **自动失效**：模型对象被垃圾回收时，缓存条目自动清除
- **避免内存地址复用污染**：跨测试运行或模型重建时无需手动清理

---

## 3. PEFTVariantProfile — 变体系数表

每个 PEFT 变体都有预校准的 `xi` 系数，来源于论文 Table 1 的最小二乘拟合。

| 变体 | xi | supports_conv | supports_linear | supports_attention | supports_text_fusion |
|------|-----|---------------|-----------------|-------------------|---------------------|
| **lora** | 0.0 | ✅ | ✅ | ✅ | ❌ |
| **dora** | +0.0050 | ✅ | ✅ | ✅ | ❌ |
| **loha** | -0.0208 | ✅ | ✅ | ✅ | ✅ |
| **lokr** | -0.0055 | ✅ | ✅ | ✅ | ❌ |
| **adalora** | 0.0 | ❌ | ✅ | ✅ | ❌ |
| **ia3** | -0.0117 | ✅ | ✅ | ✅ | ✅ |
| **hra** | +0.0152 | ✅ | ✅ | ✅ | ❌ |
| oft | -0.1 | ✅ | ✅ | ✅ | ❌ | *未校准* |
| boft | -0.08 | ✅ | ✅ | ✅ | ❌ | *未校准* |

---

## 4. PEFTPlanner — 规划器核心

### 4.1 决策流程

`plan()` 方法的三阶段决策流程：

```
┌─────────────────────────────────────────────────────────────┐
│ Phase 1: 回归主导评估 —— 对所有兼容变体计算预测 ΔmAP            │
│   → 生成 variant_scores 字典                                 │
├─────────────────────────────────────────────────────────────┤
│ Phase 2: 硬安全护栏 —— 无条件拦截已知灾难性组合                  │
│   Guardrail A: DoRA + attention-rich (φ_attn > 0.3) → downgrade to LoRA
│   Guardrail B: RT-DETR-like (φ_attn > 0.7) + LoRA-family → REFUSE
├─────────────────────────────────────────────────────────────┤
│ Phase 3: 回归主导决策                                         │
│   → 不兼容 → ADAPT (推荐最佳变体) 或 REFUSE                   │
│   → 预测灾难 (Δ < -0.05) → ADAPT 或 REFUSE                   │
│   → Attention-rich → rank cap (8) + safe attention           │
│   → CNN → 基于参数量 tiered rank cap                          │
│   → 无实质变化 → ACCEPT                                      │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 关键方法签名

```python
class PEFTPlanner:
    DEFAULT_COEFFS: Tuple[float, ...] = (
        0.0656, 0.0026, 0.0, 0.0054, 1.0,  # 原始 5 维
        0.0, 0.0, 0.0, 0.0, 0.0,           # 扩展 5 维
        0.0,                               # log(r)
    )
    REFUSE_THRESHOLD: float = -0.05

    def __init__(
        self,
        calibration_data: Optional[Path] = None,
        audit_dir: Optional[Path] = None,
        lovo_collector: Optional[LOVODataCollector] = None,
        lovo_validator: Optional[LOVOValidator] = None,
    )

    def fit(
        self,
        history: List[Tuple[ArchitectureFingerprint, str, float]],
        ranks: Optional[List[int]] = None,
    ) -> None
    # 使用 np.linalg.lstsq 拟合 11 维回归系数

    def predict(
        self,
        fingerprint: ArchitectureFingerprint,
        variant: str,
        rank: int = 8,
    ) -> float
    # 返回预测的 ΔmAP

    def plan(self, model: nn.Module, config: LoRAConfig) -> PlacementDecision
    # 主决策入口

    def plan_variant(
        self, model: nn.Module, variant: str, rank: int
    ) -> PlacementDecision
    # 快捷决策：指定变体 + 秩

    def detect_targets(
        self, model: nn.Module, config: Optional[Any] = None
    ) -> List[str]
    # 架构感知的 target 模块检测
```

### 4.3 安全护栏详解

**Guardrail A — DoRA on Attention-Rich Architectures**
- 触发条件：`variant == "dora"` 且 `phi_attn > 0.3`
- 动作：降级为 LoRA (`recommended_variant = "lora"`)
- 依据：论文 Fig. 4，YOLO12n DoRA 6/7 灾难率

**Guardrail B — RT-DETR + LoRA-Family（无条件）**
- 触发条件：`phi_attn > 0.7` 且 `variant in (lora, dora, loha, lokr)`
- 动作：**直接 REFUSE**
- 特殊性：**即使 LOVO 已拟合也触发**，防止回归未见过足够灾难数据时的误判
- 依据：论文 Fig. 4，RT-DETR-l LoRA-family 7/7 灾难率

**Rank Cap 策略**

| 架构类型 | 条件 | Rank 上限 | 说明 |
|----------|------|-----------|------|
| Attention-rich | `phi_attn > 0.3` 且 `rank > 8` | 8 | 防止注意力层 destabilization |
| CNN (大模型) | `phi_attn < 0.05`, `params > 50M` | 32 | 防止 consumer GPU OOM |
| CNN (中模型) | `phi_attn < 0.05`, `params > 10M` | 64 | 平衡内存与表达能力 |
| CNN (小模型) | `phi_attn < 0.05`, `params <= 10M` | 无上限 | 资源充足 |

### 4.4 决策审计

每次 `plan()` 调用自动生成 `DecisionAudit` 记录：
- 保存路径：`runs/planner_audit/planner_audit_YYYYMMDD_HHMMSS.json`
- 自动轮转：最多保留 100 个审计文件
- 包含：时间戳、指纹、请求配置、决策状态、推荐配置、预测 ΔmAP

---

## 5. LOVO 交叉验证

LOVO (Leave-One-Variant-Out) 是验证回归模型泛化能力的核心机制。

### 5.1 流程

```
对于每个唯一的数据点 (fingerprint, variant, ΔmAP):
    1. 将该点留出
    2. 用剩余数据拟合回归系数
    3. 预测留出点的 ΔmAP
    4. 记录 (actual, predicted, variant)
最终拟合：用全部数据拟合最终系数
```

### 5.2 LOVOValidator API

```python
from ultralytics.utils.lora.planner import LOVOValidator, LOVODataCollector, LOVODataPoint

# 构建数据集
collector = LOVODataCollector()
collector.add(LOVODataPoint(
    fingerprint=fingerprint,
    variant="lora",
    delta_mAP=0.0626,
    model_name="yolo12s",
    dataset="coco128",
))

# 运行 LOVO 交叉验证
validator = LOVOValidator(threshold=-0.05)
result = validator.validate(collector)

print(f"R²={result.lovo_r2:.3f}, RMSE={result.lovo_rmse:.3f}")
print(f"Coefficients: {result.coefficients}")

# 灾难检测评估
cat_metrics = validator.evaluate_catastrophe_detection(collector)
print(f"Precision={cat_metrics['precision']:.3f}, Recall={cat_metrics['recall']:.3f}")

# 完整报告
report = validator.full_report(collector)
```

### 5.3 论文校准指标

| 指标 | 论文值 | 说明 |
|------|--------|------|
| LOVO R² | ~0.870 | 10 个标准数据点 |
| 灾难检测 Recall | 0.944 | Table 2 |
| 灾难检测 F1 | 0.850 | Table 2 |
| 决策准确率 | 86.7% | Table 2 |

---

## 6. 使用示例

### 6.1 基础规划

```python
from ultralytics import YOLO
from ultralytics.utils.lora.planner import PEFTPlanner
from ultralytics.utils.lora.config import LoRAConfig

model = YOLO("yolov8n.pt")
config = LoRAConfig(r=16, alpha=32)

planner = PEFTPlanner()
decision = planner.plan(model.model, config)

print(f"Status: {decision.status}")
print(f"Predicted ΔmAP: {decision.predicted_delta:.4f}")
if decision.status == "ADAPT":
    print(f"Recommended variant: {decision.recommended_variant}")
    print(f"Recommended rank: {decision.recommended_rank}")
    print(f"Safety overrides: {decision.safety_overrides}")
elif decision.status == "REFUSE":
    print(f"Refusal reason: {decision.refusal_reason}")
    # 回退到 Full-SFT
```

### 6.2 带 LOVO 自动校准的规划

```python
from ultralytics.utils.lora.planner import (
    PEFTPlanner, LOVODataCollector, LOVOValidator,
    ArchitectureFingerprint
)

# 加载历史数据
collector = LOVODataCollector.load("runs/lovo_data.json")

# 创建带 LOVO 的 Planner
planner = PEFTPlanner(
    lovo_collector=collector,
    lovo_validator=LOVOValidator(threshold=-0.05),
)

# 第一次 plan() 自动调用 fit()
decision = planner.plan(model.model, config)
# 日志输出: [Planner] LOVO R²=0.873, RMSE=0.012, n=15
```

### 6.3 架构感知 Target 检测

```python
# 替代 LoRAConfigBuilder.auto_detect_targets()
targets = planner.detect_targets(model.model, config)
print(f"Detected {len(targets)} target modules")
# YOLO11s-like: 仅 conv 层
# YOLO12s-like: conv + 安全 attention（排除 qkv/proj/pe）
# RT-DETR-like: 空列表（拒绝）
```

---

## 7. 完整类索引

| 类/函数 | 职责 | 文件位置 |
|---------|------|----------|
| `ArchitectureFingerprint` | 10 维架构指纹计算 | `utils/lora/planner.py:40` |
| `PEFTVariantProfile` | PEFT 变体系数与兼容性 | `utils/lora/planner.py:362` |
| `PlacementDecision` | 决策结果数据结构 | `utils/lora/planner.py:471` |
| `DecisionAudit` | 决策审计记录（JSON 持久化） | `utils/lora/planner.py:512` |
| `LOVODataPoint` | 单个训练数据点 | `utils/lora/planner.py:607` |
| `LOVODataCollector` | 数据收集、序列化、过滤 | `utils/lora/planner.py:684` |
| `LOVOValidationResult` | LOVO 验证结果 | `utils/lora/planner.py:778` |
| `LOVOValidator` | LOVO 交叉验证引擎 | `utils/lora/planner.py:829` |
| `PEFTPlanner` | 主规划器 | `utils/lora/planner.py:1036` |
| `is_planner_enabled()` | 检查配置是否启用 Planner | `utils/lora/planner.py:1760` |
| `RefusalError` | 拒绝异常（应捕获后回退 Full-SFT） | `utils/lora/planner.py:30` |

---

## 8. 设计决策与最佳实践

### 8.1 为什么回归主导而非纯规则

- **纯规则系统**：无法处理新架构组合，维护成本高
- **纯回归系统**：需要大量灾难数据才能学习危险模式
- **YOLO-Master 方案**：回归主导 + 无条件硬护栏，兼顾泛化性与安全性

### 8.2 何时启用 Planner

| 场景 | 建议 |
|------|------|
| 标准 YOLOv8/v9/v10/v11 CNN 模型 | 可选，通常 ACCEPT |
| YOLO12 / 含 Attention 的模型 | **强烈建议启用**，防止灾难性退化 |
| RT-DETR | Planner 会 REFUSE，节省实验时间 |
| YOLO-World / 多模态 | 启用，自动检测 text-fusion 目标层 |
| 全新架构（论文未覆盖） | 启用 + 收集 LOVO 数据 → 自校准 |

### 8.3 LOVO 数据收集策略

1. **初始阶段**：使用默认系数运行一批标准实验（n/s/m/l 规模 × 主要变体）
2. **收集数据**：记录每个实验的 (fingerprint, variant, ΔmAP)
3. **校准 Planner**：将数据喂入 `LOVODataCollector`，Planner 自动拟合
4. **持续迭代**：新实验数据不断扩充 collector，提升预测精度
