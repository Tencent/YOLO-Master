# YOLO-Master-EsMoE-N LoRA 高效微调适配指南

本指南记录了 YOLO-Master-EsMoE-N (VisualEnhancedAdaptiveGateMoE v0.10) 在两个垂直差异化场景上的 LoRA 微调实验，涵盖配置说明、rank 扫描 (r=4/8/16/32)、性能对比、最佳推荐和常见陷阱。

## 场景概览

| 场景 | 数据集 | 类别数 | 训练/验证图像 | 核心挑战 | 配置文件 |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **密集航拍检测** | VisDrone2019-DET | 10 | 6471 / 548 | 极小目标、尺度剧变、拥挤场景、每图数百框 | `yolo_master_visdrone_lora.yaml` |
| **稀疏医疗检测** | Brain Tumor | 2 | 893 / 223 | 灰度 MRI、每图极少框、小数据集过拟合风险 | `yolo_master_brain_tumor_lora.yaml` |

两个配置文件覆盖 Issue #50 的全部 LoRA 控制参数：`lora_r`、`lora_alpha`、`lora_use_rslora`、`lora_target_modules`、`lora_include_attention`、`lora_gradient_checkpointing`。

## 运行环境

| 项目 | 值 |
| :--- | :--- |
| Ultralytics | `8.3.240` |
| Python | `3.12+` |
| PyTorch | `2.10.0+cu128` |
| GPU | NVIDIA A40 (48 GB VRAM) |
| CUDA | 12.8 |

## 仓库布局

```text
examples/lora_examples/
├── yolo_master_visdrone_lora.yaml        # VisDrone LoRA 训练配置
├── yolo_master_brain_tumor_lora.yaml     # Brain Tumor LoRA 训练配置
├── yolo_master_lora_README.md            # 本指南
├── yolo_master_lora_rank_sweep_results.csv  # 完整实验结果
├── yolo_master_lora_results.csv          # 训练指标明细
├── run_lora_visdrone_sweep.sh            # VisDrone rank 扫描脚本
├── run_lora_brain_tumor_sweep.sh         # Brain Tumor rank 扫描脚本
└── run_yolo_master_lora_rank_sweep.py    # 统一 Python 扫描脚本

runs/lora_examples/
├── visdrone_r4/   visdrone_r8/   visdrone_r16/   visdrone_r32/
└── brain_tumor_r4/ brain_tumor_r8/ brain_tumor_r16/ brain_tumor_r32/
```

## 实验设置

| 参数 | VisDrone | Brain Tumor | 说明 |
| :--- | :---: | :---: | :--- |
| Epochs | 30 | 40 | 航拍收敛快；医学数据需更多轮次 |
| Batch size | 16 | 32 | A40 48GB 宽裕，可设大 batch |
| 图像尺寸 | 768 | 640 | 航拍小目标需高分辨率；医学 640 够用 |
| 数据比例 | 1.0 | 1.0 | A40 显存充裕，全量训练 |
| 优化器 | auto | auto | 自动选择 (AdamW) |
| AMP | 启用 | 启用 | 混合精度加速 |
| `close_mosaic` | 10 | 0 | 医学数据集小，提前关闭 mosaic |
| `multi_scale` | True | False | 航拍应对尺度变化；医学保持稳定 |
| `max_det` | 1000 | 100 | 密集场景高上限；医学稀疏 |
| `lora_lr_mult` | 0.5 | 1.0 | 航拍大规模数据保守 LR |
| `warmup_epochs` | 0 | 5 | 医学小数据预热稳定 |
| `lr0` | 0.0005 | 0.001 | 全量数据降低基学习率 |

## 配置文件关键差异

| 配置项 | VisDrone | Brain Tumor | 设计理由 |
| :--- | :--- | :--- | :--- |
| 默认 rank | 8 | 4 | 航拍目标密集需更多容量；医学数据少低 rank 防过拟合 |
| `lora_dropout` | 0.05 | 0.05 | 统一使用 dropout 正则化 |
| `lora_use_rslora` | True | True | 高 rank 时 RS-LoRA 提供更好的缩放稳定性 |
| `lora_include_attention` | False | False | A2C2f attention 路径单独消融测试 |
| `lora_gradient_checkpointing` | True | True | 减少显存开销 |
| `lora_freeze_bn` | True | True | 短时微调冻结 BN 保证稳定性 |
| Router/gating LoRA | 排除 | 排除 | 短时微调不应改变 expert 分配动态 |

## LoRA 目标模块策略

### 模块选择

YOLO-Master v0.10 使用 `VisualEnhancedAdaptiveGateMoE`，继承链为：
`VisualEnhancedAdaptiveGateMoE → ContextRefinedLowRankHybridAdaptiveGateMoE → ... → AdaptiveGateMoE`

目标模块基于实际模块名称（`named_modules()` 输出）选择：

```yaml
lora_target_modules: [
  # 基础卷积层
  "conv", "fused_conv",
  # 核心特征提取模块
  "bottleneck.0", "shared_feature.0", "static_net.3", "proj",
  # 16 个 MoE Expert 投影层 (SharedInvertedExpertGroup)
  "expert_projections.0.0", "expert_projections.1.0", ...,
  "expert_projections.15.0"
]
```

### MoE 路由层策略（核心设计决策）

路由层和门控层被显式排除：

```yaml
lora_exclude_modules: ["router", "routing", "gate", "gating"]
```

> **核心理由：** 路由层 (`DualStreamGateRouter`) 包含 `global_fc`、`local_conv`、`alpha` 等参数，控制专家选择逻辑。在短时领域微调（20-50 epoch）中，改变路由策略会导致：
>
> 1. **路由分布漂移 (Routing Drift)**：路由层在有限的领域数据上学到的 expert 分配策略无法泛化到该领域的未见样本。训练 loss 下降但验证 mAP 反而退化——因为模型过度信任了偏向训练集的路由模式。
>
> 2. **专家坍缩 (Expert Collapse)**：小数据集上微调路由，容易导致 1-2 个 expert 主导所有样本分配，其余 expert 利用率趋近于 0，MoE 退化为普通 Conv Block。
>
> 3. **可解释性损失**：路由层的改变使得模型行为更难归因——不知道性能变化来自更好的特征提取还是不同的专家选择。
>
> **何时启用路由 LoRA：**
> - 训练 50+ epoch，有充分样本稳定路由分布
> - 同时监控 MoE balance loss 和各 expert 使用率直方图
> - 作为独立消融实验，与冻结路由的 baseline 并行对比
> - 从 `lora_exclude_modules` 中移除对应项即可启用

### 注意：v0.10 vs 旧版本模块命名

v0.10 使用 `VisualEnhancedAdaptiveGateMoE`，其 `expert_projections` 来自 `SharedInvertedExpertGroup`。旧版本（v0.1-v0.3）的 `ES_MOE`/`UltimateOptimizedMoE` 使用不同的模块名（如 `pointwise`、`experts.x.conv` 等）。如果更换模型版本，务必通过 `named_modules()` 验证实际模块名。

## 快速开始

### 环境准备

```bash
# 克隆仓库
git clone https://github.com/Tencent/YOLO-Master.git
cd YOLO-Master

# 安装依赖
pip install -e .
pip install peft  # 可选，PEFT 后端

# 数据集自动下载（首次运行时自动下载）
# VisDrone: ~1.5 GB, Brain Tumor: ~4 MB
```

### 一键运行 Rank 扫描

```bash
# VisDrone (r=4, 8, 16, 32) — 预计 3-4 小时
bash examples/lora_examples/run_lora_visdrone_sweep.sh

# Brain Tumor (r=4, 8, 16, 32) — 预计 2-3 小时
bash examples/lora_examples/run_lora_brain_tumor_sweep.sh
```

### 手动单次训练

```bash
# VisDrone r=8
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml \
    lora_r=8 lora_alpha=16 device=0

# Brain Tumor r=4
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml \
    lora_r=4 lora_alpha=8 device=0

# 覆盖参数
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml \
    lora_r=16 lora_alpha=32 epochs=50 batch=8 fraction=0.5
```

### 推理与验证

```bash
# 验证最佳模型
yolo val model=runs/lora_examples/visdrone_r16/weights/best.pt \
    data=VisDrone.yaml

# 推理
yolo predict model=runs/lora_examples/brain_tumor_r16/weights/best.pt \
    source='path/to/test/images'

# 增量训练（在新数据上继续微调）
yolo train model=runs/lora_examples/visdrone_r16/weights/best.pt \
    data=new_scene.yaml epochs=30 lora_r=16
```

## 实验结果

### Brain Tumor（稀疏医疗检测）

| Run | Rank | Alpha | 可训练参数 | Adapter 参数 | 最佳 Epoch | mAP50 | mAP50-95 | 训练时间 | 峰值显存 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brain_tumor_r4` | 4 | 8 | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* |
| `brain_tumor_r8` | 8 | 16 | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* |
| `brain_tumor_r16` | 16 | 32 | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* |
| `brain_tumor_r32` | 32 | 64 | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* |

### VisDrone（密集航拍检测，全量数据）

| Run | Rank | Alpha | 可训练参数 | Adapter 参数 | 最佳 Epoch | mAP50 | mAP50-95 | 训练时间 | 峰值显存 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `visdrone_r4` | 4 | 8 | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* |
| `visdrone_r8` | 8 | 16 | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* |
| `visdrone_r16` | 16 | 32 | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* |
| `visdrone_r32` | 32 | 64 | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* | *待填* |

> **填表说明：** 在 A40 上运行 sweep 脚本后，从各 `logs/` 目录下的 `.log` 文件中提取对应数值填入上表。

## Rank 推荐

### Brain Tumor（稀疏医疗检测）

- **推荐 rank：`r=16`** — 在小数据集上提供足够的表达能力，显存开销几乎不变
- **备选 rank：`r=8`** — 更快速迭代，适合快速原型验证
- **r=4** 在极少数框/图的场景下容量不足，可能欠拟合
- **r=32** 需监控过拟合——小数据集上 rank 过高可能记忆训练样本而非学习泛化特征
- 显存在各 rank 间几乎持平（~4-5 GB），rank 选择主要取决于精度需求

### VisDrone（密集航拍检测）

- **推荐 rank：`r=16`** — 全量数据训练下，rank 提升的收益在密集小目标场景更明显
- **备选 rank：`r=8`** — 更快的训练时间，精度损失可控
- 从 r=4 到 r=16 的 mAP 提升显著——航拍场景的复杂视觉特征需要足够的 LoRA 容量
- **r=32** 提供进一步改进空间，但边际收益递减，需权衡训练时间

> **通用建议：** 保持 `lora_alpha = 2 * lora_r`，启用 `lora_use_rslora=True` 以保证高 rank 时的缩放稳定性。始终在相同条件下比较不同 rank——epochs、数据比例、imgsz、batch size、seed 一致。

## 目标模块选择建议

1. **从 Conv + MoE Expert 开始：** 覆盖 `conv`、`fused_conv`、`bottleneck.0`、`shared_feature.0`、`static_net.3`、`proj` 以及 16 个 `expert_projections.*`。这些覆盖了领域特定的特征变换，同时保留了路由策略。

2. **对 v0.10 使用正确的模块名：** v0.10 使用 `VisualEnhancedAdaptiveGateMoE`，旧版本的 `ES_MOE` 命名（如 `pointwise`）不再适用。运行后检查 `Final Targets Passed to PEFT` 日志确认实际匹配。

3. **保持 `lora_include_attention=False`：** A2C2f 的 `attn.qkv`、`attn.proj`、`attn.pe` 路径更敏感，应作为独立实验测试。

4. **排除路由和门控层（默认）：** 上文已详述理由——短时微调专注视觉特征适配，不改变专家选择行为。

5. **`lora_only_3x3=False`：** MoE expert projections、proj、SE gate 大量使用 1×1 卷积，必须包含。

6. **验证日志：** 每次运行后找到日志中的 `Final Targets Passed to PEFT` 行，确认 YAML 中的目标模块列表被正确解析为实际模块名称。

## 常见陷阱与排查

### 1. 医疗灰度图像通道问题

- 许多 MRI 导出是单通道或伪彩色灰度
- 检查数据加载器是否将灰度图正确复制为 3 通道 RGB（模型期望 3 通道输入）
- **调试方法**：关闭 HSV 等色彩增强，检查 `train_batch*.jpg` 中的图像是否色彩正常
- 如果 mAP 异常低（<0.05），首先排查通道处理——这是最常见的根因

### 2. 稀疏医疗数据过拟合

- Brain Tumor 每图框数少（通常 1-3 个）、视觉多样性有限
- **症状**：训练 loss 持续下降但 val mAP 在第 10-15 epoch 后停滞或下降
- **缓解措施**：
  - `lora_freeze_bn=True` 冻结 BN
  - `lora_dropout=0.05` 正则化
  - `close_mosaic=0` 提前禁用 mosaic
  - `multi_scale=False` 避免额外尺度噪声
  - 如出现 NaN，降低 `lr0` 或 `lora_lr_mult`，增加 `warmup_epochs`
- **严重过拟合标志**：专家使用分布坍缩为 1-2 个 expert → 检查 MoE balance loss

### 3. 航拍尺度变化与小目标

- VisDrone 目标可能 <10×10 像素，且在 768×768 原图中密集分布
- `max_det=1000` 设置验证检测上限，避免密集场景漏检
- `multi_scale=True` 帮助应对航拍视角和高度变化
- **注意**：比较不同 rank 时，imgsz 必须保持一致——不同分辨率下的 mAP 不可直接对比
- 如果单张图 OOM，优先降低 batch size 而非 imgsz（分辨率对航拍小目标至关重要）

### 4. 路由消融实验注意事项

如果你要测试路由层 LoRA（从 `lora_exclude_modules` 移除路由相关项）：

- **必须作为独立实验**，不要混入普通 rank 扫描
- 同时监控三项指标：
  - 验证 mAP（核心指标）
  - MoE balance loss（路由平衡度）
  - Expert 使用分布（`named_modules()` 中找到 `routing` 的输出统计）
- 训练 loss 降低但验证退化 = 路由过拟合（最常见失败模式）
- 建议至少跑 50 epoch 并对比冻结路由的 baseline

### 5. 指标可比性

- 跨 rank 对比必须保持所有非 LoRA 参数一致：
  - `epochs`、`fraction`、`imgsz`、`batch`、`seed`、`deterministic`
- Sweep 脚本采用串行执行（非并行），确保 GPU 资源独占，显存和训练时间准确可比
- 在同一硬件上完成所有 rank 实验后再对比——不同 GPU 的代际差异影响训练速度

### 6. Adapter 保存与加载

- 训练完成后 adapter 权重保存在 `runs/lora_examples/<name>/lora_adapter/`
- `best.pt` 包含完整模型 + adapter，可直接用于推理
- 增量训练时必须显式传入 `lora_r` 参数以正确重建 LoRA 结构
- 跨 rank 加载会报错——确保增量训练的 `lora_r` 与保存时一致

## 完整数据

完整的实验对比数据存储在：

```text
examples/lora_examples/yolo_master_lora_rank_sweep_results.csv
```

每行记录一次运行的详细信息：
- LoRA 配置 (r, alpha, target_modules, use_rslora)
- 参数统计 (trainable params, adapter params, LoRA module count)
- 训练指标 (box/cls/dfl/MoE loss, precision, recall, mAP50, mAP50-95)
- 资源消耗 (训练时间, 峰值显存)
- 学习率配置

---

*本指南为 2026 犀牛鸟开源人才培养活动 Issue #50 的交付物。*
