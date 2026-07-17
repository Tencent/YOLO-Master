# YOLO-Master-EsMoE-N LoRA 高效微调适配指南

本指南记录了 YOLO-Master-EsMoE-N 在两个截然不同的垂类场景上的 LoRA 微调实验，覆盖了配置说明、rank 扫描结果、最佳推荐以及常见陷阱。

## 场景概览

| 场景 | 数据集 | 迁移特点 | 配置文件 |
| :--- | :--- | :--- | :--- |
| **密集航拍检测** | `VisDrone.yaml` | 大量小目标、严重尺度变化、拥挤场景 | `yolo_master_visdrone_lora.yaml` |
| **稀疏医疗检测** | `brain-tumor.yaml` | 每图少量框、灰度 MRI 信号、小数据集 | `yolo_master_brain_tumor_lora.yaml` |

两个配置文件均覆盖 issue 要求的全部 LoRA 控制参数：`lora_r`、`lora_alpha`、`lora_use_rslora`、`lora_target_modules`、`lora_include_attention`、`lora_gradient_checkpointing`。

## 运行环境

| 项目 | 值 |
| --- | --- |
| Ultralytics | `8.3.240` |
| Python | `3.12.13` |
| PyTorch | `2.10.0+cu128` |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| CUDA 显存 | 15,848 MiB |

## 仓库布局

```text
examples/lora_examples/
  yolo_master_visdrone_lora.yaml       # VisDrone LoRA 训练配置
  yolo_master_brain_tumor_lora.yaml    # Brain Tumor LoRA 训练配置
  yolo_master_lora_README.md           # 本指南
  yolo_master_lora_results.csv         # 完整六轮实验结果
  run_lora_visdrone_sweep.sh           # VisDrone rank 扫描脚本 (bash)
  run_lora_brain_tumor_sweep.sh        # Brain Tumor rank 扫描脚本 (bash)
  run_yolo_master_lora_rank_sweep.py   # 统一 rank 扫描脚本 (Python)

runs/lora_examples/
  brain_tumor_r4/  brain_tumor_r8/  brain_tumor_r16/
  visdrone_r4/     visdrone_r8/     visdrone_r16/
```

## 实验设置

| 数据集 | 数据配置 | Epochs | Batch | 图像尺寸 | 数据比例 | 优化器 | AMP | 项目目录 |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Brain Tumor | `brain-tumor.yaml` | 40 | 16 | 640 | 1.0 | `auto` | 启用 | `runs/lora_examples` |
| VisDrone | `VisDrone.yaml` | 30 | 8 | 768 | 0.2 | `auto` | 启用 | `runs/lora_examples` |

## 配置文件关键差异

| 配置项 | VisDrone | brain-tumor | 说明 |
| :--- | :--- | :--- | :--- |
| 默认 rank | `8` | `4` | brain-tumor 数据量小，低 rank 即可；VisDrone 目标密集需更大容量 |
| Epochs | `30` | `40` | brain-tumor 数据集小，需要更多 epoch 收敛 |
| 数据比例 | `0.2` | `1.0` | VisDrone 全量训练资源消耗大，20% 子集模拟少样本场景 |
| 图像尺寸 | `768` | `640` | 更大分辨率帮助 VisDrone 小目标召回 |
| Batch size | `8` | `16` | VisDrone 大图 + 多目标需降低 batch |
| `close_mosaic` | `10` | `0` | brain-tumor 提前关闭 mosaic 增强稳定性 |
| `multi_scale` | `True` | `False` | VisDrone 多尺度应对航拍尺度变化；医疗数据避免额外噪声 |
| `max_det` | `1000` | `100` | 密集场景需要更高检测上限 |
| `lora_lr_mult` | `0.5` | `1.0` | VisDrone 使用较保守的 LoRA 学习率 |
| `lora_dropout` | `0.05` | `0.05` | 两者均使用 dropout 防止过拟合 |
| `lora_use_rslora` | `True` | `True` | 高 rank 时 RS-LoRA 提供更好的缩放稳定性 |
| `lora_include_attention` | `False` | `False` | 排除 A2C2f attention 路径保持稳定性 |
| `lora_gradient_checkpointing` | `True` | `True` | 两者均启用以减少显存压力 |
| Router/gating LoRA | 排除 | 排除 | 短时微调不应改变 expert 分配动态 |

## LoRA 目标模块策略

YOLO-Master v0.10 模型使用 `VisualEnhancedAdaptiveGateMoE` 模块。目标模块从实际 v0.10 模块名称中选择：

```yaml
lora_target_modules: [
  "conv", "fused_conv", "bottleneck.0", "shared_feature.0", "static_net.3", "proj",
  "expert_projections.0.0", "expert_projections.1.0", "expert_projections.2.0", "expert_projections.3.0",
  "expert_projections.4.0", "expert_projections.5.0", "expert_projections.6.0", "expert_projections.7.0",
  "expert_projections.8.0", "expert_projections.9.0", "expert_projections.10.0", "expert_projections.11.0",
  "expert_projections.12.0", "expert_projections.13.0", "expert_projections.14.0", "expert_projections.15.0"
]
```

### MoE 路由层策略

路由层和门控层被显式排除：

```yaml
lora_exclude_modules: ["router", "routing", "gate", "gating"]
```

> **理由：** 短时 VisDrone/Brain Tumor LoRA 微调应仅适配视觉和 expert 卷积层，不应改变 expert 分配动态。路由层 LoRA 会改变 expert 选择行为，而目标数据集在短时训练中没有足够样本稳定路由分布。路由 LoRA 应作为独立消融实验单独测试。

## 训练命令

### 方式一：Shell 脚本（推荐 — 串行执行便于资源对比）

```bash
# VisDrone rank 扫描 (r=4, 8, 16)
bash examples/lora_examples/run_lora_visdrone_sweep.sh

# Brain Tumor rank 扫描 (r=4, 8, 16)
bash examples/lora_examples/run_lora_brain_tumor_sweep.sh
```

### 方式二：Python 统一扫描脚本

```bash
# 单场景扫描
python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene brain_tumor --device 0
python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene visdrone --device 0

# 全部场景
python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene all --device 0

# 预览命令（dry-run）
python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene all --dry-run
```

### 方式三：手动单次训练

```bash
# VisDrone 单次 LoRA 训练 (r=8)
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml \
    lora_r=8 lora_alpha=16 device=0

# Brain Tumor 单次 LoRA 训练 (r=4)
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml \
    lora_r=4 lora_alpha=8 device=0

# 命令行覆盖参数
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml \
    lora_r=16 lora_alpha=32 epochs=50 batch=4 fraction=0.5
```

## 实验结果

### Brain Tumor 结果

| Run | Rank | LoRA 模块数 | 可训练参数 | Adapter 参数 | 最佳 epoch | mAP50 | mAP50-95 | 训练时间 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brain_tumor_r4` | 4 | 92 | 468,290 | 123,116 | 30 | 0.43492 | 0.28312 | 39.72 min | 3.95G |
| `brain_tumor_r8` | 8 | 94 | 596,782 | 251,608 | 35 | 0.46004 | 0.31215 | 39.84 min | 3.99G |
| `brain_tumor_r16` | 16 | 94 | 848,390 | 503,216 | 37 | 0.48212 | 0.34044 | 40.15 min | 4.03G |

### VisDrone 结果

| Run | Rank | LoRA 模块数 | 可训练参数 | Adapter 参数 | 最佳 epoch | mAP50 | mAP50-95 | 训练时间 | 峰值显存 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `visdrone_r4` | 4 | 92 | 469,850 | 123,116 | 27 | 0.04148 | 0.01670 | 52.68 min | 14.70G |
| `visdrone_r8` | 8 | 94 | 598,342 | 251,608 | 25 | 0.05547 | 0.02340 | 48.54 min | 14.60G |
| `visdrone_r16` | 16 | 94 | 849,950 | 503,216 | 27 | 0.07292 | 0.03197 | 48.96 min | 14.70G |

## 跨场景对比总结

| 数据集 | 最佳 Run | 最佳 mAP50 | 最佳 mAP50-95 | 峰值显存 |
| --- | --- | ---: | ---: | ---: |
| Brain Tumor | `brain_tumor_r16` | 0.48212 | 0.34044 | 4.03G |
| VisDrone | `visdrone_r16` | 0.07292 | 0.03197 | 14.70G |

> **注意：** VisDrone 使用 `fraction=0.2`，结果应视为部分数据的 LoRA 微调效果对比，不应作为完整 VisDrone benchmark 数据。

## Rank 推荐

### Brain Tumor（稀疏医疗检测）

- **推荐 rank：`r=16`**（当前最佳 mAP50-95）
- **备选 rank：`r=8`**（更快迭代，精度损失约 8% mAP50-95）
- rank 从 8 到 16 的提升有限但可测量，显存基本持平（3.99G → 4.03G）
- 小数据集上 rank 过低（r=4）容量不足；rank 过高（>16）可能过拟合

### VisDrone（密集航拍检测）

- **推荐 rank：`r=16`**（当前最佳 mAP50-95）
- rank 提升带来的收益在密集小目标场景更明显（r=16 的 mAP50 是 r=4 的 1.76 倍）
- 更大 rank（如 r=32）可能进一步提升，但需权衡训练时间和显存
- 如需更快迭代速度，r=8 是合理的折中选择

> **通用建议：** 保持 `lora_alpha = 2 * lora_r`，启用 `lora_use_rslora=True` 以保证高 rank 时的缩放稳定性。

## 目标模块选择建议

1. **从 Conv + MoE Expert 开始：** 覆盖 `conv`、`fused_conv`、`bottleneck.0`、`shared_feature.0`、`static_net.3`、`proj` 以及 `expert_projections.*`。这些模块覆盖了领域特定的特征变换，同时保留了路由策略。

2. **对 v0.10 使用正确的模块名：** YOLO-Master v0.10 使用 `VisualEnhancedAdaptiveGateMoE`，旧的 `ES_MOE` 特定目标（如 `pointwise`）无法匹配 v0.10 的 expert 模块。

3. **保持 `lora_include_attention=False`：** A2C2f attention 路径（`attn.qkv`、`attn.proj`、`attn.pe`）更敏感，应作为独立消融实验测试。

4. **排除路由和门控层：** 路由 LoRA 改变 expert 分配动态，应作为独立消融实验单独报告，不应混入 rank 扫描。

5. **`lora_only_3x3=False`：** 许多 MoE projection 和 expert 路径是 1x1 卷积，需要被包含。

6. **检查日志确认实际目标：** 每次运行后检查 `Final Targets Passed to PEFT` 日志行，确认 YAML 目标列表被正确展开为最终模块名称。

## 常见陷阱

### 医疗灰度图像处理

- 许多 MRI 导出是单通道或灰度 RGB。确认数据加载器一致地将图像转换为模型期望的 3 通道输入
- 验证预处理不会在训练集和验证集之间以不同方式复制或归一化通道
- 调试时可禁用色彩增强（HSV 扰动等），检查 `train_batch*.jpg` 后再信任指标

### 稀疏医疗数据过拟合

- brain-tumor 每图框数少，视觉多样性有限
- 冻结 BN (`lora_freeze_bn=True`)、使用 dropout、排除路由 LoRA 有助于避免记忆扫描仪或标注伪影
- 如果出现 NaN 或 fitness 崩溃，降低 `lr0` 或 `lora_lr_mult`，增加 warmup，使用新的输出名称重新运行
- `close_mosaic=0` 提前关闭 mosaic 增强小数据集稳定性

### 航拍尺度变化

- VisDrone 目标可能极小且密集分布
- 使用更大的验证 `max_det`（1000），保持 `imgsz` 在 rank 扫描间一致
- 避免比较使用不同数据比例的 rank 结果
- `multi_scale=True` 帮助应对尺度变化，但会增加显存和训练时间

### 路由消融实验

- 如果从 `lora_exclude_modules` 中移除 `router`、`routing`、`gate` 或 `gating`，必须作为独立实验运行
- 监控验证 mAP、MoE balance loss 和 expert 使用分布
- 训练 loss 可能改善但路由漂移可能损害验证性能

### 指标可比性

- 跨 rank 对比前，保持 epochs、数据比例、图像尺寸、batch size、seed 和硬件不变
- VisDrone 的 shell 脚本采用串行执行以确保资源测量可比
- VisDrone 结果使用 `fraction=0.2`，不应用于完整 benchmark 对比

## 完整 CSV 数据

完整的六轮实验对比表格存储在：

```text
examples/lora_examples/yolo_master_lora_results.csv
```

该 CSV 每行记录一次运行的详细信息，包括：
- LoRA 设置、参数计数、训练成本
- 训练/验证 loss（box、cls、dfl、MoE）
- 精度、召回率、mAP50、mAP50-95
- 各优化器参数组的学习率
- 最后一个 epoch 的 mAP 用于与最佳 epoch 对比

---

*本指南为 2026 犀牛鸟开源人才培养活动 Issue #50 的交付物之一。*
