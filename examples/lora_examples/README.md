# YOLO-Master-EsMoE-N LoRA 高效微调适配指南

本指南提供 YOLO-Master-EsMoE-N 在两个差异化垂类场景上的 LoRA 微调方案：
**VisDrone（密集航拍检测）** 和 **Brain Tumor（稀疏医疗检测）**。

---

## 场景说明

| 场景 | 数据集 | 类别数 | 训练图像 | 迁移特点 |
| :--- | :--- | :---: | :---: | :--- |
| **密集航拍检测** | VisDrone2019-DET | 10 | 6,471 | 小目标、严重尺度变化、密集遮挡 |
| **稀疏医疗检测** | brain-tumor | 2 | 893 | 每图少量框、灰度 MRI、小数据集 |

## 训练环境

| 项目 | 值 |
| --- | --- |
| Ultralytics | 8.3.240+ |
| Python | ≥3.10 |
| PyTorch | ≥2.0 |
| GPU 显存 | ≥24 GB (RTX 4090D 实测) |

## 配置文件说明

### 文件结构

```
examples/lora_examples/
├── yolo_master_visdrone_lora.yaml        # VisDrone LoRA 训练配置
├── yolo_master_brain_tumor_lora.yaml     # Brain Tumor LoRA 训练配置
├── run_lora_visdrone_sweep.sh            # VisDrone rank 扫描脚本
├── run_lora_brain_tumor_sweep.sh         # Brain Tumor rank 扫描脚本
└── README.md                             # 本指南
```

### 核心参数对比

| 参数 | VisDrone | Brain Tumor | 说明 |
| :--- | :---: | :---: | :--- |
| 默认 rank | 8 | 8 | 推荐值 |
| lora_alpha | r×4=32 | r×2=16 | VisDrone 需要更强更新信号 |
| imgsz | 640 | 640 | 统一输入尺寸 |
| epochs | 40 | 40 | 少样本快速迭代 |
| lr0 | 0.001 | 0.0005 | 航拍领域偏移大需高学习率 |
| close_mosaic | 10 | 0 | BT 小数据集提前关闭 mosaic |
| max_det | 1000 | 300 | 密集场景需更高检测上限 |
| workers | 4 | 4 | 避免 DDP 死锁 |

### LoRA 目标模块策略

```yaml
lora_target_modules: [
  "conv", "fused_conv", "bottleneck.0",
  "static_net.0", "static_net.3",
  "shared_feature.0", "shared_feature.3", "proj",
  "feature_refiner.0",
  "pool_projections.0.0", "pool_projections.1.0",
  "expert_projections.0.0",  # MoE Expert #0
  "expert_projections.1.0",  # MoE Expert #1
  ...
  "expert_projections.15.0", # MoE Expert #15
]
```

覆盖 131 个 Conv2d 模块 + 16 个 MoE Expert 投影 + 检测头。选择策略：
- 全量 Conv2d 层：适应通用特征提取
- MoE Expert 投影：适配专家特征分布
- 检测头 (lora_include_head=True)：适配输出分布

### MoE 路由层策略

| 场景 | 策略 | 理由 |
| :--- | :--- | :--- |
| **VisDrone** | `lora_exclude_modules: []` **包含路由层** | 航拍场景领域偏移大，expert 路由分配需调整 |
| **Brain Tumor** | `lora_exclude_modules: ["router","routing"]` **排除路由层** | 医疗特征分布稳定，修改路由无收益 |

## 实验结果

### 完整对比表格

| 场景 | r | Epochs | mAP50-95 | mAP50 | Precision | Recall | 训练时间(m) | GPU显存 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Brain Tumor** | 4 | 40 | 0.2660 | 0.436 | 0.442 | 0.682 | 32.4 | ~4 GB |
| **Brain Tumor** | 8 | 40 | **0.3039** | 0.465 | 0.419 | 0.716 | 29.8 | ~4 GB |
| **Brain Tumor** | 16 | 40 | **0.3198** | 0.489 | 0.456 | 0.720 | 31.2 | ~4 GB |
| **VisDrone** | 4 | 40 | 0.0586 | 0.109 | 0.600 | 0.123 | 71.1 | ~7 GB |
| **VisDrone** | 8 | 40 | 0.0632 | 0.117 | 0.596 | 0.129 | 71.1 | ~7 GB |
| **VisDrone** | 16 | 40 | **0.0669** | 0.120 | 0.606 | 0.154 | 74.4 | ~7 GB |

### 关键发现

1. **Brain Tumor**: LoRA 有效适配医疗检测（mAP50-95 > 0.30）。**r=8 性价比最优**（比 r=16 参数少30%，仅差5%）
2. **VisDrone**: rank 越高效果越好，**r=16 最佳**（0.0669 mAP50-95）。r=8 是性价比选择
3. **训练效率**: 使用 imgsz=640 在 RTX 4090D 上约 1.8 min/epoch

### 消融实验结论

| 消融变量 | 配置 | 效果 | 结论 |
| :--- | :--- | :---: | :--- |
| 路由层 LoRA | lora_exclude_modules=[] | 持平 | BT 场景无需路由适配 |
| RS-LoRA | lora_use_rslora=False | 持平 | 低 rank 时影响有限 |
| 注意力层 | lora_include_attention=True | 持平 | A2C2f 注意力对适配无贡献 |
| 梯度检查点 | lora_gradient_checkpointing=False | 持平 | 精度不受影响，但显存增加 |
| Alpha 4倍 | lora_alpha=32 | ↓0.009 | 过大更新信号导致不稳定 |

## 训练命令

### 方式一：Shell 脚本（推荐 — 串行 rank sweep）

```bash
# VisDrone rank 扫描 (r=4, 8, 16)
bash examples/lora_examples/run_lora_visdrone_sweep.sh

# Brain Tumor rank 扫描 (r=4, 8, 16)
bash examples/lora_examples/run_lora_brain_tumor_sweep.sh
```

### 方式二：单次训练

```bash
# VisDrone 推荐配置 (r=8)
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml

# 指定 rank
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml \
       lora_r=16 lora_alpha=64 epochs=40

# Brain Tumor 推荐配置 (r=8)
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml

# 指定 rank
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml \
       lora_r=8 lora_alpha=16 epochs=40
```

## 常见陷阱

### 1. 医疗灰度通道处理
Brain Tumor 数据集为灰度 MRI（单通道），YOLO 在预处理阶段自动复制为 3 通道（`img2rgb.py`），无需手动处理。

### 2. 航拍尺度变化
VisDrone 目标大小从 10px 到 500px 不等，建议：
- 默认 imgsz=640 平衡计算与效果
- 密集场景开启 `max_det=1000` 防止漏检
- 可实验性使用 `close_mosaic=10` 在后 30 轮不使用 mosaic

### 3. DDP 死锁
- 单卡训练时指定 `device: 0` 避免 DDP 多进程死锁
- `workers: 4` 为安全值，`workers: 8` 可能引起数据加载阻塞

### 4. 梯度检查点
大 rank 配置必须开启 `lora_gradient_checkpointing=True`，否则 r≥16 可能 OOM

### 5. 峰值显存估算
| 配置 | 显存 | 说明 |
| :--- | :---: | :--- |
| VisDrone r=4/8/16 | ~7 GB | 24GB 显卡可同时跑 batch=8 |
| Brain Tumor r=4/8/16 | ~4 GB | 24GB 显卡可同时跑 batch=16 |

## 最佳实践推荐

### Brain Tumor
- **推荐 rank**: r=8（mAP50-95=0.3039，性价比最优）
- **路由层**: 排除（`lora_exclude_modules: ["router","routing"]`）
- **注意**: 小数据集 epoch=40 足够，过早停止可能欠拟合

### VisDrone
- **推荐 rank**: r=16（mAP50-95=0.0669，效果最佳）或 r=8（性价比最优）
- **路由层**: 包含（`lora_exclude_modules: []`）
- **注意**: 航拍领域偏移显著，需更高学习率（lr0=0.001）和更强的 LoRA 信号（alpha=r×4）
