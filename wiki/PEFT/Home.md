# PEFT — Parameter-Efficient Fine-Tuning

YOLO-Master 的 PEFT 体系提供从标准 LoRA 到 Mixture-of-LoRA (MoLoRA) 的完整参数高效微调能力，并配备架构条件化的自动部署规划器（PEFT Planner）。

---

## 文档索引

| 文档 | 说明 | 适合读者 |
|------|------|----------|
| [LoRA 核心实现](PEFT_LoRA_Core.md) | LoRAConfig、apply_lora、双后端架构 | 理解底层实现 |
| [Mixture-of-LoRA](PEFT_MoLoRA.md) | MoLoRA 配置、路由、持续学习 | 使用 MoLoRA 进行微调 |
| [训练策略与 IO](PEFT_Training_IO.md) | LoraTrainingStrategy、Save/Load/Merge | 训练流程与模型管理 |
| [PEFT Planner](PEFT_Planner.md) | 架构条件化规划、LOVO 验证、三态决策 | 自动配置与安全性保障 |

---

## 快速选型指南

| 场景 | 推荐方案 | 关键参数 |
|------|----------|----------|
| **标准数据集微调** | LoRA (PEFT Backend) | `r=8, alpha=16` |
| **MPS / 无 PEFT 环境** | LoRA (Fallback Backend) | `use_fallback=True` |
| **多域持续学习** | MoLoRA | `num_experts=8, top_k=2` |
| **小样本检测** | FewShotLoRAConv | `drop_connect_schedule` |
| **未知架构安全性** | LoRA + PEFT Planner | `planner_enabled=True` |
| **与 MoE 协同** | MoLoRA + MoE | `share_moe_registry=True` |

---

## LoRA 快速开始

### 命令行训练

```bash
yolo detect train model=yolov8n.pt data=coco128.yaml \
  lora_r=8 lora_alpha=16 epochs=100
```

### 程序化使用

```python
from ultralytics import YOLO
from ultralytics.utils.lora.config import LoRAConfig

model = YOLO("yolov8n.pt")
config = LoRAConfig(r=8, alpha=16)
model = model.apply_lora(config)
model.train(data="coco128.yaml", epochs=100)
```

### 权重合并（零推理开销）

```python
from ultralytics.utils.lora import merge_lora_weights

merge_lora_weights(model, output_path="merged.pt")
# 合并后等价于标准 Conv，支持 ONNX/TensorRT 导出
```

---

## MoLoRA 快速开始

### 命令行训练

```bash
yolo detect train model=yolov8n.pt data=coco128.yaml \
  molora_num_experts=4 molora_top_k=2 molora_r=8 molora_alpha=16 \
  epochs=100
```

### 程序化使用

```python
from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAConfig, get_peft_molora_model

model = YOLO("yolov8n.pt").model
cfg = MoLoRAConfig(r=8, alpha=16, num_experts=4, top_k=2)
model = get_peft_molora_model(model, cfg)
```

### 多域持续学习

```python
from ultralytics.nn.peft.molora import allocate_domain_experts

# 将 8 个专家分配给 4 个域
alloc = allocate_domain_experts(8, ["day", "night", "fog", "rain"])
# {"day": [0,1], "night": [2,3], "fog": [4,5], "rain": [6,7]}

cfg = MoLoRAConfig(num_experts=8, top_k=2, domain_experts=alloc)
wrapper = MoLoRAModel(model, cfg)

# 训练 day 域
wrapper.set_domain("day")
wrapper.model.train()

# 冻结 day 专家，训练 night
wrapper.freeze_experts([0, 1])
wrapper.set_domain("night")
```

---

## PEFT Planner 快速开始

### 启用 Planner

```python
from ultralytics import YOLO
from ultralytics.utils.lora.planner import PEFTPlanner
from ultralytics.utils.lora.config import LoRAConfig

model = YOLO("yolo12s.pt")  # Attention-rich 架构
config = LoRAConfig(r=16, alpha=32, lora_planner_enabled=True)

planner = PEFTPlanner()
decision = planner.plan(model.model, config)

# Attention-rich 架构 + 高 rank → ADAPT（rank cap 到 8）
print(decision.status)  # "ADAPT"
print(decision.recommended_rank)  # 8
print(decision.safety_overrides)  # {"r": 8, "include_attention": True}
```

### RT-DETR 安全拒绝

```python
model = YOLO("rtdetr-l.pt")
config = LoRAConfig(r=8, alpha=16)

decision = planner.plan(model.model, config)
print(decision.status)  # "REFUSE"
print(decision.refusal_reason)
# "RT-DETR-like architecture (φ_attn=0.85): LoRA-family variants destabilize..."

# 回退到 Full-SFT
```

---

## 架构总览

```
PEFT 体系
├── LoRA 核心 (ultralytics/utils/lora/)
│   ├── api.py          — apply_lora() 主入口、LoRADetectionModel
│   ├── config.py       — LoRAConfig、LoRAConfigBuilder（自动 target 检测）
│   ├── fallback.py     — FewShotLoRAConv、ManualLoRAConv、PeftProxy
│   ├── io.py           — save/load/merge adapters
│   ├── training.py     — LoraTrainingStrategy（4 种策略）
│   └── planner.py      — PEFTPlanner、ArchitectureFingerprint、LOVOValidator
│
└── MoLoRA 扩展 (ultralytics/nn/peft/molora/)
    ├── config.py       — MoLoRAConfig、预设工厂
    ├── layer.py        — MoLoRAExpert、MoLoRALayer（Merge/Unmerge）
    ├── router.py       — LinearRouter、SpatialRouter、HybridRouter
    ├── loss.py         — MoLoRALoss（balance + z-loss + diversity）
    ├── model.py        — MoLoRAModel（域管理、专家冻结、回放）
    └── utils.py        — 工具函数
```

---

## 双后端架构

| 特性 | PEFT Backend | Fallback Backend |
|------|-------------|------------------|
| 依赖 | 需要 `peft` 库 | 纯 PyTorch |
| 适用场景 | 标准环境 | MPS / 边缘设备 / 无 peft 环境 |
| 实现 | `PeftProxy` (PeftModel) | `ManualLoRAConv` |
| 兼容性 | 全功能 | 核心 LoRA 功能 |
| 自动选择 | `select_lora_backend()` 自动判断 | — |
