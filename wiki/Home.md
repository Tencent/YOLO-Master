# YOLO-Master Wiki

欢迎来到 YOLO-Master 技术 Wiki。本文档库从源码级别系统整理了项目中 **Mixture-of-Experts (MoE)** 与 **Parameter-Efficient Fine-Tuning (PEFT)** 两大核心技术体系的设计原理、模块实现与使用指南。

---

## 📚 文档导航

### 🔷 MoE — Mixture-of-Experts

稀疏专家混合架构，为 YOLO 检测模型提供动态计算能力与参数量扩展。

| 文档 | 内容 |
|------|------|
| [MoE 总览](MoE/Home.md) | MoE 体系全景与快速导航 |
| [核心模块详解](MoE/MoE_Core_Modules.md) | 40+ MoE 变体模块（ES_MOE → VisualEnhancedAdaptiveGateMoE） |
| [路由与专家模块](MoE/MoE_Routers_Experts.md) | 路由器（7种）与专家网络（7种）详解 |
| [训练、损失与剪枝](MoE/MoE_Training_Loss_Pruning.md) | MoELoss、动态调度器、专家剪枝 |
| [诊断与分析工具](MoE/MoE_Diagnostics_Analysis.md) | 路由崩溃检测、专家使用追踪、诊断持久化 |
| [Mixture of Attention](MoE/Mixture_of_Attention.md) | MoA 模块与版本演进路线 |
| [版本演进指南](MoE/MoE_Modules_Explanation.md) | v0.1 → v0.15 演进历史与部署建议（中文） |
| [Version Evolution (EN)](MoE/MoE_Modules_Explanation_EN.md) | English version of the evolution guide |

### 🔶 PEFT — Parameter-Efficient Fine-Tuning

参数高效微调体系，支持 LoRA、MoLoRA 及架构条件化自动配置。

| 文档 | 内容 |
|------|------|
| [PEFT 总览](PEFT/Home.md) | PEFT 体系全景与快速导航 |
| [LoRA 核心实现](PEFT/PEFT_LoRA_Core.md) | LoRAConfig、apply_lora、双后端架构、安全机制 |
| [Mixture-of-LoRA](PEFT/PEFT_MoLoRA.md) | MoLoRA 配置、路由、专家层、持续学习、Merge/Unmerge |
| [训练策略与 IO](PEFT/PEFT_Training_IO.md) | LoraTrainingStrategy、Save/Load/Merge、Fallback |
| [PEFT Planner](PEFT/PEFT_Planner.md) | 架构条件化部署规划器、LOVO 交叉验证、三态决策 |

---

## 🏗️ 架构概览

```
YOLO-Master
├── ultralytics/nn/modules/moe/      # MoE 核心实现（~7500 行）
│   ├── modules.py                   # 40+ MoE 变体
│   ├── routers.py                   # 7 种路由器
│   ├── experts.py                   # 7 种专家网络
│   ├── loss.py                      # MoELoss + 辅助损失
│   ├── pruning.py                   # 专家剪枝
│   ├── scheduler.py                 # 动态调度器
│   ├── analysis.py                  # 专家使用追踪与崩溃检测
│   ├── diagnostics.py               # 轻量级诊断快照
│   ├── history.py                   # 诊断持久化与绘图
│   └── utils.py                     # BatchedExpertComputation 等工具
│
├── ultralytics/nn/modules/moa/      # Mixture of Attention（~730 行）
│   └── moa.py                       # MoABlock, C2fMoA, NeckMoAFusion
│
├── ultralytics/nn/peft/molora/      # Mixture-of-LoRA（~1500 行）
│   ├── config.py                    # MoLoRAConfig / ConfigBuilder
│   ├── layer.py                     # MoLoRAExpert / MoLoRALayer
│   ├── router.py                    # LinearRouter / SpatialRouter / HybridRouter
│   ├── loss.py                      # MoLoRALoss
│   ├── model.py                     # MoLoRAModel 包装器
│   └── utils.py                     # 工具函数
│
└── ultralytics/utils/lora/          # LoRA 核心与 PEFT Planner（~6000 行）
    ├── api.py                       # apply_lora, LoRADetectionModel
    ├── config.py                    # LoRAConfig / LoRAConfigBuilder
    ├── fallback.py                  # FewShotLoRAConv, ManualLoRAConv, PeftProxy
    ├── io.py                        # save/load/merge adapters
    ├── training.py                  # LoraTrainingStrategy
    └── planner.py                   # PEFTPlanner, LOVOValidator, ArchitectureFingerprint
```

---

## 🚀 快速开始

### MoE 快速配置

```yaml
# yolo-master.yaml 中配置 MoE 层
- [-1, 1, HybridAdaptiveGateMoE, [512, 8, 2]]  # 8 专家选 2，输出 512 通道
```

### PEFT 快速配置

```python
from ultralytics import YOLO
from ultralytics.utils.lora.config import LoRAConfig

model = YOLO("yolov8n.pt")
config = LoRAConfig(r=8, alpha=16)
model = model.apply_lora(config)
model.train(data="coco128.yaml", epochs=100)
```

### MoLoRA 快速配置

```bash
yolo detect train model=yolov8n.pt data=coco128.yaml \
  molora_num_experts=4 molora_top_k=2 molora_r=8 \
  epochs=100
```

---

## 📖 相关文档

- [README.md](/README.md) — 项目主文档
- [README_CN.md](/README_CN.md) — 中文主文档
- [CONTRIBUTING.md](/CONTRIBUTING.md) — 贡献指南
- [docs/molora_guide.md](/docs/molora_guide.md) — MoLoRA 使用指南（源码级）
