<img width="960" height="180" alt="Image" src="https://github.com/user-attachments/assets/5d2ab671-cf2f-4697-9c1b-1dfe611111e3" />

<p align="center">
  <a href="https://huggingface.co/spaces/gatilin/YOLO-Master-WebUI-Demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces"></a>
  <a href="https://colab.research.google.com/drive/1gTKkCsE4sXIOWpu1cdNBjdFHEahBoZD0?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://arxiv.org/abs/2512.23273"><img src="https://img.shields.io/badge/arXiv-2512.23273-b31b1b.svg" alt="arXiv"></a>
  <a href="#-引用"><img src="https://img.shields.io/badge/CVPR-2026-6420AA.svg" alt="CVPR 2026"></a>
  <a href="https://github.com/Tencent/YOLO-Master/releases/tag/YOLO-Master-v26.02"><img src="https://img.shields.io/badge/%F0%9F%93%A6-Model%20Zoo-orange" alt="Model Zoo"></a>
  <a href="./LICENSE"><img src="https://img.shields.io/badge/License-AGPL%203.0-blue.svg" alt="AGPL 3.0"></a>
  <a href="https://github.com/ultralytics/ultralytics"><img src="https://img.shields.io/badge/Ultralytics-YOLO-blue" alt="Ultralytics"></a>
</p>

<p align="center">
  <b>YOLO-Master: <u>M</u>OE-<u>A</u>ccelerated with <u>S</u>pecialized <u>T</u>ransformers for <u>E</u>nhanced <u>R</u>eal-time Detection.</b>
</p>

<p align="center">
  <a href="https://github.com/isLinXu">Xu Lin</a><sup>1*</sup>, 
  <a href="https://pjl1995.github.io/">Jinlong Peng</a><sup>1*</sup>, 
  <a href="https://scholar.google.com/citations?user=fa4NkScAAAAJ">Zhenye Gan</a><sup>1</sup>, 
  <a href="https://scholar.google.com/citations?hl=en&user=cU0UfhwAAAAJ">Jiawen Zhu</a><sup>2</sup>, 
  <a href="https://scholar.google.com/citations?user=JIKuf4AAAAAJ&hl=zh-TW">Jun Liu</a><sup>1</sup>
  <br>
  <sup>1</sup><b>Tencent Youtu Lab</b> &nbsp;&nbsp; <sup>2</sup><b>Singapore Management University</b>
  <br>
  <sup>*</sup>Equal Contribution
</p>

<div align="center">
<h5>🎉 Accepted by CVPR 2026</h5>
</div>

---

<img
  width="224"
  alt="YOLO-Master Mascot"
  src="https://github.com/user-attachments/assets/bbf751ea-af27-465d-a8a9-7822db343638"
  align="left"
/>

`YOLO-Master` 是一个面向实时目标检测（RTOD）的 YOLO-like 框架，首次在通用数据集上将 **Mixture-of-Experts (MoE)** 深度融合进 YOLO 架构，通过 **Efficient Sparse MoE (ES‑MoE)** 与轻量级 **动态路由（Dynamic Routing）** 实现 **instance‑conditional adaptive computation**：让模型按场景复杂度“按需分配算力（compute-on-demand）”，在高精度与超低延迟之间取得更优平衡。  

**主要亮点：**
- **方法创新（ES‑MoE + Dynamic Routing）**：通过动态路由网络引导专家分工训练，并在推理时激活最相关专家，减少冗余计算并提升检测表现。
- **性能验证（精度 × 延迟）**：在 MS COCO 上，YOLO‑Master‑N 达到 **42.4% AP @ 1.62ms latency**；相较 YOLOv13‑N **+0.8% mAP 且 17.8% 更快**。
- **Compute‑on‑Demand**：从“静态致密计算”走向“按输入内容自适应分配算力”，密集/困难场景收益更显著。
- **开箱即用全流程**：提供安装、验证、训练、推理与导出（ONNX/TensorRT 等）完整链路。 
- **持续工程化演进**：包含 MoE 剪枝与分析工具（diagnose_model / prune_moe_model）、CW‑NMS、Sparse SAHI 推理模式等增强。 

---

## 💡 初心 (Introduction)

> **"探索 YOLO 中动态智能的前沿。"**

这项工作代表了我们对实时目标检测 (RTOD) 演进的热情探索。据我们所知，**YOLO-Master 是首个在通用数据集上将混合专家 (MoE) 架构与 YOLO 深度融合的工作。**

大多数现有的 YOLO 模型依赖于静态的密集计算——即对简单的天空背景和复杂的拥挤路口分配相同的计算预算。我们认为检测模型应该更加“自适应”，就像人类视觉系统一样。虽然这次初步探索可能并不完美，但它展示了 **高效稀疏 MoE (ES-MoE)** 在平衡高精度与超低延迟方面的巨大潜力。我们将致力于持续迭代和优化，以进一步完善这一方法。

展望未来，我们从 LLM 和 VLM 的变革性进步中汲取灵感。我们将致力于完善这一方法，并将这些见解扩展到基础视觉任务中，最终目标是解决更具雄心的前沿问题，如开放词汇检测和开放集分割。

<details>
  <summary>
  <font size="+1"><b>摘要 (Abstract)</b></font>
  </summary>
现有的实时目标检测 (RTOD) 方法通常采用类 YOLO 架构，因为它们在精度和速度之间取得了良好的平衡。然而，这些模型依赖于静态密集计算，对所有输入应用统一的处理，导致表示能力和计算资源的分配不当，例如在简单场景上过度分配，而在复杂场景上服务不足。这种不匹配导致了计算冗余和次优的检测性能。

为了克服这一限制，我们提出了 YOLO-Master，这是一种新颖的类 YOLO 框架，为 RTOD 引入了实例条件自适应计算。这是通过高效稀疏混合专家 (ES-MoE) 块实现的，该块根据场景复杂度动态地为每个输入分配计算资源。其核心是一个轻量级的动态路由网络，通过多样性增强目标指导专家在训练期间的专业化，鼓励专家之间形成互补的专业知识。此外，路由网络自适应地学习仅激活最相关的专家，从而在提高检测性能的同时，最大限度地减少推理过程中的计算开销。

在五个大规模基准测试上的综合实验证明了 YOLO-Master 的优越性。在 MS COCO 上，我们的模型实现了 42.4% 的 AP 和 1.62ms 的延迟，比 YOLOv13-N 高出 +0.8% mAP，推理速度快 17.8%。值得注意的是，在具有挑战性的密集场景中收益最为明显，同时模型在典型输入上保持了效率并维持了实时推理速度。代码: [Tencent/YOLO-Master](https://github.com/Tencent/YOLO-Master)
</details>

---

## 🎨 架构

<div align="center">
  <img width="90%" alt="YOLO-Master Architecture" src="https://github.com/user-attachments/assets/6caa1065-af77-4f77-8faf-7551c013dacd" />
  <p><i>YOLO-Master 引入 ES-MoE 块，通过动态路由实现“按需计算”。</i></p>
</div>

### 📚 深度文档
关于 MoE 模块的设计理念、路由机制详解以及针对不同硬件（GPU/CPU/NPU）的部署优化指南，请参阅我们的 Wiki 文档：
👉 **[Wiki: MoE 模块详解与演进](wiki/MoE_Modules_Explanation.md)**

## 📖 目录

- [初心](#-初心-introduction)
- [架构](#-架构)
- [更新](#-更新-latest-first)
- [新特性 (v2026.02)](#-新特性-v202602)
  - [混合专家 (MoE)](#1%EF%B8%8F⃣-混合专家-moe-支持)
  - [LoRA 微调](#2%EF%B8%8F⃣-lora-支持---参数高效微调)
  - [Sparse SAHI](#3%EF%B8%8F⃣-sparse-sahi-稀疏推理模式)
  - [聚类加权 NMS](#4%EF%B8%8F⃣-聚类加权-nms-cw-nms)
- [主要结果](#-主要结果)
  - [检测](#检测)
  - [分割](#分割)
  - [分类](#分类)
- [模型库](#-模型库与基准测试)
- [检测示例](#-检测示例)
- [支持的任务](#-支持的任务)
- [快速开始](#-快速开始)
  - [安装](#安装)
  - [验证](#验证)
  - [训练](#训练)
  - [推理](#推理)
  - [导出](#导出)
  - [Gradio 演示](#gradio-演示)
- [社区与贡献](#-社区与贡献)
- [许可证](#-许可证)
- [致谢](#-致谢)
- [引用](#-引用)

## 🚀 更新 (Latest First)
- **2026/02/21**: 🎉🎉 **我们的论文已被 CVPR 2026 接收！** 感谢所有贡献者和社区成员的支持！
- **2026/02/13**: 🧨🚀 为模型训练添加 LoRA 支持，并发布 [v2026.02 版本](https://github.com/Tencent/YOLO-Master/releases/tag/YOLO-Master-v26.02)。[新年快乐！]
- **2026/01/16**: [feature] 新增 MoE 模型剪枝与分析工具。
  > diagnose_model：可视化专家利用率与路由行为，用于识别冗余专家。
  > prune_moe_model：物理切除冗余专家并重构路由，无需重训即可实现高效推理。
- **2026/01/16**: 仓库 [isLinXu/YOLO-Master](https://github.com/isLinXu/YOLO-Master) 迁移到 [Tencent](https://github.com/Tencent/YOLO-Master) 组织下。
- **2026/01/14**: [ncnn-YOLO-Master-android](https://github.com/mpj1234/ncnn-YOLO-Master-android)为YOLO-Master提供部署，感谢贡献！
- **2026/01/09**: [feature] 新增Cluster-Weighted NMS (CW-NMS)来优化与平衡mAP和推理速度。
  > cluster: False # (bool) cluster NMS (MoE optimized)
- **2026/01/07**: [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) 为 YOLO-Master 提供加速，感谢贡献！
- **2026/01/07**: 新增MoE loss显式加入到training中
  > Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   **moe_loss**  Instances  Size
- **2026/01/04**: MoE模块重构
  > Split MoE script into separate modules (routers, experts)
- **2026/01/03**: [feature] 新增 Sparse SAHI 推理模式：通过全局粗筛生成的 Objectness Mask 实现内容自适应的稀疏切片推理，显著提升高分辨率图像中小目标的检测速度与显存利用率。
- **2025/12/31**: 发布演示[YOLO-Master-WebUI-Demo](https://huggingface.co/spaces/gatilin/YOLO-Master-WebUI-Demo)
- **2025/12/31**: 发布 YOLO-Master v0.1 版本，包含检测、分割和分类模型及训练代码。
- **2025/12/30**: arXiv 论文发布。

## 🔥 新特性 (v2026.02)

### 1️⃣ 混合专家 (MoE) 支持

YOLO-Master 首次将混合专家架构深度融合到 YOLO 中，实现实例条件自适应计算。

<div align="center">
  <img width="90%" alt="MoE Architecture" src="https://github.com/user-attachments/assets/5c51a886-e81d-43a4-bf4d-d37991e35cd2" />
  <img width="90%" alt="MoE Module Details" src="https://github.com/user-attachments/assets/0c2d2689-72c2-47fb-97c6-002fefa99c73" />
</div>

**核心组件：**

| 组件 | 描述 | 实现路径 |
|:-----|:----|:---------|
| **MoE 损失 (MoELoss)** | 负载均衡损失 + Z-Loss，确保稳定训练 | `ultralytics/nn/modules/moe/loss.py` |
| **MoE 剪枝 (MoEPruner)** | 自动剪枝低利用率专家（20-30% 推理加速） | `ultralytics/nn/modules/moe/pruning.py` |
| **模块化架构** | 解耦路由器、专家网络和门控机制 | `ultralytics/nn/modules/moe/` |

**使用方法：**

```python
from ultralytics import YOLO

# 加载 MoE 配置
model = YOLO("ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml")

# MoE 训练
results = model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    moe_num_experts=8,      # 专家数量
    moe_top_k=2,            # 每个 token 激活的专家数
    moe_balance_loss=0.01,  # 负载均衡损失权重
)

# 专家利用率分析与剪枝
model.prune_experts(threshold=0.15)
```

---

### 2️⃣ LoRA 支持 - 参数高效微调

架构无关的 LoRA 适配，**零架构开销** —— 纯配置驱动，无需修改模型结构。

<div align="center">
  <img width="90%" alt="LoRA Training Comparison" src="https://github.com/user-attachments/assets/98c6cada-ddc7-4723-877d-59d16ee0fdb2" />
  <p><i>LoRA vs Full SFT vs DoRA vs LoHa：YOLOv11-s 上的训练曲线对比（COCO val2017，300 epochs）</i></p>
</div>

**核心优势：**
- 🎯 仅需约 10% 可训练参数，即可达到全量微调 **95-98%** 的性能
- ⚡ **40-60%** 训练加速，**70%** 显存节省
- 📦 超紧凑适配器（如 YOLO11x：14.1 MB 适配器 vs 114.6 MB 完整模型）

**支持模型：**

| 模型系列 | 架构类型 | LoRA 集成方式 | 需要修改 |
|:---------|:---------|:-------------|:---------|
| YOLOv3 / v5 / v6 | CNN | 纯配置驱动 | 无需 ✅ |
| YOLOv8 / v9 / v10 | CNN | 纯配置驱动 | 无需 ✅ |
| YOLO11 / YOLO12 | CNN / 混合 | 纯配置驱动 | 无需 ✅ |
| RT-DETR | Transformer | 纯配置驱动 | 无需 ✅ |
| YOLO-World | 多模态 | 纯配置驱动 | 无需 ✅ |
| YOLO-Master | MoE | 纯配置驱动 | 无需 ✅ |

**使用方法：**

```python
from ultralytics import YOLO

model = YOLO("yolo11s.pt")

# LoRA 训练（一键激活）
results = model.train(
    data="coco8.yaml",
    epochs=300,
    imgsz=640,
    batch=32,
    lora_r=16,                # rank=16，最佳性价比
    lora_alpha=32,            # alpha = 2×r
    lora_dropout=0.1,
    lora_gradient_checkpointing=True,
)

# 仅保存 LoRA 适配器（YOLO11s 约 4.1MB）
model.save_lora_only("yolo11s_lora_r16.pt")
```

<details>
<summary><b>📊 GPU 显存与存储基准测试（点击展开）</b></summary>

**YOLO11 系列（LoRA rank=8）：**

| 模型 | 基础参数 (M) | LoRA 参数 | 基础模型大小 (MB) | 适配器大小 (MB) | 参数比例 (%) |
|:-----|:------------|:---------|:----------------|:---------------|:------------|
| YOLO11n | 2.6 | 527,536 | 5.6 | 2.1 | 20.29% |
| YOLO11s | 9.4 | 1,016,240 | 19.3 | 4.1 | 10.81% |
| YOLO11m | 20.1 | 1,639,856 | 40.7 | 6.6 | 8.16% |
| YOLO11l | 25.3 | 2,350,512 | 51.4 | 9.4 | 9.29% |
| YOLO11x | 56.9 | 3,525,552 | 114.6 | 14.1 | 6.20% |

**YOLO12 系列（LoRA rank=8）：**

| 模型 | 基础参数 (M) | LoRA 参数 | 基础模型大小 (MB) | 适配器大小 (MB) | 参数比例 (%) |
|:-----|:------------|:---------|:----------------|:---------------|:------------|
| YOLO12n | 2.6 | 632,752 | 5.6 | 2.3 | 24.34% |
| YOLO12s | 9.3 | 1,077,680 | 19.0 | 4.3 | 11.59% |
| YOLO12m | 20.2 | 1,684,912 | 40.9 | 6.8 | 8.34% |
| YOLO12l | 26.4 | 2,442,160 | 53.7 | 9.8 | 9.25% |
| YOLO12x | 59.1 | 3,662,768 | 119.3 | 14.7 | 6.20% |

**实际部署意义（以 YOLO11-X 为例）：**
- 🚀 **云端部署**：部署 14.1 MB 适配器而非 114.6 MB 完整模型，节省约 87.7% 存储与传输成本
- 📱 **边缘设备**：1 个基础模型 + N 个轻量适配器，实现多场景快速切换
- 🔄 **版本管理**：14.1 MB 适配器通过 Git 管理远比 114.6 MB 完整模型高效
- 💡 **多任务部署**：10 个任务仅需 255.6 MB（1×基础 + 10×适配器），传统方式需 1,146 MB

</details>

---

### 3️⃣ Sparse SAHI 稀疏推理模式

**稀疏切片辅助超推理（Sparse SAHI）** —— 针对超大分辨率图像（4K/8K）检测的革命性优化，通过智能跳过空白区域实现 **3-5 倍加速**。

<div align="center">
  <img width="90%" alt="Sparse SAHI Pipeline" src="https://github.com/user-attachments/assets/f86a1f41-7538-4168-b4b4-112dafcd80d5" />
  <p><i>Sparse SAHI 流水线：Objectness Mask → 自适应切片 → 高分辨率推理 → CW-NMS 融合</i></p>
</div>

<div align="center">
  <img width="45%" alt="Skip Ratio Analysis" src="https://github.com/user-attachments/assets/0aece4ee-f693-40bd-8164-2c7bcd954fd5" />
  <img width="45%" alt="Sparse SAHI Real-world Example" src="https://github.com/user-attachments/assets/7d41de53-7e58-472a-a6ad-15830b8744c6" />
  <p><i>左图：不同场景下的跳过比例分析。右图：真实检测效果示例。</i></p>
</div>

**工作原理：**
1. 🗺️ 低分辨率全图推理生成 objectness 热力图
2. ✂️ 自适应切片，跳过 objectness < 0.15 的区域
3. 🎯 仅对感兴趣区域进行高分辨率推理
4. 🔗 多切片结果通过 CW-NMS 融合

**使用方法：**

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.predict(
    source="large_aerial_image.jpg",
    sparse_sahi=True,
    slice_size=640,
    overlap_ratio=0.2,
    objectness_threshold=0.15,
)
```

---

### 4️⃣ 聚类加权 NMS (CW-NMS)

基于聚类理论的检测框融合算法，使用**高斯加权平均**代替硬抑制，显著提升定位精度。

<div align="center">
  <img width="90%" alt="CW-NMS Performance Comparison" src="https://github.com/user-attachments/assets/93d9252c-506a-4cf4-a0f1-ff864e0d721b" />
  <p><i>CW-NMS vs 传统 NMS vs Soft-NMS：密集场景下的性能对比</i></p>
</div>

| 方法 | 策略 | 优点 | 缺点 |
|:----|:----|:----|:----|
| 传统 NMS | 直接丢弃重叠框 | 速度快 | 可能丢失精确定位 |
| Soft-NMS | 置信度衰减 | 保留更多候选框 | 参数敏感 |
| **CW-NMS** | **高斯加权融合** | **高精度、鲁棒** | 略微增加计算量 |

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model.predict(
    source="dense_objects.jpg",
    cluster=True,     # 启用 CW-NMS
    sigma=0.1,        # 高斯加权 σ
)
```

---

## 📊 主要结果
### 检测
<div align="center">
  <img width="450" alt="Radar chart comparing YOLO models on various datasets" src="https://github.com/user-attachments/assets/743fa632-659b-43b1-accf-f865c8b66754"/>
</div>


<div align="center">
  <p><b>表 1. 五个基准测试上与最先进 Nano 级检测器的比较。</b></p>
  <table style="border-collapse:collapse; width:100%; font-family:sans-serif; text-align:center; border-top:2px solid #000; border-bottom:2px solid #000; font-size:0.9em;">
    <thead>
      <tr style="border-bottom:1px solid #ddd;">
        <th style="padding:8px; border-right:1px solid #ddd;">数据集</th>
        <th colspan="2" style="border-right:1px solid #ddd;">COCO</th>
        <th colspan="2" style="border-right:1px solid #ddd;">PASCAL VOC</th>
        <th colspan="2" style="border-right:1px solid #ddd;">VisDrone</th>
        <th colspan="2" style="border-right:1px solid #ddd;">KITTI</th>
        <th colspan="2" style="border-right:1px solid #ddd;">SKU-110K</th>
        <th>效率</th>
      </tr>
      <tr style="border-bottom:1px solid #000;">
        <th style="padding:8px; border-right:1px solid #ddd;">方法</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>mAP<br>(%)</th>
        <th style="border-right:1px solid #ddd;">mAP<sub>50</sub><br>(%)</th>
        <th>延迟<br>(ms)</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv10</td>
        <td>38.5</td><td style="border-right:1px solid #ddd;">53.8</td>
        <td>60.6</td><td style="border-right:1px solid #ddd;">80.3</td>
        <td>18.7</td><td style="border-right:1px solid #ddd;">32.4</td>
        <td>66.0</td><td style="border-right:1px solid #ddd;">88.3</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.84</td>
      </tr>
      <tr>
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv11-N</td>
        <td>39.4</td><td style="border-right:1px solid #ddd;">55.3</td>
        <td>61.0</td><td style="border-right:1px solid #ddd;">81.2</td>
        <td>18.5</td><td style="border-right:1px solid #ddd;">32.2</td>
        <td>67.8</td><td style="border-right:1px solid #ddd;">89.8</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.50</td>
      </tr>
      <tr>
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv12-N</td>
        <td>40.6</td><td style="border-right:1px solid #ddd;">56.7</td>
        <td>60.7</td><td style="border-right:1px solid #ddd;">80.8</td>
        <td>18.3</td><td style="border-right:1px solid #ddd;">31.7</td>
        <td>67.6</td><td style="border-right:1px solid #ddd;">89.3</td>
        <td>57.4</td><td style="border-right:1px solid #ddd;">90.0</td>
        <td>1.64</td>
      </tr>
      <tr style="border-bottom:1px solid #000;">
        <td style="padding:6px; text-align:left; border-right:1px solid #ddd;">YOLOv13-N</td>
        <td>41.6</td><td style="border-right:1px solid #ddd;">57.8</td>
        <td>60.7</td><td style="border-right:1px solid #ddd;">80.3</td>
        <td>17.5</td><td style="border-right:1px solid #ddd;">30.6</td>
        <td>67.7</td><td style="border-right:1px solid #ddd;">90.6</td>
        <td>57.5</td><td style="border-right:1px solid #ddd;">90.3</td>
        <td>1.97</td>
      </tr>
      <tr style="background-color:#f9f9f9;">
        <td style="padding:8px; text-align:left; border-right:1px solid #ddd;"><b>YOLO-Master-N</b></td>
        <td><b>42.4</b></td><td style="border-right:1px solid #ddd;"><b>59.2</b></td>
        <td><b>62.1</b></td><td style="border-right:1px solid #ddd;"><b>81.9</b></td>
        <td><b>19.6</b></td><td style="border-right:1px solid #ddd;"><b>33.7</b></td>
        <td><b>69.2</b></td><td style="border-right:1px solid #ddd;"><b>91.3</b></td>
        <td><b>58.2</b></td><td style="border-right:1px solid #ddd;"><b>90.6</b></td>
        <td><b>1.62</b></td>
      </tr>
    </tbody>
  </table>
</div>

### 分割

| **模型**             | **尺寸** | **mAPbox (%)** | **mAPmask (%)** | **增益 (mAPmask)** |
| --------------------- | -------- | -------------- | --------------- | ------------------ |
| YOLOv11-seg-N         | 640      | 38.9           | 32.0            | -                  |
| YOLOv12-seg-N         | 640      | 39.9           | 32.8            | Baseline           |
| **YOLO-Master-seg-N** | **640**  | **42.9**       | **35.6**        | **+2.8%** 🚀        |

### 分类

| **模型**             | **数据集**  | **输入尺寸** | **Top-1 Acc (%)** | **Top-5 Acc (%)** | **对比**    |
| --------------------- | ------------ | -------------- | ----------------- | ----------------- | ----------------- |
| YOLOv11-cls-N         | ImageNet     | 224            | 70.0              | 89.4              | Baseline          |
| YOLOv12-cls-N         | ImageNet     | 224            | 71.7              | 90.5              | +1.7% Top-1       |
| **YOLO-Master-cls-N** | **ImageNet** | **224**        | **76.6**          | **93.4**          | **+4.9% Top-1** 🔥 |

## 📦 模型库与基准测试

<div align="center">
  <img width="45%" alt="Model Performance 1" src="https://github.com/user-attachments/assets/9bd46c20-f4e3-4680-ad59-fcbab4b870f5" />
  <img width="45%" alt="Model Performance 2" src="https://github.com/user-attachments/assets/6f1b13c2-651f-4579-8a34-833c4753322a" />
</div>
<div align="center">
  <img width="45%" alt="Model Performance 3" src="https://github.com/user-attachments/assets/b6680e38-b206-438f-b693-4c7f858fb8b7" />
  <img width="45%" alt="Model Performance 4" src="https://github.com/user-attachments/assets/9f17ac3e-f839-4950-8661-76a5d4714443" />
</div>

### YOLO-Master-EsMoE 系列

| 模型 | 参数量(M) | GFLOPs(G) | Box(P) | R | mAP50 | mAP50-95 | 速度 (4090 TRT) FPS |
|:-----|:---------|:---------|:------|:--|:------|:---------|:-------------------|
| YOLO-Master-EsMoE-N | 2.68 | 8.7 | 0.684 | 0.536 | 0.587 | 0.427 | 640.18 |
| YOLO-Master-EsMoE-S | 9.69 | 29.1 | 0.699 | 0.603 | 0.603 | 0.489 | 423.87 |
| YOLO-Master-EsMoE-M | 34.88 | 97.4 | 0.737 | 0.640 | 0.697 | 0.530 | 243.79 |
| YOLO-Master-EsMoE-L | 🔥训练中 | TBD | TBD | TBD | TBD | TBD | TBD |
| YOLO-Master-EsMoE-X | 🔥训练中 | TBD | TBD | TBD | TBD | TBD | TBD |

### YOLO-Master-v0.1 系列

| 模型 | 参数量(M) | GFLOPs(G) | Box(P) | R | mAP50 | mAP50-95 | 速度 (4090 TRT) FPS |
|:-----|:---------|:---------|:------|:--|:------|:---------|:-------------------|
| YOLO-Master-v0.1-N | 7.54 | 10.1 | 0.684 | 0.542 | 0.592 | 0.429 | 528.84 |
| YOLO-Master-v0.1-S | 29.15 | 36.0 | 0.724 | 0.607 | 0.662 | 0.489 | 345.24 |
| YOLO-Master-v0.1-M | 52.17 | 116.7 | 0.729 | 0.641 | 0.696 | 0.528 | 170.72 |
| YOLO-Master-v0.1-L | 58.41 | 138.1 | 0.739 | 0.646 | 0.705 | 0.539 | 149.86 |
| YOLO-Master-v0.1-X | 🔥训练中 | TBD | TBD | TBD | TBD | TBD | TBD |

## 🖼️ 检测示例

<div align="center">
  <img width="1416" height="856" alt="Detection Examples" src="https://github.com/user-attachments/assets/0e1fbe4a-34e7-489e-b936-6d121ede5cf6" /> </div>
<table border="0"> <tr> <td align="center" style="font-weight: bold; background-color: #f6f8fa;"> <b>检测</b> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/db350acd-1d91-4be6-96b2-6bdf8aac57e8" alt="Detection 1" style="width:100%; display:block; border-radius:4px;"> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/b6c80dbd-120e-428b-8d26-ea2b38a40b47" alt="Detection 2" style="width:100%; display:block; border-radius:4px;"> </td> </tr> <tr> <td align="center" style="font-weight: bold; background-color: #f6f8fa;"> <b>分割</b> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/edb05e3c-cd83-41db-89f8-8ef09fc22798" alt="Segmentation 1" style="width:100%; display:block; border-radius:4px;"> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/ea138674-d7c7-48fb-b272-3ec211d161bf" alt="Segmentation 2" style="width:100%; display:block; border-radius:4px;"> </td> </tr> </table>



## 🧩 支持的任务

YOLO-Master 建立在强大的 Ultralytics 框架之上，继承了对各种计算机视觉任务的支持。虽然我们的研究主要集中在实时目标检测，但代码库支持：

| 任务 | 状态 | 描述 |
|:-----|:------:|:------------|
| **目标检测** | ✅ | 具有 ES-MoE 加速的实时目标检测。 |
| **实例分割** | ✅ | 实验性支持 (继承自 Ultralytics)。 |
| **姿态估计** | 🚧 | 实验性支持 (继承自 Ultralytics)。 |
| **OBB 检测** | 🚧 | 实验性支持 (继承自 Ultralytics)。 |
| **图像分类** | ✅ | 图像分类支持。 |

## ⚙️ 快速开始

### 安装

<details open>
<summary><strong>通过 pip 安装 (推荐)</strong></summary>

```bash
# 1. 创建并激活新环境
conda create -n yolo_master python=3.11 -y
conda activate yolo_master

# 2. 克隆仓库
git clone https://github.com/Tencent/YOLO-Master
cd YOLO-Master

# 3. 安装依赖
pip install -r requirements.txt
pip install -e .

# 4. 可选: 安装 FlashAttention 以加速训练 (需要 CUDA)
pip install flash_attn
```
</details>

### 验证

在 COCO 数据集上验证模型精度。

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO("yolo_master_n.pt") 

# 运行验证
metrics = model.val(data="coco.yaml", save_json=True)
print(metrics.box.map)  # map50-95
```

### 训练

在自定义数据集或 COCO 上训练新模型。

```python
from ultralytics import YOLO

# 加载模型
model = YOLO('cfg/models/master/v0/det/yolo-master-n.yaml')  # 从 YAML 构建新模型

# 训练模型
results = model.train(
    data='coco.yaml',
    epochs=600, 
    batch=256, 
    imgsz=640,
    device="0,1,2,3", # 使用多 GPU
    scale=0.5, 
    mosaic=1.0,
    mixup=0.0, 
    copy_paste=0.1
)
```

### 推理

对图像或视频进行推理。

**Python:**
```python
from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")
results = model("path/to/image.jpg")
results[0].show()
```

**CLI:**
```bash
yolo predict model=yolo_master_n.pt source='path/to/image.jpg' show=True
```

### 导出

将模型导出为其他格式以进行部署 (TensorRT, ONNX 等)。

```python
from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")
model.export(format="engine", half=True)  # 导出为 TensorRT
# 格式: onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs
```

### Gradio 演示

启动本地 Web 界面以交互式测试模型。此应用程序提供了一个用户友好的 Gradio 仪表板，用于模型推理，支持自动模型扫描、任务切换（检测、分割、分类）和实时可视化。

```bash
python app.py
# 在浏览器中打开 http://127.0.0.1:7860
```

## 🤝 社区与贡献

我们欢迎贡献！有关如何参与的详细信息，请查看我们的 [贡献指南](CONTRIBUTING.md)。

- **Issues**: 在 [这里](https://github.com/Tencent/YOLO-Master/issues) 报告错误或请求功能。
- **Pull Requests**: 提交您的改进。

## 📄 许可证

本项目采用 [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE) 许可证。

## 🙏 致谢

这项工作建立在优秀的 [Ultralytics](https://github.com/ultralytics/ultralytics) 框架之上。非常感谢社区的贡献、部署和教程！

## 📝 引用

如果您在研究中使用 YOLO-Master，请引用我们的论文：

```bibtex
@inproceedings{lin2026yolomaster,
  title={{YOLO-Master}: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection},
  author={Lin, Xu and Peng, Jinlong and Gan, Zhenye and Zhu, Jiawen and Liu, Jun},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

⭐ **如果您觉得这项工作有用，请给仓库点个星！**
