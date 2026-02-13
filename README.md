<img width="960" height="180" alt="Image" src="https://github.com/user-attachments/assets/5d2ab671-cf2f-4697-9c1b-1dfe611111e3" />

<p align="center">
  <a href="https://huggingface.co/spaces/gatilin/YOLO-Master-WebUI-Demo"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue" alt="Hugging Face Spaces"></a>
  <a href="https://colab.research.google.com/drive/1gTKkCsE4sXIOWpu1cdNBjdFHEahBoZD0?usp=sharing"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
  <a href="https://arxiv.org/abs/2512.23273"><img src="https://img.shields.io/badge/arXiv-2512.23273-b31b1b.svg" alt="arXiv"></a>
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

---

<img
  width="224"
  alt="YOLO-Master Mascot"
  src="https://github.com/user-attachments/assets/bbf751ea-af27-465d-a8a9-7822db343638"
  align="left"
/>

`YOLO-Master` is a YOLO-style framework tailored for **Real-Time Object Detection (RTOD)**. It marks the first deep integration of **Mixture-of-Experts (MoE)** into the YOLO architecture for general datasets. By leveraging **Efficient Sparse MoE (ES-MoE)** and lightweight **Dynamic Routing**, the framework achieves **instance-conditional adaptive computation**. This "compute-on-demand" paradigm allows the model to allocate FLOPs based on scene complexity, reaching a superior Pareto frontier between high precision and ultra-low latency.

**Key Highlights:**
- **Methodological Innovation (ES-MoE + Dynamic Routing)**: Utilizes dynamic routing networks to guide expert specialization during training and activates only the most relevant experts during inference, significantly reducing redundant computation while boosting detection performance.
- **Performance Validated (Accuracy × Latency)**: On MS COCO, YOLO-Master-N achieves **42.4% AP @ 1.62ms latency**, outperforming YOLOv13-N with a **+0.8% mAP gain while being 17.8% faster**.
- **Compute-on-Demand Intuition**: Transitions from "static dense computation" to "input-adaptive compute allocation," yielding more pronounced gains in dense or challenging scenarios.
- **Out-of-the-Box Pipeline**: Provides a complete end-to-end workflow including installation, validation, training, inference, and deployment (ONNX, TensorRT, etc.).
- **Continuous Engineering Evolution**: Includes advanced utilities such as MoE pruning and diagnostic tools (`diagnose_model` / `prune_moe_model`), CW-NMS, and Sparse SAHI inference modes.

<br clear="left" />

---

## 💡 A Humble Beginning (Introduction)

> **"Exploring the frontiers of Dynamic Intelligence in YOLO."**

This work represents our passionate exploration into the evolution of Real-Time Object Detection (RTOD). To the best of our knowledge, **YOLO-Master is the first work to deeply integrate Mixture-of-Experts (MoE) with the YOLO architecture on general-purpose datasets.**

Most existing YOLO models rely on static, dense computation—allocating the same computational budget to a simple sky background as they do to a complex, crowded intersection. We believe detection models should be more "adaptive", much like the human visual system. While this initial exploration may be not perfect, it demonstrates the significant potential of **Efficient Sparse MoE (ES-MoE)** in balancing high precision with ultra-low latency. We are committed to continuous iteration and optimization to refine this approach further.

Looking forward, we draw inspiration from the transformative advancements in LLMs and VLMs. We are committed to refining this approach and extending these insights to fundamental vision tasks, with the ultimate goal of tackling more ambitious frontiers like Open-Vocabulary Detection and Open-Set Segmentation.

<details>
  <summary>
  <font size="+1"><b>Abstract</b></font>
  </summary>
Existing Real-Time Object Detection (RTOD) methods commonly adopt YOLO-like architectures for their favorable trade-off between accuracy and speed. However, these models rely on static dense computation that applies uniform processing to all inputs, misallocating representational capacity and computational resources such as over-allocating on trivial scenes while under-serving complex ones. This mismatch results in both computational redundancy and suboptimal detection performance.

To overcome this limitation, we propose YOLO-Master, a novel YOLO-like framework that introduces instance-conditional adaptive computation for RTOD. This is achieved through an Efficient Sparse Mixture-of-Experts (ES-MoE) block that dynamically allocates computational resources to each input according to its scene complexity. At its core, a lightweight dynamic routing network guides expert specialization during training through a diversity enhancing objective, encouraging complementary expertise among experts. Additionally, the routing network adaptively learns to activate only the most relevant experts, thereby improving detection performance while minimizing computational overhead during inference.

Comprehensive experiments on five large-scale benchmarks demonstrate the superiority of YOLO-Master. On MS COCO, our model achieves 42.4\% AP with 1.62ms latency, outperforming YOLOv13-N by +0.8\% mAP and 17.8\% faster inference. Notably, the gains are most pronounced on challenging dense scenes, while the model preserves efficiency on typical inputs and maintains real-time inference speed. Code: [isLinXu/YOLO-Master](https://github.com/isLinXu/YOLO-Master)
</details>

---

## 🎨 Architecture

<div align="center">
  <img width="90%" alt="YOLO-Master Architecture" src="https://github.com/user-attachments/assets/6caa1065-af77-4f77-8faf-7551c013dacd" />
  <p><i>YOLO-Master introduces ES-MoE blocks to achieve "compute-on-demand" via dynamic routing.</i></p>
</div>

### 📚 In-Depth Documentation
For a deep dive into the design philosophy of MoE modules, detailed routing mechanisms, and optimization guides for deployment on various hardware (GPU/CPU/NPU), please refer to our Wiki:
👉 **[Wiki: MoE Modules Explained](wiki/MoE_Modules_Explanation_EN.md)**


## 📖 Table of Contents

- [A Humble Beginning](#-a-humble-beginning-introduction)
- [Architecture](#-architecture)
- [Updates](#-updates-latest-first)
- [Main Results](#-main-results)
  - [Detection](#detection)
  - [Segmentation](#segmentation)
  - [Classification](#classification)
- [Detection Examples](#-detection-examples)
- [Supported Tasks](#-supported-tasks)
- [Quick Start](#-quick-start)
  - [Installation](#installation)
  - [Validation](#validation)
  - [Training](#training)
  - [Inference](#inference)
  - [Export](#export)
  - [Gradio Demo](#gradio-demo)
- [Community & Contributing](#-community--contributing)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Citation](#-citation)



## 🚀 Updates (Latest First)
- **2026/02/13**: 🧨🚀add LoRA support for model training and release [v2026.02 version](https://github.com/Tencent/YOLO-Master/releases/tag/YOLO-Master-v26.02).[Happy New Year!]
- **2026/01/16**: [feature] Add pruning and analysis tools for MoE models.
  > 1. diagnose_model: Visualize expert utilization and routing behavior to identify redundant experts.
  > 2. prune_moe_model: Physically excise redundant experts and reconstruct routers for efficient inference without retraining.
- **2026/01/16**: Repo[isLinXu/YOLO-Master](https://github.com/isLinXu/YOLO-Master) Transferred to [Tencent](https://github.com/Tencent/YOLO-Master).
- **2026/01/14**: [ncnn-YOLO-Master-android](https://github.com/mpj1234/ncnn-YOLO-Master-android) support deploy YOLO-Master. Thanks to them!
- **2026/01/09**: [feature] Add Cluster-Weighted NMS (CW-NMS) to trade mAP vs speed.
  > cluster: False # (bool) cluster NMS (MoE optimized)
- **2026/01/07**: [TensorRT-YOLO](https://github.com/laugh12321/TensorRT-YOLO) accelerates YOLO-Master. Thanks to them!
- **2026/01/07**: Add MoE loss explicitly into training.
  > Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   **moe_loss**  Instances  Size
- **2026/01/04**: Split MoE script into modules
  > Split MoE script into separate modules (routers, experts)
- **2026/01/03**: [feature] Added Sparse SAHI Inference Mode: Introduced a content-adaptive sparse slicing mechanism guided by a global Objectness Mask, significantly accelerating small object detection in high-resolution images while optimizing GPU memory efficiency.
- **2025/12/31**: Released the demo [YOLO-Master-WebUI-Demo](https://huggingface.co/spaces/gatilin/YOLO-Master-WebUI-Demo).
- **2025/12/31**: Released YOLO-Master v0.1 with code, pre-trained weights, and documentation.
- **2025/12/30**: arXiv paper published.


## 📊 Main Results
### Detection
<div align="center">
  <img width="450" alt="Radar chart comparing YOLO models on various datasets" src="https://github.com/user-attachments/assets/743fa632-659b-43b1-accf-f865c8b66754"/>
</div>


<div align="center">
  <p><b>Table 1. Comparison with state-of-the-art Nano-scale detectors across five benchmarks.</b></p>
  <table style="border-collapse:collapse; width:100%; font-family:sans-serif; text-align:center; border-top:2px solid #000; border-bottom:2px solid #000; font-size:0.9em;">
    <thead>
      <tr style="border-bottom:1px solid #ddd;">
        <th style="padding:8px; border-right:1px solid #ddd;">Dataset</th>
        <th colspan="2" style="border-right:1px solid #ddd;">COCO</th>
        <th colspan="2" style="border-right:1px solid #ddd;">PASCAL VOC</th>
        <th colspan="2" style="border-right:1px solid #ddd;">VisDrone</th>
        <th colspan="2" style="border-right:1px solid #ddd;">KITTI</th>
        <th colspan="2" style="border-right:1px solid #ddd;">SKU-110K</th>
        <th>Efficiency</th>
      </tr>
      <tr style="border-bottom:1px solid #000;">
        <th style="padding:8px; border-right:1px solid #ddd;">Method</th>
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
        <th>Latency<br>(ms)</th>
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

### Segmentation

| **Model**             | **Size** | **mAPbox (%)** | **mAPmask (%)** | **Gain (mAPmask)** |
| --------------------- | -------- | -------------- | --------------- | ------------------ |
| YOLOv11-seg-N         | 640      | 38.9           | 32.0            | -                  |
| YOLOv12-seg-N         | 640      | 39.9           | 32.8            | Baseline           |
| **YOLO-Master-seg-N** | **640**  | **42.9**       | **35.6**        | **+2.8%** 🚀        |

### Classification

| **Model**             | **Dataset**  | **Input Size** | **Top-1 Acc (%)** | **Top-5 Acc (%)** | **Comparison**    |
| --------------------- | ------------ | -------------- | ----------------- | ----------------- | ----------------- |
| YOLOv11-cls-N         | ImageNet     | 224            | 70.0              | 89.4              | Baseline          |
| YOLOv12-cls-N         | ImageNet     | 224            | 71.7              | 90.5              | +1.7% Top-1       |
| **YOLO-Master-cls-N** | **ImageNet** | **224**        | **76.6**          | **93.4**          | **+4.9% Top-1** 🔥 |

## 🖼️ Detection Examples

<div align="center">
  <img width="1416" height="856" alt="Detection Examples" src="https://github.com/user-attachments/assets/0e1fbe4a-34e7-489e-b936-6d121ede5cf6" /> </div>
<table border="0"> <tr> <td align="center" style="font-weight: bold; background-color: #f6f8fa;"> <b>Detection</b> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/db350acd-1d91-4be6-96b2-6bdf8aac57e8" alt="Detection 1" style="width:100%; display:block; border-radius:4px;"> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/b6c80dbd-120e-428b-8d26-ea2b38a40b47" alt="Detection 2" style="width:100%; display:block; border-radius:4px;"> </td> </tr> <tr> <td align="center" style="font-weight: bold; background-color: #f6f8fa;"> <b>Segmentation</b> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/edb05e3c-cd83-41db-89f8-8ef09fc22798" alt="Segmentation 1" style="width:100%; display:block; border-radius:4px;"> </td> <td width="45%"> <img src="https://github.com/user-attachments/assets/ea138674-d7c7-48fb-b272-3ec211d161bf" alt="Segmentation 2" style="width:100%; display:block; border-radius:4px;"> </td> </tr> </table>



## 🧩 Supported Tasks

YOLO-Master builds upon the robust Ultralytics framework, inheriting support for various computer vision tasks. While our research primarily focuses on Real-Time Object Detection, the codebase is capable of supporting:

| Task | Status | Description |
|:-----|:------:|:------------|
| **Object Detection** | ✅ | Real-time object detection with ES-MoE acceleration. |
| **Instance Segmentation** | ✅ | Experimental support (inherited from Ultralytics). |
| **Pose Estimation** | 🚧 | Experimental support (inherited from Ultralytics). |
| **OBB Detection** | 🚧 | Experimental support (inherited from Ultralytics). |
| **Classification** | ✅ | Image classification support. |

## ⚙️ Quick Start

### Installation

<details open>
<summary><strong>Install via pip (Recommended)</strong></summary>

```bash
# 1. Create and activate a new environment
conda create -n yolo_master python=3.11 -y
conda activate yolo_master

# 2. Clone the repository
git clone https://github.com/isLinXu/YOLO-Master
cd YOLO-Master

# 3. Install dependencies
pip install -r requirements.txt
pip install -e .

# 4. Optional: Install FlashAttention for faster training (CUDA required)
pip install flash_attn
```
</details>

### Validation

Validate the model accuracy on the COCO dataset.

```python
from ultralytics import YOLO

# Load the pretrained model
model = YOLO("yolo_master_n.pt") 

# Run validation
metrics = model.val(data="coco.yaml", save_json=True)
print(metrics.box.map)  # map50-95
```

### Training

Train a new model on your custom dataset or COCO.

```python
from ultralytics import YOLO

# Load a model
model = YOLO('cfg/models/master/v0/det/yolo-master-n.yaml')  # build a new model from YAML

# Train the model
results = model.train(
    data='coco.yaml',
    epochs=600, 
    batch=256, 
    imgsz=640,
    device="0,1,2,3", # Use multiple GPUs
    scale=0.5, 
    mosaic=1.0,
    mixup=0.0, 
    copy_paste=0.1
)
```

### Inference

Run inference on images or videos.

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

### Export

Export the model to other formats for deployment (TensorRT, ONNX, etc.).

```python
from ultralytics import YOLO

model = YOLO("yolo_master_n.pt")
model.export(format="engine", half=True)  # Export to TensorRT
# formats: onnx, openvino, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs
```

### Gradio Demo

Launch a local web interface to test the model interactively. This application provides a user-friendly Gradio dashboard for model inference, supporting automatic model scanning, task switching (Detection, Segmentation, Classification), and real-time visualization.

```bash
python app.py
# Open http://127.0.0.1:7860 in your browser
```

## 🤝 Community & Contributing

We welcome contributions! Please check out our [Contribution Guidelines](CONTRIBUTING.md) for details on how to get involved.

- **Issues**: Report bugs or request features [here](https://github.com/isLinXu/YOLO-Master/issues).
- **Pull Requests**: Submit your improvements.

## 📄 License

This project is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](LICENSE).

## 🙏 Acknowledgements

This work builds upon the excellent [Ultralytics](https://github.com/ultralytics/ultralytics) framework. Huge thanks to the community for contributions, deployments, and tutorials!

## 📝 Citation

If you use YOLO-Master in your research, please cite our paper:

```bibtex
@article{lin2025yolomaster,
  title={{YOLO-Master}: MOE-Accelerated with Specialized Transformers for Enhanced Real-time Detection},
  author={Lin, Xu and Peng, Jinlong and Gan, Zhenye and Zhu, Jiawen and Liu, Jun},
  journal={arXiv preprint arXiv:2512.23273},
  year={2025}
}
```

⭐ **If you find this work useful, please star the repository!**
