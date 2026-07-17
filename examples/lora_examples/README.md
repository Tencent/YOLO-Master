# Ultralytics LoRA Examples

This directory contains configuration files and examples for training various Ultralytics models using Low-Rank Adaptation (LoRA). All configurations are ready to run with the `yolo` CLI and have been standardized to match the full Ultralytics configuration structure.

## 📦 Supported Models

We provide optimized LoRA configurations for the following model families:

| Model Family | Config File | Architecture | Key LoRA Settings |
| :--- | :--- | :--- | :--- |
| **YOLOv8** | `yolov8_lora.yaml` | Conv-based | `gradient_checkpointing=True` |
| **YOLOv3** | `yolov3_lora.yaml` | Conv-based | `gradient_checkpointing=True` |
| **YOLOv5** | `yolov5_lora.yaml` | Conv-based | `gradient_checkpointing=True` |
| **YOLOv6** | `yolov6_lora.yaml` | Conv-based | `gradient_checkpointing=True` |
| **YOLOv9** | `yolov9_lora.yaml` | Conv-based | `gradient_checkpointing=True` |
| **YOLOv10** | `yolov10_lora.yaml` | Conv-based | `gradient_checkpointing=True` |
| **YOLO11** | `yolo11_lora.yaml` | Conv-based | `gradient_checkpointing=True` |
| **YOLO12** | `yolo12_lora.yaml` | Hybrid (CNN+Attn) | `include_attention=True` |
| **RT-DETR** | `rtdetr_lora.yaml` | Transformer | `include_attention=True` |
| **YOLO-World** | `yoloworld_lora.yaml` | Multi-modal | `include_attention=True` |
| **YOLO-Master** | `yolo_master_visdrone_lora.yaml` / `yolo_master_brain_tumor_lora.yaml` | MoE (Conv+Expert) | `include_moe=True`, routing excluded |

## 🔬 YOLO-Master 垂类场景 LoRA 微调

针对 MoE 架构的 YOLO-Master-EsMoE-N 模型，我们提供了两个差异化垂类场景的完整 LoRA 微调方案：

| 场景 | 数据集 | 配置文件 | 详细指南 |
| :--- | :--- | :--- | :--- |
| 密集航拍检测 | VisDrone | `yolo_master_visdrone_lora.yaml` | [yolo_master_lora_README.md](yolo_master_lora_README.md) |
| 稀疏医疗检测 | Brain Tumor | `yolo_master_brain_tumor_lora.yaml` | [yolo_master_lora_README.md](yolo_master_lora_README.md) |

该方案包含 rank 扫描对比（r=4/8/16）、MoE 路由层 LoRA 策略、目标模块选择指导和常见陷阱说明。详见 [YOLO-Master LoRA 适配指南](yolo_master_lora_README.md)。

```bash
# 快速开始
bash examples/lora_examples/run_lora_brain_tumor_sweep.sh   # Brain Tumor rank 扫描
bash examples/lora_examples/run_lora_visdrone_sweep.sh      # VisDrone rank 扫描
```

## 🚀 Usage Guide

### 1. Basic Training
Train any model by referencing its config file:

```bash
# Example: Train YOLOv9 with LoRA
yolo train cfg=examples/lora_examples/yolov9_lora.yaml

# Example: Train YOLO11 with LoRA
yolo train cfg=examples/lora_examples/yolo11_lora.yaml
```

### 2. Overriding Parameters
You can override any parameter from the CLI without modifying the YAML:

```bash
# Train YOLOv8n with a larger LoRA rank (r=32)
yolo train cfg=examples/lora_examples/yolov8_lora.yaml lora_r=32
```

### 3. Training on Custom Data
Change the `data` argument to point to your dataset YAML:

```bash
yolo train cfg=examples/lora_examples/rtdetr_lora.yaml data=/path/to/custom_dataset.yaml
```

### 4. AdaLoRA on RT-DETR
`AdaLoRA` is supported in this repository, but the current PEFT implementation only works on `nn.Linear` targets. In practice this makes `RT-DETR` the recommended family for AdaLoRA, while conv-heavy YOLO backbones should continue using standard `LoRA` or `RS-LoRA`.

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 yolo train \
  cfg=examples/lora_examples/rtdetr_lora.yaml \
  model=rtdetr-l.pt \
  data=coco128.yaml \
  lora_type=adalora \
  lora_target_modules=linear \
  lora_include_attention=True \
  lora_target_r=4 \
  lora_init_r=6
```

Notes:
- `lora_total_step` can be left at `0`; the trainer will resolve it from the run iterations and persist the resolved value into `args.yaml`.
- On Apple Silicon, `PYTORCH_ENABLE_MPS_FALLBACK=1` avoids MPS backward kernel gaps during RT-DETR training.
- If all requested targets are non-linear layers, AdaLoRA target selection will be filtered to an empty set and adapter creation will stop.

---

## 🛠️ Configuration Guide

Each `.yaml` file follows the standard Ultralytics configuration structure, divided into four main sections:

1.  **Global settings**: Task, mode, and device selection.
2.  **Train settings**: Model path, epochs, batch size, optimizer, etc.
3.  **Val/Test settings**: Validation split, metrics, and plotting options.
4.  **LoRA settings**: Specific hyperparameters for Low-Rank Adaptation.

### Key LoRA Hyperparameters

| Parameter | Description | Recommended (YOLO) | Recommended (RT-DETR) |
| :--- | :--- | :--- | :--- |
| `lora_r` | Rank of the update matrices. | 16 - 32 | 8 - 16 |
| `lora_alpha` | Scaling factor. | 2x `lora_r` | 2x `lora_r` |
| `lora_use_rslora` | Use `alpha / sqrt(r)` scaling for better high-rank stability. | **True** | **True** |
| `lora_init_lora_weights` | Adapter initialization strategy. | `"pissa"` | `"pissa"` |
| `lora_gradient_checkpointing` | Enables gradient checkpointing. | **True** (Critical) | **True** (Critical) |
| `lora_include_attention` | Target Attention layers. | False | **True** |
| `lora_target_modules` | Regex for modules to target. | `["conv"]` | `["linear", "conv"]` |
| `lora_only_3x3` | Skip `1x1` convs during auto target detection. | **True** | False |
| `lora_total_step` | AdaLoRA total steps. `0` lets the trainer auto-resolve it. | N/A | `0` |

## Backend Behavior

- Requested backend: the backend requested by the user, for example `auto`, `peft`, or `fallback`.
- Effective backend: the backend that actually ran after capability checks.
- Requested init: the init mode requested by the user, such as `pissa`.
- Effective init: the init mode that actually ran after compatibility downgrade.
- In `auto` mode, the repository prefers `PEFT` first and uses the in-repo fallback path only when the request is unsupported on the active PEFT path.

## 🔄 Incremental Learning & Inference

### Resume / Incremental Training
To continue training or fine-tune on new data, simply load the trained `.pt` file (which includes LoRA adapters) and run training again.

```bash
# Load trained weights and train on new data
yolo train model=runs/lora_examples/yolov8n_lora/weights/best.pt data=new_dataset.yaml epochs=50 lora_r=16
```
> **Note**: You must explicitly pass `lora_r` again to ensure the LoRA structure is correctly initialized.

### Inference / Validation
LoRA models can be used for inference just like standard models. The adapter weights are automatically loaded.

```bash
# Predict
yolo predict model=runs/lora_examples/yolov8n_lora/weights/best.pt source='path/to/images'

# Validate
yolo val model=runs/lora_examples/yolov8n_lora/weights/best.pt data=coco8.yaml
```
