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
| `lora_gradient_checkpointing` | Enables gradient checkpointing. | **True** (Critical) | **True** (Critical) |
| `lora_include_attention` | Target Attention layers. | False | **True** |
| `lora_target_modules` | Regex for modules to target. | `["conv"]` | `["linear", "conv"]` |

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
