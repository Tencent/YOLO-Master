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
| **YOLO-Master-EsMoE-N / VisDrone** | `yolo_master_visdrone_lora.yaml` | MoE dense aerial detection | `r=16`, `include_moe=True`, `include_attention=True` |
| **YOLO-Master-EsMoE-N / brain-tumor** | `yolo_master_brain_tumor_lora.yaml` | MoE sparse medical detection | `r=8`, `include_moe=True`, `include_attention=False` |

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

### 5. YOLO-Master-EsMoE-N Domain LoRA Sweeps

Two domain-specific configs are provided for quick 20-50 epoch LoRA iteration on visually different scenarios:

```bash
# Dense aerial small-object detection
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml

# Sparse medical detection
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml
```

Run the standard rank comparison (`r=4,8,16`) and collect a CSV summary:

```bash
python examples/lora_examples/domain_lora_rank_sweep.py --scenario visdrone --device 0
python examples/lora_examples/domain_lora_rank_sweep.py --scenario brain_tumor --device 0
```

Use `--dry-run` to inspect the exact `yolo train` commands without launching training.

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

## YOLO-Master-EsMoE-N Domain Adaptation Guide

### Scenario Defaults

| Scenario | Config | Dataset | Epochs | Default rank | Recommended rank | Target modules | Routing policy |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VisDrone dense aerial | `yolo_master_visdrone_lora.yaml` | `VisDrone.yaml` | 30 | 16 | 16 | `conv`, `linear` | Do not explicitly target `routing`; adapt features/experts while keeping top-k routing stable. |
| Brain-tumor sparse medical | `yolo_master_brain_tumor_lora.yaml` | `brain-tumor.yaml` | 40 | 8 | 8 | `conv`, `linear` | Do not explicitly target `routing`; sparse data can overfit router logits quickly. |

### Rank Comparison Table

The sweep script writes `examples/lora_examples/domain_lora_rank_results.csv`. Record the final validation row from each run using the following format:

| Scenario | Rank | mAP50-95 | Trainable params | Train time | Peak VRAM | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VisDrone | 4 | fill from sweep output | fill from trainer log | fill from sweep output | fill from `GPU_mem`/monitor | Lowest memory, may underfit tiny dense objects. |
| VisDrone | 8 | fill from sweep output | fill from trainer log | fill from sweep output | fill from `GPU_mem`/monitor | Balanced fallback if r=16 exceeds VRAM. |
| VisDrone | 16 | fill from sweep output | fill from trainer log | fill from sweep output | fill from `GPU_mem`/monitor | Default recommendation for scale and density shift. |
| brain-tumor | 4 | fill from sweep output | fill from trainer log | fill from sweep output | fill from `GPU_mem`/monitor | Strong regularization, safest for very small subsets. |
| brain-tumor | 8 | fill from sweep output | fill from trainer log | fill from sweep output | fill from `GPU_mem`/monitor | Default recommendation for sparse medical detection. |
| brain-tumor | 16 | fill from sweep output | fill from trainer log | fill from sweep output | fill from `GPU_mem`/monitor | Use only if validation mAP improves without overfitting. |

### Target Module Selection

- `lora_include_moe=True` is enabled in both configs so LoRA can adapt YOLO-Master v0.5 MoE feature transforms and expert/projection paths.
- `lora_target_modules=["conv", "linear"]` keeps the search broad enough for Conv-heavy YOLO blocks and Linear SE/gating projections.
- Routing layers are not explicitly named in `lora_target_modules`. The router controls expert assignment and can collapse under short few-shot training if aggressively adapted.
- VisDrone enables `lora_include_attention=True` because dense aerial detection benefits from context and scale adaptation.
- Brain-tumor keeps `lora_include_attention=False` to reduce overfitting risk on a small grayscale-like medical dataset.

### Common Pitfalls

- Medical grayscale inputs: keep the dataset loader/image conversion consistent with YOLO's expected 3-channel input. If custom DICOM/PNG preprocessing is used, convert grayscale to RGB before label alignment.
- Medical augmentations: avoid mosaic-heavy augmentation on sparse lesions; `close_mosaic=0` is set in the brain-tumor config.
- Aerial scale variation: keep larger `imgsz` and `multi_scale=True` for VisDrone when GPU memory allows; reduce `imgsz` before lowering rank if OOM occurs.
- Dense scenes: increase `max_det` for VisDrone because many frames contain hundreds of small objects.
- MoE stability: do not combine high rank, router adaptation, and very short schedules unless you also monitor expert usage and MoE balance loss.

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
