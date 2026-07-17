# YOLO-Master Extra LoRA Scenarios

This document tracks two additional YOLO-Master-EsMoE-N LoRA adaptation
experiments. It is intended to make the SKU-110K and Construction-PPE rank
sweeps reproducible and provide a compact place for result tables and run
links.

## Runtime Environment

| Item | Value |
| --- | --- |
| GPU | NVIDIA L20 |
| Public curves | [Weights & Biases](https://wandb.ai/zhangjch98-sun-yat-sen-university/yolo-master-issue50) |

## Scope

- Model family: YOLO-Master detection models
- Primary model config: `ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml`
- LoRA configs:
  - `examples/lora_examples/yolo_master_sku110k_lora.yaml`
  - `examples/lora_examples/yolo_master_construction_ppe_lora.yaml`
- Completed experiment set:
  - SKU-110K LoRA rank sweep: `r=4`, `r=8`, `r=16`
  - Construction-PPE LoRA rank sweep: `r=4`, `r=8`, `r=16`

## Repository Layout

```text
examples/lora_examples/
  yolo_master_sku110k_lora.yaml
  yolo_master_construction_ppe_lora.yaml
  yolo_master_lora_scenarios_README.md
  run_yolo_master_lora_rank_sweep.py
```

## Experimental Setup

| Dataset | Data config | Epochs | Batch | Image size | Fraction | AMP |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| SKU-110K | `ultralytics/cfg/datasets/SKU-110K.yaml` | 30 | 16 | 640 | 1.0 | Disabled |
| Construction-PPE | `ultralytics/cfg/datasets/construction-ppe.yaml` | 40 | 16 | 640 | 1.0 | Enabled |

## LoRA Hyperparameters

Both configs use the same LoRA hyperparameter sweep.

| Setting | SKU-110K | Construction-PPE |
| --- | --- | --- |
| `lora_r` | `4`, `8`, `16` | `4`, `8`, `16` |
| `lora_alpha` | `8`, `16`, `32` | `8`, `16`, `32` |
| `lora_use_rslora` | `True` | `True` |
| `lora_include_attention` | `False` | `False` |
| `lora_gradient_checkpointing` | `True` | `True` |

## Commands

The rank list is executed sequentially on the selected device. Set `--device`
to an available CUDA device id for the local machine.

Run the SKU-110K rank sweep:

```bash
python examples/lora_examples/run_yolo_master_lora_rank_sweep.py \
  --scene sku110k \
  --ranks 4 8 16 \
  --device 0
```

Run the Construction-PPE rank sweep:

```bash
python examples/lora_examples/run_yolo_master_lora_rank_sweep.py \
  --scene construction_ppe \
  --ranks 4 8 16 \
  --device 0
```

## Result Summary

| Run | Rank | Trainable params | Adapter params | Best epoch | mAP50-95 | Train time | Peak GPU mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `sku110k_r4` | 4 | 444,543 | 99,564 | 20 | 0.17160 | 91.51 min | 18.30G |
| `sku110k_r8` | 8 | 549,483 | 204,504 | 22 | 0.20903 | 79.47 min | 18.30G |
| `sku110k_r16` | 16 | 753,987 | 409,008 | 24 | 0.24394 | 93.44 min | 18.20G |
| `construction_ppe_r4` | 4 | 446,493 | 99,564 | 37 | 0.03018 | 11.84 min | 3.81G |
| `construction_ppe_r8` | 8 | 551,433 | 204,504 | 35 | 0.04816 | 11.88 min | 3.86G |
| `construction_ppe_r16` | 16 | 755,937 | 409,008 | 38 | 0.06579 | 12.65 min | 3.88G |
