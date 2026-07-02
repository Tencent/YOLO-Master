# YOLO-Master Domain-Specific LoRA Tuning

This document tracks the domain-specific LoRA tuning experiments for YOLO-Master models. It is intended to make the `Brain Tumor` and `VisDrone` runs reproducible, document the LoRA target policy for MoE-based YOLO-Master variants, and provide a single place for result tables, logs, and follow-up analysis.

## Runtime Environment

| Item | Value |
| --- | --- |
| Ultralytics | `8.3.240` |
| Python | `3.12.13` |
| PyTorch | `2.10.0+cu128` |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| CUDA device memory | 15,848 MiB |

## Scope

- Model family: YOLO-Master detection models
- Primary model config: `ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml`
- LoRA configs:
  - `examples/lora_examples/yolo_master_brain_tumor_lora.yaml`
  - `examples/lora_examples/yolo_master_visdrone_lora.yaml`
- Completed experiment set:
  - Brain Tumor LoRA rank sweep: `r=4`, `r=8`, `r=16`
  - VisDrone LoRA rank sweep: `r=4`, `r=8`, `r=16`

## Repository Layout

```text
examples/lora_examples/
  yolo_master_brain_tumor_lora.yaml
  yolo_master_visdrone_lora.yaml
  yolo_master_lora_README.md

logs/
  brain_tumor_r4.log
  brain_tumor_r8.log
  brain_tumor_r16.log
  visdrone_r4.log
  visdrone_r8.log
  visdrone_r16.log

runs/lora_examples/
  brain_tumor_r4/
    args.yaml
    results.csv
    results.png
    weights/
      best.pt
      last.pt
      lora_adapter_best/
  brain_tumor_r8/
  brain_tumor_r16/
  visdrone_r4/
  visdrone_r8/
  visdrone_r16/
```

## Experimental Setup

| Dataset | Data config | Epochs | Batch | Image size | Fraction | Optimizer | AMP | Project dir |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| Brain Tumor | `ultralytics/cfg/datasets/brain-tumor.yaml` | 40 | 16 | 640 | 1.0 | `auto` | Enabled | `runs/lora_examples` |
| VisDrone | `ultralytics/cfg/datasets/VisDrone.yaml` | 30 | 8 | 768 | 0.2 | `auto` | Enabled | `runs/lora_examples` |

## LoRA Target Policy

The YOLO-Master v0.10 model uses `VisualEnhancedAdaptiveGateMoE` blocks instead of the older `ES_MOE` blocks. Therefore, the LoRA targets are selected from the actual v0.10 module names.

## LoRA Hyperparameters

| Setting | Brain Tumor | VisDrone | Notes |
| --- | --- | --- | --- |
| `lora_r` | `4`, `8`, `16` | `4`, `8`, `16` | Rank sweep for domain adaptation capacity. |
| `lora_alpha` | `8`, `16`, `32` | `8`, `16`, `32` | Set to `2 * lora_r` in the recorded sweeps. |
| `lora_use_rslora` | `True` | `True` | Uses rank-stabilized LoRA scaling for higher-rank stability. |
| `lora_target_modules` | See target list below | See target list below | Shared v0.10 YOLO-Master visual/expert target policy. |
| `lora_include_attention` | `False` | `False` | Excludes A2C2f attention paths such as `attn.qkv`, `attn.proj`, and `attn.pe` for stability. |
| `lora_gradient_checkpointing` | `True` | `True` | Reduces memory pressure during LoRA training. |

Main tuning targets:

```yaml
lora_target_modules: [
  "conv", "fused_conv", "bottleneck.0", "shared_feature.0", "static_net.3", "proj",
  "expert_projections.0.0", "expert_projections.1.0", "expert_projections.2.0", "expert_projections.3.0",
  "expert_projections.4.0", "expert_projections.5.0", "expert_projections.6.0", "expert_projections.7.0",
  "expert_projections.8.0", "expert_projections.9.0", "expert_projections.10.0", "expert_projections.11.0",
  "expert_projections.12.0", "expert_projections.13.0", "expert_projections.14.0", "expert_projections.15.0"
]
```

Routing and gate layers are excluded from the main LoRA target set:

```yaml
lora_exclude_modules: ["router", "routing", "gate", "gating"]
```

> **Rationale:** short VisDrone or Brain Tumor LoRA runs should adapt visual/expert convolutions without changing expert-assignment dynamics; routing-LoRA should be tested only as a separate ablation.

## Brain Tumor Runs

### Commands

The Brain Tumor rank sweep is executed by:

```bash
bash examples/lora_examples/run_lora_brain_tumor_sweep.sh
```

The script runs the following experiments sequentially on `GPU_ID=0`, writes logs to `logs/`, and stores outputs under `runs/lora_examples/`.

| Run | `lora_r` | `lora_alpha` | Config | Log |
| --- | ---: | ---: | --- | --- |
| `brain_tumor_r4` | 4 | 8 | `examples/lora_examples/yolo_master_brain_tumor_lora.yaml` | `logs/brain_tumor_r4.log` |
| `brain_tumor_r8` | 8 | 16 | `examples/lora_examples/yolo_master_brain_tumor_lora.yaml` | `logs/brain_tumor_r8.log` |
| `brain_tumor_r16` | 16 | 32 | `examples/lora_examples/yolo_master_brain_tumor_lora.yaml` | `logs/brain_tumor_r16.log` |

### Result Summary

| Run | Rank | LoRA modules | Trainable params | Adapter params | Best epoch | mAP50 | mAP50-95 | Train time | Peak GPU mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brain_tumor_r4` | 4 | 92 | 468,290 | 123,116 | 30 | 0.43492 | 0.28312 | 39.72 min | 3.95G |
| `brain_tumor_r8` | 8 | 94 | 596,782 | 251,608 | 35 | 0.46004 | 0.31215 | 39.84 min | 3.99G |
| `brain_tumor_r16` | 16 | 94 | 848,390 | 503,216 | 37 | 0.48212 | 0.34044 | 40.15 min | 4.03G |

Notes:

- `LoRA modules` is taken from the `Final Targets Passed to PEFT` log line.
- `Trainable params` and `Adapter params` are taken from the `[LoRA] Stats` log line.
- `Train time` is taken from the `epochs completed in` log line.
- `Peak GPU mem` is the maximum `GPU_mem` value observed in the epoch progress lines.

## Observations

- Higher LoRA rank improved the Brain Tumor validation metrics in the completed sweep.
- `r16` achieved the best recorded Brain Tumor mAP50-95 among the three runs.
- MoE routing-collapse warnings appeared during training and were handled by the existing recovery/noise adjustment logic.
- The recorded runs used `optimizer=auto`; the trainer resolved the effective optimizer and learning rate in the run logs.



## VisDrone Runs

### Commands

The VisDrone runs use the same v0.10 LoRA target policy as the Brain Tumor runs, with larger image size and a higher `max_det` for dense aerial scenes.

The VisDrone rank sweep is executed by:

```bash
bash examples/lora_examples/run_lora_visdrone_sweep.sh
```

The script runs the following experiments sequentially on `GPU_ID=0`, writes logs to `logs/`, and stores outputs under `runs/lora_examples/`.

| Run | `lora_r` | `lora_alpha` | Config | Log |
| --- | ---: | ---: | --- | --- |
| `visdrone_r4` | 4 | 8 | `examples/lora_examples/yolo_master_visdrone_lora.yaml` | `logs/visdrone_r4.log` |
| `visdrone_r8` | 8 | 16 | `examples/lora_examples/yolo_master_visdrone_lora.yaml` | `logs/visdrone_r8.log` |
| `visdrone_r16` | 16 | 32 | `examples/lora_examples/yolo_master_visdrone_lora.yaml` | `logs/visdrone_r16.log` |

### Result Summary

| Run | Rank | LoRA modules | Trainable params | Adapter params | Best epoch | mAP50 | mAP50-95 | Train time | Peak GPU mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `visdrone_r4` | 4 | 92 | 469,850 | 123,116 | 27 | 0.04148 | 0.01670 | 52.68 min | 14.70G |
| `visdrone_r8` | 8 | 94 | 598,342 | 251,608 | 25 | 0.05547 | 0.02340 | 48.54 min | 14.60G |
| `visdrone_r16` | 16 | 94 | 849,950 | 503,216 | 27 | 0.07292 | 0.03197 | 48.96 min | 14.70G |

## Cross-Domain Summary

| Dataset | Best run | Best mAP50 | Best mAP50-95 | Peak GPU mem |
| --- | --- | ---: | ---: | ---: |
| Brain Tumor | `brain_tumor_r16` | 0.48212 | 0.34044 | 4.03G |
| VisDrone | `visdrone_r16` | 0.07292 | 0.03197 | 14.70G |

Notes:

- `r16` is the best rank in both completed sweeps by mAP50-95.
- VisDrone used `fraction=0.2`, so these values should be treated as partial-data LoRA tuning results rather than full VisDrone benchmark results.
- VisDrone peak GPU memory is much higher than Brain Tumor because it uses `imgsz=768`, `batch=8`, dense scenes, and multi-scale training.

## Full CSV Comparison

The full six-run comparison table is stored at:

```text
examples/lora_examples/yolo_master_lora_results.csv
```

This CSV records one row per run and uses the best validation epoch selected by `metrics/mAP50-95(B)`. It includes:

- LoRA settings: `lora_r`, `lora_alpha`, `lora_use_rslora`, `lora_include_attention`, `lora_gradient_checkpointing`, and LoRA module count.
- Parameter counts: trainable parameters, adapter parameters, and percentages.
- Training cost: best-epoch time, total training time in minutes, and peak GPU memory.
- Training losses: box, cls, dfl, and MoE loss.
- Validation metrics: precision, recall, mAP50, mAP50-95.
- Validation losses: box, cls, dfl, and MoE loss.
- Learning rates for all optimizer parameter groups.
- Last-epoch mAP values for comparison with best-epoch selection.

## Target Module Selection Guidance

- For YOLO-Master v0.10, use the actual `VisualEnhancedAdaptiveGateMoE` module names. The older `ES_MOE`-specific `pointwise` target is not useful for this config because it does not match v0.10 MoE expert modules.
- Keep the main experiment focused on visual and expert adaptation: `conv`, `fused_conv`, `bottleneck.0`, `shared_feature.0`, `static_net.3`, `proj`, and `expert_projections.0.0` through `expert_projections.15.0`.
- Keep `lora_include_attention=False` for the default sweep. A2C2f attention paths such as `attn.qkv`, `attn.proj`, and `attn.pe` are more sensitive and should be tested only as a separate stability ablation.
- Exclude routing and gate layers from the main comparison. Routing-LoRA changes expert assignment dynamics and should be reported as its own ablation rather than mixed into the rank sweep.
- Use `lora_only_3x3=False` when targeting v0.10 MoE modules. Many useful MoE projections and expert paths are 1x1 convolutions.
- Check the log line `Final Targets Passed to PEFT` for each run. A YAML target list is a set of matching rules; the trainer expands it into final module names after structural filtering.

## Common Pitfalls

- Medical datasets often contain grayscale images. Confirm the loader converts them consistently to the model's expected 3-channel input, and verify that preprocessing does not silently duplicate or normalize channels differently between train and val.
- Brain Tumor data is small and sparse, so aggressive learning rates, high-rank LoRA, attention targets, or routing targets can cause overfitting or numerical instability. If NaN or fitness collapse appears, reduce `lr0` or `lora_lr_mult`, increase warmup, and rerun with a fresh output name.
- VisDrone has strong scale variation and dense small objects. Larger `imgsz`, higher `max_det`, and multi-scale training affect memory and wall-clock time, so compare runs only when these settings are fixed.
- VisDrone results in this document use `fraction=0.2`. They are useful for LoRA behavior comparison but should not be presented as full-dataset benchmark numbers.
- The provided scripts run experiments sequentially to keep resource measurements comparable.
