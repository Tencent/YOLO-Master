# YOLO-Master-EsMoE-N LoRA Scenario Guide

This guide documents two deliberately different LoRA adaptation scenes for YOLO-Master-EsMoE-N:

| Scene | Dataset | Transfer setting | Config |
| :--- | :--- | :--- | :--- |
| Dense aerial detection | `VisDrone.yaml` | many tiny objects, heavy scale variation, crowded images | `yolo_master_visdrone_lora.yaml` |
| Sparse medical detection | `brain-tumor.yaml` | few boxes per image, grayscale-like MRI signal, small dataset | `yolo_master_brain_tumor_lora.yaml` |

Both configs expose the requested LoRA controls: `lora_r`, `lora_alpha`, `lora_use_rslora`, `lora_target_modules`, `lora_include_attention`, and `lora_gradient_checkpointing`.

## Config Choices

| Setting | VisDrone | brain-tumor |
| :--- | :--- | :--- |
| Default rank | `8` | `4` |
| Epochs | `30` | `40` |
| Data fraction | `0.25` | `1.0` |
| Image size | `768` | `640` |
| Batch size | `8` | `16` |
| `lora_use_rslora` | `True` | `True` |
| `lora_include_attention` | `False` | `False` |
| `lora_gradient_checkpointing` | `True` | `True` |
| Router/gating LoRA | excluded | excluded |

The target modules include backbone convolutions and EsMoE expert paths:

```yaml
lora_target_modules:
  - conv
  - fused_conv
  - bottleneck.0
  - shared_feature.0
  - static_net.3
  - proj
  - expert_projections.0.0
  ...
  - expert_projections.15.0
```

Routing and gating layers are intentionally excluded via:

```yaml
lora_exclude_modules: ["router", "routing", "gate", "gating"]
```

For short few-shot runs, router LoRA can change expert assignment before the target dataset has enough examples to stabilize the routing distribution. The default recipes adapt the expert/backbone feature transforms while keeping the router priors fixed. Router adaptation should be treated as a separate ablation.

## Rank Sweep

Run the same `r=4,8,16` comparison for both scenes:

```bash
python examples/lora_examples/run_yolo_master_lora_rank_sweep.py --scene all --device 0
```

The helper writes `examples/lora_examples/yolo_master_lora_rank_sweep_results.csv` and logs to `runs/lora_rank_sweeps/logs/`. Rank overrides use `lora_alpha=2*r`.

## Current Results

All runs use YOLO-Master-EsMoE-N with the scene configs above, `lora_use_rslora=True`, `lora_include_attention=False`, and router/gating LoRA excluded.

| Scene | Rank | Epochs | Fraction | mAP50-95 | Best epoch | Trainable params | Adapter params | Train time | Peak VRAM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |
| brain-tumor | 4 | 40 | 1.0 | 0.27129 | 36 | 468,290 | 123,116 | 13.31 min | 3.20 GB |
| brain-tumor | 8 | 40 | 1.0 | 0.31259 | 40 | 596,782 | 251,608 | 13.74 min | 3.21 GB |
| brain-tumor | 16 | 40 | 1.0 | 0.33535 | 37 | 848,390 | 503,216 | 14.82 min | 3.30 GB |
| VisDrone | 4 | 30 | 0.25 | 0.01239 | 20 | 472,410 | 123,116 | 34.82 min | 12.50 GB |
| VisDrone | 8 | 30 | 0.25 | pending | pending | 600,902 | 251,608 | running | 11.10 GB so far |
| VisDrone | 16 | 30 | 0.25 | 0.02615 | 26 | 852,510 | 503,216 | 36.04 min | 12.60 GB |

## Rank Recommendations

For brain-tumor, `r=16` is currently the best completed rank by mAP50-95. The gain from `r=8` to `r=16` is modest but measurable, while VRAM remains nearly flat on this small dataset. Use `r=8` if faster iteration is preferred, and use `r=16` for the current best accuracy.

For VisDrone, compare `r=8` after it finishes before locking the recommendation. Among completed runs, `r=16` is better than `r=4` on dense aerial detection. This matches the expectation that crowded small-object scenes need more adapter capacity.

## Target Module Guidance

Use convolution and MoE expert modules first. These cover the domain-specific feature transforms while preserving the routing policy. Keep `lora_include_attention=False` for the default YOLO-Master-EsMoE-N configs because the current stable recipes focus on conv and expert-projection adaptation; attention LoRA can be revisited as a separate ablation.

Keep `lora_use_rslora=True` when sweeping rank. RS-LoRA improves scaling stability as rank increases, especially for `r=16`.

Keep `lora_gradient_checkpointing=True` for both scenes. VisDrone is memory-heavy because of larger images and many instances, and checkpointing keeps the sweep practical on a single 24 GB GPU.

## Common Pitfalls

Medical grayscale handling: many MRI exports are single-channel or grayscale RGB. Confirm that dataset preprocessing feeds the expected channel format, disable unnecessary color-heavy augmentation when debugging, and inspect `train_batch*.jpg` before trusting metrics.

Sparse medical overfitting: brain-tumor has few boxes and limited visual diversity. Freezing BN, using dropout, and keeping router LoRA excluded help avoid memorizing scanner or annotation artifacts.

Aerial scale variation: VisDrone objects can be extremely small and densely packed. Use larger validation `max_det`, keep `imgsz` consistent across rank sweeps, and avoid comparing ranks trained with different data fractions.

Router ablations: if you remove `router`, `routing`, `gate`, or `gating` from `lora_exclude_modules`, run it as a separate experiment and watch validation mAP, MoE balance losses, and expert usage. Training loss can improve while routing drift hurts validation.

Metric comparability: keep epochs, fraction, image size, batch size, seed, and hardware fixed across ranks before comparing train time or peak VRAM.
