# VisDrone MoT Hybrid Architecture Ablation: Stable Training, Negative Hybrid Synergy, and Layer-Specific Occlusion Routing

This post summarizes the YOLO-Master v0.10 MoT hybrid architecture experiment on VisDrone. The goal was to compare the MoE baseline, MoT, MoA, and a MoA+MoT hybrid; benchmark accuracy, latency, FLOPs, and stability; and inspect MoT expert routing behavior across scene groups.

Published experiment branch (four-model results and plotting assets):

```text
https://github.com/kimariyb/YOLO-Master/tree/feat/mot-hybrid-architecture
```

Experiment scripts:

- `scripts/compare_mot_ablation.py`
- `scripts/prepare_mot_routing_scenes.py`
- `scripts/diagnose_mot_routing.py`
- `examples/mot_hybrid_architecture/plot_mot_results.py`

## Compared Models

| Key | Model | Role |
| --- | --- | --- |
| `v10` | YOLO-Master-v0.10-EsMoE-N | MoE baseline |
| `v10_mot` | YOLO-Master-v0.10-MoT-N | MoT experimental module |
| `v10_moa` | YOLO-Master-v0.10-MoA-N | MoA comparison group |
| `v10_moa_mot` | YOLO-Master-v0.10-MoA+MoT-N | Hybrid architecture exploration |

All runs used VisDrone, 50 epochs, `imgsz=640`, `amp=False`, `device=0`, and `workers=8`.

| Item | Value |
| --- | --- |
| Python version | 3.12.13 |
| Ultralytics version | 8.3.240 |
| PyTorch version | 2.10.0+cu128 |
| GPU | Quadro RTX 6000 |
| GPU memory reported by Ultralytics | 22684 MiB |
| AMP | `False` |
| Image size | 640 |
| Data workers | 8 |
| Benchmark reps | 200 |
| FLOPs method | `torch_profile_actual` |

The required three models used batch 16. **The heavy MoA+MoT hybrid hit CUDA OOM at batch 16 and was completed with batch 8**. Latency and actual FLOPs were benchmarked consistently with `imgsz=640`, `cuda:0`, 200 repetitions, and `torch_profile_actual`.

## Accuracy, Latency, FLOPs, Params, Stability

| Model | mAP50-95 | mAP50 | P50 ms | P95 ms | P99 ms | GFLOPs | Params M | Final train loss | NaN | Diverged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| EsMoE-N | 0.12324 | 0.22356 | 34.355 | 35.184 | 37.371 | 8.671 | 3.450 | 3.70817 | No | No |
| MoT-N | 0.12081 | 0.22248 | 61.971 | 63.168 | 64.935 | 12.270 | 4.055 | 3.75120 | No | No |
| MoA-N | 0.11933 | 0.21697 | 58.788 | 59.924 | 63.708 | 10.072 | 3.577 | 3.74733 | No | No |
| MoA+MoT-N | 0.11789 | 0.21942 | 63.807 | 64.557 | 65.267 | 15.568 | 4.057 | 3.85280 | No | No |

The EsMoE baseline is the best accuracy/latency point in this experiment. MoT-N is stable but loses 0.00243 mAP50-95 and adds 27.616 ms P50 latency. MoA+MoT-N does not produce hybrid synergy: it loses 0.00535 mAP50-95 and adds 29.452 ms P50 latency relative to EsMoE-N.

## Figures

The Seaborn figures are embedded below and stored in the example result directory.

![Model tradeoff](https://raw.githubusercontent.com/kimariyb/YOLO-Master/feat/mot-hybrid-architecture/examples/mot_hybrid_architecture/results/figures/mot_model_tradeoff.svg)

![Training curves](https://raw.githubusercontent.com/kimariyb/YOLO-Master/feat/mot-hybrid-architecture/examples/mot_hybrid_architecture/results/figures/mot_training_curves.svg)

![Routing heatmap](https://raw.githubusercontent.com/kimariyb/YOLO-Master/feat/mot-hybrid-architecture/examples/mot_hybrid_architecture/results/figures/mot_routing_heatmap.svg)

![Deformable lift](https://raw.githubusercontent.com/kimariyb/YOLO-Master/feat/mot-hybrid-architecture/examples/mot_hybrid_architecture/results/figures/mot_deformable_lift.svg)

The ground-truth token follow-up also produces a spatial activation map. When
publishing this Markdown as a GitHub Discussion, upload
`examples/mot_hybrid_architecture/results/routing_interpretability/heatmap_spatial.png`
as an attachment at this point so GitHub replaces it with a hosted image URL.

Figure files:

- `examples/mot_hybrid_architecture/results/figures/mot_model_tradeoff.svg`
- `examples/mot_hybrid_architecture/results/figures/mot_training_curves.svg`
- `examples/mot_hybrid_architecture/results/figures/mot_routing_heatmap.svg`
- `examples/mot_hybrid_architecture/results/figures/mot_deformable_lift.svg`

They visualize model tradeoffs, per-epoch mAP/loss, MoT expert top-1 routing share, and DeformableTransformer activation lift.

## Routing Analysis

MoT routing was analyzed with a custom hook on each `MoTBlock.router`. The hook records token routing for:

- `LocalConvTransformer`
- `WindowTransformer`
- `DeformableTransformer`

Scene folders were generated from VisDrone label statistics: dense, sparse, small-object, large-object, dense-small, sparse-large, and an irregular/occluded proxy based on high box scale/aspect-ratio variation.

| Scene | Local top1 | Window top1 | Deformable top1 | Deformable mean weight |
| --- | ---: | ---: | ---: | ---: |
| dense | 0.333 | 0.000 | 0.667 | 0.339506 |
| sparse | 0.333 | 0.000 | 0.667 | 0.339495 |
| small_objects | 0.333 | 0.000 | 0.667 | 0.339502 |
| large_objects | 0.333 | 0.000 | 0.667 | 0.339498 |
| dense_small | 0.333 | 0.000 | 0.667 | 0.339506 |
| sparse_large | 0.333 | 0.000 | 0.667 | 0.339497 |
| irregular_occluded | 0.333 | 0.000 | 0.667 | 0.339505 |

The 50-epoch proxy-scene pattern is almost scene-invariant. Deformable is top-1 for two thirds of tokens in every proxy scene, LocalConv receives one third, and Window is never top-1. This result is scoped to that checkpoint and proxy grouping.

## DeformableTransformer In Irregular/Occluded Scenes

The issue asked whether `DeformableTransformer` activation rises significantly in occlusion or irregular-object scenes. The result is mixed statistically but negative practically:

- Top-1 share: no increase. `irregular_occluded` is 0.666667, identical to the pooled non-irregular baseline.
- Mean router weight: pooled non-irregular comparison has `mean_diff=0.000004`, `p=0.0002`, and relative lift about 0.0013%.
- The measured mean-weight lift is statistically detectable because variance is tiny, but the effect size is too small to call a meaningful scene-specific routing preference.

For this 50-epoch proxy run, Deformable is globally preferred but the proxy scene split does not validate a meaningful occlusion-specific increase.

### Ground-truth token-level follow-up

A separate, converged 100-epoch MoT checkpoint was analysed on all 548 VisDrone
validation images with ground-truth occlusion labels. The hook compares tokens
within images, matches object size, and controls the FDR test family.

At `model.20.m.1`, 488 size-matched single-object pairs give Deformable mean
weights 0.154569 (occluded) versus 0.136347 (clear): +13.36%, `p=8.16e-05`,
BH `q=2.35e-04`. The hypothesis is therefore supported in this layer. It is not
model-wide: `model.23.m.1` is fully collapsed to LocalConv. Forced single-expert
routing reduces mAP50-95 by 2.98%-5.25%, supporting learned mixed routing overall.

The routing checkpoint's natural mAP50-95 is 0.169861. It is a separate mechanism
study and is not substituted into the controlled 50-epoch comparison. Provenance
and reviewable outputs are in
`examples/mot_hybrid_architecture/results/routing_interpretability/`.

## Boundary Tests And Fixes

The branch also hardens the test surface:

- `MoTBlock` and `_WindowTransformerExpert` handle `window_size` larger than the feature map.
- `_WindowTransformerExpert` handles shifted windows on odd spatial sizes.
- MoT disables `exploration_eps` in eval mode.
- v0.10 MoT and MoA+MoT YAMLs parse successfully.
- `ultralytics/engine/validator.py` now imports the standalone validation helpers it uses and provides `convert_ndjson_to_yolo_if_needed`.

Current verification (Python 3.12, CPU; includes the two-rank Gloo DDP gate):

```text
144 passed, 1 skipped
```

## Scenario Recommendations

1. Use EsMoE-N as the VisDrone small-model default for this setting. It has the highest mAP50-95 (0.12324), lowest P50 latency (34.355 ms), lowest actual FLOPs (8.671 G), and stable training.
2. Avoid treating MoT-N as a free accuracy upgrade. It is stable, but mAP50-95 decreases by 0.00243 and P50 latency increases by 80.38% versus EsMoE-N.
3. Do not claim WindowTransformer specialization from the 50-epoch proxy run; its top-1 share is 0.000 there, and this is checkpoint-specific.
4. For occlusion-sensitive routing, inspect `model.20.m.1`: Deformable rises 13.36% after size matching (`q=2.35e-04`). Do not generalize to every layer.
5. Do not merge the current heavy MoA+MoT layout as a performance improvement. It is a useful negative result: -0.00535 mAP50-95 and +85.73% P50 latency versus EsMoE-N.
