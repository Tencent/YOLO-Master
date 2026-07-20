# MoA Training Validation Report — Issue #53

## Full Training: MoA vs MoE on VisDrone (50 epochs, GPU)

| Config | MoA (this work) | MoE Baseline |
|--------|-----------------|--------------|
| Model | YOLO-Master v0.10 MoA-N | YOLO-Master v0.10 N |
| Dataset | VisDrone (6471 train, 548 val, 10 cls) | VisDrone |
| Image size | 320×320 | 320×320 |
| Batch size | 24 | 24 |
| Device | RTX 5070 Ti Laptop (12 GB) | RTX 5070 Ti Laptop (12 GB) |
| Epochs | 50 | 50 |
| AMP | Disabled (FP32 NaN recovery) | Disabled (FP32 NaN recovery) |
| Train time | 4667s (78 min) | 3507s (58 min) |

### Final Results (Epoch 50)

| Metric | MoA | MoE | Delta |
|--------|-----|-----|-------|
| mAP50 | 0.133 | 0.133 | 0.000 |
| mAP50-95 | 0.069 | 0.070 | -0.001 |
| box_loss | 1.688 | 1.697 | -0.009 |
| cls_loss | 1.247 | 1.259 | -0.012 |
| dfl_loss | 0.905 | 0.905 | 0.000 |
| Train time | 78 min | 58 min | +20 min (+34%) |

### Loss Convergence

| Epoch | MoA mAP50-95 | MoE mAP50-95 | MoA box_loss | MoE box_loss |
|-------|-------------|-------------|-------------|-------------|
| 1 | 0.012 | 0.013 | 3.738 | 3.744 |
| 10 | 0.036 | 0.039 | 2.183 | 2.181 |
| 20 | 0.049 | 0.048 | 1.997 | 2.000 |
| 30 | 0.061 | 0.061 | 1.887 | 1.895 |
| 40 | 0.067 | 0.067 | 1.806 | 1.811 |
| 50 | 0.069 | 0.070 | 1.688 | 1.697 |

### Observations

1. **MoA training is stable**: No NaN divergence after initial AMP->FP32 recovery. All metrics decrease smoothly over 50 epochs.

2. **MoA router converges**: Aux loss quickly stabilizes at 2.0 (vs MoE's 1.0), reflecting the 3-group soft-routing mechanism.

3. **Comparable accuracy**: MoA and MoE achieve nearly identical mAP (0.133/0.069 vs 0.133/0.070). At 320×320 resolution with 50 epochs, MoA attention routing shows no advantage over MoE FFN routing.

4. **MoA training overhead**: +34% training time due to additional attention computation (window-partitioned SDPA, random-feature linear attention, cross-attention fusion).

5. **Both converge smoothly**: No overfitting at 50 epochs; both would benefit from longer training at higher resolution.

### Conclusions

- MoA module trains successfully on VisDrone from scratch without NaN/Inf
- MoA achieves parity with MoE baseline; attention routing is a viable alternative to FFN routing
- Training overhead is acceptable (34% slower) for potential benefits at larger scales
- Future work: higher resolution (640), longer training (100+ epochs), larger model variants

### Artifacts

- MoA results: `runs/issue-53/moa-n-visdrone/`
- MoE results: `runs/issue-53/moe-n-visdrone/`
- Loss curves: `runs/issue-53/*/results.png`
- Best checkpoints: `runs/issue-53/*/weights/best.pt`
