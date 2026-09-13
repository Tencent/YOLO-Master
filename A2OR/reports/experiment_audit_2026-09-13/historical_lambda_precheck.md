> 来源：git HEAD (`cea362b`) 中的 `A2/runs/p1_fixedk10_baseline_vd10pct_s42_b2_w0/DYNAMIC_TOPK_COMPARISON.md`。仅复制历史文档，未恢复原实验目录。

# Matched 10% VisDrone: Fixed K=10 vs Dynamic TopK

All variants use seed=42, 648 training images, 55 validation images, 10 epochs, batch=2, imgsz=640, and workers=0.

Fixed K=10 baseline mean APs: **1.408** points; epoch-10 APs: **1.922** points.

| Lambda | Mean APs | Mean delta | Epoch-10 delta | Best per-epoch delta | Epochs ahead | All epochs >= +1.0? |
|---:|---:|---:|---:|---:|---:|:---:|
| 0.50 | 1.327 | -0.081 | -0.121 | +0.119 | 3/10 | no |
| 0.55 | 1.346 | -0.062 | -0.202 | +0.071 | 2/10 | no |
| 0.60 | 1.345 | -0.063 | -0.096 | +0.034 | 1/10 | no |
| 0.65 | 1.360 | -0.048 | -0.007 | +0.083 | 2/10 | no |
| 0.70 | 1.300 | -0.108 | -0.114 | -0.005 | 0/10 | no |
| 0.75 | 1.373 | -0.035 | -0.035 | +0.089 | 3/10 | no |

## Strict +1.0-point stability check

Qualified lambdas: none.

The criterion requires delta APs >= +1.0 point at every epoch 1-10, not only at the best checkpoint.
