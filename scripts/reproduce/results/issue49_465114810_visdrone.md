# Issue #49 VisDrone reproduction results

Git commit: `096f09a`

Configuration: `epochs=100`, `imgsz=640`, `batch=32`, `workers=8`, `seed=42`

GPU: NVIDIA GeForce RTX 4090 D

W&B project: `yolo-master-issue49-465114810`


| Model | Evaluation | Best epoch | Precision | Recall | mAP50 | mAP50-95 | Hours | Best weight | W&B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| v0.1-N | standard | 88 | 0.43025 | 0.33008 | 0.30384 | 0.17009 | 2.178 | 14.8 MB | https://wandb.ai/465114810-ab/yolo-master-issue49-465114810/runs/o2ur8w66 |
| EsMoE-N | dense | 89 | 0.43094 | 0.33943 | 0.30861 | 0.17344 | 2.962 | 5.8 MB | https://wandb.ai/465114810-ab/yolo-master-issue49-465114810/runs/6qd0dn71 |

Notes: EsMoE-N uses corrected dense validation with `--no-sparse-eval`.

The routed auxiliary loss is also logged under the canonical name `moe_loss`.
