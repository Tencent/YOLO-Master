# Vertical Dataset Baseline Reproduction

This directory provides reproducible scripts for comparing `YOLO-Master-v0.1-N` and `YOLO-Master-EsMoE-N` on dense vertical detection datasets.

## Datasets

The scripts use the built-in dataset definitions:

| Dataset | Config | Scenario | Notes |
| :--- | :--- | :--- | :--- |
| VisDrone | `ultralytics/cfg/datasets/VisDrone.yaml` | Aerial dense small objects | Downloads about 2.3 GB and converts annotations to YOLO format. |
| SKU-110K | `ultralytics/cfg/datasets/SKU-110K.yaml` | Retail dense products | Downloads about 13.6 GB and converts CSV boxes to YOLO labels. |

Dataset download is handled by Ultralytics when training starts. To pre-check dataset loading, run a short smoke command:

```bash
yolo train model=YOLO-Master-v0.1-N.pt data=VisDrone.yaml imgsz=640 epochs=1 batch=4
yolo train model=YOLO-Master-v0.1-N.pt data=SKU-110K.yaml imgsz=640 epochs=1 batch=4
```

## Training Commands

Recommended reproduction settings follow the issue requirement: `imgsz=640`, `epochs=100~300`.

```bash
python scripts/reproduce/reproduce_visdrone.py --epochs 100 --imgsz 640 --device 0 --wandb-project yolo-master-reproduce
python scripts/reproduce/reproduce_sku110k.py --epochs 100 --imgsz 640 --device 0 --wandb-project yolo-master-reproduce
```

For larger GPU budgets, increase epochs:

```bash
python scripts/reproduce/reproduce_visdrone.py --epochs 300 --imgsz 640 --device 0
python scripts/reproduce/reproduce_sku110k.py --epochs 300 --imgsz 640 --device 0
```

Use dry-run mode to inspect the exact commands:

```bash
python scripts/reproduce/reproduce_visdrone.py --dry-run
python scripts/reproduce/reproduce_sku110k.py --dry-run
```

## Logging

Each script launches two runs:

| Model key | Weights |
| :--- | :--- |
| `v0.1-n` | `YOLO-Master-v0.1-N.pt` from the `YOLO-Master-v26.02` release |
| `esmmoe-n` | `YOLO-Master-EsMoE-N.pt` from the `YOLO-Master-v26.02` release |

Ultralytics writes per-epoch logs to each run directory, including:

- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`
- `train/box_loss`
- `train/cls_loss`
- `train/moe_loss` when the model reports MoE auxiliary loss
- `val/box_loss`
- `val/cls_loss`
- `val/moe_loss` when available

When W&B is installed and authenticated, pass `--wandb-project <name>` and publish the resulting W&B run URLs in the table below.

## Result Tables

The scripts write summary CSV files after successful training:

- `runs/reproduce/visdrone/visdrone_summary.csv`
- `runs/reproduce/sku110k/sku110k_summary.csv`

Record final results here after full training:

| Dataset | Model | Epochs | mAP50 | mAP50-95 | box_loss | cls_loss | moe_loss | W&B URL |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| VisDrone | YOLO-Master-v0.1-N | 100~300 | fill from summary | fill from summary | fill from summary | fill from summary | N/A or fill from summary | fill after run |
| VisDrone | YOLO-Master-EsMoE-N | 100~300 | fill from summary | fill from summary | fill from summary | fill from summary | fill from summary | fill after run |
| SKU-110K | YOLO-Master-v0.1-N | 100~300 | fill from summary | fill from summary | fill from summary | fill from summary | N/A or fill from summary | fill after run |
| SKU-110K | YOLO-Master-EsMoE-N | 100~300 | fill from summary | fill from summary | fill from summary | fill from summary | fill from summary | fill after run |

## Expected Trends

- VisDrone stresses tiny objects and dense scenes; EsMoE should be monitored for better recall/mAP50-95 under high object density.
- SKU-110K stresses extreme retail crowding; high `max_det` is set to avoid clipping dense predictions during validation.
- `YOLO-Master-v0.1-N` provides the non-EsMoE nano baseline from the same release family.
- `YOLO-Master-EsMoE-N` should additionally report MoE auxiliary loss when enabled by the model/trainer path.

## Known Issues And Fixes

- Dataset download timeout: manually download the archives referenced in the dataset YAML and place them under `datasets/` before rerunning.
- SKU-110K dependency errors: install dataset conversion dependencies such as `polars` and `numpy` before first run.
- CUDA OOM: lower `batch`, use `--batch 8`, or reduce dataloader workers before changing `imgsz`.
- W&B not logging: run `wandb login` first, or set `--wandb-mode offline` and sync later.
- Missing `moe_loss`: non-MoE baselines may not emit this metric; leave the table cell as `N/A`.
