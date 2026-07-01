# Issue #49 Reproduction Results

This file summarizes the completed vertical-dataset reproduction runs for issue #49.

## Scope

- Datasets:
  - `ultralytics/cfg/datasets/VisDrone.yaml`
  - `ultralytics/cfg/datasets/SKU-110K.yaml`
- Models:
  - `YOLO-Master-EsMoE-N`: `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml`
  - `YOLO-Master-v0.1-N`: `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml`
- Training config:
  - `imgsz=640`
  - `epochs=100`
  - `batch=16`
  - `seed=42`

## Commands

```bash
bash scripts/run_reproduce.sh --dataset visdrone --model all --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8 --wandb-mode online
bash scripts/run_reproduce.sh --dataset sku110k --model all --epochs 100 --imgsz 640 --batch 16 --device 0 --workers 8 --wandb-mode online
```

## Result Table

Final metrics are from the last row of each `results.csv`. Best metrics are selected by maximum `metrics/mAP50-95(B)`.

| Dataset | Model | Final epoch | Final mAP50 | Final mAP50-95 | Best epoch | Best mAP50 | Best mAP50-95 | Final box_loss | Final cls_loss | Final moe_loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VisDrone | YOLO-Master-EsMoE-N | 100 | 0.07506 | 0.03235 | 45 | 0.18737 | 0.12825 | 1.30428 | 0.92438 | 0 |
| VisDrone | YOLO-Master-v0.1-N | 100 | 0.30367 | 0.17375 | 88 | 0.30385 | 0.17398 | 1.31942 | 0.94919 | 0 |
| SKU-110K | YOLO-Master-EsMoE-N | 100 | 0.27438 | 0.11400 | 6 | 0.39718 | 0.17727 | 1.31196 | 0.56701 | 0 |
| SKU-110K | YOLO-Master-v0.1-N | 100 | 0.89353 | 0.56935 | 98 | 0.89347 | 0.56978 | 1.30742 | 0.55998 | 0 |

## Per-Epoch Logs

Each CSV contains one row per epoch with `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `train/box_loss`, `train/cls_loss`, `train/moe_loss`, validation losses, and learning rates.

| Dataset | Model | Per-epoch CSV |
| --- | --- | --- |
| VisDrone | YOLO-Master-EsMoE-N | `scripts/reproduce/results/visdrone_esmoe_n_e100_results.csv` |
| VisDrone | YOLO-Master-v0.1-N | `scripts/reproduce/results/visdrone_v0_1_n_e100_results.csv` |
| SKU-110K | YOLO-Master-EsMoE-N | `scripts/reproduce/results/sku110k_esmoe_n_e100_results.csv` |
| SKU-110K | YOLO-Master-v0.1-N | `scripts/reproduce/results/sku110k_v0_1_n_e100_results.csv` |

## Raw Terminal Logs

The wrapper generated raw stdout/stderr logs under `runs/`, which is ignored by git to avoid committing large runtime artifacts:

| Dataset | Raw log |
| --- | --- |
| VisDrone | `runs/issue49_visdrone/logs/visdrone_all_e100_20260701_003909.log` |
| SKU-110K | `runs/issue49_sku110k/logs/sku110k_all_e100_20260701_095737.log` |

## Known Issues

- W&B public URLs were not written into `wandb_url.txt`, so `wandb_url` columns are empty in the generated summaries.
- `moe_loss` is present in all `results.csv` files and is `0` for these runs.
- Raw terminal logs are large and live under ignored `runs/`; the committed per-epoch CSV files provide the required epoch-level metrics.
