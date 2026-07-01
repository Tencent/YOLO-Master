# Rhino-Bird Issue #49 Reproduction

This directory contains reproducible training entry points for Tencent Rhino-Bird issue #49:

- `reproduce_visdrone.py`
- `reproduce_sku110k.py`

The scripts compare two official Nano models:

| script key | model | config |
| --- | --- | --- |
| `v01` | YOLO-Master-v0.1-N | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` |
| `esmoe` | YOLO-Master-EsMoE-N | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` |

## Environment

```bash
conda activate yolo_master
cd /path/to/YOLO-Master
export YOLO_CONFIG_DIR="$PWD/runs/reproduce/_runtime/ultralytics"
export MPLCONFIGDIR="$PWD/runs/reproduce/_runtime/matplotlib"
```

## Preflight

```bash
python scripts/reproduce/reproduce_visdrone.py --dry-run
python scripts/reproduce/reproduce_visdrone.py --check-build
python scripts/reproduce/reproduce_sku110k.py --dry-run
python scripts/reproduce/reproduce_sku110k.py --check-build
```

## W&B Setup

Online W&B logging requires a local login. Do not commit API keys into this repository.

```bash
wandb login
wandb status
```

The scripts initialize one W&B run per model and print the run URL to the training log:

- default project: `yolo-master-issue-49`
- default group: dataset name, for example `visdrone` or `sku-110k`
- default tags: dataset name, model key, `rhino-bird-issue-49`

Use `--wandb online` after login. Use `--wandb offline` when the server has no public network or W&B credentials.

## Dataset Preparation

VisDrone is about 2.3 GB to download and about 3.7 GB after conversion:

```bash
python scripts/reproduce/reproduce_visdrone.py --download-dataset --prepare-data-only
```

SKU-110K is about 13.6 GB and should be downloaded only when enough disk space is available:

```bash
python scripts/reproduce/reproduce_sku110k.py --download-dataset --prepare-data-only
```

The scripts rewrite the dataset YAML `path:` field into a local generated file under `runs/reproduce/<dataset>/_data/`, so dataset files stay under:

- `datasets/VisDrone`
- `datasets/SKU-110K`

## Full Training

Recommended issue #49 settings are `imgsz=640` and `epochs=100~300`. Start with VisDrone because it is smaller and directly relevant to dense small-object detection.

```bash
python scripts/reproduce/reproduce_visdrone.py \
  --model all \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --plots \
  --wandb online \
  --wandb-project yolo-master-issue-49 \
  --wandb-tags visdrone,100e
```

```bash
python scripts/reproduce/reproduce_sku110k.py \
  --model all \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --plots \
  --wandb online \
  --wandb-project yolo-master-issue-49 \
  --wandb-tags sku110k,100e
```

For multi-GPU DDP:

```bash
python scripts/reproduce/reproduce_visdrone.py \
  --model all \
  --epochs 100 \
  --imgsz 640 \
  --batch 64 \
  --device 0,1,2,3 \
  --workers 16 \
  --plots \
  --wandb online \
  --wandb-project yolo-master-issue-49 \
  --wandb-tags visdrone,100e,ddp
```

## Resume

```bash
python scripts/reproduce/reproduce_visdrone.py --model all --resume --epochs 100 --device 0 --exist-ok
```

## Result Summary

Each training run writes the standard Ultralytics `results.csv`. The scripts also aggregate final metrics into:

- `runs/reproduce/visdrone/summary.csv`
- `runs/reproduce/visdrone/summary.md`
- `runs/reproduce/sku-110k/summary.csv`
- `runs/reproduce/sku-110k/summary.md`

To refresh summaries without training:

```bash
python scripts/reproduce/reproduce_visdrone.py --summary-only
python scripts/reproduce/reproduce_sku110k.py --summary-only
```

The summary includes mAP50, mAP50-95, box loss, cls loss, dfl loss, and `moe_loss` when present.
Summary refresh scans existing `results.csv` files below the dataset project directory, so smoke runs and reruns with custom `--name-prefix` values can be reported together. The generated summary reports both the final epoch metrics and best mAP metrics, because EarlyStopping or late-epoch degradation can make the final epoch misleading.

## Reproduced Results

Settings used for the current reproduction:

- `imgsz=640`
- `epochs=100`
- `batch=8`
- single RTX 3090 per run
- `wandb=online`
- project: <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49>

### VisDrone

Both VisDrone runs completed 100 epochs.

| model | run | last epoch | last mAP50 | last mAP50-95 | best mAP50 | best mAP50-95 | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO-Master-v0.1-N | `visdrone_100e_wandb_v01` | 100 | 0.29555 | 0.17046 | 0.29739 | 0.17080 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/5fi8avsl> |
| YOLO-Master-EsMoE-N | `visdrone_100e_wandb_esmoe` | 100 | 0.08387 | 0.04225 | 0.09132 | 0.04318 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/mhoislnw> |

### SKU-110K

YOLO-Master-v0.1-N completed 100 epochs. YOLO-Master-EsMoE-N trained successfully but stopped at epoch 54 due to the default Ultralytics `patience=50`; its best checkpoint is from epoch 4.

| model | run | last epoch | last mAP50 | last mAP50-95 | best mAP50 | best mAP50-95 | W&B |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO-Master-v0.1-N | `sku110k_100e_wandb_v01` | 100 | 0.89253 | 0.56774 | 0.89259 | 0.56782 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/6np8a4lu> |
| YOLO-Master-EsMoE-N | `sku110k_100e_wandb_esmoe` | 54 | 0.09703 | 0.04207 | 0.42847 | 0.19691 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/oddia0uf> |

## Known Issues

- SKU-110K is large; verify disk space before using `--download-dataset`.
- If `WANDB_API_KEY` is not configured, use `--wandb offline` or omit `--wandb online`.
- SKU-110K contains JPEG warnings such as `Corrupt JPEG data`; training and validation continue, but the warnings are preserved in the logs.
- W&B can print `Tried to log to step ...` warnings during final Ultralytics artifact upload. The warnings are non-fatal; metrics, media, and model artifacts were synced.
- For a strict 100-epoch SKU-110K EsMoE rerun, disable EarlyStopping with `--patience 0` or set a larger value such as `--patience 300`.
- Concurrent dataset-cache creation can corrupt `labels.cache` if multiple independent training processes prepare the same dataset simultaneously. This reproduction includes atomic cache writes and corrupted-cache fallback handling.
- The upstream checkout imports `ultralytics.utils.lora.sensitivity`, but that module is absent in this checkout. A lightweight compatibility shim is included so `from ultralytics import YOLO` works.

## Strict Rerun Commands

Use these commands if reviewers request strict no-EarlyStopping SKU-110K numbers:

```bash
python scripts/reproduce/reproduce_sku110k.py \
  --model esmoe \
  --name-prefix sku110k_100e_strict_ \
  --epochs 100 \
  --patience 0 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --plots \
  --wandb online \
  --wandb-project yolo-master-issue-49 \
  --wandb-tags sku110k,100e,no-early-stop
```
