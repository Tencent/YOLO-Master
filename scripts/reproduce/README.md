# Rhino-Bird Issue #49 Reproduction

This directory contains reproducible training entry points for Tencent Rhino-Bird issue #49. The scripts compare two official Nano models:

| script key | model | config |
| --- | --- | --- |
| `v01` | YOLO-Master-v0.1-N | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` |
| `esmoe` | YOLO-Master-EsMoE-N | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` |

## Files Added Or Updated

| file | purpose |
| --- | --- |
| `scripts/reproduce/reproduce_visdrone.py` | Reproduce VisDrone training and summary generation. |
| `scripts/reproduce/reproduce_sku110k.py` | Reproduce SKU-110K training and summary generation. |
| `scripts/reproduce/reproduce_common.py` | Shared dataset preparation, W&B setup, training, and summary helpers. |
| `scripts/reproduce/README.md` | Commands, W&B links, results, and known issues for issue #49. |
| `ultralytics/nn/modules/moe/modules.py` | Makes EsMoE validation use dense expert aggregation by default; sparse inference remains explicit opt-in. |
| `tests/test_moe.py` | Regression coverage for the dense-by-default eval path and explicit sparse inference opt-in. |

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

Online W&B logging requires a local login. Do not commit API keys.

```bash
wandb login
wandb status
```

The scripts initialize one W&B run per model and print the run URL to the log. Use `--wandb offline` if online upload is unstable, then sync after training:

```bash
wandb sync wandb/offline-run-*
```

## Dataset Preparation

VisDrone is about 2.3 GB to download and about 3.7 GB after conversion:

```bash
python scripts/reproduce/reproduce_visdrone.py --download-dataset --prepare-data-only
```

SKU-110K is about 13.6 GB and should be downloaded only when enough disk space is available:

```bash
python scripts/reproduce/reproduce_sku110k.py --download-dataset --prepare-data-only
```

Dataset files stay under `datasets/VisDrone` and `datasets/SKU-110K`. The scripts generate local dataset YAML files under `runs/reproduce/<dataset>/_data/`.

## Full Training

Recommended issue #49 settings are `imgsz=640` and `epochs=100~300`.

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

## Result Summary

Each run writes Ultralytics `results.csv`. The scripts also aggregate final and best metrics into `runs/reproduce/<dataset>/summary.csv` and `summary.md`.

To refresh summaries without training:

```bash
python scripts/reproduce/reproduce_visdrone.py --summary-only
python scripts/reproduce/reproduce_sku110k.py --summary-only
```

The summary includes mAP50, mAP50-95, box loss, cls loss, dfl loss, and `moe_loss` when present.

## Reproduction Run Records

Settings: `imgsz=640`, `epochs=100`, `batch=8`, one RTX 3090 per run. W&B project: <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49>.

The table reports final-epoch metrics. `default, pre-fix sparse` records the old EsMoE eval behavior; `dense/no-sparse` records the fixed validation path.

| Model | Dataset | Eval | Epochs | mAP50 | mAP50-95 | box_loss | cls_loss | moe_loss | W&B |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| YOLO-Master-v0.1-N | VisDrone | default | 100 | 0.29555 | 0.17046 | 1.34371 | 0.97464 | 0 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/5fi8avsl> |
| YOLO-Master-EsMoE-N | VisDrone | default, pre-fix sparse | 100 | 0.08387 | 0.04225 | 1.32176 | 0.94683 | 0 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/mhoislnw> |
| YOLO-Master-EsMoE-N | VisDrone | dense/no-sparse | 100 | 0.30612 | 0.17615 | 1.32176 | 0.94683 | 0 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/4v15e3o8> |
| YOLO-Master-v0.1-N | SKU-110K | default | 100 | 0.89253 | 0.56774 | 1.32078 | 0.56805 | 0 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/6np8a4lu> |
| YOLO-Master-EsMoE-N | SKU-110K | default, pre-fix sparse | 54 | 0.09703 | 0.04207 | 1.36981 | 0.61071 | 0.00982 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/oddia0uf> |
| YOLO-Master-EsMoE-N | SKU-110K | dense/no-sparse | 100 | 0.89498 | 0.57162 | 1.30965 | 0.55994 | 0 | <https://wandb.ai/2063055270-harbin-institute-of-technology/yolo-master-issue-49/runs/oiomzvte> |

Best-checkpoint metrics:

| Model | Dataset | Eval | best mAP50 epoch | best mAP50 | best mAP50-95 epoch | best mAP50-95 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| YOLO-Master-v0.1-N | VisDrone | default | 90 | 0.29739 | 99 | 0.17080 |
| YOLO-Master-EsMoE-N | VisDrone | default, pre-fix sparse | 45 | 0.09132 | 63 | 0.04318 |
| YOLO-Master-EsMoE-N | VisDrone | dense/no-sparse | 90 | 0.30735 | 96 | 0.17664 |
| YOLO-Master-v0.1-N | SKU-110K | default | 99 | 0.89259 | 99 | 0.56782 |
| YOLO-Master-EsMoE-N | SKU-110K | default, pre-fix sparse | 5 | 0.42847 | 4 | 0.19691 |
| YOLO-Master-EsMoE-N | SKU-110K | dense/no-sparse | 82 | 0.89740 | 82 | 0.57266 |

## Known Issues

- SKU-110K is large; verify disk space before `--download-dataset`.
- Use `--wandb offline` if W&B credentials or network are unavailable.
- SKU-110K contains JPEG warnings such as `Corrupt JPEG data`; training and validation continue, but the warnings are preserved in the logs.
- W&B can print `Tried to log to step ...` warnings during final artifact upload; these are non-fatal.
- If W&B online upload times out, use `--wandb offline` and run `wandb sync` after training.
- For strict 100-epoch EsMoE runs, disable EarlyStopping with `--patience 0` or set a larger value such as `--patience 300`.
- EsMoE validation must use dense expert aggregation unless sparse inference is explicitly being benchmarked. The sparse path prunes low-confidence experts and can under-report validation accuracy.
- Concurrent dataset-cache creation can corrupt `labels.cache` if multiple independent training processes prepare the same dataset simultaneously. This reproduction includes atomic cache writes and corrupted-cache fallback handling.
- The upstream checkout imports `ultralytics.utils.lora.sensitivity`, but that module is absent in this checkout. A lightweight compatibility shim is included so `from ultralytics import YOLO` works.

## Dense-Eval Rerun Commands

Use these commands to reproduce the fixed EsMoE dense-eval runs:

```bash
python scripts/reproduce/reproduce_visdrone.py \
  --model esmoe \
  --name-prefix visdrone_100e_denseeval_ \
  --epochs 100 \
  --patience 0 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --plots \
  --wandb online \
  --wandb-project yolo-master-issue-49 \
  --wandb-tags visdrone,100e,dense-eval,esmoe,rerun
```

```bash
python scripts/reproduce/reproduce_sku110k.py \
  --model esmoe \
  --name-prefix sku110k_100e_denseeval_offline_ \
  --epochs 100 \
  --patience 0 \
  --imgsz 640 \
  --batch 8 \
  --device 1 \
  --workers 8 \
  --plots \
  --wandb offline \
  --wandb-project yolo-master-issue-49 \
  --wandb-tags sku110k,100e,dense-eval,esmoe,rerun,offline-sync
```
