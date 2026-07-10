# YOLO-Master Reproduction on VisDrone and SKU-110K

This directory provides reproduction scripts and result summaries for
YOLO-Master Nano variants on VisDrone and SKU-110K. The scripts use the
repository model configs, train with the built-in Ultralytics dataset YAMLs, and
write per-epoch metrics plus compact CSV summaries.

## Files

| Path | Contents |
|------|----------|
| `reproduce_visdrone.py` | VisDrone2019-DET reproduction entrypoint. |
| `reproduce_sku110k.py` | SKU-110K reproduction entrypoint. |
| `_reproduce_common.py` | Shared training, build-check, logging, and summary utilities. |
| `_reproduce_trainer.py` | DDP-safe evaluation and metric callbacks. |

## Supported Models

| Name | Config |
|------|--------|
| `v0.1-N` | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` |
| `EsMoE-N` | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` |

## Supported Datasets

The scripts use the built-in Ultralytics dataset YAML files.

| Dataset | YAML | Train | Val | Test |
|---------|------|------:|----:|-----:|
| VisDrone | `VisDrone.yaml` | 6471 | 548 | 1610 |
| SKU-110K | `SKU-110K.yaml` | 8219 | 588 | 2936 |

## Usage

### 1. Installation

Install the repository in editable mode.

```bash
pip install -e .
```

### 2. Download Data

Download and validate each dataset with its built-in YAML configuration:

```bash
python scripts/reproduce/reproduce_visdrone.py --download-only
python scripts/reproduce/reproduce_sku110k.py --download-only
```

### 3. Check Model Build

```bash
python scripts/reproduce/reproduce_visdrone.py --check-build
python scripts/reproduce/reproduce_sku110k.py --check-build
```

### 4. Training

The reported configuration uses `imgsz=640` and 300 training epochs. AMP is
disabled because it produced non-finite loss or metric values in these runs.

```bash
# VisDrone
python scripts/reproduce/reproduce_visdrone.py \
  --model v0.1-N \
  --epochs 300 \
  --imgsz 640 \
  --batch 32 \
  --device 0,1 \
  --workers 8 \
  --no-amp

python scripts/reproduce/reproduce_visdrone.py \
  --model EsMoE-N \
  --epochs 300 \
  --imgsz 640 \
  --batch 32 \
  --device 0,1 \
  --workers 8 \
  --no-amp

python scripts/reproduce/reproduce_visdrone.py \
  --model EsMoE-N \
  --epochs 300 \
  --imgsz 640 \
  --batch 32 \
  --device 0,1 \
  --workers 8 \
  --no-amp \
  --no-sparse-eval

# SKU-110K
python scripts/reproduce/reproduce_sku110k.py \
  --model v0.1-N \
  --epochs 300 \
  --imgsz 640 \
  --batch 64 \
  --device 0,1 \
  --workers 8 \
  --no-amp

python scripts/reproduce/reproduce_sku110k.py \
  --model EsMoE-N \
  --epochs 300 \
  --imgsz 640 \
  --batch 64 \
  --device 0,1 \
  --workers 8 \
  --no-amp

python scripts/reproduce/reproduce_sku110k.py \
  --model EsMoE-N \
  --epochs 300 \
  --imgsz 640 \
  --batch 64 \
  --device 0,1 \
  --workers 8 \
  --no-amp \
  --no-sparse-eval
```

## Metrics

Ultralytics writes the following per-epoch metrics to each run's `results.csv`:

- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`
- `train/box_loss`, `train/cls_loss`, `train/moe_loss`
- `val/box_loss`, `val/cls_loss`, `val/moe_loss`

Each dataset output directory also contains:

| File | Contents |
|------|----------|
| `summary.csv` | Precision, recall, mAP, and loss values for each run. |

## Results

Training curves for all reported runs are available in the public result project
[YOLO-Master Reproduction Results](https://wandb.ai/zhangjch98-sun-yat-sen-university/yolo-master-issue49).
Each run uses 2 NVIDIA L20 GPUs, `imgsz=640`, 300 epochs, and `--no-amp`.
VisDrone uses batch 32; SKU-110K uses batch 64. mAP values are reported as
percentages.

| Dataset | Model | Eval | Epoch | mAP50:95 | mAP50 |
|---------|-------|------|------:|---------:|------:|
| VisDrone | v0.1-N | default | 300 | 20.08 | 34.20 |
| VisDrone | EsMoE-N | default | 300 | 3.28 | 6.81 |
| VisDrone | EsMoE-N | nosparse | 300 | 20.18 | 34.40 |
| SKU-110K | v0.1-N | default | 300 | 58.09 | 90.40 |
| SKU-110K | EsMoE-N | default | 300 | 1.80 | 5.28 |
| SKU-110K | EsMoE-N | nosparse | 300 | 58.35 | 90.73 |

## Known Issues

For `EsMoE-N`, the default validation path uses the shipped sparse routing.
It produces substantially lower validation mAP on these dense datasets.
`--no-sparse-eval` uses dense validation to match the training path; the two
modes are kept in separate run directories and reported separately above.

AMP produced non-finite values in these runs. The reported commands use
`--no-amp` for numerical stability.

## Output Layout

```text
runs/reproduce/<dataset>/<Dataset>_v0.1-N/
+-- args.yaml
+-- results.csv
+-- weights/

runs/reproduce/<dataset>/<Dataset>_EsMoE-N_<default|nosparse>/
+-- args.yaml
+-- results.csv
+-- weights/

runs/reproduce/<dataset>/
+-- summary.csv
```
