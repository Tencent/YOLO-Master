# Baseline Training for Vertical Datasets (Issue #49)

Reproducible training workflow for [Tencent/YOLO-Master issue #49](https://github.com/Tencent/YOLO-Master/issues/49), focused on the dense-scene vertical datasets `VisDrone` and `GlobalWheat2020`, with per-epoch logging of the required metrics (`mAP50`, `mAP50-95`, `box_loss`, `cls_loss`, `moe_loss`).

Files in this directory:

- [`yolo_master_issue_49.py`](./yolo_master_issue_49.py): command-line training entry used by the shell commands in this README.
- [`yolo_master_issue_49.ipynb`](./yolo_master_issue_49.ipynb): notebook version of the same workflow and functionality; after completing the environment setup in [Prerequisites](#prerequisites), open it in Jupyter or an IDE notebook runner and execute the cells directly, without using the command-line commands below.

## Prerequisites

This reproduction was run with the following setup:

| Item | Value                                   |
| --- |-----------------------------------------|
| OS | `Windows 10 10.0.19045`                 |
| CPU | `6` physical cores / `12` logical cores |
| GPU | `NVIDIA GeForce RTX 2070 (8GB)`         |
| Python | `CPython 3.11.15`                       |
| PyTorch | `2.5.1`                                 |
| CUDA | `12.1`                                  |
| W&B CLI | `0.28.0`                                |

For environment setup, follow the official repository installation flow and install the current repo in editable mode:

```bash
python -m pip install -e .
```

Optionally, enable Weights & Biases tracking:

```bash
python -m pip install wandb
wandb login
```

Notes:

- If you enable W&B, `wandb login` will prompt for your API key. If needed, create or copy it from your W&B account settings page.
- If you train on GPU, make sure your `torch` build matches the local CUDA environment.

## Dataset Download

The training command already includes dataset download and validation.

```bash
python scripts/issue49/yolo_master_issue_49.py --dataset VisDrone --model YOLO-Master-v0.1-N
```

The script inherits the repository dataset preparation flow and calls `check_det_dataset(..., autodownload=True)` internally, so dataset download and path checks are completed before training starts.

If you want to inspect the built-in datasets or prepare them separately, use the commands below.

List the built-in datasets and models:

```bash
python scripts/issue49/yolo_master_issue_49.py --list-datasets
python scripts/issue49/yolo_master_issue_49.py --list-models
```

Notes:

- `VisDrone` and `GlobalWheat2020` are prepared under `datasets/` by default.
- `VisDrone` requires about `2.3 GB` of disk space, and `GlobalWheat2020` requires about `7.0 GB` of disk space.


## Training Commands

Base training command:

```bash
python scripts/issue49/yolo_master_issue_49.py --dataset VisDrone --model YOLO-Master-v0.1-N
```

This script keeps LoRA and MoLoRA disabled by default so issue49 runs stay baseline-only. If you explicitly want to test the repo-wide LoRA defaults, add `--enable-lora`.

Common optional arguments:

| Flag | Default | Explanation |
| --- | --- | --- |
| `--dataset {VisDrone,SKU-110K,GlobalWheat2020}` | `VisDrone` | Select a built-in dataset. |
| `--dataset path/to/data.yaml --dataset-name MyDataset` | off | Use a custom dataset YAML. |
| `--model {YOLO-Master-v0.1-N,YOLO-Master-EsMoE-N}` | `YOLO-Master-v0.1-N` | Select a built-in model. |
| `--model path/to/model.yaml --model-name MyModel` | off | Use a custom model YAML. |
| `--uses-esmoe` | off | Enable ES-MoE-specific handling for a custom ES-MoE model. |
| `--dense-eval-for-esmoe` | off | Enable dense evaluation during validation for ES-MoE models. |
| `--enable-lora` | off | Opt into the repo-wide LoRA defaults. Default behavior keeps baseline training free of LoRA/MoLoRA. |
| `--epochs / --imgsz / --batch` | `100 / 640 / 16` | Training epochs, input image size, and batch size. |
| `--device / --workers` | `0 if CUDA else cpu / 4` | Device and DataLoader worker count. On Windows, `0` or `2` is often a good starting point. |
| `--run-tag <tag>` | auto | Custom run tag; defaults to auto-generated names such as `run001`, `run002`, and so on. |
| `--wandb-group <group>` | `visdrone` | W&B group name. |
| `--wandb-mode {online,offline,disabled}` | `online` | W&B mode. `offline` is often useful to reduce online syncing overhead. |
| `--no-wandb` | off | Disable W&B logging. |


## Expected Results

W&B report：[Issue_49 VisDrone_and_GlobalWheat2020](https://api.wandb.ai/links/zheliang-/ljtd5vog)

| Dataset | Model | Key Hparams | mAP50 | mAP50-95 | Runtime |
| --- | --- | --- | --- | --- | --- |
| `VisDrone` | `YOLO-Master-v0.1-N` | `Default` | `0.30571` | `0.17569` | `8h 46m 33s` |
| `VisDrone` | `YOLO-Master-EsMoE-N` | `Default` | `0.09246` | `0.03875` | `7h 54m 0s` |
| `VisDrone` | `YOLO-Master-EsMoE-N` | `dense eval` | `0.32282` | `0.18649` | `7h 49m 9s` |
| `GlobalWheat2020` | `YOLO-Master-v0.1-N` | `Default` | `0.96843` | `0.63372` | `2h 49m 49s` |
| `GlobalWheat2020` | `YOLO-Master-EsMoE-N` | `Default` | `0.82249` | `0.44471` | `2h 15m 2s` |
| `GlobalWheat2020` | `YOLO-Master-EsMoE-N` | `dense eval` | `0.96473` | `0.62405` | `2h 15m 56s` |

## Reproduction Results (This PR)

**Hardware:** NVIDIA GeForce RTX 4080 Laptop GPU (12GB), Windows 11, Python 3.12, PyTorch 2.5.1+cu124

**Training Configuration:** `imgsz=640`, `batch=12~13`, `epochs=100`, `workers=2`

**Commands:**
```bash
# VisDrone + v0.1-N
python scripts/issue49/yolo_master_issue_49.py --dataset VisDrone --model YOLO-Master-v0.1-N --batch 12 --epochs 100 --workers 2

# VisDrone + EsMoE-N (dense eval — REQUIRED)
python scripts/issue49/yolo_master_issue_49.py --dataset VisDrone --model YOLO-Master-EsMoE-N --batch 12 --epochs 100 --workers 2 --dense-eval-for-esmoe

# SKU-110K + v0.1-N
python scripts/issue49/yolo_master_issue_49.py --dataset SKU-110K --model YOLO-Master-v0.1-N --batch 13 --epochs 100 --workers 2

# SKU-110K + EsMoE-N (dense eval — REQUIRED)
python scripts/issue49/yolo_master_issue_49.py --dataset SKU-110K --model YOLO-Master-EsMoE-N --batch 13 --epochs 100 --workers 2 --dense-eval-for-esmoe
```

**Results:**

| Dataset | Model | mAP50 | mAP50-95 | box_loss | cls_loss | moe_loss* | Runtime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `VisDrone` | `YOLO-Master-v0.1-N` | `0.3044` | `0.1712` | `1.388` | `0.936` | `1.000` | ~4.9h |
| `VisDrone` | `YOLO-Master-EsMoE-N` | `0.3098` | `0.1744` | `1.397` | `0.952` | `1.000` | ~4.6h |
| `SKU-110K` | `YOLO-Master-v0.1-N` | `0.8827` | `0.5415` | `1.301` | `0.553` | `1.000` | ~4.1h |
| `SKU-110K` | `YOLO-Master-EsMoE-N` | `0.8835` | `0.5442` | `1.305` | `0.550` | `1.000` | ~4.7h |

*\*moe_loss recorded as `mixture_aux_loss` in CSV/W&B — this is the MoE auxiliary loss.*

**W&B Project:** https://wandb.ai/acjojo2-communication-university-of-china-nanjing/yolo_master_issue49

### Key Observations

1. **Dense eval is critical for EsMoE-N on dense scenes.** Without `--dense-eval-for-esmoe`, VisDrone mAP50 drops from ~0.31 to ~0.09 due to sparse inference approximating away location-specific expert specialization.

2. **Batch size is GPU-memory sensitive.** RTX 4080 Laptop (12GB) OOMs at `batch=16`. `batch=12~13` is the safe upper bound (~8.5-9G VRAM).

## Known Issues

### 1. Unexpected LoRA gets enabled in baseline runs

**Symptom.** A baseline issue49 run unexpectedly enables LoRA and may later hit instability patterns such as `NaN` losses or `Fitness collapse detected`.

**Cause.** Before commit `2eb330e`, the repository-wide defaults did not enable LoRA. Since commit `2eb330e`, `ultralytics/cfg/default.yaml` sets `lora_r: 16`, so baseline runs can silently enable LoRA.

**Resolution.** This issue49 script now disables LoRA and MoLoRA by default. Keep using the default command line for baseline runs, and only pass `--enable-lora` when you intentionally want a LoRA experiment.

### 2. `GlobalWheat2020 + YOLO-Master-EsMoE-N` may fail in late training with NaN/Inf checkpoints

**Symptom.** `GlobalWheat2020 + YOLO-Master-EsMoE-N` may run normally for many epochs and then fail with `Loss NaN/Inf detected`, `Fitness collapse detected`, or `Checkpoint ... last.pt is corrupted with NaN/Inf weights`.

**Cause.** This is a training-stability issue under the current repository defaults. In the observed failures, `NaN/Inf` values entered checkpoint EMA weights through corrupted BatchNorm running statistics inside `ES_MOE`, so the trainer could no longer recover from `last.pt`.

**Resolution.** Try a smaller base learning rate with `--lr0`, such as `0.001`. If instability persists, make `amp` off.

### 3. `YOLO-Master-EsMoE-N` default eval can underperform sharply on `VisDrone`

**Symptom.** `YOLO-Master-EsMoE-N` can show a large gap between default eval and `dense eval`, especially on `VisDrone`. In the recorded runs, `VisDrone` `mAP50-95` changed from `0.03875` under the default setting to `0.18649` with `dense eval`, while `GlobalWheat2020` showed a smaller gap.

**Cause.** `ES_MOE` trains with dense expert aggregation but validates with sparse inference by default. In the sparse path, routing weights are first averaged over space, then only Top-K experts are kept, and low-confidence experts can be pruned again by `dynamic_threshold`. This approximation removes location-specific expert specialization, which hurts dense, multi-scale, small-object scenes such as `VisDrone` much more than simpler single-class datasets such as `GlobalWheat2020`.

**Resolution.** When comparing `YOLO-Master-EsMoE-N` against other models, treat default eval and `dense eval` as different settings rather than interchangeable measurements.


