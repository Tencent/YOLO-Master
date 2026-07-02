# Reproducing YOLO-Master on VisDrone & SKU-110K

Reproduction of **YOLO-Master-v0.1-N** and **YOLO-Master-EsMoE-N** training on the
**VisDrone** and **SKU-110K** detection benchmarks (issue #49).

Per-epoch `mAP50 / mAP50-95 / box_loss / cls_loss / dfl_loss / moe_loss` are logged
to a public Weights & Biases project.

## Models

| Name | Config YAML | Params |
|------|-------------|--------|
| YOLO-Master-v0.1-N  | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` | ~7.54M |
| YOLO-Master-EsMoE-N | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml`   | ~2.68M |

## Environment

Reproduced on:

| Item | Value |
|------|-------|
| GPU | NVIDIA A100-SXM4-80GB (Colab Pro) |
| Python | 3.12.13 |
| PyTorch / CUDA | torch 2.11.0+cu128 (CUDA 12.8) |
| ultralytics | 8.3.240 |
| OS | Ubuntu 22.04 (Google Colab) |

Peak GPU memory was ~21 GB at `--batch 16, imgsz 640`.

## Setup

```bash
git clone -b reproduce/issue49 https://github.com/SophRsR/YOLO-Master.git
cd YOLO-Master
pip install -e . wandb seaborn      # seaborn is required by the plotting code
wandb login
```

## Datasets

Both datasets are downloaded automatically by Ultralytics on first run via their
`.yaml` configs (`VisDrone.yaml`, `SKU-110K.yaml`) — no manual download needed.
They are cached under the Ultralytics `datasets/` directory.

## Training

Each script trains **both** models on its dataset and prints a final results table.

```bash
# VisDrone (both models, 100 epochs)
python scripts/reproduce/reproduce_visdrone.py --models all --epochs 100 --batch 16 --device 0

# SKU-110K (both models, 100 epochs)
python scripts/reproduce/reproduce_sku110k.py  --models all --epochs 100 --batch 16 --device 0
```

Useful flags: `--models v0.1|esmoe|all`, `--epochs`, `--batch` (raise to 32/48 on an
A100), `--imgsz` (default 640), `--device`, `--workers`, `--project` (wandb project name).
Runs are saved under `<project>/<dataset>_<model>/`.

## Results

Trained with `imgsz=640`, `epochs=100`, `batch=16`. Official reference numbers are
from Table 1 ("Main Results") of the repo README; note the official table reports a
single "YOLO-Master-N" without splitting the v0.1 / EsMoE variants, so both rows
share the same reference.

### VisDrone

| Model | mAP50 (ours) | mAP50-95 (ours) | Official mAP50 | Official mAP50-95 |
|-------|--------------|-----------------|----------------|-------------------|
| YOLO-Master-v0.1-N  | 0.2996 | 0.1714 | 0.337 | 0.196 |
| YOLO-Master-EsMoE-N | 0.0810* | 0.0395* | 0.337 | 0.196 |

### SKU-110K

| Model | mAP50 (ours) | mAP50-95 (ours) | Official mAP50 | Official mAP50-95 |
|-------|--------------|-----------------|----------------|-------------------|
| YOLO-Master-v0.1-N  | 0.8928 | 0.5694 | 0.906 | 0.582 |
| YOLO-Master-EsMoE-N | 0.2814* | 0.1349* | 0.906 | 0.582 |

\* EsMoE-N numbers are depressed by a **validation-time bug**, not a training
failure — see Known issues #1. With dense evaluation the same training recipe
reaches parity with the official numbers (PR #81: VisDrone ≈0.203, SKU-110K
≈0.583 mAP50-95).

YOLO-Master-v0.1-N reproduces the official results closely on both datasets
(87% / 98% of the official mAP50-95, at 100 epochs vs. the official recipe).

**Weights & Biases (public):** <https://wandb.ai/sophier4-uci/yolo-master-reproduce>
(runs: `visdrone_v0_1_n`, `visdrone_esmoe_n`, `sku110k_v0_1_n`, `sku110k_esmoe_n`)

## Known issues

1. **EsMoE-N validation mAP collapses under default sparse inference** — training
   is healthy (dense forward, aux loss correctly registered), but validation runs
   `ES_MOE._sparse_forward` (`use_sparse_inference=True` by default), which prunes
   experts against `dynamic_threshold=0.4` and accumulates the survivors with their
   **raw softmax weights, without renormalizing**
   (`ultralytics/nn/modules/moe/modules.py:674`). With 3 experts the mean routing
   weight is ~0.33 < 0.4, so typically only the rank-0 expert survives and the
   output magnitude is ~1/3 of what the BatchNorm calibrated on the dense training
   path expects — mAP collapses (VisDrone 0.0395, SKU-110K 0.1349 in our runs).
   Notably, the correct renormalization pattern already exists twice in
   `routers.py` (`_soft_top_k`, `_hard_top_k`); only `ES_MOE._sparse_forward` skips
   it.

   This was also diagnosed by @skywalker-lt in **PR #81**, whose reproduce scripts
   work around it with a `--no-sparse-eval` flag (a callback setting
   `use_sparse_inference=False` on all `ES_MOE` modules of the model and its EMA,
   so per-epoch validation and final eval use the dense path that matches
   training). With that workaround, EsMoE-N reaches parity with the official
   numbers (#81: VisDrone 0.203, SKU-110K 0.583 mAP50-95) — our collapsed
   sparse-eval numbers independently confirm the diagnosis (our SKU-110K 0.1349 ≈
   #81's 0.136). The underlying `_sparse_forward` bug itself remains unfixed in
   the core library.

   **I did a trial with renormalizing `_sparse_forward` and it is not enough.**
   Adding the missing renormalization fixes the magnitude (unit-test verified) but
   still does not recover eval mAP (VisDrone dense 0.139 vs renormalized-sparse
   0.026). The router is near-uniform (experts didn't specialize), so it's a
   dense-train / sparse-eval mismatch, not just a scaling bug. Therefore, the proposal made by #81 (`use_sparse_inference=False`) might be more reliable

2. **SKU-110K `Corrupt JPEG data` warnings** — messages like
   `premature end of data segment` / `N extraneous bytes before marker 0xd9` come from
   a handful of slightly-corrupted JPEGs in the dataset. They are **warnings, not
   errors**: the decoder recovers the readable part and training/validation continues
   normally. Safe to ignore.

3. **Missing `seaborn`** — the results-plotting code imports `seaborn`; install it
   (included in the Setup step above) or plotting will fail at the end of training.
