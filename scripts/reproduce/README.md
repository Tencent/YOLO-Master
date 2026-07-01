# Reproduction Methodology for Training the Baseline YOLO-Master-v0.1-N & YOLO-Master-EsMoE-N on VisDrone & SKU-110K
 

Reproducible training strategy for the two YOLO-Master nano variants on two dense-scene vertical scenes, with per-epoch logging of the required metrics (mAP50, mAP50-95, box_loss, cls_loss, moe_loss)

| Model | Config | # Params | MoE characteristics |
| --- | --- | --- | --- |
| `YOLO-Master-v0.1-N` | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` | 7.55 M | `ModularRouterExpertMoE` |
| `YOLO-Master-EsMoE-N` | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` | 2.69 M | `ES_MOE` |

Weights:

| Dataset | Model | mAP50 | mAP50-95 | Weights |
| --- | --- | --- | --- | --- |
| VisDrone | `YOLO-Master-v0.1-N` | 0.3443  | 0.2009 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-visdrone.pt) |
| VisDrone | `YOLO-Master-EsMoE-N` | 0.3499 | 0.2029 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-visdrone.pt) |
| SKU-110K | `YOLO-Master-v0.1-N` | 0.9059 | 0.5821 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-sku110k.pt) |
| SKU-110K | `YOLO-Master-EsMoE-N` | 0.9041 | 0.5829  | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-sku110k.pt) |

Below is a comprehensive guide on how to reproduce the full training pipeline

## 1. Setup

Below is the physical server setup I used to train the two models:

| Category | My setup | Recommended |
| --- | --- | --- |
| OS | Ubuntu Server 22.04 LTS | Linux |
| CPU | Intel Xeon 8568Y 96C192T | 8C16T |
| Memory | 2,048GB | >32GB |
| GPU | Nvidia H200 SXM (only enabled 1) | Nvidia Ampere architecture (`sm_80`) or newer |
| GPU VRAM | 144GB / GPU | ≥16GB |
| Driver | `570.211.01` |  |
| Python | `3.14.6` | `3.13` or newer |
| CUDA | `12.8` | `12.8` |
| PyTorch | `2.11.0+cu128` | `2.11.0+cu128` |

Follow the official setup guides for environmental setup. Or directly install the exact conda environment I used: [download here](https://drive.google.com/file/d/1gskbzdVQ56pZBgungk9HcKtb2ft5WaVf/view?usp=share_link)

Download it (don't extract yet!), then run:

```bash
# 1) download the pack from Google Drive, then extract into your conda's envs dir
pip install gdown
gdown 1gskbzdVQ56pZBgungk9HcKtb2ft5WaVf -O yolo_master.tar.gz
ENV_DIR="$(conda info --base)/envs/yolo_master"
mkdir -p "$ENV_DIR"
tar -xzf yolo_master.tar.gz -C "$ENV_DIR"

# 2) activate, then rewrite the packed paths for THIS machine (conda-unpack ships inside the pack; run once)
conda activate yolo_master
conda-unpack

# 3) install this repo's ultralytics into the env (editable pkg was NOT bundled in the pack)
pip install -e .
```

## 2. Dataset Download

Datasets shall download automatically the first time training initializes. To fetch them manually, execute:

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```

The dataset will be stored under the default Ultralytics `datasets_dir` , usually under `../datasets` . VisDrone is approximately 2.3GB and SKU-110K is 13.6GB

## 3. Training

Recommended hyperparam settings: `--imgsz 640` ,`--epochs 300` , and adjust batch size `--batch` based on your GPU memory. 

### Full commands

```bash
# Adjust the batch size and # of epochs based on your computer's capability.

# ------ VisDrone ------
# YOLO-Master-v0.1-N
python scripts/reproduce/reproduce_visdrone.py --model v0.1-N  --epochs <epoch> --batch <batch-size> 
# YOLO-Master-EsMoE-N
python scripts/reproduce/reproduce_visdrone.py --model EsMoE-N --epochs <epoch> --batch <batch-size>  --no-sparse-eval

# ------ SKU-110K ------
# YOLO-Master-v0.1-N
python scripts/reproduce/reproduce_sku110k.py  --model v0.1-N  --epochs <epoch> --batch <batch-size> 
# YOLO-Master-EsMoE-N
python scripts/reproduce/reproduce_sku110k.py  --model EsMoE-N --epochs <epoch> --batch <batch-size>  --no-sparse-eval
```

### Key flags

| Flag | Default | Explanation |
| --- | --- | --- |
| `--model {v0.1-N,EsMoE-N,both}` | `both` | which model to train |
| `--no-sparse-eval` | **off** | **opt-in** correct evaluation for `EsMoE-N` **(see Known issue 1 below)**. Off = reproduce the model exactly as shipped. No-op for `v0.1-N`. |
| `--epochs / --imgsz / --batch` | `300 / 640 / 64` | training hyps |
| `--wandb / --no-wandb` | on | stream per-epoch metrics to Weights & Biases |
| `--wandb-entity <e>` | **default** | W&B entity/team to log under |
| `--wandb-mode {online,offline,disabled}` | `online` | W&B mode. To use `online` , you must login first. |

Tune batch size smaller if you encountered OOM (CUDA out of memory) errors.

A training of 100 epochs can already achieve a high mAP. **Only train for 300 or more epochs if you have enough GPU memory or want to challenge the SOTA.** 

### Expected results

`v0.1-N` trains and validates cleanly on both datasets. `EsMoE-N` **also trains
correctly** (its train losses track `v0.1-N`), ***but with the default sparse
evaluation its validation mAP collapses***; `--no-sparse-eval` restores it to the
`v0.1-N` level **(see Known issue 1 for the mechanism)**

| Model | Dataset | Eval Method | mAP50 | mAP50-95 | Raw Results (I'm blocked by W&B) |
| --- | --- | --- | --- | --- | --- |
| v0.1-N | VisDrone | default | 0.344 | 0.201 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-v0.1n-visdrone.zip) |
| EsMoE-N | VisDrone | default (sparse) | 0.010 | 0.003 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sparse-visdrone.zip) |
| EsMoE-N | VisDrone | `--no-sparse-eval` | 0.350 | 0.203 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-visdrone.zip) |
| v0.1-N | SKU-110K | default | 0.906 | 0.582 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-v0.1n-sku110k.zip) |
| EsMoE-N | SKU-110K | default (sparse) | 0.305 | 0.136 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sparse-sku110k.zip) |
| EsMoE-N | SKU-110K | `--no-sparse-eval` | 0.904 | 0.583 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sku110k.zip) |

### Visualization

| Model | VisDrone | SKU-110K |
| --- | --- | --- |
| **v0.1-N** | <img width="2234" height="882" alt="v0.1-visdrone" src="https://github.com/user-attachments/assets/7d076b6f-48aa-48a2-8d0c-31f55164d76b" /> | <img width="2234" height="882" alt="v0.1-sku110k" src="https://github.com/user-attachments/assets/15f98b56-0c47-4665-878f-0fc13e657381" /> |
| **EsMoE-N (sparse eval)** | <img width="2234" height="882" alt="esmoe-sparse-visdrone" src="https://github.com/user-attachments/assets/e9a0dd9d-e760-4b41-8b06-c076f9793ad9" /> | <img width="2234" height="882" alt="esmoe-sparse-sku110k" src="https://github.com/user-attachments/assets/9fbf7230-fa11-4f55-9609-644a7b973762" /> |
| **EsMoE-N (`--no-sparse-eval`)** | <img width="2234" height="882" alt="esmoe-visdrone" src="https://github.com/user-attachments/assets/1258edb0-bc03-4f50-84d3-b86507d663f6" /> | <img width="2234" height="882" alt="esmoe-sku110k" src="https://github.com/user-attachments/assets/cc50bd4c-1079-4abd-8b4d-9408d10d01a9" /> |

***Expected qualitative trend: `--no-sparse-eval` lifts `EsMoE-N` from collapsed (VisDrone) / far-below-baseline (SKU-110K) up to outperform the `v0.1-N` mAP, only with ~1/3 of its parameters.***

## 4. Known issues + solutions ‼️Very important‼️

### 1. **`EsMoE-N` validation mAP collapses (ES_MOE sparse inference)**

**Symptom.** `EsMoE-N` train losses (`box/cls/dfl`) descend normally — identical to `v0.1-N` — yet its **validation** mAP is near zero (VisDrone only ~0.01) or far below the `v0.1-N` (SKU-110K 0.31 vs 0.91). On VisDrone the mAP peaks mid-training then decays toward zero. 

**Why it happens? The machemism:**

the function `ES_MOE.forward` in `ultralytics/nn/modules/moe/modules.py`) uses two different code paths: 

- **Training** → `_dense_forward`: it computes **all** experts and sums them weighted by the softmax routing weights, which sum to 1 → output at the correct magnitude.
- **Inference** → `_sparse_forward` (taken because `use_sparse_inference=True` by default). For these configs (`top_k=None`, i.e. dense softmax over all experts), it:
    1. **Prunes** every expert whose routing weight `< dynamic_threshold` (`0.4`), keeping only the top-ranked one. With ~4 experts whose softmax weights average ~0.25, this reduces the block to ≈ top-1.
    2. **Does not renormalize**: the surviving expert's output is scaled by its raw softmax weight (~0.3) and never rescaled to sum-1.

So at inference the block emits roughly **one** expert at **~1/N** the activation magnitude that the trained `BatchNorm` (`self.norm`) was fitted to during dense training. The downstream head then sees mis-scaled, wrong-expert features → degenerate detections. It gets worse as training sharpens the router.

**Proof.** Re-validating the **same** trained checkpoint with the two paths:

- sparse (default) → mAP50 0.06
- forced dense → mAP50 0.35 (≈ the `v0.1-N` result)

The weights are fine; but the inference path is wrongly is configured.

**Solution.** `--no-sparse-eval` registers a callback that sets `ES_MOE.use_sparse_inference=False` on both the live model and its EMA at `on_pretrain_routine_end` / `on_train_start`, before any validation and before the EMA-derived checkpoints are written. Per-epoch validation, the saved `.pt`, and the final evaluation then all use the dense forward that matches training.

**Why `v0.1-N` is unaffected.** Its MoE block (`OptimizedMOEImproved`) runs the **same** top-k routing in train and eval (no dense→sparse switch), and adds an always-on shared expert plus a residual — a mode-invariant dense path that keeps the output scale stable.

**Be careful:** 

This is fixed at run time (a script flag), not in the library — `ES_MOE`'s default `use_sparse_inference=True` and `_sparse_forward` are unchanged. A plain `yolo val` or an exported `EsMoE-N` model will still exhibit the same collapse.

### 2. SKU-110K extraction error (`tar ... Operation not permitted`)

**Mechanism.** The dataset downloader extracts the SKU-110K archive with `tar xfz`, which tries to restore the files' archived ownership (`uid` / `gid`). On filesystems that disallow `chown` — for example, many networked, rootless, or container mounts — `tar` prints:

```bash
Cannot change ownership ... Operation not permitted
```

and exits non-zero, leaving the dataset unprepared.

**Solution.** Extract the archive while ignoring ownership and permissions:

```bash
tar -xzf SKU110K_fixed.tar.gz --no-same-owner --no-same-permissions -C <datasets_dir>
```

Then let Ultralytics re-run `check_det_dataset('SKU-110K.yaml')` to build the labels and the `train.txt`, `val.txt`, and `test.txt` split files.

### 3. Unable to make public W&B URL on organization accounts.

This is a W&B problem I encountered when I try to log my trainings to wandb and make it public, **not affiliated with the repo.**

**Mechanism.** W&B accounts bound to an organization — for example, through an institutional email — restrict project visibility to **Team / Restricted**. The **Public / Open** option is disabled by organization policy.

The personal-profile entity is not a loggable target in the organization model. Attempts to log there via CLI can fail with:

```bash
CommError: entity <user> not found during upsertBucket
```

As a result, a publicly viewable project cannot be created from such an account.

**Solution.** Log to W&B **privately** under the team entity:

```bash
--wandb-entity <team>
```

Alternatively, publish the run logs directly by committing each run's `results.csv` and `results.png`.

A public W&B project requires a non-organization personal account. **However, due to the bad frontend design of their website, I somehow cannot create any project without linking to a W&B team. And after trying to fix the issue, I'm accidentally blocked by W&B for no reason. 
Screenshot:

<img width="512" alt="Screenshot 2026-07-01 at 12 56 46 PM" src="https://github.com/user-attachments/assets/a9b46591-d203-4147-870b-61469d1f6d05" />

I've already contacted their support and will update the W&B public URL as soon as they unblock me.**

### 4. `model.val()` hangs/crashes with dataloader workers on Python 3.14 (minor)

**Mechanism.** A standalone `model.val()` call with `workers >0` can hit a multiprocessing forkserver `ConnectionResetError` on Python 3.14.

Full training is unaffected because it validates each epoch through the training path.

**Solution.** Pass `workers=0` for standalone validation invocations:

```python
model.val(workers=0)
```

## 5. Directory for Run logs

Per run, Ultralytics writes to:

```
runs/reproduce/<dataset>/<Dataset>_<model>/
```

Each run directory contains:

- `results.csv` — per-epoch `mAP50`, `mAP50-95`, and `box/cls/dfl/moe_loss` metrics for both training and validation.
- `results.png` — the corresponding metric curves.
- `weights/best.pt` and `weights/last.pt` — the best and latest model checkpoints.
- `args.yaml` — the exact resolved training configuration.

The dataset-level summary file:

```
runs/reproduce/<dataset>/summary.csv
```

aggregates the final metrics for both models, including a `dense_eval` column that records whether `--no-sparse-eval` was applied.

**Should you have any questions or doubts, feel free to make a comment or contact: rlici@connect.ust.hk**
