# Reproduction Methodology for Training the Baseline YOLO-Master-v0.1-N & YOLO-Master-EsMoE-N on VisDrone, SKU-110K, AI-TOD-v2, construction-ppe, and brain-tumor
 

Reproducible training strategy for the two YOLO-Master nano variants on three vertical scenes, with per-epoch logging of the required metrics (mAP50, mAP50-95, box_loss, cls_loss, moe_loss)

📊 **Live training curves for all eight runs (Weights & Biases):** https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce

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
| AI-TOD-v2 | `YOLO-Master-v0.1-N` | 0.2822 | 0.1204 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-aitodv2.pt) |
| AI-TOD-v2 | `YOLO-Master-EsMoE-N` | ≈0 (collapsed) | ≈0 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-esmoe-n-aitodv2.pt) |

Below is a comprehensive guide on how to reproduce the full training pipeline

---

### 🚀 Updates (Latest First)
- **🦺 2026-07-24: construction-ppe & brain-tumor Datasets:** Added two new vertical datasets — **construction-ppe** (industrial safety PPE detection, 11 classes, 1,132 images) and **brain-tumor** (medical imaging, 2 classes, 893 images) — with reproduce scripts and baseline results. See [this section](#3-training).
- **⭐️ 2026-07-19: Optimized P2 and UoMoE Models for Tiny Object Detection:** Four new nano variants (`v0.1-P2`, `EsMoE-P2`, `UoMoE`, `UoMoE-P2`) that attack the sub-8px floor from two orthogonal angles — a stride-4 **P2 head** (spatial resolution) and **UoMoE** routing (feature/compute allocation). `UoMoE-N`, `v0.1-P2-N` and `UoMoE-P2-N` are the only real-time N-tier detectors to break **≥20% AP on AI-TOD-v2**, outperforming YOLOv12-X by ~5 AP at ~10% of its FLOPs. See [this section](#6-p2--uomoe-variants-for-tiny-object-detection).
- **⚡️ 2026-07-18: Multi-GPU DDP Training:** All baseline models and datasets now support DDP training with up to **8 GPUs**. See [this section](#new-ddp-training).
- **🔎 2026-07-14: AI-TOD-v2 Dataset:** [AI-TOD-v2](https://github.com/Chasel-Tsui/AI-TOD-v2) is a much harder aerial **tiny-object** benchmark: 8 classes, ~800px crops, **mean object size ≈ 12px**, and a heavy class imbalance (one class dominates ~88% of the boxes). It pushes the two nano variants well past VisDrone/SKU-110K, and cleanly exposes a difference between their two MoE designs. See [this section](#new-ddp-training).

## Table of Contents
- [1. Setup](#1-setup)
- [2. Dataset Download](#2-dataset-download)
- [3. Training](#3-training)
  - [⚡️ (NEW!) DDP Training](#new-ddp-training)
- [4. Known issues + solutions/takeaways](#4-known-issues--solutionstakeaways)
- [5. Directory for Run logs](#5-directory-for-run-logs)
- [6. P2 & UoMoE Variants for Tiny Object Detection](#p2--uomoe-variants-for-tiny-object-detection)

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

### VisDrone & SKU-110k:

Datasets shall download automatically the first time training initializes. To fetch them manually, execute:

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```
### AI-TOD-v2:

Since this dataset is constructed upon the [xVIEW](https://xviewdataset.org) dataset, please refer to the offical [AI-TOD repo](https://github.com/jwwangchn/AI-TOD) for raw image downloads and generation scripts. 

The dataset should be stored under the default Ultralytics `datasets_dir` , usually under `../datasets`. VisDrone is ~2.3GB, SKU-110K ~13.6GB, AI-TOD-v2 is ~27GB, construction-ppe is ~170MB, and brain-tumor is ~6MB.

### construction-ppe & brain-tumor:

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('construction-ppe.yaml', autodownload=True)"
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('brain-tumor.yaml', autodownload=True)"
```

## 3. Training

Recommended hyperparam settings: `--imgsz 640` (`--imgsz 800` is the native resolution of AI-TOD-v2) ,`--epochs 300` , and adjust batch size `--batch` based on your GPU memory. 

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

# ------ AI-TOD-v2 ------
# YOLO-Master-v0.1-N
python scripts/reproduce/reproduce_aitodv2.py --model v0.1-N  --imgsz 800 --batch 64 --epochs 300
# YOLO-Master-EsMoE-N
python scripts/reproduce/reproduce_aitodv2.py --model EsMoE-N --imgsz 800 --batch 64 --epochs 300 --no-sparse-eval

# ------ construction-ppe ------
# YOLO-Master-v0.1-N
python scripts/reproduce/reproduce_ppe.py --model v0.1-N  --epochs <epoch> --batch <batch-size>
# YOLO-Master-EsMoE-N
python scripts/reproduce/reproduce_ppe.py --model EsMoE-N --epochs <epoch> --batch <batch-size> --no-sparse-eval

# ------ brain-tumor ------
# YOLO-Master-v0.1-N
python scripts/reproduce/reproduce_brain_tumor.py --model v0.1-N  --epochs <epoch> --batch <batch-size>
# YOLO-Master-EsMoE-N
python scripts/reproduce/reproduce_brain_tumor.py --model EsMoE-N --epochs <epoch> --batch <batch-size> --no-sparse-eval
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

---

### (NEW!) DDP Training 
The default scripts `reproduce_visdrone.py`, `reproduce_sku110k.py`, and `reproduce_aitodv2.py` only work with single GPU. Here, we also provide a script dedicated for distributed training. It's compatible with all of three models and the three datasets in the default scripts and runs smoothly on NVIDIA's mainstream datacenter GPUs (we have tested A100, H100, H200, and B200). 

> Note: it can only run **within** the node and will not work accross multiple physical servers even if they joined by an IB switch. 

Commands
```
# VisDrone — EsMoE-N on 4 GPUs
python scripts/reproduce/reproduce_ddp.py --dataset VisDrone --model EsMoE-N \
    --device 0,1,2,3 --batch 128 --epochs 300 --no-sparse-eval --workers 0

# SKU-110K — v0.1-N on 2 GPUs
python scripts/reproduce/reproduce_ddp.py --dataset SKU-110K --model v0.1-N \
    --device 0,1 --batch 64 --epochs 300 --workers 0

# AI-TOD-v2 — UoMoE-N on 8 GPUs (imgsz auto-set to 800)
python scripts/reproduce/reproduce_ddp.py --dataset AI-TOD-v2 --model UoMoE-N \
    --device 0,1,2,3,4,5,6,7 --batch 256 --epochs 300 --workers 0

# Train every model on a dataset, sequentially
python scripts/reproduce/reproduce_ddp.py --dataset VisDrone --model all --device 0,1,2,3 --batch 128 --workers 0

# Preview the plan without launching (also validates the model builds with --check-build)
python scripts/reproduce/reproduce_ddp.py --dataset VisDrone --model UoMoE-N --device 0,1 --dry-run
```

**Key flags**
  
| Flag | Meaning |
| --- | --- |
| `--dataset` | `VisDrone` / `SKU-110K` / `AI-TOD-v2` |
| `--model` | `v0.1-N`, `EsMoE-N`, `UoMoE-N`, `UoMoE-P2-N`, `EsMoE-P2-N`, `v0.1-P2-N`, or `all` |
| `--device` | comma-separated GPU ids, **≥ 2** (single GPU/CPU is refused) |
| `--batch` | **total** batch across all GPUs (keep it divisible by the GPU count) |
| `--no-sparse-eval` | corrected dense evaluation for the ES_MOE models (no-op for the rest) |
| `--lr0` / `--optimizer` | LR override for large-batch scaling (e.g. `--lr0 0.04` at large batch; forces SGD) |
 `--cache ram \| disk` | cache decoded images in RAM or on disk; omit to read on the fly |
| `--workers` | dataloader workers **per GPU** (see note below) |

> **Tip — `--workers 0`:** on the Python 3.14 stack, dataloader worker processes can stall DDP setup on some hosts. `--workers 0` loads data in the main process (a little slower per step) and is the reliable default; you can increase it once a run is confirmed stable.

**Notes**
- Runs are **resumable** — re-issuing the same command continues from `last.pt`.
- Uses the single-node convention (GPUs `0..N-1`). To pin specific GPUs, set `CUDA_VISIBLE_DEVICES=4,5` and pass `--device 0,1`.
- For a large batch, scale the learning rate (linear rule, e.g. `--batch 512 --lr0 0.04`).

---

### Expected results

`v0.1-N` trains and validates cleanly all three datasets. `EsMoE-N` **trains correctly** on VisDrone & SKU-110k (its train losses track `v0.1-N`), ***but with the default sparse evaluation its validation mAP collapses***; `--no-sparse-eval` restores it to the `v0.1-N` level **(see Known issue 1 for the mechanism)** on VisDrone & SKU-110k. Moreover, `EsMoE-N`'s routering mechanism **completely collapsed** on AI-TOD-v2 even with `--no-sparse-eval` due to the scale and high complexity of the dataset **(see Known issue 4)**. 

| Model | Dataset | Eval Method | mAP50 | mAP50-95 | W&B Run | Raw Results |
| --- | --- | --- | --- | --- | --- | --- |
| v0.1-N | VisDrone | default | 0.344 | 0.201 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/rbmyjy6b) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-v0.1n-visdrone.zip) |
| EsMoE-N | VisDrone | default (sparse) | 0.010 | 0.003 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/49bmlyp2) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sparse-visdrone.zip) |
| EsMoE-N | VisDrone | `--no-sparse-eval` | 0.350 | 0.203 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/6rsdhsn9) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-visdrone.zip) |
| v0.1-N | SKU-110K | default | 0.906 | 0.582 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/rogiamt4) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-v0.1n-sku110k.zip) |
| EsMoE-N | SKU-110K | default (sparse) | 0.305 | 0.136 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/7nofdfnb) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sparse-sku110k.zip) |
| EsMoE-N | SKU-110K | `--no-sparse-eval` | 0.904 | 0.583 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/yiz22jp3) | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/result-esmoen-sku110k.zip) |
| v0.1-N | AI-TOD-v2 | default | 0.282 | 0.120 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/x8447xku) | N/A |
| EsMoE-N | AI-TOD-v2 | `--no-sparse-eval` | ≈0 (collapsed) | ≈0 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/x8447xku) | N/A | 
| v0.1-N | construction-ppe | default | 0.527 | 0.253 | [View](https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/3z8vcfuy) | N/A |
| EsMoE-N | construction-ppe | `--no-sparse-eval` | 0.535 | 0.267 | [View](https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/cvsslgdu) | N/A |
| v0.1-N | brain-tumor | default | 0.531 | 0.364 | [View](https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/n1qozc3y) | N/A |
| EsMoE-N | brain-tumor | `--no-sparse-eval` | 0.552 | 0.381 | [View](https://wandb.ai/delei-kong-szu/yolo-master-reproduce/runs/hpi4dvhj) | N/A | 

*The raw results of the AI-TOD-v2 training runs are unavailable. Please check the W&B runs instead.

### Visualization

| Model | VisDrone | SKU-110K | AI-TOD-v2 |
| --- | --- | --- | --- |
| **v0.1-N** | <img width="2234" height="882" alt="v0.1-visdrone" src="https://github.com/user-attachments/assets/7d076b6f-48aa-48a2-8d0c-31f55164d76b" /> | <img width="2234" height="882" alt="v0.1-sku110k" src="https://github.com/user-attachments/assets/15f98b56-0c47-4665-878f-0fc13e657381" /> | <img width="2234" height="882" alt="v0 1-aitodv2" src="https://github.com/user-attachments/assets/e10de077-22ac-4744-ab45-89e56a4cddd2" /> |
| **EsMoE-N (sparse eval)** | <img width="2234" height="882" alt="esmoe-sparse-visdrone" src="https://github.com/user-attachments/assets/e9a0dd9d-e760-4b41-8b06-c076f9793ad9" /> | <img width="2234" height="882" alt="esmoe-sparse-sku110k" src="https://github.com/user-attachments/assets/9fbf7230-fa11-4f55-9609-644a7b973762" /> | N/A (experiment not conducted) |
| **EsMoE-N (`--no-sparse-eval`)** | <img width="2234" height="882" alt="esmoe-visdrone" src="https://github.com/user-attachments/assets/1258edb0-bc03-4f50-84d3-b86507d663f6" /> | <img width="2234" height="882" alt="esmoe-sku110k" src="https://github.com/user-attachments/assets/cc50bd4c-1079-4abd-8b4d-9408d10d01a9" /> | <img width="2234" height="882" alt="esmoe-aitodv2" src="https://github.com/user-attachments/assets/8b9bb8d4-f32b-46d3-9176-0a9364889d8e" /> |

***Expected qualitative trend: `--no-sparse-eval` lifts `EsMoE-N` from collapsed (VisDrone) / far-below-baseline (SKU-110K) up to outperform the `v0.1-N` mAP, only with ~1/3 of its parameters. However, it may not help on AI-TOD-v2 since the rounting mechanism itself is incapable of handling it.***

## 4. Known issues + solutions/takeaways

### 1. **`EsMoE-N` validation mAP collapses on VisDrone & SKU-110k (ES_MOE sparse inference)**

**Symptom.** `EsMoE-N` train losses (`box/cls/dfl`) descend normally — identical to `v0.1-N` — yet its **validation** mAP is near zero (VisDrone only ~0.01) or far below the `v0.1-N` (SKU-110K 0.31 vs 0.91). On VisDrone the mAP peaks mid-training then decays toward zero. 

**Why it happens? The machanism:**

the function `ES_MOE.forward` in `ultralytics/nn/modules/moe/modules.py` uses two different code paths: 

- **Training** → `_dense_forward`: it computes **all** experts and sums them weighted by the softmax routing weights, which sum to 1 → output at the correct magnitude.
- **Inference** → `_sparse_forward` (taken because `use_sparse_inference=True` by default). For these configs (`top_k=None`, i.e. dense softmax over all experts), it:
    1. **Prunes** every expert whose routing weight `< dynamic_threshold` (`0.4`), keeping only the top-ranked one. With ~4 experts whose softmax weights average ~0.25, this reduces the block to ≈ top-1.
    2. **Does not renormalize**: the surviving expert's output is scaled by its raw softmax weight (~0.3) and never rescaled to sum-1.

So at inference the block emits roughly **one** expert at **~1/N** the activation magnitude that the trained `BatchNorm` (`self.norm`) was fitted to during dense training. The downstream head then sees mis-scaled, wrong-expert features → degenerate detections. It gets worse as training sharpens the router.

**Proof.** Re-validating the **same** trained checkpoint with the two paths:

- sparse (default) → mAP50 0.06
- forced dense → mAP50 0.35 (≈ the `v0.1-N` result)

The weights are fine; but the inference path is wrongly configured.

**Solution.** `--no-sparse-eval` registers a callback that sets `ES_MOE.use_sparse_inference=False` on both the live model and its EMA at `on_pretrain_routine_end` / `on_train_start`, before any validation and before the EMA-derived checkpoints are written. Per-epoch validation, the saved `.pt`, and the final evaluation then all use the dense forward that matches training.

**Why `v0.1-N` is unaffected.** Its MoE block (`OptimizedMOEImproved`) runs the **same** top-k routing in train and eval (no dense→sparse switch), and adds an always-on shared expert plus a residual — a mode-invariant dense path that keeps the output scale stable.

**Be careful:** 

This is fixed at run time (a script flag), not in the library — `ES_MOE`'s default `use_sparse_inference=True` and `_sparse_forward` are unchanged. A plain `yolo val` or an exported `EsMoE-N` model will still exhibit the same collapse.

### 2. SKU-110K extraction error (`tar ... Operation not permitted`)

**The mechanism.** The dataset downloader extracts the SKU-110K archive with `tar xfz`, which tries to restore the files' archived ownership (`uid` / `gid`). On filesystems that disallow `chown` — for example, many networked, rootless, or container mounts — `tar` prints:

```bash
Cannot change ownership ... Operation not permitted
```

and exits non-zero, leaving the dataset unprepared.

**Solution.** Extract the archive while ignoring ownership and permissions:

```bash
tar -xzf SKU110K_fixed.tar.gz --no-same-owner --no-same-permissions -C <datasets_dir>
```

Then let Ultralytics re-run `check_det_dataset('SKU-110K.yaml')` to build the labels and the `train.txt`, `val.txt`, and `test.txt` split files.

### 3. `model.val()` hangs/crashes with dataloader workers on Python 3.14 (minor)

**Mechanism.** A standalone `model.val()` call with `workers >0` can hit a multiprocessing forkserver `ConnectionResetError` on Python 3.14.

Full training is unaffected because it validates each epoch through the training path.

**Solution.** Pass `workers=0` for standalone validation invocations:

```python
model.val(workers=0)
```
### 4. `ES_MOE` routing collapses on AI-TOD-v2

**Mechanism.** On AI-TOD-v2's homogeneous tiny objects, `EsMoE-N`'s pure top-k router collapses onto a single expert — one early MoE layer reaches `>0.8` max-usage — and validation mAP freezes at the noise floor (≈1e-5), even with the built-in routing-collapse recovery. `v0.1-N`'s `ModularRouterExpertMoE` keeps an always-on **shared expert** (a guaranteed signal path independent of the router), so on the *identical* data it trains cleanly to 0.28 mAP50.

**Takeaway.** On tiny-object / low-diversity data, prefer the shared-expert `v0.1-N`; `ES_MOE` needs a larger, more diverse distribution to keep its experts balanced. The AI-TOD-v2 `EsMoE-N` weights above are included for completeness — they are a collapsed model. 

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

## 6. P2 & UoMoE Variants for Tiny Object Detection

This formally integrates the improved models in issues [#98](https://github.com/Tencent/YOLO-Master/issues/98) & [#126](https://github.com/Tencent/YOLO-Master/issues/126). 

Four additional nano models, `v0.1-P2`, `EsMoE-P2`, `UoMoE` and `UoMoE-P2` derived from the two baselines, now available on every dataset and every reproduce script. Same recipe as the baselines (`optimizer=auto` → SGD@0.01, `lora_r=0`, `deterministic`, `seed 42`, `patience 0`); just pass `--model <name>`.

<img width="1821" height="1090" alt="model_evolution" src="https://github.com/user-attachments/assets/648ea44c-460a-4190-870b-d79c1ad47b28" />


### How and Why they works?

**Two independent levers on the tiny-object floor.** Both families attack the same failure mode — sub-8 px targets that a standard detector cannot recover — but from different angles. The **P2/stride-4 head** is a *spatial-resolution* fix (#98), and **UoMoE** (`UltraOptimizedMoE`) is a *feature/compute-allocation* fix (#126).

#### Optimization 1: P2 — spatial resolution ([Issue #98](https://github.com/Tencent/YOLO-Master/issues/98))

The stock head predicts on P3/P4/P5 (stride 8/16/32). Detection quality scales with an object's grid footprint, `footprint = object_px / stride`, so a 7 px object at 640 lands on ≈0.9 of a P3 cell and stays sub-grid — unlearnable no matter how good the features. Adding a stride-4 head halves the finest stride and doubles that footprint, moving tiny targets into the learnable regime. The P2 head is a pure neck/head extension (backbone untouched): the FPN top-down path is carried one level further to stride-4, the PAN bottom-up path is re-rooted at P2, and `Detect` becomes 4-level `(P2, P3, P4, P5)`.

VisDrone, imgsz 640, 300 epochs, dense eval (`--no-sparse-eval`) for EsMoE models:

| Model | Params | GFLOPs@640 | mAP50 | mAP50-95 | W&B Runs |
|---|---|---|---|---|---|
| EsMoE-N (baseline) | 2.69 M | 8.8 | 0.350 | 0.203 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/6rsdhsn9) |
| **EsMoE-N-P2** | **2.81 M** (+4.4%) | **12.2** | **0.381 (+0.031)** | **0.225 (+0.022)** | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/s44aqxgp) |
| v0.1-N (baseline) | 7.52 M | 9.9 | 0.344 | 0.201 | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/rbmyjy6b) | 
| v0.1-N-P2 | 7.67 M | 14.7 | 0.369 (+0.025) | 0.218 (+0.017) | [View](https://wandb.ai/yolo-master-reproduce/yolo-master-reproduce/runs/qkf5d0je) |

The per-class Δ is the textbook finer-head fingerprint — gains concentrate on the smallest classes and the largest class regresses slightly:

| Class | Size | v0.1-N | v0.1-N-P2 | Δ | EsMoE-N | EsMoE-N-P2 | Δ |
|---|---|---|---|---|---|---|---|
| people | tiny |0.106 | 0.136 | +0.030 | 0.101 | 0.143 | **+0.042** |
| pedestrian | tiny |0.160 | 0.198 | +0.038 | 0.165 | 0.203 | **+0.038** |
| motor | tiny/small | 0.164 | 0.190 | +0.026 | 0.166 | 0.200 | **+0.034** |
| car | small |0.533 | 0.566 | +0.033 | 0.538 | 0.572 | **+0.033** |
| bus | medium |0.329 | 0.346 | +0.017 | 0.351 | 0.378 | **+0.027** |
| bicycle | tiny/small |0.038 | 0.047 | +0.009 | 0.036 | 0.059 | **+0.022** |
| awning-tricycle | tiny/small | 0.091 | 0.092 | +0.001 | 0.073 | 0.093 | **+0.021** |
| van | small| 0.278 | 0.293 | +0.015 | 0.278 | 0.296 | **+0.018** |
| tricycle | tiny/small |0.117 | 0.131 | +0.014 | 0.122 | 0.136 | **+0.014** |
| truck | medium/large | 0.198 | 0.191 | −0.007 | 0.205 | 0.184 | **−0.021** |

| Comparison | Visualization |
| --- | --- |
| Es-MoE-N-P2 vs EsMoE-N | <img width="2383" height="875" alt="Image" src="https://github.com/user-attachments/assets/d264c8f4-b2c2-4cb8-9bdc-c7881e77c940" /> |
| v0.1-N-P2 vs v0.1-N | <img width="2383" height="875" alt="Image" src="https://github.com/user-attachments/assets/ce3aaf2c-a496-4296-9b33-3aba7beba27a" /> |

Because imgsz and stride are *substitutes* (raising input resolution raises `object_px` proportionally), P2 is a low-resolution optimization: at 640 it clearly wins, but at 1280 the pixels already supply the footprint and the gain collapses to ~+0.8 mAP50. Evaluate and deploy at 640 (the real-time regime). Adding P2 also keeps the model firmly sub-S-tier — N→P2 costs +4.8 GFLOPs whereas N→S costs +24.8 GFLOPs (~1/5 of a scale-up), and it targets the tiny-object bottleneck that scaling up does not:

| Model | Params | GFLOPs@640 | Stride-4 head? |
|---|---|---|---|
| **EsMoE-N-P2** | 2.8 M | **12.2** | ✅ |
| v0.1-N-P2 | 7.7 M | 14.7 | ✅ |
| YOLO11-S | 9.5 M | 21.7 | ✗ |
| YOLO12-S | 9.3 M | 21.7 | ✗ |
| YOLOv10-S | 8.1 M | 25.1 | ✗ |
| YOLOv8-S | 11.2 M | 28.8 | ✗ |
| YOLO-Master-S | 29.2 M | 34.7 | ✗ |

#### Optimization2: UoMoE — feature/compute allocation ([Issue #126](https://github.com/Tencent/YOLO-Master/issues/126))

Replacing v0.1-N's three `ModularRouterExpertMoE` blocks with `UltraOptimizedMoE` (backbone/head unchanged, identical `[out, num_experts, top_k]`) is a pure architecture win — neither more resolution nor more compute, just better routing of expert compute to densely-packed tiny-object regions, i.e. higher feature quality read out on the *same* grid.

AI-TOD-v2 test split (14,018 imgs / 376,121 anns, 8 classes, mean object ~12 px; imgsz 800, 300 epochs):

| model | params (M) | GFLOPs@800 | AP | AP50 | AP75 | AP_vt | AP_t | AP_s | AP_m | Weights | 
|---|---|---|---|---|---|---|---|---|---|---|
| v0.1-N | 7.52 | 14.65 | 11.0 | 27.0 | 6.7 | 3.6 | 10.9 | 14.6 | 19.1 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.0/yolo-master-v01-n-aitodv2.pt) |
| **UoMoE-N** | 7.42 | 14.61 | **20.6** | **47.0** | **14.5** | 5.9 | **18.9** | **28.0** | **35.1** | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.1/yolo-master-uomoe-n-aitodv2.pt) |
| v0.1-P2-N | 7.60 | 20.95 | **21.4** | **48.6** | **15.0** | 7.8 | **21.1** | 27.2 | 32.7 | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.1/yolo-master-v01-p2-n-aitodv2.pt) |
| UoMoE-P2-N | 7.50 | 20.91 | 21.1 | 47.8 | 14.7 | **8.6** | 20.4 | **27.6** | **33.8** | [Download](https://github.com/skywalker-lt/YOLO-Master/releases/download/v0.1.1/yolo-master-uomoe-p2-n-aitodv2.pt) |
| YOLOv12-X | 59.3 | 185.4 | 16.1 | 33.5 | 13.4 | 5.4 | 17.0 | 21.4 | 28.5 | N/A |

`UoMoE-N` jumps **+9.6 AP / +20.0 AP50** over `v0.1-N` at *fewer* params (7.42 vs 7.52 M) and iso-FLOPs (14.61 vs 14.65 GF).

<img width="1300" height="845" alt="Image" src="https://github.com/user-attachments/assets/52991fcb-b5ae-4f9a-8560-23a556de4163" />

**Based on the experiments above, v0.1-P2-N, UoMoE-N and UoMoE-P2-N are by far the only three real-time N-tier detectors which achieved ≥20% AP on the AI-TOD-v2 benchmark, outperforming YOLOv12-X by ~5 points AP with approx. 8~11% of its FLOPs.**

#### How they interact - orthogonal on a weak base, substitutive on a strong one

The two fixes target the *same* tiny-object bottleneck through different terms (P2 = sampling density, MoE = feature quality), so whether they compound depends on how much of that bottleneck the MoE has already closed:

- **Weak base → they stack.** With ES-MoE (#98, VisDrone) P2 and MoE fix orthogonal gaps and add up: `EsMoE-N-P2` beats both the MoE-only and the P2-on-dense-backbone models (0.381 mAP50 at 2.8 M params / 12.2 GFLOPs).
- **Strong base → P2 is largely redundant.** On AI-TOD-v2 (#126) P2 lifts the weak MoE (`v0.1-N`) by **+21.6 AP50** (27.0→48.6) but the strong `UoMoE-N` by only **+0.8** (47.0→47.8) — the stronger router has already resolved most of what the finer grid would have recovered. P2's residual value is confined entirely to the very-tiny (<8 px) bin (AP_vt 5.9→8.6, +2.7), making `UoMoE-P2-N` the best AP_vt of all four while flat elsewhere.

**Practical takeaway.** `UoMoE-N` (no P2, 14.6 GF) lands within 1.6 AP50 of `v0.1-P2-N` (21.0 GF) — P2-variant-level accuracy at ~30% fewer FLOPs and no extra head. Pick `UoMoE-N` for the best accuracy/FLOPs trade-off; add P2 only when the <8 px bin is the priority.

> **Reproducibility note:** `ultralytics.utils.torch_utils.get_flops` profiles an uninitialised `torch.empty` 32×32 input scaled ×625, which for input-dependent MoE triggers a different expert mix each call and makes identical runs report different GFLOPs (one UoMoE eval swung from 10.7 to 14.7). All FLOPs above are forced to a deterministic full-res measure.

### Model Specs

| Name | Derived from | Params | Notes |
|---|---|---|---|
| `v0.1-P2-N` | `v0.1-N` + P2/4 head | 7.60M | tiny-object variant |
| `EsMoE-P2-N` | `EsMoE-N` + P2/4 head | 2.81M | tiny-object; **needs `--no-sparse-eval`** |
| `UoMoE-N` | `v0.1-N`, MoE blocks → UltraOptimizedMoE | 7.42M | ~20–30% fewer GFLOPs at equal params |
| `UoMoE-P2-N` | `UoMoE-N` + P2/4 head | 7.50M | UoMoE + tiny-object head |

> **`--no-sparse-eval`** matters only for `EsMoE-P2-N` (an ES_MOE model): its default sparse eval
> collapses mAP, so pass the flag for correct dense evaluation. It is a no-op for the v0.1/UoMoE variants.

Sanity-check (instant, no training):

```bash
python scripts/reproduce/reproduce_visdrone.py --check-build --model UoMoE-P2-N
```

### Training Commands


#### Single-GPU / per-dataset scripts

```bash
# P2 variants
python scripts/reproduce/reproduce_aitodv2.py  --model v0.1-P2-N  --epochs 300 --batch 64
python scripts/reproduce/reproduce_aitodv2.py  --model EsMoE-P2-N --epochs 300 --batch 64 --no-sparse-eval

# UoMoE variants
python scripts/reproduce/reproduce_visdrone.py --model UoMoE-N    --epochs 300 --batch 64
python scripts/reproduce/reproduce_visdrone.py --model UoMoE-P2-N --epochs 300 --batch 64
```

#### Multi-GPU DDP (`reproduce_ddp.py`)

`--device` must list ≥2 GPUs; `--batch` is the TOTAL across GPUs. Use `--workers 0` on the py-3.14 stack.

```bash
python scripts/reproduce/reproduce_ddp.py --dataset AI-TOD-v2 --model v0.1-P2-N  --device 0,1,2,3 --batch 128 --workers 0
python scripts/reproduce/reproduce_ddp.py --dataset AI-TOD-v2 --model EsMoE-P2-N --device 0,1,2,3 --batch 128 --no-sparse-eval --workers 0
python scripts/reproduce/reproduce_ddp.py --dataset VisDrone  --model UoMoE-N    --device 0,1,2,3 --batch 128 --workers 0
python scripts/reproduce/reproduce_ddp.py --dataset VisDrone  --model UoMoE-P2-N --device 0,1     --batch 512 --lr0 0.04 --workers 0
```

> SKU-110K is *large-object-dense* dataset — the P2/UoMoE variants are provided for completeness there but are not tuned/tested on that dataset.

---

**Should you have any questions or doubts, feel free to make a comment or contact: rlici@connect.ust.hk**
