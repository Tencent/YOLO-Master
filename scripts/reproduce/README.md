# Reproduction Scripts — VisDrone & SKU-110K

This directory contains reproducible training scripts for **YOLO-Master** on two
dense-object detection benchmarks:

| Dataset | Domain | Classes | Train / Val | Script |
|---------|--------|---------|-------------|--------|
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | Aerial small-object detection | 10 | 6 471 / 548 | [`reproduce_visdrone.py`](reproduce_visdrone.py) |
| [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) | Retail shelf product detection | 1 | 8 219 / 588 | [`reproduce_sku110k.py`](reproduce_sku110k.py) |

Each script trains **two model variants** for comparison:

| Variant | Config file | MoE module | Params | GFLOPs |
|---------|-------------|------------|--------|--------|
| **YOLO-Master-EsMoE-N** (v0) | `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml` | `ES_MOE` | 2.66M | 8.7 |
| **YOLO-Master-v0.1-N** | `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml` | `ModularRouterExpertMoE` (= `OptimizedMOEImproved`) | 7.50M | 9.9 |

---

## Results

Training was conducted with **imgsz=640, epochs=300, SGD(lr=0.01, momentum=0.9)**,
following the paper's recommended hyper-parameters.
Hardware: 4× or 8× NVIDIA H100 80 GB HBM3.

### VisDrone2019-DET

| Model | mAP50 | mAP50-95 | Best Epoch | Precision | Recall | moe_loss (final) | Wandb |
|-------|-------|----------|------------|-----------|--------|-------------------|-------|
| YOLO-Master-EsMoE-N | **0.3440** | **0.1998** | 236 | 0.4398 | 0.3399 | 9.0e-05 | [link](https://wandb.ai/2267450832-/reproduce-visdrone/runs/cfae6k0k) |
| YOLO-Master-v0.1-N | 0.3362 | 0.1951 | 229 | 0.4478 | 0.3343 | 8.8e-03 | [link](https://wandb.ai/2267450832-/reproduce-visdrone/runs/18slp9wp) |

### SKU-110K

| Model | mAP50 | mAP50-95 | Best Epoch | Precision | Recall | moe_loss (final) | Wandb |
|-------|-------|----------|------------|-----------|--------|-------------------|-------|
| YOLO-Master-EsMoE-N | **0.9049** | **0.5821** | 211 | 0.9062 | 0.8423 | 1.0e-05 | [link](https://wandb.ai/2267450832-/reproduce-sku110k/runs/upwvudkh) |
| YOLO-Master-v0.1-N | 0.9025 | 0.5819 | 256 | 0.9058 | 0.8416 | 5.6e-03 | [link](https://wandb.ai/2267450832-/reproduce-sku110k-4gpu/runs/xoc3uc23) |

### Key Findings

1. **EsMoE-N (v0) is the overall winner** — with only 2.66M params, it slightly
   outperforms the 7.5M-param v0.1-N on both datasets. The margin is larger on
   VisDrone (+0.47 mAP50-95) where the multi-class dense small-object scenario
   benefits more from `ES_MOE`'s simpler, lighter routing.

2. **SKU-110K: both variants converge to similar levels** (~0.90 mAP50,
   ~0.58 mAP50-95). The single-class dense retail scenario is less sensitive to
   MoE architecture choice.

3. **`ES_MOE` load-balancing is 100× lighter** — moe_loss ~1e-5 to ~9e-5 vs
   v0.1's ~5e-3 to ~9e-3, with no accuracy penalty. The `ES_MOE` module uses
   a buffer-based load-balancing loss rather than the full `MoELoss` module.

4. **Slight overfitting on VisDrone** — best epoch ~230 for both models, with
   final epoch performance slightly lower. SKU-110K EsMoE-N peaked at epoch 211
   while v0.1-N continued improving until ~256.

---

## Quick Start

### 1. Environment setup

```bash
# Create & activate conda environment (requires Python ≥ 3.8, PyTorch ≥ 1.8)
conda create -n yolo_master python=3.10 -y
conda activate yolo_master

# Install PyTorch (CUDA 12.4 example; adjust to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install project dependencies
cd /path/to/YOLO-Master
pip install -e .

# Install W&B for logging (optional but recommended)
pip install wandb
wandb login
```

### 2. Dataset downloads

Both datasets are **automatically downloaded** on the first training run — no
manual steps required.

<details>
<summary>Manual download (alternative)</summary>

**VisDrone** (~2.3 GB compressed, ~25 GB extracted):
```bash
# The built-in download script fetches from Ultralytics assets
python -c "
from ultralytics.utils.downloads import download
from pathlib import Path
from ultralytics.utils import ASSETS_URL
urls = [f'{ASSETS_URL}/VisDrone2019-DET-{s}.zip' for s in ('train','val','test-dev')]
download(urls, dir=Path('VisDrone'))
"
```

**SKU-110K** (~13.6 GB):
```bash
wget http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
tar -xzf SKU110K_fixed.tar.gz
mv SKU110K_fixed SKU-110K
# Then run the conversion script from the dataset YAML or:
python -c "
import numpy as np, polars as pl
from pathlib import Path
from ultralytics.utils import TQDM
from ultralytics.utils.ops import xyxy2xywh

dir = Path('SKU-110K')
names = 'image', 'x1', 'y1', 'x2', 'y2', 'class', 'image_width', 'image_height'
dir.joinpath('labels').mkdir(exist_ok=True)
for d in 'annotations_train.csv', 'annotations_val.csv', 'annotations_test.csv':
    x = pl.read_csv(dir/'annotations'/d, has_header=False, new_columns=names, infer_schema_length=None).to_numpy()
    images, unique_images = x[:, 0], np.unique(x[:, 0])
    txt_name = d.replace('annotations_', '')
    with open(dir/txt_name, 'w') as f:
        f.writelines(f'./images/{s}\n' for s in unique_images)
    for im in TQDM(unique_images, desc=f'Converting {d}'):
        with open((dir/'labels'/im).with_suffix('.txt'), 'a') as f:
            for r in x[images == im]:
                w, h = r[6], r[7]
                xywh = xyxy2xywh(np.array([[r[1]/w, r[2]/h, r[3]/w, r[4]/h]]))[0]
                f.write(f'0 {xywh[0]:.5f} {xywh[1]:.5f} {xywh[2]:.5f} {xywh[3]:.5f}\n')
"
```
</details>

### 3. Training commands

**Single GPU, default settings:**
```bash
# VisDrone (300 epochs each variant)
python scripts/reproduce/reproduce_visdrone.py --device 0

# SKU-110K (300 epochs each variant)
python scripts/reproduce/reproduce_sku110k.py --device 0
```

**Multi-GPU (DDP, 4 cards):**
```bash
# VisDrone on 4 GPUs, batch 128 per GPU
python scripts/reproduce/reproduce_visdrone.py --device 0,1,2,3 --batch 128 --workers 16

# SKU-110K on 4 GPUs, batch 64 per GPU (dense instances → lower batch to avoid OOM)
python scripts/reproduce/reproduce_sku110k.py --device 0,1,2,3 --batch 64 --workers 16
```

**Quick test run (100 epochs, single variant):**
```bash
python scripts/reproduce/reproduce_visdrone.py --epochs 100 --variants esmoe-n
```

**Resume interrupted training:**
```bash
python scripts/reproduce/reproduce_visdrone.py --resume
```

**Disable W&B logging:**
```bash
python scripts/reproduce/reproduce_sku110k.py --no-wandb
```

### 4. Monitoring

Training metrics are logged to:

- **Console:** per-epoch `box_loss`, `cls_loss`, `dfl_loss`, `moe_loss`, `mAP50`, `mAP50-95`
- **Weights & Biases:** [wandb.ai](https://wandb.ai) (when enabled)
- **Local CSV:** `reproduce/<dataset>/<variant>/results.csv`

Key metrics tracked:
```
train/box_loss    train/cls_loss    train/dfl_loss    train/moe_loss
val/box_loss      val/cls_loss      val/dfl_loss      val/moe_loss
metrics/mAP50(B)  metrics/mAP50-95(B)
```

---

## Known Issues & Solutions

### 1. DDP `dist.gather_object()` hangs during validation (SKU-110K only)

**Problem:** SKU-110K has ~90k instances in the validation set.
`dist.gather_object()` in `val.py:243` can deadlock when the stats tensor
exceeds NCCL's default timeout, especially with 8+ GPUs.

**Solution:** Reduce GPU count to 4, and set increased NCCL timeout:
```bash
NCCL_BLOCKING_WAIT=0 NCCL_TIMEOUT=1800 \
python scripts/reproduce/reproduce_sku110k.py --device 0,1,2,3 --batch 64
```
Or run single-GPU to completely avoid the issue.

### 2. CUDA OOM in `TaskAlignedAssigner` on SKU-110K

**Problem:** SKU-110K images contain 200–500 annotations each. With large batch
sizes on a single GPU, the assignment matrix can exceed VRAM.

**Solution:** Reduce `--batch` to 16–32 for single-GPU training, or use
multi-GPU DDP which distributes the assignment across devices.

### 3. Model scale auto-detection limitation

**Problem:** `guess_model_scale()` in `tasks.py` uses the regex
`r"yolo(e-)?[v]?\d+([nslmx])"` which does **not** match `yolo-master-n.yaml`
(because `master` is not a digit). The scale falls back to the **first** key
in the `scales:` dict, which is always `n`.

**Impact:** Only affects non-Nano models. For `yolo-master-s.yaml`, the
scale defaults to `n` instead of `s`, producing incorrect channel dimensions.

**Workaround:** Pass `scale="s"` explicitly when training S/M/L/X variants:
```python
model = YOLO("ultralytics/cfg/models/master/v0_1/det/yolo-master-s.yaml")
model.model.yaml["scale"] = "s"  # force S scale
model.train(data="VisDrone.yaml", ...)
```

### 4. Corrupt JPEG warnings on SKU-110K

SKU-110K contains dozens of images with minor JPEG header corruption. OpenCV/PIL
auto-restore these on first load. The warnings (`Corrupt JPEG data: ...`) are
benign and do not affect training. A few severely corrupted images (e.g.
`train_4846.jpg`) are skipped automatically.

### 5. W&B `moe_loss` logging

The `moe_loss` is the **fourth** component in the detection loss vector
(`box_loss, cls_loss, dfl_loss, moe_loss`). It is computed by aggregating
`aux_loss` from each MoE module in the model. The W&B callback
(`ultralytics/utils/callbacks/wb.py`) logs all four components automatically.

If `moe_loss` stays at `0.0` throughout training:
- Check that the model config **actually instantiates** MoE modules.
- Verify `self.hyp.moe` is non-zero (default: `0.15` in `ultralytics/cfg/default.yaml`).
- For `ES_MOE`, the load-balancing loss uses a **registered buffer** — ensure
  the forward pass is in training mode.

### 6. GPU memory requirements

| Batch size (per GPU) | Approximate VRAM (Nano) |
|----------------------|------------------------|
| 8 | ~3.5 GB |
| 16 | ~6 GB |
| 64 | ~11 GB (dense datasets: ~18 GB) |
| 128 | ~15 GB (dense datasets: ~25 GB) |

---

## File Structure

```
scripts/reproduce/
├── README.md                  # This file
├── reproduce_visdrone.py      # VisDrone training script
└── reproduce_sku110k.py       # SKU-110K training script
```

## References

- YOLO-Master paper: https://arxiv.org/abs/2512.23273
- YOLO-Master GitHub: https://github.com/Tencent/YOLO-Master
- VisDrone dataset: https://github.com/VisDrone/VisDrone-Dataset
- SKU-110K dataset: https://github.com/eg4000/SKU110K_CVPR19
