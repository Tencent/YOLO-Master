# VisDrone & SKU-110K Baseline Training

Reproducible training scripts for **YOLO-Master** on two dense-object detection
benchmarks, comparing `ES_MOE` (v0) vs `ModularRouterExpertMoE` (v0.1) MoE
architectures.

| Dataset | Domain | Classes | Train / Val | Script |
|---------|--------|---------|-------------|--------|
| [VisDrone2019-DET](https://github.com/VisDrone/VisDrone-Dataset) | Aerial small objects | 10 | 6 471 / 548 | [`reproduce_visdrone.py`](reproduce_visdrone.py) |
| [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19) | Retail shelf products | 1 | 8 219 / 588 | [`reproduce_sku110k.py`](reproduce_sku110k.py) |

## Model Variants

| Variant | Config | MoE Module | Params | GFLOPs |
|---------|--------|------------|--------|--------|
| YOLO-Master-EsMoE-N (v0) | `cfg/models/master/v0/det/yolo-master-n.yaml` | `ES_MOE` | 2.66 M | 8.7 |
| YOLO-Master-v0.1-N | `cfg/models/master/v0_1/det/yolo-master-n.yaml` | `ModularRouterExpertMoE` | 7.50 M | 9.9 |

## MoE Design Notes

The two variants differ fundamentally in expert routing and load-balancing:

- **ES_MOE (v0)** uses `DynamicRoutingLayer` with a per-expert kernel-size schedule
  (3/5/7), a shared-expert bypass, and a lightweight buffer-based load-balancing
  loss. Expert counts are fixed at 3.
- **v0.1 (OptimizedMOEImproved)** uses `EfficientSpatialRouter` with pluggable
  router/expert types, configurable expert count (4/8/16 across backbone stages),
  and the full `MoELoss` module (balance loss + z-loss).

The key trade-off: v0.1's richer expert capacity adds 2.8× parameters but its
heavier auxiliary loss (`moe_loss` ~5e-3 to ~9e-3 vs v0's ~1e-5 to ~9e-5) does
not translate into accuracy gains on these benchmarks.

## Results

Training config: **imgsz=640, epochs=300, SGD(lr=0.01, momentum=0.9), 4× H100 80 GB**.

### VisDrone2019-DET

| Model | mAP50 | mAP50-95 | Best Ep | Wandb |
|-------|-------|----------|---------|-------|
| YOLO-Master-EsMoE-N | **0.3440** | **0.1998** | 236 | [link](https://wandb.ai/2267450832-/reproduce-visdrone/runs/cfae6k0k) |
| YOLO-Master-v0.1-N | 0.3362 | 0.1951 | 229 | [link](https://wandb.ai/2267450832-/reproduce-visdrone/runs/18slp9wp) |

### SKU-110K

| Model | mAP50 | mAP50-95 | Best Ep | Wandb |
|-------|-------|----------|---------|-------|
| YOLO-Master-EsMoE-N | **0.9049** | **0.5821** | 211 | [link](https://wandb.ai/2267450832-/reproduce-sku110k/runs/upwvudkh) |
| YOLO-Master-v0.1-N | 0.9025 | 0.5819 | 256 | [link](https://wandb.ai/2267450832-/reproduce-sku110k-4gpu/runs/xoc3uc23) |

### Summary

- **EsMoE-N wins on both datasets** with 65% fewer params. The margin is larger
  on VisDrone (+0.47 mAP50-95), where multi-class dense small-object detection
  benefits more from ES_MOE's simpler routing.
- On SKU-110K both variants converge to ~0.90 mAP50 / ~0.58 mAP50-95 — the
  single-class scenario is less sensitive to MoE architecture.
- Both models overfit slightly on VisDrone (best ~230, final lower). SKU-110K
  EsMoE-N peaked at 211; v0.1-N continued improving until 256.

---

## Quick Start

### 1. Environment

```bash
conda create -n yolo_master python=3.10 -y && conda activate yolo_master
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
cd YOLO-Master && pip install -e .
pip install wandb && wandb login
```

### 2. Datasets

Both datasets auto-download on first run. For manual SKU-110K conversion, see
the `download:` script in `ultralytics/cfg/datasets/SKU-110K.yaml`.

Disk requirements: VisDrone ~25 GB extracted, SKU-110K ~14 GB.

### 3. Training

```bash
# VisDrone (300 epochs, 4 GPUs)
python scripts/reproduce/reproduce_visdrone.py --device 0,1,2,3 --batch 128 --workers 16

# SKU-110K (300 epochs, 4 GPUs — use smaller batch for dense instances)
python scripts/reproduce/reproduce_sku110k.py --device 0,1,2,3 --batch 64 --workers 16

# Quick test
python scripts/reproduce/reproduce_visdrone.py --epochs 10 --variants esmoe-n

# Resume
python scripts/reproduce/reproduce_visdrone.py --resume
```

### 4. Monitoring

Per-epoch metrics logged to console, local CSV (`reproduce/<dataset>/<variant>/results.csv`),
and W&B. Columns: `train/box_loss`, `train/cls_loss`, `train/dfl_loss`,
`train/moe_loss`, `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, plus val losses.

---

## Known Issues

### Model scale auto-detection

`guess_model_scale()` in `tasks.py` uses regex `r"yolo(e-)?[v]?\d+([nslmx])"`
which does not match `yolo-master-*.yaml` (because `master` is not a digit).
Scale defaults to the first key in `scales:`, which is always `n`. When
training S/M/L/X variants, force the scale explicitly:

```python
model = YOLO("cfg/models/master/v0/det/yolo-master-s.yaml")
model.model.yaml["scale"] = "s"
model.train(data="VisDrone.yaml", ...)
```

### SKU-110K: CUDA OOM in TaskAlignedAssigner

SKU-110K images contain 200–500 annotations each. With large per-GPU batch
sizes, the assignment matrix can exceed VRAM. Use `--batch 16` for single-GPU,
or multi-GPU DDP which distributes the assignment across devices.

### SKU-110K: dataset naming

The upstream tarball extracts to `SKU110K_fixed/` but `SKU-110K.yaml` expects
`SKU-110K/`. Rename after extraction: `mv SKU110K_fixed SKU-110K`.

### moe_loss stays at 0.0

- Verify the model config instantiates MoE modules (`ES_MOE` or `MoE` variants).
- Check `moe: 0.15` is set in `ultralytics/cfg/default.yaml`.
- `val/moe_loss` is expected to be 0 — auxiliary loss only applies in training mode.

---

## References

- YOLO-Master paper: https://arxiv.org/abs/2512.23273
- VisDrone dataset: https://github.com/VisDrone/VisDrone-Dataset
- SKU-110K dataset: https://github.com/eg4000/SKU110K_CVPR19
