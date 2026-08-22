# Pull Request: Reproduce YOLO-Master on VisDrone and SKU-110K

## ① Summary & Scope

This PR addresses Issue #49 by providing reproducible training scripts for YOLO-Master on vertical domain datasets (VisDrone & SKU-110K), completing baseline data alignment for both `v0.1-N` and `EsMoE-N` models. The implementation ensures:

- Full training pipeline support for VisDrone (drone-captured object detection) and SKU-110K (retail shelf object detection) datasets
- Configurable training parameters for different hardware configurations
- Comprehensive per-epoch logging of `mAP50`, `mAP50-95`, `box_loss`, `cls_loss`, and `moe_loss`
- Public Weights & Biases project for full training curve visualization

---

## ② Benchmark Results

### VisDrone Dataset (Aerial, dense small objects)

| Model | Eval Mode | mAP50 | mAP50-95 | Params | Config / Script | W&B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO-Master-v0.1-N** | default | 0.2916 | 0.1666 | 7.5M | `reproduce_visdrone.py` | [run](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/501cw1pq) |
| **YOLO-Master-EsMoE-N** | default (sparse) | 0.3008 | 0.1731 | 3.4M | `reproduce_visdrone.py` | [run](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/h9doimnp) |

> **Note**: Post-training dense eval via `eval_dense.py` on the saved checkpoint yields mAP50=0.3037 / mAP50-95=0.1758 — a marginal +0.003 gain over sparse inference, confirming VisDrone's expert routing is less susceptible to the sparse inference artifact (see Key Observations).

### SKU-110K Dataset (Retail, dense products)

| Model | Eval Mode | mAP50 | mAP50-95 | Δ | Params | Config / Script | W&B |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLO-Master-v0.1-N** | default | 0.8877 | 0.5636 | — | 7.5M | `reproduce_sku110k.py` | [run](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/hkqclc61) |
| **YOLO-Master-EsMoE-N** | default (sparse) | 0.2577 | 0.1106 | ↓71% | 3.4M | `reproduce_sku110k.py` | [run](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/fivhxn27) |
| **YOLO-Master-EsMoE-N** | dense (`--no-sparse-eval`) | **0.8970** | **0.5728** | ↑+248% | 3.4M | `reproduce_sku110k.py` | [run](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/l0cdknhd) |

> **Note**: The SKU-110K EsMoE-N sparse eval result (mAP50=0.2577) is **not** a training failure — train losses converge normally (see §④). The collapse is caused by ES_MOE's default sparse inference pruning experts without normalization (see §③ Engineering Robustness). Training with `--no-sparse-eval` flag (dense eval) restores mAP to **0.8970**, surpassing the v0.1-N baseline (0.8877) with only ~45% parameters. This is a separate training run (`l0cdknhd`) where validation and checkpoint saving all use dense inference.

**Training Environment**: RTX 4050 6GB, PyTorch 2.11.0+cu128, Windows 11, `batch=4`, `workers=4-8`

**WandB Project**: [https://wandb.ai/1853979230-company-/yolo-master-reproduce](https://wandb.ai/1853979230-company-/yolo-master-reproduce)

**Key Observations**:
- v0.1-N reproduces closely with official reference (VisDrone: ~83% of official mAP50-95, SKU-110K: ~97% of official mAP50-95) at 100 epochs
- EsMoE-N shows depressed mAP on SKU-110K under default sparse inference mode
- With `--no-sparse-eval` (dense eval), EsMoE-N mAP50 recovers dramatically on SKU-110K: 0.2577 → 0.8970, reaching parity with v0.1-N (0.8877) and even slightly surpassing it
- On VisDrone, dense eval provides marginal improvement (0.3008 → 0.3037), as sparse inference already performs comparably on this dataset
- This matches the observation in [#81](https://github.com/Tencent/YOLO-Master/pull/81): ES_MOE's default sparse inference prunes experts with `dynamic_threshold=0.4` without proper renormalization, causing feature magnitude mismatch vs dense training path
- Note: Our results are at 100 epochs on RTX 4050 6GB; PR #81 reports results at 300 epochs on H200

---

## ③ Engineering Robustness & Fixes

### Offline Pretrained Weights Adaptation

To address GitHub download timeouts for the pretrained weights (`yolo11n.pt`) in restricted network environments, `_reproduce_common.py` sets `pretrained=False` and initializes the model directly from the configuration YAML, avoiding runtime GitHub fetches during training.

### VRAM Optimization & DataLoader Configuration

For 6GB VRAM GPUs (e.g., RTX 4050):
- **batch size**: 4 (reduced from default 16 to fit GPU memory)
- **workers**: 4–8 
- Adjust `--batch` and `--workers` based on available GPU resources

### ES_MOE Sparse Inference Mitigation

`eval_dense.py` provides post-training evaluation with `ES_MOE.use_sparse_inference=False`, ensuring consistent behavior between training (dense path) and inference (dense path). `_reproduce_common.py` supports `--no-sparse-eval` to enable dense validation during training, making per-epoch validation and saved checkpoints all use the dense path.

---

## ④ Training Loss Metrics (Final Epoch)

### VisDrone v0.1-N (Epoch 100)

| Metric | Value |
| :--- | :--- |
| Train Box Loss | 1.4238 |
| Train Cls Loss | 1.0672 |
| Train DFL Loss | 0.9983 |
| Train MoE Loss | 1.0004 |
| Val Box Loss | 1.3631 |
| Val Cls Loss | 1.0190 |
| Val DFL Loss | 0.9729 |

### VisDrone EsMoE-N (Epoch 100)

| Metric | Value |
| :--- | :--- |
| Train Box Loss | 1.3420 |
| Train Cls Loss | 0.9772 |
| Train DFL Loss | 0.9784 |
| Train MoE Loss | 1.0000 |
| Val Box Loss | 1.3409 |
| Val Cls Loss | 0.9846 |
| Val DFL Loss | 0.9618 |

### SKU-110K v0.1-N (Epoch 100)

| Metric | Value |
| :--- | :--- |
| Train Box Loss | 1.3382 |
| Train Cls Loss | 0.5781 |
| Train DFL Loss | 1.0460 |
| Train MoE Loss | 1.0000 |
| Val Box Loss | 1.3164 |
| Val Cls Loss | 0.5612 |
| Val DFL Loss | 1.0233 |

### SKU-110K EsMoE-N (Epoch 100, dense eval)

| Metric | Value |
| :--- | :--- |
| Train Box Loss | 1.3279 |
| Train Cls Loss | 0.5670 |
| Train DFL Loss | 1.0424 |
| Train MoE Loss | 1.0000 |
| Val Box Loss | 1.3020 |
| Val Cls Loss | 0.5491 |
| Val DFL Loss | 1.0161 |

**Note**: With `--no-sparse-eval` (dense eval), validation losses align closely with training losses, confirming that the sparse inference path was the root cause of the previous mAP collapse.

---

## ⑤ Training Curve & Reproducibility Guide

### Training Convergence

Complete `results.png` convergence curves are available in the output directories, demonstrating stable loss reduction and mAP improvement across all training epochs:

- **VisDrone v0.1-N**: `runs/reproduce/visdrone/VisDrone_v0.1-N/results.png`
- **VisDrone EsMoE-N**: `runs/reproduce/visdrone/VisDrone_EsMoE-N/results.png`
- **SKU-110K v0.1-N**: `runs/reproduce/sku110k/SKU-110K_v0.1-N/results.png`
- **SKU-110K EsMoE-N**: `runs/reproduce/sku110k/SKU-110K_EsMoE-N/results.png`

### Reproduce Commands

```bash
# VisDrone v0.1-N
python scripts/reproduce/reproduce_visdrone.py --model v0.1-N --epochs 100 --batch 4 --imgsz 640 --device 0 --workers 4

# VisDrone EsMoE-N
python scripts/reproduce/reproduce_visdrone.py --model EsMoE-N --epochs 100 --batch 4 --imgsz 640 --device 0 --workers 4

# SKU-110K v0.1-N
python scripts/reproduce/reproduce_sku110k.py --model v0.1-N --epochs 100 --batch 4 --imgsz 640 --device 0 --workers 8

# SKU-110K EsMoE-N (dense training, matches reported results)
python scripts/reproduce/reproduce_sku110k.py --model EsMoE-N --epochs 100 --batch 4 --imgsz 640 --device 0 --workers 8 --no-sparse-eval

# Optional: post-training dense eval for any EsMoE checkpoint
python scripts/reproduce/eval_dense.py --checkpoint runs/reproduce/.../weights/best.pt --data <dataset.yaml>
```

### Validation Visualization

Validation set predictions are saved in each output directory:
- `val_batch0_labels.jpg` / `val_batch0_pred.jpg`
- `val_batch1_labels.jpg` / `val_batch1_pred.jpg`
- `val_batch2_labels.jpg` / `val_batch2_pred.jpg`

---

## Checklist

- [x] VisDrone v0.1-N training completed (100 epochs)
- [x] VisDrone EsMoE-N training completed (100 epochs)
- [x] SKU-110K v0.1-N training completed (100 epochs)
- [x] SKU-110K EsMoE-N training completed (100 epochs)
- [x] All key metrics extracted (mAP50, mAP50-95, losses)
- [x] Training curves and visualization outputs generated
- [x] Reproducible scripts provided
- [x] WandB logs synced from results.csv (public project)
- [x] EsMoE-N dense eval (`--no-sparse-eval`) results obtained and documented
