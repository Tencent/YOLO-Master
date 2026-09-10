# YOLO-Master Training Results Summary

## 📊 Results Comparison Table

### VisDrone Dataset (Aerial, dense small objects)

| Model | Eval Method | mAP50 | mAP50-95 | Precision(B) | Recall(B) | Train Box Loss | Train Cls Loss | Train DFL Loss | Train MoE Loss |
|---|---|---|---|---|---|---|---|---|---|
| YOLO-Master-v0.1-N | default | 0.2916 | 0.1666 | 0.3869 | 0.3012 | 1.4238 | 1.0672 | 0.9983 | 1.0004 |
| YOLO-Master-EsMoE-N | default (sparse) | 0.3008 | 0.1731 | 0.4006 | 0.3101 | 1.3420 | 0.9772 | 0.9784 | 1.0000 |
| YOLO-Master-EsMoE-N | `--no-sparse-eval` | 0.3037 | 0.1758 | 0.4006 | 0.3109 | 1.3420 | 0.9772 | 0.9784 | 1.0000 |

### SKU-110K Dataset (Retail, dense products, single class)

| Model | Eval Method | mAP50 | mAP50-95 | Precision(B) | Recall(B) | Train Box Loss | Train Cls Loss | Train DFL Loss | Train MoE Loss |
|---|---|---|---|---|---|---|---|---|---|
| YOLO-Master-v0.1-N | default | 0.8877 | 0.5636 | 0.8941 | 0.8186 | 1.3382 | 0.5781 | 1.0460 | 1.0000 |
| YOLO-Master-EsMoE-N | default (sparse) | 0.2577 | 0.1106 | - | - | - | - | - | - |
| YOLO-Master-EsMoE-N | `--no-sparse-eval` | 0.8970 | 0.5728 | 0.9036 | 0.8251 | 1.3279 | 0.5670 | 1.0424 | 1.0000 |

## 📈 Key Observations

1. **EsMoE-N Sparse Evaluation Bug**: The default sparse evaluation path (`use_sparse_inference=True`) causes mAP to collapse dramatically. Using `--no-sparse-eval` restores performance to match or slightly surpass v0.1-N.

2. **Model Size vs Performance**: EsMoE-N has only ~45% the parameters (3.4M vs 7.5M) but achieves comparable performance when evaluated correctly with `--no-sparse-eval`.

3. **Dataset Characteristics**:
   - VisDrone: Lower overall mAP due to small object detection challenge
   - SKU-110K: High mAP due to single-class, larger objects

4. **Training Environment**: RTX 4050 6GB, PyTorch 2.11.0+cu128, Windows 11, 100 epochs, batch=4, workers=4-8

## 🔗 WandB Logs

| Model | Dataset | WandB Run |
|---|---|---|
| v0.1-N | VisDrone | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/501cw1pq) |
| EsMoE-N | VisDrone | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/h9doimnp) |
| v0.1-N | SKU-110K | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/hkqclc61) |
| EsMoE-N | SKU-110K (sparse) | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/fivhxn27) |
| EsMoE-N | SKU-110K (dense) | [View](https://wandb.ai/1853979230-company-/yolo-master-reproduce/runs/l0cdknhd) |

**WandB Project**: [https://wandb.ai/1853979230-company-/yolo-master-reproduce](https://wandb.ai/1853979230-company-/yolo-master-reproduce)
