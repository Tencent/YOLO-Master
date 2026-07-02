# Dense Vertical Dataset Reproduction

This folder contains reproducible entrypoints for comparing `YOLO-Master-v0.1-N` and `YOLO-Master-EsMoE-N` on dense vertical detection datasets:

- VisDrone: aerial dense small objects, using `ultralytics/cfg/datasets/VisDrone.yaml`
- SKU-110K: retail dense products, using `ultralytics/cfg/datasets/SKU-110K.yaml`

The scripts default to `imgsz=640`, `epochs=100`, W&B enabled, and the public `YOLO-Master-v26.02` N checkpoints. Use `--epochs 300` when GPU budget allows.

For `YOLO-Master-EsMoE-N`, use `--no-sparse-eval` to keep the corrected dense-eval path. If you explicitly pass `--sparse-eval`, the script will reproduce the legacy sparse-validation behavior, which is known to collapse VisDrone `mAP50`.

## Prepare
Optionally pre-download release checkpoints to avoid repeated downloads:

```bash
mkdir -p weights
wget -P weights https://github.com/Tencent/YOLO-Master/releases/download/YOLO-Master-v26.02/YOLO-Master-v0.1-N.pt
wget -P weights https://github.com/Tencent/YOLO-Master/releases/download/YOLO-Master-v26.02/YOLO-Master-EsMoE-N.pt
```

Dataset download/conversion is handled by the built-in dataset YAMLs when training starts. You can trigger it explicitly with a short dry training command if desired:

```bash
yolo detect train data=VisDrone.yaml model=weights/YOLO-Master-v0.1-N.pt epochs=1 imgsz=640
yolo detect train data=SKU-110K.yaml model=weights/YOLO-Master-v0.1-N.pt epochs=1 imgsz=640
```

Expected dataset locations are the Ultralytics defaults under `datasets/VisDrone` and `datasets/SKU-110K`.

## Train

Run VisDrone:

```bash
- python scripts/reproduce/reproduce_visdrone.py --models v01_n --epochs 100 --imgsz 640 --batch 8 --device 0 --wandb-project YOLO-Master
- python scripts/reproduce/reproduce_visdrone.py --models esmoe_n --epochs 100 --imgsz 640 --batch 8 --device 0 --wandb-project YOLO-Master --no-sparse-eval
```

Run SKU-110K:

```bash
- python scripts/reproduce/reproduce_sku110k.py --models v01_n --epochs 100 --imgsz 640 --batch 8 --device 0 --wandb-project YOLO-Master
- python scripts/reproduce/reproduce_sku110k.py --models esmoe_n --epochs 100 --imgsz 640 --batch 8 --device 0 --wandb-project YOLO-Master --no-sparse-eval
```

## Logs And Results

Each run writes:

- `runs/reproduce_visdrone/<dataset>_<model>/train.log`
- `runs/reproduce_visdrone/<dataset>_<model>/results.csv`
- `runs/reproduce_sku110k/<dataset>_<model>/train.log`
- `runs/reproduce_sku110k/<dataset>_<model>/results.csv`
- `summary.csv`, `summary.md`, and `status.json` in each project directory

`results.csv` contains per-epoch `metrics/mAP50(B)`, `metrics/mAP50-95(B)`, `train/box_loss`, `train/cls_loss`, and `train/moe_loss` when the model exposes the MoE auxiliary loss. W&B receives the same Ultralytics metrics when `--wandb` is enabled.

After making the W&B project public, regenerate the markdown summary with the report URL:

```bash
python scripts/reproduce/reproduce_visdrone.py --summary-only --wandb-url https://wandb.ai/<entity>/<project>
python scripts/reproduce/reproduce_sku110k.py --summary-only --wandb-url https://wandb.ai/<entity>/<project>
```

## Results

Each row below summarizes the final epoch metrics for one run. The W&B column links to the corresponding run page for full training history, and the EsMoE-N rows should be reproduced with `--no-sparse-eval`.



| Dataset | Model | Epochs | mAP50 | mAP50-95 | box_loss | cls_loss | moe_loss | W&B |
|:--|:--|:--|:--|:--|:--|:--|:--|:--|
| VisDrone | YOLO-Master-v0.1-N | 100 | 0.36446 | 0.21516 | 1.20808 | 0.78884 | 0 | [VisDrone-v0.1-N](https://wandb.ai/nicholasdezai-east-china-normal-university/YOLO-Master/runs/rgzc3mpn?nw=nwusernicholasdezai) |
| VisDrone | YOLO-Master-EsMoE-N | 100 | 0.35853 | 0.21222 | 1.19984 | 0.7674 | 0 | [VisDrone-EsMoE-N](https://wandb.ai/nicholasdezai-east-china-normal-university/YOLO-Master/runs/bjeg1pq2?nw=nwusernicholasdezai) |
| SKU-110K | YOLO-Master-v0.1-N | 100 | 0.90783 | 0.5865 | 1.26837 | 0.53381 | 0 | [SKU-110K-v0.1-N](https://wandb.ai/nicholasdezai-east-china-normal-university/YOLO-Master/runs/hj515l9s?nw=nwusernicholasdezai) |
| SKU-110K | YOLO-Master-EsMoE-N | 100 | 0.91098 | 0.58729 | 1.25772 | 0.52551 | 0 | [SKU-110K-EsMoE-N](https://wandb.ai/nicholasdezai-east-china-normal-university/YOLO-Master/runs/9kf31hbx?nw=nwusernicholasdezai) |

## Known Issues

- `ModuleNotFoundError: cv2`: install project dependencies with `python -m pip install -e ".[logging]"` or install `opencv-python`.
- SKU-110K download is large, about 13.6 GB. Ensure disk space before starting.
- If `YOLO-Master-EsMoE-N` shows near-zero VisDrone mAP, check that sparse eval is disabled. The reproduction scripts now default to dense eval; only pass `--sparse-eval` when reproducing the legacy behavior on purpose.
