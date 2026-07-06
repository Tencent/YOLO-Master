# YOLO-Master Dense Dataset Reproduction

This folder contains reproduction notes, scripts, logs summary, and PR text for training YOLO-Master nano variants on two dense-scene vertical datasets:

- VisDrone2019-DET: aerial dense small-object detection.
- SKU-110K: retail shelf dense product detection.

The experiments compare:

- `YOLO-Master-v0.1-N`: `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml`
- `YOLO-Master-EsMoE-N`: `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml`

All official runs used the built-in dataset configs:

- `ultralytics/cfg/datasets/VisDrone.yaml`
- `ultralytics/cfg/datasets/SKU-110K.yaml`

## Environment

Cloud server:

- Platform: Radeon cloud instance
- OS: Linux container
- GPU: AMD Radeon Graphics, 49136 MiB VRAM
- ROCm/PyTorch: `torch 2.10.0+rocm7.2.4`
- Python: 3.12.3
- YOLO-Master/Ultralytics: 8.3.240 in repo

Training settings:

```bash
imgsz=640
epochs=100
batch=32
workers=8
device=0
amp=False
wandb=offline, then synced
```

## Dataset Preparation

### VisDrone

Required archives:

- `VisDrone2019-DET-train.zip`
- `VisDrone2019-DET-val.zip`
- `VisDrone2019-DET-test-dev.zip`

Place them under the Ultralytics dataset directory, for example:

```bash
/workspace/datasets/VisDrone/
```

Then run:

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('VisDrone.yaml', autodownload=True)"
```

### SKU-110K

Required archive:

```text
SKU110K_fixed.tar.gz
```

If extraction fails on a rootless or network filesystem, use:

```bash
tar -xzf SKU110K_fixed.tar.gz --no-same-owner --no-same-permissions -C /workspace/datasets
```

Then run:

```bash
python -c "from ultralytics.data.utils import check_det_dataset; check_det_dataset('SKU-110K.yaml', autodownload=True)"
```

The SKU-110K source contains several JPEG files with warnings such as `Corrupt JPEG data: premature end of data segment`. These warnings did not interrupt training or validation.

## Training Commands

### VisDrone official baselines

```bash
python scripts/reproduce/reproduce_visdrone.py \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --model both \
  --wandb-mode offline \
  --project runs/reproduce/visdrone_100e_official \
  --no-amp
```

### VisDrone EsMoE dense-eval diagnostic

```bash
python scripts/reproduce/reproduce_visdrone.py \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --model EsMoE-N \
  --no-sparse-eval \
  --wandb-mode offline \
  --project runs/reproduce/visdrone_100e_dense_eval \
  --no-amp
```

### SKU-110K official baselines

```bash
python scripts/reproduce/reproduce_sku110k.py \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --model v0.1-N \
  --wandb-mode offline \
  --project runs/reproduce/sku110k_100e_official \
  --no-amp

python scripts/reproduce/reproduce_sku110k.py \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --model EsMoE-N \
  --wandb-mode offline \
  --project runs/reproduce/sku110k_100e_official \
  --no-amp
```

### SKU-110K EsMoE dense-eval diagnostic

```bash
python scripts/reproduce/reproduce_sku110k.py \
  --epochs 100 \
  --imgsz 640 \
  --batch 32 \
  --workers 8 \
  --device 0 \
  --model EsMoE-N \
  --no-sparse-eval \
  --wandb-mode offline \
  --project runs/reproduce/sku110k_100e_dense_eval \
  --no-amp
```

## Results

| Dataset | Model | Eval mode | mAP50 | mAP50-95 | W&B |
|---|---|---|---:|---:|---|
| VisDrone | YOLO-Master-v0.1-N | official | 0.30500 | 0.17571 | https://wandb.ai/yolo-master-reproduce/runs/nol27cjg |
| VisDrone | YOLO-Master-EsMoE-N | official sparse eval | 0.00777 | 0.00191 | https://wandb.ai/yolo-master-reproduce/runs/13akqv4g |
| VisDrone | YOLO-Master-EsMoE-N | `--no-sparse-eval` | 0.30845 | 0.17797 | https://wandb.ai/yolo-master-reproduce/runs/p1yab1k1 |
| SKU-110K | YOLO-Master-v0.1-N | official | 0.89113 | 0.56870 | https://wandb.ai/yolo-master-reproduce/runs/4vc21dt1 |
| SKU-110K | YOLO-Master-EsMoE-N | official sparse eval | 0.10318 | 0.04656 | https://wandb.ai/yolo-master-reproduce/runs/v14p2z5d |
| SKU-110K | YOLO-Master-EsMoE-N | `--no-sparse-eval` | 0.89557 | 0.57300 | https://wandb.ai/yolo-master-reproduce/runs/1dlyq3xv |

Full metric values are also available in:

```text
results/final_results.csv
```

## Main Finding

The `YOLO-Master-EsMoE-N` official sparse evaluation path produces extremely low validation mAP on both dense datasets, despite normal training losses. Disabling sparse eval restores the model to the expected range:

- VisDrone: `0.00777 -> 0.30845` mAP50.
- SKU-110K: `0.10318 -> 0.89557` mAP50.

This suggests the issue is not failed training. It is a train/eval mismatch in the `ES_MOE` sparse inference path.

Training uses a dense forward path that combines all experts with softmax routing weights. The default eval/inference sparse path keeps only a subset of experts and does not match the activation scale/distribution used during training. The `--no-sparse-eval` flag forces `ES_MOE.use_sparse_inference=False` at validation time for both the live model and EMA model.

## Known Issues

1. SSL certificate errors on the cloud environment prevented direct GitHub/Ultralytics downloads. Workaround: upload archives manually or use `curl -k`.
2. `/workspace` was only 10 GB. Training runs and datasets were moved to `/data`, with `/workspace/datasets` and `runs` symlinked.
3. SKU-110K emits JPEG warnings. Training and validation completed successfully.
4. EsMoE-N default sparse eval is not reliable for these dense datasets. Use `--no-sparse-eval` as a diagnostic/eval-alignment switch.

