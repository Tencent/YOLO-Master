# Issue #49 Reproduction

This directory contains reproducible training entrypoints for the issue #49 baseline runs on VisDrone and SKU-110K.

The scripts use the built-in dataset YAML files required by the issue:

- `ultralytics/cfg/datasets/VisDrone.yaml`
- `ultralytics/cfg/datasets/SKU-110K.yaml`

They train both required N-scale models:

- `YOLO-Master-EsMoE-N`: `ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml`
- `YOLO-Master-v0.1-N`: `ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml`

## Environment

```bash
source /home/cz/anaconda3/etc/profile.d/conda.sh
conda activate yolo_master
cd /home/czdata3/腾讯犀牛鸟/YOLO-Master
```

## Dataset Layout

Download and convert datasets before training. In this workspace the data root is:

```text
/home/czdata3/腾讯犀牛鸟/data
```

Expected subfolders:

```text
/home/czdata3/腾讯犀牛鸟/data/VisDrone
/home/czdata3/腾讯犀牛鸟/data/SKU-110K
```

The scripts set Ultralytics `datasets_dir` to `--datasets-dir`, so the built-in YAML `path: VisDrone` and `path: SKU-110K` resolve to the folders above.

To download through the built-in dataset YAML files:

```bash
export DATASETS_DIR=/path/to/data

python - <<'PY'
import os
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import SETTINGS

SETTINGS.update({"datasets_dir": os.environ["DATASETS_DIR"]})
check_det_dataset("ultralytics/cfg/datasets/VisDrone.yaml")
check_det_dataset("ultralytics/cfg/datasets/SKU-110K.yaml")
PY
```

After download, pass the same root to the training wrapper:

```bash
bash scripts/run_reproduce.sh --datasets-dir "$DATASETS_DIR" --dataset visdrone --model all
```

## Smoke Test

Run one epoch offline first to verify the environment, data paths, and model configs:

```bash
bash scripts/run_reproduce.sh \
  --smoke \
  --dataset visdrone \
  --model esmoe_n
```

For SKU-110K:

```bash
bash scripts/run_reproduce.sh \
  --smoke \
  --dataset sku110k \
  --model esmoe_n
```

## Full Training

Use `imgsz=640` and `epochs=100` to start. Increase to 300 if GPU time allows. The wrapper is the recommended entrypoint because it saves a complete raw terminal log automatically.

VisDrone:

```bash
wandb login
bash scripts/run_reproduce.sh \
  --dataset visdrone \
  --model all \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --wandb-mode online
```

SKU-110K:

```bash
wandb login
bash scripts/run_reproduce.sh \
  --dataset sku110k \
  --model all \
  --epochs 100 \
  --imgsz 640 \
  --batch 16 \
  --device 0 \
  --workers 8 \
  --wandb-mode online
```

If training runs out of GPU memory, reduce `--batch` to `8` or `4` and record the final value in the result table.

To run one model explicitly:

```bash
bash scripts/run_reproduce.sh --dataset visdrone --model esmoe_n --epochs 100 --batch 16
bash scripts/run_reproduce.sh --dataset visdrone --model v0_1_n --epochs 100 --batch 16
```

To force a specific raw log path:

```bash
bash scripts/run_reproduce.sh \
  --dataset visdrone \
  --model v0_1_n \
  --epochs 100 \
  --log-file runs/issue49_visdrone/logs/visdrone_v0_1_n_e100.log
```

## Outputs

Default output locations:

```text
runs/issue49_visdrone/
runs/issue49_sku110k/
```

Each run writes Ultralytics training artifacts under model-specific folders, for example:

```text
runs/issue49_visdrone/esmoe_n_e100/results.csv
runs/issue49_visdrone/v0_1_n_e100/results.csv
```

Each script also writes:

```text
summary.csv
summary.md
status.json
```

The wrapper additionally writes complete raw terminal logs:

```text
runs/issue49_visdrone/logs/*.log
runs/issue49_sku110k/logs/*.log
```

The committed reproduction result summary and per-epoch logs are stored in:

```text
scripts/reproduce/RESULTS.md
scripts/reproduce/results/
```

The summary includes:

- `metrics/mAP50(B)`
- `metrics/mAP50-95(B)`
- `train/box_loss`
- `train/cls_loss`
- `train/moe_loss`
- `val/box_loss`
- `val/cls_loss`
- `val/moe_loss`

If a public W&B URL is needed in `summary.csv` and `summary.md`, put it in the run directory as:

```text
wandb_url.txt
```

Then refresh the summary:

```bash
bash scripts/run_reproduce.sh \
  --dataset visdrone \
  --model all \
  --summary-only
```

## Utility Commands

Dry-run without importing heavy training code:

```bash
python scripts/reproduce/reproduce_visdrone.py --dry-run
```

Instantiate both model configs without training:

```bash
python scripts/reproduce/reproduce_visdrone.py \
  --datasets-dir /home/czdata3/腾讯犀牛鸟/data \
  --check-build
```

Resume existing runs:

```bash
python scripts/reproduce/reproduce_visdrone.py \
  --datasets-dir /home/czdata3/腾讯犀牛鸟/data \
  --run-suffix e100 \
  --resume-existing
```

Skip runs that already have `results.csv`:

```bash
python scripts/reproduce/reproduce_visdrone.py \
  --datasets-dir /home/czdata3/腾讯犀牛鸟/data \
  --run-suffix e100 \
  --skip-existing
```

## Known Issues

| Issue | Solution |
| --- | --- |
| GitHub release downloads may be slow or blocked in some networks. | Configure `http_proxy` / `https_proxy`, or download the archives on a machine with stable access and place the prepared folders under `--datasets-dir`. |
| SKU-110K is large and downloads may be interrupted. | Re-run the dataset download command; the local downloader used for these runs supports archive validation and HTTP range resume. Keep enough free disk space for both the archive and extracted dataset. |
| CUDA out-of-memory during training. | Reduce `--batch` from `16` to `8` or `4`, keep `imgsz=640`, and record the actual batch size in `RESULTS.md`. |
| Final validation may fail with missing validator helpers in older local code. | Apply the validator helper/import fix included with this reproduction before re-running training or validation. |
| Raw terminal logs are large and `runs/` is git-ignored. | Keep raw logs under `runs/issue49_<dataset>/logs/` for local review, and commit the compact per-epoch CSV logs under `scripts/reproduce/results/`. |
| W&B URLs are empty in the summary if no public URL is recorded. | Make the W&B run public, write the URL to `<run_dir>/wandb_url.txt`, then rerun `bash scripts/run_reproduce.sh --dataset <dataset> --model all --summary-only`. |
| `moe_loss` is `0` in the completed runs. | The column is still preserved in every `results.csv` and summary table to satisfy the issue logging requirement. |
| Datasets and model weights are large. | Do not commit dataset folders, archives, `runs/` checkpoints, `best.pt`, or `last.pt`; commit scripts, README, `RESULTS.md`, and per-epoch CSV logs only. |
