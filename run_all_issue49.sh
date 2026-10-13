#!/bin/bash
# Issue #49: Full reproduction pipeline
# 4 experiments: VisDrone/SKU-110K × v0.1-N/EsMoE-N(dense)
# epochs=100 (issue: 100~300)  batch=16 (RTX 4080 12GB)
set -e

REPO_ROOT="/d/yolo_experiments/YOLO-Master"
cd "$REPO_ROOT"

echo "=========================================="
echo "Issue #49 Reproduction Pipeline"
echo "epochs=100  batch=16  workers=4"
echo "Started: $(date)"
echo "=========================================="

# ── 1. VisDrone + v0.1-N ──────────────────────────
echo ""
echo "[1/4] VisDrone + YOLO-Master-v0.1-N  $(date)"
python -u scripts/issue49/yolo_master_issue_49.py \
    --dataset VisDrone \
    --model YOLO-Master-v0.1-N \
    --batch 16 --epochs 100 --workers 4 \
    --wandb-project yolo-master-reproduce \
    --wandb-group visdrone
echo "[1/4] Done: $(date)"

# ── 2. VisDrone + EsMoE-N (dense eval) ────────────
echo ""
echo "[2/4] VisDrone + YOLO-Master-EsMoE-N (dense eval)  $(date)"
python -u scripts/issue49/yolo_master_issue_49.py \
    --dataset VisDrone \
    --model YOLO-Master-EsMoE-N \
    --batch 16 --epochs 100 --workers 4 \
    --dense-eval-for-esmoe \
    --wandb-project yolo-master-reproduce \
    --wandb-group visdrone
echo "[2/4] Done: $(date)"

# ── 3. SKU-110K + v0.1-N ──────────────────────────
echo ""
echo "[3/4] SKU-110K + YOLO-Master-v0.1-N  $(date)"
python -u scripts/issue49/yolo_master_issue_49.py \
    --dataset SKU-110K \
    --model YOLO-Master-v0.1-N \
    --batch 16 --epochs 100 --workers 4 \
    --wandb-project yolo-master-reproduce \
    --wandb-group sku110k
echo "[3/4] Done: $(date)"

# ── 4. SKU-110K + EsMoE-N (dense eval) ────────────
echo ""
echo "[4/4] SKU-110K + YOLO-Master-EsMoE-N (dense eval)  $(date)"
python -u scripts/issue49/yolo_master_issue_49.py \
    --dataset SKU-110K \
    --model YOLO-Master-EsMoE-N \
    --batch 16 --epochs 100 --workers 4 \
    --dense-eval-for-esmoe \
    --wandb-project yolo-master-reproduce \
    --wandb-group sku110k
echo "[4/4] Done: $(date)"

echo ""
echo "=========================================="
echo "ALL DONE! $(date)"
echo "=========================================="
