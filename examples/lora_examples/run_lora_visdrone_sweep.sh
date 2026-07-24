#!/bin/bash
# VisDrone LoRA Rank Sweep (r=4, 8, 16)
# 使用: bash examples/lora_examples/run_lora_visdrone_sweep.sh
set -e
cd "$(dirname "$0")/../.."

echo "$(date): Starting VisDrone LoRA Rank Sweep"

for r in 4 8 16; do
    alpha=$((r * 4))
    echo "$(date): ===== r=${r} (alpha=${alpha}) ====="
    yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml \
           lora_r=${r} lora_alpha=${alpha} epochs=40 \
           project=runs/lora_final name=visdrone_r${r}
    echo "$(date): visdrone_r${r} DONE"
done

echo "$(date): VisDrone Sweep COMPLETED"
