#!/bin/bash
# Brain Tumor LoRA Rank Sweep (r=4, 8, 16)
# 使用: bash examples/lora_examples/run_lora_brain_tumor_sweep.sh
set -e
cd "$(dirname "$0")/../.."

echo "$(date): Starting Brain Tumor LoRA Rank Sweep"

for r in 4 8 16; do
    alpha=$((r * 2))
    echo "$(date): ===== r=${r} (alpha=${alpha}) ====="
    yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml \
           lora_r=${r} lora_alpha=${alpha} epochs=40 \
           project=runs/lora_final name=braintumor_r${r}
    echo "$(date): braintumor_r${r} DONE"
done

echo "$(date): Brain Tumor Sweep COMPLETED"
