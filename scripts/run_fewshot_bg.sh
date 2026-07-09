#!/bin/bash
# Few-shot LoRA background validation on COCO128
# Runs baseline val + MoLoRA training, saves results to JSON

cd /Users/gatilin/PycharmProjects/YOLO-Master-v0708/scripts

export WANDB_MODE=disabled
export WANDB_SILENT=true
export KMP_DUPLICATE_LIB_OK=TRUE
export YOLO_AUTOINSTALL=false
export YOLO_VERBOSE=false

LOG="/Users/gatilin/PycharmProjects/YOLO-Master-v0708/scripts/fewshot_lora_bg.log"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting few-shot LoRA validation..." > "$LOG"

python3 fewshot_lora_quick.py >> "$LOG" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done." >> "$LOG"
