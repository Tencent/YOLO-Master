#!/usr/bin/env bash

set -euo pipefail

# Activate conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate yolo_master

CFG="examples/lora_examples/yolo_master_brain_tumor_lora.yaml"
PROJECT="runs/lora_examples"
GPU_ID=0

mkdir -p logs

echo "Starting LoRA sweep on single GPU: ${GPU_ID}"

CUDA_VISIBLE_DEVICES=${GPU_ID} yolo train \
  cfg=${CFG} \
  device=0 \
  lora_r=4 \
  lora_alpha=8 \
  name=brain_tumor_r4 \
  project=${PROJECT} \
  > logs/brain_tumor_r4.log 2>&1 &

PID_R4=$!

CUDA_VISIBLE_DEVICES=${GPU_ID} yolo train \
  cfg=${CFG} \
  device=0 \
  lora_r=8 \
  lora_alpha=16 \
  name=brain_tumor_r8 \
  project=${PROJECT} \
  > logs/brain_tumor_r8.log 2>&1 &

PID_R8=$!

CUDA_VISIBLE_DEVICES=${GPU_ID} yolo train \
  cfg=${CFG} \
  device=0 \
  lora_r=16 \
  lora_alpha=32 \
  name=brain_tumor_r16 \
  project=${PROJECT} \
  > logs/brain_tumor_r16.log 2>&1 &

PID_R16=$!

echo "Started experiments:"
echo "r=4  alpha=8   PID=${PID_R4}"
echo "r=8  alpha=16  PID=${PID_R8}"
echo "r=16 alpha=32  PID=${PID_R16}"

wait ${PID_R4}
wait ${PID_R8}
wait ${PID_R16}

echo "All LoRA experiments finished."