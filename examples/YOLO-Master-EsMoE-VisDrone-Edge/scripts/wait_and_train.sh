#!/usr/bin/env bash
# Polls GPU memory; when a card frees up (>15GB free), launches the dense-fix
# training (balance-loss patched, batch=16) on that card. Waits politely instead
# of contending with other users' jobs.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate yolomaster
cd /data/matengyu/csy/yolo-master-edge/edge-src
export PYTHONPATH=$PWD/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[waiter] start $(date); polling GPU every 90s for a card with >15GB free..."
while true; do
  # pick the GPU with the most free memory
  read -r gpu free < <(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
    | awk -F', ' '{gsub(/ /,"",$2); if($2+0>15000) print $1,$2}' | sort -k2 -nr | head -1)
  if [ -n "$gpu" ]; then
    echo "[waiter] GPU $gpu has ${free}MiB free -> launching batch=16 dense-fix training at $(date)"
    nohup python scripts/train_visdrone.py --epochs 50 --imgsz 640 --batch 16 --device "$gpu" \
      --workers 8 --dense --name esmoe_n_visdrone_dense_fix \
      > /data/matengyu/csy/yolo-master-edge/runs_train_densefix.log 2>&1 &
    echo "[waiter] launched training PID $!"
    exit 0
  fi
  sleep 90
done
