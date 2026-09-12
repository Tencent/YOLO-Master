#!/usr/bin/env bash
# A3 at acce839c: implicit-INT8 pin-set diagnosis on the v0.1-N graph. The full always-pin set
# (head + routers) makes TensorRT's OBEY build infeasible on the 8.4.101 exporter's graph;
# this isolates which set is responsible.
set -uo pipefail
cd "${EDGE:-/data/yolo-master-edge}"
. "${VENV:-/root/a3venv}/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH=/data/YOLO-Master YOLO_AUTOINSTALL=False PATH=/usr/local/cuda-12.4/bin:$PATH
ONNX=runs/a3/v01n/trt/YOLO-Master-v0.1-N.onnx
for pins in head routers none; do
  echo "[diag] --pins $pins"
  python scripts/a3/quantize_trt.py --model $ONNX --data a3/config/coco-a3.yaml \
      --out runs/a3/v01n/trt-pins-$pins --calib-images /data/datasets/coco/images/train2017 \
      --calib-n 1024 --modes int8 --pins $pins --max-rounds 0 2>&1 | tee a3/logs/v01n_int8_pins_$pins.log
  cp runs/a3/v01n/trt-pins-$pins/ladder.csv a3/results/ladder_v01n_pins_$pins.csv 2>/dev/null || true
done
echo "[diag] done"
