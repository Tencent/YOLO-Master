#!/usr/bin/env bash
# A3 at acce839c: clean v0.1-N ladder re-run (the first pass crashed on the INT8 rung before
# writing ladder.csv) plus the implicit-INT8 pin-set diagnosis: which always-pin set makes
# TensorRT's OBEY build infeasible on the 8.4.101 exporter's graph.
set -uo pipefail
cd "${EDGE:-/data/yolo-master-edge}"
. "${VENV:-/root/a3venv}/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH=/data/YOLO-Master YOLO_AUTOINSTALL=False PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn, nvidia.cublas, os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib:'+os.path.dirname(nvidia.cublas.__file__)+'/lib')"):${LD_LIBRARY_PATH:-}
V01=/data/YOLO-Master/YOLO-Master-v0.1-N.pt
DATA=a3/config/coco-a3.yaml
CAL=/data/datasets/coco/images/train2017
t0=$(date +%s); say() { echo "[diag $(( $(date +%s) - t0 ))s] $*"; }

rm -f runs/a3/v01n/trt/ladder.csv a3/results/ladder_v01n.csv
say "v0.1-N ladder re-run (fp32, fp16, implicit int8 head+routers; a failed int8 is recorded, not fatal)"
python scripts/a3/quantize_trt.py --model $V01 --data $DATA --out runs/a3/v01n/trt \
    --calib-images $CAL --calib-n 1024 --modes fp32,fp16,int8 --max-rounds 0 \
    --tf32-baseline off 2>&1 | tee a3/logs/v01n_trt_ladder.log
cp runs/a3/v01n/trt/ladder.csv a3/results/ladder_v01n.csv 2>/dev/null || true

ONNX=runs/a3/v01n/trt/YOLO-Master-v0.1-N.onnx
for pins in head routers none; do
  say "implicit int8 pin-set diagnosis: --pins $pins"
  python scripts/a3/quantize_trt.py --model $ONNX --data $DATA --out runs/a3/v01n/trt-pins-$pins \
      --calib-images $CAL --calib-n 1024 --modes int8 --pins $pins --max-rounds 0 \
      2>&1 | tee a3/logs/v01n_int8_pins_$pins.log
  cp runs/a3/v01n/trt-pins-$pins/ladder.csv a3/results/ladder_v01n_pins_$pins.csv 2>/dev/null || true
done
say done
