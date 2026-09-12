#!/usr/bin/env bash
# A3 follow-up chain (runs after run_all_l4.sh): EsMoE-N ladder re-run (legacy
# property-shadow repair), the implicit-INT8 stem-pair rung (head + routers + model.0/1,
# the best implicit recipe from project03) for both models, the smoke re-runs under the
# final parity gate, and the PyTorch-vs-ORT stage with ORT's CUDA EP actually loadable.
set -uo pipefail
cd "${EDGE:-/data/yolo-master-edge}"
. "${VENV:-/root/a3venv}/bin/activate"
export PYTHONPATH=/data/YOLO-Master YOLO_AUTOINSTALL=False PATH=/usr/local/cuda-12.4/bin:$PATH
# ORT 1.20 dlopens libcudnn_adv.so.9 etc. - they ship inside torch's nvidia wheels
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn, nvidia.cublas, os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib:'+os.path.dirname(nvidia.cublas.__file__)+'/lib')"):${LD_LIBRARY_PATH:-}
V01=/data/YOLO-Master/YOLO-Master-v0.1-N.pt
ES=/data/YOLO-Master/YOLO-Master-EsMoE-N.pt
DATA=a3/config/coco-a3.yaml
CAL=/data/datasets/coco/images/train2017
CALN=${CALN:-1024}
t0=$(date +%s); say() { echo "[followup $(( $(date +%s) - t0 ))s] $*"; }

say "TRT ladder esmoen (re-run)"
python scripts/a3/quantize_trt.py --model $ES --data $DATA --out runs/a3/esmoen/trt \
    --calib-images $CAL --calib-n $CALN --modes fp32,fp16,int8 --max-rounds 0 \
    --tf32-baseline off 2>&1 | tee a3/logs/esmoen_trt_ladder.log
cp runs/a3/esmoen/trt/ladder.csv a3/results/ladder_esmoen.csv 2>/dev/null || true

for pair in "v01n:YOLO-Master-v0.1-N" "esmoen:YOLO-Master-EsMoE-N"; do
  tag=${pair%%:*}; stem=${pair#*:}
  say "implicit INT8 stem-pair rung $tag"
  python scripts/a3/quantize_trt.py --model runs/a3/$tag/trt/$stem.onnx --data $DATA \
      --out runs/a3/$tag/trt-stempair --calib-images $CAL --calib-n $CALN \
      --modes int8 --pin-modules 0,1 --max-rounds 0 2>&1 | tee a3/logs/${tag}_trt_stempair.log
  cp runs/a3/$tag/trt-stempair/ladder.csv a3/results/ladder_${tag}_stempair.csv 2>/dev/null || true
done

say "INT8 QDQ v01n (re-run: export-equivalent dense calibration)"
python scripts/a3/qat_moe.py --model $V01 --data $DATA --out runs/a3/v01n/qdq \
    --calib-n $CALN --skip-train 2>&1 | tee a3/logs/v01n_int8_qdq.log
cp runs/a3/v01n/qdq/result.json a3/results/qdq_v01n.json 2>/dev/null || true

say "smoke re-runs (final parity gate, honest ORT provider)"
python scripts/a3/smoke_export.py --model $V01 --out runs/a3/v01n --data $DATA \
    --json a3/results/smoke_v01n.json --log a3/logs/v01n_smoke.log; say "smoke v0.1-N exit=$?"
python scripts/a3/smoke_export.py --model $ES --out runs/a3/esmoen --data $DATA \
    --json a3/results/smoke_esmoen.json --log a3/logs/esmoen_smoke.log; say "smoke EsMoE-N exit=$?"

for pair in "v01n:$V01:YOLO-Master-v0.1-N" "esmoen:$ES:YOLO-Master-EsMoE-N"; do
  IFS=: read -r tag pt stem <<< "$pair"
  say "PyTorch vs ORT $tag"
  python scripts/a3/backend_val.py --model $pt --onnx runs/a3/$tag/$stem.onnx --data $DATA \
      --json a3/results/backend_$tag.json 2>&1 | tee a3/logs/${tag}_backend_val.log
done

python scripts/a3/env_matrix.py --out a3/env > a3/logs/env_matrix.log 2>&1
pip freeze > a3/env/pip-freeze.after.txt
diff -q a3/env/pip-freeze.txt a3/env/pip-freeze.after.txt && say "env unchanged (no auto-install)" || say "WARNING env changed during the run"
say done
