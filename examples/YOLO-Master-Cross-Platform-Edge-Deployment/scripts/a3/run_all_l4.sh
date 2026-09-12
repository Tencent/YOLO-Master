#!/usr/bin/env bash
# A3 8.24 entry check: the whole smoke chain on the L4, in order, every stage tee'd to
# a3/logs. /data is the shared volume, so results land in the checkout directly.
#   nohup bash scripts/a3/run_all_l4.sh > a3/logs/run_all.log 2>&1 &
set -uo pipefail
cd "${EDGE:-/data/yolo-master-edge}"
. "${VENV:-/root/a3venv}/bin/activate"
export PYTHONUNBUFFERED=1 PYTHONPATH=/data/YOLO-Master YOLO_AUTOINSTALL=False PATH=/usr/local/cuda-12.4/bin:$PATH
# ORT 1.20 dlopens libcudnn_adv.so.9 etc. - they ship inside torch's nvidia wheels; without
# this the CUDA EP is "available" but fails to load and ORT silently runs on CPU
export LD_LIBRARY_PATH=$(python -c "import nvidia.cudnn, nvidia.cublas, os; print(os.path.dirname(nvidia.cudnn.__file__)+'/lib:'+os.path.dirname(nvidia.cublas.__file__)+'/lib')"):${LD_LIBRARY_PATH:-}
V01=/data/YOLO-Master/YOLO-Master-v0.1-N.pt
ES=/data/YOLO-Master/YOLO-Master-EsMoE-N.pt
DATA=a3/config/coco-a3.yaml
CAL=/data/datasets/coco/images/train2017      # TRAIN split only (val calibration = leakage)
CALN=${CALN:-1024}                            # >=1024: the measured coverage law
STAGES=${STAGES:-env,smoke,ladder,qdq,backend}
has() { [[ ",$STAGES," == *",$1,"* ]]; }
t0=$(date +%s); say() { echo "[chain $(( $(date +%s) - t0 ))s] $*"; }

if has env; then
  say env_matrix
  python scripts/a3/env_matrix.py --out a3/env 2>&1 | tee a3/logs/env_matrix.log
fi

if has smoke; then
  say smoke v0.1-N
  python scripts/a3/smoke_export.py --model $V01 --out runs/a3/v01n --data $DATA \
      --json a3/results/smoke_v01n.json --log a3/logs/v01n_smoke.log
  say "smoke v0.1-N exit=$?"
  say smoke EsMoE-N
  python scripts/a3/smoke_export.py --model $ES --out runs/a3/esmoen --data $DATA \
      --json a3/results/smoke_esmoen.json --log a3/logs/esmoen_smoke.log
  say "smoke EsMoE-N exit=$?"
  say "smoke v0.1-N --dynamic (policy rejection = the failure log, exit 2 expected)"
  python scripts/a3/smoke_export.py --model $V01 --out runs/a3/v01n-dynamic --dynamic --data $DATA \
      --json a3/results/smoke_v01n_dynamic_reject.json --log a3/logs/v01n_smoke_dynamic_reject.log
  say "dynamic-axes run exit=$? (2 expected)" | tee -a a3/logs/v01n_smoke_dynamic_reject.log
fi

if has ladder; then
  # P0 + contrast: fp32 (TF32 off = honest), fp16, implicit entropy INT8 with head +
  # routers pinned (base build only: --max-rounds 0, no greedy search)
  for pair in "v01n:$V01" "esmoen:$ES"; do
    tag=${pair%%:*}; pt=${pair#*:}
    say "TRT ladder $tag"
    python scripts/a3/quantize_trt.py --model $pt --data $DATA --out runs/a3/$tag/trt \
        --calib-images $CAL --calib-n $CALN --modes fp32,fp16,int8 --max-rounds 0 \
        --tf32-baseline off 2>&1 | tee a3/logs/${tag}_trt_ladder.log
    cp runs/a3/$tag/trt/ladder.csv a3/results/ladder_$tag.csv 2>/dev/null || true
  done
fi

if has qdq; then
  # P1 deliverable recipe: calibrate-only explicit per-channel Q/DQ (no QAT)
  for pair in "v01n:$V01" "esmoen:$ES"; do
    tag=${pair%%:*}; pt=${pair#*:}
    say "INT8 QDQ $tag"
    python scripts/a3/qat_moe.py --model $pt --data $DATA --out runs/a3/$tag/qdq \
        --calib-n $CALN --skip-train 2>&1 | tee a3/logs/${tag}_int8_qdq.log
    cp runs/a3/$tag/qdq/result.json a3/results/qdq_$tag.json 2>/dev/null || true
  done
fi

if has backend; then
  for pair in "v01n:$V01:YOLO-Master-v0.1-N" "esmoen:$ES:YOLO-Master-EsMoE-N"; do
    IFS=: read -r tag pt stem <<< "$pair"
    say "PyTorch vs ORT $tag"
    python scripts/a3/backend_val.py --model $pt --onnx runs/a3/$tag/$stem.onnx --data $DATA \
        --json a3/results/backend_$tag.json 2>&1 | tee a3/logs/${tag}_backend_val.log
  done
fi

pip freeze > a3/env/pip-freeze.after.txt
# the editable YOLO-Master install renders either as a VCS URL or as ultralytics==8.3.240
# depending on pip's cwd; the matrix records its file path + commit, so ignore that line
diff -q <(grep -v ultralytics a3/env/pip-freeze.txt) <(grep -v ultralytics a3/env/pip-freeze.after.txt) \
  && say "env unchanged (no auto-install)" || say "WARNING env changed during the run"
say done
