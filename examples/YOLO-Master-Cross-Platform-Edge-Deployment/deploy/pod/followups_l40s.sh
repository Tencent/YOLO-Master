#!/usr/bin/env bash
# Post-grid follow-ups, sequential so measurements never overlap:
#  1. rescore txt parity of every cell with the final matcher
#  2. pruned ncnn cell rerun with the dense (SDPA) export
#  3. GPU worker-scaling run (c=8 with 4 workers) for the GPU backends
#  4. Docker image: bundle -> bazel tarball -> push
set -uo pipefail
REPO=/data/yolo-master-edge; OUT=/data/results/api_bench; cd $REPO
log() { echo "[follow $(date +%H:%M:%S)] $*"; }
log "parity rescore"
for c in $OUT/*__*; do
  [ -d $c/api_txt ] || continue
  p=$(python3 scripts/server/parity_txt.py $c/cli_txt $c/api_txt --tol 0.01)
  grep -v "^parity:" $c/score.txt > $c/score.tmp; echo "parity: $p" >> $c/score.tmp; mv $c/score.tmp $c/score.txt
done
log "pruned ncnn rerun with the dense export"
rm -rf /data/models_api/v01n-pruned/ncnn && cp -r /data/xfer/models/api/v01n-pruned_ncnn /data/models_api/v01n-pruned/ncnn && rm -f /data/models_api/v01n-pruned/ncnn/*.npz
mv $OUT/v01n-pruned-fp32__ncnn-cpu $OUT/../api_bench_pruned_ncnn_oldexport
CELLS="v01n-pruned-fp32:ncnn-cpu" bash scripts/server/run_comparison.sh $OUT 2>&1 | grep -E "score|cli:|api:|parity" 
log "GPU worker scaling (c=8, 4 workers)"
API_WORKERS_GPU=4 PATHS=c8 CELLS="v01n-fp32:trt v01n-fp16:trt v01n-pruned-fp32:trt esmoen-fp32:trt esmoen-fp16:trt v01n-fp32:ort-cuda v01n-fp16:ort-cuda v01n-pruned-fp32:ort-cuda esmoen-fp32:ort-cuda esmoen-fp16:ort-cuda v01n-fp32:mnn-cuda v01n-pruned-fp32:mnn-cuda esmoen-fp32:mnn-cuda" \
  bash scripts/server/run_comparison.sh /data/results/api_bench_gpu4 2>&1 | grep -E "^\[cmp" | tail -3
python3 scripts/server/summarize_comparison.py $OUT > /dev/null
log "docker: stage + bazel tarball + push"
bash deploy/docker/stage_bundle.sh cpp/build_l40s /data/models_api 2>&1 | tail -2
DOCKERHUB_USER=skywalker0501 bash deploy/docker/push.sh both 2>&1 | grep -vE "^\s*$" | tail -15
log "done"
