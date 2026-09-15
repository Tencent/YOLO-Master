#!/usr/bin/env bash
# Score a pulled OrtDumpTest tree (results/int8_bench/device/run_s26_dump.sh) on the pod: one
# mAP line per <model>/<runtime>-<unit>-<precision> row via scripts/eval_map.py, class names from
# the model dir's metadata.yaml, first 200 images (the smoke set, sorted as the test sorts them).
#
#   scripts/score_device_dumps.sh <pulled dump dir> <images dir> <labels dir> [--only <substr>] [--limit N]
#   e.g. scripts/score_device_dumps.sh results/int8_bench/s26_dump \
#            /tmp/.../cert416/imgs /data/datasets/coco/labels/val2017
#        scripts/score_device_dumps.sh results/int8_bench/s26_dump <visdrone imgs> \
#            /data/datasets/VisDrone/labels/val --only esmoe
#
# The dump layout is <dump>/<modelId>/<row>/<stem>.txt with modelId = model dir name minus `_ncnn`
# (v0.1-seg-n, v0.1-n, esmoe_n_visdrone, yolo11n), so the names come from models/<modelId>_ncnn/
# metadata.yaml. Rows of models whose images are not the given set (COCO rows scored against
# VisDrone labels or vice versa) are meaningless: use --only to pick one domain per call.
# Gate (tests/certify_ncnn_int8.py rule): every ONNX/NPU row within 1.0 pt mAP50-95 of the same
# model's ncnn/CPU and ONNX/CPU rows. Interpreter: the yolo_master conda env (ultralytics DetMetrics).
set -euo pipefail
DUMP=${1:?pulled dump dir}; IMAGES=${2:?images dir}; LABELS=${3:?labels dir}; shift 3
ONLY=""; LIMIT=200
while [ $# -gt 0 ]; do
  case "$1" in
    --only) ONLY="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY=${PY:-/root/anaconda3/envs/yolo_master/bin/python}
[ -x "$PY" ] || PY=python3

printf '%-22s %-22s %6s  %s\n' model row images "mAP50 / mAP50-95"
for mdir in "$DUMP"/*/; do
  [ -d "$mdir" ] || continue
  model=$(basename "$mdir")
  [ -z "$ONLY" ] || [[ "$model" == *"$ONLY"* ]] || continue
  meta="$REPO/models/${model}_ncnn/metadata.yaml"
  [ -f "$meta" ] || { echo "$model: no $meta (names unknown), skipped" >&2; continue; }
  for rdir in "$mdir"*/; do
    [ -d "$rdir" ] || continue
    row=$(basename "$rdir")
    n=$(ls "$rdir"/*.txt 2>/dev/null | wc -l)
    out=$("$PY" "$REPO/scripts/eval_map.py" --preds "$rdir" --images "$IMAGES" --labels "$LABELS" \
          --names-yaml "$meta" --limit "$LIMIT" 2>/dev/null | grep -E "^images=" || echo "images=0 mAP50=ERR mAP50-95=ERR")
    printf '%-22s %-22s %6d  %s\n' "$model" "$row" "$n" "${out#images=* }"
  done
done
