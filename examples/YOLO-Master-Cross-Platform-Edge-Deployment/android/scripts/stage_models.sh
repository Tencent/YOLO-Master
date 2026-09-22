#!/usr/bin/env bash
# Stage ncnn model dirs from the repo's (gitignored) models/ into a module's assets so the
# on-device harness (:runtime) or the app (:app) can load them.
#
#   scripts/stage_models.sh                 # -> android/runtime/src/main/assets/models (harness)
#   scripts/stage_models.sh --module app    # -> android/app/src/main/assets/models     (the app)
#
# Models are NOT committed (they ship as GitHub Release assets); this copies your local
# working-tree copies into assets/. Only the runtime files are copied: the three ncnn files
# (param, bin, metadata.yaml) plus, when present, the ONNX Runtime files that live in the same
# dir (model.onnx, model-a16w8.onnx, model-a8w8.onnx). Quantizer leftovers (quant/,
# quant_manifest.json, __pycache__/, model_ncnn.py) must never land in an APK.
set -euo pipefail

MODULE=runtime
while [ $# -gt 0 ]; do
  case "$1" in
    --module) MODULE="$2"; shift 2 ;;
    -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done
case "$MODULE" in runtime|app) ;; *) echo "--module must be runtime or app" >&2; exit 2 ;; esac

REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ASSETS="$REPO/android/$MODULE/src/main/assets"
DST="$ASSETS/models"
mkdir -p "$DST"

# name-in-assets  <-  path-under-repo/models   (float models)
STAGE=(
  "v0.1-seg-n_ncnn:v0.1-seg-n_ncnn"
  "v0.1-seg-n-416_ncnn:v0.1-seg-n-416_ncnn"
  "v0.1-n_ncnn:v0.1-n_ncnn"
  "moa-n_ncnn:mixture/moa-n_ncnn"
  "esmoe_n_visdrone_ncnn:esmoe_n_visdrone_ncnn"
  "p03_v01n_ncnn:p03_v01n_ncnn"
  "yolo11n_ncnn:yolo11n_ncnn"
  "yolo11n-seg_ncnn:yolo11n-seg_ncnn"
)
# Pre-quantized mixed-INT8 siblings (scripts/quantize_ncnn_int8.py). The runtime resolves
# Precision.INT8 to "<name>-int8_ncnn" next to the float dir, so they MUST be staged under
# exactly that name. Same three files as a float dir; optional - a missing sibling only skips
# the INT8 rows of the harness (and makes Precision.INT8 fail cleanly with lastError).
STAGE_INT8=(
  "v0.1-seg-n-int8_ncnn:v0.1-seg-n-int8_ncnn"
  "esmoe_n_visdrone-int8_ncnn:esmoe_n_visdrone-int8_ncnn"
  "p03_v01n-int8_ncnn:p03_v01n-int8_ncnn"
  "v0.1-n-int8_ncnn:v0.1-n-int8_ncnn"
)
FILES=(model.ncnn.param model.ncnn.bin metadata.yaml)
# ONNX Runtime siblings (scripts/export_onnx_dense.py / quantize_onnx_qnn.py): optional, copied
# when the source dir has them, so an ncnn-only dir stages exactly as before.
OPTIONAL_FILES=(model.onnx model-a16w8.onnx model-a8w8.onnx)

stage_one() {  # name-in-assets  path-under-repo/models  kind
  local name="$1" rel="$2" kind="$3" src="$REPO/models/$2"
  if [ -d "$src" ]; then
    rm -rf "$DST/$name"; mkdir -p "$DST/$name"
    for f in "${FILES[@]}"; do
      [ -f "$src/$f" ] || { echo "ERROR: $src/$f missing" >&2; exit 1; }
      cp "$src/$f" "$DST/$name/$f"
    done
    local extra=""
    for f in "${OPTIONAL_FILES[@]}"; do
      [ -f "$src/$f" ] && { cp "$src/$f" "$DST/$name/$f"; extra="$extra +$f"; }
    done
    echo "staged  $name  ($kind$extra)"
  elif [ "$kind" = int8 ]; then
    echo "WARN: $src not found - build it with scripts/quantize_ncnn_int8.py (INT8 rows will be skipped)" >&2
  else
    echo "WARN: $src not found - fetch the ncnn models first" >&2
  fi
}
for pair in "${STAGE[@]}";      do stage_one "${pair%%:*}" "${pair##*:}" float; done
for pair in "${STAGE_INT8[@]}"; do stage_one "${pair%%:*}" "${pair##*:}" int8;  done
du -sh "$DST" | sed 's/^/assets: /'

if [ "$MODULE" = runtime ]; then
  PROBE="$ASSETS/probe.jpg"
  [ -f "$PROBE" ] || echo "NOTE: drop a test image at $PROBE (any scene with objects)"
  echo "      and a VisDrone-domain image at $ASSETS/probe_visdrone.jpg (used for esmoe/mixture rows)"
  echo "done. now: cd android && gradle :runtime:connectedAndroidTest"
else
  # The app's measured default (DefaultPolicy.kt: ncnn-CPU vs ONNX-NPU on first selection) runs on
  # the same probe images as the harness; copy them from the runtime module when they are there.
  for p in probe.jpg probe_visdrone.jpg; do
    src="$REPO/android/runtime/src/main/assets/$p"
    if [ -f "$src" ]; then cp "$src" "$ASSETS/$p"; echo "staged  $p  (measured-default probe)"
    else echo "NOTE: $src missing - the app will default to ncnn-CPU unmeasured (drop it there and re-run)"; fi
  done
  echo "done. now: cd android && gradle :app:assembleRelease"
fi
