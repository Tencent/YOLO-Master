#!/usr/bin/env bash
# Assemble the API model layout on the pod from the rsynced artifacts and convert MNN files.
#   /data/models_api/<id>/{model.onnx, model-fp16.onnx, model.mnn, model-fp16.mnn, ncnn/, metadata.yaml}
# ids: v01n (v0.1-N COCO, A3 export), v01n-pruned (MoEPruner t0.10 surgery export), esmoen (EsMoE-N COCO).
set -euo pipefail
REPO="${REPO:-/data/yolo-master-edge}"
SRC="${SRC:-/data/xfer/models}"
OUT="${OUT:-/data/models_api}"
MNNCONVERT="${MNNCONVERT:-$REPO/third_party/mnn-src/build/MNNConvert}"
log() { echo "[prepare $(date +%H:%M:%S)] $*"; }
[ -x "$MNNCONVERT" ] || { log "MNNConvert not built: $MNNCONVERT"; exit 2; }
mkdir -p "$OUT"
for id in v01n v01n-pruned esmoen; do
  d="$OUT/$id"; mkdir -p "$d"
  cp -u "$SRC/api/$id/model.onnx" "$SRC/api/$id/model-fp16.onnx" "$SRC/api/$id/metadata.yaml" "$d/"
  # multi-path sidecar (fp32 routing segments for MNN CUDA fp16), one per ONNX; MnnBackend looks for
  # "<model>.mnn.paths.json", "<stem>.paths.json" or "model.paths.json" next to the .mnn
  [ -f "$SRC/api/$id/model.paths.json" ] && cp -u "$SRC/api/$id/model.paths.json" "$d/model.paths.json"
  # sidecars for the fp16 onnx and the engines the server will build
  cp -u "$d/metadata.yaml" "$d/model.metadata.yaml"; cp -u "$d/metadata.yaml" "$d/model-fp16.metadata.yaml"
  # MNN: fp32 and fp16-weight variants from the same fp32 ONNX
  if [ ! -f "$d/model.mnn" ]; then
    log "MNNConvert $id fp32"
    "$MNNCONVERT" -f ONNX --modelFile "$d/model.onnx" --MNNModel "$d/model.mnn" --bizCode yolomaster --keepInputFormat=1 > "$d/mnnconvert.log" 2>&1 || { tail -5 "$d/mnnconvert.log"; log "MNN fp32 conversion FAILED for $id"; }
  fi
  if [ ! -f "$d/model-fp16.mnn" ]; then
    log "MNNConvert $id fp16"
    "$MNNCONVERT" -f ONNX --modelFile "$d/model.onnx" --MNNModel "$d/model-fp16.mnn" --bizCode yolomaster --keepInputFormat=1 --fp16 > "$d/mnnconvert-fp16.log" 2>&1 || { tail -5 "$d/mnnconvert-fp16.log"; log "MNN fp16 conversion FAILED for $id"; }
  fi
  # C2 candidate: the routing-protected fp16 ONNX (Cast nodes keep the router fp32) converted as is
  if [ ! -f "$d/model-fp16-routed.mnn" ]; then
    log "MNNConvert $id fp16-routed"
    "$MNNCONVERT" -f ONNX --modelFile "$d/model-fp16.onnx" --MNNModel "$d/model-fp16-routed.mnn" --bizCode yolomaster --keepInputFormat=1 > "$d/mnnconvert-fp16-routed.log" 2>&1 || { tail -5 "$d/mnnconvert-fp16-routed.log"; log "MNN fp16-routed conversion FAILED for $id"; }
  fi
  # the routed fp16 conversion protects its routing inside the graph (Cast nodes), so its sidecar
  # says fp16_safe: true and the CUDA fp16 request is honoured (that is the C2 experiment)
  [ -f "$d/model-fp16-routed.mnn" ] && { grep -v "^fp16_safe:" "$d/metadata.yaml" > "$d/model-fp16-routed.metadata.yaml"; echo "fp16_safe: true" >> "$d/model-fp16-routed.metadata.yaml"; }
  [ -f "$d/model.mnn" ] && cp -u "$d/metadata.yaml" "$d/model.metadata.yaml"
  [ -f "$d/model-fp16.mnn" ] && cp -u "$d/metadata.yaml" "$d/model-fp16.metadata.yaml"
done
# ncnn dirs: v0.1-N (shipped SDPA export), pruned (p03), EsMoE-N (dense export from prepare step)
rm -rf "$OUT/v01n/ncnn" "$OUT/v01n-pruned/ncnn" "$OUT/esmoen/ncnn"
cp -r "$SRC/v0.1-n_ncnn" "$OUT/v01n/ncnn"
cp -r "$SRC/api/v01n-pruned_ncnn" "$OUT/v01n-pruned/ncnn"   # dense (SDPA) export of the pruned checkpoint, same rewrite as v01n
cp -r "$SRC/api/esmoen_ncnn" "$OUT/esmoen/ncnn"
rm -f "$OUT"/*/ncnn/*.onnx "$OUT"/*/ncnn/*.npz
log "layout:"; find "$OUT" -maxdepth 2 | sort | sed 's#^#  #'
du -sh "$OUT"
