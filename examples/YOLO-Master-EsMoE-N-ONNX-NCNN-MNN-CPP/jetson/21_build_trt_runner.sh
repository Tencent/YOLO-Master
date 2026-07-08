#!/usr/bin/env bash
# Build the C++ runner WITH the TensorRT backend -> real GPU inference from a .engine.
# TRT/CUDA come from JetPack (no SDK download). The engine is the GPU path, so ORT/ncnn are off.
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"
cd "$ROOT/cpp"
rm -rf build_trt && mkdir build_trt && cd build_trt

cmake .. -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON \
         -DUSE_ORT=OFF -DUSE_NCNN=OFF -DUSE_TRT=ON 2>&1 | grep -iE "backend:|Found OpenCV|error" || true
make -j"$(nproc)" 2>&1 | grep -iE "error|Built target" | tail -2

BIN="$ROOT/cpp/build_trt/yolomaster_edge"
ENG="$ROOT/jetson/engines/esmoe_n_fp16.engine"
echo
echo "built: $BIN"
echo
echo "== run on the GPU =="
echo "  $BIN --model $ENG --source <image_or_dir> --classes visdrone --conf 0.25 --out out"
echo
echo "== dump preds for on-device mAP (then scp preds/ to the server and run scripts/eval_map.py) =="
echo "  $BIN --model $ENG --source <val_images_dir> --classes visdrone \\"
echo "       --conf 0.001 --iou 0.7 --multi-label --save-txt preds --no-save --quiet"
echo "  (.engine has no embedded class names, so pass --classes visdrone)"
