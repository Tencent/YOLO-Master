#!/usr/bin/env bash
# Package a locally built TensorRT runner for Jetson Orin / JetPack.
# Bundles OpenCV (+ any non-system deps) with an $ORIGIN/lib rpath. DEPENDS on JetPack's
# TensorRT + CUDA, which must be supplied by the version-matched target JetPack -> not bundled
# (keeps it small + robust). The .engine is device-specific, so we ship the .onnx + build_engine.sh
# (builds a device-specific FP16 engine on first setup). Verify compatibility on the target device.
set -e
cd "$(dirname "$0")"; ROOT="$(cd .. && pwd)"
VERSION="${1:-1.1.0}"
BIN="$ROOT/cpp/build_trt/yolomaster_edge"
ONNX="${ONNX:-$ROOT/jetson/models/esmoe_n_visdrone_sim.onnx}"
MODEL_BASENAME="$(basename -- "$ONNX")"
case "$MODEL_BASENAME" in
  *.onnx) ;;
  *) echo "ONNX must point to a .onnx file: $ONNX" >&2; exit 2 ;;
esac
MODEL_STEM="${MODEL_BASENAME%.onnx}"
[ -n "$MODEL_STEM" ] || { echo "ONNX filename has no model stem: $ONNX" >&2; exit 2; }
ENGINE_BASENAME="${MODEL_STEM}_fp16.engine"
[ -x "$BIN" ] || { echo "build first:  bash jetson/21_build_trt_runner.sh"; exit 1; }
command -v patchelf >/dev/null 2>&1 || sudo apt install -y patchelf

OUT="$ROOT/dist/yolomaster-edge-jetson-orin-$VERSION"
rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/models"
cp "$BIN" "$OUT/yolomaster_edge"

echo "== bundling non-JetPack libs (depend on JetPack TRT/CUDA + base system) =="
ldd "$BIN" | awk '/=> \//{print $3}' | sort -u | while read -r lib; do
  base=$(basename "$lib")
  case "$base" in
    # JetPack (TensorRT + CUDA + Jetson driver stack) — supplied by the target image
    libnvinfer*|libnvonnxparser*|libnvparsers*|libcudart*|libcuda.*|libcublas*|libcudnn*|\
    libcufft*|libcurand*|libcusparse*|libcusolver*|libnpp*|libnv*|libcupti*)
      echo "  [jetpack] $base" ;;
    # base system (JetPack-provided Ubuntu aarch64 userland) — present everywhere
    ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread*|librt.so*|libresolv*|libstdc++*|\
    libgcc_s*|libgomp*|libz.so*)
      echo "  [system]  $base" ;;
    # everything else (OpenCV + odd deps) — bundle it
    *) cp -v "$lib" "$OUT/lib/" ;;
  esac
done
patchelf --set-rpath '$ORIGIN/lib' "$OUT/yolomaster_edge"

if [ -f "$ONNX" ]; then
  cp "$ONNX" "$OUT/models/$MODEL_BASENAME"
else
  echo "  [warn] no .onnx at $ONNX — add it to models/ before shipping"
fi
# metadata sidecar: the runner reads class names + imgsz from metadata.yaml next to the
# engine (v1.1.0), so --classes is no longer needed when this ships.
for MD in "${ONNX%.onnx}.metadata.yaml" "$(dirname "$ONNX")/metadata.yaml"; do
  [ -f "$MD" ] && { cp "$MD" "$OUT/models/metadata.yaml"; break; }
done

{
  printf '%s\n' '#!/usr/bin/env bash'
  printf 'MODEL_FILE=%q\n' "$MODEL_BASENAME"
  printf 'ENGINE_FILE=%q\n' "$ENGINE_BASENAME"
  cat <<'EOS'
# Build the FP16 TensorRT engine on THIS Jetson (engines are device + TRT-version specific).
# OPT=3 is a conservative default for TensorRT 10; tune it on the target and
# record any change in the deployment log. The swap covers the 4GB Nano.
set -e; cd "$(dirname "$0")"
TRTEXEC=$(find /usr -name trtexec -type f 2>/dev/null | head -1)
[ -n "$TRTEXEC" ] || { echo "trtexec not found — install: sudo apt install nvidia-jetpack"; exit 1; }
MODEL_PATH="models/$MODEL_FILE"
ENGINE_PATH="models/$ENGINE_FILE"
[ -f "$MODEL_PATH" ] || { echo "model not found: $MODEL_PATH" >&2; exit 1; }
if ! swapon --show | grep -q .; then
  echo "adding 8G swap for the build (remove later with: sudo swapoff /swapfile && sudo rm /swapfile)"
  sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
fi
"$TRTEXEC" --onnx="$MODEL_PATH" --fp16 \
  --saveEngine="$ENGINE_PATH" \
  --memPoolSize=workspace:256 --builderOptimizationLevel=3 --maxAuxStreams=0
echo "engine -> $ENGINE_PATH"
echo "run:  ./yolomaster_edge --model $ENGINE_PATH --source <img|dir|video> --out out"
EOS
} > "$OUT/build_engine.sh"
chmod +x "$OUT/build_engine.sh"

cat > "$OUT/README.md" <<'EOS'
# YOLO-Master-EsMoE-N — Jetson Orin native TensorRT runner

Locally packaged aarch64 runner. It requires a compatible **Jetson Orin** and
the target's JetPack TensorRT + CUDA installation; OpenCV is bundled when found
by the packaging step. Verify the generated binary and engine on the target.

## 1. Build the engine — once per device (engines are device-specific)
    ./build_engine.sh
    # writes models/<model-stem>_fp16.engine (FP16). Adds an 8G swapfile if none exists.

## 2. Run on the GPU
    ./yolomaster_edge --model models/<model-stem>_fp16.engine --source <image|dir|video> \
        --conf 0.25 --out out

The package contains the selected `.onnx` file under `models/`; use `ls models/*.onnx`
to see its name. The generated build script uses that exact filename, so custom
`ONNX=/path/to/model.onnx` packages remain self-consistent.

Class names + input size come from `models/metadata.yaml` (included when available); `--classes visdrone`
still overrides. Video sources decode through the bundled ffmpeg-based OpenCV.

## 3. v1.1.0 features
    --slicing off|dense|sparse    sliced inference (Sparse SAHI) for small objects
    --tile-size N                 tile edge in source px (0 = model input)
    --slicing-masks               keep global-pass masks in sliced runs (seg engines)
    --cw-nms --sigma S            Cluster-Weighted NMS refinement
    --export-labels DIR           write annotations (YOLO TXT / COCO JSON / Pascal VOC XML)
    --label-format yolo|coco|voc  --sampling all|1s|N (video frame sampling)

Segmentation engines work too: build one from a seg .onnx (e.g. v0.1-seg-n.onnx) with
build_engine.sh and put its metadata.yaml next to it — masks render and label export
emits real polygons.

Performance and accuracy are intentionally left unspecified here. Populate
them only from a fixed validation manifest, per-image predictions, and archived
target-device logs; external Jetson measurements are not valid evidence for
this package.
EOS

tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "== bundle ready =="; du -sh "$OUT.tar.gz"; echo "$OUT.tar.gz"
echo "sanity:  cd $OUT && ldd ./yolomaster_edge | grep -i 'not found' || echo 'all deps resolve'"
