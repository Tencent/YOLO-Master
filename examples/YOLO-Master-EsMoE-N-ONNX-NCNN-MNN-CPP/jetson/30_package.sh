#!/usr/bin/env bash
# Package the native TensorRT runner into a portable bundle for Jetson Orin / JetPack 7.
# Bundles OpenCV (+ any non-system deps) with an $ORIGIN/lib rpath. DEPENDS on JetPack's
# TensorRT + CUDA, which are present (version-matched) on every JetPack 7 device -> not bundled
# (keeps it small + robust). The .engine is device-specific, so we ship the .onnx + build_engine.sh
# (builds a clean FP16 engine on first setup). Runs on any Orin (Nano/NX/AGX, sm87) on JetPack 7.
set -e
cd "$(dirname "$0")"; ROOT="$(cd .. && pwd)"
BIN="$ROOT/cpp/build_trt/yolomaster_edge"
ONNX="${ONNX:-$ROOT/jetson/models/esmoe_n_visdrone_sim.onnx}"
[ -x "$BIN" ] || { echo "build first:  bash jetson/21_build_trt_runner.sh"; exit 1; }
command -v patchelf >/dev/null 2>&1 || sudo apt install -y patchelf

OUT="$ROOT/dist/yolomaster_edge-jetson-orin-jp7"
rm -rf "$OUT"; mkdir -p "$OUT/lib" "$OUT/models"
cp "$BIN" "$OUT/yolomaster_edge"

echo "== bundling non-JetPack libs (depend on JetPack TRT/CUDA + base system) =="
ldd "$BIN" | awk '/=> \//{print $3}' | sort -u | while read -r lib; do
  base=$(basename "$lib")
  case "$base" in
    # JetPack (TensorRT + CUDA + Jetson driver stack) — present on every JP7 device
    libnvinfer*|libnvonnxparser*|libnvparsers*|libcudart*|libcuda.*|libcublas*|libcudnn*|\
    libcufft*|libcurand*|libcusparse*|libcusolver*|libnpp*|libnv*|libcupti*)
      echo "  [jetpack] $base" ;;
    # base system (Ubuntu 24.04 aarch64) — present everywhere
    ld-linux*|libc.so*|libm.so*|libdl.so*|libpthread*|librt.so*|libresolv*|libstdc++*|\
    libgcc_s*|libgomp*|libz.so*)
      echo "  [system]  $base" ;;
    # everything else (OpenCV + odd deps) — bundle it
    *) cp -v "$lib" "$OUT/lib/" ;;
  esac
done
patchelf --set-rpath '$ORIGIN/lib' "$OUT/yolomaster_edge"

cp "$ONNX" "$OUT/models/" 2>/dev/null || echo "  [warn] no .onnx at $ONNX — add it to models/ before shipping"

cat > "$OUT/build_engine.sh" <<'EOS'
#!/usr/bin/env bash
# Build the FP16 TensorRT engine on THIS Jetson (engines are device + TRT-version specific).
# OPT=3 sidesteps the KTM FP16 build bug on Orin/TRT10; swap covers the 4GB Nano.
set -e; cd "$(dirname "$0")"
TRTEXEC=$(find /usr -name trtexec -type f 2>/dev/null | head -1)
[ -n "$TRTEXEC" ] || { echo "trtexec not found — install: sudo apt install nvidia-jetpack"; exit 1; }
if ! swapon --show | grep -q .; then
  echo "adding 8G swap for the build (remove later with: sudo swapoff /swapfile && sudo rm /swapfile)"
  sudo fallocate -l 8G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
fi
"$TRTEXEC" --onnx=models/esmoe_n_visdrone_sim.onnx --fp16 \
  --saveEngine=models/esmoe_n_fp16.engine \
  --memPoolSize=workspace:256 --builderOptimizationLevel=3 --maxAuxStreams=0
echo "engine -> models/esmoe_n_fp16.engine"
echo "run:  ./yolomaster_edge --model models/esmoe_n_fp16.engine --source <img|dir> --classes visdrone --out out"
EOS
chmod +x "$OUT/build_engine.sh"

cat > "$OUT/README.md" <<'EOS'
# YOLO-Master-EsMoE-N — Jetson Orin (JetPack 7) native TensorRT runner

Prebuilt aarch64 GPU runner. Runs on any **Jetson Orin** (Nano / NX / AGX, sm87) on **JetPack 7**
(CUDA 13 / TensorRT 10). Depends on JetPack's TensorRT + CUDA (already installed); OpenCV is bundled.

## 1. Build the engine — once per device (engines are device-specific)
    ./build_engine.sh
    # ~10-15 min; writes models/esmoe_n_fp16.engine (FP16). Adds an 8G swapfile if none exists.

## 2. Run on the GPU
    ./yolomaster_edge --model models/esmoe_n_fp16.engine --source <image|dir> \
        --classes visdrone --conf 0.25 --out out

Validated on Orin Nano 4GB: 35.7 FPS, 0.3488 mAP50 (VisDrone val, -0.46% vs FP32).
EOS

tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
echo "== bundle ready =="; du -sh "$OUT.tar.gz"; echo "$OUT.tar.gz"
echo "sanity:  cd $OUT && ldd ./yolomaster_edge | grep -i 'not found' || echo 'all deps resolve'"
