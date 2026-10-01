#!/usr/bin/env bash
# Stage the relocatable /opt/yolomaster bundle and pack it as deploy/docker/bundle.tar (one image layer).
#   bash deploy/docker/stage_bundle.sh [BUILD_DIR=cpp/build_l40s] [MODELS=/data/models_api]
# Contents: bin/{yolomaster_server,yolomaster_edge}, lib/ (ORT-GPU, ncnn, MNN, OpenCV-lean, TensorRT,
# ffmpeg 4.4 for videoio, and every non-base shared lib ldd resolves), models/<id>/..., server.json,
# entrypoint.sh, LICENSES/. cuDNN, cuBLAS and cudart come from the nvidia/cuda base image.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
SERVER_VERSION="$(tr -d '[:space:]' < "$REPO/cpp/server/SERVER_VERSION")"
echo "SERVER_VERSION = \"$SERVER_VERSION\"" > "$(dirname "$0")/version.bzl"   # consumed by BUILD.bazel (gitignored)
BUILD="${1:-$REPO/cpp/build_l40s}"
MODELS="${2:-/data/models_api}"
STAGE="$REPO/deploy/docker/stage"; ROOT="$STAGE/opt/yolomaster"
rm -rf "$STAGE"; mkdir -p "$ROOT/bin" "$ROOT/lib" "$ROOT/models" "$ROOT/LICENSES" "$ROOT/cache/trt"
cp "$BUILD/server/yolomaster_server" "$BUILD/yolomaster_edge" "$ROOT/bin/"
cp "$REPO/deploy/docker/entrypoint.sh" "$ROOT/"; cp "$REPO/deploy/server.json" "$ROOT/server.json"
cp "$REPO/docs/API.md" "$ROOT/API.md"
# ---- shared-library closure: everything ldd resolves outside the base image's own set ----
BASE_KEEP='^(linux-vdso|ld-linux|libc\.so|libm\.so|libdl\.so|libpthread\.so|librt\.so|libgcc_s|libstdc\+\+|libresolv|libnss_|libutil\.so|libcuda\.so|libnvidia|libcudart|libcublas|libcublasLt|libcudnn|libnvrtc|libcufft|libcurand|libcusolver|libcusparse|libnvJitLink)'
closure() {
  ldd "$1" 2>/dev/null | awk '/=>/ {print $3} /^\s*\// {print $1}' | sort -u | while read -r so; do
    [ -f "$so" ] || continue
    bn="$(basename "$so")"
    echo "$bn" | grep -Eq "$BASE_KEEP" && continue
    [ -e "$ROOT/lib/$bn" ] || cp -L "$so" "$ROOT/lib/$bn"
  done
}
closure "$ROOT/bin/yolomaster_server"; closure "$ROOT/bin/yolomaster_edge"
# ORT loads its EP libraries with dlopen: copy them explicitly, and the TensorRT builder resource
ORT_LIB="$(dirname "$(ldd "$ROOT/bin/yolomaster_server" | awk '/libonnxruntime\.so/ {print $3}')")"
cp -L "$ORT_LIB"/libonnxruntime_providers_*.so "$ROOT/lib/" 2>/dev/null || true
for f in /usr/lib/x86_64-linux-gnu/libnvinfer_plugin.so.10 /usr/lib/x86_64-linux-gnu/libnvonnxparser.so.10; do
  [ -e "$f" ] && cp -L "$f" "$ROOT/lib/$(basename "$f")"
done
# TensorRT 10.16 builder resources are per SM (libnvinfer_builder_resource_sm89.so.10.x, 165-670 MB
# each) and are only needed to BUILD engines (first start). Default set: Ampere/Ada/Hopper data-center
# and workstation parts; override with YM_TRT_SMS="80 86 89 90 ptx" (ptx = any other GPU, 490 MB).
for sm in ${YM_TRT_SMS:-80 86 89 90}; do
  for f in /usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource_${sm}.so.10* /usr/lib/x86_64-linux-gnu/libnvinfer_builder_resource_sm${sm}.so.10*; do
    [ -e "$f" ] && cp -L "$f" "$ROOT/lib/$(basename "$f")"
  done
done
for so in "$ROOT"/lib/*.so*; do closure "$so"; done   # second-level deps (ffmpeg, x264, ...)
for so in "$ROOT"/lib/*.so*; do closure "$so"; done   # third level
# ---- rpaths: binaries -> ../lib, libs -> $ORIGIN ----
for b in "$ROOT"/bin/*; do patchelf --set-rpath '$ORIGIN/../lib' "$b"; done
for so in "$ROOT"/lib/*.so*; do patchelf --set-rpath '$ORIGIN' "$so" 2>/dev/null || true; done
# ---- models: only the files the server can load ----
for id in "$MODELS"/*/; do
  id="$(basename "$id")"; [ "$id" = ".trt_cache" ] && continue
  mkdir -p "$ROOT/models/$id"
  for f in model.onnx model-fp16.onnx model.mnn model-fp16.mnn metadata.yaml model.metadata.yaml model-fp16.metadata.yaml; do
    [ -f "$MODELS/$id/$f" ] && cp "$MODELS/$id/$f" "$ROOT/models/$id/$f"
  done
  [ -d "$MODELS/$id/ncnn" ] && { mkdir -p "$ROOT/models/$id/ncnn"; cp "$MODELS/$id/ncnn"/model.ncnn.* "$MODELS/$id/ncnn/metadata.yaml" "$ROOT/models/$id/ncnn/"; }
done
# ---- licenses ----
cp "$REPO/LICENSE" "$ROOT/LICENSES/yolo-master-edge.txt" 2>/dev/null || true
cp "$REPO/third_party/uWebSockets/LICENSE" "$ROOT/LICENSES/uWebSockets.txt" 2>/dev/null || true
cp "$REPO/cpp/third_party/json.LICENSE.MIT" "$ROOT/LICENSES/nlohmann-json.txt" 2>/dev/null || true
cp "$ORT_LIB/../LICENSE" "$ROOT/LICENSES/onnxruntime.txt" 2>/dev/null || true
cp "$REPO/third_party/ncnn-20260526-ubuntu-2204-shared/LICENSE" "$ROOT/LICENSES/ncnn.txt" 2>/dev/null || true
cp "$REPO/third_party/mnn-src/LICENSE.txt" "$ROOT/LICENSES/MNN.txt" 2>/dev/null || true
cp /usr/share/doc/libnvinfer10/copyright "$ROOT/LICENSES/TensorRT.txt" 2>/dev/null || true
# ---- sanity: the staged server must start with a clean env ----
env -i "$ROOT/bin/yolomaster_server" --help >/dev/null
# ---- pack ----
tar --owner=0 --group=0 --numeric-owner -cf "$REPO/deploy/docker/bundle.tar" -C "$STAGE" opt
du -sh "$ROOT/lib" "$ROOT/models" "$REPO/deploy/docker/bundle.tar"
ls "$ROOT/lib" | tr '\n' ' '; echo
