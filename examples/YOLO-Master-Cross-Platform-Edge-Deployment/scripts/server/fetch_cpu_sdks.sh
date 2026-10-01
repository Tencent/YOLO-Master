#!/usr/bin/env bash
# Fetch the CPU inference SDKs a CPU-only build needs (CI and fresh workstations), pinned by
# sha256, into third_party/ (gitignored like every other SDK there). Idempotent.
#   ONNX Runtime 1.18.1 linux-x64 (MIT)      -> third_party/onnxruntime-linux-x64-1.18.1
#   ncnn 20260526 ubuntu-2204 shared (BSD-3) -> third_party/ncnn-20260526-ubuntu-2204-shared
# Usage: scripts/server/fetch_cpu_sdks.sh [third_party dir]
set -euo pipefail
TP="${1:-$(cd "$(dirname "$0")/../.." && pwd)/third_party}"
ORT_VER="${ORT_VER:-1.18.1}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz"
ORT_SHA="${ORT_SHA:-a0994512ec1e1debc00c18bfc7a5f16249f6ebd6a6128ff2034464cc380ea211}"
NCNN_TAG="${NCNN_TAG:-20260526}"
NCNN_URL="https://github.com/Tencent/ncnn/releases/download/${NCNN_TAG}/ncnn-${NCNN_TAG}-ubuntu-2204-shared.zip"
NCNN_SHA="${NCNN_SHA:-69174c845eaf0e7b592f1e032b700d1b0ffda2915ebf69ee98b2d87411578d30}"
mkdir -p "$TP"
fetch() {  # <url> <sha256> <dest file>
  local url="$1" sha="$2" out="$3"
  if [ -f "$out" ] && echo "$sha  $out" | sha256sum -c --quiet 2>/dev/null; then return 0; fi
  echo "downloading $(basename "$url")"
  curl -sSL --retry 3 -o "$out.part" "$url" && mv "$out.part" "$out"
  echo "$sha  $out" | sha256sum -c --quiet || { echo "sha256 mismatch for $url" >&2; rm -f "$out"; exit 3; }
}
ORT_DIR="$TP/onnxruntime-linux-x64-${ORT_VER}"
if [ ! -f "$ORT_DIR/lib/libonnxruntime.so" ]; then
  fetch "$ORT_URL" "$ORT_SHA" "$TP/onnxruntime-linux-x64-${ORT_VER}.tgz"
  tar xzf "$TP/onnxruntime-linux-x64-${ORT_VER}.tgz" -C "$TP"
fi
NCNN_DIR="$TP/ncnn-${NCNN_TAG}-ubuntu-2204-shared"
if [ ! -f "$NCNN_DIR/lib/libncnn.so" ] && [ ! -d "$NCNN_DIR/lib" ]; then
  fetch "$NCNN_URL" "$NCNN_SHA" "$TP/ncnn-${NCNN_TAG}-ubuntu-2204-shared.zip"
  (cd "$TP" && unzip -qo "ncnn-${NCNN_TAG}-ubuntu-2204-shared.zip")
fi
echo "ONNXRUNTIME_ROOT=$ORT_DIR"
echo "NCNN_ROOT=$NCNN_DIR"
