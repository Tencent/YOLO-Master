#!/usr/bin/env bash
# Build MNN (CPU) from source on the Jetson, matching the x86 SDK version (3.6.1) so
# converted .mnn files behave identically across platforms. Installs nothing: the
# cpp/CMakeLists MNN_ROOT convention is <root>/include + <root>/build/libMNN.so, which
# is exactly what an in-tree build produces. One-time (~15 min on an Orin), cached.
# 20_build_runner.sh picks the build up automatically.
set -e
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

VER=3.6.1
SRC="$ROOT/third_party/mnn-src-aarch64"

if [ -f "$SRC/build/libMNN.so" ]; then
  echo "MNN aarch64 already built: $SRC/build/libMNN.so"
  exit 0
fi

echo "==================== MNN source (tag $VER) ===================="
if [ ! -d "$SRC/.git" ]; then
  git clone --depth 1 --branch "$VER" https://github.com/alibaba/MNN.git "$SRC"
fi

echo "==================== build (CPU, shared) ===================="
cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release \
  -DMNN_BUILD_SHARED_LIBS=ON -DMNN_SEP_BUILD=OFF \
  -DMNN_BUILD_CONVERTER=OFF -DMNN_BUILD_TOOLS=OFF -DMNN_BUILD_DEMO=OFF \
  -DMNN_BUILD_QUANTOOLS=OFF -DMNN_BUILD_TEST=OFF
cmake --build "$SRC/build" -j"$(nproc)"

[ -f "$SRC/build/libMNN.so" ] || { echo "ERROR: libMNN.so not produced"; exit 1; }
echo "  built: $SRC/build/libMNN.so"
echo "  next: bash jetson/20_build_runner.sh   (auto-detects this build)"
