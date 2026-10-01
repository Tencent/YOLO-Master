#!/usr/bin/env bash
# Build ncnn tag 20260526 from source on THIS x86-64 host, with the quantization tools.
#
# Why: the vendored third_party/ncnn is an Ubuntu-22.04 prebuilt (needs GLIBCXX_3.4.29 / GLIBC_2.34),
# so on a 20.04 host neither libncnn.so nor bin/ncnn2table / bin/ncnn2int8 can run. This mirrors
# jetson/24_build_ncnn.sh (same tag, so .param/.bin stay byte-parity with the Android SDK) with three
# deltas: x86, NCNN_BUILD_TOOLS=ON (ncnn2table / ncnn2int8 / ncnnoptimize), and cmake>=3.17 via pip
# when the host cmake is older. NCNN_VULKAN is enabled only when Vulkan headers exist (the tools and
# the CPU runtime do not need it). Never touches third_party/ncnn (the Android-SDK parity reference).
#
# Usage:  scripts/build_ncnn_x86.sh            (installs to third_party/ncnn-x86-20260526)
#         OPENCV_DIR=/path/to/cmake/opencv4 scripts/build_ncnn_x86.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VER=20260526
SRC="$ROOT/third_party/ncnn-src"
DST="$ROOT/third_party/ncnn-x86-$VER"
VENV="${VENV:-/data/tmp/venv-v2608}"
OPENCV_DIR="${OPENCV_DIR:-/usr/lib/x86_64-linux-gnu/cmake/opencv4}"

[ -f "$OPENCV_DIR/OpenCVConfig.cmake" ] || {
  echo "ERROR: no OpenCV cmake config at $OPENCV_DIR (ncnn2table needs OpenCV). apt-get install -y libopencv-dev, or set OPENCV_DIR."; exit 1; }

# ---- cmake >= 3.17 (glslang / modern ncnn CMake) ---------------------------------------------
CMAKE=cmake
ver="$(cmake --version | head -1 | awk '{print $3}')"
if [ "$(printf '%s\n' 3.17 "$ver" | sort -V | head -1)" != "3.17" ]; then
  echo "== host cmake $ver < 3.17: installing cmake+ninja into $VENV =="
  "$VENV/bin/pip" install -q "cmake>=3.17" ninja
  CMAKE="$VENV/bin/cmake"
fi
echo "== using $($CMAKE --version | head -1) =="

# ---- source ------------------------------------------------------------------------------------
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  echo "== fetching Tencent/ncnn tag $VER =="
  rm -rf "$SRC"
  # Anonymous HTTPS clones get rate-limited / credential-prompted from datacenter IPs. SSH first
  # (public repo; any GitHub key works), then the release tarball (no auth, no submodules - fine
  # because the glslang submodule is only needed for NCNN_VULKAN=ON).
  if ! GIT_TERMINAL_PROMPT=0 git clone --depth 1 --branch "$VER" git@github.com:Tencent/ncnn.git "$SRC" 2>&1; then
    echo "   ssh clone failed; fetching release tarball"
    mkdir -p "$SRC"
    curl -fsSL "https://github.com/Tencent/ncnn/archive/refs/tags/$VER.tar.gz" | tar xz -C "$SRC" --strip-components=1
  fi
fi
VK=OFF
if [ -f /usr/include/vulkan/vulkan.h ] && [ -d "$SRC/.git" ]; then
  VK=ON; (cd "$SRC" && git submodule update --init --depth 1 glslang)
fi
echo "== NCNN_VULKAN=$VK  OpenCV_DIR=$OPENCV_DIR  prefix=$DST =="

# ---- configure / build / install ----------------------------------------------------------------
"$CMAKE" -S "$SRC" -B "$SRC/build-x86" -DCMAKE_BUILD_TYPE=Release \
  -DNCNN_VULKAN="$VK" -DNCNN_SHARED_LIB=ON \
  -DNCNN_BUILD_TOOLS=ON -DNCNN_BUILD_BENCHMARK=ON \
  -DNCNN_BUILD_EXAMPLES=OFF -DNCNN_BUILD_TESTS=OFF -DNCNN_PYTHON=OFF \
  -DOpenCV_DIR="$OPENCV_DIR" -DCMAKE_INSTALL_PREFIX="$DST"
"$CMAKE" --build "$SRC/build-x86" -j"$(nproc)"
"$CMAKE" --install "$SRC/build-x86"

# The quantize tools are not part of `cmake --install` on every tag: copy whatever was built.
mkdir -p "$DST/bin"
for t in tools/quantize/ncnn2table tools/quantize/ncnn2int8 tools/ncnnoptimize tools/ncnnmerge benchmark/benchncnn; do
  [ -x "$SRC/build-x86/$t" ] && cp -v "$SRC/build-x86/$t" "$DST/bin/"
done

# ---- provenance + assertions ---------------------------------------------------------------------
COMMIT="$(git -C "$SRC" rev-parse HEAD 2>/dev/null || echo tarball)"
echo "tag=$VER commit=$COMMIT vulkan=$VK opencv=$OPENCV_DIR host=$(uname -r) built=$(date -u +%FT%TZ)" > "$DST/VERSION.txt"
test -f "$DST/include/ncnn/net.h"   || { echo "ERROR: net.h missing"; exit 1; }
test -f "$DST/lib/libncnn.so"       || { echo "ERROR: libncnn.so missing"; exit 1; }
test -x "$DST/bin/ncnn2table"       || { echo "ERROR: ncnn2table was not built (OpenCV not found by ncnn?)"; exit 1; }
test -x "$DST/bin/ncnn2int8"        || { echo "ERROR: ncnn2int8 was not built"; exit 1; }
# Must print usage, not a loader error, on this host:
"$DST/bin/ncnn2int8" 2>&1 | head -2 || true
echo "== ncnn x86 $VER installed to $DST =="; cat "$DST/VERSION.txt"
