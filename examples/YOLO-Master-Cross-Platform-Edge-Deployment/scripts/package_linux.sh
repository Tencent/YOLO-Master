#!/usr/bin/env bash
# Assemble a self-contained, relocatable Linux x86_64 bundle of the yolomaster_edge CLI.
#
#   usage: package_linux.sh [cpu|gpu] [version]
#          package_linux.sh cpu 1.1.0   -> dist/yolomaster-edge-linux-x64-1.1.0.tar.gz
#          package_linux.sh gpu 1.1.0   -> dist/yolomaster-edge-linux-x64-gpu_cuda12-1.1.0.tar.gz
#   optional SDK overrides:
#          NCNN_ROOT=/opt/ncnn MNN_ROOT=/opt/mnn package_linux.sh cpu 1.1.0
#
# Bundles carry ONNX Runtime plus every optional backend available in the staging
# tree (ncnn and/or MNN) and full video support. Issue #51 requires at least one
# of ncnn or MNN; the script selects that set automatically instead of requiring
# an SDK the caller did not install. OpenCV comes from a LEAN source build
# (core/imgproc/imgcodecs/videoio,
# ffmpeg on, GStreamer/GDAL/GUI off) that this script builds once into
# third_party/opencv-lean - the stock Ubuntu OpenCV would drag ~237 shared libraries
# (GStreamer/GDAL/MySQL/X11) into the closure. The ffmpeg codec stack is bundled from
# the system.
#
# cpu: ONNX Runtime CPU. GPU still available via ncnn (Vulkan, dlopened - works when
#      the target has a driver, degrades gracefully when not) and MNN (OpenCL).
# gpu: ONNX Runtime GPU (CUDA 12 EP) with the CUDA/cuDNN runtime bundled, so ONNX runs
#      on NVIDIA GPUs with nothing installed on the target but a driver. All 14 CUDA
#      libraries the provider hard-links are REQUIRED on Linux (unlike Windows, where
#      some are lazily loaded); expect a ~3 GB stage. Assembly needs no GPU; the
#      --device cuda smoke test runs only where nvidia-smi reports one.
#
# Bundle layout: <name>/yolomaster_edge + lib/ (full transitive .so closure except the
# glibc/loader core, $ORIGIN-rpath'd via patchelf) + models/ + README.txt.
# Runs on any glibc>=2.35 (Ubuntu 22.04+) x86_64.
set -euo pipefail

VARIANT="${1:-cpu}"
VERSION="${2:-1.1.0}"
case "$VARIANT" in cpu|gpu) ;; *) echo "usage: package_linux.sh [cpu|gpu] [version]"; exit 2;; esac

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ "$VARIANT" = gpu ]; then
  NAME="yolomaster-edge-linux-x64-gpu_cuda12-$VERSION"
  DEFAULT_ORT_ROOT="$ROOT/third_party/onnxruntime-linux-x64-gpu-1.20.1"
else
  NAME="yolomaster-edge-linux-x64-$VERSION"
  DEFAULT_ORT_ROOT="$ROOT/third_party/onnxruntime-linux-x64-1.18.1"
fi
ORT_ROOT="${ORT_ROOT:-$DEFAULT_ORT_ROOT}"
DIST="$ROOT/dist/$NAME"
BUILD="$ROOT/cpp/build_pkg-$VARIANT"
OCV="$ROOT/third_party/opencv-lean"
NCNN_ROOT="${NCNN_ROOT:-$ROOT/third_party/ncnn}"
MNN_ROOT="${MNN_ROOT:-$ROOT/third_party/mnn-src}"

command -v patchelf >/dev/null 2>&1 || {
  echo "ERROR: patchelf is required to create a relocatable bundle." >&2
  echo "       Install it with the host package manager, then rerun this script." >&2
  exit 1
}
[ -d "$ORT_ROOT/lib" ] || { echo "ERROR: ONNX Runtime not found at $ORT_ROOT"; exit 1; }

# Resolve optional secondary backends before configuring CMake.  A release is
# valid with either NCNN or MNN (the Issue #51 requirement); if both are staged,
# both are included.  Header-only trees are not sufficient because CMake needs a
# linkable library as well. Accept both installed SDKs and the uninstalled layouts
# produced by upstream CMake builds (NCNN build/src/, MNN build/{,Release,Debug}/).
# Keep this probe in sync with cpp/CMakeLists.txt so discovery cannot pass while
# configuration subsequently fails.
find_backend_library() {
  local root="$1" name="$2"
  find "$root" -maxdepth 4 \( -type f -o -type l \) \
    \( -name "lib${name}.a" -o -name "lib${name}.so" -o -name "lib${name}.so.*" \) \
    -print -quit 2>/dev/null || true
}
NCNN_LIB_PATH="$(find_backend_library "$NCNN_ROOT" ncnn)"
NCNN_AVAILABLE=0
if [ -f "$NCNN_ROOT/include/ncnn/net.h" ] && [ -n "$NCNN_LIB_PATH" ]; then
  NCNN_AVAILABLE=1
fi
MNN_LIB_PATH="$(find_backend_library "$MNN_ROOT" MNN)"
MNN_AVAILABLE=0
if [ -f "$MNN_ROOT/include/MNN/Interpreter.hpp" ] && [ -n "$MNN_LIB_PATH" ]; then
  MNN_AVAILABLE=1
fi
if [ "$NCNN_AVAILABLE" -eq 0 ] && [ "$MNN_AVAILABLE" -eq 0 ]; then
  echo "ERROR: neither NCNN nor MNN SDK is available; stage one backend and rerun." >&2
  echo "       NCNN_ROOT=$NCNN_ROOT" >&2
  echo "       MNN_ROOT=$MNN_ROOT" >&2
  exit 1
fi
if [ "$NCNN_AVAILABLE" -eq 0 ]; then
  echo "  [warn] NCNN SDK unavailable; packaging MNN only"
fi
if [ "$MNN_AVAILABLE" -eq 0 ]; then
  echo "  [warn] MNN SDK unavailable; packaging NCNN only"
fi
if [ "$NCNN_AVAILABLE" -eq 1 ]; then
  echo "  [ok] NCNN library: $NCNN_LIB_PATH"
fi
if [ "$MNN_AVAILABLE" -eq 1 ]; then
  echo "  [ok] MNN library: $MNN_LIB_PATH"
fi

# ---- 0/6: lean OpenCV (built once, cached) ---------------------------------------
if [ ! -f "$OCV/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
  echo "== 0/6  building lean OpenCV (one-time, ~15 min) =="
  SRC="$ROOT/third_party/opencv-lean-src"
  [ -d "$SRC" ] || git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git "$SRC"
  cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OCV" \
    -DBUILD_LIST=core,imgproc,imgcodecs,videoio -DBUILD_SHARED_LIBS=ON \
    -DWITH_FFMPEG=ON -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_V4L=OFF -DWITH_1394=OFF \
    -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_EIGEN=OFF -DWITH_IPP=ON \
    -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_ZLIB=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=OFF -DBUILD_opencv_python_bindings_generator=OFF >/dev/null
  cmake --build "$SRC/build" -j"$(nproc)"
  cmake --install "$SRC/build" >/dev/null
fi

# ---- 1/6: clean release build -----------------------------------------------------
echo "== 1/6  clean $VARIANT release build (ORT: $(basename "$ORT_ROOT")) =="
rm -rf "$BUILD"
BACKEND_ARGS=(-DREQUIRE_ORT=ON -DALLOW_NO_BACKENDS=OFF)
if [ "$NCNN_AVAILABLE" -eq 1 ]; then
  BACKEND_ARGS+=( -DNCNN_ROOT="$NCNN_ROOT" -DREQUIRE_NCNN=ON )
else
  BACKEND_ARGS+=( -DUSE_NCNN=OFF -DREQUIRE_NCNN=OFF )
fi
if [ "$MNN_AVAILABLE" -eq 1 ]; then
  BACKEND_ARGS+=( -DMNN_ROOT="$MNN_ROOT" -DREQUIRE_MNN=ON )
else
  BACKEND_ARGS+=( -DUSE_MNN=OFF -DREQUIRE_MNN=OFF )
fi
cmake -S "$ROOT/cpp" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT="$ORT_ROOT" "${BACKEND_ARGS[@]}" \
  -DOpenCV_DIR="$OCV/lib/cmake/opencv4"
cmake --build "$BUILD" -j"$(nproc)"

# ---- 2/6: stage binary + .so closure ----------------------------------------------
echo "== 2/6  stage binary + library closure =="
rm -rf "$DIST"; mkdir -p "$DIST/lib" "$DIST/models"
cp "$BUILD/yolomaster_edge" "$DIST/yolomaster_edge"

# glibc / dynamic-loader core: MUST come from the target system, never bundle.
EXCLUDE='libc\.so|libm\.so|libdl\.so|librt\.so|libpthread\.so|ld-linux|libresolv\.so|linux-vdso'
# Walk the complete ELF dependency closure. A single ldd pass misses libraries
# needed by a backend or by a codec library several levels below the executable.
declare -A SEEN_LIBS=()
copy_closure() {
  local object="$1" so base dep
  [ -f "$object" ] || return 0
  while read -r so; do
    [ -n "$so" ] || continue
    base="$(basename "$so")"
    echo "$base" | grep -qE "$EXCLUDE" && continue
    if [ -z "${SEEN_LIBS[$base]+x}" ]; then
      SEEN_LIBS[$base]=1
      cp -L "$so" "$DIST/lib/$base"
      copy_closure "$so"
    fi
  done < <(ldd "$object" 2>/dev/null | awk '/=> \/|^[[:space:]]*\// {for (i=1;i<=NF;i++) if ($i ~ /^\//) {print $i; break}}' | sort -u)
}
copy_closure "$DIST/yolomaster_edge"

# ---- 3/6: GPU extras (dlopened provider + the CUDA/cuDNN runtime it hard-links) ----
if [ "$VARIANT" = gpu ]; then
  echo "== 3/6  bundle ONNX CUDA provider + CUDA 12 / cuDNN 9 runtime =="
  cp -L "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so"   "$DIST/lib/"
  cp -L "$ORT_ROOT/lib/libonnxruntime_providers_shared.so" "$DIST/lib/"
  # (providers_tensorrt deliberately NOT bundled - mirrors the Windows strip list)

  # every NEEDED of libonnxruntime_providers_cuda.so (readelf -d); on Linux ALL are
  # mandatory - a single missing one makes the provider dlopen fail and ORT silently
  # falls back to CPU. libnvJitLink is cublasLt's optional companion; bundle if found.
  REQUIRED="libcublasLt.so.12 libcublas.so.12 libcurand.so.10 libcufft.so.11 libcudart.so.12
            libcudnn.so.9 libcudnn_adv.so.9 libcudnn_ops.so.9 libcudnn_cnn.so.9 libcudnn_graph.so.9
            libcudnn_engines_runtime_compiled.so.9 libcudnn_engines_precompiled.so.9
            libcudnn_heuristic.so.9 libnvrtc.so.12"
  OPTIONAL="libnvJitLink.so.12"

  # search order: explicit override, CUDA toolkit installs, distro dir, pip nvidia-*-cu12
  # wheels (any python env; the cu13/ wheel subtree is explicitly skipped).
  find_cuda_lib() {  # $1 = soname -> prints full path or nothing
    local so="$1" d
    for d in ${CUDA_LIB_DIRS:-} /usr/local/cuda-12*/targets/x86_64-linux/lib /usr/local/cuda-12*/lib64 \
             /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
      [ -e "$d/$so" ] && { echo "$d/$so"; return; }
    done
    # -print -quit: first match without a pipe (find|head dies of SIGPIPE under pipefail)
    find /root/anaconda3 /opt/conda /usr/lib/python3* -path '*/nvidia/*/lib/'"$so" \
         -not -path '*/cu13/*' -print -quit 2>/dev/null || true
  }
  MISSING=""
  for so in $REQUIRED; do
    src="$(find_cuda_lib "$so")"
    if [ -n "$src" ]; then cp -L "$src" "$DIST/lib/$so"; echo "  [cuda] $so  <- $src"
    else MISSING="$MISSING $so"; fi
  done
  for so in $OPTIONAL; do
    src="$(find_cuda_lib "$so")"
    [ -n "$src" ] && { cp -L "$src" "$DIST/lib/$so"; echo "  [cuda] $so  <- $src (optional)"; }
  done
  if [ -n "$MISSING" ]; then
    echo "ERROR: required CUDA/cuDNN libraries not found:$MISSING"
    echo "       set CUDA_LIB_DIRS=\"/path/one /path/two\" and re-run."
    exit 1
  fi
  # Resolve dependencies introduced by the provider itself (and by CUDA/cuDNN
  # libraries found above), not only those visible from the main executable.
  # Otherwise a missing transitive .so can make ORT silently fall back to CPU.
  copy_closure "$DIST/lib/libonnxruntime_providers_cuda.so"
  copy_closure "$DIST/lib/libonnxruntime_providers_shared.so"
fi

# ---- 4/6: rpaths + models + README ------------------------------------------------
echo "== 4/6  rpaths, models, README =="
patchelf --set-rpath '$ORIGIN/lib' "$DIST/yolomaster_edge"
for l in "$DIST"/lib/*.so*; do patchelf --set-rpath '$ORIGIN' "$l" 2>/dev/null || true; done

for m in v0.1-seg-n.onnx v0.1-seg-n.metadata.yaml; do
  cp "$ROOT/models/$m" "$DIST/models/" 2>/dev/null || echo "  [warn] model missing: $m"
done
if [ "$MNN_AVAILABLE" -eq 1 ]; then
  cp "$ROOT/models/v0.1-seg-n.mnn" "$DIST/models/" 2>/dev/null || echo "  [warn] model missing: v0.1-seg-n.mnn"
fi
if [ "$NCNN_AVAILABLE" -eq 1 ]; then
  cp -r "$ROOT/models/v0.1-seg-n_ncnn" "$DIST/models/" 2>/dev/null || echo "  [warn] model missing: v0.1-seg-n_ncnn"
fi
[ -f "$DIST/models/v0.1-seg-n.onnx" ] || { echo "ERROR: default model v0.1-seg-n.onnx missing - the README quick start would not work"; exit 1; }

GPU_NOTE=""
[ "$VARIANT" = gpu ] && GPU_NOTE="
This is the CUDA bundle: ONNX Runtime's CUDA 12 execution provider plus the cuDNN and
CUDA runtime libraries are included, so ONNX runs on NVIDIA GPUs with nothing installed
on the target except an NVIDIA driver (R525+). Add --device cuda to use it. The very
first CUDA run on a machine can take up to a minute (one-time kernel autotuning, cached
on disk; later runs start in a couple of seconds). The lean (non-CUDA) bundle is the
right choice when you do not need ONNX-on-GPU."
cat > "$DIST/README.txt" <<EOF
YOLO-Master edge runner $VERSION -- portable Linux x86_64 bundle.
Self-contained: runs on any glibc>=2.35 (Ubuntu 22.04+) x86_64, no install needed.
Backends: ONNX Runtime$([ "$NCNN_AVAILABLE" -eq 1 ] && printf ' / ncnn (GPU via Vulkan when a driver is present)')$([ "$MNN_AVAILABLE" -eq 1 ] && printf ' / MNN').
Detection and segmentation; image, folder, newline-delimited .txt list,
dataset.yaml and video sources (ffmpeg).
$GPU_NOTE
Quick start:
  ./yolomaster_edge -m models/v0.1-seg-n.onnx -s <image|dir|video> --out out
$(if [ "$NCNN_AVAILABLE" -eq 1 ]; then printf '  ./yolomaster_edge -m models/v0.1-seg-n_ncnn -s <image|dir|video> --out out\n'; fi)
$(if [ "$MNN_AVAILABLE" -eq 1 ]; then printf '  ./yolomaster_edge -m models/v0.1-seg-n.mnn  -s <image|dir|video> --out out\n'; fi)

New in 1.1.0:
  --slicing off|dense|sparse   sliced inference (Sparse SAHI) for small objects
  --tile-size N                tile edge in source px (0 = model input)
  --slicing-masks              keep global-pass masks in sliced runs (seg models)
  --cw-nms --sigma S           Cluster-Weighted NMS refinement
  --export-labels DIR          write annotations (WYSIWYG at current settings)
  --label-format yolo|coco|voc --sampling all|1s|N (video frame sampling)

All flags: ./yolomaster_edge --help
License: AGPL-3.0. See the repository LICENSE file.
EOF

# ---- 5/6: self-test the staged bundle ----------------------------------------------
echo "== 5/6  self-test (clean env, staged tree) =="
TESTDIR="$(mktemp -d)"; cp -r "$DIST" "$TESTDIR/b"
run_clean() { env -i PATH=/usr/bin:/bin HOME="$TESTDIR" "$TESTDIR/b/yolomaster_edge" "$@"; }
run_clean --help >/dev/null || { echo "SELF-TEST FAILED: --help"; exit 1; }
if ldd "$TESTDIR/b/yolomaster_edge" | grep -q "not found"; then
  echo "SELF-TEST FAILED: unresolved libraries:"; ldd "$TESTDIR/b/yolomaster_edge" | grep "not found"; exit 1
fi
TEST_IMG="$(ls "$ROOT"/visdrone50/images/val/*.jpg 2>/dev/null | head -1 || true)"
if [ -n "$TEST_IMG" ]; then
  MODELS=(models/v0.1-seg-n.onnx)
  [ "$NCNN_AVAILABLE" -eq 1 ] && MODELS+=(models/v0.1-seg-n_ncnn)
  [ "$MNN_AVAILABLE" -eq 1 ] && MODELS+=(models/v0.1-seg-n.mnn)
  for mdl in "${MODELS[@]}"; do
    [ -e "$TESTDIR/b/$mdl" ] || { echo "  [warn] model missing, skipping: $mdl"; continue; }
    run_clean -m "$TESTDIR/b/$mdl" -s "$TEST_IMG" --no-save --quiet >/dev/null \
      && echo "  [ok] inference: $mdl" \
      || { echo "SELF-TEST FAILED: $mdl"; exit 1; }
  done
else
  echo "  [warn] no test image found (visdrone50/ absent) - inference self-test skipped"
fi
if [ "$VARIANT" = gpu ]; then
  if ldd "$TESTDIR/b/lib/libonnxruntime_providers_cuda.so" | grep -q "not found"; then
    echo "SELF-TEST FAILED: CUDA provider has unresolved deps:"
    ldd "$TESTDIR/b/lib/libonnxruntime_providers_cuda.so" | grep "not found"; exit 1
  fi
  echo "  [ok] CUDA provider closure resolves"
  if command -v nvidia-smi >/dev/null 2>&1 && [ -n "$TEST_IMG" ]; then
    OUT="$(run_clean -m "$TESTDIR/b/models/v0.1-seg-n.onnx" -s "$TEST_IMG" -d cuda --no-save --quiet 2>&1 || true)"
    echo "$OUT" | grep -qiE "ep=(cuda|tensorrt)" && echo "  [ok] --device cuda runs on the GPU" \
      || { echo "SELF-TEST FAILED: --device cuda fell back:"; echo "$OUT" | head -5; exit 1; }
  else
    echo "  [warn] no GPU on this host - --device cuda validation deferred"
  fi
fi
rm -rf "$TESTDIR"

# ---- 6/6: tar -----------------------------------------------------------------------
echo "== 6/6  tar =="
tar czf "$ROOT/dist/$NAME.tar.gz" --owner=0 --group=0 -C "$ROOT/dist" "$NAME"
echo
echo "Done."
echo "  libs bundled: $(ls "$DIST/lib" | wc -l)"
echo "  folder: $DIST  ($(du -sh "$DIST" | cut -f1))"
echo "  tarball: $ROOT/dist/$NAME.tar.gz  ($(du -h "$ROOT/dist/$NAME.tar.gz" | cut -f1))"
