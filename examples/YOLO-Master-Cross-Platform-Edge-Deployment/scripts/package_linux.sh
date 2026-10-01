#!/usr/bin/env bash
# Assemble a self-contained, relocatable Linux x86_64 bundle of the yolomaster_edge CLI.
#
#   usage: package_linux.sh [variant] [version]
#
#   variant   backends in the bundle                       tarball
#   onnx-cpu  ONNX Runtime (CPU)                           yolomaster-edge-linux-x64-onnx-cpu-<v>.tar.gz
#   onnx-cuda ONNX Runtime CUDA 12 EP (+ CUDA preproc)     yolomaster-edge-linux-x64-onnx-cuda12-<v>.tar.gz
#   ncnn      ncnn (Vulkan GPU when the host has a driver) yolomaster-edge-linux-x64-ncnn-<v>.tar.gz
#   mnn       MNN (CPU)                                    yolomaster-edge-linux-x64-mnn-<v>.tar.gz
#   trt       TensorRT 10 (+ CUDA preproc), every GPU      yolomaster-edge-linux-x64-trt10-cuda12-<v>.tar.gz
#             family's engine builder (TRT_ARCHS=all)
#   all       every backend above in one bundle            yolomaster-edge-linux-x64-all-cuda12-<v>.tar.gz
#   cpu       legacy 1.x layout: ONNX CPU + ncnn + MNN     yolomaster-edge-linux-x64-<v>.tar.gz
#   gpu       legacy 1.x layout: ONNX CUDA + ncnn + MNN    yolomaster-edge-linux-x64-gpu_cuda12-<v>.tar.gz
#
#   env: ORT_ROOT NCNN_ROOT MNN_ROOT   SDK overrides (defaults under third_party/)
#        TENSORRT_LIB_DIR              where libnvinfer.so.10 lives (default /usr/lib/x86_64-linux-gnu)
#        TRT_ARCHS                     "all" (default: every libnvinfer_builder_resource_* found) or a
#                                      list like "89" / "86 89 90"; engines are BUILT on the user's
#                                      machine from the .onnx, so the builder resource for its GPU
#                                      family must be in the bundle
#        CUDA_ARCHS                    archs of the CUDA preprocessing kernel (default: 75..120 real + PTX)
#        CUDA_LIB_DIRS                 extra dirs to search for the CUDA / cuDNN / TensorRT runtime
#        SKIP_GPU_SMOKE=1              do not run the --device cuda / -b trt self-tests even with a GPU
#
# Every bundle: the CLI, the full transitive .so closure (lean OpenCV, ffmpeg codec stack; never
# glibc, never the NVIDIA driver), $ORIGIN rpaths via patchelf, the default v0.1-seg-N model in the
# formats the bundle can run, README.txt. Runs on any glibc>=2.35 (Ubuntu 22.04+) x86_64.
#
# GPU bundles ship the CUDA 12 runtime they need (onnx-cuda: the 14 libraries the ORT provider
# hard-links plus every cuDNN 9 sublibrary; trt: libnvinfer + parser + plugin, libcudart, libnvrtc,
# the builder resources), so the target needs nothing but an NVIDIA driver (R525+ / R570+ for
# CUDA 12.9 builds). Assembly of the onnx-cuda bundle needs no GPU; trt / all need nvcc on the build
# host; the GPU self-tests run only where nvidia-smi reports a device.
set -eo pipefail

VARIANT="${1:-cpu}"
VERSION="${2:-$(tr -d "[:space:]" < "$(cd "$(dirname "$0")/.." && pwd)/VERSION")}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---- variant -> wanted backends ---------------------------------------------------
WANT_ORT=none; WANT_NCNN=0; WANT_MNN=0; WANT_TRT=0
case "$VARIANT" in
  onnx-cpu)  WANT_ORT=cpu;  SUFFIX="-onnx-cpu" ;;
  onnx-cuda) WANT_ORT=cuda; SUFFIX="-onnx-cuda12" ;;
  ncnn)      WANT_NCNN=1;   SUFFIX="-ncnn" ;;
  mnn)       WANT_MNN=1;    SUFFIX="-mnn" ;;
  trt)       WANT_TRT=1;    SUFFIX="-trt10-cuda12" ;;
  all)       WANT_ORT=cuda; WANT_NCNN=1; WANT_MNN=1; WANT_TRT=1; SUFFIX="-all-cuda12" ;;
  cpu)       WANT_ORT=cpu;  WANT_NCNN=1; WANT_MNN=1; SUFFIX="" ;;
  gpu)       WANT_ORT=cuda; WANT_NCNN=1; WANT_MNN=1; SUFFIX="-gpu_cuda12" ;;
  *) echo "usage: package_linux.sh [onnx-cpu|onnx-cuda|ncnn|mnn|trt|all|cpu|gpu] [version]"; exit 2 ;;
esac
WANT_CUDA=0; { [ "$WANT_ORT" = cuda ] || [ "$WANT_TRT" = 1 ]; } && WANT_CUDA=1
NAME="yolomaster-edge-linux-x64${SUFFIX}-$VERSION"
DIST="$ROOT/dist/$NAME"
BUILD="$ROOT/cpp/build_pkg-$VARIANT"
OCV="$ROOT/third_party/opencv-lean"
if [ "$WANT_ORT" = cuda ]; then DEFAULT_ORT_ROOT="$ROOT/third_party/onnxruntime-linux-x64-gpu-1.20.1"
else DEFAULT_ORT_ROOT="$ROOT/third_party/onnxruntime-linux-x64-1.18.1"; fi
ORT_ROOT="${ORT_ROOT:-$DEFAULT_ORT_ROOT}"
# ncnn: third_party/ncnn, else the newest staged ncnn-*-shared SDK
if [ -z "${NCNN_ROOT:-}" ]; then
  NCNN_ROOT="$ROOT/third_party/ncnn"
  [ -d "$NCNN_ROOT" ] || NCNN_ROOT="$(ls -d "$ROOT"/third_party/ncnn-*-shared 2>/dev/null | sort | tail -1 || true)"
fi
MNN_ROOT="${MNN_ROOT:-$ROOT/third_party/mnn-src}"
TENSORRT_LIB_DIR="${TENSORRT_LIB_DIR:-/usr/lib/x86_64-linux-gnu}"

command -v patchelf >/dev/null 2>&1 || {
  echo "ERROR: patchelf is required to create a relocatable bundle." >&2; exit 1; }

# ---- SDK probes (a linkable library, not only headers) ----------------------------
find_backend_library() {
  local root="$1" name="$2"
  [ -n "$root" ] || return 0
  find "$root" -maxdepth 4 \( -type f -o -type l \) \
    \( -name "lib${name}.a" -o -name "lib${name}.so" -o -name "lib${name}.so.*" \) -print -quit 2>/dev/null || true
}
if [ "$WANT_ORT" != none ]; then
  [ -d "$ORT_ROOT/lib" ] || { echo "ERROR: ONNX Runtime not found at $ORT_ROOT"; exit 1; }
  echo "  [ok] ONNX Runtime: $ORT_ROOT"
fi
if [ "$WANT_NCNN" = 1 ]; then
  NCNN_LIB_PATH="$(find_backend_library "$NCNN_ROOT" ncnn)"
  { [ -f "$NCNN_ROOT/include/ncnn/net.h" ] && [ -n "$NCNN_LIB_PATH" ]; } || { echo "ERROR: ncnn SDK not found (NCNN_ROOT=$NCNN_ROOT)"; exit 1; }
  echo "  [ok] ncnn: $NCNN_LIB_PATH"
fi
if [ "$WANT_MNN" = 1 ]; then
  MNN_LIB_PATH="$(find_backend_library "$MNN_ROOT" MNN)"
  { [ -f "$MNN_ROOT/include/MNN/Interpreter.hpp" ] && [ -n "$MNN_LIB_PATH" ]; } || { echo "ERROR: MNN SDK not found (MNN_ROOT=$MNN_ROOT)"; exit 1; }
  echo "  [ok] MNN: $MNN_LIB_PATH"
fi
NVCC=""
if [ "$WANT_CUDA" = 1 ]; then
  # the newest CUDA 12.x toolkit on the host (TensorRT 10.16 is built against 12.9; the unversioned
  # /usr/local/cuda symlink is only a fallback)
  NVCC="$(ls -d /usr/local/cuda-12*/bin/nvcc 2>/dev/null | sort -V | tail -1 || true)"
  [ -n "$NVCC" ] || NVCC="$(ls /usr/local/cuda/bin/nvcc 2>/dev/null || true)"
fi
if [ "$WANT_TRT" = 1 ]; then
  [ -f "$TENSORRT_LIB_DIR/libnvinfer.so.10" ] || { echo "ERROR: libnvinfer.so.10 not found in $TENSORRT_LIB_DIR (TENSORRT_LIB_DIR=...)"; exit 1; }
  [ -n "$NVCC" ] || { echo "ERROR: the trt / all bundles need nvcc (CUDA preprocessing kernel + TensorRT build)"; exit 1; }
  echo "  [ok] TensorRT: $(readlink -f "$TENSORRT_LIB_DIR/libnvinfer.so.10")"
fi

# ---- 0/6: lean OpenCV (built once, cached) ---------------------------------------
if [ ! -f "$OCV/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
  echo "== 0/6  building lean OpenCV (one-time, ~15 min) =="
  SRC="$ROOT/third_party/opencv-lean-src"
  [ -d "$SRC" ] || git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git "$SRC"
  cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OCV" \
    -DBUILD_LIST=core,imgproc,imgcodecs,videoio,video,calib3d -DBUILD_SHARED_LIBS=ON \
    -DWITH_FFMPEG=ON -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_V4L=OFF -DWITH_1394=OFF \
    -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_EIGEN=OFF -DWITH_IPP=ON \
    -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_ZLIB=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=OFF -DBUILD_opencv_python_bindings_generator=OFF >/dev/null
  cmake --build "$SRC/build" -j"$(nproc)"
  cmake --install "$SRC/build" >/dev/null
fi

# ---- 1/6: clean release build -----------------------------------------------------
echo "== 1/6  clean $VARIANT release build =="
rm -rf "$BUILD"
ARGS=( -DCMAKE_BUILD_TYPE=Release -DOpenCV_DIR="$OCV/lib/cmake/opencv4" )
if [ "$WANT_ORT" != none ]; then ARGS+=( -DUSE_ORT=ON -DONNXRUNTIME_ROOT="$ORT_ROOT" ); else ARGS+=( -DUSE_ORT=OFF ); fi
if [ "$WANT_NCNN" = 1 ]; then ARGS+=( -DUSE_NCNN=ON -DNCNN_ROOT="$NCNN_ROOT" ); else ARGS+=( -DUSE_NCNN=OFF ); fi
if [ "$WANT_MNN" = 1 ]; then ARGS+=( -DUSE_MNN=ON -DMNN_ROOT="$MNN_ROOT" ); else ARGS+=( -DUSE_MNN=OFF ); fi
if [ "$WANT_TRT" = 1 ]; then ARGS+=( -DUSE_TRT=ON -DTENSORRT_ROOT=/usr ); else ARGS+=( -DUSE_TRT=OFF ); fi
if [ "$WANT_CUDA" = 1 ] && [ -n "$NVCC" ]; then
  # the CUDA preprocessing kernel for every GPU family nvcc knows (real SASS) plus PTX of the newest
  # for JIT on later ones; CUDA_ARCHS="89" narrows it
  if [ -z "${CUDA_ARCHS:-}" ]; then
    KNOWN="$("$NVCC" --list-gpu-arch 2>/dev/null | sed -n 's/compute_//p' | tr '\n' ' ')"
    CUDA_ARCHS=""
    for a in 75 80 86 89 90 100 120; do case " $KNOWN " in *" $a "*) CUDA_ARCHS="$CUDA_ARCHS $a";; esac; done
  fi
  CMAKE_ARCHS=""; LAST=""
  for a in $CUDA_ARCHS; do CMAKE_ARCHS="$CMAKE_ARCHS;$a-real"; LAST="$a"; done
  CMAKE_ARCHS="${CMAKE_ARCHS#;};$LAST-virtual"
  ARGS+=( -DUSE_CUDA_PREPROC=ON -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CUDA_ARCHITECTURES="$CMAKE_ARCHS" )
  echo "  [ok] CUDA preprocessing kernel: $NVCC (archs $CMAKE_ARCHS)"
elif [ "$WANT_CUDA" = 1 ]; then
  echo "  [warn] nvcc not found: $VARIANT bundle without the CUDA preprocessing kernel (CPU letterbox)"
fi
cmake -S "$ROOT/cpp" -B "$BUILD" "${ARGS[@]}"
cmake --build "$BUILD" -j"$(nproc)"

# ---- 2/6: stage binary + .so closure ----------------------------------------------
echo "== 2/6  stage binary + library closure =="
rm -rf "$DIST"; mkdir -p "$DIST/lib" "$DIST/models"
cp "$BUILD/yolomaster_edge" "$DIST/yolomaster_edge"

# glibc / dynamic-loader core: MUST come from the target system, never bundle. Same for the
# NVIDIA driver's user-space libraries (libcuda, libnvidia-*): they must match the host's kernel
# driver, and a bundled copy would shadow the host's via $ORIGIN (cuDNN 9.27's
# libcudnn_engines_runtime_compiled links libcuda.so.1 directly, which is how one slipped in once).
EXCLUDE='libc\.so|libm\.so|libdl\.so|librt\.so|libpthread\.so|ld-linux|libresolv\.so|linux-vdso|libmvec\.so|libcuda\.so|libnvidia-|libnvcuvid|libnvoptix'
declare -A SEEN_LIBS=()
copy_closure() {   # walk the complete ELF dependency closure of one object
  local object="$1" so base
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

# search order for CUDA / cuDNN / TensorRT runtime files: explicit dirs, CUDA toolkit installs,
# the distro dir, pip nvidia-*-cu12 wheels (the cu13 subtree is skipped)
CUDA_DIRS_NEWEST_FIRST="$(ls -d /usr/local/cuda-12* 2>/dev/null | sort -V -r | tr '\n' ' ')"   # 12.9 before 12.4
find_cuda_lib() {  # $1 = soname -> prints full path or nothing
  local so="$1" d c
  for d in ${CUDA_LIB_DIRS:-} "$TENSORRT_LIB_DIR" \
           $(for c in $CUDA_DIRS_NEWEST_FIRST; do echo "$c/targets/x86_64-linux/lib $c/lib64"; done) \
           /usr/local/cuda/lib64 /usr/lib/x86_64-linux-gnu; do
    [ -e "$d/$so" ] && { echo "$d/$so"; return; }
  done
  find /root/anaconda3 /opt/conda /usr/lib/python3* -path '*/nvidia/*/lib/'"$so" -not -path '*/cu13/*' -print -quit 2>/dev/null || true
}
# copy a runtime library under its soname AND its fully versioned real name (cuDNN and TensorRT
# dlopen sublibraries by the versioned name)
stage_cuda_lib() {  # $1 = soname, $2 = note
  local so="$1" src real
  src="$(find_cuda_lib "$so")"
  [ -n "$src" ] || return 1
  [ -e "$DIST/lib/$so" ] || { cp -L "$src" "$DIST/lib/$so"; echo "  [cuda] $so  <- $src${2:+ ($2)}"; }
  real="$(basename "$(readlink -f "$src")")"
  [ "$real" != "$so" ] && [ ! -e "$DIST/lib/$real" ] && ln -sf "$so" "$DIST/lib/$real"
  return 0
}

# ---- 3/6: GPU runtime ---------------------------------------------------------------
if [ "$WANT_ORT" = cuda ]; then
  echo "== 3/6  ONNX CUDA provider + CUDA 12 / cuDNN 9 runtime =="
  cp -L "$ORT_ROOT/lib/libonnxruntime_providers_cuda.so"   "$DIST/lib/"
  cp -L "$ORT_ROOT/lib/libonnxruntime_providers_shared.so" "$DIST/lib/"
  # every NEEDED of libonnxruntime_providers_cuda.so: on Linux ALL are mandatory, one missing makes
  # the provider dlopen fail and ORT silently falls back to CPU
  MISSING=""
  for so in libcublasLt.so.12 libcublas.so.12 libcurand.so.10 libcufft.so.11 libcudart.so.12 \
            libcudnn.so.9 libcudnn_adv.so.9 libcudnn_ops.so.9 libcudnn_cnn.so.9 libcudnn_graph.so.9 \
            libcudnn_engines_runtime_compiled.so.9 libcudnn_engines_precompiled.so.9 libcudnn_heuristic.so.9 libnvrtc.so.12; do
    stage_cuda_lib "$so" || MISSING="$MISSING $so"
  done
  stage_cuda_lib libnvJitLink.so.12 optional || true
  [ -z "$MISSING" ] || { echo "ERROR: required CUDA/cuDNN libraries not found:$MISSING (set CUDA_LIB_DIRS)"; exit 1; }
  # cuDNN dlopens engine sublibraries that are NOT in the provider's NEEDED list (9.26 added
  # libcudnn_engines_tensor_ir, 9.27 libcudnn_ext); without them cuDNN's frontend fails at the
  # first Conv on a machine with no system cuDNN, and on one that has a different version the
  # foreign copy is loaded instead (Integer overflow / SUBLIBRARY_VERSION_MISMATCH). Take every
  # libcudnn*.so.9 beside the libcudnn.so.9 we bundled.
  cudnn_dir="$(dirname "$(find_cuda_lib libcudnn.so.9)")"
  for f in "$cudnn_dir"/libcudnn*.so.9; do stage_cuda_lib "$(basename "$f")" "cuDNN sublibrary" || true; done
  copy_closure "$DIST/lib/libonnxruntime_providers_cuda.so"
  copy_closure "$DIST/lib/libonnxruntime_providers_shared.so"
fi
if [ "$WANT_TRT" = 1 ]; then
  echo "== 3/6  TensorRT 10 runtime + engine builder resources =="
  for so in libnvinfer.so.10 libnvonnxparser.so.10 libcudart.so.12 libnvrtc.so.12; do
    stage_cuda_lib "$so" || { echo "ERROR: $so not found (TENSORRT_LIB_DIR / CUDA_LIB_DIRS)"; exit 1; }
  done
  stage_cuda_lib libnvinfer_plugin.so.10 "optional, plugin ops" || true
  # nvrtc's builtins are dlopened by the versioned name beside libnvrtc
  for f in $(ls "$(dirname "$(find_cuda_lib libnvrtc.so.12)")"/libnvrtc-builtins.so.12* 2>/dev/null); do
    b="$(basename "$f")"; [ -e "$DIST/lib/$b" ] || { cp -L "$f" "$DIST/lib/$b"; echo "  [cuda] $b  <- $f (nvrtc builtins)"; }
  done
  # the engine builder resources: one per GPU family, dlopened by full name when an .onnx is
  # compiled into an engine on the user's machine. TRT_ARCHS=all ships every family found
  # (~2.9 GB), TRT_ARCHS="89" only Ada (L4 / L40 / RTX 40)
  TRT_REAL="$(readlink -f "$(find_cuda_lib libnvinfer.so.10)")"; TRT_DIR="$(dirname "$TRT_REAL")"
  TRT_VER="${TRT_REAL##*libnvinfer.so.}"
  n=0
  for f in "$TRT_DIR"/libnvinfer_builder_resource_*.so."$TRT_VER"; do
    [ -e "$f" ] || continue
    fam="$(basename "$f" | sed -E 's/libnvinfer_builder_resource_(.*)\.so\..*/\1/')"
    case "${TRT_ARCHS:-all}" in
      all) ;;
      *) keep=0; for a in ${TRT_ARCHS}; do [ "$fam" = "sm$a" ] && keep=1; done; [ "$fam" = ptx ] && keep=1; [ "$keep" = 1 ] || continue ;;
    esac
    cp -L "$f" "$DIST/lib/$(basename "$f")"; n=$((n + 1)); echo "  [trt]  $(basename "$f") ($(du -h "$f" | cut -f1))"
  done
  [ "$n" -gt 0 ] || echo "  [warn] no libnvinfer_builder_resource_* found: engines cannot be built from .onnx on the target"
  copy_closure "$DIST/lib/libnvinfer.so.10"
  copy_closure "$DIST/lib/libnvonnxparser.so.10"
fi
[ "$WANT_CUDA" = 1 ] || echo "== 3/6  (no GPU runtime in this variant) =="

# ---- 4/6: rpaths + models + README ------------------------------------------------
echo "== 4/6  rpaths, models, README =="
patchelf --set-rpath '$ORIGIN/lib' "$DIST/yolomaster_edge"
for l in "$DIST"/lib/*.so*; do patchelf --set-rpath '$ORIGIN' "$l" 2>/dev/null || true; done

MODELS=(); QUICK=""
if [ "$WANT_ORT" != none ] || [ "$WANT_TRT" = 1 ]; then
  for m in v0.1-seg-n.onnx v0.1-seg-n.metadata.yaml; do cp "$ROOT/models/$m" "$DIST/models/" || { echo "ERROR: models/$m missing"; exit 1; }; done
  if [ "$WANT_ORT" != none ]; then
    MODELS+=(models/v0.1-seg-n.onnx)
    QUICK+="  ./yolomaster_edge -m models/v0.1-seg-n.onnx -s <image|dir|video> --out out"
    [ "$WANT_ORT" = cuda ] && QUICK+="            # add --device cuda for the GPU"
    QUICK+=$'\n'
  fi
  if [ "$WANT_TRT" = 1 ]; then
    QUICK+="  ./yolomaster_edge -m models/v0.1-seg-n.onnx -s <image|dir|video> --out out -b trt --precision fp16"$'\n'
  fi
fi
if [ "$WANT_NCNN" = 1 ]; then
  cp -r "$ROOT/models/v0.1-seg-n_ncnn" "$DIST/models/" || { echo "ERROR: models/v0.1-seg-n_ncnn missing"; exit 1; }
  rm -f "$DIST/models/v0.1-seg-n_ncnn"/*.onnx
  MODELS+=(models/v0.1-seg-n_ncnn)
  QUICK+="  ./yolomaster_edge -m models/v0.1-seg-n_ncnn -s <image|dir|video> --out out"$'\n'
fi
if [ "$WANT_MNN" = 1 ]; then
  cp "$ROOT/models/v0.1-seg-n.mnn" "$DIST/models/" || { echo "ERROR: models/v0.1-seg-n.mnn missing"; exit 1; }
  cp "$ROOT/models/v0.1-seg-n.metadata.yaml" "$DIST/models/" 2>/dev/null || true
  MODELS+=(models/v0.1-seg-n.mnn)
  QUICK+="  ./yolomaster_edge -m models/v0.1-seg-n.mnn  -s <image|dir|video> --out out"$'\n'
fi

BACKENDS=""
[ "$WANT_ORT" = cpu ]  && BACKENDS+="ONNX Runtime (CPU)"
[ "$WANT_ORT" = cuda ] && BACKENDS+="ONNX Runtime (CPU, CUDA 12 EP with --device cuda)"
[ "$WANT_TRT" = 1 ]    && BACKENDS+="${BACKENDS:+ / }TensorRT 10 (-b trt)"
[ "$WANT_NCNN" = 1 ]   && BACKENDS+="${BACKENDS:+ / }ncnn (Vulkan GPU when a driver is present)"
[ "$WANT_MNN" = 1 ]    && BACKENDS+="${BACKENDS:+ / }MNN"
NOTES=""
[ "$WANT_ORT" = cuda ] && NOTES+="
ONNX on the GPU: the CUDA 12 execution provider plus the cuDNN 9 and CUDA runtime it needs are
included, so --device cuda works with nothing installed on the target except an NVIDIA driver.
The first CUDA run on a machine can take up to a minute (one-time kernel autotuning, cached on
disk). Preprocessing (letterbox + tensor build) runs on the GPU too; --cpu-preproc switches it off.
"
[ "$WANT_TRT" = 1 ] && NOTES+="
TensorRT: pass -b trt (add --precision fp16 for half precision). An .onnx model is compiled into a
TensorRT engine on first use and cached beside it (named by model hash, GPU and TensorRT version;
a few minutes once per model and GPU). The bundle ships TensorRT $TRT_VER with the engine builder
for ${TRT_ARCHS:-every} GPU famil$([ "${TRT_ARCHS:-all}" = all ] && echo "y it supports (Turing sm75 to Blackwell sm120, plus PTX)" || echo "ies $TRT_ARCHS"); a prebuilt .engine file also loads directly. --cuda-graph replays
the per-frame stream as a CUDA graph. Needs an NVIDIA driver R570+ (CUDA 12.9).
"
[ "$WANT_NCNN" = 1 ] && NOTES+="
ncnn: runs on the CPU, or on any GPU through Vulkan when the host has a Vulkan driver (loaded at
runtime, nothing to install for the CPU path).
"
cat > "$DIST/README.txt" <<EOF
YOLO-Master edge runner $VERSION -- portable Linux x86_64 bundle ($VARIANT).
Self-contained: runs on any glibc>=2.35 (Ubuntu 22.04+) x86_64, no install needed.
Backends: $BACKENDS.
Detection and segmentation; image, folder, newline-delimited .txt list, dataset.yaml and
video sources (ffmpeg); benchmark mode (--bench), in-process mAP (--accuracy), tracking (--track).
$NOTES
Quick start:
$QUICK
All flags: ./yolomaster_edge --help
License: AGPL-3.0. (c) 2026 Thomas Li. https://github.com/skywalker-lt/yolo-master-edge
EOF

# ---- 5/6: self-test the staged bundle ----------------------------------------------
echo "== 5/6  self-test (clean env, staged tree) =="
TESTDIR="$(mktemp -d)"; cp -r "$DIST" "$TESTDIR/b"
run_clean() { env -i PATH=/usr/bin:/bin HOME="$TESTDIR" "$TESTDIR/b/yolomaster_edge" "$@"; }
run_clean --help >/dev/null || { echo "SELF-TEST FAILED: --help"; exit 1; }
if ldd "$TESTDIR/b/yolomaster_edge" | grep -q "not found"; then
  echo "SELF-TEST FAILED: unresolved libraries:"; ldd "$TESTDIR/b/yolomaster_edge" | grep "not found"; exit 1
fi
for l in libonnxruntime_providers_cuda.so libnvinfer.so.10 libnvonnxparser.so.10; do
  [ -e "$TESTDIR/b/lib/$l" ] || continue
  if ldd "$TESTDIR/b/lib/$l" | grep -q "not found"; then echo "SELF-TEST FAILED: $l has unresolved deps:"; ldd "$TESTDIR/b/lib/$l" | grep "not found"; exit 1; fi
  echo "  [ok] $l closure resolves"
done
TEST_IMG="$(ls "$ROOT"/visdrone50/images/val/*.jpg 2>/dev/null | head -1 || true)"
HAVE_GPU=0; command -v nvidia-smi >/dev/null 2>&1 && [ "${SKIP_GPU_SMOKE:-0}" != 1 ] && HAVE_GPU=1
if [ -n "$TEST_IMG" ]; then
  for mdl in "${MODELS[@]}"; do
    run_clean -m "$TESTDIR/b/$mdl" -s "$TEST_IMG" --no-save --quiet >/dev/null \
      && echo "  [ok] inference: $mdl" || { echo "SELF-TEST FAILED: $mdl"; exit 1; }
  done
  if [ "$WANT_ORT" = cuda ] && [ "$HAVE_GPU" = 1 ]; then
    OUT="$(run_clean -m "$TESTDIR/b/models/v0.1-seg-n.onnx" -s "$TEST_IMG" -d cuda --no-save --quiet 2>&1 || true)"
    echo "$OUT" | grep -qiE "ep=ort-CUDA" && ! echo "$OUT" | grep -q "\[skip\]" \
      && echo "  [ok] --device cuda runs on the GPU ($(echo "$OUT" | grep -oE "ep=[^ ]+" | head -1))" \
      || { echo "SELF-TEST FAILED: --device cuda:"; echo "$OUT" | head -5; exit 1; }
  fi
  if [ "$WANT_TRT" = 1 ] && [ "$HAVE_GPU" = 1 ]; then
    echo "  [..] TensorRT smoke: building an fp32 engine for the test (a few minutes)"
    cp "$TESTDIR/b/models/v0.1-seg-n.onnx" "$TESTDIR/trt-test.onnx"; cp "$TESTDIR/b/models/v0.1-seg-n.metadata.yaml" "$TESTDIR/trt-test.metadata.yaml"
    OUT="$(run_clean -m "$TESTDIR/trt-test.onnx" -s "$TEST_IMG" -b trt --no-save --quiet 2>&1 || true)"
    echo "$OUT" | grep -qE "ep=TRT" && ! echo "$OUT" | grep -q "\[skip\]" \
      && echo "  [ok] -b trt runs on the GPU ($(echo "$OUT" | grep -oE "ep=[^ ]+" | head -1))" \
      || { echo "SELF-TEST FAILED: -b trt:"; echo "$OUT" | tail -5; exit 1; }
  fi
  [ "$WANT_CUDA" = 1 ] && [ "$HAVE_GPU" = 0 ] && echo "  [warn] no GPU on this host - GPU self-tests deferred"
else
  echo "  [warn] no test image found (visdrone50/ absent) - inference self-test skipped"
fi
rm -rf "$TESTDIR"

# ---- 6/6: tar + checksum -------------------------------------------------------------
echo "== 6/6  tar =="
tar czf "$ROOT/dist/$NAME.tar.gz" --owner=0 --group=0 -C "$ROOT/dist" "$NAME"
SUMS="$ROOT/dist/SHA256SUMS-linux-$VERSION.txt"
SUM="$(cd "$ROOT/dist" && sha256sum "$NAME.tar.gz")"
touch "$SUMS"; grep -v " $NAME.tar.gz\$" "$SUMS" > "$SUMS.tmp" || true; echo "$SUM" >> "$SUMS.tmp"; sort -k2 "$SUMS.tmp" > "$SUMS"; rm -f "$SUMS.tmp"
echo
echo "Done."
echo "  libs bundled: $(ls "$DIST/lib" | wc -l)"
echo "  folder: $DIST  ($(du -sh "$DIST" | cut -f1))"
echo "  tarball: $ROOT/dist/$NAME.tar.gz  ($(du -h "$ROOT/dist/$NAME.tar.gz" | cut -f1))"
echo "  sha256: $SUM  (recorded in $(basename "$SUMS"))"
