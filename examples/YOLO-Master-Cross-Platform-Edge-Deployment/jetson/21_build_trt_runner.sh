#!/usr/bin/env bash
# Build the C++ runner WITH the TensorRT 10 backend -> real GPU inference from a .engine.
# TRT/CUDA come from the target JetPack (no SDK download). The native source uses
# TensorRT's named-I/O API; TensorRT 8 targets should use 22_build_ort_trt.sh.
#
# v1.1.0: builds against a LEAN local OpenCV (core/imgproc/imgcodecs/videoio, ffmpeg on,
# GStreamer/GUI off) so video sources + video label export work without dragging the
# JetPack OpenCV's GStreamer closure into the bundle. Built once into
# third_party/opencv-lean (~30 min on an Orin Nano), cached afterward.
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(cd .. && pwd)"

OCV="$ROOT/third_party/opencv-lean"
if [ ! -f "$OCV/lib/cmake/opencv4/OpenCVConfig.cmake" ]; then
  echo "== one-time lean OpenCV build (video support without the GStreamer closure) =="
  SRC="$ROOT/third_party/opencv-lean-src"
  [ -d "$SRC" ] || git clone --depth 1 --branch 4.10.0 https://github.com/opencv/opencv.git "$SRC"
  cmake -S "$SRC" -B "$SRC/build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$OCV" \
    -DBUILD_LIST=core,imgproc,imgcodecs,videoio -DBUILD_SHARED_LIBS=ON \
    -DWITH_FFMPEG=ON -DWITH_GSTREAMER=OFF -DWITH_GTK=OFF -DWITH_QT=OFF -DWITH_V4L=OFF -DWITH_1394=OFF \
    -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_OPENJPEG=OFF -DWITH_JASPER=OFF -DWITH_OPENEXR=OFF -DWITH_EIGEN=OFF \
    -DBUILD_JPEG=ON -DBUILD_PNG=ON -DBUILD_ZLIB=ON \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF \
    -DOPENCV_GENERATE_PKGCONFIG=OFF -DBUILD_opencv_python_bindings_generator=OFF >/dev/null
  cmake --build "$SRC/build" -j"$(nproc)"
  cmake --install "$SRC/build" >/dev/null
fi

cd "$ROOT/cpp"
rm -rf build_trt && mkdir build_trt && cd build_trt

# JetPack installs TensorRT/CUDA in the standard system locations.  For a
# container, cross-build, or unpacked SDK, callers may provide explicit roots;
# keep the arguments conditional so the target's normal discovery still works.
CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DOpenCV_DIR="$OCV/lib/cmake/opencv4"
  -DUSE_ORT=OFF -DUSE_NCNN=OFF -DUSE_MNN=OFF -DUSE_TRT=ON
)
if [ -n "${TENSORRT_ROOT:-}" ]; then CMAKE_ARGS+=("-DTENSORRT_ROOT=$TENSORRT_ROOT"); fi
if [ -n "${CUDA_ROOT:-}" ]; then CMAKE_ARGS+=("-DCUDA_ROOT=$CUDA_ROOT"); fi
cmake .. "${CMAKE_ARGS[@]}" 2>&1 | tee configure.log
cmake --build . --parallel "$(nproc)" 2>&1 | tee build.log

BIN="$ROOT/cpp/build_trt/yolomaster_edge"
[ -x "$BIN" ] || { echo "build completed without an executable: $BIN" >&2; exit 1; }
ENG="$ROOT/jetson/engines/esmoe_n_fp16.engine"
echo
echo "built: $BIN"
echo
echo "== run on the GPU =="
echo "  $BIN --model $ENG --source <image|dir|video> --conf 0.25 --out out"
echo "  (class names come from a metadata.yaml sidecar next to the engine when present;"
echo "   otherwise pass --classes visdrone)"
echo
echo "== v1.1.0 features =="
echo "  slicing:      --slicing sparse [--tile-size N] [--slicing-masks]"
echo "  CW-NMS:       --cw-nms [--sigma 0.1]"
echo "  label export: --export-labels out_labels --label-format yolo|coco|voc [--sampling all|1s|N]"
echo
echo "== dump preds for on-device mAP (then scp preds/ to the server and run scripts/eval_map.py) =="
echo "  $BIN --model $ENG --source <val_images_dir> --classes visdrone \\"
echo "       --conf 0.001 --iou 0.7 --multi-label --save-txt preds --no-save --quiet"
