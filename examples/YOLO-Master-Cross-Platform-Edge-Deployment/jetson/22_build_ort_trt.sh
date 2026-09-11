#!/usr/bin/env bash
# Build the runner with ONNXRuntime + the TensorRT Execution Provider.
# Runs the .onnx directly (no hand-built engine): ORT builds a TRT engine internally on the first
# run and CACHES it, uses INT8 where the model has QDQ nodes + FP16 elsewhere, and falls back to
# CUDA/CPU for unsupported subgraphs. More portable than a per-device .engine.
#
# Needs a Jetson ONNXRuntime build that INCLUDES the TensorRT EP (CUDA + TRT), matched to your
# JetPack CUDA/TRT. Point ORT_ROOT at it (headers in $ORT_ROOT/include, libonnxruntime.so in lib):
#   ORT_ROOT=/path/to/onnxruntime-jetson bash 22_build_ort_trt.sh
#
# Getting that ORT (the provisioning, not this script, is the gate):
#   * Use a Jetson/aarch64 distribution that explicitly includes the CUDA and
#     TensorRT execution providers for the installed JetPack.  Availability and
#     index URLs are release-specific; verify the wheel's provider list before
#     building and record the exact source in DEPLOYMENT_LOG.md.
#   * The public PyPI onnxruntime-gpu package is generally x86_64-only.  The C++
#     runner also needs headers: pair the wheel's libonnxruntime.so with matching
#     version headers from the ONNX Runtime release under $ORT_ROOT/{lib,include}.
#   * If no compatible wheel exists, build ORT from source with --use_tensorrt,
#     or use the native TensorRT 10 backend (jetson/21_build_trt_runner.sh).
set -euo pipefail
cd "$(dirname "$0")"; ROOT="$(cd .. && pwd)"
: "${ORT_ROOT:?set ORT_ROOT to a Jetson ONNXRuntime with the TensorRT EP (see README)}"
[ -f "$ORT_ROOT/include/onnxruntime_cxx_api.h" ] || { echo "no ORT headers at $ORT_ROOT/include"; exit 1; }

cd "$ROOT/cpp"; rm -rf build_ort_trt && mkdir build_ort_trt && cd build_ort_trt
cmake .. -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON -DUSE_NCNN=OFF -DUSE_TRT=OFF -DUSE_ORT=ON \
         -DONNXRUNTIME_ROOT="$ORT_ROOT" 2>&1 | tee configure.log
cmake --build . --parallel "$(nproc)" 2>&1 | tee build.log

BIN="$ROOT/cpp/build_ort_trt/yolomaster_edge"
[ -x "$BIN" ] || { echo "build completed without an executable: $BIN" >&2; exit 1; }
echo
echo "built: $BIN"
echo "run (ORT + TensorRT EP; first run builds+caches the engine in ./trt_engine_cache):"
echo "  $BIN --model $ROOT/jetson/models/esmoe_n_visdrone_sim.onnx --source <img|dir> \\"
echo "       --device trt --classes visdrone --out out"
echo "  # for INT8: --model esmoe_n_visdrone_int8_qdq.onnx  (QDQ nodes drive INT8, FP16 fallback)"
