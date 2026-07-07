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
#   * Standard JetPack (jp6 / CUDA 12.x): NVIDIA ships prebuilt onnxruntime-gpu WITH the CUDA+TRT EPs
#     via the Jetson index, e.g.
#       pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.dev/jp6/cu126
#     Note: PyPI's onnxruntime-gpu is x86_64-only (no aarch64 wheel) — you must use the Jetson index.
#     The C++ runner also needs headers: pair the wheel's libonnxruntime.so with the matching-version
#     headers from the onnxruntime GitHub release (arch-independent) under $ORT_ROOT/{lib,include}.
#   * Bleeding-edge CUDA (e.g. 13.x, no prebuilt wheel yet): build ORT from source with --use_tensorrt,
#     OR just use the native TRT backend (jetson/21_build_trt_runner.sh) — same TensorRT under the hood,
#     needs only JetPack, and is the validated path (35.7 FPS / 0.3488 mAP50, see DEPLOYMENT_LOG.md).
set -e
cd "$(dirname "$0")"; ROOT="$(cd .. && pwd)"
: "${ORT_ROOT:?set ORT_ROOT to a Jetson ONNXRuntime with the TensorRT EP (see README)}"
[ -f "$ORT_ROOT/include/onnxruntime_cxx_api.h" ] || { echo "no ORT headers at $ORT_ROOT/include"; exit 1; }

cd "$ROOT/cpp"; rm -rf build_ort_trt && mkdir build_ort_trt && cd build_ort_trt
cmake .. -DCMAKE_BUILD_TYPE=Release -DPORTABLE=ON -DUSE_NCNN=OFF -DUSE_TRT=OFF -DUSE_ORT=ON \
         -DONNXRUNTIME_ROOT="$ORT_ROOT" 2>&1 | grep -iE "backend:|error" || true
make -j"$(nproc)" 2>&1 | grep -iE "error|Built target" | tail -2

BIN="$ROOT/cpp/build_ort_trt/yolomaster_edge"
echo
echo "built: $BIN"
echo "run (ORT + TensorRT EP; first run builds+caches the engine in ./trt_engine_cache):"
echo "  $BIN --model $ROOT/jetson/models/esmoe_n_visdrone_sim.onnx --source <img|dir> \\"
echo "       --device trt --classes visdrone --out out"
echo "  # for INT8: --model esmoe_n_visdrone_int8_qdq.onnx  (QDQ nodes drive INT8, FP16 fallback)"
