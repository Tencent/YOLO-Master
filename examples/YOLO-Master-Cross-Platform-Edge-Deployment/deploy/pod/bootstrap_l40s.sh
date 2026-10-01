#!/usr/bin/env bash
# Idempotent toolchain bootstrap for a bare RunPod L40S container (Ubuntu 22.04, CUDA 12.x driver).
# Installs: build tools, TensorRT 10.9 (dev + runtime), cuDNN 9, cmake/ninja (pip), bazelisk, rsync.
# Usage: bash deploy/pod/bootstrap_l40s.sh [TRT_VERSION]   (default 10.9.0.34-1+cuda12.8)
set -euo pipefail
TRT_VER="${1:-10.16.1.11-1+cuda12.9}"
# CUDA minor the TensorRT build was linked against (…+cuda12.9 -> 12-9): its runtime libs are
# installed beside whatever toolkit the pod image ships (12.4 on the RunPod L40S image).
CUDA_MM="$(echo "$TRT_VER" | sed -nE 's/.*\+cuda([0-9]+)\.([0-9]+)$/\1-\2/p')"
CUDA_DOT="${CUDA_MM/-/.}"
export DEBIAN_FRONTEND=noninteractive
log() { echo "[bootstrap $(date +%H:%M:%S)] $*"; }
echo $$ > /root/bootstrap.pid; trap 'rm -f /root/bootstrap.pid' EXIT

log "apt update"
apt-get update -qq
log "apt base tools"
apt-get install -y -qq --no-install-recommends \
  build-essential ninja-build patchelf zlib1g-dev pkg-config rsync unzip wget curl ca-certificates \
  libssl-dev git python3-dev python3-pip jq bc \
  libavcodec58 libavformat58 libavutil56 libswscale5 libswresample3 \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev libswresample-dev >/dev/null   # opencv-lean videoio (ffmpeg 4.4 ABI + headers to rebuild it)

if ! dpkg -s libnvinfer-dev >/dev/null 2>&1; then
  log "TensorRT ${TRT_VER}"
  apt-cache madison libnvinfer-dev > /tmp/trt_versions.txt   # (no grep -q in a pipefail pipeline)
  grep -q "${TRT_VER}" /tmp/trt_versions.txt || { log "pinned TRT version not in repo; available 10.x:"; grep -E " 10\." /tmp/trt_versions.txt | head -5; exit 3; }
  apt-get install -y -qq --no-install-recommends \
    "libnvinfer10=${TRT_VER}" "libnvinfer-plugin10=${TRT_VER}" "libnvonnxparsers10=${TRT_VER}" \
    "libnvinfer-headers-dev=${TRT_VER}" "libnvinfer-headers-plugin-dev=${TRT_VER}" "libnvinfer-safe-headers-dev=${TRT_VER}" \
    "libnvinfer-dev=${TRT_VER}" "libnvinfer-plugin-dev=${TRT_VER}" "libnvonnxparsers-dev=${TRT_VER}" \
    "libnvinfer-dispatch10=${TRT_VER}" "libnvinfer-lean10=${TRT_VER}" "libnvinfer-vc-plugin10=${TRT_VER}" >/dev/null
  apt-mark hold libnvinfer10 libnvinfer-dev libnvinfer-headers-dev >/dev/null
else
  log "TensorRT already installed: $(dpkg -s libnvinfer-dev | awk '/^Version/{print $2}')"
fi

if ! dpkg -s libcudnn9-cuda-12 >/dev/null 2>&1; then
  log "cuDNN 9 (ORT CUDA EP)"
  apt-get install -y -qq --no-install-recommends libcudnn9-cuda-12 libcudnn9-dev-cuda-12 >/dev/null
fi

# CUDA runtime libs matching the TensorRT build (cudart/cublas/nvrtc), beside the pod toolkit.
if ! ls /usr/local/cuda-${CUDA_DOT}/lib64/libcudart.so.12 >/dev/null 2>&1; then
  log "CUDA ${CUDA_DOT} runtime libs beside the existing toolkit"
  apt-get install -y -qq --no-install-recommends cuda-cudart-${CUDA_MM} libcublas-${CUDA_MM} cuda-nvrtc-${CUDA_MM} \
    cuda-cudart-dev-${CUDA_MM} cuda-crt-${CUDA_MM} cuda-nvcc-${CUDA_MM} cuda-cccl-${CUDA_MM} \
    libcusolver-dev-${CUDA_MM} libcublas-dev-${CUDA_MM} libcusparse-dev-${CUDA_MM} libcurand-dev-${CUDA_MM} cuda-nvrtc-dev-${CUDA_MM} >/dev/null   # headers + nvcc + cuSOLVER (MNN CUDA backend)
fi
echo "/usr/local/cuda-${CUDA_DOT}/lib64" > /etc/ld.so.conf.d/cuda-trt.conf && ldconfig

log "pip tools"
pip install -q --upgrade "cmake>=3.27" ninja onnx onnxconverter-common "onnxruntime>=1.18,<1.21" websocket-client requests pyyaml \
  opencv-python-headless pytest psutil >/dev/null   # scoring (eval_map.py via ultralytics), server tests

if ! command -v bazel >/dev/null; then
  log "bazelisk -> /usr/local/bin/bazel"
  wget -q https://github.com/bazelbuild/bazelisk/releases/download/v1.28.1/bazelisk-linux-amd64 -O /usr/local/bin/bazel
  chmod +x /usr/local/bin/bazel
fi

log "versions"
cmake --version | head -1; ninja --version; bazel --version 2>/dev/null || true
dpkg -s libnvinfer-dev | awk '/^Version/{print "tensorrt", $2}'
dpkg -s libcudnn9-cuda-12 | awk '/^Version/{print "cudnn", $2}'
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
log "done"
