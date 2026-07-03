#!/usr/bin/env bash
# Portability proof: cross-compile the C++ sources to an ARM64 (aarch64) object file
# on the x86_64 host, using the conda-forge aarch64 cross-compiler (no sudo needed).
# A full link would require aarch64 builds of OpenCV/ORT; the native ARM64 build is
# covered by Dockerfile.arm64. This check proves the code has no x86-only assumptions.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ORT_INC="${ONNXRUNTIME_ROOT_DIR:-${HERE}/third_party/onnxruntime}/include"
CP="${CONDA_PREFIX:-$(conda info --base 2>/dev/null)/envs/yolomaster}"
INC_FLAGS="-I${HERE}/include -I${ORT_INC} -I${CP}/include/opencv4 -I${CP}/include"

CXX=$(command -v aarch64-conda-linux-gnu-g++ || command -v aarch64-linux-gnu-g++)
[ -z "${CXX}" ] && { echo "no aarch64 cross-compiler found"; exit 1; }
echo "[cross] using ${CXX}"

OUT="${HERE}/build-cross/main.aarch64.o"
mkdir -p "$(dirname "${OUT}")"
"${CXX}" -std=c++17 -O3 -c "${HERE}/src/main.cpp" ${INC_FLAGS} -o "${OUT}"
echo "[cross] compiled -> ${OUT}"
file "${OUT}"
file "${OUT}" | grep -q "aarch64\|ARM aarch64" && echo "[cross] OK: ARM64 object produced" || { echo "[cross] FAIL"; exit 2; }
