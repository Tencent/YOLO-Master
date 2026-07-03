#!/usr/bin/env bash
# Download + extract the ONNX Runtime C/C++ release archive for the current arch.
# Version matches the python onnxruntime wheel used for export (1.23.2).
set -euo pipefail

VERSION=${ORT_VERSION:-1.23.2}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${HERE}/third_party/onnxruntime"
MIRROR=${ORT_MIRROR:-https://github.com/microsoft/onnxruntime/releases/download}

uname_m="$(uname -m)"
case "${uname_m}" in
  x86_64)  ART="onnxruntime-linux-x64-${VERSION}.tgz" ;;
  aarch64|arm64) ART="onnxruntime-linux-aarch64-${VERSION}.tgz" ;;
  *) echo "unsupported arch: ${uname_m}" >&2; exit 1 ;;
esac

if [ -f "${DEST}/include/onnxruntime_cxx_api.h" ]; then
  echo "[setup_ort] already present at ${DEST}"; exit 0
fi

URL="${MIRROR}/v${VERSION}/${ART}"
echo "[setup_ort] downloading ${URL}"
TMP="$(mktemp -d)"
curl -fL "${URL}" -o "${TMP}/${ART}"
tar -xzf "${TMP}/${ART}" -C "${TMP}"
rm -rf "${DEST}"; mkdir -p "${DEST}"
mv "${TMP}"/onnxruntime-*/* "${DEST}/"
rm -rf "${TMP}"
echo "[setup_ort] installed -> ${DEST}"
ls "${DEST}/include/onnxruntime_cxx_api.h" "${DEST}/lib/"libonnxruntime.so* && echo OK
