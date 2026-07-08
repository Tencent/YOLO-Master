#!/usr/bin/env bash
# Jetson Orin Nano (Super) — verify the platform, lock max performance, install build deps.
# Safe to re-run. Run after every reboot to restore the power/clock state.
set -e
cd "$(dirname "$0")"

echo "==================== platform ===================="
if [ -f /etc/nv_tegra_release ]; then head -1 /etc/nv_tegra_release; else echo "  (not a Tegra device?)"; fi
echo "-- CUDA --";     (/usr/local/cuda/bin/nvcc --version 2>/dev/null || nvcc --version 2>/dev/null) | grep -i release || echo "  nvcc not found (add /usr/local/cuda/bin to PATH)"
echo "-- TensorRT --"; dpkg -l 2>/dev/null | grep -iE "tensorrt " | awk '{print "  "$2" "$3}' | head -1 || echo "  (check: dpkg -l | grep tensorrt)"
echo "-- cuDNN --";    dpkg -l 2>/dev/null | grep -iE "libcudnn" | awk '{print "  "$2" "$3}' | head -1
TRTEXEC=$(command -v trtexec || echo /usr/src/tensorrt/bin/trtexec)
echo "-- trtexec --";  [ -x "$TRTEXEC" ] && echo "  $TRTEXEC" || echo "  NOT FOUND (expected /usr/src/tensorrt/bin/trtexec)"

echo "==================== max performance ===================="
echo "  setting MAXN power mode + locking clocks (needs sudo)"
sudo nvpmodel -m 0 || echo "  (nvpmodel -m 0 failed; check available modes with: sudo nvpmodel -q)"
sudo jetson_clocks   || echo "  (jetson_clocks failed)"
sudo nvpmodel -q 2>/dev/null | grep -i "power mode" || true

echo "==================== build deps ===================="
sudo apt-get update -qq
sudo apt-get install -y cmake build-essential libopencv-dev git wget

echo "==================== model check ===================="
if [ -f models/esmoe_n_visdrone_sim.onnx ]; then
  echo "  models/esmoe_n_visdrone_sim.onnx  ✓ ($(du -h models/esmoe_n_visdrone_sim.onnx | cut -f1))"
else
  echo "  ⚠ models/esmoe_n_visdrone_sim.onnx MISSING."
  echo "    scp it from your server:  scp user@host:/data/yolo-master-edge/models/esmoe_n_visdrone_sim.onnx models/"
fi
echo "done. next: bash 10_trt_bench.sh"
