#!/usr/bin/env bash
# Build TensorRT engines from the ONNX and report GPU throughput on the Orin.
# FP16 = the safe deploy precision; INT8 = the tensor-core speed ceiling (uncalibrated here).
set -e
cd "$(dirname "$0")"
M=models/esmoe_n_visdrone_sim.onnx
# locate trtexec across the common JetPack paths (env override wins)
TRTEXEC="${TRTEXEC:-}"
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
  for c in "$(command -v trtexec 2>/dev/null)" \
           /usr/src/tensorrt/bin/trtexec \
           /usr/src/tensorrt/bin/aarch64-linux-gnu/trtexec \
           "$(find /usr -name trtexec -type f 2>/dev/null | head -1)"; do
    [ -n "$c" ] && [ -x "$c" ] && { TRTEXEC="$c"; break; }
  done
fi
[ -f "$M" ]        || { echo "missing $M — see 00_setup.sh"; exit 1; }
if [ -z "$TRTEXEC" ] || [ ! -x "$TRTEXEC" ]; then
  echo "trtexec not found. Build it:  cd /usr/src/tensorrt/samples/trtexec && sudo make -j\$(nproc)"
  echo "or install it:  sudo apt install -y tensorrt   (then re-run)"
  exit 1
fi
echo "using trtexec: $TRTEXEC"
mkdir -p models engines

# TensorRT builder workspace (MB). 4GB Orin Nano: use 256 and build headless
# (sudo systemctl isolate multi-user.target) or the builder OOMs on tactic profiling.
WORKSPACE="${WORKSPACE:-512}"
OPT="${OPT:-2}"           # builder optimization level 0-5 (lower=faster build; 2 is plenty for this model)
FREE_MB=$(free -m 2>/dev/null | awk '/Mem:/{print $7}')
if [ -n "$FREE_MB" ] && [ "$FREE_MB" -lt 1200 ]; then
  echo "  ⚠ only ${FREE_MB}MB RAM free — low. Go headless: sudo systemctl isolate multi-user.target"
  echo "    and/or lower the workspace:  WORKSPACE=256 bash $0"
fi
echo "  builder workspace: ${WORKSPACE} MB (override with WORKSPACE=<mb>)"

bench() {  # name  extra-flags
  local name="$1"; shift
  local log="engines/trtexec_${name}.log"
  echo "==================== $name (building — full log streams below) ===================="
  # stream the FULL trtexec output (so engine-build progress is visible) AND save it
  "$TRTEXEC" --onnx="$M" --saveEngine="engines/esmoe_n_${name}.engine" \
             --memPoolSize=workspace:"${WORKSPACE}" \
             --builderOptimizationLevel="${OPT}" "$@" 2>&1 | tee "$log"
  echo "-------------------- $name RESULT --------------------"
  grep -iE "Throughput|GPU Compute Time:|Latency: min|Total Host Walltime|error|failed|Engine built" "$log" \
    | tail -10 | sed 's/^/  /'
  echo
}

bench fp16  --fp16
bench int8  --int8 --fp16          # INT8 kernels where available, FP16 fallback (dynamic ranges)

echo
echo "==================== summary ===================="
echo "  Throughput = frames/sec (qps). GPU Compute mean = per-inference ms."
echo "  Engines saved to engines/.  FP16 is the deploy engine; for accurate INT8 build from a"
echo "  calibrated model with the detection head pinned to FP16 (see TECHNICAL_REPORT.md §3)."
