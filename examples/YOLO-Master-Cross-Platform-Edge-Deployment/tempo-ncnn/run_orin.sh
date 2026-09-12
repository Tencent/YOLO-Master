#!/usr/bin/env bash
# Orin CPU tempo: fp16 vs INT8, 1/2/6 threads. Fetches models from the mdb drop
# if absent, builds against the aarch64 ncnn install (jetson/24_build_ncnn.sh).
set -e
cd "$(dirname "$0")"
MDB="${MDB:-root@185.76.11.54:/yolotmp/jetson-p03}"
NCNN="${NCNN:-$(cd .. && pwd)/third_party/ncnn-aarch64-20260526}"

mkdir -p models
for f in p03_v01n.ncnn.param p03_v01n.ncnn.bin p03_v01n-int8.param p03_v01n-int8.bin; do
  [ -f "models/$f" ] || scp "$MDB/$f" "models/$f"
done
[ -f "$NCNN/include/ncnn/net.h" ] || { echo "ncnn install missing: $NCNN (run jetson/24_build_ncnn.sh)"; exit 1; }

cmake -B build -DNCNN_INSTALL="$NCNN" >/dev/null && cmake --build build -j"$(nproc)" >/dev/null
echo "== Orin CPU: fp16-path float model vs INT8 model =="
for T in 1 2 6; do
  ./build/ncnn_tempo models/p03_v01n.ncnn.param  models/p03_v01n.ncnn.bin  "$T" 100 1
  ./build/ncnn_tempo models/p03_v01n-int8.param  models/p03_v01n-int8.bin  "$T" 100 1
done
echo "== fp32 reference (fp16 path off), 6 threads =="
./build/ncnn_tempo models/p03_v01n.ncnn.param models/p03_v01n.ncnn.bin 6 100 0
