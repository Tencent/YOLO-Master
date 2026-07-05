#!/usr/bin/env bash
# Assemble a self-contained, relocatable Linux x86_64 bundle of yolomaster_edge.
# Bundles the full transitive .so closure (except the glibc/loader core, which must
# match the target kernel) into lib/, and rewrites rpaths to $ORIGIN so the binary
# runs on any glibc>=build-host x86_64 with no install and no LD_LIBRARY_PATH.
#   usage: package_linux.sh <path-to-built-yolomaster_edge>
set -euo pipefail
BIN="${1:?usage: package_linux.sh <path-to-yolomaster_edge>}"
ROOT="/data/yolo-master-edge"
DIST="$ROOT/dist/linux-x64"
rm -rf "$DIST"; mkdir -p "$DIST/lib" "$DIST/models"
cp "$BIN" "$DIST/yolomaster_edge"

# glibc / dynamic-loader core: MUST come from the target system, never bundle.
EXCLUDE='libc\.so|libm\.so|libdl\.so|librt\.so|libpthread\.so|ld-linux|libresolv\.so|linux-vdso'

# copy the full transitive closure (ldd) except the excluded core into lib/
ldd "$DIST/yolomaster_edge" | awk '{print $3}' | grep -E '^/' | sort -u | while read -r so; do
  base="$(basename "$so")"
  echo "$base" | grep -qE "$EXCLUDE" && continue
  cp -L "$so" "$DIST/lib/$base"
done

# rpath: binary searches ./lib ; each bundled lib searches its own dir for siblings
patchelf --set-rpath '$ORIGIN/lib' "$DIST/yolomaster_edge"
for l in "$DIST"/lib/*.so*; do patchelf --set-rpath '$ORIGIN' "$l" 2>/dev/null || true; done

# deploy models (ONNX + ncnn) so the bundle runs out of the box
cp "$ROOT/models/esmoe_n_visdrone_sim.onnx" "$DIST/models/"
cp -r "$ROOT/models/esmoe_n_visdrone_ncnn" "$DIST/models/"

cat > "$DIST/README.txt" <<'EOF'
YOLO-Master-EsMoE-N edge runner -- portable Linux x86_64 bundle (CPU; ONNX + ncnn).
Self-contained: runs on any glibc>=2.35 (Ubuntu 22.04+) x86_64, no install needed.
  ./yolomaster_edge --model models/esmoe_n_visdrone_sim.onnx --source <img|dir|video> --out out
  ./yolomaster_edge --model models/esmoe_n_visdrone_ncnn      --source <img|dir|video> --out out
Flags: --conf --iou --imgsz --multi-label --no-save --quiet  (--help for all).
EOF

tar czf "$ROOT/dist/yolomaster_edge-linux-x64.tar.gz" -C "$ROOT/dist" linux-x64
echo "libs bundled: $(ls "$DIST/lib" | wc -l)"
echo "bundle : $DIST"
echo "tarball: $ROOT/dist/yolomaster_edge-linux-x64.tar.gz ($(du -h "$ROOT/dist/yolomaster_edge-linux-x64.tar.gz" | cut -f1))"
