#!/usr/bin/env bash
# Fetch the CPU test fixtures (EsMoE-N VisDrone ONNX + ncnn dir, the moa-n router-emulated ncnn
# graph, the 50-image VisDrone set with labels) that tests/run_tests.sh and
# tests/run_server_tests.sh expect under models/ and visdrone50/. Release asset, sha256 pinned.
# Usage: scripts/server/fetch_ci_fixtures.sh [repo root]
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")/../.." && pwd)}"
ASSET="${FIXTURES_ASSET:-ci-fixtures-v1.tar.gz}"
URL="${FIXTURES_URL:-https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.1.0/$ASSET}"
SHA="${FIXTURES_SHA:-33ab282614e2378bad475a9e2563c81a26483b8fdb9aaeda14b7e29939eeffbc}"
if [ -f "$ROOT/models/esmoe_n_visdrone_sim.onnx" ] && [ -f "$ROOT/models/moa-n_ncnn/model.ncnn.param" ] && [ -f "$ROOT/visdrone50/visdrone50.yaml" ]; then
  echo "fixtures already present"; exit 0
fi
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
echo "downloading $ASSET"
curl -sSL --retry 3 -o "$TMP/$ASSET" "$URL"
echo "$SHA  $TMP/$ASSET" | sha256sum -c --quiet || { echo "sha256 mismatch for $URL" >&2; exit 3; }
tar xzf "$TMP/$ASSET" -C "$ROOT"
# the dataset yaml carries a relative path; make it absolute for this checkout
sed -i "s#^path: .*#path: $ROOT/visdrone50#" "$ROOT/visdrone50/visdrone50.yaml"
echo "fixtures -> $ROOT/models, $ROOT/visdrone50"
