#!/usr/bin/env bash
# One-click run of the published image on any machine with Docker + the NVIDIA container toolkit.
#   bash deploy/run.sh [IMAGE=<user>/yolomaster-api:<SERVER_VERSION>] [PORT=8080]
# Volumes: ./cache (TensorRT engines built on first start), optional ./models (extra models) and
# ./server.json (custom config, mounted over the bundled one).
set -euo pipefail
SV="$(tr -d '[:space:]' < "$(dirname "$0")/../cpp/server/SERVER_VERSION" 2>/dev/null || echo latest)"
IMAGE="${1:-${IMAGE:-yolomaster-api:$SV}}"; PORT="${2:-${PORT:-8080}}"
# YM_API_KEYS=k1,k2 in the environment turns authentication on inside the container
[ -n "${YM_API_KEYS:-}" ] && EXTRA_ENV=(-e "YM_API_KEYS=$YM_API_KEYS") || EXTRA_ENV=()
mkdir -p cache
ARGS=(--gpus all -p "$PORT:8080" -v "$PWD/cache:/opt/yolomaster/cache/trt" --name yolomaster-api --rm -d)
[ -f server.json ] && ARGS+=(-v "$PWD/server.json:/opt/yolomaster/server.json:ro")
[ -d models ] && ARGS+=(-v "$PWD/models:/opt/yolomaster/models/extra:ro")
docker run "${ARGS[@]}" "${EXTRA_ENV[@]}" "$IMAGE"
echo "waiting for /readyz (first start builds TensorRT engines: 1 to 3 minutes)"
for i in $(seq 1 600); do curl -sf "localhost:$PORT/readyz" >/dev/null && break; sleep 1; done
curl -s "localhost:$PORT/readyz"; echo
curl -s "localhost:$PORT/v1/models" | python3 -c "import sys,json; [print(' ', m['id'], m.get('ep',''), 'ready' if m.get('ready') else 'lazy') for m in json.load(sys.stdin)['models']]" 2>/dev/null || true
echo "try: curl -X POST 'localhost:$PORT/v1/infer?model=v01n-trt' --data-binary @image.jpg"
