#!/usr/bin/env bash
# Container entrypoint: engines for TensorRT models are built on first start into YM_ENGINE_CACHE
# (mount it as a volume to keep them), then the server runs in the foreground (PID 1 friendly).
set -euo pipefail
cd /opt/yolomaster
mkdir -p "${YM_ENGINE_CACHE:-/opt/yolomaster/cache/trt}"
if [ "${1:-}" = "cli" ]; then shift; exec /opt/yolomaster/bin/yolomaster_edge "$@"; fi
if [ "${1:-}" = "shell" ]; then exec /bin/bash; fi
exec /opt/yolomaster/bin/yolomaster_server --config "${YM_CONFIG:-/opt/yolomaster/server.json}" \
  --engine-cache "${YM_ENGINE_CACHE:-/opt/yolomaster/cache/trt}" "$@"
