#!/usr/bin/env bash
# Build the image layer, then push (and/or write the loadable tarball) with Bazel + rules_oci.
#   DOCKERHUB_USER=<user> bash deploy/docker/push.sh [tarball|push|both]   (default: both)
# Docker Hub auth: ~/.docker/config.json with {"auths":{"https://index.docker.io/v1/":{"auth":"<base64 user:token>"}}}
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; MODE="${1:-both}"
[ -f "$HERE/bundle.tar" ] || { echo "bundle.tar missing: run stage_bundle.sh first"; exit 2; }
cd "$HERE"
if [ "$MODE" = tarball ] || [ "$MODE" = both ]; then
  bazel build //:tarball && cp -L "$(bazel cquery --output=files //:tarball 2>/dev/null | head -1)" "$HERE/yolomaster-api.tar" && ls -la "$HERE/yolomaster-api.tar"
fi
if [ "$MODE" = push ] || [ "$MODE" = both ]; then
  [ -n "${DOCKERHUB_USER:-}" ] || { echo "set DOCKERHUB_USER"; exit 2; }
  bazel run //:push -- --repository "index.docker.io/${DOCKERHUB_USER}/yolomaster-api"
fi
