#!/usr/bin/env bash
# Fetch the pinned uWebSockets + uSockets sources into third_party/uWebSockets (gitignored, like
# every other SDK under third_party/). Used by cpp/server/CMakeLists.txt.
#   uWebSockets v20.80.0 (Apache-2.0), uSockets @ 86097c490263ab662d62e8e7b541390bdec7d149 (Apache-2.0)
set -euo pipefail
UWS_TAG="${UWS_TAG:-v20.80.0}"
US_SHA="${US_SHA:-86097c490263ab662d62e8e7b541390bdec7d149}"
DST="${1:-$(cd "$(dirname "$0")/../.." && pwd)/third_party/uWebSockets}"
if [ -f "$DST/VENDORED_VERSION" ] && [ "$(cat "$DST/VENDORED_VERSION")" = "$UWS_TAG" ] && [ -f "$DST/uSockets/src/libusockets.h" ]; then
  echo "uWebSockets $UWS_TAG already present at $DST"; exit 0
fi
rm -rf "$DST"; mkdir -p "$DST"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
curl -sSL "https://github.com/uNetworking/uWebSockets/archive/refs/tags/${UWS_TAG}.tar.gz" | tar xz -C "$TMP"
cp -r "$TMP/uWebSockets-${UWS_TAG#v}/src" "$TMP/uWebSockets-${UWS_TAG#v}/LICENSE" "$TMP/uWebSockets-${UWS_TAG#v}/README.md" "$DST/"
mkdir -p "$DST/uSockets"
curl -sSL "https://github.com/uNetworking/uSockets/archive/${US_SHA}.tar.gz" | tar xz -C "$TMP"
cp -r "$TMP/uSockets-${US_SHA}/src" "$TMP/uSockets-${US_SHA}/LICENSE" "$DST/uSockets/"
echo "$UWS_TAG" > "$DST/VENDORED_VERSION"; echo "$US_SHA" > "$DST/uSockets/VENDORED_COMMIT"
echo "fetched uWebSockets $UWS_TAG + uSockets ${US_SHA:0:12} -> $DST"
