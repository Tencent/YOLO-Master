#!/usr/bin/env bash
# Package the labelled eval sets the bench mode scores (release assets, never committed):
#   datasets/coco500  (scripts/make_coco_subset.py)   visdrone50/  (50-image VisDrone val cut)
# into dist/eval-sets/<name>.tar.gz + SHA256SUMS, and optionally upload them to a GitHub release.
#   scripts/package_eval_sets.sh                 # tar + checksums only
#   scripts/package_eval_sets.sh --upload v1.1.0 # also: gh release upload <tag> (needs gh auth / GH_TOKEN)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIST="$ROOT/dist/eval-sets"
UPLOAD_TAG=""
[ "${1:-}" = "--upload" ] && UPLOAD_TAG="${2:?tag}"
mkdir -p "$DIST"
pack() {  # <name> <dir>
  local name="$1" dir="$2"
  [ -d "$dir" ] || { echo "skip $name: $dir missing" >&2; return 0; }
  echo "packing $name from $dir"
  tar -C "$(dirname "$dir")" --exclude='*.cache' -czf "$DIST/$name.tar.gz" "$(basename "$dir")"
}
pack eval-set-coco500 "$ROOT/datasets/coco500"
pack eval-set-visdrone50 "$ROOT/visdrone50"
( cd "$DIST" && sha256sum *.tar.gz > SHA256SUMS && cat SHA256SUMS )
if [ -n "$UPLOAD_TAG" ]; then
  command -v gh >/dev/null || { echo "gh not installed" >&2; exit 2; }
  gh release upload "$UPLOAD_TAG" "$DIST"/*.tar.gz "$DIST/SHA256SUMS" --clobber
  echo "uploaded to release $UPLOAD_TAG"
fi
