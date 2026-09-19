#!/usr/bin/env bash
# STAL small-object label-assignment smoke test — one-command reproduction.
#
# Usage:
#   bash A2/smoke/reproduce.sh          # GPU (device=0)
#   bash A2/smoke/reproduce.sh cpu      # CPU fallback
#
# Effect:
#   1. Generates the tiny VisDrone-smoke subset (A2/smoke/smoke_test.py make_subset)
#   2. Trains YOLO-Master v0.1-N (MoE) from scratch for 1 epoch (data -> assigner -> loss -> val)
#   3. Prints the metric log line (results.csv) and the run directory.
set -euo pipefail
cd "$(dirname "$0")/../.."
DEVICE="${1:-0}"
python A2/smoke/smoke_test.py --device "$DEVICE"
