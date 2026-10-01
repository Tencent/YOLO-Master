#!/usr/bin/env bash
# macOS battery for the v1.2.0 Core ML runner: the Mac counterparts of tests/run_tests.sh T19-T21
# plus the core handshake. Run ON A MAC from the repo root (or anywhere: paths are absolute).
#
#   MODEL=/path/to/v0.1-N.mlpackage COCO500=/path/to/coco500 bash mac/tests/run_mac_tests.sh
#
# Env:  MODEL     .mlpackage / .mlmodelc (default: mac/Resources/v0.1-seg-N.mlpackage, detect models score;
#                 the seg default is fine for M1, M2, M4 but M3 wants a detect model such as v0.1-N)
#       COCO500   extracted eval-set-coco500.tar.gz (images/ + labels/ + coco500.yaml); M3 skipped without it
#       IMAGES    any image folder for the smoke passes (default: $COCO500/images, else mac/tests/fixtures)
#       PAN_MP4   synthetic pan clip (default: generated here with ffmpeg from the first image, if present)
#       COMPUTE   cpuAndGPU | all | cpu (default cpuAndGPU)
#       SKIP_BUILD=1 to reuse the last build.
# Needs: Xcode command line tools (swift), python3 with numpy (for scripts/eval_map_standalone.py), ffmpeg
# (brew install ffmpeg) for the synthetic clips.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
MODEL=${MODEL:-$ROOT/mac/Resources/v0.1-seg-N.mlpackage}
COCO500=${COCO500:-}
COMPUTE=${COMPUTE:-cpuAndGPU}
OUT=$(mktemp -d /tmp/ymtests.XXXXXX)
P=0; F=0; S=0
ok(){ echo "  PASS  $1"; P=$((P+1)); }
no(){ echo "  FAIL  $1"; F=$((F+1)); }
skip(){ echo "  SKIP  $1"; S=$((S+1)); }

echo "== build =="
if [ "${SKIP_BUILD:-0}" != "1" ]; then
  swift build -c release --package-path "$ROOT/mac" --product yolomaster-coreml > "$OUT/build.log" 2>&1 \
    && ok "M1 swift build (release, CLI)" || { no "M1 swift build"; tail -40 "$OUT/build.log"; exit 1; }
fi
BIN=$(swift build -c release --package-path "$ROOT/mac" --show-bin-path)/yolomaster-coreml
[ -x "$BIN" ] || { echo "no binary at $BIN"; exit 1; }
run(){ "$BIN" "$@" 2>&1; }

echo "== portable core handshake =="
run --core-selftest > "$OUT/selftest.txt" && grep -q "SELFTEST OK" "$OUT/selftest.txt" \
  && ok "M1b --core-selftest (mAP, stats, sha256, both trackers agree with the Linux core)" \
  || { no "M1b --core-selftest"; cat "$OUT/selftest.txt"; }

# ---- images for the smoke passes ----
if [ -n "$COCO500" ] && [ -d "$COCO500/images" ]; then IMAGES=${IMAGES:-$COCO500/images}
else IMAGES=${IMAGES:-$HERE/fixtures}; fi
if [ ! -d "$IMAGES" ] || [ -z "$(ls "$IMAGES"/*.jpg 2>/dev/null | head -1)" ]; then
  echo "no images (set COCO500 or IMAGES)"; exit 1
fi
IMG=$(ls "$IMAGES"/*.jpg | sort | head -1)

echo "== bench mode =="
rm -rf "$OUT/bench"
run --model "$MODEL" --source "$IMAGES" --limit 4 --compute "$COMPUTE" --no-save \
    --bench cold --bench-iters 5 --bench-warmup 2 --bench-json "$OUT/bench/b.json" > "$OUT/bench_out.txt"
if grep -q "frames=4" "$OUT/bench_out.txt" \
   && python3 "$ROOT/scripts/bench_schema_check.py" "$OUT/bench/b.json" --iters 5 --frames 4 --images 4 > /dev/null \
   && python3 -c "import json,sys; j=json.load(open(sys.argv[1])); assert j['tool']=='macos' and j['model']['runtime']=='coreml' and j['environment']['version']" "$OUT/bench/b.json"
then ok "M2 --bench cold writes a valid yolomaster-bench/v1 JSON (tool=macos, [summary] intact)"
else no "M2 bench JSON"; cat "$OUT/bench_out.txt" | tail -5; fi
grep "^\[summary\]" "$OUT/bench_out.txt"

echo "== save-txt format =="
rm -rf "$OUT/txt"
run --model "$MODEL" --source "$IMAGES" --limit 3 --compute "$COMPUTE" --no-save --save-txt "$OUT/txt" > /dev/null
if [ "$(ls "$OUT/txt"/*.txt 2>/dev/null | wc -l | tr -d ' ')" = "3" ] \
   && python3 - "$OUT/txt" <<'PY'
import glob, re, sys
num = r"-?\d+(?:\.\d+)?(?:e[+-]?\d+)?"
line = re.compile(rf"^\d+ {num} {num} {num} {num} {num}$")
rows = [l.rstrip("\n") for f in sorted(glob.glob(sys.argv[1] + "/*.txt")) for l in open(f)]
assert all(line.match(r) for r in rows), [r for r in rows if not line.match(r)][:3]
for r in rows:                                   # six significant digits at most, like the C++ writer
    for tok in r.split()[1:]:
        digits = re.sub(r"[-.e+]", "", tok.split("e")[0]).lstrip("0")
        assert len(digits) <= 6, tok
PY
then ok "M2b --save-txt: 'class conf x1 y1 x2 y2' lines, six significant digits"; else no "M2b save-txt"; fi

echo "== in-process accuracy (coco500) =="
if [ -n "$COCO500" ] && [ -d "$COCO500/labels" ]; then
  rm -rf "$OUT/acc" "$OUT/valtxt"
  run --model "$MODEL" --source "$COCO500/images" --compute "$COMPUTE" --no-save \
      --accuracy "$COCO500/labels" --bench-iters 1 --bench-warmup 0 \
      --save-txt "$OUT/valtxt" --bench-json "$OUT/acc/acc.json" > "$OUT/acc_out.txt" 2>&1
  GOT=$(grep "^\[accuracy\]" "$OUT/acc_out.txt" | sed 's/^\[accuracy\] //')
  REF=$(python3 "$ROOT/scripts/eval_map_standalone.py" --preds "$OUT/valtxt/val" --images "$COCO500/images" --labels "$COCO500/labels" 2>/dev/null | grep "^images=")
  # ultralytics (and the standalone port of it) sorts confidences and IoU matches with an UNSTABLE
  # np.argsort; Core ML fp16 outputs carry many exact ties after the six-digit rounding, so the
  # reference's tie order is implementation-defined and the 4th decimal can differ (Linux fp32
  # outputs have no ties, which is why T20 is exact). Agreement within 0.0005 is the Mac contract.
  if [ -n "$REF" ] && python3 - "$REF" "$GOT" <<'PY'
import re, sys
def parse(s):
    m = re.search(r"images=(\d+)\s+mAP50=([\d.]+)\s+mAP50-95=([\d.]+)", s); return int(m.group(1)), float(m.group(2)), float(m.group(3))
(ri, r50, r95), (gi, g50, g95) = parse(sys.argv[1]), parse(sys.argv[2])
assert ri == gi and abs(r50 - g50) <= 0.0005 and abs(r95 - g95) <= 0.0005, (sys.argv[1], sys.argv[2])
PY
  then ok "M3 in-process accuracy agrees with eval_map_standalone.py on the val dump within 0.0005 (got $GOT; ref $REF)"
  else echo "  ref: $REF"; echo "  got: $GOT"; no "M3 accuracy"; fi
  python3 "$ROOT/scripts/bench_schema_check.py" "$OUT/acc/acc.json" --accuracy > /dev/null && ok "M3b accuracy block in the bench JSON" || no "M3b"
else skip "M3 (set COCO500=/path/to/coco500 with labels/)"; fi

echo "== multi-object tracking (synthetic pan clips) =="
PAN=${PAN_MP4:-}
if [ -z "$PAN" ] && command -v ffmpeg > /dev/null; then
  # 30 frames: a crop window sliding LEFT/UP over the still image makes the scene move RIGHT by 3 px
  # per frame (the T21 clip) and, in the second clip, right 3 px + down 2 px per frame
  ffmpeg -loglevel error -y -loop 1 -framerate 10 -i "$IMG" -vf "scale=640:480,crop=540:420:'90-3*n':30,pad=640:480:50:30:color=#727272" \
      -frames:v 30 -r 10 -pix_fmt yuv420p "$OUT/pan.mp4" && PAN="$OUT/pan.mp4"
  ffmpeg -loglevel error -y -loop 1 -framerate 10 -i "$IMG" -vf "scale=640:480,crop=540:400:'90-3*n':'60-2*n',pad=640:480:50:40:color=#727272" \
      -frames:v 30 -r 10 -pix_fmt yuv420p "$OUT/pan_xy.mp4" > /dev/null 2>&1
fi
if [ -n "$PAN" ] && [ -f "$PAN" ]; then
  for TM in bytetrack botsort; do
    rm -rf "$OUT/trk_$TM"
    run --model "$MODEL" --source "$PAN" --compute "$COMPUTE" --track $TM --save-txt "$OUT/trk_$TM" --no-save > "$OUT/trk_$TM.log"
    # persistence: the most persistent id must be present on >= 27 of the 30 frames (the scene only
    # translates, so a tracked object never leaves; the Linux tracker keeps several ids on all 30)
    if grep -q "frames=30" "$OUT/trk_$TM.log" && python3 - "$OUT/trk_$TM" <<'PY'
import glob, sys, collections
files = sorted(glob.glob(sys.argv[1] + "/*.txt")); presence = collections.Counter(); bad = 0
for f in files:
    rows = [l.split() for l in open(f) if l.strip()]
    bad += sum(1 for r in rows if len(r) != 7)
    presence.update({r[6] for r in rows})
assert len(files) == 30 and bad == 0 and presence, (len(files), bad)
best, frames = presence.most_common(1)[0]
print(f"  most persistent id {best}: {frames}/30 frames, {len(presence)} ids total")
assert frames >= 27, f"most persistent id present on only {frames}/30 frames"
assert len(presence) >= 2
PY
    then ok "M4 --track $TM: 7-column txt, one id persists on >= 27/30 frames"; else no "M4 $TM"; tail -3 "$OUT/trk_$TM.log"; fi
  done
  if [ -f "$OUT/pan_xy.mp4" ]; then
    YM_MOTION_DEBUG=1 run --model "$MODEL" --source "$OUT/pan_xy.mp4" --compute "$COMPUTE" --track botsort --no-save > "$OUT/motion.log"
    # the scene moves +3 px in x and +2 px in y per frame (top-down pixels): the estimate must have both signs positive
    if python3 - "$OUT/motion.log" <<'PY'
import re, sys, statistics
tx, ty = [], []
for l in open(sys.argv[1]):
    m = re.match(r"\[motion\] frame \d+ tx=(\S+) ty=(\S+)", l)
    if m: tx.append(float(m.group(1))); ty.append(float(m.group(2)))
assert len(tx) >= 20, len(tx)
mx, my = statistics.median(tx), statistics.median(ty)
print(f"  median motion tx={mx:.2f} ty={my:.2f} (expected about +3, +2 in top-down pixels)")
assert 1.5 < mx < 4.5 and 0.5 < my < 3.5, (mx, my)
PY
    then ok "M4b Vision camera motion: sign and magnitude match the diagonal pan"; else no "M4b camera motion (check the y-axis convention in Motion.swift)"; fi
  fi
else skip "M4 (no PAN_MP4 and no ffmpeg)"; fi

echo "== GPU (Metal) preprocessing =="
rm -rf "$OUT/dump_gpu" "$OUT/dump_cpu"
run --model "$MODEL" --source "$IMAGES" --limit 10 --compute "$COMPUTE" --no-save --dump-input "$OUT/dump_gpu" > "$OUT/dump_gpu.log"
run --model "$MODEL" --source "$IMAGES" --limit 10 --compute "$COMPUTE" --no-save --cpu-preproc --dump-input "$OUT/dump_cpu" > "$OUT/dump_cpu.log"
if grep -q "preproc=gpu" "$OUT/dump_gpu.log" && [ "$(ls "$OUT/dump_gpu"/*.f32 2>/dev/null | wc -l | tr -d ' ')" = "10" ]; then
  ok "M5 Metal preprocessing active (10 input tensors dumped, preproc=gpu)"
  grep "^\[summary\]" "$OUT/dump_gpu.log" | sed 's/^/  gpu /'; grep "^\[summary\]" "$OUT/dump_cpu.log" | sed 's/^/  cpu /'
  if python3 -c "import cv2, numpy" 2>/dev/null; then
    python3 "$ROOT/scripts/preproc_compare.py" "$IMAGES" "$OUT/dump_gpu" --imgsz "$(python3 -c "import json;print(json.load(open('$OUT/bench/b.json'))['model']['imgsz'])")" > "$OUT/cmp_gpu.txt" 2>&1 \
      && ok "M5b Metal tensor within 1/255 of the Linux preprocess_nchw rule on the same decoded pixels ($(tail -1 "$OUT/cmp_gpu.txt"))" \
      || { no "M5b Metal tensor parity"; tail -4 "$OUT/cmp_gpu.txt"; }
  else
    skip "M5b (no cv2 here: copy $OUT/dump_gpu to a box with opencv and run scripts/preproc_compare.py $IMAGES <dump_gpu>)"
  fi
else no "M5 Metal preprocessing"; tail -3 "$OUT/dump_gpu.log"; fi
if [ -n "$COCO500" ] && [ -d "$COCO500/labels" ] && [ -f "$OUT/acc_out.txt" ]; then
  run --model "$MODEL" --source "$COCO500/images" --compute "$COMPUTE" --no-save --cpu-preproc \
      --accuracy "$COCO500/labels" --bench-iters 1 --bench-warmup 0 --bench-json "$OUT/acc/acc_cpu.json" > "$OUT/acc_cpu_out.txt" 2>&1
  if python3 - "$OUT/acc/acc.json" "$OUT/acc/acc_cpu.json" <<'PY'
import json, sys
g, c = [json.load(open(p))["accuracy"]["map5095"] for p in sys.argv[1:]]
print(f"  mAP50-95 gpu-preproc {g:.4f} vs cpu-preproc {c:.4f} (diff {g - c:+.4f})")
assert abs(g - c) <= 0.001, "preprocessing device moved mAP50-95 by more than 0.001"
PY
  then ok "M5c coco500 mAP50-95 unchanged between Metal and CPU preprocessing (within 0.001)"; else no "M5c mAP gpu vs cpu"; fi
else skip "M5c (needs COCO500)"; fi

echo "RESULT: $P passed, $F failed, $S skipped   (outputs in $OUT)"
[ "$F" = "0" ]
