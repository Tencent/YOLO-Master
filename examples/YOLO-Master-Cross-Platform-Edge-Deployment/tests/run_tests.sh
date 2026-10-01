#!/usr/bin/env bash
# Robustness battery for yolomaster_edge. Re-runnable on any platform (x86_64 / Jetson).
# Usage: BIN=./build/yolomaster_edge ONNX=... NCNN=... DIR=... YAML=... ./run_tests.sh
set -u
ROOT=${ROOT:-$(cd "$(dirname "$0")/.." && pwd)}
BIN=${BIN:-$ROOT/cpp/build/yolomaster_edge}
ONNX=${ONNX:-$ROOT/models/esmoe_n_visdrone_sim.onnx}
NCNN=${NCNN:-$ROOT/models/esmoe_n_visdrone_ncnn}
DIR=${DIR:-$ROOT/visdrone50/images/val}
YAML=${YAML:-$ROOT/visdrone50/visdrone50.yaml}
OUT=$(mktemp -d)
IMG=$(ls "$DIR"/*.jpg | sort | head -1)
P=0; F=0
ok(){ echo "  PASS  $1"; P=$((P+1)); }
no(){ echo "  FAIL  $1"; F=$((F+1)); }
run(){ "$BIN" "$@" 2>&1; }

# python with cv2 for the synthetic videos (PYCV=... to point at a venv; skips the video tests otherwise)
PYCV=${PYCV:-python3}
"$PYCV" -c "import cv2" 2>/dev/null || PYCV=""
# build a 6-frame test video if opencv-python is present
[ -n "$PYCV" ] && "$PYCV" - "$DIR" "$OUT/test.mp4" <<'PY' 2>/dev/null || true
import cv2,glob,sys
imgs=sorted(glob.glob(sys.argv[1]+"/*.jpg"))[:6]
vw=cv2.VideoWriter(sys.argv[2],cv2.VideoWriter_fourcc(*'mp4v'),5,(640,480))
for p in imgs: vw.write(cv2.resize(cv2.imread(p),(640,480)))
vw.release()
PY

echo "== sources & auto-detection =="
run -m "$ONNX" -s "$IMG" --no-save | grep -q "backend=onnx.*model-metadata" && ok "T1 onnx auto backend+classes" || no T1
run -m "$NCNN" -s "$IMG" --no-save | grep -q "backend=ncnn.*model-metadata" && ok "T2 ncnn auto backend+classes" || no T2
run -m "$ONNX" -s "$DIR" --limit 4 --quiet --no-save | grep -q "frames=4" && ok "T3 directory source" || no T3
run -m "$NCNN" -s "$YAML" --limit 3 --quiet --no-save | grep -q "frames=3" && ok "T4 dataset.yaml source" || no T4
[ -f "$OUT/test.mp4" ] && { run -m "$ONNX" -s "$OUT/test.mp4" --quiet --no-save | grep -q "frames=6" && ok "T5 video source" || no T5; } || echo "  SKIP  T5 (no video)"

echo "== bench mode + in-process accuracy =="
rm -rf "$OUT/bench" "$OUT/valtxt"
run -m "$ONNX" -s "$DIR" --limit 4 --quiet --no-save --bench cold --bench-iters 5 --bench-warmup 2 --bench-json "$OUT/bench/b.json" > "$OUT/bench_out.txt" 2>&1
if grep -q "frames=4" "$OUT/bench_out.txt" && python3 "$ROOT/scripts/bench_schema_check.py" "$OUT/bench/b.json" --iters 5 --frames 4 --images 4 > /dev/null \
   && python3 -c "import json,sys; assert json.load(open(sys.argv[1]))['environment']['version']" "$OUT/bench/b.json"
then ok "T19 --bench cold writes a valid yolomaster-bench/v1 JSON ([summary] intact; scripts/bench_schema_check.py)"; else no T19; fi
LABELS="$ROOT/visdrone50/labels/val"
if [ -d "$LABELS" ]; then
  run -m "$ONNX" -s "$DIR" --quiet --no-save --conf 0.001 --iou 0.7 --multi-label --save-txt "$OUT/valtxt" > /dev/null 2>&1
  REF=$(python3 "$ROOT/scripts/eval_map_standalone.py" --preds "$OUT/valtxt" --images "$DIR" --labels "$LABELS" 2>/dev/null | grep "^images=")
  GOT=$(run -m "$ONNX" -s "$DIR" --quiet --no-save --accuracy "$LABELS" --bench-iters 1 --bench-warmup 0 --bench-json "$OUT/bench/acc.json" 2>&1 | grep "^\[accuracy\]" | sed 's/^\[accuracy\] //')
  [ -n "$REF" ] && [ "$REF" = "$GOT" ] && ok "T20 in-process accuracy == eval_map_standalone.py ($GOT)" || { echo "  ref: $REF"; echo "  got: $GOT"; no T20; }
else echo "  SKIP  T20 (no visdrone50 labels)"; fi

echo "== multi-object tracking (synthetic panning clip) =="
if [ -n "$PYCV" ] && "$PYCV" - "$IMG" "$OUT/pan.mp4" <<'PY' 2>/dev/null
import cv2, sys, numpy as np
img = cv2.resize(cv2.imread(sys.argv[1]), (640, 480))
vw = cv2.VideoWriter(sys.argv[2], cv2.VideoWriter_fourcc(*'mp4v'), 10, (640, 480))
for f in range(30):
    M = np.float32([[1, 0, 3 * f], [0, 1, 0]])
    vw.write(cv2.warpAffine(img, M, (640, 480), borderValue=(114, 114, 114)))
vw.release()
PY
then
  for TM in bytetrack botsort; do
    rm -rf "$OUT/trk_$TM"
    run -m "$ONNX" -s "$OUT/pan.mp4" --track $TM --save-txt "$OUT/trk_$TM" --no-save --quiet > "$OUT/trk_$TM.log" 2>&1
    if grep -q "frames=30" "$OUT/trk_$TM.log" && python3 - "$OUT/trk_$TM" <<'PY'
import glob, sys, collections
files = sorted(glob.glob(sys.argv[1] + "/*.txt")); top = []; ids = set(); bad = 0
for f in files:
    rows = [l.split() for l in open(f) if l.strip()]
    bad += sum(1 for r in rows if len(r) != 7)
    if rows:
        top.append(max(rows, key=lambda r: float(r[1]))[6]); ids.update(r[6] for r in rows)
assert len(files) == 30 and bad == 0 and top, (len(files), bad)
stable = collections.Counter(top).most_common(1)[0][1]
assert stable >= 27, f"top detection kept one id on only {stable}/30 frames"
assert len(ids) >= 2
PY
    then ok "T21 --track $TM: 7-column txt, top detection keeps one id on >= 27/30 frames"; else no "T21 $TM"; fi
  done
else echo "  SKIP  T21 (no python with cv2; set PYCV=/path/to/python)"; fi

echo "== ncnn per-layer fp32 pin (router-emulated mixture graph) =="
MOA=${MOA:-$ROOT/models/moa-n_ncnn}
if [ -f "$MOA/model.ncnn.param" ]; then
  PIN=$(YOLOMASTER_NCNN_VERBOSE=1 "$BIN" -m "$MOA" -s "$IMG" --no-save --precision fp16 2>&1 | grep "^\[ncnn\] fp32 pin set" || true)
  NPIN=$(echo "$PIN" | sed -n 's/.*pin set (\([0-9]*\) of.*/\1/p')
  if echo "$PIN" | grep -q "pinnable=1" && [ -n "$NPIN" ] && [ "$NPIN" -ge 30 ] && [ "$NPIN" -le 200 ] \
     && echo "$PIN" | grep -q " add_18 " && echo "$PIN" | grep -q " mul_20 " && echo "$PIN" | grep -q " ceil_19 " \
     && echo "$PIN" | grep -q " amax_841 " && ! echo "$PIN" | grep -q " amax_840 "; then
    ok "T22 ncnn router pin set = the 1e-9/1e30 chains only ($NPIN layers)"
  else no T22; fi
else echo "  SKIP  T22 (no models/moa-n_ncnn)"; fi

echo "== parity (post-refactor) =="
c1=$(run -m "$ONNX" -s "$IMG" --no-save | grep -oE "total_dets=[0-9]+")
c2=$(run -m "$NCNN" -s "$IMG" --no-save | grep -oE "total_dets=[0-9]+")
[ -n "$c1" ] && [ "$c1" = "$c2" ] && ok "T6 onnx==ncnn ($c1)" || no "T6 parity ($c1 vs $c2)"

echo "== overrides =="
run -m "$ONNX" -s "$IMG" --classes sku --conf 0.5 --no-save | grep -qE "nc=1 \(flag:sku\)  conf=0.5" && ok "T7 --classes/--conf override" || no T7
run -m "$ONNX" -s "$IMG" --imgsz 512 --no-save | grep -q "requires fixed imgsz" && ok "T8 imgsz auto-align warn" || no T8

echo "== error handling / robustness =="
run -m /nope/x.onnx -s "$IMG" --no-save >/dev/null 2>&1; [ $? -ne 0 ] && ok "T9 missing model -> nonzero" || no T9
run -m "$ONNX" -s /nope/x.jpg --no-save >/dev/null 2>&1; [ $? -ne 0 ] && ok "T10 missing source -> nonzero" || no T10
run -m model.bin -s "$IMG" --no-save 2>&1 | grep -qi "cannot infer backend" && ok "T11 unknown ext -> ask backend" || no T11
"$BIN" -m "$ONNX" >/dev/null 2>&1; [ $? -ne 0 ] && ok "T12 missing --source -> CLI error" || no T12
run --help 2>&1 | grep -q "universal YOLO-Master" && ok "T13 --help" || no T13
mkdir -p "$OUT/corrupt"; cp "$IMG" "$OUT/corrupt/good.jpg"; echo x > "$OUT/corrupt/bad.jpg"
run -m "$ONNX" -s "$OUT/corrupt" --no-save 2>&1 | grep -q "skip. unreadable.*bad.jpg" && ok "T14 corrupt image skipped" || no T14
"$BIN" -m "$ONNX" -s "$IMG" --imgsz 512 --no-save >/dev/null 2>&1; [ $? -eq 0 ] && ok "T15 no crash on imgsz mismatch" || no T15
run -m "$ONNX" -s "$IMG" --out "$OUT/w" >/dev/null 2>&1; ls "$OUT"/w/*.jpg >/dev/null 2>&1 && ok "T16 writes annotated output" || no T16

echo "== output-shape assertions (count what actually lands on disk) =="
# T17: a video source must yield ONE annotated mp4 (no per-frame jpg spam / overwrites)
# and one --save-txt file PER FRAME (frame-indexed names).
if [ -f "$OUT/test.mp4" ]; then
  run -m "$ONNX" -s "$OUT/test.mp4" --quiet --out "$OUT/v" --save-txt "$OUT/vtxt" >/dev/null 2>&1
  NFRAMES=$(run -m "$ONNX" -s "$OUT/test.mp4" --quiet --no-save 2>&1 | grep -o "frames=[0-9]*" | head -1 | cut -d= -f2)
  MP4S=$(ls "$OUT"/v/*_annotated.mp4 2>/dev/null | wc -l)
  JPGS=$(ls "$OUT"/v/*.jpg 2>/dev/null | wc -l)
  TXTS=$(ls "$OUT"/vtxt/*.txt 2>/dev/null | wc -l)
  [ "$MP4S" = 1 ] && [ "$JPGS" = 0 ] && [ "$TXTS" = "$NFRAMES" ] \
    && ok "T17 video -> one mp4 + per-frame txt ($NFRAMES frames)" \
    || no "T17 video output shape (mp4=$MP4S jpg=$JPGS txt=$TXTS frames=$NFRAMES)"
else
  echo "  SKIP  T17 (no test video; opencv-python missing)"
fi
# T18: duplicate stems (1.jpg + 1.png in one dir) must yield one DISTINCT output per input
# in every stem-keyed writer: annotated jpgs, --save-txt, YOLO label export.
mkdir -p "$OUT/dup"
cp "$IMG" "$OUT/dup/1.jpg"
python3 - "$IMG" "$OUT/dup/1.png" <<'PY' 2>/dev/null || cp "$IMG" "$OUT/dup/1.png"
import sys
try:
    import cv2
    cv2.imwrite(sys.argv[2], cv2.imread(sys.argv[1]))
except Exception:
    raise SystemExit(1)
PY
run -m "$ONNX" -s "$OUT/dup" --quiet --out "$OUT/dout" --save-txt "$OUT/dtxt" --export-labels "$OUT/dlbl" >/dev/null 2>&1
DJ=$(ls "$OUT"/dout/*.jpg 2>/dev/null | wc -l)
DT=$(ls "$OUT"/dtxt/*.txt 2>/dev/null | wc -l)
DL=$(ls "$OUT"/dlbl/*.txt 2>/dev/null | grep -vc classes)
[ "$DJ" = 2 ] && [ "$DT" = 2 ] && [ "$DL" = 2 ] \
  && ok "T18 duplicate stems -> distinct outputs" \
  || no "T18 duplicate-stem collision (jpg=$DJ txt=$DT lbl=$DL, want 2 each)"

rm -rf "$OUT"
echo "======================================"
echo "RESULT: $P passed, $F failed"
[ $F -eq 0 ]
