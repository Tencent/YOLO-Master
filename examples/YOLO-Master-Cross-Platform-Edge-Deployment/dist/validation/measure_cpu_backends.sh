#!/usr/bin/env bash
# ncnn / MNN: 200-frame warm timing (pre/model/post), bench cold median, coco500 mAP (val protocol)
# usage: BIN=<cli> C=<coco500 images> MODELS="label|path|args ..." bash measure_cpu_backends.sh > out.log
set -uo pipefail
for spec in $MODELS; do
  IFS='|' read -r label model args <<< "$spec"
  args="${args//_/ }"
  t="$(timeout 1800 $BIN -m $model -s $C $args --no-save --quiet --warmup 5 --limit 200 2>&1 | grep -E '^\[summary\]' | sed -E 's/.*avg\/frame: //; s/ +wall=.*//')"
  b="$(timeout 1800 $BIN -m $model -s $C $args --bench cold --bench-iters 50 --bench-warmup 10 --bench-json /tmp/b_$label.json --no-save --quiet 2>&1 | grep -oE 'median=[0-9.]+ms' | head -1)"
  a="$(timeout 3600 $BIN -m $model -s $C $args --accuracy auto --no-save --quiet 2>&1 | grep -E '^\[accuracy\]|note=' | sed -E 's/.*(note=[^ ]+ [^ ]+ [^ ]+).*/\1/; s/\[accuracy\] images=500 +//' | tr '\n' ' ')"
  echo "$label | $t | bench $b | $a"
done
echo "MEASURE DONE $(date +%H:%M:%S)"
