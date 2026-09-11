# Edge Validation Protocol

This document specifies the validation procedure associated with issue #51.
It separates dependency-free contract checks from measurements that require a
checkpoint, a dataset, and an inference SDK. A result is considered complete
only when the command line, software versions, model digest, and image-list
digest are recorded with it.

## Contract checks

From the repository root, install the helper dependencies and run the tests:

```bash
python -m pip install -r examples/YOLO-Master-Cross-Platform-Edge-Deployment/requirements-edge.txt
python -m pytest tests/test_edge_deployment_utils.py \
  tests/test_edge_deployment_contract.py \
  tests/test_issue51_runtime_contract.py -q
```

The dependency-light scaffold can be configured and built as follows:

```bash
cmake -S examples/YOLO-Master-Edge-Deployment \
  -B /tmp/yolo-master-edge-scaffold
cmake --build /tmp/yolo-master-edge-scaffold
```

This build verifies the CLI and CSV interfaces only. It does not execute a
neural-network graph. For backend execution, configure
`examples/YOLO-Master-Cross-Platform-Edge-Deployment/cpp` with the relevant
SDK roots and run its `run_tests.sh` harness.

## Export checks

Use a fixed checkpoint and input shape. The cross-platform exporter validates a
static ONNX graph, preserves model metadata, checks the NCNN parameter/binary
pair, and writes `export_summary.json`:

```bash
python examples/YOLO-Master-Cross-Platform-Edge-Deployment/scripts/export_models.py \
  --model /path/to/checkpoint.pt \
  --formats onnx ncnn \
  --imgsz 640 \
  --out-dir artifacts/exports
```

The default ONNX path is simplified with `onnxsim` and checked with the ONNX
checker. `--no-simplify --allow-unsimplified` is available for diagnosis only;
the resulting artifact must not be reported as an acceptance export. MNN
conversion consumes the canonical ONNX file and is marked incomplete until a
runtime parity check has been performed.

## Tensor parity

Run PyTorch and every exported backend on one identical, ordered image list.
Save raw output tensors before decoding and NMS. For MNN/ONNX comparison:

```bash
python examples/YOLO-Master-Cross-Platform-Edge-Deployment/scripts/mnn_parity.py \
  --mnn artifacts/exports/model.mnn \
  --onnx artifacts/exports/model.onnx \
  --images /data/VisDrone/images/val \
  --limit 100 \
  --tolerance 0.1 \
  --json artifacts/mnn_parity.json \
  --debug-dir artifacts/mnn_parity_debug
```

The parity tool normalizes the two common feature/anchor layouts, rejects
shape mismatches and non-finite values, and returns a non-zero status when the
maximum absolute error exceeds the declared tolerance. The debug directory is
intended to retain the first input and tensor mismatch for diagnosis.
The example uses 100 images as a diagnostic subset; it is not a substitute for
the full validation image list or the 500-image accuracy gate below.

## Per-image prediction diagnosis

When an accuracy gate fails, retain the decoded predictions from both backends
and compare them without rerunning inference:

```bash
python examples/YOLO-Master-Cross-Platform-Edge-Deployment/scripts/prediction_diff.py \
  --reference artifacts/onnx_txt \
  --candidate artifacts/ncnn_txt \
  --images /data/VisDrone/images/val \
  --iou 0.50 --min-iou 0.90 --top-k 20 \
  --json artifacts/onnx_ncnn_prediction_diff.json \
  --csv artifacts/onnx_ncnn_prediction_diff.csv \
  --debug-dir artifacts/onnx_ncnn_prediction_diff_images
```

The report matches same-class boxes by IoU and lists, for every image, the
detection-count delta, unmatched boxes, matched IoU, confidence deltas, and
maximum coordinate delta. It also records nearest-rank IoU P05/P50/P95/P99 in
both the per-image rows and the aggregate summary. `--min-iou` (also accepted
as `--min-match-iou`), `--max-unmatched`, `--max-box-delta`, and
`--max-conf-delta` are optional diagnostic gates; they do not replace the mAP
acceptance gate. The script uses only the standard library unless
`--debug-dir` is requested (Pillow then provides the overlays).

## Accuracy gate

Use at least 500 validation images (the issue target; 548 is a valid larger
VisDrone split). The image set must be fixed, ordered, and free of duplicate
stems. Use identical confidence, IoU, class mapping, multi-label policy, and
maximum-detection settings for all backends.

```bash
python examples/YOLO-Master-Cross-Platform-Edge-Deployment/scripts/eval_map.py \
  --preds artifacts/onnx_txt \
  --images /data/VisDrone/images/val \
  --labels /data/VisDrone/labels/val \
  --classes visdrone --label-format yolo \
  --imgsz 640 --conf 0.001 --iou 0.70 --max-det 300 --multi-label \
  --min-images 500 \
  --reference-json artifacts/pytorch_map.json \
  --max-abs-delta-pp 0.5 \
  --json artifacts/onnx_map.json
```

The evaluator accepts normalized YOLO labels and native VisDrone rows. Native
rows are intended for diagnostics: `score=0` and task-external categories are
excluded, but ignored-region matching is not inferred. For an acceptance run,
convert annotations with the official `visdrone2yolo` procedure first. Outside
`--smoke`, the evaluator requires one label and one prediction file per image,
rejects extra stems and enforces the non-negotiable 500-image floor.
`--max-abs-delta-pp` applies an absolute percentage-point gate (the Issue #51
convention); `--max-abs-delta-pct` remains available for a relative percentage
gate. They are mutually exclusive. The relative gate requires a positive
PyTorch reference mAP; the absolute gate records the reference/protocol
metadata used for the comparison. Every result JSON records both
`delta_mAP50-95_pp` and `delta_mAP50-95_pct` to make the units explicit. The
recommended budgets are below 0.5 pp for non-quantized exports and below 1.0
pp for INT8; these are decision thresholds, not results claimed by this
repository.

## INT8 calibration

Calibration must use training images only and must be disjoint from validation.
Pass `--validation-images` to the quantizer when both splits are available; it
compares content hashes and aborts on overlap. The quantizer's JSON is
intentionally marked `acceptance_ready: false`; a separate evidence manifest
may claim INT8 acceptance only after the prediction directory passes the 1.0
percentage-point mAP gate.
The quantizer enforces a minimum of 300 images, uses the same letterbox/RGB/NCHW
preprocessing as the C++ runner, writes a deterministic calibration manifest,
and records its SHA-256 digest:

```bash
python examples/YOLO-Master-Cross-Platform-Edge-Deployment/scripts/quantize_int8.py \
  --fp32 artifacts/exports/model.onnx \
  --train /data/VisDrone/images/train \
  --validation-images /data/VisDrone/images/val \
  --n-calib 300 \
  --format QOperator \
  --out artifacts/exports/model_int8.onnx
```

The default exclusion set keeps the detection head, attention, and routing
nodes in FP32. The script refuses to label the mixed-precision recipe when a
declared exclusion pattern matches no graph node. Use
`--no-default-exclude` only for a diagnostic comparison and report the changed
recipe explicitly.

## MNN prediction export

To produce C++-compatible per-image predictions from an MNN model:

```bash
python examples/YOLO-Master-Cross-Platform-Edge-Deployment/scripts/mnn_val.py \
  --mnn artifacts/exports/model.mnn \
  --images /data/VisDrone/images/val \
  --out artifacts/mnn_txt \
  --limit 500
```

The decoder applies strict finite/positive-geometry checks, class-aware NMS,
and a 300-detection cap. Its output can be passed directly to `eval_map.py`.

## Latency reporting

Warm up each backend and use the same image list, input size, precision, and
thread count. Report preprocessing, inference, and postprocessing separately,
then report count, mean, P50, P95, P99, and FPS. Include CPU/GPU model and
runtime versions; latency values without this metadata are not comparable.

The C++ runner writes per-image `total_ms` values and a `#summary` row. The
shared Python helper accepts that format as well as the scaffold's
`latency_ms` column and ignores blank aggregate cells.

## Platform evidence

For each target platform, retain the exact CMake configure command, compiler,
runtime SDK versions, generated binary name, and a clean-environment smoke
log. A Linux x86_64 build and a second target (Windows x64 or Linux ARM64/
Jetson) provide a minimal cross-platform record. Do not place model weights,
datasets, SDKs, build directories, or generated evidence archives in Git.
