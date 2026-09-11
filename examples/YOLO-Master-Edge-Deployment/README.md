# YOLO-Master Edge Deployment Example

This example defines a reproducible protocol for issue #51. It covers model
export, preprocessing and postprocessing consistency, cross-runtime numerical
comparison, and latency measurement. The example is independent of a specific
checkpoint or dataset and does not bundle model weights or runtime SDKs.

## Scope and files

- `edge_utils.py` contains shared preprocessing, box scaling, tensor comparison,
  and latency-summary utilities.
- `export_edge_models.py` is a lightweight wrapper around the Ultralytics exporter.
- `validate_edge_outputs.py` compares saved `.npy` tensors before decoding and NMS.
- `CMakeLists.txt` builds the dependency-light benchmark scaffold.
- `cpp/` contains the scaffold benchmark and optional backend adapters.

The end-to-end C++ runner for ONNX Runtime, NCNN, MNN, and native TensorRT 10 is
maintained in
[`../YOLO-Master-Cross-Platform-Edge-Deployment/`](../YOLO-Master-Cross-Platform-Edge-Deployment/).
Its export and validation scripts are described in that directory and in
[`VALIDATION.md`](VALIDATION.md).
Targets that ship TensorRT 8 should use the runner's ONNX Runtime TensorRT
execution-provider route rather than the native TensorRT backend.

## Reference profiles

The Python utilities expose two explicit profiles:

- `visdrone`: aspect-ratio-preserving input preparation, `conf=0.001`, and
  `iou=0.70`, `max_det=300`, and multi-label decoding at `640x640`, matching
  the small-object evaluation protocol used by the production runner.
- `sku110k`: high-resolution shelf-image preparation at `1280x1280`,
  `conf=0.25`, `iou=0.60`, `max_det=300`, and multi-label decoding. This is
  the same static-input protocol used by the production runner and evaluators.
  A rectangular deployment must be requested with an explicit `--imgsz`
  override and reported as a separate protocol.

These values are defaults rather than hidden global state. Experiments should
record any command-line overrides with the resulting metrics.

## Export

For the lightweight wrapper:

```bash
python export_edge_models.py \
  --model runs/train/weights/best.pt \
  --formats onnx ncnn \
  --profile visdrone \
  --half
```

The cross-platform exporter performs additional graph checks and emits a JSON
summary:

```bash
python examples/YOLO-Master-Cross-Platform-Edge-Deployment/scripts/export_models.py \
  --model runs/train/weights/best.pt \
  --formats onnx ncnn \
  --imgsz 640 \
  --out-dir artifacts/exports
```

ONNX exports are static and simplified by default. `--no-simplify` is reserved
for diagnostics and must be paired with `--allow-unsimplified`; that artifact
is not a validation result.

## Tensor consistency

Save raw outputs from PyTorch and each exported backend using the same ordered
input list, then compare them before decoding:

```bash
python validate_edge_outputs.py \
  --reference artifacts/pytorch.npy \
  --candidate artifacts/onnx.npy \
  --tolerance 0.005
```

The comparison reports maximum and mean absolute error, RMSE, and a Boolean
decision for the declared tolerance. Shape mismatches and non-finite values
are failures. Keep the first failing tensor and image as diagnostic evidence.

## C++ scaffold build

The scaffold can be compiled without an inference SDK. This verifies the CLI
and CSV contract; it does not constitute neural-network inference evidence.

```bash
cmake -S examples/YOLO-Master-Edge-Deployment \
  -B build/edge-scaffold
cmake --build build/edge-scaffold
```

For a real backend, configure the corresponding SDK in the CMake invocation.
The cross-platform runner's CMake options and platform-specific toolchains are
documented in its README.

The benchmark accepts either a directory or a fixed, newline-delimited image
list. Relative list entries are resolved against the list file, blank/comment
lines are ignored, and duplicate filename stems are rejected so prediction
artifacts cannot overwrite one another:

```bash
./build/edge-scaffold/yolo_master_edge_benchmark \
  --backend onnx --model artifacts/model.onnx \
  --images artifacts/visdrone_val.txt --profile visdrone \
  --min-images 500 --warmup 10 --runs 3 \
  --output artifacts/onnx_benchmark.csv --json artifacts/onnx_benchmark.json
```

The JSON sidecar records the resolved profile, image count, warm-up/repeat
counts, thread count, host/compiler information, and mean/P50/P95/P99/FPS for
preprocess, inference, postprocess, and end-to-end latency. It is metadata for
the benchmark and does not replace the SHA256 evidence manifest.

## Benchmark output

Per-image CSV output uses the following columns:

```text
image,preprocess_ms,inference_ms,postprocess_ms,total_ms,detections,run
```

The current runner appends a `run` column when repeated measurements are
requested. For acceptance measurements, retain the CSV together with the JSON
sidecar and evidence manifest; do not average values copied from console output.

`preprocess_ms` includes image loading, letterbox, color conversion, and tensor
packing. `inference_ms` is runtime execution, `postprocess_ms` is decoding and
NMS, and `total_ms` is their end-to-end sum. Aggregate statistics are reported
as mean, P50, P95, P99, and FPS. The shared `read_latency_csv` helper accepts
both the scaffold's `latency_ms` column and the runner's `total_ms` column.

## Reproducibility workflow

1. Record the checkpoint revision, class mapping, input size, and export opset.
2. Export ONNX and at least one mobile format (NCNN or MNN).
3. Validate graph structure, preprocessing, and conversion artifacts.
4. Generate a SHA256-pinned evidence manifest with
   `../YOLO-Master-Cross-Platform-Edge-Deployment/scripts/evidence_manifest.py`.
5. Evaluate all formats on the same ordered validation image list.
6. Enforce the image-count and accuracy gates in [`VALIDATION.md`](VALIDATION.md).
7. Report latency with platform, runtime version, precision, and thread count.

For VisDrone, use the C++ runner's `--profile visdrone` and the evaluator's
`--max-abs-delta-pp 0.5` (FP32) or `1.0` (INT8). Quantitative results are
meaningful only when the image manifest, model artifact, command line and
software versions are retained. A smoke run or a reference table without those
artifacts is not an acceptance result.
