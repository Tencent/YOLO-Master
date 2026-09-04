# YOLO-Master Cross-Platform Edge Deployment

This example provides a C++17 command-line runtime and a reproducible
validation workflow for Issue #51. The runtime shares one preprocessing,
decoding, NMS, timing, and output contract across ONNX Runtime, NCNN, MNN,
and (when the SDK is installed) TensorRT.

The repository contains code, schemas, and validation tools. It intentionally
does not contain private checkpoints, datasets, or generated predictions. A
metric or device claim is valid only when its model, image list, raw output,
environment record, and SHA256 manifest are archived together.

## Scope and current evidence

The implementation includes:

- CMake discovery for ONNX Runtime, NCNN, MNN, and optional TensorRT;
- UTF-8-safe model and image paths on Windows;
- model metadata and output-shape validation before decoding;
- deterministic image-list handling with duplicate-stem and root-escape checks;
- ONNX/NCNN/MNN export, parity, mAP, INT8, benchmark, and manifest tools;
- Linux, Windows, macOS, ARM64 cross-build, and Jetson build entry points.

The validated local smoke record is an Ubuntu 22.04 x86_64 YOLOv5s ONNX
single-image run (six detections, approximately 970.873 ms end to end). This
is an L1 functional check. It is not an EsMoE-N VisDrone/SKU-110K accuracy
result, an INT8 result, or a native ARM64/Jetson measurement.

## Directory layout

| Path | Purpose |
| --- | --- |
| `cpp/` | C++17 runtime and CMake build |
| `scripts/export_models.py` | Checked ONNX, NCNN, and MNN conversion |
| `scripts/eval_map.py` | Ultralytics-backed mAP evaluation |
| `scripts/eval_map_standalone.py` | Dependency-light mAP evaluation |
| `scripts/prediction_diff.py` | Per-image and per-box parity diagnostics |
| `scripts/quantize_int8.py` | Training-only INT8 calibration and manifest |
| `scripts/evidence_manifest.py` | Evidence creation, validation, and hash verification |
| `scripts/collect_environment.py` | Reproducible host and SDK record |
| `scripts/package_linux.sh` | Relocatable Linux bundle assembly |
| `environment.schema.json` | Host-record schema |
| `evidence-manifest.schema.json` | Dataset, model, prediction, and report schema |
| `TECHNICAL_REPORT.md` | Full protocol and rationale |
| `TECHNICAL_SUMMARY_ZH.md` | Chinese technical summary |

## Dependencies

Required for the image CLI:

- CMake 3.16 or newer;
- a C++17 compiler;
- OpenCV 4.5 or newer (`core`, `imgproc`, and `videoio` for video input);
- ONNX Runtime for ONNX models.

NCNN and MNN are optional at configuration time. A complete Issue #51 build
should set `REQUIRE_ORT=ON` and at least one of `REQUIRE_NCNN=ON` or
`REQUIRE_MNN=ON`; missing required SDKs then fail configuration instead of
silently producing a partial binary.

## Build

From this directory, configure a minimal image-inference build:

```bash
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release \
  -DPORTABLE=ON \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime \
  -DNCNN_ROOT=/opt/ncnn \
  -DUSE_MNN=OFF \
  -DREQUIRE_ORT=ON -DREQUIRE_NCNN=ON \
  -DALLOW_NO_BACKENDS=OFF
cmake --build cpp/build --parallel
```

For MNN as the secondary backend, use `-DMNN_ROOT=/opt/mnn
-DUSE_MNN=ON -DREQUIRE_MNN=ON`. For a full video build, omit `-DPORTABLE=ON`.
The configure summary lists every enabled backend and its resolved library.

The same source builds on Windows with MSVC or MinGW. Replace the SDK paths
with Windows paths and pass `-DOpenCV_DIR=<opencv-build>`. For ARM64 cross
compilation, use `-DCMAKE_TOOLCHAIN_FILE=cpp/aarch64-toolchain.cmake` and
record the sysroot and compiler in the environment manifest. A cross-compiled
binary is not evidence of a native device run.

## Runtime profiles

The generic defaults are intended for interactive inference. For an Issue #51
accuracy run, always select an explicit profile so the resolved protocol is
printed in the log and benchmark sidecar.

### VisDrone

`--profile visdrone` resolves to:

| Parameter | Value |
| --- | ---: |
| input | 640 x 640 |
| confidence | 0.001 |
| NMS IoU | 0.70 |
| max detections | 300 |
| class policy | 10-class VisDrone mapping |
| multi-label | enabled |
| resize | aspect-preserving letterbox, RGB/NCHW, float32/255 |

### SKU-110K

`--profile sku110k` resolves to a 1280-square input, confidence 0.25, NMS IoU
0.60, max detections 300, and the one-class SKU-110K mapping. Explicit command
line values are recorded and take precedence over profile defaults.

## Quick smoke run

```bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:$LD_LIBRARY_PATH
./cpp/build/yolomaster_edge \
  --model artifacts/model.onnx \
  --source test-data/image.jpg \
  --backend onnx \
  --profile visdrone \
  --out runs/smoke \
  --save-txt runs/smoke/labels \
  --csv runs/smoke/timing.csv \
  --benchmark-json runs/smoke/benchmark.json
```

Use `--no-save` for a load-only check. A missing model, invalid output shape,
unsupported device, or incomplete NCNN pair returns a non-zero status with a
diagnostic message.

## Issue #51 acceptance workflow

The following commands are the canonical order. Run all backends against the
same ordered image list and with the same class mapping and profile.

### 1. Freeze the validation set

Convert native annotations to the repository's YOLO label format, then create
an ordered UTF-8 list. The standard VisDrone validation split contains 548
images; SKU-110K must report its actual split and count.

```bash
find /data/VisDrone/images/val -type f \\
  \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' \) \\
  | sort > artifacts/visdrone-val.list
```

Use the list file for every PyTorch and edge run. Directory enumeration is not
an equivalent protocol. The manifest tool rejects missing files, unsupported
suffixes, duplicate stems, and paths outside the declared image root.

### 2. Export and check models

```bash
python scripts/export_models.py \
  --weights runs/esmoe_n/weights/best.pt \
  --out artifacts/export \
  --imgsz 640 --opset 17
```

The exporter runs ONNX checker and simplification by default and records the
conversion command, output names, input shape, and SHA256. NCNN requires a
matching `.param` and `.bin`; MNN conversion remains explicitly marked
incomplete until its predictions pass the same parity gate.

### 3. Run the C++ backends

```bash
./cpp/build/yolomaster_edge --model artifacts/export/model.onnx \
  --source artifacts/visdrone-val.list --profile visdrone \
  --backend onnx --threads 4 --warmup 10 --runs 100 \
  --out runs/onnx --save-txt runs/onnx/labels \
  --csv runs/onnx/timing.csv --benchmark-json runs/onnx/benchmark.json

./cpp/build/yolomaster_edge --model artifacts/export/model.mnn \
  --source artifacts/visdrone-val.list --profile visdrone \
  --backend mnn --threads 4 --warmup 10 --runs 100 \
  --out runs/mnn --save-txt runs/mnn/labels \
  --csv runs/mnn/timing.csv --benchmark-json runs/mnn/benchmark.json
```

Use `--backend ncnn` with the exported NCNN directory when that SDK is enabled.
Keep the `--profile`, image list, thread count, warm-up count, and repeat count
identical across backends.

### 4. Evaluate accuracy

Use converted YOLO labels for the formal gate. Native VisDrone annotation rows
are supported only for diagnostics because their ignored-region semantics are
not identical to YOLO labels.

```bash
python scripts/eval_map.py \
  --preds runs/onnx/labels \
  --images artifacts/visdrone-val.list \
  --image-root /data/VisDrone/images/val \
  --labels /data/VisDrone/labels/val \
  --classes visdrone --label-format yolo \
  --routing-semantics dense_fallback \
  --imgsz 640 --conf 0.001 --iou 0.70 --max-det 300 --multi-label \
  --min-images 500 \
  --reference-json runs/pytorch/map.json \
  --max-abs-delta-pp 0.5 \
  --json runs/onnx/map.json
```

The FP32 budget is 0.5 percentage points. Use 1.0 percentage point only for
an explicitly identified INT8 run. The evaluator reports both absolute
percentage-point and relative-percent deltas; do not mix the two units.

### 5. Check per-image parity

```bash
python scripts/prediction_diff.py \
  --reference runs/pytorch/labels \
  --candidate runs/onnx/labels \
  --images artifacts/visdrone-val.list \
  --image-root /data/VisDrone/images/val \
  --iou 0.50 --min-iou 0.99 \
  --json runs/onnx/prediction-diff.json \
  --csv runs/onnx/prediction-diff.csv
```

The report includes matched and unmatched counts, coordinate and confidence
deltas, and IoU percentiles. A parity claim must cite this report, not only a
single visual example.

### 6. Archive and verify evidence

```bash
python scripts/evidence_manifest.py create \
  --dataset visdrone --split val \
  --images artifacts/visdrone-val.list \
  --image-root /data/VisDrone/images/val \
  --labels /data/VisDrone/labels/val \
  --predictions runs/onnx/labels \
  --checkpoint runs/esmoe_n/weights/best.pt \
  --training-metadata artifacts/training-provenance.json \
  --model onnx=artifacts/export/model.onnx \
  --report map=runs/onnx/map.json \
  --report timing=runs/onnx/timing.csv \
  --command "./cpp/build/yolomaster_edge --profile visdrone ..." \
  --acceptance --output runs/onnx/evidence.json

python scripts/evidence_manifest.py verify runs/onnx/evidence.json \
  --acceptance \
  --images-root /data/VisDrone/images/val \
  --labels-root /data/VisDrone/labels/val \
  --predictions-root runs/onnx/labels \
  --models-root artifacts/export \
  --checkpoint-root runs/esmoe_n/weights \
  --reports-root runs/onnx
```

Publish the verified evidence directory as an immutable Release artifact or
equivalent. Keep large models, images, and predictions out of the source PR.

## INT8 protocol

Calibration images must come from the training split, contain at least 300
files, and be disjoint from validation by content SHA256. Quantization produces
an explicit calibration manifest and reports `acceptance_ready: false` until an
INT8 prediction run has passed `eval_map.py` with the 1.0 percentage-point gate.

## Performance protocol

Use the same host, CPU affinity, input size, thread count, warm-up count, and
repeat count for all backends. Archive preprocessing, inference, postprocessing,
end-to-end mean/P50/P95/P99, FPS, compiler, runtime versions, and host data.
Virtual-machine results must be labelled as VM measurements and must not be
presented as Jetson or ARM64 performance.

## Evidence levels

| Level | Minimum evidence | Permitted conclusion |
| --- | --- | --- |
| L0 | Contract tests, parser checks, CMake diagnostics | Interfaces and static gates work |
| L1 | Real model plus one image or a small subset | Load, preprocess, decode, and output work |
| L2 | At least 500 fixed images, reference JSON, predictions, and hashes | Auditable FP32 accuracy |
| L3 | L2 plus 300 disjoint calibration images and INT8 gate | Auditable INT8 accuracy |
| L4 | Native builds on two platforms with raw logs and parity artifacts | Cross-platform deployment result |

Do not upgrade an evidence level by filling a template with estimated values.

## License

The example follows the license of the surrounding YOLO-Master repository.
