# YOLO-Master Cross-Platform Edge Runtime

This example provides a C++17 inference runner and an auditable validation
workflow for YOLO-Master detection models. The runner supports ONNX Runtime,
NCNN and MNN; a native TensorRT 10 backend is optional on Jetson. The same preprocessing,
decoding and class-aware NMS are used by every compiled backend.

The implementation is intended for Issue #51 (vertical-domain edge inference
and consistency validation). It does not include a checkpoint, a dataset, or
generated predictions. Consequently, this repository documents the procedure
and supplies executable checks; numerical accuracy and latency claims require
an evidence manifest and the raw artifacts described below.

## Verification scope

The current checkout deliberately separates software verification from a
model-specific experiment. This makes the status visible before a reviewer
attempts a reproduction:

| Level | Evidence available here | Permitted conclusion |
| --- | --- | --- |
| L0 | Contract tests, exporter checks, evaluator and manifest validation | Interfaces and acceptance gates are executable |
| L1 | Ubuntu 22.04 x86_64 ONNX single-image smoke | The runtime can load and execute a compatible graph |
| L2 | 500+ fixed validation images and reference metrics | Requires a user-supplied EsMoE-N checkpoint and dataset |
| L3 | 300+ disjoint calibration images and INT8 metrics | Requires a completed INT8 run |
| L4 | Native second-platform build and raw benchmark log | Requires the target platform |

The L1 smoke model is YOLOv5s, not EsMoE-N. No mAP, FPS, hardware or
quantization value from another Issue #51 submission is reused as a result for
this checkout.

## Contents

| Path | Purpose |
| --- | --- |
| `cpp/` | Full C++17 runner, backend adapters and annotation export |
| `scripts/export_models.py` | Static ONNX and NCNN export checks |
| `scripts/eval_map.py` | Ultralytics-compatible mAP50/mAP50-95 evaluator |
| `scripts/eval_map_standalone.py` | Dependency-light evaluator for offline use |
| `scripts/mnn_val.py`, `scripts/mnn_parity.py` | MNN predictions and tensor parity |
| `scripts/prediction_diff.py` | Per-image box matching and discrepancy report |
| `scripts/quantize_int8.py` | Deterministic, training-only calibration helper |
| `scripts/evidence_manifest.py` | SHA256-pinned experiment manifest and verifier |
| `scripts/collect_environment.py` | Dependency-free host, compiler, SDK and runtime metadata collector |
| `evidence-manifest.schema.json` | Machine-readable evidence contract |
| `environment.schema.json` | Schema for the host/toolchain record |
| `jetson/` | Optional native TensorRT build and packaging scripts |

The sibling [`YOLO-Master-Edge-Deployment`](../YOLO-Master-Edge-Deployment/)
directory is a dependency-light scaffold. It is useful for checking the
Python and CMake interfaces, but it does not replace the full runner.

## Canonical evaluation protocol

The following values are the defaults of the explicit profiles. They are
recorded in the console header and in every JSON result so that an override is
visible in a review.

| Profile | Input | Confidence | IoU | Max detections | Decode |
| --- | ---: | ---: | ---: | ---: | --- |
| `visdrone` | 640 x 640 | 0.001 | 0.70 | 300 | multi-label, letterbox |
| `sku110k` | 1280 x 1280 | 0.25 | 0.60 | 300 | model-specific |

For VisDrone, inputs are resized with centered letterbox padding (value 114),
converted BGR to RGB, packed as NCHW and normalized by 255.0. The profile also
selects the canonical ten-class mapping when `--classes auto` is used. If model
metadata declares a different class count, the runner stops with an actionable
error instead of silently decoding the tensor with the wrong mapping.

These values define the conservative `visdrone` profile implemented by this
runner. Public Issue #51 reports use more than one input size and threshold
configuration, so their numerical results are not interchangeable with this
profile. Record any override in the metric JSON and compare only runs with
matching protocol metadata.

The canonical profile records `small_conf=-1` (disabled) and
`small_area=1024` in the evidence manifest. If a small-object NMS sweep is
enabled, both values must be supplied to the manifest and to every evaluator;
otherwise the comparison is considered a different protocol.

For EsMoE checkpoints, also record the routing semantics. The export helper
uses `dense_fallback` for static ONNX/NCNN graphs; a PyTorch reference run must
use the same semantics before an accuracy delta is interpreted. Use
`native_sparse` only when the backend preserves top-k dispatch and the run has
been independently validated.

## Dependencies

For an Ubuntu 22.04 x86_64 build, install the system tools and development
headers first:

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake pkg-config libopencv-dev python3
```

Provide SDK roots for the backends that will be compiled:

* ONNX Runtime: an SDK containing `include/onnxruntime_cxx_api.h` and
  `lib/libonnxruntime.so*`.
* NCNN: an install tree containing `include/ncnn/net.h` and `lib/libncnn.*`.
* MNN (optional): an install tree containing `include/MNN/Interpreter.hpp` and
  `lib/libMNN.*`.

The exact SDK versions, compiler, operating system and CPU must be recorded in
the evidence manifest. CUDA, Vulkan, OpenCL and TensorRT are optional and must
match the runtime SDK used for the build. The native TensorRT runner uses the
TensorRT 10 named-I/O API; TensorRT 8 targets should use the ONNX Runtime
TensorRT execution provider instead.

Capture the host and toolchain record before a build or benchmark with the
dependency-free collector below. It records values exposed by the current
machine and marks unavailable optional tools as `available=false`; it does not
infer GPU or SDK versions.

```bash
python3 scripts/collect_environment.py \
  --repo-root . --backend onnx --execution-provider cpu \
  --threads 4 --warmup 2 --runs 20 \
  --onnxruntime-root /opt/onnxruntime --ncnn-root /opt/ncnn \
  --output artifacts/environment.json
```

The output follows [`environment.schema.json`](environment.schema.json). Keep it
with the immutable evidence bundle and add it to the manifest with
`--report environment=artifacts/environment.json`; it complements, rather than
replaces, the manifest's model, image, prediction and report hashes.

## Build the runner

From this directory, configure only the SDKs that are present. The
`REQUIRE_*` options are recommended for a release build because they turn a
missing backend into a configuration error.

```bash
cmake -S cpp -B build-linux -DCMAKE_BUILD_TYPE=Release \
  -DONNXRUNTIME_ROOT=/opt/onnxruntime \
  -DNCNN_ROOT=/opt/ncnn \
  -DREQUIRE_ORT=ON -DREQUIRE_NCNN=ON \
  -DALLOW_NO_BACKENDS=OFF
cmake --build build-linux -j"$(nproc)"
```

Add `-DMNN_ROOT=/opt/mnn -DREQUIRE_MNN=ON` when MNN is part of the comparison.
For a dependency-light argument/CSV check, omit the SDK roots and set
`-DALLOW_NO_BACKENDS=ON`. `-DPORTABLE=ON` removes video support and links only
the OpenCV modules needed for still images.

The supplied `aarch64-toolchain.cmake` can be used with an external ARM64
sysroot. A native Jetson TensorRT build is documented in `jetson/README.md`.
The Windows build uses the same CMake target; pass `-DOpenCV_DIR` and the
Windows SDK roots from an x64 developer prompt.

## Run inference

The backend is inferred from the model suffix (`.onnx`, `.mnn`, `.engine` or
`.trt`) or from an NCNN directory/`.param` file. NCNN directories may contain
any single unambiguous `.param`/`.bin` pair; the conventional
`model.ncnn.param`/`model.ncnn.bin` names remain supported.

```bash
export LD_LIBRARY_PATH=/opt/onnxruntime/lib:/opt/ncnn/lib:${LD_LIBRARY_PATH:-}
./build-linux/yolomaster_edge \
  --model /data/models/esmoe_n_visdrone.onnx \
  --source artifacts/visdrone-val.list \
  --profile visdrone --classes auto \
  --threads 4 --out artifacts/onnx_rendered \
  --save-txt artifacts/onnx_txt \
  --csv artifacts/onnx_timing.csv
```

The list in this example is generated in step 2 of the evidence workflow below;
for an exploratory run it may be replaced with `/data/VisDrone/images/val`.

`--source` accepts a directory, a single image, a dataset YAML, or a UTF-8
newline-delimited `.txt`/`.list` image list. A list is resolved relative to its own directory
and preserves line order; blank lines and lines beginning with `#` are ignored.
For dataset YAML input, the `val` field may name one directory/list/image or
provide a YAML sequence of those entries; all resolved images are checked for
unique stems before inference.
The runner rejects missing files, unsupported extensions and duplicate filename
stems so that per-image prediction files remain one-to-one with the manifest.

Useful options are:

```text
--backend auto|onnx|ncnn|mnn|trt
--device cpu|cuda|vulkan|opencl|trt|coreml
--profile default|visdrone|sku110k
--imgsz N --conf F --iou F --max-det N
--small-conf F --small-area A  (optional area-adaptive floor for small objects)
--multi-label or --single-label
--warmup N --runs N --threads N --csv PATH
--save-txt DIR --out DIR --no-save --quiet
```

The run header prints the resolved backend, execution provider, class source,
input size and all post-processing parameters. A non-zero exit status means
that at least one input or output could not be processed; the summary lists
the failed paths.

For dense small-object splits, `--small-conf F --small-area A` applies `F`
(clamped to the global `--conf` value) to boxes whose original-image area is
below `A` pixels squared. The rule is evaluated before NMS and is implemented
identically by the C++ runner and `scripts/mnn_val.py`; it is disabled by
default. Record both values in the metric manifest when using an area-adaptive
NMS sweep.

## Export models

Use a real YOLO-Master checkpoint and keep its SHA256. The exporter fixes the
input shape, runs ONNX checking/simplification by default and records whether
the NCNN pair was loaded successfully.

```bash
python3 scripts/export_models.py \
  --model /data/checkpoints/best.pt \
  --formats onnx ncnn --imgsz 640 \
  --out-dir artifacts/exports
```

MNN conversion is intentionally separate because converter output alone is not
runtime evidence:

```bash
mnnconvert -f ONNX \
  --modelFile artifacts/exports/best.onnx \
  --MNNModel artifacts/exports/best.mnn --bizCode edge
```

Run `mnn_val.py` and the parity tool after conversion. Do not report an MNN or
NCNN result from a file-existence check alone.

## Evidence-first validation

A reproducible submission records the training provenance, evaluated image set,
post-processing policy and raw outputs so that an independent reviewer can
recompute each reported number. The workflow below follows that order and
treats every metric as a function of the recorded inputs.

1. Record the training provenance before export. At minimum, retain the
   repository/model revision, dataset release and split, ten-class mapping,
   epoch count, optimizer and learning-rate schedule, random seed,
   deterministic-setting, software versions, routing semantics, and the
   best-checkpoint SHA256.
   A command skeleton is useful for the archive (replace values with the
   actual run, rather than copying this example):

   ```bash
   yolo train model=<base-or-esmoe-yaml> data=<visdrone.yaml> \
     epochs=<N> imgsz=640 seed=<SEED> deterministic=True \
     project=artifacts/train name=esmoe_n_visdrone
   sha256sum artifacts/train/esmoe_n_visdrone/weights/best.pt
   ```

   A final `--acceptance` manifest must additionally supply
   `--training-metadata`, the exact `--command`, and at least one content-hashed
   `--report` containing metrics or benchmark output. The validator also
   requires non-empty Python/platform/machine/source-revision fields captured
   from the execution environment.

2. Materialize one ordered validation list and reuse it for every backend. A
   standard VisDrone validation split has 548 images; an acceptance run must
   contain at least 500. The list is part of the evidence, not an implicit
   directory walk:

   ```bash
   mkdir -p artifacts
   LC_ALL=C find /data/VisDrone/images/val -type f \
     \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) \
     | LC_ALL=C sort > artifacts/visdrone-val.list
   test "$(wc -l < artifacts/visdrone-val.list)" -eq 548
   ```

   Reject duplicate filename stems. If the selected split is not the 548-image
   release, record the actual count and the list digest in the manifest.
3. Generate a diagnostic manifest after the input/export paths exist. Add the
   label, prediction, calibration and report paths when producing the final
   acceptance candidate:

   ```bash
   python3 scripts/evidence_manifest.py create \
     --dataset visdrone --split val \
      --images artifacts/visdrone-val.list \
      --image-root /data/VisDrone/images/val \
      --checkpoint /data/checkpoints/best.pt \
      --model onnx_fp32=artifacts/exports/best.onnx \
      --routing-semantics dense_fallback \
      --output artifacts/onnx-evidence.json
   ```

4. Generate PyTorch/reference and backend predictions on exactly that list.
   Keep the command line, runtime versions, raw TXT files and timing CSV. Use a
   stable artifact naming scheme, for example `pytorch`, `onnx_fp32`,
   `ncnn_fp32`, `mnn_fp32` and `mnn_int8`; each directory must contain one TXT
   file per listed image, including an empty file when no detection is emitted.
   Both mAP evaluators accept the directory or the frozen list via `--images`;
   the list form preserves the recorded order verbatim.
5. Evaluate the same files with an explicit absolute percentage-point budget:

   ```bash
   python3 scripts/eval_map.py \
     --preds artifacts/onnx_txt \
      --images artifacts/visdrone-val.list \
      --image-root /data/VisDrone/images/val \
      --labels /data/VisDrone/labels/val \
      --classes visdrone --label-format yolo \
      --routing-semantics dense_fallback \
      --min-images 500 --reference-json artifacts/pytorch_map.json \
      --max-abs-delta-pp 0.5 --json artifacts/onnx_map.json
   ```

   The evaluator reports both `delta_mAP50-95_pp` (percentage points) and
   `delta_mAP50-95_pct` (relative percent). It records
   `image_manifest_sha256` for the ordered relative path list and
   `image_list_sha256` for ordered relative paths plus each image's SHA256.
   The latter matches `evidence_manifest.py`, so a same-name replacement
   cannot pass the reference-metadata gate. The thresholds are gates, not
   claims about an unmeasured model.
6. For INT8, use at least 300 training images and pass
   `--validation-images`; the quantizer checks content hashes and rejects any
   overlap. Only a subsequent mAP run can mark an INT8 artifact acceptable.
7. Verify the diagnostic bundle and publish the verification output together
   with the metrics. For the final acceptance candidate, add
   `--acceptance` here and when creating the manifest, after every required
   artifact and provenance field is present:

   ```bash
   python3 scripts/evidence_manifest.py verify artifacts/onnx-evidence.json \
       --acceptance \
       --images-root /data/VisDrone/images/val \
       --labels-root /data/VisDrone/labels/val \
       --predictions-root artifacts/onnx_txt \
       --models-root artifacts/exports \
       --reports-root artifacts \
       --checkpoint-root /data/checkpoints
   ```

   The release record should expose a compact comparison table whose cells are
   populated only from the archived JSON/CSV files:

   | Backend | Model SHA256 | Images | mAP50-95 | Delta (pp) | E2E P50/P95/P99 (ms) | FPS | Host |
   | --- | --- | ---: | ---: | ---: | --- | --- |
   | PyTorch reference | recorded in manifest | 548 | from JSON | -- | from CSV | from CSV | recorded in manifest |
   | ONNX Runtime | recorded in manifest | 548 | from JSON | from JSON | from CSV | from CSV | recorded in manifest |
   | NCNN or MNN | recorded in manifest | 548 | from JSON | from JSON | from CSV | from CSV | recorded in manifest |

   Do not fill a cell with a value that cannot be traced to a model hash,
   image-list digest and raw per-image output.

The machine-readable contract is in
[`evidence-manifest.schema.json`](evidence-manifest.schema.json); the example
file is a non-acceptance template. Store the manifest, model hashes, JSON
metrics, raw predictions and logs together in a Release or another immutable
artifact store. Do not commit datasets, weights or generated run directories.
Use repeatable `--report NAME=PATH` options for mAP JSON, timing CSV or
benchmark sidecar JSON, and parity logs; each report collection is recorded
with a deterministic file-list SHA256 and can be checked with
`verify --reports-root`.

## Benchmark protocol

Use the same ordered list, input size, precision, thread count and backend
provider for every comparison. Warm up before timing and report preprocessing,
inference, postprocessing and end-to-end mean/P50/P95/P99 plus FPS. Include the
CPU/GPU model, OS, compiler, SDK versions, warm-up count and repeat count. A
virtual-machine measurement must be labelled as such and must not be presented
as an ARM or Jetson result.

`--csv` writes one row per image and a `#summary` row. Add
`--benchmark-json artifacts/onnx_benchmark.json` to write a machine-readable
sidecar containing the resolved protocol, execution provider, host OS and
architecture, compiler, CPU model, logical CPU count, build date and aggregate
timing statistics. The sidecar is useful for reviewing a run, but it does not
replace the evidence manifest: model, image and report hashes remain the source
of truth. When both options are supplied, the JSON records the CSV path as
`timing_csv`; either option enables the configured warm-up and timed repeats.
Use `scripts/prediction_diff.py` to localize confidence, coordinate or missing
box differences before changing thresholds. Pass the same frozen list and
normalization root used by the mAP evaluator; BOMs, quoted paths and root
escapes are handled consistently:

```bash
python3 scripts/prediction_diff.py \
  --reference artifacts/onnx_txt \
  --candidate artifacts/ncnn_txt \
  --images artifacts/visdrone-val.list \
  --image-root /data/VisDrone/images/val \
  --json artifacts/onnx-vs-ncnn-diff.json \
  --csv artifacts/onnx-vs-ncnn-diff.csv \
  --min-iou 0.90 --max-unmatched 0
```

The report includes nearest-rank IoU P05/P50/P95/P99 and explicit unmatched,
confidence and coordinate-delta gates. It is a diagnostic artifact; it does
not replace the mAP and evidence-manifest acceptance checks.

## Tests

The dependency-light Python contract suite covers profile resolution, strict
label parsing, evidence manifests, output-shape handling and prediction diff.
From the repository root:

```bash
python3 -m pip install -r examples/YOLO-Master-Cross-Platform-Edge-Deployment/requirements-edge.txt
python3 -m pytest -q tests/test_edge_deployment_utils.py \
  tests/test_edge_deployment_contract.py \
  tests/test_issue51_runtime_contract.py
```

The C++ robustness harness is `cpp/run_tests.sh`. It requires a built runner
and the relevant SDKs; a dependency-light CMake build only checks the CLI
contract.

## Scope and limitations

The repository contains implementation and validation infrastructure, not the
training run behind any public Issue #51 submission. Do not copy another
author's mAP, FPS, hardware or INT8 values into a result table. A result is
publishable only when a reviewer can recompute it from the archived manifest,
model, image list, predictions and logs.

The optional `gui/`, `mac/` and `jetson/` integrations remain available for
platform-specific work. Their platform measurements must follow the same
evidence rules; the source tree itself makes no hardware-performance claim.

When the acceptance run is complete, use `TECHNICAL_SUMMARY_ZH.md` as the
technical basis for the public summary. Every reported number must point to an
archived JSON, CSV, prediction set and SHA256 manifest.

## Contributing

Please open an issue or pull request in the
[Tencent/YOLO-Master repository](https://github.com/Tencent/YOLO-Master) with
the exact protocol and reproducibility artifacts for any reported result.
