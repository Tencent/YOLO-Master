# Technical Note: Auditable Edge Deployment for Issue #51

## Abstract

This note describes the implementation and the measurement protocol supplied
for Issue #51, which concerns edge inference of vertically trained YOLO-Master
models. The contribution is a backend-independent C++ runtime together with
export, validation, quantization and evidence tooling. The design treats
reproducibility as part of the deployment interface: a metric is considered a
result only when the model, ordered image set, command line, software
environment and raw predictions can be verified from a content-addressed
manifest.

The source tree contains no EsMoE-N checkpoint, VisDrone images or generated
prediction directory. This document therefore specifies an executable method
and the implementation boundaries; it does not claim a new mAP, FPS or ARM64
measurement.

## Contribution and verification status

The table below is the short reviewer-facing record. It separates implementation
evidence from measurements that require a user-supplied checkpoint and data.

| Contribution | Reproducible artifact | Status in this checkout |
| --- | --- | --- |
| Unified C++17 inference path | ONNX Runtime, NCNN and MNN adapters with shared preprocessing, decoding and NMS | L0 contract-checked |
| Export and conversion checks | ONNX checker/simplifier, NCNN pair/sidecar validation, MNN conversion diagnostics | L0 structural checks |
| Accuracy and parity protocol | Ordered image manifest, mAP evaluator, percentage-point gate and per-image prediction diff | L0 tooling |
| INT8 protocol | Training-only calibration selection (>=300 images), hash disjointness and protected nodes | L0 tooling |
| Linux functional smoke | Ubuntu 22.04 x86_64, YOLOv5s ONNX, one image and six detections | L1 smoke evidence |
| EsMoE-N acceptance result | VisDrone/SKU-110K full split, mAP, INT8 and a second native platform | Pending checkpoint/data/logs |

The smoke run is a functional check of the runtime and is not an EsMoE-N
accuracy claim. No result is promoted to an acceptance claim until its model,
image set, predictions and raw logs are available for independent verification.

## 1. Evaluation objective

The target comparison is a single trained checkpoint evaluated through
PyTorch and at least two deployment formats (ONNX Runtime plus NCNN or MNN).
Every path must consume the same ordered validation images and the same
post-processing parameters. The minimum acceptance record is:

* a fixed validation list with at least 500 images (the common VisDrone split
  contains 548 images);
* a PyTorch/reference metric JSON and per-image predictions for each backend;
* a model and image-list SHA256 digest;
* an explicit EsMoE routing-semantic record shared by the reference and export;
* a latency log with fixed thread count, warm-up, repeat count and host details.

Optional INT8 evaluation adds a training-only calibration list of at least 300
images. The calibration and validation sets are checked for content overlap,
not merely for different filenames.

These requirements make the evaluation record independent of a particular
machine, exporter ordering or directory traversal. The same protocol is used
for every backend and platform so that a reported difference has a traceable
cause.

### 1.1 Training provenance and image-manifest control

The checkpoint is treated as an experimental input rather than as an
interchangeable model file. A submission record should therefore include the
base-model or repository revision, dataset release and split, class mapping,
epoch count, optimizer and learning-rate schedule, random seed,
deterministic-setting, training software versions and the SHA256 of the best
checkpoint. The exact training command and the selected checkpoint should be
archived alongside the exported graphs.

The validation population is materialized before inference. For the standard
VisDrone validation split this is an ordered list of 548 image paths; the list
is reused unchanged by PyTorch, ONNX Runtime, NCNN and MNN. The manifest stores
one record per image (relative path, byte count and SHA256), rejects duplicate
stems and computes an ordered-list digest. A run with fewer than 500 images is
diagnostic only. This distinction prevents a convenient subset or a stale
prediction directory from being presented as a full-split result.

Metric JSON records both the path-only `image_manifest_sha256` and the
content-aware `image_list_sha256`. The latter hashes ordered
`relative-path SHA256` rows and therefore matches the evidence manifest. The
reference gate compares both fields, so replacing an image while retaining its
filename invalidates the comparison. For a list stored outside the dataset
tree, `--image-root` defines the normalization boundary; entries outside that
boundary are rejected.

The same control applies to INT8 calibration. Calibration images are selected
from the training split, deterministically ordered and content-hash compared
with the validation records. At least 300 images are required, and any hash
overlap invalidates the calibration evidence even when filenames differ.

## 2. Runtime architecture

The C++ runner is organized into four layers:

1. **Input and timing.** `main.cpp` resolves image, directory, video, dataset
   YAML and newline-delimited image-list sources. Dataset `val` accepts a
   scalar or YAML sequence; list inputs preserve their declared order, while
   directory inputs are sorted deterministically. The runner records failures
   and emits per-image timing rows.
2. **Preprocessing.** `common.cpp` implements centered letterbox (padding
   value 114), BGR-to-RGB conversion and NCHW `float32 / 255` packing. Stretch
   mode is available only as an explicit diagnostic override.
3. **Backend adapters.** `ort_backend.cpp`, `ncnn_backend.cpp` and
   `mnn_backend.cpp` load a graph once and expose the same `Backend` interface.
   Optional native TensorRT support is compiled only with a TensorRT 10.x SDK;
   older TensorRT releases should use the ONNX Runtime TensorRT execution
   provider path.
4. **Shared decoding.** Raw tensors are normalized to the feature-major
   `[features, anchors]` layout, decoded to original-image pixel coordinates,
   filtered with class-aware NMS and capped at `max_det`. Segmentation
   prototypes are carried separately so annotation export does not alter box
   results.

The backend factory infers a format from the model path, accepts case-insensitive
suffixes, and reports an error for an ambiguous NCNN directory. Metadata is
used for names and input size when available; explicit profiles take precedence
for domain-critical class mappings.

## 3. Canonical post-processing profiles

The profile is part of the run contract and is printed in the console header.
The VisDrone profile is:

| Parameter | Value |
| --- | ---: |
| Input | 640 x 640 |
| Confidence threshold | 0.001 |
| NMS IoU threshold | 0.70 |
| Maximum detections | 300 |
| Small-object confidence floor | disabled (`--small-conf=-1`); area threshold 1024 px^2 |
| Class policy | ten canonical VisDrone classes |
| Decode | multi-label per anchor |
| Resize | centered letterbox, pad 114 |

The SKU-110K profile uses a 1280-square input, confidence 0.25, IoU 0.60 and
the one-class mapping. Callers may override thresholds for deployment, but the
resulting values are recorded and must not be compared with a run using a
different protocol.

The runner also exposes an optional area-adaptive confidence floor for dense
small-object scenes. When `--small-conf` is non-negative, candidates whose
decoded area in the original image is below `--small-area` use
`min(conf, small-conf)` before class-aware NMS. This setting is intentionally
disabled in the canonical profile; enable it only for a separately identified
NMS sweep and record both values in the manifest. The implementation matches
the thresholding rule in `scripts/mnn_val.py`.

The evidence manifest and metric JSON carry both numeric values, including the
disabled sentinel (`small_conf=-1`). A reference/candidate delta gate rejects
reports that omit or change either field.

### 3.1 EsMoE routing semantics

EsMoE has two materially different inference paths. Eager PyTorch inference
may dispatch only the top-k experts (`native_sparse`), whereas static export
must evaluate and blend all experts when the exporter cannot lower the
data-dependent dispatch (`dense_fallback`). These paths are not interchangeable
baselines: a backend comparison is valid only when the reference and exported
run declare the same `protocol.routing_semantics`. The export summary records
the selected path and the number of layers whose routing flags were changed;
the evidence manifest and both metric evaluators preserve that field for the
strict delta gate. Models without an MoE block use `not_applicable`.

### 3.2 Class-count safety

Class metadata is a frequent source of silent parity failures. For an explicit
vertical profile, the runner selects the canonical mapping even when
`--classes auto` is supplied. If the loaded model also exposes names and its
class count disagrees with that profile, startup fails with the expected and
observed counts. A generic/default run continues to use model metadata and
requires the evaluator's `--classes` choice to match the checkpoint.

### 3.3 Tensor-shape safety

ONNX outputs are accepted only when a floating-point rank-3 tensor has batch
one, a plausible feature dimension (`4 + nc` or larger) and a positive anchor
dimension. Both `[1, features, anchors]` and `[1, anchors, features]` are
normalized. When several rank-3 tensors remain equally plausible after the
feature/anchor checks, the ONNX and MNN adapters fail explicitly instead of
depending on exporter ordering. Rank-4 outputs are treated as segmentation
prototypes only after their dimensions are validated. MNN and NCNN apply the
same rank, dimension, finite-value and layout checks before entering the shared
decoder; MNN status-returning API calls are checked and an unavailable
accelerator is retried with a CPU session. This turns a wrong export into an
explicit diagnostic instead of a plausible-looking empty prediction file.
The MNN adapter requires float32 public input/output tensors; quantized graphs
remain eligible when quantization is internal and the graph boundary stays
float32.

## 4. Export and conversion

`scripts/export_models.py` is a wrapper around the model's native exporter. It
uses a static square input, runs ONNX checker/simplification by default and
writes an export summary containing the checkpoint digest, requested formats,
graph checks and NCNN pair status. `--no-simplify` is retained for diagnosis
only and requires `--allow-unsimplified`.

NCNN conversion may emit names other than `in0` and `out0`. The exporter writes
the actual input/output/prototype names to both `<param-stem>.metadata.yaml` and
the shared `metadata.yaml` (the latter retains legacy compatibility). The runtime validates
each declared name against the parsed `.param` graph before inference. When no
sidecar is present it resolves a unique graph endpoint, retains the historical
`in0`/`out0`/`out1` fallback, and fails closed when multiple terminal tensors make
the roles ambiguous. A prototype explicitly declared by metadata is mandatory;
missing it is an error rather than a silent box-only result. For a directory input,
exactly one matching `.param`/`.bin` pair is required unless the conventional
`model.ncnn.*` pair is present.

MNN conversion is intentionally not treated as acceptance evidence. The
converter output must load in the MNN runtime, produce a finite detection
tensor, and pass the same per-image metric gate before it is listed as a
validated backend.

## 5. Accuracy evaluation

There are two evaluators because a deployment host may not have the full
training environment:

* `eval_map.py` delegates AP matching to Ultralytics and is the preferred
  formal path when PyTorch is available;
* `eval_map_standalone.py` implements the same ten IoU thresholds with only
  NumPy and standard-library dependencies.

Both parsers are strict about column count, finite values, class range and
positive geometry. The formal path requires one prediction and one label file
for every image outside `--smoke`, rejects duplicate stems and records the
ordered image-list digest. Native VisDrone rows are accepted for diagnosis;
formal runs should use the official `visdrone2yolo` conversion so ignored
regions have defined semantics.

The result JSON exposes two distinct units:

```text
delta_mAP50-95_pp  = (candidate - reference) * 100
delta_mAP50-95_pct = (candidate - reference) / reference * 100
```

Use `--max-abs-delta-pp` for an absolute percentage-point budget. The
relative `--max-abs-delta-pct` option is retained for compatibility and cannot
be combined with the absolute gate. The relative gate requires a positive
reference metric; either gate requires the same image-list/protocol metadata.

When a gate fails, `scripts/prediction_diff.py` matches same-class boxes by IoU
and reports missing boxes, confidence differences and coordinate differences
per image. It accepts the same BOM-tolerant, quoted ordered image list as the
metric evaluators and can enforce an explicit `--image-root`; this prevents a
diagnostic report from silently analyzing a different file set. The report
separates preprocessing/decode errors from genuine model quality changes
without rerunning inference.

## 6. INT8 calibration

The quantization helper is deliberately conservative. It:

1. selects a deterministic, sorted calibration list;
2. requires at least 300 images;
3. applies the same letterbox/RGB/NCHW preprocessing contract;
4. compares calibration image content hashes with the validation list;
5. records the selected list digest, quantizer settings and exclusion patterns.

The default exclusion patterns protect detection-head, attention and routing
nodes when they exist. A pattern matching no graph node is an error, preventing
a command-line typo from silently changing the precision recipe. The generated
summary is always marked `acceptance_ready: false`; only a subsequent full
prediction/evaluation run can establish an INT8 result.

## 7. Benchmark methodology

Latency comparisons are meaningful only when the following are held constant:

* ordered image list and image decoding path;
* input size, precision, confidence/IoU policy and maximum detections;
* CPU/GPU device, runtime build and thread count;
* warm-up count and timed repeat count.

The runner reports preprocessing, inference, postprocessing and end-to-end
times per image, followed by mean, P50, P95, P99 and FPS. With
`--benchmark-json`, an optional sidecar records the resolved protocol, host
platform, compiler, CPU model, logical CPU count and build date; `--csv` retains
the per-image timing rows. The evidence manifest additionally records exact file
hashes. Virtual-machine
measurements are valid diagnostics but must be labelled as VM results and must
not be generalized to ARM or Jetson hardware.

Capture a separate host/toolchain snapshot before the run:

```bash
python3 scripts/collect_environment.py \
  --repo-root . --backend onnx --execution-provider cpu \
  --threads 4 --warmup 2 --runs 20 \
  --output artifacts/environment.json
```

The collector is dependency-free and reports missing optional tools explicitly.
Its output follows [`environment.schema.json`](environment.schema.json) and can
be attached as `--report environment=artifacts/environment.json` when creating
the evidence manifest.

For publication, report the results in a table whose values point to the
corresponding manifest and raw logs. The table is intentionally a schema, not
a set of default numbers:

| Backend | Export/checkpoint digest | Image count/list digest | mAP50-95 | Delta (pp) | End-to-end P50/P95/P99 | FPS | Host and runtime |
| --- | --- | --- | ---: | ---: | --- | ---: | --- |
| PyTorch reference | recorded in manifest | recorded in manifest | from metric JSON | -- | from timing log | from timing log | recorded in manifest |
| ONNX Runtime | recorded in manifest | recorded in manifest | from metric JSON | from metric JSON | from timing log | from timing log | recorded in manifest |
| NCNN or MNN | recorded in manifest | recorded in manifest | from metric JSON | from metric JSON | from timing log | from timing log | recorded in manifest |

No cell is a result until the reviewer can recompute it from the stated model
digest, image-list digest and per-image predictions. Compute latency and
end-to-end latency are reported separately; a virtual-machine value is labelled
as such.

## 8. Evidence manifest

`evidence_manifest.py` and `evidence-manifest.schema.json` define the release
boundary. An acceptance manifest contains:

* dataset name/split, ordered image records and a list digest;
* required training provenance, including base-model revision, dataset
  version, epoch/seed configuration and the exact training command;
* the complete protocol and class profile;
* checkpoint and every exported model, each with file hashes;
* labels and predictions with matching counts;
* calibration records and an explicit disjointness assertion for INT8;
* environment, source revision, command line, content-hashed metric/benchmark
  reports and gate values.

`validate` checks the structure; `verify` additionally recomputes hashes under
the supplied roots. The template intentionally leaves labels and predictions
null and cannot pass the acceptance validator. This prevents a release note or
an empty directory from being mistaken for a completed experiment.

## 9. Build and portability controls

The CMake target enables each backend only when its headers and library are
found. `REQUIRE_ORT`, `REQUIRE_NCNN` and `REQUIRE_MNN` turn missing SDKs into
configuration errors; `ALLOW_NO_BACKENDS` is reserved for dependency-light
CLI diagnostics. On Windows, model and image paths are converted from UTF-8 to
UTF-16 before opening; the JPEG writer uses the same wide-path handling. On
Linux, the release script computes the recursive shared-library closure and
sets an `$ORIGIN/lib` RPATH while leaving system glibc and accelerator drivers
to the target host.

The ARM64 toolchain file describes cross-compilation but does not claim that a
cross-compiled binary has run on hardware. A native Jetson run must archive the
binary, engine, device/software versions and raw log together with the same
manifest.

## 10. Reproducibility status of this checkout

The repository-level contract tests cover profile resolution, parser failures,
shape normalization, evidence-manifest gates and prediction diagnostics. A
previous Ubuntu 22.04 smoke run loaded a YOLOv5s ONNX model on one image; that
is a functional L1 check, not an EsMoE-N VisDrone accuracy result. The full
Issue #51 acceptance record remains intentionally pending until a real
EsMoE-N checkpoint, dataset split and target-platform logs are supplied.

This boundary is important: a reproducible procedure is useful only when its
limitations are stated as precisely as its successes.

## 11. Publication record

The public submission should contain a compact result table followed by links to
the machine-readable evidence. Use one row per backend and keep the protocol
identical across rows:

| Backend | Model/checkpoint SHA256 | Image-list SHA256 | Images | mAP50-95 | Delta (pp) | P50/P95/P99 (ms) | FPS | Platform |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| PyTorch reference | evidence manifest | evidence manifest | N | metric JSON | -- | timing CSV | timing CSV | environment JSON |
| ONNX Runtime | evidence manifest | evidence manifest | N | metric JSON | metric JSON | timing CSV | timing CSV | environment JSON |
| NCNN or MNN | evidence manifest | evidence manifest | N | metric JSON | metric JSON | timing CSV | timing CSV | environment JSON |

The accompanying text should state the dataset release and split, checkpoint
provenance, preprocessing and NMS parameters, runtime versions, thread policy,
warm-up/repeat counts, and the exact commands used. A platform is listed as
validated only when both compilation and an inference run were executed on that
platform. Cross-compilation, CI compilation, or a virtual-machine smoke test
must be labelled accordingly and must not be presented as native device
evidence.

For a short discussion post, use `TECHNICAL_SUMMARY_ZH.md` as the narrative and
attach the evidence manifest, metric JSON, timing CSV/JSON, prediction archive,
environment snapshot and model/export summaries. Keep all unavailable fields
explicitly marked as pending until the corresponding files can be verified.
