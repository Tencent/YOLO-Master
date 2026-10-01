# Upstream sync: examples/YOLO-Master-Cross-Platform-Edge-Deployment (issue 51 branch)

Date: 2026-09-12. Source: Tencent/YOLO-Master `main` at 44de0bc, contributor commits by
lsl-arch (18 commits, 2026-08-26 to 2026-09-04) on top of the example as synced from this repo
at v1.1.1 (edge commit b5ebe71). Their base predates the Android runtime, the per-model fp16
policy, the API server, the core library split and the TensorRT engine builder in this repo,
so the contributor patch was reviewed hunk by hunk instead of merged wholesale.

## Taken

| area | what | why |
|---|---|---|
| `cpp/src/slicing.cpp`, `cpp/include/slicing.hpp` | `SliceOutput.pre_ms` / `post_ms` accumulated over the global pass and every tile | real fix: sliced `[summary] pre=/post=` used to report the last tile only |
| `cpp/src/annotate_export.cpp` | UTF-8 paths on Windows, checked JPEG writes, deterministic polygon order | same bytes and formats, fewer silent failures |
| `cpp/src/common.cpp` `meta::parse_names_dict` | key/value parser for Python repr, JSON object and JSON list forms | real fix: JSON-quoted numeric keys used to become class names and shift the class ids |
| `cpp/aarch64-toolchain.cmake` | cross toolchain file | self-contained, unreferenced |
| `gui/src/app.cpp`, `gui/src/main_win.cpp` | folder-scan and single-image timings use the sliced aggregate; `NOMINMAX` guard | correct timing display; harmless |
| `jetson/00_setup.sh`, `21_build_trt_runner.sh` | TensorRT version print; `set -euo pipefail`, build logs, executable check, optional `TENSORRT_ROOT`/`CUDA_ROOT` | additive hardening |
| `jetson/20_build_runner.sh`, `22_build_ort_trt.sh` | the same hardening, merged by hand | our ncnn/MNN/lean-OpenCV detection and the concrete ORT provisioning recipe are kept |
| `jetson/30_package.sh` | model stem derived from `ONNX=`, quoted generated script, safe copy | correct for custom models; the measured Orin numbers are restored in the bundle README |
| `scripts/package_linux.sh` | optional ncnn/MNN staging, recursive ELF closure (also for the CUDA provider), no unattended `apt-get` | same bundle layout; the `REQUIRE_*` CMake flags that do not exist here were dropped |
| `scripts/mnn_val.py` | parameterized (`--limit --imgsz --threads --nc --max-det --small-conf --small-area`) | decode unchanged at defaults |
| `scripts/collect_environment.py`, `scripts/prediction_diff.py`, `scripts/evidence_manifest.py`, `environment.schema.json`, `evidence-manifest.*.json` | environment capture, per-image prediction diff, hash-pinned evidence manifests | new, self-contained, no network; schema ids retargeted to this repo |
| `requirements-edge.txt` | Python deps for the scripts | `ultralytics` pinned to the 8.4.101 lineage this repo is validated on |
| `TECHNICAL_SUMMARY_ZH.md` | Chinese protocol summary | the unlogged "970.873 ms" smoke number removed; points at this repo's measured reports |

## Rejected, with the reason

| area | reason |
|---|---|
| `cpp/src/main.cpp` rewrite | exit code 6 on any unreadable input, exit 2 on an imgsz mismatch (was a warning), `[model]` banner format change (breaks the macOS/iOS parity parsers), `flag:sku` renamed, duplicate stems fatal (the runtime already disambiguates), `--profile` presets that silently change conf/iou/imgsz |
| `cpp/src/common.cpp` other hunks | recursive directory scan (changes `frames=` for every existing run), `.tif/.tiff/.webp` dropped from sources, 30000-candidate NMS cap (numerics change), objectness auto-detection that misreads a 1-coefficient seg head, `read_ncnn_yaml` returning true with an empty name list |
| `cpp/src/ncnn_backend.cpp`, `ort_backend.cpp`, `mnn_backend.cpp`, `trt_backend.cpp`, `backend_factory.hpp`, `CMakeLists.txt` | conflict with the per-model fp16 policy, the engine builder, the MNN precision knob, the core library split and the server; hard throws on a class-count mismatch where the runtime used to decode; per-frame full-tensor copies and NaN scans on the hot path; TensorRT 8 (JetPack 5) support removed by a header `#error`; ORT SONAME chosen by sort order; `trt_int8_enable` default flipped |
| `cpp/run_tests.sh` | weakens T6 (onnx vs ncnn parity) to a smoke, rewrites T8/T15/T18 to pass under the new exit codes, leaves T7 broken; `tests/run_tests.sh` keeps the strict contract |
| `scripts/eval_map.py`, `scripts/eval_map_standalone.py` | JSON-only output, `--names-yaml`, `evaluate()` and `load_names_yaml()` removed, mandatory `--routing-semantics` and a 500-image floor: breaks `scripts/server/run_comparison.sh` and `tests/certify_ncnn_int8.py`; the per-class image count is also miscomputed |
| `scripts/quantize_int8.py` | defaults changed to QOperator and a head/attention/routing exclusion list, so every recorded INT8 number would stop reproducing |
| `scripts/mnn_parity.py` | duplicate of `tests/mnn_parity.py` with an arbitrary 0.1 absolute pass/fail tolerance |
| `scripts/export_models.py` | thin `YOLO.export` wrapper without the SDPA/router rewrites, numerical gates or legacy shims of `export_onnx_dense.py` / `export_ncnn_dense.py`; writes pnnx output next to the checkpoint; opset 12 |
| `TECHNICAL_REPORT.md` | full replacement that deletes every measured table (VisDrone 548-image accuracy, latency, Jetson 35.7 FPS) for empty schemas |
| `jetson/DEPLOYMENT_LOG.md`, `jetson/README.md` | the real Orin log replaced by an all-TBD template; the README rewrite drops the v1.1.0 section and points at paths that only exist in the upstream monorepo |
| `README.md` | out of scope by instruction |

## Verification

Workstation (Ubuntu 20.04, ORT 1.18 CPU, ncnn x86): `tests/run_tests.sh` 18/18, `tests/run_server_tests.sh` 16/16,
`parse_names_dict` unit check on the four name-map forms, and old-vs-new binary parity on 20 COCO
val images for ONNX and ncnn (2214 detections each, zero delta). The sliced-run summary now reports
accumulated pre/post times, which is the intended change.
