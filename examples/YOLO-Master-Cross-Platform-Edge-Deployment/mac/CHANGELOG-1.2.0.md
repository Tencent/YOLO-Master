# macOS Core ML Runner - change log, v1.1.x -> v1.2.0

Compiled from `git diff v1.1.1..HEAD -- mac/` on branch `dev/v1.2.0-mac` (four commits on top of
the v1.2.0 Linux phase). The user-facing summary is `RELEASE_NOTES-1.2.0.md`; this file is the
engineering record.

| file | change |
|---|---|
| Sources/YOLOMasterCore/ (new) | the portable C++17 core shared with `cpp/`: `metrics_core`, `tracker_core`, `bench_stats`, C shim `include/ymcore.h` |
| Package.swift | target `YOLOMasterCore` (C header only, `cxxLanguageStandard .cxx17`), Kit depends on it |
| Sources/YOLOMasterKit/Core.swift (new) | Swift face of the core: `YMCore`, `StageStats`, `SustainedSummary`, `MapEvaluator`, `CoreSelfTest` |
| Sources/YOLOMasterKit/Tracker.swift (new) | `Tracker`, `TrackerConfig`, `CameraMotion`, `CameraMotionEstimator` |
| Sources/YOLOMasterKit/Motion.swift (new) | `VisionCameraMotion` (translational registration) |
| Sources/YOLOMasterKit/Bench.swift (new) | `BenchDocument` (+ optional `model.moe_export`, `host_meters`), `BenchEnvironment`, `BenchRunner`, `AccuracyRunner` |
| Sources/YOLOMasterKit/TxtDump.swift (new) | `TxtDump`, `TxtDumpWriter` |
| Sources/YOLOMasterKit/Preproc.swift (new) | `PreprocDevice`, `MetalPreprocessor` (runtime-compiled kernel) |
| Sources/YOLOMasterKit/Detector.swift | `Detection.trackId`, `RawOutput.preMs/preGpuMs`, `Result.preMs/postMs`, `preprocDevice`, GPU `forward`, `inputTensorBytes`, `modelURL`, `metadata` |
| Sources/YOLOMasterKit/Pipeline.swift | `runFolder` / `runVideo` hooks (`onResult`, `limit`, `tracker`, optional output), `MotionLog`, `trackCached`, `exportVideoCached(tracked:)`, `InferSummary` stage means |
| Sources/YOLOMasterKit/Annotate.swift | `#id` labels and per-id colours |
| Sources/YOLOMasterCoreML/main.swift | `--save-txt`, `--limit`, `--bench*`, `--accuracy`, `--track*`, `--cpu-preproc`, `--dump-input`, `--core-selftest` |
| Sources/YOLOMasterApp/YOLOMasterApp.swift | Bench section, Tracking picker, Preprocess > Device, stage / tracks stat rows, engine tracking and bench methods |
| Sources/YOLOMasterApp/BenchMode.swift (new) | Bench mode: models / units / protocol sidebar, the dashboard (charts, comparison, results), meters (SMC die temperature, IOKit battery power), history store, CSV (45 columns) and JSON (per cell, per run) export |
| Sources/YOLOMasterApp/Camera.swift | preprocessing device hot-swap |
| Sources/YOLOMasterApp/Info.swift, make_app.sh, scripts/release.sh | version 1.2.0 from the repo `VERSION` file |
| tests/run_mac_tests.sh (new) | the Mac battery (M1 to M5c) |

---

## 1. The portable core (commit `core: extract ...`)

`cpp/src/map_metrics.cpp`, `tracker.cpp` and `bench.cpp` were split: everything that does not
need OpenCV, nlohmann or a `Backend` moved verbatim into `mac/Sources/YOLOMasterCore/` (SwiftPM
needs its targets inside the package; CMake reaches across and builds the same directory as the
static library `yolomaster_ccore`). The OpenCV-typed wrappers keep their public API, so the Linux
CLI and the API server are unchanged; the Linux outputs were proven identical before and after
(tracker txt dumps for both trackers on the synthetic clip, `[accuracy]` on visdrone50 and
coco500, `yolomaster_score --per-class`, bench JSON minus timing keys; the 23 CLI and 22 + 5
server tests pass). `yolomaster_score` now links the core alone, which fails the build if an
OpenCV include ever leaks in.

The C shim (`ymcore.h`) is plain C11: opaque tracker handle, flat arrays for the mAP input,
`ym_round6`, label loading and the ultralytics label-path rule, floor-rank `ym_stats_reduce`,
`ym_sustained_summary`, sha256 helpers. Swift imports it as a C module; no C++ interop mode, so
the iOS project needs no change.

## 2. Timing seam

`RawOutput.preMs` (letterbox + tensor build, or texture upload + kernel + wait on the GPU path)
and `Result.preMs / postMs` (decode + NMS measured in `detect`). `InferSummary` carries the
mean preprocess / postprocess when the caller collected them; the stats panel shows the rows
only then. "Model-only" and "Overall" keep their meaning.

## 3. `--save-txt`

`TxtDump.values` computes the five numbers in Float the way the C++ writer does (`x2 = x + w` in
float), `TxtDump.g` prints `%g` of the float promoted to double; `TxtDump.predBox` feeds the same
values through `ym_round6` to the scorer. Video frames are named `<stem>_%06d`, stems are made
unique with `uniqueStem`. Empty files are written for images without detections.

## 4. Bench and accuracy

`BenchRunner.coldRun` and `sustainedLoop` mirror `bench::cold_run` / `sustained_loop`
(gray-114 probe, `inferOnly`, one median per second, `summarize_sustained` from the core).
`AccuracyRunner.run` uses the confidence floor `Float(0.001).nextDown` because the Kit keeps
candidates with score `>` floor where the C++ keeps `>=` conf; the Kit's decode is already
multi-label. `BenchDocument` is `Codable` with the same keys as `bench::to_json`, `tool =
"macos"`, `model.execution_provider` = `CoreML-ANE | CoreML-GPU | CoreML-CPU | CoreML-ALL` (ANE = `cpuAndNeuralEngine`, ALL = Core ML's own partitioning), `ep_note` =
`preproc=cpu|gpu`. The app accumulates cold, sustained and accuracy into one document per
model + compute unit.

## 5. Tracking

`Tracker.update(dets, motion:)` wraps `ym_tracker_update`; the returned detections carry
`trackId`, the Kalman box, the last matched score and the mask coefficients of the matched
detection (coasting tracks have none). `trackCached` re-runs NMS at `min(conf, track_low)` and
the tracker over cached candidates in frame order, filtering the output back to the user's conf.
`runVideo` (CLI) tracks inline; the app records `MotionLog` during `inferVideo` and recomputes
tracks off-main on every settings change (superseding runs are cancelled), feeding playback,
scrubbing and `exportVideoCached(tracked:)`. `VisionCameraMotion` registers the previous frame
onto the current one with `VNTranslationalImageRegistrationRequest` at a long side of up to 1280 px (y negated to top-down) and
scales the translation back; `YM_MOTION_DEBUG=1` prints every estimate.

## 6. Metal preprocessing

`MetalPreprocessor` compiles the kernel from a string at first use (SwiftPM does not build
`.metal` files), keeps two shared-storage output buffers and one upload texture, and wraps the
output as `MLMultiArray(dataPointer:...)` with the buffer retained by the deallocator.
`Detector.forward` takes the GPU path when `preprocDevice == .gpu` and Metal is available, and
falls back to the CPU path on any Metal failure. Camera pixel buffers go through
`CVMetalTextureCache` (BGRA, zero copy). The Kit default is `.cpu` (iOS unchanged); the macOS
CLI and app set `.gpu`. The geometry differs from the CPU path on purpose (integer pads,
bilinear), which is what makes `scripts/preproc_compare.py` meaningful.

## 7. Known limits

- Tiles (`forwardPadded`) stay on the CPU path in this release.
- BoT-SORT compensation is translational on macOS (no rotation / zoom term); Vision quantizes the
  translation to whole pixels of the registered frame (registered at up to 1280 px on the long side).
- The in-process mAP agrees with `eval_map*.py` within 0.0005 on Core ML fp16 outputs, not to the
  4th decimal: ultralytics sorts confidences and IoU matches with an unstable `np.argsort`, and fp16
  outputs carry many exact ties after the six-digit rounding, so the reference's tie order is
  implementation-defined (Linux fp32 outputs have no ties; T20 is exact there).
- `scripts/preproc_compare.py` is a kernel check, not a decoder check: `--dump-input` also writes
  the decoded source pixels (ImageIO and libjpeg differ by several levels on chroma) and the
  reference letterbox runs on those.
- `--accuracy` and the app's Accuracy pass take detection models only.
- The Mac battery and the first Core ML mAP row require a Mac; nothing here was compiled on the
  Linux side except the portable core (g++ -std=c++17, the CMake build and the C-cleanliness
  check of `ymcore.h`).
