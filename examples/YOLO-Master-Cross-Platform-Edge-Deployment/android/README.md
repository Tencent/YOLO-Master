# YOLO-Master Android runtime (ncnn)

The Android backend runtime for the cross-platform edge example. It runs YOLO-Master
detection and segmentation on-device with **ncnn**, reusing the shared C++ core in
`../cpp` verbatim (the same letterbox -> ncnn -> decode -> NMS path as the Linux, Windows,
Jetson, and macOS runners) behind a small JNI bridge.

This milestone is the **runtime + JNI + on-device test harness only** — there is no app UI
yet. A future app module depends on `:runtime`.

## What it provides

- `libyolomaster_ncnn.so` (arm64-v8a, x86_64) wrapping the ncnn backend.
- A thin Kotlin API, `dev.yolomaster.ncnn.YoloMasterNcnn`:
  - `init(modelDir, useVulkan, threads, precision)` / `initBest(modelDir, probe, threads, precision)`
    (precision-with-parity-fallback, then Vulkan-with-parity-fallback)
  - `setConfig(conf, iou, maxDet)` (cheap retune, reuses the cached forward)
  - `infer(bitmap) -> List<Detection>`, `inferSeg(bitmap) -> SegResult`
  - `activeBackend` (the REAL resolved precision: "ncnn-CPU-fp32" | "ncnn-CPU-fp16" |
    "ncnn-CPU-int8+fp32" | "ncnn-CPU-int8+fp16" | "ncnn-Vulkan" | "ncnn-Vulkan-fp32"),
    `backendNote` (why a requested precision was downgraded), `lastError`, `close()`
  - `Precision { AUTO, FP32, FP16, INT8 }` - see "Precision modes and mixed-INT8" below
  - the app-shape API (the iOS Kit contract) - see "Forward once, tune cheap" below:
    `forwardRaw(bitmap, confFloor) -> RawOutput`, `forwardRaw(rgbaBuffer, w, h, rowStride, crop,
    rotationDegrees, confFloor) -> RawOutput`, `RawOutput.decode(conf, iou, maxDet)`,
    `RawOutput.maskOverlay(dets, maxSide, alpha, reuse) -> Bitmap?`, `inferOnly(bitmap) -> ms`,
    `isSeg`, `classNames`, `imgsz`, `lastTimings`, `YoloMasterNcnn.setPowersave(mode)`,
    `YoloMasterNcnn.palette` / `classColor(i)`
- An instrumented parity/robustness harness (`ParityTest`, `SegRawTest`, `LatencyBenchTest`).

## Forward once, tune cheap (the app-shape API)

The harness shape (`infer` / `inferSeg`) does forward + NMS in one call. An app wants the iOS
Kit shape instead: one forward per frame, then arbitrarily many cheap re-decodes (the Photo
tab's conf/IoU sliders) and mask re-renders at display size (the Live overlay), with nothing
big crossing JNI at 30 fps. That is `RawOutput`:

```kotlin
val rt = YoloMasterNcnn()
YoloMasterNcnn.setPowersave(2)                       // on the inference thread, BEFORE init
rt.init(dir, useVulkan = true, threads = 2)
rt.forwardRaw(bitmap, confFloor = 0.05f).use { raw ->  // forward + candidate decode, no NMS
    val dets = raw.decode(conf = 0.25f, iou = 0.45f)  // same C++ nms_and_cap as infer()
    val mask = raw.maskOverlay(dets, maxSide = 640)   // premultiplied ARGB_8888, null on det models
    // raw.preMs / inferMs / decodeMs / origW / origH / candidateCount / isSeg
}
```

- `RawOutput` is a **native handle** (`AutoCloseable`): the candidates (score >= `confFloor`)
  and the segmentation proto are moved out of the backend with zero copies and live on the
  native heap, not the Java heap the Photo bitmaps need. A seg raw is ~3.3 MB proto + up to
  ~1.6 MB candidates, a det raw <= 0.4 MB. `close()` is idempotent; a leaked raw is freed by the
  finalizer with a `YMNcnn` warning. Decode and mask need **no model handle**: a screen can
  close the runtime (releasing Vulkan) and keep its raws.
- `decode(conf, iou, maxDet)` returns `Detection`s with `candIndex` set (the candidate each
  came from); `maskOverlay(dets, ...)` reads that index, so masks are rendered for exactly the
  chosen subset and no coefficient array ever crosses JNI. The overlay is rendered by the
  sized `seg_overlay` overload in `cpp/src/common.cpp` at `orig * min(1, maxSide/max(orig))`
  (same box clipping and smoothstep edge as the CLI) and written **premultiplied** (Android
  composites premultiplied; straight RGBA would render the tints too bright). Pass `reuse` (a
  mutable ARGB_8888 bitmap of the same size) to avoid per-frame allocation.
- `forwardRaw(ByteBuffer, ...)` takes the CameraX `ImageAnalysis` RGBA_8888 frame directly
  (`planes[0].buffer` must be direct, `planes[0].rowStride`, `cropRect`,
  `imageInfo.rotationDegrees`): crop first, then rotate, so `origW/origH` and every box are in
  the upright frame the preview shows. The RGBA->BGR conversion is the single copy and the
  buffer is not retained: `imageProxy.close()` right after the call.
- `inferOnly(bitmap)` is the kernel-only number for benchmarks (letterbox + extractor, no
  decode, no NMS); `lastTimings` gives `preMs / inferMs / postMs` of the last forward (after
  `infer()` `postMs` = decode + NMS, after `forwardRaw()` decode only). `isSeg` reads
  `metadata.yaml` `task:` (missing = detect) without a forward; `imgsz` is the model's fixed
  input size.

**Thread policy.** The runtime is not thread-safe: own it from ONE inference thread (a
single-thread executor). On phones with two prime cores the fastest configuration is
`threads = 2` on those cores, but an app process has no affinity by default and ran 2-4x
slower than the shell bench at >= 2 threads. `YoloMasterNcnn.setPowersave(2)`
(`ncnn::set_cpu_powersave`, process-global) pins the **calling** thread and the OpenMP team
ncnn spawns from it to the big cores, so call it on the inference thread before `init` and
pass `threads = 2` explicitly (`threads = 0` still means "ncnn's big-core count"). The first
Vulkan forward includes the pipeline/shader build: run one warm-up forward behind a loading
card and do not count it.

## Robustness model (why this is not a naive ncnn wrapper)

Two ARM-only hazards silently break these models on phones; both are handled:

1. **fp16 underflow (ARM CPU) - handled PER MODEL.** ncnn enables fp16 CPU kernels on
   armv8.2 (every modern phone). The emulated mixture-of-experts router uses `1e-9` mask
   nudges and `1e30` expert masks that are unrepresentable in fp16 (1e-9 flushes to zero
   under ARM FZ16, 1e30 overflows fp16's 65504 max) and return **no detections**. The
   runtime reads that fingerprint from the `.param` itself (`meta::scan_ncnn_param` in
   `cpp/src/common.cpp`, applied in `cpp/src/ncnn_backend.cpp`): mixture models are pinned
   fp32 (with the reason in `backendNote`), while the dense models (v0.1, EsMoE-N, whose
   graphs carry no such constants) run **fp16** on armv8.2 - about -45% latency vs fp32
   (Orin measurement). An explicit `Precision.FP16` request on a mixture model is
   downgraded and explained, never run as a silent zero-detection model.
2. **Unregistered router ops.** ncnn has no `TopK/Gather/Where`, so raw gated-router MoE
   models will not load. The models are exported with `scripts/export_ncnn_mixture.py`,
   which rewrites the router into stock ncnn ops (census-gated). The Android runtime loads
   the resulting `.param/.bin` unchanged.

Vulkan (fp16, ~4x faster) is opt-in: it is used only if a GPU is actually present
(`ncnn::get_gpu_count()`), otherwise the runtime transparently falls back to the CPU
choice. The same per-model rule applies on the GPU: a mixture model runs Vulkan with the
fp16 flags off (`ncnn-Vulkan-fp32`). INT8 never uses Vulkan (no ncnn Vulkan int8 kernels).
`initBest()` additionally verifies the GPU agrees with the CPU reference on a probe image
before trusting it, so per-device driver differences cannot ship silent garbage.

## Supported models (validated ncnn set)

v0.1 (all sizes incl. the `v0.1-seg-N` default), EsMoE-N, and the four supported mixture
models (`moa-n`, `mot-n`, `moa-mot-n`, `molora-merged`). Out of scope for now: UoMoE-N
(no ncnn export exists), `molora-routed`, and yolo26 end2end-on-ncnn.

## Build prerequisites

1. **ncnn Android SDK**, tag **20260526**, `-android-vulkan` variant (matches the desktop
   SDK so `.param/.bin` are byte-parity). Download from the ncnn releases, or cross-build
   with the NDK mirroring `../jetson/24_build_ncnn.sh`.
2. **opencv-mobile** Android SDK (core + imgproc).
3. **Android NDK** r26+ and a JDK 17.

Keep both SDKs out of git (they are large). Then:

```bash
cd android
cp sdk-paths.example.properties sdk-paths.properties
$EDITOR sdk-paths.properties     # set NCNN_ANDROID_ROOT and OPENCV_ANDROID_ROOT
./gradlew :runtime:assembleRelease
```

## Run the on-device harness

The models are not committed. Stage your local ncnn model dirs and a probe image, then run
the instrumented tests on a connected arm64 device:

```bash
./scripts/stage_models.sh                 # copies models/*_ncnn -> runtime assets (gitignored)
# put any scene image at runtime/src/main/assets/probe.jpg
./gradlew :runtime:connectedAndroidTest
```

The harness (`ParityTest`) asserts:
- the `v0.1-seg-N` default with explicit `FP32` reports **ncnn-CPU-fp32** and detects (the
  original fp16-underflow regression guard), and with `AUTO` reports **ncnn-CPU-fp16** on
  arm64 (fp32 on the x86_64 emulator, where ncnn's fp16 kernels do not exist) and still
  detects within +/-1 of the fp32 count - the fp16 guard for a dense model;
- an emulated-router mixture model (`moa-n`) is pinned **ncnn-CPU-fp32** by fingerprint
  (`backendNote` says why), detects on ARM, and an explicit `FP16` request is refused;
- Vulkan agrees with CPU within +/-1 detection, or falls back to the CPU choice cleanly;
- a missing model fails with an error, not a crash; `INT8` with no `-int8_ncnn` sibling
  fails with an error naming int8 (never a silent float fallback).

`SegRawTest` covers the app-shape API on `v0.1-seg-N`: `forwardRaw` + `decode(0.25, 0.45)`
equals `infer()` box for box (1e-3) on CPU and on Vulkan (det count within +/-1 of CPU when a
GPU exists), the mask overlay at `maxSide = 640` is non-empty and premultiplied, the direct-RGBA
path equals the Bitmap path, and `setPowersave(2)` + `threads = 2` leaves the detections
unchanged.

Latencies are logged under the `ParityTest` / `SegRawTest` / `YMNcnn` tags (`adb logcat`).

## Precision modes and mixed-INT8

`Precision.AUTO` is the default and the right choice for apps: the precision is derived from
the model (see hazard 1 above). `FP32` / `FP16` are explicit overrides (downgraded with a
`backendNote` when the model cannot honour them). `INT8` loads a **pre-quantized sibling
directory** named `<name>-int8_ncnn` next to the float dir - the same three files - produced
by `scripts/quantize_ncnn_int8.py` (mixed per-layer int8: the conv trunk is int8, the stem
pair, DFL and any router layers stay float; see its `metadata.yaml` `quant:` block for the
exact provenance). INT8 is CPU-only; its float remainder follows the same per-model rule
(fp16 for dense models, fp32 for mixture models). ncnn loads a mixed int8/float graph
natively - there is no runtime switch to flip; the mode simply selects which directory is
opened. Stage the siblings with `scripts/stage_models.sh` (they are optional: a missing one
only skips the INT8 rows).

`LatencyBenchTest` runs `(model, precision) x threads {1, 2, 4, big}` and logs one parseable
line per configuration (warmup 10, 50 timed end-to-end infers on the real probe):

```
YM_LAT model=.. precision=.. backend=.. abi=.. threads=.. n=.. median_ms=.. p90_ms=.. dets=.. fp32_dets=.. note=".."
```

Capture with `adb logcat -s ParityTest | grep YM_LAT`. `cpp/tools/ncnn_bench` gives the
kernel-only number for the same variants. Speed verdicts come **only** from an arm64 device:
x86 cannot run ncnn's fp16 kernels at all, and its int8 path (AVX512-VNNI) says nothing about
ARM `sdot`/`i8mm`. INT8 accuracy, on the other hand, is certified on Linux first
(`tests/certify_ncnn_int8.py`, gate: mAP50-95 within 1.0 point of fp32) before any speed
claim is made.

## Layout

```
android/
  settings.gradle.kts  build.gradle.kts  gradle.properties
  sdk-paths.example.properties            # template (real one gitignored)
  scripts/stage_models.sh
  app/                                    # com.android.application = the YOLO-Master app (Compose)
    build.gradle.kts                      # Compose 1.6 / Kotlin 1.9.24, CameraX 1.3.4, Haze 0.7.3
    src/main/assets/models/               # staged by scripts/stage_models.sh --module app (gitignored)
    src/main/kotlin/dev/yolomaster/app/
      ui/{live,photo,bench,settings}/     # the four tabs of the iOS app, screen for screen
      ui/{hud,overlay,common,theme}/      # StatsHUD widgets, the 5 box styles, materials, tokens
      detect/Detector.kt                  # iOS-Kit-shaped wrapper: open/forward/decode/maskOverlay/inferOnly
      model/, system/                     # model catalog + naming, haptics/thermal/gallery/picker
  runtime/                                # com.android.library = the runtime
    build.gradle.kts
    src/main/cpp/{CMakeLists.txt, jni_bridge.cpp}
    src/main/kotlin/dev/yolomaster/ncnn/{YoloMasterNcnn,RawOutput,Types}.kt
    src/main/assets/models/.gitkeep       # staged models land here (payloads gitignored)
    src/androidTest/kotlin/.../{ParityTest,SegRawTest,LatencyBenchTest}.kt
```

The `cpp/` core is not duplicated; `runtime/src/main/cpp/CMakeLists.txt` compiles
`../../../../../cpp/src/{common,ncnn_backend,stb_impl}.cpp` directly, the same way the iOS
app reuses `mac/Sources/YOLOMasterKit`.

## The app (`:app`)

A function-for-function port of the iOS app (`dev/ios`): tabs Live (CameraX preview + async
overlay, lens stops, tap-to-focus, torch, full-res shutter with the overlay baked and saved to
`Pictures/YOLO-Master`, stats HUD with the thermal tachometer), Photo (up to 100 images, 3-up
gallery / zoomable pager, conf/IoU retune from cached raw outputs, export), Bench (cold run of
every model x GPU/CPU, sustained runs with sparkline + thermal bar, history, CSV share) and
Settings (about, licenses, privacy, the CPU toggle, custom model import, erase history).

Defaults: compute = GPU (ncnn Vulkan; fp16 for fp16-safe models, fp32 pinned for the
router-emulated mixture models), 2 CPU threads pinned to the big cores, INT8 siblings listed as
their own models (`YOLO-Master-<stem>-int8`, CPU only). Build:

```
scripts/stage_models.sh --module app       # copies the 7 model dirs into app assets (~67 MB)
gradle :app:assembleRelease                # -> app/build/outputs/apk/release/app-release.apk
adb install -r -g app/build/outputs/apk/release/app-release.apk
```

`local.properties` may carry `YM_KEYSTORE=/path/to/dev.jks` (+ `YM_KEYSTORE_PASSWORD`,
`YM_KEY_ALIAS`, `YM_KEY_PASSWORD`) so release builds from one machine install over each other;
without it the AGP debug key is used.

## Not in this milestone

UoMoE-N ncnn export, AAR publishing, Play Store signing, upstream PR.
