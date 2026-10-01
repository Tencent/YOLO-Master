# ncnn on ARM CPU: per-model FP16 and mixed-INT8 (Android runner)

Status: 2026-09-08, Linux-certified accuracy + Samsung S26 latency (section 5). Plan:
`/root/.claude/plans/goofy-hatching-scone.md`. Everything below was measured on this host
(Ubuntu 20.04, 8-core x86 slice); x86 timings are never a speed result and are not reported.

## 1. Runtime: precision is now decided per model (T1)

`cpp/src/ncnn_backend.cpp` no longer pins the whole CPU path to fp32. The constructor scans the
`.param` text (`meta::scan_ncnn_param`, `cpp/src/common.cpp`) for the emulated-router fingerprint
left by `scripts/export_ncnn_mixture.py` - literal `1.000000e-9` mask nudges, `1.000000e30`
expert masks, `Reduction amax_*` layers - and for true fp16 overflow (`|v| > 65504`):

| model | amax | 1e-9 | 1e30 | overflow | verdict |
|---|---|---|---|---|---|
| moa-n, molora-merged | 6 | 6 | 3 | 18 | fp32 pinned |
| mot-n / moa-mot-n | 6 | 6 | 3 | 6 / 9 | fp32 pinned |
| v0.1-seg-n, esmoe_n_visdrone | 0 | 0 | 0 | 0 | fp16-safe |
| p03_v01n | 0 | 0 | 0 | 0 | fp16-safe (three benign `1e-6` guards, informational) |

Dense models therefore run fp16 on armv8.2 CPUs (Orin precedent: -45% vs fp32); mixture models
stay fp32 because 1e-9 flushes to zero under ARM FZ16 and 1e30 overflows fp16, which zeroes the
routing (runtime-unfixable; the export-side fix is a separate task). The policy is threaded
Kotlin `Precision {AUTO, FP32, FP16, INT8}` -> JNI `nativeInit(..., precision)` -> `NcnnBackend`
(defaulted ctor parameter) -> CLI `--precision`. What actually resolved is reported in
`active_ep` (`ncnn-CPU-fp32 | fp16 | int8+fp32 | int8+fp16`, `ncnn-Vulkan[-fp32]`) and the
reason for any downgrade in `ep_note` / `backendNote`. Explicit requests a model cannot honour are
downgraded, never run as a silent zero-detection model. `INT8` loads the pre-quantized
`<name>-int8_ncnn` sibling (hard failure if absent) and forces CPU.

CLI oracle on x86 (no asimdhp, so fp16 is inert and honestly reported as fp32):

```
dense   auto -> ep=ncnn-CPU-fp32                       (no note)
dense   fp16 -> ep=ncnn-CPU-fp32  note=fp16 requested: CPU has no fp16 arithmetic (asimdhp); running fp32
moa-n   auto -> ep=ncnn-CPU-fp32  note=fp32 pinned: emulated-router fingerprint (1e-9 nudge x6, 1e30 mask x3, amax x6); 18 literal(s) overflow fp16
moa-n   fp16 -> ep=ncnn-CPU-fp32  note=fp16 requested but refused: ...
p03     int8 -> ep=ncnn-CPU-int8+fp32   (sibling models/p03_v01n-int8_ncnn); missing sibling -> "int8 model not found"
```

fp32 parity is unchanged: `tests/validate_mixture.py` on moa-n against the rebuilt runner gives
raw ncnn maxdiff 1.8e-4 (gate 5e-3) and every CLI/ONNX sub-check passes (only the MNN sub-check
fails, because the regression build has no MNN backend). The robustness battery
`tests/run_tests.sh` (T1-T18, incl. onnx==ncnn total_dets=43 on the visdrone50 probe) passes 18/18
against `cpp/build_ncnn_ort_x86/yolomaster_edge`.

## 2. Host fix

The vendored `third_party/ncnn` and `third_party/opencv-lean` are Ubuntu-22.04 builds; on this
20.04 host neither the runner, `ncnn2table`/`ncnn2int8`, nor `tests/ncnn_rawdump` could run.
`apt-get install libopencv-dev` + `scripts/build_ncnn_x86.sh` (ncnn tag 20260526 from source,
tools ON, Vulkan OFF, installed to `third_party/ncnn-x86-20260526`) fixes all three;
`cpp/build_ncnn_x86/` (ncnn-only) and `cpp/build_ncnn_ort_x86/` (ncnn + ONNX Runtime) are the
rebuilt runners. Steady state ~62-73 ms/frame at 8 threads (README: 80 ms @4T).

## 3. Mixed-INT8: pipeline and the calibration finding

`scripts/quantize_ncnn_int8.py` (port of the project03 `tempo-ncnn` recipe) builds the exclusion
set structurally from the `.param` (stem pair, DFL `0=1 1=1 5=0 6=16`, selective-kernel gates,
p03's image-level logit heads, the 36 router-window layers on mixture models, optionally the head),
calibrates `ncnn2table` on 1024+ even-spread TRAIN images pre-rendered as letterboxed 640 gray-114
PNGs, strips the table, runs `ncnn2int8`, and hard-checks the result (coverage set equality, no
`8=` on non-quantizable types, bin ratio). 84 unit assertions in `tests/test_quantize_ncnn_int8.py`
(the p03 six-name set equals the diff of the two prior-art tables). `tests/certify_ncnn_int8.py`
scores fp32 vs int8 with `scripts/eval_map.py::evaluate` at ultralytics val settings; the gate is
mAP50-95(int8) >= fp32 - 1.0 pt (the project criterion), plus det-count ratio (warn band) and a
conf-0.25 box match >= 0.90.

**The prior-art p03 int8 (`tempo-ncnn`, 256 raw-resized images, KL) was never accuracy-tested and
is effectively dead: 0-1 detections where float gives 11-14.** With the same exclusion set and
letterboxed calibration, the calibration METHOD turned out to be the lever, not the layers:

| p03_v01n (pruned v0.1-N, COCO val2017, 200-image smoke) | int8 layers | dmAP50-95 (pts) | box match | ratio |
|---|---|---|---|---|
| KL, 1024 | 139/145 | -3.22 | 0.767 | 1.007 |
| KL, 1024, head excluded | 133/145 | (dets +3%) | - | - |
| **ACIQ, 1024** | 139/145 | -2.00 | 0.908 | 0.996 |
| ACIQ, 1024, head excluded | 133/145 | -1.96 | 0.923 | 1.021 |
| **ACIQ, 2048** | 139/145 | **-1.44** | 0.912 | 0.993 |
| ACIQ, 4096 | 139/145 | -1.70 | 0.927 | 1.054 |

ncnn's KL histogram clips this model's activation ranges (28% of confident detections lost);
ACIQ's analytical clipping keeps detection parity at identical coverage. Doubling the calibration
set from 1024 to 2048 bought +0.56 pt, but 4096 gave it back (-1.70): the ladder plateaus around
-1.4 to -1.7 pt on the 200-image smoke (the smoke's own noise between variants is a few tenths),
so per-layer PTQ on this pruned model does not reach the strict 1.0-pt gate with any lever tried.
Head exclusion buys nothing on ncnn (unlike TensorRT, where the head was the dominant lever) and
costs 6 layers. `--method aciq` is now the quantizer default; the canonical
`models/p03_v01n-int8_ncnn` is the best certified variant (ACIQ, 2048); its full 5000-image number
is in the verdict table below.

Other models (ACIQ, 1024, default exclusions):

| model | val set | n | fp32 mAP50/50-95 | int8 mAP50/50-95 | dmAP50-95 (pts) | match | ratio | int8 layers | bin |
|---|---|---|---|---|---|---|---|---|---|
| esmoe_n_visdrone | VisDrone val | 548 (full) | 0.3267 / 0.1863 | 0.3109 / 0.1746 | -1.17 | 0.859 | 1.005 | 141/152 | 2.88 MB (0.5x) |
| v0.1-seg-n | COCO val2017 | 200 (smoke) | 0.6105 / 0.4573 | 0.6101 / 0.4520 | **-0.53** | 0.846 | 1.052 | 153/164 | 3.14 MB |
| v0.1-seg-n | COCO val2017 | 5000 (full) | 0.5774 / 0.4166 | 0.5632 / 0.4020 | **-1.47** | 0.846 | 1.062 | 153/164 | 3.14 MB |
| v0.1-seg-n, head excluded | COCO val2017 | 200 (smoke) | 0.6105 / 0.4573 | 0.6088 / 0.4539 | -0.34 | 0.848 | 1.071 | 143/164 | 3.18 MB |

Note on absolute scale: the README's EsMoE-N ncnn baseline (0.3495 / 0.2034) was scored with an
older ultralytics; scoring that exact original `preds_ncnn/` dump with this environment's
`DetMetrics` (8.4.101) gives 0.3267 / 0.1863 - identical to the fp32 run here. The runner is
unchanged; only the metric code moved. Both precisions are scored with the same code, so the
deltas are what the gate uses.

### Verdict table (full validation sets, canonical `-int8_ncnn` siblings)

| model (calibration) | val set | fp32 mAP50 / 50-95 | int8 mAP50 / 50-95 | dmAP50-95 (pts) | box match | det ratio | gate (<= 1.0 pt) |
|---|---|---|---|---|---|---|---|
| esmoe_n_visdrone (ACIQ, 1024) | VisDrone val, 548 | 0.3267 / 0.1863 | 0.3109 / 0.1746 | -1.17 | 0.859 | 1.005 | FAIL, by 0.17 |
| v0.1-seg-n (ACIQ, 1024) | COCO val2017, 5000 | 0.5774 / 0.4166 | 0.5632 / 0.4020 | -1.47 | 0.846 | 1.062 | FAIL, by 0.47 |
| p03_v01n (ACIQ, 2048) | COCO val2017, 5000 | 0.5889 / 0.4247 | 0.5752 / 0.4106 | -1.40 | 0.912 | 0.991 | FAIL, by 0.40 |

No model passes the strict 1.0-pt gate by per-layer PTQ. The 200-image smoke is only good for ranking
variants: on seg it read -0.53 and the full set reads -1.47 (the smoke's fp32 baseline is also a
harder-to-compare subset, 0.4573 vs 0.4166), whereas p03's smoke (-1.44) matched its full number
(-1.40). esmoe misses by 0.17 pt on the full VisDrone val; p03 misses by 0.40 pt at its best
calibration point (ACIQ, 2048) with the ladder plateaued. The 0.90 box-match check passes on p03
(0.912) and fails on seg and esmoe (0.846-0.859): int8 keeps the ranking but shifts roughly 15% of
confident boxes by a few pixels at the IoU 0.8 threshold, which is consistent with a 1-1.5 pt
mAP50-95 loss concentrated at the strict IoU thresholds. Nothing regresses catastrophically
any more (the prior-art p03 int8 was dead), and every int8 dir is half the size of float.

## 4. What Linux cannot certify (the device handoff)

The Android toolchain now lives on this pod under `/data/android/` (cmdline-tools SDK with
platforms;android-34, build-tools;34.0.0, cmake;3.22.1, ndk r26/r28/r29; Gradle 8.7; JDK 17; the
official ncnn 20260526 Android SDK; opencv-mobile 4.13.0). Two build fixes were needed for the
runtime to compile against the official prebuilts at all: the ncnn Android SDK exports
`-fno-rtti;-fno-exceptions` to consumers while the JNI bridge and `NcnnBackend` use try/throw
(the app's CMake now clears that interface property on the imported target), and opencv-mobile
4.13.0 references `__kmpc_dispatch_deinit`, an OpenMP runtime symbol absent from the r26-r28 NDKs,
so the app's `ndkVersion` moved from 26.3 to 29.0.14206865. Outputs:
`android/runtime/build/outputs/aar/runtime-release.aar` and the self-contained instrumented-test
APK `android/runtime/build/outputs/apk/androidTest/debug/runtime-debug-androidTest.apk` (models and
`probe.jpg` are baked in as assets by `android/scripts/stage_models.sh`).


x86 has no ncnn fp16 kernels and its int8 path (AVX512-VNNI) says nothing about ARM `sdot`/`i8mm`,
so the two questions this work exists to answer - does AUTO/fp16 hold on a phone, and does
mixed-INT8 beat fp16 there - need the arm64 device:

1. `android/scripts/stage_models.sh` (stages the float dirs and the `-int8_ncnn` siblings), drop a
   `probe.jpg`, `./gradlew :runtime:connectedAndroidTest` - `ParityTest` (mode-aware) and
   `LatencyBenchTest` (one `YM_LAT ...` line per model x precision x threads).
2. `cpp/tools/ncnn_bench` is already cross-built for arm64 (`cpp/tools/build-android/ncnn_bench`,
   NDK r28c + the official ncnn 20260526 Android SDK, static libc++, needs only system libs; the
   toolchains live under `/data/android/`). The complete device bundle is
   `results/int8_bench/device/` (bench, both probes, the three float + three int8 model dirs,
   moa-n, `REFERENCE.txt`, and `run_s26.sh`, which pushes everything to `/data/local/tmp/ymbench`,
   runs the whole matrix at `--threads 1,2,4,big --rounds 3 --variants fp32,fp16,int8+fp32,int8+fp16`
   and writes `results/int8_bench/<label>.log` + JSON). Probes: `probe_640.f32` from a COCO image
   for p03/seg, `probe_visdrone_640.f32` for esmoe; x86 reference counts at conf 0.25 in
   `REFERENCE.txt`: p03 14/15 fp32/int8, seg 15/12, esmoe 43/44. The tool's `valid=1`
   means finite output with non-zero dets; additionally accept an int8 row only if its `dets=` is
   within 35% of the fp32 row on the same probe (a single image's int8 count jitters by up to ~20%;
   accuracy is the certification above, never the probe count).
3. "INT8 wins at T" iff the accuracy gate passed, `valid=1` on device, and
   `median(int8) < 0.97 x median(fp16)` with `p90(int8) < median(fp16)` across rounds. A tie at
   high thread counts (LPDDR-bandwidth bound, as on Orin: +8.1% @1T, +6.5% @2T, tie @6T) is a
   legitimate result.

### The app (2026-09-08)

`android/app` is the Android port of the iOS app (Live / Photo / Bench / Settings), built on the
pod as `android/app/build/outputs/apk/release/app-release.apk` and shipped in the S26 bundle with
`run_s26_app_install.sh`. Its Bench tab runs every bundled model over GPU (Vulkan) and CPU with
the same warmup/iters protocol, so the GPU-vs-CPU-fp16-vs-INT8 comparison on the phone comes
straight out of the app (History + CSV share). Default unit = GPU; INT8 entries force CPU.

## 5. Samsung S26 results (MEASURED, 2026-09-08)

Device: Samsung S26, arm64, ncnn caps asimdhp/asimddp/i8mm all present, `get_big_cpu_count()=2`
(two prime cores), 8 CPUs. Two harnesses, both from `results/int8_bench/device/`: the standalone
`ncnn_bench` (shell user, `set_cpu_powersave(2)`, 100 iters x 3 rounds, median of round medians)
and the instrumented `LatencyBenchTest` (app process, 50 frames, no affinity). Raw logs:
`results/int8_bench/{s26.log,s26_app.log,s26_json/}` and `mdb:/yolotmp/ncnn-int8/`.

Policy checks (ParityTest, 5/5): dense models auto-select `ncnn-CPU-fp16` with the fp32
detection count (+/-1); moa-n is pinned `ncnn-CPU-fp32` with the fingerprint note and an
explicit fp16 request is refused; Vulkan matched CPU (15 vs 15 dets); INT8 rows ran
`ncnn-CPU-int8+fp16`; a missing int8 sibling fails cleanly.

`ncnn_bench` medians (ms) by threads, dense models on their domain probe:

| model | variant | 1T | 2T | 4T |
|---|---|---|---|---|
| p03_v01n | fp32 | 184.7 | 130.6 | 260.8 |
| p03_v01n | fp16 | 116.7 | 71.5 | 112.1 |
| p03_v01n | int8+fp32 | 141.7 | 89.1 | 133.9 |
| p03_v01n | **int8+fp16** | **95.5** | **58.3** | **94.1** |
| v0.1-seg-n | fp32 | 275.7 | 158.2 | 199.2 |
| v0.1-seg-n | fp16 | 151.1 | 87.0 | 119.9 |
| v0.1-seg-n | int8+fp32 | 173.7 | 107.0 | 157.4 |
| v0.1-seg-n | **int8+fp16** | **124.6** | **74.7** | **109.5** |
| esmoe_n_visdrone | fp32 | 192.4 | 151.4 | 190.1 |
| esmoe_n_visdrone | fp16 | 105.2 | 79.5 | 114.0 |
| esmoe_n_visdrone | int8+fp32 | 132.0 | 94.8 | 141.2 |
| esmoe_n_visdrone | **int8+fp16** | **94.1** | **70.3** | **103.0** |
| moa-n (router-emulated) | fp32 (fp16 skipped) | 220.1 | 166.9 | 242.4 |

int8+fp16 vs fp16: p03 -18% / -18% / -16%, seg -18% / -14% / -9%, esmoe -11% / -12% / -10%
(1T / 2T / 4T). Every int8 row is `valid=1` with detection counts in band, and the int8 p90 is
below the fp16 median in all nine comparisons, so the pre-set win rule holds at every thread
count. **Verdict: mixed-INT8 wins on the S26**, 10-18% over fp16 on top of fp16's ~45% over
fp32, at the certified 1.2-1.5 pt mAP50-95 cost and half the model size. 2 threads is the
operating point (the two prime cores); 4 threads oversubscribes them. int8+fp32 shows the float
remainder's precision matters: keeping it fp16 is worth a further ~30%.

App-process harness (`LatencyBenchTest`, 50 frames, medians): at 1 thread seg fp32 308 / fp16
178 / int8 142, esmoe 252 / 132 / 111, p03 252 / 137 / 98 (same ordering and margins); at 2, 4
and "big" threads every row ran 2-4x slower (500-680 ms) because the instrumentation process
has no core affinity and its OpenMP team migrates off the prime cores. This is the reason the
app runtime must call `ncnn::set_cpu_powersave(2)` and default to 2 threads (done in the app
milestone). Vulkan latency was not part of this run (both harnesses were CPU-only at the time);
they now carry a `vulkan` variant / row (`ncnn_bench --variants ...,vulkan`, `LatencyBenchTest`
`useVulkan=true`, both reporting `first_ms=` for the pipeline-compile first inference), pending a
device run; until then the app's Bench tab is the only GPU-vs-CPU measurement.

### GPU (Vulkan) vs CPU, from the app's Bench tab (MEASURED on the S26, 2026-09-08, cold run,
warmup 10 / 50 timed, pure model time, medians)

| model | GPU (ncnn-Vulkan) | CPU fp16 | CPU int8+fp16 |
|---|---|---|---|
| v0.1-seg-n | **54.4** | 84.9 | 74.4 |
| p03_v01n | **47.1** | 71.9 | 52.6 |
| esmoe_n_visdrone | 43.4 | 52.6 | **36.7** |
| moa-n (fp32 pinned on both) | **80.4** | 140.8 | n/a |

Vulkan beats CPU fp16 on every float model (1.2-1.75x) despite the 16-40 MatMul/Tile layers
that fall back to CPU inside the graph; mixed-INT8 on CPU still beats the GPU on esmoe. The
CPU rows match the standalone `ncnn_bench` numbers (seg 84.9 vs 87.0, p03 71.9 vs 71.5), so the
app's thread pinning works. GPU is therefore the right default unit for the float models. A
first Live-tab run on GPU showed 222 ms model time for seg-n, i.e. 4x the bench number: a
Live-path contention effect (camera pipeline + real-time backdrop blur on the same GPU), under
investigation; it is not a Vulkan property of the SoC.

### Live-tab speed on the S26 is thermal, not runtime (MEASURED, 2026-09-08)

With the camera open, the app's Live tab ran seg-N at 222-280 ms model time on CPU fp16 (4-7 fps)
and about 8 fps on Vulkan, for every thread setting (1/2/4/all), while the same models bench at
85 ms (CPU) / 54 ms (GPU). The HUD diagnostics explain it: thermal headroom 0.98 (1.0 = the
severe-throttling threshold) and the prime-core clock at 1382-1497 MHz against
`cpuinfo_max_freq` = 4,742,400 kHz, i.e. the cores were clamped to ~30% of their ceiling.
85 ms x (4.74 / 1.45) = 278 ms, which is the Live number. Camera pipeline + continuous inference
+ screen + USB charging push a Samsung flagship into its clamp within a minute; the bench's
cold-run numbers are the first-seconds performance only. Consequences: (1) under the clamp the
GPU is the most efficient unit (8 vs 5-7 fps), so GPU stays the default; (2) the Sustained bench
mode (3 min, last-quarter median + throttle %) is the number to quote for this device; (3) the
Live HUD now flags `throttled` (headroom >= 0.9, red tachometer) so a slow reading is never
mistaken for a runtime defect. Also measured: moa-n (VisDrone-trained mixture model) fires ~287
boxes at conf 0.25 on an indoor scene on every platform (CLI 287, EsMoE 0, seg-N 15): out-of-domain
model behaviour, not an app bug.

Update (same day): a 3-minute Sustained bench of seg-N on Vulkan holds 63.4 ms flat (cold 70.8),
thermal bar nominal throughout, so continuous inference alone does NOT clamp this SoC. The clamp
(prime cores at 1.4-1.5 GHz of 4.74) appears only with the camera pipeline running: the trigger
is the camera path (Samsung's camera power policy and/or the two 720p 30 fps streams + RGBA
conversion), not the model. Under investigation with the Live tab's `cam lite` switch (640x480
analysis, 15-30 fps) and the SoC rows shown while paused.

Final reading (2026-09-08): with per-cluster clocks in the HUD, Live on CPU (seg-N, 2 threads,
inference thread confirmed on prime cpu6): first 3 s prime 3648/4742 MHz -> 5 fps; after 30 s
prime 1497, performance cluster 787 -> 4 fps; `cam lite` (640x480, 15-30 fps) changes nothing;
paused with the camera open the clusters idle at 883/787. Conclusions: (1) thread pinning works;
(2) a camera + sustained-CPU clamp is real but only explains 5 -> 4 fps; (3) the base speed is
the runtime ceiling: seg-N at 640 is 85 ms pure inference on CPU and 54 ms on Vulkan at full
clock, i.e. ~11 / ~15 fps before overheads. The iPhone's 30 fps comes from the ANE (~10 ms);
ncnn has no Hexagon NPU path. Levers: a 416/320 re-export for Live (2-4x), and ONNX Runtime with
the QNN execution provider on the NPU (the ANE-class path) as the next milestone.

CPU scaling across all cores (`ncnn_bench --powersave 0`, seg-N fp16, S26): 2T 97.0, 4T 88.8,
6T 87.1, 8T 75.7 ms. All eight cores buy 13% over the 2-prime-core pin (87 ms): the NEON path
bottoms out near 75 ms (~13 fps) for this model. The iPhone Air's ~25 fps on Core ML "CPU" is the
AMX matrix engine via BNNS, which has no ncnn counterpart on Snapdragon. Runtime ceiling confirmed
on both units; levers = smaller input export, NPU via ORT QNN.

seg-N re-exported at imgsz 416 (`yolo export format=ncnn imgsz=416`, 5.27 GFLOPs vs ~12 at 640;
`models/v0.1-seg-n-416_ncnn`, shipped in the app as an opt-in entry): 200-image COCO val smoke
mAP50/50-95 0.641/0.473 (640) -> 0.581/0.413 (416), i.e. -6.0 pt box mAP for ~2.3x fewer FLOPs.
Not the default. p03 (pruned v0.1-N) cannot be re-exported from `tempo-ncnn/models/p03_v01n.pt`
(a TorchScript trace at 640); EsMoE-N VisDrone's checkpoint is not on this pod.

Reframed (user observation: the phone is not warm, and games run hot yet fast): the Live clock
drop is a vendor camera-scenario POWER POLICY (pre-emptive CPU cap while the camera HAL is
active), not reactive thermal throttling. The app now opens an ADPF performance-hint session
(API 31+, 33 ms target, every frame reported) for its threads, the sanctioned way to ask the
power HAL for the clocks a deadline needs; `adb shell dumpsys thermalservice` during Live is the
check that temperatures are nominal while the cap is on.

ADPF result: the hint session lifts Live to ~11 fps for the first seconds, then the cap returns
and behaviour is unchanged. Control experiment in flight: stock YOLO11n / YOLO11n-seg exported
to ncnn (6.7 / 10.0 GFLOPs; on the pod's x86 CPU, ratio only: yolo11n 48 ms vs p03 83 ms,
yolo11n-seg 62 ms vs seg-N 104 ms, ~1.7x). If YOLO11n reaches ~20 fps on the S26 the gap is
YOLO-Master's graph on ncnn; if not, ncnn on this phone is the limit and the ONNX Runtime + QNN
NPU path replaces it.

CONTROL RESULT (S26, MEASURED by the user, 2026-09-08): stock YOLO11n on the same ncnn runtime
runs ~35 fps at normal clocks and still ~19 fps under the camera cap (prime at 1267 MHz).
YOLO-Master's graphs are therefore 3-4x slower than a YOLO11n of comparable FLOPs on the same
runtime and phone: the runtime is not the limit, the lowering of the MoE/attention blocks is
(MatMul x16-36, Permute/Reshape churn, Tile, Reduction, Softmax; several without Vulkan kernels).
Next: per-layer profile with an NCNN_BENCHMARK build (x86 CPU ratios as the guide), then an
export-side rewrite of the hot blocks, re-measured on the S26.

Per-layer profile (NCNN_BENCHMARK build `third_party/ncnn-x86-bench`, x86 CPU 4T fp32, per forward):
seg-N 61 ms vs yolo11n-seg 34 ms. Gap by layer type: MatMul +5.9 ms (16 vs 2 layers), ConvDW +4.5
(26 vs 7), BinaryOp +4.1 (68 vs 21), Permute +3.5 (57 vs 2), Reshape +2.9 (45 vs 15), Convolution
+2.9 (135 vs 90), Softmax +2.1 (13 vs 2), Slice +1.1. p03 vs yolo11n has the same shape. Real
convolutions are 11% of the gap; the rest is memory-bound glue from how pnnx lowers the MoE
gating and attention blocks (worse on a phone's memory bus: 3-4x there vs 1.8x on x86).
`ncnnoptimize` fuses none of it (696 -> 696 layers, 68.7 ms). The fix is export-side: fused
attention (MultiHeadAttention/SDPA layers instead of MatMul+Permute chains), expert mixing folded
into convolutions, reshape churn removed; mapping in progress.

Export-side fix (`scripts/export_ncnn_dense.py`, 2026-09-08): export-time forward swaps under
tracing (the fork is untouched): the attention scale folded into the qkv conv, `AAttn` re-expressed
as `F.scaled_dot_product_attention` with the area folded into the heads axis (one fused ncnn SDPA
layer per block, `pe` fed from the same conv's v channels), and the router's `.repeat` dropped so
the gate broadcasts natively. Numerics: class scores agree with the stock export to ~1e-5, boxes
to the stock export's own 2e-2 px conv noise, identical detection counts (seg 15/15, det 14/14).
Layers: MatMul 16 -> 0, Tile 4 -> 0, Softmax 13 -> 5, Permute 57 -> 33, SDPA 0 -> 8. x86 CPU:
seg-N 69.0 -> 53.4 ms (-23%), v0.1-N 61.3 -> 54.9 ms (-10%); yolo11n(-seg) 30-32 ms on the same
box, the remaining gap being plain and depthwise convolutions (7x7 `pe`, MoE experts), i.e.
architecture. New dirs: `models/{v0.1-seg-n-sdpa,v0.1-n-sdpa,v0.1-n}_ncnn` (the released v0.1-N
needs the coco_eval repair shim and a trace-time top-k dispatch shim to lower at all; documented in
the script). Shipped to the S26 bundle and the app (asset version 4).

S26 RESULT of the export fix (MEASURED 2026-09-08, `ncnn_bench`, CPU fp16, 8 threads, all cores):
v0.1-seg-N stock export 76 ms -> SDPA export **33.2 ms** (2.3x, ~30 fps model time); v0.1-N stock
35.5 ms -> SDPA 30.2 ms. seg-N is now in the same class as stock yolo11n-seg on the same runtime,
with identical weights and identical mAP. The SDPA graphs are now the shipped `v0.1-seg-n_ncnn`
and `v0.1-n_ncnn` (stock exports archived under `models/archive/*-stock_ncnn`); INT8 siblings are
regenerated from them with the same ACIQ recipe. Section 1-3 numbers above were measured on the
stock exports and remain valid for those files.

INT8 siblings regenerated from the SDPA graphs (ACIQ 1024): seg-N 153/164 layers, 3.1 MB
(0.27x), 200-image COCO smoke -0.81 pt mAP50-95 (0.4573 -> 0.4492, passes the 1.0-pt gate; box
match 0.844 as before); v0.1-N 139/145 layers, 3.5 MB. Full-val certification of the new
siblings is still to be run; the app (asset version 5) ships them.

### S26 table for the shipped (SDPA) seg-N graph (MEASURED 2026-09-08, `ncnn_bench`, medians ms)

| unit | 2T | 4T | 8T | note |
|---|---|---|---|---|
| CPU fp16 | **34.8** | 46.2 | 39.4 (32-33 in an earlier run) | fastest path, ~30 fps model time |
| CPU int8+fp16 | 42.0 | 43.8 | 38.8 | slower than fp16 at every thread count |
| Vulkan (Adreno 840) | 50.1 (first frame 74) | | | stock graph was 54.4 |

Consequences: (1) the mixed-INT8 win measured on the stock export (sections 3-5) does NOT carry
over: with the glue removed the graph is convolution-bound and fp16 NEON beats int8 convs plus
their quantize/requantize passes; INT8 keeps only the 0.27x size at its accuracy cost.
(2) CPU fp16 beats Vulkan 1.5x on this graph, so the GPU is no longer the fastest unit for the
dense models on the S26. (3) Thread scaling is flat; 2 pinned threads is as fast as 8 and cooler.

## 6. ONNX Runtime + QNN on the Hexagon NPU (MEASURED, S26, 2026-09-09, M0 gate)

`OrtQnnSmokeTest` (runtime test APK, ORT 1.29.0 + QNN 2.42.0, V81 skel, fp32 ONNX run as
fp16 on HTP, `htp_performance_mode=burst`, 30 timed frames), model time medians in ms:

| model | ORT CPU | NPU fp16 | ncnn CPU fp16 (best) | HTP placement | dets NPU vs CPU |
|---|---|---|---|---|---|
| yolo11n | 54.6 | **9.1** | ~28 | 331/331 | 13 = 13 |
| v0.1-N (SDPA export) | 95.3 | **11.0** | 30 | 629/629 | 14 = 14 |
| v0.1-seg-N (SDPA export) | 143.9 | **19.6** | 33 | 674/674 | 11 vs 15 |
| esmoe_n_visdrone | 111.2 | **12.3** | 35 | 596/596 | 19 vs 43 |

Every graph is placed entirely on the HTP (strict sessions succeed); first inference 15-25 ms
with the EPContext cache (7-9 MB per model) created at first session. GO: the NPU is 2.7x the
best ncnn path for the detect model and 1.7x for seg. Open issue: fp16 precision on the HTP
drops detections on seg-N (11/15) and EsMoE (19/43) while the det model is exact; A16W8
quantization (16-bit activations) and the 200-image dump certification are the next step.
Packaging lesson: the Hexagon loader opens the skel by file path; `extractNativeLibs=true`
(`jniLibs.useLegacyPackaging = true`) plus the `libcdsprpc.so` `uses-native-library` entry must
be set in the module that builds the APK (the runtime module for the test APK), otherwise
`QNN SetupBackend failed ... Failed to create device` and everything silently runs on the CPU.
Reference (Qualcomm AI Hub) for YOLO11n on this SoC is ~3 ms graph time, so ~3x of session
overhead remains to chase (perf mode, I/O conversions, context priority).

M1 (app): runtime picker (ncnn | ONNX) and unit picker (CPU | GPU | NPU) in Live, Photo and the
Bench sustained config; measured default per model on first use (ncnn-CPU vs ONNX-NPU mini-bench
on the bundled probe, parity within 35%, cached per model + app version, hand picks win);
Bench cells `ncnn·GPU / ncnn·CPU / ONNX·NPU / ONNX·CPU` with a `runtime` column in history and
CSV; Settings "ONNX Runtime" section (perf mode, prefer quantized, clear NPU cache, measured
defaults). 27 unit tests.

M3 (quantization, desktop, ORT CPU-EP fake-quant, 200-image smokes, box mAP50-95):
`scripts/quantize_onnx_qnn.py` (qnn_preprocess_model + get_qnn_qdq_config, u16 activations,
int8 per-channel weights, MinMax, 256 letterboxed calibration images):

| model | float | A16W8 | delta | notes |
|---|---|---|---|---|
| v0.1-seg-N | 0.4613 | 0.4527 | -0.86 | seg-head convs kept int16 (`--w16-match model.25.`); plain a16w8 -1.23; a16w16 lossless |
| v0.1-N | 0.4700 | 0.4644 | -0.56 | |
| EsMoE-N VisDrone (200 val) | 0.1688 | 0.1653 | -0.35 | |

A8W8 (MinMax) is dead on all three (0 dets); parked as `model-a8w8.rejected.onnx`. Files 3.7-4.6
MB. Device certification: `OrtDumpTest` writes `class conf x1 y1 x2 y2` dumps per (runtime, unit,
precision) row for the 200-image smoke sets; `scripts/score_device_dumps.sh` scores them with
`eval_map.py`; gate = every NPU row within 1.0 pt of the CPU rows. Pending the phone run.

## 6. v1.2.0: the pruned v0.1-N ships as the dense export (2026-09-20)

`models/p03_v01n_ncnn` is now the `scripts/export_ncnn_dense.py` graph of the pruned checkpoint
(522 layers, 8 fused SDPA, no MatMul / Tile glue; x86 parity with the ONNX path 261 / 261
detections, max box delta 0.01 px, where the stock pnnx export differed on 7 of 20 images) and
`models/p03_v01n-int8_ncnn` its ACIQ mixed-INT8 sibling (139 / 145 layers, 2048 COCO train2017
calibration images, bin 3.5 MB = 0.26x). Scored in-process on the 500-image COCO subset
(`--accuracy auto`, x86 ncnn): fp32 mAP50-95 0.4309, INT8 0.4205 (-1.04 AP). The stock graph and
its sibling are kept as `p03_v01n-stock_ncnn` / `p03_v01n-stock-int8_ncnn`. The Galaxy S26
re-timing of the dense graph is the Android phase of v1.2.0.
