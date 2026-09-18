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
milestone). Vulkan latency was not measured by these harnesses (CPU-only by design); the app's
Bench tab measures GPU vs CPU.
