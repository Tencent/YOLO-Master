# Jetson Orin Deployment Kit — YOLO-Master-EsMoE-N

aarch64 / JetPack deployment procedure for the edge runner and TensorRT. The
scripts must be run on a compatible Jetson device; this checkout does not
contain a device binary, engine, model, or benchmark evidence.

## Prerequisites

- A Jetson Orin device with a supported JetPack image. CUDA, TensorRT and
  cuDNN are supplied by JetPack; do **not** mix libraries from another release.
  The native TensorRT runner in `cpp/` requires the TensorRT 10 named-I/O API.
  On images that ship TensorRT 8, use the ORT + TensorRT-EP route instead.
- The VisDrone model files placed in `jetson/models/`:
  - `esmoe_n_visdrone_sim.onnx`  (for TensorRT + the ONNX backend)
  - `esmoe_n_visdrone_ncnn/`     (for the ncnn backend, optional)
  - Supply them from the experiment's model archive and record their SHA256.
- Internet (for `apt` build deps + fetching the ONNXRuntime aarch64 SDK).

## Quick start (in order)

```bash
cd examples/YOLO-Master-Cross-Platform-Edge-Deployment/jetson
# put the model in models/  (scp esmoe_n_visdrone_sim.onnx here)

bash 00_setup.sh          # verify JetPack, set MAX power, install build deps
bash 10_trt_bench.sh        # TensorRT FP16 + INT8 engines and timing output
bash 20_build_runner.sh   # build the C++ runner (aarch64) + run it
```

## What each step gives you

| Step | Output | Why |
|---|---|---|
| `00_setup.sh` | versions, MAXN power mode, `cmake`/OpenCV installed | reproducible perf (clocks locked) |
| `10_trt_bench.sh` | `*.engine` + timing output (FP16, INT8) | compare modes on the target and archive the raw log |
| `20_build_runner.sh` | `yolomaster_edge` (aarch64) + per-frame latency | the portable runner, same binary as Linux/Windows |

Before selecting the native route, record the installed TensorRT major version:

```bash
grep -E 'NV_TENSORRT_(MAJOR|MINOR|PATCH)' \
  /usr/include/aarch64-linux-gnu/NvInferVersion.h 2>/dev/null || \
grep -E 'NV_TENSORRT_(MAJOR|MINOR|PATCH)' /usr/include/NvInferVersion.h
```

The native build discovers the JetPack installation under `/usr` and
`/usr/local/cuda` by default.  When the headers and libraries are staged in a
non-standard SDK or sysroot, pass the roots explicitly; the same values are
written to `configure.log` for later audit:

```bash
TENSORRT_ROOT=/opt/tensorrt CUDA_ROOT=/opt/cuda bash 21_build_trt_runner.sh
```

## GPU inference — two routes

| Route | Script | Ships | Pros | Needs |
|---|---|---|---|---|
| **Native TRT 10** | `21_build_trt_runner.sh` | a device-local `.engine` | direct TensorRT execution | TensorRT 10.x and CUDA from the target JetPack; build the engine per device via `trtexec` |
| **ORT + TRT-EP** | `22_build_ort_trt.sh` | the `.onnx` | portable; auto engine build+cache; auto INT8(QDQ)/FP16/CUDA fallback | a Jetson ONNXRuntime **with the TensorRT EP**, matched to your CUDA/TRT |

> **Version boundary:** the native backend is intentionally limited to TensorRT
> 10.x because it uses the named-I/O (`enqueueV3`) API. TensorRT 8.x targets
> should use an ONNXRuntime build with the TensorRT EP, matched to the target
> CUDA/TensorRT versions, or build a separate legacy binding backend.

> **ORT provisioning caveat:** the TRT-EP path needs a Jetson ONNXRuntime built
> with the TensorRT EP. Use a version matched to the installed CUDA and
> TensorRT, or build ORT from source.

Both routes execute on the GPU. Native TRT requires a device-local engine;
ORT+TRT-EP accepts the ONNX model and can build/cache an engine on first run.
The selected route, CUDA version and TensorRT version must be recorded with the
benchmark log.

## Notes

- **Builder diagnostics:** TensorRT tactic selection and memory requirements are
  version- and device-dependent. If an engine build fails, record the exact
  TensorRT error, builder optimization level and workspace in
  [`DEPLOYMENT_LOG.md`](DEPLOYMENT_LOG.md); do not generalize one device's
  workaround to another version.

- **4 GB Orin Nano:** the TensorRT *builder* can be memory-hungry. Build **headless**
  (`sudo systemctl isolate multi-user.target`) and use a small workspace (`WORKSPACE=256 bash 10_trt_bench.sh`),
  if tactic profiling runs out of memory; record any swap and workspace changes.

- **INT8 accuracy:** `trtexec --int8` without a calibration/QDQ model is a speed
  diagnostic only. For an INT8 acceptance run, use a calibrated model and keep
  the detection head in FP16
  (`--precisionConstraints`/`--layerPrecisions`), mirroring the mixed-precision recipe from `TECHNICAL_REPORT.md §3`.
- **Power:** `00_setup.sh` sets `nvpmodel -m 0` (MAXN) + `jetson_clocks`. Re-run after every reboot for stable numbers.
- **GPU via the C++ runner:** `20_build_runner.sh` builds the CPU path (functional + portable). GPU acceleration
  through the runner is a follow-up (ncnn-Vulkan or the ONNXRuntime TensorRT EP); `trtexec` already gives the GPU ceiling.

## Evidence and reporting

This directory contains procedures only. Do not copy FPS or mAP values from an
external deployment into a result table. For a target-device run, retain the
exact model and engine hashes, JetPack/CUDA/TensorRT versions, power mode,
workspace and builder options, image-list hash, and raw `trtexec`/runner logs.
Report GPU compute latency and FPS separately from end-to-end latency, and
report mAP50 and mAP50-95 only with the fixed validation manifest and per-image
predictions. A result is publishable only after it is referenced by the shared
`evidence_manifest.py` output.
