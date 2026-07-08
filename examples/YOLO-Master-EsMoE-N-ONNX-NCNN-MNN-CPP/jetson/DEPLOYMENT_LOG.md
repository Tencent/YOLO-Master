# Jetson Orin Nano — Deployment Log

Deploying YOLO-Master-EsMoE-N to a Jetson Orin Nano (4 GB) via TensorRT: the deployment numbers,
on-device accuracy, and the reproducible build issues on Orin / TensorRT 10.16.2.

## Platform
| | |
|---|---|
| Device | Jetson Orin Nano 4 GB (`Orin`, sm87, 4 SMs @ 0.624 GHz, ~3.4 GB usable) |
| Software | JetPack 7 — Ubuntu 24.04, CUDA 13.2, TensorRT 10.16.2 |
| Model | `esmoe_n_visdrone_sim.onnx` (opset 12, `images[1,3,640,640] → output0[1,14,8400]`) |

## Result

Deployment engine: **clean FP16** (`trtexec --fp16 --builderOptimizationLevel=3`).

| Engine | GPU compute | FPS | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **FP16 (shipped)** | **27.8 ms** | **35.7** | **0.3488** | **0.2029** |
| QDQ INT8 (+FP32 fallback) | 45.4 ms | 21.7 | 0.3202 | 0.1834 |
| uncalibrated `--int8` | 31.4 ms | 31.6 | 0.128 | — |

mAP over the full 548 VisDrone val images (`eval_map_standalone.py`); FP16 is **−0.46% / −0.34%** vs the
FP32 baseline (0.3504 / 0.2036) — near-lossless. Context: x86 CPU (ORT) 25 FPS · Orin FP16 35.7 FPS ·
H200 CUDA 128 FPS.

**FP16, not INT8.** The mixed-precision recipe keeps the area-attention, head, and router out of INT8, so
INT8 leaves the compute-heavy attention on FP32/FP16 kernels; combined with TensorRT's lossier INT8, the
calibrated engine is both slower and less accurate than FP16. The `uncalibrated --int8` row is a trap:
`--int8` on a plain FP32 ONNX (no calibration/QDQ) silently builds a broken INT8 engine that benchmarks
fast but collapses to 0.128 mAP — a speed benchmark can't catch this, so validate mAP.

## Build issues (Orin / TensorRT 10.16.2)

**1. Pure-FP16 build fails at low opt levels.** At `--builderOptimizationLevel<=2` TensorRT's kernel timing
model references an `sm80` FP16 conv shader with no `sm87` base and asserts, producing an empty engine
(`KTM assertion failure: convolutionTimingModel.cpp:65 shader != nullptr` → `Created engine with size: 0 MiB`).
Fix: **`--builderOptimizationLevel=3`** (profiles tactics on-device instead of estimating) — needed for any
build that keeps FP16 layers (pure FP16 and mixed-precision QDQ INT8).

**2. ORT-quantized QDQ won't parse in TensorRT.** An `onnxruntime.quantization` QDQ model needs three things
the parser requires: (a) **symmetric** activations (ORT defaults to asymmetric → `Non-zero zero point is not
supported`); (b) **no int32-bias DQ** (ORT quantizes bias to int32, TensorRT handles bias internally →
`DequantizeLayer can only run in kINT8/…`); (c) **opset ≥ 13** for per-channel DQ. All handled by
`scripts/quantize_int8.py --symmetric` (`ActivationSymmetric` + `QuantizeBias=False` + opset upgrade).

**3. 4 GB build OOM.** The builder profiles tactics wanting 100s of MB each; on a 4 GB module it OOMs or
skips every fast tactic. Fix for the *build* (inference itself needs ~20 MB): go headless
(`sudo systemctl isolate multi-user.target`), add swap (`fallocate -l 8G /swapfile`), and cap
`--memPoolSize=workspace:256 --maxAuxStreams=0`.

## Reproduce
```bash
# on the Jetson (headless + 8 GB swap on the 4 GB Nano)
trtexec --onnx=esmoe_n_visdrone_sim.onnx --fp16 \
  --saveEngine=esmoe_n_fp16.engine \
  --memPoolSize=workspace:256 --builderOptimizationLevel=3 --maxAuxStreams=0
```
Scripts: `jetson/{00_setup,10_trt_bench,21_build_trt_runner,30_package}.sh`. Power: `nvpmodel -m 0 && jetson_clocks`.
