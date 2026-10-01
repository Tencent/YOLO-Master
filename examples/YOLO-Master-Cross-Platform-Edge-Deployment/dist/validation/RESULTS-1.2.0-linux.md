# v1.2.0 Linux: mAP and speed, as measured on the two pods (2026-09-30 / 10-01)

All mAP figures are the in-process val protocol (conf 0.001, IoU 0.7, max_det 300, multi-label)
on coco500 (500 COCO val images, release eval set), boxes only. Model time = the CLI's `infer`
stage per frame; "bench median" = `--bench cold` (gray probe, model-only, floor-rank median).

## 1. Prebuilt bundles on a plain A100 80GB (driver 570.172, no cuDNN / TensorRT on the box)

Model: v0.1-seg-N (the bundled default), 640. Full log: `a100-bundle-acceptance-1.2.0.log`.

| bundle | path | bench median ms | model ms, 50-image run ¹ | mAP50-95 |
|---|---|---|---|---|
| ncnn | ncnn CPU fp32 | 149.1 | 149.3 | 0.4270 |
| mnn | MNN CPU | 125.3 | 99.2 | 0.4270 |
| onnx-cpu | ORT CPU fp32 | 138.1 | 194.2 | 0.4270 |
| onnx-cuda12 | ORT CUDA EP + CUDA preprocessing | 5.50 | 53.6 ¹ | 0.4271 |
| onnx-cuda12 | ORT CUDA EP, CPU preprocessing | 5.87 | 47.0 ¹ | 0.4269 |
| onnx-cuda12 | ORT CPU fp32 | 125.7 | 162.2 | 0.4270 |
| trt10-cuda12 | TensorRT fp16 + CUDA preprocessing (engine built on the A100) | 2.48 | 2.60 | 0.4268 |
| trt10-cuda12 | TensorRT fp32 + CUDA graph | 2.44 | 2.48 | 0.4271 |
| all-cuda12 | TensorRT fp16 | 2.53 | 2.63 | 0.4271 |
| all-cuda12 | ORT CUDA EP + CUDA preprocessing | 6.12 | 57.5 ¹ | 0.4271 |
| all-cuda12 | ncnn CPU | 144.2 | 185.1 | 0.4270 |
| all-cuda12 | MNN CPU | 130.9 | 127.5 | 0.4270 |

¹ The 50-image figure is an average that includes the first frame; on the ORT CUDA EP the first
frame carries the one-time cuDNN autotuning (about two seconds), which is why that column reads
50 ms there while the bench median is 5.5 ms. The bench median is the steady-state number.

Extras on the A100: TensorRT fp16 sustained 30 s throttle +0.12 %; CPU-vs-GPU preprocessing parity
0.4269 vs 0.4271; both TensorRT engines (fp16, fp32) compiled on the box from the sm80 builder
resource, cache reused on the second run.

## 2. Source build on the L40S (driver 580, TensorRT 10.16, CUDA 12.9), v0.1-N COCO, 640

Warm, 200 coco500 frames, per frame. Source: `l40s-v120-cuda-receipts.tar.gz` / `post_validate.log`.

| path | pre ms | model ms | post ms | total ms | model FPS | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|
| TensorRT fp16 + CUDA preprocessing | 0.09 | 1.45 | 0.68 | 2.22 | 451 | 0.5963 | 0.4314 |
| TensorRT fp16 + CUDA preprocessing + CUDA graph | 0.10 | 1.16 | 0.67 | 1.92 | 519 | | |
| TensorRT fp16, CPU preprocessing | 0.94 | 1.45 | 0.67 | 3.06 | 326 | | |
| ORT CUDA EP + CUDA preprocessing (IoBinding) | 0.11 | 3.45 | 0.68 | 4.23 | 236 | 0.5940 | 0.4308 |
| ORT CPU fp32 | 0.87 | 125.8 | 0.61 | 127.3 | 7.9 | 0.5940 | 0.4309 |
| MNN CPU fp32 | 0.86 | 203.2 | 0.90 | 204.9 | 4.9 | 0.5940 | 0.4309 |
| MNN CUDA fp32 ² | 0.73 | 2.11 | 10.29 | 13.13 | 76 | 0.5938 | 0.4309 |

bench cold (TensorRT fp16, gray probe, 100 iters): median 1.435 ms, p90 1.439, min 1.430.

² MNN CUDA fp16 is refused for this routed model (`fp16_safe: false` in the metadata, the
documented MNN 3.6.1 limitation) and the CLI runs fp32 with a note. The 10 ms post stage on the
MNN CUDA path is the output readback through MNN's host tensor copy, not the decoder; the CPU
path has no such copy.

Reference: the same checkpoint on the Mac (Core ML, M4 Max GPU) scores 0.4306, the Linux ORT-CPU
fp32 number in the README is 0.4309.

## 3. Bundle on the L40S with the system cuDNN hidden (the "driver only" proof), v0.1-seg-N

| path | model ms (visdrone50) | model ms (coco500, 500 frames) | mAP50 | mAP50-95 |
|---|---|---|---|---|
| onnx-cuda12 bundle, ORT CUDA EP + CUDA preprocessing | 3.60 | 3.70 | 0.5854 | 0.4271 |

## 4. validate_gpu.sh on the L40S (EsMoE-N COCO and v0.1-N, 50 visdrone images, conf 0.001)

| model | path | pre ms | model ms | post ms |
|---|---|---|---|---|
| EsMoE-N | TensorRT fp16, CPU preprocessing | 1.22 | 1.34 | 1.81 |
| EsMoE-N | TensorRT fp16, CUDA preprocessing | 0.39 | 1.33 | 1.70 |
| EsMoE-N | TensorRT fp16, CUDA preprocessing + graph | 0.37 | 1.07 | 1.69 |
| EsMoE-N | ORT CUDA EP, CPU preprocessing | 1.85 | 3.44 | 1.68 |
| EsMoE-N | ORT CUDA EP, CUDA preprocessing | 0.34 | 2.90 | 1.65 |
| v0.1-N | TensorRT fp16, CPU preprocessing | 1.37 | 1.44 | 1.92 |
| v0.1-N | TensorRT fp16, CUDA preprocessing | 0.38 | 1.44 | 1.93 |
| v0.1-N | TensorRT fp16, CUDA preprocessing + graph | 0.33 | 1.16 | 1.92 |

(The VisDrone mAP printed in that log, 0.033, is a COCO model scored against VisDrone labels and
is only used there to prove the three preprocessing paths agree. The MNN CUDA rows in that log
read 0.0000 because the .mnn conversion did not exist yet when the script ran; section 2 has the
MNN numbers measured afterwards.)

## Reading the numbers

- Every backend agrees on mAP to within 0.0003 on v0.1-seg-N (0.4268 to 0.4271) and within 0.0006
  on v0.1-N (0.4308 to 0.4314, TensorRT fp16 slightly above fp32 on this set).
- On the A100, TensorRT fp16 is 2.2x faster than the ORT CUDA EP (2.48 vs 5.50 ms); on the L40S the
  gap is 2.4x (1.45 vs 3.45 ms). CUDA preprocessing saves 0.8 to 1.5 ms per frame; the CUDA graph
  another 0.3 ms of model time.
- CPU numbers depend on the pod's CPU share (EPYC, 13 to 16 effective cores per container) and
  are not comparable across pods.

## 5. ncnn and MNN, both pods, measured 2026-10-01 (coco500, 640, val protocol)

Warm 200-frame timing (pre / model / post, ms per frame), `--bench cold` median, mAP. CPU rows use
the pod's CPU share (threads 4, the CLI default), which differs between the two pods.

### L40S, source build (`cpp/build_l40s`: ncnn CPU, MNN CPU and MNN CUDA)

| model | backend | pre | model | post | total | bench median | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| v0.1-seg-N | ncnn CPU fp32 | 1.11 | 113.1 | 1.03 | 115.2 | 109.0 | 0.5853 | 0.4270 |
| EsMoE-N | ncnn CPU fp32 | 1.29 | 93.6 | 0.68 | 95.6 | 93.6 | 0.6053 | 0.4353 |
| v0.1-N pruned | ncnn CPU fp32 | 1.08 | 97.7 | 0.67 | 99.4 | 131.6 | 0.5940 | 0.4309 |
| v0.1-seg-N | MNN CPU | 0.70 | 128.0 | 1.42 | 130.1 | 131.0 | 0.5853 | 0.4270 |
| v0.1-seg-N | MNN CUDA fp32 | 0.74 | 1.72 | 8.31 | 10.77 | 1.87 | 0.5853 | 0.4269 |
| EsMoE-N | MNN CPU | 0.71 | 162.6 | 0.89 | 164.2 | 164.1 | 0.6053 | 0.4353 |
| EsMoE-N | MNN CUDA fp32 | 0.69 | 1.65 | 7.17 | 9.51 | 1.62 | 0.6053 | 0.4351 |
| v0.1-N | MNN CPU | 0.72 | 192.2 | 0.87 | 193.8 | 191.6 | 0.5940 | 0.4309 |
| v0.1-N | MNN CUDA fp32 | 0.87 | 2.31 | 10.28 | 13.45 | 1.98 | 0.5938 | 0.4309 |

### A100, the all-cuda12 bundle (ncnn CPU, MNN CPU; the bundle's MNN is the CPU build)

| model | backend | pre | model | post | total | bench median | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| v0.1-seg-N | ncnn CPU fp32 | 1.25 | 150.5 | 1.62 | 153.3 | 116.5 | 0.5853 | 0.4270 |
| EsMoE-N | ncnn CPU fp32 | 1.69 | 158.7 | 1.21 | 161.6 | 167.4 | 0.6053 | 0.4353 |
| v0.1-N pruned | ncnn CPU fp32 | 1.59 | 130.6 | 1.00 | 133.2 | 178.2 | 0.5940 | 0.4309 |
| v0.1-seg-N | MNN CPU | 1.05 | 96.0 | 1.52 | 98.6 | 97.3 | 0.5853 | 0.4270 |
| EsMoE-N | MNN CPU | 0.97 | 86.3 | 1.11 | 88.4 | 127.5 | 0.6053 | 0.4353 |
| v0.1-N | MNN CPU | 1.00 | 166.0 | 0.93 | 167.9 | 201.9 | 0.5940 | 0.4309 |
| v0.1-N | MNN, `-d cuda` asked ³ | 0.90 | 212.7 | 1.02 | 214.6 | 207.3 | 0.5940 | 0.4309 |

³ The bundles ship MNN's CPU build, so `-b mnn -d cuda` silently runs on the CPU there (same mAP,
CPU speed). MNN on CUDA exists only in the source build (`deploy/pod/build_gpu.sh` builds the
second libMNN.so with `MNN_CUDA=ON`). A future bundle variant could carry it (a few MB plus
libcudart), but this release does not.

Reading section 5: every backend lands on the same mAP for a given model (seg-N 0.4270, EsMoE-N
0.4353, v0.1-N 0.4309, within 0.0002), and the pruned v0.1-N on ncnn scores exactly the unpruned
0.4309. MNN on CUDA runs the model in 1.6 to 2.3 ms but pays 7 to 10 ms of host readback per frame,
so it is a 10 to 13 ms path end to end, far behind TensorRT (2.2 ms) and ORT CUDA (4.2 ms). CPU
figures swing by up to 1.7x between the two pods for the same backend; they measure the pod's CPU
share, not the backends.
