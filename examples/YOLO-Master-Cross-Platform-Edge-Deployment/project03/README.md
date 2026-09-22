# Project03: MoE pruning and edge deployment

Submission for 实战题目三 (模型剪枝与端侧部署实践), kept inside the cross-platform edge deployment
example because the runtime, the Jetson scripts, the Android app and the API server it builds on
are this example. Paths below are relative to the example root
(`examples/YOLO-Master-Cross-Platform-Edge-Deployment/`).

Ownership note. The MoE architectures, routers, `MoEPruner`, the exporter and the training
pipeline are upstream YOLO-Master and Ultralytics work. What this submission adds is the
measurement, the pruning surgery for the released checkpoints, the quantization recipes, the
device benchmarks, and the serving layer. Every number below is MEASURED by us unless marked
REFERENCE. The full sources (C++ runtime, Android and iOS apps, API server) are on
[skywalker-lt/yolo-master-edge](https://github.com/skywalker-lt/yolo-master-edge) `main`.

## Acceptance criteria and status

| criterion | bar | result | status |
|---|---|---|---|
| 基础功能 | MoEPruner pruning + TensorRT export; batch automation | `scripts/project03/diagnose_moe.py` and `prune_moe.py` sweep any set of checkpoints x datasets x thresholds in one run; `scripts/project03/quantize_trt.py` builds the FP32/FP16/INT8 ladders | 优秀 |
| 剪枝效果 | params -30%, mAP loss < 0.5% | v0.1-N -55.9% params, mAP50-95 -0.00001 (COCO); UoMoE-N -56.7%, +0.00002; worst case across 3 models x 3 datasets -0.0012 | 优秀 |
| 推理加速 | FP16 -25%; INT8 -40% with < 1% loss | FP16 -36.5% (A100), -47.3% (L4), -41.9% (Orin Nano); explicit Q/DQ INT8 -0.80 AP at -37.8% (L4); on Orin Nano INT8 beats FP16 at S/M/L (-5.4% to -7.2%) | FP16 通过, INT8 通过 (优秀 latency bar missed by 2.2 points on L4) |
| 端侧部署 | Android < 200 ms; < 100 ms + live video | Samsung S26: v0.1-N 30 ms per frame on the CPU (ncnn fp16), 11 to 20 ms on the Hexagon NPU (ONNX Runtime + QNN); Compose app with live camera, photo batch, bench and settings | 优秀 |
| 系统完整性 | docs + benchmarks; Docker + one-click | REST + WebSocket inference server on ONNX Runtime / TensorRT / ncnn / MNN with Prometheus metrics and latency percentiles; Docker image `skywalker0501/yolomaster-api:1.2.0`; one-click `deploy/run.sh` and compose file; full COCO val comparison of API vs bare runtime | 优秀 |

Not implemented by design: adaptive batching (the targets are batch-1 real-time paths; the
multi-file endpoint runs independent batch-1 jobs). Scene-dependent expert analysis exists
(`project03/results/pruning/*/scene_usage.json`) and shows no divergence on the released
checkpoints (see Diagnosis).

## Layout

```
scripts/project03/diagnose_moe.py      expert usage statistics, histograms, simple/medium/complex scene split, Gini
scripts/project03/prune_moe.py         progressive pruning sweep with the checkpoint repairs the released weights need
scripts/project03/quantize_trt.py      TensorRT FP32 / FP16 / implicit INT8 ladders, pin sets, bisect / ablate, engine audit
scripts/project03/qat_moe.py           explicit Q/DQ calibration (modelopt) and the QAT attempt
scripts/project03/measure_latency.py   model-only vs end-to-end latency, mAP re-validation
jetson/11_p03_trt_bench.sh             Orin Nano engine builds + benchmarks for pruned v0.1 N/S/M/L
tempo-ncnn/                            Orin-CPU ncnn fp16 vs INT8 bench (pruned v0.1-N)
coreml_export/export_coreml_p03.py     Core ML export of the pruned model with the dense k==E MoE rewrite (ANE-resident)
android/                               Android runtime module (ncnn + ONNX Runtime with the QNN NPU provider) and the Compose app; the C++ core it compiles against is vendored under android/runtime/src/main/cpp/core
scripts/export_ncnn_dense.py           dense ncnn export (fused SDPA attention, native gate broadcast)
scripts/export_onnx_dense.py           dense ONNX export for the ONNX Runtime + QNN path
scripts/quantize_onnx_qnn.py           A16W8 quantization of the dense ONNX for the Hexagon NPU
scripts/score_device_dumps.sh          scores the on-device output dumps against the reference
project03/results/pruning/             diagnosis reports, scene analysis, sweep CSVs, plans, plots
project03/results/quant/               INT8_RESULTS.md, ladders, sensitivity reports, pin sets, bisect/ablation, QAT results
project03/results/orin/                Jetson Orin Nano deployment log
project03/results/android/             ncnn INT8/FP16 policy and the Samsung S26 measurements
project03/results/api/                 API server vs bare runtime on COCO val2017 (L40S)
```

Prerequisites: the pinned YOLO-Master fork lineage (ultralytics 8.4.101, upstream commit
acce839c) on `PYTHONPATH`, the released checkpoints (`YOLO-Master-v0.1-N.pt`,
`YOLO-Master-EsMoE-N.pt`), COCO / VisDrone / AI-TOD-v2 in YOLO layout, and for the INT8 rungs
TensorRT 10.x with `nvidia-modelopt`.

## 1. Diagnosis

```
python scripts/project03/diagnose_moe.py --model YOLO-Master-v0.1-N.pt --data coco.yaml --scene-split --out out/diag/v01-n-coco
```

Collects per-layer expert hit shares and average routing weights with the upstream tracker,
writes the usage histogram per MoE layer, the utilization heatmap, and splits the validation
set into simple / medium / complex scenes by ground-truth instance count (thresholds 3 and 8)
to compare routing between scenes.

| model (COCO val) | max layer Gini | routing | scene divergence (L1, simple vs complex) |
|---|---|---|---|
| v0.1-N (`OptimizedMOEImproved`, E = 4/8/16, k = 2) | 0.875 | 2 of E experts carry 100% of the hits in every layer | 0.0005 at model.6, 0 at model.9 and model.12 |
| UoMoE-N (ours) | 0.823 | same concentration pattern | see `project03/results/pruning/uomoe-n-coco` |
| EsMoE-N (`ES_MOE`, E = 3, k = 3) | 0.0 | soft mixture, every expert active on every token (degenerate hit shares) | 0 everywhere |

The released checkpoints route the same experts on simple and complex scenes: the
concentration is a property of the trained router, not of the input, which is what makes the
pruning below free. Full reports: `project03/results/pruning/*/report.txt`, `usage.json`,
`scene_usage.json`, `expert_usage_heatmap.png`, `scene_model_*_routing.png`.

## 2. Pruning

```
python scripts/project03/prune_moe.py --model YOLO-Master-v0.1-N.pt --data coco.yaml \
    --stats out/diag/v01-n-coco/usage.json --thresholds 0.05,0.10,0.15,0.20,0.30 --final-eval --out out/prune/v01-n-coco
```

Progressive sweep (threshold 5% to 30%), one diagnosis feeding every threshold, full
validation set per rung, best plan selected under a 0.005 mAP50-95 budget. Three repairs the
released checkpoints need, applied at load and documented in the script: the `add_residual`
default for January-2026 COCO weights (without it the model scores 0.007), BatchNorm-terminated
routers that upstream's projection finder silently skips (rebuilt by a backwards scan), and
per-expert buffers that must shrink with the surgery.

| model / dataset | base params | base mAP50-95 | best cut | pruned params | delta mAP50-95 | kept experts |
|---|---|---|---|---|---|---|
| v0.1-N / COCO | 7.5546M | 0.42916 | 55.89% | 3.3321M | -0.00001 | 2 / 2 / 2 |
| v0.1-N / VisDrone | 7.5167M | 0.17071 | 56.18% | 3.2942M | +0.00004 | 2 / 2 / 2 |
| v0.1-N / AI-TOD-v2 | 7.5164M | 0.12024 | 56.18% | 3.2938M | -0.00118 | 2 / 2 / 2 |
| UoMoE-N / COCO | 7.4477M | 0.41379 | 56.69% | 3.2255M | +0.00002 | 2 / 2 / 2 |
| UoMoE-N / VisDrone | 7.4175M | 0.19939 | 56.92% | 3.1952M | -0.00023 | 2 / 2 / 2 |
| UoMoE-N / AI-TOD-v2 | 7.4171M | 0.19809 | 56.93% | 3.1948M | +0.00006 | 2 / 2 / 2 |
| EsMoE-N / COCO | 2.6944M | 0.4272 | 0% | unchanged | removing one expert costs -0.30, two -0.43 | 3 / 3 / 3 / 3 |

The plan is identical at every threshold from 5% to 30% (`project03/results/pruning/v01-n-coco/sweep.csv`):
the unused experts are exactly unused. EsMoE-N is unprunable by construction (k equals E, balanced
soft mixture); the sweep quantifies the cost instead of claiming a win. Pruned checkpoints load
with plain `YOLO()`. Scale invariance: pruned v0.1-S/M/L converge to the same 2/2/2 plan at
-57.7% / -44.3% / -39.6% params with no mAP change (`project03/results/quant/trt/scaling`).

## 3. TensorRT FP16 and INT8 (pruned checkpoints)

Full write-up with every table: `project03/results/quant/INT8_RESULTS.md`. Headline numbers (batch 1,
640, model-only median of 200 executions, full COCO val):

| device | fp32 (TF32 off) | fp16 | INT8 explicit Q/DQ | INT8 mAP50-95 (fp32 0.4286) |
|---|---|---|---|---|
| A100 | 3.331 ms | 2.115 ms (-36.5%) | 2.359 ms | 0.4206 (-0.80 AP) |
| L4 | 3.047 ms | 1.607 ms (-47.3%) | 1.896 ms (-37.8%) | 0.4210 |
| Orin Nano 10 W | 37.62 ms | 21.87 ms (-41.9%, 45.7 FPS) | 24.77 ms | same engine ONNX |

What the INT8 investigation established, in order: implicit TensorRT PTQ collapses on this
architecture (-12.9 AP at 256 val images; calibration coverage recovers to -2.3 AP at 2048 train
images and then floors); explicit per-channel Q/DQ with the sensitive set kept in fp16 (stem
pair, routers, DFL) reaches -0.80 AP; a 3-epoch QAT regressed to -3.2 AP (negative result,
recorded with the retry recipe); on Orin Nano explicit INT8 beats fp16 at S/M/L (-5.4% / -6.7% /
-7.2%, S passes the accuracy bar at -0.46 AP) while N is the one scale where fp16 stays faster.
Sensitive sets are per model (bisect / ablation CSVs in `project03/results/quant/trt/*`).

```
python scripts/project03/quantize_trt.py --model pruned.pt --data coco.yaml --calib-n 1024 --pins head --out out/trt/v01n   # add --bisect / --ablate for the sensitive set
python scripts/project03/qat_moe.py --model pruned.pt --data coco.yaml --calib-n 1024 --skip-train --out out/qat/v01n   # explicit Q/DQ, calibrate only
python scripts/project03/measure_latency.py --engine out/qat/v01n/model.engine --data coco.yaml --revalidate
bash jetson/11_p03_trt_bench.sh n s m l                                                                              # on the Orin
```

## 4. Android (Samsung Galaxy S26, Snapdragon 8 Elite Gen 5)

Two runtimes in one Kotlin/Compose app on the shared C++ core: ncnn (CPU fp16 / int8, Vulkan)
and ONNX Runtime with the Qualcomm QNN execution provider on the Hexagon NPU. Model time per
frame, medians, from the app's bench and the NPU gate (`project03/results/android/NCNN_INT8_RESULTS.md`):

| model | ncnn CPU fp16 | ncnn Vulkan | ONNX Runtime QNN (NPU) fp16 | HTP placement |
|---|---|---|---|---|
| v0.1-N | 30 ms | | 11.0 ms | 629 / 629 nodes |
| pruned v0.1-N (stock ncnn export) | 71.5 ms at 4 threads (int8+fp16 58.3 ms) | 47 ms | | |
| v0.1-seg-N | 33 to 35 ms | 50 ms | 19.6 ms | 674 / 674 |
| EsMoE-N | 35 ms | | 12.3 ms | 596 / 596 |

The ncnn graphs are exported with a dense rewrite (fused attention, native gate broadcast)
that removed the MatMul / Permute / Tile glue of the stock export; seg-N went from 76 ms to 33 ms
on the same phone. The pruned model was timed on the phone with the stock export only; its dense
re-export runs at the unpruned model's speed on x86 (`project03/results/api/API_SERVER_RESULTS.md`) and is
not yet re-timed on the S26. On this graph CPU fp16 beats Vulkan and beats mixed INT8, so INT8 is
kept for size only. A16W8 quantized ONNX for the NPU: v0.1-N -0.56 mAP, seg-N -0.86, EsMoE-N -0.35
(MEASURED on Linux; device certification of the A16W8 dumps is the open item). App and runtime sources: `android/` in this example (mirror of yolo-master-edge `android/` with the C++ core vendored under `android/runtime/src/main/cpp/core`); build instructions in `android/README.md`. A release build of the app (`yolomaster-edge-android-app-1.1.0.apk`) and the runtime module (`yolomaster-edge-android-runtime-1.1.0.aar`) are attached to the [yolo-master-edge v1.1.0 release](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.1.0).

## 5. Inference service and Docker

`yolomaster_server` (C++, uWebSockets) serves any number of models on ONNX Runtime CPU / CUDA,
TensorRT (engines built from ONNX and cached), ncnn and MNN, batch 1 per request, one backend
instance per worker thread, bounded queues, deadlines, graceful drain, `/metrics` (Prometheus)
and `/v1/stats` (rolling p50 / p95 / p99 per stage). Endpoints: `/v1/infer` (JSON, YOLO txt, COCO
JSON or annotated JPEG), `/v1/infer/batch`, `/v1/video` (NDJSON), `WS /v1/stream` (keep-latest
backpressure), model load / unload, `/healthz`, `/readyz`.

```
docker run --gpus all -p 8080:8080 -v $PWD/cache:/opt/yolomaster/cache/trt skywalker0501/yolomaster-api:1.2.0
curl -X POST 'localhost:8080/v1/infer?model=v01n-trt&conf=0.25' --data-binary @image.jpg
```

End-to-end comparison on an NVIDIA L40S over the full COCO val2017 (`project03/results/api/API_SERVER_RESULTS.md`,
23 cells): the API adds 0.8 to 1.5 ms per request over the bare CLI runtime, detections are
numerically identical on every backend, TensorRT fp16 serves v0.1-N at 7.7 ms per request end to
end (1.8 ms model time), the pruned model is 14% faster than the unpruned one on TensorRT at
identical mAP, and four GPU workers reach 345 to 480 images/s. The image is built with Bazel and
rules_oci (no Docker daemon needed on the build pod); `deploy/run.sh` and `docker-compose.yml`
in yolo-master-edge are the one-click entry points.

## 6. What is not here

Model weights, engines and calibration caches (rebuild from the released checkpoints with the
scripts above); the C++ server sources and the Android and iOS apps (yolo-master-edge `main`);
the unpruned N/S/M/L quantization study, which belongs to topic A3 and lives in `a3/` of this
example.
