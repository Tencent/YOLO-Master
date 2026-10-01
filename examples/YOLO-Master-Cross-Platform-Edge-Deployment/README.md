# YOLO-Master Cross-Platform Edge Inference Runtime

<img alt="Docker" src="https://img.shields.io/badge/Docker-yolomaster--api-2496ED.svg?logo=docker&logoColor=white"> <img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white"> <img alt="NCNN" src="https://img.shields.io/badge/NCNN-Tencent-blue.svg"> <img alt="MNN" src="https://img.shields.io/badge/MNN-Alibaba-orange.svg"> <img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-NVIDIA-76B900.svg"> <img alt="Core ML" src="https://img.shields.io/badge/CoreML-Apple-black.svg">  <img alt="Linux" src="https://img.shields.io/badge/Linux-FCC624.svg?logo=linux&logoColor=black"> <img alt="Windows" src="https://img.shields.io/badge/Windows-0078D6.svg?logo=windows&logoColor=white"> <img alt="Jetson" src="https://img.shields.io/badge/Jetson%20Orin-76B900.svg?logo=nvidia&logoColor=white"> <img alt="macOS" src="https://img.shields.io/badge/macOS-000000.svg?logo=apple&logoColor=white"> <img alt="iOS" src="https://img.shields.io/badge/iOS-000000.svg?logo=apple&logoColor=white"> <img alt="Android" src="https://img.shields.io/badge/Android-3DDC84.svg?logo=android&logoColor=white"> 

This project provides a universal inference runtime for [YOLO-Master](https://github.com/Tencent/YOLO-Master) object-detection models, leveraging, [ONNX Runtime](https://onnxruntime.ai/), [NCNN](https://github.com/Tencent/ncnn), [MNN](https://github.com/alibaba/mnn), [TensorRT](https://github.com/nvidia/tensorrt), and [CoreML](https://github.com/apple/coremltools) backends. It runs on almost every platform: Linux, Windows (10/11), Jetson, MacOS, iOS, and **Android (NEW!)**; supports CPU, [CUDA](https://developer.nvidia.com/cuda-toolkit), [MPS](https://developer.apple.com/documentation/metalperformanceshaders), and NPU (on Android and Apple devices). It's capable of auto-detecting the model format, class names, and input size -- designed for real-time, end-to-end edge deployment in some of the most challenging tasks (VisDrone, SKU-110K, AI-TOD-v2, etc.).

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skywalker-lt/yolo-master-edge/main/assets/edge_deployment_architecture_dark.png">
    <img src="https://raw.githubusercontent.com/skywalker-lt/yolo-master-edge/main/assets/edge_deployment_architecture_light.png" width="900" alt="Edge Deployment Bundle, Architecture">
  </picture>
</p>

---

## 💡 Major Update (1-10-2026): YOLO-Master Edge v1.2.0 Linux, API Server macOS Preview

**We now have six Linux builds:** [ncnn](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-ncnn-1.2.0.tar.gz) (0.10 GB) / [MNN](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-mnn-1.2.0.tar.gz) (0.09 GB) / [ONNX CPU](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-onnx-cpu-1.2.0.tar.gz) (0.10 GB) / [ONNX CUDA 12](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-onnx-cuda12-1.2.0.tar.gz) (1.9 GB) / TensorRT 10, every GPU family: [part 0](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0.tar.gz.part-0) + [part 1](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-trt10-cuda12-1.2.0.tar.gz.part-1) (2.3 GB) / everything in one: [part 0](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-all-cuda12-1.2.0.tar.gz.part-0) + [part 1](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-all-cuda12-1.2.0.tar.gz.part-1) + [part 2](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/yolomaster-edge-linux-x64-all-cuda12-1.2.0.tar.gz.part-2) (4.1 GB). **And macOS:** [YOLO-Master 1.2.0](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/YOLO-Master-macOS-1.2.0.zip) (universal App, macOS 27 Golden Gate compatible, stapled and notarized). Checksums: [bundles](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/SHA256SUMS-linux-1.2.0.txt), [parts](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.2.0/SHA256SUMS-linux-1.2.0-parts.txt). The two large bundles are split for GitHub's 2 GiB asset limit; `cat <name>.tar.gz.part-* > <name>.tar.gz` concats them back together.

In v1.2.0, **we unified the measurements in every runtime**. The C++ core now has a benchmark mode, an in-process mAP scorer and multi-object tracking, the OpenCV-free part of that core now compiles into the Swift package too, and GPU preprocessing moved the letterbox off the CPU on both NVIDIA and Apple silicon.

### Features

- **📊 Bench mode:** `--bench cold` (10 warm-up and 100 timed model-only predictions on a gray probe, floor-rank percentiles) and `--bench sustained` (a timed loop, one median per second, the throttle figure). The Linux CLI, the API server and the macOS app all write the same `yolomaster-bench/v1` JSON。
- **🎯 On-device accuracy:** `--accuracy` scores a labelled folder with the val protocol (conf 0.001, IoU 0.7, max_det 300, multi-label). The scorer is the core's own C++ and equals `eval_map.py` to four decimals; the macOS app runs the very same code, which gives the **first Core ML mAP numbers** (v0.1-N 0.4306 on coco500 vs 0.4309 on ORT CPU). 
- **🎥 Tracking:** `--track botsort|bytetrack` on video (Kalman, two-stage IoU association, BoT-SORT camera-motion compensation); ids on the annotated video and as a 7th column of `--save-txt`.
- **⚡ GPU preprocessing:** one CUDA kernel builds the input tensor straight into TRT or an ORT IoBinding, plus `--cuda-graph` replay. L40S, v0.1-N at 640: TensorRT fp16 **1.45 ms** model time (1.16 ms with the graph), ORT CUDA 3.45 ms, mAP unchanged. On macOS a Metal kernel does the same job.

<img width="5733" height="1879" alt="v1 2 0-mac-screenshot-5" src="https://github.com/user-attachments/assets/8874d4a6-fb68-45f0-bb5c-c6a2e2d38c63" />

<br> 

- **🖥️ macOS app completely rebuilt in two modes:** **Inference** (everything from 1.1 plus tracking, the preprocessing device and a System card with the die temperature and battery power) and **Bench** (up to five models on Auto / GPU / CPU, cold / sustained / accuracy protocols, a live dashboard with the latency trace, per-run histogram, FPS over time coloured by thermal state, per-class AP chart, side-by-side comparison and results table, a thermometer reading the real die temperature from the SMC, a power meter from IOKit, run history with CSV and JSON export). The icon 
- **🧠 Neural Engine export:** the Core ML exporter writes the MoE top-k mixture with arithmetic only, so v0.1-N runs entirely on the Neural Engine instead of bouncing its routers to the CPU; `compute_plan.py` shows where Core ML places every op and `quantize_w8a16.py` makes int8-weight packages.
- **🌐 API server 1.2.1:** `track=` on `/v1/video` and `/v1/stream`, `POST /v1/bench`, API keys, rate limiting, request-id and `traceparent` propagation, JSON access log, `GET /openapi.json`, GPU preprocessing on the TensorRT and ORT-CUDA workers; image `skywalker0501/yolomaster-api:1.2.1`.

**Measured numbers:**

| v0.1-seg-N, 640, coco500 | 1x A100 bundle, end-to-end median | mAP50-95 |
|---|---|---|
| TensorRT fp16 (engine built on the box) | 2.48 ms | 0.4268 |
| TensorRT fp32 + CUDA graph | 2.44 ms | 0.4271 |
| ONNX Runtime CUDA EP | 5.50 ms | 0.4271 |
| ONNX Runtime CPU | 138 ms | 0.4270 |
| ncnn CPU | 149 ms | 0.4270 |
| MNN CPU | 125 ms | 0.4270 |

| v0.1-N, 640, coco500 | 1x L40S source build, model ms | mAP50-95 |
|---|---|---|
| TensorRT fp16, CUDA preprocessing | 1.45 (1.16 with graph) | 0.4314 |
| ONNX Runtime CUDA EP, CUDA preprocessing | 3.45 | 0.4308 |
| MNN CUDA fp32 | 2.31 (+10 ms host readback) | 0.4309 |
| ONNX Runtime CPU | 125.8 | 0.4309 |
| macOS, Core ML GPU, Metal preprocessing (M4 Max) | 4.3 | 0.4306 |

```bash
# any Linux bundle
tar xzf yolomaster-edge-linux-x64-trt10-cuda12-1.2.0.tar.gz && cd yolomaster-edge-linux-x64-trt10-cuda12-1.2.0
./yolomaster_edge -m models/v0.1-seg-n.onnx -s images/ -b trt --precision fp16 --out out          # engine built on first run, cached
./yolomaster_edge -m models/v0.1-seg-n.onnx -s images/ -b trt --precision fp16 --bench cold --bench-json bench.json --no-save
./yolomaster_edge -m models/v0.1-seg-n.onnx -s coco500/images -b trt --precision fp16 --accuracy auto --no-save
./yolomaster_edge -m models/v0.1-seg-n.onnx -s clip.mp4 -b trt --precision fp16 --track botsort --out out
```

Two findings worth noting here. 1. On Apple silicon with a strong GPU, the NPU (ANE) does not outperform the GPU on these models: a static Core ML graph computes every expert of v0.1-N, and yolov12x's area attention is refused by the ANE compiler. The app's forced-ANE unit was therefore replaced by **Auto**, Core ML's own split, which is also what the iPhone app's "ANE" numbers measure, which will be corrected later as a naming issue. 

Windows, Jetson, iOS and Android stay on v1.1.x in this preview; the same features land there in their own phases. Full notes: [Release Page](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.2.0); macOS details in [`mac/RELEASE_NOTES-1.2.0.md`](mac/RELEASE_NOTES-1.2.0.md); all Linux numbers and logs in [`dist/validation/RESULTS-1.2.0-linux.md`](dist/validation/RESULTS-1.2.0-linux.md).

> Moreover, I'll be using Claude Code and Codex to measure all the CLI and API Server numbers from now on. Hence an `AGENTS.md` and a `CLAUDE.md` will be added later to help contributors better understand the structure and rules of this repo and automate the development.

---

## 🌐 Update (11-09-2026): YOLO-Master Edge API Server v1.2.0 (REST + WebSocket, four backends)

[Click here](https://hub.docker.com/repository/docker/skywalker0501/yolomaster-api/general) to check the image on DockerHub!

`yolomaster_server` turns the v1.1.0 runtime into a production inference service: one C++ process, any number of models, each on **ONNX Runtime (CPU/CUDA), TensorRT, ncnn or MNN**, in fp32, fp16 or int8, batch 1 per request by design. It ships with a Docker image built with Bazel + rules_oci (no Docker daemon needed on the build pod), a one-click `deploy/run.sh`, Prometheus metrics and a
Python client.

- `POST /v1/infer` (JSON, YOLO txt, COCO JSON or annotated JPEG), `POST /v1/infer/batch`, `POST /v1/video` (NDJSON stream), `WS /v1/stream` (keep-latest frame backpressure), `GET /metrics`, `GET /v1/stats`, `/healthz`, `/readyz`, model load/unload at runtime
- One `Backend` per worker thread, bounded queues (503 + Retry-After), request deadlines (504), graceful drain on SIGTERM, TensorRT engines built from `.onnx` and cached per GPU
- Measured on an L40S over the full COCO val2017 (5000 images), API vs bare CLI on every backend: see [`API_SERVER_RESULTS.md`](API_SERVER_RESULTS.md); the v1.2.0 / image 1.2.1 rerun on an L4 with GPU preprocessing is Tables 4 to 6 of the same document
- Docs: [`docs/API.md`](docs/API.md), [`deploy/README.md`](deploy/README.md); build: `cmake -DBUILD_SERVER=ON` (run `scripts/server/fetch_uws.sh` once for uWebSockets)

```bash
yolomaster_server -p 8080 -m v01n=/models/v01n/model.onnx,backend=trt,device=cuda,precision=fp16
curl -X POST 'localhost:8080/v1/infer?model=v01n&conf=0.3' --data-binary @image.jpg
```

---

## 🤳 Update (11-09-2026): YOLO-Master Edge v1.1.0 Android Alpha build

**The 6th platform of YOLO-Master Edge, and the first with two runtimes.**

Native Android app (Kotlin, Jetpack Compose) for on-device YOLO-Master detection and segmentation. It is a function-for-function port of the iOS app on top of the shared C++ core (the same letterbox, decode and NMS code as the Linux, Windows,Jetson and macOS runners) behind a JNI bridge, with the runtime and the compute unit selectable per model:

- **ncnn** on the CPU (per-model fp16 policy, mixed-INT8 siblings) or the GPU through Vulkan
- **ONNX Runtime with the Qualcomm QNN execution provider** on the Hexagon NPU (Snapdragon 8 Elite Gen 5 and other HTP-capable SoCs), with A16W8 quantized models and a measured per-model default that picks the faster runtime on first launch

Everything runs on the phone; nothing you capture leaves it. Requires Android 7.0 (API 24) or later, arm64; the NPU path needs a Snapdragon SoC with a Hexagon HTP. Preliminary build: sideload only, no store listing yet.

### Features

- **📹 Live** - Real-time CameraX detection with an async overlay, FPS tachometer, per-stage latency and thermal state, lens switching, tap-to-focus, torch, live conf/IoU tuning without pausing inference, pause/resume, and a full-resolution
shutter that bakes the boxes and masks into the saved photo (`Pictures/YOLO-Master`). The HUD shows the resolved backend (`ncnn-CPU-fp16`, `ncnn-Vulkan`, `ort-QNN-htp-fp16`, ...) and, on the NPU, how many graph nodes run on the HTP.
- **📸 Photo** - Batch detection over up to 100 library images, 3-up gallery and zoomable pager, per-image and batch stats, segmentation masks, conf/IoU retune from cached raw outputs (one forward, many decodes), export of annotated images.
- **🎛️  Bench** - Cold run of every bundled model across runtime and unit (ncnn CPU, ncnn GPU, ONNX CPU, ONNX NPU) with pre/inference/decode breakdowns, sustained thermal runs with a latency sparkline and thermal timeline, run history with
per-run graphs, CSV share.
- **⚙️  Settings** - About, licenses, privacy, CPU-inference toggle, ONNX Runtime section (NPU performance mode, prefer-quantized switch, NPU cache reset, the measured-default table), custom model import (ncnn dirs or ONNX), erase history.

### Models

Seven bundled models: YOLO-Master v0.1-N and v0.1-seg-N (COCO), YOLO-Master-EsMoE-N (VisDrone), the MoEPruner-pruned v0.1-N, their mixed-INT8 ncnn siblings, and YOLO11n as a control. The ncnn graphs are exported with a dense rewrite (fused
SDPA attention, native gate broadcast) that removes the MatMul/Permute/Tile glue the stock export produced; on the S26 that took seg-N from 76 ms to 33 ms on the CPU.

### Performance

MEASURED on a Samsung Galaxy S26 (Snapdragon 8 Elite Gen 5, Hexagon V81, Adreno 840), model time per frame, medians, 2026-09-08/09.

| model | ncnn CPU fp16 | ncnn GPU (Vulkan) | ONNX NPU fp16 | HTP placement |
|---|---|---|---|---|
| YOLO11n (control) | 28 ms | | 9.1 ms | 331/331 nodes |
| YOLO-Master v0.1-N | 30 ms | | 11.0 ms | 629/629 |
| YOLO-Master v0.1-seg-N | 33 to 35 ms | 50 ms | 19.6 ms | 674/674 |
| YOLO-Master-EsMoE-N (VisDrone) | 35 ms | | 12.3 ms | 596/596 |

Two things to know before reading the table. On this graph the CPU fp16 path beats Vulkan and beats the mixed-INT8 siblings, so INT8 is kept for its size only. And while the camera is open the SoC's power policy clamps the prime cores to
about 1.4 GHz even when the phone is cool, which is why the NPU path matters: it is the unit that holds its speed in Live mode. fp16 on the HTP is exact for v0.1-N and YOLO11n; seg-N and EsMoE-N lose detections in fp16 on the HTP, which is
what the A16W8 models are for (v0.1-N -0.56 mAP, seg-N -0.86 with the seg head kept int16, EsMoE-N -0.35, MEASURED on Linux; device certification of the A16W8 dumps is the open item).

### Build from source
 
```bash
cd android
scripts/stage_models.sh --module app         # copies the bundled models into app assets
gradle :app:assembleRelease                  # -> app/build/outputs/apk/release/app-release.apk
adb install -r -g app/build/outputs/apk/release/app-release.apk
```

Needs the Android SDK (API 34), NDK r29 and a prebuilt ncnn for Android under android/sdk-paths.properties; the ONNX Runtime QNN package and the Qualcomm QNN runtime are pulled from Maven by the runtime module's extractOrt task. See
android/README.md for the runtime API, the precision policy and the on-device test harness.
 
Privacy & License (Android App Update)
 
The app has no internet permission. No data leaves the device, and we don't collect anything. See PRIVACY.md.
 
The app is licensed under AGPL-3.0, consistent with YOLO-Master and Ultralytics; ncnn is BSD-3-Clause, ONNX Runtime is MIT. The Qualcomm QNN runtime libraries bundled in test builds are proprietary (Qualcomm Technologies license, "not a
contribution"); this preliminary build is for research and personal experience only, and public distribution of an NPU-enabled build is pending a license review. Any direct commercial use of this app is prohibited.

---

## 📱 Major Update (27-08-2026): YOLO-Master Edge for iOS v1.1.0 Beta Build 1

<img width="4812" height="2291" alt="screnshots-framed" src="https://github.com/user-attachments/assets/e858e07b-eeff-40c2-b11a-dcb4dba44577" />

<br>
<br>

**🍾 Welcome to the 5th platform of YOLO-Master Edge. [Try it now.](https://testflight.apple.com/join/EVExpVHD)**

Native iOS SwiftUI app for on-device YOLO-Master detection and segmentation, powered by Apple Core ML and the same YOLOMasterKit inference path as the macOS runtime. Everything runs locally on the Neural Engine, GPU, or CPU. Nothing you capture leaves the iPhone. Requires iPhone on iOS 17 or later. iPhone 13 and later models are recommended. 

### Features

- **📹 Live** - Real-time camera detection with a dynamic performance visuliaztion (FPS tachometer, per-stage latency measures, and phone's thermal state), multi-cam with automatic lens switching and pinch zoom, tap-to-focus, torch, and a full-resolution shutter that renders the live overlay (boxes and segmentation masks) into the saved photo. Also supports live IoU/conf tuning without pausing the inference. 

- **📸 Photo** - Batch detection over images you pick from your library, up to 100 images at a time. Per-image and batch stats, segmentation masks, live conf/IoU tuning, and export of annotated images back to Photos.

- **🎛️ Bench** - On-device benchmarking on all devices. A Cold run measures every bundled model across compute units (Neural Engine (ANE), GPU, CPU) with expandable pre/inference/decode stage breakdowns, and a Sustained mode runs a thermal-throttle test with a live latency heatbeat sparkline and a colored state timeline. Pause/resume, a persistent run History vault (with per-run graphs) and CSV export.

- **⚙️ Settings** - App info and an expandable About card (wihich explains the MoE architecture summary with Paper, Model / App Repo links), Licenses and Acknowledgements (from macOS build), a Privacy and Security summary, a CPU-inference opt-in for Live and Photo modes, erase-all benchmark history, and a Beta importer for your own trained Core ML models (.mlpackage / .mlmodelc / .mlmodel).

### Performance

All data are measured under Live mode on ANE with `YOLO-Master-v0.1-N` (COCO) for 3 min.

| iPhone Model | SoC | End-to-end FPS | Thermal State after 1 min |
|---|---|---|---|
| iPhone 17 Pro Max | A19 Pro | 51 | 🟢 |
| iPhone Air | A19 Pro | 36 | 🟠 |
| iPhone 15 Pro Max | A17 Pro | 42 | 🟠 |

### Install (Public Beta)

The app is distributed through TestFlight. Install the TestFlight app on App Store, then open the invite link below to start testing it: 

https://testflight.apple.com/join/EVExpVHD

### Build from source

```zsh
brew install xcodegen
cd ios && xcodegen # this generates YOLOMasterIOS.xcodeproj
```

Drop your Core ML models into directory `ios/Models/`, set your signing team ID in `ios/project.yml`, open the project, and run. See `ios/README.md` for details.

### Privacy & License (iOS App Update)

The app has no internet connection. No data leaves the device, and we don't collect anything. See PRIVACY.md for detailed terms. 

The app is licensed under AGPL-3.0, consistent with YOLO-Master and Ultralytics; coremltools is BSD-3-Clause. It is released for research and personal experience only, and any direct commercial use of this app is prohibited. 

---

## 🚀 Major Update (12-08-2026): YOLO-Master Edge v1.1.0

**One release, every platform: [macOS](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.1.0/yolomaster-edge-mac-1.1.0.zip) / Windows [CPU](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.1.0/yolomaster-edge-win-x64-1.1.0.zip) + [CUDA](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.1.0/yolomaster-edge-win-x64-gpu_cuda12-1.1.0.zip) / Linux [CPU](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.1.0/yolomaster-edge-linux-x64-1.1.0.tar.gz) + [CUDA](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.1.0/yolomaster-edge-linux-x64-gpu_cuda12-1.1.0.tar.gz) / [Jetson Orin](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.1.0/yolomaster-edge-jetson-orin-1.1.0.tar.gz).**

v1.1.0 brings the same feature set to all runners (macOS / Windows GUIs, Linux and Jetson TensorRT CLIs):

- **🔪 Slicing (Sparse SAHI):** sliced inference for small objects on large images, a faithful port of upstream [YOLO-Master](https://github.com/Tencent/YOLO-Master)'s Sparse SAHI Mode plus a traditional dense-tiling variant, with adjustable tile size and per-run statistics.
- **🍇 Cluster-Weighted NMS:** a new NMS mode that refines each box as the weighted average of its detection cluster; tunable sigma, live everywhere including the webcam.
- **📤 Annotation export:** turn detections into training data as **YOLO TXT / COCO JSON / Pascal VOC XML** from images, folders, or videos (with frame sampling); segmentation models export real mask polygons. Rendered images and annotated videos export too.
- **🔎 Zoom & pan on GUI:** cursor-anchored zoom up to 8x on images and paused video in both GUIs.
- **📦 New: prebuilt Linux x86_64 bundles** (self-contained, glibc 2.35+, all three backends + ffmpeg video) and a **Jetson Orin bundle** with the TensorRT backend now supporting the full feature set.

<br>

<img height="380" alt="Screenshot 2026-08-12 at 5 35 43 PM" src="https://github.com/user-attachments/assets/6cb59d23-76e1-43b4-93f6-4302290a0e8a" /> <img height="380" alt="Screenshot 2026-08-12 at 5 36 12 PM" src="https://github.com/user-attachments/assets/27a60aee-a587-47e1-be93-e404e70b932d" />

<br>

**Full notes: [Release Page](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.1.0).**

---

## ✨ Update (27-07-2026): YOLO-Master Windows 10/11 Runner (**GUI**) on ONNX/ncnn/MNN backends with **GPU Acceleration**
**Download the [CPU runner](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.0.0-windows/YOLO-Master-Windows-1.0.0.zip) / [CUDA runner](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.0.0-windows/YOLO-Master-Windows-CUDA-1.0.0.zip).**

Now the Windows C++ edge runner has an improved backend and dedicated GUI! **YOLO-Master Windows Runner GUI** provides a refined C++ edge inference backend that bundles [ONNX](https://onnxruntime.ai/), [ncnn](https://github.com/Tencent/ncnn) and [MNN](https://github.com/alibaba/MNN) with **GPU acceleration**, + a polished frontend built with [Dear ImGui](https://github.com/ocornut/imgui) with all functionalities from the MacOS Core ML Runner below. It also bundles a default `YOLO-Master-v0.1-seg-N` segmentation model as the Mac runner. 

<br>

<img width="400" alt="48 2" src="https://github.com/user-attachments/assets/fa96097b-1014-46c0-8692-2c3656f4f763" />  <img width="400" alt="50 2" src="https://github.com/user-attachments/assets/db1e2194-584e-4898-b2b3-e7370fa325c5" />
<img width="800" alt="49 2" src="https://github.com/user-attachments/assets/6a438094-b352-4f59-bb55-9e3c600837ad" />

<br>

- **Three Backends in One App** ONNX, ncnn, and MNN all ship in single executable. Inference backends can be switched with a single click.
- **GPU Acceleration for All Backends** up to **4x speedup** with CUDA-accelerated inference on consumer devices. (please refer to the inference speed comparison table in [Relases](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.0.0-windows)) 

Please check our [Release Page](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.0.0-windows) for more details.

---

## 🍎 Update (17-07-2026): YOLO-Master CoreML Runner for MacOS (GUI)

**[Download](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.0.0-macos/YOLO-Master-CoreML-Runner-1.0.0.zip) and try it now!**

Alongside the Linxu and Windows C++ runtime, we now provide a native, user-friendly macOS runner, **YOLO-Master CoreML Runner**, a [SwiftUI](https://developer.apple.com/xcode/swiftui/) frontend over an Apple [Core ML](https://developer.apple.com/documentation/coreml) backend for on-device [YOLO-Master](https://github.com/Tencent/YOLO-Master) inference, no command line required. It ships with a default `YOLO-Master-v0.1-seg-N` segmentation model, so it runs out of the box. 

<img width="400" alt="Screenshot1" src="https://github.com/user-attachments/assets/d1747b4d-0961-458e-99c5-2a9870b8df96" /> <img width="400" alt="Screenshot2" src="https://github.com/user-attachments/assets/5f71d80a-6238-49bd-a230-95ccd4020d29" /> 
<img width="400" alt="Screenshot3" src="https://github.com/user-attachments/assets/9cc60636-b795-4326-992c-06239a77db55" /> <img width="400" alt="Screenshot4" src="https://github.com/user-attachments/assets/b5ee48bb-52dc-4ff7-b0bd-f2461b34ad7c" />

- **Detection & Segmentation:** Runs both bounding-box detectors and instance-segmentation models, with anti-aliased mask overlays and a Masks / Boxes / Both toggle.
- **Images, Video & Live Camera:** Infers single images, whole folders (batch), and MP4 video, plus a low-latency **live webcam** mode with a real-time FPS / ms-per-frame readout.
- **⭐️ Real-Time Tuning:** Confidence, IoU, box style, labels, and letterbox/stretch preprocessing are all adjustable live:  the forward pass is cached, so tuning re-draws without re-inferring.
- **Signed & Notarized:** A **universal** (Apple Silicon + Intel) bundle, **Developer-ID signed and notarized by Apple**: it installs by a simple double-click on any Mac with MacOS 14+.

For more details, please check the [Release](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.0.0-macos) page.

---

## ✨ Benefits

- **Universal CLI Binary for Linux and Windows:** A single executable integrates **ONNX Runtime**, **NCNN** and **MNN** backends; the backend, class names, and input size are auto-detected from the model, no recompilation or any dataset YAML needed at runtime.
- **Verified Accuracy:** Reproduces the PyTorch original to **< 0.5%** mAP50-95 across ONNX / NCNN / MNN, and **< 1.0%** under INT8 quantization, on 548 VisDrone validation images.
- **Deployment-Friendly:** Cross-platform [CMake](https://cmake.org/) build producing **self-contained and relocatable bundles** for Linux x86_64 and Windows 10/11, installable by unzip, no dependencies on the target.
- **GUI:** On Windows 10/11 and MacOS, there are user-friendly GUI runners which integarates all functions of the CLI bundles and supports GPU acceleration.
- **GPU Acceleration:** Supports FP32 CPU inference & [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) GPU acceleration through the ONNX Runtime CUDA Execution Provider on both Linux and Windows, on Windows via NCNN's [Vulkan](https://vulkan.org) & MNN's [OpenCL](https://opencl.org), on [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) Orin via a native TensorRT backend (JetPack 7), and accelerated on MacOS via [MPS](https://developer.apple.com/documentation/metalperformanceshaders) beind Core ML.

## ☕ Note

The exported models embed their class names, input size, and stride as ONNX/NCNN/MNN metadata, so the runtime configures itself from the model file. Post-processing is tuned for the vertical domain, aspect-ratio-preserving letterbox, per-class **multi-label** NMS, and a low default confidence threshold appropriate for VisDrone's small, dense objects.

## 📦 Exporting Models

Pre-built models (trained on VisDrone) are attached to the [Releases](https://github.com/skywalker-lt/yolo-master-edge/releases) page. To export your own trained [YOLO-Master](https://github.com/Tencent/YOLO-Master) checkpoint, use the Ultralytics `export` mode.

### ONNX

```python
from ultralytics import YOLO

# Load a trained YOLO-Master-EsMoE-N checkpoint
model = YOLO("EsMoE-N_VisDrone.pt")

# opset=12 for broad compatibility (ORT + NCNN + MNN)
# simplify=True runs onnxsim; dynamic=False fixes the input shape for C++ deployment
model.export(format="onnx", opset=12, simplify=True, dynamic=False, imgsz=640)
```

### NCNN (via pnnx) and MNN

```bash
# NCNN: Ultralytics uses pnnx under the hood
yolo export model=EsMoE-N_VisDrone.pt format=ncnn imgsz=640

# MNN: convert the exported ONNX with MNN's converter
mnnconvert -f ONNX --modelFile esmoe_n_visdrone_sim.onnx --MNNModel esmoe_n_visdrone.mnn --bizCode edge
```

For more details on exporting, refer to the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export/).

### Core ML

```zsh
# detector or segmenter (task auto-detected)
python coreml_export/export_coreml.py --weights model.pt --imgsz 640 --out model.mlpackage

# YOLO-Master default imgsz is 800 for AI-TOD models: pass --imgsz accordingly
python coreml_export/export_coreml.py --weights yolo-master-v0.1-N_aitodv2.pt --imgsz 800 --out v0.1-N.mlpackage

# sunsmarterjie/yolov12 checkpoints (split qk+v area-attention): stock ultralytics + the flag
python coreml_export/export_coreml.py --weights yolov12x.pt --imgsz 640 --out yolov12x.mlpackage --yolov12-aattn

# a LoRA fine-tune: merge the trained adapters first
python coreml_export/export_coreml.py --weights base.pt --merge-lora-dir lora_adapter/ --imgsz 640 --out ft.mlpackage
```


## ⚙️ Dependencies

Ensure you have the following dependencies installed （not required if you only want to smoke-test the pre-built bundles):

### Linux & Windows

| Dependency                                                          | Version       | Notes                                                                                                          |
| :------------------------------------------------------------------ | :------------ | :------------------------------------------------------------------------------------------------------------- |
| [ONNX Runtime](https://onnxruntime.ai/docs/install/)                | >=1.18        | Download pre-built binaries or build from source. Use the GPU build for the CUDA Execution Provider.           |
| [NCNN](https://github.com/Tencent/ncnn/releases)                    | recent        | Tencent NCNN; on Windows use the `windows-vs2022` prebuilt.                                                     |
| [OpenCV](https://opencv.org/releases/)                              | >=4.5.0       | Used for image preprocessing (`core` + `imgproc`).                                                             |
| C++ Compiler                                                        | C++17 Support | Needed for `<filesystem>`. ([GCC](https://gcc.gnu.org/), [Clang](https://clang.llvm.org/), MSVC 2022/2026)      |
| [CMake](https://cmake.org/download/)                                | >=3.16        | Cross-platform build system generator.                                                                         |
| [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (Optional)| 12.x          | Required for GPU acceleration via ONNX Runtime's CUDA Execution Provider (match your ONNX Runtime GPU build).   |
| [MNN](https://github.com/alibaba/MNN) (Optional)                    | >=3.0         | Only for the third export format / benchmarking.                                                               |

> **Note:** The CUDA Execution Provider is ABI-coupled to a CUDA major version, use the ONNX Runtime GPU build that matches your CUDA Toolkit (e.g. the CUDA-12 build with CUDA 12.x), or you'll hit loader errors.


### MacOS
|                                                         | Version       | Notes                                                                                                          |
| :------------------------------------------------------------------ | :------------ | :------------------------------------------------------------------------------------------------------------- |
| MacOS | Sonoma or newer (14.0+) | SwiftUI API floor (onKeyPress, zero-param onChange)  |
| [Xcode Command Line Tools](https://developer.apple.com/documentation/xcode/installing-the-command-line-tools/) | Xcode 15+    | Install with xcode-select --install. Provides swift, codesign, ditto. Full Xcode GUI not required for a build.   |
| [Swift toolchain](https://www.swift.org/swiftly/documentation/swiftly/install-toolchains/) | 5.9+  | swift-tools-version:5.9 in Package.swift; ships with the CLT/Xcode above. Build: swift build -c release --package-path mac. |       
| Apple SDK frameworks | macOS 14+ SDK (system) | SwiftUI, AppKit, AVFoundation, Core ML, Core Image, Core Video, ImageIO, QuartzCore, etc. |


## 🛠️ Build Instructions

### CLI (Linux & Windows)

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/skywalker-lt/yolo-master-edge.git
    cd yolo-master-edge/cpp
    ```

2.  **Create Build Directory:**

    ```bash
    mkdir build && cd build
    ```

3.  **Configure with CMake:**
    Point CMake at your extracted ONNX Runtime and NCNN SDKs via `ONNXRUNTIME_ROOT` and `NCNN_ROOT`.

    ```bash
    # Example for Linux (adjust paths as needed)
    cmake .. -DCMAKE_BUILD_TYPE=Release \
      -DONNXRUNTIME_ROOT=/path/to/onnxruntime \
      -DNCNN_ROOT=/path/to/ncnn
    ```

    ```bat
    :: Example for Windows, from the "x64 Native Tools Command Prompt"
    cmake .. -DCMAKE_BUILD_TYPE=Release ^
      -DOpenCV_DIR=C:/dev/opencv/build/x64/vc16/lib ^
      -DONNXRUNTIME_ROOT=C:/dev/onnxruntime-win-x64 ^
      -DNCNN_ROOT=C:/dev/ncnn-windows-vs2022/x64
    ```

    **CMake Options:**
    - `-DONNXRUNTIME_ROOT=<path>`: **(Required)** Path to the extracted ONNX Runtime library.
    - `-DNCNN_ROOT=<path>`: **(Required)** Path to the extracted NCNN library.
    - `-DCMAKE_BUILD_TYPE=Release`: (Optional) Build with optimizations.
    - `-DPORTABLE=ON`: (Optional, Linux) Slim build for a small self-contained bundle (image inference only).
    - If CMake struggles to find OpenCV, set `-DOpenCV_DIR=/path/to/opencv/build`.

4.  **Build the Project:**
    Use the build tool generated by CMake (Make, Ninja, or Visual Studio).

    ```bash
    # Using CMake's generic build command (works with Make, Ninja, MSBuild)
    cmake --build . --config Release
    ```

5.  **Locate Executable:**
    The compiled executable (`yolomaster_edge`, or `yolomaster_edge.exe` on Windows) is located in the `build` directory. On Windows the required backend and OpenCV DLLs are auto-copied next to it.

### Windows GUI (with CUDA)

1. **Clone the Repository**
   ```shell
   git clone https://github.com/skywalker-lt/yolo-master-edge.git
   cd yolo-master-edge/gui
   ```
   
2. **Copy and Edit the Paths**
   ```bat
   copy sdk-paths.example.cmd sdk-paths.cmd
   ```
   Edit `sdk-paths.cmd` with your locations. It is gitignored. Leave a backend blank to skip it.
   
4. **Build**
   ```bat
   build.cmd            :: configure + build Release
   build.cmd run        :: build, then launch
   build.cmd clean      :: wipe build\ first
   ```
   Output: `gui\build\Release\yolomaster_gui.exe`

   > If PowerShell blocks `.ps1` scripts, use `.cmd` scripts as they are not subject to execution policy. `build.ps1` is equivalent and takes the same paths as parameters.   
   
### MacOS 

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/skywalker-lt/yolo-master-edge.git
    cd yolo-master-edge/cpp
    ```

2.  **Build the App and Run** 
    ```zsh
    xcode-select --install
    swift run -c release --package-path mac YOLOMasterApp
    ```

## 🚀 Usage (CLI)

Run the executable, pointing it at a model and a source (image, directory, video, or `dataset.yaml`):

```bash
./yolomaster_edge --model ../../models/esmoe_n_visdrone_sim.onnx \
                  --source path/to/image_or_dir \
                  --conf 0.25 --out out
```

The backend is inferred from the model (`.onnx` → ONNX Runtime, an NCNN directory → NCNN), and class names and input size are read from the model metadata. Common options:

```text
--backend      auto | onnx | ncnn        (default: auto-detect)
--device       cpu | cuda                (ONNX backend; falls back to CPU)
--conf         confidence threshold      (default 0.25; lower for dense scenes)
--iou          NMS IoU threshold         (default 0.50)
--multi-label  one detection per class > conf per anchor (matches Ultralytics val mAP)
--save-txt     dir to write predictions  ('class conf x1 y1 x2 y2')
--out          dir for annotated outputs  --no-save / --quiet
```

See `tests/run_tests.sh` for the 16-test robustness battery.

### Benchmark mode, on-device accuracy and tracking (v1.2.0)

```bash
# cold probe run (10 warm-up + 50 timed forwards on a gray 640 probe) plus per-stage stats of the
# dataset pass, written as one yolomaster-bench/v1 JSON (the schema the Android / iOS Bench tabs will share)
yolomaster_edge -m model.onnx -s images/ --bench cold --bench-json bench.json --no-save --quiet
# sustained: a 2-minute loop, cold vs slowest-quarter median, throttle percentage, one-second sparkline
yolomaster_edge -m model.onnx -s images/ --bench sustained --bench-minutes 2 --bench-json sustained.json
# accuracy: a second pass at the val protocol (conf 0.001, iou 0.7, multi-label, max_det 300) scored
# in-process; equals scripts/eval_map.py on the txt dump to four decimals. Labels dir or 'auto'
# (.../images/... -> .../labels/...) for dataset.yaml sources such as datasets/coco500/coco500.yaml
yolomaster_edge -m model.onnx -s datasets/coco500/coco500.yaml --accuracy auto --bench-json acc.json
# score an existing txt dump the same way
yolomaster_score preds_dir images_dir labels_dir [--per-class]
# multi-object tracking on video: ids on the annotated mp4, --save-txt gains a 7th column
yolomaster_edge -m model.onnx -s clip.mp4 --track botsort --save-txt tracks/
# TensorRT / ORT-CUDA builds: --cpu-preproc (reference path), --cuda-graph (TensorRT graph replay)
```

GPU builds (`-DUSE_CUDA_PREPROC=ON`, needs nvcc) preprocess on the GPU for TensorRT and the ORT
CUDA EP and can replay the TensorRT frame as a CUDA graph; numbers and the timing contract are in
`GPU_PREPROC_RESULTS.md`. `scripts/make_coco_subset.py` builds the 500-image COCO val subset and `scripts/package_eval_sets.sh`
packages it with `visdrone50/` as release assets. Percentiles are floor rank
(`sorted[min(int(q * n), n - 1)]`), the convention of the phone Bench tabs; `sustained` is the
median of the slowest quarter. `botsort` adds sparse-optical-flow camera motion compensation
(needs OpenCV `video` + `calib3d`, `-DUSE_GMC=ON`, the default); `bytetrack` is the same
association without it. The same tracker runs in the API server (`track=` on `/v1/video` and
`/v1/stream`), and `POST /v1/bench` returns the same JSON from a server worker.

**macOS (Core ML Runner 1.2.0, branch `dev/v1.2.0-mac`).** The same features on the Swift runner,
sharing the C++ scorer, tracker and statistics through the portable core in
`mac/Sources/YOLOMasterCore` (compiled by both CMake and the Swift package):

```bash
swift build -c release --package-path mac
B=$(swift build -c release --package-path mac --show-bin-path)/yolomaster-coreml
$B --model v0.1-N.mlpackage --source coco500/images --accuracy coco500/labels --save-txt val/ --bench-json acc.json
$B --model v0.1-N.mlpackage --source images/ --bench sustained --bench-minutes 2 --bench-json sustained.json
$B --model v0.1-N.mlpackage --source clip.mp4 --track botsort --save-txt tracks/ --out tracked.mp4
$B --model v0.1-N.mlpackage --source images/ --cpu-preproc --dump-input dumps/   # Metal vs CPU preprocessing parity
bash mac/tests/run_mac_tests.sh   # MODEL=... COCO500=... : the Mac battery
```

The app gains a Bench section (cold / sustained / accuracy, Save JSON), a Tracking picker for
video and a Preprocess device picker (Metal GPU letterbox straight into the Core ML input, zero
copy from the camera). BoT-SORT's camera motion on macOS comes from Vision's translational
registration. See `mac/RELEASE_NOTES-1.2.0.md`.

Measured on an Apple M4 Max (macOS 26.3, Core ML GPU compute unit, Metal preprocessing,
`mac/tests/run_mac_tests.sh`, 12/12):

| model | coco500 mAP50 / mAP50-95 on Core ML | Linux ORT-CPU fp32, same images | preprocess CPU -> Metal |
|---|---|---|---|
| v0.1-N (COCO) | 0.5931 / 0.4306 | 0.5940 / 0.4309 | 2.59 -> 1.83 ms |
| v0.1-seg-N (COCO, boxes) | 0.5860 / 0.4271 | - | 2.56 -> 1.93 ms |

The in-process number equals `scripts/eval_map_standalone.py` on the runner's own `--save-txt`
dump within 0.0001 (ultralytics' unstable `argsort` on the many fp16 confidence ties decides the
4th decimal), the Metal input tensor is within 0.00294 of the Linux `preprocess_nchw` rule on the
same decoded pixels (the CUDA kernel: 0.0029) and switching the preprocessing device moves
coco500 mAP50-95 by 0.0005 at most.

## 🤖 Jetson Orin (Native TensorRT)

A prebuilt aarch64 runner for **Jetson Orin** (Nano / NX / AGX) on **JetPack 7** is attached to the [Releases](https://github.com/skywalker-lt/yolo-master-edge/releases) page. It bundles OpenCV and uses JetPack's TensorRT + CUDA; the per-device FP16 engine is built once with the included script.

```bash
tar xzf yolomaster_edge-jetson-orin-jp7.tar.gz && cd yolomaster_edge-jetson-orin-jp7
./build_engine.sh    # builds the FP16 engine for this device (once, ~10-15 min)
./yolomaster_edge --model models/esmoe_n_fp16.engine --source <img|dir> --classes visdrone --out out
```

On an Orin Nano 4 GB the FP16 engine runs at **35.7 FPS** (27.8 ms) with **mAP50-95 0.2029 (−0.34% vs FP32)**. FP16 is the recommended target on this model, its area-attention does not quantize, so INT8 is both slower and less accurate here. To build from source, the [`jetson/`](jetson/) scripts drive the engine build and packaging; see [`jetson/README.md`](jetson/README.md) and [`jetson/DEPLOYMENT_LOG.md`](jetson/DEPLOYMENT_LOG.md).

## 📊 Results

Inference performed on full 548 VisDrone validation images against the PyTorch original (`mAP50-95 = 0.2036`), using identical settings (conf 0.001, NMS IoU 0.7, multi-label).

| Inference Backend | Device | mAP50-95 | Δ vs PyTorch | End-to-end Latency | FPS |
| :------------------------ | :------- | :------- | :----------- | :------ | :---- |
| ONNX | CPU | 0.2034 | −0.02%  | 40 ms | 25.0  |
| ONNX (CUDA) | H200 SXM | 0.2033 | −0.03% | 7.8 ms  | 128   |
| ONNX (CUDA) | RTX 5070Ti Laptop | 0.2033 | −0.03%  | 9.0 ms  | 111   |
| NCNN | CPU | 0.2034 | −0.02% | 80 ms | 12.5  |
| NCNN (Vulkan) | RTX 5070Ti Laptop | 0.2034 | −0.02%  | 20.2 ms | 49.5  | 
| MNN | CPU | 0.2034 | −0.02% | 74 ms | 13.5  |
| MNN (OpenCL) | RTX 5070Ti Laptop | 0.2034 | −0.02%  | 19.1 | 52.4  | 
| INT8 mixed ¹ | CPU | 0.1952 | −0.84% | 137 ms  | 7.2   |
| TensorRT FP16 | Jetson Orin Nano 4GB | 0.2029 | −0.34% | 27.8 ms | 35.7 |
| Core ML | Apple M4 Max | see ² | see ² | 17.4 ms | 57.4  |

CPU latencies are x86 @ 4 threads on one host; mAP is identical across FP32 formats because they are of the same graph. The Jetson row is a native TensorRT FP16 engine, measured on-device.

> ¹ INT8 is *slower* than FP32 on CPU, its throughput payoff needs INT8 tensor cores, not x86 CPUs. The CPU INT8 result is an **accuracy** proof (−0.84%, within budget); on the actual accelerator, note that even on the Orin's tensor cores FP16 wins here (the attention doesn't quantize, see the TensorRT row and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) Section 9).

> ² Since Core ML Runner 1.2.0 the macOS runner scores itself in process (the same C++ scorer as the Linux CLI): on coco500 v0.1-N reaches 0.4306 mAP50-95 on Core ML vs 0.4309 on ORT-CPU fp32 (see the v1.2.0 section). The VisDrone EsMoE-N figure for this row is not measured yet; the latency stays the 1.1 measurement.

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full methodology, INT8 quantization deep-dive, and numerical parity analysis.


## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the [project repository](https://github.com/skywalker-lt/yolo-master-edge).
