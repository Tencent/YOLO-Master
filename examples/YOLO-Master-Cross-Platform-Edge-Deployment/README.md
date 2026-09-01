# YOLO-Master Cross-Platform Edge Inference Runtime

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white"> <img alt="NCNN" src="https://img.shields.io/badge/NCNN-Tencent-blue.svg"> <img alt="MNN" src="https://img.shields.io/badge/MNN-Alibaba-orange.svg"> <img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-NVIDIA-76B900.svg"> <img alt="Core ML" src="https://img.shields.io/badge/CoreML-Apple-black.svg"> <img alt="Linux" src="https://img.shields.io/badge/Linux-FCC624.svg?logo=linux&logoColor=black"> <img alt="Windows" src="https://img.shields.io/badge/Windows-0078D6.svg?logo=windows&logoColor=white"> <img alt="Jetson" src="https://img.shields.io/badge/Jetson%20Orin-76B900.svg?logo=nvidia&logoColor=white"> <img alt="macOS" src="https://img.shields.io/badge/macOS-000000.svg?logo=apple&logoColor=white"> <img alt="iOS" src="https://img.shields.io/badge/iOS-000000.svg?logo=apple&logoColor=white"> 

This project provides a cross-platform inference runtime for [YOLO-Master](https://github.com/Tencent/YOLO-Master) object-detection models using [ONNX Runtime](https://onnxruntime.ai/), [NCNN](https://github.com/Tencent/ncnn), [MNN](https://github.com/alibaba/mnn), [TensorRT](https://github.com/nvidia/tensorrt), and [Core ML](https://github.com/apple/coremltools) backends. The supported targets are Linux, Windows 10/11, Jetson, and macOS, with CPU, [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit), and [Apple Metal Performance Shaders](https://developer.apple.com/documentation/metalperformanceshaders) execution where available. The runtime can infer the model format and read class names and input size from model metadata.
This project provides a universal inference runtime for [YOLO-Master](https://github.com/Tencent/YOLO-Master) object-detection models, leveraging, [ONNX Runtime](https://onnxruntime.ai/), [NCNN](https://github.com/Tencent/ncnn), [MNN](https://github.com/alibaba/mnn), [TensorRT](https://github.com/nvidia/tensorrt), and [CoreML](https://github.com/apple/coremltools) backends. It runs on almost every platform: Linux, Windows (10/11), Jetson, MacOS, and **iOS (NEW!)**; supports CPU, [CUDA](https://developer.nvidia.com/cuda-toolkit), [MPS](https://developer.apple.com/documentation/metalperformanceshaders), and [ANE (on Apple devices）](https://machinelearning.apple.com). It's capable of auto-detecting the model format, class names, and input size -- designed for real-time, end-to-end edge deployment in some of the most challenging tasks (VisDrone, SKU-110K, AI-TOD-v2, etc.).

## Issue #51 validation bundle

This checkout also contains an auditable validation workflow for Issue #51.
The [Chinese technical summary](TECHNICAL_SUMMARY_ZH.md) and
[technical report](TECHNICAL_REPORT.md) define the reproducible protocol,
machine-readable manifests, and platform records. The repository provides the
runner and validation gates; it does not include private checkpoints, datasets,
or generated predictions. Therefore, numerical accuracy, latency, INT8, and
ARM64 claims must be reported only with the corresponding hashes and raw logs.
The release notes elsewhere in this upstream README describe separate published
artifacts; they are not measurements produced by this validation checkout. The
sections marked **upstream reference** are retained for provenance only; their
models, datasets, hardware and measurements are not part of this branch.

### Verification status of this checkout

The repository currently provides the runner, validation scripts, schemas and
contract tests. A Ubuntu 22.04 x86_64 smoke run has exercised a YOLOv5s ONNX
model on one image. The EsMoE-N checkpoint, the VisDrone/SKU-110K data splits,
full mAP comparison, INT8 calibration run and native ARM64 measurements are
not bundled; those claims require the corresponding model/data hashes and raw
logs before they can be reported.

<p align="left">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/skywalker-lt/yolo-master-edge/dev/ios/assets/edge_deployment_architecture_dark.png">
    <img src="https://raw.githubusercontent.com/skywalker-lt/yolo-master-edge/dev/ios/assets/edge_deployment_architecture_light.png" width="900" alt="Edge Deployment Bundle — Architecture">
  </picture>
</p>

---

## 📱 Update (27-08-2026): YOLO-Master for iPhone v1.1.0 Beta Build 1

<img width="4812" height="2291" alt="screnshots-framed" src="https://github.com/user-attachments/assets/e858e07b-eeff-40c2-b11a-dcb4dba44577" />

<br>
<br>

**🍾 Welcome to the 5th platform of YOLO-Master Edge. [Try it now.](https://testflight.apple.com/join/EVExpVHD)**

Native iOS SwiftUI app for on-device YOLO-Master detection and segmentation, powered by Apple Core ML and the same YOLOMasterKit inference path as the macOS runtime. Everything runs locally on the Neural Engine, GPU, or CPU. Nothing you capture leaves the iPhone. Requires iPhone on iOS 17 or later. iPhone 13 and later models are recommended. 

### Features

- **📹 Live** - Real-time camera detection with a dynamic performance visuliaztion (FPS tachometer, per-stage latency measures, and phone's thermal state), multi-cam with automatic lens switching and pinch zoom, tap-to-focus, torch, and a full-resolution shutter that renders the live overlay (boxes and segmentation masks) into the saved photo. Also supports live IoU/conf tuning without pausing the inference. 

- **📸 Photo** - Batch detection over images you pick from your library, up to 100 images at a time. Per-image and batch stats, segmentation masks, live conf/IoU tuning, and export of annotated images back to Photos.

- **🎛️ Bench** - On-device benchmarking on all devices. A Cold Sweep measures every bundled model across compute units (Neural Engine (ANE), GPU, CPU) with expandable pre/inference/decode stage breakdowns, and a Sustained mode runs a thermal-throttle test with a live latency heatbeat sparkline and a colored state timeline. Pause/resume, a persistent run History vault (with per-run graphs) and CSV export.

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

## 🚀 Update (12-08-2026): YOLO-Master Edge v1.1.0 is up!

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

Now the Windows C++ edge runner has an improved backend and dedicated GUI. **YOLO-Master Windows Runner GUI** provides a C++ edge inference backend that bundles [ONNX](https://onnxruntime.ai/), [ncnn](https://github.com/Tencent/ncnn), and [MNN](https://github.com/alibaba/MNN) with **GPU acceleration**, together with a [Dear ImGui](https://github.com/ocornut/imgui) frontend. It includes the same default `YOLO-Master-v0.1-seg-N` segmentation model as the macOS runner.

<br>

<img width="400" alt="48 2" src="https://github.com/user-attachments/assets/fa96097b-1014-46c0-8692-2c3656f4f763" />  <img width="400" alt="50 2" src="https://github.com/user-attachments/assets/db1e2194-584e-4898-b2b3-e7370fa325c5" />
<img width="800" alt="49 2" src="https://github.com/user-attachments/assets/6a438094-b352-4f59-bb55-9e3c600837ad" />

<br>

- **Three Backends in One App** ONNX, ncnn, and MNN all ship in single executable. Inference backends can be switched with a single click.
- **GPU Acceleration for All Backends** up to **4x speedup** with CUDA-accelerated inference on consumer devices. (please refer to the inference speed comparison table in [Relases](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.0.0-windows)) 

See the [Windows release notes](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.0.0-windows) for build and benchmark details.

---

## 🍎 Update (17-07-2026): YOLO-Master Core ML Runner for macOS (GUI)

**[Download](https://github.com/skywalker-lt/yolo-master-edge/releases/download/v1.0.0-macos/YOLO-Master-CoreML-Runner-1.0.0.zip) and try it now!**

Alongside the Linux and Windows C++ runtimes, this project provides a native macOS runner, **YOLO-Master Core ML Runner**: a [SwiftUI](https://developer.apple.com/xcode/swiftui/) frontend over an Apple [Core ML](https://developer.apple.com/documentation/coreml) backend for on-device [YOLO-Master](https://github.com/Tencent/YOLO-Master) inference. It ships with a default `YOLO-Master-v0.1-seg-N` segmentation model for an immediate smoke test.

<img width="400" alt="Screenshot1" src="https://github.com/user-attachments/assets/d1747b4d-0961-458e-99c5-2a9870b8df96" /> <img width="400" alt="Screenshot2" src="https://github.com/user-attachments/assets/5f71d80a-6238-49bd-a230-95ccd4020d29" /> 
<img width="400" alt="Screenshot3" src="https://github.com/user-attachments/assets/9cc60636-b795-4326-992c-06239a77db55" /> <img width="400" alt="Screenshot4" src="https://github.com/user-attachments/assets/b5ee48bb-52dc-4ff7-b0bd-f2461b34ad7c" />

- **Detection & Segmentation:** Runs both bounding-box detectors and instance-segmentation models, with anti-aliased mask overlays and a Masks / Boxes / Both toggle.
- **Images, Video & Live Camera:** Infers single images, whole folders (batch), and MP4 video, plus a low-latency **live webcam** mode with a real-time FPS / ms-per-frame readout.
- **⭐️ Real-Time Tuning:** Confidence, IoU, box style, labels, and letterbox/stretch preprocessing are adjustable at runtime. The forward pass is cached, so tuning redraws without rerunning inference.
- **Signed & Notarized:** A **universal** (Apple Silicon + Intel) bundle, **Developer-ID signed and notarized by Apple**; it installs by double-clicking on macOS 14+.

See the [macOS release notes](https://github.com/skywalker-lt/yolo-master-edge/releases/tag/v1.0.0-macos) for build and distribution details.

---

## ✨ Benefits

- **Universal CLI Binary for Linux and Windows:** A single executable integrates **ONNX Runtime**, **NCNN** and **MNN** backends; the backend, class names, and input size are auto-detected from the model — no recompilation or any dataset YAML needed at runtime.
- **Verified Accuracy:** Reproduces the PyTorch original to **< 0.5%** mAP50-95 across ONNX / NCNN / MNN, and **< 1.0%** under INT8 quantization, on 548 VisDrone validation images.
- **Deployment-Friendly:** Cross-platform [CMake](https://cmake.org/) build producing **self-contained and relocatable bundles** for Linux x86_64 and Windows 10/11 — installable by unzip, no dependencies on the target.
- **GUI:** Windows 10/11 and macOS provide GUI runners that integrate the CLI bundle functionality and support GPU acceleration.
- **GPU Acceleration:** Supports FP32 CPU inference and [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) through the ONNX Runtime CUDA Execution Provider on Linux and Windows; Windows also supports NCNN [Vulkan](https://vulkan.org) and MNN [OpenCL](https://opencl.org), NVIDIA Jetson Orin supports a native TensorRT backend (JetPack 7), and macOS uses [MPS](https://developer.apple.com/documentation/metalperformanceshaders) through Core ML.

## ☕ Note

The exported models embed their class names, input size, and stride as ONNX/NCNN/MNN metadata, so the runtime configures itself from the model file. Post-processing is tuned for the vertical domain — aspect-ratio-preserving letterbox, per-class **multi-label** NMS, and a low default confidence threshold appropriate for VisDrone's small, dense objects.

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
# NCNN — Ultralytics uses pnnx under the hood
yolo export model=EsMoE-N_VisDrone.pt format=ncnn imgsz=640

# MNN — convert the exported ONNX with MNN's converter
mnnconvert -f ONNX --modelFile esmoe_n_visdrone_sim.onnx --MNNModel esmoe_n_visdrone.mnn --bizCode edge
```

For more details on exporting, refer to the [Ultralytics Export documentation](https://docs.ultralytics.com/modes/export/).

### Core ML

```zsh
# detector or segmenter (task auto-detected)
python coreml_export/export_coreml.py --weights model.pt --imgsz 640 --out model.mlpackage

# YOLO-Master default imgsz is 800 for AI-TOD models — pass --imgsz accordingly
python coreml_export/export_coreml.py --weights yolo-master-v0.1-N_aitodv2.pt --imgsz 800 --out v0.1-N.mlpackage

# sunsmarterjie/yolov12 checkpoints (split qk+v area-attention) — stock ultralytics + the flag
python coreml_export/export_coreml.py --weights yolov12x.pt --imgsz 640 --out yolov12x.mlpackage --yolov12-aattn

# a LoRA fine-tune: merge the trained adapters first
python coreml_export/export_coreml.py --weights base.pt --merge-lora-dir lora_adapter/ --imgsz 640 --out ft.mlpackage
```


## ⚙️ Dependencies

Ensure you have the following dependencies installed (not required if you only want to smoke-test the pre-built bundles):

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

> **Note:** The CUDA Execution Provider is ABI-coupled to a CUDA major version — use the ONNX Runtime GPU build that matches your CUDA Toolkit (e.g. the CUDA-12 build with CUDA 12.x), or you'll hit loader errors.


### macOS
|                                                         | Version       | Notes                                                                                                          |
| :------------------------------------------------------------------ | :------------ | :------------------------------------------------------------------------------------------------------------- |
| macOS | Sonoma or newer (14.0+) | SwiftUI API floor (onKeyPress, zero-param onChange)  |
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
   
### macOS

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

The backend is inferred from the model (`.onnx` → ONNX Runtime, an NCNN directory or `.param` → NCNN, `.mnn` → MNN, `.engine`/`.trt` → TensorRT), and class names and input size are read from model metadata. Common options:

```text
--backend      auto | onnx | ncnn | mnn | trt (default: auto-detect)
--device       backend-dependent: cpu, cuda, vulkan, opencl, trt, coreml
--conf         confidence threshold      (default 0.25; lower for dense scenes)
--iou          NMS IoU threshold         (default 0.50)
--multi-label  one detection per class >= conf per anchor (matches Ultralytics val mAP)
--save-txt     dir to write predictions  ('class conf x1 y1 x2 y2')
--out          dir for annotated outputs  --no-save / --quiet
```

See `tests/run_tests.sh` for the 16-test robustness battery.

## 🤖 Jetson Orin (Native TensorRT)

A prebuilt aarch64 runner for **Jetson Orin** (Nano / NX / AGX) on **JetPack 7** is attached to the [Releases](https://github.com/skywalker-lt/yolo-master-edge/releases) page. It bundles OpenCV and uses JetPack's TensorRT + CUDA; the per-device FP16 engine is built once with the included script.

```bash
tar xzf yolomaster_edge-jetson-orin-jp7.tar.gz && cd yolomaster_edge-jetson-orin-jp7
./build_engine.sh    # builds the FP16 engine for this device (once, ~10-15 min)
./yolomaster_edge --model models/esmoe_n_fp16.engine --source <img|dir> --classes visdrone --out out
```

On an Orin Nano 4 GB the FP16 engine runs at **35.7 FPS** (27.8 ms) with **mAP50-95 0.2029 (−0.07 pp vs FP32)**. FP16 is the recommended target for this model because the area-attention path is not quantized, making INT8 both slower and less accurate here. To build from source, the [`jetson/`](jetson/) scripts drive the engine build and packaging; see [`jetson/README.md`](jetson/README.md) and [`jetson/DEPLOYMENT_LOG.md`](jetson/DEPLOYMENT_LOG.md).

## 📊 Results

Inference performed on full 548 VisDrone validation images against the PyTorch original (`mAP50-95 = 0.2036`), using identical settings (conf 0.001, NMS IoU 0.7, multi-label).

| Inference Backend | Device | mAP50-95 | Δ vs PyTorch | End-to-end Latency | FPS |
| :------------------------ | :------- | :------- | :----------- | :------ | :---- |
| ONNX | CPU | 0.2034 | −0.02 pp  | 40 ms | 25.0  |
| ONNX (CUDA) | H200 SXM | 0.2033 | −0.03 pp | 7.8 ms  | 128   |
| ONNX (CUDA) | RTX 5070Ti Laptop | 0.2033 | −0.03 pp  | 9.0 ms  | 111   |
| NCNN | CPU | 0.2034 | −0.02 pp | 80 ms | 12.5  |
| NCNN (Vulkan) | RTX 5070Ti Laptop | 0.2034 | −0.02 pp  | 20.2 ms | 49.5  |
| MNN | CPU | 0.2034 | −0.02 pp | 74 ms | 13.5  |
| MNN (OpenCL) | RTX 5070Ti Laptop | 0.2034 | −0.02 pp  | 19.1 ms | 52.4  |
| INT8 mixed ¹ | CPU | 0.1952 | −0.84 pp | 137 ms  | 7.2   |
| TensorRT FP16 | Jetson Orin Nano 4GB | 0.2029 | −0.07 pp | 27.8 ms | 35.7 |
| Core ML | Apple M4 Max | N/A (no validator bundled) | N/A | 17.4 ms | 57.4  |

CPU latencies are x86 @ 4 threads on one host; mAP is identical across FP32 formats because they are of the same graph. The Jetson row is a native TensorRT FP16 engine, measured on-device.

> ¹ INT8 is *slower* than FP32 on CPU — its throughput payoff needs INT8 tensor cores, not x86 CPUs. The CPU INT8 result is **accuracy evidence** (−0.84 percentage points, within budget); on the actual accelerator, FP16 also wins on Orin because the attention path is not quantized (see the TensorRT row and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) Section 9).

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full methodology, INT8 quantization deep-dive, and numerical parity analysis.


## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the [project repository](https://github.com/skywalker-lt/yolo-master-edge).
