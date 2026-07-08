# YOLO-Master-EsMoE-N Edge Inference C++ Runtime

<img alt="C++" src="https://img.shields.io/badge/C++-17-blue.svg?style=flat&logo=c%2B%2B"> <img alt="Onnx-runtime" src="https://img.shields.io/badge/OnnxRuntime-717272.svg?logo=Onnx&logoColor=white"> <img alt="NCNN" src="https://img.shields.io/badge/NCNN-Tencent-blue.svg"> <img alt="MNN" src="https://img.shields.io/badge/MNN-Alibaba-orange.svg"> <img alt="Platform" src="https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey.svg">

This project provides a universal C++ inference runtime for [YOLO-Master](https://github.com/Tencent/YOLO-Master) **EsMoE-N** object-detection models, leveraging both the [ONNX Runtime](https://onnxruntime.ai/) and [NCNN](https://github.com/Tencent/ncnn) backends together with the [OpenCV](https://opencv.org/) library. A single binary runs either backend on CPU and [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit), auto-detecting the model format, class names, and input size — designed for edge deployment of vertical-domain detectors (VisDrone aerial imagery, SKU-110K, etc.).

## ✨ Benefits

- **One Universal Binary:** A single executable integrates both **ONNX Runtime** and **NCNN** backends; the backend, class names, and input size are auto-detected from the model — no recompilation or any dataset YAML needed at runtime.
- **Verified Accuracy:** Reproduces the PyTorch original to **< 0.5%** mAP50-95 across ONNX / NCNN / MNN, and **< 1.0%** under INT8 quantization, on 548 VisDrone validation images.
- **Deployment-Friendly:** Cross-platform [CMake](https://cmake.org/) build producing **self-contained and relocatable bundles** for Linux x86_64 and Windows 10/11 — installable by unzip, no dependencies on the target.
- **GPU Acceleration:** Supports FP32 CPU inference and [NVIDIA CUDA](https://developer.nvidia.com/cuda-toolkit) GPU acceleration through the ONNX Runtime CUDA Execution Provider on Linux, and on [NVIDIA Jetson](https://developer.nvidia.com/embedded-computing) Orin via a native TensorRT backend (JetPack 7).

## ☕ Note

The exported models embed their class names, input size, and stride as ONNX/NCNN metadata, so the runtime configures itself from the model file. Post-processing is tuned for the vertical domain — aspect-ratio-preserving letterbox, per-class **multi-label** NMS, and a low default confidence threshold appropriate for VisDrone's small, dense objects.

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

## ⚙️ Dependencies

Ensure you have the following dependencies installed （not required if you only want to smoke-test the pre-built bundles):

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

## 🛠️ Build Instructions

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

## 🚀 Usage

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

See `cpp/run_tests.sh` for the 16-test robustness battery.

## 🤖 Jetson Orin (Native TensorRT)

A prebuilt aarch64 runner for **Jetson Orin** (Nano / NX / AGX) on **JetPack 7** is attached to the [Releases](https://github.com/skywalker-lt/yolo-master-edge/releases) page. It bundles OpenCV and uses JetPack's TensorRT + CUDA; the per-device FP16 engine is built once with the included script.

```bash
tar xzf yolomaster_edge-jetson-orin-jp7.tar.gz && cd yolomaster_edge-jetson-orin-jp7
./build_engine.sh    # builds the FP16 engine for this device (once, ~10-15 min)
./yolomaster_edge --model models/esmoe_n_fp16.engine --source <img|dir> --classes visdrone --out out
```

On an Orin Nano 4 GB the FP16 engine runs at **35.7 FPS** (27.8 ms) with **mAP50-95 0.2029 (−0.34% vs FP32)**. FP16 is the recommended target on this model — its area-attention does not quantize, so INT8 is both slower and less accurate here. To build from source, the [`jetson/`](jetson/) scripts drive the engine build and packaging; see [`jetson/README.md`](jetson/README.md) and [`jetson/DEPLOYMENT_LOG.md`](jetson/DEPLOYMENT_LOG.md).

## 📊 Results

Validated on full 548 VisDrone validation images against the PyTorch original (`mAP50-95 = 0.2036`), using identical settings (conf 0.001, NMS IoU 0.7, multi-label).

| Model                     | mAP50-95 | Δ vs PyTorch | Latency | FPS   |
| :------------------------ | :------- | :----------- | :------ | :---- |
| ONNX (ONNX Runtime, CPU)  | 0.2034   | −0.02%       | 40 ms   | 25.0  |
| NCNN (CPU)                | 0.2034   | −0.02%       | ~80 ms  | ~12.5 |
| MNN (CPU)                 | 0.2034   | −0.02%       | 74 ms   | 13.5  |
| INT8 mixed (CPU) ¹        | 0.1952   | −0.84%       | 137 ms  | 7.2   |
| ONNX CUDA (H200 GPU)      | 0.2033   | −0.03%       | 7.8 ms  | ~128  |
| TensorRT FP16 (Jetson Orin Nano 4GB) | 0.2029 | −0.34% | 27.8 ms | 35.7  |

CPU latencies are x86 @ 4 threads on one host; mAP is identical across FP32 formats because they are of the same graph. The Jetson row is a native TensorRT FP16 engine, measured on-device (see below).

> ¹ INT8 is *slower* than FP32 on CPU — its throughput payoff needs INT8 tensor cores, not x86 CPUs. The CPU INT8 result is an **accuracy** proof (−0.84%, within budget); on the actual accelerator, note that even on the Orin's tensor cores FP16 wins here (the attention doesn't quantize — see the TensorRT row and [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) §8).

See [`TECHNICAL_REPORT.md`](TECHNICAL_REPORT.md) for the full methodology, INT8 quantization deep-dive, and numerical parity analysis.

## 🤝 Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the [project repository](https://github.com/skywalker-lt/yolo-master-edge).
