YOLO-Master Windows Runner 1.1.0
================================

On-device YOLO-Master detection and segmentation. Self-contained: unzip anywhere
and run yolomaster_gui.exe. No installer, no admin rights, nothing else to set up.

Two bundles are published:

* YOLO-Master-Windows-<version>.zip (lean, ~150 MB)
  Runs everywhere (Windows 10/11 x64). CPU inference via ONNX Runtime / ncnn /
  MNN; GPU inference via ncnn (Vulkan) and MNN (OpenCL) using nothing but your
  graphics driver.

* YOLO-Master-Windows-CUDA-<version>.zip (~2 GB)
  Adds ONNX Runtime's CUDA execution provider with the cuDNN and CUDA runtime
  DLLs bundled, so ONNX runs on NVIDIA GPUs with only a driver installed.
  Choose this only if you want maximum ONNX throughput on an NVIDIA card.

Quick start
-----------
1. Unzip.
2. Run yolomaster_gui.exe. The bundled v0.1-seg-N model loads automatically.
3. Open an image, folder or video - or start the webcam.

If Windows SmartScreen warns about an unrecognized app: More info -> Run anyway.

What's in 1.1.0
---------------
* Slicing (Sparse SAHI + dense tiling) for small-object detection on large
  images, with adjustable tile size and per-run statistics.
* Cluster-Weighted NMS (mode picker + sigma) for refined box coordinates.
* Zoom and pan on images and paused video (mouse wheel, drag, Ctrl+0 to reset).
* Annotation export: YOLO TXT, COCO JSON and Pascal VOC XML for images, folders
  and videos (with frame sampling), plus rendered image / annotated video export.

See gui/CHANGELOG-1.1.0.md in the source repository for the complete list.

Notes
-----
* Backends: ONNX Runtime, ncnn and MNN are all included; the model picker
  auto-detects the format. Bundled models live in models/.
* The Device combo switches CPU/GPU per backend (ONNX: CUDA build only;
  ncnn: Vulkan; MNN: OpenCL).
* Source code, CLI and the macOS runner: https://github.com/skywalker-lt/yolo-master-edge
  License: AGPL-3.0. (c) 2026 Thomas Li.
