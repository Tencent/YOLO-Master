# YOLO-Master Edge Deployment Example

This example supports issue #51: vertical-model edge inference acceleration and consistency validation.

It provides a lightweight, reproducible scaffold for exporting YOLO-Master models to ONNX plus NCNN/MNN, running vertical-domain preprocessing, comparing backend outputs, and summarizing edge benchmark logs.

## Files

- `edge_utils.py` - shared preprocessing, postprocessing, consistency, and benchmark utilities.
- `export_edge_models.py` - export helper for ONNX, NCNN, and MNN.
- `validate_edge_outputs.py` - compare PyTorch/exported backend outputs saved as `.npy` tensors.
- `cpp/edge_benchmark.cpp` - C++ benchmark runner with backend selection, OpenCV letterbox preprocessing, CSV latency output, and summary stats.
- `cpp/backends/` - backend interface plus ONNX, NCNN, and MNN implementation slots.
- `CMakeLists.txt` - portable CMake target for the C++ benchmark entry.

## Vertical Profiles

The example includes two profiles:

- `visdrone`: keeps long/short aspect ratio, uses lower confidence for small objects.
- `sku110k`: supports high-resolution shelf images and a slightly higher NMS IoU.

## Export

```bash
python export_edge_models.py --model runs/train/weights/best.pt --formats onnx ncnn --imgsz 960 --half
python export_edge_models.py --model runs/train/weights/best.pt --formats onnx mnn --imgsz 960
```

For ONNX simplification, install export dependencies and pass `--simplify`.

## Consistency Validation

Save backend outputs as `.npy` tensors with compatible shapes, then run:

```bash
python validate_edge_outputs.py --reference pytorch.npy --candidate onnx.npy --tolerance 0.005
python validate_edge_outputs.py --reference pytorch.npy --candidate ncnn.npy --tolerance 0.01
```

The tool reports max absolute error, mean absolute error, RMSE, and whether the configured tolerance is met.

## CMake Benchmark Build

The C++ runner requires OpenCV for image loading and letterbox preprocessing. Backend SDK calls are currently isolated behind `cpp/backends/` so ONNX Runtime, NCNN, and MNN can be wired independently.

```bash
cmake -S . -B build
cmake --build build
./build/yolo_master_edge_benchmark \
  --backend onnx \
  --model best.onnx \
  --images /path/to/VisDrone/images/val \
  --profile visdrone \
  --imgsz 960 \
  --limit 500 \
  --output benchmark_onnx.csv
```

`--images` accepts either a directory of images or a text file with one image path per line. Use `--limit 500` for the issue validation subset.

To enable real ONNX Runtime inference, provide an ONNX Runtime package root:

```bash
cmake -S . -B build-ort \
  -DWITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build build-ort
```

Without `WITH_ONNXRUNTIME=ON`, the ONNX backend remains a buildable stub so the benchmark CLI, preprocessing, and CSV/report plumbing can be developed without backend SDKs installed.

For a local ONNX Runtime C/C++ package, point `ONNXRUNTIME_ROOT` at the directory containing `include/` and `lib/`:

```bash
cmake -S examples/YOLO-Master-Edge-Deployment \
  -B examples/YOLO-Master-Edge-Deployment/build-ort \
  -DWITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build examples/YOLO-Master-Edge-Deployment/build-ort
```

For example, Homebrew users can use `-DONNXRUNTIME_ROOT="$(brew --prefix onnxruntime)"`.

Smoke test the ONNX backend with bundled sample images:

```bash
examples/YOLO-Master-Edge-Deployment/build-ort/yolo_master_edge_benchmark \
  --backend onnx \
  --model YOLO-Master-EsMoE-N.onnx \
  --images ultralytics/assets \
  --profile visdrone \
  --imgsz 960 \
  --limit 2 \
  --output benchmark_onnx_assets.csv
```

Example local CPU result with `imgsz=960`:

```text
backend=onnx model=YOLO-Master-EsMoE-N.onnx profile=visdrone imgsz=960 conf=0.2 iou=0.55 output=benchmark_onnx_assets.csv
count,mean_ms,p50_ms,p95_ms,p99_ms,fps
2,402.451,400.866,400.866,400.866,2.48478
```

The CSV contains one row per image:

```text
image,preprocess_ms,inference_ms,postprocess_ms,total_ms,detections
```

## Recommended Issue #51 Workflow

1. Train or reuse a YOLO-Master-EsMoE-N checkpoint on VisDrone or SKU-110K.
2. Export ONNX plus NCNN or MNN.
3. Validate ONNX opset/simplification and NCNN/MNN conversion files.
4. Run the same 500-image validation list through PyTorch and exported backends.
5. Compare mAP50-95 deltas and tensor/output differences.
6. Report latency P50/P95/P99 and FPS per backend/platform.
