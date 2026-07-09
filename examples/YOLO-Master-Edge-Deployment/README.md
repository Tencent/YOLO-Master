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

The C++ runner requires OpenCV for image loading and letterbox preprocessing. Build with the backend SDKs you want to benchmark:

```bash
cmake -S . -B build
cmake --build build
```

Without a backend flag, ONNX/NCNN/MNN compile in stub mode so the benchmark CLI and CSV plumbing remain buildable without SDKs installed.

### ONNX Runtime

```bash
cmake -S examples/YOLO-Master-Edge-Deployment \
  -B examples/YOLO-Master-Edge-Deployment/build-ort \
  -DWITH_ONNXRUNTIME=ON \
  -DONNXRUNTIME_ROOT=/path/to/onnxruntime
cmake --build examples/YOLO-Master-Edge-Deployment/build-ort
```

If needed, pass explicit paths instead of `ONNXRUNTIME_ROOT`:

```bash
-DONNXRUNTIME_INCLUDE_DIR=/path/to/onnxruntime/include
-DONNXRUNTIME_LIB=/path/to/onnxruntime/lib/libonnxruntime.so
```

Run:

```bash
examples/YOLO-Master-Edge-Deployment/build-ort/yolo_master_edge_benchmark \
  --backend onnx \
  --model /path/to/model.onnx \
  --images /path/to/VisDrone/images/val \
  --profile visdrone \
  --imgsz 960 \
  --limit 500 \
  --output benchmark_onnx.csv
```

### NCNN

```bash
cmake -S examples/YOLO-Master-Edge-Deployment \
  -B examples/YOLO-Master-Edge-Deployment/build-ncnn \
  -DWITH_NCNN=ON \
  -DNCNN_ROOT=/path/to/ncnn/install
cmake --build examples/YOLO-Master-Edge-Deployment/build-ncnn
```

If NCNN was built but not installed, pass explicit paths. Include both source and build include dirs so generated headers such as `platform.h` are visible:

```bash
-DNCNN_INCLUDE_DIR="/path/to/ncnn/src;/path/to/ncnn/build/src"
-DNCNN_LIB=/path/to/ncnn/build/src/libncnn.a
```

Run:

```bash
examples/YOLO-Master-Edge-Deployment/build-ncnn/yolo_master_edge_benchmark \
  --backend ncnn \
  --model /path/to/exported_ncnn_model_dir \
  --images /path/to/VisDrone/images/val \
  --profile visdrone \
  --imgsz 960 \
  --limit 500 \
  --output benchmark_ncnn.csv
```

### MNN

```bash
cmake -S examples/YOLO-Master-Edge-Deployment \
  -B examples/YOLO-Master-Edge-Deployment/build-mnn \
  -DWITH_MNN=ON \
  -DMNN_ROOT=/path/to/MNN
cmake --build examples/YOLO-Master-Edge-Deployment/build-mnn
```

If needed, pass explicit paths instead of `MNN_ROOT`:

```bash
-DMNN_INCLUDE_DIR=/path/to/MNN/include
-DMNN_LIB=/path/to/MNN/lib/libMNN.so
```

Run:

```bash
examples/YOLO-Master-Edge-Deployment/build-mnn/yolo_master_edge_benchmark \
  --backend mnn \
  --model /path/to/model.mnn \
  --images /path/to/VisDrone/images/val \
  --profile visdrone \
  --imgsz 960 \
  --limit 500 \
  --output benchmark_mnn.csv
```

`--images` accepts either a directory of images or a text file with one image path per line.

## Benchmark CSV Output

ONNX Runtime, NCNN, and MNN write the same per-image CSV structure:

```text
image,preprocess_ms,inference_ms,postprocess_ms,total_ms,detections
```

Column meanings:

- `image`: source image path
- `preprocess_ms`: OpenCV image load, letterbox, RGB conversion, and tensor packing time
- `inference_ms`: backend runtime execution time
- `postprocess_ms`: YOLO decode and NMS time
- `total_ms`: end-to-end time for one image
- `detections`: number of detections after confidence filtering and NMS

The aggregate latency summary is printed to stdout, not written to the CSV:

```text
count,mean_ms,p50_ms,p95_ms,p99_ms,fps
```

## Recommended Issue #51 Workflow

1. Train or reuse a YOLO-Master-EsMoE-N checkpoint on VisDrone or SKU-110K.
2. Export ONNX plus NCNN or MNN.
3. Validate ONNX opset/simplification and NCNN/MNN conversion files.
4. Run the same 500-image validation list through PyTorch and exported backends.
5. Compare mAP50-95 deltas and tensor/output differences.
6. Report latency P50/P95/P99 and FPS per backend/platform.
