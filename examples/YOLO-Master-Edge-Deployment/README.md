# YOLO-Master Edge Deployment Example

This example supports issue #51: vertical-model edge inference acceleration and consistency validation.

It provides a lightweight, reproducible scaffold for exporting YOLO-Master models to ONNX plus NCNN/MNN, running vertical-domain preprocessing, comparing backend outputs, and summarizing edge benchmark logs.

## Files

- `edge_utils.py` - shared preprocessing, postprocessing, consistency, and benchmark utilities.
- `export_edge_models.py` - export helper for ONNX, NCNN, and MNN.
- `validate_edge_outputs.py` - compare PyTorch/exported backend outputs saved as `.npy` tensors.
- `cpp/edge_benchmark_stub.cpp` - minimal C++ benchmark entry that can be extended with ONNX Runtime, NCNN, or MNN runtime calls.
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

## CMake Smoke Build

```bash
cmake -S . -B build
cmake --build build
./build/yolo_master_edge_benchmark --backend onnx --model best.onnx --images val.txt
```

The C++ file is intentionally dependency-light. It establishes the CLI contract and build target, so backend-specific runtime calls can be added without changing benchmark automation.

## Recommended Issue #51 Workflow

1. Train or reuse a YOLO-Master-EsMoE-N checkpoint on VisDrone or SKU-110K.
2. Export ONNX plus NCNN or MNN.
3. Validate ONNX opset/simplification and NCNN/MNN conversion files.
4. Run the same 500-image validation list through PyTorch and exported backends.
5. Compare mAP50-95 deltas and tensor/output differences.
6. Report latency P50/P95/P99 and FPS per backend/platform.
