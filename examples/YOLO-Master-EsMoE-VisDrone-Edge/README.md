# YOLO-Master-EsMoE-N · VisDrone edge inference + export-consistency verification

A vertical-domain (VisDrone, UAV small-object, 10 classes) edge-deployment example for
**YOLO-Master-EsMoE-N**: fine-tune → export to **ONNX + MNN + NCNN** → Python & C++ (CMake)
edge inference with domain-tuned preprocessing/NMS → **PyTorch-vs-export mAP50-95
consistency verification** and latency/throughput benchmark.

Targets [Tencent/YOLO-Master#51](https://github.com/Tencent/YOLO-Master/issues/51).

## What this example adds

| Path | Contents |
|------|----------|
| `configs/visdrone.yaml` | VisDrone2019-DET dataset config (absolute path, auto-download). |
| `scripts/train_visdrone.py` | Fine-tune EsMoE-N on VisDrone (COCO pretrained → nc=10). |
| `scripts/recalibrate_bn.py` | BatchNorm running-stat recalibration (fixes the sparse-eval BN drift — see below). |
| `scripts/export_models.py` | Export ONNX (onnxslim+onnxsim) + MNN (+INT8 weight-quant) + NCNN. |
| `scripts/export_ncnn_dense.py` | NCNN via dense (full-softmax) routing — pnnx cannot lower MoE `topk`. |
| `scripts/patch_dense_forward.py` | **Export-consistency patch** for `ES_MOE._dense_forward` (see below). |
| `python/` | Letterbox / area-adaptive NMS / unified ONNX+MNN+NCNN backends / mAP eval / consistency / benchmark. |
| `cpp/` | CMake C++ (ONNXRuntime) edge inference, Linux x86_64 + Windows x86_64 + ARM64 cross. |

## Usage

```bash
# 0. apply the one-line export-consistency patch to a YOLO-Master checkout
python scripts/patch_dense_forward.py path/to/YOLO-Master/ultralytics/nn/modules/moe/modules.py

# 1. fine-tune EsMoE-N on VisDrone
python scripts/train_visdrone.py --epochs 50 --imgsz 640 --batch 16 --device 0

# 2. recalibrate BN (the sparse-eval BN drift) on the resulting checkpoint
python scripts/recalibrate_bn.py --src runs/.../weights/best.pt --dst best_recal.pt --device 0

# 3. export to ONNX + MNN (+NCNN)
python scripts/export_models.py --model best_recal.pt --imgsz 640

# 4. consistency verification (PyTorch vs ONNX vs MNN, 548 val images)
cd python && python eval_consistency.py \
    --pytorch ../best_recal.pt --onnx ../exports/best_recal.onnx --mnn ../exports/best_recal.mnn \
    --imgsz 640 --num-classes 10 --device cuda:0

# 5. C++ edge inference (Linux)
cd ../cpp && bash scripts/setup_ort.sh
cmake -S . -B build -DONNXRUNTIME_ROOT_DIR=$(pwd)/third_party/onnxruntime && cmake --build build -j
./build/yolo_edge ../exports/best_recal.onnx image.jpg --nc 10 --bench 200
```

## Export-consistency patch (`patch_dense_forward.py`) — why it exists

`ES_MOE` has two expert-combine paths:

- `_sparse_forward` (eager eval): computes only the **top-k** experts *and* applies a
  **dynamic-threshold pruning** (`dynamic_threshold=0.4`) — weak experts are dropped.
- `_dense_forward` (forced during ONNX/TorchScript tracing, because the sparse path has
  data-dependent control flow that tracing can't follow): keeps **all** experts.

Because `_sparse_forward`'s pruning is not replicated in `_dense_forward`, the **exported
graph produces different (worse) features than eager eval** → exported mAP collapses to ~0
even though the PyTorch model reaches 6% mAP50-95.

`patch_dense_forward.py` edits `ES_MOE._dense_forward` so that, **during export only**, it
replicates `_sparse_forward`'s pruning with traceable ops (`topk` / `one_hot` / `argmax` /
comparisons). After the patch, the exported ONNX/MNN graph matches eager eval to ~1e-4 and
recovers the full mAP. The training path is unchanged.

> This is the **export-side** counterpart to the **eval-side** `--no-sparse-eval` flag added
> in #81 (which makes *validation* use the dense path). Both stem from the same sparse/dense
> mismatch; this patch makes the *exported* model consistent with the model's mAP.
>
> **No core library change is required to run the example** — `patch_dense_forward.py` is a
> standalone script the user runs against their YOLO-Master checkout. (A maintainer may
> prefer to fold it into `ultralytics/` directly; the patch is intentionally minimal.)

## Results (VisDrone val, 548 images, imgsz=640, EsMoE-N fine-tuned)

Consistency (issue #51 target ΔmAP50-95 < 0.5%):

| Backend | mAP50 | mAP50-95 | ΔmAP50-95 vs PyTorch |
|---------|-------|----------|----------------------|
| PyTorch (native) | 12.09% | 6.00% | — |
| ONNX (opset 17) | 12.08% | 5.99% | **−0.004%** ✅ |
| MNN | 12.08% | 5.99% | **−0.006%** ✅ |
| NCNN (dense) | — | ~0% | — | (MoE pruning can't lower to NCNN — see export_ncnn_dense.py) |

Edge latency / throughput (same image → identical 33 detections across platforms):

| Backend / platform | mean (ms) | FPS |
|--------------------|-----------|-----|
| C++ ONNXRuntime — Linux x86_64 | 53.0 | **18.9** |
| C++ ONNXRuntime — Windows x86_64 (MSVC 19.44) | 83.6 | **12.0** |
| MNN (Python, CPU) | 56.1 | 17.8 |
| NCNN (Python, CPU) | 100.0 | 10.0 |

## Notes

- **NCNN**: pnnx cannot lower the MoE routing's `topk` or comparison ops (they become
  no-op layers NCNN refuses to register), so the pruned path is not representable in NCNN.
  `export_ncnn_dense.py` exports a dense (full-softmax, no-topk) graph that NCNN can load
  and run, at the cost of accuracy (mAP ~0). ONNX/MNN are the recommended edge paths and
  both meet the <0.5% consistency target with non-zero mAP.
- Full artifacts (checkpoints, exported models, training/eval logs):
  https://github.com/Naloam/YOLO-Master-EsMoE-VisDrone-Edge (release v1.0).
