# 05 - 完整日志

> 提交日期：2026-08-24
> 锁定版本：main @ e9ac08b
> 对应登记表列：完整日志

> 本目录收录 smoke 全流程的关键命令输出（截选最末 20 行 + 关键状态行）。完整 stdout/stderr 在本地 `runs/` 和 yolo CLI 历史中可查。
> 按规则 3.2："截图、口头说明不能单独作证"——所以这里提供**文本日志**（可 grep / diff）而非图片。

---

## 1. preflight 单测

**命令**：
```
python -m pytest tests/test_export_preflight.py -v
```

**输出**：
```
============================= test session starts ==============================
platform win32 -- Python 3.13.5, pytest-9.1.1, pluggy-1.6.0
cachedir: .pytest_cache
rootdir: D:\YOLO_MASTER
configfile: pyproject.toml
collected 3 items

tests/test_export_preflight.py::test_mixture_export_preflight_selects_dense_fallback PASSED [ 33%]
tests/test_export_preflight.py::test_export_preflight_reports_safe_eager_strategy PASSED [ 66%]
tests/test_export_preflight.py::test_exporter_invokes_preflight_before_device_setup PASSED [100%]

============================== slowest 30 durations ===============================
0.08s call  tests/test_export_preflight.py::test_mixture_export_preflight_selects_dense_fallback
0.06s call  tests/test_export_preflight.py::test_exporter_invokes_preflight_before_device_setup
0.02s call  tests/test_export_preflight.py::test_export_preflight_reports_safe_eager_strategy

============================== 3 passed in 1.51s ==============================
```

---

## 2. yolo export 端到端

**命令**：
```
yolo export model=yolo_master_n.pt format=onnx imgsz=640 device=cpu
```

**输出**：
```
ONNX: starting export with onnx 1.22.0 opset 19...
ONNX: slimming with onnxslim 0.1.96...
ONNX: export success 19.1s, saved as 'yolo_master_n.onnx' (10.6 MB)

Export complete (21.9s)
Results saved to D:\YOLO_MASTER\yolo_master_n.onnx
Predict:     yolo predict task=detect model=yolo_master_n.onnx imgsz=640
Validate:    yolo val task=detect model=yolo_master_n.onnx imgsz=640 data=/.../coco.yaml
```

**产物 SHA256**（`sha256sum yolo_master_n.onnx`）：
```
586cf6feeee5bbadddc2cd7cb5852ed49e06f4008e024859904fffc4394c1055  yolo_master_n.onnx
```

---

## 3. ONNX Runtime 端到端推理

**命令**：
```
yolo predict model=yolo_master_n.onnx source=ultralytics/assets/bus.jpg device=cpu
```

**输出**：
```
Ultralytics 8.4.101  Python-3.13.5 torch-2.7.1+cpu CPU (12th Gen Intel Core i5-1240P)
Loading yolo_master_n.onnx for ONNX Runtime inference...
Using ONNX Runtime 1.29.0 with CPUExecutionProvider

image 1/1 D:\YOLO_MASTER\ultralytics\assets\bus.jpg: 640x640 4 persons, 1 bus, 574.9ms
Speed: 80.1ms preprocess, 574.9ms inference, 11.5ms postprocess per image at shape (1, 3, 640, 640)
Results saved to D:\YOLO_MASTER\runs\detect\predict-3
```

> 注：574.9ms 为**单张首帧推理**（含模型加载 + warmup），非稳态吞吐；稳态推理耗时见下方 val 段（ONNX 71.5ms / image）。

**视觉证据**：`06_result_evidence/ort_predict_sample/bus_pred.jpg`（带检测框的 bus.jpg，4 persons + 1 bus）

---

## 4. PyTorch val（COCO8）

**命令**：
```
yolo val model=yolo_master_n.pt data=C:\Users\86133\datasets\coco8\data.yaml device=cpu imgsz=640
```

**输出**：
```
val: Scanning C:\Users\86133\datasets\coco8\labels\val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%

               Class    Images  Instances    Box(P)     R    mAP50   mAP50-95
                 all        4         17      0.803   0.927    0.956      0.743
Speed: 2.7ms preprocess, 233.6ms inference, 0.0ms loss, 1.8ms postprocess per image
Results saved to D:\YOLO_MASTER\runs\detect\val-6

> 注：本段为 CLI 首次跑（val-6）的输出；inference 时间以 `06_result_evidence/val_metrics.json`（Python API 稳态重跑）的 233.6ms 为最终口径。
```

**全 metrics**：`06_result_evidence/val_metrics.json`（PT 块）
**可视化**：`06_result_evidence/val_pt_plots/`（Box PR/F1/P/R + 混淆矩阵 + 批次样本）

---

## 5. ONNX Runtime val（COCO8）

**命令**：
```
yolo val model=yolo_master_n.onnx data=C:\Users\86133\datasets\coco8\data.yaml device=cpu imgsz=640
```

**输出**：
```
val: Scanning C:\Users\86133\datasets\coco8\labels\val.cache... 4 images, 0 backgrounds, 0 corrupt: 100%

               Class    Images  Instances    Box(P)     R    mAP50   mAP50-95
                 all        4         17      0.855   0.891    0.952      0.711
Speed: 2.2ms preprocess, 71.5ms inference, 0.0ms loss, 2.7ms postprocess per image
Results saved to D:\YOLO_MASTER\runs\detect\val-7
```

**全 metrics**：`06_result_evidence/val_metrics.json`（ONNX 块）
**可视化**：`06_result_evidence/val_onnx_plots/`

---

## 6. 差值与判定

```
delta_mAP50-95 = -0.0324   (PT 0.7432 → ORT 0.7108, < 0.05 阈值，导出无损)
delta_mAP50    = -0.0040   (PT 0.9562 → ORT 0.9522, 远 < 0.01 阈值)
speedup_x      =  3.27     (PT 233.6ms → ORT 71.5ms, ORT 加速符合 Intel CPU 典型)
```

**结论**：YOLO-Master EsMoE-N 在 PyTorch → ONNX → ORT CPU 路径上达到 "导出基本无损、部署侧显著加速" 的标准。
