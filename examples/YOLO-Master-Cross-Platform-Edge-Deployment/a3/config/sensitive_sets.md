# A3 quantization configuration (inherited from the project03 INT8 study)

## Calibration policy
- Images: `/data/datasets/coco/images/train2017`, TRAIN split only. Calibrating on the
  val split is leakage and, measured, also worse.
- Count: 1024 (`--calib-n 1024`), even-spread sampling over the sorted file list
  (reproducible without a stored list). Coverage law measured on COCO: 256 -> 1024 -> 2048
  images closes most of the implicit-PTQ gap; beyond 2048 the method floor dominates.
- Preprocess: letterbox to 640x640, gray 114, BGR->RGB, /255 (identical to the runners).
- Implicit TensorRT INT8: `IInt8EntropyCalibrator2` (never MinMax).
- Explicit INT8: modelopt `INT8_DEFAULT_CFG` (per-channel weights, per-tensor activations),
  calibrate-only, then Q/DQ ONNX -> TensorRT `int8-qdq` build (the graph is the precision spec).

## Sensitive set (kept FP16 / not quantized), `SENSITIVE_SETS["default"]` in qat_moe.py
```
*model.0.*   stem conv            (first downsample: ~83% of backbone INT8 damage with model.1)
*model.1.*   first downsample
*routing*    MoE routers          (linear + softmax + top-k: precision-critical decisions)
*router*
*dfl*        distribution-focal-loss decode tail
```
The implicit TensorRT rung additionally pins the detect head subgraph and every router
subgraph to FP16 with `OBEY_PRECISION_CONSTRAINTS` (PREFER silently ignores pins when a
calibrator is present).

## Data yaml
`a3/config/coco-a3.yaml`: `path: /data/datasets/coco`, `train: images/train2017`,
`val: images/val2017`, 80 COCO names. Eval protocol everywhere: ultralytics `.val`,
imgsz 640, batch 1, split val (5000 images).

## Not used, by design
QAT: verified harmful on these MoE CNNs in project03 (3 epochs on frozen scales regressed
the v0.1-N COCO engine by -3.2 AP against its own calibrate-only init of -0.80).
