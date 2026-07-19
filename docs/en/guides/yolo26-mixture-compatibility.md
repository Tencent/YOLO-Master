---
comments: true
description: Train, infer, checkpoint, and export YOLO26 models with YOLO-Master mixture and adapter methods on the Ultralytics v8.4.101 baseline.
keywords: YOLO26, YOLO-Master, mixture of experts, MoE, MoA, MoT, MoLoRA, LoRA, VPEFT, export, checkpoint
---

# YOLO26 Mixture Compatibility

This branch uses Ultralytics `v8.4.101` as its upstream baseline and ports the existing YOLO-Master mixture and PEFT methods as additive extensions. Official YOLO26 model YAML files and native training, prediction, validation, and export flows remain intact.

## Compatibility Boundary

The integration preserves these YOLO26 graph invariants:

- Detection-style heads use `reg_max=1` and `end2end=True`.
- Detect, Segment26, Pose26, and OBB26 retain both one-to-many and one-to-one branches.
- Specialized heads are excluded from automatic LoRA, MoLoRA, and VPEFT placement unless `include_head=True` is explicitly selected.
- Official files such as `yolo26.yaml`, `yolo26-seg.yaml`, `yolo26-pose.yaml`, `yolo26-obb.yaml`, and `yolo26-sem.yaml` are not replaced by mixture configurations.
- Newly saved checkpoints include additive `mixture_checkpoint` metadata without changing the native checkpoint fields.

The upstream file hashes recorded in `docs/governance/upstream-v8.4.101-manifest.json` provide the integrity boundary for official YOLO26 configs and export/backend modules.

## Validated Model Matrix

| Task | Dense | MoE | MoA | MoT | MoA + MoT | MoLoRA |
|---|---:|---:|---:|---:|---:|---:|
| Detect | Train/infer/export | Train/infer/export | Train/infer/export | Train/infer/export | Train/infer/export | Train/infer/export |
| Instance segment | Train/infer/export | Not configured | Not configured | Not configured | Not configured | Routed smoke/export |
| Pose | Train/infer/export | Not configured | Not configured | Not configured | Not configured | Routed smoke |
| OBB | Train/infer/export | Not configured | Not configured | Not configured | Not configured | Routed smoke |
| Semantic segment | Train/infer/export | Not configured | Not configured | Not configured | Not configured | Routed smoke |

"Not configured" means this branch does not ship a dedicated task YAML for that combination. It is not a claim that the architecture is impossible.

## Detection Configurations

The following configs are additive and keep the official YOLO26 backbone, feature indices, and Detect head contract:

| Method | Configuration |
|---|---|
| Dense | `ultralytics/cfg/models/26/yolo26.yaml` |
| MoE | `ultralytics/cfg/models/26/yolo26-master-n.yaml` |
| MoA | `ultralytics/cfg/models/26/yolo26-master-moa-n.yaml` |
| MoT | `ultralytics/cfg/models/26/yolo26-master-mot-n.yaml` |
| MoA + MoT | `ultralytics/cfg/models/26/yolo26-master-moa-mot-n.yaml` |
| MoLoRA | Official `yolo26.yaml` plus `molora_*` training arguments |

### Train a Routed Architecture

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-master-moa-mot-n.yaml")
model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

The standard CLI works with the same additive config:

```bash
yolo detect train model=ultralytics/cfg/models/26/yolo26-master-moa-mot-n.yaml data=coco8.yaml epochs=100 imgsz=640
```

### Train with MoLoRA

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26.yaml")
model.train(
    data="coco8.yaml",
    epochs=100,
    imgsz=640,
    molora_num_experts=4,
    molora_top_k=2,
    molora_r=8,
    molora_alpha=16,
)
```

When `lora_target_modules` is not supplied, MoLoRA selects compatible Conv2d/Linear layers and records the resolved names in checkpoint adapter metadata. YOLO26 head layers are excluded by default. Set `lora_include_head=True` only when head adaptation is intentional.

### Specialized Tasks

Use the official task YAML and enable MoLoRA through trainer arguments when routed adapter training is required:

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-seg.yaml")
model.train(
    data="coco8-seg.yaml",
    epochs=100,
    imgsz=640,
    molora_num_experts=2,
    molora_top_k=1,
    molora_r=4,
    molora_alpha=8,
)
```

The same pattern applies to `yolo26-pose.yaml`, `yolo26-obb.yaml`, and `yolo26-sem.yaml` with their corresponding datasets.

## Inference and Full Checkpoints

Native checkpoint loading remains the default workflow:

```python
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")
results = model("image.jpg")
```

New full checkpoints include:

```text
mixture_checkpoint.schema_version
mixture_checkpoint.graph.head_class
mixture_checkpoint.graph.reg_max
mixture_checkpoint.graph.end2end
mixture_checkpoint.graph.one2many
mixture_checkpoint.graph.one2one
mixture_checkpoint.graph.routed_families
mixture_checkpoint.adapter
```

These fields support compatibility audits. They do not replace `model`, `ema`, `optimizer`, `train_args`, or other native Ultralytics fields.

### Inspect or Convert a Legacy Artifact

Inspection is read-only:

```python
from ultralytics.utils.checkpoint_compat import inspect_checkpoint_artifact

report = inspect_checkpoint_artifact("legacy-best.pt")
print(report.to_dict())
```

Conversion always writes a different destination and records a machine-readable audit:

```python
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.checkpoint_compat import convert_checkpoint_artifact

target = DetectionModel("ultralytics/cfg/models/26/yolo26.yaml", verbose=False)
report = convert_checkpoint_artifact(
    "legacy-best.pt",
    "converted-yolo26.pt",
    target_model=target,
)
print(report.to_dict())
```

Head class, `reg_max`, end-to-end mode, and branch mismatches are rejected by default. `allow_head_mismatch=True` must be an explicit migration decision, not a routine upgrade flag. Adapter topology mismatches are also rejected; attach a matching adapter graph or merge the source adapter first.

## Adapter Facade

The model facade exposes one backend-neutral lifecycle for standard LoRA and MoLoRA:

```python
model.save_adapters("adapters/run-a")
model.load_adapters("adapters/run-a")
model.merge_adapters(mode="uniform")
```

For standard LoRA compatibility, the existing methods remain available:

```python
model.save_lora_only("adapters/lora-a")
model.load_lora("adapters/lora-a", trainable=True)
model.merge_lora()
```

MoLoRA merge modes are `ema`, `uniform`, and `calibrated`. A merged MoLoRA model is a deployment approximation of its dynamic experts; use a calibration set when routing behavior must be represented more faithfully.

## Export Preflight and Fallback

Run preflight before exporting a routed model:

```python
from ultralytics.utils.export_preflight import export_preflight

report = export_preflight(model.model, "onnx", strict=True)
print(report)
artifact = model.export(format="onnx", imgsz=640, opset=17)
```

PyTorch retains dynamic routing. TorchScript, ONNX, OpenVINO, TensorRT, CoreML, and several downstream formats use the declared dense fallback when both the capability matrix and concrete routed module permit it. A successful fallback export is not an exact preservation of data-dependent sparse dispatch.

The authoritative policy is `ultralytics/cfg/export-capability-matrix.yaml`; its generated report is `docs/governance/export-capability-matrix.md`. Unsupported routed targets are refused before exporter execution. Hardware-specific formats still require validation on their target runtime.

Validated roundtrips in this branch include:

- Detect: dense, MoE, MoA, MoT, MoA + MoT, and MoLoRA with TorchScript and ONNX/AutoBackend.
- Segment26, Pose26, OBB26, and SemanticSegment with TorchScript and ONNX/AutoBackend.
- Segment26 with MoLoRA using ONNX/AutoBackend.

Semantic export output differs by format: TorchScript returns upsampled class logits, while ONNX returns the exported class map. Use the task predictor or AutoBackend instead of assuming that every format exposes the same raw tensor rank.

## Verification Commands

```bash
PYTHONNOUSERSITE=1 python -m pytest -q \
  tests/test_yolo26_mixture_matrix.py \
  tests/test_yolo26_task_matrix.py

PYTHONNOUSERSITE=1 python -m pytest -q --slow -m slow \
  tests/test_yolo26_mixture_matrix.py \
  tests/test_yolo26_task_matrix.py
```

The fast suite covers graph contracts, inference, native losses, routed gradients, and preflight. The slow suite covers full checkpoint reload and official TorchScript/ONNX exporter roundtrips.
