# Multi-Task Validation Metrics Implementation Plan

**Goal:** Report genuine box, instance-mask, and OKS pose metrics from one COCO-aligned multi-task validation pass.

**Architecture:** Keep the detector's existing end-to-end candidate selection and NMS, then map retained candidates back
to their source anchors before decoding mask coefficients and keypoints from the matching raw task outputs. A compact
composite metrics object shares box statistics while keeping mask and pose AP evidence in distinct namespaces.

**Tech Stack:** Python, PyTorch, Ultralytics detection validation, COCO-aligned multi-task datasets.

---

### Task 1: Define candidate alignment coverage

- Add focused validator tests with deterministic raw mask and keypoint outputs.
- Verify NMS-retained predictions retain their matching mask and keypoint payloads.

### Task 2: Compose active task metrics

- Initialize mask and pose metrics only when the validated model enables their branch.
- Reuse the existing mask IoU and OKS matching definitions and publish standard `(M)` and `(P)` result namespaces.
- Keep non-metric dense branches represented solely by named validation losses.

### Task 3: Preserve distributed and detection behavior

- Gather image metrics for every active auxiliary metric after shared detection statistics are merged.
- Keep existing detection JSON/export behavior and loss reporting untouched.

### Task 4: Verify

- Run targeted multi-task and P0 gates, MoT/DDP regressions, formatting/lint, diff validation, and the Agent Skill quick suite.

## Stage 2 Execution Record

**Scope:** End-to-end smoke validation on local COCO 2017 using the MPS-friendly MoT multi-task model, with
`detect`, `segment`, `pose`, `classify`, `depth`, and `normal` enabled. This is a functional contract check, not a
benchmark: each split has two real images, and random initialization makes zero mAP expected.

**Data correction:** The original dense smoke YAML used its `train2017` image list for validation while selecting
`val2017` instance/keypoint JSON files. `COCOMultiTaskDataset` now rejects any requested images absent from its
selected annotation JSON, preventing validation from silently dropping every image. The smoke YAML now uses:

- train: `coco2017_multitask_dense_smoke.txt` (two `train2017` images)
- validation: `coco2017_multitask_val_smoke.txt` (two `val2017` images)

The validated batches contained 23 train instances with masks/visible keypoints and 35 validation instances with
masks/visible keypoints.

**Executed commands:**

```bash
python - <<'PY'
from ultralytics import YOLO

model = YOLO("scripts/yolo26-master-mt-dense-mps-local.yaml", task="multitask")
model.train(
    data="scripts/coco2017_multitask_dense_smoke.yaml",
    epochs=1,
    imgsz=128,
    batch=2,
    device="mps",
    workers=0,
    optimizer="AdamW",
    amp=False,
    mosaic=0.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    project="runs/multitask_stage2",
    name="coco2-e1-mps",
    save_period=1,
    val=True,
    plots=False,
)
PY

python - <<'PY'
from ultralytics import YOLO

checkpoint = "runs/multitask_stage2/coco2-e1-mps/weights/last_healthy.pt"
YOLO(checkpoint, task="multitask").train(resume=checkpoint, epochs=2, device="mps", workers=0, plots=False)
PY
```

**Observed outcome:** The first run produced finite non-zero train losses for box (3.93547), segmentation (33.169),
pose (24.1393), image classification (0.68868), depth (0.03613), and normals (1.13714). Its validation row contained
finite non-zero box (4.31983), segmentation (10.1347), pose (23.4246), and classification (0.63686) losses.
Depth/normal validation losses were zero because these two local auxiliary maps do not exist for the chosen `val2017`
images; their validity masks correctly suppress loss computation. The Box, Mask, and Pose metric namespaces all
appeared against the 35 validation instances.

The resume run logged `Resuming ... from epoch 2 to 2 total epochs`, appended only epoch 2, and generated `epoch1.pt`.
Its train and validation losses remained finite, with `results.csv` containing both epochs and all `(B)`, `(M)`, and
`(P)` metric columns. Artifacts are retained under `runs/multitask_stage2/coco2-e1-mps/`.

## Stage 3 Execution Record

**Scope:** Make the COCO `detect`/`segment`/`pose` MoT model deployable through an explicit ONNX contract.

**Export endpoint:** `MultiTaskHead` now emits stable ordered tensors named `detections`, `mask_coefficients`,
`mask_prototypes`, and `keypoints`. The exporter writes the matching `multitask_output_schema` metadata and carries
`kpt_shape` for pose consumers. `nms=True` is deliberately rejected for multi-task export because the generic embedded
NMS wrapper only returns detections and would silently discard aligned mask, pose, and dense tensors.

**MoT compatibility:** Sparse MoT routing remains eager-only. Trace and ONNX paths use the model's dense routing
semantics, preserving runnable deployment behavior without claiming exact sparse dispatch in the exported graph.

**Verification:** A head-level ONNX Runtime round-trip verifies all four tensors against eager mode at `1e-4` absolute
and relative tolerance. A slow whole-model test verifies ONNX checker acceptance, stable output names and metadata,
finite shape-preserving outputs, and numerical agreement for mask prototypes. Detection and keypoint tensors are not
compared position-by-position in the untrained full-model case: equal classification scores make PyTorch and ONNX
choose different but valid Top-K anchor orderings. This ambiguity is isolated from strict numerical coverage by the
head-level test, which uses non-degenerate FPN features.

## Stage 4 Execution Record (P1 MoT ablation and training telemetry)

**Scope:** Close the first real MPS training blocker, run the three-task MoT matrix, and make the distributed
telemetry evidence contract explicit before CUDA/NCCL acceptance.

**Training blocker and fix:** The first MPS optimizer step failed in `ModelEMA.update()` while calling
`source.state_dict()`. `MultiTaskLoss` is an `nn.Module` and its constructor stored `self.model = model`; once the
criterion was lazily attached to the model this registered the parent as a child and created
`model -> criterion -> model`. `state_dict()` then recursed indefinitely. The constructor now stores
`object.__setattr__(self, "model", model)`, preserving `criterion.model` access as a non-owning host reference without
registering a child module. A regression test attaches the criterion after EMA creation and exercises both
`model.state_dict()` and `ema.update(model)`.

**Targeted verification:**

```text
pytest -q tests/test_multitask.py tests/test_training_telemetry.py \
  tests/test_compare_mot_ablation.py tests/test_aggregate_mot_ablation_seeds.py
103 passed, 1 xfailed
```

The CPU/Gloo two-rank telemetry smoke also passed. It verifies rank-local artifacts, equal step counts, and rank-0
aggregation, but it is not CUDA or NCCL evidence.

**Build matrix (imgsz=64, CPU construction):**

| Variant | Parameters | GFLOPs | C2fMoT | Dispatch contract |
|---|---:|---:|---:|---|
| `mt_off` | 6.039456M | 0.973144 | 0 | No MoT modules |
| `mt_mot_dense` | 6.300864M | 0.986549 | 4 | Dense MoT |
| `mt_mot_sparse` | 6.300864M | 0.986549 | 4 | Runtime sparse request |

`mt_off` is a functional control, not an exact parameter/FLOPs-matched control: it has about 4.3% fewer parameters
and 1.4% fewer reported GFLOPs than the MoT variants. Accuracy or latency deltas must not be attributed solely to
MoT without a matched-capacity follow-up.

**Real MPS smoke:** All three variants ran one epoch on the two-image COCO smoke split with `AdamW`, `imgsz=128`,
`batch=2`, `workers=0`, `amp=False`, and mosaic/mixup/cutmix/copy-paste disabled. Each run wrote `results.csv`,
`telemetry.json`, `telemetry_rank_0.json`, `best.pt`, `last.pt`, and `last_healthy.pt`.

| Variant | Train total loss | Step P50 (ms) | Samples/s | MoT dispatch | Expert calls | Memory evidence |
|---|---:|---:|---:|---|---:|---|
| `mt_off` | 78.156600 | 6050.85 | 0.3305 | none | 0 | MPS sampled current, not true peak |
| `mt_mot_dense` | 67.719750 | 7175.55 | 0.2787 | `dense` | 24 | MPS sampled current, not true peak |
| `mt_mot_sparse` | 67.717930 | 2847.34 | 0.7024 | `sample_sparse` | 19 | MPS sampled current, not true peak |

All three single-rank telemetry records report `rank_step_counts_consistent=true`, finite losses, and
`world_size=1`. MPS exposes no public allocator peak equivalent in this runtime, so
`is_true_peak=false`, `peak_device_memory_bytes=null`, and `peak_device_memory_fraction=null` are intentional. The
MPS timing and routing values are smoke evidence only; they are not multi-seed quality claims or CUDA memory claims.

**CUDA/NCCL acceptance commands:** Run these on a CUDA host after copying the repository and COCO subset. The same
three variants and seeds must be completed for each requested world size; use separate output roots per world size.

```bash
# 2 GPU gate
CUDA_VISIBLE_DEVICES=0,1 python scripts/compare_mot_ablation.py \
  --train --models mt_off mt_mot_dense mt_mot_sparse \
  --data scripts/coco2017_multitask_train2000.yaml \
  --seeds 42 123 3407 --epochs 1 --imgsz 128 --batch 2 --workers 0 \
  --device 0,1 --optimizer AdamW --no-amp \
  --mosaic 0 --mixup 0 --cutmix 0 --copy-paste 0 \
  --telemetry-loss-steps 20 --patience 0 --exist-ok \
  --project runs/p1-mot-cuda2

# 8 GPU gate (repeat on a host with eight visible GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python scripts/compare_mot_ablation.py \
  --train --models mt_off mt_mot_dense mt_mot_sparse \
  --data scripts/coco2017_multitask_train2000.yaml \
  --seeds 42 123 3407 --epochs 1 --imgsz 128 --batch 2 --workers 0 \
  --device 0,1,2,3,4,5,6,7 --optimizer AdamW --no-amp \
  --mosaic 0 --mixup 0 --cutmix 0 --copy-paste 0 \
  --telemetry-loss-steps 20 --patience 0 --exist-ok \
  --project runs/p1-mot-cuda8
```

The CUDA/NCCL gate is accepted only when every seed/model has complete artifacts and all of the following hold in
each `telemetry.json`: `rank_step_counts_consistent == true`; the first 20 rank losses have
`rank_loss_relative_spread_max <= 0.05`; `rank_peak_device_memory_fraction_max <= 0.90`; and the sparse run's
manifest request is corroborated by runtime `dispatch_mode` values containing `sample_sparse`. Use
`python scripts/aggregate_mot_ablation_seeds.py --root <run-root> --baseline mt_off --expected-seeds 42 123 3407`
only after all three seeds are present. A three-seed result is the minimum evidence gate, not a claim of statistical
significance.

**Remaining evidence boundary:** This Apple M1 Pro host has zero CUDA devices, so CUDA allocator peaks, NCCL
collectives, multi-rank loss spread, TensorRT export, and three-seed accuracy conclusions remain unverified here.
