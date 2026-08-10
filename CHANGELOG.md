# Changelog

All notable changes to YOLO-Master are documented in this file.

---

## [YOLO-Master-v26.08] — 2026-08-07

<div align="center">
  <img width="260" height="260" alt="YOLO-Master Logo" src="https://github.com/user-attachments/assets/847ce41b-7282-4e98-b8be-240a572dd87a" />

  <h1>YOLO-Master v2026.08</h1>
  <p><strong>Ultralytics 8.4.101 · YOLO26 · Mixture Architectures · PEFT · MultiTask · Edge Runtime</strong></p>

  [![Release](https://img.shields.io/badge/release-v26.08-7c3aed.svg)](https://github.com/Tencent/YOLO-Master/releases)
  [![Validation](https://img.shields.io/badge/focused_gate-377%20passed%2C%201%20xfailed-16a34a.svg)](#-validation)
  [![Ultralytics](https://img.shields.io/badge/Ultralytics-8.4.101-111827.svg)](#-upstream-modernization)
  [![Python](https://img.shields.io/badge/Python-3.8+-2563eb.svg)](https://www.python.org/)
  [![PyTorch](https://img.shields.io/badge/PyTorch-1.8+-ee4c2c.svg)](https://pytorch.org/)
  [![License](https://img.shields.io/badge/License-AGPL--3.0-0f766e.svg)](https://github.com/Tencent/YOLO-Master/blob/main/LICENSE)
</div>

> [!IMPORTANT]
> v26.08 is a cumulative release from `YOLO-Master-v26.02`. It upgrades the upstream Ultralytics baseline from `8.3.240` to `8.4.101` while preserving YOLO-Master's mixture and PEFT extensions. Published metrics below are limited to results stored in the repository's model catalog or deployment reports.

<p align="center">
  <img width="100%" alt="YOLO-Master v2026.08 release overview: MultiTask, Shared Expert MoE, MoA and MoT, PEFT Planner, MoLoRA, Latent Mixture, and edge deployment" src="https://github.com/user-attachments/assets/0d3b951b-bc83-4a81-9865-9fb40370a912" />
</p>

### Release at a glance

| | |
|---|---|
| **Release range** | `YOLO-Master-v26.02...YOLO-Master-v26.08` |
| **Audited history** | 622 commits · 96 merged PRs · 38 author identities, including bots and aliases |
| **Upstream upgrade** | Ultralytics `8.3.240` → `8.4.101` |
| **Native model family** | YOLO26 detect · segment · semantic · pose · OBB · classify · YOLOE |
| **Focused release gate** | 377 passed · 1 xfailed (mixture/P0-P2 scope) |
| **Model catalog** | 7 evaluated checkpoints · 3 pending/evaluating variants |
| **License** | AGPL-3.0 |

### Why upgrade

- **New upstream foundation:** move from Ultralytics `8.3.240` to `8.4.101` and gain native YOLO26 task, checkpoint, and export contracts.
- **One routing stack:** use a shared protocol for MoE, MoA, MoT, Latent Mixture, and MoLoRA auxiliary losses, temperature scheduling, diagnostics, and export capability reporting.
- **More adaptation paths:** choose fixed-rank LoRA, the architecture-conditioned PEFT Planner, FewShot-LoRA, or routed MoLoRA adapters.
- **Deployment beyond Python:** package models for Windows, Linux, Jetson, and macOS through ONNX Runtime, NCNN, MNN, TensorRT, and Core ML workflows.

[What's new](#-whats-new) · [Choose a model](#choose-a-v2608-architecture) · [Upstream upgrade](#-upstream-modernization) · [Model Zoo](#-model-zoo) · [Validation](#-validation) · [Migration](#-migration-guide) · [Development diff](https://github.com/Tencent/YOLO-Master/compare/YOLO-Master-v26.02...main)

> **Canonical release notes:** [docs/release-notes/v26.08.md](docs/release-notes/v26.08.md) contains the current P0-P2 hardening summary, validation evidence, migration steps, and known limitations.

---

## ✨ What's new

| Area | Release status | What v26.08 adds |
|---|---|---|
| **Ultralytics 8.4.101 / YOLO26** | **Stable upstream base** | Native task flows, checkpoint compatibility, export integrity, and additive mixture registration |
| **MultiTask** | **Preview** | Unified detection, segmentation, pose, classification, depth, normal, and semantic branches with optional task routing; OBB uses its dedicated task model |
| **Shared Expert MoE** | **Validated component** | Model-scoped expert-pool reuse with cross-model isolation |
| **MoA / MoT** | **Experimental profiles** | Routed attention and transformer blocks, sparse paths, scene-aware routing, and shared temperature scheduling |
| **PEFT Planner / LOVO** | **Opt-in** | Architecture-conditioned placement, V-PEFT solvers, validation, and FewShot-LoRA controls |
| **MoLoRA** | **Opt-in** | Sparse routing over low-rank adapter experts with routing-aware merge contracts |
| **Latent Mixture** | **Experimental profiles** | Dense latent routing, configurable initialization/noise, auxiliary losses, and inference top-k |
| **Edge Runtime** | **Platform-validated** | Windows GUI, ONNX Runtime, NCNN, MNN, Jetson TensorRT, and macOS Core ML workflows |
| **Reliability** | **Release gate** | NaN recovery, DDP checkpoint hardening, EMA synchronization, AMP-safe sparse dispatch, and release audits |

> [!TIP]
> **Status vocabulary:** _Stable upstream base_ follows the project model registry; _validated component_ means focused build/forward/contract tests pass; _preview_ and _experimental profiles_ expose working interfaces without a release-level accuracy claim; _opt-in_ means disabled by default.

---

## 🚀 Quick start

### Install from source

```bash
git clone https://github.com/Tencent/YOLO-Master.git
cd YOLO-Master
pip install -e .
yolo version
yolo checks
```

### Train a standard YOLO-Master model

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26-master-n.yaml")
model.train(data="coco8.yaml", epochs=100, imgsz=640)
```

### Choose a v26.08 architecture

| If you need… | Start with | Status |
|---|---|---|
| Native YOLO26 behavior | `ultralytics/cfg/models/26/yolo26.yaml` | Stable upstream profile |
| YOLO26 with MoE blocks | `ultralytics/cfg/models/26/yolo26-master-n.yaml` | Experimental; component export verified |
| Joint task training | `ultralytics/cfg/models/26/yolo26-master-mt-n.yaml` | Preview; training/validation focused |
| Shared expert parameters | `ultralytics/cfg/models/master/v0_8/det/yolo-master-moe-mot-shared-n.yaml` | Build/forward/reuse verified |
| Combined attention + transformer routing | `ultralytics/cfg/models/master/v0_10/det/yolo-master-moa-mot-n.yaml` | Experimental; component export/compile verified |
| Latent-space routing | `ultralytics/cfg/models/26/yolo26-master-latent-n.yaml` | Experimental; focused tests verified |
| Routed low-rank adapters | Native `yolo26.yaml` + `molora_*` arguments | Opt-in |

```python
from ultralytics import YOLO

multitask = YOLO(
    "ultralytics/cfg/models/26/yolo26-master-mt-n.yaml",
    task="multitask",
)

shared_moe = YOLO(
    "ultralytics/cfg/models/master/v0_8/det/"
    "yolo-master-moe-mot-shared-n.yaml"
)

moa_mot = YOLO(
    "ultralytics/cfg/models/master/v0_10/det/"
    "yolo-master-moa-mot-n.yaml"
)

latent = YOLO(
    "ultralytics/cfg/models/26/yolo26-master-latent-n.yaml"
)
```

### WebUI preview

The repository-level WebUI provides task selection, image/batch/video/webcam inputs, inference controls, result tables, and Agent-oriented workflows in one interface.

<p align="center">
  <img width="100%" alt="YOLO-Master WebUI showing single-image object detection, confidence controls, detections, and inference results" src="https://github.com/user-attachments/assets/282bcece-8d88-4157-b3f5-390cc25aa24b" />
</p>

> [!NOTE]
> Shared Expert MoE, MoA/MoT, Latent Mixture, and MultiTask are selected through model YAML files. They are not enabled by undocumented flags such as `moe_shared_expert=True` or `latent_init_perturb=True`.

---

## 🔄 Upstream modernization

v26.02 reported Ultralytics `8.3.240`. v26.08 rebases YOLO-Master on **Ultralytics `8.4.101`** and ports the existing MoE, MoA, MoT, MoLoRA, V-PEFT, and Agent integrations as additive extensions rather than replacing upstream task implementations.

| Compatibility boundary | v26.08 behavior |
|---|---|
| **Official YOLO26 configs** | Native detect, segment, pose, OBB, semantic, classification, and YOLOE YAML files remain available |
| **End-to-end heads** | YOLO26 detection-style heads retain `reg_max=1`, `end2end=True`, and one-to-many/one-to-one branches |
| **Train / predict / val / export** | Native Ultralytics flows remain intact for official YOLO26 models |
| **Mixture models** | Registered as additive YAML profiles instead of overwriting official files |
| **PEFT targeting** | Specialized heads remain excluded unless `lora_include_head=True` is explicitly selected |
| **Checkpoints** | Native fields are preserved; mixture metadata is added without replacing the upstream schema |
| **Integrity boundary** | Official config and backend hashes are recorded in `docs/governance/upstream-v8.4.101-manifest.json` |

The repository includes a deterministic native baseline for eight upstream task configurations—detect, segment, semantic segment, pose, OBB, classification, YOLOE detect, and YOLOE segment—in `reports/migration/v8.4.101-native-baseline.json`.

| Native task | Reference config | Baseline evidence |
|---|---|---|
| Detect | `ultralytics/cfg/models/26/yolo26.yaml` | Build + finite forward |
| Instance segment | `ultralytics/cfg/models/26/yolo26-seg.yaml` | Build + finite forward |
| Semantic segment | `ultralytics/cfg/models/26/yolo26-sem.yaml` | Build + finite forward |
| Pose | `ultralytics/cfg/models/26/yolo26-pose.yaml` | Build + finite forward |
| OBB | `ultralytics/cfg/models/26/yolo26-obb.yaml` | Build + finite forward |
| Classification | `ultralytics/cfg/models/26/yolo26-cls.yaml` | Build + finite forward |
| YOLOE detect | `ultralytics/cfg/models/26/yoloe-26.yaml` | Prompt-aware build + finite forward |
| YOLOE segment | `ultralytics/cfg/models/26/yoloe-26-seg.yaml` | Prompt-aware build + finite forward |

```bash
# Discover additive mixture profiles without replacing native YOLO26 configs.
yolo mixtures kind=mot task=detect
yolo mixtures kind=latent format=json
```

**Migration evidence:** `docs/en/guides/yolo26-mixture-compatibility.md` · `docs/governance/upstream-v8.4.101-manifest.json` · `reports/migration/v8.4.101-native-baseline.json`

---

## 🧩 Architecture highlights

<details open>
<summary><strong>1. MultiTask Learning — one feature hierarchy, configurable task branches</strong></summary>

`MultiTaskHead` combines a shared backbone and neck with task-specific branches. `TaskRouter` is optional and performs content-based spatial-token routing between task-specific and shared features.

```text
                     Shared Backbone + Neck
                               │
                      Optional TaskRouter
                               │
        ┌──────────┬─────────┬──────┬──────────┬───────┬────────┬──────────┐
        │ Detect   │ Segment │ Pose │ Classify │ Depth │ Normal │ Semantic │
        └──────────┴─────────┴──────┴──────────┴───────┴────────┴──────────┘
```

| Component | Role |
|---|---|
| `MultiTaskHead` | Builds branches enabled by the model and dataset task lists |
| `TaskRouter` | Routes spatial tokens to task and shared feature channels |
| `MultiTaskLoss` | Combines losses only for tasks with valid supervision |
| Unified data path | Preserves missing supervision instead of converting it to negative targets |

```bash
yolo multitask train \
  model=ultralytics/cfg/models/26/yolo26-master-mt-n.yaml \
  data=ultralytics/cfg/datasets/coco8-multitask.yaml \
  epochs=100 \
  imgsz=640
```

> [!NOTE]
> The current unified COCO pipeline has aligned trainable labels for detection, instance segmentation, and human pose. Classification, depth, normal, and semantic branches require suitable aligned labels before they contribute a training loss. Unified MultiTask OBB training is unsupported; use the dedicated OBB model. The current `multitask` prediction map uses `DetectionPredictor`; a public `tasks=[...]` multi-output inference API is not documented in this release.

**Implementation:** `ultralytics/nn/modules/multitask/` · `ultralytics/models/yolo/multitask/` · `ultralytics/nn/tasks.py`

</details>

<details open>
<summary><strong>2. Shared Expert MoE — model-scoped parameter reuse</strong></summary>

`SharedExpertMoE` uses `pool_id` to share one `fused_experts` module across compatible blocks built as part of the same model. Model parsing clears the temporary registry at model boundaries, so separately constructed models do not share parameters or devices.

```yaml
backbone:
  - [-1, 1, SharedExpertMoE, [512, 4, 2, 0.5, 8, 1.2, 0.5, 1.0, 1.0, 0.01, 8, 2, 0.5, "p3_p4"]]
  - [-1, 1, SharedExpertMoE, [512, 4, 2, 0.5, 8, 1.2, 0.5, 1.0, 1.0, 0.01, 8, 2, 0.5, "p3_p4"]]
```

The v0.8 shared model now uses the current `C2fMoT` argument order (`num_heads=8`, `top_k=2`). Regression tests verify model construction, a minimal forward pass, in-model object reuse, and cross-model isolation.

> [!NOTE]
> Source comments estimate a 25–50% reduction in expert parameters when sharing applies. v26.08 does not publish that estimate as a measured end-to-end model result.

**Implementation:** `ultralytics/nn/modules/moe/shared_expert_moe.py` · `ultralytics/nn/tasks.py` · `ultralytics/cfg/models/master/v0_8/det/yolo-master-moe-mot-shared-n.yaml`

</details>

<details>
<summary><strong>3. MoA and MoT — routed attention and transformer experts</strong></summary>

### Mixture of Attention

MoA combines local, regional, and global attention paths behind a router.

| Setting | Purpose |
|---|---|
| `moa_local_window_size` | Local attention window size |
| `moa_regional_max_kv_tokens` | Regional key/value token cap |
| `moa_sparse_inference` | Skip low-weight head groups during evaluation |
| `moa_sparse_inference_threshold` | Sparse-evaluation threshold |

### Mixture of Transformers

MoT routes spatial tokens through transformer-style experts.

| Setting | Purpose |
|---|---|
| `mot_balance_loss` | GShard balance-loss coefficient |
| `mot_router_z_loss` | Router z-loss coefficient |
| `mot_sparse_train` | Sparse expert dispatch during training |
| `mot_scene_aware_router` | Experimental scene-aware routing branch |

```python
from ultralytics import YOLO

model = YOLO(
    "ultralytics/cfg/models/master/v0_10/det/"
    "yolo-master-moa-mot-n.yaml"
)
model.train(
    data="coco8.yaml",
    epochs=100,
    moa_sparse_inference=False,
    mot_balance_loss=0.01,
    moa_mot_temperature_factor=0.97,
    moa_mot_min_temperature=0.3,
)
```

**Implementation:** `ultralytics/nn/modules/moa/` · `ultralytics/nn/modules/mot/` · `ultralytics/nn/modules/routing_protocol.py`

</details>

<details>
<summary><strong>4. PEFT Planner, LOVO, and FewShot-LoRA</strong></summary>

`PEFTPlanner` evaluates model structure and returns an `ACCEPT`, `ADAPT`, or `REFUSE` placement decision. The V-PEFT backend provides `ao`, `dco`, and `mip` budget solvers.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(
    data="coco8.yaml",
    epochs=100,
    lora_r=16,
    lora_planner_enabled=True,
    lora_adapter_budget=500_000,
    lora_planner_solver="ao",
    lora_planner_backend="vpeft",
)
```

LOVO is a Python API rather than a `yolo lora lovo` CLI command:

```python
from ultralytics.utils.lora import LOVODataCollector, LOVOValidator

collector = LOVODataCollector.load("reports/lovo_data.json")
result = LOVOValidator().validate(collector)
result.save("reports/lovo_validation.json")
```

FewShot-LoRA adds scheduled DropConnect, optional teacher distillation, and variational rank selection:

```python
model.train(
    data="fewshot.yaml",
    epochs=200,
    lora_r=16,
    lora_few_shot_mode=True,
    lora_few_shot_dropconnect_schedule="cosine",
    lora_few_shot_dropconnect_max=0.3,
    lora_few_shot_distill_weight=0.5,
    lora_few_shot_variational_rank=True,
)
```

**Implementation:** `ultralytics/utils/lora/planner.py` · `ultralytics/vpeft/graph.py` · `ultralytics/utils/lora/fallback.py`

</details>

<details>
<summary><strong>5. MoLoRA — routed low-rank adapter experts</strong></summary>

MoLoRA adds sparse routing over multiple low-rank adapter experts. It includes balance, z, and diversity losses; expert dropout and warmup; optional domain mappings; expert freezing; and routing-aware merge contracts.

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")
model.train(
    data="coco8.yaml",
    epochs=100,
    molora_num_experts=4,
    molora_top_k=2,
    molora_r=8,
    molora_alpha=16,
    molora_router_type="linear",
)
```

`molora_num_experts=0` disables MoLoRA. A positive standard `lora_r` request and `molora_num_experts>0` cannot be used together.

**Implementation:** `ultralytics/nn/peft/molora/` · `ultralytics/engine/extensions/adapters.py`

</details>

<details>
<summary><strong>6. Latent Mixture — dense latent-space routing</strong></summary>

`LatentMixture` projects one or more feature maps into a shared latent space, routes them through channel experts, and publishes balance and router z-loss terms through the common routing protocol.

```python
from ultralytics import YOLO

model = YOLO(
    "ultralytics/cfg/models/26/"
    "yolo26-master-latent-n-initperturb020-temp05.yaml"
)
model.train(
    data="coco8.yaml",
    epochs=100,
    latent_inference_top_k=2,
    moa_mot_temperature_factor=0.97,
    moa_mot_min_temperature=0.3,
)
```

The selected YAML uses `router_init_std=0.02` and `temperature=0.5`. The shared mixture extension applies a multiplicative temperature schedule with a configured floor. Ablation configs under `ultralytics/cfg/models/26/` cover router initialization, temperature, noise, and residual initialization.

**Implementation:** `ultralytics/nn/modules/latent_mixture.py` · `ultralytics/nn/modules/routing_protocol.py` · `ultralytics/engine/extensions/mixture.py`

</details>

---

## 🖥️ Edge deployment

The cross-platform example combines a shared C++ runtime with platform-specific applications and packaging workflows.

| Backend | Repository integration | Primary targets |
|---|---|---|
| **ONNX Runtime** | C++ backend + Windows GUI | Linux and Windows, CPU/CUDA |
| **NCNN** | C++ backend + Windows GUI | x86 and ARM, Vulkan where available |
| **MNN** | C++ backend + Windows GUI | x86 and ARM, OpenCL where available |
| **TensorRT** | Native C++ backend + Jetson scripts | NVIDIA GPU and Jetson Orin |
| **Core ML** | Export scripts + Swift application | macOS on Apple Silicon and Intel |

### Windows GUI

The Windows 10/11 application uses Dear ImGui and Direct3D 11. It supports image, folder, video, and webcam input; segmentation overlays; backend switching; and live confidence/IoU controls.

<p align="center">
  <img width="100%" alt="YOLO-Master Windows Runner using the ONNX Runtime CUDA backend for dense aerial vehicle detection" src="https://github.com/user-attachments/assets/187e04da-9abd-4d83-aab7-f5c48a89fd8c" />
</p>

The cross-platform edge and reproduction work includes merged contributions for the C++ ONNX/NCNN/MNN runtime ([#97](https://github.com/Tencent/YOLO-Master/pull/97)), Jetson TensorRT deployment ([#105](https://github.com/Tencent/YOLO-Master/pull/105)), the macOS Core ML runner ([#134](https://github.com/Tencent/YOLO-Master/pull/134)), and the Windows GUI ([#176](https://github.com/Tencent/YOLO-Master/pull/176)). [View all PRs by `skywalker-lt`](https://github.com/Tencent/YOLO-Master/pulls?q=is%3Apr+author%3Askywalker-lt+).

```powershell
cd examples/YOLO-Master-Cross-Platform-Edge-Deployment/gui
./build.ps1
./build.ps1 -Run
```

The build requires Visual Studio 2022, CMake 3.16 or newer, and at least one configured inference backend. Packaging copies required runtime DLLs beside `yolomaster_gui.exe`.

### Verified Jetson result

| Device | Backend | Precision | Dataset/model scope | Latency | FPS | mAP50-95 |
|---|---|---:|---|---:|---:|---:|
| Jetson Orin Nano 4 GB | TensorRT | FP16 | Documented VisDrone model, 548-image validation | 27.8 ms | 35.7 | 0.2029 |

The corresponding PyTorch FP32 baseline is `0.2036` mAP50-95. These values apply only to the model, data, and device documented in `examples/YOLO-Master-Cross-Platform-Edge-Deployment/jetson/DEPLOYMENT_LOG.md`.

> [!NOTE]
> `cpu_threads` and `cpu_affinity` are not Python `YOLO.predict()` arguments in v26.08.

---

## 🛡️ Reliability and recovery

| Area | v26.08 change | Verification scope |
|---|---|---|
| **NaN handling** | Pre-batch checks, component guards, and recovery paths | Routed training and recovery regression tests |
| **AMP safety** | Dtype-aligned sparse accumulation | MoE/MoA/MoT mixed-precision tests |
| **DDP lifecycle** | Bootstrap and pre-epoch checkpoint coordination | DDP lifecycle and static-graph tests |
| **EMA** | Buffer and PEFT scaling synchronization | Checkpoint/EMA regression tests |
| **MPS** | Native bilinear `grid_sample` path and numerical fixes | Apple Silicon regression tests |
| **Export** | Routed capability matrix and pruning metadata | Export and pruning contract tests |
| **Upstream integrity** | Ultralytics `8.4.101` file manifest, native baseline, and additive registry | Integrity, checkpoint, and model-registry tests |
| **Agent runtime** | Profile manifests, release audits, and structured fallback | Agent quick/contract validation suites |

### Selected critical fixes

| PR | Fix |
|---:|---|
| [#74](https://github.com/Tencent/YOLO-Master/pull/74) | ONNX export compatibility for MoE expert loss |
| [#116](https://github.com/Tencent/YOLO-Master/pull/116) | P0/P1/P2 fixes across MoE, MoA, MoT, and PEFT |
| [#124](https://github.com/Tencent/YOLO-Master/pull/124) | AMP dtype alignment for sparse `index_add_` paths |
| [#127](https://github.com/Tencent/YOLO-Master/pull/127), [#140](https://github.com/Tencent/YOLO-Master/pull/140) | DDP static-graph and checkpoint coordination |
| [#158](https://github.com/Tencent/YOLO-Master/pull/158) | Released router checkpoint compatibility |
| [#161](https://github.com/Tencent/YOLO-Master/pull/161) | YOLOE released-checkpoint execution semantics |
| [#162 (merge)](https://github.com/Tencent/YOLO-Master/commit/9a93a786d2c3a35af506e2bc8121b07f5dd00586) | GitHub Actions pinned to commit SHAs |
| [#177](https://github.com/Tencent/YOLO-Master/pull/177) | LoRA alpha warmup across the EMA lifecycle |
| [#188](https://github.com/Tencent/YOLO-Master/pull/188) | Routing dataset statistics weighted by sample count |
| [#192](https://github.com/Tencent/YOLO-Master/pull/192), [#194](https://github.com/Tencent/YOLO-Master/pull/194) | Pruned expert architecture preserved for retraining |
| [#211](https://github.com/Tencent/YOLO-Master/pull/211) | PEFT scaling state synchronized to EMA |

---

## 📦 Model Zoo

### YOLO-Master-EsMoE

| Model | Params | GFLOPs | mAP50-95 | FPS¹ | Assets |
|---|---:|---:|---:|---:|---|
| **EsMoE-N** | 2.68M | 8.7 | 0.427 | 640.18 | [Weights](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0/resolve/main/YOLO-Master-EsMoE-N/YOLO-Master-EsMoE-N.pt?download=true) · [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml) |
| **EsMoE-S** | 9.69M | 29.1 | 0.489 | 423.87 | [Weights](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0/resolve/main/YOLO-Master-EsMoE-S/YOLO-Master-EsMoE-S.pt?download=true) · [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-s.yaml) |
| **EsMoE-M** | 34.88M | 97.4 | 0.530 | 243.79 | [Weights](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0/resolve/main/YOLO-Master-EsMoE-M/YOLO-Master-EsMoE-M.pt?download=true) · [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-m.yaml) |
| **EsMoE-L** | _evaluating_ | — | — | — | [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-l.yaml) |
| **EsMoE-X** | _pending_ | — | — | — | [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_10/det/yolo-master-x.yaml) |

### YOLO-Master-v0.1

| Model | Params | GFLOPs | mAP50-95 | FPS¹ | Assets |
|---|---:|---:|---:|---:|---|
| **v0.1-N** | 7.54M | 10.1 | 0.429 | 528.84 | [Weights](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-N/YOLO-Master-v0.1-N.pt?download=true) · [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml) |
| **v0.1-S** | 29.15M | 36.0 | 0.489 | 345.24 | [Weights](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-S/YOLO-Master-v0.1-S.pt?download=true) · [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_1/det/yolo-master-s.yaml) |
| **v0.1-M** | 52.17M | 116.7 | 0.528 | 170.72 | [Weights](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-M/YOLO-Master-v0.1-M.pt?download=true) · [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_1/det/yolo-master-m.yaml) |
| **v0.1-L** | 58.41M | 138.1 | 0.539 | 149.86 | [Weights](https://huggingface.co/gatilin/YOLO-Master-ckpts-v0_1/resolve/main/YOLO-Master-v0.1-L/YOLO-Master-v0.1-L.pt?download=true) · [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_1/det/yolo-master-l.yaml) |
| **v0.1-X** | _evaluating_ | — | — | — | [YAML](https://github.com/Tencent/YOLO-Master/blob/main/ultralytics/cfg/models/master/v0_1/det/yolo-master-x.yaml) |

<details>
<summary><strong>Full precision, recall, and mAP50 metrics</strong></summary>

| Model | P | R | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| EsMoE-N | 0.684 | 0.536 | 0.587 | 0.427 |
| EsMoE-S | 0.699 | 0.603 | 0.603 | 0.489 |
| EsMoE-M | 0.737 | 0.640 | 0.697 | 0.530 |
| v0.1-N | 0.684 | 0.542 | 0.592 | 0.429 |
| v0.1-S | 0.724 | 0.607 | 0.662 | 0.489 |
| v0.1-M | 0.729 | 0.641 | 0.696 | 0.528 |
| v0.1-L | 0.739 | 0.646 | 0.705 | 0.539 |

</details>

<sub>¹ FPS values are the RTX 4090 results recorded in `model-zoo/models.json` (updated 2026-07-22). Dataset and evaluation fields follow that catalog. L/X results and PEFT efficiency figures are not published because matching evaluated artifacts are not available.</sub>

---

## ✅ Validation

The release-focused gate covers the corrected Shared Expert path, routed architecture contracts, and the P0-P2 hardening surface. The command below is the historical architecture subset; the current focused totals and system gates are listed in the table.

```bash
pytest \
  tests/test_moe.py \
  tests/test_moa.py \
  tests/test_mot.py \
  tests/test_mixture_aux_loss.py \
  tests/test_routing_aux_contract.py \
  tests/test_master_model_configs.py \
  tests/test_default_config_integrity.py \
  tests/test_mixture_catalog.py \
  tests/test_upstream_integrity.py \
  tests/test_checkpoint_compat.py \
  tests/test_mixture_model_registry.py -q
```

| Gate | Result |
|---|---:|
| MoE, MoA, and MoT modules | Passed |
| Routed auxiliary-loss contracts | Passed |
| Shared Expert build, forward, reuse, and isolation | Passed |
| Master model configuration regression | Passed |
| Default configuration integrity | Passed |
| Mixture catalog integrity | Passed |
| Ultralytics `8.4.101` upstream integrity | Passed |
| Checkpoint conversion and compatibility | Passed |
| Additive mixture model registry | Passed |
| **Mixture/P0-P2 focused gate** | **377 passed · 1 xfailed** |
| **MultiTask + Latent + P0 system gates** | **107 passed · 1 xfailed** |
| **Agent Skill quick suite** | **36/36 passed** |

Static release-note checks also verify referenced repository paths, Python syntax, registered configuration keys, model catalog parity, and patch whitespace.

> [!NOTE]
> This is a **targeted release gate**, not the repository's entire test suite. It covers the features and compatibility claims promoted in these notes.

The current focused result supersedes the earlier 230-test draft count. The `xfailed` case is retained as an explicit expected failure; it is not counted as a pass.

### Release boundaries

- Routed model profiles remain experimental unless the model registry marks them stable.
- MultiTask training and validation are implemented; task-specific multi-output prediction beyond the detection predictor remains preview functionality.
- MoA/MoT, Latent Mixture, MoLoRA, and FewShot-LoRA do not have release-level accuracy or latency tables because matching evaluated artifacts are not stored in the repository.
- TensorRT export remains unverified for the routed profiles listed in `docs/governance/model-registry.yaml`; component-level ONNX round trips do not imply full-model TensorRT validation.
- EsMoE-L, EsMoE-X, and v0.1-X remain pending or under evaluation.

---

## 🔄 Migration guide

### From v26.02 to v26.08

> [!IMPORTANT]
> **Breaking changes:** none are formally declared for v26.08. Existing fixed-rank LoRA calls and the documented `sparse_sahi`, `lora_auto_r_ratio`, and `moe_balance_loss` settings remain registered. Custom code that imports Ultralytics internals should still be retested against the new `8.4.101` baseline.

#### Upstream baseline: `8.3.240` → `8.4.101`

The main migration is an upstream Ultralytics upgrade, not only a feature addition. Existing YOLO-Master mixture and adapter methods were ported onto the `8.4.101` parser, trainer, checkpoint, task-head, and export contracts.

- Use the packaged YOLO26 YAML files for the new native model family.
- Keep mixture architectures additive; do not replace official `yolo26*.yaml` files.
- Preserve native checkpoint fields when converting old artifacts; the project adds `mixture_checkpoint` metadata separately.
- Revalidate custom integrations that depended directly on the old `8.3.240` parser, trainer lifecycle, or internal head signatures.
- Use `tools/migration/check_upstream_integrity.py` and `tests/test_upstream_integrity.py` when rebasing further upstream changes.

```bash
# Confirm that the active checkout, import path, and CLI resolve to v8.4.101.
python -c "import ultralytics; print(ultralytics.__version__, ultralytics.__file__)"
yolo version
yolo checks
```

#### Existing LoRA calls remain valid

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26.yaml")
model.train(
    data="coco8.yaml",
    epochs=100,
    lora_r=16,
    lora_alpha=32,
)
```

#### Enable the PEFT Planner explicitly

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26.yaml")
model.train(
    data="coco8.yaml",
    lora_r=16,
    lora_planner_enabled=True,
    lora_planner_backend="vpeft",
    lora_planner_solver="ao",
)
```

#### Enable MoLoRA with a positive expert count

```python
from ultralytics import YOLO

model = YOLO("ultralytics/cfg/models/26/yolo26.yaml")
model.train(
    data="coco8.yaml",
    molora_num_experts=4,
    molora_top_k=2,
    molora_r=8,
)
```

> [!WARNING]
> Do not combine a positive standard `lora_r` request with `molora_num_experts>0`; the adapter extension rejects the ambiguous request.

### Compatibility notes

- PRs [#158](https://github.com/Tencent/YOLO-Master/pull/158) and [#161](https://github.com/Tencent/YOLO-Master/pull/161) preserve released router and YOLOE checkpoint behavior. There is no `legacy_routing` argument.
- MoLoRA merge semantics have dedicated regression tests. There is no `molora_compat` argument.
- On macOS, the Agent runtime prefers MPS when available and no device is specified. Set `runtime.device` or disable `runtime.prefer_mps` to force CPU execution.

---

## 👥 Contributors

Thanks to every contributor who shaped this release. Commit counts below follow the audited release range and retain Git author identities as recorded.

| Contributor | Commits | Focus |
|---|---:|---|
| **isLinXu** | 364 | Project direction, MoE architecture, DDP hardening, release integration |
| **Hertz** | 102 | MoA/MoT integration and mixture optimization |
| **gatilin** | 24 | Agent system and release management |
| **13ewat3r** | 15 | MoA tests and vertical validation |
| **kimariyb** | 15 | MoT hybrid architecture and domain LoRA |
| **Thomas** | 13 | Project contributions across the release range |
| **SidKC** | 12 | LoRA/V-PEFT lifecycle and routing dataset fixes |

Additional contributions came from [**skywalker-lt**](https://github.com/Tencent/YOLO-Master/pulls?q=is%3Apr+author%3Askywalker-lt+) for edge deployment and reproduction workflows, plus **Lfan-ke**, **vankari**, **delei-kong**, **Cooryn**, **Ricky-7-Yan**, **Zviolin**, and the wider YOLO-Master community.

Special thanks to the Ultralytics team for the `8.4.101` upstream release. YOLO-Master v26.08 carries its mixture, PEFT, multi-task, and Agent extensions forward from the older `8.3.240` baseline.

---

## 🔗 Resources

- [Documentation site](https://tencent.github.io/YOLO-Master/)
- [GitHub Wiki](https://github.com/Tencent/YOLO-Master/wiki)
- [Model Zoo](https://github.com/Tencent/YOLO-Master/tree/main/model-zoo)
- [Discussions](https://github.com/Tencent/YOLO-Master/discussions)
- [Issues](https://github.com/Tencent/YOLO-Master/issues)
- [Development diff from v26.02](https://github.com/Tencent/YOLO-Master/compare/YOLO-Master-v26.02...main)

**License:** [GNU AGPL-3.0](https://github.com/Tencent/YOLO-Master/blob/main/LICENSE)<br>
**Contact:** [gatilin@tencent.com](mailto:gatilin@tencent.com) · [islinxu@163.com](mailto:islinxu@163.com)

---

## [YOLO-Master-v26.02] — 2026-02-13

- Based on Ultralytics `8.3.240`.
- Added LoRA support for model training.
- Established the Mixture-of-Experts module foundation.
- Added Sparse SAHI inference.
- Added Cluster-Weighted NMS (CW-NMS).
- Added MoE auxiliary-loss support.
- Added MoE pruning and analysis tools.

[View the v26.02 release](https://github.com/Tencent/YOLO-Master/releases/tag/YOLO-Master-v26.02)
