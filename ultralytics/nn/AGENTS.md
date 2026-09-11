# AGENTS.md — `ultralytics/nn/`

Neural network core for YOLO-Master: MoE/MoA/MoT modules, routing layers, detection heads, PEFT, and multi-backend inference.

> Cross-project commands, code style, and environment notes: see [root AGENTS.md](../../AGENTS.md).

## Responsibilities

- **`modules/moe/`** — MoE module family: routers, experts, pruning, dynamic scheduling, config, diagnostics
- **`modules/moa/`** — Mixture-of-Attention: blocks, heads, router, wrappers
- **`modules/mot/`** — Mixture-of-Transformers: block, experts, router, wrappers
- **`modules/multitask/`** — Multi-task heads and routers (detection, segmentation, pose)
- **`peft/`** — V-PEFT compiler and MoLoRA parameter-efficient fine-tuning
- **`backends/`** — Inference backends (ONNX, TensorRT, CoreML, NCNN, MNN, OpenVINO, etc.)
- **`foundation/`** — Foundation model integration: losses, routing, preprocessing, teachers, knowledge distillation
- **`modules/head.py`** — Detection/segmentation/pose heads
- **`modules/block.py`**, **`modules/conv.py`** — Shared building blocks

## High-Risk Files

Changes to these files require extra caution and targeted verification:

| File | Risk | Verification |
|------|------|--------------|
| `modules/moe/routers.py` | MoE routing core — affects all expert selection logic | `pytest tests/test_moe_router_boundaries.py -v` |
| `modules/moe/pruning.py` | MoE pruning & dynamic scheduling | `pytest tests/test_moe_dynamic_schedule.py -v` |
| `peft/molora/` (entire subpackage) | MoLoRA layers & model — affects parameter-efficient fine-tuning | `pytest tests/test_molora_routing_aware_merge.py tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py -v` |
| `peft/molora/model.py` | MoLoRA model orchestration | `pytest tests/test_vpeft.py -v` |
| `modules/routing_protocol.py` | Unified routing protocol across MoE/MoA/MoT | `pytest tests/test_moe_router_boundaries.py tests/test_master_model_configs.py -v` |
| `mixture_registry.py` | Mixture component registration | `pytest tests/test_master_model_configs.py -v` |

## Verification

```bash
# MoE routing & boundaries
pytest tests/test_moe_router_boundaries.py -v

# MoE dynamic scheduling
pytest tests/test_moe_dynamic_schedule.py -v

# MoLoRA merge & routing
pytest tests/test_molora_routing_aware_merge.py tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py -v

# V-PEFT compiler
pytest tests/test_vpeft.py -v

# Model config integrity (after architecture changes)
pytest tests/test_master_model_configs.py -v

# CI mixture regression gate
pytest tests/test_molora_dtype.py tests/test_molora_backend_roundtrip.py \
       tests/test_molora_merge_semantics.py tests/test_adapter_backend_contract.py --tb=long

# Doctest
pytest --doctest-modules ultralytics/nn/
```

## Task Routing

| If you are changing… | Start here | Then verify |
|----------------------|------------|-------------|
| MoE routing logic | `modules/moe/routers.py`, `modules/moe/config.py` | `test_moe_router_boundaries.py` |
| MoE experts or pruning | `modules/moe/experts.py`, `modules/moe/pruning.py` | `test_moe_dynamic_schedule.py` |
| MoA blocks or heads | `modules/moa/block.py`, `modules/moa/heads.py` | `test_master_model_configs.py` |
| MoT blocks or experts | `modules/mot/block.py`, `modules/mot/experts.py` | `test_master_model_configs.py` |
| MoLoRA layers or config | `peft/molora/layer.py`, `peft/molora/config.py` | `test_molora_*.py` + `test_vpeft.py` |
| Detection/segmentation heads | `modules/head.py` | `test_master_model_configs.py` |
| Inference backends | `backends/` | `test_exports.py --export-env base -v` |
| Foundation / distillation | `foundation/`, `foundation_distill_model.py` | `pytest --doctest-modules ultralytics/nn/` |
