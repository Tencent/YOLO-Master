# MoE Governance and Engineering Debt Design

## Scope

This phase implements four compatibility-first changes:

1. Add a version-aware retirement audit for YAML-visible experimental MoE variants.
2. Move internal auxiliary-loss consumers to the step-aware routing protocol.
3. Make `MoLoRAMoEAwareLayer` reuse `MoLoRALayer.__init__`.
4. Give `capacity_factor` one documented meaning across MoLoRA configuration and execution.

Router hooks, pre-export pruning, and routing-preserved ONNX export are intentionally deferred. They change model or
export contracts and need separate calibration and checkpoint compatibility designs.

## Variant Lifecycle

`DEPRECATED_MOE_CLASSES` becomes a fourth, disjoint API tier beside stable, experimental, and legacy. It starts empty:
the repository has only a current-state audit and cannot prove that a class was absent from YAML for two releases.

`scripts/audit_moe_usage.py` remains the single audit entry point. It will additionally scan the keys registered in
`MIXTURE_MODULES`, collect their YAML references, record explicit version snapshots in a JSON ledger, and report an
experimental class as deprecation-eligible only when it is absent from the latest two distinct snapshots. Recording a
snapshot does not edit the runtime tier automatically. Release owners must review checkpoint and external Python API
compatibility, then update `DEPRECATED_MOE_CLASSES` deliberately. A check mode detects drift between eligible classes
and the declared deprecated tier.

## Auxiliary-Loss Migration

The step-aware records in `routing_protocol.py` become the only internal source used to compute training loss.
`MoLoRAModel.compute_aux_loss` and the composite MoE collector call `collect_aux_loss` directly. Module properties use
the canonical record or their local last value. Recovery diagnostics inspect canonical records.

`MOE_LOSS_REGISTRY` and `_registry_set` remain for one compatibility window: legacy integrations can still read values
written during forward. Internal training code will no longer accept values injected only into the legacy dictionary,
preventing stale or out-of-step tensors from entering the loss.

## MoE-Aware Initialization

`MoLoRALayer.__init__` accepts optional `expert_ranks` and `router_calibration` inputs. The parent validates the rank
list and builds either uniform-rank or per-expert adapters while initializing every shared field exactly once.
`MoLoRAMoEAwareLayer` becomes a thin specialization that calls `super().__init__` and keeps only its calibrated-router
forward behavior. Existing state-dict names and expert module ordering remain unchanged.

## Capacity Semantics

`capacity_factor` is the multiplier in `ceil(capacity_factor * batch * top_k / num_experts)`. Values `<= 0.0` or
`>= num_experts` mean unlimited. Values in `(0.0, num_experts)` enable the soft capacity penalty. The default changes
from `1.0` to `0.0` so users who relied on the previous default-unlimited behavior keep that behavior, while an explicit
`1.0` now means one ideal-share capacity. Configuration rejects non-finite values.

## Verification

Focused tests cover lifecycle history, tier separation, canonical-only internal aux collection, backward-compatible
legacy publication, inherited initialization/state keys, per-expert ranks, calibration gradients, and all capacity
boundaries. The existing MoE, MoLoRA, V-PEFT, configuration-integrity, Ruff, formatting, and codespell gates are run in
proportion to the touched files.
