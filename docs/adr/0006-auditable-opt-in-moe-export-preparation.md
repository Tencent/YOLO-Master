# ADR-0006: Use Auditable Opt-In MoE Export Preparation

## Status

Accepted for the first deployment slice.

## Decision

MoE expert pruning before export is opt-in through `pre_export_prune=True`.
The exporter deep-copies the model, performs a bounded synthetic calibration
pass, applies the existing structural pruning surgery to the copy, and writes
a JSON manifest beside the exported artifact. The manifest records calibration
steps, threshold, retained experts, and the output artifact. No source model or
checkpoint is mutated, and the default export path remains unchanged.

MoLoRA exposes `molora_export_mode=dynamic` (the existing behavior) and the
explicit `routing_preserved` mode. The latter is accepted only for ONNX and
TorchScript and is surfaced as a distinct preflight strategy; it keeps router
and expert parameters in the graph rather than silently merging them. Backend
operator support and numerical parity remain verification requirements.
The export forward path omits runtime-only diagnostics and auxiliary-loss
publication so both the project TorchScript-based ONNX wrapper and PyTorch's
dynamo ONNX exporter can capture the same router-preserved tensor graph.

## Consequences

The deployment artifact can be smaller and cheaper after calibration, while
the manifest makes the structural change reviewable and reproducible. Pruning
quality depends on calibration inputs, so the default is conservative and
disabled. Routing-preserved export avoids approximate merge semantics but can
still execute all LoRA experts in a static graph.
