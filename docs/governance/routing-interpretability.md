# Routing Interpretability Toolkit

YOLO-Master provides a shared diagnostic API for MoE, MoA, MoT, Latent, and MoLoRA routing. The toolkit observes existing router outputs and `last_routing_snapshot` state through temporary hooks; it does not add model parameters, checkpoint fields, or deployment behavior.

## Python API

```python
from ultralytics.utils.routing_interpreter import RoutingInterpreter

interpreter = RoutingInterpreter(model)

# Run one batch and capture normalized [B, E, H, W] or [B, E] probabilities.
heatmaps = interpreter.capture_routing(batch)

# Spatial routers write continuous color-value overlays aligned to `batch`.
# Global routers write a distribution plot because they have no spatial evidence.
visualizations = interpreter.save_routing_visualizations(
    heatmaps,
    "runs/routing_interpreter",
    input_image=batch,
)

# Read normalized usage, entropy, Gini, dominant expert, and dead experts.
summaries = interpreter.collect_layer_summaries(heatmaps=heatmaps)
collapse = interpreter.detect_routing_collapse(heatmaps=heatmaps)

# Export one versioned, JSON-safe record per observed routed layer.
records = interpreter.routing_snapshot_records(heatmaps=heatmaps, step=0, mode="eval")

# Aggregate which input characteristics activate each expert.
specialization = interpreter.analyze_expert_specialization(dataloader, num_samples=1000)

# Compare natural routing with a temporary forced-expert counterfactual.
causal = interpreter.routing_causal_analysis(batch, "model.4.m.0", expert_idx=2)
```

The default specialization descriptors are per-sample activation mean, standard deviation, RMS, spatial size, and high-frequency energy. Dataset-specific semantics such as object density, scale distribution, class mix, or occlusion can be supplied with `feature_fn(batch) -> dict[str, Tensor]`.

Tensor batches call `model(batch)`, tuple/list batches call `model(*batch)`, and dictionaries call `model(**batch)`. Use `forward_fn(model, batch)` when a YOLO dataloader dictionary needs custom image extraction or preprocessing.

Dense router tensors are normalized along the expert axis. Sparse routers that return `(topk_weights, topk_indices, metadata)` are expanded with `scatter_add` into the full expert space before summaries, collapse checks, or rendering. Image-level singleton dimensions such as `[B, E, 1, 1]` are reported as `[B, E]`; genuine spatial routers retain `[B, E, H, W]`.

Passing the captured `heatmaps` mapping to `collect_layer_summaries()` or `detect_routing_collapse()` limits the result to layers observed in that capture. Without it, these methods read the most recent valid `last_routing_snapshot` from every discovered leaf routed layer. Snapshot expert vectors must contain exactly `num_experts` entries; malformed vectors are ignored instead of being truncated.

## Checkpoint CLI

```bash
python tools/routing_interpreter.py \
  runs/train/weights/best.pt \
  assets/bus.jpg \
  --device cuda:0 \
  --output runs/routing_interpreter
```

The command writes spatial `*_confidence_heatmap.png`, `*_expert_<n>_heatmap.png`, and `*_assignment_map.png` overlays for spatial routers. Global routers write `*_routing_distribution.png` instead. It also writes a dashboard and a `routing_report.json` containing shapes, mean usage, visualization artifact paths, collapse metrics, and layer summaries. Use an exact layer name to narrow the capture and optionally measure a counterfactual:

Single-image mode also writes `routing_snapshot.jsonl`. Each `yolo_master.routing_snapshot.v1` row contains the family, layer, execution context, expert count, top-k, normalized usage, optional probability shape, spatial capability, entropy, Gini, dominant share, dead experts, collapse status, and auxiliary-loss availability. Consumers should reject unsupported schema versions instead of guessing field meanings.

```bash
python tools/routing_interpreter.py best.pt image.jpg \
  --layer model.4.m.0 \
  --expert 2
```

## Interpretation Boundaries

- Forced routing is a diagnostic ablation, not a supported deployment mode. It temporarily replaces router outputs and restores hooks and training flags afterward.
- For sparse top-k routers, forced routing changes both weights and expert indices, assigning all probability mass to the requested expert while preserving the router output structure and metadata.
- Output-distance metrics show sensitivity, not causal attribution to a semantic concept. Semantic claims require controlled dataset slices or caller-provided descriptors.
- A collapse flag combines dominant share, normalized Gini, normalized entropy, and dead-expert checks. Thresholds should be calibrated for dense MoA and sparse top-k routers separately.
- Image-level routers produce expert distributions (`[B, E]`), not image heatmaps; without spatial evidence, assigning colors to pixels would be misleading.
- Spatial routers produce continuous confidence and per-expert activation heatmaps plus a categorical top-1 assignment map (`[B, E, H, W]`). Values are upsampled from the router grid into the actual model-input geometry before being overlaid.

