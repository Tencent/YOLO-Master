# Foundation Distillation Evaluation Plan

This document is the review contract for the Foundation teacher experiment. It freezes the design and decision line
before an accuracy result is inspected.

## P0: minimum viable distillation

- Student location: the P4 source consumed by the YOLO `Detect` head. `StudentFeatureTap` resolves it from the head's
  graph metadata instead of relying on a model-specific hard-coded layer number.
- Teacher location: DINOv3 dense patch tokens, normalized to BCHW and exposed as `dense["p4"]` by the teacher protocol.
- Projection: trainable student 1x1 convolution plus optional normalization; detached teacher identity or frozen 1x1
  projection. Only the teacher grid is resized to the live student grid.
- Loss: cosine feature loss plus optional sampled relational loss. The teacher is detached, kept out of the optimizer,
  state dict, EMA, DDP graph, prediction path, and export graph.
- Smoke evidence: `python scripts/foundation_p0_smoke.py`. Passing requires finite loss, decreasing single-batch loss,
  aligned stage shapes, a trainable student projector, and no teacher gradient.

The production entry point remains `FoundationDistillationModel`; the offline smoke is deliberately a dependency-light
acceptance probe, not a second training implementation.

## P1: paired on/off experiment

Use `scripts/foundation_f08_effect_gate.py` with at least three seeds. Each baseline/Foundation pair must share the
model config, initialization seed, dataset split, epochs, image size, batch size, optimizer, augmentation, validation
split, and checkpoint-selection rule. Only `foundation_*` settings and run identity may differ.

The primary metric is `metrics/mAP50-95(B)`. Repository metrics use a 0..1 fraction; the decision report converts them
to percentage points. The preregistered rule is:

> NO-GO when `abs(mean paired delta) < 0.3 mAP points` and the two-sided 95% paired Student-t confidence interval
> contains zero.

A positive GO requires both a mean delta of at least +0.3 points and a confidence interval strictly above zero. A
negative interval is NO-GO. All other outcomes are INCONCLUSIVE and require more paired seeds without changing the
decision line.

Generate the review artifacts with:

```bash
python scripts/foundation_distill_decision.py \
  --input reports/foundation/v0.1/f08-effect-gate-mps.json \
  --output-json reports/foundation/v0.1/distill-decision.json \
  --output-md reports/foundation/v0.1/distill-recommendation.md
```

## No-confound table

| Variable | Baseline | Foundation | Allowed to differ |
|---|---|---|---|
| Student YAML and initialization | same model + seed | same model + seed | no |
| Dataset and train/validation split | frozen | frozen | no |
| Epochs, batch, image size, optimizer | frozen | frozen | no |
| Augmentation and checkpoint selection | frozen | frozen | no |
| Teacher/projector/loss settings | disabled | preregistered recipe | yes |
| Decision threshold | 0.3 mAP points | 0.3 mAP points | no |

The decision tool audits the serialized run plan and returns `INSUFFICIENT` if a non-Foundation field differs.

## P2: interpretation and recommendation

For GO, report paired AP deltas for small/medium/large objects and optional per-class AP. A category claim is allowed
only when the evaluator supplied `per_class_ap`, `per_category_ap`, or `category_ap`; missing values are never imputed.

For NO-GO or INCONCLUSIVE, the generated recommendation covers four hypotheses:

- capacity: compare the next student size only when scale evidence supports a backbone bottleneck;
- dimension: compare one smaller and one larger `foundation_align_dim` under the same paired budget;
- optimization: inspect Foundation/task loss ratio and raw cosine/relational curves before retuning the weight;
- data: add paired seeds when the confidence interval is wide, without moving the 0.3-point decision line.

The Markdown output is the formal go/no-go recommendation. It must accompany the machine-readable JSON and the source
effect-gate report in review.
