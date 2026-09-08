# C2 tiered coverage experiment

## Hypothesis

C1 improves candidate-starved short-side `[8,16)` GTs, especially elongated shapes, but its uniform rescue budget can
shift too much supervision away from other shapes. C2 keeps the same candidate region and nearest-ring selection while
changing only the number of supplemental candidates.

## Assignment rule

The baseline/core candidates are always preserved. Only GTs with at least one original side in `[8,16)` are eligible.

| Baseline candidates | Ordinary GT (long side `<32`) | Elongated GT (long side `>=32`) |
|---:|---:|---:|
| 0 | target 3 | target 3 |
| 1 | target 2 | target 3 |
| 2 | keep 2 | target 3 |
| `>=3` | fixed STAL | fixed STAL |

The expansion target stays at 20 pixels, fixed TopK stays at 10, and TAL `alpha=0.5` / `beta=6.0` stay unchanged.
No IoU, alignment-quality, or crowding gate is included in this experiment.

## Controlled comparison

- Baseline: fixed STAL.
- C1: uniform coverage-triggered target of 3.
- C2: deficit-tiered rule above.
- Use the same model, dataset, initialization policy, seed, optimizer, augmentations, image size, batch/nbs, epoch budget,
  validation split, and checkpoint cadence for all three runs.

## Required evidence

- Effective `args.yaml`, full console log, `results.csv`, and per-epoch checkpoints.
- Official checkpoint-based AP/AP50/APs/APm/APl and recall metrics.
- Shape AP table using the same evaluator and bins as C1/baseline.
- Assignment counts by baseline-candidate bucket (`0`, `1`, `2`, `>=3`), ordinary/elongated group, and epoch.
- Supplemental-candidate count, selected-positive count, and multi-GT conflict count.

## Launch boundary

`run_c2_tiered.ps1` is prepared for the constrained-memory batch-4/nbs-64 protocol. Audit its printed effective
configuration and GPU availability before explicitly authorizing a 120-epoch launch.
