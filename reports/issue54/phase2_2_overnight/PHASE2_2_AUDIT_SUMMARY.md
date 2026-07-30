# Issue #54 Phase 2.2 Overnight Acceptance Audit

Evidence classification: `diagnostic_not_formal_evidence`

## Outcome

The controller ended with `implementation_failed`; no 30-epoch pilot run was started. The cloud instance remained online, and the termination was not caused by AutoDL scheduled shutdown.

Both full-VisDrone, three-epoch MoT calibration trainings completed and produced valid checkpoints. Post-processing then failed at `payload.update(parsed, {...})` because `dict.update()` received two positional mappings. Consequently, the generated controller report conservatively records zero completed epochs, no metrics, and non-loadable checkpoints even though `results.csv` and checkpoint inspection prove otherwise. This audit does not reclassify either run as `passed_pilot`, because the required routing gate was never executed.

## Reconstructed calibration evidence

| Run | Epochs | Batch | Final mAP50 | Final mAP50-95 | Final train loss (box/cls/dfl/aux) | Training time | Seconds/epoch | Peak GPU memory | NaN/Inf/OOM |
|---|---:|---:|---:|---:|---|---:|---:|---:|---|
| MoT AMP | 3/3 | 8/8 | 0.02993 | 0.01124 | 3.08790 / 2.67434 / 1.61065 / 1.89156 | 1121.210 s | 373.737 s | 8,482,816,512 B | No |
| MoT FP32 | 3/3 | 8/8 | 0.04069 | 0.01672 | 2.77757 / 2.42454 / 1.39933 / 1.99077 | 875.465 s | 291.822 s | 8,459,157,504 B | No |

The two runs use the same seed and differ only in precision; they are not independent seeds. AMP was 28.1% slower per epoch than FP32. The protocol rule therefore points to FP32 on timing, but the formal precision selection remains unset because no Phase 2.2 routing export or repeat-determinism check completed.

## Checkpoint and artifact audit

- All 12 generated `.pt` files exist, have distinct paths, and load successfully with the repository's `YOLO` loader.
- `best.pt` is the accepted final artifact for each calibration; neither accepted artifact is `last_healthy.pt` or a partial checkpoint.
- `results.csv` contains epochs 1–3 with finite metrics; `args.yaml` confirms `epochs=3`, `batch=8`, `fraction=1.0`, `cache=False`, and the requested precision.
- No automatic batch reduction, OOM, dtype error, NaN, or Inf was found.
- Cache files were confined to this run's copied-label execution view. Earlier Phase 2/2.1 result files were not overwritten.
- The data inventory is frozen at SHA256 `4a7a03b08cf21d913ab85a86b40a75eee13579881fba9bc0d979b72b7f0a96fa`.

## Missing acceptance evidence

- Routing export directory is empty: no six-layer export, repeat inference, hook-cleanup evidence, or AMP/FP32 routing comparison.
- No second MoT seed completed, so no cross-seed result exists.
- MoE and MoA 30-epoch pilot runs were never scheduled.
- Planned long runs are absent from the controller manifest instead of being explicitly recorded as not started.

## Readiness decision

Large-scale formal training is **not yet admitted**. Before launch, fix and test the controller post-processing call, replay post-processing/routing against the existing calibration checkpoints without retraining, verify deterministic six-layer routing, and obtain trustworthy full-data timing for MoE and MoA. Subject to those gates, the recommended target is the 9-run plan: MoT 5 seeds, MoE 3 seeds, and MoA 1 seed.
