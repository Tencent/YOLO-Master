# Issue #54 Phase 2.3 Controller Recovery and Protocol Freeze

Evidence classification: `diagnostic_not_formal_evidence`.

## Controller defect and recovery

The Phase 2.2 child completed training and validation, then called `payload.update(parsed, metadata)`. Python `dict.update` accepts only one positional mapping, so post-processing raised `TypeError` and the controller incorrectly recorded 0/3 epochs.

The fix explicitly merges `parsed` and `metadata` in two operations. Recovery mode reconstructs results from `args.yaml` and `results.csv`, verifies the expected best-checkpoint SHA256 and loadability, indexes the existing checkpoint into the new result directory, and never launches a completed MoT calibration again. Partial or invalid recovery artifacts cannot be marked `passed_pilot`.

An additional recovery-only path issue was found before inference: the routing exporter requires a checkpoint path under the new output root. Recovery now passes its verified checkpoint-index symlink while retaining the original checkpoint identity and SHA256. The failed recovery attempt performed no inference or training and remains isolated.

## Verification

- Controller regressions: 8 passed in 0.10 s.
- Issue #54 plus `tests/test_mot.py`: 80 passed in 10.14 s.
- Changed-file Ruff lint/format, Python compile, and `git diff --check`: passed.
- Repository-wide Ruff remains blocked by 450 pre-existing violations outside the Phase 2.3 changes; none are introduced by the changed files.
- Environment: RTX 4090 24 GB, PyTorch 2.5.1+cu124, CUDA 12.4.
- Dataset inventory SHA256: `4a7a03b08cf21d913ab85a86b40a75eee13579881fba9bc0d979b72b7f0a96fa`.

## Calibration results

All runs used full VisDrone, seed 0, image size 640, requested/actual batch 8/8, eight workers, deterministic mode, and no cache. Metrics are three-epoch calibration diagnostics only.

| Model | Precision | Epochs | s/epoch | Peak GiB | Loss | mAP50 | mAP50-95 | Checkpoint SHA256 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| MoT | AMP | 3/3 | 373.74 | 7.90 | 7.37289 | 0.02993 | 0.01124 | `70ff9e97539a772cb009539f16159d7eb9a623d1a8df28c8f5ee68af2ab19b8b` |
| MoT | FP32 | 3/3 | 291.82 | 7.88 | 6.60144 | 0.04069 | 0.01672 | `d2d81d184519ca0dc14a78b93e1c2ca310a6aeb7557cbe505639842903d99361` |
| EsMoE | AMP | 3/3 | 215.08 | 4.47 | 7.33244 | 0.03148 | 0.01221 | `06021f02c538d640a1426c6a29bbc8b0a677b0e846179381b0addc97a610bdd6` |
| MoA | AMP | 3/3 | 294.19 | 4.87 | 7.38659 | 0.02862 | 0.01049 | `ab8bc2bdb47f2bb4c4f64e53c489bfece17d1578eb3694c4575a7ee29b0fe38b` |

All four best checkpoints exist, load with the repository YOLO loader, contain the expected three-epoch training arguments, and have finite `results.csv` values. No dtype failure, NaN, Inf, OOM, cache contamination, or batch reduction occurred.

## Routing and precision freeze

Each MoT checkpoint produced 384 records: 32 fixed validation images × 6 layers × 2 repeats. Expert names were `DeformableTransformer`, `LocalConvTransformer`, and `WindowTransformer`. Probabilities were finite, non-negative, normalized, repeat deterministic, and hooks returned to their initial counts.

AMP/FP32 comparison aligned 192 image-layer rows. Overall top-1 agreement was 0.500, mean JSD was `1.29584e-05`, and mean entropy differed by `3.42612e-05`. The checkpoints are same-seed precision conditions, not independent seeds. Both routing paths are stable, but AMP was 28.1% slower; the frozen MoT precision is therefore **FP32** under the predeclared 10% rule.

## Formal protocol B

The recommended protocol remains diagnostic until explicitly launched:

- MoT FP32: 5 seeds × 30 epochs.
- EsMoE AMP: 3 seeds × 30 epochs.
- MoA AMP: 1 seed × 30 epochs.
- Total: 9 runs and 270 epochs.

Measured training-only estimates are 12.16 GPU-hours for MoT, 5.38 for EsMoE, and 2.45 for MoA: 19.99 GPU-hours total. Including observed process overhead and a 20% admission reserve requires approximately 26.0 hours. At an account rate of `R` currency units per GPU-hour, budget `26.0 × R`.

Sequential order: MoT seeds 0–1, EsMoE seed 0, MoA seed 0, then remaining MoT and EsMoE seeds. This exposes model-specific failures early while retaining enough completed MoT seeds for preliminary variance checks. Each run must use an isolated directory and process, record requested/actual batch separately, and fail independently.

With every-epoch checkpoints, reserve 9–12 GB. The recommended retention policy is `best.pt`, `last.pt`, `last_healthy.pt`, and periodic checkpoints every five epochs, with complete logs and manifests retained; reserve 6 GB plus a safety margin. Automatic recovery may retry one recognized transient failure and resume only the same experiment/seed from a verified checkpoint. Partial or resumed failures remain non-passed, and no retry may change model mathematics or protocol fields.

`mot_seed0_30e`, `mot_seed1_30e`, `esmoe_seed0_30e`, and `moa_seed0_30e` are consistently recorded as `not_started` in the manifest and CSV. No formal training was started in Phase 2.3.
