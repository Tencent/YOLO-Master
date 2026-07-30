# Issue #54 Phase 2.1 MoT AMP Fix Report

Evidence classification: `diagnostic_not_formal_evidence`.

## Root cause and fix

RTX 4090 tracing reproduced the failure only in eval sparse dispatch. The incoming feature, router weights, and accumulator were FP16, while all expert outputs were FP32. FP32 LayerScale parameters (`ls1`/`ls2`) promote AMP expert residuals to FP32. Multiplication therefore produced FP32 and indexed assignment into the FP16 accumulator failed.

`MoTBlock._blend_experts` now creates its accumulator from the first actual expert output and converts only each local blend-weight view to that output's dtype/device immediately before fusion. It does not mutate router outputs, parameters, probabilities, top-k indices, expert selection, or loss formulas. The FP32 reference is bit-identical in the regression test.

## Validation

| Check | Result |
|---|---|
| Local Issue #54 + MoT tests | 77 passed, 20.94s |
| Cloud Issue #54 + MoT tests | 77 passed, 11.01s |
| CUDA AMP block forward/backward | finite output/gradients; 0.92s; 16.82 MiB |
| Official MoT YAML AMP eval | 6 MoT layers; 0.25s; 38.31 MiB |
| Python compile / changed-file Ruff / diff check | passed |

The repository-wide Ruff baseline remains non-clean independently of this patch: 450 existing lint findings and 172 files outside this change would be reformatted.

## Targeted smoke

Protocol: VisDrone inventory SHA256 `4a7a03b08cf21d913ab85a86b40a75eee13579881fba9bc0d979b72b7f0a96fa`, seed 0, deterministic requested, 3 epochs, fraction 0.05, imgsz 640, requested/actual batch 8/8, workers 8, cache disabled, AMP enabled.

| Precision | Status | Train/val s | s/epoch | Peak GiB | Loss | mAP50 / mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| AMP fixed | passed_diagnostic | 81.09 | 27.03 | 7.868 | 13.54409 | 0 / 0 |
| Existing FP32 | passed_diagnostic | 74.87 | 24.96 | 7.873 | 13.12377 | 0.00001 / 0 |

AMP was 8.31% slower and used 0.07% less peak allocated memory in this short diagnostic. No dtype error, NaN/Inf, OOM, batch reduction, or stall occurred. Low short-run mAP is not a failure criterion.

AMP checkpoint: `${REMOTE_PHASE2_1_RESULTS}/training/A_mot_amp_fixed/weights/best.pt`, 8,710,926 bytes, SHA256 `9f40d2d56e0b6019deef25a7f12b0629dfd2e4c801c849075b0f16ca51d13d5f`.

## Routing comparison

The fixed 32-image manifest produced 384 records per checkpoint: 32 images × 6 layers × 2 exact repeats. Expert names were `LocalConvTransformer`, `WindowTransformer`, and `DeformableTransformer`. Probabilities were finite, non-negative, normalized; repeats were identical and all hooks were removed.

Across 192 aligned image/layer rows, AMP versus FP32 top-1 agreement was 0.625, mean JSD was `4.8452e-08`, and mean entropy difference (AMP − FP32) was `3.3025e-07`. Both routers remained nearly uniform after only three epochs, so tiny probability changes can flip argmax; top-1 agreement must not be interpreted as instability or cross-seed evidence. AMP and FP32 are precision conditions from separate same-seed diagnostic runs, not independent seeds.

## Recommendation and remaining risks

The dtype blocker is resolved and AMP is eligible for a deliberately approved formal protocol. This smoke does not validate long-run determinism, real convergence, multi-seed variance, final mAP, or AMP speed advantage. Because AMP was not faster here and CUDA emitted warnings about nondeterministic attention/pooling kernels under `warn_only`, choose AMP only after a longer timing/determinism check; FP32 remains the conservative baseline.

Linear 30-epoch estimates from the measured AMP time are 1.126 GPU-hours for 5 runs and 2.027 GPU-hours for 9 runs, excluding setup and retry overhead.
