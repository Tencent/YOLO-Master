# Issue #54 Phase 2 Diagnostic Smoke Report

Evidence classification: `diagnostic_not_formal_evidence`.

This run validates execution paths only. It is not formal accuracy, stability, or cross-seed evidence.

## Environment

- Host: `autodl-container-29e34aa8c9-7f86c093`
- Python: `<REMOTE_PYTHON>`
- PyTorch/CUDA: `2.5.1+cu124` / `12.4`
- GPU: `NVIDIA GeForce RTX 4090`, 24564 MiB
- Git commit: `675659e2b13a03f5c3c5c421d3eb0656255a2eb5`

## VisDrone

- Source YAML: `<READ_ONLY_VISDRONE_SOURCE_YAML>`
- Source YAML SHA256: `f3ac89c439ea06ee456876fb6dab98390b899d0af305e44b01395ecfa9429fc0`
- Data inventory SHA256: `4a7a03b08cf21d913ab85a86b40a75eee13579881fba9bc0d979b72b7f0a96fa`
- Counts: `{'test': {'images': 1610, 'labels': 1610}, 'train': {'images': 6471, 'labels': 6471}, 'val': {'images': 548, 'labels': 548}}`
- Images were reused read-only. Labels were copied into the MoT result directory so generated cache files did not modify the old dataset.
- The old train cache is absent after completion; the pre-existing val cache was preserved.

## Runs

| Run | Variant | Precision | Batch | Epochs | Train/val s/epoch | Process s | Peak GiB | mAP50 | mAP50-95 | Loss | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| A_mot_amp | mot | amp | 8/8 | 3 |  | 46.41 | 6.88 | None | None | None | failed |
| B_mot_fp32 | mot | fp32 | 8/8 | 3 | 24.96 | 111.09 | 7.87 | 1e-05 | 0.0 | 13.12377 | passed_diagnostic |
| C_esmoe_amp | esmoe | amp | 8/8 | 2 | 23.29 | 82.24 | 4.61 | 0.0 | 0.0 | 15.58225 | passed_diagnostic |
| D_moa_amp | moa | amp | 8/8 | 2 | 23.84 | 94.96 | 5.03 | 0.0 | 0.0 | 15.577850000000002 | passed_diagnostic |

Run A completed one AMP training epoch but failed during validation with `expected scalar type Float but found Half` in `MoTBlock._blend_experts`. It has only a partial `last_healthy.pt`; it is not a successful checkpoint. No OOM, NaN/Inf, batch reduction, or stall occurred. The optimizer request `auto` resolved to AdamW at lr=0.000714 for all runs.

## Routing

- MoT FP32: 32 fixed validation images, 6 layers, 2 exact repeats, 384 records.
- Experts: `WindowTransformer`, `DeformableTransformer`, `LocalConvTransformer`.
- Probabilities are finite, non-negative, normalized; repeated inference is identical; hooks are fully removed.
- MoT AMP routing was not exported because Run A did not produce a successful checkpoint. AMP/FP32 route comparison is therefore unavailable and was not fabricated.

## Time and estimate

- Controller wall time: `401.54` seconds.
- Aggregate child process + routing time: `340.75` seconds (includes model/data setup).
- Mean successful measured train/val time: `24.03` seconds/epoch.
- MVP 5 runs × 30 epochs: `1.001` GPU-hours.
- Recommended 9 runs × 30 epochs: `1.802` GPU-hours.

## Recommendation

This original Phase 2 run identified a reproducible MoT AMP validation defect. Phase 2.1 fixed and retested that defect; see `../PHASE2_1_MOT_AMP_FIX_REPORT.md`. Formal multi-seed training still requires explicit protocol approval.
