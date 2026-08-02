# Issue #54 Phase 3 Results Index

## Formal evidence

- [MoT five-seed formal evidence](phase3_architecture_controls/phase3_architecture_report_manifest.json): five
  independent FP32 seeds; the frozen manifest records the source runs and checkpoint identities.
- [Architecture controls report](phase3_architecture_controls/PHASE3_ARCHITECTURE_CONTROLS_REPORT.md): EsMoE,
  MoA, and MoT descriptive comparison.
- Supporting files: [per-run metrics](phase3_architecture_controls/phase3_architecture_run_metrics.csv),
  [architecture summary](phase3_architecture_controls/phase3_architecture_summary.csv), and
  [integrity checksums](phase3_architecture_controls/SHA256SUMS).

## Protocol summary

- MoT: 5 seeds, FP32.
- EsMoE: 3 seeds, AMP.
- MoA: 1 seed, AMP.
- All runs: 30 epochs, batch 8, image size 640, VisDrone2019-DET.

## Scientific boundary

MoA has only one independent seed. The architecture comparison is descriptive: it does not claim statistical
significance, causal superiority, or that one architecture is universally better. EsMoE and MoA do not produce or
claim MoT routing evidence.
