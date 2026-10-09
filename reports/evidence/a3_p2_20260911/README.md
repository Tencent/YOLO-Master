# A3 P2 evidence snapshot (2026-09-11)

This directory is the review-sized snapshot for the A3 five-family FP32/INT8 experiment. The original uploaded bundle is intentionally not committed as a binary archive.

## Files

- `p2_evidence_summary.json`: compact derived metrics and the explicit claim boundary.
- `accuracy_summary.csv` and `accuracy_summary.json`: raw accuracy-run summaries.
- `route_consistency_summary.csv`: raw per-layer route-consistency metrics.
- `experiment.lock.json`: immutable dataset, checkpoint, repository and export-environment lock.
- `cpu_export.yaml`: quantization and evaluation configuration.

## Source bundle

```text
name: a3_p2_evidence_bundle.zip
size: 946,882 bytes
files: 28
uncompressed: 11,926,523 bytes
SHA256: 468523B784E9FCFAF5E92521E81785C55D16569A44F9A7799E4120B9A829BF6B
```

The full bundle additionally contains all five per-family route JSON files and the MoLoRA structural audit. Its hash is the integrity anchor for evidence omitted from ordinary Git history.

## Interpretation warning

The top-level `accuracy_summary.json` records `selected_families=["latent"]` and `completed=4/4` because it reflects the final resumed invocation. The preserved per-family result files and `accuracy_summary.csv` together contain successful FP32 and full-INT8 results for all five families. Therefore the report claims a **10/10 core FP32/INT8 matrix**, not a 20/20 four-variant matrix.

`elapsed_seconds` is complete validation wall time, not an isolated latency benchmark. Dense-probability route rows are Top-2 ranking proxies and do not prove conditional expert execution.
