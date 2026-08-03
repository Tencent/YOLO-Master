# Phase 3 MoT Cross-Seed Report

## Performance

| Metric | Mean | Sample SD | Min | Max |
|---|---:|---:|---:|---:|
| mAP50 | 0.160366 | 0.002928 | 0.157010 | 0.163640 |
| mAP50-95 | 0.083112 | 0.001834 | 0.080560 | 0.084690 |

## Routing evidence

- Mean dominant-expert agreement: 0.526042
- Mean token top-1 agreement: 0.435361
- Checkpoint repeated-inference rows passing determinism: 960/960.
- Five checkpoint SHA256 values were verified distinct before report generation.

## Supported conclusion

Performance across seeds is relatively stable, but internal routing shows only moderate or lower agreement and clear layer-level differences.

## Not supported by this evidence

- Routing instability necessarily reduces detection performance.
- A given expert has a fixed responsibility for a target type.
- Occlusion or object size has been proven to cause routing changes.
- Higher route entropy means routing is more stable.

## Reproducibility limitation

Deterministic CUDA warnings remain a reproducibility limitation: deterministic settings do not guarantee bitwise equivalence for every CUDA kernel or environment.

## Layer stability ranking

| Rank | Layer | Dominant agreement | Token top-1 agreement | Route entropy | Normalized route entropy |
|---:|---|---:|---:|---:|---:|
| 1 | model.23.m.0 | 1.000000 | 0.876156 | 1.098612 | 1.000000 |
| 2 | model.20.m.0 | 0.737500 | 0.534416 | 1.098034 | 0.999474 |
| 3 | model.14.m.0 | 0.621875 | 0.339736 | 1.098574 | 0.999965 |
| 4 | model.14.m.1 | 0.346875 | 0.340252 | 1.098604 | 0.999992 |
| 5 | model.20.m.1 | 0.250000 | 0.321607 | 1.098611 | 0.999999 |
| 6 | model.23.m.1 | 0.200000 | 0.200000 | 1.098612 | 1.000000 |
