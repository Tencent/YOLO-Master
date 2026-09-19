# E3 unified routing observability for YOLO-Master

This directory combines the three E3 acceptance levels into one reviewable contribution. It observes real routed
modules through removable forward hooks and does not alter model forward signatures or routing decisions.

## Acceptance map

| Level | Deliverable | Audited result |
| --- | --- | --- |
| P0 | One schema, structured logs and static plots for MoE/MoT/Latent | 52 events, 13 modules, 52/52 repeat match, batch 2/4 max load delta 0 |
| P1 | Live dashboard for at least three families and <10% slowdown | Three-family HTTP dashboard; 108 paired full steps; all median and CI upper bounds <10% |
| P2 | Original-image token overlays, two-minute demo, more families | True MoT/MoA overlays; five-family audit; 1:55 real UI demo |

P2 deliberately reports MoE, Latent and MoLoRA as spatially `UNSUPPORTED`: their observed contracts do not expose
a reversible two-dimensional token axis. Sample-level or expert-axis vectors are never reshaped into plausible but
false heatmaps.

## Layout

```text
tools/e3_routing/
├── src/e3_p0/             # unified event schema, collection and static plots
├── src/e3_p1/             # live dashboard and paired overhead gates
├── src/e3_p2/             # spatial capture, overlays and mechanism analyses
├── schemas/               # e3.routing/v1.0.0 JSON Schema
├── configs/               # current-main configs plus untouched frozen configs
├── tests/                 # 113 phase and CLI-contract tests
├── scripts/               # latest-main compatibility smoke
└── artifacts/             # locally generated outputs (ignored by Git)
```

## Review-first checks

From the YOLO-Master repository root:

```bat
cd tools\e3_routing
python -m pip install -e ".[test]"
run_tests.cmd
run_smoke.cmd --output artifacts\latest-main-smoke.json
```

The compatibility smoke loads MoE, MoT, Latent and MoA detector configs from the containing checkout, runs real
forward passes, checks hook output equality, verifies true MoT/MoA `[B,E,H,W]` captures, and re-audits the explicit
non-spatial boundary for MoE, Latent and MoLoRA.

## Formal evidence

The archived experiments under `configs/frozen/` are bound to source commit
`07d330325b5a26b75aabfc75389f9bcbc0d40245`; the historical
increment boundary is `acce839c7e895d6b179de7f7093fa879e237cc7b`. See [RESULTS.md](docs/RESULTS.md) for the numerical
claims and [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for the distinction between archived evidence and the
latest-main compatibility gate.

The active configs use the containing checkout and were refreshed against upstream main
`7bfbfd374a6b44720f98366e088f3a962e16321d`. Run P0 once before the P1 benchmark/dashboard so the local ignored
`artifacts/p0/LATEST.txt` and event stream exist.

Phase design and acceptance details are summarized in [P0.md](docs/P0.md), [P1.md](docs/P1.md), and
[P2.md](docs/P2.md). The full manifest-bound evidence repositories remain available separately:

- P0: <https://github.com/Ricky-7-Yan/YOLO-Master-E3-P0>
- P1: <https://github.com/Ricky-7-Yan/YOLO-Master-E3-P1>
- P2: <https://github.com/Ricky-7-Yan/YOLO-Master-E3-P2>

## Scope and limitations

- Formal results are CPU/coco8 or a deterministic 32-image coco128 subset.
- Models are randomly initialized; observations are cold-start mechanism evidence, not trained accuracy claims.
- Negative timing values are scheduler noise, not observer speedups.
- Correlations and detector-output coupling do not establish routing causality.
- Generated reruns are ignored; only compact, manifest-bound formal evidence is included for review.
