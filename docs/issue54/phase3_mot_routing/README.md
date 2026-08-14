# Phase 3 MoT routing evidence

This directory publishes the compact formal five-seed MoT routing evidence used by Tencent/YOLO-Master PR #216. The
recorded formal root is `/root/autodl-tmp/MoT/results/phase3_formal_20260731T214515Z`. The highest-level experimental
unit is an independently trained seed; images, tokens, layers, repeated exports, and seed pairs are not independent
training repetitions.

## Raw source publication policy

The verified raw analyzer output `phase3_cross_seed_routing.json` is omitted from Git to keep the PR diff reviewable,
not because the result is invalid. Its formal identity remains public:

- size: 3,112,098 bytes;
- SHA256: `b7049cdbac25c346ac8deb37b505d92bd37406f197458a8f045677c1eba9f7f2`;
- formal root: `/root/autodl-tmp/MoT/results/phase3_formal_20260731T214515Z`;
- provenance: `SOURCE_PROVENANCE.json` and `reports/phase3_mot_report_manifest.json`.

The complete raw JSON remains in the private formal archive and is not distributed through Git. Checkpoints and the
VisDrone dataset are also not included.

## Public evidence and integrity

- `phase3_mot_global_summary.json` is a script-derived scalar and experiment-identity summary of the verified raw
  analyzer output.
- `reports/phase3_mot_layer_stability.csv` supplies the six formal layer rows used by the routing figure.
- `reports/phase3_mot_pairwise_agreement.csv` retains all 1,920 formal comparison rows for audit.
- `reports/phase3_mot_expert_utilization.csv` contains 18 cross-seed rows and embeds 90 per-seed utilization values:
  five seeds by six layers by three experts. Seed pairs are not utilization repetitions.
- `SOURCE_SHA256SUMS` is the unchanged checksum index for the complete formal source bundle, including the omitted raw
  JSON.
- `PUBLIC_SHA256SUMS` verifies only files that are actually present in this public evidence directory.

The default public plotting path uses the compact global summary, the formal layer CSV, and the architecture-controls
summary; it does not require the raw JSON:

```bash
python scripts/issue54/build_pr216_visuals.py
```

Maintainers with the verified private source can fail-closed refresh the compact summary without copying the raw JSON
into Git or recording its private path:

```bash
python scripts/issue54/build_pr216_visuals.py \
  --raw-cross-seed-json <PRIVATE_JSON> \
  --refresh-global-summary
```
