# Phase 3 MoT routing evidence

This directory archives the formal five-seed MoT routing evidence used by Tencent/YOLO-Master PR #216. The recorded
formal root is `/root/autodl-tmp/MoT/results/phase3_formal_20260731T214515Z`.

`SOURCE_PROVENANCE.json` records the source paths and hashes, while `SHA256SUMS` verifies the archived formal files.
The highest-level experimental unit is an independently trained seed.

The utilization artifacts use two complementary, non-pairwise representations:

- `reports/phase3_mot_expert_utilization.csv` contains 18 cross-seed summary rows: six layers by three experts.
- `phase3_cross_seed_routing.json` contains 90 per-seed utilization entries: five seeds by six layers by three experts.

Seed pairs are used only for agreement and Jensen-Shannon-divergence comparisons. Pairwise utilization is neither
defined nor required. Checkpoints and datasets are not included in this archive.

Generate the PR figures from the verified evidence with:

```bash
python scripts/issue54/build_pr216_visuals.py
```
