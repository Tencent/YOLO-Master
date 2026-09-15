# Project03 UoMoE/EsMoE pruning results (full-val, A100 pod + CPU cross-check)

Scripts: scripts/project03/{diagnose_moe,prune_moe}.py @ dev/project03.
Budget: mAP50-95 loss < 0.005. All numbers are full validation sets.

## Sweep results

| model / dataset | base params | base mAP50-95 | best cut | pruned params | dmAP50-95 | kept experts |
|---|---|---|---|---|---|---|
| esmoe-n-visdrone | 2.6641M | 0.20363 | 0.0% | 2.6641M | +0.0 | {"3": 3, "6": 3, "9": 3, "12": 3} |
| uomoe-n-aitodv2 | 7.4171M | 0.19809 | 56.93% | 3.1948M | +6e-05 | {"5": 2, "8": 2, "11": 2} |
| uomoe-n-coco | 7.4477M | 0.41379 | 56.69% | 3.2255M | +2e-05 | {"5": 2, "8": 2, "11": 2} |
| uomoe-n-visdrone | 7.4175M | 0.19939 | 56.92% | 3.1952M | -0.00023 | {"5": 2, "8": 2, "11": 2} |
| uomoe-n-aitodv2 (CPU x-check) | 7.4171M | 0.19807 | 56.93% | 3.1948M | +0.00011 | {"5": 2, "8": 2, "11": 2} |

## Diagnosis highlights

- esmoe-n-visdrone: baseline {'mAP50': 0.3504, 'mAP50_95': 0.20363}, max layer Gini 0.0, DEGENERATE hit shares (dense router; usage_weight importance used)
- uomoe-n-coco: baseline {'mAP50': 0.57322, 'mAP50_95': 0.41379}, max layer Gini 0.822858
- uomoe-n-visdrone: baseline {'mAP50': 0.34099, 'mAP50_95': 0.19939}, max layer Gini 0.570908
- yolo-master-UoMoE-N_aitodv2_best: baseline {'mAP50': 0.44268, 'mAP50_95': 0.19809}, max layer Gini 0.817978

Full artifacts (usage stats, scene analysis, per-threshold plans/ckpt metadata,
sweep CSVs and curves, upstream heatmaps) in the accompanying tarball.
