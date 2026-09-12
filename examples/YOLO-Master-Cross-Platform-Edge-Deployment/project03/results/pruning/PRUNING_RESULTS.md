# Project03 MoE pruning results (full-val, A100 pod; CPU cross-check for AI-TOD)

Scripts: scripts/project03/{diagnose_moe,prune_moe}.py @ dev/project03.
Budget: mAP50-95 loss < 0.005. Exploratory EsMoE rows use min-keep=1 +
usage_weight importance (k==E soft mixture: expert removal is NOT free by
construction; those rows quantify the cost rather than claim a win).

| model / dataset | base params | base mAP50-95 | best cut | pruned params | dmAP50-95 | kept experts |
|---|---|---|---|---|---|---|
| esmoe-n-coco-explore | 2.6944M | 0.4272 | 0.0% | 2.6944M | +0.0 | {"3": 3, "6": 3, "9": 3, "12": 3} |
| esmoe-n-visdrone-explore | 2.6641M | 0.20363 | 0.0% | 2.6641M | +0.0 | {"3": 3, "6": 3, "9": 3, "12": 3} |
| esmoe-n-visdrone | 2.6641M | 0.20363 | 0.0% | 2.6641M | +0.0 | {"3": 3, "6": 3, "9": 3, "12": 3} |
| uomoe-n-aitodv2 | 7.4171M | 0.19809 | 56.93% | 3.1948M | +6e-05 | {"5": 2, "8": 2, "11": 2} |
| uomoe-n-coco | 7.4477M | 0.41379 | 56.69% | 3.2255M | +2e-05 | {"5": 2, "8": 2, "11": 2} |
| uomoe-n-visdrone | 7.4175M | 0.19939 | 56.92% | 3.1952M | -0.00023 | {"5": 2, "8": 2, "11": 2} |
| v01-n-aitodv2 | 7.5164M | 0.12024 | 56.18% | 3.2938M | -0.00118 | {"5": 2, "8": 2, "11": 2} |
| v01-n-coco | 7.5546M | 0.42916 | 55.89% | 3.3321M | -1e-05 | {"6": 2, "9": 2, "12": 2} |
| v01-n-visdrone | 7.5167M | 0.17071 | 56.18% | 3.2942M | +4e-05 | {"5": 2, "8": 2, "11": 2} |
| uomoe-n-aitodv2 (CPU x-check) | 7.4171M | 0.19807 | 56.93% | 3.1948M | +0.00011 | {"5": 2, "8": 2, "11": 2} |

## Exploratory EsMoE full sweep rows (cost quantification)

### esmoe-n-coco-explore
- t=0.1: kept {"3": 3, "6": 3, "9": 3, "12": 3}, cut 0.0%, dmAP50-95 +0.0
- t=0.15: kept {"3": 2, "6": 3, "9": 3, "12": 3}, cut 0.22%, dmAP50-95 -0.29862
- t=0.2: kept {"3": 1, "6": 3, "9": 3, "12": 3}, cut 0.49%, dmAP50-95 -0.42644
- t=0.3: kept {"3": 1, "6": 2, "9": 2, "12": 2}, cut 5.11%, dmAP50-95 -0.4272
### esmoe-n-visdrone-explore
- t=0.1: kept {"3": 3, "6": 3, "9": 3, "12": 3}, cut 0.0%, dmAP50-95 +0.0
- t=0.15: kept {"3": 3, "6": 3, "9": 3, "12": 3}, cut 0.0%, dmAP50-95 +0.0
- t=0.2: kept {"3": 3, "6": 3, "9": 3, "12": 3}, cut 0.0%, dmAP50-95 +0.0
- t=0.3: kept {"3": 2, "6": 2, "9": 3, "12": 3}, cut 1.08%, dmAP50-95 -0.08005

## Diagnosis highlights

- YOLO-Master-EsMoE-N: baseline {'mAP50': 0.58829, 'mAP50_95': 0.4272}, max layer Gini 0.0, DEGENERATE hit shares (dense router)
- esmoe-n-coco: baseline {'mAP50': 0.58829, 'mAP50_95': 0.4272}, max layer Gini 0.0, DEGENERATE hit shares (dense router)
- esmoe-n-visdrone: baseline {'mAP50': 0.3504, 'mAP50_95': 0.20363}, max layer Gini 0.0, DEGENERATE hit shares (dense router)
- uomoe-n-coco: baseline {'mAP50': 0.57322, 'mAP50_95': 0.41379}, max layer Gini 0.822858
- uomoe-n-visdrone: baseline {'mAP50': 0.34099, 'mAP50_95': 0.19939}, max layer Gini 0.570908
- v01-n-coco: baseline {'mAP50': 0.59311, 'mAP50_95': 0.42916}, max layer Gini 0.875
- v01-n-visdrone: baseline {'mAP50': 0.29615, 'mAP50_95': 0.17071}, max layer Gini 0.708863
- yolo-master-UoMoE-N_aitodv2_best: baseline {'mAP50': 0.44268, 'mAP50_95': 0.19809}, max layer Gini 0.817978
- yolo-master-v0.1-N_aitodv2_best: baseline {'mAP50': 0.27989, 'mAP50_95': 0.12024}, max layer Gini 0.693091
