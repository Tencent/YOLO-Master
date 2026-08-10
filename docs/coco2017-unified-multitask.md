# COCO 2017 Unified Multi-Task Dataset

`scripts/prepare_coco2017_unified_multitask.py` creates a non-destructive, reproducible index over local COCO 2017
data. It leaves the original images and official annotations in place and produces only image lists, JSONL manifests,
a report, and a training YAML.

```bash
python scripts/prepare_coco2017_unified_multitask.py \
  --dataset-root "${YOLO_MASTER_COCO_ROOT:-$HOME/datasets/coco2017}" \
  --output-dir "${YOLO_MASTER_COCO_ROOT:-$HOME/datasets/coco2017}/unified_multitask" \
  --overwrite
```

The resulting `coco2017_mot_multitask.yaml` is directly supported by the Mixture-of-Transformers model. It enables
COCO-derived classification and available local dense labels immediately; after the official Panoptic archive is
installed, it also enables the 133-class semantic branch.

| Task | Source | Current status |
| --- | --- | --- |
| Detection | COCO instances | Enabled |
| Instance segmentation | COCO polygon instances | Enabled |
| Human pose | COCO person keypoints (17 x 3) | Enabled |
| Image multi-label classification | Derived from instance classes | Enabled as COCO-80 multi-hot BCE |
| Captioning | COCO captions | Manifest-only; no caption decoder/loss |
| Depth and surface normals | Local auxiliary PNGs | Enabled where a PNG exists; missing samples are masked |
| Panoptic | Official Panoptic PNG + JSON | Enabled as 133-class semantic CE after installation |
| COCO Stuff | Official Stuff PNG + JSON | Optional 171-class semantic source, with JSON category-ID mapping |
| Oriented boxes | Not native to COCO | Requires a different labelled source |

The generator records task availability for every image. This is important because keypoints are present only for
person instances and missing supervision must not be converted into negative targets.

Install official semantic annotations before Panoptic training:

```bash
python scripts/download_coco2017_dense_annotations.py \
  --dataset-root "${YOLO_MASTER_COCO_ROOT:-$HOME/datasets/coco2017}" --tasks panoptic stuff
```

Train the complete MPS-friendly MoT configuration with:

```bash
python -c '
from ultralytics import YOLO
model = YOLO("scripts/yolo26-master-mt-dense-mps-local.yaml", task="multitask")
model.train(
    data="scripts/coco2017_multitask_panoptic.yaml",
    epochs=100,
    imgsz=320,
    batch=8,
    device="mps",
    workers=0,
    optimizer="AdamW",
    project="runs/coco2017_mps",
    name="mot-unified-panoptic-2k",
)
'
```

Use the supplied two-image train and validation lists for a dense smoke run while the full archive is installing. The
lists intentionally use their respective COCO splits; a train-image list cannot be validated against the `val2017`
annotation JSON.

```bash
python -c '
from ultralytics import YOLO
model = YOLO("scripts/yolo26-master-mt-dense-mps-local.yaml", task="multitask")
model.train(data="scripts/coco2017_multitask_dense_smoke.yaml", epochs=1, imgsz=128, batch=2,
            device="mps", workers=0, optimizer="AdamW", val=False, project="runs/coco2017_mps",
            name="mot-dense-aux-smoke")
'
```

The full dataset has 118,287 training images and 5,000 validation images. On MPS, use staged resolution training
rather than a single long 640-pixel run. The local depth and normal PNGs currently align with train images only, so
those two losses are expected to be zero on the standard val split; this is correct masking, not a failure.
