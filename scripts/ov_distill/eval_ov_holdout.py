#!/usr/bin/env python3
"""OV-Distill: open-vocabulary evaluation under a true protocol (seen/unseen class split).

Closed-set mAP (val1k.yaml, all 80 classes) only reflects detection quality on
training classes, while the core capability of an OV detector is **generalizing
text prompts to unseen classes**. This script:

  1. deterministically splits the 80 COCO classes into seen/unseen
     (``--unseen-ratio``, default 1/4, i.e. 60 seen / 20 unseen; fixed seed,
     reproducible);
  2. builds unseen-only labels from the full GT label directory (class ids
     remapped to 0..n_unseen-1), written as a standalone validation directory
     plus a yaml;
  3. sets the text prompts with ``set_classes(unseen_names)`` and runs
     ``model.val`` to obtain unseen-class mAP50 / mAP50-95 -- the true
     open-vocabulary metrics.

Usage:
  python scripts/ov_distill/eval_ov_holdout.py \
      --weights runs/ov_s2_train/s5k_q16/weights/best.pt \
      --labels runs/ov_s2_5k/gt_labels --images datasets/coco/val2017 \
      --out runs/ov_s2_5k/ov_holdout --device mps

Note: evaluation images come from the training pool (full val2017); what is
measured is "text generalization to unseen classes", not "unseen images".
Image-level generalization is covered by the closed-set val1k protocol; the
two complement each other.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.make_coco128_manifest import COCO_NAMES  # noqa: E402


def split_classes(unseen_ratio: float = 0.25, seed: int = 0) -> Tuple[List[str], List[str]]:
    """Deterministically split class names into seen/unseen (sampled after sorting by name; independent of COCO id order)."""
    names = sorted(COCO_NAMES)
    rng = random.Random(seed)
    n_unseen = max(1, round(len(names) * unseen_ratio))
    unseen = sorted(rng.sample(names, n_unseen))
    seen = [n for n in names if n not in set(unseen)]
    return seen, unseen


def build_unseen_val(
    labels: Path, images: Path, out: Path, unseen: List[str], max_images: int = 0
) -> Path:
    """Build an unseen-only validation set from the full GT (label remapping + image symlinks + yaml).

    Returns:
        Path of the generated data yaml.
    """
    name2old = {n: i for i, n in enumerate(COCO_NAMES)}
    remap = {name2old[n]: i for i, n in enumerate(unseen)}
    out_labels = out / "val" / "labels"
    out_images = out / "val" / "images"
    out_labels.mkdir(parents=True, exist_ok=True)
    out_images.mkdir(parents=True, exist_ok=True)

    kept = 0
    for lab in sorted(labels.glob("*.txt")):
        lines = []
        for line in lab.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            cid = int(p[0])
            if cid in remap:
                lines.append(f"{remap[cid]} " + " ".join(p[1:5]))
        if not lines:
            continue  # images without unseen instances are excluded from the validation set
        (out_labels / lab.name).write_text("\n".join(lines) + "\n")
        src = images / f"{lab.stem}.jpg"
        dst = out_images / src.name
        if src.exists() and not dst.exists():
            dst.symlink_to(src.resolve())
        kept += 1
        if max_images and kept >= max_images:
            break

    yaml_path = out / "ov_unseen.yaml"
    names_yaml = "\n".join(f"  {i}: {n}" for i, n in enumerate(unseen))
    yaml_path.write_text(
        # check_det_dataset mandates a train key; placeholder only, never read during validation
        f"path: {out.resolve()}\ntrain: val/images\nval: val/images\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )
    return yaml_path


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="OV seen/unseen class-split evaluation")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--labels", required=True, help="full GT label directory (YOLO txt, COCO 80-class ids)")
    parser.add_argument("--images", required=True, help="image directory (.jpg files matching label names)")
    parser.add_argument("--out", required=True, help="output directory (unseen validation set + results)")
    parser.add_argument("--unseen-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=0, help="limit the number of validation images (0 = all)")
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--build-only", action="store_true", help="only build the unseen validation set; skip val")
    args = parser.parse_args()

    seen, unseen = split_classes(args.unseen_ratio, args.seed)
    print(f"[ov-holdout] seen={len(seen)} unseen={len(unseen)}: {unseen}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    yaml_path = build_unseen_val(Path(args.labels), Path(args.images), out, unseen, args.max_images)
    print(f"[ov-holdout] unseen val yaml: {yaml_path}")

    if args.build_only:
        return

    from ultralytics import YOLOE

    model = YOLOE(args.weights)
    model.set_classes(unseen, model.get_text_pe(unseen))
    r = model.val(
        data=str(yaml_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(out),
        name="val_unseen",
        exist_ok=True,
        verbose=False,
        plots=False,
    )
    report = {
        "weights": args.weights,
        "unseen_classes": unseen,
        "n_unseen": len(unseen),
        "mAP50": round(float(r.box.map50), 4),
        "mAP50-95": round(float(r.box.map), 4),
        "precision": round(float(r.box.mp), 4),
        "recall": round(float(r.box.mr), 4),
    }
    (out / "ov_holdout_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
