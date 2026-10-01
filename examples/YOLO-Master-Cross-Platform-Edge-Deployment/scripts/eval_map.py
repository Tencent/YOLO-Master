#!/usr/bin/env python3
"""Compute mAP50-95 from dumped predictions (lines: 'class conf x1 y1 x2 y2', pixel
xyxy) vs YOLO-format GT, reusing ultralytics' matching + DetMetrics so the number is
directly comparable to ultralytics `.val()` (PyTorch / ONNX).

Class names default to the VisDrone 10-class set (historic behaviour); pass
`--names-yaml <metadata.yaml|dataset.yaml>` to score any other model (e.g. the COCO
80-class `names:` block of an ncnn export sidecar).

`evaluate()` is importable (no work happens at import time) so other tools, e.g.
`tests/certify_ncnn_int8.py`, can score two prediction dirs in-process.
"""
import argparse, glob, os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from ultralytics.utils.metrics import DetMetrics, box_iou

NAMES = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
         5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor"}
IOUV = torch.linspace(0.5, 0.95, 10)


def match_predictions(pred_cls, true_cls, iou):
    """Exact copy of ultralytics BaseValidator.match_predictions (non-scipy)."""
    correct = np.zeros((pred_cls.shape[0], IOUV.shape[0]), dtype=bool)
    correct_class = true_cls[:, None] == pred_cls               # (M gt, N pred)
    iou = (iou * correct_class).cpu().numpy()
    for i, thr in enumerate(IOUV.tolist()):
        m = np.array(np.nonzero(iou >= thr)).T                  # (K, 2) [gt, pred]
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), i] = True
    return torch.tensor(correct)


def load_gt(path, w, h):
    b, c = [], []
    if os.path.exists(path):
        with open(path) as fh:
            for ln in fh:
                p = ln.split()
                if len(p) < 5:
                    continue
                c.append(int(float(p[0])))
                cx, cy, bw, bh = map(float, p[1:5])
                b.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return torch.tensor(b, dtype=torch.float32).reshape(-1, 4), torch.tensor(c, dtype=torch.int64)


def load_pred(path):
    b, s, c = [], [], []
    if os.path.exists(path):
        with open(path) as fh:
            for ln in fh:
                p = ln.split()
                if len(p) < 6:
                    continue
                c.append(int(float(p[0]))); s.append(float(p[1])); b.append([float(x) for x in p[2:6]])
    return (torch.tensor(b, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(s, dtype=torch.float32), torch.tensor(c, dtype=torch.int64))


def _unquote(s):
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "'\"":
        s = s[1:-1]
    return s


def load_names_yaml(path):
    """Parse the `names:` block of an ultralytics sidecar / dataset yaml into an ordered list.

    Handles the exported-metadata form (`names:` then `  0: person` per line), a dataset
    yaml list form (`  - person`), and the inline `names: [a, b]` form. Quotes around the
    value are stripped. Deliberately line-based (no pyyaml dependency and immune to the
    odd scalars that appear elsewhere in ultralytics sidecars); only the `names:` block
    is read.

    Args:
        path: metadata.yaml (ncnn/onnx export sidecar) or dataset yaml.

    Returns:
        list[str]: class names indexed by class id (0..nc-1).

    Raises:
        SystemExit: no `names:` block, or non-contiguous ids.
    """
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    mapping, listed = {}, []
    in_block, block_indent = False, -1
    for raw in lines:
        line = raw.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if not in_block:
            if indent == 0 and line.startswith("names:"):
                rest = line[len("names:"):].strip()
                if rest.startswith("[") and rest.endswith("]"):          # inline list
                    listed = [_unquote(x) for x in rest[1:-1].split(",") if x.strip()]
                    break
                in_block, block_indent = True, indent
            continue
        if indent <= block_indent:                                      # dedent = end of block
            break
        body = line.strip()
        if body.startswith("- "):
            listed.append(_unquote(body[2:]))
        elif ":" in body:
            k, v = body.split(":", 1)
            k = _unquote(k)
            if k.lstrip("-").isdigit():
                mapping[int(k)] = _unquote(v)
    if listed:
        return listed
    if not mapping:
        raise SystemExit(f"no `names:` block found in {path}")
    ids = sorted(mapping)
    if ids != list(range(len(ids))):
        raise SystemExit(f"non-contiguous class ids in {path}: {ids[:5]}...")
    return [mapping[i] for i in ids]


def evaluate(preds_dir, images_dir, labels_dir, names, limit=0):
    """Score a dir of save-txt predictions against YOLO-format labels with ultralytics DetMetrics.

    Args:
        preds_dir: per-image `<stem>.txt` files with rows `class conf x1 y1 x2 y2` (pixel xyxy).
            A missing file counts as zero predictions for that image.
        images_dir: the `*.jpg` images (sorted; sizes are read to de-normalise the labels).
        labels_dir: YOLO-format `<stem>.txt` labels; a missing file is an empty ground truth.
        names: class names, list indexed by id or `{id: name}` dict.
        limit: score only the first `limit` images in sorted order (0 = all). Matches the
            edge CLI's `--limit N`, which truncates the same sorted listing.

    Returns:
        tuple[float, float, int]: (mAP50, mAP50-95, number of images scored).
    """
    metrics = DetMetrics()
    metrics.names = dict(enumerate(names)) if isinstance(names, (list, tuple)) else dict(names)
    imgs = sorted(glob.glob(os.path.join(images_dir, "*.jpg")))
    if not imgs:
        raise SystemExit(f"no *.jpg images found under {images_dir}")
    if limit and limit > 0:
        imgs = imgs[:limit]
    for img in imgs:
        stem = Path(img).stem
        w, h = Image.open(img).size
        gt_b, gt_c = load_gt(os.path.join(labels_dir, stem + ".txt"), w, h)
        pb, ps, pc = load_pred(os.path.join(preds_dir, stem + ".txt"))
        N, M = pb.shape[0], gt_b.shape[0]
        tp = (np.zeros((N, 10), dtype=bool) if (M == 0 or N == 0)
              else match_predictions(pc, gt_c, box_iou(gt_b, pb)).cpu().numpy())
        metrics.update_stats({
            "tp": tp,
            "target_cls": gt_c.numpy(),
            "target_img": np.unique(gt_c.numpy()),
            "conf": ps.numpy() if N else np.zeros(0),
            "pred_cls": pc.numpy() if N else np.zeros(0),
            "im_name": stem,            # required by ultralytics >= 8.4 (per-image P/R); ignored by older
        })
    metrics.process()
    return float(metrics.box.map50), float(metrics.box.map), len(imgs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preds", required=True, help="dir of per-image prediction txts")
    ap.add_argument("--images", default="/data/datasets/VisDrone/images/val")
    ap.add_argument("--labels", default="/data/datasets/VisDrone/labels/val")
    ap.add_argument("--names-yaml", default="",
                    help="metadata.yaml / dataset yaml whose `names:` block gives the class names "
                         "(default: the VisDrone 10-class set)")
    ap.add_argument("--limit", type=int, default=0, help="score only the first N images (sorted; 0 = all)")
    args = ap.parse_args()

    names = load_names_yaml(args.names_yaml) if args.names_yaml else NAMES
    map50, map5095, n = evaluate(args.preds, args.images, args.labels, names, limit=args.limit)
    print(f"images={n}  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")


if __name__ == "__main__":
    main()
