#!/usr/bin/env python3
"""Compute mAP50-95 from dumped predictions (lines: 'class conf x1 y1 x2 y2', pixel
xyxy) vs VisDrone YOLO-format GT, reusing ultralytics' matching + DetMetrics so the
number is directly comparable to ultralytics `.val()` (PyTorch / ONNX)."""
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
        for ln in open(path):
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
        for ln in open(path):
            p = ln.split()
            if len(p) < 6:
                continue
            c.append(int(float(p[0]))); s.append(float(p[1])); b.append([float(x) for x in p[2:6]])
    return (torch.tensor(b, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(s, dtype=torch.float32), torch.tensor(c, dtype=torch.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="dir of per-image prediction txts")
    ap.add_argument("--images", default="/data/datasets/VisDrone/images/val")
    ap.add_argument("--labels", default="/data/datasets/VisDrone/labels/val")
    args = ap.parse_args()

    metrics = DetMetrics()
    metrics.names = NAMES
    imgs = sorted(glob.glob(os.path.join(args.images, "*.jpg")))
    for img in imgs:
        stem = Path(img).stem
        w, h = Image.open(img).size
        gt_b, gt_c = load_gt(os.path.join(args.labels, stem + ".txt"), w, h)
        pb, ps, pc = load_pred(os.path.join(args.preds, stem + ".txt"))
        N, M = pb.shape[0], gt_b.shape[0]
        tp = (np.zeros((N, 10), dtype=bool) if (M == 0 or N == 0)
              else match_predictions(pc, gt_c, box_iou(gt_b, pb)).cpu().numpy())
        metrics.update_stats({
            "tp": tp,
            "target_cls": gt_c.numpy(),
            "target_img": np.unique(gt_c.numpy()),
            "conf": ps.numpy() if N else np.zeros(0),
            "pred_cls": pc.numpy() if N else np.zeros(0),
        })
    metrics.process()
    print(f"images={len(imgs)}  mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
