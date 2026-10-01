#!/usr/bin/env python3
"""Standalone mAP50 / mAP50-95 — numpy only (no ultralytics/torch/cv2/PIL required).

Scores per-image prediction txts ('class conf x1 y1 x2 y2', pixel xyxy) against VisDrone
YOLO-format labels, replicating ultralytics' matching + AP (so numbers match eval_map.py).
Runs on-device (e.g. Jetson) where preds+labels+images already live.

  python3 eval_map_standalone.py --preds preds_fp16 --images images/val --labels labels/val
"""
import argparse, glob, os, struct
import numpy as np

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2.0 renamed trapz->trapezoid


def jpeg_size(path):
    """(w, h) from a JPEG header — pure python, no image libs."""
    with open(path, "rb") as f:
        f.read(2)  # SOI
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            m = marker[0]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                f.read(3)                      # len(2)+precision(1)
                h, w = struct.unpack(">HH", f.read(4))
                return w, h
            else:
                (seg_len,) = struct.unpack(">H", f.read(2))
                f.seek(seg_len - 2, 1)


def load_gt(path, w, h):
    if not os.path.exists(path):
        return np.zeros((0, 4)), np.zeros((0,), int)
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    if not lines:
        return np.zeros((0, 4)), np.zeros((0,), int)
    if "," in lines[0]:
        # original VisDrone: x,y,w,h,score,category,trunc,occ (pixel). Matches ultralytics'
        # visdrone2yolo: skip score==0 (ignored regions), class = category-1, keep classes 0..9.
        boxes, cls = [], []
        for l in lines:
            v = l.split(",")
            if v[4] == "0":
                continue
            c = int(v[5]) - 1
            if c < 0 or c > 9:
                continue
            x, y, bw, bh = float(v[0]), float(v[1]), float(v[2]), float(v[3])
            boxes.append([x, y, x + bw, y + bh]); cls.append(c)
        return (np.array(boxes, float).reshape(-1, 4), np.array(cls, int))
    # YOLO: class cx cy w h (normalized)
    a = np.array([l.split() for l in lines], float)
    cls = a[:, 0].astype(int)
    cx, cy, bw, bh = a[:, 1] * w, a[:, 2] * h, a[:, 3] * w, a[:, 4] * h
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    return xyxy, cls


def load_pred(path):
    if not os.path.exists(path):
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
    rows = [r.split() for r in open(path).read().splitlines() if r.strip()]
    if not rows:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,), int)
    a = np.array(rows, float)
    return a[:, 2:6], a[:, 1], a[:, 0].astype(int)   # xyxy, conf, cls


def box_iou(a, b):                                    # (N,4),(M,4) -> (N,M)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


IOUV = np.linspace(0.5, 0.95, 10)


def match(pred_cls, true_cls, iou):                   # -> (N_pred, 10) correct matrix
    correct = np.zeros((pred_cls.shape[0], 10), bool)
    cc = true_cls[:, None] == pred_cls[None, :]        # (M_gt, N_pred)
    iou = iou.T * cc                                   # iou passed (N_pred,M_gt) -> (M,N)
    for k, thr in enumerate(IOUV):
        gt_i, pr_i = np.nonzero(iou >= thr)
        if gt_i.size:
            m = np.stack([gt_i, pr_i, iou[gt_i, pr_i]], 1)
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), k] = True
    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return _trapz(np.interp(x, mrec, mpre), x)


def ap_per_class(tp, conf, pred_cls, target_cls):
    i = np.argsort(-conf)
    tp, pred_cls = tp[i], pred_cls[i]
    classes = np.unique(target_cls)
    ap = np.zeros((len(classes), tp.shape[1]))
    for ci, c in enumerate(classes):
        m = pred_cls == c
        n_gt = int((target_cls == c).sum())
        if m.sum() == 0 or n_gt == 0:
            continue
        fpc = (1 - tp[m]).cumsum(0)
        tpc = tp[m].cumsum(0)
        recall = tpc / (n_gt + 1e-16)
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[ci, j] = compute_ap(recall[:, j], precision[:, j])
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--images", default="images/val")
    ap.add_argument("--labels", default="labels/val")
    a = ap.parse_args()
    imgs = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    all_tp, all_conf, all_pcls, all_tcls = [], [], [], []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        w, h = jpeg_size(p)
        gtb, gtc = load_gt(os.path.join(a.labels, stem + ".txt"), w, h)
        pb, ps, pc = load_pred(os.path.join(a.preds, stem + ".txt"))
        all_tcls.append(gtc)
        if pb.shape[0] == 0:
            continue
        tp = (match(pc, gtc, box_iou(pb, gtb)) if gtb.shape[0]
              else np.zeros((pb.shape[0], 10), bool))
        all_tp.append(tp); all_conf.append(ps); all_pcls.append(pc)
    tp = np.concatenate(all_tp); conf = np.concatenate(all_conf)
    pcls = np.concatenate(all_pcls); tcls = np.concatenate(all_tcls)
    APc = ap_per_class(tp, conf, pcls, tcls)
    print(f"images={len(imgs)}  mAP50={APc[:,0].mean():.4f}  mAP50-95={APc.mean():.4f}")


if __name__ == "__main__":
    main()
