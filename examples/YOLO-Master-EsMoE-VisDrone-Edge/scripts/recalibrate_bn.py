#!/usr/bin/env python3
"""Recalibrate BatchNorm running stats for a finetuned ES-MoE checkpoint.

Diagnosis
---------
ES-MoE's dynamic routing produces DIFFERENT feature distributions in train vs eval
mode (stochastic routing during training, deterministic top-k during eval). Standard
finetuning updates BN running stats from *train-mode* features, so at inference
(eval mode) the stored running_mean/running_var no longer match the actual feature
distribution -> BN output collapses -> class scores -> ~0 -> 0 detections.

Isolation proof (best.pt, same image):
  A. full eval (running stats)            cls max = 0.0001   (broken)
  B. eval routing + BN using batch stats  cls max = 0.7592   (healthy)
  C. full train (batch stats)             cls logit = 12.5   (healthy)
=> eval-mode features are FINE; only the stored BN running stats are wrong.

Fix
---
Run the model in EVAL mode (correct MoE routing) but with BN layers in TRAIN mode
(momentum=None -> cumulative average), pushing batches of real VisDrone data through
so the running stats converge to the eval-mode feature distribution. No weight update.

Usage:
    python scripts/recalibrate_bn.py --src runs/train/esmoe_n_visdrone/weights/best.pt \
        --dst runs/train/esmoe_n_visdrone/weights/best_recal.pt --device 0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from ultralytics import YOLO

PROJ_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--dst", type=Path, required=True)
    p.add_argument("--images", type=Path, default=PROJ_ROOT.parent / "datasets" / "VisDrone" / "images" / "train")
    p.add_argument("--n", type=int, default=512, help="number of calibration images")
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", type=str, default="0")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    import cv2
    import numpy as np

    imgs = sorted(p for p in args.images.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg"))[: args.n]
    assert len(imgs) >= 128, f"need >=128 calibration images, got {len(imgs)}"
    print(f"[recal] {len(imgs)} images, src={args.src}")

    model = YOLO(str(args.src)).model.to(device)
    model.eval()  # MoE in eval (deterministic) routing mode

    # Put every BN into TRAIN mode with momentum=None (cumulative average over all
    # calibration batches) so running stats track the eval-mode feature distribution.
    bn_count = 0
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.training = True
            m.momentum = None
            m.reset_running_stats()
            bn_count += 1
    print(f"[recal] {bn_count} BN layers in recalibration mode (momentum=None)")

    def letterbox(bgr, sz=640):
        h, w = bgr.shape[:2]
        r = min(sz / h, sz / w)
        nw, nh = int(w * r), int(h * r)
        rs = cv2.resize(bgr, (nw, nh))
        pad = np.full((sz, sz, 3), 114, np.uint8)
        pad[:nh, :nw] = rs
        return torch.from_numpy(pad[:, :, ::1].copy() if False else pad[:, :, ::-1].copy()).permute(2, 0, 1).float() / 255.0

    with torch.no_grad():
        for i in range(0, len(imgs), args.batch):
            chunk = imgs[i : i + args.batch]
            batch = torch.stack([letterbox(cv2.imread(str(p))) for p in chunk]).to(device)
            model(batch)  # forward only; BN running stats update
            if (i // args.batch) % 5 == 0:
                print(f"[recal] batch {i // args.batch}/{(len(imgs) + args.batch - 1) // args.batch}")

    # Lock in: full eval mode, verify class scores recovered.
    model.eval()
    smp = cv2.imread(str(imgs[0]))
    x = letterbox(smp)[None].to(device)
    with torch.no_grad():
        out = model(x)
        out = out[0] if isinstance(out, (list, tuple)) else out
    print(f"[recal] post-recal eval cls max = {out[:, 4:, :].max().item():.4f} (was ~0.0001)")

    # Save via the YOLO API so metadata/weights round-trip cleanly.
    yolo = YOLO(str(args.src))
    yolo.model.load_state_dict(model.state_dict())
    args.dst.parent.mkdir(parents=True, exist_ok=True)
    yolo.save(str(args.dst))
    print(f"[recal] saved -> {args.dst}")


if __name__ == "__main__":
    main()
