#!/usr/bin/env python3
"""Run all backends on the VisDrone val set and compare mAP to the PyTorch baseline.

Meets issue #51:
  * >=500 validation images (full VisDrone val split = 548).
  * mAP50-95 error target: <0.5% (non-quantized), <1.0% (INT8).
  * On excess error: per-image divergence analysis + saved diff visualizations.

Usage (from edge-src/python):
    python eval_consistency.py --pytorch runs/train/.../weights/best.pt \
        --onnx exports/best.onnx --ncnn exports/best_ncnn_model --mnn exports/best.mnn
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np

from backends import create_detector
from map_eval import evaluate_map
from postprocess import NmsConfig, decode_and_nms
from preprocess import letterbox, scale_boxes


PROJ_ROOT = Path(__file__).resolve().parents[1]
VISDRONE = PROJ_ROOT.parent / "datasets" / "VisDrone"
RESULTS = PROJ_ROOT.parent / "results"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--pytorch", type=Path, required=True)
    p.add_argument("--onnx", type=Path, default=None)
    p.add_argument("--ncnn", type=Path, default=None)
    p.add_argument("--mnn", type=Path, default=None)
    p.add_argument("--val-images", type=Path, default=VISDRONE / "images" / "val")
    p.add_argument("--val-labels", type=Path, default=VISDRONE / "labels" / "val")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--num-classes", type=int, default=10)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--limit", type=int, default=0, help="0 = all val images")
    p.add_argument("--conf", type=float, default=0.15)
    p.add_argument("--small-conf", type=float, default=0.05)
    p.add_argument(
        "--mode",
        choices=["sparse", "combine", "full"],
        default="sparse",
        help="PyTorch baseline routing: sparse(native eval) | combine(dense expert combine, "
        "keep topk — matches ONNX/MNN export) | full(dense + full-softmax routing — matches NCNN export)",
    )
    p.add_argument("--out", type=Path, default=RESULTS / "consistency")
    return p.parse_args()


def load_image_list(img_dir: Path, limit: int) -> List[Path]:
    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in (".jpg", ".png", ".jpeg")])
    return imgs[:limit] if limit > 0 else imgs


def load_gt(label_path: Path, img_w: int, img_h: int, num_classes: int) -> List[Tuple[int, np.ndarray]]:
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        c = int(parts[0])
        if c >= num_classes:
            continue
        cx, cy, w, h = map(float, parts[1:5])
        x1 = (cx - w / 2) * img_w
        y1 = (cy - h / 2) * img_h
        x2 = (cx + w / 2) * img_w
        y2 = (cy + h / 2) * img_h
        out.append((c, np.array([x1, y1, x2, y2], dtype=np.float32)))
    return out


def run_backend(kind: str, model_path: Path, images: List[Path], args) -> Tuple[Dict, float]:
    """Returns (detections dict, total forward seconds)."""
    cfg = NmsConfig(conf=args.conf, small_conf=args.small_conf, num_classes=args.num_classes)
    if kind == "pytorch":
        from backends import PyTorchDetector
        det = PyTorchDetector(str(model_path), args.device, args.num_classes, mode=args.mode)
    else:
        det = create_detector(kind, str(model_path), args.device, args.num_classes)
    detections: Dict[str, List] = {}
    total_fwd = 0.0
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        lb = letterbox(img, args.imgsz)
        t0 = time.perf_counter()
        raw = det.forward(lb.image)
        total_fwd += time.perf_counter() - t0
        boxes, scores, cls = decode_and_nms(raw, cfg)
        boxes = scale_boxes(boxes, lb)
        key = img_path.name
        detections[key] = [(int(c), float(s), b) for c, s, b in zip(cls, scores, boxes)]
    return detections, total_fwd


def collect_gt(images: List[Path], args) -> Dict[str, List]:
    gts = {}
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        gts[img_path.name] = load_gt(args.val_labels / (img_path.stem + ".txt"), w, h, args.num_classes)
    return gts


def divergence_images(base: Dict, other: Dict, topk: int = 5) -> List[Tuple[str, int]]:
    """Images with the largest detection-count divergence between two backends."""
    keys = set(base) | set(other)
    scored = [(k, abs(len(base.get(k, [])) - len(other.get(k, [])))) for k in keys]
    scored.sort(key=lambda x: -x[1])
    return scored[:topk]


def visualize_divergence(img_path: Path, base_dets, other_dets, out_path: Path, label: str):
    img = cv2.imread(str(img_path))
    panel = img.copy()
    for _, _, b in base_dets:
        b = b.astype(int)
        cv2.rectangle(panel, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2)  # green = pytorch
    for _, _, b in other_dets:
        b = b.astype(int)
        cv2.rectangle(panel, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)  # red = other
    cv2.putText(panel, f"green=pytorch red={label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)


def main():
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    images = load_image_list(args.val_images, args.limit)
    assert len(images) >= 500 or args.limit > 0, f"only {len(images)} val images (need >=500)"
    print(f"[eval] {len(images)} val images, imgsz={args.imgsz}, device={args.device}")

    gts = collect_gt(images, args)
    print(f"[eval] collected GT for {len(gts)} images")

    backends = [("pytorch", args.pytorch)]
    if args.onnx:
        backends.append(("onnx", args.onnx))
    if args.ncnn:
        backends.append(("ncnn", args.ncnn))
    if args.mnn:
        backends.append(("mnn", args.mnn))

    results = {}
    for kind, path in backends:
        if path is None or not Path(path).exists():
            print(f"[eval] skip {kind} (no model)")
            continue
        print(f"[eval] running {kind}: {path}")
        dets, fwd = run_backend(kind, Path(path), images, args)
        m = evaluate_map(dets, gts, args.num_classes)
        m["latency_ms"] = fwd / max(len(dets), 1) * 1000
        m["fps"] = len(dets) / fwd if fwd > 0 else 0
        results[kind] = m
        print(f"   {kind}: mAP50={m['mAP50']:.4f} mAP50-95={m['mAP50-95']:.4f} "
              f"latency={m['latency_ms']:.2f}ms fps={m['fps']:.1f}")

    # Delta vs PyTorch baseline
    if "pytorch" in results:
        base = results["pytorch"]["mAP50-95"]
        summary = {}
        for kind, r in results.items():
            delta = r["mAP50-95"] - base
            r["delta_map5095"] = delta
            summary[kind] = {
                "mAP50-95": r["mAP50-95"],
                "delta_vs_pytorch": delta,
                "within_target_0.5pct": abs(delta) < 0.005,
                "latency_ms": r["latency_ms"],
                "fps": r["fps"],
            }
            if kind != "pytorch":
                tag = "OK" if abs(delta) < 0.005 else "EXCEED"
                print(f"[delta] {kind:8s} ΔmAP50-95={delta*100:+.3f}%  [{tag}]")

        (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\n[eval] summary -> {args.out / 'summary.json'}")

    (args.out / "full_metrics.json").write_text(json.dumps(results, indent=2, default=str))
    print(f"[eval] full metrics -> {args.out / 'full_metrics.json'}")


if __name__ == "__main__":
    main()
