#!/usr/bin/env python3
"""Plot a side-by-side comparison of MoE versions from their results.csv.

Reads runs/<project>/<version>/results.csv for each version and renders
mAP50-95, mAP50 and moe_loss curves plus a final-epoch summary table.

Usage:
    python3 scripts/plot_moe_voc_compare.py \
        --project runs/moe_voc_smoke --versions v0_6 v0_11
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path: Path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return [{k.strip(): v.strip() for k, v in r.items()} for r in rows]


def col(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (ValueError, TypeError):
            out.append(float("nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", type=Path, default=ROOT / "runs/moe_voc_smoke")
    ap.add_argument("--versions", nargs="+", default=["v0_6", "v0_11"])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    project = args.project if args.project.is_absolute() else ROOT / args.project
    data = {v: read_csv(project / v / "results.csv") for v in args.versions}

    metrics = [
        ("metrics/mAP50-95(B)", "mAP50-95"),
        ("metrics/mAP50(B)", "mAP50"),
        ("train/moe_loss", "train moe_loss"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, (key, title) in zip(axes, metrics):
        for v in args.versions:
            rows = data[v]
            if not rows:
                continue
            ep = col(rows, "epoch")
            ax.plot(ep, col(rows, key), marker="o", ms=3, label=v)
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle("YOLO-Master MoE VOC-subset comparison (v0_6 vs v0_11)")
    fig.tight_layout()
    out = args.out or (project / "compare_curves.png")
    fig.savefig(out, dpi=130)
    print(f"[plot] wrote {out}")

    print("\n=== Final-epoch summary ===")
    hdr = f"{'version':<8} {'epoch':>5} {'mAP50-95':>9} {'mAP50':>8} {'precision':>9} {'recall':>8} {'moe_loss':>9}"
    print(hdr)
    print("-" * len(hdr))
    for v in args.versions:
        rows = data[v]
        if not rows:
            print(f"{v:<8} (no results.csv)")
            continue
        # best mAP50-95 row
        best = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", "nan") or "nan")
                   if (r.get("metrics/mAP50-95(B)", "") not in ("", "nan")) else float("-inf"))
        def g(k):
            try:
                return float(best.get(k, ""))
            except ValueError:
                return float("nan")
        print(f"{v:<8} {int(g('epoch')):>5} {g('metrics/mAP50-95(B)'):>9.5f} "
              f"{g('metrics/mAP50(B)'):>8.5f} {g('metrics/precision(B)'):>9.5f} "
              f"{g('metrics/recall(B)'):>8.5f} {g('train/moe_loss'):>9.4f}")


if __name__ == "__main__":
    main()
