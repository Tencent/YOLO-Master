#!/usr/bin/env python3
"""Plot per-epoch mAP curves for the Issue #49 VisDrone baseline runs.

Reads the two per-epoch results.csv logs and draws mAP50 / mAP50-95 over
epochs for both YOLO-Master-v0.1-N and YOLO-Master-EsMoE-N. Pure local
matplotlib, no network needed.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FILES = {
    "v0.1-N": HERE / "visdrone_v01_results.csv",
    "EsMoE-N": HERE / "visdrone_esmoe_results.csv",
}
COLORS = {"v0.1-N": "#1f77b4", "EsMoE-N": "#d62728"}


def load(path: Path):
    epochs, m50, m5095 = [], [], []
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                epochs.append(int(float(row["epoch"])))
                m50.append(float(row["metrics/mAP50(B)"]))
                m5095.append(float(row["metrics/mAP50-95(B)"]))
            except (KeyError, ValueError):
                continue
    return epochs, m50, m5095


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
for ax, (key, label) in zip(axes, [("mAP50", "mAP50"), ("mAP50-95", "mAP50-95")]):
    for name, path in FILES.items():
        ep, m50, m5095 = load(path)
        y = m50 if key == "mAP50" else m5095
        ax.plot(ep, y, marker="o", color=COLORS[name], label=name, linewidth=2)
        for x, yy in zip(ep, y):
            ax.annotate(f"{yy:.4f}", (x, yy), textcoords="offset points",
                        xytext=(0, 8), fontsize=7, color=COLORS[name], ha="center")
    ax.set_title(f"Issue #49 — VisDrone {label} vs epoch")
    ax.set_xlabel("epoch")
    ax.set_ylabel(label)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

fig.suptitle("YOLO-Master Issue #49: VisDrone baseline (v0.1-N vs EsMoE-N, 5 epochs, CPU)",
             fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.95))
out = HERE / "issue49_training_curves.png"
fig.savefig(out, dpi=130)
print("saved", out)
