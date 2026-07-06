#!/usr/bin/env python3
"""Generate comparison plots from VOC training results."""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def load_csv(name):
    p = ROOT / "runs" / "moe_voc_compare" / name / "results.csv"
    if not p.exists():
        return []
    with open(p) as f:
        return [{k.strip(): float(v) for k, v in r.items() if k.strip()} for r in csv.DictReader(f)]

v6 = load_csv("moe_v0_6_voc")
v12 = load_csv("moe_v0_12_voc")

if not v6 or not v12:
    print("Missing data"); sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib not available, skipping plots"); sys.exit(0)

e6 = [r["epoch"] for r in v6]
e12 = [r["epoch"] for r in v12]

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MoE v0_6 vs v0_12 — VOC Subset (30 epochs, MPS, batch=16)", fontsize=14, fontweight="bold")

# mAP50-95
ax = axes[0, 0]
ax.plot(e6, [r["metrics/mAP50-95(B)"] for r in v6], "b-o", ms=3, label="v0_6 (HybridAdaptiveGateMoE)")
ax.plot(e12, [r["metrics/mAP50-95(B)"] for r in v12], "r-s", ms=3, label="v0_12 (OptimalHybridGateMoE)")
ax.set_title("mAP50-95 Convergence")
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP50-95"); ax.legend(); ax.grid(True, alpha=0.3)

# mAP50
ax = axes[0, 1]
ax.plot(e6, [r["metrics/mAP50(B)"] for r in v6], "b-o", ms=3, label="v0_6")
ax.plot(e12, [r["metrics/mAP50(B)"] for r in v12], "r-s", ms=3, label="v0_12")
ax.set_title("mAP50 Convergence")
ax.set_xlabel("Epoch"); ax.set_ylabel("mAP50"); ax.legend(); ax.grid(True, alpha=0.3)

# Box loss
ax = axes[1, 0]
ax.plot(e6, [r["train/box_loss"] for r in v6], "b-o", ms=3, label="v0_6 train")
ax.plot(e12, [r["train/box_loss"] for r in v12], "r-s", ms=3, label="v0_12 train")
ax.plot(e6, [r["val/box_loss"] for r in v6], "b--", ms=2, label="v0_6 val", alpha=0.6)
ax.plot(e12, [r["val/box_loss"] for r in v12], "r--", ms=2, label="v0_12 val", alpha=0.6)
ax.set_title("Box Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(True, alpha=0.3)

# Per-epoch delta
ax = axes[1, 1]
min_len = min(len(v6), len(v12))
deltas = [v12[i]["metrics/mAP50-95(B)"] - v6[i]["metrics/mAP50-95(B)"] for i in range(min_len)]
colors = ["#27ae60" if d >= 0 else "#e74c3c" for d in deltas]
ax.bar(range(1, min_len+1), deltas, color=colors, alpha=0.7)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("Per-Epoch mAP50-95 Delta (v0_12 - v0_6)")
ax.set_xlabel("Epoch"); ax.set_ylabel("Delta mAP50-95"); ax.grid(True, alpha=0.3)

plt.tight_layout()
out = ROOT / "runs" / "moe_voc_compare" / "comparison_plots.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Plots saved to: {out}")

# Final summary table
print(f"\n{'='*60}")
print(f"  FINAL RESULTS (30 epochs, VOC subset 3000/800)")
print(f"{'='*60}")
print(f"{'Metric':<15} {'v0_6':>12} {'v0_12':>12} {'Delta':>12} {'Delta%':>10}")
print("-" * 60)
for col, label in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"),
                   ("metrics/precision(B)", "Precision"), ("metrics/recall(B)", "Recall")]:
    b = v6[-1][col]; n = v12[-1][col]; d = n - b
    print(f"{label:<15} {b:>12.5f} {n:>12.5f} {d:>+12.5f} {100*d/b:>+9.1f}%")
t6 = v6[-1]["time"]; t12 = v12[-1]["time"]
print(f"{'Time(s)':<15} {t6:>12.1f} {t12:>12.1f} {t12-t6:>+12.1f} {100*(t12-t6)/t6:>+9.1f}%")
print(f"{'Time(h)':<15} {t6/3600:>12.2f} {t12/3600:>12.2f} {(t12-t6)/3600:>+12.2f}")
print(f"{'='*60}")
