"""
读取每个变体的 results.csv，画 4 张对比图：
- train/box_loss
- train/cls_loss
- val/cls_loss
- metrics/mAP50-95(B)
保存到 peft_compare_curves.png
"""
import json
from pathlib import Path
import csv
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
RUNS = HERE / "runs"
VARIANTS = ["full", "lora", "dora", "loha", "ia3"]
COLORS = {"full": "#444", "lora": "#1f77b4", "dora": "#d62728", "loha": "#2ca02c", "ia3": "#ff7f0e"}


def read_csv(path):
    rows = list(csv.DictReader(open(path)))
    return rows


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
panels = [
    ("train/box_loss",       "Train Box Loss",       axes[0][0]),
    ("train/cls_loss",       "Train Cls Loss",       axes[0][1]),
    ("val/cls_loss",         "Val Cls Loss",         axes[1][0]),
    ("metrics/mAP50-95(B)",  "Val mAP50-95",         axes[1][1]),
]

for v in VARIANTS:
    csv_path = RUNS / f"v_{v}" / "results.csv"
    if not csv_path.exists():
        print(f"[skip] {csv_path} not found")
        continue
    rows = read_csv(csv_path)
    epochs = [int(r["epoch"]) for r in rows]
    for col, _title, ax in panels:
        col_clean = col.strip()
        ys = [float(r[col_clean]) for r in rows]
        ax.plot(epochs, ys, marker="o", linewidth=2, label=v.upper(), color=COLORS[v])

for col, title, ax in panels:
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.grid(alpha=0.3); ax.legend(fontsize=9)

fig.suptitle("PEFT Variants on COCO128 (yolo11n, 2 epochs, MPS)", fontsize=14, fontweight="bold")
fig.tight_layout()
out = HERE / "peft_compare_curves.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print(f"Saved: {out}")

# 同时把汇总打成 markdown 表
data = json.load(open(HERE / "peft_compare_results.json"))
md = ["# PEFT Variants Comparison (coco128, yolo11n, 2 epochs)\n",
      "| Variant | OK | Total Params | Δ vs Full | adapter signature | mAP50 | mAP50-95 | Time(s) |",
      "|---------|----|--------------|-----------|-------------------|-------|----------|---------|"]
for r in data:
    sig = r["adapter_sig"]
    sig_str = []
    if sig["has_lora_A"]:         sig_str.append("lora_A/B")
    if sig["has_dora_magnitude"]: sig_str.append("DoRA-magnitude")
    if sig["has_loha"]:           sig_str.append("hada")
    if sig["has_ia3"]:            sig_str.append("ia3")
    if not sig_str:               sig_str = ["-"]
    fm = r["final_metrics"]
    delta = r["params_total"] - 2624080
    md.append(f"| **{r['name'].upper()}** | {'✅' if r['ok'] else '❌'} | "
              f"{r['params_total']:,} | {'+' if delta>=0 else ''}{delta:,} | "
              f"{', '.join(sig_str)} | "
              f"{fm.get('metrics/mAP50(B)', 0):.4f} | "
              f"{fm.get('metrics/mAP50-95(B)', 0):.4f} | "
              f"{r['elapsed_sec']:.0f} |")
md_path = HERE / "peft_compare_summary.md"
md_path.write_text("\n".join(md))
print(f"Saved: {md_path}")
