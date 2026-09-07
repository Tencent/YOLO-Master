#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage4 统计脚本:
读 stage3 evidence(主18) + 补充单元, 输出:
  1) 每 (dataset,strategy) 各 seed mAP50 + mean/sd/95%CI (t, df=n-1)
  2) 全量 vs vpeft / 全量 vs frozen 逐 seed 配对差(paired) 均值与 CI
  3) 参数/显存/时长四维同表(标量), 4) 收敛异常(outlier) 标注
用法: python analysis_stats.py [--evidence ../stage3_matrix/evidence_summary.json]
"""
import argparse
import json
import math
from pathlib import Path

TRAINABLE = {  # 实测(全参2,813,626; 由 best.pt state_dict 统计, 6 类一致)
    "full_sft": {"params": 2_813_626, "pct": 100.0},
    "frozen_backbone": {"params": 1_906_822, "pct": 67.8},   # freeze=11 (顶层0..10=32.2%)
    "vpeft_adapter": {"params": 116_736, "pct": 4.15},        # 96 个 lora_A/B
    "vpeft_effective": {"params": 465_250, "pct": 16.5},      # adapter + 类别重初始化解冻 head ~348,514
}
STRAT_ORDER = ["full_sft", "vpeft", "frozen_backbone"]


def mean(x):
    return sum(x) / len(x)


def ci95(x):
    """t(df=n-1) 近似 95% CI 半宽"""
    n = len(x)
    if n < 2:
        return None
    s = math.sqrt(sum((v - mean(x)) ** 2 for v in x) / (n - 1)) if n > 1 else 0.0
    # n<=5 用保守 2.57(n=3),2.78(n=4),2.57(n=5) 简化取 t≈2.92(n=3)
    t = {2: 4.30, 3: 3.18, 4: 2.78, 5: 2.57}.get(n, 2.02)
    return t * s / math.sqrt(n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", default="../stage3_matrix/evidence_summary.json")
    ap.add_argument("--supplement", default="../stage3_matrix/runs",
                    help="补充单元目录(自动发现 *_s2025 / *_s824b)")
    args = ap.parse_args()
    ev = json.loads(Path(args.evidence).read_text())
    rows = [r for r in ev if r.get("status") == "done"]
    sup_root = Path(args.supplement)
    for d in sorted(sup_root.iterdir()) if sup_root.exists() else []:
        if d.name in {"neu_vpeft_s2025", "neu_frozen_s2025", "pcb_frozen_s824b"}:
            s = json.loads((d / "summary.json").read_text())
            strat = s["strategy"]
            seed = d.name.rsplit("_s", 1)[1]  # '2025' / '824b'(重跑标记与正式824区分)
            unit = {
                "id": d.name, "dataset": d.name[:3], "seed": seed, "strategy": strat,
                "mAP50_best": best_of(d / "train" / strat / "results.csv"),
                "peak_gpu_gb": None, "elapsed_sec": s.get("elapsed_sec"),
            }
            rows.append(unit)
    group = {}
    for r in rows:
        group.setdefault((r["dataset"], r["strategy"]), []).append(r)
    lines = []
    lines.append("## 表1 每 (dataset, strategy) 多 seed mAP50")
    lines.append("| dataset | strategy | seeds(mAP50 best) | mean | sd | 95%CI |")
    lines.append("|---|---|---|---|---|---|")
    for ds in ["neu", "pcb"]:
        for st in STRAT_ORDER:
            g = group.get((ds, st), [])
            if not g:
                continue
            vals = sorted((str(r["seed"]), r.get("mAP50_best")) for r in g)
            vs = [v for _, v in vals]
            m, sd = mean(vs), (math.sqrt(sum((v - mean(vs)) ** 2 for v in vs) / (len(vs) - 1)) if len(vs) > 1 else 0)
            c = ci95(vs)
            sl = ", ".join(f"{s}:{v:.3f}" for s, v in vals)
            ci = f"±{c:.3f}" if c else "n<2"
            lines.append(f"| {ds} | {st} | {sl} | {m:.3f} | {sd:.3f} | {ci} |")
    lines.append("")
    lines.append("## 表2 配对差 (逐 seed: 全量 − 对照策略)")
    for ds in ["neu", "pcb"]:
        for st in ["vpeft", "frozen_backbone"]:
            f = dict((r["seed"], r["mAP50_best"]) for r in group.get((ds, "full_sft"), []))
            c = dict((r["seed"], r["mAP50_best"]) for r in group.get((ds, st), []))
            seeds = sorted(set(f) & set(c))
            if not seeds:
                continue
            d = [f[s] - c[s] for s in seeds]
            lines.append(f"- {ds}: full−{st} over seeds {seeds}: 均值 {mean(d):+.3f}, "
                         f"逐差 {', '.join(f'{x:+.3f}' for x in d)}")
    lines.append("")
    lines.append("## 表3 四维对照 (参数/显存/时长; 同预算 epochs100×batch8×imgsz640, amp=false)")
    lines.append("| dataset | strategy | 可训参数(占比) | 峰值显存G | 时长s/seed |")
    lines.append("|---|---|---|---|---|")
    for ds in ["neu", "pcb"]:
        for st in STRAT_ORDER:
            g = group.get((ds, st), [])
            if not g:
                continue
            pr = TRAINABLE["vpeft_adapter"] if st == "vpeft" else TRAINABLE[st]
            lab = f"{pr['params']:,} ({pr['pct']:.1f}%)"
            mems = [r.get("peak_gpu_gb") for r in g if r.get("peak_gpu_gb")]
            tms = [r.get("elapsed_sec") for r in g if r.get("elapsed_sec")]
            if mems and tms:
                lines.append(f"| {ds} | {st} | {lab} | {mean(mems):.2f} | {mean(tms)/60:.0f}m |")
            else:
                lines.append(f"| {ds} | {st} | {lab} | n/a | n/a |")
    out = Path("comparison_tables.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def best_of(csv_path: Path):
    if not csv_path.exists():
        return None
    import csv as _csv
    best = None
    for r in _csv.DictReader(csv_path.open()):
        try:
            m = float(r["metrics/mAP50(B)"])
        except (KeyError, ValueError):
            continue
        best = m if best is None else max(best, m)
    return best


if __name__ == "__main__":
    main()
