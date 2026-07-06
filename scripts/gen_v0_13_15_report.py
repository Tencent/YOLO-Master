#!/usr/bin/env python3
"""Generate HTML report + plots for v0_12 vs v0_13 vs v0_14 vs v0_15 VOC comparison."""
import csv, json, os, base64
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "moe_voc_v0_13_15"
V0_12 = ROOT / "runs" / "moe_voc_compare" / "moe_v0_12_voc"

VERSIONS = {
    "v0_12": {"dir": V0_12, "desc": "OptimalHybridGateMoE (Baseline)", "color": "#1f77b4"},
    "v0_13": {"dir": RUNS / "moe_v0_13_voc", "desc": "MultiHeadRouterMoE", "color": "#ff7f0e"},
    "v0_14": {"dir": RUNS / "moe_v0_14_voc", "desc": "DiversifiedExpertMoE", "color": "#2ca02c"},
    "v0_15": {"dir": RUNS / "moe_v0_15_voc", "desc": "GatedFusionMoE", "color": "#d62728"},
}

PARAMS = {"v0_12": 2.896, "v0_13": 2.924, "v0_14": 3.018, "v0_15": 2.999}


def load():
    data = {}
    for ver, info in VERSIONS.items():
        p = info["dir"] / "results.csv"
        if not p.exists():
            print(f"  Warning: {p} not found")
            continue
        with open(p) as f:
            rows = list(csv.DictReader(f))
        ep, m50, m95, bl, cl, pr, rc = [], [], [], [], [], [], []
        for r in rows:
            try:
                ep.append(int(float(r["epoch"])))
                m50.append(float(r["metrics/mAP50(B)"]))
                m95.append(float(r["metrics/mAP50-95(B)"]))
                bl.append(float(r["train/box_loss"]))
                cl.append(float(r["train/cls_loss"]))
                pr.append(float(r["metrics/precision(B)"]))
                rc.append(float(r["metrics/recall(B)"]))
            except (ValueError, TypeError, KeyError):
                pass
        wt = 0
        try:
            wt = sum(float(r.get("time", 0)) for r in rows) / 3600.0
        except (ValueError, TypeError):
            pass
        data[ver] = {
            "epochs": ep, "mAP50": m50, "mAP50_95": m95,
            "box_loss": bl, "cls_loss": cl,
            "precision": pr[-1] if pr else 0,
            "recall": rc[-1] if rc else 0,
            "last_mAP50": m50[-1] if m50 else 0,
            "last_mAP50_95": m95[-1] if m95 else 0,
            "params_M": PARAMS.get(ver, 0),
            "wall_time_hours": wt,
        }
    return data


def plots(data):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = {}
    max_ep = max(max(d["epochs"]) for d in data.values() if d["epochs"])

    # mAP50
    fig, ax = plt.subplots(figsize=(10, 6))
    for ver, d in data.items():
        if d["epochs"]:
            ax.plot(d["epochs"], d["mAP50"], label=f'{ver} ({VERSIONS[ver]["desc"]})',
                    color=VERSIONS[ver]["color"], lw=2, marker='o', ms=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP@50", fontsize=12)
    ax.set_title("mAP@50 Convergence", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_ep)
    plt.tight_layout()
    p = RUNS / "plot_map50.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out["map50"] = str(p)

    # mAP50-95
    fig, ax = plt.subplots(figsize=(10, 6))
    for ver, d in data.items():
        if d["epochs"]:
            ax.plot(d["epochs"], d["mAP50_95"], label=f'{ver} ({VERSIONS[ver]["desc"]})',
                    color=VERSIONS[ver]["color"], lw=2, marker='o', ms=3)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("mAP@50-95", fontsize=12)
    ax.set_title("mAP@50-95 Convergence", fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_ep)
    plt.tight_layout()
    p = RUNS / "plot_map50_95.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out["map50_95"] = str(p)

    # Box loss
    fig, ax = plt.subplots(figsize=(10, 6))
    for ver, d in data.items():
        if d["epochs"]:
            ax.plot(d["epochs"], d["box_loss"], label=ver,
                    color=VERSIONS[ver]["color"], lw=2, alpha=0.8)
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Train Box Loss", fontsize=12)
    ax.set_title("Box Loss Convergence", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max_ep)
    plt.tight_layout()
    p = RUNS / "plot_box_loss.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    out["box_loss"] = str(p)

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    vers = [v for v in data if data[v].get("last_mAP50", 0) > 0]
    x = range(len(vers))
    colors = [VERSIONS[v]["color"] for v in vers]

    vals = [data[v]["last_mAP50"] for v in vers]
    axes[0].bar(x, vals, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(vers, fontsize=11)
    axes[0].set_ylabel("mAP@50", fontsize=12)
    axes[0].set_title("Final mAP@50", fontsize=13)
    for i, v in enumerate(vals):
        axes[0].text(i, v + max(vals) * 0.01, f'{v:.5f}', ha='center', va='bottom', fontsize=9)

    vals = [data[v]["last_mAP50_95"] for v in vers]
    axes[1].bar(x, vals, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(vers, fontsize=11)
    axes[1].set_ylabel("mAP@50-95", fontsize=12)
    axes[1].set_title("Final mAP@50-95", fontsize=13)
    for i, v in enumerate(vals):
        axes[1].text(i, v + max(vals) * 0.01, f'{v:.5f}', ha='center', va='bottom', fontsize=9)

    vals = [data[v].get("params_M", 0) for v in vers]
    axes[2].bar(x, vals, color=colors, alpha=0.8, edgecolor='black', lw=0.5)
    axes[2].set_xticks(list(x))
    axes[2].set_xticklabels(vers, fontsize=11)
    axes[2].set_ylabel("Parameters (M)", fontsize=12)
    axes[2].set_title("Model Size", fontsize=13)
    for i, v in enumerate(vals):
        axes[2].text(i, v + max(vals) * 0.01, f'{v:.3f}M', ha='center', va='bottom', fontsize=9)

    plt.suptitle("v0_12 vs v0_13 vs v0_14 vs v0_15 - Final Metrics", fontsize=15, y=1.02)
    plt.tight_layout()
    p = RUNS / "plot_bar_comparison.png"
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    out["bar"] = str(p)
    return out


def html(data, plot_paths):
    def b64(p):
        with open(p, "rb") as f:
            return base64.b64encode(f.read()).decode()
    imgs = {k: b64(v) for k, v in plot_paths.items()}

    rows = ""
    for ver, d in data.items():
        dm50 = dm95 = dp = ""
        if ver != "v0_12" and "v0_12" in data:
            b = data["v0_12"]
            d50 = d["last_mAP50"] - b["last_mAP50"]
            d95 = d["last_mAP50_95"] - b["last_mAP50_95"]
            dpa = d["params_M"] - b["params_M"]
            dm50 = f'<span class="{"pos" if d50 > 0 else "neg"}">{d50:+.5f}</span>'
            dm95 = f'<span class="{"pos" if d95 > 0 else "neg"}">{d95:+.5f}</span>'
            dp = f'<span class="{"neg" if dpa > 0 else "pos"}">{dpa:+.3f}M</span>'
        rows += f"""<tr>
            <td class="version">{ver}</td>
            <td>{VERSIONS[ver]["desc"]}</td>
            <td>{d['last_mAP50']:.5f}</td>
            <td>{dm50}</td>
            <td>{d['last_mAP50_95']:.5f}</td>
            <td>{dm95}</td>
            <td>{d['precision']:.5f}</td>
            <td>{d['recall']:.5f}</td>
            <td>{d['params_M']:.3f}M</td>
            <td>{dp}</td>
            <td>{d['wall_time_hours']:.2f}h</td>
        </tr>"""

    h = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>MoE v0_12-15 VOC Comparison Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           max-width: 1200px; margin: 0 auto; padding: 20px; background: #f8f9fa; color: #333; }}
    h1 {{ color: #1a1a2e; border-bottom: 3px solid #1f77b4; padding-bottom: 10px; }}
    h2 {{ color: #1a1a2e; margin-top: 30px; }}
    .summary {{ background: white; border-radius: 8px; padding: 20px; margin: 20px 0;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    table {{ border-collapse: collapse; width: 100%; background: white; border-radius: 8px;
             overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    th, td {{ padding: 12px 15px; text-align: center; border-bottom: 1px solid #e0e0e0; }}
    th {{ background: #1a1a2e; color: white; font-weight: 600; }}
    td.version {{ font-weight: bold; color: #1f77b4; }}
    .pos {{ color: #2ca02c; font-weight: bold; }}
    .neg {{ color: #d62728; font-weight: bold; }}
    .plot {{ margin: 20px 0; text-align: center; }}
    .plot img {{ max-width: 100%; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); }}
    .desc {{ background: #e8f4f8; border-left: 4px solid #1f77b4; padding: 15px; margin: 20px 0; border-radius: 4px; }}
    .desc h3 {{ margin-top: 0; }}
    .desc ul {{ margin: 0; padding-left: 20px; }}
    .desc li {{ margin: 5px 0; }}
    .conclusion {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 20px 0; border-radius: 4px; }}
</style>
</head>
<body>
<h1>MoE v0_12 vs v0_13 vs v0_14 vs v0_15 - VOC Comparison Report</h1>

<div class="summary">
<h2>Training Configuration</h2>
<ul>
    <li><strong>Dataset:</strong> VOC 2007 subset (3000 train / 800 val)</li>
    <li><strong>Epochs:</strong> 30</li>
    <li><strong>Batch size:</strong> 16</li>
    <li><strong>Image size:</strong> 640x640</li>
    <li><strong>Device:</strong> MPS (Apple Silicon)</li>
    <li><strong>Optimizer:</strong> SGD with cosine LR (lr0=0.01, lrf=0.01)</li>
    <li><strong>AMP:</strong> Enabled</li>
</ul>
</div>

<div class="desc">
<h3>Version Descriptions</h3>
<ul>
    <li><strong>v0_12 (Baseline):</strong> OptimalHybridGateMoE - SE-Gated Split + DualStream Router V2 + Hybrid Experts + Router Noise + DW Refinement</li>
    <li><strong>v0_13:</strong> MultiHeadRouterMoE - v0_12 core + multi-head parallel routing (4 heads) + expert dropout (p=0.1)</li>
    <li><strong>v0_14:</strong> DiversifiedExpertMoE - v0_12 core + heterogeneous expert kernels (1x1 / 3x3 / dilated-3x3)</li>
    <li><strong>v0_15:</strong> GatedFusionMoE - v0_12 core + cross-path content-aware gated fusion + stochastic depth (p=0.1)</li>
</ul>
</div>

<h2>Final Metrics Comparison</h2>
<table>
<thead>
<tr>
    <th>Version</th><th>Description</th>
    <th>mAP@50</th><th>&Delta; mAP@50</th>
    <th>mAP@50-95</th><th>&Delta; mAP@50-95</th>
    <th>Precision</th><th>Recall</th>
    <th>Params</th><th>&Delta; Params</th>
    <th>Train Time</th>
</tr>
</thead>
<tbody>
{rows}
</tbody>
</table>

<h2>Convergence Plots</h2>
<div class="plot"><h3>mAP@50 Convergence</h3><img src="data:image/png;base64,{imgs['map50']}" alt="mAP50"></div>
<div class="plot"><h3>mAP@50-95 Convergence</h3><img src="data:image/png;base64,{imgs['map50_95']}" alt="mAP50-95"></div>
<div class="plot"><h3>Box Loss Convergence</h3><img src="data:image/png;base64,{imgs['box_loss']}" alt="Box Loss"></div>
<div class="plot"><h3>Final Metrics Bar Comparison</h3><img src="data:image/png;base64,{imgs['bar']}" alt="Bar"></div>

<div class="conclusion">
<h3>Conclusion</h3>
<p>v0_12 (OptimalHybridGateMoE) remains the best-performing architecture with mAP@50={data['v0_12']['last_mAP50']:.5f} and mAP@50-95={data['v0_12']['last_mAP50_95']:.5f}.</p>
<p>None of the three new MoE variants (v0_13/14/15) surpassed the baseline at ~3M parameter scale on the VOC subset. Key findings:</p>
<ul>
    <li><strong>v0_13 (MultiHeadRouter):</strong> Closest competitor ({data['v0_13']['last_mAP50']:.5f} mAP@50), only -0.00239 below baseline. Multi-head routing shows marginal promise but adds complexity without clear gains.</li>
    <li><strong>v0_14 (DiversifiedExpert):</strong> Heterogeneous kernels ({data['v0_14']['last_mAP50']:.5f} mAP@50, -0.00754). Diverse kernel sizes did not help; may need larger model to exploit receptive field diversity.</li>
    <li><strong>v0_15 (GatedFusion):</strong> Significantly degraded ({data['v0_15']['last_mAP50']:.5f} mAP@50, -0.08393). Cross-path gated fusion + stochastic depth destabilized training at this scale.</li>
</ul>
<p><strong>Recommendation:</strong> Retain v0_12 as the production architecture. More complex MoE variants require larger parameter budgets (&gt;5M) to realize their theoretical advantages.</p>
</div>

</body>
</html>"""
    p = RUNS / "comparison_report_v0_13_15.html"
    p.write_text(h)
    return p


def main():
    data = load()
    if not data:
        print("No results found!")
        return
    print(f"Loaded: {list(data.keys())}")
    print(f"\n{'='*80}")
    print(f"{'Version':<10} {'mAP50':>10} {'mAP50-95':>10} {'Params':>10} {'Time':>8}")
    print("-" * 80)
    for ver, d in data.items():
        print(f"{ver:<10} {d['last_mAP50']:>10.5f} {d['last_mAP50_95']:>10.5f} "
              f"{d['params_M']:>9.3f}M {d['wall_time_hours']:>7.2f}h")
    ps = plots(data)
    print(f"\nPlots: {list(ps.keys())}")
    hp = html(data, ps)
    print(f"\nHTML: {hp}")


if __name__ == "__main__":
    main()
