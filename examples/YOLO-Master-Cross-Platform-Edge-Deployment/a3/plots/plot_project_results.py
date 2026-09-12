#!/usr/bin/env python
"""Project03 + a3-smoke quantization results, plotted in the same visual language as the
two heatmaps (DejaVu Sans/Mono, the same green/sand/red palette, thin marks, direct labels,
no cringe legends). Three figures, each wider-than-tall for an issue:

  a3_int8_unpruned   - INT8 Q/DQ on unpruned YOLO-Master (COCO): accuracy + latency
  p03_int8_accuracy  - reaching the <1 AP bar (pruned, COCO): calibration sweep + per-model
  p03_int8_latency   - FP16 beats INT8 on every GPU, and QAT regressed

Numbers: a3 read live from runs/a3/*/{trt/ladder.csv, qdq/result.json}; project03 curated
constants transcribed from PROJECT03_QUANT_RESULTS.md (tables 2.1-3.7).
"""
import json, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUTDIR = "/data/yolo-quant-work"
A3 = "/data/yolo-master-edge/runs/a3"

# palette (consistent with the heatmaps)
INK, INK2, MUTED, FAINT = "#1c1c1c", "#3a414d", "#69707e", "#9a988f"
GREY = "#b9b6ad"      # fp32 reference
GREEN = "#2f8862"     # fp16 / pass
TEAL = "#12655a"      # fastest
SAND = "#d1913f"      # int8 (the quantized one)
RED = "#a5382d"       # collapse / fail
GRID = "#e7e5df"
MONO = "DejaVu Sans Mono"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "figure.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "#c9c7c0", "axes.linewidth": 0.9, "axes.grid": False,
})


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def keymark(ax, items, loc="lower left", anchor=(0.0, 1.005)):
    """minimal frameless colour key above the axes."""
    from matplotlib.patches import Patch
    h = [Patch(facecolor=c, label=l) for c, l in items]
    ax.legend(handles=h, frameon=False, ncol=len(items), loc=loc, bbox_to_anchor=anchor,
              fontsize=8.6, handlelength=1.1, handleheight=1.0, handletextpad=0.45,
              columnspacing=1.4, labelcolor=INK2, borderpad=0)


# ----------------------------------------------------------------- a3 data
def load_a3():
    out = []
    for m, label in [("v01n", "v0.1-n"), ("v01s", "v0.1-s"), ("v01m", "v0.1-m"),
                     ("v01l", "v0.1-l"), ("esmoen", "EsMoE-n")]:
        lad = os.path.join(A3, m, "trt", "ladder.csv")
        fp32 = fp16 = None
        for r in csv.DictReader(open(lad)):
            if r["mode"] == "fp32":
                fp32 = (float(r["mAP50_95"]), float(r["lat_ms_median"]))
            if r["mode"] == "fp16":
                fp16 = (float(r["mAP50_95"]), float(r["lat_ms_median"]))
        j = json.load(open(os.path.join(A3, m, "qdq", "result.json")))["model"]
        i8 = (float(j["mAP50_95"]), float(j["lat_ms_median"]))
        out.append((label, fp32, fp16, i8))
    return out


def fig_a3():
    data = load_a3()
    labels = [d[0] for d in data]
    x = np.arange(len(labels))
    fig, (axA, axL) = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=300,
                                   gridspec_kw={"wspace": 0.22})

    # --- panel A: accuracy, FP32 vs INT8 Q/DQ ---
    w = 0.38
    fp32 = [d[1][0] for d in data]
    i8 = [d[3][0] for d in data]
    axA.bar(x - w / 2, fp32, w, color=GREY, zorder=3)
    for k, (xi, v) in enumerate(zip(x, i8)):
        loss = (fp32[k] - v) * 100
        c = GREEN if loss < 1 else (SAND if loss < 6 else RED)
        axA.bar(xi + w / 2, v, w, color=c, zorder=3)
        axA.text(xi + w / 2, v + 0.008, f"−{loss:.2f}", ha="center", va="bottom",
                 fontsize=8.2, color=c, fontweight="bold")
    for xi, v in zip(x, fp32):
        axA.text(xi - w / 2, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6, color=MUTED, family=MONO)
    axA.set_ylim(0, 0.60)
    axA.set_ylabel("COCO mAP$_{50-95}$", fontsize=10)
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9.2)
    axA.set_yticks(np.arange(0, 0.61, 0.1))
    axA.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    style(axA)
    keymark(axA, [(GREY, "FP32"), (GREEN, "INT8 (Δ<1)"), (RED, "INT8 collapse")])

    # --- panel L: latency FP32 / FP16 / INT8 ---
    w = 0.26
    fp32l = [d[1][1] for d in data]
    fp16l = [d[2][1] for d in data]
    i8l = [d[3][1] for d in data]
    axL.bar(x - w, fp32l, w, color=GREY, zorder=3)
    axL.bar(x, fp16l, w, color=TEAL, zorder=3)
    axL.bar(x + w, i8l, w, color=SAND, zorder=3)
    for xi, a, b, c in zip(x, fp32l, fp16l, i8l):
        for dx, v, col in [(-w, a, MUTED), (0, b, TEAL), (w, c, SAND)]:
            axL.text(xi + dx, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=7.0, color=col)
    axL.set_ylabel("model-only latency (ms, batch=1)", fontsize=9.6)
    axL.set_xticks(x); axL.set_xticklabels(labels, fontsize=9.2)
    axL.set_ylim(0, max(fp32l) * 1.16)
    axL.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    style(axL)
    keymark(axL, [(GREY, "FP32"), (TEAL, "FP16"), (SAND, "INT8")])

    fig.savefig(f"{OUTDIR}/a3_int8_unpruned.png", dpi=300, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(f"{OUTDIR}/a3_int8_unpruned.pdf", bbox_inches="tight", pad_inches=0.16)
    print("wrote a3_int8_unpruned")


# ----------------------------------------------------------------- project03 accuracy
def fig_p03_accuracy():
    # 2.1 calibration coverage (v0.1-N pruned COCO, fp32 0.4286)
    calib = [256, 1024, 2048, 4096]
    surgical = [0.2995, 0.3756, 0.4044, 0.4038]
    recipe = [None, 0.3959, 0.4055, 0.4058]
    FP32 = 0.4286
    QDQ = 0.4206  # explicit Q/DQ calibrate-only (64 imgs)
    # 2.2 explicit Q/DQ per model
    models = ["v0.1-N", "UoMoE-N\nexp-excl", "UoMoE-N"]
    loss = [0.80, 1.39, 5.93]
    lcol = [GREEN, SAND, RED]

    fig, (axC, axB) = plt.subplots(1, 2, figsize=(10.4, 4.3), dpi=300,
                                   gridspec_kw={"wspace": 0.28, "width_ratios": [1.25, 1]})

    # calibration sweep
    axC.axhline(FP32, color=FAINT, lw=1.1, ls=(0, (5, 3)), zorder=2)
    axC.text(4096, FP32 + 0.002, "FP32  0.4286", ha="right", va="bottom", fontsize=8, color=MUTED)
    xr = [c for c, r in zip(calib, recipe) if r is not None]
    yr = [r for r in recipe if r is not None]
    axC.plot(calib, surgical, "-o", color=SAND, lw=2, ms=6, zorder=3, label="surgical PTQ")
    axC.plot(xr, yr, "-o", color=GREEN, lw=2, ms=6, zorder=3, label="+ stem-pair recipe")
    axC.plot([64], [QDQ], marker="*", ms=15, color=TEAL, zorder=4, ls="none")
    axC.text(64, QDQ + 0.004, "explicit Q/DQ\n0.4206  (−0.80)", ha="left", va="bottom", fontsize=8, color=TEAL)
    axC.set_xscale("log", base=2)
    axC.set_xticks(calib); axC.set_xticklabels([str(c) for c in calib], fontsize=9)
    axC.set_xlim(52, 5200)
    axC.set_ylim(0.28, 0.44)
    axC.set_xlabel("calibration images (train2017)", fontsize=9.6)
    axC.set_ylabel("COCO mAP$_{50-95}$", fontsize=10)
    axC.grid(color=GRID, lw=0.8, zorder=0)
    style(axC)
    keymark(axC, [(SAND, "surgical PTQ"), (GREEN, "+ stem-pair"), (TEAL, "explicit Q/DQ")])

    # per-model Q/DQ loss
    y = np.arange(len(models))[::-1]
    axB.barh(y, loss, 0.6, color=lcol, zorder=3)
    for yi, v, c in zip(y, loss, lcol):
        axB.text(v + 0.08, yi, f"−{v:.2f}", va="center", ha="left", fontsize=9, color=c, fontweight="bold")
    axB.axvline(1.0, color=FAINT, lw=1.1, ls=(0, (5, 3)), zorder=2)
    axB.text(1.0, len(models) - 0.35, "1.0 AP bar", ha="center", va="bottom", fontsize=8, color=MUTED)
    axB.set_yticks(y); axB.set_yticklabels(models, fontsize=9)
    axB.set_xlim(0, 6.6)
    axB.set_xlabel("AP points lost to INT8 Q/DQ  (pruned, COCO)", fontsize=9.2)
    axB.grid(axis="x", color=GRID, lw=0.8, zorder=0)
    style(axB)

    fig.savefig(f"{OUTDIR}/p03_int8_accuracy.png", dpi=300, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(f"{OUTDIR}/p03_int8_accuracy.pdf", bbox_inches="tight", pad_inches=0.16)
    print("wrote p03_int8_accuracy")


# ----------------------------------------------------------------- project03 latency
def fig_p03_latency():
    # 3.6 / 3.7 hardware ladders, model-only ms (honest fp32 baseline)
    hw = ["A100", "L4 (Ada)", "Orin Nano\n(10W)"]
    fp32 = [2.422, 3.047, 37.62]
    fp16 = [2.115, 1.607, 21.87]
    int8 = [2.359, 1.896, 24.77]
    cut16 = [(1 - b / a) * 100 for a, b in zip(fp32, fp16)]
    cut8 = [(1 - c / a) * 100 for a, c in zip(fp32, int8)]

    fig, (axH, axQ) = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=300,
                                   gridspec_kw={"wspace": 0.32, "width_ratios": [1.5, 1]})

    x = np.arange(len(hw)); w = 0.36
    axH.bar(x - w / 2, cut16, w, color=TEAL, zorder=3)
    axH.bar(x + w / 2, cut8, w, color=SAND, zorder=3)
    for xi, a, b in zip(x, cut16, cut8):
        axH.text(xi - w / 2, a + 0.8, f"{a:.0f}%", ha="center", va="bottom", fontsize=8.4, color=TEAL, fontweight="bold")
        axH.text(xi + w / 2, b + 0.8, f"{b:.0f}%", ha="center", va="bottom", fontsize=8.4, color=SAND, fontweight="bold")
    axH.set_xticks(x); axH.set_xticklabels(hw, fontsize=9.2)
    axH.set_ylabel("latency cut vs FP32  (higher = faster)", fontsize=9.4)
    axH.set_ylim(0, 58)
    axH.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    style(axH)
    keymark(axH, [(TEAL, "FP16"), (SAND, "INT8 Q/DQ")])

    # QAT negative result (v0.1-N COCO)
    eng = ["Q/DQ\ncalib-only", "QAT\nema", "QAT\nraw"]
    dap = [-0.80, -3.17, -3.32]
    col = [GREEN, RED, RED]
    xx = np.arange(len(eng))
    axQ.bar(xx, dap, 0.6, color=col, zorder=3)
    for xi, v, c in zip(xx, dap, col):
        axQ.text(xi, v - 0.12, f"{v:.2f}", ha="center", va="top", fontsize=8.8, color=c, fontweight="bold")
    axQ.axhline(0, color="#c9c7c0", lw=0.9)
    axQ.set_xticks(xx); axQ.set_xticklabels(eng, fontsize=8.6)
    axQ.set_ylabel("ΔAP vs FP32", fontsize=9.6)
    axQ.set_ylim(-3.9, 0.5)
    axQ.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    style(axQ)

    fig.savefig(f"{OUTDIR}/p03_int8_latency.png", dpi=300, bbox_inches="tight", pad_inches=0.16)
    fig.savefig(f"{OUTDIR}/p03_int8_latency.pdf", bbox_inches="tight", pad_inches=0.16)
    print("wrote p03_int8_latency")


if __name__ == "__main__":
    fig_a3()
    fig_p03_accuracy()
    fig_p03_latency()
