#!/usr/bin/env python3
"""Build MoT ablation tables and Seaborn figures from completed experiment runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


MODEL_ORDER = ["v10", "v10_mot", "v10_moa", "v10_moa_mot"]
MODEL_NAMES = {
    "v10": "EsMoE-N",
    "v10_mot": "MoT-N",
    "v10_moa": "MoA-N",
    "v10_moa_mot": "MoA+MoT-N",
}
EXPERT_ORDER = ["LocalConvTransformer", "WindowTransformer", "DeformableTransformer"]
SCENE_ORDER = ["dense", "sparse", "small_objects", "large_objects", "dense_small", "sparse_large", "irregular_occluded"]


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def total_loss(df: pd.DataFrame, prefix: str) -> pd.Series:
    keys = [f"{prefix}/box_loss", f"{prefix}/cls_loss", f"{prefix}/dfl_loss", f"{prefix}/moe_loss"]
    values = [pd.to_numeric(df[k], errors="coerce") for k in keys if k in df.columns]
    if not values:
        return pd.Series([float("nan")] * len(df), index=df.index)
    out = values[0].fillna(0.0)
    for item in values[1:]:
        out = out + item.fillna(0.0)
    return out


def prepare_summary(run_dir: Path, out_dir: Path) -> pd.DataFrame:
    summary = read_csv(run_dir / "summary.csv")
    summary["model"] = summary["key"].map(MODEL_NAMES).fillna(summary["key"])
    summary["mAP50-95"] = pd.to_numeric(summary["metrics/mAP50-95(B)"], errors="coerce")
    summary["mAP50"] = pd.to_numeric(summary["metrics/mAP50(B)"], errors="coerce")
    summary["latency_p50_ms"] = pd.to_numeric(summary["latency_ms_p50"], errors="coerce")
    summary["latency_p95_ms"] = pd.to_numeric(summary["latency_ms_p95"], errors="coerce")
    summary["latency_p99_ms"] = pd.to_numeric(summary["latency_ms_p99"], errors="coerce")
    summary["params_m"] = pd.to_numeric(summary["params_m"], errors="coerce")
    summary["flops_g"] = pd.to_numeric(summary["flops_g"], errors="coerce")
    summary["final_train_total_loss"] = pd.to_numeric(summary["final_train_total_loss"], errors="coerce")
    summary["loss_diverged"] = summary["loss_diverged"].astype(str)
    summary["nan_detected"] = summary["nan_detected"].astype(str)
    keep = [
        "key",
        "model",
        "mAP50-95",
        "mAP50",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
        "flops_g",
        "params_m",
        "final_train_total_loss",
        "nan_detected",
        "loss_diverged",
    ]
    out = summary[keep].copy()
    out.to_csv(out_dir / "mot_model_comparison.csv", index=False)
    return out


def prepare_curves(run_dir: Path, out_dir: Path) -> pd.DataFrame:
    frames = []
    for key in MODEL_ORDER:
        path = run_dir / key / "results.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["key"] = key
        df["model"] = MODEL_NAMES[key]
        df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
        df["mAP50-95"] = pd.to_numeric(df["metrics/mAP50-95(B)"], errors="coerce")
        df["mAP50"] = pd.to_numeric(df["metrics/mAP50(B)"], errors="coerce")
        df["train_total_loss"] = total_loss(df, "train")
        df["val_total_loss"] = total_loss(df, "val")
        frames.append(df[["key", "model", "epoch", "mAP50-95", "mAP50", "train_total_loss", "val_total_loss"]])
    curves = pd.concat(frames, ignore_index=True)
    curves.to_csv(out_dir / "mot_training_curves.csv", index=False)
    return curves


def prepare_routing(run_dir: Path, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    routing_dir = run_dir / "routing"
    scenarios = read_csv(routing_dir / "mot_routing_scenarios.csv")
    checks = read_csv(routing_dir / "mot_deformable_activation_check.csv")
    scenarios.to_csv(out_dir / "mot_routing_scenarios.csv", index=False)
    checks.to_csv(out_dir / "mot_deformable_activation_check.csv", index=False)
    return scenarios, checks


def setup_theme() -> None:
    sns.set_theme(style="whitegrid", context="paper", font="Arial", font_scale=1.15)
    plt.rcParams["svg.fonttype"] = "none"


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_model_tradeoff(summary: pd.DataFrame, fig_dir: Path) -> None:
    data = summary.copy()
    data["model"] = pd.Categorical(data["model"], [MODEL_NAMES[k] for k in MODEL_ORDER], ordered=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    sns.barplot(data=data, x="model", y="mAP50-95", hue="model", palette="colorblind", legend=False, ax=axes[0])
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("mAP50-95")
    sns.barplot(data=data, x="model", y="latency_p50_ms", hue="model", palette="colorblind", legend=False, ax=axes[1])
    axes[1].set_title("Latency")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("P50 latency (ms)")
    sns.barplot(data=data, x="model", y="flops_g", hue="model", palette="colorblind", legend=False, ax=axes[2])
    axes[2].set_title("Actual FLOPs")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("GFLOPs")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, fig_dir / "mot_model_tradeoff.svg")


def plot_training_curves(curves: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.lineplot(data=curves, x="epoch", y="mAP50-95", hue="model", hue_order=[MODEL_NAMES[k] for k in MODEL_ORDER], ax=axes[0])
    axes[0].set_title("Validation mAP50-95")
    axes[0].set_ylabel("mAP50-95")
    sns.lineplot(data=curves, x="epoch", y="train_total_loss", hue="model", hue_order=[MODEL_NAMES[k] for k in MODEL_ORDER], ax=axes[1])
    axes[1].set_title("Training total loss")
    axes[1].set_ylabel("train box+cls+dfl+aux")
    for ax in axes:
        ax.set_xlabel("Epoch")
    fig.tight_layout()
    save(fig, fig_dir / "mot_training_curves.svg")


def plot_routing(scenarios: pd.DataFrame, checks: pd.DataFrame, fig_dir: Path) -> None:
    scenarios = scenarios.copy()
    scenarios["scene"] = pd.Categorical(scenarios["scene"], SCENE_ORDER, ordered=True)
    scenarios["expert"] = pd.Categorical(scenarios["expert"], EXPERT_ORDER, ordered=True)
    pivot = scenarios.pivot_table(index="scene", columns="expert", values="top1_share_mean", observed=False)
    pivot = pivot.reindex(index=SCENE_ORDER, columns=EXPERT_ORDER).dropna(how="all")
    fig, ax = plt.subplots(figsize=(9, 4.8))
    sns.heatmap(pivot, annot=True, fmt=".2f", vmin=0, vmax=1, cmap="viridis", linewidths=0.4, ax=ax)
    ax.set_title("MoT expert top-1 routing share by scene")
    ax.set_xlabel("")
    ax.set_ylabel("")
    save(fig, fig_dir / "mot_routing_heatmap.svg")

    mean_weight = checks[checks["metric"] == "mean_weight"].copy()
    mean_weight = mean_weight[mean_weight["baseline"] != "non_irregular_pooled"]
    mean_weight["baseline"] = pd.Categorical(mean_weight["baseline"], SCENE_ORDER, ordered=True)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.barplot(data=mean_weight, x="baseline", y="mean_diff", hue="deformable_significantly_higher", palette="colorblind", ax=ax)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Deformable mean-weight lift: irregular/occluded vs baseline")
    ax.set_xlabel("Baseline scene")
    ax.set_ylabel("Mean weight difference")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    save(fig, fig_dir / "mot_deformable_lift.svg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/mot_ablation/visdrone_v10_mot_hybrid_50ep"))
    parser.add_argument("--output-dir", type=Path, default=Path("examples/mot_hybrid_architecture/results"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.output_dir
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    setup_theme()
    summary = prepare_summary(args.run_dir, out_dir)
    curves = prepare_curves(args.run_dir, out_dir)
    scenarios, checks = prepare_routing(args.run_dir, out_dir)
    plot_model_tradeoff(summary, fig_dir)
    plot_training_curves(curves, fig_dir)
    plot_routing(scenarios, checks, fig_dir)
    print(f"[plots] wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
