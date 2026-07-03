#!/usr/bin/env python3
"""Plot VisDrone MoA ablation curves with seaborn."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "runs/moa_ablation/visdrone_v10_50ep"
OUT_DIR = Path(__file__).resolve().parent / "assets"

MODELS = {
    "v10": "YOLO-Master v0.10 MoE Baseline",
    "v10_moa": "YOLO-Master v0.10 MoA-N",
}


def load_results() -> pd.DataFrame:
    frames = []
    for key, label in MODELS.items():
        path = RUN_ROOT / key / "results.csv"
        df = pd.read_csv(path)
        df.columns = [c.strip() for c in df.columns]
        df["model_key"] = key
        df["Model"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def configure_style() -> None:
    sns.set_theme(style="whitegrid", context="paper", font="Arial", font_scale=1.15)
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.sans-serif": ["Arial"],
            "axes.unicode_minus": False,
            "figure.dpi": 140,
            "savefig.dpi": 300,
        }
    )


def plot_map_curve(df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    sns.lineplot(
        data=df,
        x="epoch",
        y="metrics/mAP50-95(B)",
        hue="Model",
        style="Model",
        linewidth=2.2,
        ax=ax,
    )
    ax.set_title("VisDrone Validation mAP50-95", pad=12)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP50-95")
    ax.set_xlim(df["epoch"].min(), df["epoch"].max())
    ax.legend(title="", frameon=True, loc="lower right")
    sns.despine(fig=fig)
    fig.tight_layout()

    out = OUT_DIR / "visdrone_map50_95_curve.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_loss_curves(df: pd.DataFrame) -> Path:
    loss_rows = []
    components = [
        ("Box Loss", "train/box_loss", "val/box_loss"),
        ("Class Loss", "train/cls_loss", "val/cls_loss"),
        ("DFL Loss", "train/dfl_loss", "val/dfl_loss"),
    ]
    for component, train_col, val_col in components:
        for split, col in (("Train", train_col), ("Validation", val_col)):
            part = df[["epoch", "Model", col]].rename(columns={col: "Loss"})
            part["Split"] = split
            part["Component"] = component
            loss_rows.append(part)
    loss_df = pd.concat(loss_rows, ignore_index=True)

    g = sns.relplot(
        data=loss_df,
        x="epoch",
        y="Loss",
        hue="Model",
        style="Model",
        col="Component",
        row="Split",
        kind="line",
        linewidth=1.9,
        height=3.0,
        aspect=1.35,
        facet_kws={"sharey": False, "margin_titles": True},
    )
    g.set_axis_labels("Epoch", "Loss")
    g.set_titles(row_template="{row_name}", col_template="{col_name}")
    g.figure.suptitle("VisDrone Training and Validation Loss Curves", y=1.03)
    sns.move_legend(g, "upper center", bbox_to_anchor=(0.5, 1.0), ncol=2, title="", frameon=True)
    g.figure.tight_layout()

    out = OUT_DIR / "visdrone_loss_curves.png"
    g.figure.savefig(out, bbox_inches="tight")
    plt.close(g.figure)
    return out


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    df = load_results()
    map_path = plot_map_curve(df)
    loss_path = plot_loss_curves(df)
    print(map_path.relative_to(ROOT))
    print(loss_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
