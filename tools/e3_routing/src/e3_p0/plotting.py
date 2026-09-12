"""Static evidence plots for the P0 artifact bundle."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .aggregation import aggregate_events


def _load_matrix(events: list[dict[str, Any]]) -> tuple[np.ndarray, list[str]]:
    width = max(len(event["routing"]["expert_load"]) for event in events)
    matrix = np.full((len(events), width), np.nan, dtype=np.float64)
    labels = []
    for row, event in enumerate(events):
        load = event["routing"]["expert_load"]
        matrix[row, : len(load)] = load
        labels.append(event["module"]["name"])
    return matrix, labels


def save_family_plot(family: str, events: list[dict[str, Any]], path: Path, subtitle: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    aggregates = aggregate_events(events)
    width = max(len(item["expert_load_mean"]) for item in aggregates)
    matrix = np.full((len(aggregates), width), np.nan, dtype=np.float64)
    for row, item in enumerate(aggregates):
        matrix[row, : len(item["expert_load_mean"])] = item["expert_load_mean"]
    labels = [item["module"] for item in aggregates]
    entropies = [item["per_forward_metric_stats"]["entropy_normalized"]["mean"] for item in aggregates]
    entropy_std = [item["per_forward_metric_stats"]["entropy_normalized"]["std"] for item in aggregates]
    ginis = [item["per_forward_metric_stats"]["load_gini"]["mean"] for item in aggregates]
    gini_std = [item["per_forward_metric_stats"]["load_gini"]["std"] for item in aggregates]
    height = max(4.8, 0.55 * len(labels) + 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(13, height), constrained_layout=True)
    image = axes[0].imshow(matrix, aspect="auto", cmap="Blues", vmin=0.0, vmax=max(0.5, float(np.nanmax(matrix))))
    axes[0].set_title(f"{family.upper()} expert load")
    axes[0].set_xlabel("Expert index")
    axes[0].set_ylabel("Leaf routed module")
    axes[0].set_xticks(range(matrix.shape[1]), [f"E{i}" for i in range(matrix.shape[1])])
    axes[0].set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=axes[0], label="Normalized load share")

    positions = np.arange(len(labels))
    axes[1].barh(
        positions - 0.18,
        entropies,
        height=0.34,
        xerr=entropy_std,
        label="Normalized entropy (mean ± std)",
        color="#2F6FAE",
    )
    axes[1].barh(
        positions + 0.18,
        ginis,
        height=0.34,
        xerr=gini_std,
        label="Load Gini (mean ± std)",
        color="#E19A35",
    )
    axes[1].set_yticks(positions, labels)
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_xlabel("Metric value")
    axes[1].set_title("Balance diagnostics")
    axes[1].grid(axis="x", linewidth=0.7, color="#D8DEE6")
    axes[1].legend(loc="lower right")
    fig.suptitle(f"E3 P0 unified routing view — {family.upper()}\n{subtitle}", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cross_family_plot(all_events: dict[str, list[dict[str, Any]]], path: Path, subtitle: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = list(all_events)
    aggregates = {name: aggregate_events(all_events[name]) for name in families}
    entropy_mean = [
        np.mean([item["aggregate_metrics"]["entropy_normalized"] for item in aggregates[name]]) for name in families
    ]
    gini_mean = [np.mean([item["aggregate_metrics"]["load_gini"] for item in aggregates[name]]) for name in families]
    dominant_mean = [
        np.mean([item["aggregate_metrics"]["dominant_expert_share"] for item in aggregates[name]])
        for name in families
    ]
    positions = np.arange(len(families))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    ax.bar(positions - width, entropy_mean, width, label="Mean normalized entropy", color="#2F6FAE")
    ax.bar(positions, gini_mean, width, label="Mean load Gini", color="#E19A35")
    ax.bar(positions + width, dominant_mean, width, label="Mean dominant share", color="#50A47B")
    ax.set_xticks(positions, [family.upper() for family in families])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Metric value")
    ax.set_title(f"E3 P0 cross-family routing summary\n{subtitle}")
    ax.grid(axis="y", linewidth=0.7, color="#D8DEE6")
    ax.legend(loc="upper right")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_sample_variation_plot(all_events: dict[str, list[dict[str, Any]]], path: Path, subtitle: str) -> None:
    """Plot per-sample normalized entropy for every routed module."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = list(all_events)
    fig, axes = plt.subplots(len(families), 1, figsize=(12, 3.7 * len(families)), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for axis, family in zip(axes, families):
        modules = sorted({event["module"]["name"] for event in all_events[family]})
        for module in modules:
            selected = [event for event in all_events[family] if event["module"]["name"] == module]
            selected.sort(key=lambda event: event["runtime"]["sample_indices"])
            x = [event["runtime"]["sample_indices"][0] for event in selected]
            y = [event["routing"]["entropy_normalized"] for event in selected]
            axis.plot(x, y, marker="o", linewidth=1.5, label=module)
        axis.set_ylim(-0.03, 1.03)
        axis.set_xticks(sorted({index for event in all_events[family] for index in event["runtime"]["sample_indices"]}))
        axis.set_ylabel("Normalized entropy")
        axis.set_title(f"{family.upper()} per-sample routing variation")
        axis.grid(linewidth=0.7, color="#D8DEE6")
        axis.legend(fontsize=7, loc="best")
    axes[-1].set_xlabel("coco8 val sample index")
    fig.suptitle(f"E3 P0 per-sample routing evidence\n{subtitle}", fontsize=13)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_batch_consistency_plot(comparisons: list[dict[str, Any]], path: Path) -> None:
    """Plot maximum load deltas for each tested batch size and family."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = []
    for comparison in comparisons:
        batch_size = comparison["candidate_batch_size"]
        for family in sorted({item["family"] for item in comparison["modules"]}):
            values = [
                item["max_abs_load_delta"]
                for item in comparison["modules"]
                if item["family"] == family and item["max_abs_load_delta"] is not None
            ]
            rows.append((batch_size, family, max(values, default=0.0)))
    labels = [f"B{batch_size}-{family.upper()}" for batch_size, family, _ in rows]
    values = [value for _, _, value in rows]
    tolerance = comparisons[0]["tolerance"] if comparisons else 1e-5
    fig, ax = plt.subplots(figsize=(10, 5.2), constrained_layout=True)
    bars = ax.bar(labels, values, color="#2F6FAE")
    ax.axhline(tolerance, color="#C83E4D", linestyle="--", label=f"Tolerance {tolerance:g}")
    upper = max(tolerance * 1.25, max(values, default=0.0) * 1.15)
    ax.set_ylim(0.0, upper)
    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.1e}",
            (bar.get_x() + bar.get_width() / 2, value),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Maximum absolute expert-load delta")
    ax.set_title("E3 P0 batch-size consistency against batch=1")
    ax.grid(axis="y", linewidth=0.7, color="#D8DEE6")
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)
