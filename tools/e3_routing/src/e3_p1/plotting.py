"""Evidence plots for the P1 paired training-path benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def save_benchmark_plot(families: dict[str, dict[str, Any]], path: Path, target_percent: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(families)
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)
    x = np.arange(len(names))
    off = [families[name]["off_ms"]["median"] for name in names]
    on = [families[name]["on_ms"]["median"] for name in names]
    width = 0.34
    axes[0].bar(x - width / 2, off, width, label="Hook off", color="#61758A")
    axes[0].bar(x + width / 2, on, width, label="Hook on", color="#2F6FAE")
    axes[0].set_xticks(x, [name.upper() for name in names])
    axes[0].set_ylabel("Median train-path forward+backward (ms)")
    axes[0].set_title("Paired observer timing")
    axes[0].grid(axis="y", color="#D8DEE6")
    axes[0].legend()

    slowdowns = [families[name]["paired_slowdown_percent"]["median"] for name in names]
    lower = [families[name]["paired_slowdown_percent"]["bootstrap_95_ci"][0] for name in names]
    upper = [families[name]["paired_slowdown_percent"]["bootstrap_95_ci"][1] for name in names]
    errors = np.asarray([[value - low for value, low in zip(slowdowns, lower)], [high - value for value, high in zip(slowdowns, upper)]])
    colors = ["#50A47B" if value < target_percent else "#C83E4D" for value in slowdowns]
    axes[1].bar(x, slowdowns, yerr=errors, capsize=5, color=colors)
    axes[1].axhline(target_percent, color="#C83E4D", linestyle="--", label=f"Target < {target_percent:g}%")
    axes[1].axhline(0.0, color="#61758A", linewidth=0.9)
    axes[1].set_xticks(x, [name.upper() for name in names])
    axes[1].set_ylabel("Paired slowdown (%)")
    axes[1].set_title("Median and bootstrap 95% CI")
    axes[1].grid(axis="y", color="#D8DEE6")
    axes[1].legend()
    figure.suptitle("E3 P1 observer-overhead benchmark · real train-mode forward+backward")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_strengthened_plot(families: dict[str, dict[str, Any]], path: Path, target_percent: float) -> None:
    """Plot full-step condition timings and layered paired slowdowns."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(families)
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    width = 0.25
    condition_styles = (
        ("off", "Off", "#61758A"),
        ("capture", "Capture", "#2F6FAE"),
        ("capture_jsonl", "Capture + JSONL", "#50A47B"),
    )
    for offset, (condition, label, color) in zip((-width, 0.0, width), condition_styles):
        values = [families[name]["condition_ms"][condition]["median"] for name in names]
        axes[0].bar(x + offset, values, width, label=label, color=color)
    axes[0].set_xticks(x, [name.upper() for name in names])
    axes[0].set_ylabel("Median two-batch full-step unit (ms)")
    axes[0].set_title("Detection loss + backward + optimizer")
    axes[0].grid(axis="y", color="#D8DEE6")
    axes[0].legend()

    comparison_styles = (
        ("capture_vs_off_percent", "Capture vs off", "#2F6FAE", -0.12),
        ("capture_jsonl_vs_off_percent", "Capture + JSONL vs off", "#50A47B", 0.12),
    )
    for key, label, color, offset in comparison_styles:
        values = [families[name]["comparisons"][key]["median"] for name in names]
        lower = [families[name]["comparisons"][key]["bootstrap_95_ci"][0] for name in names]
        upper = [families[name]["comparisons"][key]["bootstrap_95_ci"][1] for name in names]
        errors = np.asarray(
            [[value - low for value, low in zip(values, lower)], [high - value for value, high in zip(values, upper)]]
        )
        axes[1].errorbar(x + offset, values, yerr=errors, fmt="o", capsize=5, color=color, label=label)
    axes[1].axhline(target_percent, color="#C83E4D", linestyle="--", label=f"Task target < {target_percent:g}%")
    axes[1].axhline(0.0, color="#61758A", linewidth=0.9)
    axes[1].set_xticks(x, [name.upper() for name in names])
    axes[1].set_ylabel("Paired slowdown (%)")
    axes[1].set_title("Median and bootstrap 95% CI")
    axes[1].grid(axis="y", color="#D8DEE6")
    axes[1].legend()
    figure.suptitle("E3 P1 strengthened layered observer benchmark")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_engine_crosscheck_plot(families: dict[str, dict[str, Any]], path: Path, target_percent: float) -> None:
    """Plot descriptive Trainer epoch-window pairs without making a new formal verdict."""

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(families)
    x = np.arange(len(names))
    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    values = [families[name]["epoch_window_slowdown_percent"]["values"] for name in names]
    axes[0].boxplot(values, tick_labels=[name.upper() for name in names], showmeans=True)
    for index, samples in enumerate(values, start=1):
        axes[0].scatter(np.full(len(samples), index), samples, color="#2F6FAE", s=24, zorder=3)
    axes[0].axhline(target_percent, color="#C83E4D", linestyle="--", label=f"P1 context: {target_percent:g}%")
    axes[0].axhline(0.0, color="#61758A", linewidth=0.9)
    axes[0].set_ylabel("Paired epoch-window slowdown (%)")
    axes[0].set_title("Six Trainer pairs per family (descriptive)")
    axes[0].grid(axis="y", color="#D8DEE6")
    axes[0].legend()

    event_counts = [families[name]["writer_event_count"] for name in names]
    # Event volume is an integrity metric, not a pass/fail result for the descriptive timing panel.
    axes[1].bar(x, event_counts, color=["#3B82F6", "#14B8A6", "#8B5CF6"])
    axes[1].set_xticks(x, [name.upper() for name in names])
    axes[1].set_ylabel("Trainer-generated JSONL events")
    axes[1].set_title("Writer stream consumed by dashboard HTTP API")
    axes[1].grid(axis="y", color="#D8DEE6")
    for index, count in enumerate(event_counts):
        axes[1].text(index, count, str(count), ha="center", va="bottom")
    figure.suptitle("E3 P1 · real DetectionTrainer integration cross-check (not the formal verdict)")
    figure.savefig(path, dpi=180)
    plt.close(figure)
