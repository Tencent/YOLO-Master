"""Plot P1 results with default matplotlib styling."""

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")

import matplotlib.pyplot as plt
import numpy as np

plt.switch_backend("Agg")

ROOT = Path(__file__).resolve().parents[3]
RESULTS = ROOT / "experiments/d2/results"
OUTPUT = ROOT / "experiments/d2/docs/p1/assets"
CELLS = ("off", "a", "b", "c")
SEEDS = (17, 29, 43)
LABELS = ("off", "A: DINOv3 P4", "B: DINOv3 multiscale", "C: SigLIP2 P4")


def load_data():
    """Read aligned epoch metrics for all twelve runs."""
    curves = []
    reference = None
    for cell in CELLS:
        runs = []
        for seed in SEEDS:
            with (RESULTS / f"p1voc_{cell}-s{seed}" / "metrics.csv").open(newline="") as file:
                rows = list(csv.DictReader(file))
            epochs = np.array([int(row["epoch"]) for row in rows])
            if reference is None:
                reference = epochs
            if not np.array_equal(epochs, reference) or len(epochs) != 400:
                raise ValueError(f"Expected aligned 400-epoch runs: {cell}, seed {seed}")
            runs.append([float(row["metrics/mAP50-95(B)"]) for row in rows])
        curves.append(runs)
    data = np.array(curves)
    if not np.isfinite(data).all():
        raise ValueError("Metrics contain non-finite values")
    return reference, data


def save(figure, filename):
    """Save a report figure as PNG."""
    figure.tight_layout()
    figure.savefig(OUTPUT / filename, dpi=200)
    plt.close(figure)


def main():
    """Generate learning curves, paired differences, and final seed results."""
    plt.style.use("default")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    epochs, data = load_data()
    means = data.mean(axis=1)

    figure, axis = plt.subplots()
    for index, label in enumerate(LABELS):
        (line,) = axis.plot(epochs, means[index], label=label)
        spread = data[index].std(axis=0, ddof=1)
        axis.fill_between(epochs, means[index] - spread, means[index] + spread, color=line.get_color(), alpha=0.1)
    axis.set(xlabel="Epoch", ylabel="mAP50-95", title="P1 learning curves (3 seeds)")
    axis.legend()
    save(figure, "p1-learning-curves.png")

    figure, axis = plt.subplots()
    for index in range(1, len(CELLS)):
        paired_delta = 100 * (data[index] - data[0])
        axis.plot(epochs, paired_delta.mean(axis=0), label=LABELS[index])
    axis.axhline(0, color="black", linewidth=1)
    axis.set(xlabel="Epoch", ylabel="Delta mAP50-95 (points)", title="KD minus off (paired seed mean)")
    axis.legend()
    save(figure, "p1-paired-deltas.png")

    figure, axis = plt.subplots()
    positions = np.arange(len(CELLS))
    final = data[:, :, -1]
    for index, (seed, marker) in enumerate(zip(SEEDS, ("o", "s", "^"))):
        axis.scatter(positions + (index - 1) * 0.1, final[:, index], label=f"Seed {seed}", marker=marker)
    # Two-sided t interval for each cell mean, df=2; not a paired-effect interval.
    margins = 4.302652729911275 * final.std(axis=1, ddof=1) / np.sqrt(len(SEEDS))
    axis.errorbar(positions, final.mean(axis=1), yerr=margins, fmt="k_", capsize=4, label="Mean and 95% CI")
    axis.set_xticks(positions, ("off", "A", "B", "C"))
    axis.set(xlabel="Configuration", ylabel="mAP50-95", title="Final results (epoch 400)")
    axis.legend()
    save(figure, "p1-final-seeds.png")
    print(f"Wrote three figures to {OUTPUT}")


if __name__ == "__main__":
    main()
