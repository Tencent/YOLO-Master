#!/usr/bin/env python3
"""Archive and plot locally executed COCO experiments for Issue #52."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCAL_RESULTS = ROOT / "runs/issue52_coco128_corrected/pruning/results.csv"
DYNAMIC_SUMMARY = ROOT / "runs/issue52_coco128_dynamic_selfrun_v2/dynamic_schedule_summary.csv"
DYNAMIC_TRACE = ROOT / "runs/issue52_coco128_dynamic_selfrun_v2/issue52_gini_balance/moe_dynamic_schedule.csv"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def num(row: dict[str, object], key: str) -> float:
    return float(row[key])


def relative_checkpoint(value: str) -> str:
    try:
        return str(Path(value).relative_to(ROOT))
    except ValueError:
        return value


def archive_pruning(source: Path, output: Path) -> list[dict[str, str]]:
    archived = output / "coco128-pruning-results.csv"
    if source.exists():
        rows: list[dict[str, object]] = []
        for row in read_csv(source):
            rows.append(
                {
                    "threshold": row["threshold"],
                    "stage": row["recovery"],
                    "dataset": "coco128",
                    "checkpoint": relative_checkpoint(row["checkpoint"]),
                    "mAP50-95": row["map50_95"],
                    "mAP50": row["map50"],
                    "GFLOPs": row["gflops"],
                    "latency_mean_ms": row["latency_ms"],
                    "params_M": row["params_m"],
                    "mean_gini": row["mean_gini"],
                    "experts_per_layer": row["experts_per_layer"],
                    "layer_gini": row["layer_gini"],
                }
            )
        write_csv(archived, rows)
    if not archived.exists():
        raise FileNotFoundError(f"Local COCO128 results not found: {source}")
    return read_csv(archived)


def derive_layers(output: Path, pruning: list[dict[str, str]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in pruning:
        for layer, retained in sorted(
            json.loads(row["experts_per_layer"]).items(), key=lambda item: int(item[0].split(".")[-1])
        ):
            rows.append(
                {
                    "threshold": row["threshold"],
                    "stage": row["stage"],
                    "layer_name": layer.replace("model.base_model.model.", "model."),
                    "retained_experts": retained,
                    "original_experts": 3,
                }
            )
    write_csv(output / "coco128-per-layer-experts.csv", rows)
    return rows


def derive_pareto(output: Path, pruning: list[dict[str, str]]) -> list[dict[str, object]]:
    dense = next(row for row in pruning if row["stage"] == "dense")
    ranked = sorted(pruning, key=lambda row: (num(row, "latency_mean_ms"), -num(row, "mAP50-95")))
    rows: list[dict[str, object]] = []
    best_map = -1.0
    for row in ranked:
        is_pareto = num(row, "mAP50-95") > best_map
        best_map = max(best_map, num(row, "mAP50-95"))
        pruned = row["experts_per_layer"] != dense["experts_per_layer"]
        rows.append(
            {
                "threshold": row["threshold"],
                "stage": row["stage"],
                "mAP50-95": row["mAP50-95"],
                "latency_mean_ms": row["latency_mean_ms"],
                "structurally_pruned": str(pruned).lower(),
                "quality_gate_pass": str(
                    pruned and num(dense, "mAP50-95") - num(row, "mAP50-95") <= 0.01
                ).lower(),
                "pareto": str(is_pareto).lower(),
            }
        )
    write_csv(output / "coco128-pareto-accuracy-latency.csv", rows)
    return rows


def archive_dynamic(summary: Path, trace: Path, output: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    summary_out = output / "coco128-dynamic-schedule-results.csv"
    trace_out = output / "coco128-dynamic-gini-trace.csv"
    if summary.exists():
        write_csv(summary_out, read_csv(summary))
    if trace.exists():
        write_csv(trace_out, read_csv(trace))
    missing = [path for path in (summary_out, trace_out) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required local dynamic experiment outputs are missing: {missing}")
    return read_csv(summary_out), read_csv(trace_out)


def archive_auxiliary(output: Path) -> None:
    write_csv(
        output / "coco128-resource-profile.csv",
        [
            {
                "probe": "oom_boundary",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition 97GB",
                "batch": 128,
                "imgsz": 1600,
                "result": "OOM; automatic batch fallback reached 16",
                "peak_allocated_GiB": "",
                "peak_reserved_GiB": "",
            },
            {
                "probe": "stable_high_utilization",
                "gpu": "NVIDIA RTX PRO 6000 Blackwell Server Edition 97GB",
                "batch": 36,
                "imgsz": 1344,
                "result": "completed without batch fallback",
                "peak_allocated_GiB": 87.28,
                "peak_reserved_GiB": 92.89,
            },
        ],
    )


def build_plots(
    pruning: list[dict[str, str]],
    layers: list[dict[str, object]],
    pareto: list[dict[str, object]],
    dynamic: list[dict[str, str]],
    trace: list[dict[str, str]],
    output: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    dense = next(row for row in pruning if row["stage"] == "dense")
    colors = {"direct": "#2878B5", "lora10": "#F28E2B"}
    for key, ylabel, filename in (
        ("mAP50-95", "mAP50-95", "coco128-threshold-map.png"),
        ("GFLOPs", "Whole-model GFLOPs", "coco128-threshold-gflops.png"),
        ("latency_mean_ms", "Latency (ms/image)", "coco128-threshold-latency.png"),
    ):
        fig, axis = plt.subplots(figsize=(7.2, 4.4))
        axis.axhline(num(dense, key), color="#555555", linestyle="--", label="dense baseline")
        for stage in ("direct", "lora10"):
            group = sorted((row for row in pruning if row["stage"] == stage), key=lambda row: num(row, "threshold"))
            axis.plot(
                [num(row, "threshold") for row in group],
                [num(row, key) for row in group],
                marker="o",
                linewidth=2,
                label=stage,
                color=colors[stage],
            )
        axis.set(xlabel="Pruning threshold", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)

    points = [row for row in pruning if row["stage"] != "dense"]
    fig = plt.figure(figsize=(8.2, 6.2))
    axis = fig.add_subplot(111, projection="3d")
    scatter = axis.scatter(
        [num(row, "threshold") for row in points],
        [num(row, "GFLOPs") for row in points],
        [num(row, "latency_mean_ms") for row in points],
        c=[num(row, "mAP50-95") for row in points],
        cmap="viridis",
        s=55,
    )
    axis.set(xlabel="Threshold", ylabel="Whole-model GFLOPs", zlabel="Latency (ms/image)")
    fig.colorbar(scatter, ax=axis, label="mAP50-95", shrink=0.72)
    fig.tight_layout()
    fig.savefig(output / "coco128-threshold-map-flops-latency-3d.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.5, 5.0))
    for row in pareto:
        marker = "s" if row["structurally_pruned"] == "true" else "o"
        axis.scatter(num(row, "latency_mean_ms"), num(row, "mAP50-95"), marker=marker, s=72, alpha=0.8)
        axis.annotate(
            f"t={float(row['threshold']):.2f} {row['stage']}",
            (num(row, "latency_mean_ms"), num(row, "mAP50-95")),
            xytext=(4, 5),
            textcoords="offset points",
            fontsize=7,
        )
    front = [row for row in pareto if row["pareto"] == "true"]
    axis.plot(
        [num(row, "latency_mean_ms") for row in front],
        [num(row, "mAP50-95") for row in front],
        color="#D62728",
        linewidth=1.8,
        label="Pareto front",
    )
    axis.set(xlabel="Latency (ms/image)", ylabel="mAP50-95", title="COCO128: no pruned point passes 0.01 mAP gate")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output / "coco128-pareto-accuracy-latency.png", dpi=180)
    plt.close(fig)

    direct = [row for row in layers if row["stage"] in {"dense", "direct"}]
    thresholds = sorted({float(row["threshold"]) for row in direct})
    names = sorted({str(row["layer_name"]) for row in direct}, key=lambda value: int(value.split(".")[-1]))
    matrix = np.array(
        [
            [
                int(
                    next(
                        row
                        for row in direct
                        if float(row["threshold"]) == threshold and row["layer_name"] == layer
                    )["retained_experts"]
                )
                for layer in names
            ]
            for threshold in thresholds
        ]
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.0))
    image = axis.imshow(matrix, cmap="Blues", vmin=1, vmax=3, aspect="auto")
    axis.set_xticks(range(len(names)), names)
    axis.set_yticks(range(len(thresholds)), [f"{value:.2f}" for value in thresholds])
    axis.set(xlabel="MoE layer", ylabel="Threshold (0=dense)", title="COCO128 retained experts")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]}/3", ha="center", va="center")
    fig.colorbar(image, ax=axis, label="Retained experts")
    fig.tight_layout()
    fig.savefig(output / "coco128-per-layer-retained-experts.png", dpi=180)
    plt.close(fig)

    if dynamic:
        labels = [row["variant"] for row in dynamic]
        x = np.arange(len(dynamic))
        width = 0.34
        fig, left = plt.subplots(figsize=(7.4, 4.8))
        left.bar(x - width / 2, [num(row, "final_mAP50-95") for row in dynamic], width, label="Final")
        left.bar(x + width / 2, [num(row, "best_mAP50-95") for row in dynamic], width, label="Best")
        left.set_xticks(x, labels)
        left.set_ylabel("mAP50-95")
        left.grid(axis="y", alpha=0.2)
        right = left.twinx()
        right.plot(
            x,
            [float(row["convergence_epoch_ratio"] or "nan") for row in dynamic],
            color="#D62728",
            marker="D",
            label="Epoch ratio",
        )
        right.set_ylabel("Epoch ratio to 95% baseline-final target")
        handles_l, labels_l = left.get_legend_handles_labels()
        handles_r, labels_r = right.get_legend_handles_labels()
        left.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "coco128-dynamic-schedule-comparison.png", dpi=180)
        plt.close(fig)

    if trace:
        fig, left = plt.subplots(figsize=(7.2, 4.4))
        epochs = [num(row, "epoch") for row in trace]
        left.plot(epochs, [num(row, "mean_gini") for row in trace], marker="o", label="mean Gini")
        left.plot(epochs, [num(row, "ema_gini") for row in trace], marker="s", label="EMA Gini")
        left.set(xlabel="Epoch", ylabel="Gini")
        left.grid(alpha=0.25)
        right = left.twinx()
        right.plot(
            epochs,
            [num(row, "balance_loss_coeff") for row in trace],
            color="#D62728",
            marker="D",
            label="coefficient",
        )
        right.set_ylabel("Balance loss coefficient")
        handles_l, labels_l = left.get_legend_handles_labels()
        handles_r, labels_r = right.get_legend_handles_labels()
        left.legend(handles_l + handles_r, labels_l + labels_r, fontsize=8)
        fig.tight_layout()
        fig.savefig(output / "coco128-dynamic-gini-trace.png", dpi=180)
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-results", type=Path, default=LOCAL_RESULTS)
    parser.add_argument("--dynamic-summary", type=Path, default=DYNAMIC_SUMMARY)
    parser.add_argument("--dynamic-trace", type=Path, default=DYNAMIC_TRACE)
    parser.add_argument("--data-output", type=Path, default=ROOT / "reports/moe-pruning")
    parser.add_argument("--figure-output", type=Path, default=ROOT / "reports/issue52-figs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_output = args.data_output.resolve()
    figure_output = args.figure_output.resolve()
    data_output.mkdir(parents=True, exist_ok=True)
    pruning = archive_pruning(args.local_results.resolve(), data_output)
    layers = derive_layers(data_output, pruning)
    pareto = derive_pareto(data_output, pruning)
    dynamic, trace = archive_dynamic(args.dynamic_summary.resolve(), args.dynamic_trace.resolve(), data_output)
    archive_auxiliary(data_output)
    build_plots(pruning, layers, pareto, dynamic, trace, figure_output)
    print(f"Local COCO128 CSV files written to {data_output}")
    print(f"Local COCO128 figures written to {figure_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
