#!/usr/bin/env python3
"""Aggregate multi-seed MoE/MoT/MoA ablations with completeness checks.

The training summaries contain seed-dependent accuracy and stability metrics. A
single optional benchmark CSV supplies architecture-dependent latency, FLOPs,
and parameter counts, which do not need to be remeasured for every seed.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = {
    "map50_95": ("metrics/mAP50-95(B)", "mAP50-95"),
    "map50": ("metrics/mAP50(B)", "mAP50"),
    "final_train_total_loss": ("final_train_total_loss",),
}
PROFILE_FIELDS = ("latency_ms_p50", "latency_ms_p95", "latency_ms_p99", "flops_g", "params_m")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def as_float(value: object) -> float | None:
    try:
        result = float(value) if value not in {None, ""} else None
    except (TypeError, ValueError):
        return None
    return result if result is not None and math.isfinite(result) else None


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def seed_sort_key(seed: str) -> tuple[int, int | str]:
    return (0, int(seed)) if seed.isdigit() else (1, seed)


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def load_optional_by_key(path: Path | None) -> dict[str, dict[str, str]]:
    if not path:
        return {}
    if not path.exists():
        raise FileNotFoundError(path)
    rows = read_csv(path)
    if any(not row.get("key") for row in rows):
        raise ValueError(f"profile CSV has a row without key: {path}")
    return {row["key"]: row for row in rows}


def collect_seed_rows(
    root: Path,
    expected_seeds: list[str] | None = None,
    allow_incomplete: bool = False,
) -> dict[str, list[dict[str, str]]]:
    summaries = sorted(root.glob("seed_*/summary.csv"), key=lambda path: seed_sort_key(path.parent.name[5:]))
    if not summaries:
        raise ValueError(f"no seed_*/summary.csv files found under {root}")

    discovered_seeds = [path.parent.name[5:] for path in summaries]
    if expected_seeds is not None:
        expected = sorted({str(seed) for seed in expected_seeds}, key=seed_sort_key)
        if discovered_seeds != expected:
            raise ValueError(f"seed mismatch: expected {expected}, found {discovered_seeds}")

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    model_sets: dict[str, set[str]] = {}
    for summary in summaries:
        seed = summary.parent.name[5:]
        rows = read_csv(summary)
        keys = [row.get("key", "") for row in rows]
        if not rows or any(not key for key in keys):
            raise ValueError(f"empty or malformed summary: {summary}")
        if len(keys) != len(set(keys)):
            raise ValueError(f"duplicate model key in {summary}")
        model_sets[seed] = set(keys)
        for row in rows:
            grouped[row["key"]].append({**row, "seed": seed, "source": str(summary)})

    if not allow_incomplete:
        expected_models = set.union(*model_sets.values())
        incomplete = {
            seed: sorted(expected_models - keys)
            for seed, keys in model_sets.items()
            if keys != expected_models
        }
        if incomplete:
            raise ValueError(f"incomplete model coverage by seed: {incomplete}")
    return grouped


def aggregate(
    root: Path,
    latency_csv: Path | None = None,
    build_csv: Path | None = None,
    *,
    baseline_key: str = "v10",
    expected_seeds: list[str] | None = None,
    allow_incomplete: bool = False,
    map_gain_threshold: float = 0.01,
    latency_reduction_threshold_pct: float = 10.0,
) -> list[dict[str, object]]:
    grouped = collect_seed_rows(root, expected_seeds=expected_seeds, allow_incomplete=allow_incomplete)
    latency = load_optional_by_key(latency_csv)
    builds = load_optional_by_key(build_csv)
    output: list[dict[str, object]] = []

    for key, rows in sorted(grouped.items()):
        ordered_rows = sorted(rows, key=lambda row: seed_sort_key(row["seed"]))
        item: dict[str, object] = {
            "key": key,
            "label": ordered_rows[0].get("label", key),
            "n_seeds": len(ordered_rows),
            "seeds": ",".join(row["seed"] for row in ordered_rows),
            "nan_any": any(as_bool(row.get("nan_detected")) for row in ordered_rows),
            "loss_diverged_any": any(as_bool(row.get("loss_diverged")) for row in ordered_rows),
        }
        for output_name, source_names in METRICS.items():
            values = []
            for row in ordered_rows:
                value = next((parsed for name in source_names if (parsed := as_float(row.get(name))) is not None), None)
                if value is not None:
                    values.append(value)
            if values:
                mean, std = mean_std(values)
                item.update(
                    {
                        f"{output_name}_mean": mean,
                        f"{output_name}_std": std,
                        f"{output_name}_min": min(values),
                        f"{output_name}_max": max(values),
                    }
                )

        profile = {**builds.get(key, {}), **latency.get(key, {})}
        fallback = ordered_rows[0]
        for field in PROFILE_FIELDS:
            value = as_float(profile.get(field))
            if value is None:
                value = as_float(fallback.get(field))
            if value is not None:
                item[field] = value
        output.append(item)

    baseline = next((row for row in output if row["key"] == baseline_key), None)
    if baseline is None:
        raise ValueError(f"baseline key {baseline_key!r} is absent")
    baseline_map = as_float(baseline.get("map50_95_mean"))
    baseline_p50 = as_float(baseline.get("latency_ms_p50"))
    for row in output:
        current_map = as_float(row.get("map50_95_mean"))
        current_p50 = as_float(row.get("latency_ms_p50"))
        map_delta = current_map - baseline_map if current_map is not None and baseline_map is not None else None
        latency_delta = (
            (current_p50 - baseline_p50) / baseline_p50 * 100
            if current_p50 is not None and baseline_p50 not in {None, 0}
            else None
        )
        if map_delta is not None:
            row["map50_95_delta_vs_baseline"] = map_delta
        if latency_delta is not None:
            row["latency_delta_pct_vs_baseline"] = latency_delta
        row["meaningful_gain"] = bool(
            (map_delta is not None and map_delta > map_gain_threshold)
            or (latency_delta is not None and latency_delta < -latency_reduction_threshold_pct)
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt(row: dict[str, object], key: str, digits: int = 4) -> str:
    value = as_float(row.get(key))
    return f"{value:.{digits}f}" if value is not None else "N/A"


def write_markdown(path: Path, rows: list[dict[str, object]], title: str, note: str | None = None) -> None:
    lines = [
        f"# {title}",
        "",
        "| Model | Seeds | mAP50-95 mean±std | mAP50 mean±std | P50/P95/P99 ms | "
        "Actual FLOPs (G) | Params (M) | Stable | Gain gate |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    for row in rows:
        map95 = f"{fmt(row, 'map50_95_mean')}±{fmt(row, 'map50_95_std')}"
        map50 = f"{fmt(row, 'map50_mean')}±{fmt(row, 'map50_std')}"
        latency = "/".join(fmt(row, key, 2) for key in ("latency_ms_p50", "latency_ms_p95", "latency_ms_p99"))
        stable = "yes" if not row.get("nan_any") and not row.get("loss_diverged_any") else "no"
        gain = "pass" if row.get("meaningful_gain") else "fail"
        lines.append(
            f"| {row.get('label', row['key'])} | {row['n_seeds']} | {map95} | {map50} | {latency} | "
            f"{fmt(row, 'flops_g', 3)} | {fmt(row, 'params_m', 3)} | {stable} | {gain} |"
        )
    if note:
        lines.extend(["", f"> {note}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Directory containing seed_*/summary.csv.")
    parser.add_argument("--latency-csv", type=Path, help="One hardware-controlled benchmark CSV shared by all seeds.")
    parser.add_argument("--build-csv", type=Path)
    parser.add_argument("--baseline", default="v10")
    parser.add_argument("--expected-seeds", nargs="+")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--map-gain-threshold", type=float, default=0.01)
    parser.add_argument("--latency-reduction-threshold-pct", type=float, default=10.0)
    parser.add_argument("--title", default="MoE/MoT/MoA multi-seed ablation")
    parser.add_argument("--note")
    parser.add_argument("--out-csv", type=Path)
    parser.add_argument("--out-md", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = aggregate(
        args.root,
        args.latency_csv,
        args.build_csv,
        baseline_key=args.baseline,
        expected_seeds=args.expected_seeds,
        allow_incomplete=args.allow_incomplete,
        map_gain_threshold=args.map_gain_threshold,
        latency_reduction_threshold_pct=args.latency_reduction_threshold_pct,
    )
    out_csv = args.out_csv or args.root / "aggregate_multiseed.csv"
    out_md = args.out_md or args.root / "aggregate_multiseed.md"
    write_csv(out_csv, rows)
    write_markdown(out_md, rows, title=args.title, note=args.note)
    print(f"wrote {out_csv}")
    print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
