#!/usr/bin/env python3
"""Collect and analyze final training results for VisDrone ablation study."""

import json
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_mot_ablation import (
    SPECS,
    select_specs,
    write_summary,
    benchmark_row,
)


def main():
    project = Path("/root/autodl-tmp/runs/visdrone_mot_ablation")
    specs = select_specs(["v10", "v10_mot", "v10_moa"])

    print("=" * 80)
    print("VisDrone MoT/MoA Ablation Study - Final Results")
    print("=" * 80)
    print()

    # Check training completion
    all_complete = True
    for spec in specs:
        results_csv = project / spec.key / "results.csv"
        if not results_csv.exists():
            print(f"⚠️  {spec.label}: Training not started or results missing")
            all_complete = False
        else:
            with open(results_csv) as f:
                lines = f.readlines()
                if len(lines) < 101:  # header + 100 epochs
                    print(f"⚠️  {spec.label}: Training incomplete ({len(lines)-1}/100 epochs)")
                    all_complete = False
                else:
                    print(f"✓  {spec.label}: Training complete (100/100 epochs)")

    print()

    if not all_complete:
        print("Training is not complete. Run benchmarking anyway? (results may be incomplete)")
        print()

    # Run benchmarking
    print("Running latency benchmarks...")
    print()

    benchmark_specs = []
    for device in ["cpu", "0"]:  # CPU and GPU
        for imgsz in [640]:
            print(f"Benchmarking on device={device}, imgsz={imgsz}")
            rows = []
            for spec in specs:
                try:
                    row = benchmark_row(
                        spec,
                        device=device,
                        imgsz=imgsz,
                        warmup=5,
                        reps=100,
                        actual_flops=True
                    )
                    rows.append(row)
                    print(f"  {spec.label}: {row['latency_ms_p50']}ms (P50)")
                except Exception as e:
                    print(f"  {spec.label}: ERROR - {e}")

            if rows:
                from scripts.compare_mot_ablation import write_csv
                out = project / f"latency_{device}_{imgsz}.csv"
                write_csv(out, rows)
                print(f"  Saved: {out}")
            print()

    # Generate summary
    print("Generating summary...")
    summary_path = write_summary(project, specs)
    print(f"✓ Summary written to: {summary_path}")
    print()

    # Print key metrics
    import csv
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print("=" * 80)
    print("KEY METRICS COMPARISON")
    print("=" * 80)
    print()

    headers = ["Model", "Params(M)", "FLOPs(G)", "mAP50", "mAP50-95", "Latency(ms)", "NaN", "Diverged"]
    col_widths = [25, 12, 12, 10, 12, 14, 8, 10]

    # Print header
    header_row = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_row)
    print("-" * sum(col_widths))

    # Print data
    for row in rows:
        label = row.get("label", "")[:24]
        params_m = row.get("params_m", "")[:11]
        flops = row.get("flops_g", "")[:11]
        map50 = row.get("metrics/mAP50(B)", "")[:9]
        map50_95 = row.get("metrics/mAP50-95(B)", "")[:11]
        latency = row.get("latency_ms_p50", "")[:13]
        nan = row.get("nan_detected", "")[:7]
        diverged = row.get("loss_diverged", "")[:9]

        values = [label, params_m, flops, map50, map50_95, latency, nan, diverged]
        data_row = "".join(str(v).ljust(w) for v, w in zip(values, col_widths))
        print(data_row)

    print()
    print(f"Full results in: {project}")
    print()


if __name__ == "__main__":
    main()
