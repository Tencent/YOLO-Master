"""Compare zero-candidate, pre-conflict zero-positive, and final zero-positive rates from training CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


SIZES = ("all", "small", "medium", "large")


def _read_rows(path: Path) -> list[dict[str, str]]:
    """Read an Ultralytics results CSV while normalizing whitespace in column names."""
    with path.open(encoding="utf-8-sig", newline="") as file:
        return [{key.strip(): value.strip() for key, value in row.items()} for row in csv.DictReader(file)]


def _select_row(rows: list[dict[str, str]], epoch: int | None, path: Path) -> dict[str, str]:
    """Select the requested one-based epoch, or the final recorded epoch."""
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    if epoch is None:
        return rows[-1]
    for row in rows:
        if int(float(row["epoch"])) == epoch:
            return row
    raise ValueError(f"Epoch {epoch} not found in {path}")


def _detect_base(row: dict[str, str], branch: str) -> str:
    """Resolve the assignment metric prefix for standard or end-to-end detection losses."""
    if branch != "auto":
        return "assign" if branch == "root" else f"assign/{branch}"
    if "assign/gt_total" in row:
        return "assign"
    if "assign/o2m/gt_total" in row:
        return "assign/o2m"
    raise ValueError("No assignment telemetry found (expected assign/gt_total or assign/o2m/gt_total)")


def _number(row: dict[str, str], key: str) -> float:
    """Read one required numeric field with an actionable error for legacy CSV files."""
    if key not in row or row[key] == "":
        raise ValueError(
            f"Missing column '{key}'. This CSV predates the zero-candidate probe; "
            "candidate-stage counts cannot be reconstructed from final positives."
        )
    return float(row[key])


def _stats(row: dict[str, str], branch: str) -> tuple[str, dict[str, dict[str, float]]]:
    """Calculate the three zero-rate stages for every COCO area bucket."""
    base = _detect_base(row, branch)
    output = {}
    for size in SIZES:
        suffix = "gt" if size == "all" else size
        denominator_key = "gt_total" if size == "all" else f"gt_{size}"
        denominator = _number(row, f"{base}/{denominator_key}")
        zero_candidate = _number(row, f"{base}/zero_candidate_{suffix}")
        zero_preconflict = _number(row, f"{base}/zero_preconflict_{suffix}")
        zero_final_key = "zero_gt" if size == "all" else f"zero_{size}"
        zero_final = _number(row, f"{base}/{zero_final_key}")
        divisor = max(denominator, 1.0)
        output[size] = {
            "gt": denominator,
            "candidate": zero_candidate / divisor,
            "preconflict": zero_preconflict / divisor,
            "final": zero_final / divisor,
            "topk_added": (zero_preconflict - zero_candidate) / divisor,
            "conflict_added": (zero_final - zero_preconflict) / divisor,
        }
    return base, output


def main() -> None:
    """Print a stage-by-stage assignment funnel for one or more result files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path, help="One or more run results.csv files")
    parser.add_argument("--epoch", type=int, help="One-based epoch to inspect; defaults to the final CSV row")
    parser.add_argument("--branch", choices=("auto", "root", "o2m", "o2o"), default="auto")
    args = parser.parse_args()

    print("Rates are percentages of valid GTs; +topk and +conflict show newly-zero GTs at each stage.")
    print(
        f"{'run':<30} {'epoch':>5} {'size':>7} {'GT':>9} {'0-candidate':>12} "
        f"{'0-after-topk':>13} {'+topk':>9} {'0-final':>9} {'+conflict':>10}"
    )
    for path in args.results:
        row = _select_row(_read_rows(path), args.epoch, path)
        _, summary = _stats(row, args.branch)
        run_name = path.parent.name
        epoch = int(float(row["epoch"]))
        for size in SIZES:
            item = summary[size]
            print(
                f"{run_name:<30} {epoch:>5} {size:>7} {item['gt']:>9.0f} {100 * item['candidate']:>11.3f}% "
                f"{100 * item['preconflict']:>12.3f}% {100 * item['topk_added']:>+8.3f} "
                f"{100 * item['final']:>8.3f}% {100 * item['conflict_added']:>+9.3f}"
            )


if __name__ == "__main__":
    main()
