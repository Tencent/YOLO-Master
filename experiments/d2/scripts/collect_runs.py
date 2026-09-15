# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Archive finished training runs into the D2 evidence directory and summarize them into one comparable table.

A run leaves ``results.csv``, ``args.yaml`` and ~60 MB of checkpoints in ``runs/detect/...``. None of that is
committable as-is: the repository's root ``.gitignore`` filters both filenames, and the checkpoints do not belong
in Git at all. Reading several runs also means opening several CSVs and eyeballing the last row of each, which is
how transcription mistakes get into reports.

This copies the per-epoch metrics, audits the resolved args, then writes a summary table and compact CSV manifest.
Full resolved args remain useful as local audit inputs but are ignored by Git to avoid committing near-identical files.

It also re-checks the no-confound constraint **after the fact**: ``validate_pair.py`` proves the configs agree,
but only the ``args.yaml`` of a finished run shows what the trainer actually resolved -- defaults, ``optimizer:
auto``, CLI overrides and all. A field that differs across runs but is not the variable under study is reported
as a confound, whatever the configs promised.

Usage
    python experiments/d2/scripts/collect_runs.py runs/detect/d2/p1voc/* --label p1voc
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # experiments/d2, one level up from scripts/
RESULTS = HERE / "results"

# Fields expected to differ between runs by design; anything else differing is a confound.
DEFAULT_AXIS_KEYS = {
    "name",
    "seed",
    "save_dir",
    "foundation_enabled",
    "foundation_teacher",
    "foundation_model",
    "foundation_loss_weight",
    "foundation_target_levels",
    "foundation_multiscale",
}

# Within one named arm only run identity may vary. This catches an accidental weight or model change between seeds,
# even though those fields are valid axes when different arms are compared.
WITHIN_ARM_AXIS_KEYS = {"name", "seed", "save_dir"}

# Columns worth putting in the summary, mapped to the short headers used in the table.
SUMMARY_COLUMNS = {
    "metrics/mAP50-95(B)": "mAP50-95",
    "metrics/mAP50(B)": "mAP50",
    "train/foundation_task_ratio": "task_ratio",
    "train/foundation_relational_raw": "kd_raw",
    "train/foundation_loss": "kd_weighted",
    "train/mixture_aux_loss": "moe_aux",
    "train/box_loss": "box",
    "train/cls_loss": "cls",
    "time": "seconds",
}

MANIFEST_ARG_KEYS = (
    "seed",
    "data",
    "model",
    "imgsz",
    "batch",
    "optimizer",
    "lr0",
    "lrf",
    "pretrained",
    "amp",
    "deterministic",
    "foundation_enabled",
    "foundation_teacher",
    "foundation_model",
    "foundation_target_levels",
    "foundation_multiscale",
    "foundation_loss_weight",
)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_flat_yaml(path: Path) -> dict[str, str]:
    """Parse the top-level ``key: value`` pairs of an Ultralytics ``args.yaml``.

    Deliberately not a YAML parser: this only needs scalar top-level fields for comparison, and avoiding the
    dependency keeps the script runnable in a bare environment. Nested list values are skipped, so a field like
    ``foundation_target_levels`` is compared only when the trainer wrote it inline.

    Args:
        path (Path): Path to an ``args.yaml``.

    Returns:
        (dict[str, str]): Field name to raw string value.

    Examples:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     sample = pathlib.Path(directory) / "args.yaml"
        ...     _ = sample.write_text("epochs: 3\\nname: a\\n  indented: skipped\\n")
        ...     read_flat_yaml(sample)
        {'epochs': '3', 'name': 'a'}
    """
    fields: dict[str, str] = {}
    active_list = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line or line[0].isspace() or line.lstrip().startswith("#"):
            continue
        if line.startswith("- ") and active_list:
            item = line[2:].split("#")[0].strip()
            fields[active_list] = "+".join(filter(None, (fields[active_list], item)))
            continue
        key, separator, value = line.partition(":")
        if separator:
            value = value.split("#")[0].strip()
            fields[key.strip()] = value
            active_list = key.strip() if not value else None
    return fields


def last_row(csv_path: Path) -> dict[str, str]:
    """Return the final epoch's row of a trainer ``results.csv``, with whitespace-stripped keys."""
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
    if not rows:
        raise ValueError(f"{csv_path} has no epochs")
    return {key.strip(): (value or "").strip() for key, value in rows[-1].items()}


def as_float(value: str) -> float | None:
    """Return a float, or None when the field is absent or not numeric.

    Examples:
        >>> as_float("0.0042"), as_float(""), as_float("n/a")
        (0.0042, None, None)
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_confounds(runs: list[dict], axis_keys: set[str]) -> list[str]:
    """Report resolved-arg fields that differ across runs but are not the declared axis.

    Args:
        runs (list[dict]): Collected run records, each carrying an ``args`` mapping.
        axis_keys (set[str]): Fields allowed to differ.

    Returns:
        (list[str]): Human-readable descriptions, empty when the comparison is clean.
    """
    if len(runs) < 2:
        return []
    keys = set().union(*(set(run["args"]) for run in runs))
    problems = []
    for key in sorted(keys - axis_keys):
        values = {run["args"].get(key) for run in runs}
        if len(values) > 1:
            shown = ", ".join(sorted(str(value) for value in values))
            problems.append(f"{key}: {shown}")
    return problems


def find_within_arm_confounds(runs: list[dict]) -> list[str]:
    """Report resolved-arg differences between seeds assigned to the same arm.

    A run name ending in ``-s<seed>`` identifies its arm. Runs without that suffix remain independent and are not
    compared as a group.

    Args:
        runs (list[dict]): Collected run records, each carrying a run name and an ``args`` mapping.

    Returns:
        (list[str]): Arm-prefixed descriptions of fields that drifted between seeds.
    """
    groups: dict[str, list[dict]] = {}
    for run in runs:
        arm = re.sub(r"-s\d+(?:-\d+)?$", "", run["run"])
        groups.setdefault(arm, []).append(run)

    problems = []
    for arm, arm_runs in sorted(groups.items()):
        if len(arm_runs) < 2:
            continue
        problems.extend(f"{arm}.{problem}" for problem in find_confounds(arm_runs, WITHIN_ARM_AXIS_KEYS))
    return problems


def collect(run_dir: Path, label: str | None) -> dict:
    """Archive one run's evidence files and extract its final-epoch numbers."""
    results_csv = run_dir / "results.csv"
    args_yaml = run_dir / "args.yaml"
    if not results_csv.is_file():
        raise FileNotFoundError(f"{run_dir} has no results.csv (did the run finish?)")

    # Renamed on copy: the root .gitignore filters 'results.csv' and 'args.yaml' by basename.
    destination = RESULTS / (f"{label}_{run_dir.name}" if label else run_dir.name)
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(results_csv, destination / "metrics.csv")
    if args_yaml.is_file():
        shutil.copy2(args_yaml, destination / "resolved_args.yaml")

    row = last_row(results_csv)
    return {
        "run": run_dir.name,
        "archived": destination,
        "epochs": row.get("epoch", "?"),
        "args": read_flat_yaml(args_yaml) if args_yaml.is_file() else {},
        "metrics": {short: as_float(row.get(column, "")) for column, short in SUMMARY_COLUMNS.items()},
        "metrics_sha256": sha256_file(results_csv),
        "resolved_args_sha256": sha256_file(args_yaml) if args_yaml.is_file() else "",
    }


def render_table(runs: list[dict]) -> str:
    """Render the collected runs as a Markdown table, omitting columns no run reported."""
    columns = [short for short in SUMMARY_COLUMNS.values() if any(run["metrics"][short] is not None for run in runs)]
    header = "| run | epochs | " + " | ".join(columns) + " |"
    divider = "|---" * (len(columns) + 2) + "|"
    lines = [header, divider]
    for run in runs:
        cells = []
        for short in columns:
            value = run["metrics"][short]
            cells.append("-" if value is None else (f"{value:.5g}" if abs(value) < 1000 else f"{value:.1f}"))
        lines.append(f"| {run['run']} | {run['epochs']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_manifest(runs: list[dict], path: Path) -> None:
    """Write a compact audit manifest so full resolved args need not be committed for every run."""
    fields = ["run", "epochs", *MANIFEST_ARG_KEYS, "metrics_sha256", "resolved_args_sha256"]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for run in runs:
            row = {"run": run["run"], "epochs": run["epochs"]}
            row.update({key: run["args"].get(key, "") for key in MANIFEST_ARG_KEYS})
            for key in ("data", "model"):
                if row[key] and Path(row[key]).is_absolute():
                    row[key] = Path(row[key]).name
            row["metrics_sha256"] = run["metrics_sha256"]
            row["resolved_args_sha256"] = run["resolved_args_sha256"]
            writer.writerow(row)


def main() -> int:
    """Archive the given run directories and write a combined summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", help="training run directories, e.g. runs/detect/d2/p0/wsweep_*")
    parser.add_argument("--label", default=None, help="prefix for archived directory names, e.g. 'wsweep'")
    parser.add_argument("--out", default=None, help="summary path (default: results/<label>_summary.md)")
    parser.add_argument(
        "--axis",
        action="append",
        default=[],
        help="additional field allowed to differ across runs; repeatable",
    )
    args = parser.parse_args()

    runs = []
    for pattern in args.run_dirs:
        path = Path(pattern)
        if path.is_dir():
            runs.append(collect(path, args.label))
        else:
            print(f"skip  {path} (not a directory)")
    if not runs:
        print("no runs collected")
        return 1
    runs.sort(key=lambda run: run["run"])

    table = render_table(runs)
    confounds = find_confounds(runs, DEFAULT_AXIS_KEYS | set(args.axis))
    confounds += find_within_arm_confounds(runs)

    body = [f"# {args.label or 'D2'} run summary", "", table, "", "## 事后无混杂核查", ""]
    if confounds:
        body.append("以下字段在各 run 的源 `args.yaml` 中不一致，且不在声明的对照轴内：")
        body.append("")
        body += [f"- `{problem}`" for problem in confounds]
        body.append("")
        body.append("**在解释上表任何差异之前，必须先解释这些字段为何不同。**")
    else:
        body.append("各 run 的源 `args.yaml` 仅在声明的对照轴内不同；关键字段和哈希另存于 manifest。")
    body += ["", "## 归档", ""] + [f"- `{run['run']}` → `{run['archived'].relative_to(HERE)}/`" for run in runs]

    out = Path(args.out) if args.out else RESULTS / f"{args.label or 'runs'}_summary.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(body) + "\n", encoding="utf-8")
    manifest = RESULTS / f"{args.label or 'runs'}_manifest.csv"
    write_manifest(runs, manifest)

    print(table)
    print()
    if confounds:
        print(f"CONFOUND  {len(confounds)} field(s) differ outside the declared axis:")
        for problem in confounds:
            print(f"  - {problem}")
    else:
        print("no-confound check: PASS")
    print(f"\nsummary -> {out}")
    print(f"manifest -> {manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
