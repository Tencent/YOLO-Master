"""Run a P1 matrix, one training run per row of the matrix csv.

The matrix is the single source of truth: N rows means N runs, each named after its
``run_id`` so the archived directories line up with the matrix without any string surgery.
``--matrix`` / ``--configs`` point at a different matrix and config directory, so a second
experiment can reuse this runner without copying it.

Already-finished runs are skipped, so a batch interrupted halfway can simply be re-invoked.

Examples:
    $ python experiments/d2/scripts/run_p1.py --dry-run
    $ python experiments/d2/scripts/run_p1.py --device 0
    $ python experiments/d2/scripts/run_p1.py --device 0 --only c-s17
    $ python experiments/d2/scripts/run_p1.py --device 0 --only off-s17,a-s17
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent.parent  # experiments/d2
ROOT = HERE.parents[1]


def build_command(row: dict, device: str, configs: str = ".") -> list[str]:
    """Return the yolo command for one matrix row.

    ``seed`` and ``name`` are always overridden: the configs hard-code seed 17, so without the
    override every seed would write into the same directory under a mislabelled name.

    Args:
        row (dict): One matrix record.
        device (str): Value for the ``device`` argument.
        configs (str): Sub-directory of ``experiments/d2/configs`` holding this matrix's arms.

    Returns:
        (list[str]): Argument vector, safe to pass to subprocess without a shell.

    Examples:
        >>> build_command({"run_id": "a-s29", "config": "p1_a.yaml", "seed": "29"}, "0")[:2]
        ['yolo', 'train']
        >>> "name=a-s29" in build_command({"run_id": "a-s29", "config": "p1_a.yaml", "seed": "29"}, "0")
        True
        >>> "p1_coco" in build_command({"run_id": "a-s29", "config": "a.yaml", "seed": "29"}, "0", "p1_coco")[2]
        True
    """
    return [
        "yolo",
        "train",
        f"cfg={HERE / 'configs' / configs / row['config']}",
        f"seed={row['seed']}",
        f"name={row['run_id']}",
        f"device={device}",
    ]


def main() -> None:
    """Run every matrix row that has not completed yet."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="0")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and exit")
    parser.add_argument(
        "--only",
        default=None,
        help="run just these run_ids, comma-separated (e.g. off-s17,a-s17) -- for the staged pilot",
    )
    parser.add_argument("--project", default="d2/p1voc", help="must match `project` in the configs")
    parser.add_argument(
        "--matrix",
        default=str(HERE / "p1_voc_matrix.csv"),
        help="matrix csv to drive; one training run per row",
    )
    parser.add_argument(
        "--configs",
        default="p1_voc",
        help="sub-directory of experiments/d2/configs holding this matrix's arms (e.g. p1_coco)",
    )
    args = parser.parse_args()

    with open(args.matrix) as matrix_file:
        rows = list(csv.DictReader(matrix_file))
    if args.only:
        # Order follows the matrix, not the flag, so a pilot's runs land in a stable order
        # however the ids were typed.
        wanted = [name.strip() for name in args.only.split(",") if name.strip()]
        unknown = sorted(set(wanted) - {r["run_id"] for r in rows})
        if unknown:
            sys.exit(f"no matrix row with run_id in {unknown}")
        rows = [r for r in rows if r["run_id"] in set(wanted)]

    # Logs are namespaced by the project's last segment so two matrices (coco128 and COCO) can
    # be driven from the same checkout without one batch's logs overwriting the other's.
    label = args.project.rsplit("/", 1)[-1]
    out_root = ROOT / "runs" / "detect" / args.project
    log_dir = HERE / "results" / f"{label}_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    planned, done = [], []
    for row in rows:
        # results.csv is written throughout training; its presence means the run got going and
        # a finished run keeps it, so it is the cheapest "already done" marker available.
        if (out_root / row["run_id"] / "results.csv").is_file():
            done.append(row["run_id"])
        else:
            planned.append(row)

    if done:
        print(f"skipping {len(done)} already-present run(s): {', '.join(done)}\n")
    print(f"{len(planned)} run(s) to go\n")

    if args.dry_run:
        for row in planned:
            print(" ", " ".join(build_command(row, args.device, args.configs)))
        return

    failures = []
    for index, row in enumerate(planned, 1):
        run_id = row["run_id"]
        cmd = build_command(row, args.device, args.configs)
        log_path = log_dir / f"{run_id}.log"
        print(f"[{index}/{len(planned)}] {run_id}  -> {log_path.relative_to(ROOT)}")
        started = time.time()
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.time() - started
        if result.returncode == 0:
            print(f"      ok   ({elapsed:.0f}s)")
        else:
            failures.append(run_id)
            print(f"      FAIL (exit {result.returncode}, {elapsed:.0f}s) -- tail of the log:")
            for line in log_path.read_text().splitlines()[-15:]:
                print("        ", line)

    print()
    if failures:
        print(f"{len(failures)} run(s) failed: {', '.join(failures)}")
        print("Fix the cause and re-invoke; finished runs are skipped automatically.")
        sys.exit(1)
    print(f"all {len(planned)} run(s) finished. Next:")
    print(f"  python experiments/d2/scripts/collect_runs.py runs/detect/{args.project}/* --label {label}")


if __name__ == "__main__":
    main()
