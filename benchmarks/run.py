"""Run standardized YOLO-Master benchmark suites."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATALOG = ROOT / "benchmarks/suites.yaml"


def build_parser() -> argparse.ArgumentParser:
    """Build CLI options without loading PyTorch or model modules."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default="mixture_smoke", help="suite name from benchmarks/suites.yaml")
    parser.add_argument("--catalog", type=Path, default=CATALOG, help="suite catalog YAML")
    parser.add_argument("--list", action="store_true", help="list available suites and exit")
    parser.add_argument("--case", dest="cases", action="append", help="run one case ID; repeat to select several")
    parser.add_argument("--device", help="cpu, mps, cuda:0, numeric CUDA ID, or auto")
    parser.add_argument("--dtype", choices=("fp32", "fp16", "bf16"), help="model and input dtype")
    parser.add_argument("--batch", type=int, help="batch size override")
    parser.add_argument("--imgsz", type=int, help="square synthetic input size override")
    parser.add_argument("--warmup", type=int, help="warmup iterations override")
    parser.add_argument("--iterations", type=int, help="timed iterations override")
    parser.add_argument("--seed", type=int, help="random seed override")
    parser.add_argument("--no-flops", action="store_true", help="disable THOP FLOPs collection")
    parser.add_argument("--no-routing", action="store_true", help="disable routing health collection")
    parser.add_argument("--output", type=Path, help="output directory")
    parser.add_argument("--force", action="store_true", help="rerun successful matching cases instead of resuming")
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    mapping = {
        "device": args.device,
        "dtype": args.dtype,
        "batch_size": args.batch,
        "image_size": args.imgsz,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
    }
    values = {key: value for key, value in mapping.items() if value is not None}
    if args.no_flops:
        values["collect_flops"] = False
    if args.no_routing:
        values["collect_routing"] = False
    return values


def main(argv: list[str] | None = None) -> int:
    """Resolve a suite, execute it, and print a compact result table."""
    args = build_parser().parse_args(argv)
    from benchmarks.suite import BenchmarkSuiteRunner, list_suites, load_suite

    if args.list:
        for name, description in list_suites(args.catalog).items():
            print(f"{name}: {description}")
        return 0

    suite = load_suite(
        args.catalog,
        args.suite,
        root=ROOT,
        overrides=_overrides(args),
        case_ids=args.cases,
    )
    output = args.output or ROOT / "runs/benchmarks" / suite.name
    report = BenchmarkSuiteRunner(suite, output).run(resume=not args.force)

    print(f"Suite: {report.suite_name}")
    print(f"Reports: {output / 'results.json'}, {output / 'results.csv'}, {output / 'report.md'}")
    print(f"{'case':<20} {'status':<8} {'params':>12} {'P50 ms':>10} {'P95 ms':>10} {'items/s':>10}")
    for case in report.cases:
        metrics = case.metrics
        print(
            f"{case.case_id:<20} {case.status:<8} "
            f"{_display(metrics.get('params_total'), 0):>12} "
            f"{_display(metrics.get('latency_p50_ms'), 3):>10} "
            f"{_display(metrics.get('latency_p95_ms'), 3):>10} "
            f"{_display(metrics.get('throughput_items_s'), 2):>10}"
        )
        if case.error:
            print(f"  error: {case.error}")
    return 1 if any(case.status == "failed" for case in report.cases) else 0


def _display(value: Any, digits: int) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


if __name__ == "__main__":
    raise SystemExit(main())
