#!/usr/bin/env python3
"""LOVO Data Collection & Validation Script.

Collects (fingerprint, variant, ΔmAP) data points from training artifacts,
validates them via Leave-One-Variant-Out (LOVO) cross-validation, and
benchmarks against the paper's claimed metrics.

Usage:
    python scripts/collect_lovo_data.py collect --from-paper --output lovo_data.json
    python scripts/collect_lovo_data.py validate --input lovo_data.json --report report.json
    python scripts/collect_lovo_data.py benchmark --report benchmark.json
    python scripts/collect_lovo_data.py fit --from-paper --coefficients coeffs.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add project root to sys.path so that `ultralytics` is importable
# when the script is run directly.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from ultralytics.utils.lora.planner import (
    ArchitectureFingerprint,
    LOVODataCollector,
    LOVODataPoint,
    LOVOValidator,
    PEFTPlanner,
)


# =============================================================================
# Paper canonical data points
# =============================================================================

def _make_paper_history_10() -> List[LOVODataPoint]:
    """Return the 10 canonical paper data points (Table 1, no catastrophic)."""
    return [
        # YOLO11s (φ_attn=0, φ_text=0, φ_dw=0)
        LOVODataPoint(
            ArchitectureFingerprint(0.0, 0.0, 0.0, 0.0, 0.25),
            "lora", 0.0710, model_name="YOLO11s", notes="canonical",
        ),
        LOVODataPoint(
            ArchitectureFingerprint(0.0, 0.0, 0.0, 0.0, 0.25),
            "dora", 0.0710, model_name="YOLO11s", notes="canonical",
        ),
        LOVODataPoint(
            ArchitectureFingerprint(0.0, 0.0, 0.0, 0.0, 0.25),
            "loha", 0.0359, model_name="YOLO11s", notes="canonical",
        ),
        LOVODataPoint(
            ArchitectureFingerprint(0.0, 0.0, 0.0, 0.0, 0.25),
            "lokr", 0.0605, model_name="YOLO11s", notes="canonical",
        ),
        LOVODataPoint(
            ArchitectureFingerprint(0.0, 0.0, 0.0, 0.0, 0.25),
            "ia3", 0.0552, model_name="YOLO11s", notes="canonical",
        ),
        LOVODataPoint(
            ArchitectureFingerprint(0.0, 0.0, 0.0, 0.0, 0.25),
            "hra", 0.0848, model_name="YOLO11s", notes="canonical",
        ),
        # YOLO12s (φ_attn≈0.45, φ_text=0, φ_dw=0)
        LOVODataPoint(
            ArchitectureFingerprint(0.45, 0.0, 0.0, 0.0, 0.333),
            "lora", 0.0645, model_name="YOLO12s", notes="canonical",
        ),
        LOVODataPoint(
            ArchitectureFingerprint(0.45, 0.0, 0.0, 0.0, 0.333),
            "loha", 0.0560, model_name="YOLO12s", notes="canonical",
        ),
        LOVODataPoint(
            ArchitectureFingerprint(0.45, 0.0, 0.0, 0.0, 0.333),
            "ia3", 0.0548, model_name="YOLO12s", notes="canonical",
        ),
        LOVODataPoint(
            ArchitectureFingerprint(0.45, 0.0, 0.0, 0.0, 0.333),
            "hra", 0.0791, model_name="YOLO12s", notes="canonical",
        ),
    ]


def _make_paper_history_full() -> List[LOVODataPoint]:
    """Return the full 12-point matrix including catastrophic cases (Fig. 4)."""
    points = _make_paper_history_10()
    points.extend([
        # Catastrophic: RT-DETR-like + LoRA (φ_attn≈0.85)
        LOVODataPoint(
            ArchitectureFingerprint(0.85, 0.0, 0.0, 0.0, 0.25),
            "lora", -0.600, model_name="RT-DETR", notes="catastrophic",
        ),
        # Catastrophic: YOLO12s + DoRA no-rs (φ_attn≈0.45)
        LOVODataPoint(
            ArchitectureFingerprint(0.45, 0.0, 0.0, 0.0, 0.333),
            "dora", -0.055, model_name="YOLO12s", notes="catastrophic",
        ),
    ])
    return points


# =============================================================================
# CLI Commands
# =============================================================================

def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run LOVO benchmark against paper claims and print a structured report."""
    # Canonical 10 points (no catastrophic)
    points_10 = _make_paper_history_10()
    collector_10 = LOVODataCollector(points_10)
    validator_10 = LOVOValidator(threshold=-0.05)
    result_10 = validator_10.cross_validate(collector_10.data_points)

    # Full 12-point matrix (with catastrophic)
    points_full = _make_paper_history_full()
    collector_full = LOVODataCollector(points_full)
    validator_full = LOVOValidator(threshold=-0.05)
    result_full = validator_full.cross_validate(collector_full.data_points)

    cat_metrics = validator_full.evaluate_catastrophe_detection(collector_full)
    decision_metrics = validator_full.evaluate_decision_boundary(collector_full)

    report: Dict[str, Any] = {
        "paper_claims": {
            "accuracy": 0.867,
            "recall": 0.944,
            "f1": 0.850,
            "r2_fitted_matrix": 0.762,
            "r2_canonical_10": 0.870,
        },
        "canonical_10_points": {
            "n_samples": result_10.n_samples,
            "n_variants": result_10.n_variants,
            "lovo_r2": result_10.lovo_r2,
            "lovo_mse": result_10.lovo_mse,
            "lovo_mae": result_10.lovo_mae,
            "lovo_rmse": result_10.lovo_rmse,
            "coefficients": result_10.coefficients,
        },
        "full_matrix_12_points": {
            "n_samples": result_full.n_samples,
            "n_variants": result_full.n_variants,
            "lovo_r2": result_full.lovo_r2,
            "lovo_mse": result_full.lovo_mse,
            "lovo_mae": result_full.lovo_mae,
            "lovo_rmse": result_full.lovo_rmse,
            "coefficients": result_full.coefficients,
        },
        "catastrophe_detection": cat_metrics,
        "decision_boundary": decision_metrics,
        "threshold": validator_full.threshold,
    }

    if args.report:
        out_path = Path(args.report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[LOVO] Benchmark report saved to {out_path}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0


def cmd_collect(args: argparse.Namespace) -> int:
    """Collect LOVO data points and persist to JSON."""
    collector = LOVODataCollector()

    if args.from_paper:
        points = _make_paper_history_full() if args.include_catastrophic else _make_paper_history_10()
        for p in points:
            collector.add(p)
        print(f"[LOVO] Collected {len(collector)} data points from paper benchmark.")
    elif args.from_dir:
        # Future: scan runs/ directory for training results.json
        print(f"[LOVO] Scanning {args.from_dir} for training artifacts … (not yet implemented)")
        return 1
    else:
        print("[LOVO] No data source specified. Use --from-paper or --from-dir.")
        return 1

    if args.output:
        out_path = Path(args.output)
        collector.save(out_path)
        print(f"[LOVO] Saved to {out_path}")
    else:
        print("[LOVO] No --output specified; skipping save.")

    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate LOVO data and generate a structured report."""
    if not args.input:
        print("[LOVO] Error: --input required for validate command.")
        return 1

    input_path = Path(args.input)
    collector = LOVODataCollector.load(input_path)
    validator = LOVOValidator(threshold=args.threshold)

    result = validator.cross_validate(collector.data_points)
    cat_metrics = validator.evaluate_catastrophe_detection(collector)
    decision_metrics = validator.evaluate_decision_boundary(collector)

    report: Dict[str, Any] = {
        "lovo": result.to_dict(),
        "catastrophe_detection": cat_metrics,
        "decision_boundary": decision_metrics,
        "summary": {
            "n_samples": result.n_samples,
            "n_variants": result.n_variants,
            "lovo_r2": result.lovo_r2,
            "lovo_rmse": result.lovo_rmse,
            "lovo_mae": result.lovo_mae,
            "catastrophe_recall": cat_metrics["recall"],
            "catastrophe_precision": cat_metrics["precision"],
            "catastrophe_f1": cat_metrics["f1"],
            "decision_accuracy": decision_metrics["accuracy"],
        },
    }

    if args.report:
        out_path = Path(args.report)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"[LOVO] Validation report saved to {out_path}")
    else:
        print(json.dumps(report, indent=2, ensure_ascii=False))

    return 0


def cmd_fit(args: argparse.Namespace) -> int:
    """Fit PEFTPlanner coefficients from data and print/save them."""
    if args.from_paper:
        points = _make_paper_history_full() if args.include_catastrophic else _make_paper_history_10()
        collector = LOVODataCollector(points)
    elif args.input:
        collector = LOVODataCollector.load(args.input)
    else:
        print("[LOVO] Error: --from-paper or --input required for fit command.")
        return 1

    planner = PEFTPlanner()
    planner.fit(collector.to_history())

    print(f"[LOVO] Fitted coefficients (β₀, β₁, β₂, β₃, β₄): {planner._coeffs}")
    print(f"[LOVO] Number of data points: {len(collector)}")

    if args.coefficients:
        out_path = Path(args.coefficients)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "coefficients": planner._coeffs,
                    "n_samples": len(collector),
                    "default_coeffs": list(PEFTPlanner.DEFAULT_COEFFS),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[LOVO] Coefficients saved to {out_path}")

    return 0


# =============================================================================
# Argument Parser
# =============================================================================

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="collect_lovo_data",
        description="LOVO Data Collection & Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ------------------------------------------------------------------
    # collect
    # ------------------------------------------------------------------
    collect_p = subparsers.add_parser("collect", help="Collect LOVO data points")
    collect_p.add_argument(
        "--from-paper", action="store_true",
        help="Use the paper's canonical benchmark data",
    )
    collect_p.add_argument(
        "--include-catastrophic", action="store_true",
        help="Include catastrophic data points (default: 10 canonical only)",
    )
    collect_p.add_argument(
        "--from-dir", type=str, metavar="DIR",
        help="Scan a directory for training artifacts (future feature)",
    )
    collect_p.add_argument(
        "--output", "-o", type=str, metavar="PATH",
        help="Output JSON file path",
    )
    collect_p.set_defaults(func=cmd_collect)

    # ------------------------------------------------------------------
    # validate
    # ------------------------------------------------------------------
    validate_p = subparsers.add_parser("validate", help="Validate LOVO data")
    validate_p.add_argument(
        "--input", "-i", type=str, required=True, metavar="PATH",
        help="Input JSON file with LOVO data points",
    )
    validate_p.add_argument(
        "--threshold", type=float, default=-0.05,
        help="Catastrophe detection threshold (default: -0.05)",
    )
    validate_p.add_argument(
        "--report", "-r", type=str, metavar="PATH",
        help="Report output JSON file",
    )
    validate_p.set_defaults(func=cmd_validate)

    # ------------------------------------------------------------------
    # benchmark
    # ------------------------------------------------------------------
    benchmark_p = subparsers.add_parser("benchmark", help="Run paper benchmark")
    benchmark_p.add_argument(
        "--report", "-r", type=str, metavar="PATH",
        help="Report output JSON file",
    )
    benchmark_p.set_defaults(func=cmd_benchmark)

    # ------------------------------------------------------------------
    # fit
    # ------------------------------------------------------------------
    fit_p = subparsers.add_parser("fit", help="Fit regression coefficients")
    fit_p.add_argument(
        "--from-paper", action="store_true",
        help="Use the paper's canonical data for fitting",
    )
    fit_p.add_argument(
        "--include-catastrophic", action="store_true",
        help="Include catastrophic data points when --from-paper is set",
    )
    fit_p.add_argument(
        "--input", "-i", type=str, metavar="PATH",
        help="Input JSON file with LOVO data points",
    )
    fit_p.add_argument(
        "--coefficients", "-c", type=str, metavar="PATH",
        help="Output JSON file for fitted coefficients",
    )
    fit_p.set_defaults(func=cmd_fit)

    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
