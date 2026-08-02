#!/usr/bin/env python3
"""Build a read-only five-seed Phase 3 MoT report from completed formal evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.issue54.build_experiment_registry import validate_registry  # noqa: E402
from scripts.issue54.schema import (  # noqa: E402
    ANALYSIS_SCHEMA_VERSION,
    SchemaValidationError,
    canonical_payload_sha256,
    load_json,
    validate_experiment_manifest,
)


MODEL = "v10_mot"
SEEDS = (0, 1, 2, 3, 4)
REPORT_NAMES = (
    "PHASE3_MOT_CROSS_SEED_REPORT.md",
    "phase3_mot_seed_metrics.csv",
    "phase3_mot_layer_stability.csv",
    "phase3_mot_pairwise_agreement.csv",
    "phase3_mot_expert_utilization.csv",
    "phase3_mot_report_manifest.json",
)


def sha256_file(path: Path) -> str:
    """Return a streaming SHA256 for an evidence artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one report table without changing any formal input artifact."""
    fields = sorted({field for row in rows for field in row})
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write one new deterministic report JSON artifact."""
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def mean(values: list[float]) -> float | None:
    """Return a mean only where an input population exists."""
    return statistics.fmean(values) if values else None


def sample_stats(values: list[float]) -> dict[str, float | None]:
    """Summarize formal seed values with sample rather than population spread."""
    return {
        "mean": mean(values),
        "sample_standard_deviation": statistics.stdev(values) if len(values) > 1 else None,
        "minimum": min(values) if values else None,
        "maximum": max(values) if values else None,
    }


def results_metrics(path: Path) -> dict[str, float]:
    """Read the final row of a completed Ultralytics results CSV."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"results.csv has no data rows: {path}")
    row = {str(key).strip(): value for key, value in rows[-1].items()}
    try:
        metrics = {
            "mAP50": float(row["metrics/mAP50(B)"]),
            "mAP50-95": float(row["metrics/mAP50-95(B)"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"results.csv lacks final mAP metrics: {path}") from error
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"results.csv contains non-finite final mAP metrics: {path}")
    return metrics


def formal_paths(root: Path, seed: int) -> tuple[Path, Path]:
    """Return explicit manifest and results paths for one required MoT seed."""
    run = root / f"phase3_{MODEL}_seed{seed}"
    return run / "experiment_manifest.json", run / "training" / MODEL / "results.csv"


def resolve_checkpoint(run_root: Path, checkpoint_path: str | Path) -> Path:
    """Resolve an absolute checkpoint directly or a manifest path under its run directory."""
    checkpoint = Path(checkpoint_path)
    return checkpoint if checkpoint.is_absolute() else run_root / checkpoint


def validate_analysis(payload: dict[str, Any], registry_sha256: str) -> dict[str, Any]:
    """Validate the frozen analysis identity and checksum before reporting it."""
    if payload.get("schema_version") != ANALYSIS_SCHEMA_VERSION:
        raise SchemaValidationError("unsupported Phase 3 routing analysis schema")
    if payload.get("registry_sha256") != registry_sha256:
        raise SchemaValidationError("routing analysis does not reference the formal registry")
    expected = canonical_payload_sha256(payload, exclude=("analysis_sha256",))
    if payload.get("analysis_sha256") != expected:
        raise SchemaValidationError("analysis_sha256 does not match the canonical analysis")
    if payload.get("formal_seed_counts_from_registry", {}).get(MODEL) != len(SEEDS):
        raise SchemaValidationError("routing analysis does not contain five formal MoT seeds")
    if payload.get("analyzed_seed_counts", {}).get(MODEL) != len(SEEDS):
        raise SchemaValidationError("routing analysis does not analyze five MoT seeds")
    if payload.get("validation_issues"):
        raise SchemaValidationError("routing analysis contains unresolved validation issues")
    return payload


def validate_unique_checkpoints(checkpoint_hashes: list[str]) -> None:
    """Reject checkpoint reuse that would inflate the independent-seed count."""
    if len(checkpoint_hashes) != len(SEEDS) or len(set(checkpoint_hashes)) != len(SEEDS):
        raise SchemaValidationError("formal MoT checkpoint SHA256 values are not unique")


def validate_inputs(root: Path) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], list[Path]]:
    """Load all required formal inputs and reject partial or non-formal evidence."""
    registry_path = root / "phase3_formal_registry.json"
    analysis_path = root / "phase3_cross_seed_routing.json"
    registry = validate_registry(load_json(registry_path))
    analysis = validate_analysis(load_json(analysis_path), registry["registry_sha256"])
    manifests, inputs = [], [registry_path, analysis_path]
    actual_checkpoint_hashes = []
    for seed in SEEDS:
        manifest_path, results_path = formal_paths(root, seed)
        manifest = validate_experiment_manifest(load_json(manifest_path))
        if manifest["model_variant"] != MODEL or manifest["seed"] != seed or manifest["status"] != "passed":
            raise SchemaValidationError(f"seed {seed} is not a passed formal {MODEL} manifest")
        if not results_path.is_file():
            raise FileNotFoundError(f"missing results.csv for seed {seed}: {results_path}")
        results_metrics(results_path)
        run_root = manifest_path.parent
        checkpoint = resolve_checkpoint(run_root, manifest["checkpoint_path"])
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing checkpoint for seed {seed}: {checkpoint}")
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != manifest["checkpoint_sha256"]:
            raise SchemaValidationError(f"checkpoint SHA256 mismatch for seed {seed}")
        actual_checkpoint_hashes.append(checkpoint_hash)
        manifests.append(manifest)
        inputs.extend((manifest_path, results_path))
    validate_unique_checkpoints(actual_checkpoint_hashes)
    expected_ids = {manifest["experiment_id"] for manifest in manifests}
    registry_by_id = {item["experiment_id"]: item for item in registry["experiments"]}
    if set(registry_by_id) != expected_ids:
        raise SchemaValidationError("formal registry does not contain exactly the five required MoT experiments")
    for manifest in manifests:
        registered = registry_by_id[manifest["experiment_id"]]
        if registered["manifest_sha256"] != manifest["manifest_sha256"]:
            raise SchemaValidationError(f"registry manifest mismatch for {manifest['experiment_id']}")
    return registry, analysis, manifests, inputs


def build_tables(
    root: Path, analysis: dict[str, Any], manifests: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], ...]:
    """Derive report tables from final metrics and cross-seed routing analysis."""
    seed_rows = []
    for manifest in manifests:
        _, results_path = formal_paths(root, manifest["seed"])
        seed_rows.append(
            {
                "seed": manifest["seed"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                **results_metrics(results_path),
            }
        )
    pairwise = list(analysis.get("pairwise_seed_comparisons", []))
    pairwise_rows = [
        {
            key: row.get(key)
            for key in (
                "seed_a",
                "seed_b",
                "layer_name",
                "layer_index",
                "dominant_expert_agreement",
                "token_top1_agreement",
                "jensen_shannon_divergence",
                "route_entropy_a",
                "route_entropy_b",
                "normalized_route_entropy_a",
                "normalized_route_entropy_b",
            )
        }
        for row in pairwise
    ]
    entropy_by_layer: dict[tuple[Any, Any], list[dict[str, float | None]]] = {}
    for row in pairwise:
        entropy_by_layer.setdefault((row.get("layer_name"), row.get("layer_index")), []).append(
            {
                "route_entropy": mean(
                    [value for value in (row.get("route_entropy_a"), row.get("route_entropy_b")) if value is not None]
                ),
                "normalized_route_entropy": mean(
                    [
                        value
                        for value in (row.get("normalized_route_entropy_a"), row.get("normalized_route_entropy_b"))
                        if value is not None
                    ]
                ),
            }
        )
    layers = list(analysis.get("per_layer_summary", []))
    layer_rows = sorted(
        [
            {
                "layer_name": row.get("layer_name"),
                "layer_index": row.get("layer_index"),
                "mean_dominant_expert_agreement": row.get("mean_dominant_expert_agreement"),
                "mean_token_top1_agreement": row.get("mean_token_top1_agreement"),
                "mean_jensen_shannon_divergence": row.get("mean_jensen_shannon_divergence"),
                "mean_route_entropy": mean(
                    [
                        item["route_entropy"]
                        for item in entropy_by_layer.get((row.get("layer_name"), row.get("layer_index")), [])
                        if item["route_entropy"] is not None
                    ]
                ),
                "mean_normalized_route_entropy": mean(
                    [
                        item["normalized_route_entropy"]
                        for item in entropy_by_layer.get((row.get("layer_name"), row.get("layer_index")), [])
                        if item["normalized_route_entropy"] is not None
                    ]
                ),
            }
            for row in layers
        ],
        key=lambda row: (row["mean_dominant_expert_agreement"] is None, row["mean_dominant_expert_agreement"]),
        reverse=True,
    )
    for rank, row in enumerate(layer_rows, start=1):
        row["stability_rank"] = rank
    if len(layer_rows) != 6:
        raise SchemaValidationError(f"formal MoT analysis must contain exactly six layers, got {len(layer_rows)}")
    utilization = list(analysis.get("expert_utilization_between_seed", []))
    return seed_rows, layer_rows, pairwise_rows, utilization


def report_markdown(seed_rows: list[dict[str, Any]], layers: list[dict[str, Any]], analysis: dict[str, Any]) -> str:
    """Render cautious, evidence-bounded Markdown conclusions."""
    metrics = {key: sample_stats([row[key] for row in seed_rows]) for key in ("mAP50", "mAP50-95")}
    global_summary = analysis.get("global_summary", [{}])[0]
    determinism = analysis.get("checkpoint_determinism", [])
    passed_determinism = sum(item.get("status") == "passed" for item in determinism)
    lines = [
        "# Phase 3 MoT Cross-Seed Report",
        "",
        "## Performance",
        "",
        "| Metric | Mean | Sample SD | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, values in metrics.items():
        lines.append(
            f"| {name} | {values['mean']:.6f} | {values['sample_standard_deviation']:.6f} | "
            f"{values['minimum']:.6f} | {values['maximum']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Routing evidence",
            "",
            f"- Mean dominant-expert agreement: {global_summary.get('mean_dominant_expert_agreement', float('nan')):.6f}",
            f"- Mean token top-1 agreement: {global_summary.get('mean_token_top1_agreement', float('nan')):.6f}",
            f"- Checkpoint repeated-inference rows passing determinism: {passed_determinism}/{len(determinism)}.",
            "- Five checkpoint SHA256 values were verified distinct before report generation.",
            "",
            "## Supported conclusion",
            "",
            "Performance across seeds is relatively stable, but internal routing shows only moderate or lower agreement and clear layer-level differences.",
            "",
            "## Not supported by this evidence",
            "",
            "- Routing instability necessarily reduces detection performance.",
            "- A given expert has a fixed responsibility for a target type.",
            "- Occlusion or object size has been proven to cause routing changes.",
            "- Higher route entropy means routing is more stable.",
            "",
            "## Reproducibility limitation",
            "",
            "Deterministic CUDA warnings remain a reproducibility limitation: deterministic settings do not guarantee bitwise equivalence for every CUDA kernel or environment.",
            "",
            "## Layer stability ranking",
            "",
            "| Rank | Layer | Dominant agreement | Token top-1 agreement | Route entropy | Normalized route entropy |",
            "|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in layers:
        lines.append(
            f"| {row['stability_rank']} | {row['layer_name']} | {row['mean_dominant_expert_agreement']:.6f} | "
            f"{row['mean_token_top1_agreement']:.6f} | {row['mean_route_entropy']:.6f} | "
            f"{row['mean_normalized_route_entropy']:.6f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    """Generate new report artifacts without touching formal evidence inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.formal_root
    if not root.is_dir():
        raise FileNotFoundError(f"formal root not found: {root}")
    reports = root / "reports"
    outputs = [reports / name for name in REPORT_NAMES]
    if any(path.exists() for path in outputs):
        raise FileExistsError("report output already exists; refusing to overwrite formal results")
    registry, analysis, manifests, inputs = validate_inputs(root)
    seed_rows, layer_rows, pairwise_rows, utilization_rows = build_tables(root, analysis, manifests)
    reports.mkdir(exist_ok=True)
    markdown = reports / REPORT_NAMES[0]
    with markdown.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(report_markdown(seed_rows, layer_rows, analysis))
    write_csv(reports / REPORT_NAMES[1], seed_rows)
    write_csv(reports / REPORT_NAMES[2], layer_rows)
    write_csv(reports / REPORT_NAMES[3], pairwise_rows)
    write_csv(reports / REPORT_NAMES[4], utilization_rows)
    manifest = {
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "model_variant": MODEL,
        "seeds": list(SEEDS),
        "registry_sha256": registry["registry_sha256"],
        "analysis_sha256": analysis.get("analysis_sha256"),
        "input_sha256": {path.relative_to(root).as_posix(): sha256_file(path) for path in inputs},
        "output_sha256": {path.name: sha256_file(path) for path in outputs[:-1]},
    }
    write_json(reports / REPORT_NAMES[5], manifest)
    print(f"[issue54-phase3-report] wrote {len(outputs)} files under {reports}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, OSError, SchemaValidationError, ValueError) as error:
        print(f"[issue54-phase3-report] error: {error}", file=sys.stderr)
        raise SystemExit(2)
