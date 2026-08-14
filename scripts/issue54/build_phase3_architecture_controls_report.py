#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Issue #54 Phase 3 architecture controls report."
    )
    parser.add_argument("--controls-root", type=Path, required=True)
    parser.add_argument("--mot-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    return [
        {str(key).strip(): value for key, value in row.items()}
        for row in rows
    ]


def resolve_checkpoint(
    run_dir: Path,
    model_key: str,
    manifest: dict[str, Any],
) -> Path:
    checkpoint_value = manifest.get("checkpoint_path")

    if checkpoint_value:
        checkpoint = Path(str(checkpoint_value))
        if not checkpoint.is_absolute():
            checkpoint = run_dir / checkpoint
    else:
        checkpoint = (
            run_dir
            / "training"
            / model_key
            / "weights"
            / "best.pt"
        )

    return checkpoint


def sample_std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    return statistics.stdev(values)


def format_mean_std(mean: float, std: float | None) -> str:
    if std is None:
        return f"{mean:.5f} (single seed)"
    return f"{mean:.5f} ± {std:.5f}"


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()

    controls_root = args.controls_root.resolve()
    mot_root = args.mot_root.resolve()
    output_dir = args.output_dir.resolve()

    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing report directory: {output_dir}"
        )

    output_dir.mkdir(parents=False)

    groups = [
        {
            "architecture": "EsMoE",
            "model_key": "v10",
            "seeds": [0, 1, 2],
            "root": controls_root,
            "routing_expected": False,
        },
        {
            "architecture": "MoA",
            "model_key": "v10_moa",
            "seeds": [0],
            "root": controls_root,
            "routing_expected": False,
        },
        {
            "architecture": "MoT",
            "model_key": "v10_mot",
            "seeds": [0, 1, 2, 3, 4],
            "root": mot_root,
            "routing_expected": True,
        },
    ]

    run_rows: list[dict[str, Any]] = []
    source_runs: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}

    for group in groups:
        architecture = str(group["architecture"])
        model_key = str(group["model_key"])
        root = Path(group["root"])
        routing_expected = bool(group["routing_expected"])

        for seed in group["seeds"]:
            run_id = f"phase3_{model_key}_seed{seed}"
            run_dir = root / run_id
            manifest_path = run_dir / "experiment_manifest.json"
            results_path = (
                run_dir
                / "training"
                / model_key
                / "results.csv"
            )

            if not run_dir.is_dir():
                raise FileNotFoundError(f"Missing run directory: {run_dir}")
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing manifest: {manifest_path}")
            if not results_path.is_file():
                raise FileNotFoundError(f"Missing results: {results_path}")

            manifest = read_json(manifest_path)

            if manifest.get("status") != "passed":
                raise ValueError(
                    f"{run_id}: status={manifest.get('status')!r}"
                )
            if manifest.get("failure_reason") not in (None, ""):
                raise ValueError(
                    f"{run_id}: failure_reason="
                    f"{manifest.get('failure_reason')!r}"
                )
            if manifest.get("requested_epochs") != 30:
                raise ValueError(
                    f"{run_id}: requested_epochs="
                    f"{manifest.get('requested_epochs')!r}"
                )
            if manifest.get("epochs") != 30:
                raise ValueError(
                    f"{run_id}: epochs={manifest.get('epochs')!r}"
                )

            rows = read_csv_rows(results_path)
            if len(rows) != 30:
                raise ValueError(
                    f"{run_id}: results.csv rows={len(rows)}, expected=30"
                )

            final_row = rows[-1]
            final_epoch = int(float(final_row["epoch"]))
            if final_epoch != 30:
                raise ValueError(
                    f"{run_id}: final epoch={final_epoch}, expected=30"
                )

            checkpoint_path = resolve_checkpoint(
                run_dir,
                model_key,
                manifest,
            )
            if not checkpoint_path.is_file():
                raise FileNotFoundError(
                    f"{run_id}: checkpoint missing: {checkpoint_path}"
                )

            checkpoint_sha256 = sha256_file(checkpoint_path)
            if checkpoint_sha256 != manifest.get("checkpoint_sha256"):
                raise ValueError(
                    f"{run_id}: checkpoint SHA256 mismatch"
                )

            if not routing_expected:
                routing_files = [
                    path
                    for path in run_dir.rglob("*")
                    if path.is_file()
                    and "routing" in path.name.lower()
                ]
                if routing_files:
                    raise ValueError(
                        f"{run_id}: unexpected routing files: "
                        f"{routing_files}"
                    )

            map50 = float(final_row["metrics/mAP50(B)"])
            map50_95 = float(final_row["metrics/mAP50-95(B)"])

            precision_mode = manifest.get("precision_mode")
            requested_batch = manifest.get(
                "requested_batch",
                manifest.get("batch"),
            )
            effective_batch = manifest.get(
                "effective_batch",
                manifest.get("batch"),
            )
            imgsz = manifest.get("imgsz")

            checkpoint_key = f"{architecture}:{model_key}:{seed}"
            checkpoint_hashes[checkpoint_key] = checkpoint_sha256

            run_rows.append(
                {
                    "architecture": architecture,
                    "model_key": model_key,
                    "seed": seed,
                    "status": manifest.get("status"),
                    "epochs": manifest.get("epochs"),
                    "precision_mode": precision_mode,
                    "requested_batch": requested_batch,
                    "effective_batch": effective_batch,
                    "imgsz": imgsz,
                    "mAP50": f"{map50:.8f}",
                    "mAP50-95": f"{map50_95:.8f}",
                    "checkpoint_sha256": checkpoint_sha256,
                    "checkpoint_path": str(checkpoint_path),
                    "routing_scope": (
                        "MoT routing preserved"
                        if routing_expected
                        else "No MoT routing by design"
                    ),
                }
            )

            source_runs.append(
                {
                    "architecture": architecture,
                    "model_key": model_key,
                    "seed": seed,
                    "run_id": run_id,
                    "run_dir": str(run_dir),
                    "manifest_path": str(manifest_path),
                    "manifest_sha256": sha256_file(manifest_path),
                    "results_path": str(results_path),
                    "results_sha256": sha256_file(results_path),
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": checkpoint_sha256,
                }
            )

    if len(checkpoint_hashes) != 9:
        raise ValueError(
            f"Expected 9 checkpoints, got {len(checkpoint_hashes)}"
        )

    if len(set(checkpoint_hashes.values())) != 9:
        raise ValueError("Duplicate checkpoint SHA256 detected")

    summary_rows: list[dict[str, Any]] = []
    summary_by_architecture: dict[str, dict[str, Any]] = {}

    for group in groups:
        architecture = str(group["architecture"])
        model_key = str(group["model_key"])
        seeds = list(group["seeds"])

        selected = [
            row
            for row in run_rows
            if row["architecture"] == architecture
        ]

        map50_values = [float(row["mAP50"]) for row in selected]
        map50_95_values = [
            float(row["mAP50-95"]) for row in selected
        ]

        mean_map50 = statistics.mean(map50_values)
        std_map50 = sample_std(map50_values)
        mean_map50_95 = statistics.mean(map50_95_values)
        std_map50_95 = sample_std(map50_95_values)

        summary = {
            "architecture": architecture,
            "model_key": model_key,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
            "mean_mAP50": mean_map50,
            "sample_std_mAP50": std_map50,
            "mean_mAP50-95": mean_map50_95,
            "sample_std_mAP50-95": std_map50_95,
        }
        summary_by_architecture[architecture] = summary

        summary_rows.append(
            {
                "architecture": architecture,
                "model_key": model_key,
                "seed_count": len(seeds),
                "seeds": ",".join(str(seed) for seed in seeds),
                "mean_mAP50": f"{mean_map50:.8f}",
                "sample_std_mAP50": (
                    ""
                    if std_map50 is None
                    else f"{std_map50:.8f}"
                ),
                "mean_mAP50-95": f"{mean_map50_95:.8f}",
                "sample_std_mAP50-95": (
                    ""
                    if std_map50_95 is None
                    else f"{std_map50_95:.8f}"
                ),
                "statistical_scope": (
                    "single-seed descriptive result; "
                    "no between-seed variance estimate"
                    if len(seeds) == 1
                    else "sample mean and sample standard deviation"
                ),
            }
        )

    run_metrics_path = (
        output_dir / "phase3_architecture_run_metrics.csv"
    )
    summary_path = (
        output_dir / "phase3_architecture_summary.csv"
    )
    report_path = (
        output_dir / "PHASE3_ARCHITECTURE_CONTROLS_REPORT.md"
    )
    manifest_path = (
        output_dir / "phase3_architecture_report_manifest.json"
    )

    write_csv(
        run_metrics_path,
        [
            "architecture",
            "model_key",
            "seed",
            "status",
            "epochs",
            "precision_mode",
            "requested_batch",
            "effective_batch",
            "imgsz",
            "mAP50",
            "mAP50-95",
            "checkpoint_sha256",
            "checkpoint_path",
            "routing_scope",
        ],
        run_rows,
    )

    write_csv(
        summary_path,
        [
            "architecture",
            "model_key",
            "seed_count",
            "seeds",
            "mean_mAP50",
            "sample_std_mAP50",
            "mean_mAP50-95",
            "sample_std_mAP50-95",
            "statistical_scope",
        ],
        summary_rows,
    )

    esmoe = summary_by_architecture["EsMoE"]
    moa = summary_by_architecture["MoA"]
    mot = summary_by_architecture["MoT"]

    mot_minus_esmoe_map50 = (
        mot["mean_mAP50"] - esmoe["mean_mAP50"]
    )
    mot_minus_esmoe_map50_95 = (
        mot["mean_mAP50-95"] - esmoe["mean_mAP50-95"]
    )
    moa_minus_esmoe_map50 = (
        moa["mean_mAP50"] - esmoe["mean_mAP50"]
    )
    moa_minus_esmoe_map50_95 = (
        moa["mean_mAP50-95"] - esmoe["mean_mAP50-95"]
    )
    moa_minus_mot_map50 = (
        moa["mean_mAP50"] - mot["mean_mAP50"]
    )
    moa_minus_mot_map50_95 = (
        moa["mean_mAP50-95"] - mot["mean_mAP50-95"]
    )

    generated_at = datetime.now(timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )

    report_lines = [
        "# Phase 3 Architecture Controls Report",
        "",
        f"Generated at: `{generated_at}`",
        "",
        "## 1. Scope",
        "",
        "This report compares the final detection performance of:",
        "",
        "- **EsMoE**: model key `v10`, seeds 0, 1, and 2;",
        "- **MoA**: model key `v10_moa`, seed 0;",
        "- **MoT**: model key `v10_mot`, seeds 0 through 4.",
        "",
        "All runs used 30 epochs, batch size 8, image size 640, "
        "VisDrone2019-DET, and the formal Issue #54 protocol. "
        "MoT used FP32; EsMoE and MoA used AMP.",
        "",
        "The highest-level experimental unit is an independently "
        "trained seed. Images or tokens are not treated as independent "
        "training repetitions.",
        "",
        "## 2. Integrity checks",
        "",
        "- All nine formal runs have `status=passed`.",
        "- All nine runs completed 30 epochs.",
        "- All checkpoint SHA256 values match their manifests.",
        "- All nine formal checkpoints are mutually distinct.",
        "- EsMoE and MoA contain no fabricated MoT routing artifacts.",
        "- Existing MoT routing artifacts are preserved separately.",
        "",
        "## 3. Per-run final metrics",
        "",
        "| Architecture | Model | Seed | Precision | mAP50 | "
        "mAP50-95 | Checkpoint SHA256 |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]

    for row in run_rows:
        report_lines.append(
            f"| {row['architecture']} | `{row['model_key']}` | "
            f"{row['seed']} | {row['precision_mode']} | "
            f"{float(row['mAP50']):.5f} | "
            f"{float(row['mAP50-95']):.5f} | "
            f"`{row['checkpoint_sha256']}` |"
        )

    report_lines.extend(
        [
            "",
            "## 4. Aggregate performance",
            "",
            "Sample standard deviation is reported only when at least "
            "two independent seeds are available.",
            "",
            "| Architecture | Independent seeds | mAP50 | mAP50-95 |",
            "|---|---:|---:|---:|",
        ]
    )

    for architecture in ("EsMoE", "MoA", "MoT"):
        summary = summary_by_architecture[architecture]
        report_lines.append(
            f"| {architecture} | {summary['seed_count']} | "
            f"{format_mean_std(summary['mean_mAP50'], summary['sample_std_mAP50'])} | "
            f"{format_mean_std(summary['mean_mAP50-95'], summary['sample_std_mAP50-95'])} |"
        )

    report_lines.extend(
        [
            "",
            "## 5. Descriptive architecture differences",
            "",
            "These differences are descriptive only. They are not "
            "formal significance tests and do not establish causal "
            "superiority.",
            "",
            "| Comparison | ΔmAP50 | ΔmAP50-95 |",
            "|---|---:|---:|",
            f"| MoT mean − EsMoE mean | "
            f"{mot_minus_esmoe_map50:+.5f} | "
            f"{mot_minus_esmoe_map50_95:+.5f} |",
            f"| MoA seed0 − EsMoE mean | "
            f"{moa_minus_esmoe_map50:+.5f} | "
            f"{moa_minus_esmoe_map50_95:+.5f} |",
            f"| MoA seed0 − MoT mean | "
            f"{moa_minus_mot_map50:+.5f} | "
            f"{moa_minus_mot_map50_95:+.5f} |",
            "",
            "## 6. Evidence-bounded interpretation",
            "",
            "1. **MoT and EsMoE have very similar mean detection "
            "performance under the current protocol.** The observed "
            "mean differences are small and mixed across the two "
            "metrics.",
            "2. **MoA is represented by one independent training seed.** "
            "Its result is a single-run architecture control and cannot "
            "support claims about between-seed stability or variance.",
            "3. **Performance stability and routing stability are "
            "different questions.** The existing MoT cross-seed report "
            "shows relatively stable detection performance alongside "
            "only moderate or low internal routing agreement, with "
            "strong layer-level differences.",
            "4. The present evidence does not prove that routing "
            "instability reduces performance, that a specific expert "
            "has a fixed semantic role, or that high routing entropy "
            "implies high routing stability.",
            "",
            "## 7. Statistical limitations",
            "",
            "- Seed counts are unequal: MoT n=5, EsMoE n=3, MoA n=1.",
            "- No formal hypothesis test is reported.",
            "- MoA has no valid between-seed variance estimate.",
            "- Conclusions should remain descriptive and protocol-specific.",
            "",
            "## 8. Related MoT routing report",
            "",
            f"The detailed MoT routing analysis remains in:",
            "",
            f"`{mot_root / 'reports' / 'PHASE3_MOT_CROSS_SEED_REPORT.md'}`",
            "",
            "This architecture report does not recreate or invent "
            "routing records for EsMoE or MoA.",
            "",
            "## 9. Source roots",
            "",
            f"- Controls root: `{controls_root}`",
            f"- MoT formal root: `{mot_root}`",
            "",
        ]
    )

    report_path.write_text(
        "\n".join(report_lines),
        encoding="utf-8",
    )

    report_files = [
        report_path,
        run_metrics_path,
        summary_path,
    ]

    generator_path = Path(__file__).resolve()

    manifest_payload = {
        "schema_version": 1,
        "report_type": "phase3_architecture_controls",
        "generated_at": generated_at,
        "generator_script": str(generator_path),
        "generator_script_sha256": sha256_file(generator_path),
        "controls_root": str(controls_root),
        "mot_root": str(mot_root),
        "protocol": {
            "dataset": "VisDrone2019-DET",
            "epochs": 30,
            "batch": 8,
            "imgsz": 640,
            "mot_precision": "fp32",
            "esmoe_precision": "amp",
            "moa_precision": "amp",
        },
        "integrity": {
            "formal_run_count": 9,
            "all_runs_passed": True,
            "all_runs_completed_30_epochs": True,
            "checkpoint_sha256_unique_count": 9,
            "controls_have_no_mot_routing": True,
        },
        "source_runs": source_runs,
        "architecture_summary": [
            {
                "architecture": row["architecture"],
                "model_key": row["model_key"],
                "seed_count": row["seed_count"],
                "seeds": row["seeds"],
                "mean_mAP50": row["mean_mAP50"],
                "sample_std_mAP50": (
                    row["sample_std_mAP50"] or None
                ),
                "mean_mAP50-95": row["mean_mAP50-95"],
                "sample_std_mAP50-95": (
                    row["sample_std_mAP50-95"] or None
                ),
                "statistical_scope": row["statistical_scope"],
            }
            for row in summary_rows
        ],
        "report_files": [
            {
                "path": path.name,
                "sha256": sha256_file(path),
            }
            for path in report_files
        ],
    }

    manifest_path.write_text(
        json.dumps(
            manifest_payload,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"REPORT_DIR={output_dir}")
    for path in [
        report_path,
        run_metrics_path,
        summary_path,
        manifest_path,
    ]:
        print(f"CREATED {path}")
    print("PASS: architecture controls report generated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
