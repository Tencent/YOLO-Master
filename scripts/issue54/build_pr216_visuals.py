"""Build deterministic PR #216 figures from verified Phase 3 evidence."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_DIR = ROOT / "docs" / "issue54" / "phase3_architecture_controls"
ROUTING_DIR = ROOT / "docs" / "issue54" / "phase3_mot_routing"
OUTPUT_DIR = ROOT / "docs" / "issue54" / "pr_assets"
GLOBAL_SUMMARY_PATH = ROUTING_DIR / "phase3_mot_global_summary.json"
PUBLIC_CHECKSUM_NAME = "PUBLIC_SHA256SUMS"
SOURCE_CHECKSUM_NAME = "SOURCE_SHA256SUMS"

ARCHITECTURES = ("EsMoE", "MoA", "MoT")
SEED_COUNTS = {"EsMoE": 3, "MoA": 1, "MoT": 5}
LAYERS = (
    "model.14.m.0",
    "model.14.m.1",
    "model.20.m.0",
    "model.20.m.1",
    "model.23.m.0",
    "model.23.m.1",
)
EXPERTS = {"DeformableTransformer", "LocalConvTransformer", "WindowTransformer"}
PNG_METADATA = {"Software": "YOLO-Master Issue #54"}


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_sha256sums(checksum_path: Path) -> dict[str, str]:
    """Read a safe, non-empty SHA256SUMS-style file."""
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty checksum file: {checksum_path}")
    entries: dict[str, str] = {}
    for line in lines:
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        expected = expected.lower()
        relative = relative.lstrip("*")
        pure = PurePosixPath(relative)
        if (
            len(expected) != 64
            or any(character not in "0123456789abcdef" for character in expected)
            or pure.is_absolute()
            or ".." in pure.parts
            or relative in entries
        ):
            raise ValueError(f"unsafe or duplicate checksum entry: {relative}")
        entries[relative] = expected
    if not entries:
        raise ValueError(f"empty checksum file: {checksum_path}")
    return entries


def validate_sha256sums(directory: Path, checksum_name: str = "SHA256SUMS") -> dict[str, str]:
    """Fail unless every checksum entry matches a file in ``directory``."""
    checksum_path = directory / checksum_name
    entries = read_sha256sums(checksum_path)
    for relative, expected in entries.items():
        pure = PurePosixPath(relative)
        target = directory.joinpath(*pure.parts)
        if not target.is_file() or sha256_file(target) != expected:
            raise ValueError(f"SHA256 mismatch: {target}")
    return entries


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read a non-empty CSV into dictionaries."""
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows


def finite_float(value: str | int | float, *, field: str) -> float:
    """Parse a finite floating-point value."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} is not numeric: {value!r}") from error
    if not math.isfinite(parsed):
        raise ValueError(f"{field} is not finite: {value!r}")
    return parsed


def load_json_object(path: Path) -> dict[str, object]:
    """Load a JSON object and reject other top-level types."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_source_metadata() -> dict[str, object]:
    """Cross-check the omitted raw source identity across public provenance records."""
    provenance = load_json_object(ROUTING_DIR / "SOURCE_PROVENANCE.json")
    source_files = provenance.get("source_files")
    if not isinstance(source_files, list):
        raise ValueError("SOURCE_PROVENANCE.json has no source_files list")
    raw_entries = [row for row in source_files if row.get("relative_path") == "phase3_cross_seed_routing.json"]
    if len(raw_entries) != 1:
        raise ValueError("provenance must identify the raw cross-seed source exactly once")
    raw_entry = raw_entries[0]
    source_checksums = read_sha256sums(ROUTING_DIR / SOURCE_CHECKSUM_NAME)
    source_sha256 = str(raw_entry.get("sha256", "")).lower()
    if source_checksums.get("phase3_cross_seed_routing.json") != source_sha256:
        raise ValueError("raw source SHA256 differs between provenance and SOURCE_SHA256SUMS")
    formal_root = (ROUTING_DIR / "PHASE3_FORMAL_ROOT.txt").read_text(encoding="utf-8").strip()
    if not formal_root or provenance.get("formal_root") != formal_root:
        raise ValueError("formal root differs between provenance and PHASE3_FORMAL_ROOT.txt")
    return {
        "formal_root": formal_root,
        "source_file": "phase3_cross_seed_routing.json",
        "source_sha256": source_sha256,
        "source_size_bytes": int(raw_entry["size_bytes"]),
    }


def load_formal_registry() -> tuple[dict[str, object], list[dict[str, object]]]:
    """Load and validate the five-run formal MoT registry."""
    registry = load_json_object(ROUTING_DIR / "phase3_formal_registry.json")
    experiments = registry.get("experiments")
    if not isinstance(experiments, list) or not all(isinstance(row, dict) for row in experiments):
        raise ValueError("formal registry must contain experiment objects")
    if len(experiments) != 5 or registry.get("status_counts") != {"passed": 5}:
        raise ValueError("formal registry must contain exactly five passed experiments")
    expected_seeds = list(range(5))
    if sorted(int(row["seed"]) for row in experiments) != expected_seeds:
        raise ValueError("formal registry must contain seeds 0-4 exactly once")
    identity_fields = {
        "model_variant": "v10_mot",
        "status": "passed",
        "epochs": 30,
        "precision_mode": "fp32",
        "batch": 8,
        "imgsz": 640,
        "dataset": "VisDrone2019-DET",
        "dataset_version": "2019-DET",
        "split": "val-fixed32",
    }
    for row in experiments:
        for field, expected in identity_fields.items():
            if row.get(field) != expected:
                raise ValueError(f"unexpected formal registry identity: {field}={row.get(field)!r}")
    checkpoint_sha256s = {str(row.get("checkpoint_sha256", "")) for row in experiments}
    if len(checkpoint_sha256s) != 5 or any(len(digest) != 64 for digest in checkpoint_sha256s):
        raise ValueError("formal checkpoints must have five distinct SHA256 identities")
    return registry, experiments


def build_compact_summary(raw_path: Path) -> dict[str, object]:
    """Extract a compact formal identity and scalar summary from the private raw JSON."""
    metadata = load_source_metadata()
    if not raw_path.is_file():
        raise ValueError(f"raw cross-seed JSON does not exist: {raw_path}")
    if raw_path.stat().st_size != metadata["source_size_bytes"] or sha256_file(raw_path) != metadata["source_sha256"]:
        raise ValueError("selected raw JSON does not match the recorded formal source identity")
    analysis = load_json_object(raw_path)
    registry, experiments = load_formal_registry()
    if analysis.get("schema_version") != 1 or analysis.get("analysis_mode") != "formal_passed_only":
        raise ValueError("raw analysis is not the expected formal schema")
    if analysis.get("validation_issues") != []:
        raise ValueError("raw analysis contains validation issues")
    if analysis.get("registry_sha256") != registry.get("registry_sha256"):
        raise ValueError("raw analysis and formal registry identities differ")

    groups = analysis.get("analysis_groups")
    global_rows = analysis.get("global_summary")
    layer_rows = analysis.get("per_layer_summary")
    pairwise_rows = analysis.get("pairwise_seed_comparisons")
    utilization_rows = analysis.get("expert_utilization_by_seed")
    determinism_rows = analysis.get("checkpoint_determinism")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(groups[0], dict):
        raise ValueError("raw analysis must contain one formal analysis group")
    if not isinstance(global_rows, list) or len(global_rows) != 1 or not isinstance(global_rows[0], dict):
        raise ValueError("raw analysis must contain one formal global summary")
    collections = (layer_rows, pairwise_rows, utilization_rows, determinism_rows)
    if any(not isinstance(rows, list) for rows in collections):
        raise ValueError("raw analysis is missing a required formal row collection")

    group = groups[0]
    global_row = global_rows[0]
    seeds = sorted(int(seed) for seed in group.get("seeds", []))
    if seeds != list(range(5)) or int(group.get("seed_count", -1)) != len(seeds):
        raise ValueError("raw analysis group must contain five seeds 0-4")
    model = str(group.get("model_variant"))
    if model != "v10_mot" or global_row.get("model_variant") != model:
        raise ValueError("raw analysis is not the formal v10_mot group")
    dataset = str(group.get("dataset"))
    dataset_version = str(group.get("dataset_version"))
    split = str(group.get("split"))
    if any(
        global_row.get(field) != value
        for field, value in (("dataset", dataset), ("dataset_version", dataset_version), ("split", split))
    ):
        raise ValueError("global summary identity differs from the analysis group")

    image_count = int(analysis.get("image_count", -1))
    layer_names = {str(row["layer_name"]) for row in layer_rows}
    expert_names = {str(row["expert_name"]) for row in utilization_rows}
    seed_pairs = {(int(row["seed_a"]), int(row["seed_b"])) for row in pairwise_rows}
    expected_pairs = set(combinations(seeds, 2))
    if seed_pairs != expected_pairs:
        raise ValueError("raw pairwise rows do not contain the ten seed pairs exactly")
    pairwise_row_count = len(pairwise_rows)
    if pairwise_row_count != image_count * len(layer_names) * len(seed_pairs):
        raise ValueError("raw pairwise row count does not match images x layers x seed pairs")
    if (
        analysis.get("record_count") != pairwise_row_count
        or global_row.get("aligned_comparisons") != pairwise_row_count
    ):
        raise ValueError("raw global comparison counts are inconsistent")
    if int(global_row.get("distinct_seed_pairs", -1)) != len(seed_pairs):
        raise ValueError("raw global seed-pair count is inconsistent")
    if len(utilization_rows) != len(seeds) * len(layer_names) * len(expert_names):
        raise ValueError("raw per-seed utilization dimensions are inconsistent")

    determinism_passed = sum(
        row.get("status") == "passed"
        and isinstance(row.get("repeat_comparisons"), list)
        and bool(row["repeat_comparisons"])
        and all(repeat.get("passed") is True for repeat in row["repeat_comparisons"])
        for row in determinism_rows
    )
    if determinism_passed != len(determinism_rows) or len(determinism_rows) != len(seeds) * image_count * len(
        layer_names
    ):
        raise ValueError("raw determinism rows are incomplete or contain failures")

    protocol_values: dict[str, object] = {}
    for field in ("epochs", "precision_mode", "batch", "imgsz", "dataset", "dataset_version", "split"):
        values = {row[field] for row in experiments}
        if len(values) != 1:
            raise ValueError(f"formal registry does not have one protocol value for {field}")
        protocol_values[field] = values.pop()
    if (protocol_values["dataset"], protocol_values["dataset_version"], protocol_values["split"]) != (
        dataset,
        dataset_version,
        split,
    ):
        raise ValueError("raw and registry protocol identities differ")

    dominant_global = finite_float(
        global_row["mean_dominant_expert_agreement"], field="global.mean_dominant_expert_agreement"
    )
    token_global = finite_float(global_row["mean_token_top1_agreement"], field="global.mean_token_top1_agreement")
    if not 0.0 <= dominant_global <= 1.0 or not 0.0 <= token_global <= 1.0:
        raise ValueError("global routing agreement is outside [0, 1]")
    return {
        "artifact_type": "formal_cross_seed_routing_summary",
        "dataset": dataset,
        "dataset_version": dataset_version,
        "derivation": "lossless scalar and identity summary extracted from the verified formal analyzer output",
        "determinism_rows_passed": determinism_passed,
        "determinism_rows_total": len(determinism_rows),
        "expert_count": len(expert_names),
        "expert_utilization_by_seed_count": len(utilization_rows),
        "formal_root": metadata["formal_root"],
        "image_count": image_count,
        "layer_count": len(layer_names),
        "mean_dominant_expert_agreement": dominant_global,
        "mean_token_top1_agreement": token_global,
        "model": model,
        "pairwise_row_count": pairwise_row_count,
        "protocol": {
            "batch": protocol_values["batch"],
            "epochs": protocol_values["epochs"],
            "imgsz": protocol_values["imgsz"],
            "precision": protocol_values["precision_mode"],
        },
        "publication_status": "raw_source_omitted_from_git",
        "raw_source_location": "private formal archive; not distributed in Git",
        "schema_version": 1,
        "seed_count": len(seeds),
        "seed_pair_count": len(seed_pairs),
        "seeds": seeds,
        "source_file": metadata["source_file"],
        "source_sha256": metadata["source_sha256"],
        "source_size_bytes": metadata["source_size_bytes"],
        "split": split,
    }


def validate_compact_summary(summary: dict[str, object]) -> None:
    """Fail closed unless the compact summary matches all public identity records."""
    metadata = load_source_metadata()
    _, experiments = load_formal_registry()
    expected_identity = {
        "schema_version": 1,
        "artifact_type": "formal_cross_seed_routing_summary",
        "source_file": metadata["source_file"],
        "source_sha256": metadata["source_sha256"],
        "source_size_bytes": metadata["source_size_bytes"],
        "publication_status": "raw_source_omitted_from_git",
        "formal_root": metadata["formal_root"],
        "model": "v10_mot",
        "seeds": list(range(5)),
        "seed_count": 5,
        "dataset": "VisDrone2019-DET",
        "dataset_version": "2019-DET",
        "split": "val-fixed32",
    }
    for field, expected in expected_identity.items():
        if summary.get(field) != expected:
            raise ValueError(f"unexpected compact summary identity: {field}={summary.get(field)!r}")
    protocol = summary.get("protocol")
    if not isinstance(protocol, dict):
        raise ValueError("compact summary is missing protocol identity")
    expected_protocol = {
        "epochs": experiments[0]["epochs"],
        "precision": experiments[0]["precision_mode"],
        "batch": experiments[0]["batch"],
        "imgsz": experiments[0]["imgsz"],
    }
    if protocol != expected_protocol:
        raise ValueError("compact summary protocol differs from the formal registry")
    for field in (
        "image_count",
        "layer_count",
        "expert_count",
        "seed_pair_count",
        "pairwise_row_count",
        "expert_utilization_by_seed_count",
        "determinism_rows_passed",
        "determinism_rows_total",
    ):
        if not isinstance(summary.get(field), int) or int(summary[field]) <= 0:
            raise ValueError(f"invalid compact summary count: {field}")
    if summary["determinism_rows_passed"] != summary["determinism_rows_total"]:
        raise ValueError("compact summary records failed determinism rows")
    for field in ("mean_dominant_expert_agreement", "mean_token_top1_agreement"):
        value = finite_float(summary[field], field=field)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"compact summary agreement is outside [0, 1]: {field}")


def load_architecture_summary() -> list[dict[str, object]]:
    """Load and validate the formal architecture summary."""
    validate_sha256sums(ARCHITECTURE_DIR)
    csv_path = ARCHITECTURE_DIR / "phase3_architecture_summary.csv"
    manifest_path = ARCHITECTURE_DIR / "phase3_architecture_report_manifest.json"
    rows = read_csv(csv_path)
    if len(rows) != len(ARCHITECTURES) or {row["architecture"] for row in rows} != set(ARCHITECTURES):
        raise ValueError("architecture summary must contain EsMoE, MoA, and MoT exactly once")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_rows = {row["architecture"]: row for row in manifest.get("architecture_summary", [])}
    result: list[dict[str, object]] = []
    by_name = {row["architecture"]: row for row in rows}
    for architecture in ARCHITECTURES:
        row = by_name[architecture]
        if int(row["seed_count"]) != SEED_COUNTS[architecture]:
            raise ValueError(f"unexpected seed count for {architecture}")
        mean_50 = finite_float(row["mean_mAP50"], field=f"{architecture}.mean_mAP50")
        mean_95 = finite_float(row["mean_mAP50-95"], field=f"{architecture}.mean_mAP50-95")
        std_50 = (
            None
            if row["sample_std_mAP50"] == ""
            else finite_float(row["sample_std_mAP50"], field=f"{architecture}.sample_std_mAP50")
        )
        std_95 = (
            None
            if row["sample_std_mAP50-95"] == ""
            else finite_float(row["sample_std_mAP50-95"], field=f"{architecture}.sample_std_mAP50-95")
        )
        if not 0.0 <= mean_50 <= 1.0 or not 0.0 <= mean_95 <= 1.0:
            raise ValueError(f"architecture metric outside [0, 1]: {architecture}")
        if architecture == "MoA":
            if std_50 is not None or std_95 is not None:
                raise ValueError("single-seed MoA must not have sample standard deviations")
        elif std_50 is None or std_95 is None or std_50 < 0.0 or std_95 < 0.0:
            raise ValueError(f"missing or invalid sample standard deviation for {architecture}")
        manifest_row = manifest_rows.get(architecture)
        if manifest_row is None or int(manifest_row["seed_count"]) != SEED_COUNTS[architecture]:
            raise ValueError(f"architecture manifest mismatch for {architecture}")
        if not math.isclose(float(manifest_row["mean_mAP50"]), mean_50, abs_tol=1e-12):
            raise ValueError(f"mAP50 differs between CSV and manifest for {architecture}")
        if not math.isclose(float(manifest_row["mean_mAP50-95"]), mean_95, abs_tol=1e-12):
            raise ValueError(f"mAP50-95 differs between CSV and manifest for {architecture}")
        result.append(
            {
                "architecture": architecture,
                "seed_count": SEED_COUNTS[architecture],
                "mean_mAP50": mean_50,
                "std_mAP50": std_50,
                "mean_mAP50-95": mean_95,
                "std_mAP50-95": std_95,
            }
        )
    return result


def validate_utilization(summary: dict[str, object]) -> None:
    """Validate the public 18-row summary and its embedded 90 per-seed values."""
    rows = read_csv(ROUTING_DIR / "reports" / "phase3_mot_expert_utilization.csv")
    keys: set[tuple[str, str]] = set()
    layer_means: defaultdict[str, float] = defaultdict(float)
    by_seed_count = 0
    expected_seeds = [int(seed) for seed in summary["seeds"]]
    for row in rows:
        key = (row["layer_name"], row["expert_name"])
        if key in keys:
            raise ValueError(f"duplicate utilization summary row: {key}")
        keys.add(key)
        mean = finite_float(row["mean_utilization"], field=f"{key}.mean_utilization")
        variance = finite_float(row["between_seed_variance"], field=f"{key}.variance")
        standard_deviation = finite_float(row["between_seed_standard_deviation"], field=f"{key}.std")
        if not 0.0 <= mean <= 1.0 or variance < 0.0 or standard_deviation < 0.0:
            raise ValueError(f"invalid utilization summary values: {key}")
        if int(row["seed_count"]) != summary["seed_count"]:
            raise ValueError(f"unexpected utilization seed count: {key}")
        by_seed = ast.literal_eval(row["utilization_by_seed"])
        if not isinstance(by_seed, list) or sorted(int(item["seed"]) for item in by_seed) != expected_seeds:
            raise ValueError(f"utilization summary does not contain seeds 0-4: {key}")
        for item in by_seed:
            value = finite_float(item["utilization"], field=f"{key}.{item['seed']}.utilization")
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"invalid embedded per-seed utilization: {key}")
        by_seed_count += len(by_seed)
        layer_means[row["layer_name"]] += mean
    expected_keys = {(layer, expert) for layer in LAYERS for expert in EXPERTS}
    if len(rows) != summary["layer_count"] * summary["expert_count"] or keys != expected_keys:
        raise ValueError("utilization summary must contain six layers by three experts")
    if any(not math.isclose(total, 1.0, abs_tol=1e-12) for total in layer_means.values()):
        raise ValueError("mean expert utilization must sum to one within each layer")
    if by_seed_count != summary["expert_utilization_by_seed_count"]:
        raise ValueError("embedded per-seed utilization count differs from the compact summary")


def load_routing_stability() -> tuple[list[dict[str, float | str]], float, float]:
    """Load and validate formal per-layer and global routing agreement."""
    validate_sha256sums(ROUTING_DIR, PUBLIC_CHECKSUM_NAME)
    summary = load_json_object(GLOBAL_SUMMARY_PATH)
    validate_compact_summary(summary)
    validate_utilization(summary)
    dominant_global = finite_float(
        summary["mean_dominant_expert_agreement"], field="global.mean_dominant_expert_agreement"
    )
    token_global = finite_float(summary["mean_token_top1_agreement"], field="global.mean_token_top1_agreement")

    rows = read_csv(ROUTING_DIR / "reports" / "phase3_mot_layer_stability.csv")
    by_layer: dict[str, dict[str, float | str]] = {}
    for row in rows:
        layer = row["layer_name"]
        if layer in by_layer:
            raise ValueError(f"duplicate layer stability row: {layer}")
        dominant = finite_float(row["mean_dominant_expert_agreement"], field=f"{layer}.dominant")
        token = finite_float(row["mean_token_top1_agreement"], field=f"{layer}.token")
        if not 0.0 <= dominant <= 1.0 or not 0.0 <= token <= 1.0:
            raise ValueError(f"layer agreement is outside [0, 1]: {layer}")
        by_layer[layer] = {"layer": layer, "dominant": dominant, "token": token}
    if len(rows) != summary["layer_count"] or set(by_layer) != set(LAYERS):
        raise ValueError("layer stability CSV must contain the six expected MoT layers")

    pairwise_rows = read_csv(ROUTING_DIR / "reports" / "phase3_mot_pairwise_agreement.csv")
    if len(pairwise_rows) != summary["pairwise_row_count"]:
        raise ValueError("pairwise CSV row count differs from the compact summary")
    pair_counts: defaultdict[tuple[int, int], int] = defaultdict(int)
    layer_values: defaultdict[str, list[tuple[float, float]]] = defaultdict(list)
    all_dominant: list[float] = []
    all_token: list[float] = []
    for row in pairwise_rows:
        seed_pair = (int(row["seed_a"]), int(row["seed_b"]))
        layer = row["layer_name"]
        dominant = finite_float(row["dominant_expert_agreement"], field=f"{seed_pair}.{layer}.dominant")
        token = finite_float(row["token_top1_agreement"], field=f"{seed_pair}.{layer}.token")
        js_divergence = finite_float(row["jensen_shannon_divergence"], field=f"{seed_pair}.{layer}.jsd")
        if not 0.0 <= dominant <= 1.0 or not 0.0 <= token <= 1.0 or js_divergence < 0.0:
            raise ValueError(f"invalid pairwise routing metric: {seed_pair}.{layer}")
        pair_counts[seed_pair] += 1
        layer_values[layer].append((dominant, token))
        all_dominant.append(dominant)
        all_token.append(token)
    expected_pairs = set(combinations((int(seed) for seed in summary["seeds"]), 2))
    expected_rows_per_pair = summary["image_count"] * summary["layer_count"]
    if set(pair_counts) != expected_pairs or any(count != expected_rows_per_pair for count in pair_counts.values()):
        raise ValueError("pairwise CSV does not contain the expected ten complete seed pairs")
    if len(pair_counts) != summary["seed_pair_count"] or set(layer_values) != set(LAYERS):
        raise ValueError("pairwise CSV dimensions differ from the compact summary")
    expected_rows_per_layer = summary["image_count"] * summary["seed_pair_count"]
    for layer, values in layer_values.items():
        if len(values) != expected_rows_per_layer:
            raise ValueError(f"pairwise CSV has an incomplete layer: {layer}")
        mean_dominant = sum(value[0] for value in values) / len(values)
        mean_token = sum(value[1] for value in values) / len(values)
        if not math.isclose(mean_dominant, float(by_layer[layer]["dominant"]), abs_tol=1e-12):
            raise ValueError(f"pairwise and layer dominant agreement differ: {layer}")
        if not math.isclose(mean_token, float(by_layer[layer]["token"]), abs_tol=1e-12):
            raise ValueError(f"pairwise and layer token agreement differ: {layer}")
    if not math.isclose(sum(all_dominant) / len(all_dominant), dominant_global, abs_tol=1e-12):
        raise ValueError("pairwise and global dominant agreement differ")
    if not math.isclose(sum(all_token) / len(all_token), token_global, abs_tol=1e-12):
        raise ValueError("pairwise and global token agreement differ")
    return [by_layer[layer] for layer in LAYERS], dominant_global, token_global


def annotate_vertical_bars(axis: plt.Axes, bars: object) -> None:
    """Add compact numerical labels above vertical bars."""
    for bar in bars:
        value = float(bar.get_height())
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.003,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )


def plot_architecture_performance(rows: list[dict[str, object]]) -> Path:
    """Plot formal architecture detection performance."""
    positions = list(range(len(rows)))
    width = 0.34
    mean_50 = [float(row["mean_mAP50"]) for row in rows]
    mean_95 = [float(row["mean_mAP50-95"]) for row in rows]
    figure, axis = plt.subplots(figsize=(16, 9), dpi=100, facecolor="white")
    bars_50 = axis.bar([position - width / 2 for position in positions], mean_50, width, label="mAP50", color="#4C78A8")
    bars_95 = axis.bar(
        [position + width / 2 for position in positions], mean_95, width, label="mAP50-95", color="#F58518"
    )
    for index, row in enumerate(rows):
        std_50 = row["std_mAP50"]
        std_95 = row["std_mAP50-95"]
        if std_50 is not None:
            axis.errorbar(
                positions[index] - width / 2,
                mean_50[index],
                yerr=float(std_50),
                fmt="none",
                ecolor="#222222",
                capsize=5,
                linewidth=1.5,
            )
        if std_95 is not None:
            axis.errorbar(
                positions[index] + width / 2,
                mean_95[index],
                yerr=float(std_95),
                fmt="none",
                ecolor="#222222",
                capsize=5,
                linewidth=1.5,
            )
    annotate_vertical_bars(axis, bars_50)
    annotate_vertical_bars(axis, bars_95)
    moa_index = ARCHITECTURES.index("MoA")
    axis.text(moa_index, max(mean_50[moa_index], mean_95[moa_index]) + 0.018, "single seed\n(no SD)", ha="center")
    labels = [f"{row['architecture']}\nn={row['seed_count']}" for row in rows]
    axis.set_xticks(positions, labels)
    upper_values = [
        float(row[metric]) + float(row[standard_deviation] or 0.0)
        for row in rows
        for metric, standard_deviation in (
            ("mean_mAP50", "std_mAP50"),
            ("mean_mAP50-95", "std_mAP50-95"),
        )
    ]
    axis.set_ylim(0.0, max(0.2, max(upper_values) * 1.28))
    axis.set_ylabel("Detection metric")
    axis.set_title("Phase 3 architecture performance")
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="upper right")
    figure.text(0.5, 0.045, "VisDrone2019-DET, 30 epochs, batch 8, image size 640.", ha="center", fontsize=11)
    figure.subplots_adjust(left=0.08, right=0.97, top=0.88, bottom=0.16)
    output = OUTPUT_DIR / "architecture_performance.png"
    figure.savefig(output, dpi=100, facecolor="white", metadata=PNG_METADATA)
    plt.close(figure)
    return output


def annotate_horizontal_bars(axis: plt.Axes, bars: object) -> None:
    """Add numerical labels without clipping the right edge."""
    for bar in bars:
        value = float(bar.get_width())
        inside = value > 0.92
        axis.text(
            value - 0.015 if inside else value + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="right" if inside else "left",
            va="center",
            fontsize=9,
            color="white" if inside else "#222222",
        )


def plot_routing_stability(rows: list[dict[str, float | str]], dominant_global: float, token_global: float) -> Path:
    """Plot formal per-layer routing agreement in architecture order."""
    positions = list(range(len(rows)))
    height = 0.34
    dominant = [float(row["dominant"]) for row in rows]
    token = [float(row["token"]) for row in rows]
    figure, axis = plt.subplots(figsize=(16, 9), dpi=100, facecolor="white")
    dominant_bars = axis.barh(
        [position - height / 2 for position in positions],
        dominant,
        height,
        label="Dominant expert agreement",
        color="#4C78A8",
    )
    token_bars = axis.barh(
        [position + height / 2 for position in positions],
        token,
        height,
        label="Token top-1 agreement",
        color="#F58518",
    )
    annotate_horizontal_bars(axis, dominant_bars)
    annotate_horizontal_bars(axis, token_bars)
    axis.axvline(
        dominant_global,
        color="#2F5597",
        linestyle="--",
        linewidth=1.5,
        label=f"Global dominant: {dominant_global:.3f}",
    )
    axis.axvline(
        token_global,
        color="#C45A00",
        linestyle=":",
        linewidth=1.8,
        label=f"Global token top-1: {token_global:.3f}",
    )
    axis.set_yticks(positions, [str(row["layer"]) for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Cross-seed agreement")
    axis.set_title("MoT layer routing stability across five seeds")
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.legend(frameon=False, loc="lower right")
    figure.text(0.5, 0.045, "Entropy and routing agreement measure different properties.", ha="center", fontsize=11)
    figure.subplots_adjust(left=0.16, right=0.97, top=0.88, bottom=0.16)
    output = OUTPUT_DIR / "mot_layer_routing_stability.png"
    figure.savefig(output, dpi=100, facecolor="white", metadata=PNG_METADATA)
    plt.close(figure)
    return output


def parse_args() -> argparse.Namespace:
    """Parse public plotting and private provenance-refresh options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-cross-seed-json",
        type=Path,
        help="Private formal raw JSON used only to verify and refresh the compact summary.",
    )
    parser.add_argument(
        "--refresh-global-summary",
        action="store_true",
        help="Rebuild the compact public summary after verifying the selected raw JSON.",
    )
    args = parser.parse_args()
    if args.refresh_global_summary != (args.raw_cross_seed_json is not None):
        parser.error("--refresh-global-summary and --raw-cross-seed-json must be supplied together")
    return args


def main() -> int:
    """Refresh the compact summary or build figures from public formal evidence."""
    args = parse_args()
    if args.refresh_global_summary:
        summary = build_compact_summary(args.raw_cross_seed_json)
        GLOBAL_SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Wrote: {GLOBAL_SUMMARY_PATH.relative_to(ROOT).as_posix()}")
        return 0

    architecture_rows = load_architecture_summary()
    routing_rows, dominant_global, token_global = load_routing_stability()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    architecture_output = plot_architecture_performance(architecture_rows)
    routing_output = plot_routing_stability(routing_rows, dominant_global, token_global)
    print(f"Architecture source: {ARCHITECTURE_DIR.relative_to(ROOT).as_posix()}/phase3_architecture_summary.csv")
    print(f"Layer source: {ROUTING_DIR.relative_to(ROOT).as_posix()}/reports/phase3_mot_layer_stability.csv")
    print(f"Global source: {GLOBAL_SUMMARY_PATH.relative_to(ROOT).as_posix()}")
    print(f"Wrote: {architecture_output.relative_to(ROOT).as_posix()}")
    print(f"Wrote: {routing_output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
