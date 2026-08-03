"""Build deterministic PR #216 figures from verified Phase 3 evidence."""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path, PurePosixPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


ROOT = Path(__file__).resolve().parents[2]
ARCHITECTURE_DIR = ROOT / "docs" / "issue54" / "phase3_architecture_controls"
ROUTING_DIR = ROOT / "docs" / "issue54" / "phase3_mot_routing"
OUTPUT_DIR = ROOT / "docs" / "issue54" / "pr_assets"

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


def validate_sha256sums(directory: Path) -> None:
    """Fail unless every entry in ``SHA256SUMS`` matches its local file."""
    checksum_path = directory / "SHA256SUMS"
    lines = checksum_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError(f"empty checksum file: {checksum_path}")
    seen: set[str] = set()
    for line in lines:
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("*")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or relative in seen:
            raise ValueError(f"unsafe or duplicate checksum entry: {relative}")
        seen.add(relative)
        target = directory.joinpath(*pure.parts)
        if not target.is_file() or sha256_file(target) != expected.lower():
            raise ValueError(f"SHA256 mismatch: {target}")


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


def validate_utilization(analysis: dict[str, object]) -> None:
    """Validate the canonical 18-row summary and 90-item per-seed utilization semantics."""
    rows = read_csv(ROUTING_DIR / "reports" / "phase3_mot_expert_utilization.csv")
    keys: set[tuple[str, str]] = set()
    layer_means: defaultdict[str, float] = defaultdict(float)
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
        if int(row["seed_count"]) != 5:
            raise ValueError(f"unexpected utilization seed count: {key}")
        by_seed = ast.literal_eval(row["utilization_by_seed"])
        if sorted(int(item["seed"]) for item in by_seed) != list(range(5)):
            raise ValueError(f"utilization summary does not contain seeds 0-4: {key}")
        layer_means[row["layer_name"]] += mean
    expected_keys = {(layer, expert) for layer in LAYERS for expert in EXPERTS}
    if len(rows) != 18 or keys != expected_keys:
        raise ValueError("utilization summary must contain six layers by three experts")
    if any(not math.isclose(total, 1.0, abs_tol=1e-12) for total in layer_means.values()):
        raise ValueError("mean expert utilization must sum to one within each layer")

    raw_rows = analysis.get("expert_utilization_by_seed")
    if not isinstance(raw_rows, list):
        raise ValueError("analysis is missing expert_utilization_by_seed")
    raw_keys: set[tuple[int, str, str]] = set()
    raw_sums: defaultdict[tuple[int, str], float] = defaultdict(float)
    for row in raw_rows:
        seed = int(row["seed"])
        layer = str(row["layer_name"])
        expert = str(row["expert_name"])
        value = finite_float(row["utilization"], field=f"{seed}.{layer}.{expert}.utilization")
        key = (seed, layer, expert)
        if key in raw_keys or not 0.0 <= value <= 1.0:
            raise ValueError(f"duplicate or invalid per-seed utilization: {key}")
        raw_keys.add(key)
        raw_sums[(seed, layer)] += value
    expected_raw_keys = {(seed, layer, expert) for seed in range(5) for layer in LAYERS for expert in EXPERTS}
    if len(raw_rows) != 90 or raw_keys != expected_raw_keys:
        raise ValueError("per-seed utilization must contain five seeds by six layers by three experts")
    if any(not math.isclose(total, 1.0, abs_tol=1e-12) for total in raw_sums.values()):
        raise ValueError("per-seed expert utilization must sum to one within each seed and layer")


def load_routing_stability() -> tuple[list[dict[str, float | str]], float, float]:
    """Load and validate formal per-layer and global routing agreement."""
    validate_sha256sums(ROUTING_DIR)
    analysis = json.loads((ROUTING_DIR / "phase3_cross_seed_routing.json").read_text(encoding="utf-8"))
    validate_utilization(analysis)
    if analysis.get("analysis_mode") != "formal_passed_only" or analysis.get("image_count") != 32:
        raise ValueError("routing analysis is not the expected formal 32-image analysis")
    if len(analysis.get("pairwise_seed_comparisons", [])) != 1920:
        raise ValueError("routing analysis must contain 1,920 pairwise rows")
    global_rows = analysis.get("global_summary")
    if not isinstance(global_rows, list) or len(global_rows) != 1:
        raise ValueError("routing analysis must contain one formal global summary")
    global_row = global_rows[0]
    if global_row.get("model_variant") != "v10_mot" or global_row.get("split") != "val-fixed32":
        raise ValueError("unexpected routing analysis identity")
    dominant_global = finite_float(
        global_row["mean_dominant_expert_agreement"], field="global.mean_dominant_expert_agreement"
    )
    token_global = finite_float(global_row["mean_token_top1_agreement"], field="global.mean_token_top1_agreement")
    if not 0.0 <= dominant_global <= 1.0 or not 0.0 <= token_global <= 1.0:
        raise ValueError("global routing agreement is outside [0, 1]")

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
    if len(rows) != 6 or set(by_layer) != set(LAYERS):
        raise ValueError("layer stability CSV must contain the six expected MoT layers")
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


def main() -> int:
    """Validate formal sources and write the two deterministic PR figures."""
    architecture_rows = load_architecture_summary()
    routing_rows, dominant_global, token_global = load_routing_stability()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    architecture_output = plot_architecture_performance(architecture_rows)
    routing_output = plot_routing_stability(routing_rows, dominant_global, token_global)
    print(f"Architecture source: {ARCHITECTURE_DIR.relative_to(ROOT).as_posix()}/phase3_architecture_summary.csv")
    print(f"Layer source: {ROUTING_DIR.relative_to(ROOT).as_posix()}/reports/phase3_mot_layer_stability.csv")
    print(f"Global source: {ROUTING_DIR.relative_to(ROOT).as_posix()}/phase3_cross_seed_routing.json")
    print(f"Wrote: {architecture_output.relative_to(ROOT).as_posix()}")
    print(f"Wrote: {routing_output.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
