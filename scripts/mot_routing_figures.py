#!/usr/bin/env python3
"""Render the MoT routing-interpretability figures and dashboard.

Consumes the artifacts written by ``scripts/run_mot_routing_interpret.py`` and
produces the expert-activation heatmaps plus a self-contained HTML report.

Two visual-encoding rules are enforced here rather than left to taste:

1. **Magnitude uses one hue, light to dark.** Every routing share is a magnitude
   on a common 0-1 scale, so the heatmaps use a single blue ramp. A multi-hue
   ("rainbow") ramp would invent category boundaries the data does not have; that
   is why the usual ``viridis``/``jet`` default is not used.
2. **Expert identity uses a fixed categorical order.** The three experts always
   take the same three validated hues in ``EXPERT_NAMES`` order, in every figure,
   so a colour learned in one chart transfers to the next. The palette is checked
   with the data-viz validator (all-pairs, both modes); the light-mode aqua slot
   is below 3:1 on the light surface, which is why every series is also
   direct-labelled and mirrored in a table view.

Example:
    python scripts/mot_routing_figures.py \\
        --input /root/autodl-tmp/mot_routing_interpret/visdrone \\
        --checkpoint /root/autodl-tmp/runs/visdrone_mot_ablation/v10_mot/weights/best.pt \\
        --data /root/autodl-tmp/datasets/VisDrone/VisDrone.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yolo_master_matplotlib")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from scripts.mot_routing_interpret import EXPERT_NAMES

SERIES_LIGHT: tuple[str, ...] = ("#2a78d6", "#eb6834", "#1baf7a")
"""Categorical hues for the three experts, in ``EXPERT_NAMES`` order (light mode).

Validated all-pairs on the light surface ``#fcfcfb``: worst CVD ΔE 9.2 (deutan),
worst normal-vision ΔE 24.0, aqua at 2.74:1 contrast (relief rule → direct labels
and table view).
"""

SERIES_DARK: tuple[str, ...] = ("#3987e5", "#d95926", "#199e70")
"""The same three hues stepped for the dark surface ``#1a1a19`` (all checks pass)."""

SEQUENTIAL_STEPS: tuple[str, ...] = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)
"""Single-hue blue ramp (light to dark) for magnitude encoding."""

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
SURFACE_LIGHT = "#fcfcfb"
STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"
NEUTRAL_MID = "#f0efec"

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("moe_blue", SEQUENTIAL_STEPS)
DIVERGING_CMAP = LinearSegmentedColormap.from_list("moe_div", ("#1c5cab", "#2a78d6", NEUTRAL_MID, "#e34948", "#a52222"))

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE_LIGHT,
        "axes.facecolor": SURFACE_LIGHT,
        "savefig.facecolor": SURFACE_LIGHT,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "text.color": INK_PRIMARY,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": BASELINE,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.linewidth": 0.8,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.8,
        "figure.dpi": 140,
    }
)

SHORT_EXPERT = {
    "LocalConvTransformer": "LocalConv",
    "WindowTransformer": "Window",
    "DeformableTransformer": "Deformable",
}
SCENE_LABELS = {
    "dense": "Dense",
    "sparse": "Sparse",
    "small_objects": "Small objects",
    "large_objects": "Large objects",
    "irregular_objects": "Irregular objects",
    "regular_objects": "Regular objects",
    "high_occlusion": "High occlusion",
    "low_occlusion": "Low occlusion",
    "heavy_occlusion": "Heavy occlusion",
}
SCENE_ORDER = (
    "dense",
    "sparse",
    "small_objects",
    "large_objects",
    "irregular_objects",
    "regular_objects",
    "low_occlusion",
    "high_occlusion",
    "heavy_occlusion",
)
CONTRAST_LABELS = {
    "dense_vs_sparse": "密集 vs 稀疏",
    "small_vs_large": "小目标 vs 大目标",
    "irregular_vs_regular": "不规则 vs 规则",
    "occluded_vs_clear": "遮挡 vs 清晰",
}


def short_layer(name: str) -> str:
    """Abbreviate ``model.14.m.0`` to ``14.0`` for axis labels.

    Examples:
        >>> short_layer("model.14.m.0")
        '14.0'
        >>> short_layer("__pooled__")
        'pooled'
    """
    if name == "__pooled__":
        return "pooled"
    parts = name.split(".")
    return f"{parts[1]}.{parts[-1]}" if len(parts) >= 4 else name


def annotate_cells(axis: plt.Axes, matrix: np.ndarray, *, threshold: float = 0.55, fmt: str = "{:.2f}") -> None:
    """Write each cell's value in ink that stays legible against the cell fill.

    A heatmap encodes magnitude in lightness, so the label colour has to flip on
    dark cells; this is also what satisfies the "never colour-only" requirement.
    """
    span = float(np.nanmax(matrix) - np.nanmin(matrix)) or 1.0
    low = float(np.nanmin(matrix))
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            if not np.isfinite(value):
                continue
            normalized = (value - low) / span
            axis.text(
                column,
                row,
                fmt.format(value),
                ha="center",
                va="center",
                fontsize=7.5,
                color="#ffffff" if normalized > threshold else INK_PRIMARY,
            )


def figure_scene_layer_heatmap(payload: dict, output: Path) -> Path:
    """Heatmap of each expert's mean router weight per scene and per MoT layer.

    This is the figure the analysis is named for: rows are scenes, columns are MoT
    blocks, one panel per expert, all three panels on a shared 0-1 colour scale so
    panels are comparable by eye.
    """
    layers = payload["layers"]
    scenes = [scene for scene in SCENE_ORDER if scene in payload["scene_mean_weight"]]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
    matrices = [
        np.asarray([[payload["scene_mean_weight"][scene][layer][index] for layer in layers] for scene in scenes])
        for index in range(len(EXPERT_NAMES))
    ]
    for index, (axis, matrix) in enumerate(zip(axes, matrices)):
        image = axis.imshow(matrix, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_xticks(range(len(layers)), [short_layer(name) for name in layers], fontsize=8)
        axis.set_yticks(
            range(len(scenes)),
            [SCENE_LABELS[scene] for scene in scenes] if index == 0 else [""] * len(scenes),
            fontsize=8,
        )
        axis.set_title(SHORT_EXPERT[EXPERT_NAMES[index]], fontsize=10, color=SERIES_LIGHT[index], pad=8)
        axis.set_xlabel("MoT block", fontsize=8)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        annotate_cells(axis, matrix)
    figure.colorbar(image, ax=axes, shrink=0.82, label="mean router weight", pad=0.015)
    figure.suptitle(
        "Expert activation by scene and MoT block — one hue, darker = higher routing weight",
        fontsize=11,
        y=1.06,
    )
    path = output / "heatmap_scene_layer.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_token_group_heatmap(payload: dict, output: Path) -> Path:
    """Heatmap of the within-image occluded-minus-clear routing shift, per layer.

    Two panels on ONE shared diverging scale: density-controlled (all single-box
    tokens) and additionally size-matched. Both are shown because size matching is
    what changes the conclusion — plotting only the first would credit occlusion for
    a shift that is mostly object size, and the two panels are the same measure on the
    same units, so they belong on one scale rather than in two figures.

    Diverging because the quantity has a meaningful zero: positive means the expert
    is preferred on occluded tokens, negative means it is avoided. The neutral gray
    midpoint reads as "no shift".
    """
    layers = payload["layers"]
    panels = (
        ("occluded_solo", "clear_solo", "density-controlled (single-box tokens)"),
        ("occluded_solo_sized", "clear_solo_sized", "+ size-matched boxes"),
    )

    def lookup(group_a: str, group_b: str, key: str, default: object) -> np.ndarray:
        rows = [
            row
            for row in payload["token_level_tests"]
            if row["group_a"] == group_a and row["group_b"] == group_b and row["layer"] != "__pooled__"
        ]
        return np.asarray(
            [
                [
                    next((row[key] for row in rows if row["expert"] == expert and row["layer"] == layer), default)
                    for layer in layers
                ]
                for expert in EXPERT_NAMES
            ]
        )

    matrices = [lookup(a, b, "hodges_lehmann", float("nan")) for a, b, _ in panels]
    flags = [lookup(a, b, "significant_after_fdr", False) for a, b, _ in panels]
    limit = max((float(np.nanmax(np.abs(matrix))) for matrix in matrices), default=1.0) or 1.0

    figure, axes = plt.subplots(2, 1, figsize=(8.2, 5.4), constrained_layout=True)
    image = None
    for index, (axis, matrix, significant, (_, _, label)) in enumerate(zip(axes, matrices, flags, panels)):
        image = axis.imshow(matrix, cmap=DIVERGING_CMAP, vmin=-limit, vmax=limit, aspect="auto")
        # Columns are shared and vertically aligned, so only the bottom panel is ticked —
        # top-panel labels would otherwise collide with the second panel's title.
        last = index == len(panels) - 1
        axis.set_xticks(range(len(layers)), [short_layer(name) if last else "" for name in layers], fontsize=8)
        axis.set_yticks(range(len(EXPERT_NAMES)), [SHORT_EXPERT[name] for name in EXPERT_NAMES], fontsize=8)
        axis.set_title(label, fontsize=9, color=INK_SECONDARY, pad=5)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix[row, column]
                if not np.isfinite(value):
                    continue
                mark = "✦" if significant[row, column] else ""
                axis.text(
                    column,
                    row,
                    f"{value:+.3f}{mark}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#ffffff" if abs(value) > 0.62 * limit else INK_PRIMARY,
                )
    axes[-1].set_xlabel("MoT block", fontsize=8)
    figure.colorbar(image, ax=axes, shrink=0.75, label="occluded − clear (Hodges–Lehmann)", pad=0.015)
    figure.suptitle(
        "Within-image routing shift on occluded tokens  ✦ = significant after BH FDR",
        fontsize=10,
    )
    path = output / "heatmap_token_shift.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_spatial_maps(spatial: dict, output: Path) -> Path | None:
    """Per-token spatial activation maps for one occluded and one clear exemplar."""
    if not spatial.get("examples"):
        return None
    examples = spatial["examples"]
    figure, axes = plt.subplots(
        len(examples),
        len(EXPERT_NAMES) + 1,
        figsize=(3.0 * (len(EXPERT_NAMES) + 1), 2.7 * len(examples)),
        constrained_layout=True,
    )
    axes = np.atleast_2d(axes)
    for row, example in enumerate(examples):
        image_rgb = np.asarray(example["thumbnail"], dtype=np.float32) / 255.0
        axes[row, 0].imshow(image_rgb)
        axes[row, 0].set_ylabel(
            f"{example['label']}\nocclusion rate {example['occlusion_rate']:.2f}",
            fontsize=8,
            color=INK_SECONDARY,
        )
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        if row == 0:
            axes[row, 0].set_title(f"input ({example['layer']})", fontsize=9, pad=6)
        for index, expert in enumerate(EXPERT_NAMES):
            grid = np.asarray(example["maps"][expert], dtype=np.float64)
            axis = axes[row, index + 1]
            handle = axis.imshow(grid, cmap=SEQUENTIAL_CMAP, vmin=0.0, vmax=1.0)
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title(SHORT_EXPERT[expert], fontsize=9, color=SERIES_LIGHT[index], pad=6)
            axis.set_xlabel(f"mean {grid.mean():.3f}", fontsize=7.5, color=INK_MUTED)
    figure.colorbar(handle, ax=axes[:, 1:].ravel().tolist(), shrink=0.7, label="router weight", pad=0.015)
    figure.suptitle("Per-token expert activation maps (same colour scale as the summary heatmaps)", fontsize=11)
    path = output / "heatmap_spatial.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_causal_map(payload: dict, output: Path) -> Path | None:
    """Bars for the mAP cost of each routing intervention.

    Emphasis encoding: the natural baseline is the reference (gray) and each
    intervention's loss is a single-hue bar, because the story is one comparison
    repeated, not four competing identities.
    """
    rows = payload.get("causal_map")
    if not rows:
        return None
    interventions = [row for row in rows if row["intervention"] != "natural"]
    labels = [
        row["intervention"].replace("forced_", "forced → ").replace("shuffled_routing", "routing shuffled")
        for row in interventions
    ]
    labels = [
        SHORT_EXPERT.get(label.replace("forced → ", ""), label) if label.startswith("forced → ") else label
        for label in labels
    ]
    labels = [
        f"forced → {SHORT_EXPERT[row['intervention'].removeprefix('forced_')]}"
        if row["intervention"].startswith("forced_")
        else "routing shuffled"
        for row in interventions
    ]
    losses = [-100.0 * row["relative_mAP50_95_vs_natural"] for row in interventions]

    figure, axis = plt.subplots(figsize=(7.4, 2.9), constrained_layout=True)
    positions = np.arange(len(labels))
    axis.barh(positions, losses, height=0.52, color=SEQUENTIAL_STEPS[7], zorder=3)
    axis.set_yticks(positions, labels, fontsize=9)
    axis.invert_yaxis()
    axis.set_xlabel("mAP50-95 lost vs natural routing (%)", fontsize=9)
    axis.xaxis.grid(True, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    for position, (loss, row) in enumerate(zip(losses, interventions)):
        axis.text(
            loss + max(losses) * 0.02,
            position,
            f"−{loss:.2f}%  (mAP {row['mAP50_95']:.4f})",
            va="center",
            fontsize=8,
            color=INK_SECONDARY,
        )
    axis.set_xlim(0, max(losses) * 1.42)
    natural = next(row for row in rows if row["intervention"] == "natural")
    axis.set_title(
        f"Causal cost of overriding the router — natural mAP50-95 = {natural['mAP50_95']:.4f}",
        fontsize=10,
        pad=8,
    )
    path = output / "causal_map.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_redundancy(payload: dict, output: Path) -> Path:
    """Grouped bars contrasting output cosine with delta cosine and delta CKA.

    The three measures share one 0-1 axis on purpose: the whole point is that
    output cosine sits near 1.0 for every pair while the delta measures do not, and
    a second axis would destroy that comparison.
    """
    rows = payload["redundancy"]
    layers = [row["layer"] for row in rows]
    pair_keys = list(rows[0]["delta_cosine"].keys())
    figure, axes = plt.subplots(
        1, len(pair_keys), figsize=(4.6 * len(pair_keys), 3.2), constrained_layout=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    width = 0.26
    positions = np.arange(len(layers))
    measures = (
        ("output cosine (misleading)", "output_cosine_uninformative", SERIES_LIGHT[1]),
        ("delta cosine", "delta_cosine", SERIES_LIGHT[0]),
        ("delta CKA", "delta_cka", SERIES_LIGHT[2]),
    )
    for axis, pair in zip(axes, pair_keys):
        for offset, (label, key, colour) in enumerate(measures):
            values = [row[key][pair] for row in rows]
            axis.bar(
                positions + (offset - 1) * width,
                values,
                width=width - 0.02,
                label=label if axis is axes[0] else None,
                color=colour,
                zorder=3,
            )
        axis.set_xticks(positions, [short_layer(name) for name in layers], fontsize=8)
        axis.set_title(" ↔ ".join(SHORT_EXPERT[name] for name in pair.split("|")), fontsize=9.5, pad=6)
        axis.yaxis.grid(True, zorder=0)
        axis.set_axisbelow(True)
        axis.set_ylim(-0.05, 1.05)
        for spine in ("top", "right"):
            axis.spines[spine].set_visible(False)
    axes[0].set_ylabel("similarity", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    figure.suptitle(
        "Expert similarity: raw outputs look identical; the learned residual deltas do not",
        fontsize=11,
    )
    path = output / "expert_redundancy.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def figure_scene_profile(payload: dict, output: Path) -> Path:
    """Stacked bars of the pooled expert mix per scene.

    Part-to-whole: the three router weights sum to 1 per scene, so a stacked bar is
    the honest form. A 2px surface gap separates segments instead of a border.
    """
    scenes = [scene for scene in SCENE_ORDER if scene in payload["scene_mean_weight"]]
    shares = np.asarray([payload["scene_mean_weight"][scene]["__pooled__"] for scene in scenes])
    figure, axis = plt.subplots(figsize=(8.6, 3.4), constrained_layout=True)
    left = np.zeros(len(scenes))
    positions = np.arange(len(scenes))
    for index, expert in enumerate(EXPERT_NAMES):
        values = shares[:, index]
        axis.barh(
            positions, values, left=left, height=0.58, color=SERIES_LIGHT[index], zorder=3, label=SHORT_EXPERT[expert]
        )
        for position, (value, start) in enumerate(zip(values, left)):
            if value > 0.08:
                axis.text(
                    start + value / 2,
                    position,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="#ffffff" if index != 2 else INK_PRIMARY,
                )
        left = left + values
    axis.set_yticks(positions, [SCENE_LABELS[scene] for scene in scenes], fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(0, 1)
    axis.set_xlabel("mean router weight (pooled over MoT blocks)", fontsize=9)
    axis.xaxis.grid(True, zorder=0)
    axis.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        axis.spines[spine].set_visible(False)
    axis.legend(frameon=False, fontsize=8.5, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.42))
    axis.set_title("Expert mix per scene — pooled over all six MoT blocks", fontsize=10.5, pad=8)
    path = output / "scene_profile.png"
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)
    return path


def significant_rows(payload: dict, key: str, expert: str | None = None) -> list[dict[str, Any]]:
    """Return the FDR-significant rows of a test family, optionally for one expert."""
    return [
        row for row in payload[key] if row.get("significant_after_fdr") and (expert is None or row["expert"] == expert)
    ]


def scene_contrast_summary(payload: dict) -> list[dict[str, Any]]:
    """Summarise each scene contrast by how many of its tests survived FDR.

    Reported per contrast rather than pooled, because "some scene test somewhere was
    significant" is not a claim about any particular scene axis.

    Args:
        payload (dict): Parsed ``routing_analysis.json``.

    Returns:
        (list[dict[str, Any]]): One entry per contrast, in declaration order.
    """
    image_rows = [row for row in payload.get("scene_contrast_image_tests", []) if row["layer"] != "__pooled__"]
    token_rows = [row for row in payload.get("scene_contrast_token_tests", []) if row["layer"] != "__pooled__"]
    summary: list[dict[str, Any]] = []
    for contrast in dict.fromkeys(row["contrast"] for row in image_rows):
        image = [row for row in image_rows if row["contrast"] == contrast]
        token = [row for row in token_rows if row["contrast"] == contrast]
        significant_image = [row for row in image if row["significant_after_fdr"]]
        significant_token = [row for row in token if row["significant_after_fdr"]]
        summary.append(
            {
                "contrast": contrast,
                "n_image_tests": len(image),
                "n_image_significant": len(significant_image),
                "n_token_tests": len(token),
                "n_token_significant": len(significant_token),
                "experts_with_token_effect": sorted({row["expert"] for row in significant_token}),
                "largest_token_effect": max(
                    significant_token, key=lambda row: abs(row["hodges_lehmann"]), default=None
                ),
                "largest_image_effect": max(
                    significant_image, key=lambda row: abs(row["stratified_difference"]), default=None
                ),
            }
        )
    return summary


def build_verdict(payload: dict) -> dict[str, Any]:
    """Derive the headline claims, each tied to the test that supports it."""
    token_sized = [
        row
        for row in payload["token_level_tests"]
        if row["group_a"] == "occluded_solo_sized" and row["layer"] != "__pooled__"
    ]
    deformable_sized = [row for row in token_sized if row["expert"] == "DeformableTransformer"]
    deformable_solo = [
        row
        for row in payload["token_level_tests"]
        if row["expert"] == "DeformableTransformer"
        and row["group_a"] == "occluded_solo"
        and row["layer"] != "__pooled__"
    ]
    image_deformable = [
        row
        for row in payload["image_level_occlusion_tests"]
        if row["expert"] == "DeformableTransformer" and row["layer"] != "__pooled__"
    ]
    causal = {row["intervention"]: row for row in payload.get("causal_map", [])}
    collapsed = [row["layer"] for row in payload["collapse"] if row["dead_experts"]]
    return {
        "deformable_rises_with_occlusion": any(
            row["significant_after_fdr"] and row["hodges_lehmann"] > 0 for row in deformable_sized
        ),
        "deformable_sized_significant_layers": [
            row["layer"] for row in deformable_sized if row["significant_after_fdr"] and row["hodges_lehmann"] > 0
        ],
        "deformable_solo_significant_layers": [
            row["layer"] for row in deformable_solo if row["significant_after_fdr"] and row["hodges_lehmann"] > 0
        ],
        "deformable_image_significant_layers": [
            row["layer"]
            for row in image_deformable
            if row["significant_after_fdr"] and row["stratified_difference"] > 0
        ],
        "largest_sized_effect": max(deformable_sized, key=lambda row: row["hodges_lehmann"], default=None),
        "max_forced_cost_relative": min(
            (row["relative_mAP50_95_vs_natural"] for name, row in causal.items() if name.startswith("forced_")),
            default=float("nan"),
        ),
        "shuffle_cost_relative": causal.get("shuffled_routing", {}).get("relative_mAP50_95_vs_natural", float("nan")),
        "collapsed_layers": collapsed,
        "n_token_tests_significant": len(significant_rows(payload, "token_level_tests")),
        "n_image_tests_significant": len(significant_rows(payload, "image_level_occlusion_tests")),
        "scene_contrasts": scene_contrast_summary(payload),
    }


def html_table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, caption: str = "") -> str:
    """Render a table view — the WCAG-clean twin of every figure."""
    head = "".join(f"<th>{header}</th>" for header in headers)
    body = "".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
    label = f"<caption>{caption}</caption>" if caption else ""
    return f"<table>{label}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_dashboard(payload: dict, figures: dict[str, Path], verdict: dict, output: Path) -> Path:
    """Write the self-contained HTML report."""
    scenes = [scene for scene in SCENE_ORDER if scene in payload["scene_mean_weight"]]
    scene_rows = [
        [
            SCENE_LABELS[scene],
            str(int(payload["scene_mean_weight"][scene]["__n_images__"][0])),
            *[f"{value:.4f}" for value in payload["scene_mean_weight"][scene]["__pooled__"]],
        ]
        for scene in scenes
    ]
    token_rows = [
        [
            SHORT_EXPERT[row["expert"]],
            short_layer(row["layer"]),
            str(row["n_pairs"]),
            f"{row['mean_a']:.4f}",
            f"{row['mean_b']:.4f}",
            f"{row['hodges_lehmann']:+.4f}",
            f"{row['superiority']:.2f}",
            f"{row['q_value_bh']:.3g}",
            "yes" if row["significant_after_fdr"] else "no",
        ]
        for row in payload["token_level_tests"]
        if row["group_a"] == "occluded_solo_sized" and row["layer"] != "__pooled__"
    ]
    image_rows = [
        [
            SHORT_EXPERT[row["expert"]],
            short_layer(row["layer"]),
            f"{row['raw_difference']:+.4f}",
            f"{row['stratified_difference']:+.4f}",
            f"{row['q_value_bh']:.3g}",
            "yes" if row["significant_after_fdr"] else "no",
        ]
        for row in payload["image_level_occlusion_tests"]
        if row["layer"] != "__pooled__"
    ]
    scene_summary_rows = [
        [
            CONTRAST_LABELS.get(entry["contrast"], entry["contrast"]),
            f"{entry['n_image_significant']} / {entry['n_image_tests']}",
            f"{entry['n_token_significant']} / {entry['n_token_tests']}",
            ", ".join(SHORT_EXPERT[name] for name in entry["experts_with_token_effect"]) or "—",
            (
                f"{SHORT_EXPERT[entry['largest_token_effect']['expert']]} @ "
                f"{short_layer(entry['largest_token_effect']['layer'])} "
                f"{entry['largest_token_effect']['hodges_lehmann']:+.4f}"
                if entry["largest_token_effect"]
                else "—"
            ),
        ]
        for entry in verdict["scene_contrasts"]
    ]
    scene_image_rows = [
        [
            CONTRAST_LABELS.get(row["contrast"], row["contrast"]),
            SHORT_EXPERT[row["expert"]],
            short_layer(row["layer"]),
            f"{row['raw_difference']:+.4f}",
            f"{row['stratified_difference']:+.4f}",
            f"{row['q_value_bh']:.3g}",
            "yes" if row["significant_after_fdr"] else "no",
        ]
        for row in payload.get("scene_contrast_image_tests", [])
        if row["layer"] != "__pooled__"
    ]
    scene_token_rows = [
        [
            CONTRAST_LABELS.get(row["contrast"], row["contrast"]),
            f"{row['group_a']} vs {row['group_b']}",
            SHORT_EXPERT[row["expert"]],
            short_layer(row["layer"]),
            str(row["n_pairs"]),
            f"{row['mean_a']:.4f}",
            f"{row['mean_b']:.4f}",
            f"{row['hodges_lehmann']:+.4f}",
            f"{row['q_value_bh']:.3g}",
            "yes" if row["significant_after_fdr"] else "no",
        ]
        for row in payload.get("scene_contrast_token_tests", [])
        if row["layer"] != "__pooled__"
    ]
    causal_rows = [
        [
            row["intervention"],
            f"{row['mAP50_95']:.5f}",
            f"{row['mAP50']:.5f}",
            f"{row['delta_mAP50_95_vs_natural']:+.5f}",
            f"{row['relative_mAP50_95_vs_natural']:+.2%}",
        ]
        for row in payload.get("causal_map", [])
    ]
    redundancy_rows = [
        [
            short_layer(row["layer"]),
            pair.replace("|", " ↔ "),
            f"{row['output_cosine_uninformative'][pair]:.4f}",
            f"{row['delta_cosine'][pair]:.4f}",
            f"{row['delta_cka'][pair]:.4f}",
        ]
        for row in payload["redundancy"]
        for pair in row["delta_cosine"]
    ]
    collapse_rows = [
        [
            short_layer(row["layer"]),
            *[f"{value:.4f}" for value in row["mean_weight"]],
            f"{row['gini']:.3f}",
            f"{row['entropy']:.3f}",
            ", ".join(EXPERT_NAMES[index] for index in row["dead_experts"]) or "—",
        ]
        for row in payload["collapse"]
    ]

    largest = verdict["largest_sized_effect"]
    if largest and verdict["deformable_rises_with_occlusion"]:
        largest_note = (
            f"+{largest['hodges_lehmann']:.4f} 权重（{largest['mean_b']:.4f} → {largest['mean_a']:.4f}，"
            f"n={largest['n_pairs']} 图像配对，q={largest['q_value_bh']:.2g}）"
        )
        largest_layer = largest["layer"]
    else:
        largest_note = "无显著正向效应"
        largest_layer = "—"
    uncontrolled = verdict["deformable_solo_significant_layers"]
    controlled = set(verdict["deformable_sized_significant_layers"])
    lost_to_size = [layer for layer in uncontrolled if layer not in controlled]
    size_note = (
        f"未控制尺寸时 <code>{'</code>、<code>'.join(short_layer(name) for name in lost_to_size)}</code> "
        "也显著，但尺寸匹配后消失，说明那部分差异主要由“遮挡物体更小”驱动，而不是遮挡本身。"
        if lost_to_size
        else "未控制尺寸的比较没有额外的显著层，因此尺寸混淆在本次数据上不是主要来源。"
    )
    occlusion_note = (
        "ground-truth VisDrone occlusion labels"
        if payload.get("dataset_kind") == "visdrone"
        else "occlusion derived from box overlap and iscrowd (proxy — weaker evidence)"
    )
    verdict_state = "confirmed-in-part" if verdict["deformable_rises_with_occlusion"] else "not-supported"
    verdict_colour = STATUS_GOOD if verdict["deformable_rises_with_occlusion"] else STATUS_CRITICAL

    def image_block(key: str, title: str, note: str) -> str:
        if key not in figures:
            return ""
        return f'<figure><img src="{figures[key].name}" alt="{title}"><figcaption>{note}</figcaption></figure>'

    palette_css = "\n".join(f"      --series-{index + 1}: {colour};" for index, colour in enumerate(SERIES_LIGHT))
    palette_css_dark = "\n".join(f"      --series-{index + 1}: {colour};" for index, colour in enumerate(SERIES_DARK))

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MoT routing interpretability — {Path(payload["checkpoint"]).parent.parent.name}</title>
<style>
  .viz-root {{
    color-scheme: light;
    --surface-1: {SURFACE_LIGHT};
    --page: #f9f9f7;
    --text-primary: {INK_PRIMARY};
    --text-secondary: {INK_SECONDARY};
    --text-muted: {INK_MUTED};
    --gridline: {GRIDLINE};
    --baseline: {BASELINE};
    --good: {STATUS_GOOD};
    --critical: {STATUS_CRITICAL};
{palette_css}
  }}
  @media (prefers-color-scheme: dark) {{
    :root:where(:not([data-theme="light"])) .viz-root {{
      color-scheme: dark;
      --surface-1: #1a1a19;
      --page: #0d0d0d;
      --text-primary: #ffffff;
      --text-secondary: #c3c2b7;
      --text-muted: #898781;
      --gridline: #2c2c2a;
      --baseline: #383835;
      --good: #0ca30c;
      --critical: #d03b3b;
{palette_css_dark}
    }}
  }}
  :root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
    --surface-1: #1a1a19;
    --page: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --baseline: #383835;
{palette_css_dark}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--page); }}
  .viz-root {{
    background: var(--page);
    color: var(--text-primary);
    font: 14px/1.62 system-ui, -apple-system, "Segoe UI", sans-serif;
    padding: 40px 28px 72px;
  }}
  .wrap {{ max-width: 1120px; margin: 0 auto; }}
  h1 {{ font-size: 25px; font-weight: 600; margin: 0 0 6px; letter-spacing: -0.01em; }}
  h2 {{ font-size: 17px; font-weight: 600; margin: 44px 0 10px; }}
  h3 {{ font-size: 14px; font-weight: 600; margin: 26px 0 8px; color: var(--text-secondary); }}
  p {{ margin: 0 0 12px; color: var(--text-secondary); max-width: 78ch; }}
  .sub {{ color: var(--text-muted); font-size: 12.5px; margin-bottom: 26px; }}
  .kpis {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 14px; margin: 22px 0 8px; }}
  .tile {{ background: var(--surface-1); border: 1px solid var(--gridline); border-radius: 10px; padding: 15px 17px; }}
  .tile .label {{ font-size: 11.5px; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }}
  .tile .value {{ font-size: 27px; font-weight: 600; margin-top: 5px; letter-spacing: -0.02em; }}
  .tile .foot {{ font-size: 11.5px; color: var(--text-muted); margin-top: 3px; }}
  .verdict {{ background: var(--surface-1); border: 1px solid var(--gridline); border-left: 3px solid {verdict_colour};
              border-radius: 10px; padding: 17px 20px; margin: 26px 0; }}
  .verdict .head {{ font-weight: 600; color: {verdict_colour}; margin-bottom: 7px; }}
  figure {{ margin: 18px 0 8px; background: var(--surface-1); border: 1px solid var(--gridline);
            border-radius: 10px; padding: 15px; }}
  figure img {{ width: 100%; height: auto; display: block; border-radius: 4px; }}
  figcaption {{ font-size: 12.5px; color: var(--text-muted); margin-top: 11px; }}
  details {{ margin: 12px 0 22px; }}
  summary {{ cursor: pointer; font-size: 13px; color: var(--text-secondary); padding: 5px 0; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 12.5px; margin-top: 10px;
           font-variant-numeric: tabular-nums; background: var(--surface-1); }}
  caption {{ text-align: left; font-size: 12px; color: var(--text-muted); padding: 6px 0 9px; }}
  th, td {{ text-align: right; padding: 6px 10px; border-bottom: 1px solid var(--gridline); }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  thead th {{ color: var(--text-secondary); font-weight: 600; border-bottom: 1px solid var(--baseline); }}
  code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px;
          background: var(--gridline); padding: 1px 5px; border-radius: 3px; }}
  .swatch {{ display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 6px; }}
  ul {{ color: var(--text-secondary); max-width: 78ch; padding-left: 20px; }}
  li {{ margin-bottom: 6px; }}
</style>
</head>
<body>
<div class="viz-root">
<div class="wrap">

<h1>MoT 路由行为可解释性分析</h1>
<div class="sub">
  checkpoint <code>{payload["checkpoint"]}</code> · {payload["n_images"]} images from
  <code>{payload["data"]}</code> ({payload["split"]}) · imgsz {payload["imgsz"]} ·
  occlusion label: {occlusion_note}
</div>

<div class="verdict">
  <div class="head">结论：DeformableTransformer 在遮挡场景的激活率上升 — {verdict_state}</div>
  <p style="margin-bottom:0">
    在<strong>控制了物体尺寸与图像内密度</strong>之后，Deformable 的路由权重在
    {len(controlled)} / {len(payload["layers"])} 个 MoT block 上于遮挡 token 显著升高
    （FDR 校正后），最大效应出现在 <code>{short_layer(largest_layer)}</code>：{largest_note}。
    这是一个<strong>真实但很小</strong>的效应，且并非全局性：{size_note}
  </p>
</div>

<div class="kpis">
  <div class="tile">
    <div class="label">Token 级显著检验</div>
    <div class="value">{verdict["n_token_tests_significant"]}</div>
    <div class="foot">BH FDR 校正后 · 图像内配对 Wilcoxon</div>
  </div>
  <div class="tile">
    <div class="label">强制单专家最大代价</div>
    <div class="value">{-100 * verdict["max_forced_cost_relative"]:.2f}%</div>
    <div class="foot">mAP50-95 相对下降</div>
  </div>
  <div class="tile">
    <div class="label">打乱路由代价</div>
    <div class="value">{-100 * verdict["shuffle_cost_relative"]:.2f}%</div>
    <div class="foot">保留权重分布、破坏内容依赖</div>
  </div>
  <div class="tile">
    <div class="label">坍缩的 block</div>
    <div class="value">{len(verdict["collapsed_layers"])} / {len(payload["layers"])}</div>
    <div class="foot">{", ".join(short_layer(name) for name in verdict["collapsed_layers"]) or "无"}</div>
  </div>
</div>

<h2>1 · 专家激活热力图（场景 × MoT block）</h2>
<p>
  行为场景分组，列为六个 MoT block，三个面板分别对应三个 expert，共用同一 0–1 色阶。
  <span class="swatch" style="background:var(--series-1)"></span>LocalConv
  <span class="swatch" style="background:var(--series-2)"></span>Window
  <span class="swatch" style="background:var(--series-3)"></span>Deformable
</p>
{image_block("scene_layer", "Expert activation by scene and layer", "单色阶（蓝，越深权重越高）表示量级；每格标注数值，颜色不是唯一编码通道。")}
<details><summary>表格视图 — 按场景的 pooled 专家权重</summary>
{html_table(["场景", "图像数", *[SHORT_EXPERT[name] for name in EXPERT_NAMES]], scene_rows, caption="mean router weight, pooled over all MoT blocks")}
</details>

{image_block("scene_profile", "Expert mix per scene", "堆叠条形：三个专家权重按定义和为 1，因此这是部分-整体关系。")}

<h2>2 · 遮挡假设的检验</h2>
<p>
  这里区分三种越来越严格的比较。VisDrone 中遮挡与<strong>密度</strong>和<strong>物体尺寸</strong>高度相关，
  所以只有最后一种比较能把遮挡本身分离出来：
</p>
<ul>
  <li><strong>图像级</strong>高遮挡 vs 低遮挡：受密度和尺寸混淆，需分层置换检验控制协变量。</li>
  <li><strong>图像内 token 级</strong>（<code>occluded_solo</code> vs <code>clear_solo</code>）：同一张图内比较，密度已被控制，但尺寸未控制。</li>
  <li><strong>尺寸匹配 token 级</strong>：再限制到遮挡框与清晰框共同的面积区间 —
      <code>[{payload["size_matched_area_band"][0]:.6f}, {payload["size_matched_area_band"][1]:.6f}]</code>，
      这是唯一能支持因果解读的比较。</li>
</ul>
{image_block("token_shift", "Within-image routing shift", "双色发散色阶（蓝↔红，中点为中性灰）：正值表示该专家在遮挡 token 上被更多使用。✦ 标记 FDR 校正后显著。")}
<details><summary>表格视图 — 尺寸匹配的 token 级配对检验（全部专家 × block）</summary>
{html_table(["专家", "block", "n", "遮挡均值", "清晰均值", "HL 效应", "优势比", "q (BH)", "显著"], token_rows, caption="occluded_solo_sized vs clear_solo_sized, paired Wilcoxon within image, one-sided")}
</details>
<details><summary>表格视图 — 图像级分层置换检验</summary>
{html_table(["专家", "block", "原始差", "分层后差", "q (BH)", "显著"], image_rows, caption="high vs low occlusion, stratified by n_objects x median_area quantile bins")}
</details>

<h2>3 · 其余三组场景对比（与遮挡同级别的检验）</h2>
<p>
  密集 vs 稀疏、小目标 vs 大目标、不规则 vs 规则这三组，采用与遮挡完全相同的两级检验：
  <strong>图像级</strong>分层置换检验（把另外两个场景维度分箱后只在层内置换标签），以及
  <strong>图像内 token 级</strong>配对 Wilcoxon（两臂都只取被恰好一个框覆盖的 token）。
  区别只有一处：遮挡的假设有方向（Deformable 应当上升），因此是单侧；这三组没有先验方向，用双侧检验。
  三组共用一个 BH-FDR 检验族。
</p>
<ul>
  <li><strong>密集 vs 稀疏</strong>：图像级按 <code>n_objects</code> 分组，控制 <code>median_area × occlusion_rate</code>；
      token 级用局部密度 —— 以 {payload.get("density_radius_tokens", 1)} 个 token 为半径统计邻域内的框中心数，
      在同一张图内比较拥挤邻域与孤立邻域的单框 token。</li>
  <li><strong>小目标 vs 大目标</strong>：图像级按 <code>median_area</code> 分组，控制 <code>n_objects × occlusion_rate</code>；
      token 级取全数据集面积三分位
      <code>{payload.get("size_contrast_area_terciles", [0, 0])[0]:.6f}</code> /
      <code>{payload.get("size_contrast_area_terciles", [0, 0])[1]:.6f}</code> 的两端，中间一档只计入占用、不参与比较。</li>
  <li><strong>不规则 vs 规则</strong>：像素长宽比 ≥3 或 ≤1/3 判为不规则（按像素而非归一化坐标，否则非方形图像会被整体拉伸）。
      图像级按 <code>irregular_rate</code> 分组，控制 <code>n_objects × median_area</code>；token 级另有一版把两臂限制在共同面积区间
      <code>[{payload.get("shape_matched_area_band", [0, 1])[0]:.6f}, {payload.get("shape_matched_area_band", [0, 1])[1]:.6f}]</code>，
      以排除“不规则框本身更大/更小”的解释。</li>
</ul>
{html_table(["场景对比", "图像级显著", "token 级显著", "有 token 效应的专家", "最大 token 效应"], scene_summary_rows, caption="significant tests after BH-FDR, pooled-layer rows excluded from the family")}
<details><summary>表格视图 — 三组场景的图像级分层置换检验（全部）</summary>
{html_table(["场景对比", "专家", "block", "原始差", "分层后差", "q (BH)", "显著"], scene_image_rows, caption="two-sided stratified permutation test, covariates held fixed by quantile bins")}
</details>
<details><summary>表格视图 — 三组场景的 token 级配对检验（全部）</summary>
{html_table(["场景对比", "token 组", "专家", "block", "n", "A 均值", "B 均值", "HL 效应", "q (BH)", "显著"], scene_token_rows, caption="two-sided paired Wilcoxon within image, single-box tokens on both arms")}
</details>

<h2>4 · 三个专家是否学到了不同的函数</h2>
<p>
  每个 expert 都是 <code>x + ls1·attn(x) + ls2·ffn(x)</code>（<code>ls</code> 初始化为 0.1），
  MoTBlock 之后还有一次块级残差。因此直接对 expert <strong>输出</strong>做余弦相似度必然接近 1，
  这个数字不能作为"专家同质化"的证据。下图把恒等路径减掉，只比较学到的增量 <code>expert(x) − x</code>。
</p>
{image_block("redundancy", "Expert redundancy on residual deltas", "三个度量共用同一 0–1 纵轴（不使用双轴）：橙色为会误导人的输出余弦，蓝/绿为去掉残差后的真实相似度。")}
<details><summary>表格视图 — 相似度</summary>
{html_table(["block", "专家对", "输出余弦（误导）", "delta 余弦", "delta CKA"], redundancy_rows, caption="cosine on raw outputs vs cosine and linear CKA on expert(x) - x")}
</details>

<h2>5 · 因果验证：覆盖路由器的代价</h2>
{image_block("causal", "Causal mAP cost", "强调式编码：只有一个量（相对 mAP 损失）重复四次，因此用单色条形而非四种类别色。")}
<details><summary>表格视图 — mAP 干预结果</summary>
{html_table(["干预", "mAP50-95", "mAP50", "Δ mAP50-95", "相对变化"], causal_rows, caption="full-split re-validation under each routing intervention")}
</details>

<h2>6 · 路由健康度</h2>
<details open><summary>表格视图 — 每个 block 的专家使用集中度</summary>
{html_table(["block", *[SHORT_EXPERT[name] for name in EXPERT_NAMES], "Gini", "熵", "死亡专家"], collapse_rows, caption="mean router weight per expert, with concentration measures")}
</details>

{image_block("spatial", "Per-token spatial activation", "同一色阶下的逐 token 激活图，用于目视核对上面的统计结论。")}

<h2>方法与局限</h2>
<ul>
  <li>路由权重通过 <code>MoTBlock.router</code> 的 forward hook 采集，模型处于 <code>eval()</code>，
      因此 <code>exploration_eps</code> 的训练期稠密下限不生效，读到的是真实推理期路由。</li>
  <li>图像按 <code>imgsz</code> 直接缩放而非 letterbox，这样归一化标注坐标可以无偏移地映射到 token 网格。</li>
  <li>token 归属规则：框中心落入判定；覆盖不到任何 token 中心的小框归到最近的一个 token，避免小目标被静默丢弃。</li>
  <li>多重比较用 Benjamini–Hochberg 控制 FDR，跨层 pooled 行不计入检验族（它们是族内行的聚合）。
      遮挡的图像级、遮挡的 token 级、三组场景的图像级、三组场景的 token 级各自构成一个检验族。</li>
  <li><code>model.23.m.1</code> 已完全坍缩到单一专家，其所有检验结果恒为零差异，不构成证据。</li>
  <li>本页所有配色经数据可视化调色板校验器（all-pairs、明暗两种模式）验证；浅色模式下 aqua 对比度低于 3:1，
      因此每个图形都配有直接数值标注与表格视图，颜色从不是唯一的编码通道。</li>
</ul>

</div>
</div>
</body>
</html>
"""
    path = output / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Directory written by run_mot_routing_interpret.py.")
    parser.add_argument("--spatial", type=Path, default=None, help="Optional spatial-maps JSON from --emit-spatial.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for figure and dashboard rendering."""
    args = parse_args(argv)
    payload = json.loads((args.input / "routing_analysis.json").read_text())
    output = args.input
    figures: dict[str, Path] = {
        "scene_layer": figure_scene_layer_heatmap(payload, output),
        "scene_profile": figure_scene_profile(payload, output),
        "token_shift": figure_token_group_heatmap(payload, output),
        "redundancy": figure_redundancy(payload, output),
    }
    causal = figure_causal_map(payload, output)
    if causal is not None:
        figures["causal"] = causal

    spatial_path = args.spatial or (args.input / "spatial_maps.json")
    if spatial_path.exists():
        spatial = json.loads(spatial_path.read_text())
        rendered = figure_spatial_maps(spatial, output)
        if rendered is not None:
            figures["spatial"] = rendered

    verdict = build_verdict(payload)
    (output / "verdict.json").write_text(json.dumps(verdict, indent=2, default=float), encoding="utf-8")
    dashboard = render_dashboard(payload, figures, verdict, output)
    for name, path in figures.items():
        print(f"[figures] {name}: {path}")
    print(f"[figures] dashboard: {dashboard}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
