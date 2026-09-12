#!/usr/bin/env python3
"""Validate and render the B1 controlled-intervention result figures."""

from __future__ import annotations

import argparse
import csv
import html
from pathlib import Path
from typing import Iterable


SEEDS = (0, 1, 2)
DIRECT_CONDITIONS = ("T_NATIVE", "T", "GZ", "GW", "E0", "E1", "FZ", "FW")
PALETTE = ("#2563eb", "#ea580c", "#059669")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, ValueError) as exc:
        raise ValueError(f"invalid numeric field {key!r}: {row.get(key)!r}") from exc


def validate_conditions(rows: Iterable[dict[str, str]]) -> dict[tuple[int, str], dict[str, str]]:
    indexed: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["seed"]), row["condition"])
        if key in indexed:
            raise ValueError(f"duplicate condition row: {key}")
        indexed[key] = row
    expected = {(seed, condition) for seed in SEEDS for condition in DIRECT_CONDITIONS}
    if set(indexed) != expected:
        missing = sorted(expected - set(indexed))
        extra = sorted(set(indexed) - expected)
        raise ValueError(f"condition matrix mismatch; missing={missing}, extra={extra}")
    for seed in SEEDS:
        native = indexed[(seed, "T_NATIVE")]
        patched = indexed[(seed, "T")]
        if _number(native, "new17_AP") != _number(patched, "new17_AP"):
            raise ValueError(f"T_NATIVE and T New17 AP differ for seed {seed}")
        if int(patched["expert0_count"]) + int(patched["expert1_count"]) != 5000:
            raise ValueError(f"native expert counts do not cover 5000 images for seed {seed}")
    return indexed


def validate_contrasts(
    rows: Iterable[dict[str, str]], conditions: dict[tuple[int, str], dict[str, str]]
) -> dict[tuple[int, str, str], dict[str, str]]:
    indexed: dict[tuple[int, str, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["seed"]), row["contrast"], row["metric"])
        if key in indexed:
            raise ValueError(f"duplicate contrast row: {key}")
        indexed[key] = row
        low = _number(row, "bootstrap_p2_5")
        high = _number(row, "bootstrap_p97_5")
        if low > high:
            raise ValueError(f"reversed bootstrap interval: {key}")
        if int(row["bootstrap_replicates"]) != 1000:
            raise ValueError(f"unexpected replicate count: {key}")
    for seed in SEEDS:
        for contrast, left, right in (
            ("GZ-T", "GZ", "T"),
            ("GW-T", "GW", "T"),
            ("E1-E0", "E1", "E0"),
        ):
            row = indexed[(seed, contrast, "AP")]
            observed = _number(row, "point_delta")
            expected = _number(conditions[(seed, left)], "new17_AP") - _number(conditions[(seed, right)], "new17_AP")
            if abs(observed - expected) > 1e-12:
                raise ValueError(f"point delta mismatch for seed {seed} {contrast}")
        for required in ("FT-T", "intFZ", "intFW"):
            if (seed, required, "AP") not in indexed:
                raise ValueError(f"missing AP contrast for seed {seed} {required}")
    return indexed


def validate_opportunities(rows: Iterable[dict[str, str]]) -> dict[int, dict[str, str]]:
    indexed = {int(row["seed"]): row for row in rows}
    if set(indexed) != set(SEEDS):
        raise ValueError(f"opportunity seeds mismatch: {sorted(indexed)}")
    for seed, row in indexed.items():
        event_count = int(row["event_count"])
        opportunity_count = int(row["opportunity_count"])
        action_count = int(row["nontrivial_action_count"])
        partition = int(row["images_E0_better"]) + int(row["images_E1_better"]) + int(row["images_tie"])
        if event_count != 5000 or partition != event_count:
            raise ValueError(f"invalid opportunity partition for seed {seed}")
        if not 0 < action_count <= opportunity_count <= event_count:
            raise ValueError(f"invalid opportunity/action counts for seed {seed}")
    return indexed


def _escape(value: object) -> str:
    return html.escape(str(value), quote=True)


def _forest_panel(
    rows: list[tuple[int, str, float, float, float]],
    *,
    x: float,
    y: float,
    width: float,
    title: str,
    low: float,
    high: float,
) -> str:
    height = 212

    def sx(value: float) -> float:
        return x + (value - low) * width / (high - low)

    parts = [f'<text class="panel-title" x="{x}" y="{y}">{_escape(title)}</text>']
    zero = sx(0)
    parts.append(f'<line class="zero" x1="{zero:.1f}" y1="{y + 20}" x2="{zero:.1f}" y2="{y + height}" />')
    for tick in (low, 0.0, high):
        tx = sx(tick)
        parts.append(f'<text class="tick" x="{tx:.1f}" y="{y + height + 18}" text-anchor="middle">{tick:+.2f}</text>')
    for index, (seed, label, point, ci_low, ci_high) in enumerate(rows):
        yy = y + 40 + index * 27
        color = PALETTE[seed]
        parts.append(f'<text class="row-label" x="{x - 8}" y="{yy + 4}" text-anchor="end">S{seed} {label}</text>')
        parts.append(
            f'<line x1="{sx(ci_low):.1f}" y1="{yy}" x2="{sx(ci_high):.1f}" y2="{yy}" stroke="{color}" stroke-width="3" />'
        )
        parts.append(f'<circle cx="{sx(point):.1f}" cy="{yy}" r="4.5" fill="{color}" />')
    return "\n".join(parts)


def render_controlled_svg(
    conditions: dict[tuple[int, str], dict[str, str]],
    contrasts: dict[tuple[int, str, str], dict[str, str]],
) -> str:
    gate_rows: list[tuple[int, str, float, float, float]] = []
    selection_rows: list[tuple[int, str, float, float, float]] = []
    for seed in SEEDS:
        for label in ("GZ−T", "GW−T"):
            key = label.replace("−", "-")
            row = contrasts[(seed, key, "AP")]
            gate_rows.append(
                (
                    seed,
                    label,
                    100 * _number(row, "point_delta"),
                    100 * _number(row, "bootstrap_p2_5"),
                    100 * _number(row, "bootstrap_p97_5"),
                )
            )
        for label in ("E1−E0", "FT−T"):
            key = label.replace("−", "-")
            row = contrasts[(seed, key, "AP")]
            selection_rows.append(
                (
                    seed,
                    label,
                    100 * _number(row, "point_delta"),
                    100 * _number(row, "bootstrap_p2_5"),
                    100 * _number(row, "bootstrap_p97_5"),
                )
            )
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1320" height="430" viewBox="0 0 1320 430" role="img" aria-labelledby="title desc">',
        '<title id="title">Controlled routing interventions across three checkpoints</title>',
        '<desc id="desc">Paired New17 AP effects for gate substitutions, fixed experts, flipped native choices, and native expert use.</desc>',
        """<style>
          .bg{fill:#fff}.title{font:700 22px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}
          .subtitle,.note{font:12px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#4b5563}
          .panel-title{font:700 15px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}
          .row-label,.tick{font:11px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#374151}
          .zero{stroke:#111827;stroke-width:1;stroke-dasharray:3 3}.frame{fill:none;stroke:#d1d5db}.bar-bg{fill:#e5e7eb}
        </style>""",
        '<rect class="bg" width="1320" height="430"/>',
        '<text class="title" x="32" y="34">同一检查点的门控干预与专家选择质量</text>',
        '<text class="subtitle" x="32" y="56">点为完整 5,000 图 New17 AP 点差；线为 1,000 次配对图像重采样的逐比较 95% 区间</text>',
        '<rect class="frame" x="28" y="76" width="1264" height="305" rx="8"/>',
        _forest_panel(gate_rows, x=185, y=105, width=260, title="(a) 固定编号后替换 gate", low=-0.18, high=0.08),
        _forest_panel(selection_rows, x=620, y=105, width=300, title="(b) 固定/翻转专家选择", low=-3.2, high=3.2),
        '<text class="panel-title" x="1020" y="105">(c) 原生专家使用</text>',
    ]
    for seed in SEEDS:
        row = conditions[(seed, "T")]
        e0 = int(row["expert0_count"])
        share = e0 / 5000
        yy = 145 + seed * 62
        parts.extend(
            [
                f'<text class="row-label" x="1012" y="{yy + 13}" text-anchor="end">S{seed}</text>',
                f'<rect class="bar-bg" x="1020" y="{yy}" width="230" height="20" rx="3"/>',
                f'<rect x="1020" y="{yy}" width="{230 * share:.1f}" height="20" rx="3" fill="{PALETTE[seed]}"/>',
                f'<text class="row-label" x="1135" y="{yy + 15}" text-anchor="middle">E0 {e0} / E1 {5000 - e0}</text>',
            ]
        )
    parts.extend(
        [
            '<text class="note" x="32" y="410">GZ/GW：保留原生专家编号，分别使用零文本/错类文本 gate；FT：按逐图原生编号的相反专家组装完整预测。</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def render_opportunity_svg(opportunities: dict[int, dict[str, str]]) -> str:
    parts = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="1080" height="410" viewBox="0 0 1080 410" role="img" aria-labelledby="title desc">',
        '<title id="title">Local expert opportunity and target-metric outcome</title>',
        '<desc id="desc">Per-image diagnostic preference counts and New17 AP difference from the best fixed expert.</desc>',
        """<style>
          .bg{fill:#fff}.title{font:700 22px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}
          .subtitle,.note{font:12px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#4b5563}
          .panel-title{font:700 15px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#111827}
          .label{font:12px -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;fill:#374151}.frame{fill:none;stroke:#d1d5db}
          .zero{stroke:#111827;stroke-width:1;stroke-dasharray:3 3}
        </style>""",
        '<rect class="bg" width="1080" height="410"/>',
        '<text class="title" x="32" y="34">局部专家机会与目标指标收益</text>',
        '<text class="subtitle" x="32" y="56">逐图诊断选择器用于检验机会，不是可部署路由器，也不构成 AP 上界</text>',
        '<rect class="frame" x="28" y="76" width="1024" height="270" rx="8"/>',
        '<text class="panel-title" x="60" y="108">(a) 逐图诊断偏好计数</text>',
        '<text class="panel-title" x="650" y="108">(b) 相对最佳固定专家</text>',
    ]
    count_scale = 450 / 5000
    for seed in SEEDS:
        row = opportunities[seed]
        yy = 140 + seed * 60
        x = 100.0
        parts.append(f'<text class="label" x="92" y="{yy + 14}" text-anchor="end">S{seed}</text>')
        for key, color in (("images_E0_better", "#2563eb"), ("images_E1_better", "#ea580c"), ("images_tie", "#d1d5db")):
            value = int(row[key])
            width = value * count_scale
            parts.append(f'<rect x="{x:.1f}" y="{yy}" width="{width:.1f}" height="20" fill="{color}"/>')
            x += width
        parts.append(
            f'<text class="label" x="325" y="{yy + 42}" text-anchor="middle">机会 {row["opportunity_count"]} · 改选 {row["nontrivial_action_count"]}</text>'
        )
    low, high, left, width = -0.5, 0.5, 680.0, 300.0
    zero = left + (0 - low) * width / (high - low)
    parts.append(f'<line class="zero" x1="{zero:.1f}" y1="125" x2="{zero:.1f}" y2="310"/>')
    for seed in SEEDS:
        row = opportunities[seed]
        delta = 100 * _number(row, "oracle_minus_best_fixed_new17_AP")
        xx = left + (delta - low) * width / (high - low)
        yy = 155 + seed * 60
        parts.extend(
            [
                f'<text class="label" x="672" y="{yy + 4}" text-anchor="end">S{seed}</text>',
                f'<circle cx="{xx:.1f}" cy="{yy}" r="7" fill="{PALETTE[seed]}"/>',
                f'<text class="label" x="{xx + (12 if delta < 0.42 else -12):.1f}" y="{yy + 4}" text-anchor="{"start" if delta < 0.42 else "end"}">{delta:+.3f} AP</text>',
            ]
        )
    parts.extend(
        [
            '<text class="label" x="680" y="330">−0.5</text><text class="label" x="830" y="330" text-anchor="middle">0</text><text class="label" x="980" y="330" text-anchor="end">+0.5 AP 点</text>',
            '<text class="note" x="32" y="382">计数将 5,000 张图分为 E0 更优、E1 更优和平局；右侧点差使用完整预测的 COCO New17 AP。</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path("reports/text_router_analysis/controlled_interventions")
    parser.add_argument("--conditions", type=Path, default=default_root / "conditions.csv")
    parser.add_argument("--contrasts", type=Path, default=default_root / "paired_bootstrap.csv")
    parser.add_argument("--opportunities", type=Path, default=default_root / "opportunity_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=default_root / "figures")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    conditions = validate_conditions(read_rows(args.conditions))
    contrasts = validate_contrasts(read_rows(args.contrasts), conditions)
    opportunities = validate_opportunities(read_rows(args.opportunities))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "fig5_controlled_interventions.svg").write_text(
        render_controlled_svg(conditions, contrasts) + "\n", encoding="utf-8"
    )
    (args.output_dir / "fig6_local_opportunity.svg").write_text(
        render_opportunity_svg(opportunities) + "\n", encoding="utf-8"
    )
    print(f"validated 24 conditions and wrote 2 SVG files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
