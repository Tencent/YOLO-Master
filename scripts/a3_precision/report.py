"""Aggregate machine-readable and Markdown evidence for the precision matrix."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from .common import json_dump
from .manifest import HarnessConfig


def _read(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _fmt(value: Any, digits: int = 4) -> str:
    return "—" if not isinstance(value, (int, float)) else f"{value:.{digits}f}"


def build_report(config: HarnessConfig, failures: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Build the final four-way comparison and all-family failure inventory."""
    failures = list(failures or [])
    operator_failures = []
    families = {}
    rows = []
    for spec in config.models:
        family_dir = config.output_dir / spec.family
        for evidence_path in family_dir.rglob("*.quantization.json"):
            evidence = _read(evidence_path)
            if evidence and evidence.get("status") == "failed":
                handled_probe = "sensitivity_probes" in evidence_path.parts
                operator_failures.append(
                    {
                        "family": spec.family,
                        "stage": evidence.get("label", "quantization"),
                        "evidence": str(evidence_path),
                        "handled_by_fallback": handled_probe,
                        "error_type": evidence.get("error_type"),
                        "error": evidence.get("error"),
                        "failed_operators": evidence.get("failed_operators", []),
                    }
                )
        comparison = _read(family_dir / "comparison.final.json")
        sensitivity = _read(family_dir / "sensitivity.json")
        route_drift = _read(family_dir / "route_drift.int8_per_layer.json")
        families[spec.family] = {
            "comparison": comparison,
            "sensitivity": sensitivity,
            "full_int8_onnx_route_drift": route_drift,
        }
        if comparison is None:
            failures.append({"family": spec.family, "stage": "report", "error": "final comparison missing"})
            continue
        for variant, result in comparison.items():
            accuracy = result.get("accuracy_result", {}).get("accuracy", {})
            latency = result.get("latency_result", {}).get("latency_ms", {})
            assignment = result.get("latency_result", {}).get("provider_assignment", {}).get("providers", {})
            failed_ops = result.get("failed_operators", [])
            if result.get("status") != "success":
                accuracy_error = result.get("accuracy_result", {}).get("error")
                latency_error = result.get("latency_result", {}).get("error")
                failures.append(
                    {
                        "family": spec.family,
                        "stage": f"final_validation::{variant}",
                        "error": accuracy_error or latency_error or result.get("error") or "validation failed",
                        "failed_operators": failed_ops,
                    }
                )
            rows.append(
                {
                    "family": spec.family,
                    "variant": variant,
                    "status": result.get("status"),
                    "map50": accuracy.get("map50"),
                    "map50_95": accuracy.get("map50_95"),
                    "map_loss_pp": result.get("map50_95_loss_percentage_points"),
                    "latency_median_ms": latency.get("median"),
                    "latency_p95_ms": latency.get("p95"),
                    "latency_change_pct": result.get("latency_change_pct"),
                    "provider_assignment": json.dumps(assignment, ensure_ascii=False, sort_keys=True),
                    "cpu_executed_nodes": assignment.get("CPUExecutionProvider"),
                    "size_mb": result.get("size_mb"),
                    "size_reduction_pct": result.get("size_reduction_pct"),
                    "failed_operators": ",".join(failed_ops),
                    "sensitive_groups": len(sensitivity.get("selected_groups", [])) if sensitivity else None,
                }
            )
    stage0 = _read(config.output_dir / "stage0_trained_mot_drift" / "trained_mot_routing_drift.json")
    payload = {
        "schema_version": 1,
        "status": (
            "failed"
            if failures or any(not failure.get("handled_by_fallback", False) for failure in operator_failures)
            else "success"
        ),
        "claim_boundary": (
            "ONNX comparisons use masked-dense graphs. They measure numerical precision behavior, "
            "not true conditional expert execution; dynamic runtime evidence is maintained separately."
        ),
        "stage0_trained_mot_routing_drift": stage0,
        "families": families,
        "table": rows,
        "operator_failures": operator_failures,
        "failures": failures,
    }
    json_dump(config.output_dir / "five_family_precision_summary.json", payload)
    fieldnames = list(rows[0]) if rows else ["family", "variant", "status"]
    with (config.output_dir / "five_family_precision_table.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# A3 五族精度、路由漂移与自动回退报告",
        "",
        f"状态：**{payload['status']}**",
        "",
        "> 当前 ONNX 图用于 masked-dense 数值评估，不代表真正的条件专家执行；动态运行时在另一任务中独立验收。",
        "",
        "## 第 0 阶段：真实训练 MoT 路由漂移",
        "",
    ]
    if stage0:
        lines.extend(
            [
                f"- checkpoint：`{stage0['model']['path']}`",
                f"- SHA-256：`{stage0['model']['sha256']}`",
                f"- 真实训练元数据门禁：`{stage0['model'].get('trained_checkpoint_gate', False)}`",
                f"- Router 非退化门禁：`{stage0['router_gate']['passed']}`",
                f"- 图像数：`{stage0['dataset']['images']}`",
            ]
        )
    else:
        lines.append("- 缺少第 0 阶段证据。")
    lines.extend(
        [
            "",
            "## 四方案与 FP16 对比",
            "",
            "| 族 | 方案 | 状态 | mAP50-95 | 损失/百分点 | 中位时延/ms | CPU 执行节点 | 体积/MB | 体积下降 |",
            "|---|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['family']} | {row['variant']} | {row['status']} | {_fmt(row['map50_95'], 6)} | "
            f"{_fmt(row['map_loss_pp'], 4)} | {_fmt(row['latency_median_ms'], 3)} | "
            f"{_fmt(row['cpu_executed_nodes'], 0)} | "
            f"{_fmt(row['size_mb'], 3)} | {_fmt(row['size_reduction_pct'], 2)}% |"
        )
    lines.extend(["", "## 自动敏感层与相关性", ""])
    for family, info in families.items():
        sensitivity = info["sensitivity"]
        if not sensitivity:
            lines.append(f"- `{family}`：缺少 sensitivity.json。")
            continue
        correlation = sensitivity.get("correlation", {})
        route_drift = info.get("full_int8_onnx_route_drift") or {}
        route_layers = list(route_drift.get("layers", {}).values())
        route_summary = (
            f"ONNX层数={len(route_layers)}，最差有序Top-K={_fmt(min(row['topk_ordered_exact_rate'] for row in route_layers))}，"
            f"最差集合Top-K={_fmt(min(row['topk_exact_rate'] for row in route_layers))}，"
            f"最低Jaccard={_fmt(min(row['jaccard_rate'] for row in route_layers))}，"
            f"最大token翻转={_fmt(max(row['token_flip_rate'] for row in route_layers))}；"
            if route_layers
            else "缺少真实ONNX逐层路由；"
        )
        lines.append(
            f"- `{family}`：选择 {len(sensitivity.get('selected_groups', []))} 组；"
            f"{route_summary}漂移-mAP Pearson={_fmt(correlation.get('pearson'))}，"
            f"Spearman={_fmt(correlation.get('spearman'))}。"
        )
    lines.extend(["", "## 失败算子/阶段", ""])
    all_failures = [*failures, *operator_failures]
    if all_failures:
        for failure in all_failures:
            operators = ",".join(failure.get("failed_operators", [])) or "未解析"
            handling = "（已由保守回退处理）" if failure.get("handled_by_fallback") else ""
            lines.append(
                f"- `{failure.get('family', 'global')}` / `{failure.get('stage', 'unknown')}` "
                f"(算子: {operators}){handling}：{failure.get('error')}"
            )
    else:
        lines.append("- 无失败。")
    (config.output_dir / "five_family_precision_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload
