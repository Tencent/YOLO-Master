#!/usr/bin/env python3
"""Turn paired Foundation on/off runs into a preregistered go/no-go recommendation.

The input is the JSON report emitted by ``foundation_f08_effect_gate.py``. Metrics are
normally stored on a 0..1 scale; decisions are reported in mAP percentage points.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

SCHEMA_VERSION = 1
DEFAULT_METRIC = "metrics/mAP50-95(B)"
SCALE_METRICS = {
    "small": "metrics/mAP_small(B)",
    "medium": "metrics/mAP_medium(B)",
    "large": "metrics/mAP_large(B)",
}
_T95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def _finite(value: Any) -> float | None:
    """Return a finite float, otherwise ``None``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _metrics(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve the metric container used by supported experiment reports."""
    for key in ("observed", "validation_metrics", "metrics"):
        value = record.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _record_arm(record: Mapping[str, Any]) -> str | None:
    """Resolve an explicit arm or the generic baseline/Foundation branch name."""
    if record.get("arm") is not None:
        return str(record["arm"])
    if isinstance(record.get("foundation"), bool):
        return "foundation" if record["foundation"] else "baseline"
    return None


def _plan_arm(spec: Mapping[str, Any]) -> str | None:
    """Resolve a plan arm using the same rules as result records."""
    return _record_arm(spec)


def _metric_multiplier(records: list[Mapping[str, Any]], metric: str, metric_scale: str) -> float:
    """Convert repository-native 0..1 AP values into percentage points."""
    if metric_scale == "fraction":
        return 100.0
    if metric_scale == "points":
        return 1.0
    values = [_finite(_metrics(record).get(metric)) for record in records]
    finite = [abs(value) for value in values if value is not None]
    return 100.0 if finite and max(finite) <= 1.5 else 1.0


def _paired_records(
    payload: Mapping[str, Any], baseline_arm: str, treatment_arm: str
) -> tuple[list[tuple[int, Mapping[str, Any], Mapping[str, Any]]], list[dict[str, Any]]]:
    """Pair baseline and treatment records by seed without imputing missing runs."""
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("report must contain a records list")
    by_key: dict[tuple[int, str], Mapping[str, Any]] = {}
    issues: list[dict[str, Any]] = []
    for record in records:
        if not isinstance(record, Mapping) or _record_arm(record) not in {baseline_arm, treatment_arm}:
            continue
        try:
            key = (int(record["seed"]), str(_record_arm(record)))
        except (KeyError, TypeError, ValueError):
            issues.append({"kind": "invalid_record", "record": dict(record) if isinstance(record, Mapping) else record})
            continue
        if key in by_key:
            raise ValueError(f"duplicate record for seed={key[0]} arm={key[1]}")
        by_key[key] = record
    seeds = sorted({seed for seed, _ in by_key})
    pairs = []
    for seed in seeds:
        baseline = by_key.get((seed, baseline_arm))
        treatment = by_key.get((seed, treatment_arm))
        if baseline is None or treatment is None:
            issues.append(
                {
                    "kind": "incomplete_pair",
                    "seed": seed,
                    "baseline_present": baseline is not None,
                    "treatment_present": treatment is not None,
                }
            )
            continue
        pairs.append((seed, baseline, treatment))
    return pairs, issues


def _clean_overrides(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only the preregistered treatment and run-identity fields."""
    overrides = spec.get("overrides")
    if not isinstance(overrides, Mapping):
        return {}
    return {
        str(key): value
        for key, value in overrides.items()
        if not str(key).startswith("foundation_") and key not in {"name", "resume"}
    }


def audit_equal_budget(payload: Mapping[str, Any], baseline_arm: str, treatment_arm: str) -> dict[str, Any]:
    """Audit that paired plan arms differ only in Foundation-specific fields."""
    plan = payload.get("plan")
    if not isinstance(plan, list):
        return {"available": False, "passed": False, "mismatches": ["missing plan"]}
    by_key = {}
    for spec in plan:
        if not isinstance(spec, Mapping) or _plan_arm(spec) not in {baseline_arm, treatment_arm}:
            continue
        try:
            by_key[(int(spec["seed"]), str(_plan_arm(spec)))] = spec
        except (KeyError, TypeError, ValueError):
            continue
    seeds = sorted({seed for seed, _ in by_key})
    mismatches = []
    checked = 0
    for seed in seeds:
        baseline = by_key.get((seed, baseline_arm))
        treatment = by_key.get((seed, treatment_arm))
        if baseline is None or treatment is None:
            mismatches.append({"seed": seed, "fields": ["missing paired plan arm"]})
            continue
        checked += 1
        left, right = _clean_overrides(baseline), _clean_overrides(treatment)
        fields = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        if fields:
            mismatches.append(
                {
                    "seed": seed,
                    "fields": fields,
                    "baseline": {key: left.get(key) for key in fields},
                    "treatment": {key: right.get(key) for key in fields},
                }
            )
    return {
        "available": True,
        "passed": checked > 0 and not mismatches,
        "paired_specs_checked": checked,
        "mismatches": mismatches,
    }


def paired_interval(values: list[float], *, confidence: float = 0.95) -> dict[str, Any]:
    """Calculate a two-sided Student-t interval for paired deltas."""
    if not values:
        return {"n": 0, "mean": None, "sample_std": None, "low": None, "high": None, "confidence": confidence}
    mean = statistics.fmean(values)
    if len(values) == 1:
        return {"n": 1, "mean": mean, "sample_std": None, "low": None, "high": None, "confidence": confidence}
    sample_std = statistics.stdev(values)
    if confidence != 0.95:
        raise ValueError("only the preregistered 95% confidence interval is supported")
    critical = _T95.get(len(values) - 1, 1.96)
    half_width = critical * sample_std / math.sqrt(len(values))
    return {
        "n": len(values),
        "mean": mean,
        "sample_std": sample_std,
        "low": mean - half_width,
        "high": mean + half_width,
        "confidence": confidence,
    }


def _summarize_metric(
    pairs: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]], metric: str, multiplier: float
) -> dict[str, Any]:
    """Summarize complete paired deltas for one metric in mAP points."""
    observations = []
    for seed, baseline, treatment in pairs:
        baseline_value = _finite(_metrics(baseline).get(metric))
        treatment_value = _finite(_metrics(treatment).get(metric))
        if baseline_value is None or treatment_value is None:
            continue
        observations.append(
            {
                "seed": seed,
                "baseline": baseline_value * multiplier,
                "treatment": treatment_value * multiplier,
                "delta": (treatment_value - baseline_value) * multiplier,
            }
        )
    interval = paired_interval([item["delta"] for item in observations])
    return {"metric": metric, "unit": "mAP_points", "observations": observations, **interval}


def _class_ap(record: Mapping[str, Any]) -> Mapping[str, Any]:
    """Resolve optional per-category AP emitted by an evaluator extension."""
    for container in (record, _metrics(record)):
        for key in ("per_class_ap", "per_category_ap", "category_ap"):
            value = container.get(key) if isinstance(container, Mapping) else None
            if isinstance(value, Mapping):
                return value
    return {}


def _category_summary(
    pairs: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]], multiplier: float
) -> list[dict[str, Any]]:
    """Aggregate optional paired per-category AP deltas."""
    values: dict[str, list[float]] = defaultdict(list)
    for _, baseline, treatment in pairs:
        baseline_ap, treatment_ap = _class_ap(baseline), _class_ap(treatment)
        for name in sorted(set(baseline_ap) & set(treatment_ap)):
            left, right = _finite(baseline_ap[name]), _finite(treatment_ap[name])
            if left is not None and right is not None:
                values[str(name)].append((right - left) * multiplier)
    result = [{"category": name, "unit": "mAP_points", **paired_interval(deltas)} for name, deltas in values.items()]
    return sorted(result, key=lambda item: float(item["mean"]), reverse=True)


def _benefit_summary(scales: Mapping[str, Mapping[str, Any]], categories: list[Mapping[str, Any]]) -> dict[str, Any]:
    """Separate observed positive deltas from confidence-supported benefits."""

    def positive(item: Mapping[str, Any]) -> bool:
        return item.get("mean") is not None and float(item["mean"]) > 0

    def confirmed(item: Mapping[str, Any]) -> bool:
        return int(item.get("n", 0)) >= 3 and item.get("low") is not None and float(item["low"]) > 0

    ranked_scales = sorted(
        ({"scale": name, **dict(item)} for name, item in scales.items() if item.get("mean") is not None),
        key=lambda item: float(item["mean"]),
        reverse=True,
    )
    ranked_categories = [dict(item) for item in categories]
    return {
        "scale_evidence_available": bool(ranked_scales),
        "observed_positive_scales": [item["scale"] for item in ranked_scales if positive(item)],
        "confirmed_positive_scales": [item["scale"] for item in ranked_scales if confirmed(item)],
        "ranked_scales": ranked_scales,
        "category_evidence_available": bool(ranked_categories),
        "category_claim_allowed": bool(ranked_categories) and all(int(item.get("n", 0)) >= 3 for item in categories),
        "observed_positive_categories": [item["category"] for item in ranked_categories if positive(item)],
        "confirmed_positive_categories": [item["category"] for item in ranked_categories if confirmed(item)],
        "ranked_categories": ranked_categories,
    }


def _treatment_metric_mean(pairs: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]], metric: str) -> float | None:
    values = [_finite(_metrics(treatment).get(metric)) for _, _, treatment in pairs]
    finite = [value for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def _diagnostics(
    decision: str,
    primary: Mapping[str, Any],
    scales: Mapping[str, Mapping[str, Any]],
    pairs: list[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
    payload: Mapping[str, Any],
    treatment_arm: str,
) -> list[dict[str, Any]]:
    """Produce evidence-linked P2 diagnoses without pretending missing data are measurements."""
    diagnostics = []
    width = None
    if primary.get("low") is not None and primary.get("high") is not None:
        width = float(primary["high"]) - float(primary["low"])
    diagnostics.append(
        {
            "area": "data",
            "status": "risk" if int(primary["n"]) < 5 or (width is not None and width > 0.6) else "checked",
            "evidence": (
                f"{primary['n']} paired seeds; 95% CI width={width:.4f} mAP points"
                if width is not None
                else f"{primary['n']} paired seeds; CI unavailable"
            ),
            "next_action": (
                "Add paired seeds with the same frozen budget and decision line; do not retune the threshold."
            ),
        }
    )
    task_ratio = _treatment_metric_mean(pairs, "train/foundation_task_ratio")
    if task_ratio is None:
        optimization_status, optimization_evidence = "open", "foundation/task loss ratio was not reported"
    elif task_ratio > 0.3:
        optimization_status, optimization_evidence = "risk", f"mean Foundation/task loss ratio={task_ratio:.4f} (>0.3)"
    elif task_ratio < 1e-4:
        optimization_status, optimization_evidence = "risk", f"mean Foundation/task loss ratio={task_ratio:.6f} (<1e-4)"
    else:
        optimization_status, optimization_evidence = "checked", f"mean Foundation/task loss ratio={task_ratio:.4f}"
    diagnostics.append(
        {
            "area": "optimization",
            "status": optimization_status,
            "evidence": optimization_evidence,
            "next_action": "Inspect task ratio and raw cosine/relational curves before changing only the loss weight.",
        }
    )
    plan = payload.get("plan") or []
    treatment_specs = [item for item in plan if isinstance(item, Mapping) and _plan_arm(item) == treatment_arm]
    dims = sorted(
        {
            int(item.get("overrides", {}).get("foundation_align_dim"))
            for item in treatment_specs
            if _finite(item.get("overrides", {}).get("foundation_align_dim")) is not None
        }
    )
    diagnostics.append(
        {
            "area": "dimension",
            "status": "open",
            "evidence": f"tested align_dim={dims}" if dims else "alignment dimension metadata unavailable",
            "next_action": (
                "If no-go persists, compare one smaller and one larger align_dim under the identical paired budget."
            ),
        }
    )
    worst_scale = min(
        ((name, item.get("mean")) for name, item in scales.items() if item.get("mean") is not None),
        key=lambda item: float(item[1]),
        default=None,
    )
    diagnostics.append(
        {
            "area": "capacity",
            "status": "open" if decision != "GO" else "checked",
            "evidence": (
                f"weakest scale={worst_scale[0]} ({float(worst_scale[1]):+.4f} points)"
                if worst_scale
                else "scale AP unavailable"
            ),
            "next_action": (
                "Run the same recipe on the next student size only if scale evidence suggests a backbone capacity "
                "bottleneck."
            ),
        }
    )
    return diagnostics


def analyze_report(
    payload: Mapping[str, Any],
    *,
    baseline_arm: str = "B0",
    treatment_arm: str = "D2",
    metric: str = DEFAULT_METRIC,
    min_effect_points: float = 0.3,
    metric_scale: str = "auto",
) -> dict[str, Any]:
    """Analyze one paired report using the preregistered P1 decision rule."""
    if min_effect_points <= 0 or not math.isfinite(min_effect_points):
        raise ValueError("min_effect_points must be finite and positive")
    if metric_scale not in {"auto", "fraction", "points"}:
        raise ValueError("metric_scale must be auto, fraction, or points")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError("report must contain a records list")
    pairs, issues = _paired_records(payload, baseline_arm, treatment_arm)
    multiplier = _metric_multiplier(records, metric, metric_scale)
    primary = _summarize_metric(pairs, metric, multiplier)
    budget = audit_equal_budget(payload, baseline_arm, treatment_arm)
    ci_contains_zero = (
        primary["low"] is not None and primary["high"] is not None and primary["low"] <= 0 <= primary["high"]
    )
    small_effect = primary["mean"] is not None and abs(primary["mean"]) < min_effect_points
    if not budget["passed"] or primary["n"] < 3:
        decision, reason = "INSUFFICIENT", "need at least 3 complete paired seeds and a clean same-budget audit"
    elif primary["high"] < 0:
        decision, reason = "NO-GO", "paired confidence interval is entirely negative"
    elif small_effect and ci_contains_zero:
        decision, reason = "NO-GO", "preregistered negligible-effect rule matched"
    elif primary["mean"] >= min_effect_points and primary["low"] > 0:
        decision, reason = "GO", "effect exceeds 0.3 mAP points and the paired interval excludes zero"
    else:
        decision, reason = "INCONCLUSIVE", "add paired seeds without changing the preregistered decision line"
    scales = {name: _summarize_metric(pairs, key, multiplier) for name, key in SCALE_METRICS.items()}
    categories = _category_summary(pairs, multiplier)
    benefits = _benefit_summary(scales, categories)
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis": "foundation_distillation_go_no_go",
        "baseline_arm": baseline_arm,
        "treatment_arm": treatment_arm,
        "metric_scale": metric_scale,
        "metric_multiplier_to_points": multiplier,
        "preregistered_rule": {
            "min_effect_points": min_effect_points,
            "no_go": "abs(mean_delta_points) < min_effect_points and 95% paired CI contains 0",
            "minimum_paired_seeds": 3,
        },
        "budget_audit": budget,
        "pair_issues": issues,
        "primary": primary,
        "decision": decision,
        "reason": reason,
        "criteria": {"small_effect": small_effect, "ci_contains_zero": ci_contains_zero},
        "scale_analysis": scales,
        "category_analysis": categories,
        "benefit_summary": benefits,
    }
    diagnostics = _diagnostics(decision, primary, scales, pairs, payload, treatment_arm)
    result["diagnostics"] = diagnostics
    if decision == "GO":
        action = "advance_to_confirmation"
        recommendation = "Proceed to scale/category confirmation; keep the student deployment contract unchanged."
        required_followups = [
            "Confirm reported scale benefits with the same paired seeds.",
            "Claim category benefits only where per-class AP is present and its paired interval excludes zero.",
        ]
    elif decision == "NO-GO":
        action = "stop_default_distillation_path"
        recommendation = (
            "Do not claim an accuracy gain or enable Foundation KD by default. Follow the evidence-linked diagnostics."
        )
        required_followups = [item["next_action"] for item in diagnostics if item["status"] in {"risk", "open"}]
    else:
        action = "hold_and_collect_evidence"
        recommendation = (
            "Do not claim an accuracy gain. Resolve the budget/evidence gaps and add paired seeds without changing "
            "the preregistered threshold."
        )
        required_followups = [item["next_action"] for item in diagnostics if item["status"] in {"risk", "open"}]
    result["p2"] = {
        "path": "positive_benefit_analysis" if decision == "GO" else "negative_or_uncertain_diagnosis",
        "benefits": benefits,
        "diagnostics": [] if decision == "GO" else diagnostics,
        "formal_recommendation": {
            "decision": decision,
            "action": action,
            "accuracy_claim_allowed": decision == "GO",
            "category_claim_allowed": decision == "GO" and benefits["category_claim_allowed"],
            "recommendation": recommendation,
            "required_followups": required_followups,
        },
    }
    result["recommendation"] = recommendation
    return result


def render_markdown(result: Mapping[str, Any], source: str) -> str:
    """Render a formal, reviewable recommendation from a decision result."""
    primary = result["primary"]
    low = "n/a" if primary["low"] is None else f"{primary['low']:+.4f}"
    high = "n/a" if primary["high"] is None else f"{primary['high']:+.4f}"
    mean = "n/a" if primary["mean"] is None else f"{primary['mean']:+.4f}"
    lines = [
        "# Foundation 蒸馏 Go/No-Go 建议书",
        "",
        f"- 数据来源：`{source}`",
        f"- 对照：`{result['baseline_arm']}` vs `{result['treatment_arm']}`",
        f"- 结论：**{result['decision']}**",
        f"- 原因：{result['reason']}",
        "",
        "## P1 预注册判读",
        "",
        f"主指标为 `{primary['metric']}`；配对 seed 数为 {primary['n']}；平均差为 {mean} mAP 点；95% CI 为 [{low}, {high}]。",
        f"判读线固定为 `|ΔmAP| < {result['preregistered_rule']['min_effect_points']} 点` 且置信区间含 0 时 NO-GO。",
        f"同预算审计：{'通过' if result['budget_audit']['passed'] else '未通过'}。",
        "",
        "## P2 尺度与类别",
        "",
    ]
    for name, item in result["scale_analysis"].items():
        value = "n/a" if item["mean"] is None else f"{item['mean']:+.4f} mAP 点"
        lines.append(f"- {name}: {value}（n={item['n']}）")
    categories = result["category_analysis"]
    if categories:
        lines.extend(("", "类别收益前五："))
        for item in categories[:5]:
            lines.append(f"- {item['category']}: {item['mean']:+.4f} mAP 点（n={item['n']}）")
    else:
        lines.append("- 当前报告没有 per-class AP；不得声称具体类别受益。")
    benefits = result["benefit_summary"]
    if result["decision"] == "GO":
        lines.extend(("", "### 置信区间支持的受益项", ""))
        lines.append(f"- 尺度：{', '.join(benefits['confirmed_positive_scales']) or '无'}")
        lines.append(f"- 类别：{', '.join(benefits['confirmed_positive_categories'][:5]) or '无'}")
    else:
        lines.extend(("", "## 负结果/不确定结果诊断", ""))
        for item in result["diagnostics"]:
            lines.append(f"- **{item['area']} / {item['status']}**：{item['evidence']}。下一步：{item['next_action']}")
    formal = result["p2"]["formal_recommendation"]
    lines.extend(
        (
            "",
            "## 正式建议",
            "",
            f"行动：`{formal['action']}`。",
            str(formal["recommendation"]),
            "",
            "必要后续：",
        )
    )
    lines.extend(f"- {item}" for item in formal["required_followups"])
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    """Analyze a completed report and write JSON plus Markdown evidence."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--baseline-arm", default="B0")
    parser.add_argument("--treatment-arm", default="D2")
    parser.add_argument("--metric", default=DEFAULT_METRIC)
    parser.add_argument("--min-effect-points", type=float, default=0.3)
    parser.add_argument("--metric-scale", choices=("auto", "fraction", "points"), default="auto")
    args = parser.parse_args(argv)
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    result = analyze_report(
        payload,
        baseline_arm=args.baseline_arm,
        treatment_arm=args.treatment_arm,
        metric=args.metric,
        min_effect_points=args.min_effect_points,
        metric_scale=args.metric_scale,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result, str(args.input)), encoding="utf-8")
    print(json.dumps({"decision": result["decision"], "paired_seeds": result["primary"]["n"]}, sort_keys=True))


if __name__ == "__main__":
    main()
