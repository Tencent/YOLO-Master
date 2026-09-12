"""Contracts for the preregistered Foundation distillation decision gate."""

import json

import pytest

from scripts.foundation_distill_decision import analyze_report, main, render_markdown


def _report(deltas, *, contaminate=False, categories=False):
    records, plan = [], []
    for seed, delta_points in enumerate(deltas):
        baseline_metrics = {
            "metrics/mAP50-95(B)": 0.30,
            "metrics/mAP_small(B)": 0.10,
            "metrics/mAP_medium(B)": 0.20,
            "metrics/mAP_large(B)": 0.40,
        }
        treatment_metrics = {
            "metrics/mAP50-95(B)": 0.30 + delta_points / 100,
            "metrics/mAP_small(B)": 0.10 + (delta_points + 0.2) / 100,
            "metrics/mAP_medium(B)": 0.20 + delta_points / 100,
            "metrics/mAP_large(B)": 0.40 + (delta_points - 0.2) / 100,
            "train/foundation_task_ratio": 0.02,
        }
        baseline = {"seed": seed, "arm": "B0", "observed": baseline_metrics}
        treatment = {"seed": seed, "arm": "D2", "observed": treatment_metrics}
        if categories:
            baseline["per_class_ap"] = {"person": 0.20, "car": 0.30}
            treatment["per_class_ap"] = {"person": 0.21, "car": 0.295}
        records.extend((baseline, treatment))
        common = {"epochs": 10, "imgsz": 256, "batch": 4, "seed": seed, "optimizer": "SGD"}
        baseline_overrides = {**common, "name": f"b0-s{seed}", "foundation_enabled": False}
        treatment_overrides = {
            **common,
            "name": f"d2-s{seed}",
            "foundation_enabled": True,
            "foundation_loss": "hybrid",
            "foundation_align_dim": 32,
        }
        if contaminate and seed == 1:
            treatment_overrides["batch"] = 8
        plan.extend(
            (
                {"seed": seed, "arm": "B0", "overrides": baseline_overrides},
                {"seed": seed, "arm": "D2", "overrides": treatment_overrides},
            )
        )
    return {"records": records, "plan": plan}


def test_preregistered_small_effect_with_zero_in_ci_is_no_go():
    result = analyze_report(_report([0.1, -0.1, 0.2]))
    assert result["decision"] == "NO-GO"
    assert result["criteria"] == {"small_effect": True, "ci_contains_zero": True}
    assert result["budget_audit"]["passed"] is True


def test_positive_effect_requires_margin_and_ci_above_zero():
    result = analyze_report(_report([0.5, 0.6, 0.7]))
    assert result["decision"] == "GO"
    assert result["primary"]["mean"] == pytest.approx(0.6)
    assert result["primary"]["low"] > 0


def test_exact_effect_margin_is_not_classified_as_negligible():
    result = analyze_report(_report([0.3, 0.3, 0.3]))
    assert result["decision"] == "GO"
    assert result["criteria"]["small_effect"] is False


def test_budget_contamination_blocks_decision():
    result = analyze_report(_report([0.5, 0.6, 0.7], contaminate=True))
    assert result["decision"] == "INSUFFICIENT"
    assert result["budget_audit"]["passed"] is False
    assert result["budget_audit"]["mismatches"][0]["fields"] == ["batch"]


def test_scale_and_category_analysis_are_paired():
    result = analyze_report(_report([0.5, 0.6, 0.7], categories=True))
    assert result["scale_analysis"]["small"]["mean"] == pytest.approx(0.8)
    assert result["scale_analysis"]["large"]["mean"] == pytest.approx(0.4)
    assert result["category_analysis"][0]["category"] == "person"
    assert result["category_analysis"][0]["mean"] == pytest.approx(1.0)
    assert result["p2"]["path"] == "positive_benefit_analysis"
    assert result["benefit_summary"]["confirmed_positive_scales"] == ["small", "medium", "large"]
    assert result["benefit_summary"]["confirmed_positive_categories"] == ["person"]
    assert result["p2"]["formal_recommendation"]["category_claim_allowed"] is True
    assert "Foundation 蒸馏 Go/No-Go 建议书" in render_markdown(result, "report.json")


def test_no_go_emits_four_evidence_linked_diagnostic_areas():
    result = analyze_report(_report([0.1, -0.1, 0.2]))
    diagnostics = result["p2"]["diagnostics"]
    assert result["p2"]["path"] == "negative_or_uncertain_diagnosis"
    assert {item["area"] for item in diagnostics} == {"capacity", "dimension", "optimization", "data"}
    assert result["p2"]["formal_recommendation"]["action"] == "stop_default_distillation_path"
    assert result["p2"]["formal_recommendation"]["accuracy_claim_allowed"] is False
    assert result["p2"]["formal_recommendation"]["category_claim_allowed"] is False


def test_missing_category_metrics_prohibit_category_claim_even_for_go():
    result = analyze_report(_report([0.5, 0.6, 0.7]))
    assert result["decision"] == "GO"
    assert result["benefit_summary"]["category_evidence_available"] is False
    assert result["p2"]["formal_recommendation"]["category_claim_allowed"] is False


def test_cli_writes_json_and_formal_recommendation(tmp_path):
    source = tmp_path / "report.json"
    output_json = tmp_path / "decision.json"
    output_md = tmp_path / "recommendation.md"
    source.write_text(json.dumps(_report([0.1, -0.1, 0.2])), encoding="utf-8")
    main(["--input", str(source), "--output-json", str(output_json), "--output-md", str(output_md)])
    assert json.loads(output_json.read_text(encoding="utf-8"))["decision"] == "NO-GO"
    assert "不得声称具体类别受益" in output_md.read_text(encoding="utf-8")
