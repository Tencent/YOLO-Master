from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "examples" / "mot_cross_domain_audit" / "results"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def keyed(rows: list[dict[str, str]], key: str) -> dict[str, dict[str, str]]:
    return {row[key]: row for row in rows}


def test_model_summary_matches_training_validation_and_latency_evidence() -> None:
    summary = keyed(read_csv(RESULTS / "model_comparison_v2.csv"), "key")
    validation = keyed(read_csv(RESULTS / "benchmark_v2" / "post_merge_validation.csv"), "key")
    latency = read_csv(RESULTS / "benchmark_v2" / "latency_rounds.csv")
    training_files = {
        "v10": "v10_results.csv",
        "v10_mot": "v10_mot_results.csv",
        "v10_moa": "v10_moa_results.csv",
        "v10_mot_p5": "v10_mot_p5_results.csv",
    }

    assert set(summary) == set(validation) == set(training_files)
    for model_key, row in summary.items():
        training = read_csv(RESULTS / "training" / training_files[model_key])
        assert len(training) == 30
        numeric_values = [float(value) for epoch in training for name, value in epoch.items() if name != "epoch"]
        assert all(math.isfinite(value) for value in numeric_values)
        selected = max(training, key=lambda epoch: float(epoch["metrics/mAP50-95(B)"]))
        assert selected["epoch"] == "30"
        assert float(row["training_map50"]) == pytest.approx(float(selected["metrics/mAP50(B)"]))
        assert float(row["training_map50_95"]) == pytest.approx(float(selected["metrics/mAP50-95(B)"]))

        current = validation[model_key]
        assert float(row["current_map50"]) == pytest.approx(float(current["map50"]))
        assert float(row["current_map50_95"]) == pytest.approx(float(current["map50_95"]))
        assert row["weights_sha256"] == current["weights_sha256"]

        rounds = [item for item in latency if item["key"] == model_key]
        assert len(rounds) == int(row["benchmark_rounds"]) == 3
        for percentile in ("p50", "p95", "p99"):
            values = [float(item[f"{percentile}_ms"]) for item in rounds]
            assert float(row[f"latency_ms_{percentile}"]) == pytest.approx(statistics.median(values))
        p50_values = [float(item["p50_ms"]) for item in rounds]
        assert float(row["p50_run_min"]) == pytest.approx(min(p50_values))
        assert float(row["p50_run_max"]) == pytest.approx(max(p50_values))
        assert {item["weights_sha256"] for item in rounds} == {row["weights_sha256"]}


def test_adaptive_k_summary_matches_raw_rounds_and_detection_metrics() -> None:
    directory = RESULTS / "utility_routing" / "adaptive_k"
    summary = keyed(read_csv(directory / "adaptive_k_benchmark.csv"), "variant")
    detection = keyed(read_csv(directory / "detection_metrics.csv"), "variant")
    latency = read_csv(directory / "latency_rounds.csv")
    baseline_calls = float(summary["baseline_fixed_k"]["mean_expert_sample_calls"])

    assert set(summary) == set(detection)
    for variant, row in summary.items():
        rounds = [item for item in latency if item["variant"] == variant]
        assert len(rounds) == 3
        for percentile in ("p50", "p95", "p99"):
            values = [float(item[f"latency_{percentile}_ms"]) for item in rounds]
            assert float(row[f"latency_{percentile}_ms"]) == pytest.approx(statistics.median(values))
            assert float(row[f"latency_{percentile}_ms_run_min"]) == pytest.approx(min(values))
            assert float(row[f"latency_{percentile}_ms_run_max"]) == pytest.approx(max(values))
        metric = detection[variant]
        assert float(row["metrics/mAP50-95(B)"]) == pytest.approx(float(metric["metrics/mAP50-95(B)"]))
        calls = float(row["mean_expert_sample_calls"])
        expected_saving = (baseline_calls - calls) / baseline_calls
        assert float(row["expert_sample_saving_vs_baseline_fixed_dispatch"]) == (pytest.approx(expected_saving))


def test_utility_guard_reports_calibration_gain_and_test_fallback() -> None:
    directory = RESULTS / "utility_routing" / "utility_router"
    validation = json.loads((directory / "val" / "evaluation_report.json").read_text(encoding="utf-8"))
    test = json.loads((directory / "test" / "evaluation_report.json").read_text(encoding="utf-8"))

    assert validation["drift_guard_triggered"] is False
    assert validation["effective_blend_alpha"] == pytest.approx(0.4)
    baseline_regret = validation["baseline"]["mean_regret"]
    deployed_regret = validation["deployment_router"]["mean_regret"]
    assert deployed_regret < baseline_regret
    assert validation["relative_regret_reduction"] == pytest.approx(
        (baseline_regret - deployed_regret) / baseline_regret
    )

    assert test["drift_guard_triggered"] is True
    assert test["effective_blend_alpha"] == pytest.approx(0.0)
    assert test["deployment_router"] == test["baseline"]
    assert test["mean_utility_to_baseline_kl"] > test["max_mean_router_kl"]


def test_public_result_text_is_desensitized() -> None:
    forbidden = ("/home/", "\\Users\\", "sk-")
    text_suffixes = {".csv", ".json", ".md", ".txt"}
    offenders = []
    for path in RESULTS.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == []
