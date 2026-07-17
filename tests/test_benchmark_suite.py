from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks.suite import (
    BenchmarkCaseResult,
    BenchmarkRunReport,
    BenchmarkSuiteRunner,
    case_fingerprint,
    load_suite,
    summarize_latency,
    write_reports,
)


def _write_catalog(path: Path, model_path: Path) -> None:
    path.write_text(
        f"""
schema_version: 1
defaults:
  task: inference
  device: cpu
  dtype: fp32
  batch_size: 1
  image_size: 32
  warmup: 1
  iterations: 3
  seed: 7
suites:
  smoke:
    description: test suite
    cases:
      - id: a
        label: A
        model: {model_path.as_posix()}
      - id: b
        label: B
        model: {model_path.as_posix()}
        iterations: 5
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _result(case_id: str, fingerprint: str, *, status: str = "ok") -> BenchmarkCaseResult:
    return BenchmarkCaseResult(
        case_id=case_id,
        label=case_id.upper(),
        model=f"{case_id}.yaml",
        fingerprint=fingerprint,
        status=status,
        error=None if status == "ok" else "boom",
        metrics={"params_total": 10, "latency_p50_ms": 2.0, "throughput_items_s": 500.0},
        routing={"collapsed_layers": 0},
        duration_s=0.1,
    )


def test_load_suite_merges_defaults_and_resolves_paths(tmp_path):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("nc: 1\n", encoding="utf-8")
    catalog = tmp_path / "suites.yaml"
    _write_catalog(catalog, model_path)

    suite = load_suite(catalog, "smoke", root=tmp_path)

    assert suite.name == "smoke"
    assert suite.cases[0].model == model_path.resolve()
    assert suite.cases[0].settings.iterations == 3
    assert suite.cases[1].settings.iterations == 5
    assert suite.cases[0].settings.seed == 7


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("id: b", "duplicate"),
        ("task: training", "task"),
        ("iterations: 0", "iterations"),
        ("duplicate key", "duplicate benchmark catalog key"),
    ],
)
def test_load_suite_rejects_invalid_catalog(tmp_path, replacement, message):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("nc: 1\n", encoding="utf-8")
    catalog = tmp_path / "suites.yaml"
    _write_catalog(catalog, model_path)
    text = catalog.read_text(encoding="utf-8")
    if replacement == "id: b":
        text = text.replace("id: a", "id: b")
    elif replacement == "task: training":
        text = text.replace("task: inference", replacement)
    elif replacement == "duplicate key":
        text = text.replace("  seed: 7", "  seed: 7\n  seed: 8")
    else:
        text = text.replace("iterations: 3", replacement)
    catalog.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_suite(catalog, "smoke", root=tmp_path)


def test_summarize_latency_computes_percentiles_and_throughput():
    summary = summarize_latency([1.0, 2.0, 3.0, 4.0, 100.0], batch_size=2)

    assert summary["latency_mean_ms"] == pytest.approx(22.0)
    assert summary["latency_p50_ms"] == pytest.approx(3.0)
    assert summary["latency_p95_ms"] == pytest.approx(80.8)
    assert summary["latency_p99_ms"] == pytest.approx(96.16)
    assert summary["latency_min_ms"] == pytest.approx(1.0)
    assert summary["latency_max_ms"] == pytest.approx(100.0)
    assert summary["throughput_items_s"] == pytest.approx(2000.0 / 3.0)


def test_case_fingerprint_changes_with_runtime_environment(tmp_path):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("nc: 1\n", encoding="utf-8")
    catalog = tmp_path / "suites.yaml"
    _write_catalog(catalog, model_path)
    case = load_suite(catalog, "smoke", root=tmp_path).cases[0]

    first = case_fingerprint(case, environment={"torch": "2.9", "device": "cpu-a"})
    second = case_fingerprint(case, environment={"torch": "2.9", "device": "cpu-b"})

    assert first != second


def test_write_reports_produces_stable_json_csv_and_markdown(tmp_path):
    report = BenchmarkRunReport(
        schema_version=1,
        suite_name="smoke",
        description="test",
        settings={"device": "cpu"},
        environment={"python": "3.11"},
        started_at="2026-07-17T00:00:00Z",
        completed_at="2026-07-17T00:00:01Z",
        cases=[_result("b", "fb"), _result("a", "fa")],
    )

    paths = write_reports(report, tmp_path)

    payload = json.loads(paths["json"].read_text(encoding="utf-8"))
    with paths["csv"].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    markdown = paths["markdown"].read_text(encoding="utf-8")
    assert [case["case_id"] for case in payload["cases"]] == ["b", "a"]
    assert [row["case_id"] for row in rows] == ["b", "a"]
    assert "| b | B | ok |" in markdown
    assert "latency_p50_ms" in rows[0]


def test_runner_continues_failures_and_resumes_matching_success(tmp_path):
    model_path = tmp_path / "model.yaml"
    model_path.write_text("nc: 1\n", encoding="utf-8")
    catalog = tmp_path / "suites.yaml"
    _write_catalog(catalog, model_path)
    suite = load_suite(catalog, "smoke", root=tmp_path)
    calls: list[str] = []

    def execute(case, fingerprint):
        calls.append(case.case_id)
        if case.case_id == "a":
            raise RuntimeError("boom")
        return _result(case.case_id, fingerprint)

    runner = BenchmarkSuiteRunner(suite, tmp_path / "out", case_executor=execute)
    first = runner.run(resume=True)
    assert [case.status for case in first.cases] == ["failed", "ok"]
    assert calls == ["a", "b"]

    calls.clear()
    second = runner.run(resume=True)
    assert calls == ["a"]
    assert [case.status for case in second.cases] == ["failed", "reused"]

    changed = load_suite(catalog, "smoke", root=tmp_path, overrides={"image_size": 48})
    calls.clear()
    BenchmarkSuiteRunner(changed, tmp_path / "out", case_executor=execute).run(resume=True)
    assert calls == ["a", "b"]
