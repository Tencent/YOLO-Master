"""Regression tests for strict multi-seed MoT ablation aggregation."""

from pathlib import Path

import pytest

from scripts.aggregate_mot_ablation_seeds import aggregate, write_markdown


HEADER = "key,label,metrics/mAP50(B),metrics/mAP50-95(B),final_train_total_loss,nan_detected,loss_diverged\n"


def write_seed(root: Path, seed: int, rows: list[str]) -> None:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    (seed_dir / "summary.csv").write_text(HEADER + "".join(rows), encoding="utf-8")


def test_aggregate_combines_seed_metrics_and_one_profile(tmp_path: Path):
    root = tmp_path / "runs"
    write_seed(root, 42, ["v10,baseline,0.2,0.1,5.0,False,False\n", "v10_mot,mot,0.3,0.2,6.0,False,False\n"])
    write_seed(root, 123, ["v10,baseline,0.4,0.3,5.5,False,False\n", "v10_mot,mot,0.5,0.4,6.5,False,False\n"])
    profile = tmp_path / "latency.csv"
    profile.write_text(
        "key,latency_ms_p50,latency_ms_p95,latency_ms_p99,flops_g,params_m\n"
        "v10,10,11,12,8.5,3.4\n"
        "v10_mot,20,21,22,12.2,4.0\n",
        encoding="utf-8",
    )

    rows = aggregate(root, profile, expected_seeds=["42", "123"])
    by_key = {row["key"]: row for row in rows}

    assert by_key["v10"]["seeds"] == "42,123"
    assert by_key["v10"]["map50_95_mean"] == pytest.approx(0.2)
    assert by_key["v10"]["map50_95_std"] == pytest.approx(2**0.5 / 10)
    assert by_key["v10_mot"]["latency_ms_p99"] == 22.0
    assert by_key["v10_mot"]["meaningful_gain"] is True


def test_aggregate_rejects_missing_seed(tmp_path: Path):
    root = tmp_path / "runs"
    write_seed(root, 42, ["v10,baseline,0.2,0.1,5.0,False,False\n"])

    with pytest.raises(ValueError, match="seed mismatch"):
        aggregate(root, expected_seeds=["42", "123"])


def test_aggregate_rejects_incomplete_model_coverage(tmp_path: Path):
    root = tmp_path / "runs"
    write_seed(root, 42, ["v10,baseline,0.2,0.1,5.0,False,False\n", "v10_mot,mot,0.3,0.2,6.0,False,False\n"])
    write_seed(root, 123, ["v10,baseline,0.4,0.3,5.5,False,False\n"])

    with pytest.raises(ValueError, match="incomplete model coverage"):
        aggregate(root)


def test_markdown_uses_uncertainty_and_pilot_note(tmp_path: Path):
    output = tmp_path / "summary.md"
    rows = [
        {
            "key": "v10",
            "label": "baseline",
            "n_seeds": 3,
            "map50_95_mean": 0.2,
            "map50_95_std": 0.01,
            "map50_mean": 0.3,
            "map50_std": 0.02,
            "nan_any": False,
            "loss_diverged_any": False,
            "meaningful_gain": False,
        }
    ]

    write_markdown(output, rows, title="pilot", note="smoke only")
    text = output.read_text(encoding="utf-8")

    assert "mean±std" in text
    assert "0.2000±0.0100" in text
    assert "> smoke only" in text
