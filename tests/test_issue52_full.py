"""Regression tests for the complete Issue #52 experiment orchestrator."""

import csv
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from scripts.run_issue52_full import ROOT
from scripts.run_issue52_full import (
    DEFAULT_BATCH,
    DEFAULT_IMGSZ,
    PRUNE_FIELDS,
    SCHEDULE_VARIANTS,
    _analyze_pruning,
    _canonical_moe_layer_name,
    _expert_signature,
    _freeze_detection_heads,
    _summarize_schedule,
    _tracker_gini,
    _write_csv,
)
from ultralytics import YOLO
from ultralytics.nn.modules.moe.analysis import ExpertStats
from ultralytics.nn.modules.moe.modules import ES_MOE


def _pruning_row(threshold, recovery, map_value, latency, experts):
    row = {field: "" for field in PRUNE_FIELDS}
    row.update(
        {
            "threshold": threshold,
            "recovery": recovery,
            "checkpoint": f"{recovery}.pt",
            "map50_95": map_value,
            "map50": map_value + 0.1,
            "gflops": 8.0,
            "latency_ms": latency,
            "params_m": 2.5,
            "mean_gini": 0.1,
            "experts_per_layer": json.dumps({"model.3": experts}),
            "layer_gini": json.dumps({"model.3": 0.1}),
        }
    )
    return row


def test_pruning_analysis_selects_only_structural_quality_gated_sweet_spot(tmp_path):
    results = tmp_path / "results.csv"
    _write_csv(
        results,
        [
            _pruning_row(0.0, "dense", 0.50, 10.0, 3),
            _pruning_row(0.10, "direct", 0.495, 8.0, 2),
            _pruning_row(0.20, "direct", 0.30, 7.0, 1),
        ],
    )

    recommendation_path = _analyze_pruning(results, tmp_path, max_map_drop=0.01)
    recommendation = json.loads(recommendation_path.read_text(encoding="utf-8"))
    assert recommendation["sweet_spot_status"] == "observed"
    assert float(recommendation["sweet_spot"]["threshold"]) == pytest.approx(0.10)
    assert (tmp_path / "pareto.csv").exists()


def test_pruning_analysis_rejects_noop_threshold_as_sweet_spot(tmp_path):
    results = tmp_path / "results.csv"
    _write_csv(
        results,
        [
            _pruning_row(0.0, "dense", 0.50, 10.0, 3),
            _pruning_row(0.10, "direct", 0.50, 9.0, 3),
        ],
    )
    recommendation = json.loads(_analyze_pruning(results, tmp_path, 0.01).read_text(encoding="utf-8"))
    assert recommendation["sweet_spot_status"] == "not_observed"
    assert recommendation["server"]["recovery"] == "dense"


def _write_training_results(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metrics/mAP50-95(B)"])
        writer.writeheader()
        writer.writerows({"metrics/mAP50-95(B)": value} for value in values)


def test_schedule_summary_uses_gini_trace_and_reports_speedup(tmp_path):
    _write_training_results(tmp_path / "baseline_fixed" / "results.csv", [0.10, 0.20])
    _write_training_results(tmp_path / "dynamic_gini_balance" / "results.csv", [0.19, 0.21])
    _write_training_results(tmp_path / "ablation_low_coeff" / "results.csv", [0.10, 0.19])
    trace = tmp_path / "dynamic_gini_balance" / "moe_dynamic_schedule.csv"
    with trace.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["mean_gini", "balance_loss_coeff"])
        writer.writeheader()
        writer.writerows(
            [
                {"mean_gini": 0.30, "balance_loss_coeff": 1.05},
                {"mean_gini": 0.20, "balance_loss_coeff": 0.95},
            ]
        )

    summary = _summarize_schedule(tmp_path)
    with summary.open(newline="", encoding="utf-8") as handle:
        rows = {row["variant"]: row for row in csv.DictReader(handle)}
    assert float(rows["dynamic"]["convergence_speedup"]) == pytest.approx(2.0)
    assert float(rows["dynamic"]["mean_gini"]) == pytest.approx(0.25)
    assert float(rows["dynamic"]["final_balance_loss_coeff"]) == pytest.approx(0.95)


def test_full_runner_uses_real_gini_dynamic_variant():
    assert SCHEDULE_VARIANTS["dynamic"]["args"]["moe_dynamic_schedule"] == "gini"
    assert SCHEDULE_VARIANTS["baseline"]["args"]["moe_dynamic_schedule"] == "none"
    assert (DEFAULT_BATCH, DEFAULT_IMGSZ) == (36, 1344)


def test_structure_signature_normalizes_peft_wrapper_prefix():
    assert _canonical_moe_layer_name("model.3") == "model.3"
    assert _canonical_moe_layer_name("model.base_model.model.3") == "model.3"
    assert _canonical_moe_layer_name("base_model.model.12") == "model.12"

    model = nn.Module()
    model.model = nn.ModuleList([nn.Identity(), ES_MOE(8, 8, num_experts=2, top_k=1)])
    assert _expert_signature(model) == {"model.1": {"num_experts": 2, "top_k": 1}}


def test_structure_preserving_recovery_locks_detection_head_only():
    class Detect(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(4, 2)

    backbone = nn.Linear(4, 4)
    head = Detect()
    model = nn.Sequential(backbone, head)
    frozen = _freeze_detection_heads(model)

    assert frozen == sum(parameter.numel() for parameter in head.parameters())
    assert all(parameter.requires_grad for parameter in backbone.parameters())
    assert not any(parameter.requires_grad for parameter in head.parameters())


def test_tracker_gini_ignores_nested_lora_router_name_matches():
    model = nn.Sequential(ES_MOE(8, 8, num_experts=3, top_k=2))
    tracker = SimpleNamespace(
        usage_stats={
            "0.routing": {
                0: ExpertStats(hits=8),
                1: ExpertStats(hits=1),
                2: ExpertStats(hits=1),
            },
            "0.routing.routing_network.0.lora_dropout.default": {
                0: ExpertStats(hits=1),
                1: ExpertStats(hits=9),
            },
        }
    )
    ginis = _tracker_gini(tracker, model)
    assert set(ginis) == {"0"}
    assert ginis["0"] > 0.4


def test_visdrone_esmoe_model_cfg_builds_and_forwards():
    config = ROOT / "ultralytics/cfg/models/master/v0/det/yolo-master-esmoe-n-visdrone.yaml"
    model = YOLO(str(config)).model.eval()
    with torch.inference_mode():
        output = model(torch.zeros(1, 3, 64, 64))
    assert output is not None
