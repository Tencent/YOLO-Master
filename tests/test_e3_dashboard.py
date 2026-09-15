"""Contract, adapter, and HTTP tests for the local Routing Observatory."""

from __future__ import annotations

import json
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from scripts.e3_overhead_benchmark import (
    _image_files,
    paired_bootstrap_ci,
    parse_model_overrides,
    percentile,
    prepare_benchmark_dataset,
    summarize_pairs,
)
from scripts.e3_routing_dashboard import (
    ObservatoryHandler,
    WorkspaceIndex,
    _normalise_routing_map,
    _normalise_training_smoke,
)

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD = ROOT / "reports/e3_routing_observability/dashboard"


def _benchmark_runs(overheads: list[float]) -> list[dict]:
    records = []
    for pair, overhead in enumerate(overheads, 1):
        records.extend(
            [
                {"pair": pair, "condition": "baseline", "mean_milliseconds": 100.0},
                {"pair": pair, "condition": "e3", "mean_milliseconds": 100.0 + overhead},
            ]
        )
    return records


def test_percentile_interpolates_and_rejects_empty_values():
    assert percentile([1.0, 2.0, 3.0], 0.5) == 2.0
    assert percentile([0.0, 10.0], 0.95) == pytest.approx(9.5)
    with pytest.raises(ValueError, match="at least one"):
        percentile([], 0.5)


def test_paired_bootstrap_is_reproducible():
    first = paired_bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], iterations=500, seed=17)
    second = paired_bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], iterations=500, seed=17)
    assert first == second
    assert first[0] <= 3.0 <= first[1]


def test_pair_summary_passes_only_when_mean_and_ci_upper_are_below_threshold():
    passing = summarize_pairs(_benchmark_runs([1.0, 2.0, 3.0, 4.0, 5.0]), bootstrap_iterations=1_000)
    failing_mean = summarize_pairs(_benchmark_runs([11.0] * 5), bootstrap_iterations=100)
    failing_ci = summarize_pairs(_benchmark_runs([-5.0, -5.0, -5.0, -5.0, 30.0]), bootstrap_iterations=2_000)
    assert passing["passed"] is True
    assert failing_mean["passed"] is False
    assert failing_ci["mean_overhead_percent"] < 10.0
    assert failing_ci["paired_bootstrap_95_ci_percent"][1] > 10.0
    assert failing_ci["passed"] is False


def test_pair_summary_requires_complete_pairs():
    with pytest.raises(ValueError, match="exactly one"):
        summarize_pairs([{"pair": 1, "condition": "baseline", "mean_milliseconds": 100.0}])


def test_native_dataset_is_used_without_creating_repeated_files(tmp_path, monkeypatch):
    image_dir = tmp_path / "dataset/images/train"
    image_dir.mkdir(parents=True)
    for name in ("a.jpg", "b.jpg"):
        (image_dir / name).write_bytes(b"image")
    data_yaml = tmp_path / "dataset.yaml"
    data_yaml.write_text("path: .\ntrain: dataset/images/train\nval: dataset/images/train\nnames: [item]\n")
    monkeypatch.setattr("ultralytics.data.utils.check_det_dataset", lambda *_args, **_kwargs: {"train": str(image_dir)})
    resolved, count = prepare_benchmark_dataset(str(data_yaml), tmp_path / "output", images=20, native=True)
    assert resolved == data_yaml.resolve()
    assert count == 2
    assert not (tmp_path / "output/generated_dataset").exists()
    assert _image_files(str(image_dir)) == sorted(image_dir.iterdir())


def test_model_overrides_validate_family_and_path(tmp_path):
    checkpoint = tmp_path / "best.pt"
    checkpoint.write_bytes(b"checkpoint")
    models = parse_model_overrides([f"latent={checkpoint}"])
    assert models["latent"] == checkpoint.resolve()
    with pytest.raises(ValueError, match="FAMILY=PATH"):
        parse_model_overrides([f"unknown={checkpoint}"])
    with pytest.raises(FileNotFoundError):
        parse_model_overrides([f"moe={tmp_path / 'missing.pt'}"])


def _snapshot(family: str = "switchboard") -> dict:
    return {
        "schema_version": "e3.routing_snapshot.v1",
        "device": "test-device",
        "families": {
            family: {
                "layers": [
                    {
                        "layer_name": "model.7.router",
                        "module_type": "SyntheticRouter",
                        "num_experts": 4,
                        "top_k": 2,
                        "routing_axis": "token",
                        "dispatch_policy": "topk",
                        "expert_usage": [0.2, 0.3, 0.1, 0.4],
                        "mean_router_probs": [0.22, 0.28, 0.12, 0.38],
                        "normalized_entropy": 0.91,
                        "aux_loss": {
                            "status": "active_training",
                            "configured": True,
                            "observed": 0.02,
                        },
                    }
                ]
            }
        },
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_snapshot_and_benchmark_are_discovered_and_bounded(tmp_path: Path):
    run = tmp_path / "cuda-run"
    snapshot = _snapshot("moe")
    snapshot["families"]["moe"]["static_figure"] = "moe_expert_usage.png"
    _write_json(run / "route_stats.json", snapshot)
    (run / "moe_expert_usage.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    _write_json(
        run / "summary.json",
        {
            "schema_version": "e3.overhead_benchmark.v1",
            "passed": True,
            "runs": [
                {"family": "moe", "condition": "baseline", "raw_milliseconds": [1.0]},
                {"family": "moe", "condition": "e3", "raw_milliseconds": [1.1]},
            ],
            "families": {"moe": {"passed": True, "mean_overhead_percent": 1.0}},
        },
    )
    index = WorkspaceIndex([tmp_path])

    assert len(index.workspaces) == 1
    workspace = next(iter(index.workspaces.values()))
    assert workspace["summary"]["layers"] > 0
    assert workspace["benchmark"]["passed"] is True
    assert workspace["summary"]["has_benchmark"] is True
    assert workspace["evidence"]
    assert all("relative_path" in item for item in workspace["evidence"])
    assert {item["name"] for item in workspace["evidence"] if item["media_type"] == "image/png"} == {
        "moe_expert_usage.png"
    }
    assert all("raw_milliseconds" not in run for run in workspace["benchmark"]["runs"])


def test_final_manifest_merges_nested_training_routing_and_overhead(tmp_path: Path):
    package = tmp_path / "cuda-final"
    _write_json(
        package / "manifest.json",
        {
            "schema_version": "e3.cuda_final.v1",
            "batch": 2,
            "config": {"data": "dataset.yaml", "imgsz": 640, "device": 0},
            "environment": {"gpu": "RTX"},
            "git": {"commit": "abc", "dirty": False},
            "training": {"moe": {"epochs": 50, "metrics": {"metrics/mAP50-95(B)": 0.42}}},
            "evaluation": {"moe": {"metrics": {"metrics/mAP50-95(B)": 0.50}}},
            "routing": {"families": {"moe": {"captured_layers": 1}}},
        },
    )
    _write_json(package / "visualizations/route_stats.json", _snapshot("moe"))
    _write_json(
        package / "overhead/summary.json",
        {
            "schema_version": "e3.overhead_benchmark.v1",
            "passed": True,
            "runs": [],
            "families": {"moe": {"passed": True, "mean_overhead_percent": 1.0}},
        },
    )

    index = WorkspaceIndex([tmp_path])

    assert len(index.workspaces) == 1
    workspace = next(iter(index.workspaces.values()))
    assert workspace["name"] == "cuda final"
    assert workspace["summary"]["layers"] == 1
    assert workspace["summary"]["has_benchmark"] is True
    assert workspace["training_runs"]["moe"] == {
        "epochs": 50,
        "routed_layers": 1,
        "metrics": {"metrics/mAP50-95(B)": 0.42},
        "test_metrics": {"metrics/mAP50-95(B)": 0.50},
    }


def test_unknown_family_is_dynamic_and_duplicate_layers_are_removed(tmp_path: Path):
    run = tmp_path / "run-a"
    _write_json(run / "snapshot-a.json", _snapshot())
    _write_json(run / "snapshot-b.json", _snapshot())

    workspace = next(iter(WorkspaceIndex([tmp_path]).workspaces.values()))

    assert workspace["summary"]["families"] == {"switchboard": 1}
    assert workspace["summary"]["layers"] == 1
    assert workspace["layers"][0]["family"] == "switchboard"


def test_native_routing_map_and_input_image_are_bounded_and_discovered(tmp_path: Path):
    snapshot = _snapshot()
    snapshot["families"]["switchboard"]["routing_map"] = {
        "kind": "spatial",
        "layer_name": "model.7.router",
        "num_experts": 2,
        "height": 2,
        "width": 2,
        "input_image": "input.jpg",
        "probabilities": [[[0.8, 0.2], [0.4, 0.6]], [[0.2, 0.8], [0.6, 0.4]]],
    }
    _write_json(tmp_path / "native" / "snapshot.json", snapshot)
    (tmp_path / "native" / "input.jpg").write_bytes(b"jpeg")

    workspace = next(iter(WorkspaceIndex([tmp_path]).workspaces.values()))

    assert workspace["routing_maps"]["switchboard"]["probabilities"][0][0] == [0.8, 0.2]
    assert any(item["name"] == "input.jpg" and item["media_type"] == "image/jpeg" for item in workspace["evidence"])
    assert (
        _normalise_routing_map({"kind": "spatial", "num_experts": 1, "height": 999, "width": 1, "probabilities": []})
        is None
    )


def test_baseline_telemetry_without_routing_is_an_empty_run(tmp_path: Path):
    _write_json(
        tmp_path / "baseline" / "telemetry.json",
        {
            "schema_version": 1,
            "ranks": [
                {
                    "metadata": {"device": "cpu"},
                    "steps": {"first": 1, "last": 2},
                    "routing": {"observations": 0, "unsupported_layers": 0, "last": {}},
                }
            ],
        },
    )

    workspace = next(iter(WorkspaceIndex([tmp_path]).workspaces.values()))

    assert workspace["status"] == "empty"
    assert workspace["summary"]["layers"] == 0
    assert workspace["benchmark"] is None


def test_compact_live_training_smoke_is_normalised_for_dashboard():
    runs = _normalise_training_smoke(
        {
            "families": {
                "moe": {
                    "routed_layers": 6,
                    "routing_observations": 4,
                    "routing_scalar_tags": 136,
                    "invalid_layers": 0,
                    "unsupported_layers": 0,
                    "aux_statuses": ["active_training"],
                }
            }
        }
    )

    assert runs == {
        "moe": {
            "routed_layers": 6,
            "routing_observations": 4,
            "tensorboard_routing_scalar_tags": 136,
            "invalid_layers": 0,
            "unsupported_layers": 0,
            "aux_statuses": ["active_training"],
        }
    }


def test_committed_example_combines_routing_training_and_performance():
    index = WorkspaceIndex([ROOT / "reports/e3_routing_observability/results"])
    workspace = next(item for item in index.workspaces.values() if item["relative_path"] == ".")

    assert index.scan_errors == []
    assert workspace["status"] == "healthy"
    assert workspace["schemas"] == [
        "e3.overhead_benchmark.v1",
        "e3.routing_snapshot.v1",
        "e3.training_smoke.v1",
    ]
    assert workspace["summary"]["families"] == {"moe": 6, "mot": 4, "latent": 3}
    assert workspace["summary"]["has_benchmark"] is True
    assert set(workspace["training_runs"]) == {"moe", "mot", "latent"}
    assert set(workspace["benchmark"]["families"]) == {"moe", "mot", "latent"}
    assert workspace["summary"]["visual_artifacts"] == 3


def test_missing_benchmark_does_not_degrade_valid_snapshot(tmp_path: Path):
    _write_json(tmp_path / "routing-only" / "snapshot.json", _snapshot())

    workspace = next(iter(WorkspaceIndex([tmp_path]).workspaces.values()))

    assert workspace["status"] == "healthy"
    assert workspace["benchmark"] is None


def test_non_finite_layer_values_are_not_emitted_as_json_nan(tmp_path: Path):
    snapshot = _snapshot()
    snapshot["families"]["switchboard"]["layers"][0]["expert_usage"][0] = float("nan")
    _write_json(tmp_path / "routing" / "snapshot.json", snapshot)

    workspace = next(iter(WorkspaceIndex([tmp_path]).workspaces.values()))

    assert workspace["status"] == "warning"
    assert "non_finite_expert_value" in workspace["layers"][0]["issues"]
    assert "NaN" not in json.dumps(workspace)


def test_invalid_json_and_symlink_escape_are_skipped(tmp_path: Path):
    root = tmp_path / "registered"
    root.mkdir()
    (root / "broken.json").write_text("{not-json", encoding="utf-8")
    outside = tmp_path / "outside.json"
    _write_json(outside, _snapshot())
    try:
        (root / "escape.json").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not available on this platform")

    index = WorkspaceIndex([root])

    assert not index.workspaces
    assert {error["reason"] for error in index.scan_errors} == {"invalid_json", "outside_root"}


def test_unrelated_json_and_non_telemetry_schema_one_are_ignored(tmp_path: Path):
    (tmp_path / "predictions.json").write_text("[]", encoding="utf-8")
    _write_json(tmp_path / "manifest.json", {"schema_version": 1, "status": "complete"})

    index = WorkspaceIndex([tmp_path])

    assert not index.workspaces
    assert not index.scan_errors


@pytest.fixture
def observatory_url(tmp_path: Path):
    snapshot = _snapshot()
    snapshot["families"]["switchboard"]["static_figure"] = "routing.png"
    _write_json(tmp_path / "run-a" / "snapshot.json", snapshot)
    (tmp_path / "run-a" / "routing.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    index = WorkspaceIndex([tmp_path])
    handler = type("TestObservatoryHandler", (ObservatoryHandler,), {"index": index})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", index
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=2) as response:
        return json.load(response)


def test_http_list_detail_refresh_and_evidence_routes(observatory_url):
    base_url, index = observatory_url
    listing = _get_json(f"{base_url}/api/v1/runs")
    run_id = listing["runs"][0]["id"]
    detail = _get_json(f"{base_url}/api/v1/runs/{run_id}")
    file_id = detail["evidence"][0]["id"]
    image_id = next(file["id"] for file in detail["evidence"] if file["media_type"] == "image/png")

    assert detail["layers"][0]["family"] == "switchboard"
    assert _get_json(f"{base_url}/api/v1/runs/{run_id}/files/{file_id}")["schema_version"]
    with urlopen(f"{base_url}/api/v1/runs/{run_id}/files/{image_id}", timeout=2) as response:
        assert response.headers["Content-Type"] == "image/png"
        assert response.read().startswith(b"\x89PNG")
    request = Request(f"{base_url}/api/v1/refresh", method="POST")
    with urlopen(request, timeout=2) as response:
        assert json.load(response)["runs"][0]["id"] == run_id
    assert index.workspaces

    evidence = index.files[(run_id, file_id)]
    outside = evidence.root.parent / "outside-after-scan.json"
    _write_json(outside, _snapshot())
    evidence.path.unlink()
    evidence.path.symlink_to(outside)
    with pytest.raises(HTTPError) as error:
        urlopen(f"{base_url}/api/v1/runs/{run_id}/files/{file_id}", timeout=2)
    assert error.value.code == 404


@pytest.mark.parametrize(
    "path",
    ["/api/v1/runs/0000000000000000", "/api/v1/runs/../../AGENTS.md", "/../pyproject.toml"],
)
def test_http_rejects_unknown_ids_and_path_traversal(observatory_url, path: str):
    base_url, _ = observatory_url

    with pytest.raises(HTTPError) as error:
        urlopen(f"{base_url}{path}", timeout=2)

    assert error.value.code == 404


def test_frontend_is_offline_generic_and_uses_versioned_api():
    html = (DASHBOARD / "index.html").read_text(encoding="utf-8")
    javascript = (DASHBOARD / "app.js").read_text(encoding="utf-8")
    stylesheet = (DASHBOARD / "styles.css").read_text(encoding="utf-8")
    source = html + javascript + stylesheet

    assert "http://" not in source
    assert "https://" not in source
    assert 'src="/app.js"' in html
    assert 'href="/styles.css"' in html
    assert "/api/v1/runs" in javascript
    assert "/api/v1/refresh" in javascript
    assert "benchmark-locate" in html + javascript
    assert "E3 路由观测" in html
    assert "均衡度越接近 1" in html
    assert "训练采集验证" in html
    assert "性能门结果" in html
    assert "跨层专家负载偏差" in html
    assert "路由均衡二维分布" in html
    assert "空间路由证据" in html
    assert 'data-view="matrix"' in html
    assert 'data-view="balance"' in html
    assert 'data-view="spatial"' in html
    assert "可视分析" not in html
    assert "const auxLabel" in javascript
    assert "renderTrainingSummary" in javascript
    assert "renderPerformanceSummary" in javascript
    assert "renderLoadMatrix" in javascript
    assert "relativeDelta" in javascript
    assert "Math.log10(point.entropyGap)" in javascript
    assert "balance-svg" in stylesheet
    assert "routing-canvas" not in html + javascript + stylesheet
    assert "context.drawImage" not in javascript
    assert "routing_dashboard" in javascript
    assert "spatial-card" in stylesheet
    assert "属于全局路由，没有空间坐标" not in javascript
    assert "仅显示会影响判断的问题" not in html
    assert "grid-template-columns: minmax(0, 1fr) 104px 64px" in stylesheet
    assert "mean_router_probs" in javascript
    assert "Aux 观测值" in javascript
    assert "usage * 100" in javascript
    assert "usage / maxValue" not in javascript
    for decorative_label in ("WORKSPACE", "FAMILY COVERAGE", "LOWEST ENTROPY", "DIAGNOSTIC NOTES"):
        assert decorative_label not in html
    for project_specific_value in ("Apple M3", "MoE", "MoT", "Latent"):
        assert project_specific_value not in source
