from __future__ import annotations

import json

import pytest

from e3_p1.dashboard import build_snapshot, load_jsonl, load_jsonl_window, render_dashboard_html, smoke_test_server


def _event(family="mot"):
    return {
        "family": family,
        "module": {"name": "route"},
        "routing": {
            "expert_load": [0.6, 0.4],
            "entropy_normalized": 0.9,
            "load_gini": 0.1,
            "dominant_expert_share": 0.6,
            "mixing_weights": {"state": "observed"},
        },
        "aux_loss": {"state": "observed"},
        "runtime": {"batch_size": 1, "sample_indices": [0]},
    }


def test_jsonl_loader_ignores_partial_tail_and_dashboard_updates(tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text(json.dumps(_event()) + "\n{partial", encoding="utf-8")

    first = build_snapshot(load_jsonl(source))
    source.write_text(json.dumps(_event()) + "\n" + json.dumps(_event("moe")) + "\n", encoding="utf-8")
    second = build_snapshot(load_jsonl(source))

    assert first["event_count"] == 1
    assert second["event_count"] == 2
    assert sorted(second["families"]) == ["moe", "mot"]


def test_dashboard_html_and_real_http_smoke(tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text(json.dumps(_event()) + "\n", encoding="utf-8")

    html = render_dashboard_html(refresh_ms=250)
    smoke = smoke_test_server(source)

    assert "fetch('/api/snapshot'" in html
    assert "refresh 250 ms" in html
    assert smoke == {
        "status": "PASS",
        "health_status": 200,
        "html_contains_dashboard": True,
        "api_event_count": 1,
        "api_source_event_count": 1,
        "api_window_limit": 1000,
        "api_truncated": False,
        "api_families": ["mot"],
    }


def test_jsonl_window_reports_truncation_without_losing_source_count(tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text("".join(json.dumps(_event()) + "\n" for _ in range(1001)), encoding="utf-8")

    window, source_count = load_jsonl_window(source)
    snapshot = build_snapshot(window, source_event_count=source_count, window_limit=1000)

    assert len(window) == 1000
    assert snapshot["event_count"] == 1000
    assert snapshot["source_event_count"] == 1001
    assert snapshot["truncated"] is True


def test_jsonl_loader_rejects_corruption_before_final_line(tmp_path):
    source = tmp_path / "events.jsonl"
    source.write_text(json.dumps(_event()) + "\n{broken}\n" + json.dumps(_event()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSONL before final line"):
        load_jsonl(source)
