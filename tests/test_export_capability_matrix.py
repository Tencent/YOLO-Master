"""Tests for the canonical mixture export capability matrix."""

from pathlib import Path

import pytest
import torch
import yaml

from ultralytics.nn.modules.moa import MoABlock
from ultralytics.nn.modules.block import DyMoEBlock
from ultralytics.nn.modules.moe import AdaptiveGateMoE
from ultralytics.nn.modules.moe.modules import OptimizedMOE
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.engine.exporter import export_formats
from ultralytics.utils.export_capabilities import (
    classify_routed_module,
    load_export_capability_matrix,
    normalize_export_format,
    render_export_capability_markdown,
    validate_export_capability_matrix,
)
from ultralytics.utils.export_preflight import export_preflight


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_export_matrix_is_valid_and_covers_routed_families():
    matrix = load_export_capability_matrix()

    assert matrix["schema_version"] == 1
    assert {"MoE", "MoA", "MoT", "MoLoRA"} <= matrix["modules"].keys()
    assert {"supported", "dense_fallback", "requires_merge", "known_error"} <= matrix["modules"]["MoT"].keys()


def test_canonical_export_matrix_covers_every_exporter_format():
    matrix = load_export_capability_matrix()
    exporter_formats = {normalize_export_format(fmt) for fmt in export_formats()["Argument"]}

    assert exporter_formats == set(matrix["formats"])


def test_matrix_validation_rejects_missing_required_fields():
    with pytest.raises(ValueError, match="dense_fallback"):
        validate_export_capability_matrix(
            {
                "schema_version": 1,
                "formats": {"onnx": {"supported": True, "default": "dense_fallback", "known_error": None}},
                "modules": {name: {"supported": True} for name in ("MoE", "MoA", "MoT", "MoLoRA")},
            }
        )


def test_format_aliases_and_routed_module_classification():
    assert normalize_export_format("TensorRT") == "engine"
    assert normalize_export_format("trt") == "engine"
    assert classify_routed_module(OptimizedMOE(16, 16, num_experts=2, top_k=1)) == "MoE"
    assert classify_routed_module(AdaptiveGateMoE(16, 16, num_experts=2, top_k=1)) == "MoE"
    assert classify_routed_module(DyMoEBlock(16, num_experts=2, top_k=1)) == "MoE"
    assert classify_routed_module(MoABlock(16, num_heads=3)) == "MoA"
    assert classify_routed_module(MoTBlock(16, num_heads=2, top_k=1)) == "MoT"
    assert classify_routed_module(MoLoRALayer(torch.nn.Linear(16, 16), r=2, num_experts=2, top_k=1)) == "MoLoRA"


def test_matrix_validation_rejects_invalid_override_boolean():
    matrix = _matrix()
    matrix["modules"]["MoT"]["formats"] = {"onnx": {"dense_fallback": "yes"}}

    with pytest.raises(ValueError, match="dense_fallback must be bool"):
        validate_export_capability_matrix(matrix)


def test_matrix_validation_rejects_unknown_override_format():
    matrix = _matrix()
    matrix["modules"]["MoT"]["formats"] = {"engine": {"supported": True}}

    with pytest.raises(ValueError, match="unknown formats"):
        validate_export_capability_matrix(matrix)


def _matrix(*, mot_supported=True, molora_requires_merge=False):
    return {
        "schema_version": 1,
        "formats": {
            "pytorch": {"supported": True, "default": "dynamic", "known_error": None},
            "onnx": {"supported": True, "default": "dense_fallback", "known_error": None},
        },
        "modules": {
            "MoE": {"supported": True, "dense_fallback": True, "requires_merge": False, "known_error": None},
            "MoA": {"supported": True, "dense_fallback": True, "requires_merge": False, "known_error": None},
            "MoT": {
                "supported": mot_supported,
                "dense_fallback": True,
                "requires_merge": False,
                "known_error": None if mot_supported else "MoT blocked by policy",
            },
            "MoLoRA": {
                "supported": True,
                "dense_fallback": True,
                "requires_merge": molora_requires_merge,
                "known_error": None,
            },
        },
    }


def test_preflight_refuses_matrix_blocked_module():
    report = export_preflight(MoTBlock(16, num_heads=2, top_k=1), "onnx", strict=False, matrix=_matrix(mot_supported=False))

    assert report["supported"] is False
    assert report["decisions"][0]["strategy"] == "refuse"
    assert "MoT blocked by policy" in report["decisions"][0]["known_error"]


def test_preflight_enforces_matrix_merge_requirement():
    layer = MoLoRALayer(torch.nn.Linear(16, 16), r=2, num_experts=2, top_k=1)
    matrix = _matrix(molora_requires_merge=True)

    refused = export_preflight(layer, "onnx", strict=False, matrix=matrix)
    assert refused["supported"] is False
    assert refused["decisions"][0]["requires_merge"] is True

    layer.merge_weights()
    accepted = export_preflight(layer, "onnx", strict=True, matrix=matrix)
    assert accepted["supported"] is True


def test_preflight_intersects_runtime_dense_fallback_declaration(monkeypatch):
    layer = MoTBlock(16, num_heads=2, top_k=1)
    monkeypatch.setattr(
        layer,
        "export_capabilities",
        lambda: {"supported": True, "dynamic_routing": True, "export_safe_dense_fallback": False},
    )

    report = export_preflight(layer, "onnx", strict=False, matrix=_matrix())

    assert report["supported"] is False
    assert "not allowed by both matrix and runtime" in report["errors"][0]


def test_preflight_loads_matrix_path(tmp_path):
    path = tmp_path / "export-matrix.yaml"
    path.write_text(yaml.safe_dump(_matrix(), sort_keys=False), encoding="utf-8")

    report = export_preflight(MoTBlock(16, num_heads=2), "onnx", matrix_path=path)

    assert report["supported"] is True
    assert report["matrix_source"] == str(path)


def test_unknown_export_format_is_refused():
    report = export_preflight(MoTBlock(16, num_heads=2), "unknown_backend", strict=False)

    assert report["supported"] is False
    assert report["errors"]


def test_unsupported_routed_format_does_not_block_dense_models():
    matrix = _matrix()
    matrix["formats"]["axelera"] = {
        "supported": False,
        "default": "refuse",
        "known_error": "routed export is unverified",
    }

    dense_report = export_preflight(torch.nn.Conv2d(3, 8, 3), "axelera", matrix=matrix)
    routed_report = export_preflight(MoTBlock(16, num_heads=2), "axelera", strict=False, matrix=matrix)

    assert dense_report["supported"] is True
    assert dense_report["decisions"] == []
    assert routed_report["supported"] is False


def test_preflight_reports_matrix_metadata():
    report = export_preflight(MoTBlock(16, num_heads=2), "onnx", strict=True, matrix=_matrix())

    assert report["matrix_schema_version"] == 1
    assert report["matrix_source"] == "<in-memory>"
    assert report["decisions"][0]["module_family"] == "MoT"


def test_checked_in_export_matrix_markdown_is_generated():
    matrix = load_export_capability_matrix()
    rendered = render_export_capability_markdown(matrix)
    document = ROOT / "docs/governance/export-capability-matrix.md"

    assert document.read_text(encoding="utf-8") == rendered
