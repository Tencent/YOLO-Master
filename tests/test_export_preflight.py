import torch
from unittest import mock

from ultralytics.engine.exporter import Exporter
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.nn.peft.molora.layer import MoLoRALayer
from ultralytics.utils.export_preflight import export_preflight


def test_mixture_export_preflight_selects_dense_fallback():
    model = torch.nn.Sequential(MoTBlock(16, num_heads=2, top_k=1), MoLoRALayer(torch.nn.Linear(16, 16), r=2, num_experts=2, top_k=1))
    report = export_preflight(model, "onnx", strict=True)
    assert report["supported"] is True
    assert all(item["strategy"] == "dense_fallback" for item in report["decisions"])


def test_export_preflight_reports_safe_eager_strategy():
    model = MoTBlock(16, num_heads=2, top_k=1)
    report = export_preflight(model, "pytorch", strict=True)
    assert report["decisions"][0]["strategy"] == "dynamic"


def test_exporter_invokes_preflight_before_device_setup(monkeypatch):
    model = torch.nn.Sequential(MoTBlock(16, num_heads=2, top_k=1))
    exporter = Exporter(overrides={"format": "onnx"})
    report = {
        "format": "onnx",
        "supported": True,
        "matrix_schema_version": 1,
        "matrix_source": "test",
        "decisions": [],
        "errors": [],
    }
    preflight = mock.Mock(return_value=report)
    monkeypatch.setattr("ultralytics.utils.export_preflight.export_preflight", preflight)
    monkeypatch.setattr("ultralytics.engine.exporter.select_device", mock.Mock(side_effect=RuntimeError("stop")))

    try:
        exporter(model=model)
    except RuntimeError as error:
        assert str(error) == "stop"

    preflight.assert_called_once_with(model, "onnx", strict=True)
    assert exporter.export_preflight_report is report
