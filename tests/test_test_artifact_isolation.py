"""Ensure test lifecycle hooks never remove artifacts owned by the caller."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tests import conftest


@pytest.mark.parametrize("exitstatus", [0, 1])
def test_session_finish_preserves_existing_artifacts(tmp_path, monkeypatch, exitstatus):
    """Successful and failed sessions must both preserve shared weights and working files."""
    weights = tmp_path / "weights"
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    artifacts = [
        weights / "nested" / "customer.onnx",
        weights / "nested" / "customer.torchscript",
        weights / "customer.mlpackage" / "model.bin",
        weights / "customer_openvino_model" / "model.xml",
        *[workdir / name for name in ("bus.jpg", "decelera_portrait_min.mov", "yolo26n.onnx", "yolo26n.torchscript")],
    ]
    for path in artifacts:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"existing user artifact")
    monkeypatch.chdir(workdir)
    monkeypatch.setattr("ultralytics.utils.WEIGHTS_DIR", weights)
    manager = pytest.PytestPluginManager()
    manager.register(conftest)

    manager.hook.pytest_sessionfinish(session=SimpleNamespace(config=SimpleNamespace()), exitstatus=exitstatus)

    for path in artifacts:
        assert path.is_file(), f"test lifecycle removed an existing artifact: {path}"
        assert path.read_bytes() == b"existing user artifact"


def test_export_copy_is_owned_by_the_test_directory(tmp_path):
    """An export derived from the isolated model must not overwrite a cached artifact."""
    cached_model = tmp_path / "shared" / "model.pt"
    cached_model.parent.mkdir()
    cached_model.write_bytes(b"cached weights")
    cached_export = cached_model.with_suffix(".onnx")
    cached_export.write_bytes(b"existing export")
    test_dir = tmp_path / "test_owned"
    test_dir.mkdir()

    copied_model = conftest.isolated_model_path(test_dir, cached_model)
    Path(copied_model).with_suffix(".onnx").write_bytes(b"new test export")

    assert Path(copied_model).parent == test_dir
    assert cached_model.read_bytes() == b"cached weights"
    assert cached_export.read_bytes() == b"existing export"
