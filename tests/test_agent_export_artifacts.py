"""CLI export results must identify the real final file or directory."""

import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest


@pytest.fixture
def export_runner(tmp_path, monkeypatch):
    """Exercise the real handler with a mocked CLI and no model or environment probing."""
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "agent"))
    handlers = importlib.import_module("runtime.cli.core_handlers")
    monkeypatch.setattr(handlers, "collect_environment_report", lambda *args, **kwargs: {})

    def run(stdout="", stderr="", returncode=0):
        request = {
            "skill": "yolo.export",
            "inputs": {"model": str(tmp_path / "weights" / "best.pt")},
            "params": {"device": "cpu", "format": "onnx", "project": str(tmp_path), "name": "export"},
            "runtime": {},
            "policy": {},
        }
        cli = Mock(
            return_value={
                "cmd": ["yolo", "export"],
                "cwd": str(tmp_path),
                "returncode": returncode,
                "stdout": stdout,
                "stderr": stderr,
                "install": {},
            }
        )
        deps = handlers.CoreDeps(build_model=Mock(), run_cli=cli, run_cli_with_recovery=Mock())
        return handlers.run_export(request, deps)

    return run


def test_export_reports_onnx_next_to_checkpoint(export_runner, tmp_path):
    """Export writes alongside its checkpoint even when project/name names another directory."""
    output = tmp_path / "weights" / "best.onnx"
    output.parent.mkdir()
    output.write_bytes(b"mock exported model")
    result = export_runner(stdout=f"ONNX: export success, saved as '{output}'\nResults saved to {output}\n")
    assert result["status"] == "ok"
    assert result["artifacts"] == [{"kind": "exported_model", "path": str(output)}]
    assert result["job"]["save_dir"] == str(output.parent)
    assert result["job"]["requested_save_dir"] == str(tmp_path / "export")


@pytest.mark.parametrize("directory", [False, True])
@pytest.mark.parametrize("stream", ["stdout", "stderr"])
def test_final_export_path_handles_directories_spaces_quotes_and_relative_paths(
    export_runner, tmp_path, directory, stream
):
    """Final output wins over intermediate ONNX and stale files in the requested save directory."""
    output = tmp_path / "owner's model" / ("best_openvino_model" if directory else "best.engine")
    output.parent.mkdir()
    if directory:
        output.mkdir()
        (output / "best.xml").write_text("mock openvino")
    else:
        output.write_bytes(b"mock engine")
    intermediate = output.parent / "best.onnx"
    intermediate.write_bytes(b"mock intermediate")
    stale_dir = tmp_path / "export"
    stale_dir.mkdir()
    (stale_dir / "old.onnx").write_bytes(b"stale")
    stdout = f"ONNX: export success, saved as '{intermediate}'\n"
    marker = f"Results saved to \x1b[1m{output.relative_to(tmp_path)}\x1b[0m\n"
    result = export_runner(
        stdout=stdout + (marker if stream == "stdout" else ""),
        stderr=marker if stream == "stderr" else "",
    )
    assert result["artifacts"] == [{"kind": "exported_model", "path": str(output)}]
    assert result["job"]["save_dir"] == str(output if directory else output.parent)
    assert result["job"]["requested_save_dir"] == str(tmp_path / "export")


def test_export_keeps_existing_save_directory_fallback(export_runner, tmp_path):
    """Older/custom CLIs without a final marker retain save-directory discovery."""
    output = tmp_path / "export" / "model.custom"
    output.parent.mkdir()
    output.write_bytes(b"mock export")
    (output.parent / "metadata.json").write_text("{}")
    result = export_runner(stdout="Export complete\n")
    assert result["artifacts"] == [{"kind": "exported_model", "path": str(output)}]


def test_missing_export_path_is_not_reported(export_runner, tmp_path):
    """A log message alone cannot establish that an artifact exists."""
    result = export_runner(stdout=f"Results saved to {tmp_path / 'missing.onnx'}\n")
    assert result["artifacts"] == []


def test_failed_export_does_not_claim_stale_output(export_runner, tmp_path):
    """Nonzero exits remain failures even if the model output path already exists."""
    output = tmp_path / "best.onnx"
    output.write_bytes(b"old model")
    result = export_runner(stdout=f"Results saved to {output}\n", returncode=1)
    assert result["status"] == "failed"
    assert not result.get("artifacts")
