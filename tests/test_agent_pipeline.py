"""Dependency and model-lineage contracts for Agent experiment pipelines."""

from dataclasses import fields
from pathlib import Path
from unittest.mock import Mock

import pytest

from agent.runtime.cli import pipeline
from agent.runtime.cli.contract import response, write_manifest
from agent.runtime.cli.normalize import normalize_request


def dependencies(tmp_path):
    """Use real normalization/manifests and in-memory executors, never model execution."""
    values = {
        field.name: Mock(side_effect=AssertionError(f"Unexpected dependency: {field.name}"))
        for field in fields(pipeline.PipelineDeps)
    }
    values.update(
        normalize_request=normalize_request,
        is_dry_run=lambda request: request.get("policy", {}).get("dry_run", False),
        response=response,
        write_manifest=write_manifest,
        best_checkpoint=lambda payload: next(
            (item["path"] for item in payload.get("artifacts", []) if item.get("label") == "best"), None
        ),
        run_train_like=Mock(
            return_value={
                "status": "ok",
                "artifacts": [{"kind": "checkpoint", "label": "best", "path": str(tmp_path / "trained.pt")}],
            }
        ),
        run_val=Mock(return_value={"status": "ok"}),
        run_export=Mock(return_value={"status": "ok"}),
    )
    return pipeline.PipelineDeps(**values)


def request(tmp_path, **params):
    """Build a normalized experiment whose files stay in the test directory."""
    return normalize_request(
        {
            "skill": "yolo.pipeline.experiment",
            "inputs": {"model": str(tmp_path / "initial.pt")},
            "params": params or {"train": {}, "val": {}, "export": {}},
            "artifacts": {"project": str(tmp_path), "name": "pipeline"},
        }
    )


def test_train_checkpoint_reaches_val_export_and_manifest(tmp_path):
    """The manifest and every dependent executor must agree on the model actually used."""
    import json

    deps = dependencies(tmp_path)
    result = pipeline.run_experiment_pipeline(request(tmp_path), deps)
    trained = str(tmp_path / "trained.pt")
    assert deps.run_train_like.call_args.args[0]["inputs"]["model"] == str(tmp_path / "initial.pt")
    assert deps.run_val.call_args.args[0]["inputs"]["model"] == trained
    assert deps.run_export.call_args.args[0]["inputs"]["model"] == trained
    assert result["pipeline"]["stage_input_models"] == {
        "train": str(tmp_path / "initial.pt"),
        "val": trained,
        "export": trained,
    }
    manifest = json.loads(Path(result["manifest"]).read_text())
    assert manifest["result"]["pipeline"]["stage_input_models"] == result["pipeline"]["stage_input_models"]


def test_explicit_stage_model_overrides_checkpoint_only_for_that_stage(tmp_path):
    """An explicit comparison model must not change the next stage's inherited checkpoint."""
    deps = dependencies(tmp_path)
    explicit = str(tmp_path / "comparison.pt")
    experiment = request(tmp_path, train={}, val={"inputs": {"model": explicit}}, export={})
    original = experiment["inputs"]["model"]
    pipeline.run_experiment_pipeline(experiment, deps)
    assert deps.run_val.call_args.args[0]["inputs"]["model"] == explicit
    assert deps.run_export.call_args.args[0]["inputs"]["model"] == str(tmp_path / "trained.pt")
    assert experiment["inputs"]["model"] == original


def test_normalized_initial_profile_does_not_override_trained_checkpoint(tmp_path):
    """Real profile normalization must not cause later stages to reload the initial architecture."""
    deps = dependencies(tmp_path)
    experiment = normalize_request(
        {
            "skill": "yolo.pipeline.experiment",
            "inputs": {"profile": "26/yolo26-master-mot-n"},
            "params": {"train": {}, "val": {}, "export": {}},
            "artifacts": {"project": str(tmp_path), "name": "profile-pipeline"},
        }
    )
    assert experiment["profile"]["profile_id"] == "26/yolo26-master-mot-n"
    pipeline.run_experiment_pipeline(experiment, deps)
    for executor in (deps.run_val, deps.run_export):
        actual = executor.call_args.args[0]
        assert actual["inputs"]["model"] == str(tmp_path / "trained.pt")
        assert "profile" not in actual["inputs"]
        assert "profile" not in actual


@pytest.mark.parametrize("async_value", [True, "true", "1", "yes", "on", 1])
@pytest.mark.parametrize("dry_run", [False, True])
def test_nested_async_is_rejected_before_any_stage_or_progress(tmp_path, async_value, dry_run):
    """Reject even a later asynchronous stage before training can have side effects."""
    deps = dependencies(tmp_path)
    experiment = request(tmp_path, train={}, val={"policy": {"async": async_value}})
    experiment["policy"]["dry_run"] = dry_run
    with pytest.raises(ValueError, match="stage 'val'.*whole pipeline"):
        pipeline.run_experiment_pipeline(experiment, deps)
    deps.run_train_like.assert_not_called()
    deps.run_val.assert_not_called()
    assert not list(tmp_path.rglob("progress.jsonl"))


def test_outer_async_is_not_inherited_by_pipeline_stages(tmp_path):
    """Outer async selects the dispatcher submission; dependency executors stay synchronous."""
    deps = dependencies(tmp_path)
    experiment = request(tmp_path)
    experiment["policy"]["async"] = True
    result = pipeline.run_experiment_pipeline(experiment, deps)
    assert result["status"] == "ok"
    for executor in (deps.run_train_like, deps.run_val, deps.run_export):
        assert executor.call_args.args[0]["policy"]["async"] is False
    assert experiment["policy"]["async"] is True


@pytest.mark.parametrize("stage_status", ["running", "pending", "failed", "cancelled", None])
def test_nonterminal_or_failed_stage_never_advances_dependents(tmp_path, stage_status):
    """An unexpected asynchronous result remains evidence, never a completed dependency."""
    deps = dependencies(tmp_path)
    deps.run_train_like.return_value = {"status": stage_status}
    result = pipeline.run_experiment_pipeline(request(tmp_path), deps)
    assert result["status"] == "failed"
    assert result["pipeline"]["failed_stage"] == "train"
    assert result["stages"]["train"]["status"] == stage_status
    deps.run_val.assert_not_called()
    deps.run_export.assert_not_called()
