"""Regression tests for reproduction-script W&B callbacks."""

from types import SimpleNamespace

from scripts.reproduce._reproduce_common import DatasetSpec, ModelSpec, _make_wandb_callbacks


def test_epoch_metrics_are_committed_immediately(monkeypatch):
    """The final log call for an epoch must flush all metrics for that step."""
    log_calls = []
    run = SimpleNamespace(
        url="https://wandb.invalid/run/test",
        log=lambda data, **kwargs: log_calls.append((data, kwargs)),
        finish=lambda: None,
    )
    wandb = SimpleNamespace(init=lambda **_kwargs: run)
    monkeypatch.setitem(__import__("sys").modules, "wandb", wandb)

    args = SimpleNamespace(
        epochs=2,
        imgsz=800,
        batch=8,
        seed=42,
        wandb_project="test-project",
        wandb_entity="",
        wandb_mode="online",
    )
    callbacks = _make_wandb_callbacks(
        "test-run",
        DatasetSpec(name="VisDrone", data="VisDrone.yaml", project="runs/test"),
        ModelSpec(name="v0.1-N", cfg="model.yaml"),
        args,
        dense_val=False,
    )
    trainer = SimpleNamespace(
        epoch=0,
        tloss=object(),
        metrics={"metrics/mAP50(B)": 0.25, "metrics/mAP50-95(B)": 0.125},
        label_loss_items=lambda _loss, prefix: {f"{prefix}/box_loss": 1.5, f"{prefix}/cls_loss": 0.75},
    )

    callbacks["on_train_start"](trainer)
    callbacks["on_fit_epoch_end"](trainer)

    assert len(log_calls) == 1
    data, kwargs = log_calls[0]
    assert kwargs == {"step": 1, "commit": True}
    assert data["epoch"] == 1
    assert data["mAP50"] == 0.25
    assert data["mAP50-95"] == 0.125
    assert data["train/box_loss"] == 1.5
    assert data["train/cls_loss"] == 0.75
