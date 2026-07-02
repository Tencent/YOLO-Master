from pathlib import Path
from types import SimpleNamespace

import scripts.compare_moa_ablation as compare_moa_ablation
import scripts.issue53.probe_visdrone_batch as probe_visdrone_batch


def _train_args(amp: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        epochs=1,
        imgsz=64,
        batch=1,
        device="cpu",
        workers=0,
        seed=42,
        exist_ok=True,
        plots=False,
        cache=False,
        patience=0,
        amp=amp,
        verbose=False,
        moa_temp_factor=0.97,
        moa_min_temp=0.3,
    )


def test_moa_ablation_train_spec_uses_trainer_temperature_annealing(monkeypatch, tmp_path):
    created = []

    class FakeYOLO:
        def __init__(self, cfg):
            self.cfg = cfg
            self.callbacks = []
            self.train_kwargs = None
            created.append(self)

        def add_callback(self, event, callback):
            self.callbacks.append((event, callback))

        def train(self, **kwargs):
            self.train_kwargs = kwargs

    monkeypatch.setattr(compare_moa_ablation, "YOLO", FakeYOLO)

    compare_moa_ablation.train_spec(
        _train_args(amp=False),
        compare_moa_ablation.SPECS["v10_moa"],
        tmp_path / "data.yaml",
        tmp_path,
    )

    assert created
    assert created[0].callbacks == []
    assert created[0].train_kwargs["amp"] is False


def test_issue53_train_script_supports_disabling_amp():
    script = Path("scripts/issue53/train_visdrone_issue53.sh").read_text()

    assert "AMP=" in script
    assert "--no-amp" in script


def test_issue53_batch_probe_supports_disabling_amp(monkeypatch, tmp_path):
    created = []

    class FakeYOLO:
        def __init__(self, cfg):
            self.cfg = cfg
            self.train_kwargs = None
            created.append(self)

        def train(self, **kwargs):
            self.train_kwargs = kwargs

    monkeypatch.setattr(probe_visdrone_batch, "YOLO", FakeYOLO)
    monkeypatch.setattr(
        "sys.argv",
        [
            "probe_visdrone_batch.py",
            "--model",
            "v10_moa",
            "--batch",
            "2",
            "--project",
            str(tmp_path),
            "--no-amp",
        ],
    )

    probe_visdrone_batch.main()

    assert created
    assert created[0].train_kwargs["amp"] is False
