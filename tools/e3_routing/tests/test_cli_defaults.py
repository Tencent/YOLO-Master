from __future__ import annotations

import sys
from pathlib import Path

from e3_p0.runner import PROJECT_ROOT as P0_ROOT
from e3_p0.runner import parse_args as parse_p0_args
from e3_p1.benchmark import PROJECT_ROOT as P1_ROOT
from e3_p1.benchmark import parse_args as parse_p1_args
from e3_p2 import runner as p2_runner
from e3_p2.__main__ import PROJECT_ROOT as P2_ROOT
from e3_p2.__main__ import build_parser


def test_p0_default_config_exists(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["e3_p0"])
    config = parse_p0_args().config
    assert config == P0_ROOT / "configs" / "p0" / "p0.yaml"
    assert config.is_file()


def test_p1_default_config_exists(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["e3_p1"])
    config = parse_p1_args().config
    assert config == P1_ROOT / "configs" / "p1" / "p1.yaml"
    assert config.is_file()


def test_all_p2_default_configs_exist() -> None:
    parser = build_parser()
    expected = {
        "run": "p2.yaml",
        "robustness": "robustness.yaml",
        "appearance": "appearance.yaml",
        "layer-drilldown": "layer_drilldown.yaml",
        "image-scale": "image_scale.yaml",
        "image-driver": "image_driver.yaml",
        "dose-response": "dose_response.yaml",
        "output-coupling": "output_coupling.yaml",
        "dose-holdout": "dose_holdout.yaml",
        "blur-residual": "blur_residual.yaml",
    }
    for command, filename in expected.items():
        config: Path = parser.parse_args([command]).config
        assert config == P2_ROOT / "configs" / "p2" / filename
        assert config.is_file()


def test_direct_p2_runner_uses_existing_default_config(monkeypatch) -> None:
    seen: dict[str, Path] = {}

    def fake_run(config: Path, **_: object) -> Path:
        seen["config"] = config
        return config

    monkeypatch.setattr(p2_runner, "run", fake_run)
    p2_runner.main([])
    assert seen["config"] == P2_ROOT / "configs" / "p2" / "p2.yaml"
    assert seen["config"].is_file()
