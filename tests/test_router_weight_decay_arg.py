"""`moe_router_weight_decay` / `--router-decay`: an additive knob for the router parameter group's weight decay.

The default (unset) must leave the optimizer exactly as before the knob existed: every router-named parameter in the
router group with the scaled global weight decay.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from torch import nn

from ultralytics.cfg import DEFAULT_CFG_DICT, get_cfg
from ultralytics.engine.trainer import BaseTrainer

ROOT = Path(__file__).resolve().parents[1]


class _RouterFixture(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 1)
        self.router = nn.Sequential(nn.Conv2d(4, 4, 1), nn.GroupNorm(2, 4), nn.Conv2d(4, 2, 1))


def _trainer() -> BaseTrainer:
    trainer = object.__new__(BaseTrainer)
    trainer.args = SimpleNamespace(moe_router_lr_scale=0.5, lora_lr_mult=1.0, warmup_bias_lr=0.1)
    trainer.data = {"nc": 80}
    trainer.adapter_controller = SimpleNamespace(active=False)
    return trainer


def _groups(router_decay=None):
    model = _RouterFixture()
    kwargs = {} if router_decay is None else {"router_decay": router_decay}
    opt = _trainer().build_optimizer(model, name="SGD", lr=0.01, momentum=0.9, decay=0.002, iterations=100, **kwargs)
    names = {id(p): n for n, p in model.named_parameters()}
    return {g["param_group"]: (g, {names[id(p)] for p in g["params"]}) for g in opt.param_groups}


def test_default_keeps_router_group_at_global_decay():
    """Without the argument the router group is unchanged: all router-named parameters, decay == global decay."""
    router_group, router_names = _groups()["router"]
    assert router_names == {f"router.{i}.{p}" for i in (0, 1, 2) for p in ("weight", "bias")}
    assert router_group["weight_decay"] == pytest.approx(0.002)
    assert router_group["lr"] == pytest.approx(0.005)


@pytest.mark.parametrize("router_decay", [0.0, 0.0005])
def test_router_decay_argument_only_changes_the_router_group(router_decay):
    groups = _groups(router_decay)
    assert groups["router"][0]["weight_decay"] == pytest.approx(router_decay)
    assert groups["weight"][0]["weight_decay"] == pytest.approx(0.002)
    assert groups["bias"][0]["weight_decay"] == 0.0
    assert groups["router"][1] == _groups()["router"][1]  # same membership as the default


def test_config_key_is_optional_and_unset_by_default():
    assert "moe_router_weight_decay" in DEFAULT_CFG_DICT
    assert DEFAULT_CFG_DICT["moe_router_weight_decay"] is None
    assert get_cfg(overrides={"moe_router_weight_decay": 0.0}).moe_router_weight_decay == 0.0
    assert get_cfg().moe_router_weight_decay is None
    with pytest.raises(TypeError):
        get_cfg(overrides={"moe_router_weight_decay": "zero"})


def test_reproduce_parser_flag_defaults_to_unset():
    sys.path.insert(0, str(ROOT / "scripts" / "reproduce"))
    try:
        from _reproduce_common import DatasetSpec, build_parser
    finally:
        sys.path.pop(0)
    parser = build_parser(DatasetSpec(name="COCO", data="coco.yaml", project="runs/reproduce/coco"))
    assert parser.parse_args([]).router_decay is None
    assert parser.parse_args(["--router-decay", "0"]).router_decay == 0.0
    assert parser.parse_args(["--router-decay", "0.0005"]).router_decay == pytest.approx(0.0005)


def test_scaled_router_decay_matches_weight_decay_scaling():
    """The trainer scales the argument by batch x accumulate / nbs exactly as it scales weight_decay."""
    trainer = object.__new__(BaseTrainer)
    trainer.args = get_cfg(overrides={"weight_decay": 0.0005, "moe_router_weight_decay": 0.0005, "nbs": 64})
    trainer.batch_size, trainer.accumulate = 256, 1
    scale = trainer.batch_size * trainer.accumulate / trainer.args.nbs
    assert trainer.args.weight_decay * scale == pytest.approx(0.002)
    assert float(trainer.args.moe_router_weight_decay) * scale == pytest.approx(0.002)
