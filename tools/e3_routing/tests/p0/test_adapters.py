from __future__ import annotations

import json
import math
from pathlib import Path

import jsonschema
import pytest
import torch

from e3_p0.adapters import SCHEMA_VERSION, adapt_snapshot, routing_metrics


class FakeMoE(torch.nn.Module):
    __module__ = "ultralytics.nn.modules.moe.fake"

    def __init__(self, usage=(0.7, 0.2, 0.1)):
        super().__init__()
        self.num_experts = 3
        self.top_k = 2
        self.register_buffer("aux_loss", torch.tensor(0.125))
        self.usage = usage
        self.last_routing_snapshot = {}

    def forward(self, value):
        self.last_routing_snapshot = {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "expert_usage": torch.tensor(self.usage),
            "mean_router_probs": torch.tensor(self.usage),
            "mean_topk_weight": torch.tensor([0.6, 0.4]),
            "aux_loss": self.aux_loss,
        }
        return value * 2


@pytest.mark.parametrize(
    ("usage", "expected_entropy", "expected_gini"),
    [
        ([0.25, 0.25, 0.25, 0.25], 1.0, 0.0),
        ([1.0, 0.0, 0.0, 0.0], 0.0, 0.75),
    ],
)
def test_known_routing_metrics(usage, expected_entropy, expected_gini):
    metrics = routing_metrics(usage)
    assert metrics["expert_load_sum"] == pytest.approx(1.0)
    assert metrics["entropy_normalized"] == pytest.approx(expected_entropy)
    assert metrics["load_gini"] == pytest.approx(expected_gini)


def test_snapshot_normalizes_and_validates_against_json_schema():
    module = FakeMoE()
    output = module(torch.ones(1, 3, 2, 2))
    event = adapt_snapshot(
        family="moe",
        run_id="unit-test",
        sequence=0,
        module_name="route",
        module=module,
        snapshot=module.last_routing_snapshot,
        input_shape=[1, 3, 2, 2],
        output_shapes=[list(output.shape)],
    )
    schema_path = Path(__file__).resolve().parents[2] / "schemas" / "routing-event.schema.json"
    jsonschema.Draft202012Validator(json.loads(schema_path.read_text(encoding="utf-8"))).validate(event)
    assert event["schema_version"] == SCHEMA_VERSION
    assert event["routing"]["expert_load"] == pytest.approx([0.7, 0.2, 0.1])
    assert event["routing"]["mixing_weights"]["source"] == "mean_topk_weight"
    assert event["aux_loss"]["finite"] is True
    assert math.isfinite(event["routing"]["entropy_nats"])


def test_missing_aux_loss_is_explicit_not_silently_zero():
    module = FakeMoE()
    module.last_routing_snapshot = {
        "num_experts": 3,
        "top_k": 2,
        "expert_usage": torch.tensor([0.4, 0.3, 0.3]),
    }
    event = adapt_snapshot(
        family="moe",
        run_id="unit-test",
        sequence=0,
        module_name="route",
        module=module,
        snapshot=module.last_routing_snapshot,
        input_shape=None,
        output_shapes=[],
    )
    assert event["aux_loss"] == {
        "state": "unavailable",
        "source": None,
        "value": None,
        "finite": None,
        "mode": "eval_observation",
    }
