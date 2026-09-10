from __future__ import annotations

from pathlib import Path

import pytest
import torch

from e3_p1.strengthened import (
    CONDITIONS,
    _assert_step_equivalence,
    _condition_order,
    _full_detection_step,
    _label_path,
    _load_strengthened_config,
    _load_yolo_labels,
    _validate_event_stream,
)


def test_three_condition_order_is_position_balanced():
    orders = [_condition_order(index) for index in range(6)]

    assert len({tuple(order) for order in orders}) == 6
    for condition in CONDITIONS:
        assert [sum(order[position] == condition for order in orders) for position in range(3)] == [2, 2, 2]


def test_strengthened_config_requires_balanced_measured_pairs(tmp_path: Path):
    config_path = tmp_path / "strengthened.yaml"
    config_path.write_text(
        "run_id: test\nbatch_size: 2\nwarmup_pairs: 6\nbootstrap_resamples: 10\n"
        "sample_batches: [[0, 1], [2, 3]]\nprofiles:\n  mot:\n    measured_pairs: 31\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="divisible by 6"):
        _load_strengthened_config(config_path)


def test_yolo_label_path_and_parser(tmp_path: Path):
    image_path = tmp_path / "images" / "val" / "sample.jpg"
    label_path = tmp_path / "labels" / "val" / "sample.txt"
    label_path.parent.mkdir(parents=True)
    label_path.write_text("2 0.5 0.5 0.25 0.75\n", encoding="utf-8")

    assert _label_path(image_path) == label_path
    labels = _load_yolo_labels(label_path, num_classes=3)
    assert labels.shape == (1, 5)
    assert labels.tolist()[0] == pytest.approx([2.0, 0.5, 0.5, 0.25, 0.75])


def test_yolo_label_parser_rejects_out_of_range_class(tmp_path: Path):
    label_path = tmp_path / "bad.txt"
    label_path.write_text("3 0.5 0.5 0.2 0.2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid class id"):
        _load_yolo_labels(label_path, num_classes=3)


def test_event_stream_requires_contiguous_global_sequence():
    result = _validate_event_stream(
        [{"sequence": 0}, {"sequence": 1}, {"sequence": 2}], expected_count=3, label="test"
    )
    assert result["contiguous_from_zero"] is True
    assert result["last_sequence"] == 2

    with pytest.raises(RuntimeError, match="not contiguous"):
        _validate_event_stream(
            [{"sequence": 0}, {"sequence": 1}, {"sequence": 0}], expected_count=3, label="test"
        )


class _ToyDetectionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))

    def loss(self, batch):
        value = ((self.weight * batch["img"]) - batch["target"]).square().mean()
        losses = torch.stack((value, value * 0.5, value * 0.25))
        return losses, losses.detach()


def test_full_detection_step_updates_identical_models_equivalently():
    first = _ToyDetectionModel()
    second = _ToyDetectionModel()
    second.load_state_dict(first.state_dict())
    first_optimizer = torch.optim.SGD(first.parameters(), lr=0.01, momentum=0.9)
    second_optimizer = torch.optim.SGD(second.parameters(), lr=0.01, momentum=0.9)
    batch = {"img": torch.tensor([1.0, 2.0]), "target": torch.tensor([0.0, 1.0])}

    first_result = _full_detection_step(
        first, first_optimizer, batch, seed=7, torch_module=torch, gradient_clip_norm=10.0
    )
    second_result = _full_detection_step(
        second, second_optimizer, batch, seed=7, torch_module=torch, gradient_clip_norm=10.0
    )
    _assert_step_equivalence(
        [first_result],
        [second_result],
        family="toy",
        condition="capture",
        config={
            "loss_match_rtol": 1e-6,
            "loss_match_atol": 1e-6,
            "gradient_match_rtol": 1e-6,
            "gradient_match_atol": 1e-6,
        },
    )

    assert first.weight.detach().tolist() == pytest.approx(second.weight.detach().tolist())
