from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from ultralytics.engine.validator import convert_ndjson_to_yolo_if_needed
from ultralytics.models.yolo.detect.val import DetectionValidator


def test_convert_ndjson_to_yolo_if_needed_leaves_yaml_unchanged():
    data = "ultralytics/cfg/datasets/VisDrone.yaml"
    assert convert_ndjson_to_yolo_if_needed(data) == data


def _validator_for_gather(stats, jdict):
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.metrics = SimpleNamespace(stats=stats, clear_stats=MagicMock())
    validator.jdict = jdict
    validator.dataloader = SimpleNamespace(dataset=[object()] * 5)
    return validator


def test_gather_stats_uses_symmetric_all_gather_and_merges_on_rank_zero():
    validator = _validator_for_gather({"tp": ["local"], "conf": [0.5]}, [{"image_id": 1}])
    gathered = [{"tp": ["rank0"], "conf": [0.1]}, {"tp": ["rank1"], "conf": [0.9]}]
    gathered_jdict = [[{"image_id": 0}], [{"image_id": 2}]]

    def all_gather(output, value):
        output[:] = gathered if value is validator.metrics.stats else gathered_jdict

    with (
        patch("ultralytics.models.yolo.detect.val.dist.is_available", return_value=True),
        patch("ultralytics.models.yolo.detect.val.dist.is_initialized", return_value=True),
        patch("ultralytics.models.yolo.detect.val.dist.get_world_size", return_value=2),
        patch("ultralytics.models.yolo.detect.val.dist.get_rank", return_value=0),
        patch("ultralytics.models.yolo.detect.val.dist.all_gather_object", side_effect=all_gather) as gather,
        patch("ultralytics.models.yolo.detect.val.dist.gather_object") as destination_gather,
    ):
        validator.gather_stats()

    assert gather.call_count == 2
    assert destination_gather.call_count == 0
    assert validator.metrics.stats == {"tp": ["rank0", "rank1"], "conf": [0.1, 0.9]}
    assert validator.jdict == [{"image_id": 0}, {"image_id": 2}]
    assert validator.seen == 5


def test_gather_stats_worker_enters_same_collectives_and_clears_local_state():
    validator = _validator_for_gather({"tp": ["worker"]}, [{"image_id": 1}])
    initial_stats = validator.metrics.stats

    with (
        patch("ultralytics.models.yolo.detect.val.dist.is_available", return_value=True),
        patch("ultralytics.models.yolo.detect.val.dist.is_initialized", return_value=True),
        patch("ultralytics.models.yolo.detect.val.dist.get_world_size", return_value=4),
        patch("ultralytics.models.yolo.detect.val.dist.get_rank", return_value=3),
        patch("ultralytics.models.yolo.detect.val.dist.all_gather_object") as gather,
        patch("ultralytics.models.yolo.detect.val.dist.gather_object") as destination_gather,
    ):
        validator.gather_stats()

    assert gather.call_count == 2
    assert gather.call_args_list[0].args[1] is initial_stats
    assert gather.call_args_list[1].args[1] == [{"image_id": 1}]
    assert destination_gather.call_count == 0
    assert validator.jdict == []
    validator.metrics.clear_stats.assert_called_once_with()


def test_gather_stats_skips_collectives_without_a_multi_rank_process_group():
    validator = _validator_for_gather({"tp": []}, [])

    with (
        patch("ultralytics.models.yolo.detect.val.dist.is_available", return_value=True),
        patch("ultralytics.models.yolo.detect.val.dist.is_initialized", return_value=False),
        patch("ultralytics.models.yolo.detect.val.dist.all_gather_object") as gather,
    ):
        validator.gather_stats()

    gather.assert_not_called()


def test_training_validation_accumulates_loss_items_on_accumulator_device():
    """Validation must accept model loss items from a device different from its accumulator."""
    validator = DetectionValidator.__new__(DetectionValidator)
    validator.loss = torch.zeros(2)

    class LossItems:
        def __init__(self):
            self.device = None

        def to(self, device):
            self.device = device
            return torch.tensor([1.25, 2.5])

    loss_items = LossItems()
    validator._accumulate_loss_items(loss_items)

    assert loss_items.device == validator.loss.device
    assert validator.loss.tolist() == [1.25, 2.5]
