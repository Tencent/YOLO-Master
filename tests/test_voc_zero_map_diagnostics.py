import importlib.util
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "scripts" / "diagnose_voc_zero_map.py"
SPEC = importlib.util.spec_from_file_location("diagnose_voc_zero_map", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_first_prediction_tensor_prefers_decoded_predictions_over_raw_dfl_boxes():
    decoded = torch.zeros(1, 24, 8400)
    raw_boxes = torch.zeros(1, 64, 8400)
    raw_scores = torch.zeros(1, 20, 8400)

    selected = MODULE.first_prediction_tensor((decoded, {"boxes": raw_boxes, "scores": raw_scores}), expected_nc=20)

    assert selected is decoded


def test_prediction_scores_uses_decoded_channel_layout():
    decoded = torch.zeros(1, 24, 8400)

    scores = MODULE.prediction_scores(decoded, expected_nc=20)

    assert scores is not None
    assert scores.shape == (1, 20, 8400)
