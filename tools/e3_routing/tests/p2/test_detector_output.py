import numpy as np
import pytest

from e3_p2.detector_output import detector_output_comparison, detector_output_tensors


def _output(offset: float = 0.0):
    return (
        np.full((1, 300, 6), 1.0 + offset),
        {
            "one2one": {
                "scores": np.full((1, 4, 8), 2.0 + offset),
                "boxes": np.full((1, 4, 4), 3.0 + offset),
            },
            "one2many": {
                "scores": np.full((1, 7, 8), 4.0 + offset),
                "boxes": np.full((1, 7, 4), 5.0 + offset),
            },
        },
    )


def test_detector_output_comparison_has_fixed_primary_and_known_delta():
    comparison = detector_output_comparison(_output(), _output(0.25))
    assert comparison["primary_tensor"] == "one2one_scores"
    assert set(comparison["tensors"]) == {
        "decoded_top300",
        "one2one_scores",
        "one2one_boxes",
        "one2many_scores",
        "one2many_boxes",
    }
    for metrics in comparison["tensors"].values():
        assert metrics["mean_absolute_change"] == pytest.approx(0.25)
        assert metrics["root_mean_square_change"] == pytest.approx(0.25)
        assert metrics["maximum_absolute_change"] == pytest.approx(0.25)


def test_detector_output_extraction_rejects_missing_field_and_shape_change():
    missing = _output()
    del missing[1]["one2one"]["scores"]
    with pytest.raises(TypeError, match="missing one2one.scores"):
        detector_output_tensors(missing)
    changed = _output()
    changed[1]["one2one"]["scores"] = np.zeros((1, 5, 8))
    with pytest.raises(RuntimeError, match="shape changed"):
        detector_output_comparison(_output(), changed)
