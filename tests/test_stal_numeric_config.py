"""Check numerical experiment parameters cannot silently turn mechanisms into no-ops."""

import pytest

from ultralytics.cfg import CFG_FLOAT_KEYS, CFG_FRACTION_KEYS, CFG_INT_KEYS, get_cfg
from ultralytics.utils.tal import TaskAlignedAssigner


@pytest.mark.parametrize("key", sorted(k for k in CFG_FLOAT_KEYS | CFG_FRACTION_KEYS if k.startswith("stal_")))
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), True])
def test_stal_numeric_parameters_reject_nonfinite_and_boolean(key, value):
    """NaN comparisons and bool-as-int must not bypass the central configuration contract."""
    with pytest.raises((ValueError, TypeError)):
        get_cfg(overrides={key: value})


@pytest.mark.parametrize("key", sorted(k for k in CFG_INT_KEYS if k.startswith("stal_")))
def test_stal_integer_parameters_reject_boolean(key):
    """Python booleans are integers but are not experiment counts."""
    with pytest.raises(TypeError):
        get_cfg(overrides={key: True})


@pytest.mark.parametrize(
    "key",
    [
        "stal_relaxation",
        "stal_min_size_stride_ratio",
        "stal_warmup_epochs",
        "stal_rescue_floor_decay_epochs",
        "stal_nwd_constant",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_direct_assigner_rejects_nonfinite_unbounded_parameters(key, value):
    """Research scripts constructing the assigner directly must also reject non-finite parameters."""
    with pytest.raises(ValueError, match="finite"):
        TaskAlignedAssigner(**{key: value})


@pytest.mark.parametrize(
    "key", sorted(k for k in CFG_FLOAT_KEYS | CFG_FRACTION_KEYS | CFG_INT_KEYS if k.startswith("stal_"))
)
def test_direct_assigner_rejects_boolean_numeric_parameters(key):
    """Direct construction must preserve the same bool-versus-number boundary as central configuration."""
    with pytest.raises((TypeError, ValueError)):
        TaskAlignedAssigner(**{key: True})


@pytest.mark.parametrize(
    "key", ["stal_enabled", "stal_stats", "stal_min_candidate_guarantee", "stal_zero_positive_rescue"]
)
def test_direct_assigner_rejects_nonboolean_flags(key):
    """Direct construction must not accept numeric truth values for boolean experiment switches."""
    with pytest.raises(TypeError, match="must be a boolean"):
        TaskAlignedAssigner(**{key: 1})
