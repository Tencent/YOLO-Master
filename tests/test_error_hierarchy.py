"""Tests for PEFT Planner exception hierarchy and RefusalError unification.

Verifies:
  - RefusalError inherits from PEFTRefusalError (unified hierarchy)
  - PEFTRefusalError carries reason and predicted_delta attributes
  - RefusalError is catchable by except PEFTRefusalError
  - RefusalError is catchable by except YOLOMasterError
  - PEFTPlannerError inherits YOLOMasterError
"""

import pytest

from ultralytics.utils.errors import (
    YOLOMasterError,
    PEFTPlannerError,
    PEFTRefusalError,
)
from ultralytics.utils.lora.planner import RefusalError


class TestRefusalErrorHierarchy:
    """Verify RefusalError is properly unified with PEFTRefusalError."""

    def test_refusal_inherits_peft_refusal(self):
        assert issubclass(RefusalError, PEFTRefusalError)

    def test_refusal_inherits_yolomaster(self):
        """Caller can catch RefusalError via the base YOLOMasterError."""
        assert issubclass(RefusalError, YOLOMasterError)

    def test_refusal_catchable_as_peft_refusal(self):
        try:
            raise RefusalError("test refusal")
        except PEFTRefusalError as exc:
            assert "test refusal" in str(exc) or "test" in str(exc)

    def test_refusal_catchable_as_yolomaster(self):
        try:
            raise RefusalError("test refusal")
        except YOLOMasterError:
            pass  # expected

    def test_refusal_catchable_as_peft_planner(self):
        try:
            raise RefusalError("test refusal")
        except PEFTPlannerError:
            pass  # expected


class TestPEFTRefusalErrorAttributes:
    """Verify PEFTRefusalError carries structured data."""

    def test_default_reason(self):
        err = PEFTRefusalError()
        assert err.reason == ""
        assert err.predicted_delta == 0.0

    def test_custom_reason(self):
        err = PEFTRefusalError(reason="incompatible architecture")
        assert err.reason == "incompatible architecture"

    def test_predicted_delta(self):
        err = PEFTRefusalError(reason="low delta", predicted_delta=-0.05)
        assert err.predicted_delta == -0.05

    def test_message_includes_reason(self):
        err = PEFTRefusalError(reason="test reason")
        assert "test reason" in str(err)


class TestShapeMismatchError:
    """Verify ShapeMismatchError structured attributes."""

    def test_attributes_preserved(self):
        from ultralytics.utils.errors import ShapeMismatchError
        err = ShapeMismatchError(expected=(1, 64, 16, 16), actual=(1, 32, 16, 16),
                                  context="router")
        assert err.expected == (1, 64, 16, 16)
        assert err.actual == (1, 32, 16, 16)
        assert err.context == "router"

    def test_inherits_yolomaster(self):
        from ultralytics.utils.errors import ShapeMismatchError
        assert issubclass(ShapeMismatchError, YOLOMasterError)


class TestMoERouterError:
    """Verify MoERouterError is in the hierarchy."""

    def test_inherits_yolomaster(self):
        from ultralytics.utils.errors import MoERouterError
        assert issubclass(MoERouterError, YOLOMasterError)
