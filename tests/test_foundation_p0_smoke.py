"""Offline acceptance test for the minimum Foundation distillation chain."""

from scripts.foundation_p0_smoke import run_smoke


def test_single_stage_smoke_aligns_and_decreases_loss():
    result = run_smoke(steps=12)
    assert result["passed"] is True
    assert result["stage"] == "p4"
    assert result["teacher_frozen"] is True
    assert result["teacher_grad_free"] is True
    assert result["aligned_shape"][-2:] == result["student_shape"][-2:]
    assert result["final_loss"] < result["initial_loss"]
