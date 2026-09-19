from __future__ import annotations

from pathlib import Path

import pytest
import torch

from e3_p1.benchmark import _bootstrap_median_ci, _load_config, _stats, _surrogate_training_loss


def test_surrogate_loss_reaches_nested_differentiable_tensors():
    value = torch.tensor([2.0], requires_grad=True)

    loss = _surrogate_training_loss({"one": [value], "ignored": torch.tensor([4])})
    loss.backward()

    assert float(loss.detach()) == pytest.approx(4.0)
    assert value.grad.tolist() == pytest.approx([4.0])


def test_benchmark_statistics_are_deterministic():
    stats = _stats([1.0, 2.0, 3.0])
    first = _bootstrap_median_ci([1.0, 2.0, 3.0], resamples=100, seed=7)
    second = _bootstrap_median_ci([1.0, 2.0, 3.0], resamples=100, seed=7)

    assert stats["median"] == 2.0
    assert first == second


def test_p1_config_requires_one_fixed_full_batch(tmp_path: Path):
    config_path = tmp_path / "p1.yaml"
    config_path.write_text(
        "run_id: test\nbatch_size: 2\nwarmup_pairs: 1\nmeasured_pairs: 5\n"
        "bootstrap_resamples: 10\nsample_indices: [0]\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="equal batch_size"):
        _load_config(config_path)
