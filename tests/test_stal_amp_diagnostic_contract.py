"""Regression checks for shared training initialization and complete batch fingerprints."""

import pytest
import torch

from scripts.stal.diagnose_amp_training import (
    batch_signature,
    first_batch_mismatch,
    shared_state_trainer,
    state_digest,
    tensor_digest,
)
from ultralytics.models.yolo.detect import DetectionTrainer


def test_shared_state_restores_new_training_model(monkeypatch):
    """Dataset-specific rebuilt models must use the first arm's actual state, including buffers."""
    monkeypatch.setattr(DetectionTrainer, "get_model", lambda *a, **kw: torch.nn.BatchNorm1d(10))
    shared = {}
    trainer_type = shared_state_trainer(shared)
    first = trainer_type.__new__(trainer_type).get_model()
    shared["weight"].fill_(7)
    shared["running_mean"].fill_(3)
    second = trainer_type.__new__(trainer_type).get_model()
    assert torch.all(second.weight == 7)
    assert torch.all(second.running_mean == 3)
    assert state_digest(second.state_dict()) == state_digest(shared)
    assert torch.all(first.weight == 1)  # the captured state owns its storage


@pytest.mark.parametrize("key", ["img", "bboxes", "cls", "batch_idx"])
def test_batch_signature_detects_equal_sum_permutation(key):
    """Old aggregate sums miss pixel, target, class and image-association permutations."""
    batch = {name: torch.tensor([1, 2]) for name in ("img", "bboxes", "cls", "batch_idx")}
    changed = {**batch, key: batch[key].flip(0)}
    assert batch[key].sum() == changed[key].sum()
    assert batch_signature(batch) != batch_signature(changed)


def test_tensor_digest_preserves_shape_dtype_and_empty_tensors():
    """Identical bytes with different interpretation cannot count as identical inputs."""
    value = torch.tensor([1, 2], dtype=torch.int32)
    assert tensor_digest(value) != tensor_digest(value.reshape(1, 2))
    assert tensor_digest(value) != tensor_digest(value.view(torch.float32))
    assert tensor_digest(torch.empty(0)) == tensor_digest(torch.empty(0))


@pytest.mark.parametrize("left,right,expected", [([], [], None), (["a"], [], 0), (["a"], ["a", "b"], 1)])
def test_batch_mismatch_detects_length_difference(left, right, expected):
    """A missing suffix must have a concrete first mismatch index."""
    assert first_batch_mismatch(left, right) == expected
