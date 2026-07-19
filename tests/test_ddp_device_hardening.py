import os
from unittest.mock import patch
import pytest
import torch
from ultralytics.engine.trainer import _distributed_env, _validate_cuda_ddp_device

KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE")


def env(values):
    d = {k: v for k, v in os.environ.items() if k not in KEYS}
    d.update(values)
    return patch.dict(os.environ, d, clear=True)


def test_none():
    with env({}):
        assert _distributed_env() is None


def test_valid():
    with env({"RANK": "3", "LOCAL_RANK": "1", "WORLD_SIZE": "8"}):
        assert _distributed_env() == (3, 1, 8)


@pytest.mark.parametrize(
    "v",
    [
        {"RANK": "0"},
        {"RANK": "x", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
        {"RANK": "2", "LOCAL_RANK": "0", "WORLD_SIZE": "2"},
        {"RANK": "0", "LOCAL_RANK": "-1", "WORLD_SIZE": "2"},
    ],
)
def test_invalid(v):
    with env(v), pytest.raises(RuntimeError):
        _distributed_env()


def test_cpu_rejected():
    with patch("torch.cuda.is_available", return_value=False), pytest.raises(RuntimeError):
        _validate_cuda_ddp_device(torch.device("cpu"), (0, 0, 2))


def test_ordinal_rejected():
    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.device_count", return_value=1
    ), pytest.raises(RuntimeError):
        _validate_cuda_ddp_device(torch.device("cuda", 0), (0, 1, 2))
