from unittest.mock import patch

import pytest
import torch
from torch import nn

from ultralytics.nn.modules.moa import MoABlock
from ultralytics.nn.modules.moe.modules import OptimizedMOE
from ultralytics.utils.torch_utils import attempt_compile, unwrap_model


@pytest.mark.parametrize(
    ("module", "sample"),
    [
        (MoABlock(32, num_heads=3).eval(), torch.randn(1, 32, 8, 8)),
        (OptimizedMOE(32, 32, num_experts=2, top_k=1).eval(), torch.randn(1, 32, 8, 8)),
    ],
)
def test_mixture_modules_compile_through_existing_entrypoint(module, sample):
    compiled = attempt_compile(
        module,
        device=torch.device("cpu"),
        mode="default",
        backend="eager",
        dynamic=True,
        warmup=True,
        warmup_input=sample,
    )

    with torch.no_grad():
        eager_output = module(sample)
        compiled_output = compiled(sample)
    assert unwrap_model(compiled) is module
    assert torch.allclose(eager_output, compiled_output, atol=1e-5, rtol=1e-4)


def test_attempt_compile_returns_eager_model_when_lazy_compile_fails():
    eager = nn.Conv2d(3, 4, 1).eval()

    class LazyFailure(nn.Module):
        def __init__(self, original):
            super().__init__()
            self._orig_mod = original

        def forward(self, _):
            raise RuntimeError("backend failed during first graph execution")

    with patch("torch.compile", side_effect=lambda model, **_: LazyFailure(model)):
        result = attempt_compile(
            eager,
            device=torch.device("cpu"),
            warmup=True,
            warmup_input=torch.zeros(1, 3, 8, 8),
        )

    assert result is eager
