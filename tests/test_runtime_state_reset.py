import torch

from ultralytics.nn.modules.moa import MoABlock
from ultralytics.nn.modules.routing_protocol import get_aux_record, reset_routing_runtime_state


def test_runtime_state_reset_removes_graph_and_snapshot():
    module = MoABlock(48, num_heads=6).train()
    module(torch.randn(1, 48, 4, 4))
    assert get_aux_record(module) is not None
    assert module.last_routing_snapshot
    reset_routing_runtime_state(module, step=99)
    assert get_aux_record(module) is None
    assert module.last_routing_snapshot == {}
    assert not module.last_aux_loss.requires_grad
