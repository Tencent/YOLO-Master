from copy import deepcopy

import torch
from torch import nn

from ultralytics.nn.modules.moe.weight_verify import safe_load_with_verify, verify_moe_weights


class LoadAwareLinear(nn.Linear):
    def load(self, weights, verbose=True):
        assert isinstance(weights, nn.Module)
        self.load_state_dict(weights.state_dict())


def test_moe_weight_verify_uses_ema_from_standard_ultralytics_checkpoint(tmp_path):
    source = nn.Linear(2, 2)
    checkpoint = tmp_path / "standard.pt"
    torch.save({"model": None, "ema": deepcopy(source)}, checkpoint)

    target = LoadAwareLinear(2, 2)
    report = verify_moe_weights(target, checkpoint, verbose=False)
    assert report.matched_keys == len(target.state_dict())
    assert not report.missing_in_checkpoint

    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()
    safe_load_with_verify(target, checkpoint, verbose=False)
    assert torch.equal(target.weight, source.weight)
    assert torch.equal(target.bias, source.bias)
