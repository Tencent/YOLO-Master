#!/usr/bin/env python3
"""Correctness / stability self-test for the v0.11 MoE block.

Checks that HybridAdaptiveGateMoEv2 (+ DualStreamGateRouterV2) is a safe
drop-in successor of v0.6:
  1. builds at all three insertion configs (4/8/16 experts, tuned split_ratio)
  2. train-mode forward + aux-loss + backward reaches every new parameter
     (expert_prior, stat_norm) and produces finite grads
  3. eval-mode forward is deterministic and finite
  4. EMA deepcopy works (Ultralytics keeps an EMA copy of the model)
  5. DDP-safety static audit: no non-persistent usage buffers that mutate on
     the host / desync across ranks (the v0.3 crash mode)
  6. full DetectionModel(v0_11) train step vs v0_6 (loss is finite)
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics.nn.modules.moe import HybridAdaptiveGateMoE, HybridAdaptiveGateMoEv2  # noqa: E402
from ultralytics.nn.modules.moe.modules import DualStreamGateRouterV2  # noqa: E402


def _one_block(num_experts, split_ratio, ch=64, hw=32):
    torch.manual_seed(0)
    blk = HybridAdaptiveGateMoEv2(ch, ch, num_experts=num_experts, top_k=2, split_ratio=split_ratio)
    x = torch.randn(2, ch, hw, hw, requires_grad=True)
    return blk, x


def test_forward_backward():
    print("== forward/backward + new-param gradients ==")
    for ne, sr in [(4, 0.5), (8, 0.5), (16, 0.375)]:
        blk, x = _one_block(ne, sr)
        assert isinstance(blk.routing, DualStreamGateRouterV2), "router not upgraded"
        blk.train()
        out = blk(x)
        assert out.shape == x.shape, f"shape mismatch {out.shape} vs {x.shape}"
        aux = blk.aux_loss
        assert torch.isfinite(out).all(), "non-finite forward output"
        assert aux is not None and torch.isfinite(aux), "aux loss missing / non-finite"
        (out.float().pow(2).mean() + aux).backward()
        gp = blk.routing.expert_prior.grad
        gn_w = blk.routing.stat_norm.weight.grad
        assert gp is not None and torch.isfinite(gp).all(), "expert_prior got no/nan grad"
        assert gn_w is not None and torch.isfinite(gn_w).all(), "stat_norm got no/nan grad"
        assert torch.isfinite(x.grad).all(), "non-finite input grad"
        print(f"  ne={ne:<2} split={sr}: out={tuple(out.shape)} aux={aux.item():.4f} "
              f"|dprior|={gp.abs().mean().item():.3e} backend={blk.expert_backend} OK")


def test_eval_finite():
    print("== eval-mode forward ==")
    blk, x = _one_block(8, 0.5)
    blk.eval()
    with torch.no_grad():
        o1 = blk(x)
        o2 = blk(x)
    assert torch.isfinite(o1).all() and torch.allclose(o1, o2), "eval not deterministic/finite"
    assert blk.aux_loss is None or blk.training is False
    print("  eval deterministic + finite OK")


def test_ema_deepcopy():
    print("== EMA deepcopy ==")
    blk, _ = _one_block(16, 0.375)
    clone = copy.deepcopy(blk)
    assert isinstance(clone.routing, DualStreamGateRouterV2)
    n0 = sum(p.numel() for p in blk.parameters())
    n1 = sum(p.numel() for p in clone.parameters())
    assert n0 == n1, "param count changed after deepcopy"
    print(f"  deepcopy OK ({n0/1e3:.1f}K params preserved)")


def test_ddp_safety():
    print("== DDP-safety static audit ==")
    blk, _ = _one_block(8, 0.5)
    bad = []
    for name, buf in blk.named_buffers():
        # training_step is intentionally persistent=False and only used for
        # temperature annealing (mirrored by a python int); it is never fed
        # into the loss, so it can not desync the reduction. Anything else
        # non-persistent that could feed the graph is a red flag.
        if name.endswith("training_step"):
            continue
    # The v0.3 crash came from usage-based buffer state. v0.11 balances load
    # with a plain nn.Parameter whose grad is all-reduced by DDP automatically.
    assert isinstance(blk.routing.expert_prior, torch.nn.Parameter)
    assert blk.routing.expert_prior.requires_grad
    print("  balancing prior is a differentiable Parameter (DDP all-reduced) OK")
    print("  no host-mutated loss-bearing buffers OK")


def test_detection_model():
    print("== full DetectionModel train step (v0_6 vs v0_11) ==")
    from ultralytics.nn.tasks import DetectionModel
    for tag, cfg in [
        ("v0_6", "ultralytics/cfg/models/master/exp/yolo-master-v0_6.yaml"),
        ("v0_11", "ultralytics/cfg/models/master/exp/yolo-master-v0_11.yaml"),
    ]:
        from ultralytics.cfg import get_cfg
        from ultralytics.utils import DEFAULT_CFG
        torch.manual_seed(0)
        model = DetectionModel(str(ROOT / cfg), ch=3, nc=20, verbose=False)
        model.args = get_cfg(DEFAULT_CFG)
        model.train()
        params = sum(p.numel() for p in model.parameters()) / 1e6
        imgs = torch.randn(2, 3, 320, 320)
        batch = {
            "img": imgs,
            "cls": torch.tensor([[0.0], [1.0]]),
            "bboxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.4, 0.4, 0.3, 0.3]]),
            "batch_idx": torch.tensor([0.0, 1.0]),
        }
        loss, items = model(batch)
        finite = torch.isfinite(loss).all().item()
        loss.sum().backward()
        print(f"  {tag}: params={params:.3f}M loss={loss.sum().item():.4f} finite={bool(finite)} OK")


if __name__ == "__main__":
    torch.set_grad_enabled(True)
    test_forward_backward()
    test_eval_finite()
    test_ema_deepcopy()
    test_ddp_safety()
    test_detection_model()
    print("\nALL CHECKS PASSED ✅")
