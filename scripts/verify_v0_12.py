#!/usr/bin/env python3
"""Correctness checks for the v0.12 OptimalHybridGateMoE block and full model."""
import os
import sys
import copy
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from ultralytics.nn.modules.moe import OptimalHybridGateMoE, HybridAdaptiveGateMoE
from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY
from ultralytics.nn.tasks import DetectionModel


def block_checks():
    print("== Block-level checks (OptimalHybridGateMoE) ==")
    for experts, split in [(4, 0.5), (8, 0.5), (16, 0.375)]:
        c = 64
        blk = OptimalHybridGateMoE(c, c, num_experts=experts, top_k=2, split_ratio=split)
        blk.train()
        x = torch.randn(2, c, 32, 32, requires_grad=True)
        y = blk(x)
        assert y.shape == x.shape, f"shape mismatch {y.shape} vs {x.shape}"
        aux = blk.aux_loss
        loss = y.float().mean() + (aux if aux is not None else 0.0)
        loss.backward()
        # router prior must receive gradient (proves it is graph-connected)
        prior = blk.routing.expert_prior
        assert prior.grad is not None, "expert_prior got no gradient"
        assert blk.routing.stat_norm.weight.grad is not None, "stat_norm got no gradient"
        # refine_scale must receive gradient
        assert blk.refine_scale.grad is not None, "refine_scale got no gradient"
        # router noise buffer must be advancing
        noise_prog = float(blk.routing._noise_progress)
        assert noise_prog > 0, f"noise_progress not advancing: {noise_prog}"
        backend = blk.expert_backend
        print(f"  experts={experts:>2} split={split} backend={backend:<14} "
              f"out={tuple(y.shape)} aux={float(aux):.4f} "
              f"prior.grad_norm={prior.grad.norm():.4f} "
              f"refine.grad={blk.refine_scale.grad.item():.4f} "
              f"noise_prog={noise_prog:.4f} OK")

        # eval-mode forward (no aux loss path, no noise)
        blk.eval()
        with torch.no_grad():
            ye = blk(torch.randn(2, c, 32, 32))
        assert ye.shape == (2, c, 32, 32)

        # deepcopy must not raise
        blk_copy = copy.deepcopy(blk)
        assert isinstance(blk_copy, OptimalHybridGateMoE)
    print("  block checks passed\n")


def model_checks():
    print("== Model-level checks (v0_12 vs v0_6) ==")
    cfgs = {
        "v0_6": ROOT / "ultralytics/cfg/models/master/v0_6/det/yolo-master-n.yaml",
        "v0_12": ROOT / "ultralytics/cfg/models/master/v0_12/det/yolo-master-n.yaml",
    }
    params = {}
    for name, cfg in cfgs.items():
        model = DetectionModel(str(cfg), ch=3, nc=20, verbose=False)
        params[name] = sum(p.numel() for p in model.parameters())
        model.train()
        x = torch.randn(1, 3, 320, 320)
        out = model(x)
        aux_terms = [v for v in MOE_LOSS_REGISTRY.values() if torch.is_tensor(v)]
        print(f"  {name:<6} params={params[name]/1e6:.4f}M  "
              f"moe_blocks_with_aux={len(aux_terms)}  "
              f"out_shape={[o.shape for o in out] if isinstance(out, (list, tuple)) else out.shape}")
    delta = (params["v0_12"] - params["v0_6"]) / 1e6
    print(f"  param delta v0_12 - v0_6 = {delta:+.4f}M "
          f"({100*delta*1e6/params['v0_6']:+.3f}%)")
    print("  model checks passed\n")


if __name__ == "__main__":
    torch.manual_seed(0)
    block_checks()
    model_checks()
    print("ALL CHECKS PASSED")
