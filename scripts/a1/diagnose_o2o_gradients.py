"""CPU-only synthetic probe of the one-to-one detach boundary (not an AP evaluation)."""

import argparse
import json
from pathlib import Path

import torch

from scripts.a1.diagnostic_hooks import capture_head_inputs
from ultralytics.nn.modules.head import Detect


def diagnose(alpha=0.0, seed=260829):
    """Measure gradients on identical synthetic features without modifying a checkpoint."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head = Detect(nc=3, reg_max=1, end2end=True, ch=(8, 16, 32)).train()
        for module in head.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.eval()
        x = [torch.randn(1, c, s, s, requires_grad=True) for c, s in ((8, 8), (16, 4), (32, 2))]
        head.p2_o2o_gradient_alpha = alpha
        with capture_head_inputs(head) as captured:
            outputs = head(x)
            rows = {}
            for branch in ("one2many", "one2one"):
                # This output-energy probe isolates connectivity, not a detection loss/assignment.
                energy = sum(outputs[branch][key].square().mean() for key in ("boxes", "scores"))
                gradients = torch.autograd.grad(energy, captured, allow_unused=True, retain_graph=True)
                rows[branch] = [None if g is None else float(g.norm().detach()) for g in gradients]
        return {
            "probe": "synthetic_output_energy_not_detection_loss",
            "seed": seed,
            "alpha": alpha,
            "input_gradient_l2": rows,
            "hooks_remaining": len(head._forward_pre_hooks) + len(head._forward_hooks),
        }


def main():
    """Print a small JSON diagnostic; only write a file when explicitly requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=260829)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(diagnose(args.alpha, args.seed), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
