"""Two-rank CPU/Gloo continuous-training gate for a real routed module."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ultralytics.nn.modules.moe.modules import OptimizedMOE
from ultralytics.utils import WINDOWS
from ultralytics.utils.torchrun import disable_libuv_rendezvous


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2, f"P0 gate requires exactly two ranks, got {world}"
    torch.set_num_threads(1)
    if WINDOWS:
        disable_libuv_rendezvous()
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(1234)
        model = OptimizedMOE(8, 8, num_experts=2, top_k=2)
        ddp = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
        # A constant image is a degenerate normalization case: BatchNorm and
        # the experts' GroupNorm can legitimately remove the entire signal.
        # Keep the fixture deterministic, but include spatial/channel variation
        # so this gate measures real routed gradients on every backend.
        pattern = torch.linspace(-1.0, 1.0, steps=4 * 8 * 2 * 2, dtype=torch.float32).reshape(4, 8, 2, 2)
        pattern = (pattern - pattern.mean()) / pattern.std()
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            inputs = pattern + 0.25 * rank + 0.1 * step
            loss = ddp(inputs).square().mean()
            loss.backward()
            routed_params = [p for p in ddp.module.experts.parameters() if p.requires_grad]
            routed_grads = [p.grad for p in routed_params if p.grad is not None]
            assert len(routed_grads) == len(routed_params), "routed experts produced incomplete gradients"
            assert all(torch.isfinite(grad).all() for grad in routed_grads), "non-finite routed gradient"
            assert sum(float(grad.abs().sum()) for grad in routed_grads) > 0.0, "all routed gradients are zero"
            optimizer.step()
            flat = torch.cat([p.detach().reshape(-1) for p in ddp.module.parameters()])
            gathered = [torch.empty_like(flat) for _ in range(world)]
            dist.all_gather(gathered, flat)
            assert torch.allclose(gathered[0], gathered[1]), f"parameters diverged after step {step}"
        if rank == 0:
            print("P0 routed DDP gate passed: backend=gloo, world_size=2, steps=2")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
