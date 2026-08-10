"""Two-rank CPU/Gloo continuous-training gate for a real routed module."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ultralytics.nn.modules.moe.modules import OptimizedMOE


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    assert world == 2, f"P0 gate requires exactly two ranks, got {world}"
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(1234)
        model = OptimizedMOE(8, 8, num_experts=2, top_k=2)
        ddp = DDP(model, find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.SGD(ddp.parameters(), lr=0.05)
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.full((4, 8, 2, 2), 1.0 + rank + step * 0.25)
            loss = ddp(inputs).square().mean()
            loss.backward()
            grads = [p.grad for p in ddp.module.parameters() if p.requires_grad and p.grad is not None]
            assert grads, "routed module produced no gradients"
            assert all(torch.isfinite(grad).all() for grad in grads), "non-finite routed gradient"
            assert sum(float(grad.abs().sum()) for grad in grads) > 0.0, "all routed gradients are zero"
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
