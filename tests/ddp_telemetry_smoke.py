"""Two-rank CPU/Gloo smoke for the opt-in telemetry aggregation contract."""

import json
import os
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from ultralytics.engine.telemetry import TrainingTelemetry
from ultralytics.nn.modules.mot import MoTBlock


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    out_dir = Path(os.environ["TELEMETRY_SMOKE_DIR"])
    assert world == 2, f"telemetry gate requires exactly two ranks, got {world}"
    torch.set_num_threads(1)
    dist.init_process_group("gloo", timeout=timedelta(seconds=60))
    try:
        torch.manual_seed(9000 + rank)
        model = DDP(MoTBlock(24, num_heads=3, top_k=1, sparse_train=True), find_unused_parameters=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        trainer = SimpleNamespace(
            device=torch.device("cpu"),
            args=SimpleNamespace(device="cpu", deterministic=True),
            batch_size=2,
            model=model,
            optimizer=optimizer,
            save_dir=out_dir,
            wdir=out_dir / "weights",
            world_size=world,
        )
        trainer.wdir.mkdir(parents=True, exist_ok=True)
        telemetry = TrainingTelemetry(enabled=True, loss_steps=2)
        telemetry.on_pretrain_routine_end(trainer)
        for step in range(2):
            trainer.batch = {"img": torch.full((2, 24, 4, 4), 1.0 + rank + step * 0.1)}
            telemetry.on_train_batch_start(trainer)
            optimizer.zero_grad(set_to_none=True)
            output, aux = model(trainer.batch["img"])
            loss = output.square().mean() + aux
            trainer.loss_items = loss.detach()
            loss.backward()
            optimizer.step()
            telemetry.on_train_batch_end(trainer)
        telemetry.on_teardown(trainer)
        dist.barrier()
        if rank == 0:
            payload = json.loads((out_dir / "telemetry.json").read_text(encoding="utf-8"))
            assert payload["aggregation"]["rank_step_counts_consistent"] is True
            assert len(payload["ranks"]) == 2
            assert {record["metadata"]["rank"] for record in payload["ranks"]} == {0, 1}
            assert (out_dir / "telemetry_rank_0.json").exists()
            assert (out_dir / "telemetry_rank_1.json").exists()
            print("P1 telemetry DDP gate passed: backend=gloo, world_size=2, steps=2")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
