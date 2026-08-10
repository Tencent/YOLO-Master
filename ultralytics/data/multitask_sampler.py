"""Deterministic task scheduling primitives for multi-source training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler


class MultiTaskBatchSampler(Sampler[list[tuple[str, int]]]):
    """Schedule task/source samples with a resumable deterministic state."""

    schema_version = 1

    def __init__(
        self,
        source_lengths: Mapping[str, int | Sequence[int]],
        batch_size: int,
        *,
        weights: Mapping[str, float] | None = None,
        mode: str = "weighted",
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
        steps_per_epoch: int | None = None,
    ) -> None:
        if not isinstance(source_lengths, Mapping) or not source_lengths:
            raise ValueError("source_lengths must be a non-empty mapping")
        self.source_indices = {
            str(task): tuple(range(int(value)))
            if isinstance(value, int) and not isinstance(value, bool)
            else tuple(int(index) for index in value)
            for task, value in source_lengths.items()
        }
        self.source_lengths = {task: len(indices) for task, indices in self.source_indices.items()}
        if any(v <= 0 for v in self.source_lengths.values()):
            raise ValueError("all source lengths must be positive")
        self.tasks = tuple(self.source_lengths)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        if self.batch_size <= 0 or self.world_size <= 0 or not 0 <= self.rank < self.world_size:
            raise ValueError("batch_size must be positive and rank/world_size must be valid")
        self.mode = str(mode).lower()
        if self.mode not in {"weighted", "round_robin"}:
            raise ValueError("mode must be 'weighted' or 'round_robin'")
        self.weights = {task: 1.0 for task in self.tasks}
        if weights is not None:
            unknown = sorted(set(weights).difference(self.tasks))
            if unknown:
                raise ValueError(f"weights contain unknown task sources: {unknown}")
            self.weights.update({str(k): float(v) for k, v in weights.items()})
        if any(value < 0 for value in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("task weights must be non-negative with a positive total")
        self.epoch = 0
        self.position = 0
        self.steps_per_epoch = max(int(steps_per_epoch or max(self.source_lengths.values()) // self.batch_size), 1)

    def __len__(self) -> int:
        """Return the number of batches scheduled per epoch."""
        return self.steps_per_epoch

    def set_epoch(self, epoch: int) -> None:
        """Set epoch and reset the intra-epoch cursor."""
        self.epoch = int(epoch)
        self.position = 0

    def state_dict(self) -> dict[str, Any]:
        """Serialize schedule configuration and cursor for checkpoint/resume."""
        return {
            "schema_version": self.schema_version,
            "source_lengths": dict(self.source_lengths),
            "source_indices": {task: list(indices) for task, indices in self.source_indices.items()},
            "weights": dict(self.weights),
            "mode": self.mode,
            "seed": self.seed,
            "rank": self.rank,
            "world_size": self.world_size,
            "batch_size": self.batch_size,
            "steps_per_epoch": self.steps_per_epoch,
            "epoch": self.epoch,
            "position": self.position,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore state only when the source contract matches."""
        if int(state.get("schema_version", 0)) != self.schema_version:
            raise ValueError("unsupported multi-task sampler schema")
        if state.get("source_lengths") != self.source_lengths or state.get("mode") != self.mode:
            raise ValueError("multi-task sampler source contract does not match")
        serialized_indices = state.get("source_indices")
        if serialized_indices is not None:
            normalized_indices = {
                str(task): tuple(int(index) for index in indices) for task, indices in serialized_indices.items()
            }
            if normalized_indices != self.source_indices:
                raise ValueError("multi-task sampler source indices do not match")
        if state.get("weights") != self.weights:
            raise ValueError("multi-task sampler weights do not match")
        if int(state.get("batch_size", -1)) != self.batch_size:
            raise ValueError("multi-task sampler batch_size does not match")
        if int(state.get("seed", self.seed)) != self.seed:
            raise ValueError("multi-task sampler seed does not match")
        if (
            int(state.get("rank", self.rank)) != self.rank
            or int(state.get("world_size", self.world_size)) != self.world_size
        ):
            raise ValueError("multi-task sampler distributed contract does not match")
        if int(state.get("steps_per_epoch", self.steps_per_epoch)) != self.steps_per_epoch:
            raise ValueError("multi-task sampler steps_per_epoch does not match")
        self.epoch = int(state.get("epoch", 0))
        self.position = max(0, int(state.get("position", 0)))

    def _schedule(self) -> list[str]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        total = self.steps_per_epoch * self.batch_size * self.world_size
        if self.mode == "round_robin":
            return [self.tasks[i % len(self.tasks)] for i in range(total)]
        probs = torch.tensor([self.weights[task] for task in self.tasks], dtype=torch.float32)
        ids = torch.multinomial(probs / probs.sum(), total, replacement=True, generator=generator)
        return [self.tasks[int(index)] for index in ids]

    def __iter__(self) -> Iterator[list[tuple[str, int]]]:
        """Yield task/source batches while advancing the resumable batch cursor."""
        schedule = self._schedule()
        for batch_index in range(self.position, self.steps_per_epoch):
            batch = []
            for item_index in range(self.batch_size):
                global_index = (batch_index * self.batch_size + item_index) * self.world_size + self.rank
                task = schedule[global_index]
                # Each rank receives a distinct sample from the selected source. The
                # rank offset must remain in the source cursor; dividing by world
                # size would make every rank repeat the same sample.
                local_index = global_index % self.source_lengths[task]
                batch.append((task, self.source_indices[task][local_index]))
            self.position = batch_index + 1
            yield batch


class TaskRoutedDataset(Dataset):
    """Adapt ``(task, index)`` sampler values to a normal YOLO dataset."""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.collate_fn = self._collate

    def __len__(self) -> int:
        return len(self.dataset)

    def __getattr__(self, name: str):
        return getattr(self.dataset, name)

    def __getitem__(self, item: tuple[str, int] | int):
        task, index = item if isinstance(item, tuple) else ("default", int(item))
        sample = dict(self.dataset[index])
        sample["task_source"] = task
        return sample

    def _collate(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        sources = [sample.pop("task_source", "default") for sample in batch]
        collate = getattr(self.dataset, "collate_fn", None)
        result = collate(batch) if callable(collate) else torch.utils.data.default_collate(batch)
        result["task_source"] = sources
        return result


__all__ = ["MultiTaskBatchSampler", "TaskRoutedDataset"]
