"""Tests for batched preloading of offline D1 features."""

from __future__ import annotations

import torch

from ultralytics.data.preloaded_feature_cache import (
    PreloadedFeatureBatchLoader,
)


class FakeCachedDataset:
    """Small deterministic cached-feature dataset."""

    def __init__(self, count: int = 5) -> None:
        self.items = []
        for index in range(count):
            self.items.append(
                {
                    "features": tuple(
                        torch.full(
                            (2, 2, 2),
                            float(index + level),
                            dtype=torch.float16,
                        )
                        for level in range(3)
                    ),
                    "cls": torch.tensor(
                        [[float(index % 2)]]
                    ),
                    "bboxes": torch.tensor(
                        [[0.5, 0.5, 0.25, 0.25]]
                    ),
                    "sample_id": f"sample-{index}",
                    "image_path": (
                        f"images/train/{index}.jpg"
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int):
        return self.items[index]


def test_cpu_preload_preserves_values_and_order():
    loader = PreloadedFeatureBatchLoader(
        FakeCachedDataset(),
        batch_size=2,
        shuffle=False,
        generator=None,
        storage_device=torch.device("cpu"),
    )

    first = next(iter(loader))

    assert first["sample_ids"] == [
        "sample-0",
        "sample-1",
    ]
    assert first["features"][0].dtype == torch.float16
    assert first["features"][0].shape == (
        2,
        2,
        2,
        2,
    )
    assert first["features"][0][0].eq(0).all()
    assert first["features"][0][1].eq(1).all()
    assert first["batch_idx"].tolist() == [0, 1]
    assert loader.preloaded_bytes > 0
    assert loader.preload_seconds >= 0


def test_iteration_does_not_call_torch_stack(monkeypatch):
    loader = PreloadedFeatureBatchLoader(
        FakeCachedDataset(),
        batch_size=2,
        shuffle=False,
        generator=None,
        storage_device=torch.device("cpu"),
    )

    def forbidden_stack(*args, **kwargs):
        raise AssertionError(
            "torch.stack must not run during iteration"
        )

    monkeypatch.setattr(
        torch,
        "stack",
        forbidden_stack,
    )

    batches = list(loader)

    assert len(batches) == 3
    assert batches[-1]["sample_ids"] == [
        "sample-4"
    ]


def test_shuffle_is_reproducible():
    first_loader = PreloadedFeatureBatchLoader(
        FakeCachedDataset(),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
        storage_device=torch.device("cpu"),
    )
    second_loader = PreloadedFeatureBatchLoader(
        FakeCachedDataset(),
        batch_size=2,
        shuffle=True,
        generator=torch.Generator().manual_seed(7),
        storage_device=torch.device("cpu"),
    )

    first_order = [
        sample_id
        for batch in first_loader
        for sample_id in batch["sample_ids"]
    ]
    second_order = [
        sample_id
        for batch in second_loader
        for sample_id in batch["sample_ids"]
    ]

    assert first_order == second_order
