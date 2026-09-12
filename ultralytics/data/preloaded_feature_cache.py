"""Preloaded batched access to offline feature caches."""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import Dataset


class PreloadedFeatureBatchLoader:
    """Pre-stack cached FP16 features and retrieve batches by index."""

    def __init__(
        self,
        dataset: Dataset,
        *,
        batch_size: int,
        shuffle: bool,
        generator: torch.Generator | None,
        storage_device: torch.device,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                f"batch_size must be positive, got {batch_size}"
            )
        if len(dataset) <= 0:
            raise ValueError("cannot preload an empty dataset")
        if (
            storage_device.type == "cuda"
            and not torch.cuda.is_available()
        ):
            raise RuntimeError(
                "CUDA cache residency requested but CUDA is unavailable"
            )

        started = time.perf_counter()

        items = [
            dataset[index]
            for index in range(len(dataset))
        ]
        levels = len(items[0]["features"])

        if levels <= 0:
            raise ValueError(
                "cached samples contain no feature levels"
            )
        if any(
            len(item["features"]) != levels
            for item in items
        ):
            raise RuntimeError(
                "cached samples have inconsistent feature levels"
            )

        for item in items:
            for feature in item["features"]:
                if feature.dtype != torch.float16:
                    raise RuntimeError(
                        "preloaded cache expects FP16 features, "
                        f"got {feature.dtype}"
                    )

        self._features = tuple(
            torch.stack(
                [
                    item["features"][level]
                    for item in items
                ],
                dim=0,
            ).to(storage_device)
            for level in range(levels)
        )
        self._classes = tuple(
            item["cls"] for item in items
        )
        self._boxes = tuple(
            item["bboxes"] for item in items
        )
        self._sample_ids = tuple(
            item["sample_id"] for item in items
        )
        self._image_paths = tuple(
            item["image_path"] for item in items
        )

        self.batch_size = batch_size
        self.shuffle = shuffle
        self.generator = generator
        self.storage_device = storage_device
        self.preloaded_bytes = sum(
            feature.numel() * feature.element_size()
            for feature in self._features
        )
        self.preload_seconds = (
            time.perf_counter() - started
        )

    def __len__(self) -> int:
        return math.ceil(
            len(self._sample_ids) / self.batch_size
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        sample_count = len(self._sample_ids)

        if self.shuffle:
            order = torch.randperm(
                sample_count,
                generator=self.generator,
            )
        else:
            order = torch.arange(sample_count)

        for start in range(
            0,
            sample_count,
            self.batch_size,
        ):
            cpu_indices = order[
                start : start + self.batch_size
            ]
            index_values = cpu_indices.tolist()
            storage_indices = cpu_indices.to(
                self.storage_device
            )

            features = tuple(
                feature.index_select(
                    0,
                    storage_indices,
                )
                for feature in self._features
            )

            classes = [
                self._classes[index]
                for index in index_values
            ]
            boxes = [
                self._boxes[index]
                for index in index_values
            ]
            batch_indices = [
                torch.full(
                    (len(classes[local_index]),),
                    local_index,
                    dtype=torch.long,
                )
                for local_index in range(
                    len(index_values)
                )
            ]

            yield {
                "features": features,
                "cls": torch.cat(classes, dim=0),
                "bboxes": torch.cat(boxes, dim=0),
                "batch_idx": torch.cat(
                    batch_indices,
                    dim=0,
                ),
                "targets": list(
                    zip(classes, boxes)
                ),
                "sample_ids": [
                    self._sample_ids[index]
                    for index in index_values
                ],
                "image_paths": [
                    self._image_paths[index]
                    for index in index_values
                ],
            }
