# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""YOLO Multi-Task Trainer.

Extends DetectionTrainer for unified multi-task training with
TaskRouter-contextualized features and combined multi-task loss.
"""

from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ultralytics.data import TaskRoutedDataset, build_dataloader, build_yolo_dataset
from ultralytics.data.multitask_sampler import MultiTaskBatchSampler
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import MultiTaskModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK, colorstr
from ultralytics.utils.torch_utils import torch_distributed_zero_first

# ── per-task loss name kits ──────────────────────────────────────────────
_TASK_LOSS_NAMES = {
    "detect": ("box_loss", "cls_loss", "dfl_loss"),
    "segment": ("box_loss", "cls_loss", "dfl_loss", "seg_loss"),
    "pose": ("box_loss", "pose_loss", "kobj_loss", "cls_loss", "dfl_loss"),
    "classify": ("cls_global_loss",),
    "depth": ("depth_loss",),
    "normal": ("normal_loss",),
    "semantic": ("semantic_loss",),
}
_UNIFIED_LOSS_NAMES = (
    "box_loss",
    "cls_loss",
    "dfl_loss",
    "seg_loss",
    "pose_loss",
    "cls_global_loss",
    "depth_loss",
    "normal_loss",
    "semantic_loss",
)

_SUPPORTED_TASKS = frozenset((*_TASK_LOSS_NAMES, "normal", "semantic"))
_LOSS_BACKED_TASKS = frozenset(("detect", "segment", "pose", "classify", "depth", "normal", "semantic"))
_VALIDATION_EVIDENCE = {
    "detect": "box metrics",
    "segment": "mask metrics",
    "pose": "pose metrics",
    "classify": "validation cls_global_loss",
    "depth": "validation depth_loss",
    "normal": "validation normal_loss",
    "semantic": "validation semantic_loss",
}


def _resolve_data_source(data: dict[str, Any], key: str) -> Path | None:
    """Resolve a configured local source relative to the dataset root."""
    value = data.get(key)
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(data.get("path", "")) / path


def _split_source(data: dict[str, Any], name: str, split: str) -> tuple[str, Path | None]:
    """Return the split-specific source using the dataset's training-source fallback."""
    key = f"{split}_{name}"
    if key not in data:
        key = f"train_{name}"
    return key, _resolve_data_source(data, key)


def _require_existing_sources(data: dict[str, Any], task: str, names: tuple[str, ...]) -> None:
    """Require configured, readable train and validation sources for a COCO-backed task."""
    missing = []
    for name in names:
        for split in ("train", "val"):
            key, path = _split_source(data, name, split)
            if path is None:
                missing.append(f"'{key}'")
            elif not path.exists():
                missing.append(f"'{key}' ({path})")
    if missing:
        raise ValueError(
            f"Multi-task '{task}' requires configured train/val target sources: {', '.join(missing)}. "
            "Remove the task from data.tasks or provide the missing sources."
        )


def _require_dense_source(data: dict[str, Any], task: str, key: str) -> None:
    """Require a configured dense target directory before inspecting selected training labels."""
    directory = _resolve_data_source(data, key)
    if directory is None or not directory.is_dir():
        raise ValueError(
            f"Multi-task '{task}' requires '{key}' to reference an existing target directory. "
            "Remove the task from data.tasks or configure its supervision source."
        )


def _has_visible_keypoints(label: dict[str, Any]) -> bool:
    """Return whether a raw COCO label contains at least one visible pose keypoint."""
    keypoints = label.get("keypoints")
    return bool(
        keypoints is not None
        and getattr(keypoints, "size", 0)
        and len(keypoints.shape) == 3
        and keypoints.shape[-1] == 3
        and (keypoints[..., 2] > 0).any()
    )


def validate_multitask_dataset_supervision(dataset, tasks: tuple[str, ...], mode: str) -> None:
    """Reject selected training data that has no positive target for an enabled auxiliary branch."""
    if mode != "train":
        return
    labels = getattr(dataset, "labels", ())
    target_predicates = {
        "segment": lambda label: bool(len(label.get("segments", ()))),
        "pose": _has_visible_keypoints,
        "classify": lambda label: "cls_img" in label and bool(label.get("cls_img_valid", False)),
        "depth": lambda label: bool(label.get("depth_path")),
        "normal": lambda label: bool(label.get("normal_path")),
        "semantic": lambda label: bool(label.get("semantic_path") or label.get("panoptic_path")),
    }
    unsupervised = [
        task
        for task in tasks
        if task in target_predicates and not any(target_predicates[task](label) for label in labels)
    ]
    if unsupervised:
        raise ValueError(
            "The selected multi-task training split has no usable targets for "
            f"{sorted(unsupervised)}. Remove those tasks from data.tasks or provide aligned labels for the selected images."
        )


def validate_multitask_contract(data: dict[str, Any], head) -> tuple[str, ...]:
    """Validate that every requested multi-task branch has a model head, targets, loss, and validation evidence.

    Multi-task training currently uses the COCO-aligned dataset. Detection, instance segmentation, and pose report
    task-specific metrics; dense auxiliary branches contribute named validation losses until their own metrics exist.
    This gate intentionally rejects branches that would otherwise emit predictions without a supervised training path.
    """
    declared_tasks = data.get("tasks")
    if not isinstance(declared_tasks, (list, tuple, set)) or not declared_tasks:
        raise ValueError(
            "Multi-task datasets must declare a non-empty 'tasks' list. For example: tasks: [detect, segment, pose]."
        )
    if not all(isinstance(task, str) for task in declared_tasks):
        raise ValueError(f"Multi-task 'tasks' must contain only strings, got {declared_tasks!r}")

    tasks = set(declared_tasks)
    tasks.add("detect")  # MultiTaskHead and its assignment/loss path always include detection.
    if "obb" in tasks:
        raise ValueError(
            "Multi-task 'obb' is not supported: its target, loss, and validator path are incomplete. "
            "Use a dedicated OBB model or remove 'obb' from data.tasks."
        )
    unknown = tasks.difference(_SUPPORTED_TASKS)
    if unknown:
        raise ValueError(f"Unsupported multi-task branches: {sorted(unknown)}")
    without_loss = tasks.difference(_LOSS_BACKED_TASKS)
    without_validation = tasks.difference(_VALIDATION_EVIDENCE)
    if without_loss or without_validation:
        raise ValueError(
            "Multi-task branches require a criterion and validation evidence. "
            f"Missing loss={sorted(without_loss)}, validation={sorted(without_validation)}."
        )
    if data.get("multitask_format") != "coco":
        raise ValueError(
            "Multi-task training currently requires 'multitask_format: coco' so the data loader can emit aligned targets."
        )
    if not hasattr(head, "set_active_tasks"):
        raise TypeError("The selected multi-task model has no MultiTaskHead.set_active_tasks() contract.")
    head.set_active_tasks(tasks)

    _require_existing_sources(data, "detect", ("instances",))
    if "pose" in tasks:
        kpt_shape = data.get("kpt_shape")
        if (
            not isinstance(kpt_shape, (list, tuple))
            or len(kpt_shape) != 2
            or not all(isinstance(x, int) and x > 0 for x in kpt_shape)
        ):
            raise ValueError("Multi-task 'pose' requires a positive two-element 'kpt_shape', for example [17, 3].")
        if tuple(kpt_shape) != (17, 3):
            raise ValueError("The COCO multi-task pose loader currently supports only kpt_shape: [17, 3].")
        if tuple(getattr(head, "kpt_shape", ())) != tuple(kpt_shape):
            raise ValueError(
                "Multi-task pose kpt_shape does not match the selected model head: "
                f"data={list(kpt_shape)}, model={list(getattr(head, 'kpt_shape', ()))}."
            )
        _require_existing_sources(data, "pose", ("keypoints",))
    if "depth" in tasks:
        _require_dense_source(data, "depth", "depth_dir")
    if "normal" in tasks:
        _require_dense_source(data, "normal", "normal_dir")
    if "semantic" in tasks:
        source = str(data.get("semantic_source", "")).lower()
        if source not in {"stuff", "panoptic"}:
            raise ValueError("Multi-task 'semantic' requires semantic_source: stuff or semantic_source: panoptic.")
        semantic_nc = data.get("semantic_nc")
        if not isinstance(semantic_nc, int) or semantic_nc <= 0:
            raise ValueError("Multi-task 'semantic' requires a positive 'semantic_nc'.")
        if getattr(head, "semantic_nc", None) != semantic_nc:
            raise ValueError(
                "Multi-task semantic_nc does not match the selected model head: "
                f"data={semantic_nc}, model={getattr(head, 'semantic_nc', None)}."
            )
        _require_existing_sources(data, "semantic", (f"{source}_masks", f"{source}_annotations"))

    return tuple(sorted(tasks))


class MultiTaskTrainer(DetectionTrainer):
    """Trainer for YOLO Multi-Task Vision Model.

    Builds a MultiTaskModel instead of DetectionModel, and configures
    the combined multi-task loss with per-task weighting and TaskRouter
    balance loss.
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        """Initialize MultiTaskTrainer."""
        if overrides is None:
            overrides = {}
        # Ensure multitask-specific overrides
        overrides.setdefault("task", "multitask")
        super().__init__(cfg, overrides, _callbacks)
        # MultiTaskLoss always returns this fixed-width vector. Retain zeros for inactive tasks so terminal/log rows
        # cannot zip a shorter task list against the wrong loss indices.
        self.loss_names = _UNIFIED_LOSS_NAMES
        self.task_sampler: MultiTaskBatchSampler | None = None

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Return a MultiTaskModel instance."""
        model = self.set_model_names_for_load(
            MultiTaskModel(cfg, nc=self.data["nc"], ch=self.data.get("channels", 3), verbose=verbose and RANK == -1)
        )
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        """Return a validator matching the actually supervised tasks."""
        from ultralytics.models import yolo

        self.loss_names = _UNIFIED_LOSS_NAMES
        # MultiTaskValidator keeps the COCO JSON loader active and reports aligned detection, mask, and pose metrics.
        return yolo.multitask.MultiTaskValidator(
            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
        )

    def set_model_attributes(self):
        """Set MultiTaskModel attributes from dataset configuration."""
        super().set_model_attributes()
        model = unwrap_model(self.model)
        if not hasattr(model, "active_tasks"):
            raise TypeError("MultiTaskTrainer requires a model with an active_tasks contract.")
        active_tasks = validate_multitask_contract(self.data, model.model[-1])
        model.active_tasks = set(active_tasks)
        LOGGER.info(f"{colorstr('multitask:')} active_tasks={model.active_tasks}")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build YOLO Dataset for training or validation (fallback: detection)."""
        gs = max(int(unwrap_model(self.model).stride.max()), 32)
        dataset = build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)
        validate_multitask_dataset_supervision(dataset, tuple(sorted(unwrap_model(self.model).active_tasks)), mode)
        return dataset

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        """Construct a normal loader or an explicitly configured task-routed batch loader."""
        if mode not in {"train", "val"}:
            raise ValueError(f"Mode must be 'train' or 'val', not {mode}.")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        shuffle = mode == "train"
        if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
            LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
            shuffle = False
        batch_sampler = None
        if mode == "train" and isinstance(self.data.get("task_sources"), dict) and self.data["task_sources"]:
            sampler = self.build_task_sampler(rank=rank, batch_size=batch_size, dataset_length=len(dataset))
            dataset = TaskRoutedDataset(dataset)
            batch_sampler = sampler
            shuffle = False
        else:
            self.task_sampler = None
        return build_dataloader(
            dataset,
            batch=batch_size,
            workers=self.args.workers if mode == "train" else self.args.workers * 2,
            shuffle=shuffle,
            rank=rank,
            drop_last=self.args.compile and mode == "train",
            batch_sampler=batch_sampler,
        )

    def build_task_sampler(
        self, *, rank: int | None = None, batch_size: int | None = None, dataset_length: int | None = None
    ) -> MultiTaskBatchSampler | None:
        """Build an optional deterministic multi-source scheduler from ``data.task_sources``.

        The default COCO-aligned dataset remains unchanged. A sampler is only
        created when a data YAML explicitly provides source lengths, allowing
        experimental partial-label/multi-source runs to opt in without
        changing legacy single-source training.
        """
        sources = self.data.get("task_sources")
        if not isinstance(sources, dict) or not sources:
            return None
        args = self.args
        weights = getattr(args, "task_source_weights", None) or self.data.get("task_source_weights")
        mode = getattr(args, "task_sampler", None) or self.data.get("task_sampler", "weighted")
        seed = int(getattr(args, "seed", 0))
        current_rank = int(RANK if rank is None else rank)
        world_size = (
            int(torch.distributed.get_world_size())
            if torch.distributed.is_available() and torch.distributed.is_initialized()
            else 1
        )
        if dataset_length is not None:
            source_items = {
                str(task): tuple(range(int(value)))
                if isinstance(value, int) and not isinstance(value, bool)
                else tuple(value)
                for task, value in sources.items()
            }
            invalid = {
                task: [int(index) for index in indices if int(index) < 0 or int(index) >= dataset_length]
                for task, indices in source_items.items()
            }
            invalid = {task: indices[:3] for task, indices in invalid.items() if indices}
            if invalid:
                raise ValueError(f"task_sources contain indices outside the dataset: {invalid}")
        sampler = MultiTaskBatchSampler(
            sources,
            int(batch_size or getattr(args, "batch", 1)),
            weights=weights,
            mode=mode,
            seed=seed,
            rank=max(current_rank, 0),
            world_size=world_size,
            steps_per_epoch=self.data.get("task_steps_per_epoch"),
        )
        self.task_sampler = sampler
        return sampler

    def checkpoint_runtime_state(self) -> dict[str, Any]:
        """Provide multi-task scheduler progress for atomic training checkpoints."""
        return {"multitask_sampler": self.task_sampler.state_dict()} if self.task_sampler is not None else {}

    def restore_checkpoint_runtime_state(self, state: dict[str, Any]) -> None:
        """Restore the task-sampling cursor after the current data contract is constructed."""
        sampler_state = state.get("multitask_sampler") if isinstance(state, dict) else None
        if sampler_state is None:
            return
        if self.task_sampler is None:
            raise ValueError("resume checkpoint contains a multi-task sampler but no task_sources are configured")
        self.task_sampler.load_state_dict(sampler_state)

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Label multi-task loss items with readable names."""
        keys = list(self.loss_names)
        if loss_items is not None:
            # loss_items may be a dict (from MultiTaskLoss) or list/tensor
            if isinstance(loss_items, dict):
                return {f"{prefix}/{k}": round(float(v), 5) for k, v in loss_items.items()}
            if isinstance(loss_items, torch.Tensor):
                loss_items = loss_items.tolist()
            return {f"{prefix}/{k}": round(float(v), 5) for k, v in zip(keys, loss_items)}
        return keys

    def progress_string(self):
        """Return a formatted string of training progress with wider columns for multi-task losses."""
        # Use %17s to fit longest loss names: cls_global_loss (14) / mixture_aux_loss (15)
        n_loss = len(self.loss_names)
        fmt = "\n" + "%12s" + "%10s" + "%17s" * n_loss + "%11s" + "%10s"
        return fmt % ("Epoch", "GPU_mem", *self.loss_names, "Instances", "Size")


def unwrap_model(model):
    """Unwrap DDP/DistributedDataParallel wrappers."""
    from torch.nn.parallel import DistributedDataParallel as DDP

    if isinstance(model, DDP):
        return model.module
    return model
