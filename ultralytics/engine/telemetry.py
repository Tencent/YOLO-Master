"""Opt-in, device-aware training telemetry for reproducible experiments.

The collector is intentionally activated through an environment variable so it
also reaches Ultralytics' generated DDP worker entrypoint. This keeps normal
training free of profiling overhead while allowing experiment scripts to emit
one JSON record per rank and an aggregated rank-0 record.
"""

from __future__ import annotations

import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from ultralytics.nn.modules.moe.diagnostics import routing_runtime_metrics
from ultralytics.utils import LOGGER, RANK
from ultralytics.utils.torch_utils import unwrap_model

TELEMETRY_ENV = "YOLO_TRAIN_TELEMETRY"
TELEMETRY_LOSS_STEPS_ENV = "YOLO_TRAIN_TELEMETRY_LOSS_STEPS"
TELEMETRY_SCHEMA_VERSION = 1


def _env_flag(name: str, default: bool = False) -> bool:
    """Return a conventional boolean environment variable value."""
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on", "y"}


def _percentile(values: list[float], quantile: float) -> float | None:
    """Return a linear-interpolated percentile or ``None`` when no samples exist."""
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def _device_type(device: Any) -> str:
    """Normalize a PyTorch device-like value to its backend type."""
    return str(getattr(device, "type", device)).split(":", 1)[0].lower()


def sync_device(device: Any) -> None:
    """Synchronize supported accelerator work before a wall-clock measurement."""
    kind = _device_type(device)
    if kind == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))
    elif kind == "mps":
        mps = getattr(torch, "mps", None)
        if mps is not None and mps.is_available():
            mps.synchronize()


def reset_device_memory_peak(device: Any) -> None:
    """Reset CUDA's allocator peak; MPS has no equivalent public peak API."""
    if _device_type(device) == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def device_memory_sample(device: Any) -> dict[str, Any]:
    """Return memory data with an explicit backend-specific measurement contract."""
    kind = _device_type(device)
    if kind == "cuda" and torch.cuda.is_available():
        total = int(torch.cuda.get_device_properties(device).total_memory)
        peak = int(torch.cuda.max_memory_allocated(device))
        return {
            "measurement": "cuda_max_memory_allocated",
            "is_true_peak": True,
            "peak_device_memory_bytes": peak,
            "device_total_memory_bytes": total,
            "peak_device_memory_fraction": peak / total if total > 0 else None,
            "sampled_current_memory_bytes": None,
        }
    if kind == "mps":
        mps = getattr(torch, "mps", None)
        current = getattr(mps, "current_allocated_memory", None) if mps is not None else None
        if mps is not None and mps.is_available() and callable(current):
            return {
                "measurement": "mps_sampled_current_allocated_memory",
                "is_true_peak": False,
                "peak_device_memory_bytes": None,
                "device_total_memory_bytes": None,
                "peak_device_memory_fraction": None,
                "sampled_current_memory_bytes": int(current()),
            }
    return {
        "measurement": "unavailable",
        "is_true_peak": False,
        "peak_device_memory_bytes": None,
        "device_total_memory_bytes": None,
        "peak_device_memory_fraction": None,
        "sampled_current_memory_bytes": None,
    }


def _finite_loss(value: Any) -> float | None:
    """Convert a scalar/tensor/mapping of loss items to one finite total."""
    if isinstance(value, dict):
        values = [_finite_loss(item) for item in value.values()]
        finite = [item for item in values if item is not None]
        return sum(finite) if finite else None
    if isinstance(value, torch.Tensor):
        if not value.numel():
            return None
        value = value.detach().float().sum().item()
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _tensor_storage_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Estimate distinct tensor storage held by an optimizer state object."""
    seen = set() if seen is None else seen
    if isinstance(value, torch.Tensor):
        if value.numel():
            try:
                pointer = value.untyped_storage().data_ptr()
            except AttributeError:  # torch<1.12, supported by the package contract
                pointer = value.storage().data_ptr()
        else:
            pointer = id(value)
        if pointer in seen:
            return 0
        seen.add(pointer)
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_storage_bytes(item, seen) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_storage_bytes(item, seen) for item in value)
    return 0


def _checkpoint_files(trainer) -> dict[str, Any]:
    """Report serialized checkpoint sizes without presenting them as allocator memory."""
    weights = Path(getattr(trainer, "wdir", Path(getattr(trainer, "save_dir", ".")) / "weights"))
    files = {path.name: path.stat().st_size for path in sorted(weights.glob("*.pt")) if path.is_file()}
    return {
        "measurement": "serialized_checkpoint_file_size",
        "files_bytes": files,
        "max_checkpoint_file_bytes": max(files.values()) if files else None,
    }


def _relative_spread(values: list[float]) -> float | None:
    """Return max-min relative to absolute mean for a same-step rank comparison."""
    if len(values) < 2:
        return None
    denominator = max(abs(sum(values) / len(values)), 1e-12)
    return (max(values) - min(values)) / denominator


def aggregate_rank_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine rank-local records without claiming a single-device measurement is multi-device evidence."""
    ordered = sorted(records, key=lambda record: int(record.get("metadata", {}).get("rank", -1)))
    step_counts = {str(record["metadata"]["rank"]): int(record["steps"]["count"]) for record in ordered}
    samples = {str(record["metadata"]["rank"]): int(record["steps"]["samples"]) for record in ordered}
    elapsed = [float(record["steps"]["total_seconds"]) for record in ordered]
    losses = [record.get("loss", {}).get("first_steps", []) for record in ordered]
    spreads = []
    for index in range(max((len(items) for items in losses), default=0)):
        values = [float(items[index]) for items in losses if index < len(items) and items[index] is not None]
        spread = _relative_spread(values)
        if spread is not None:
            spreads.append({"step_index": index, "relative_spread": spread, "ranks": len(values)})
    total_samples = sum(samples.values())
    parallel_seconds = max(elapsed, default=0.0)
    peak_fractions = [
        float(record.get("memory", {}).get("peak_device_memory_fraction"))
        for record in ordered
        if isinstance(record.get("memory"), dict) and record["memory"].get("peak_device_memory_fraction") is not None
    ]
    return {
        "world_size": len(ordered),
        "rank_step_counts": step_counts,
        "rank_step_counts_consistent": len(set(step_counts.values())) <= 1,
        "rank_samples": samples,
        "global_samples_per_second": total_samples / parallel_seconds if parallel_seconds > 0 else None,
        "global_samples_per_second_measurement": "sum_rank_samples/max_rank_step_seconds",
        "rank_loss_relative_spread_first_steps": spreads,
        "rank_loss_relative_spread_max": max((item["relative_spread"] for item in spreads), default=None),
        "rank_loss_relative_spread_mean": (
            sum(item["relative_spread"] for item in spreads) / len(spreads) if spreads else None
        ),
        "rank_peak_device_memory_fraction_max": max(peak_fractions, default=None),
        "rank_peak_device_memory_fraction_measurement": "max_rank_cuda_allocator_peak/device_total_memory",
    }


class TrainingTelemetry:
    """Collect opt-in step timing, memory, routing, and DDP-consistency evidence."""

    def __init__(self, enabled: bool = False, loss_steps: int = 20):
        self.enabled = bool(enabled)
        self.loss_steps = max(int(loss_steps), 0)
        self._started = False
        self._step_started_at: float | None = None
        self._durations_seconds: list[float] = []
        self._samples = 0
        self._losses: list[float] = []
        self._memory: dict[str, Any] = {}
        self._optimizer_state_bytes_max = 0
        self._optimizer_samples = 0
        self._routing_observations = 0
        self._routing_summary = {
            "routed_layers_max": 0,
            "collapsed_layers_max": 0,
            "mean_gini_sum": 0.0,
            "mean_dominant_share_sum": 0.0,
            "expert_calls_sum": 0,
        }
        self._last_routing: dict[str, Any] = {}

    @classmethod
    def from_environment(cls) -> "TrainingTelemetry":
        """Construct telemetry from the DDP-inherited environment contract."""
        try:
            loss_steps = int(os.getenv(TELEMETRY_LOSS_STEPS_ENV, "20"))
        except ValueError:
            loss_steps = 20
        return cls(enabled=_env_flag(TELEMETRY_ENV), loss_steps=loss_steps)

    def on_pretrain_routine_end(self, trainer) -> None:
        """Start the measurement window after model, optimizer, and loaders are initialized."""
        if not self.enabled:
            return
        reset_device_memory_peak(trainer.device)
        self._memory = device_memory_sample(trainer.device)
        self._started = True

    def on_train_batch_start(self, trainer) -> None:
        """Synchronize supported accelerators immediately before timing a train step."""
        if self._started:
            sync_device(trainer.device)
            self._step_started_at = time.perf_counter()

    def on_train_batch_end(self, trainer) -> None:
        """Record one local-rank step after the optimizer update and callbacks complete."""
        if not self._started or self._step_started_at is None:
            return
        sync_device(trainer.device)
        self._durations_seconds.append(time.perf_counter() - self._step_started_at)
        self._step_started_at = None
        batch = getattr(trainer, "batch", {})
        image = batch.get("img") if isinstance(batch, dict) else None
        if isinstance(image, torch.Tensor) and image.ndim:
            self._samples += int(image.shape[0])
        if len(self._losses) < self.loss_steps:
            loss = _finite_loss(getattr(trainer, "loss_items", None))
            if loss is not None:
                self._losses.append(loss)
        sample = device_memory_sample(trainer.device)
        if sample["peak_device_memory_bytes"] is not None:
            self._memory["peak_device_memory_bytes"] = max(
                int(self._memory.get("peak_device_memory_bytes") or 0), int(sample["peak_device_memory_bytes"])
            )
            total = sample.get("device_total_memory_bytes")
            if total:
                self._memory["device_total_memory_bytes"] = int(total)
                self._memory["peak_device_memory_fraction"] = (
                    self._memory["peak_device_memory_bytes"] / self._memory["device_total_memory_bytes"]
                )
        if sample["sampled_current_memory_bytes"] is not None:
            self._memory["sampled_current_memory_bytes_max"] = max(
                int(self._memory.get("sampled_current_memory_bytes_max") or 0),
                int(sample["sampled_current_memory_bytes"]),
            )
        if len(self._durations_seconds) == 1 or len(self._durations_seconds) % 32 == 0:
            optimizer = getattr(trainer, "optimizer", None)
            if optimizer is not None:
                self._optimizer_samples += 1
                self._optimizer_state_bytes_max = max(
                    self._optimizer_state_bytes_max, _tensor_storage_bytes(getattr(optimizer, "state", {}))
                )
        try:
            routing = routing_runtime_metrics(unwrap_model(trainer.model))
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            self._last_routing = {"error": f"{type(exc).__name__}: {exc}"}
        else:
            self._routing_observations += 1
            self._last_routing = routing
            self._routing_summary["routed_layers_max"] = max(
                self._routing_summary["routed_layers_max"], int(routing.get("routed_layers", 0))
            )
            self._routing_summary["collapsed_layers_max"] = max(
                self._routing_summary["collapsed_layers_max"], int(routing.get("collapsed_layers", 0))
            )
            self._routing_summary["mean_gini_sum"] += float(routing.get("mean_gini", 0.0))
            self._routing_summary["mean_dominant_share_sum"] += float(routing.get("mean_dominant_share", 0.0))
            self._routing_summary["expert_calls_sum"] += int(routing.get("expert_calls", 0))

    def _record(self, trainer) -> dict[str, Any]:
        """Build a JSON-safe rank-local record with measurement semantics included."""
        optimizer = getattr(trainer, "optimizer", None)
        if optimizer is not None:
            self._optimizer_samples += 1
            self._optimizer_state_bytes_max = max(
                self._optimizer_state_bytes_max, _tensor_storage_bytes(getattr(optimizer, "state", {}))
            )
        total_seconds = sum(self._durations_seconds)
        count = len(self._durations_seconds)
        routing_count = self._routing_observations
        metadata = {
            "rank": RANK if RANK >= 0 else 0,
            "world_size": max(int(getattr(trainer, "world_size", 1) or 1), 1),
            "requested_device": str(getattr(getattr(trainer, "args", None), "device", "")),
            "resolved_device": str(getattr(trainer, "device", "")),
            "amp": bool(getattr(trainer, "amp", False)),
            "deterministic": bool(getattr(getattr(trainer, "args", None), "deterministic", False)),
            "configured_batch": getattr(trainer, "batch_size", None),
            "torch": str(torch.__version__),
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        return {
            "schema_version": TELEMETRY_SCHEMA_VERSION,
            "metadata": metadata,
            "steps": {
                "count": count,
                "samples": self._samples,
                "total_seconds": total_seconds,
                "samples_per_second": self._samples / total_seconds if total_seconds > 0 else None,
                "mean_milliseconds": total_seconds / count * 1000.0 if count else None,
                "p50_milliseconds": (_percentile(self._durations_seconds, 0.50) or 0.0) * 1000.0 if count else None,
                "p95_milliseconds": (_percentile(self._durations_seconds, 0.95) or 0.0) * 1000.0 if count else None,
            },
            "loss": {"first_steps": self._losses, "measurement": "sum_of_trainer_loss_items"},
            "memory": self._memory,
            "optimizer": {
                "measurement": "sampled_optimizer_state_tensor_storage",
                "sample_count": self._optimizer_samples,
                "sampled_max_tensor_storage_bytes": self._optimizer_state_bytes_max,
            },
            "routing": {
                "observations": routing_count,
                "routed_layers_max": self._routing_summary["routed_layers_max"],
                "collapsed_layers_max": self._routing_summary["collapsed_layers_max"],
                "mean_gini": self._routing_summary["mean_gini_sum"] / routing_count if routing_count else None,
                "mean_dominant_share": (
                    self._routing_summary["mean_dominant_share_sum"] / routing_count if routing_count else None
                ),
                "mean_expert_calls": self._routing_summary["expert_calls_sum"] / routing_count
                if routing_count
                else None,
                "last": self._last_routing,
            },
            "checkpoint": _checkpoint_files(trainer),
        }

    def on_teardown(self, trainer) -> None:
        """Write rank-local and rank-aggregated artifacts before DDP is destroyed."""
        if not self.enabled or not self._started:
            return
        record = self._record(trainer)
        save_dir = Path(getattr(trainer, "save_dir", "."))
        save_dir.mkdir(parents=True, exist_ok=True)
        rank = int(record["metadata"]["rank"])
        (save_dir / f"telemetry_rank_{rank}.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        records = [record]
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            gathered: list[dict[str, Any] | None] = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, record)
            records = [item for item in gathered if item is not None]
        if rank == 0:
            aggregate = {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "aggregation": aggregate_rank_records(records),
                "ranks": records,
            }
            (save_dir / "telemetry.json").write_text(
                json.dumps(aggregate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            LOGGER.info(f"[Telemetry] wrote {save_dir / 'telemetry.json'}")


__all__ = (
    "TELEMETRY_ENV",
    "TELEMETRY_LOSS_STEPS_ENV",
    "TrainingTelemetry",
    "aggregate_rank_records",
    "device_memory_sample",
    "reset_device_memory_peak",
    "sync_device",
)
