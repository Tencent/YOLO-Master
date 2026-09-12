"""Opt-in shared Trainer policy for matched runs and exact epoch-boundary resume."""

from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import random
import time
import uuid
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler

from scripts.d1.artifacts import write_json
from ultralytics.data.build import ContiguousDistributedSampler, seed_worker
from ultralytics.engine.extensions.recovery import TrainingRecoveryController
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

AMP_INIT_SCALE = 0.0625
AMP_GROWTH_INTERVAL = 1_000_000
VALIDATION_PRECISION = "fp32-v1"
TRAINING_PRECISION = "bf16-mixed-fp32-loss-v1"


def detached_state(value):
    """Clone nested state onto CPU without rounding floating point tensors."""
    if isinstance(value, torch.nn.Module):
        value = value.state_dict()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: detached_state(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detached_state(item) for item in value)
    return deepcopy(value)


def state_digest(value):
    """Hash tensor values and scalar state independently of serialization storage IDs."""
    digest = hashlib.sha256()

    def visit(item):
        if isinstance(item, torch.Tensor):
            item = item.detach().cpu().contiguous()
            visit((str(item.dtype), tuple(item.shape)))
            digest.update(item.reshape(-1).view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict")
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(type(item).__name__.encode())
            for child in item:
                visit(child)
        else:
            data = pickle.dumps(item, protocol=4)
            digest.update(len(data).to_bytes(8, "little"))
            digest.update(data)

    visit(value)
    return digest.hexdigest()


def atomic_torch(path, state):
    """Publish only a completely written latest snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.part")
    with temporary.open("xb") as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def rank_buffer_state(model):
    return detached_state(dict(model.named_buffers())) if model is not None else None


def restore_rank_buffers(model, state):
    buffers = dict(model.named_buffers()) if model is not None else None
    if buffers is None or state is None:
        if buffers is not state:
            raise ValueError("Rank-local EMA availability differs")
        return
    if buffers.keys() != state.keys():
        raise ValueError("Rank-local buffer keys differ")
    for name, target in buffers.items():
        source = state[name]
        if source.shape != target.shape or source.dtype != target.dtype or not torch.isfinite(source).all():
            raise ValueError(f"Rank-local buffer metadata or finiteness differs: {name}")
    with torch.no_grad():
        for name, target in buffers.items():
            target.copy_(state[name])


def rng_state(device):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


class EpochDataLoader(DataLoader):
    """Bound prefetch to one epoch and seed new workers deterministically each epoch."""

    def __init__(self, *args, seed, rank, **kwargs):
        self.epoch_seed, self.epoch_rank = seed, rank
        super().__init__(*args, generator=torch.Generator(), persistent_workers=False, **kwargs)
        self.set_epoch(0)

    def set_epoch(self, epoch):
        self.sampler.set_epoch(epoch)
        self.generator.manual_seed((self.epoch_seed + 1_000_003 * epoch + self.epoch_rank) % (2**63))
        return True


class RunMixin:
    """Cooperative policy; omit run_identity to retain the ordinary Trainer lifecycle.

    stop_after_epoch is an absolute, one-based completed epoch, never a new schedule
    length. Resume inputs are trusted local snapshots with matching run identities.
    """

    def __init__(
        self, *args, run_identity=None, run_output=None, resume_snapshot=None, stop_after_epoch=None, **kwargs
    ):
        self.run_identity = deepcopy(run_identity)
        self.run_output = Path(run_output) if run_output is not None else None
        self.resume_snapshot = Path(resume_snapshot) if resume_snapshot is not None else None
        self.stop_after_epoch = stop_after_epoch
        self._run_enabled = run_identity is not None
        self.run_complete = False
        self._run_last_epoch = -1
        self._run_resume_state = None
        if not self._run_enabled and (resume_snapshot is not None or stop_after_epoch is not None):
            raise ValueError("Resume/window requires run_identity")
        if self._run_enabled and self.resume_snapshot is not None:
            self._run_resume_state = torch.load(self.resume_snapshot, map_location="cpu", weights_only=False)
            if (
                self._run_resume_state.get("schema_version") != "d1-runtime-v1"
                or self._run_resume_state.get("identity") != self.run_identity
            ):
                raise ValueError("Resume identity or schema differs")
        if self._run_enabled and self.run_output is not None:
            self._run_check_output()
        super().__init__(*args, **kwargs)
        if self._run_enabled:
            self.run_output = self.run_output or Path(self.save_dir)
            self._run_check_output()
            if stop_after_epoch is not None and (
                type(stop_after_epoch) is not int or not 1 <= stop_after_epoch <= self.args.epochs
            ):
                raise ValueError("stop_after_epoch must lie within the full epoch schedule")
            for event in (
                "train_epoch_start",
                "train_batch_start",
                "train_batch_end",
                "train_epoch_end",
                "fit_epoch_end",
            ):
                self.add_callback("on_" + event, getattr(self, "_run_on_" + event))

    def _run_check_output(self):
        latest = self.run_output / "resume.pt"
        if latest.exists() and (self.resume_snapshot is None or latest.resolve() != self.resume_snapshot.resolve()):
            raise FileExistsError("Use a fresh run_output or explicitly resume its matching latest snapshot")

    def get_model(self, cfg=None, weights=None, verbose=True):
        if not getattr(self, "_run_enabled", False):
            return super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        # Construction is CPU-only; do not seed or consume the rank's CUDA stream.
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(self.args.seed)
            model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        model._d1_loss_fp32 = True
        return model

    def check_amp_compatibility(self):
        """Select native BF16 for measured CUDA runs before the shared AMP broadcast."""
        if not getattr(self, "_run_enabled", False):
            return super().check_amp_compatibility()
        return self.device.type == "cuda" and torch.cuda.is_bf16_supported()

    def training_autocast(self):
        """Keep BF16 local to this run; model parameters and optimizer state stay FP32."""
        if not getattr(self, "_run_enabled", False):
            return super().training_autocast()
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp)

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        if not getattr(self, "_run_enabled", False):
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        sources = dataset_path if isinstance(dataset_path, (list, tuple)) else [dataset_path]
        listed = None
        if all(Path(path).suffix.lower() == ".txt" for path in sources):
            listed = sum(sum(bool(line.strip()) for line in Path(path).read_text().splitlines()) for path in sources)
            if len(dataset) != listed:
                raise ValueError(f"{mode} dataset size {len(dataset)} differs from supplied list size {listed}")
        world = max(self.world_size, 1)
        local_rank = dist.get_rank() if dist.is_initialized() else 0
        sampler = (
            DistributedSampler(dataset, num_replicas=world, rank=local_rank, shuffle=True, seed=self.args.seed)
            if mode == "train"
            else ContiguousDistributedSampler(dataset, num_replicas=world, rank=local_rank, batch_size=batch_size)
        )
        batch = min(batch_size, len(dataset))
        if not batch:
            raise ValueError("Empty runtime dataset")
        workers = min(
            (os.cpu_count() or 1) // max(torch.cuda.device_count(), 1),
            self.args.workers,
            0 if len(sampler) <= batch else math.ceil(len(sampler) / batch),
        )
        loader = EpochDataLoader(
            dataset,
            batch_size=batch,
            sampler=sampler,
            num_workers=workers,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            pin_memory=self.device.type == "cuda",
            drop_last=False,
            prefetch_factor=1 if workers else None,
            seed=self.args.seed,
            rank=local_rank,
        )
        loader.source_manifest = {"paths": [str(path) for path in sources], "listed_samples": listed}
        return loader

    def _build_train_pipeline(self):
        if getattr(self, "_run_enabled", False) and getattr(self, "_oom_retries", 0):
            raise RuntimeError("Runtime forbids OOM batch reduction/retry")
        return super()._build_train_pipeline()

    def _bootstrap_healthy_checkpoint(self):
        if getattr(self, "_run_enabled", False) and self.resume_snapshot is not None:
            return None
        return super()._bootstrap_healthy_checkpoint()

    def _setup_train(self):
        super()._setup_train()
        if not getattr(self, "_run_enabled", False):
            return
        if self.resume or self.accumulate != 1 or self.args.time or self.args.compile or self.args.close_mosaic:
            raise ValueError("Runtime requires fresh setup, one update per batch, and a fixed full schedule")
        if self.args.amp and not self.amp:
            raise RuntimeError("Requested AMP was disabled")
        if self.amp:
            self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
        self._run_rank = dist.get_rank() if dist.is_initialized() else 0
        self._run_actual_steps = 0
        self.optimizer.register_step_post_hook(self._run_step_recorded)
        if self.resume_snapshot is not None:
            self._run_restore()
        self._run_warm_reducer()
        self._run_setup_report()

    def _run_warm_reducer(self):
        """Build DDP buckets before either fresh or resumed updates, without training."""
        if not isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            return
        model = unwrap_model(self.model)
        state = detached_state(model)
        criterion = getattr(model, "criterion", None)
        native = getattr(criterion, "native_criterion", criterion)
        criterion_progress = {
            key: deepcopy(getattr(native, key)) for key in ("updates", "o2m", "o2o") if hasattr(native, key)
        }
        rng = rng_state(self.device)
        generator = self.train_loader.generator.get_state()
        scaler = deepcopy(self.scaler.state_dict())
        was_training = model.training
        started = time.monotonic()
        iterator = None
        try:
            self.train_loader.set_epoch(self.start_epoch)
            iterator = iter(self.train_loader)
            batch = self.preprocess_batch(next(iterator))
            model.train()
            for _ in range(3):
                self.optimizer.zero_grad(set_to_none=True)
                with self.training_autocast():
                    loss, _ = self.model(batch)
                    loss = loss.sum() * self.world_size
                if not bool(torch.isfinite(loss).all()):
                    raise FloatingPointError("DDP reducer warmup loss is not finite")
                self.scaler.scale(loss).backward()
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        finally:
            del iterator
            self.optimizer.zero_grad(set_to_none=True)
            model.criterion = criterion
            for key, value in criterion_progress.items():
                setattr(native, key, value)
            model.load_state_dict(state, strict=True)
            model.train(was_training)
            self.scaler.load_state_dict(scaler)
            self.train_loader.generator.set_state(generator)
            restore_rng(rng, self.device)
        if state_digest(model.state_dict()) != state_digest(state):
            raise RuntimeError("DDP warmup changed registered model state")
        write_json(
            self.run_output / f"reducer-warmup-rank-{self._run_rank}.json",
            {
                "passes": 3,
                "optimizer_updates": 0,
                "state_restored": True,
                "seconds": time.monotonic() - started,
            },
        )

    def _run_setup_report(self):
        datasets = {}
        for mode, loader in (("train", self.train_loader), ("val", self.test_loader)):
            datasets[mode] = {
                **loader.source_manifest,
                "dataset_size": len(loader.dataset),
                "rank_samples": len(loader.sampler),
                "rank_batches": len(loader),
                "rank_batch_size": loader.batch_size,
                "workers": loader.num_workers,
                "prefetch_factor": loader.prefetch_factor,
                "persistent_workers": loader.persistent_workers,
                "drop_last": loader.drop_last,
                "sampler": type(loader.sampler).__name__,
            }
        groups = []
        for names, group in zip(self._run_groups(), self.optimizer.param_groups):
            groups.append(
                {
                    "names": names,
                    "parameter_tensors": len(group["params"]),
                    "parameters": sum(parameter.numel() for parameter in group["params"]),
                    "settings": {key: value for key, value in group.items() if key != "params"},
                }
            )
        report = {
            "rank": self._run_rank,
            "world_size": max(self.world_size, 1),
            "seed": self.args.seed,
            "total_epochs": self.args.epochs,
            "start_epoch": self.start_epoch,
            "datasets": datasets,
            "args": vars(self.args),
            "optimizer_groups": groups,
            "amp": {
                "enabled": bool(self.amp),
                "training_precision": TRAINING_PRECISION if self.amp else "fp32",
                "validation_precision": VALIDATION_PRECISION,
                "gradient_scaling": self.scaler.is_enabled(),
                "current_scale": self.scaler.get_scale(),
            },
            "identity": self.run_identity,
            "resume_snapshot": self.resume_snapshot,
        }
        write_json(
            self.run_output / f"setup-rank-{self._run_rank}.json",
            json.loads(json.dumps(report, default=str, allow_nan=False)),
        )

    def _run_step_recorded(self, optimizer, args, kwargs):
        self._run_actual_steps += 1

    def _run_fail(self, bad, message):
        flag = torch.tensor(int(bool(bad)), device=self.device)
        if dist.is_initialized():
            dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        if flag.item():
            raise RuntimeError(message)

    def optimizer_step(self):
        if not getattr(self, "_run_enabled", False):
            return super().optimizer_step()
        self._run_fail(not torch.isfinite(self.loss).all(), "Non-finite training loss")
        before, actual = self.optimizer_steps, self._run_actual_steps
        ema = getattr(self, "ema", None)
        ema_before = ema.updates if ema is not None else None
        # Invoke the shared single attempt, bypassing any legacy Scratch AMP retry policy.
        updated = BaseTrainer.optimizer_step(self)
        lost = not updated or self.optimizer_steps != before + 1 or self._run_actual_steps != actual + 1
        lost |= ema is not None and ema.updates != ema_before + 1
        if lost or self.accumulate != 1:
            write_json(
                self.run_output / f"failed-update-rank-{self._run_rank}.json",
                {
                    "epoch": getattr(self, "epoch", -1) + 1,
                    "updated": bool(updated),
                    "gradient_nonfinite": bool(getattr(self, "_gradient_nonfinite", False)),
                    "amp_scale": self.scaler.get_scale(),
                    "accumulate": self.accumulate,
                    "optimizer_steps": [before, self.optimizer_steps],
                    "actual_steps": [actual, self._run_actual_steps],
                    "ema_updates": [ema_before, ema.updates if ema is not None else None],
                },
            )
        self._run_fail(lost or self.accumulate != 1, "Missing/repeated optimizer update or AMP overflow")
        return True

    def _run_check_finite(self):
        ema = getattr(getattr(self, "ema", None), "ema", None)
        values = [unwrap_model(self.model).state_dict(), self.optimizer.state, ema.state_dict() if ema else {}]
        finite = TrainingRecoveryController.tensors_are_finite(TrainingRecoveryController.iter_floating_tensors(values))
        self._run_fail(not finite, "Non-finite model, EMA, or optimizer state")

    def _recover_before_validation(self, epoch):
        if not getattr(self, "_run_enabled", False):
            return super()._recover_before_validation(epoch)
        self._run_check_finite()
        return False

    def validate(self):
        """Keep measured validation in FP32 while preserving the training AMP policy."""
        if not getattr(self, "_run_enabled", False):
            return super().validate()
        training_amp = self.amp
        try:
            self.amp = False
            with torch.autocast(self.device.type, enabled=False):
                return super().validate()
        finally:
            self.amp = training_amp

    def _handle_nan_recovery(self, epoch):
        if not getattr(self, "_run_enabled", False):
            return super()._handle_nan_recovery(epoch)
        metrics = dict(getattr(self, "metrics", None) or {})
        if getattr(self, "fitness", None) is not None:
            metrics["fitness"] = self.fitness
        invalid = {key: str(float(value)) for key, value in metrics.items() if not math.isfinite(float(value))}
        self._run_fail(bool(invalid), f"Non-finite validation metrics at epoch {epoch + 1}: {invalid}")
        self._run_check_finite()
        return False

    def _run_groups(self):
        names = {id(p): name for name, p in unwrap_model(self.model).named_parameters()}
        return [[names[id(p)] for p in group["params"]] for group in self.optimizer.param_groups]

    def _run_criterion(self):
        model = unwrap_model(self.model)
        if getattr(model, "criterion", None) is None:
            model.criterion = model.init_criterion()
        return getattr(model.criterion, "native_criterion", model.criterion)

    def _run_local_state(self):
        ema = getattr(getattr(self, "ema", None), "ema", None)
        return {
            "rng": rng_state(self.device),
            "loader_generator": self.train_loader.generator.get_state(),
            "val_loader_generator": self.test_loader.generator.get_state(),
            "model_buffers": rank_buffer_state(unwrap_model(self.model)),
            "ema_buffers": rank_buffer_state(ema),
            "sample_sequence_sha256": self._run_samples.hexdigest(),
        }

    def _run_state(self):
        ema = getattr(self, "ema", None)
        criterion = self._run_criterion()
        return {
            "model": detached_state(unwrap_model(self.model)),
            "ema": detached_state(ema.ema) if ema is not None else None,
            "ema_updates": ema.updates if ema is not None else None,
            "optimizer": detached_state(self.optimizer.state_dict()),
            "scaler": detached_state(self.scaler.state_dict()),
            "scheduler": detached_state(self.scheduler.state_dict()),
            "criterion": {
                k: deepcopy(getattr(criterion, k)) for k in ("updates", "o2m", "o2o") if hasattr(criterion, k)
            },
            "optimizer_steps": self.optimizer_steps,
            "optimizer_groups": self._run_groups(),
            "trainer_state": {
                "best_fitness": getattr(self, "best_fitness", None),
                "fitness": self.fitness,
                "stopper": deepcopy(vars(self.stopper)) if getattr(self, "stopper", None) is not None else None,
            },
        }

    def _run_restore(self):
        state, self._run_resume_state = self._run_resume_state, None
        world = dist.get_world_size() if dist.is_initialized() else 1
        if state.get("schema_version") != "d1-runtime-v1" or state.get("identity") != self.run_identity:
            raise ValueError("Resume identity or schema differs")
        start = state["epoch"] + 1
        if state["total_epochs"] != self.args.epochs or not 0 < start < self.args.epochs:
            raise ValueError("Resume epoch or full schedule differs")
        if len(state["ranks"]) != world or state["optimizer_groups"] != self._run_groups():
            raise ValueError("Resume world size or named optimizer groups differ")
        if self.stop_after_epoch is not None and self.stop_after_epoch <= start:
            raise ValueError("Resume window has no remaining epochs")
        if any(state_digest(state[key]) != digest for key, digest in state["state_digests"].items()):
            raise ValueError("Resume state digest mismatch")
        model, ema = unwrap_model(self.model), getattr(self, "ema", None)
        local = state["ranks"][self._run_rank]
        model.load_state_dict(state["model"], strict=True)
        restore_rank_buffers(model, local["model_buffers"])
        if ema is not None:
            ema.ema.load_state_dict(state["ema"], strict=True)
            ema.updates = state["ema_updates"]
        restore_rank_buffers(ema.ema if ema is not None else None, local["ema_buffers"])
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.scheduler.load_state_dict(state["scheduler"])
        for key, value in state["criterion"].items():
            setattr(self._run_criterion(), key, deepcopy(value))
        self.optimizer_steps = state["optimizer_steps"]
        self.best_fitness = state["trainer_state"]["best_fitness"]
        self.fitness = state["trainer_state"]["fitness"]
        if state["trainer_state"]["stopper"] is not None:
            vars(self.stopper).update(deepcopy(state["trainer_state"]["stopper"]))
        self.start_epoch = start
        self._run_last_epoch = state["epoch"]
        self.train_loader.generator.set_state(local["loader_generator"])
        self.test_loader.generator.set_state(local["val_loader_generator"])
        restore_rng(local["rng"], self.device)
        expected = {key: deepcopy(state[key]) for key in self._run_state()}
        expected["model"].update({k: v for k, v in local["model_buffers"].items() if k in expected["model"]})
        if ema is not None:
            expected["ema"].update({k: v for k, v in local["ema_buffers"].items() if k in expected["ema"]})
        else:
            expected["ema"], expected["ema_updates"] = None, None
        restored = self._run_state()
        expected_digests = {key: state_digest(value) for key, value in expected.items()}
        restored_digests = {key: state_digest(value) for key, value in restored.items()}
        if expected_digests != restored_digests:
            raise ValueError("Restored state is not numerically identical")
        restored_local = {
            **local,
            "rng": rng_state(self.device),
            "loader_generator": self.train_loader.generator.get_state(),
            "val_loader_generator": self.test_loader.generator.get_state(),
            "model_buffers": rank_buffer_state(model),
            "ema_buffers": rank_buffer_state(ema.ema if ema is not None else None),
        }
        if state_digest(local) != state_digest(restored_local):
            raise ValueError("Restored rank buffers or RNG differ")
        self._run_check_finite()
        # begin_epoch skips epoch == start_epoch; apply the missing boundary exactly once.
        self.mixture_controller.anneal_temperature()
        write_json(
            self.run_output / f"resume-restored-rank-{self._run_rank}.json",
            {
                "start_epoch": start,
                "rank": self._run_rank,
                "matched": True,
                "snapshot": str(self.resume_snapshot),
                "expected_state_digests": expected_digests,
                "restored_state_digests": restored_digests,
                "rank_state_digest": state_digest(local),
                "restored_rank_state_digest": state_digest(restored_local),
                "temperature_policy": "epoch-boundary-v1",
                "next_epoch_model_sha256": state_digest(model.state_dict()),
            },
        )

    def _run_on_train_epoch_start(self, trainer):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self._run_started = self._run_previous_end = time.monotonic()
        self._run_wait = self._run_compute = 0.0
        self._run_batches = 0
        self._run_start_updates = self.optimizer_steps
        self._run_samples = hashlib.sha256()
        self.test_loader.set_epoch(self.epoch)
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)

    def _run_on_train_batch_start(self, trainer):
        self._run_batch_started = time.monotonic()
        self._run_wait += self._run_batch_started - self._run_previous_end
        files = self.batch.get("im_file")
        if files is None:
            raise ValueError("Runtime sample audit requires im_file")
        self._run_samples.update(json.dumps(list(files), ensure_ascii=True, separators=(",", ":")).encode())

    def _run_progress(self, status):
        return {
            "status": status,
            "epoch": self.epoch + 1,
            "rank": self._run_rank,
            "batches": self._run_batches,
            "optimizer_steps": self.optimizer_steps,
            "amp_scale": self.scaler.get_scale(),
            "data_wait_seconds": self._run_wait,
            "step_seconds": self._run_compute,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device) if self.device.type == "cuda" else 0,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(self.device) if self.device.type == "cuda" else 0,
            "sample_sequence_sha256": self._run_samples.hexdigest(),
            "updated_unix": time.time(),
        }

    def _run_on_train_batch_end(self, trainer):
        self._run_batches += 1
        self._run_fail(
            self.optimizer_steps - self._run_start_updates != self._run_batches or self.accumulate != 1,
            "Batch count differs from optimizer updates",
        )
        self._run_fail(not torch.isfinite(self.loss_items).all(), "Non-finite per-batch loss items")
        now = time.monotonic()
        self._run_compute += now - self._run_batch_started
        self._run_previous_end = now
        if self._run_batches == 1 or self._run_batches % 10 == 0:
            write_json(self.run_output / f"progress-rank-{self._run_rank}.json", self._run_progress("running"))

    def _run_on_train_epoch_end(self, trainer):
        self._run_fail(self._run_batches != len(self.train_loader), "Epoch did not consume the entire sampler")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        write_json(
            self.run_output / "epochs" / f"rank-{self._run_rank}-epoch-{self.epoch + 1:03d}.json",
            self._run_progress("trained"),
        )

    def _run_on_fit_epoch_end(self, trainer):
        if self.epoch <= self._run_last_epoch:
            raise RuntimeError("Duplicate epoch-boundary callback")
        self._handle_nan_recovery(self.epoch)
        local = self._run_local_state()
        ranks = [None] * (dist.get_world_size() if dist.is_initialized() else 1)
        if dist.is_initialized():
            dist.all_gather_object(ranks, local)
        else:
            ranks[0] = local
        error = None
        if self._run_rank == 0:
            try:
                state = self._run_state()
                state.update(ranks=ranks)
                digests = {key: state_digest(value) for key, value in state.items()}
                state.update(
                    schema_version="d1-runtime-v1",
                    identity=self.run_identity,
                    epoch=self.epoch,
                    total_epochs=self.args.epochs,
                    state_digests=digests,
                )
                atomic_torch(self.run_output / "resume.pt", state)
                write_json(
                    self.run_output / "validation" / f"epoch-{self.epoch + 1:03d}.json",
                    {
                        "epoch": self.epoch + 1,
                        "epoch_wall_seconds": time.monotonic() - self._run_started,
                        "metrics": {k: float(v) for k, v in (getattr(self, "metrics", None) or {}).items()},
                        "state_digests": digests,
                    },
                )
            except Exception as exc:  # noqa: BLE001 - peers must receive serialization failures before continuing.
                error = f"{type(exc).__name__}: {exc}"
        if dist.is_initialized():
            errors = [error]
            dist.broadcast_object_list(errors, src=0)
            error = errors[0]
        if error is not None:
            raise RuntimeError(f"Epoch snapshot failed: {error}")
        self._run_last_epoch = self.epoch
        self.run_complete = self.epoch + 1 == self.args.epochs
        window = self.stop_after_epoch is not None and self.epoch + 1 >= self.stop_after_epoch
        self.stop |= window or self.run_complete
        status = "completed" if self.run_complete else "window-complete" if window else "running"
        write_json(self.run_output / f"progress-rank-{self._run_rank}.json", self._run_progress(status))

    def save_model(self):
        if not getattr(self, "_run_enabled", False):
            return super().save_model()
        period, self.save_period = self.save_period, -1
        try:
            result = super().save_model()
        finally:
            self.save_period = period
        if result:
            path = Path(self.wdir) / f"epoch-{self.epoch + 1:03d}.pt" if (self.epoch + 1) % 5 == 0 else None
            checkpoint = torch.load(self.last, map_location="cpu", weights_only=False)
            for key, source in (("model", unwrap_model(self.model)), ("ema", getattr(self.ema, "ema", None))):
                checkpoint[key] = None
                if source is not None:
                    if any("teacher" in name.lower() for name in source.state_dict()):
                        raise ValueError("Teacher state is forbidden in D1 checkpoints")
                    checkpoint[key] = deepcopy(source).cpu().float()
                    checkpoint[key].criterion = None
            if path is not None and path.exists():
                previous = torch.load(path, map_location="cpu", weights_only=False)
                for key in ("model", "ema"):
                    old = previous.get(key)
                    new = checkpoint.get(key)
                    if state_digest(detached_state(old)) != state_digest(detached_state(new)):
                        raise ValueError("Replayed periodic checkpoint has different tensor state")
                if previous["epoch"] != checkpoint["epoch"]:
                    raise ValueError("Replayed periodic checkpoint has a different epoch")
            atomic_torch(self.last, checkpoint)
            if getattr(self, "best", None) is not None and self.best_fitness == self.fitness:
                atomic_torch(self.best, checkpoint)
            if path is not None:
                atomic_torch(path, checkpoint)
        return result

    def final_eval(self):
        """Keep optimizer-bearing checkpoints; main runs independent evaluation."""
        if not getattr(self, "_run_enabled", False):
            return super().final_eval()
        return None
