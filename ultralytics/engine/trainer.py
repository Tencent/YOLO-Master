# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Train a model on a dataset.

Usage:
    $ yolo mode=train model=yolo26n.pt data=coco8.yaml imgsz=640 epochs=100 batch=16
"""

from __future__ import annotations

import gc
import math
import os
import subprocess
import time
import warnings
from contextlib import nullcontext
from copy import copy, deepcopy
from datetime import timedelta
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch import distributed as dist
from torch import nn, optim

from ultralytics.cfg import _YOLO_CLI_COMMAND, get_cfg, get_save_dir
from ultralytics.engine.extensions import (
    AdapterRuntimeController,
    MixtureRuntimeController,
    TrainingRecoveryController,
    update_args_with_lora_runtime_metadata,
    validate_adapter_configuration,
)
from ultralytics.data.utils import check_cls_dataset, check_det_dataset, convert_ndjson_to_yolo_if_needed
from ultralytics.nn.distill_model import DistillationModel
from ultralytics.nn.mixture_loss import has_routed_modules
from ultralytics.nn.tasks import load_checkpoint
from ultralytics.optim import MuSGD
from ultralytics.utils import (
    DEFAULT_CFG,
    LOCAL_RANK,
    LOGGER,
    RANK,
    TQDM,
    YAML,
    callbacks,
    clean_url,
    colorstr,
    emojis,
)
from ultralytics.utils.autobatch import check_train_batch_size
from ultralytics.utils.checks import check_amp, check_file, check_imgsz, check_model_file_from_stem, print_args
from ultralytics.utils.dist import collect_ddp_error_logs, ddp_cleanup, ddp_launch_env, generate_ddp_command
from ultralytics.utils.files import get_latest_run
from ultralytics.utils.plotting import plot_results
from ultralytics.utils.torch_utils import (
    TORCH_2_4,
    EarlyStopping,
    ModelEMA,
    attempt_compile,
    autocast,
    init_seeds,
    one_cycle,
    parse_device,
    select_device,
    strip_optimizer,
    torch_distributed_zero_first,
    unset_deterministic,
    unwrap_model,
)

__all__ = [
    "BaseTrainer",
    "MultiTrainer",
    "update_args_with_lora_runtime_metadata",
    "validate_adapter_configuration",
]


def _distributed_env() -> tuple[int, int, int] | None:
    """Validate torchrun rank variables as an all-or-none contract."""
    names = ("RANK", "LOCAL_RANK", "WORLD_SIZE")
    raw = {name: os.getenv(name) for name in names}
    present = [name for name, value in raw.items() if value not in (None, "")]
    if not present:
        return None
    if len(present) != len(names):
        raise RuntimeError(f"Incomplete distributed environment: set all of {names}, got {raw}.")
    try:
        rank, local_rank, world_size = (int(raw[name]) for name in names)
    except ValueError as exc:
        raise RuntimeError(f"Distributed rank variables must be integers, got {raw}.") from exc
    if world_size < 1 or rank < 0 or rank >= world_size or local_rank < 0:
        raise RuntimeError(
            f"Invalid distributed environment: RANK={rank}, LOCAL_RANK={local_rank}, WORLD_SIZE={world_size}."
        )
    return rank, local_rank, world_size


def _validate_cuda_ddp_device(device: torch.device, env: tuple[int, int, int] | None) -> None:
    """Fail before collectives when torchrun cannot bind the local CUDA ordinal."""
    if env is None:
        return
    _, local_rank, _ = env
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError(f"torchrun DDP requires available CUDA, but resolved device is {device}.")
    count = torch.cuda.device_count()
    if local_rank >= count:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} is invalid for {count} visible CUDA device(s). "
            "Check CUDA_VISIBLE_DEVICES and --nproc_per_node."
        )


def _optimizer_family(optimizer) -> str | None:
    """Return a coarse optimizer family used to validate checkpoint state compatibility."""
    name = type(optimizer).__name__.lower()
    if name == "musgd":
        return "musgd"
    if name in {"adam", "adamax", "adamw", "nadam", "radam"}:
        return "adam"
    if name == "rmsprop":
        return "rmsprop"
    if name == "sgd":
        return "sgd"
    return None


def _optimizer_state_family(state_dict) -> str | None:
    """Infer an optimizer family from a serialized PyTorch optimizer state."""
    if not isinstance(state_dict, dict):
        return None
    groups = state_dict.get("param_groups", ())
    if any("use_muon" in group for group in groups):
        return "musgd"
    state_keys = {
        key
        for state in state_dict.get("state", {}).values()
        if isinstance(state, dict)
        for key in state
    }
    if {"exp_avg", "exp_avg_sq"} <= state_keys:
        return "adam"
    if "square_avg" in state_keys:
        return "rmsprop"
    if "momentum_buffer" in state_keys:
        return "sgd"
    return None


def _adapter_active(model, controller=None) -> bool:
    """Return whether a model or any nested student/adapter module owns trainable adapters."""
    if bool(getattr(controller, "active", False)):
        return True
    root = unwrap_model(model)
    return any(
        bool(getattr(module, "lora_enabled", False) or getattr(module, "molora_enabled", False))
        for module in root.modules()
    )


class BaseTrainer:
    """A base class for creating trainers.

    This class provides the foundation for training YOLO models, handling the training loop, validation, checkpointing,
    and various training utilities. It supports both single-GPU and multi-GPU distributed training.

    Attributes:
        args (SimpleNamespace): Configuration for the trainer.
        validator (BaseValidator): Validator instance.
        model (nn.Module): Model instance.
        callbacks (defaultdict): Dictionary of callbacks.
        save_dir (Path): Directory to save results.
        wdir (Path): Directory to save weights.
        last (Path): Path to the last checkpoint.
        best (Path): Path to the best checkpoint.
        save_period (int): Save checkpoint every x epochs (disabled if < 1).
        batch_size (int): Batch size for training.
        epochs (int): Number of epochs to train for.
        start_epoch (int): Starting epoch for training.
        device (torch.device): Device to use for training.
        amp (bool): Flag to enable AMP (Automatic Mixed Precision).
        scaler (torch.amp.GradScaler): Gradient scaler for AMP.
        data (dict): Dataset dictionary containing paths and metadata.
        ema (ModelEMA): EMA (Exponential Moving Average) of the model.
        resume (bool): Resume training from a checkpoint.
        lf (callable): Learning rate scheduling function.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
        best_fitness (float): The best fitness value achieved.
        fitness (float): Current fitness value.
        loss (torch.Tensor): Current loss value.
        tloss (torch.Tensor): Running mean of loss items.
        loss_names (list): List of loss names.
        csv (Path): Path to results CSV file.
        metrics (dict): Dictionary of metrics.
        plots (dict): Dictionary of plots.

    Methods:
        train: Execute the training process.
        validate: Run validation on the val set.
        save_model: Save model training checkpoints.
        get_dataset: Get train and validation datasets.
        setup_model: Load, create, or download model.
        build_optimizer: Construct an optimizer for the model.

    Examples:
        Initialize a trainer and start training
        >>> trainer = BaseTrainer(cfg="config.yaml")
        >>> trainer.train()
    """

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks: dict | None = None):
        """Initialize the BaseTrainer class.

        Args:
            cfg (str | dict | SimpleNamespace, optional): Path to a configuration file or configuration object.
            overrides (dict, optional): Configuration overrides.
            _callbacks (dict, optional): Dictionary of callback functions.
        """
        self.hub_session = overrides.pop("session", None)  # HUB
        self.args = get_cfg(cfg, overrides)
        self.check_resume(overrides)
        self.args.device = parse_device(self.args.device)  # canonical string, resolves '-1' auto-selection once
        self.device = select_device(self.args.device)
        self._dist_env = _distributed_env()
        _validate_cuda_ddp_device(self.device, self._dist_env)
        self.validator = None
        self.metrics = None
        self.plots = {}
        init_seeds(self.args.seed + 1 + RANK, deterministic=self.args.deterministic)

        # Dirs
        self.save_dir = get_save_dir(self.args)
        self.args.name = self.save_dir.name  # update name for loggers
        self.wdir = self.save_dir / "weights"  # weights dir
        if RANK in {-1, 0}:
            self.wdir.mkdir(parents=True, exist_ok=True)  # make dir
            self.args.save_dir = str(self.save_dir)
            self._save_run_args()
        self.last, self.best = self.wdir / "last.pt", self.wdir / "best.pt"  # checkpoint paths
        self.healthy = self.wdir / "last_healthy.pt"
        self.save_period = self.args.save_period

        self.batch_size = self.args.batch
        self.epochs = self.args.epochs or 100  # in case users accidentally pass epochs=None with timed training
        self.start_epoch = 0
        if RANK == -1:
            print_args(vars(self.args))

        # Device
        if self.device.type in {"cpu", "mps"}:
            self.args.workers = 0  # faster CPU training as time dominated by inference, not dataloading

        # Callbacks - initialize early so on_pretrain_routine_start can capture original args.data
        self.callbacks = _callbacks or callbacks.get_default_callbacks()

        if self.device.type in {"cpu", "mps"}:
            world_size = 0
        else:  # i.e. device='0', '0,1,2,3', 'npu:0', or '' auto-selecting a single GPU
            world_size = len(self.args.device.split(",")) if self.args.device else 1

        self.ddp = world_size > 1 and "LOCAL_RANK" not in os.environ
        self.world_size = world_size
        # Run on_pretrain_routine_start before get_dataset() to capture original args.data (e.g., ul:// URIs)
        if RANK in {-1, 0} and not self.ddp:
            callbacks.add_integration_callbacks(self)
            self.run_callbacks("on_pretrain_routine_start")

        # Model and Dataset
        self.model = check_model_file_from_stem(self.args.model)  # add suffix, i.e. yolo26n -> yolo26n.pt
        with torch_distributed_zero_first(LOCAL_RANK):  # avoid auto-downloading dataset multiple times
            self.data = self.get_dataset()

        self.ema = None

        # Optimization utils init
        self.lf = None
        self.scheduler = None

        # Epoch level metrics
        self.best_fitness = None
        self.fitness = None
        self.loss = None
        self.tloss = None
        self.loss_names = ["Loss"]
        self.csv = self.save_dir / "results.csv"
        if self.csv.exists() and not self.args.resume:
            self.csv.unlink()
        self.plot_idx = [0, 1, 2]
        self.nan_recovery_attempts = 0
        self.adapter_controller = AdapterRuntimeController(self)
        self.mixture_controller = MixtureRuntimeController(self)
        self.recovery_controller = TrainingRecoveryController(self)

    def _save_run_args(self) -> None:
        """Persist requested and effective run arguments on the main process."""
        if RANK not in {-1, 0}:
            return
        args_dict = vars(self.args).copy()
        if args_dict.get("augmentations") is not None:
            # Serialize Albumentations transforms as repr strings for checkpoint compatibility.
            args_dict["augmentations"] = [repr(transform) for transform in args_dict["augmentations"]]
        YAML.save(self.save_dir / "args.yaml", args_dict)

    def add_callback(self, event: str, callback):
        """Append the given callback to the event's callback list."""
        self.callbacks[event].append(callback)

    def set_callback(self, event: str, callback):
        """Override the existing callbacks with the given callback for the specified event."""
        self.callbacks[event] = [callback]

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.callbacks.get(event, []):
            callback(self)

    def train(self):
        """Execute the training process, using DDP subprocess for multi-GPU or direct training for single-GPU."""
        # Run subprocess if DDP training, else train normally
        if self.ddp:
            # Argument checks
            if self.args.rect:
                LOGGER.warning("'rect=True' is incompatible with Multi-GPU training, setting 'rect=False'")
                self.args.rect = False
            if self.args.batch < 1.0:
                raise ValueError(
                    "AutoBatch with batch<1 not supported for Multi-GPU training, "
                    f"please specify a valid batch size multiple of GPU count {self.world_size}, i.e. batch={self.world_size * 8}."
                )

            # Command
            cmd, file = None, None
            try:
                cmd, file = generate_ddp_command(self)
                LOGGER.info(f"{colorstr('DDP:')} debug command {' '.join(cmd)}")
                subprocess.run(cmd, check=True, env=ddp_launch_env())
            except subprocess.CalledProcessError as e:
                worker_errors = collect_ddp_error_logs(getattr(self, "ddp_log_dir", ""))
                if worker_errors:
                    LOGGER.error(f"DDP persisted worker root cause:\n{worker_errors}")
                LOGGER.error(
                    f"DDP worker process failed with exit code {e.returncode}. "
                    "The worker Root Cause is printed above and persisted under the DDP log directory."
                )
                raise
            finally:
                if file is not None:
                    ddp_cleanup(self, str(file))

        else:
            try:
                self._do_train()
            finally:
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()

    def _setup_scheduler(self):
        """Initialize training learning rate scheduler."""
        if self.args.cos_lr:
            self.lf = one_cycle(1, self.args.lrf, self.epochs)  # cosine 1->hyp['lrf']
        else:
            self.lf = lambda x: max(1 - x / self.epochs, 0) * (1.0 - self.args.lrf) + self.args.lrf  # linear
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=self.lf)

    def _setup_ddp(self):
        """Initialize and set the DistributedDataParallel parameters for training."""
        index = int(self.args.device.split(",")[LOCAL_RANK])  # world_size > 1 guarantees a multi-device string
        torch.cuda.set_device(index)
        self.device = torch.device("cuda", index)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"  # set to enforce timeout
        dist.init_process_group(
            backend="nccl" if dist.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),  # 3 hours
            rank=RANK,
            world_size=self.world_size,
        )

    def _build_train_pipeline(self):
        """Build dataloaders, optimizer, and scheduler for current batch size."""
        batch_size = self.batch_size // max(self.world_size, 1)
        self.train_loader = self.get_dataloader(
            self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
        )
        # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
        self.test_loader = self.get_dataloader(
            self.data.get("val") or self.data.get("test"),
            batch_size=batch_size if self.args.task in {"obb", "semantic"} else batch_size * 2,
            rank=LOCAL_RANK,
            mode="val",
        )
        self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
        weight_decay = self.args.weight_decay * self.batch_size * self.accumulate / self.args.nbs  # scale weight_decay
        iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
        self.adapter_controller.prepare_optimizer(iterations)
        self.optimizer = self.build_optimizer(
            model=self.model,
            name=self.args.optimizer,
            lr=self.args.lr0,
            momentum=self.args.momentum,
            decay=weight_decay,
            iterations=iterations,
        )
        self.adapter_controller.configure_optimizer(self.optimizer)
        self.args.effective_optimizer = type(self.optimizer).__name__
        self.args.effective_optimizer_lrs = [float(group["lr"]) for group in self.optimizer.param_groups]
        self._save_run_args()
        self._setup_scheduler()

    def _setup_train(self):
        """Configure model, optimizer, dataloaders, and training utilities before the training loop."""
        ckpt = self.setup_model()
        self.model = self.model.to(self.device)
        self.set_model_attributes()
        self.adapter_controller.setup()
        self.mixture_controller.setup()
        has_mixture_loss = has_routed_modules(unwrap_model(self.model))
        if has_mixture_loss:
            from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer

            initialize_mixture_loss_ema_buffer(unwrap_model(self.model))

        # Compile model (knowledge distillation runs the wrapped model eagerly and relies on
        # find_unused_parameters under DDP for the frozen teacher, so disable compilation when distilling)
        if self.args.distill_model is not None and self.args.compile:
            LOGGER.warning("'compile' is not supported with knowledge distillation and will be disabled.")
            self.args.compile = False
        self.model = attempt_compile(self.model, device=self.device, mode=self.args.compile)

        self._freeze_model_parameters()

        # Check AMP
        self.amp = torch.tensor(self.args.amp).to(self.device)  # True or False
        if self.amp and RANK in {-1, 0}:  # Single-GPU and DDP
            callbacks_backup = callbacks.default_callbacks.copy()  # backup callbacks as check_amp() resets them
            self.amp = torch.tensor(check_amp(self.model), device=self.device)
            callbacks.default_callbacks = callbacks_backup  # restore callbacks
        if RANK > -1 and self.world_size > 1:  # DDP
            amp_flag = self.amp.to(dtype=torch.int32)
            dist.broadcast(amp_flag, src=0)  # broadcast from rank 0 to all other ranks
            self.amp = bool(amp_flag.item())
        else:
            self.amp = bool(self.amp)
        self.scaler = (
            torch.amp.GradScaler("cuda", enabled=self.amp) if TORCH_2_4 else torch.cuda.amp.GradScaler(enabled=self.amp)
        )
        # Check imgsz
        gs = max(int(self.model.stride.max() if hasattr(self.model, "stride") else 32), 32)  # grid size (max stride)
        self.args.imgsz = check_imgsz(self.args.imgsz, stride=gs, floor=gs, max_dim=1)
        self.stride = gs  # for multiscale training

        # resume training would directly load DistillationModel so check here
        if self.args.distill_model is not None and not isinstance(unwrap_model(self.model), DistillationModel):
            self.model = DistillationModel(student_model=self.model, teacher_model=self.args.distill_model)
        if self.world_size > 1:
            # static_graph=True permits params used >1 time per forward (e.g. flow_model in
            # o2m+o2o pose loss branches) under torch.compile.
            self.model = nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.device.index],
                static_graph=bool(self.args.compile and not has_mixture_loss),
                broadcast_buffers=False,
                find_unused_parameters=bool(has_mixture_loss or not self.args.compile),
            )

        # Batch size
        if self.batch_size < 1 and RANK == -1:  # single-GPU only, estimate best batch size
            self.args.batch = self.batch_size = self.auto_batch()
        if self.world_size > 1 and self.batch_size % self.world_size:
            raise ValueError(f"batch={self.batch_size} must be divisible by world_size={self.world_size}")
        if self.batch_size // max(self.world_size, 1) == 1 and self.args.imgsz < 2 * gs:
            raise ValueError(
                f"batch=1 training at imgsz={self.args.imgsz} gives BatchNorm a single value per channel; "
                f"increase batch or use imgsz >= {2 * gs}"
            )

        self._build_train_pipeline()
        self.validator = self.get_validator()
        if has_mixture_loss:
            self.loss_names = (*self.loss_names, "mixture_aux_loss")
        if self.args.distill_model is not None and "dis_loss" not in self.loss_names:
            self.loss_names += ("dis_loss",)
        self.ema = ModelEMA(self.model)
        self.set_class_weights()  # compute class weights after dataloader is ready
        if RANK in {-1, 0}:
            metric_keys = self.validator.metrics.keys + self.label_loss_items(prefix="val")
            self.metrics = dict(zip(metric_keys, [0] * len(metric_keys)))
            if self.args.plots:
                self.plot_training_labels()

        self.stopper, self.stop = EarlyStopping(patience=self.args.patience), False
        self.resume_training(ckpt)
        self.scheduler.last_epoch = self.start_epoch - 1  # do not move
        self._bootstrap_healthy_checkpoint()
        self.run_callbacks("on_pretrain_routine_end")

    def _freeze_model_parameters(self) -> None:
        """Apply explicit layer freezing without re-enabling adapter base parameters."""
        freeze_list = (
            self.args.freeze
            if isinstance(self.args.freeze, list)
            else range(self.args.freeze)
            if isinstance(self.args.freeze, int)
            else []
        )
        adapter_active = self.adapter_controller.active
        always_freeze_names = [] if adapter_active else [".dfl"]  # adapters may need a freshly initialized head
        freeze_layer_names = [f"model.{x}." for x in freeze_list] + always_freeze_names
        if isinstance(unwrap_model(self.model), DistillationModel):
            freeze_layer_names.append("teacher_model.")
        self.freeze_layer_names = freeze_layer_names
        for name, parameter in self.model.named_parameters():
            if any(layer_name in name for layer_name in freeze_layer_names):
                LOGGER.info(f"Freezing layer '{name}'")
                parameter.requires_grad = False
            elif not parameter.requires_grad and parameter.dtype.is_floating_point and not adapter_active:
                LOGGER.warning(
                    f"setting 'requires_grad=True' for frozen layer '{name}'. "
                    "See ultralytics.engine.trainer for customization of frozen layers."
                )
                parameter.requires_grad = True
        if not any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError(
                f"'freeze={self.args.freeze}' froze the entire model with no trainable parameters left. "
                f"Reduce 'freeze' or pass a list of specific layer indices."
            )

    def _detect_moa_mot_modules(self):
        """Compatibility wrapper for routed module discovery."""
        controller = getattr(self, "mixture_controller", None)
        if controller is None:
            controller = self.mixture_controller = MixtureRuntimeController(self)
        return controller.detect_modules()

    def _resolve_mixture_runtime_config(self):
        """Compatibility wrapper for canonical mixture configuration."""
        controller = getattr(self, "mixture_controller", None)
        if controller is None:
            controller = self.mixture_controller = MixtureRuntimeController(self)
        return controller.resolve_config()

    def _anneal_moa_mot_temperature(self):
        """Compatibility wrapper for mixture temperature scheduling."""
        controller = getattr(self, "mixture_controller", None)
        if controller is None:
            controller = self.mixture_controller = MixtureRuntimeController(self)
        return controller.anneal_temperature()

    def _disable_gradient_checkpointing_for_ddp_moe_lora(self):
        """Compatibility wrapper for DDP routed-module safety."""
        controller = getattr(self, "mixture_controller", None)
        if controller is None:
            controller = self.mixture_controller = MixtureRuntimeController(self)
        return controller.prepare_ddp()

    def _finalize_moe_map_saturation_epoch(self, *, recovered: bool, validated: bool):
        """Compatibility wrapper for validation-driven MoE balance scheduling."""
        controller = getattr(self, "mixture_controller", None)
        if controller is None:
            controller = self.mixture_controller = MixtureRuntimeController(self)
        return controller.finalize_epoch(recovered=recovered, validated=validated)

    def _compute_distillation_loss(self, student_preds, teacher_preds, adaptive_temp=False):
        return self.adapter_controller.compute_distillation_loss(student_preds, teacher_preds, adaptive_temp)

    def _compute_response_distillation_loss(self, student_preds, teacher_preds):
        return self.adapter_controller.compute_response_distillation_loss(student_preds, teacher_preds)

    def _compute_prediction_entropy(self, preds):
        return self.adapter_controller.compute_prediction_entropy(preds)

    def _init_hierarchical_distill_cache(self):
        return self.adapter_controller.init_hierarchical_distill_cache()

    def _compute_hierarchical_distillation_loss(self, images, layer_indices):
        return self.adapter_controller.compute_hierarchical_distillation_loss(images, layer_indices)

    def _do_train(self):
        """Perform the full training loop including setup, epoch iteration, validation, and final evaluation."""
        if self.world_size > 1:
            self._setup_ddp()
        self._setup_train()

        nb = len(self.train_loader)  # number of batches
        nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1  # warmup iterations
        last_opt_step = -1
        self.epoch_time = None
        self.epoch_time_start = time.time()
        self.train_time_start = time.time()
        self.run_callbacks("on_train_start")
        LOGGER.info(
            f"Image sizes {self.args.imgsz} train, {self.args.imgsz} val\n"
            f"Using {self.train_loader.num_workers * (self.world_size or 1)} dataloader workers\n"
            f"Logging results to {colorstr('bold', self.save_dir)}\n"
            f"Starting training for " + (f"{self.args.time} hours..." if self.args.time else f"{self.epochs} epochs...")
        )
        if self.args.close_mosaic:
            base_idx = (self.epochs - self.args.close_mosaic) * nb
            self.plot_idx.extend([base_idx, base_idx + 1, base_idx + 2])
        epoch = self.start_epoch
        self.optimizer.zero_grad()  # zero any resumed gradients to ensure stability on train start
        self._oom_retries = 0  # OOM auto-reduce counter for first epoch
        while True:
            self.epoch = epoch
            self.mixture_controller.begin_epoch(epoch)
            self.adapter_controller.begin_epoch(epoch)
            self.run_callbacks("on_train_epoch_start")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # suppress 'Detected lr_scheduler.step() before optimizer.step()'
                self.scheduler.step()

            self._model_train()
            if RANK != -1:
                self.train_loader.sampler.set_epoch(epoch)
            pbar = enumerate(self.train_loader)
            # Update dataloader attributes (optional)
            if epoch == (self.epochs - self.args.close_mosaic):
                self._close_dataloader_mosaic()
                self.train_loader.reset()

            if RANK in {-1, 0}:
                LOGGER.info(self.progress_string())
                pbar = TQDM(enumerate(self.train_loader), total=nb)
            self.tloss = None
            for i, batch in pbar:
                self.run_callbacks("on_train_batch_start")
                # Warmup
                ni = i + nb * epoch
                if ni <= nw:
                    xi = [0, nw]  # x interp
                    self.accumulate = max(1, int(np.interp(ni, xi, [1, self.args.nbs / self.batch_size]).round()))
                    for x in self.optimizer.param_groups:
                        # Bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                        x["lr"] = float(
                            np.interp(
                                ni,
                                xi,
                                [
                                    self.args.warmup_bias_lr if x.get("param_group") == "bias" else 0.0,
                                    x["initial_lr"] * self.lf(epoch),
                                ],
                            )
                        )
                        if "momentum" in x:
                            x["momentum"] = float(np.interp(ni, xi, [self.args.warmup_momentum, self.args.momentum]))

                should_step = ni - last_opt_step >= self.accumulate or i == nb - 1
                sync_context = self.model.no_sync() if RANK != -1 and not should_step else nullcontext()
                try:
                    with sync_context:
                        with autocast(self.amp):
                            batch = self.preprocess_batch(batch)
                            if self.args.compile:
                                # Decouple inference and loss calculations for improved compile performance
                                preds = self.model(batch["img"])
                                loss, self.loss_items = unwrap_model(self.model).loss(batch, preds)
                            else:
                                loss, self.loss_items = self.model(batch)
                            self.mixture_controller.collect_routing_usage(batch_weight=batch["img"].shape[0])
                            loss = self.adapter_controller.augment_loss(loss)
                            loss = self.adapter_controller.augment_few_shot_loss(loss, batch["img"], epoch)
                            self.loss = loss.sum()
                            if RANK != -1:
                                self.loss *= self.world_size
                            self.tloss = (
                                self.loss_items if self.tloss is None else (self.tloss * i + self.loss_items) / (i + 1)
                            )

                        self.scaler.scale(self.loss).backward()
                except RuntimeError as e:
                    is_oom = "out of memory" in str(e).lower()  # torch.cuda.OutOfMemoryError requires torch>=1.13
                    if not is_oom and not any(
                        s in str(e) for s in ("CUDNN_STATUS_INTERNAL_ERROR", "unable to find an engine")
                    ):
                        raise
                    if epoch > self.start_epoch or self._oom_retries >= 3 or RANK != -1:
                        raise  # only auto-reduce during first epoch on single GPU, max 3 retries
                    self._oom_retries += 1
                    old_batch = self.batch_size
                    self.args.batch = self.batch_size = max(self.batch_size // 2, 1)
                    LOGGER.warning(
                        f"{'CUDA out of memory' if is_oom else 'CUDA backend memory error'} with batch={old_batch}. "
                        f"Reducing to batch={self.batch_size} and retrying ({self._oom_retries}/3)."
                    )
                    batch = loss = preds = None
                    self.loss = self.loss_items = self.tloss = None
                    self._clear_memory()
                    self._build_train_pipeline()  # rebuild dataloaders, optimizer, scheduler
                    self.scheduler.last_epoch = self.start_epoch - 1
                    nb = len(self.train_loader)
                    nw = max(round(self.args.warmup_epochs * nb), 100) if self.args.warmup_epochs > 0 else -1
                    last_opt_step = -1
                    self.optimizer.zero_grad()
                    break  # restart epoch loop with reduced batch size
                if should_step:
                    self.optimizer_step()
                    last_opt_step = ni

                    # Timed stopping
                    if self.args.time:
                        self.stop = (time.time() - self.train_time_start) > (self.args.time * 3600)
                        if RANK != -1:  # if DDP training
                            broadcast_list = [self.stop if RANK == 0 else None]
                            dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                            self.stop = broadcast_list[0]
                        if self.stop:  # training time exceeded
                            break

                # Log
                if RANK in {-1, 0}:
                    loss_length = self.tloss.shape[0] if len(self.tloss.shape) else 1
                    pbar.set_description(
                        ("%11s" * 2 + "%11.4g" * (2 + loss_length))
                        % (
                            f"{epoch + 1}/{self.epochs}",
                            f"{self._get_memory():.3g}G",  # (GB) GPU memory util
                            *(self.tloss if loss_length > 1 else torch.unsqueeze(self.tloss, 0)),  # losses
                            batch.get("cls", batch["img"]).shape[0],  # no. of instances
                            batch["img"].shape[-1],  # imgsz, i.e 640
                        )
                    )
                    self.run_callbacks("on_batch_end")
                    if self.args.plots and ni in self.plot_idx:
                        self.plot_training_samples(batch, ni)

                self.run_callbacks("on_train_batch_end")
                if self.stop:
                    break  # allow external stop (e.g. platform cancellation) between batches
            else:
                # for/else: this block runs only when the for loop completes without break (no OOM retry)
                self._oom_retries = 0  # reset OOM counter after successful first epoch

            if self._oom_retries and not self.stop:
                continue  # OOM recovery broke the for loop, restart with reduced batch size

            if hasattr(unwrap_model(self.model).criterion, "update"):
                unwrap_model(self.model).criterion.update()

            self.lr = {f"lr/pg{ir}": x["lr"] for ir, x in enumerate(self.optimizer.param_groups)}  # for loggers

            self.run_callbacks("on_train_epoch_end")
            if RANK in {-1, 0}:
                self.ema.update_attr(self.model, include=["yaml", "nc", "args", "names", "stride", "class_weights"])

            # Validation
            final_epoch = epoch + 1 >= self.epochs
            validated = self._sync_validation_gate(
                self.args.val or final_epoch or self.stopper.possible_stop or self.stop
            )
            if validated:
                self._clear_memory(None if self.device.type == "mps" else 0.5)  # prevent VRAM spike
                if self._recover_before_validation(epoch):
                    self._finalize_moe_map_saturation_epoch(recovered=True, validated=True)
                    continue
                self.metrics, self.fitness = self.validate()

            # NaN recovery
            if self._handle_nan_recovery(epoch):
                self._finalize_moe_map_saturation_epoch(recovered=True, validated=validated)
                continue
            self._finalize_moe_map_saturation_epoch(recovered=False, validated=validated)

            self.nan_recovery_attempts = 0
            rank0_epoch_end_error = None
            if RANK in {-1, 0}:
                try:
                    self.save_metrics(metrics={**self.label_loss_items(self.tloss), **self.metrics, **self.lr})
                    self.stop |= self.stopper(epoch + 1, self.fitness) or final_epoch
                    if self.args.time:
                        self.stop |= (time.time() - self.train_time_start) > (self.args.time * 3600)

                    # Save standard checkpoints first; otherwise keep the independent
                    # online recovery point current even when save=False.
                    if self.args.save or final_epoch:
                        if self.save_model():
                            self.run_callbacks("on_model_save")
                    else:
                        self._refresh_healthy_checkpoint()
                except Exception as exc:
                    if RANK == -1:
                        raise
                    rank0_epoch_end_error = f"{type(exc).__name__}: {exc}"
            self._sync_rank0_epoch_end_result(rank0_epoch_end_error)

            # Scheduler
            t = time.time()
            self.epoch_time = t - self.epoch_time_start
            self.epoch_time_start = t
            if self.args.time:
                mean_epoch_time = (t - self.train_time_start) / (epoch - self.start_epoch + 1)
                self.epochs = self.args.epochs = math.ceil(self.args.time * 3600 / mean_epoch_time)
                self._setup_scheduler()
                self.scheduler.last_epoch = self.epoch  # do not move
                self.stop |= epoch >= self.epochs  # stop if exceeded epochs
            self.run_callbacks("on_fit_epoch_end")
            # clear if memory utilization > 50%; always clear on MPS due to leak https://github.com/ultralytics/ultralytics/issues/22621
            self._clear_memory(None if self.device.type == "mps" else 0.5)

            # Early Stopping
            if RANK != -1:  # if DDP training
                broadcast_list = [self.stop if RANK == 0 else None]
                dist.broadcast_object_list(broadcast_list, 0)  # broadcast 'stop' to all ranks
                self.stop = broadcast_list[0]
            if self.stop:
                break  # must break all DDP ranks
            epoch += 1

        seconds = time.time() - self.train_time_start
        LOGGER.info(f"\n{epoch - self.start_epoch + 1} epochs completed in {seconds / 3600:.3f} hours.")
        # Do final val with best.pt
        self.final_eval()
        if RANK in {-1, 0}:
            if self.args.plots:
                self.plot_metrics()
            self.run_callbacks("on_train_end")
        self._clear_memory()
        for loader in (self.train_loader, self.test_loader):
            if hasattr(loader, "close"):
                loader.close()  # shut down persistent dataloader workers so none survive to interpreter exit
        unset_deterministic()
        self.run_callbacks("teardown")

    def auto_batch(self, max_num_obj=0, dataset_size=0):
        """Calculate optimal batch size based on model and device memory constraints."""
        max_imgsz = int(self.args.imgsz * (1 + self.args.multi_scale))  # need not be stride-aligned
        return check_train_batch_size(
            model=self.model,
            imgsz=max_imgsz,
            amp=self.amp,
            batch=self.batch_size,
            max_num_obj=max_num_obj,
            dataset_size=dataset_size,
        )  # returns batch size

    def _get_memory(self, fraction=False):
        """Get accelerator memory utilization in GB or as a fraction of total memory."""
        memory, total = 0, 0
        if self.device.type == "mps":
            memory = torch.mps.driver_allocated_memory()
            if fraction:
                return __import__("psutil").virtual_memory().percent / 100
        elif self.device.type != "cpu":
            memory = torch.cuda.memory_reserved()
            if fraction:
                total = torch.cuda.get_device_properties(self.device).total_memory
        return ((memory / total) if total > 0 else 0) if fraction else (memory / 2**30)

    def _clear_memory(self, threshold: float | None = None):
        """Clear accelerator memory by calling garbage collector and emptying cache."""
        if threshold:
            assert 0 <= threshold <= 1, "Threshold must be between 0 and 1."
            if self._get_memory(fraction=True) <= threshold:
                return
        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()
        elif self.device.type == "cpu":
            return
        else:
            torch.cuda.empty_cache()

    def read_results_csv(self):
        """Read results.csv into a dictionary using polars."""
        import polars as pl  # scope for faster 'import ultralytics'

        try:
            return pl.read_csv(self.csv, infer_schema_length=None).to_dict(as_series=False)
        except Exception:
            return {}

    def _model_train(self):
        """Set model in training mode."""
        self.model.train()
        # Freeze BN stat
        for n, m in self.model.named_modules():
            if any(filter(lambda f: f in n, self.freeze_layer_names)) and isinstance(m, nn.BatchNorm2d):
                m.eval()

    def save_model(self):
        """Save standard last/best checkpoints following the upstream Ultralytics lifecycle."""
        serialized_ckpt = self._serialize_checkpoint()
        self.wdir.mkdir(parents=True, exist_ok=True)
        self.last.write_bytes(serialized_ckpt)
        if self.best_fitness == self.fitness:
            self.best.write_bytes(serialized_ckpt)
        if (self.save_period > 0) and (self.epoch % self.save_period == 0):
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)  # save epoch, i.e. 'epoch3.pt'
        self._refresh_healthy_checkpoint()
        return True

    def get_dataset(self):
        """Get train and validation datasets from data dictionary.

        Returns:
            (dict): A dictionary containing the training/validation/test dataset and category names.
        """
        try:
            self.args.data = convert_ndjson_to_yolo_if_needed(self.args.data)

            # Task-specific dataset checking
            if self.args.task == "classify":
                data = check_cls_dataset(self.args.data)
            elif str(self.args.data).rsplit(".", 1)[-1] in {"yaml", "yml"} or self.args.task in {
                "detect",
                "segment",
                "pose",
                "obb",
                "semantic",
            }:
                data = check_det_dataset(self.args.data)
                if "yaml_file" in data:
                    self.args.data = data["yaml_file"]  # for validating 'yolo train data=url.zip' usage
        except Exception as e:
            raise RuntimeError(emojis(f"Dataset '{clean_url(self.args.data)}' error ❌ {e}")) from e
        if self.args.single_cls:
            LOGGER.info("Overriding class names with single class.")
            data["names"] = {0: "item"}
            data["nc"] = 1
        return data

    def setup_model(self):
        """Load, create, or download model for any task.

        Returns:
            (dict | None): Checkpoint to resume training from, or None if no checkpoint is loaded.
        """
        if isinstance(self.model, torch.nn.Module):  # if model is loaded beforehand. No setup needed
            return

        cfg, weights = self.model, None
        ckpt = None
        if str(self.model).endswith(".pt"):
            weights, ckpt = load_checkpoint(self.model)
            cfg = weights.yaml
        if isinstance(self.args.pretrained, (str, Path)):
            weights, _ = load_checkpoint(self.args.pretrained)
        elif self.args.pretrained is False and not self.resume:
            weights = None

        # rebuild DistillationModel from resuming checkpoint
        if isinstance(weights, DistillationModel):
            if RANK in {-1, 0}:
                LOGGER.info("Resuming training DistillationModel from checkpoint weights")
            student_model = self.get_model(cfg=cfg, weights=weights.student_model, verbose=RANK in {-1, 0})
            student_model.args = self.args
            # teacher is stripped from the checkpoint to save memory/disk; rebuild it from the distill_model path
            teacher_model = weights.teacher_model if weights.teacher_model is not None else self.args.distill_model
            model = DistillationModel(student_model=student_model, teacher_model=teacher_model)
            if getattr(weights, "projector", None) is not None:
                model.projector.load_state_dict(weights.projector.state_dict())  # restore the trained projector
            model.criterion = None
            self.model = model
        else:
            self.model = self.get_model(cfg=cfg, weights=weights, verbose=RANK in {-1, 0})  # calls Model(cfg, weights)
        return ckpt

    def optimizer_step(self):
        """Perform a single step of the training optimizer with gradient clipping and EMA update."""
        self.scaler.unscale_(self.optimizer)  # unscale gradients
        local_nonfinite = any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all().item())
            for parameter in self.model.parameters()
        )
        if self._sync_nonfinite_flag(local_nonfinite):
            self._gradient_nonfinite = True
            self.optimizer.zero_grad()
            self.scaler.update()
            return False
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        controller = getattr(self, "adapter_controller", None)
        if controller is not None:
            controller.after_optimizer_step()
        self.optimizer.zero_grad()
        if self.ema:
            self.ema.update(self.model)
        return True

    def preprocess_batch(self, batch):
        """Allow custom preprocessing of model inputs and ground truths depending on task type."""
        return batch

    def validate(self):
        """Run validation on val set using self.validator.

        Returns:
            (tuple): A tuple containing:
                - metrics (dict | None): Dictionary of validation metrics, or None if validation was skipped.
                - fitness (float | None): Fitness score for the validation, or None if validation was skipped.
        """
        self._sync_ema_buffers_for_validation()
        ema_model = getattr(getattr(self, "ema", None), "ema", None)
        if ema_model is not None and not self._state_is_finite(unwrap_model(ema_model)):
            self._ema_nonfinite = True
            return {}, float("nan")
        self._ema_nonfinite = False
        try:
            metrics = self.validator(self)
        except Exception as exc:
            from ultralytics.utils.errors import MoERouterError

            if isinstance(exc, MoERouterError):
                return {}, float("nan")
            raise
        if metrics is None:
            return None, None
        fitness = metrics.pop("fitness", -self.loss.detach().cpu().numpy())  # use loss as fitness measure if not found
        if self.best_fitness is None or self.best_fitness < fitness:
            self.best_fitness = fitness
        return metrics, fitness

    def _recovery_controller(self):
        """Return the recovery controller, including compatibility for lightweight test trainers."""
        controller = getattr(self, "recovery_controller", None)
        if controller is None:
            controller = self.recovery_controller = TrainingRecoveryController(self)
        return controller

    @staticmethod
    def _state_is_finite(value):
        return TrainingRecoveryController.state_is_finite(value)

    def _sync_nonfinite_flag(self, local_nonfinite):
        return self._recovery_controller().sync_nonfinite_flag(local_nonfinite)

    def _sync_validation_gate(self, local_validated):
        """Use rank 0's validation decision so every DDP rank enters the same collectives."""
        if RANK == -1 or not dist.is_initialized():
            return bool(local_validated)
        decision = [bool(local_validated) if RANK == 0 else None]
        dist.broadcast_object_list(decision, src=0)
        return bool(decision[0])

    @staticmethod
    def _sync_rank0_epoch_end_result(rank0_error=None):
        """Propagate rank-0-only save/callback failures before the next collective."""
        if RANK == -1 or not dist.is_initialized():
            if rank0_error:
                raise RuntimeError(f"Rank 0 epoch-end checkpoint stage failed: {rank0_error}")
            return
        result = [rank0_error if RANK == 0 else None]
        dist.broadcast_object_list(result, src=0)
        if result[0]:
            raise RuntimeError(f"Rank 0 epoch-end checkpoint stage failed: {result[0]}")

    def _sync_ema_buffers_for_validation(self):
        return self._recovery_controller().sync_ema_buffers()

    def _serialize_checkpoint(self, *, include_online_model=False):
        return self._recovery_controller().serialize_checkpoint(include_online_model=include_online_model)

    def _checkpoint_forward_smoke(self, checkpoint):
        return self._recovery_controller().checkpoint_forward_smoke(checkpoint)

    def _validate_checkpoint_artifact(self, path):
        return self._recovery_controller().validate_artifact(path)

    def _select_final_eval_checkpoints(self):
        return self._recovery_controller().select_final_eval_checkpoints()

    def _reset_non_checkpoint_moe_runtime_state(self):
        return self._recovery_controller().reset_runtime(getattr(self, "model", None))

    def _save_healthy_checkpoint(self, serialized_ckpt, *, verify_forward=False):
        if not self._check_mox_aux_finite(getattr(self, "model", None)):
            return False
        # Check live state before serialization so normal epoch saves do not
        # deserialize the complete checkpoint and run CPU inference again.
        model = getattr(self, "model", None)
        optimizer = getattr(self, "optimizer", None)
        scaler = getattr(self, "scaler", None)
        live_state_available = model is not None and optimizer is not None and scaler is not None
        finite_state = None
        if live_state_available:
            finite_state = all(
                self._state_is_finite(state)
                for state in (
                    unwrap_model(model),
                    getattr(getattr(self, "ema", None), "ema", None),
                    optimizer.state_dict(),
                    scaler.state_dict(),
                )
                if state is not None
            )
        return self._recovery_controller().save_healthy(
            serialized_ckpt, state_verified=finite_state, verify_forward=verify_forward
        )

    def _bootstrap_healthy_checkpoint(self):
        return self._recovery_controller().bootstrap()

    def _refresh_healthy_checkpoint(self):
        """Best-effort refresh that never blocks standard last/best checkpoint saving."""
        try:
            return self._recovery_controller().refresh_healthy()
        except (OSError, RuntimeError, ValueError) as exc:
            LOGGER.warning(f"Recovery checkpoint refresh failed; preserving the previous file: {exc}")
            return False

    @staticmethod
    def _check_mox_aux_finite(model=None):
        """Compatibility check for legacy routed auxiliary registries."""
        return TrainingRecoveryController.aux_state_is_finite()

    def _collect_prevalidation_nonfinite_flags(self):
        """Collect model and EMA non-finite flags across initialized ranks."""
        model_nonfinite = getattr(self, "model", None) is not None and not self._state_is_finite(
            unwrap_model(self.model)
        )
        ema = getattr(getattr(self, "ema", None), "ema", None)
        ema_nonfinite = ema is not None and not self._state_is_finite(unwrap_model(ema))
        return {
            "model_nonfinite": self._sync_nonfinite_flag(model_nonfinite),
            "ema_nonfinite": self._sync_nonfinite_flag(ema_nonfinite),
        }

    def _recover_before_validation(self, epoch):
        """Recover before validation if the online or EMA model is already non-finite."""
        flags = self._collect_prevalidation_nonfinite_flags()
        if flags["ema_nonfinite"]:
            self._recovery_controller().resync_nonfinite_ema()
            flags = self._collect_prevalidation_nonfinite_flags()
        if not any(flags.values()):
            return False
        self.fitness = float("nan")
        recovered = self._handle_nan_recovery(epoch)
        if recovered:
            # The live graph is finite again. Validate and checkpoint the restored state
            # instead of replaying an epoch that may repeat a deterministic callback fault.
            return False
        return any(self._collect_prevalidation_nonfinite_flags().values())

    def _record_nonfinite_diagnostic(self, component, *, epoch, step, loss_items=None, parameter=None):
        """Record the first local non-finite event for recovery diagnostics."""
        if getattr(self, "_nonfinite_diagnostic", None) is None:
            self._nonfinite_diagnostic = {
                "component": component,
                "epoch": epoch,
                "step": step,
                "loss_items": None if loss_items is None else str(loss_items),
                "parameter": parameter,
            }

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Get model and raise NotImplementedError for loading cfg files."""
        raise NotImplementedError("This task trainer doesn't support loading cfg files")

    def get_validator(self):
        """Raise NotImplementedError (must be implemented by subclasses)."""
        raise NotImplementedError("get_validator function not implemented in trainer")

    def get_dataloader(self, dataset_path, batch_size=16, rank=0, mode="train"):
        """Raise NotImplementedError (must return a `torch.utils.data.DataLoader` in subclasses)."""
        raise NotImplementedError("get_dataloader function not implemented in trainer")

    def build_dataset(self, img_path, mode="train", batch=None):
        """Build dataset."""
        raise NotImplementedError("build_dataset function not implemented in trainer")

    def label_loss_items(self, loss_items=None, prefix="train"):
        """Return a loss dict with labeled training loss items, or a list of loss names if loss_items is None.

        Notes:
            This is not needed for classification but necessary for segmentation & detection.
        """
        return {"loss": loss_items} if loss_items is not None else ["loss"]

    def set_model_attributes(self):
        """Set or update model parameters before training."""
        self.model.names = self.data["names"]

    def set_class_weights(self):
        """Compute and set class weights for handling class imbalance. Override in subclasses."""
        pass

    def build_targets(self, preds, targets):
        """Build target tensors for training YOLO model."""
        pass

    def progress_string(self):
        """Return a string describing training progress."""
        return ""

    # TODO: may need to put these following functions into callback
    def plot_training_samples(self, batch, ni):
        """Plot training samples during YOLO training."""
        pass

    def plot_training_labels(self):
        """Plot training labels for YOLO model."""
        pass

    def save_metrics(self, metrics):
        """Save training metrics to a CSV file."""
        keys, vals = list(metrics.keys()), list(metrics.values())
        n = len(metrics) + 2  # number of cols
        t = time.time() - self.train_time_start
        self.csv.parent.mkdir(parents=True, exist_ok=True)  # ensure parent directory exists
        s = "" if self.csv.exists() else ("%s," * n % ("epoch", "time", *keys)).rstrip(",") + "\n"
        with open(self.csv, "a", encoding="utf-8") as f:
            f.write(s + ("%.6g," * n % (self.epoch + 1, t, *vals)).rstrip(",") + "\n")

    def plot_metrics(self):
        """Plot metrics from a CSV file."""
        plot_results(file=self.csv, on_plot=self.on_plot)  # save results.png

    def on_plot(self, name, data=None):
        """Register plots (e.g. to be consumed in callbacks)."""
        path = Path(name)
        self.plots[path] = {"data": data, "timestamp": time.time()}

    def final_eval(self):
        """Perform final evaluation and validation for the YOLO model."""
        with torch_distributed_zero_first(LOCAL_RANK):  # strip only on GPU 0; other GPUs should wait
            if RANK in {-1, 0}:
                ckpt = strip_optimizer(self.last) if self.last.exists() else {}
                if self.best.exists():
                    # update best.pt train_metrics from last.pt
                    strip_optimizer(self.best, updates={"train_results": ckpt.get("train_results")})
        candidates, rejected = self._select_final_eval_checkpoints()
        if not candidates:
            raise RuntimeError(
                "No healthy checkpoint is available for final evaluation: " + ("; ".join(rejected) or "none exist")
            )
        self.validator.args.plots = self.args.plots
        self.validator.args.compile = False
        self.validator.args.half = False
        router_failures = []
        from ultralytics.utils.errors import MoERouterError

        for model in candidates:
            self._reset_non_checkpoint_moe_runtime_state()
            LOGGER.info(f"\nValidating {model}...")
            try:
                self.metrics = self.validator(model=model)
                self.metrics.pop("fitness", None)
                self.run_callbacks("on_fit_epoch_end")
                return
            except MoERouterError as exc:
                router_failures.append(f"{model.name}: {exc}")
        raise RuntimeError(
            "No healthy checkpoint is available for final evaluation: " + "; ".join((*rejected, *router_failures))
        )

    def check_resume(self, overrides):
        """Check if resume checkpoint exists and update arguments accordingly."""
        resume = self.args.resume
        if resume:
            try:
                exists = isinstance(resume, (str, Path)) and Path(resume).exists()
                last = Path(check_file(resume) if exists else get_latest_run())
                ckpt_args = load_checkpoint(last)[0].args
                if not isinstance(ckpt_args["data"], dict) and not Path(ckpt_args["data"]).exists():
                    ckpt_args["data"] = self.args.data

                resume = True
                self.args = get_cfg(ckpt_args)
                self.args.model = self.args.resume = str(last)  # reinstate model
                for k in (
                    "imgsz",
                    "batch",
                    "device",
                    "close_mosaic",
                    "augmentations",
                    "save_period",
                    "workers",
                    "cache",
                    "patience",
                    "time",
                    "freeze",
                    "val",
                    "plots",
                    "distill_model",
                    "save_dir",
                ):  # allow arg updates to reduce memory or update device on resume
                    if k in overrides:
                        setattr(self.args, k, overrides[k])

                # Handle augmentations parameter for resume: check if user provided custom augmentations
                if ckpt_args.get("augmentations") is not None:
                    # Augmentations were saved in checkpoint as reprs but can't be restored automatically
                    LOGGER.warning(
                        "Custom Albumentations transforms were used in the original training run but are not "
                        "being restored. To preserve custom augmentations when resuming, you need to pass the "
                        "'augmentations' parameter again to get expected results. Example: \n"
                        f"model.train(resume=True, augmentations={ckpt_args['augmentations']})"
                    )

            except Exception as e:
                raise FileNotFoundError(
                    "Resume checkpoint not found. Please pass a valid checkpoint to resume from, "
                    "i.e. 'yolo train resume model=path/to/last.pt'"
                ) from e
        self.resume = resume

    def _load_checkpoint_state(self, ckpt):
        """Load optimizer, scaler, EMA, and best_fitness from checkpoint."""
        if ckpt.get("optimizer") is not None:
            checkpoint_family = _optimizer_state_family(ckpt["optimizer"])
            runtime_family = _optimizer_family(self.optimizer)
            if checkpoint_family and runtime_family and checkpoint_family != runtime_family:
                adapter_active = _adapter_active(self.model, getattr(self, "adapter_controller", None))
                message = (
                    f"Resume optimizer state uses {checkpoint_family}, but the current policy selected "
                    f"{runtime_family}."
                )
                if not adapter_active:
                    raise ValueError(f"{message} Refusing incompatible full-SFT optimizer state.")
                LOGGER.warning(f"{message} Keeping the freshly initialized adapter optimizer.")
            else:
                try:
                    self.optimizer.load_state_dict(ckpt["optimizer"])
                except (KeyError, RuntimeError, ValueError):
                    adapter_active = _adapter_active(self.model, getattr(self, "adapter_controller", None))
                    if not adapter_active:
                        raise
                    LOGGER.warning("[PEFT] Resume optimizer state is incompatible; using the initialized optimizer.")
        if ckpt.get("scaler") is not None:
            self.scaler.load_state_dict(ckpt["scaler"])
        if self.ema and ckpt.get("ema"):
            from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer

            online_target = unwrap_model(self.model)
            online_mixture_ema = initialize_mixture_loss_ema_buffer(online_target)
            ema_state = ckpt["ema"].float().state_dict()
            checkpoint_mixture_ema = ema_state.get("_mixture_loss_ema_buf")
            if checkpoint_mixture_ema is not None:
                online_mixture_ema.copy_(
                    checkpoint_mixture_ema.to(device=online_mixture_ema.device, dtype=online_mixture_ema.dtype)
                )
            self.ema = ModelEMA(self.model)  # validation with EMA creates inference tensors that can't be updated
            ema_target = unwrap_model(self.ema.ema)
            if getattr(ema_target, "lora_enabled", False):
                from ultralytics.utils.lora import load_lora_compatible_state_dict

                load_lora_compatible_state_dict(
                    ema_target, ema_state, context="resume checkpoint EMA", adapter_only=True
                )
            else:
                ema_target.load_state_dict(ema_state, strict=False)
            self.ema.updates = ckpt["updates"]
        self.best_fitness = ckpt.get("best_fitness")

    def _restore_lora_resume_model(self, ckpt):
        """Restore adapter-only EMA weights into the online model before optimizer state loading."""
        if not ckpt or not _adapter_active(self.model, getattr(self, "adapter_controller", None)):
            return
        source_model = ckpt.get("ema") or ckpt.get("model")
        if not isinstance(source_model, nn.Module):
            raise RuntimeError("Resume checkpoint has active adapters but no EMA/model state to restore.")
        target = unwrap_model(self.model)
        source_state = source_model.float().state_dict()
        if any(bool(getattr(module, "molora_enabled", False)) for module in target.modules()):
            from ultralytics.nn.peft.molora.model import _is_molora_state_key

            target_state = target.state_dict()
            target_keys = {key for key in target_state if _is_molora_state_key(key)}
            source_keys = {key for key in source_state if _is_molora_state_key(key)}
            missing = sorted(target_keys - source_keys)
            unexpected = sorted(source_keys - target_keys)
            shape_mismatch = sorted(
                key
                for key in target_keys & source_keys
                if hasattr(target_state[key], "shape")
                and hasattr(source_state[key], "shape")
                and tuple(target_state[key].shape) != tuple(source_state[key].shape)
            )
            if missing or unexpected or shape_mismatch:
                raise RuntimeError(
                    "Resume checkpoint is incompatible with the current MoLoRA adapter topology: "
                    f"missing={missing[:5]} ({len(missing)} total), "
                    f"unexpected={unexpected[:5]} ({len(unexpected)} total), "
                    f"shape_mismatch={shape_mismatch[:5]} ({len(shape_mismatch)} total)."
                )
            adapter_state = {key: source_state[key] for key in target_keys}
            target.load_state_dict(adapter_state, strict=False)
            LOGGER.info(f"[MoLoRA] Restored {len(adapter_state)} adapter tensors from resume checkpoint EMA.")
            return

        from ultralytics.utils.lora import load_lora_compatible_state_dict

        load_lora_compatible_state_dict(
            target,
            source_state,
            context="resume checkpoint EMA",
            adapter_only=True,
        )

    def _handle_nan_recovery(self, epoch):
        """Recover globally confirmed non-finite state from a healthy online checkpoint."""
        return self._recovery_controller().recover(epoch)

    def resume_training(self, ckpt):
        """Resume YOLO training from a given checkpoint."""
        if ckpt is None or not self.resume:
            return
        start_epoch = ckpt.get("epoch", -1) + 1
        assert 0 < start_epoch < self.epochs, (
            f"{self.args.model} training to {self.epochs} epochs is finished, nothing to resume.\n"
            f"Start a new training without resuming, i.e. 'yolo train model={self.args.model}'"
        )
        LOGGER.info(f"Resuming training {self.args.model} from epoch {start_epoch + 1} to {self.epochs} total epochs")
        if self.epochs < start_epoch:
            LOGGER.info(
                f"{self.model} has been trained for {ckpt['epoch']} epochs. Fine-tuning for {self.epochs} more epochs."
            )
            self.epochs += ckpt["epoch"]  # finetune additional epochs
        self._restore_lora_resume_model(ckpt)
        self._load_checkpoint_state(ckpt)
        if getattr(unwrap_model(self.model), "end2end", False):
            # initialize loss and resume o2o and o2m args
            unwrap_model(self.model).criterion = unwrap_model(self.model).init_criterion()
            unwrap_model(self.model).criterion.updates = start_epoch - 1
            unwrap_model(self.model).criterion.update()
        self.start_epoch = start_epoch
        if start_epoch > (self.epochs - self.args.close_mosaic):
            self._close_dataloader_mosaic()

    def _close_dataloader_mosaic(self):
        """Update dataloaders to stop using mosaic augmentation."""
        if hasattr(self.train_loader.dataset, "mosaic"):
            self.train_loader.dataset.mosaic = False
        if hasattr(self.train_loader.dataset, "close_mosaic"):
            LOGGER.info("Closing dataloader mosaic")
            self.train_loader.dataset.close_mosaic(hyp=copy(self.args))

    def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
        """Construct an optimizer for the given model.

        Args:
            model (torch.nn.Module): The model for which to build an optimizer.
            name (str, optional): The name of the optimizer to use. If 'auto', the optimizer is selected based on the
                number of iterations.
            lr (float, optional): The learning rate for the optimizer.
            momentum (float, optional): The momentum factor for the optimizer.
            decay (float, optional): The weight decay for the optimizer.
            iterations (float, optional): The number of iterations, which determines the optimizer if name is 'auto'.

        Returns:
            (torch.optim.Optimizer): The constructed optimizer.
        """
        g = [{}, {}, {}, {}, {}, {}]  # decay, normalization, bias, Muon, router, adapter groups
        bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
        from ultralytics.utils.lora.api import _is_adapter_param

        router_lr_scale = float(getattr(self.args, "moe_router_lr_scale", 0.5) or 0.5)
        adapter_lr_mult = float(getattr(self.args, "lora_lr_mult", 1.0) or 1.0)
        adapter_model = unwrap_model(model)
        adapter_controller = getattr(self, "adapter_controller", None)
        adapter_active = bool(getattr(adapter_controller, "active", False)) or bool(
            getattr(adapter_model, "lora_enabled", False) or getattr(adapter_model, "molora_enabled", False)
        )
        if not adapter_active:
            adapter_active = any(
                parameter.requires_grad and _is_adapter_param(name)
                for name, parameter in adapter_model.named_parameters()
            )
        if name == "auto":
            LOGGER.info(
                f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                f"determining best 'optimizer', 'lr0' and 'momentum' automatically... "
            )
            nc = self.data.get("nc", 10)  # number of classes
            lr_fit = round(0.002 * 5 / (4 + nc), 6)  # lr0 fit equation to 6 decimal places
            if adapter_active:
                # Planned iteration count must not switch PEFT runs from AdamW to MuSGD. DoRA/LoRA update a small,
                # highly scaled parameter subspace and MuSGD's 1e-2 LR can suppress detection confidences even while
                # the training losses remain finite. Keep auto PEFT policy stable when users change only `epochs`.
                name, lr, momentum = "AdamW", lr_fit, 0.9
                LOGGER.info(
                    f"{colorstr('optimizer:')} active adapters detected, selecting AdamW(lr={lr:g}) "
                    f"instead of iteration-based MuSGD for PEFT stability."
                )
            else:
                name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
            self.args.warmup_bias_lr = 0.0  # no higher than 0.01 for Adam

        use_muon = name == "MuSGD"
        for module_name, module in unwrap_model(model).named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                fullname = f"{module_name}.{param_name}" if module_name else param_name
                if _is_adapter_param(fullname):
                    g[5][fullname] = param
                elif "routing" in fullname.lower() or "router" in fullname.lower():
                    g[4][fullname] = param
                elif param.ndim >= 2 and use_muon:
                    g[3][fullname] = param  # muon params
                elif "bias" in fullname:  # bias (no decay)
                    g[2][fullname] = param
                elif isinstance(module, bn) or "logit_scale" in fullname:  # weight (no decay)
                    # ContrastiveHead and BNContrastiveHead included here with 'logit_scale'
                    g[1][fullname] = param
                else:  # weight (with decay)
                    g[0][fullname] = param
        num_params = [len(g[0]), len(g[1]), len(g[2]), len(g[4]), len(g[5])]  # parameters by policy
        if use_muon:
            router_index, adapter_index = 4, 5
        else:
            g = [g[0].values(), g[1].values(), g[2].values(), g[4].values(), g[5].values()]
            router_index, adapter_index = 3, 4

        optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
        name = {x.lower(): x for x in optimizers}.get(str(name).lower(), str(name))
        if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
            optim_args = dict(lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
        elif name == "RMSProp":
            optim_args = dict(lr=lr, momentum=momentum)
        elif name == "SGD" or name == "MuSGD":
            optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
        else:
            raise NotImplementedError(
                f"Optimizer '{name}' not found in list of available optimizers {optimizers}. "
                "Request support for additional optimizers at https://github.com/ultralytics/ultralytics."
            )

        g[2] = {"params": g[2], **optim_args, "param_group": "bias"}
        g[0] = {"params": g[0], **optim_args, "weight_decay": decay, "param_group": "weight"}
        g[1] = {"params": g[1], **optim_args, "weight_decay": 0.0, "param_group": "bn"}
        g[router_index] = {
            "params": g[router_index],
            **optim_args,
            "lr": lr * router_lr_scale,
            "weight_decay": decay,
            "param_group": "router",
        }
        g[adapter_index] = {
            "params": g[adapter_index],
            **optim_args,
            "lr": lr * adapter_lr_mult,
            "weight_decay": 0.0,
            "param_group": "adapter",
        }
        muon, sgd = (0.2, 1.0)
        if use_muon:
            num_params[0] = len(g[3])  # update number of params
            g[3] = {"params": g[3], **optim_args, "weight_decay": decay, "use_muon": True, "param_group": "muon"}
            import re

            # higher lr for certain parameters in MuSGD when finetuning
            # proto.semseg is the checkpoint parameter name for YOLO26 semantic auxiliary heads.
            pattern = re.compile(r"(?=.*23)(?=.*cv3)|proto\.semseg|SemanticSegment")
            g_ = []  # new param groups
            for x in g:
                p = x.pop("params")
                p1 = [v for k, v in p.items() if pattern.search(k)]
                p2 = [v for k, v in p.items() if not pattern.search(k)]
                g_.extend([{"params": p1, **x, "lr": x.get("lr", lr) * 3}, {"params": p2, **x}])
            g = g_
        optimizer = getattr(optim, name, partial(MuSGD, muon=muon, sgd=sgd))(params=g)

        LOGGER.info(
            f"{colorstr('optimizer:')} {type(optimizer).__name__}(lr={lr}, momentum={momentum}) with parameter groups "
            f"{num_params[1]} weight(decay=0.0), {num_params[0]} weight(decay={decay}), "
            f"{num_params[2]} bias(decay=0.0), {num_params[3]} router(lr={router_lr_scale:g}x), "
            f"{num_params[4]} adapter(lr={adapter_lr_mult:g}x)"
        )
        return optimizer


class MultiTrainer:
    """Fine-tune a single base model across a collection of datasets and aggregate per-dataset results.

    Used automatically by Model.train() when `data` is a list or tuple, allowing one base model to be benchmarked across
    many datasets (such as the RF100 collection) in a single call. The datasets are fine-tuned in series and the same
    base weights seed each run, so every run starts from an identical model. All output is grouped under one sweep
    directory (e.g. runs/detect/multitrain): each dataset gets its own run subdirectory, and the per-dataset and mean
    metrics are written to multitrain_results.json (for post-processing) alongside a multitrain_results.png bar
    chart. The base model object is left unchanged; each dataset's fine-tuned weights live in its own run directory.

    Attributes:
        trainer (type[BaseTrainer] | None): Task trainer class for Python runs, or None for CLI subprocess runs.
        args (dict): Training arguments shared across datasets; its `data` key holds the dataset collection.
        model (torch.nn.Module): Base model whose weights seed each per-dataset fine-tune.
        callbacks (dict | None): Callbacks forwarded to each per-dataset trainer.
        trainers (list[SimpleNamespace]): Completed per-dataset run records.
        metrics (dict): Mapping of each run name (e.g. coco8, coco8-2) to its training-metrics dict from the checkpoint.
        save_dir (Path | None): Sweep directory holding the per-dataset runs and the results JSON/plot.

    Examples:
        Fine-tune one base model across several datasets and read back per-run metrics:
        >>> from ultralytics import YOLO
        >>> model = YOLO("yolo26n.pt")
        >>> results = model.train(data=["coco8.yaml", "african-wildlife.yaml"], epochs=10)
        >>> results["coco8"]["fitness"]  # final fitness on the coco8 run
    """

    def __init__(self, trainer, args, model, _callbacks: dict | None = None):
        """Initialize MultiTrainer with a task trainer class, shared training arguments, and the base model.

        Args:
            trainer (type[BaseTrainer] | None): Task trainer class to run once per dataset. None uses CLI subprocesses.
            args (dict): Training arguments; the `data` key holds the list/tuple of datasets to fine-tune on.
            model (torch.nn.Module): Base model whose weights seed each per-dataset fine-tune.
            _callbacks (dict, optional): Callback functions forwarded to each per-dataset trainer.
        """
        self.trainer = trainer
        self.args = args
        self.model = model
        self.callbacks = _callbacks
        self.trainers = []
        self.metrics = {}
        self.save_dir = None

    def train(self):
        """Fine-tune the base model on each dataset in series and return a {dataset: metrics} mapping."""
        from types import SimpleNamespace

        from ultralytics.utils.patches import torch_load, torch_save

        datasets = self.args["data"]
        # Group every per-dataset run and the summary plot under one sweep directory, e.g. runs/detect/multitrain
        sweep = SimpleNamespace(
            project=self.args.get("project"),
            task=self.args.get("task"),
            mode="train",
            exist_ok=self.args.get("exist_ok", False),
        )
        self.save_dir = get_save_dir(sweep, name="multitrain")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        base_model = self.save_dir / "multitrain_base.pt" if self.trainer is None else None
        if base_model:
            torch_save(
                {"model": deepcopy(self.model).half(), "train_args": getattr(self.model, "args", {})}, base_model
            )
        try:
            for i, data in enumerate(datasets):
                LOGGER.info(
                    f"\n{colorstr('blue', 'bold', f'MultiTrainer {i + 1}/{len(datasets)}:')} fine-tuning on {data}"
                )
                name = Path(str(data)).stem
                run_name = name
                try:
                    overrides = {
                        **self.args,
                        "data": data,
                        "project": str(self.save_dir),  # nest per-dataset runs inside the sweep directory
                        "name": name,
                        "resume": False,
                        "session": None,
                    }
                    run = SimpleNamespace(
                        project=overrides["project"],
                        name=overrides["name"],
                        task=overrides.get("task"),
                        mode="train",
                        exist_ok=overrides.get("exist_ok", False),
                        save_dir=None,
                    )
                    save_dir = get_save_dir(run)
                    save_dir.mkdir(parents=True, exist_ok=True)
                    run_name = save_dir.name
                    overrides["save_dir"] = str(save_dir)
                    if self.trainer is None:
                        overrides["model"] = str(base_model)
                        overrides["pretrained"] = True
                        subprocess.run(
                            [
                                *_YOLO_CLI_COMMAND,
                                "train",
                                *(f"{k}={v}" for k, v in overrides.items() if k != "session"),
                            ],
                            check=True,
                        )
                    else:
                        trainer = self.trainer(overrides=overrides, _callbacks=self.callbacks)
                        trainer.model = trainer.get_model(weights=self.model, cfg=self.model.yaml)
                        trainer.train()
                    best, last = save_dir / "weights" / "best.pt", save_dir / "weights" / "last.pt"
                    ckpt = best if best.exists() else last
                    metrics = None
                    if self.trainer is not None:
                        metrics = getattr(getattr(trainer, "validator", None), "metrics", None)
                        if metrics is not None:
                            metrics = metrics.results_dict
                    self.metrics[run_name] = metrics or (torch_load(ckpt)["train_metrics"] if ckpt.exists() else None)
                    self.trainers.append(SimpleNamespace(save_dir=save_dir, best=best, last=last))
                except Exception as e:  # one bad dataset should not abort the whole sweep
                    LOGGER.error(f"MultiTrainer: fine-tuning on {data} failed, skipping: {e}")
                    self.metrics[run_name] = None
        finally:
            if base_model:
                base_model.unlink(missing_ok=True)
        if RANK in {-1, 0} and self.trainers:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            self.save_results()  # JSON of per-dataset + mean metrics for programmatic post-processing
            if self.args.get("plots", True):
                self.plot_results()
        return self.metrics

    def save_results(self):
        """Write per-dataset and mean metrics to multitrain_results.json for programmatic post-processing."""
        import json

        results = {run: ({k: float(v) for k, v in m.items()} if m else None) for run, m in self.metrics.items()}
        valid = [m for m in results.values() if m]
        keys = {k for m in valid for k in m}
        mean = {k: sum(m[k] for m in valid if k in m) / sum(k in m for m in valid) for k in keys}
        file = self.save_dir / "multitrain_results.json"
        with open(file, "w", encoding="utf-8") as f:
            json.dump({"results": results, "mean": mean}, f, indent=2)
        LOGGER.info(f"MultiTrainer results saved to {colorstr('bold', file)}")
        return file

    def plot_results(self):
        """Save a cross-dataset bar chart of the per-dataset metric with the mean across all datasets."""
        from ultralytics.cfg import TASK2METRIC
        from ultralytics.utils.plotting import plot_multitrain_results

        key = TASK2METRIC.get(self.args.get("task"))
        scores = {run: float(m.get(key, m.get("fitness", 0.0))) for run, m in self.metrics.items() if m}
        if not scores:
            return None
        fname = plot_multitrain_results(scores, key=key or "fitness", save_dir=self.save_dir)
        LOGGER.info(f"MultiTrainer results saved to {colorstr('bold', fname)}")
        return fname
