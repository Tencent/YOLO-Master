# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from typing import TYPE_CHECKING

from . import MACOS, USER_CONFIG_DIR, WINDOWS
from .torch_utils import TORCH_1_9

if TYPE_CHECKING:
    from ultralytics.engine.trainer import BaseTrainer


def ddp_launch_env() -> dict[str, str]:
    """Return a torchrun environment compatible with Windows builds that omit libuv."""
    env = os.environ.copy()
    if WINDOWS:
        env["USE_LIBUV"] = "0"
    return env


def find_free_network_port() -> int:
    """Find a free port on localhost.

    It is useful in single-node training when we don't want to connect to a real main node but have to set the
    `MASTER_PORT` environment variable.

    Returns:
        (int): The available network port number.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]  # port


def generate_ddp_file(trainer: BaseTrainer) -> str:
    """Generate a DDP (Distributed Data Parallel) file for multi-GPU training.

    This function creates a temporary Python file that enables distributed training across multiple GPUs. The file
    contains the necessary configuration to initialize the trainer in a distributed environment.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing training configuration and arguments.
            Must have args attribute and be a class instance.

    Returns:
        (str): Path to the generated temporary DDP file.

    Notes:
        The generated file is saved in the USER_CONFIG_DIR/DDP directory and includes:
        - Trainer class import
        - Configuration overrides from the trainer arguments
        - Model path configuration
        - Training initialization code
    """
    module, name = f"{trainer.__class__.__module__}.{trainer.__class__.__name__}".rsplit(".", 1)

    # Serialize augmentations to JSON-safe dicts to avoid NameError in DDP subprocess
    overrides = vars(trainer.args).copy()
    if overrides.get("augmentations") is not None:
        import albumentations as A

        overrides["augmentations"] = [A.to_dict(t) for t in overrides["augmentations"]]

    content = f"""
# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
from pathlib import Path, PosixPath  # For model arguments stored as Path instead of str
from torch.distributed.elastic.multiprocessing.errors import record
overrides = {overrides}

@record
def main():
    from {module} import {name}
    from ultralytics.utils import DEFAULT_CFG_DICT

    # Deserialize augmentations from dicts back to Albumentations transform objects
    if overrides.get("augmentations") is not None:
        import albumentations as A
        overrides["augmentations"] = [A.from_dict(t) for t in overrides["augmentations"]]

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = {name}(cfg=cfg, overrides=overrides)
    trainer.args.model = "{getattr(trainer.hub_session, "model_url", trainer.args.model)}"
    return trainer.train()

if __name__ == "__main__":
    main()
"""
    (USER_CONFIG_DIR / "DDP").mkdir(exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix="_temp_",
        suffix=f"{id(trainer)}.py",
        mode="w+",
        encoding="utf-8",
        dir=USER_CONFIG_DIR / "DDP",
        delete=False,
    ) as file:
        file.write(content)
    return file.name


def collect_ddp_error_logs(log_dir, max_chars: int = 50000) -> str:
    """Collect Elastic error metadata and worker stderr after a failed launch."""
    log_dir = os.fspath(log_dir)
    if not os.path.isdir(log_dir):
        return ""
    chunks = []
    for root, _, files in os.walk(log_dir):
        for name in sorted(files):
            if name not in {"error.json", "stderr.log"}:
                continue
            path = os.path.join(root, name)
            try:
                with open(path, encoding="utf-8", errors="replace") as file:
                    text = file.read()
            except OSError:
                continue
            if text:
                chunks.append(f"--- {path} ---\n{text[-max_chars:]}")
    return "\n".join(chunks)


def generate_ddp_command(trainer: BaseTrainer) -> tuple[list[str], str]:
    """Generate command for distributed training.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer containing configuration for distributed training.

    Returns:
        cmd (list[str]): The command to execute for distributed training.
        file (str): Path to the temporary file created for DDP training.
    """
    import __main__  # noqa local import to avoid https://github.com/Lightning-AI/pytorch-lightning/issues/15218

    if not trainer.resume:
        shutil.rmtree(trainer.save_dir)  # remove the save_dir
    file = generate_ddp_file(trainer)
    dist_cmd = "torch.distributed.run" if TORCH_1_9 else "torch.distributed.launch"
    port = find_free_network_port()
    cmd = [
        sys.executable,
        "-m",
        dist_cmd,
        "--nproc_per_node",
        f"{trainer.world_size}",
        "--master_port",
        f"{port}",
    ]
    log_dir = None
    if TORCH_1_9:
        log_dir = USER_CONFIG_DIR / "DDP" / f"logs_{id(trainer)}"
        log_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--log-dir", str(log_dir)])
        if not MACOS:
            cmd.extend(["--tee", "3"])
    cmd.append(file)
    trainer.ddp_log_dir = log_dir
    return cmd, file


def ddp_cleanup(trainer: BaseTrainer, file: str) -> None:
    """Delete temporary file if created during distributed data parallel (DDP) training.

    This function checks if the provided file contains the trainer's ID in its name, indicating it was created as a
    temporary file for DDP training, and deletes it if so.

    Args:
        trainer (ultralytics.engine.trainer.BaseTrainer): The trainer used for distributed training.
        file (str): Path to the file that might need to be deleted.

    Examples:
        >>> trainer = YOLOTrainer()
        >>> file = "/tmp/ddp_temp_123456789.py"
        >>> ddp_cleanup(trainer, file)
    """
    if f"{id(trainer)}.py" in file:  # if temp_file suffix in file
        os.remove(file)
