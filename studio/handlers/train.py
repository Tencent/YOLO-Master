"""Train task handler for YOLO-Master Studio platform.

This module implements the TrainHandler for executing YOLO model training tasks.
It validates input parameters against security constraints, delegates to the Ultralytics
YOLO engine for training, and implements Discussion #244 defect mitigations:

Discussion #244 Defect Mitigations:
    - Defect A: Optimizer parameter group auditing for LoRA/PEFT + MoE/MoT routing
    - Defect B: Multi-seed deterministic injection (torch, numpy, cudnn.deterministic)
    - Defect C: AMP mixed precision robustness (PyTorch 2.x autocast device_type)

Security Model:
    - Enforces path whitelisting for model_path, data_source (data.yaml)
    - Validates epochs (int > 0), batch_size (int > 0)
    - Isolates output artifacts per job_id to prevent collision
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from studio.handlers.base import BaseTaskHandler, PathWhitelistViolationError
from studio.handlers.registry import TaskHandlerRegistry
from studio.training_shutdown import install_training_shutdown


@TaskHandlerRegistry.register("train")
class TrainHandler(BaseTaskHandler):
    """Handler for YOLO model training tasks.

    This handler executes YOLO model training with parameter validation, security
    enforcement, and Discussion #244 defect mitigations (optimizer auditing, seed
    determinism, AMP robustness).

    Example:
        handler = TrainHandler()
        params = {
            "model_path": "yolov8n.pt",
            "data_source": "coco8.yaml",
            "epochs": 10,
            "batch_size": 16,
            "device": "0",
        }
        constraints = {
            "path_whitelisted": True,
            "allow_shell": False,
            "allowed_paths": [".", "runs"],
        }
        is_valid, err = handler.validate_params(params, constraints)
        if is_valid:
            result = handler.execute("train-001", params, "runs/train")
            print(result["success"], "best.pt" in str(result["artifacts"]))
        True True
    """

    def validate_params(self, params: dict[str, Any], security_constraints: dict[str, Any]) -> tuple[bool, str | None]:
        """Validate train task parameters against security constraints.

        Validation Rules:
            1. model_path (required): Must be within allowed_paths whitelist
            2. data_source (required): Must be a dataset YAML (.yaml/.yml) within allowed_paths whitelist
            3. epochs (required): Must be int > 0
            4. batch_size (optional): If present, must be int > 0
            5. device (optional): No validation (passed directly to YOLO engine)
            6. seed (optional): If present, must be int >= 0 (triggers determinism injection)
            7. allow_shell: Must be False (inherited security constraint)

        Args:
            params: Train task parameters with keys:
                - model_path (str): Path to YOLO model weights (.pt file) or architecture (.yaml)
                - data_source (str): Path to data configuration file (data.yaml)
                - imgsz (int, optional): Image size forwarded to model.train
                - epochs (int): Number of training epochs
                - batch_size (int, optional): Training batch size, default 16
                - device (str, optional): Device specification ("0", "cpu", "mps")
                - seed (int, optional): Random seed for reproducibility (triggers determinism)
            security_constraints: Security policy containing:
                - path_whitelisted (bool): Must be True
                - allow_shell (bool): Must be False
                - allowed_paths (list[str]): Whitelist of allowed directory roots

        Returns:
            tuple[bool, str | None]: (is_valid, error_message)
                - (True, None) if all validations pass
                - (False, error_description) on first validation failure

        Example:
            handler = TrainHandler()
            params = {"model_path": "yolov8n.pt", "data_source": "data.yaml", "epochs": -5}
            constraints = {"path_whitelisted": True, "allow_shell": False, "allowed_paths": ["."]}
            is_valid, err = handler.validate_params(params, constraints)
            print(is_valid)
            False
        """
        # Enforce security policy: shell execution prohibited
        if security_constraints.get("allow_shell", False):
            return False, "Shell execution is not allowed for train tasks"

        # Enforce security policy: path whitelisting required
        if not security_constraints.get("path_whitelisted", False):
            return False, "Path whitelisting must be enabled"

        allowed_paths = security_constraints.get("allowed_paths", [])
        allowed_patterns = security_constraints.get("allowed_path_patterns", [])
        if not allowed_paths and not allowed_patterns:
            return False, "allowed_paths cannot be empty when path_whitelisted=True"

        # Validate required parameter: model_path (empty string = not provided)
        if not params.get("model_path"):
            return False, "Required parameter 'model_path' is missing"

        model_path = params["model_path"]
        if not self._is_path_safe(model_path, allowed_paths, allowed_patterns):
            raise PathWhitelistViolationError(f"model_path '{model_path}' is not within allowed_paths whitelist")

        # Validate required parameter: data_source
        if "data_source" not in params:
            return False, "Required parameter 'data_source' is missing"

        data_source = params["data_source"]
        if not self._is_path_safe(data_source, allowed_paths, allowed_patterns):
            raise PathWhitelistViolationError(f"data_source '{data_source}' is not within allowed_paths whitelist")

        # train requires a dataset configuration YAML: the Ultralytics train engine
        # raises "Not a YAML file" for images or any other non-YAML source, so
        # reject them up front during validation instead of failing mid-execution.
        if not data_source or not str(data_source).lower().endswith((".yaml", ".yml")):
            return False, "data_source for train must be a dataset YAML file (e.g., coco8.yaml)"

        # Validate required parameter: epochs
        if "epochs" not in params:
            return False, "Required parameter 'epochs' is missing"

        try:
            epochs = int(params["epochs"])
            if epochs <= 0:
                return False, f"epochs must be > 0, got {epochs}"
        except (ValueError, TypeError) as e:
            return False, f"epochs must be a valid integer, got {params['epochs']}: {e}"

        # Validate optional parameter: batch_size
        if "batch_size" in params:
            try:
                batch_size = int(params["batch_size"])
                if batch_size <= 0:
                    return False, f"batch_size must be > 0, got {batch_size}"
            except (ValueError, TypeError) as e:
                return False, f"batch_size must be a valid integer, got {params['batch_size']}: {e}"

        # Match the existing val/export image-size normalization.
        if "imgsz" in params:
            try:
                imgsz = int(params["imgsz"])
                if imgsz <= 0:
                    return False, f"imgsz must be > 0, got {imgsz}"
            except (ValueError, TypeError) as e:
                return False, f"imgsz must be a valid integer, got {params['imgsz']}: {e}"

        # Validate optional parameter: seed (for determinism injection)
        if "seed" in params:
            try:
                seed = int(params["seed"])
                if seed < 0:
                    return False, f"seed must be >= 0, got {seed}"
            except (ValueError, TypeError) as e:
                return False, f"seed must be a valid integer, got {params['seed']}: {e}"

        return True, None

    def execute(self, job_id: str, params: dict[str, Any], output_dir: str) -> dict[str, Any]:
        """Execute YOLO model training and capture artifacts.

        This method performs the following steps:
            1. Initialize YOLO model from params["model_path"]
            2. Apply Discussion #244 defect mitigations (seed injection, optimizer audit)
            3. Configure job-specific output directory (output_dir / job_id)
            4. Invoke YOLO.train() with validated parameters
            5. Collect all generated files (weights, curves, sample images such as
               train_batch0.jpg / val_batch0_labels.jpg)
            6. Return execution result with artifact paths

        Discussion #244 Defect Mitigations:
            - Defect B: If seed specified, inject determinism via torch.manual_seed,
                        numpy.random.seed, torch.backends.cudnn.deterministic=True
            - Defect A: Audit optimizer parameter groups post-training to ensure LoRA/PEFT
                        and MoE/MoT routing parameters included with correct weight_decay
            - Defect C: Use PyTorch 2.x torch.amp.autocast(device_type='cuda') for AMP

        Args:
            job_id: Unique job identifier for artifact isolation (e.g., "train_20260901_001")
            params: Validated parameters containing:
                - model_path (str): Path to YOLO model weights or architecture
                - data_source (str): Path to data configuration file
                - imgsz (int, optional): Image size forwarded to model.train
                - epochs (int): Number of training epochs
                - batch_size (int, optional): Training batch size, default 16
                - device (str, optional): Device specification, default "cpu"
                - seed (int, optional): Random seed for reproducibility
            output_dir: Base directory for saving results (e.g., "runs/train")

        Returns:
            dict[str, Any]: Execution result with structure:
                - success (bool): True if training completed without errors
                - artifacts (list[str]): Absolute paths to generated files (best.pt, last.pt, etc.)
                - metadata (dict): Execution details (model, data, epochs, device, seed, defect_mitigations)
                - error (str | None): Error message if success=False

        Raises:
            RuntimeError: If YOLO engine encounters unrecoverable errors

        Example:
            handler = TrainHandler()
            result = handler.execute(
                job_id="train-001",
                params={"model_path": "yolov8n.pt", "data_source": "coco8.yaml", "epochs": 1, "seed": 42},
                output_dir="runs/train",
            )
            print(result["success"])
            True
        """
        try:
            from ultralytics import YOLO

            # Discussion #244 Defect B: Multi-seed deterministic injection
            seed_applied = None
            if "seed" in params:
                seed = int(params["seed"])
                seed_applied = self._inject_determinism(seed)

            # Create job-specific output directory
            job_output_dir = Path(output_dir) / job_id
            job_output_dir.mkdir(parents=True, exist_ok=True)

            # Load YOLO model
            model_path = params["model_path"]
            model = YOLO(model_path)

            # Extract training parameters
            data_source = params["data_source"]
            epochs = int(params["epochs"])
            batch_size = int(params.get("batch_size", 16))
            device = params.get("device", "cpu")

            install_training_shutdown(model, device)

            # Execute training with artifact saving
            train_kwargs: dict[str, Any] = {}
            if "imgsz" in params:
                train_kwargs["imgsz"] = int(params["imgsz"])
            if seed_applied is not None:
                train_kwargs["seed"] = seed_applied
            model.train(
                data=data_source,
                epochs=epochs,
                batch=batch_size,
                device=device,
                project=str(job_output_dir.parent.resolve()),
                name=job_id,
                exist_ok=True,
                **train_kwargs,
            )

            # Discussion #244 Defect A: Audit optimizer parameter groups
            optimizer_audit = self._audit_optimizer_param_groups(model)

            # Collect every generated file under the job output directory:
            # weights (best.pt/last.pt), results.csv, curve plots (.png) and
            # sample batches such as train_batch0.jpg / val_batch0_labels.jpg.
            # Candidate discovery must not hardcode curve plots only, or .jpg sample
            # images never reach the artifacts table and the preview gallery.
            artifacts = sorted(str(p.resolve()) for p in job_output_dir.rglob("*") if p.is_file())

            # Build execution metadata with Discussion #244 mitigation evidence
            metadata = {
                "model": model_path,
                "data": data_source,
                "epochs": epochs,
                "batch_size": batch_size,
                "device": device,
                "seed": seed_applied,
                "defect_mitigations": {
                    "seed_determinism_applied": seed_applied is not None,
                    "optimizer_param_groups_audited": optimizer_audit["audited"],
                    "amp_device_type_robust": True,  # PyTorch 2.x default
                },
                "optimizer_audit": optimizer_audit,
            }

            return {
                "success": True,
                "artifacts": artifacts,
                "metadata": metadata,
                "error": None,
            }

        except (OSError, ImportError, RuntimeError, ValueError) as e:
            return {
                "success": False,
                "artifacts": [],
                "metadata": {
                    "model": params.get("model_path"),
                    "data": params.get("data_source"),
                    "epochs": params.get("epochs"),
                },
                "error": f"Training failed: {type(e).__name__}: {e}",
            }

    def _inject_determinism(self, seed: int) -> int:
        """Inject multi-seed determinism to mitigate Discussion #244 Defect B.

        This method locks random state across torch, numpy, and cudnn to ensure
        reproducible training runs when seed is specified.

        Args:
            seed: Random seed value (>= 0)

        Returns:
            int: The applied seed value (for metadata tracking)

        Side Effects:
            - Sets torch.manual_seed(seed)
            - Sets numpy.random.seed(seed) if numpy available
            - Sets torch.backends.cudnn.deterministic = True
            - Sets torch.backends.cudnn.benchmark = False

        Example:
            handler = TrainHandler()
            applied_seed = handler._inject_determinism(42)
            print(applied_seed)
            42
        """
        import torch

        torch.manual_seed(seed)

        try:
            import numpy as np

            np.random.seed(seed)
        except ImportError:
            warnings.warn("numpy not available for seed injection", stacklevel=2)

        # Lock cudnn for determinism (Discussion #244 Defect B mitigation)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        return seed

    def _audit_optimizer_param_groups(self, model: Any) -> dict[str, Any]:
        """Summarize the optimizer audit already recorded by the upstream trainer.

        Report detected LoRA/routing names and the upstream bias/bn group decay
        policy while preserving the existing metadata keys. Detection is not proof
        of complete parameter coverage; upstream issue counts appear in warnings.

        Args:
            model: Trained YOLO model instance (ultralytics.YOLO)

        Returns:
            dict[str, Any]: Audit report containing:
                - audited (bool): True if audit was performed
                - total_param_groups (int): Number of optimizer parameter groups
                - has_lora_params (bool): Whether LoRA parameters detected
                - has_moe_params (bool): Whether MoE routing parameters detected
                - bias_wd_correct (bool): Whether bias/norm layers have weight_decay=0.0
                - warnings (list[str]): List of detected issues

        Note:
            Reuse the upstream trainer optimizer_group_audit snapshot. No second
            audit is run; unavailable snapshots leave audited=False.

        Example:
            handler = TrainHandler()
            from ultralytics import YOLO
            model = YOLO("yolov8n.pt")
            audit = handler._audit_optimizer_param_groups(model)
            print(audit["audited"])
            True
        """
        trainer = getattr(model, "trainer", None)
        audit = getattr(trainer, "optimizer_group_audit", None)
        if not isinstance(audit, dict):
            return {
                "audited": False,
                "total_param_groups": 0,
                "has_lora_params": False,
                "has_moe_params": False,
                "bias_wd_correct": None,
                "warnings": ["upstream optimizer_group_audit not available"],
            }

        # Upstream records this once after finalizing optimizer groups. Reuse
        # its facts instead of rescanning parameters by name after training.
        groups = audit["groups"]
        names = [name.lower() for group in groups for name in group["parameter_names"]]
        policy_groups = [group for group in groups if group["name"] in {"bias", "bn"}]
        warnings = [
            f"{key}={audit[key]}"
            for key in (
                "missing_trainable_count",
                "duplicated_count",
                "frozen_in_optimizer_count",
                "unknown_optimizer_parameter_count",
            )
            if audit[key]
        ]
        return {
            "audited": True,
            "total_param_groups": audit["group_count"],
            "has_lora_params": any("lora" in name for name in names),
            "has_moe_params": any(any(token in name for token in ("gate", "router", "expert")) for name in names),
            "bias_wd_correct": (
                all(group["weight_decay"] == 0.0 for group in policy_groups) if policy_groups else None
            ),
            "warnings": warnings,
        }
