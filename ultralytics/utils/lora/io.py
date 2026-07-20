# 🐧Please note that this file has been modified by Tencent on 2026/02/13. All Tencent Modifications are Copyright (C) 2026 Tencent.
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Union

import torch

from ultralytics.utils import LOGGER

if TYPE_CHECKING:
    from ultralytics.nn.tasks import DetectionModel


def _lora_pkg():
    """Return the public lora package module so monkeypatches on package attrs remain effective."""
    return sys.modules[__package__]

def save_lora_adapters(model: "DetectionModel", path: Union[str, Path]) -> bool:
    """
    Save active LoRA or MoLoRA adapter weights without the frozen base model.
    
    Args:
        model: LoRADetectionModel instance.
        path: Directory path for saving.
    """
    # Unwrap DDP
    if hasattr(model, 'module'):
        model = model.module

    if getattr(model, "molora_enabled", False) or any(
        module.__class__.__name__ == "MoLoRALayer" for module in model.modules()
    ):
        from .backend import MoLoRABackend

        return MoLoRABackend().save(model, path)

    if not getattr(model, 'lora_enabled', False):
        LOGGER.debug("[LoRA] Save skipped: LoRA not enabled.")
        return False

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    backend = getattr(model, "lora_backend", "peft")
    variant = getattr(model, "lora_variant", "lora")
    
    try:
        if backend == "fallback":
            fallback_state = _lora_pkg()._collect_fallback_adapter_state(model)
            weight_file = "fallback_adapter.pt"
            torch.save(fallback_state, path / weight_file)
            payload = {
                "backend": backend,
                "variant": variant,
                "weight_file": weight_file,
                "freeze_bn": bool(getattr(model, "lora_freeze_bn", False)),
                "include_head": bool(getattr(model, "lora_include_head", False)),
                "target_modules": list(getattr(model, "lora_target_modules", sorted(fallback_state["modules"]))),
                "target_audit": getattr(model, "lora_target_audit", {}),
                "runtime_metadata": getattr(model, "lora_runtime_metadata", {}),
            }
            # FIX: write fallback metadata to a dedicated filename so it does
            # not collide with PEFT's own `adapter_config.json` when both
            # backends save into the same directory (e.g. successive
            # save_lora_adapters calls). Keep an `adapter_config.json` symlink
            # for backward compatibility with older loaders.
            (path / "fallback_meta.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=False)
            )
            # Backward compat: only write adapter_config.json if PEFT didn't
            # already create one in this directory.
            adapter_cfg_path = path / "adapter_config.json"
            if not adapter_cfg_path.exists():
                adapter_cfg_path.write_text(
                    json.dumps(payload, indent=2, ensure_ascii=False)
                )
            LOGGER.info(f"[LoRA] 💾 Fallback adapter metadata saved to {path}")
            return True

        # model.model is PeftProxy (PeftModel)
        # save_pretrained automatically saves only the adapter weights
        model.model.save_pretrained(str(path))
        runtime_payload = {
            "backend": backend,
            "variant": variant,
            "freeze_bn": bool(getattr(model, "lora_freeze_bn", False)),
            "include_head": bool(getattr(model, "lora_include_head", False)),
            "target_modules": list(getattr(model, "lora_target_modules", [])),
            "target_audit": getattr(model, "lora_target_audit", {}),
            "runtime_metadata": getattr(model, "lora_runtime_metadata", {}),
        }
        (path / "runtime_metadata.json").write_text(json.dumps(runtime_payload, indent=2, ensure_ascii=False))
        LOGGER.info(f"[LoRA] 💾 Adapters saved to {path}")
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Failed to save adapters: {e}")
        return False


def load_lora_adapters(
    model: "DetectionModel",
    path: Union[str, Path],
    merge: bool = False,
    force_replace: bool = False,
    trainable: bool = False,
) -> bool:
    """
    Loads LoRA adapter weights onto an existing Ultralytics model.

    Args:
        model: Base Ultralytics model instance.
        path: Directory containing PEFT adapter files.
        merge: Whether to merge loaded adapters into the base model immediately.
        force_replace: If True, replace existing LoRA adapters with new ones (default False).
        trainable: If True, keep PEFT adapter params trainable for continued fine-tuning.
    """
    path = Path(path)
    if not path.exists():
        LOGGER.error(f"[LoRA] Adapter path not found: {path}")
        return False

    # FIX: prefer dedicated fallback metadata file to avoid mis-classifying
    # a PEFT-saved `adapter_config.json` as a fallback config.
    fallback_meta_path = path / "fallback_meta.json"
    config_path = path / "adapter_config.json"
    payload = {}
    if fallback_meta_path.exists():
        try:
            payload = json.loads(fallback_meta_path.read_text())
        except Exception:
            payload = {}
    elif config_path.exists():
        try:
            candidate = json.loads(config_path.read_text())
            # Only treat as fallback metadata if it self-identifies as such.
            if candidate.get("backend") == "fallback":
                payload = candidate
        except Exception:
            payload = {}

    if hasattr(model, "module"):
        model = model.module

    if getattr(model, "lora_enabled", False):
        if force_replace:
            LOGGER.info("[LoRA] Force-replacing existing LoRA adapters with new ones.")
            if hasattr(getattr(model, "model", None), "merge_and_unload"):
                _lora_pkg().merge_lora_weights(model)
            else:
                _lora_pkg()._clear_lora_runtime_state(model)
        else:
            LOGGER.warning("[LoRA] Model already has LoRA enabled. Skipping. Use force_replace=True to override.")
            return True

    if payload.get("backend") == "fallback":
        model = _lora_pkg()._load_fallback_adapter_state(model, path, payload)
        LOGGER.info(f"[LoRA] 📥 Fallback adapter metadata loaded from {path}")
        if merge:
            return _lora_pkg().merge_lora_weights(model)
        return True

    if not _lora_pkg().PEFT_AVAILABLE:
        LOGGER.error("[LoRA] PEFT library not found. Please install via `pip install peft`.")
        return False

    try:
        peft_model_wrapper = _lora_pkg().PeftModel.from_pretrained(
            model.model,
            str(path),
            is_trainable=trainable,
        )
        peft_model_wrapper.__class__ = _lora_pkg().PeftProxy
        model.model = peft_model_wrapper
        _lora_pkg()._wrap_top_level_lora_model(model, getattr(peft_model_wrapper, "peft_config", None))
        runtime_path = path / "runtime_metadata.json"
        runtime_payload = {}
        if runtime_path.exists():
            try:
                runtime_payload = json.loads(runtime_path.read_text())
            except Exception:
                runtime_payload = {}
        model.lora_backend = runtime_payload.get("backend", "peft")
        model.lora_variant = runtime_payload.get("variant", "lora")
        model.lora_include_head = runtime_payload.get("include_head", False)
        model.lora_freeze_bn = runtime_payload.get("freeze_bn", False)
        model.lora_target_modules = runtime_payload.get("target_modules", [])
        model.lora_target_audit = runtime_payload.get("target_audit", {})
        model.lora_runtime_metadata = runtime_payload.get("runtime_metadata", {})
        _lora_pkg()._validate_lora_runtime_model(
            model,
            expected_targets=model.lora_target_modules or None,
            context="PEFT load_lora_adapters",
        )

        LOGGER.info(f"[LoRA] 📥 Adapters loaded from {path}")
        if merge:
            return _lora_pkg().merge_lora_weights(model)
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Failed to load adapters: {e}")
        return False


def _find_original_model_class(model: "DetectionModel"):
    """Find the original model class before LoRA wrapping by inspecting MRO."""
    from ultralytics.nn.tasks import (
        DetectionModel, SegmentationModel, PoseModel,
        ClassificationModel, OBBModel, RTDETRDetectionModel, WorldModel
    )
    
    # Known original classes
    ORIGINAL_CLASSES = {
        DetectionModel, SegmentationModel, PoseModel,
        ClassificationModel, OBBModel, RTDETRDetectionModel, WorldModel
    }
    
    # Check all bases in MRO order
    for cls in model.__class__.__mro__:
        if cls in ORIGINAL_CLASSES:
            return cls
    
    # Fallback to DetectionModel if we can't determine the original class
    return DetectionModel


def merge_lora_weights(model: "DetectionModel") -> bool:
    """
    Merges LoRA weights back into the base model and unloads adapters.
    Useful for inference acceleration or model export.
    """
    if getattr(model, "lora_backend", None) == "fallback":
        try:
            target_root = getattr(model, "model", model)
            merged_count = _lora_pkg()._merge_fallback_modules(target_root)
            if merged_count == 0:
                LOGGER.error("[LoRA] Cannot merge fallback adapters: no ManualLoRAConv modules found.")
                return False

            original_cls = getattr(model, "lora_original_class", None)
            if original_cls is not None:
                model.__class__ = original_cls

            _lora_pkg()._clear_lora_runtime_state(model)

            LOGGER.info(f"[LoRA] ✅ Fallback merge completed. Merged {merged_count} manual LoRA modules.")
            return True
        except Exception as e:
            LOGGER.error(f"[LoRA] Fallback merge failed: {e}")
            return False

    # Check if wrapped in PeftProxy
    if not hasattr(model, 'model') or not hasattr(getattr(model, 'model', None), 'merge_and_unload'):
        LOGGER.error("[LoRA] Cannot merge: Model does not appear to have LoRA adapters attached.")
        return False

    try:
        LOGGER.info("[LoRA] 🔄 Merging adapters into base model...")
        
        # merge_and_unload returns the clean base model (nn.Sequential)
        merged_base = model.model.merge_and_unload()
        
        # Restore structure
        model.model = merged_base
        
        # Restore original class using robust MRO inspection
        original_cls = _lora_pkg()._find_original_model_class(model)
        model.__class__ = original_cls
        
        # Clear flags
        _lora_pkg()._clear_lora_runtime_state(model)
            
        LOGGER.info(f"[LoRA] ✅ Merge completed. Model restored to {original_cls.__name__} architecture.")
        return True
    except Exception as e:
        LOGGER.error(f"[LoRA] Merge failed: {e}")
        return False


__all__ = ['save_lora_adapters', 'load_lora_adapters', 'merge_lora_weights']
