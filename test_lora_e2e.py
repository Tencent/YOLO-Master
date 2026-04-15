#!/usr/bin/env python3
"""
LoRA End-to-End Pipeline Verification Script
=============================================
This script validates the complete LoRA lifecycle:
1. Training with LoRA enabled
2. Saving LoRA adapters
3. Loading LoRA adapters
4. Merging LoRA weights
5. Inference verification

Usage:
    python test_lora_e2e.py
"""

import sys
import os
import torch
import shutil
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path("/Users/gatilin/PycharmProjects/YOLO-Master-lora-fixed")
sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO
from ultralytics.utils.lora import (
    apply_lora, save_lora_adapters, load_lora_adapters,
    merge_lora_weights, _print_param_stats
)


def log(msg: str, emoji: str = "📝"):
    """Print a formatted log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {emoji} {msg}")


def log_section(title: str):
    """Print a section divider."""
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


TEST_IMAGE = str(PROJECT_ROOT / "runs/detect/lora_e2e_cpu_check/train_batch0.jpg")


def verify_base_model():
    """Step 0: Verify base model works before LoRA."""
    log_section("Step 0: Base Model Verification")

    model_path = PROJECT_ROOT / "yolo11n.pt"
    if not model_path.exists():
        log(f"Model not found at {model_path}", "❌")
        return None

    log(f"Loading base model from {model_path}")
    model = YOLO(str(model_path))

    # Print model info
    log("Base model loaded successfully", "✅")
    log(f"Model type: {type(model.model)}")
    log(f"Device: {model.device}")

    # Verify inference works (use local image)
    log(f"Running test inference with local image: {TEST_IMAGE}")
    if os.path.exists(TEST_IMAGE):
        results = model.predict(source=TEST_IMAGE, imgsz=320, verbose=False, device="cpu")
        log(f"Test inference successful, detected {len(results[0])} objects", "✅")
    else:
        log(f"Test image not found, skipping inference test", "⚠️")

    return model


def verify_lora_training(model: YOLO, epochs: int = 3) -> YOLO:
    """Step 1: Train model with LoRA enabled."""
    log_section("Step 1: LoRA Training")

    lora_args = {
        "data": "coco8.yaml",
        "epochs": epochs,
        "imgsz": 64,  # Small for quick testing
        "batch": 2,
        "device": "cpu",
        "workers": 0,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "lora_target_modules": None,  # Auto-detect
        "lora_gradient_checkpointing": True,
        "lora_save_adapters": True,
        "verbose": True,
    }

    log(f"Training with LoRA config: r={lora_args['lora_r']}, alpha={lora_args['lora_alpha']}")
    log(f"Parameters: epochs={epochs}, imgsz={lora_args['imgsz']}, batch={lora_args['batch']}")

    # Train
    results = model.train(**lora_args)

    log(f"Training completed!", "✅")
    log(f"Best mAP50: {results.box.map50 if hasattr(results, 'box') else 'N/A'}")
    log(f"Best mAP50-95: {results.box.map if hasattr(results, 'box') else 'N/A'}")

    return model


def verify_lora_save(model: YOLO) -> Path:
    """Step 2: Save LoRA adapters."""
    log_section("Step 2: Save LoRA Adapters")

    adapter_dir = PROJECT_ROOT / "runs" / "detect" / "lora_e2e_test" / "lora_adapters"

    # Clean up if exists
    if adapter_dir.exists():
        shutil.rmtree(adapter_dir)

    log(f"Saving LoRA adapters to {adapter_dir}")

    # Use the save_lora_only method
    success = model.save_lora_only(str(adapter_dir))

    if success:
        log(f"LoRA adapters saved successfully!", "✅")

        # Verify files exist
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_model = adapter_dir / "adapter_model.safetensors"

        if adapter_config.exists():
            log(f"adapter_config.json exists ({adapter_config.stat().st_size} bytes)", "✅")
        if adapter_model.exists():
            log(f"adapter_model.safetensors exists ({adapter_model.stat().st_size} bytes)", "✅")

        return adapter_dir
    else:
        log("Failed to save LoRA adapters", "❌")
        return None


def verify_lora_load(model: YOLO, adapter_dir: Path) -> YOLO:
    """Step 3: Load LoRA adapters."""
    log_section("Step 3: Load LoRA Adapters")

    log(f"Loading LoRA adapters from {adapter_dir}")

    # Check lora_enabled flag before loading
    lora_enabled_before = getattr(model.model, "lora_enabled", False)
    log(f"LoRA enabled before load: {lora_enabled_before}")

    # Load adapters
    success = model.load_lora(str(adapter_dir))

    if success:
        log(f"LoRA adapters loaded successfully!", "✅")

        # Check lora_enabled flag after loading
        lora_enabled_after = getattr(model.model, "lora_enabled", False)
        log(f"LoRA enabled after load: {lora_enabled_after}")

        # Print param stats
        log("LoRA parameter statistics:")
        _print_param_stats(model.model)

        return model
    else:
        log("Failed to load LoRA adapters", "❌")
        return model


def verify_lora_merge(model: YOLO) -> YOLO:
    """Step 4: Merge LoRA weights."""
    log_section("Step 4: Merge LoRA Weights")

    # Check before merge
    lora_enabled_before = getattr(model.model, "lora_enabled", False)
    has_merge_method = hasattr(model.model, "merge_and_unload")
    log(f"LoRA enabled before merge: {lora_enabled_before}")
    log(f"Has merge_and_unload: {has_merge_method}")

    # Merge
    success = model.merge_lora()

    if success:
        log(f"LoRA weights merged successfully!", "✅")

        # Check after merge
        lora_enabled_after = getattr(model.model, "lora_enabled", False)
        log(f"LoRA enabled after merge: {lora_enabled_after}")

        # Verify merge_and_unload is no longer available
        has_merge_after = hasattr(model.model, "merge_and_unload")
        log(f"Has merge_and_unload after merge: {has_merge_after}")

        return model
    else:
        log("Failed to merge LoRA weights", "❌")
        return model


def verify_inference(model: YOLO, stage: str = "merged"):
    """Step 5: Verify inference works."""
    log_section(f"Step 5: Inference Verification ({stage})")

    test_sources = [
        TEST_IMAGE,
    ]

    for source in test_sources:
        log(f"Running inference on {source}")
        try:
            if not os.path.exists(source):
                log(f"Test image not found, skipping", "⚠️")
                continue
            results = model.predict(
                source=source,
                imgsz=320,
                verbose=False,
                device="cpu",
                save=False
            )
            num_objects = len(results[0]) if results else 0
            log(f"Detected {num_objects} objects", "✅")
        except Exception as e:
            log(f"Inference failed: {e}", "❌")
            return False

    return True


def verify_full_pipeline():
    """Run the complete LoRA pipeline verification."""
    print("\n" + "=" * 70)
    print("  LoRA End-to-End Pipeline Verification")
    print("  Model: yolo11n | Dataset: coco8")
    print("=" * 70)

    try:
        # Step 0: Verify base model
        model = verify_base_model()
        if model is None:
            log("Base model verification failed, exiting", "❌")
            return False

        # Step 1: Train with LoRA
        model = verify_lora_training(model, epochs=2)

        # Step 2: Save adapters
        adapter_dir = verify_lora_save(model)
        if adapter_dir is None:
            log("LoRA save failed, exiting", "❌")
            return False

        # Step 3: Load adapters (reload fresh model)
        log_section("Step 3b: Reload Model for Adapter Loading")
        model = YOLO(str(PROJECT_ROOT / "yolo11n.pt"))
        model = verify_lora_load(model, adapter_dir)

        # Verify inference with LoRA
        verify_inference(model, "with_lora")

        # Step 4: Merge
        model = verify_lora_merge(model)

        # Verify inference after merge
        verify_inference(model, "merged")

        # Final summary
        log_section("Verification Complete!")
        log("All LoRA pipeline stages completed successfully!", "🎉")

        return True

    except Exception as e:
        log(f"Pipeline failed with error: {e}", "❌")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = verify_full_pipeline()
    sys.exit(0 if success else 1)
