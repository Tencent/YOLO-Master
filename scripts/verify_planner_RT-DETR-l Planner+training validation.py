"""
RT-DETR-l Planner + Training Validation Script
Part A: Planner Decision Verification
Part B: 1-Epoch Training Smoke Test (if applicable)
"""

import sys
import traceback
import torch
import torch.nn as nn
from datetime import datetime

# Add project root to path if needed
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ultralytics.utils.lora.planner import ArchitectureFingerprint, PEFTPlanner, PlacementDecision
from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.utils.lora.api import apply_lora, _compute_param_stats

# Report accumulator
REPORT_LINES = []


def log(msg: str):
    print(msg)
    REPORT_LINES.append(msg)


def separator():
    log("=" * 70)


def main():
    separator()
    log("RT-DETR-l PEFT Planner + Training Validation")
    log(f"Timestamp: {datetime.now().isoformat()}")
    log(f"PyTorch: {torch.__version__}")
    log(f"Device: {torch.device('mps')}")
    separator()

    # ========================================================================
    # Part A: Planner Decision Verification
    # ========================================================================
    log("\n[Part A] Planner Decision Verification")
    separator()

    weight_path = "rtdetr-l.pt"
    log(f"Loading checkpoint: {weight_path}")

    # 1. Load the checkpoint with torch.load
    ckpt = torch.load(weight_path, map_location="cpu", weights_only=False)
    log(f"Checkpoint keys: {list(ckpt.keys())}")

    # 2. Navigate to the nn.Module
    detection_model = ckpt["model"]
    log(f"Checkpoint model type: {type(detection_model).__name__}")

    inner_model = getattr(detection_model, "model", detection_model)
    log(f"Inner model type: {type(inner_model).__name__}")

    # 3. Compute ArchitectureFingerprint on the inner model
    fingerprint = ArchitectureFingerprint.compute(inner_model)
    log(f"\nArchitecture Fingerprint:")
    log(f"  phi_attn   = {fingerprint.phi_attn:.4f}")
    log(f"  phi_text   = {fingerprint.phi_text:.4f}")
    log(f"  phi_dw     = {fingerprint.phi_dw:.4f}")
    log(f"  phi_group  = {fingerprint.phi_group:.4f}")
    log(f"  phi_linear = {fingerprint.phi_linear:.4f}")

    # 4. Call PEFTPlanner.plan() with LoRAConfig
    config = LoRAConfig(r=16, alpha=32, peft_type="lora", planner_enabled=True)
    log(f"\nLoRAConfig: r={config.r}, alpha={config.alpha}, peft_type={config.peft_type}, planner_enabled={config.planner_enabled}")

    planner = PEFTPlanner()
    decision = planner.plan(detection_model, config)

    log(f"\nPlanner Decision: {decision.status}")
    if decision.refusal_reason:
        log(f"  Refusal Reason: {decision.refusal_reason}")
    if decision.predicted_delta is not None:
        log(f"  Predicted ΔmAP: {decision.predicted_delta:.4f}")
    if decision.recommended_variant:
        log(f"  Recommended Variant: {decision.recommended_variant}")
    if decision.recommended_rank is not None:
        log(f"  Recommended Rank: {decision.recommended_rank}")
    if decision.safety_overrides:
        log(f"  Safety Overrides: {decision.safety_overrides}")
    if decision.target_modules_hint:
        log(f"  Target Modules Hint: {len(decision.target_modules_hint)} modules")

    # 5. Apply LoRA or verify REFUSE behavior
    trainable = False
    if decision.status in ("ACCEPT", "ADAPT"):
        log("\n[Part A.5] Applying LoRA with planner_enabled=True")
        modified_model = apply_lora(detection_model, config)

        # Check if model was actually modified
        stats = _compute_param_stats(modified_model)
        log(f"\nParameter Statistics:")
        log(f"  Total parameters:      {stats.total:,}")
        log(f"  Trainable parameters:  {stats.trainable:,} ({stats.trainable_pct:.2f}%)")
        log(f"  Frozen parameters:     {stats.frozen:,}")
        log(f"  Adapter parameters:    {stats.adapter:,} ({stats.adapter_pct:.4f}%)")
        log(f"  Base total (excl. adapters): {stats.base_total:,}")

        # Report targeted modules
        if hasattr(modified_model, "lora_target_modules"):
            targets = modified_model.lora_target_modules
            log(f"\nTargeted Modules ({len(targets)} total):")
            for t in targets[:10]:
                log(f"  - {t}")
            if len(targets) > 10:
                log(f"  ... and {len(targets) - 10} more")
        else:
            log("\nNo lora_target_modules attribute found (model may not be LoRA-wrapped).")

        # Check if model is ready for training
        if stats.adapter > 0 and stats.trainable > 0:
            log("\n✅ Model is ready for training (LoRA adapters applied).")
            trainable = True
        else:
            log("\n⚠️ Model has no adapter parameters or no trainable parameters.")

    elif decision.status == "REFUSE":
        log("\n[Part A.5] Verifying REFUSE behavior (apply_lora should return unmodified model)")
        original_stats = _compute_param_stats(detection_model)
        returned_model = apply_lora(detection_model, config)
        returned_stats = _compute_param_stats(returned_model)

        log(f"\nOriginal total params:  {original_stats.total:,}")
        log(f"Returned total params:  {returned_stats.total:,}")
        log(f"Original adapter params:  {original_stats.adapter:,}")
        log(f"Returned adapter params:  {returned_stats.adapter:,}")

        if returned_stats.adapter == 0 and returned_stats.total == original_stats.total:
            log("\n✅ Verified: apply_lora() returned the model unmodified (no adapters added).")
        else:
            log("\n❌ ERROR: Model was unexpectedly modified despite REFUSE decision!")

    else:
        log(f"\n⚠️ Unknown decision status: {decision.status}")

    separator()

    # ========================================================================
    # Part B: 1-Epoch Training Smoke Test
    # ========================================================================
    log("\n[Part B] 1-Epoch Training Smoke Test")
    separator()

    training_result = "NOT_RUN"
    epoch_time = None
    final_loss = None
    warnings = []
    error_traceback = None

    if decision.status in ("ACCEPT", "ADAPT"):
        log("Model is trainable (ACCEPT/ADAPT). Running LoRA smoke test...")
        try:
            from ultralytics import YOLO

            # Load fresh model for training to avoid state contamination from Part A
            model = YOLO(weight_path)
            log(f"Loaded fresh model: {type(model).__name__}")

            # Run 1-epoch training with MPS
            log(f"Starting training: epochs=1, imgsz=320, batch=4, device=mps, lora_planner_enabled=True")
            results = model.train(
                data="coco128.yaml",
                epochs=1,
                imgsz=320,
                batch=4,
                device="mps",
                lora_planner_enabled=True,
                lora_r=config.r,
                lora_alpha=config.alpha,
                lora_peft_type=config.peft_type,
            )

            training_result = "SUCCESS"
            log("\n✅ Training completed successfully!")

            # Try to extract metrics
            try:
                if hasattr(results, "results_dict"):
                    rd = results.results_dict
                    final_loss = rd.get("train/box_loss", rd.get("train/loss", None))
                elif hasattr(results, "metrics"):
                    final_loss = getattr(results.metrics, "train_loss", None)
            except Exception as e:
                warnings.append(f"Could not extract final loss: {e}")

        except Exception as e:
            error_traceback = traceback.format_exc()
            log(f"\n❌ MPS training failed: {e}")
            log(f"Traceback:\n{error_traceback}")

            # Retry with CPU
            log("\nRetrying with device='cpu'...")
            try:
                from ultralytics import YOLO
                model = YOLO(weight_path)
                results = model.train(
                    data="coco128.yaml",
                    epochs=1,
                    imgsz=320,
                    batch=4,
                    device="cpu",
                    lora_planner_enabled=True,
                    lora_r=config.r,
                    lora_alpha=config.alpha,
                    lora_peft_type=config.peft_type,
                )
                training_result = "SUCCESS (CPU fallback)"
                log("\n✅ CPU fallback training completed successfully!")
            except Exception as e2:
                error_traceback = traceback.format_exc()
                training_result = "FAILURE"
                log(f"\n❌ CPU fallback also failed: {e2}")
                log(f"Traceback:\n{error_traceback}")

    elif decision.status == "REFUSE":
        log("Model was REFUSEd. Running Full-SFT smoke test instead...")
        try:
            from ultralytics import YOLO
            model = YOLO(weight_path)
            log(f"Loaded fresh model: {type(model).__name__}")

            log(f"Starting Full-SFT training: epochs=1, imgsz=320, batch=4, device=mps, lora_r=0")
            results = model.train(
                data="coco128.yaml",
                epochs=1,
                imgsz=320,
                batch=4,
                device="mps",
                lora_r=0,
            )

            training_result = "SUCCESS (Full-SFT)"
            log("\n✅ Full-SFT training completed successfully!")

            # Check if LoRA was NOT injected
            if hasattr(model, "lora_enabled") and model.lora_enabled:
                warnings.append("Unexpected: lora_enabled=True despite lora_r=0")
            else:
                log("Verified: No LoRA injection detected (lora_enabled not set or False).")

        except Exception as e:
            error_traceback = traceback.format_exc()
            log(f"\n❌ MPS Full-SFT training failed: {e}")
            log(f"Traceback:\n{error_traceback}")

            # Retry with CPU
            log("\nRetrying Full-SFT with device='cpu'...")
            try:
                from ultralytics import YOLO
                model = YOLO(weight_path)
                results = model.train(
                    data="coco128.yaml",
                    epochs=1,
                    imgsz=320,
                    batch=4,
                    device="cpu",
                    lora_r=0,
                )
                training_result = "SUCCESS (Full-SFT, CPU fallback)"
                log("\n✅ CPU fallback Full-SFT training completed successfully!")
            except Exception as e2:
                error_traceback = traceback.format_exc()
                training_result = "FAILURE"
                log(f"\n❌ CPU fallback Full-SFT also failed: {e2}")
                log(f"Traceback:\n{error_traceback}")

    else:
        log(f"Unknown decision status '{decision.status}', skipping training.")

    separator()

    # ========================================================================
    # Summary Report
    # ========================================================================
    log("\n[Summary Report]")
    separator()
    log(f"Model:                {weight_path}")
    log(f"Architecture Family:  {ArchitectureFingerprint._detect_architecture_family(inner_model)}")
    log(f"Fingerprint:            φ_attn={fingerprint.phi_attn:.4f}, φ_text={fingerprint.phi_text:.4f}, "
        f"φ_dw={fingerprint.phi_dw:.4f}, φ_group={fingerprint.phi_group:.4f}, φ_linear={fingerprint.phi_linear:.4f}")
    log(f"Planner Decision:     {decision.status}")
    if decision.refusal_reason:
        log(f"Refusal Reason:       {decision.refusal_reason}")
    if decision.predicted_delta is not None:
        log(f"Predicted ΔmAP:       {decision.predicted_delta:.4f}")
    log(f"Training Result:      {training_result}")
    if final_loss is not None:
        log(f"Final Loss:           {final_loss:.4f}")
    if epoch_time is not None:
        log(f"Epoch Time:           {epoch_time:.2f}s")
    if warnings:
        log(f"Warnings ({len(warnings)}):")
        for w in warnings:
            log(f"  - {w}")
    if error_traceback and training_result.startswith("FAILURE"):
        log(f"\nLast Error Traceback (last 30 lines):")
        lines = error_traceback.strip().split("\n")
        for line in lines[-30:]:
            log(f"  {line}")

    separator()

    # Save report to file
    report_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "experiment_report_RT-DETR-l Planner+training validation.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"\nReport saved to: {report_path}")

    return 0 if training_result.startswith("SUCCESS") or decision.status == "REFUSE" else 1


if __name__ == "__main__":
    sys.exit(main())
