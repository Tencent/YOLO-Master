"""
PEFT Planner Verification Script for YOLO-Master
Tests: YOLO11s, YOLO12s, RT-DETR-l
Part A: Planner Decision Verification
Part B: 1-Epoch Training Smoke Test (if applicable)
"""
import sys
import traceback
import torch
import torch.nn as nn

# Add project root to path if needed
sys.path.insert(0, "/Users/gatilin/PycharmProjects/YOLO-Master-v260703")

from ultralytics.utils.lora.planner import ArchitectureFingerprint, PEFTPlanner
from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.utils.lora.api import apply_lora, _compute_param_stats


MODELS = {
    "YOLO11s": "yolo11s.pt",
    "YOLO12s": "yolo12s.pt",
    "RT-DETR-l": "rtdetr-l.pt",
}

REPORT_LINES = []


def log(msg: str):
    print(msg)
    REPORT_LINES.append(msg)


def verify_model(model_name: str, weight_path: str):
    log(f"\n{'='*60}")
    log(f"PART A: Planner Verification — {model_name}")
    log(f"{'='*60}")
    log(f"Weight file: {weight_path}")

    # 1. Load checkpoint
    checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
    log(f"Checkpoint type: {type(checkpoint).__name__}")
    if isinstance(checkpoint, dict):
        log(f"Checkpoint keys: {list(checkpoint.keys())}")
    else:
        log("WARNING: Checkpoint is not a dict")
        return None, None, None

    # 2. Navigate to nn.Module
    if "model" in checkpoint:
        model = checkpoint["model"]
    else:
        log("ERROR: No 'model' key in checkpoint")
        return None, None, None

    log(f"Loaded model type: {type(model).__name__}")
    inner_model = getattr(model, "model", model)
    log(f"Inner model type: {type(inner_model).__name__}")
    if hasattr(inner_model, "__len__"):
        log(f"Inner model length: {len(inner_model)}")

    # 3. Compute architecture fingerprint
    fingerprint = ArchitectureFingerprint.compute(inner_model)
    log(f"\nArchitecture Fingerprint:")
    log(f"  φ_attn   = {fingerprint.phi_attn:.4f}")
    log(f"  φ_text   = {fingerprint.phi_text:.4f}")
    log(f"  φ_dw     = {fingerprint.phi_dw:.4f}")
    log(f"  φ_group  = {fingerprint.phi_group:.4f}")
    log(f"  φ_linear = {fingerprint.phi_linear:.4f}")

    # 4. Planner decision
    config = LoRAConfig(r=16, alpha=32, peft_type="lora", planner_enabled=True)
    planner = PEFTPlanner()
    decision = planner.plan(model, config)

    log(f"\nPlanner Decision: {decision.status}")
    if decision.refusal_reason:
        log(f"  Refusal Reason: {decision.refusal_reason}")
    if decision.recommended_variant:
        log(f"  Recommended Variant: {decision.recommended_variant}")
    if decision.recommended_rank is not None:
        log(f"  Recommended Rank: {decision.recommended_rank}")
    if decision.predicted_delta is not None:
        log(f"  Predicted ΔmAP: {decision.predicted_delta:.4f}")
    if decision.target_modules_hint:
        log(f"  Target modules hint count: {len(decision.target_modules_hint)}")
    if decision.safety_overrides:
        log(f"  Safety overrides: {decision.safety_overrides}")

    # 5. Apply LoRA and verify
    stats_before = _compute_param_stats(model)
    log(f"\nParameters BEFORE apply_lora:")
    log(f"  Total:    {stats_before.total:,}")
    log(f"  Trainable: {stats_before.trainable:,}")
    log(f"  Frozen:   {stats_before.frozen:,}")
    log(f"  Adapter:  {stats_before.adapter:,}")

    try:
        modified_model = apply_lora(model, args=config)
    except Exception as e:
        log(f"\nERROR during apply_lora: {type(e).__name__}: {e}")
        traceback.print_exc()
        return model_name, fingerprint, None

    stats_after = _compute_param_stats(modified_model)
    log(f"\nParameters AFTER apply_lora:")
    log(f"  Total:    {stats_after.total:,}")
    log(f"  Trainable: {stats_after.trainable:,}")
    log(f"  Frozen:   {stats_after.frozen:,}")
    log(f"  Adapter:  {stats_after.adapter:,}")

    adapter_added = stats_after.adapter - stats_before.adapter

    if decision.status in ("ACCEPT", "ADAPT"):
        log(f"\n  Adapter parameters added: {adapter_added:,}")
        target_modules = getattr(modified_model, "lora_target_modules", [])
        log(f"  Target modules count: {len(target_modules)}")
        if target_modules:
            log(f"  First 10 targets: {target_modules[:10]}")
        log(f"  Model ready for training: {stats_after.adapter > 0}")

        if stats_after.adapter <= 0:
            log(f"  WARNING: Expected adapters but none found!")

    elif decision.status == "REFUSE":
        log(f"\n  REFUSE verification:")
        if adapter_added == 0 and stats_after.total == stats_before.total:
            log(f"  ✓ Verified: Model returned unmodified (no adapter parameters added)")
        else:
            log(f"  ✗ WARNING: Model was modified despite REFUSE decision!")
            log(f"    Adapter delta: {adapter_added}, Total delta: {stats_after.total - stats_before.total}")

    return model_name, fingerprint, decision


def main():
    log("=" * 60)
    log("PEFT Planner Verification Report")
    log(f"PyTorch version: {torch.__version__}")
    log(f"MPS available: {torch.backends.mps.is_available()}")
    log("=" * 60)

    results = {}
    for model_name, weight_path in MODELS.items():
        name, fp, decision = verify_model(model_name, weight_path)
        if decision:
            results[model_name] = (fp, decision)

    # Save report
    report_path = "scripts/experiment_report_YOLO12s Planner+training validation.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(REPORT_LINES) + "\n")
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
