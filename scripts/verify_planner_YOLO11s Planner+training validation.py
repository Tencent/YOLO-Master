#!/usr/bin/env python3
"""PEFT Planner + Training Validation Script.

Part A: Planner Decision Verification
- Loads model weights via torch.load
- Computes ArchitectureFingerprint
- Runs PEFTPlanner.plan()
- If ACCEPT/ADAPT: applies LoRA and reports parameter statistics
- If REFUSE: verifies apply_lora returns the model unmodified

Part B: 1-Epoch Training Smoke Test
- If trainable: runs 1 epoch on COCO128 with imgsz=320, batch=4, device=mps
- If MPS fails: retries on CPU
- If REFUSE: runs Full-SFT (lora_r=0)
"""

import sys
import os
import gc
import traceback
import time

WORKSPACE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.utils.lora.planner import ArchitectureFingerprint, PEFTPlanner
from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.utils.lora.api import apply_lora, _compute_param_stats

# -----------------------------------------------------------------------------
MODELS = [
    {
        "name": "YOLO11s",
        "path": os.path.join(WORKSPACE, "yolo11s.pt"),
        "expected_decision": "ACCEPT",
    },
    {
        "name": "YOLO12s",
        "path": os.path.join(WORKSPACE, "yolo12s.pt"),
        "expected_decision": "ADAPT",
    },
    {
        "name": "RT-DETR-l",
        "path": os.path.join(WORKSPACE, "rtdetr-l.pt"),
        "expected_decision": "REFUSE",
    },
]


def load_inner_module(path):
    """Load a .pt checkpoint and return the nn.Module (DetectionModel or inner)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model weight not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "model" in ckpt:
        obj = ckpt["model"]
        if isinstance(obj, nn.Module):
            return obj
    if isinstance(ckpt, nn.Module):
        return ckpt
    return None


def print_and_report(report_lines, msg):
    print(msg)
    report_lines.append(msg)


def main():
    report_lines = []
    print_and_report(report_lines, "=" * 80)
    print_and_report(report_lines, "PEFT Planner + Training Validation Report")
    print_and_report(report_lines, "=" * 80)
    print_and_report(report_lines, "")

    # Allow filtering by command-line argument, e.g.:
    #   python script.py YOLO12s
    target_models = [sys.argv[1]] if len(sys.argv) > 1 else None

    for spec in MODELS:
        name = spec["name"]
        if target_models and name not in target_models:
            continue
        path = spec["path"]
        print_and_report(report_lines, f"\n{'='*40}")
        print_and_report(report_lines, f"Testing {name}")
        print_and_report(report_lines, f"{'='*40}")

        # =====================================================================
        # Part A: Planner Decision Verification
        # =====================================================================
        inner = None
        try:
            inner = load_inner_module(path)
        except Exception as e:
            print_and_report(report_lines, f"❌ {name}: torch.load failed: {e}")
            continue

        if inner is None:
            print_and_report(report_lines, f"❌ {name}: Could not extract nn.Module from checkpoint")
            continue

        # Step 3: Architecture Fingerprint
        try:
            fp = ArchitectureFingerprint.compute(inner)
        except Exception as e:
            print_and_report(report_lines, f"❌ {name}: Fingerprint compute failed: {e}")
            continue

        print_and_report(report_lines, "[Part A] Architecture Fingerprint:")
        print_and_report(report_lines, f"  φ_attn  = {fp.phi_attn:.4f}")
        print_and_report(report_lines, f"  φ_text  = {fp.phi_text:.4f}")
        print_and_report(report_lines, f"  φ_dw    = {fp.phi_dw:.4f}")
        print_and_report(report_lines, f"  φ_group = {fp.phi_group:.4f}")
        print_and_report(report_lines, f"  φ_linear= {fp.phi_linear:.4f}")

        # Step 4: Planner decision
        config = LoRAConfig(r=16, alpha=32, peft_type="lora", planner_enabled=True)
        planner = PEFTPlanner()
        try:
            decision = planner.plan(inner, config)
        except Exception as e:
            print_and_report(report_lines, f"❌ {name}: Planner.plan() failed: {e}")
            continue

        print_and_report(report_lines, f"[Part A] Planner Decision: {decision.status}")
        if decision.predicted_delta is not None:
            print_and_report(report_lines, f"  predicted ΔmAP = {decision.predicted_delta:.4f}")
        if decision.recommended_rank is not None:
            print_and_report(report_lines, f"  recommended rank = {decision.recommended_rank}")
        if decision.recommended_variant:
            print_and_report(report_lines, f"  recommended variant = {decision.recommended_variant}")
        if decision.refusal_reason:
            print_and_report(report_lines, f"  refusal reason = {decision.refusal_reason}")

        # Step 5: apply_lora + parameter stats
        if decision.status in ("ACCEPT", "ADAPT"):
            print_and_report(report_lines, f"[Part A] Decision is {decision.status} — applying LoRA...")
            try:
                yolo = YOLO(path)
            except Exception as e:
                print_and_report(report_lines, f"❌ {name}: YOLO load failed: {e}")
                continue

            try:
                # Pass config with planner_enabled=True so apply_lora also runs the planner
                lora_model = apply_lora(yolo.model, config)
            except Exception as e:
                print_and_report(report_lines, f"❌ {name}: apply_lora() failed: {e}")
                traceback.print_exc()
                continue

            stats = _compute_param_stats(lora_model)
            print_and_report(report_lines, "  Parameter Statistics:")
            print_and_report(report_lines, f"    Total params:     {stats.total:,}")
            print_and_report(report_lines, f"    Trainable params: {stats.trainable:,} ({stats.trainable_pct:.3f}%)")
            print_and_report(report_lines, f"    Frozen params:    {stats.frozen:,}")
            print_and_report(report_lines, f"    Adapter params:   {stats.adapter:,} ({stats.adapter_pct:.3f}%)")

            ready = stats.adapter > 0 and stats.trainable > 0
            print_and_report(report_lines, f"  Model ready for training: {ready}")

            # =================================================================
            # Part B: 1-Epoch Training Smoke Test (ACCEPT/ADAPT only)
            # =================================================================
            if ready:
                print_and_report(report_lines, "[Part B] Running 1-epoch training smoke test on COCO128...")
                start_time = time.time()
                try:
                    yolo.model = lora_model
                    results = yolo.train(
                        data="coco128.yaml",
                        epochs=1,
                        imgsz=320,
                        batch=4,
                        device="mps",
                        lora_planner_enabled=True,
                    )
                    elapsed = time.time() - start_time
                    print_and_report(report_lines, f"  ✅ Training SUCCESS (elapsed: {elapsed:.1f}s)")
                    # Attempt to capture final loss from trainer metrics
                    try:
                        trainer = getattr(yolo, "trainer", None)
                        if trainer and hasattr(trainer, "metrics"):
                            metrics = trainer.metrics
                            loss_keys = [k for k in metrics.keys() if "loss" in k.lower()]
                            if loss_keys:
                                for k in loss_keys:
                                    print_and_report(report_lines, f"  Final {k}: {metrics[k]:.4f}")
                        else:
                            # Fallback: last epoch from results dict
                            if hasattr(results, "results") and isinstance(results.results, dict):
                                box_loss = results.results.get("train/box_loss", None)
                                if box_loss is not None:
                                    print_and_report(report_lines, f"  Final box_loss: {box_loss:.4f}")
                    except Exception:
                        pass
                except Exception as e:
                    elapsed = time.time() - start_time
                    err_lines = traceback.format_exc().splitlines()[-30:]
                    print_and_report(report_lines, f"  ❌ Training FAILURE on MPS (elapsed: {elapsed:.1f}s): {e}")
                    for line in err_lines:
                        print_and_report(report_lines, f"    {line}")

                    # Retry with CPU
                    print_and_report(report_lines, "  Retrying with CPU...")
                    try:
                        yolo.model = lora_model
                        results = yolo.train(
                            data="coco128.yaml",
                            epochs=1,
                            imgsz=320,
                            batch=4,
                            device="cpu",
                            lora_planner_enabled=True,
                        )
                        print_and_report(report_lines, "  ✅ Training SUCCESS on CPU")
                    except Exception as e2:
                        err_lines2 = traceback.format_exc().splitlines()[-30:]
                        print_and_report(report_lines, f"  ❌ Training FAILURE on CPU: {e2}")
                        for line in err_lines2:
                            print_and_report(report_lines, f"    {line}")

            # Cleanup before next model
            del yolo, lora_model
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

        else:  # REFUSE
            print_and_report(report_lines, "[Part A] Decision is REFUSE — verifying apply_lora returns unmodified...")
            try:
                yolo = YOLO(path)
            except Exception as e:
                print_and_report(report_lines, f"❌ {name}: YOLO load failed: {e}")
                continue

            original_stats = _compute_param_stats(yolo.model)
            try:
                refuse_config = LoRAConfig(r=16, alpha=32, peft_type="lora", planner_enabled=True)
                result_model = apply_lora(yolo.model, refuse_config)
            except Exception as e:
                print_and_report(report_lines, f"❌ {name}: apply_lora() on REFUSE model failed unexpectedly: {e}")
                traceback.print_exc()
                continue

            result_stats = _compute_param_stats(result_model)
            no_change = (result_stats.adapter == 0 and result_stats.total == original_stats.total)
            print_and_report(report_lines, f"  Original adapter params: {original_stats.adapter}")
            print_and_report(report_lines, f"  Result adapter params:   {result_stats.adapter}")
            print_and_report(report_lines, f"  Total params unchanged:  {no_change}")
            print_and_report(report_lines, f"  Unmodified check: {'✅ PASS' if no_change else '❌ FAIL'}")

            # =================================================================
            # Part B: Full-SFT for REFUSEd model
            # =================================================================
            print_and_report(report_lines, "[Part B] Running Full-SFT (lora_r=0) smoke test on COCO128...")
            try:
                results = yolo.train(
                    data="coco128.yaml",
                    epochs=1,
                    imgsz=320,
                    batch=4,
                    device="mps",
                    lora_r=0,
                )
                print_and_report(report_lines, "  ✅ Full-SFT SUCCESS")
            except Exception as e:
                err_lines = traceback.format_exc().splitlines()[-30:]
                print_and_report(report_lines, f"  ❌ Full-SFT FAILURE on MPS: {e}")
                for line in err_lines:
                    print_and_report(report_lines, f"    {line}")

                # Retry with CPU
                print_and_report(report_lines, "  Retrying Full-SFT with CPU...")
                try:
                    results = yolo.train(
                        data="coco128.yaml",
                        epochs=1,
                        imgsz=320,
                        batch=4,
                        device="cpu",
                        lora_r=0,
                    )
                    print_and_report(report_lines, "  ✅ Full-SFT SUCCESS on CPU")
                except Exception as e2:
                    err_lines2 = traceback.format_exc().splitlines()[-30:]
                    print_and_report(report_lines, f"  ❌ Full-SFT FAILURE on CPU: {e2}")
                    for line in err_lines2:
                        print_and_report(report_lines, f"    {line}")

            # Cleanup
            del yolo, result_model
            gc.collect()
            if torch.backends.mps.is_available():
                try:
                    torch.mps.empty_cache()
                except Exception:
                    pass

    # -------------------------------------------------------------------------
    # Save report
    # -------------------------------------------------------------------------
    if target_models:
        report_path = os.path.join(
            WORKSPACE,
            "scripts",
            f"experiment_report_{target_models[0]}.txt",
        )
    else:
        report_path = os.path.join(
            WORKSPACE,
            "scripts",
            "experiment_report_YOLO11s Planner+training validation.txt",
        )
    with open(report_path, "w", encoding="utf-8") as f:
        for line in report_lines:
            f.write(line + "\n")
    print_and_report(report_lines, f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
