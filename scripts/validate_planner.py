"""Planner validation harness — comprehensive benchmarking, correctness, and training comparison.

Usage (from project root):
    python scripts/validate_planner.py --mode benchmark
    python scripts/validate_planner.py --mode decision-check
    python scripts/validate_planner.py --mode train-compare --model yolo11s --epochs 3
    python scripts/validate_planner.py --mode all

Modes:
    benchmark      — measure plan() / detect_targets() latency and cache efficiency
    decision-check — verify ACCEPT/ADAPT/REFUSE for YOLO11s/12s/RT-DETR-l
    train-compare  — run short training experiments (Planner ON vs OFF)
    audit-check    — verify audit JSON files in runs/planner_audit/
    all            — run everything sequentially
"""

import argparse
import json
import time
import statistics
import tempfile
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn

from ultralytics.utils.lora.planner import (
    ArchitectureFingerprint,
    PEFTPlanner,
    DecisionAudit,
    _fingerprint_cache,
)
from ultralytics.utils.lora.config import LoRAConfig


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark():
    """Measure Planner execution time with/without cache, and memory overhead."""
    print("\n" + "=" * 60)
    print("  Planner Performance Benchmark")
    print("=" * 60)

    # 1. Build a representative dummy model (YOLO12-like scale)
    class AAttn(nn.Module):
        pass

    class DummyYOLO12(nn.Module):
        def __init__(self, n_conv=120, n_attn=45, n_linear=20):
            super().__init__()
            self.convs = nn.ModuleList([nn.Conv2d(3, 16, 3) for _ in range(n_conv)])
            self.attns = nn.ModuleList([AAttn() for _ in range(n_attn)])
            self.linears = nn.ModuleList([nn.Linear(64, 64) for _ in range(n_linear)])

    model = DummyYOLO12()
    config = LoRAConfig(peft_type="lora", r=16)
    planner = PEFTPlanner()

    # Warm-up (first call populates cache)
    planner.plan(model, config)

    # Cold-start timing (cache miss)
    ArchitectureFingerprint.invalidate_cache(model)
    times_cold = []
    for _ in range(5):
        t0 = time.perf_counter()
        planner.plan(model, config)
        t1 = time.perf_counter()
        times_cold.append((t1 - t0) * 1000)

    # Warm timing (cache hit)
    times_warm = []
    for _ in range(20):
        t0 = time.perf_counter()
        planner.plan(model, config)
        t1 = time.perf_counter()
        times_warm.append((t1 - t0) * 1000)

    # detect_targets timing
    times_targets = []
    for _ in range(10):
        t0 = time.perf_counter()
        planner.detect_targets(model, config)
        t1 = time.perf_counter()
        times_targets.append((t1 - t0) * 1000)

    print(f"  Cold-start plan() latency: {statistics.mean(times_cold):.3f} ms (std={statistics.stdev(times_cold):.3f})")
    print(f"  Warm   plan() latency:     {statistics.mean(times_warm):.3f} ms (std={statistics.stdev(times_warm):.3f})")
    print(f"  detect_targets() latency:  {statistics.mean(times_targets):.3f} ms (std={statistics.stdev(times_targets):.3f})")
    print(f"  Cache speedup: {statistics.mean(times_cold) / statistics.mean(times_warm):.1f}x")
    print(f"  Cache entries alive: {len(_fingerprint_cache)}")

    return {
        "cold_ms": times_cold,
        "warm_ms": times_warm,
        "targets_ms": times_targets,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Decision correctness
# ─────────────────────────────────────────────────────────────────────────────

def run_decision_check():
    """Verify Planner decisions for known architectures."""
    print("\n" + "=" * 60)
    print("  Decision Correctness Verification")
    print("=" * 60)

    # YOLO11s-like: no attention → ACCEPT
    class YOLO11s(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(80):
                setattr(self, f"c_{i}", nn.Conv2d(3, 16, 3))

    # YOLO12s-like: AAttn → ADAPT for DoRA, ADAPT (rank cap) for LoRA
    class AAttn(nn.Module):
        pass

    class YOLO12s(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(80):
                setattr(self, f"c_{i}", nn.Conv2d(3, 16, 3))
            for i in range(32):
                setattr(self, f"a_{i}", AAttn())

    # RT-DETR-like: RTDETRDecoder → REFUSE for LoRA-family, ADAPT for IA3
    class RTDETRDecoder(nn.Module):
        pass

    class RTDETR(nn.Module):
        def __init__(self):
            super().__init__()
            for i in range(100):
                setattr(self, f"c_{i}", nn.Conv2d(3, 16, 3))
            self.decoder = RTDETRDecoder()

    cases = [
        ("YOLO11s (no attn)", YOLO11s(), "lora", 16, "ACCEPT", None, None),
        ("YOLO12s (DoRA r=16)", YOLO12s(), "dora", 16, "ADAPT", "lora", 8),
        ("YOLO12s (LoRA r=16)", YOLO12s(), "lora", 16, "ADAPT", None, 8),  # rank cap, not ACCEPT
        ("RT-DETR (LoRA r=16)", RTDETR(), "lora", 16, "REFUSE", None, None),
        ("RT-DETR (LoHa r=16)", RTDETR(), "loha", 16, "REFUSE", None, None),  # LoHa is LoRA-family
        ("RT-DETR (IA3 r=16)", RTDETR(), "ia3", 16, "ADAPT", "ia3", None),  # non-LoRA-family → ADAPT
    ]

    planner = PEFTPlanner()
    all_pass = True
    for name, model, variant, rank, expected_status, expected_variant, expected_rank in cases:
        config = LoRAConfig(peft_type=variant, r=rank)
        decision = planner.plan(model, config)
        status_ok = decision.status == expected_status
        variant_ok = (expected_variant is None) or (decision.recommended_variant == expected_variant)
        rank_ok = (expected_rank is None) or (decision.recommended_rank == expected_rank)
        ok = status_ok and variant_ok and rank_ok
        all_pass = all_pass and ok
        marker = "✅" if ok else "❌"
        print(f"  {marker} {name}: {decision.status} (expected {expected_status})")
        if decision.recommended_variant:
            print(f"      variant→{decision.recommended_variant}, rank→{decision.recommended_rank}")
        if decision.refusal_reason:
            print(f"      reason: {decision.refusal_reason[:60]}")
    return all_pass


# ─────────────────────────────────────────────────────────────────────────────
# Audit log check
# ─────────────────────────────────────────────────────────────────────────────

def run_audit_check():
    """Verify audit JSON files in runs/planner_audit/."""
    print("\n" + "=" * 60)
    print("  Audit Log Verification")
    print("=" * 60)

    audit_dir = Path("runs/planner_audit")
    if not audit_dir.exists():
        print(f"  ⚠️  Audit directory not found: {audit_dir}")
        return False

    files = sorted(audit_dir.glob("*.json"))
    print(f"  Found {len(files)} audit files")

    all_valid = True
    for f in files:
        try:
            data = json.loads(f.read_text())
            required = ["timestamp", "model_name", "fingerprint", "variant", "requested_rank", "decision_status"]
            missing = [k for k in required if k not in data]
            if missing:
                print(f"  ❌ {f.name}: missing keys {missing}")
                all_valid = False
            else:
                print(f"  ✅ {f.name}: {data['decision_status']} (variant={data['variant']}, rank={data['requested_rank']})")
        except Exception as e:
            print(f"  ❌ {f.name}: parse error {e}")
            all_valid = False

    return all_valid


# ─────────────────────────────────────────────────────────────────────────────
# Training comparison (short, 3 epochs on COCO128)
# ─────────────────────────────────────────────────────────────────────────────

def run_train_compare(model_name: str, epochs: int = 3, batch: int = 8, imgsz: int = 320):
    """Compare Planner ON vs OFF for a given model."""
    print(f"\n{'=' * 60}")
    print(f"  Training Comparison: {model_name} (epochs={epochs})")
    print("=" * 60)

    from ultralytics import YOLO

    results = {}

    # --- OFF (baseline) ---
    print(f"\n  [Planner OFF] Loading {model_name}...")
    model_off = YOLO(model_name)
    print(f"  [Planner OFF] Training...")
    r_off = model_off.train(
        data="ultralytics/cfg/datasets/coco128.yaml",
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lora_r=16,
        lora_type="lora",
        lora_planner_enabled=False,
        device="mps",
        verbose=False,
    )
    results["off"] = {
        "mAP50": r_off.results.get("metrics/mAP50(B)"),
        "mAP50-95": r_off.results.get("metrics/mAP50-95(B)"),
    }
    print(f"  [Planner OFF] mAP50={results['off']['mAP50']:.4f}, mAP50-95={results['off']['mAP50-95']:.4f}")

    # --- ON ---
    print(f"\n  [Planner ON] Loading {model_name}...")
    model_on = YOLO(model_name)
    print(f"  [Planner ON] Training...")
    r_on = model_on.train(
        data="ultralytics/cfg/datasets/coco128.yaml",
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lora_r=16,
        lora_type="lora",
        lora_planner_enabled=True,
        device="mps",
        verbose=False,
    )
    results["on"] = {
        "mAP50": r_on.results.get("metrics/mAP50(B)"),
        "mAP50-95": r_on.results.get("metrics/mAP50-95(B)"),
    }
    print(f"  [Planner ON] mAP50={results['on']['mAP50']:.4f}, mAP50-95={results['on']['mAP50-95']:.4f}")

    # Delta
    d50 = results["on"]["mAP50"] - results["off"]["mAP50"]
    d5095 = results["on"]["mAP50-95"] - results["off"]["mAP50-95"]
    print(f"\n  Δ mAP50: {d50:+.4f}, Δ mAP50-95: {d5095:+.4f}")
    print(f"  Planner {'improved' if d50 > 0 else 'degraded' if d50 < 0 else 'matched'} performance")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Planner validation harness")
    parser.add_argument("--mode", choices=["benchmark", "decision-check", "train-compare", "audit-check", "all"], default="all")
    parser.add_argument("--model", default="yolo11s.pt", help="Model for train-compare")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs for train-compare")
    args = parser.parse_args()

    report = {"timestamp": datetime.now().isoformat(), "mode": args.mode}

    if args.mode in ("benchmark", "all"):
        report["benchmark"] = run_benchmark()

    if args.mode in ("decision-check", "all"):
        report["decision_correct"] = run_decision_check()

    if args.mode in ("audit-check", "all"):
        report["audit_valid"] = run_audit_check()

    if args.mode in ("train-compare", "all"):
        report["train_compare"] = run_train_compare(args.model, args.epochs)

    # Save report
    report_path = Path("runs/planner_validation_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\n📄 Report saved to {report_path}")

    # Summary
    print("\n" + "=" * 60)
    print("  Validation Summary")
    print("=" * 60)
    for k, v in report.items():
        if k in ("timestamp", "mode"):
            continue
        if isinstance(v, bool):
            print(f"  {'✅' if v else '❌'} {k}")
        elif isinstance(v, dict):
            print(f"  ✅ {k} (data collected)")


if __name__ == "__main__":
    main()
