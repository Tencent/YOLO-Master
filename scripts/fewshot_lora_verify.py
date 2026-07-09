"""Few-shot LoRA validation on COCO128 using MPS.

Compares three configurations on COCO128 (128 images):
  1. Baseline: YOLO-Master-EsMoE-N.pt without any fine-tuning
  2. Standard LoRA: single-rank LoRA on all target layers
  3. MoLoRA (MoE-aware): mixture-of-LoRA with per-expert ranks + router calibration

Usage:
    cd /Users/gatilin/PycharmProjects/YOLO-Master-v0708/scripts
    python3 fewshot_lora_verify.py
"""
import os
import sys
import json
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

import torch
from ultralytics.utils import SETTINGS
SETTINGS["wandb"] = False

from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAMoEAwareConfig, build_moe_aware_layer
from ultralytics.nn.peft.molora.model import _parent_child_name, _get_submodule


MODEL_PATH = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
DATA = "coco128.yaml"  # Ultralytics built-in dataset (128 images)
EPOCHS = 10
BATCH = 8
IMGSZ = 320

def count_params(m):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def apply_molora(model, config):
    """Apply MoLoRA (MoE-aware) to model."""
    from ultralytics.nn.peft.molora import MoLoRAConfigBuilder
    targets = MoLoRAConfigBuilder.auto_detect_targets(model, r=config.r, include_moe=True, only_backbone=False)
    wrapped = 0
    modules_dict = dict(model.named_modules())
    for name in targets:
        if name not in modules_dict: continue
        base = modules_dict[name]
        if not isinstance(base, (torch.nn.Conv2d, torch.nn.Linear)): continue
        pn, cn = _parent_child_name(name)
        parent = _get_submodule(model, pn) if pn else model
        if parent is None or not hasattr(parent, cn): continue
        layer = build_moe_aware_layer(base, config, usage_history=None)
        setattr(parent, cn, layer); wrapped += 1
    model.molora_config = config; model.molora_enabled = True
    from ultralytics.nn.peft.molora.utils import mark_only_molora_as_trainable
    mark_only_molora_as_trainable(model)
    return wrapped


def run_baseline():
    """Run baseline: no fine-tuning, just val on pretrained model."""
    print(f"\n{'='*70}")
    print("[BASELINE] YOLO-Master-EsMoE-N.pt without fine-tuning")
    print(f"{'='*70}")
    t0 = time.time()
    model = YOLO(MODEL_PATH)
    total, trainable = count_params(model.model)
    print(f"Total params: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
    
    # Validation only
    results = model.val(data=DATA, imgsz=IMGSZ, batch=BATCH, device=DEVICE, verbose=False)
    elapsed = time.time() - t0
    metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
    
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    return {"name": "baseline", "ok": True, "elapsed_sec": round(elapsed, 1),
            "params_total": total, "params_trainable": trainable,
            "trainable_pct": round(trainable/total*100, 4), "final_metrics": metrics}


def run_standard_lora():
    """Run standard LoRA: single-rank, no MoE."""
    print(f"\n{'='*70}")
    print("[STANDARD LoRA] Single-rank LoRA on all target layers")
    print(f"{'='*70}")
    t0 = time.time()
    model = YOLO(MODEL_PATH)
    base_total, base_train = count_params(model.model)
    
    # Standard LoRA: simple low-rank adapter (no MoE)
    # Use MoLoRA with num_experts=1, top_k=1 as standard LoRA proxy
    from ultralytics.nn.peft.molora import MoLoRAConfigBuilder
    targets = MoLoRAConfigBuilder.auto_detect_targets(model.model, r=8, include_moe=True, only_backbone=False)
    
    # Wrap with standard LoRA-like config (single expert = standard LoRA)
    from ultralytics.nn.peft.molora.layer import MoLoRALayer
    wrapped = 0
    modules_dict = dict(model.model.named_modules())
    for name in targets:
        if name not in modules_dict: continue
        base = modules_dict[name]
        if not isinstance(base, (torch.nn.Conv2d, torch.nn.Linear)): continue
        pn, cn = _parent_child_name(name)
        parent = _get_submodule(model.model, pn) if pn else model.model
        if parent is None or not hasattr(parent, cn): continue
        layer = MoLoRALayer(base, num_experts=1, top_k=1, r=8, alpha=16)
        setattr(parent, cn, layer); wrapped += 1
    
    model.model.molora_enabled = True
    from ultralytics.nn.peft.molora.utils import mark_only_molora_as_trainable
    mark_only_molora_as_trainable(model.model)
    
    post_total, post_train = count_params(model.model)
    print(f"Wrapped {wrapped} layers | Total: {post_total:,} | Trainable: {post_train:,} ({post_train/post_total*100:.2f}%)")
    
    # Train
    results = model.train(data=DATA, epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ, device=DEVICE,
                          project=str(REPO_ROOT / "scripts" / "runs_fewshot"), name="standard_lora",
                          exist_ok=True, verbose=False, workers=2, patience=0, plots=False, save=False)
    elapsed = time.time() - t0
    metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
    
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    return {"name": "standard_lora", "ok": True, "elapsed_sec": round(elapsed, 1),
            "params_total": post_total, "params_trainable": post_train,
            "trainable_pct": round(post_train/post_total*100, 4), "final_metrics": metrics,
            "wrapped_layers": wrapped}


def run_molora_moe():
    """Run MoLoRA (MoE-aware) with per-expert ranks + calibration."""
    print(f"\n{'='*70}")
    print("[MoLoRA MoE-aware] Per-expert ranks + router calibration")
    print(f"{'='*70}")
    t0 = time.time()
    model = YOLO(MODEL_PATH)
    base_total, base_train = count_params(model.model)
    
    cfg = MoLoRAMoEAwareConfig(
        r=8, alpha=16, num_experts=4, top_k=2, router_type="linear",
        per_expert_rank=True, rank_allocator_mode="frequency",
        rank_budget_total=32, rank_min=2,
        router_calibration=True, router_calib_rank=4,
        balance_loss_coef=0.01, z_loss_coef=0.001, use_rslora=True,
    )
    wrapped = apply_molora(model.model, cfg)
    post_total, post_train = count_params(model.model)
    print(f"Wrapped {wrapped} layers | Total: {post_total:,} | Trainable: {post_train:,} ({post_train/post_total*100:.2f}%)")
    
    # Train
    results = model.train(data=DATA, epochs=EPOCHS, batch=BATCH, imgsz=IMGSZ, device=DEVICE,
                          project=str(REPO_ROOT / "scripts" / "runs_fewshot"), name="molora_moe",
                          exist_ok=True, verbose=False, workers=2, patience=0, plots=False, save=False)
    elapsed = time.time() - t0
    metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
    
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"mAP50-95: {metrics.get('metrics/mAP50-95(B)', 'N/A')}")
    return {"name": "molora_moe", "ok": True, "elapsed_sec": round(elapsed, 1),
            "params_total": post_total, "params_trainable": post_train,
            "trainable_pct": round(post_train/post_total*100, 4), "final_metrics": metrics,
            "wrapped_layers": wrapped}


def main():
    print("=" * 70)
    print("Few-shot LoRA Validation on COCO128")
    print(f"Model: {MODEL_PATH}")
    print(f"Data: {DATA} (128 images)")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH} | ImgSize: {IMGSZ}")
    print("=" * 70)
    
    records = []
    records.append(run_baseline())
    records.append(run_standard_lora())
    records.append(run_molora_moe())
    
    # Save
    out = REPO_ROOT / "scripts" / "fewshot_lora_results.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<20} {'mAP50-95':>12} {'Trainable%':>12} {'Time(s)':>10}")
    print("-" * 70)
    for r in records:
        map_val = r["final_metrics"].get("metrics/mAP50-95(B)", "N/A")
        map_str = f"{map_val:.4f}" if isinstance(map_val, (int, float)) else str(map_val)
        print(f"{r['name']:<20} {map_str:>12} {r['trainable_pct']:>12.2f} {r['elapsed_sec']:>10.1f}")
    print(f"{'='*70}")
    print(f"\n[Saved] {out}")
    
    # Compare improvements
    baseline_map = records[0]["final_metrics"].get("metrics/mAP50-95(B)")
    if baseline_map is not None and baseline_map > 0:
        for r in records[1:]:
            m = r["final_metrics"].get("metrics/mAP50-95(B)")
            if m is not None and m > 0:
                delta = m - baseline_map
                pct = delta / baseline_map * 100
                print(f"{r['name']}: ΔmAP = {delta:+.4f} ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
