"""Few-shot LoRA quick validation on COCO128 (MPS).

Runs only 2 configs to fit in 300s:
  1. Baseline: pretrained YOLO-Master-EsMoE-N.pt val only
  2. MoLoRA (MoE-aware): 5 epochs fine-tuning

Usage:
    python3 fewshot_lora_quick.py
"""
import os, sys, json, time
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.update({"WANDB_MODE": "disabled", "WANDB_SILENT": "true", "KMP_DUPLICATE_LIB_OK": "TRUE", "YOLO_AUTOINSTALL": "false", "YOLO_VERBOSE": "false"})
import torch
from ultralytics.utils import SETTINGS
SETTINGS["wandb"] = False
from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAMoEAwareConfig, build_moe_aware_layer
from ultralytics.nn.peft.molora.model import _parent_child_name, _get_submodule

MODEL = str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATA = "coco128.yaml"

def count_params(m):
    return sum(p.numel() for p in m.parameters()), sum(p.numel() for p in m.parameters() if p.requires_grad)

def apply_molora(model, cfg):
    from ultralytics.nn.peft.molora import MoLoRAConfigBuilder
    targets = MoLoRAConfigBuilder.auto_detect_targets(model, r=cfg.r, include_moe=True, only_backbone=False)
    wrapped = 0
    md = dict(model.named_modules())
    for name in targets:
        if name not in md: continue
        base = md[name]
        if not isinstance(base, (torch.nn.Conv2d, torch.nn.Linear)): continue
        pn, cn = _parent_child_name(name)
        parent = _get_submodule(model, pn) if pn else model
        if parent is None or not hasattr(parent, cn): continue
        setattr(parent, cn, build_moe_aware_layer(base, cfg, usage_history=None))
        wrapped += 1
    model.molora_config = cfg; model.molora_enabled = True
    from ultralytics.nn.peft.molora.utils import mark_only_molora_as_trainable
    mark_only_molora_as_trainable(model)
    return wrapped

def run_baseline():
    print(f"\n{'='*60}\n[BASELINE] Pretrained YOLO-Master-EsMoE-N.pt (val only)\n{'='*60}")
    t0 = time.time()
    model = YOLO(MODEL)
    total, trainable = count_params(model.model)
    print(f"Total: {total:,} | Trainable: {trainable:,} ({trainable/total*100:.2f}%)")
    results = model.val(data=DATA, imgsz=320, batch=8, device=DEVICE, verbose=False)
    elapsed = time.time() - t0
    m = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
    print(f"mAP50-95: {m.get('metrics/mAP50-95(B)', 'N/A')} | Time: {elapsed:.1f}s")
    return {"name": "baseline", "ok": True, "elapsed_sec": round(elapsed, 1),
            "params_total": total, "params_trainable": trainable,
            "trainable_pct": round(trainable/total*100, 4), "final_metrics": m}

def run_molora():
    print(f"\n{'='*60}\n[MoLoRA] MoE-aware PEFT (5 epochs on COCO128)\n{'='*60}")
    t0 = time.time()
    model = YOLO(MODEL)
    base_total, base_train = count_params(model.model)
    cfg = MoLoRAMoEAwareConfig(
        r=8, alpha=16, num_experts=4, top_k=2, router_type="linear",
        per_expert_rank=True, rank_allocator_mode="frequency", rank_budget_total=32, rank_min=2,
        router_calibration=True, router_calib_rank=4,
        balance_loss_coef=0.01, z_loss_coef=0.001, use_rslora=True)
    wrapped = apply_molora(model.model, cfg)
    post_total, post_train = count_params(model.model)
    print(f"Wrapped {wrapped} layers | Total: {post_total:,} | Trainable: {post_train:,} ({post_train/post_total*100:.2f}%)")
    results = model.train(data=DATA, epochs=5, batch=8, imgsz=320, device=DEVICE,
                          project=str(REPO_ROOT / "scripts" / "runs_fewshot"), name="molora",
                          exist_ok=True, verbose=False, workers=2, patience=0, plots=False, save=False)
    elapsed = time.time() - t0
    m = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}
    print(f"mAP50-95: {m.get('metrics/mAP50-95(B)', 'N/A')} | Time: {elapsed:.1f}s")
    return {"name": "molora", "ok": True, "elapsed_sec": round(elapsed, 1),
            "params_total": post_total, "params_trainable": post_train,
            "trainable_pct": round(post_train/post_total*100, 4), "final_metrics": m,
            "wrapped_layers": wrapped}

def main():
    print("=" * 60)
    print("Few-shot LoRA Quick Validation on COCO128")
    print(f"Model: {MODEL}")
    print(f"Data: {DATA} (128 images)")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    records = [run_baseline(), run_molora()]
    out = REPO_ROOT / "scripts" / "fewshot_lora_results.json"
    out.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"\n{'='*60}\nSUMMARY\n{'='*60}")
    print(f"{'Config':<15} {'mAP50-95':>12} {'Trainable%':>12} {'Time(s)':>10}")
    print("-" * 60)
    for r in records:
        map_val = r["final_metrics"].get("metrics/mAP50-95(B)", "N/A")
        map_str = f"{map_val:.4f}" if isinstance(map_val, (int, float)) else str(map_val)
        print(f"{r['name']:<15} {map_str:>12} {r['trainable_pct']:>12.2f} {r['elapsed_sec']:>10.1f}")
    print(f"{'='*60}\n[Saved] {out}")

if __name__ == "__main__":
    main()
