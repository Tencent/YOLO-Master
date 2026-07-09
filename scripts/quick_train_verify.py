"""Quick training verification: 500-train + 100-val subset of COCO2017, 1 epoch.

Validates the full MoE-aware PEFT training loop:
  - MoE-aware layer wrapping
  - Forward + backward + optimizer step
  - Auxiliary loss (balance + z-loss) collection
  - Validation metrics (mAP)
  - JSON result persistence

Usage:
    python scripts/quick_train_verify.py
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


HERE = Path(__file__).parent
MODEL_PATH = "YOLO-Master-EsMoE-N.pt"
DATA_YAML = str(HERE / "coco2017_quick.yaml")
RESULTS_JSON = HERE / "quick_verify_results.json"

EPOCHS = 1
BATCH = 1
IMGSZ = 320
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)


def count_params(m: torch.nn.Module):
    total = sum(p.numel() for p in m.parameters())
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    return total, trainable


def apply_moe_aware_to_model(model, config):
    target_modules = getattr(config, "target_modules", None)
    if target_modules is None or not target_modules:
        from ultralytics.nn.peft.molora import MoLoRAConfigBuilder
        target_modules = MoLoRAConfigBuilder.auto_detect_targets(
            model, r=config.r, include_moe=True, only_backbone=False
        )
    wrapped = 0
    modules_dict = dict(model.named_modules())
    for name in target_modules:
        if name not in modules_dict:
            continue
        base_layer = modules_dict[name]
        if not isinstance(base_layer, (torch.nn.Conv2d, torch.nn.Linear)):
            continue
        parent_name, child_name = _parent_child_name(name)
        parent = _get_submodule(model, parent_name) if parent_name else model
        if parent is None or not hasattr(parent, child_name):
            continue
        layer = build_moe_aware_layer(base_layer, config, usage_history=None)
        setattr(parent, child_name, layer)
        wrapped += 1
    model.molora_config = config
    model.molora_enabled = True
    from ultralytics.nn.peft.molora.utils import mark_only_molora_as_trainable
    mark_only_molora_as_trainable(model)
    return wrapped


def run_training(name: str, config: MoLoRAMoEAwareConfig):
    print(f"\n{'='*70}\n=== Quick Verify: {name.upper()} {'='*40}\n{'='*70}")
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH} | Data: {DATA_YAML}")

    t0 = time.time()
    model = YOLO(MODEL_PATH)
    base_total, base_train = count_params(model.model)
    print(f"[Pre-train] total={base_total:,} trainable={base_train:,}")

    wrapped = apply_moe_aware_to_model(model.model, config)
    print(f"[MoE-aware] Wrapped {wrapped} layers")

    post_total, post_train = count_params(model.model)
    print(f"[Post-wrap] total={post_total:,} trainable={post_train:,} ({post_train/post_total*100:.2f}%)")

    # Verify calibration presence
    has_calib = any(
        hasattr(m, "router_calibration") and m.router_calibration is not None
        for m in model.model.modules()
    )
    print(f"[Calibration] Present={has_calib}")

    # Verify per-expert ranks
    has_per_rank = any(
        hasattr(m, "_expert_ranks") and m._expert_ranks is not None
        for m in model.model.modules()
    )
    print(f"[Per-expert rank] Present={has_per_rank}")
    if has_per_rank:
        for name, m in model.model.named_modules():
            if hasattr(m, "_expert_ranks") and m._expert_ranks is not None:
                print(f"  [Example {name}] ranks={m._expert_ranks}")
                break

    print("\n[Training start] ...")
    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            project=str(HERE / "runs_quick_verify"),
            name=f"quick_{name}",
            exist_ok=True,
            verbose=True,
            workers=2,
            patience=0,
            plots=False,
            save=True,
        )
        ok = True
        err = None
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        results = None
        print(f"[ERROR] {err}")
        import traceback
        traceback.print_exc()

    elapsed = time.time() - t0

    final_metrics = {}
    if ok and results is not None and hasattr(results, "results_dict"):
        final_metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}

    record = {
        "name": name,
        "ok": ok,
        "error": err,
        "elapsed_sec": round(elapsed, 1),
        "params_total": post_total,
        "params_trainable": post_train,
        "trainable_pct": round(post_train / post_total * 100, 4),
        "has_calibration": has_calib,
        "has_per_expert_rank": has_per_rank,
        "final_metrics": final_metrics,
    }
    print(f"\n[Final metrics] {json.dumps(final_metrics, indent=2)}")
    return record


def main():
    print(f"{'='*70}")
    print("MoE-aware PEFT Quick Training Verification")
    print(f"{'='*70}")

    config = MoLoRAMoEAwareConfig(
        r=8,
        alpha=16,
        num_experts=4,
        top_k=2,
        router_type="linear",
        per_expert_rank=True,
        rank_allocator_mode="frequency",
        rank_budget_total=32,
        rank_min=2,
        router_calibration=True,
        router_calib_rank=4,
        balance_loss_coef=0.01,
        z_loss_coef=0.001,
        use_rslora=True,
    )

    record = run_training("moe_aware_full", config)

    RESULTS_JSON.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    print(f"\n[Saved] Results: {RESULTS_JSON}")

    # Summary
    print(f"\n{'='*70}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*70}")
    status = "✅ PASSED" if record["ok"] else "❌ FAILED"
    print(f"Status: {status}")
    print(f"Elapsed: {record['elapsed_sec']:.1f}s")
    print(f"Trainable params: {record['params_trainable']:,} ({record['trainable_pct']:.2f}%)")
    print(f"Calibration: {'Y' if record['has_calibration'] else 'N'}")
    print(f"Per-expert rank: {'Y' if record['has_per_expert_rank'] else 'N'}")
    m = record["final_metrics"]
    map_val = m.get("metrics/mAP50-95(B)")
    if map_val is not None:
        print(f"mAP50-95: {map_val:.4f}")
    else:
        print(f"mAP50-95: N/A (check training output above)")
    print(f"{'='*70}")

    if not record["ok"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
