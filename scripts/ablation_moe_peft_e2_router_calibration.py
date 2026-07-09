"""E2: Router calibration ablation (with ΔW_r vs without ΔW_r).

Compares:
  - baseline:   standard MoLoRA router (no calibration)
  - calibrated: MoLoRA router + RouterCalibration low-rank correction

Usage:
    python scripts/ablation_moe_peft_e2_router_calibration.py
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
DATA_YAML = "coco2017.yaml"
PROJECT_DIR = HERE / "runs_e2"
RESULTS_JSON = HERE / "e2_router_calibration_results.json"

EPOCHS = 3
BATCH = 8
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


def run_variant(name: str, config: MoLoRAMoEAwareConfig):
    print(f"\n{'='*70}\n=== Variant: {name.upper()} {'='*40}\n{'='*70}")

    t0 = time.time()
    model = YOLO(MODEL_PATH)
    base_total, base_train = count_params(model.model)
    print(f"[Pre-train] total={base_total:,} trainable={base_train:,}")

    wrapped = apply_moe_aware_to_model(model.model, config)
    print(f"[MoE-aware] Wrapped {wrapped} layers")

    # Verify calibration presence
    has_calib = any(
        hasattr(m, "router_calibration") and m.router_calibration is not None
        for m in model.model.modules()
    )
    print(f"[Calibration] Present={has_calib}")

    post_total, post_train = count_params(model.model)
    print(f"[Post-wrap] total={post_total:,} trainable={post_train:,}")

    try:
        results = model.train(
            data=DATA_YAML,
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            project=str(PROJECT_DIR),
            name=f"e2_{name}",
            exist_ok=True,
            verbose=False,
            workers=2,
            patience=0,
            plots=False,
            save=False,
        )
        ok = True
        err = None
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        results = None
        print(f"[ERROR] {err}")

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
        "final_metrics": final_metrics,
    }
    print(f"[Final metrics] {json.dumps(final_metrics, indent=2)}")
    return record


def main():
    print(f"Device: {DEVICE} | Epochs: {EPOCHS} | Batch: {BATCH} | Imgsz: {IMGSZ}")
    PROJECT_DIR.mkdir(exist_ok=True, parents=True)

    BASE_CONFIG = dict(
        r=8,
        alpha=16,
        num_experts=4,
        top_k=2,
        router_type="linear",
        balance_loss_coef=0.01,
        z_loss_coef=0.001,
        use_rslora=True,
    )

    variants = [
        {
            "name": "baseline",
            "config": MoLoRAMoEAwareConfig(
                router_calibration=False,
                **BASE_CONFIG,
            ),
        },
        {
            "name": "calibrated_r4",
            "config": MoLoRAMoEAwareConfig(
                router_calibration=True,
                router_calib_rank=4,
                **BASE_CONFIG,
            ),
        },
        {
            "name": "calibrated_r8",
            "config": MoLoRAMoEAwareConfig(
                router_calibration=True,
                router_calib_rank=8,
                **BASE_CONFIG,
            ),
        },
    ]

    all_records = []
    for v in variants:
        rec = run_variant(v["name"], v["config"])
        all_records.append(rec)
        RESULTS_JSON.write_text(json.dumps(all_records, indent=2, ensure_ascii=False))

    print("\n" + "=" * 100)
    print(f"{'Variant':<18} {'OK':<3} {'Trainable':>11} {'%':>7} {'Calib':>6} {'mAP50-95':>10}")
    print("-" * 100)
    for r in all_records:
        m = r["final_metrics"].get("metrics/mAP50-95(B)", float("nan"))
        print(f"{r['name']:<18} {'Y' if r['ok'] else 'N':<3} "
              f"{r['params_trainable']:>11,} {r['trainable_pct']:>7.3f} "
              f"{'Y' if r['has_calibration'] else 'N':>6} {m if isinstance(m, float) else '':>10}")
    print("=" * 100)
    print(f"\n详细结果: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
