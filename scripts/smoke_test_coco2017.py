"""Quick smoke test: verify MoE-aware PEFT pipeline on COCO2017 config.

Part A: Verify coco2017.yaml is valid and COCO2017 dataset loads correctly.
Part B: Verify MoE-aware wrapping + forward + routing stats collection.
"""
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

import torch
import yaml
from ultralytics.utils import SETTINGS
SETTINGS["wandb"] = False

from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAMoEAwareConfig, build_moe_aware_layer
from ultralytics.nn.peft.molora.model import _parent_child_name, _get_submodule


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


def main():
    DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")
    DATA_YAML = REPO_ROOT / "scripts" / "coco2017.yaml"
    MODEL_PATH = "YOLO-Master-EsMoE-N.pt"

    print("=" * 60)
    print("MoE-aware PEFT COCO2017 Smoke Test")
    print("=" * 60)

    # ---- Part A: Verify dataset YAML ----
    print(f"\n[Part A] Dataset YAML: {DATA_YAML}")
    try:
        data = yaml.safe_load(open(DATA_YAML))
        path = data.get("path")
        train_dir = data.get("train")
        val_dir = data.get("val")
        num_classes = len(data.get("names", {}))
        print(f"  path={path}")
        print(f"  train={train_dir}, val={val_dir}")
        print(f"  classes={num_classes}")

        train_path = Path(path) / train_dir
        val_path = Path(path) / val_dir
        train_count = len(list(train_path.glob("*.jpg")))
        val_count = len(list(val_path.glob("*.jpg")))
        print(f"  train images: {train_count}")
        print(f"  val images: {val_count}")

        assert train_count > 0, "No training images found"
        assert val_count > 0, "No validation images found"
        print("  ✅ Dataset YAML valid and images found")
    except Exception as e:
        print(f"  ❌ Dataset validation failed: {e}")
        sys.exit(1)

    # ---- Part B: Model + MoE-aware wrapping + forward ----
    print(f"\n[Part B] Model loading: {MODEL_PATH}")
    yolo = YOLO(MODEL_PATH)
    model = yolo.model
    print(f"  ✅ Model loaded")

    cfg = MoLoRAMoEAwareConfig(
        r=8, alpha=16, num_experts=4, top_k=2,
        router_type="linear",
        per_expert_rank=True, rank_allocator_mode="frequency",
        rank_budget_total=32, rank_min=2,
        router_calibration=True, router_calib_rank=4,
        balance_loss_coef=0.01, z_loss_coef=0.001, use_rslora=True,
    )

    print("  Wrapping with MoE-aware layers...")
    wrapped = apply_moe_aware_to_model(model, cfg)
    print(f"  ✅ Wrapped {wrapped} layers")

    # Dummy forward (batch of 2, 3x320x320)
    print(f"\n[Part C] Forward pass (dummy batch) on {DEVICE}...")
    model.to(DEVICE)
    model.eval()
    dummy_input = torch.randn(2, 3, 320, 320).to(DEVICE)
    with torch.no_grad():
        output = model(dummy_input)
    print(f"  ✅ Forward pass OK — output type={type(output).__name__}")

    # Check routing stats
    stats_collected = 0
    for name, m in model.named_modules():
        if hasattr(m, "_last_routing_stats") and m._last_routing_stats is not None:
            stats = m._last_routing_stats
            stats_collected += 1
            if stats_collected == 1:  # Only print first one
                print(f"\n  [Example: {name}]")
                print(f"    effective_k={stats['effective_k']}")
                print(f"    expert_usage={stats['expert_usage'].tolist()}")
                print(f"    calibration_applied={stats.get('calibration_applied', False)}")
                if hasattr(m, "_expert_ranks") and m._expert_ranks:
                    print(f"    expert_ranks={m._expert_ranks}")

    print(f"\n  Total MoE-aware layers with routing stats: {stats_collected}")

    # ---- Part D: Trainability check ----
    print(f"\n[Part D] Trainability check...")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  total params: {total_params:,}")
    print(f"  trainable params: {trainable_params:,}")
    print(f"  trainable %: {trainable_params/total_params*100:.2f}%")

    # ---- Summary ----
    print(f"\n{'='*60}")
    all_ok = (
        train_count > 0 and val_count > 0
        and wrapped > 0
        and stats_collected > 0
        and trainable_params > 0
    )
    if all_ok:
        print("🎉 ALL CHECKS PASSED — MoE-aware PEFT pipeline is ready for COCO2017")
        print(f"\nNext: run actual experiments with:")
        print(f"  cd scripts && python3 ablation_moe_peft_e1_molora_rank.py")
        print(f"  cd scripts && python3 ablation_moe_peft_e2_router_calibration.py")
        print(f"  cd scripts && python3 ablation_moe_peft_e3_expert_load_viz.py")
        print(f"  cd scripts && python3 eval_moe_peft.py")
    else:
        print("❌ Some checks failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
