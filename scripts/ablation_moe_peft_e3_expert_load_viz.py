"""E3: Expert load visualization (activation frequency distribution + rank allocation).

Generates:
  - Bar chart of expert activation frequencies
  - Bar chart of per-expert rank allocations (if frequency-based)
  - Gini coefficient of load imbalance
  - Saves to local PNG (no WandB dependency)

Usage:
    python scripts/ablation_moe_peft_e3_expert_load_viz.py
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

import json
import math

import torch
import torch.nn as nn
from ultralytics.utils import SETTINGS
SETTINGS["wandb"] = False

from ultralytics import YOLO
from ultralytics.nn.peft.molora import (
    MoLoRAMoEAwareConfig,
    PerExpertRankAllocator,
    MoLoRAMoEAwareLayer,
    build_moe_aware_layer,
)
from ultralytics.nn.peft.molora.model import _parent_child_name, _get_submodule


HERE = Path(__file__).parent
MODEL_PATH = "/Users/gatilin/PycharmProjects/YOLO-Master-v0708/YOLO-Master-EsMoE-N.pt"
DATA_YAML = "coco2017.yaml"
OUTPUT_DIR = HERE / "e3_viz_outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

IMGSZ = 320
BATCH = 16
MODEL_PATH = "/Users/gatilin/PycharmProjects/YOLO-Master-v0708/YOLO-Master-EsMoE-N.pt"
DATA_YAML = "coco2017.yaml"
OUTPUT_DIR = HERE / "e3_viz_outputs"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

IMGSZ = 320
BATCH = 32
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")


def gini_coefficient(x: torch.Tensor) -> float:
    """Compute Gini coefficient of a 1-D tensor (0 = perfectly equal, 1 = maximally unequal)."""
    x = x.float().flatten()
    x = torch.sort(x)[0]
    n = x.numel()
    if n == 0 or x.sum() == 0:
        return 0.0
    index = torch.arange(1, n + 1, dtype=x.dtype, device=x.device)
    return (2.0 * torch.sum(index * x) / (n * x.sum()) - (n + 1.0) / n).item()


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
        if not isinstance(base_layer, (nn.Conv2d, nn.Linear)):
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


def collect_routing_stats(model, data_yaml, imgsz, batch, device):
    """Run one mini-batch forward and collect routing stats from all MoLoRA layers."""
    yolo = YOLO(MODEL_PATH)
    # Replace model
    yolo.model = model
    # Bypass fuse() so wrapped layers are not probed for Conv2d attributes
    model.fuse = lambda verbose=False: model
    # Do a quick validation forward to trigger routing
    try:
        yolo.val(data=data_yaml, imgsz=imgsz, batch=batch, device=device, verbose=False)
    except Exception:
        pass

    all_stats = []
    for name, m in model.named_modules():
        if isinstance(m, MoLoRAMoEAwareLayer) and m._last_routing_stats is not None:
            stats = m._last_routing_stats
            all_stats.append({
                "layer": name,
                "expert_usage": stats["expert_usage"].cpu().tolist(),
                "effective_k": stats["effective_k"],
                "calibration_applied": stats.get("calibration_applied", False),
                "expert_ranks": stats.get("expert_ranks"),
            })
    return all_stats


def main():
    print(f"Device: {DEVICE}")

    # Build two variants: uniform and frequency
    NUM_EXPERTS = 4
    BUDGET = 32
    MIN_RANK = 2

    configs = {
        "uniform": MoLoRAMoEAwareConfig(
            r=8, alpha=16, num_experts=NUM_EXPERTS, top_k=2,
            router_type="linear", per_expert_rank=True,
            rank_allocator_mode="uniform", rank_budget_total=BUDGET, rank_min=MIN_RANK,
            balance_loss_coef=0.01, z_loss_coef=0.001, use_rslora=True,
        ),
        "frequency": MoLoRAMoEAwareConfig(
            r=8, alpha=16, num_experts=NUM_EXPERTS, top_k=2,
            router_type="linear", per_expert_rank=True,
            rank_allocator_mode="frequency", rank_budget_total=BUDGET, rank_min=MIN_RANK,
            balance_loss_coef=0.01, z_loss_coef=0.001, use_rslora=True,
        ),
    }

    # Also test rank allocator standalone
    allocator = PerExpertRankAllocator(
        num_experts=NUM_EXPERTS, total_budget=BUDGET, min_rank=MIN_RANK, mode="frequency"
    )
    # Simulate skewed usage: expert 0 gets 50%, others share the rest
    skewed_history = torch.tensor([0.50, 0.20, 0.20, 0.10])
    allocated_ranks = allocator.allocate(skewed_history)
    print(f"\n[Allocator test] skewed_history={skewed_history.tolist()}")
    print(f"[Allocator test] allocated_ranks={allocated_ranks}")
    print(f"[Allocator test] sum={sum(allocated_ranks)} (expected {BUDGET})")

    # Validate uniform allocator
    uniform_allocator = PerExpertRankAllocator(
        num_experts=NUM_EXPERTS, total_budget=BUDGET, min_rank=MIN_RANK, mode="uniform"
    )
    uniform_ranks = uniform_allocator.allocate(skewed_history)
    print(f"[Allocator test] uniform_ranks={uniform_ranks}")

    summary = {}
    for variant_name, cfg in configs.items():
        print(f"\n{'='*60}\nVariant: {variant_name}\n{'='*60}")
        model = YOLO(MODEL_PATH).model
        apply_moe_aware_to_model(model, cfg)

        stats = collect_routing_stats(model, DATA_YAML, IMGSZ, BATCH, DEVICE)
        if not stats:
            print("[WARN] No routing stats collected. Skipping visualization.")
            continue

        # Aggregate usage across all layers
        agg_usage = torch.zeros(NUM_EXPERTS)
        for s in stats:
            agg_usage += torch.tensor(s["expert_usage"])
        agg_usage = agg_usage / len(stats)

        gini = gini_coefficient(agg_usage)
        print(f"[Aggregate usage] {agg_usage.tolist()}")
        print(f"[Gini coefficient] {gini:.4f}")

        # If frequency mode, show rank allocation based on this usage
        if variant_name == "frequency":
            freq_ranks = allocator.allocate(agg_usage)
            print(f"[Frequency-based ranks] {freq_ranks}")
        else:
            freq_ranks = uniform_ranks

        summary[variant_name] = {
            "num_layers": len(stats),
            "avg_usage": agg_usage.tolist(),
            "gini": gini,
            "ranks": freq_ranks,
        }

    # Save summary JSON
    summary_path = OUTPUT_DIR / "e3_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[Saved] Summary JSON: {summary_path}")

    # Try matplotlib visualization if available
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for variant_name, data in summary.items():
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))

            # Usage plot
            ax = axes[0]
            labels = [f"E{i}" for i in range(NUM_EXPERTS)]
            ax.bar(labels, data["avg_usage"], color="steelblue")
            ax.set_title(f"{variant_name}: Avg Expert Usage")
            ax.set_ylabel("Frequency")
            ax.set_ylim(0, max(data["avg_usage"]) * 1.3)
            ax.text(0.5, 0.95, f"Gini={data['gini']:.3f}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

            # Rank plot
            ax = axes[1]
            ax.bar(labels, data["ranks"], color="coral")
            ax.set_title(f"{variant_name}: Per-Expert Rank")
            ax.set_ylabel("Rank")
            ax.set_ylim(0, max(data["ranks"]) * 1.3)

            plt.tight_layout()
            png_path = OUTPUT_DIR / f"e3_{variant_name}.png"
            plt.savefig(png_path, dpi=150)
            plt.close(fig)
            print(f"[Saved] Visualization: {png_path}")

    except ImportError:
        print("[INFO] matplotlib not available; skipped PNG generation.")

    print("\n[E3] Done.")


if __name__ == "__main__":
    main()
