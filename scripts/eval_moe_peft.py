"""Unified evaluation script for MoE-aware PEFT experiments.

Supports multi-seed, multi-config batch evaluation.
Produces a structured JSON report with statistical aggregation.

Usage:
    python scripts/eval_moe_peft.py --config my_config.json --seeds 3
"""
import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, List, Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_SILENT"] = "true"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ.setdefault("YOLO_AUTOINSTALL", "false")
os.environ.setdefault("YOLO_VERBOSE", "false")

import torch
import numpy as np
from ultralytics.utils import SETTINGS
SETTINGS["wandb"] = False

from ultralytics import YOLO
from ultralytics.nn.peft.molora import MoLoRAMoEAwareConfig, build_moe_aware_layer
from ultralytics.nn.peft.molora.model import _parent_child_name, _get_submodule


HERE = Path(__file__).parent
DEFAULT_MODEL = "YOLO-Master-EsMoE-N.pt"
DEFAULT_DATA = "coco2017.yaml"
DEFAULT_EPOCHS = 3
DEFAULT_BATCH = 8
DEFAULT_IMGSZ = 320
DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda:0" if torch.cuda.is_available() else "cpu")


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


def run_single(config_dict: Dict[str, Any], seed: int, model_path: str, data_yaml: str,
               epochs: int, batch: int, imgsz: int, device: str, project_dir: Path) -> Dict[str, Any]:
    """Run a single training run with a given config and seed."""
    config = MoLoRAMoEAwareConfig(**config_dict)

    t0 = time.time()
    model = YOLO(model_path)
    base_total, base_train = count_params(model.model)

    wrapped = apply_moe_aware_to_model(model.model, config)
    post_total, post_train = count_params(model.model)

    try:
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch,
            imgsz=imgsz,
            device=device,
            project=str(project_dir),
            name=f"eval_seed{seed}",
            exist_ok=True,
            verbose=False,
            workers=2,
            patience=0,
            plots=False,
            save=False,
            seed=seed,
        )
        ok = True
        err = None
    except Exception as e:
        ok = False
        err = f"{type(e).__name__}: {e}"
        results = None
        print(f"[ERROR] seed={seed} | {err}")

    elapsed = time.time() - t0

    final_metrics = {}
    if ok and results is not None and hasattr(results, "results_dict"):
        final_metrics = {k: float(v) for k, v in results.results_dict.items() if isinstance(v, (int, float))}

    return {
        "seed": seed,
        "ok": ok,
        "error": err,
        "elapsed_sec": round(elapsed, 1),
        "params_total": post_total,
        "params_trainable": post_train,
        "trainable_pct": round(post_train / post_total * 100, 4),
        "final_metrics": final_metrics,
        "wrapped_layers": wrapped,
    }


def aggregate_seeds(records: List[Dict[str, Any]], metric_key: str = "metrics/mAP50-95(B)") -> Dict[str, Any]:
    """Aggregate metrics across seeds."""
    values = []
    for r in records:
        if r["ok"] and metric_key in r["final_metrics"]:
            values.append(r["final_metrics"][metric_key])
    if not values:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "n": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description="Unified MoE-aware PEFT evaluation")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to JSON config file with list of config dicts")
    parser.add_argument("--seeds", type=int, default=1, help="Number of random seeds to run")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Model path")
    parser.add_argument("--data", type=str, default=DEFAULT_DATA, help="Dataset yaml")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--output", type=str, default=str(HERE / "eval_moe_peft_results.json"))
    args = parser.parse_args()

    # Default configs if no config file provided
    if args.config and Path(args.config).exists():
        configs = json.loads(Path(args.config).read_text())
        if isinstance(configs, dict):
            configs = [configs]
    else:
        # Built-in benchmark configs
        base = dict(r=8, alpha=16, num_experts=4, top_k=2, router_type="linear",
                    balance_loss_coef=0.01, z_loss_coef=0.001, use_rslora=True)
        configs = [
            {"name": "molora_baseline", **base, "router_calibration": False, "per_expert_rank": False},
            {"name": "molora_calib_r4", **base, "router_calibration": True, "router_calib_rank": 4, "per_expert_rank": False},
            {"name": "molora_freq_rank", **base, "router_calibration": False, "per_expert_rank": True,
             "rank_allocator_mode": "frequency", "rank_budget_total": 32, "rank_min": 2},
        ]

    project_dir = HERE / "runs_eval"
    project_dir.mkdir(exist_ok=True, parents=True)

    all_results = []
    for cfg in configs:
        name = cfg.pop("name", "unnamed")
        print(f"\n{'='*70}\nConfig: {name}\n{'='*70}")
        seeds = list(range(args.seeds))
        seed_records = []
        for seed in seeds:
            rec = run_single(
                cfg.copy(), seed, args.model, args.data,
                args.epochs, args.batch, args.imgsz, args.device, project_dir
            )
            rec["config_name"] = name
            seed_records.append(rec)

        agg = aggregate_seeds(seed_records)
        print(f"[Aggregate] {name}: mean={agg['mean']:.4f} std={agg['std']:.4f} (n={agg['n']})")

        all_results.append({
            "config_name": name,
            "config": cfg,
            "seeds": seed_records,
            "aggregate": agg,
        })

    # Write final report
    output_path = Path(args.output)
    output_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
    print(f"\n[Saved] Final report: {output_path}")

    # Console summary table
    print("\n" + "=" * 100)
    print(f"{'Config':<20} {'Seeds':>5} {'mAP50-95 mean':>14} {'std':>10} {'Trainable':>11} {'%':>7}")
    print("-" * 100)
    for r in all_results:
        agg = r["aggregate"]
        mean_str = f"{agg['mean']:.4f}" if agg["mean"] is not None else "N/A"
        std_str = f"{agg['std']:.4f}" if agg["std"] is not None else "N/A"
        # Take trainable from first successful seed
        trainable = next((s["params_trainable"] for s in r["seeds"] if s["ok"]), 0)
        total = next((s["params_total"] for s in r["seeds"] if s["ok"]), 1)
        pct = trainable / total * 100 if total else 0
        print(f"{r['config_name']:<20} {agg['n']:>5} {mean_str:>14} {std_str:>10} {trainable:>11,} {pct:>7.3f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
