#!/usr/bin/env python3
"""MoT Routing Interpretability Analysis for 犀牛鸟 #54.

Analyzes expert routing distributions across MoTBlocks, generating:
- Per-block expert activation heatmaps
- Activation statistics by scenario (dense/sparse, small/large objects)
- Scene-conditioned routing patterns
"""

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
from ultralytics import YOLO
import json
import argparse


EXPERT_NAMES = ["LocalConv", "Window", "Deformable"]


def collect_routing_hooks(model):
    """Register forward hooks on all MoTBlocks to collect routing weights."""
    from ultralytics.nn.modules.mot import MoTBlock
    routing_data = defaultdict(list)

    def make_hook(layer_name):
        def hook(module, input, output):
            if hasattr(module, "last_routing_snapshot"):
                snap = module.last_routing_snapshot
                expert_usage = snap.get("expert_usage", None)
                if expert_usage is not None:
                    routing_data[layer_name].append({
                        "expert_usage": expert_usage.cpu().tolist() if isinstance(expert_usage, torch.Tensor) else expert_usage,
                        "aux_loss": snap.get("aux_loss", 0),
                    })
            # Also capture raw router weights from the forward
            if hasattr(module, "router") and hasattr(module.router, "_last_weights"):
                weights = module.router._last_weights
                routing_data[f"{layer_name}_spatial"].append(
                    weights.detach().cpu().float().mean(dim=(2, 3)).tolist()
                )
        return hook

    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, MoTBlock):
            h = module.register_forward_hook(make_hook(name))
            hooks.append(h)
    return routing_data, hooks


def analyze_scenes(model, data_yaml, device, num_samples=50):
    """Run inference on val images and collect routing statistics."""
    from ultralytics.data import build_dataloader
    from ultralytics.utils import yaml_load

    data_dict = yaml_load(data_yaml)
    routing_data = defaultdict(list)
    scene_metadata = []

    # Build dataloader for validation
    from ultralytics.data import YOLODataset
    from torch.utils.data import DataLoader

    val_path = Path(data_dict["path"]) / "images" / "val"
    if not val_path.exists():
        # Try alternative path
        val_path = Path(data_dict["path"]) / data_dict.get("val", "images/val")

    if isinstance(val_path, str) and not Path(val_path).exists():
        # Use the dataset path directly
        val_path = Path(data_dict["path"]) / "images" / "val"

    images = sorted(Path(val_path).glob("*.jpg")) if val_path.exists() else []
    if not images:
        # fallback: try parent directory
        images = sorted(Path("/home/u2120250644/zzq/hanhaoran/datasets/VisDrone/images/val").glob("*.jpg"))

    if not images:
        print("[WARN] No validation images found, using synthetic analysis")
        return routing_data, scene_metadata

    # Limit samples
    import random
    random.seed(42)
    if len(images) > num_samples:
        images = random.sample(images, num_samples)

    # Register hooks
    from ultralytics.nn.modules.mot import MoTBlock
    hooks = []

    def make_hook(name):
        def hook_fn(module, input, output):
            if hasattr(module, "last_routing_snapshot"):
                snap = module.last_routing_snapshot
                eu = snap.get("expert_usage")
                if eu is not None:
                    routing_data[name].append(eu.cpu().tolist())
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, MoTBlock):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    scene_stats = []

    for i, img_path in enumerate(images):
        try:
            results = model(str(img_path), device=device, verbose=False)
            # Count objects to classify scene
            num_objects = len(results[0].boxes) if results[0].boxes is not None else 0

            # Get object sizes from bounding boxes
            if num_objects > 0:
                boxes = results[0].boxes.xywh
                img_area = results[0].orig_shape[0] * results[0].orig_shape[1]
                obj_areas = boxes[:, 2] * boxes[:, 3]
                rel_areas = obj_areas / img_area
                avg_obj_size = rel_areas.mean().item()
                small_obj_ratio = (rel_areas < 0.01).float().mean().item()
            else:
                avg_obj_size = 0
                small_obj_ratio = 0

            scene_stats.append({
                "image": img_path.name,
                "num_objects": num_objects,
                "density": "dense" if num_objects > 30 else ("sparse" if num_objects < 10 else "medium"),
                "avg_obj_size": avg_obj_size,
                "small_obj_ratio": small_obj_ratio,
            })
        except Exception as e:
            print(f"  [WARN] {img_path.name}: {e}")

    # Clean up
    for h in hooks:
        h.remove()

    return dict(routing_data), scene_stats


def summarize_routing(routing_data, scene_stats):
    """Generate routing statistics summary."""
    summary = {}

    for layer_name, activations in routing_data.items():
        if not activations:
            continue
        arr = np.array(activations)  # [N_images, E]
        summary[layer_name] = {
            "mean": arr.mean(axis=0).tolist(),
            "std": arr.std(axis=0).tolist(),
            "expert_names": EXPERT_NAMES[: arr.shape[1]],
            "num_samples": len(activations),
        }

    # Scene-conditioned analysis
    scene_groups = defaultdict(list)
    for i, stat in enumerate(scene_stats):
        scene_groups[stat["density"]].append(i)
        size_key = "small" if stat["small_obj_ratio"] > 0.5 else "large" if stat["avg_obj_size"] > 0.05 else "mixed"
        scene_groups[f"size_{size_key}"].append(i)

    scene_analysis = {}
    for group_key, indices in scene_groups.items():
        if len(indices) < 2:
            continue
        scene_analysis[group_key] = {}
        for layer_name, activations in routing_data.items():
            if not activations:
                continue
            arr = np.array(activations)
            grouped = arr[indices]
            scene_analysis[group_key][layer_name] = {
                "mean": grouped.mean(axis=0).tolist(),
                "std": grouped.std(axis=0).tolist(),
                "num_samples": len(indices),
            }

    return summary, scene_analysis


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/mot_ablation/v10_mot/weights/best.pt")
    parser.add_argument("--data", default="ultralytics/cfg/datasets/VisDrone.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--output", default="runs/mot_ablation/routing_analysis.json")
    args = parser.parse_args()

    print(f"[1/4] Loading model: {args.model}")
    model = YOLO(args.model)
    model.model.eval()

    print(f"[2/4] Running inference on {args.num_samples} VisDrone val images...")
    routing_data, scene_stats = analyze_scenes(
        model.model, args.data, args.device, args.num_samples
    )

    print(f"[3/4] Summarizing routing patterns...")
    summary, scene_analysis = summarize_routing(routing_data, scene_stats)

    # Print summary
    print("\n" + "=" * 70)
    print("MoT EXPERT ROUTING ANALYSIS")
    print("=" * 70)
    print(f"\n{'Layer':<50} {'LocalConv':>10} {'Window':>10} {'Deformable':>10}")
    print("-" * 80)
    for layer_name, stats in summary.items():
        means = stats["mean"]
        print(f"{layer_name:<50} {means[0]:>10.4f} {means[1]:>10.4f} {means[2]:>10.4f}")

    print("\n--- Scene-Conditioned Analysis ---")
    for group, layers in scene_analysis.items():
        print(f"\n[{group}] ({layers.get(list(layers.keys())[0] if layers else '', {}).get('num_samples', 0)} samples)")
        for layer_name, stats in layers.items():
            means = stats["mean"]
            print(f"  {layer_name:<48} {means[0]:>10.4f} {means[1]:>10.4f} {means[2]:>10.4f}")

    # Save detailed results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "model": args.model,
        "per_layer_summary": {
            k: {kk: vv for kk, vv in v.items()}
            for k, v in summary.items()
        },
        "scene_analysis": {
            group: {
                layer: {k: v for k, v in stats.items()}
                for layer, stats in layers.items()
            }
            for group, layers in scene_analysis.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[4/4] Results saved to {output_path}")

    # Generate scene-based insights
    print("\n" + "=" * 70)
    print("SCENE-BASED INSIGHTS")
    print("=" * 70)

    for group, layers in scene_analysis.items():
        for layer_name, stats in layers.items():
            means = np.array(stats["mean"])
            top_expert = np.argmax(means)
            print(f"  {group:20s} | {layer_name:40s} | top_expert={EXPERT_NAMES[top_expert]:15s} | weights={means.round(4).tolist()}")


if __name__ == "__main__":
    main()
