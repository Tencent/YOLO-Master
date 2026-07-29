#!/usr/bin/env python3
"""Evaluate a utility-router bundle on an independent detection utility matrix.

The script verifies checkpoint and layer identities, extracts frozen features,
compares the original and utility-trained routers using true forced-expert loss,
and reports the K distribution implied by candidate adaptive-K thresholds.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_mot_cross_domain import load_model, normalize_torch_device, write_csv
from scripts.build_mot_detection_utility import resolve_mot_layer
from scripts.train_mot_utility_router import (
    evaluate_router,
    extract_feature_cache,
    file_sha256,
    read_utility_matrix,
    routing_metrics,
    subset_tensors,
    verify_matrix_manifest,
)


def adaptive_k_from_probabilities(
    probabilities: torch.Tensor,
    max_k: int,
    threshold: float,
) -> torch.Tensor:
    """Return the minimum K whose cumulative top probability reaches a threshold."""
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [N,E]")
    if not 1 <= max_k <= probabilities.shape[1]:
        raise ValueError("max_k must be in [1, num_experts]")
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0,1]")
    sorted_probabilities = probabilities.sort(dim=1, descending=True).values
    selected_k = (sorted_probabilities.cumsum(dim=1) < threshold).sum(dim=1) + 1
    return selected_k.clamp(min=1, max=max_k)


def threshold_summary(probabilities: torch.Tensor, max_k: int, thresholds: list[float]) -> list[dict]:
    """Summarize expected expert-sample compute for adaptive-K candidates."""
    rows = []
    for threshold in thresholds:
        selected_k = adaptive_k_from_probabilities(probabilities, max_k, threshold)
        mean_k = float(selected_k.float().mean())
        row = {
            "threshold": threshold,
            "mean_k": mean_k,
            "expert_sample_saving_vs_dense": 1.0 - mean_k / probabilities.shape[1],
            "expert_sample_saving_vs_fixed_max_k": 1.0 - mean_k / max_k,
        }
        for k in range(1, max_k + 1):
            row[f"k_{k}_share"] = float((selected_k == k).float().mean())
        rows.append(row)
    return rows


def blend_router_probabilities(
    baseline: torch.Tensor,
    utility: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """Apply a trust-region blend between the original and utility routers."""
    if baseline.shape != utility.shape:
        raise ValueError("baseline and utility probabilities must have the same shape")
    if not 0 <= alpha <= 1:
        raise ValueError("blend alpha must be in [0,1]")
    probabilities = (1.0 - alpha) * baseline + alpha * utility
    return probabilities / probabilities.sum(dim=1, keepdim=True).clamp_min(1e-8)


def mean_router_kl(baseline: torch.Tensor, utility: torch.Tensor) -> float:
    """Measure mean utility-to-baseline KL divergence for drift guarding."""
    if baseline.shape != utility.shape or baseline.ndim != 2:
        raise ValueError("router probabilities must share shape [N,E]")
    kl = utility * (utility.clamp_min(1e-8).log() - baseline.clamp_min(1e-8).log())
    return float(kl.sum(dim=1).mean())


def blend_sweep(
    baseline: torch.Tensor,
    utility: torch.Tensor,
    targets: torch.Tensor,
    forced_losses: torch.Tensor,
    alphas: list[float],
) -> list[dict]:
    """Evaluate candidate trust-region strengths against detection regret."""
    return [
        {
            "alpha": alpha,
            **routing_metrics(
                blend_router_probabilities(baseline, utility, alpha),
                targets,
                forced_losses,
            ),
        }
        for alpha in alphas
    ]


def comparison_rows(
    image_ids: tuple[str, ...],
    sequence_ids: tuple[str, ...],
    baseline_probabilities: torch.Tensor,
    utility_probabilities: torch.Tensor,
    deployment_probabilities: torch.Tensor,
    forced_losses: torch.Tensor,
    expert_names: tuple[str, ...],
) -> list[dict]:
    """Create per-image decisions and detection regret deltas."""
    baseline_selected = baseline_probabilities.argmax(dim=1)
    utility_selected = utility_probabilities.argmax(dim=1)
    deployment_selected = deployment_probabilities.argmax(dim=1)
    oracle = forced_losses.argmin(dim=1)
    oracle_losses = forced_losses.min(dim=1).values
    rows = []
    for index, image_id in enumerate(image_ids):
        baseline_regret = forced_losses[index, baseline_selected[index]] - oracle_losses[index]
        utility_regret = forced_losses[index, utility_selected[index]] - oracle_losses[index]
        deployment_regret = forced_losses[index, deployment_selected[index]] - oracle_losses[index]
        row = {
            "image_id": image_id,
            "sequence_id": sequence_ids[index],
            "oracle_expert": int(oracle[index]),
            "oracle_expert_name": expert_names[int(oracle[index])],
            "baseline_selected_expert": int(baseline_selected[index]),
            "baseline_selected_expert_name": expert_names[int(baseline_selected[index])],
            "utility_selected_expert": int(utility_selected[index]),
            "utility_selected_expert_name": expert_names[int(utility_selected[index])],
            "deployment_selected_expert": int(deployment_selected[index]),
            "deployment_selected_expert_name": expert_names[int(deployment_selected[index])],
            "baseline_regret": float(baseline_regret),
            "utility_regret": float(utility_regret),
            "deployment_regret": float(deployment_regret),
            "regret_reduction": float(baseline_regret - utility_regret),
            "deployment_regret_reduction": float(baseline_regret - deployment_regret),
        }
        for expert_id in range(len(expert_names)):
            row[f"expert_{expert_id}_forced_loss"] = float(forced_losses[index, expert_id])
            row[f"expert_{expert_id}_baseline_probability"] = float(baseline_probabilities[index, expert_id])
            row[f"expert_{expert_id}_utility_probability"] = float(utility_probabilities[index, expert_id])
            row[f"expert_{expert_id}_deployment_probability"] = float(deployment_probabilities[index, expert_id])
        rows.append(row)
    return rows


def parse_thresholds(value: str) -> list[float]:
    """Parse, validate, and deduplicate comma-separated adaptive-K thresholds."""
    try:
        thresholds = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("thresholds must be comma-separated numbers") from error
    if not thresholds or any(not math.isfinite(item) or not 0 < item <= 1 for item in thresholds):
        raise argparse.ArgumentTypeError("all thresholds must be finite and in (0,1]")
    return thresholds


def parse_blend_alphas(value: str) -> list[float]:
    """Parse comma-separated trust-region blend strengths."""
    try:
        alphas = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    except ValueError as error:
        raise argparse.ArgumentTypeError("blend alphas must be comma-separated numbers") from error
    if not alphas or any(not math.isfinite(item) or not 0 <= item <= 1 for item in alphas):
        raise argparse.ArgumentTypeError("all blend alphas must be finite and in [0,1]")
    return alphas


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--router-bundle", type=Path, required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--layer", default=None)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--blend-alpha", type=float, default=1.0)
    parser.add_argument("--max-mean-router-kl", type=float, default=None)
    parser.add_argument(
        "--blend-alpha-sweep",
        type=parse_blend_alphas,
        default=parse_blend_alphas("0,0.2,0.4,0.6,0.8,1"),
    )
    parser.add_argument("--adaptive-k-thresholds", type=parse_thresholds, default=parse_thresholds("0.35,0.4,0.45,0.5"))
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 <= args.blend_alpha <= 1:
        raise SystemExit("--blend-alpha must be in [0,1]")
    if args.max_mean_router_kl is not None and args.max_mean_router_kl <= 0:
        raise SystemExit("--max-mean-router-kl must be positive")
    model_path = args.model.expanduser().resolve()
    data_path = args.data.expanduser().resolve()
    matrix_path = args.matrix.expanduser().resolve()
    bundle_path = args.router_bundle.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    device_name = normalize_torch_device(args.device)
    device = torch.device(device_name)
    model = load_model(model_path, device_name, nc=10)
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=True)
    if bundle.get("format") != "yolo-master-mot-utility-router-v1":
        raise SystemExit("unsupported utility-router bundle format")
    layer_name = args.layer or str(bundle.get("layer_name", ""))
    layer_name, block = resolve_mot_layer(model, layer_name)
    model_sha = file_sha256(model_path)
    if bundle.get("base_checkpoint_sha256") != model_sha:
        raise SystemExit("router bundle was trained for a different base checkpoint")
    if bundle.get("layer_name") != layer_name:
        raise SystemExit("router bundle was trained for a different MoT layer")

    records, expert_names = read_utility_matrix(matrix_path)
    if tuple(bundle.get("expert_names", ())) != expert_names:
        raise SystemExit("router bundle and utility matrix expert order differ")
    verify_matrix_manifest(matrix_path, model_sha, layer_name, args.split)
    model.requires_grad_(False)
    cache = extract_feature_cache(
        model,
        block,
        records,
        data_path,
        args.split,
        device=device,
        imgsz=args.imgsz,
        workers=args.workers,
    )
    tensors = subset_tensors(cache, list(range(len(records))), device)
    baseline_metrics, baseline_probabilities = evaluate_router(block.router, tensors, args.batch_size)
    utility_config = bundle.get("config", {})
    if utility_config.get("scene_aware") and block.router.scene_projector is None:
        block.router.enable_scene_aware(int(utility_config.get("scene_hidden_dim") or 12))
    if utility_config.get("global_utility_head") and getattr(block.router, "utility_projector", None) is None:
        block.router.enable_global_utility_head(int(utility_config.get("utility_hidden_dim") or 16))
    block.router.load_state_dict(bundle["router_state_dict"], strict=True)
    utility_metrics, utility_probabilities = evaluate_router(block.router, tensors, args.batch_size)
    forced_losses = tensors[3].cpu()
    targets = tensors[1].cpu()
    router_kl = mean_router_kl(baseline_probabilities, utility_probabilities)
    drift_guard_triggered = bool(args.max_mean_router_kl is not None and router_kl > args.max_mean_router_kl)
    effective_alpha = 0.0 if drift_guard_triggered else args.blend_alpha
    deployment_probabilities = blend_router_probabilities(
        baseline_probabilities,
        utility_probabilities,
        effective_alpha,
    )
    deployment_metrics = routing_metrics(deployment_probabilities, targets, forced_losses)
    alpha_sweep = blend_sweep(
        baseline_probabilities,
        utility_probabilities,
        targets,
        forced_losses,
        args.blend_alpha_sweep,
    )
    rows = comparison_rows(
        cache.image_ids,
        cache.sequence_ids,
        baseline_probabilities,
        utility_probabilities,
        deployment_probabilities,
        forced_losses,
        expert_names,
    )
    thresholds = threshold_summary(deployment_probabilities, block.top_k, args.adaptive_k_thresholds)
    report = {
        "protocol": "independent-split utility-router evaluation",
        "model_sha256": model_sha,
        "matrix_sha256": file_sha256(matrix_path),
        "router_bundle_sha256": file_sha256(bundle_path),
        "split": args.split,
        "layer": layer_name,
        "images": len(records),
        "sequences": len(set(cache.sequence_ids)),
        "baseline": baseline_metrics,
        "raw_utility_router": utility_metrics,
        "requested_blend_alpha": args.blend_alpha,
        "effective_blend_alpha": effective_alpha,
        "mean_utility_to_baseline_kl": router_kl,
        "max_mean_router_kl": args.max_mean_router_kl,
        "drift_guard_triggered": drift_guard_triggered,
        "deployment_router": deployment_metrics,
        "blend_alpha_sweep": alpha_sweep,
        "mean_regret_reduction": baseline_metrics["mean_regret"] - deployment_metrics["mean_regret"],
        "relative_regret_reduction": (
            1.0 - deployment_metrics["mean_regret"] / baseline_metrics["mean_regret"]
            if baseline_metrics["mean_regret"] > 0
            else 0.0
        ),
        "adaptive_k_threshold_sweep": thresholds,
        "note": "K sweep reports compute selection only; detection metrics require end-to-end adaptive-K inference.",
    }
    write_csv(output / "router_comparison.csv", rows)
    write_csv(output / "blend_alpha_sweep.csv", alpha_sweep)
    write_csv(output / "adaptive_k_threshold_sweep.csv", thresholds)
    (output / "evaluation_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"[utility-eval] images={len(records)} "
        f"oracle_accuracy={baseline_metrics['oracle_accuracy']:.3f}->{deployment_metrics['oracle_accuracy']:.3f} "
        f"regret={baseline_metrics['mean_regret']:.6f}->{deployment_metrics['mean_regret']:.6f} "
        f"alpha={effective_alpha:.2f} drift_guard={drift_guard_triggered}"
    )
    print(f"[utility-eval] wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
