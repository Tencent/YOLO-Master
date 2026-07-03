#!/usr/bin/env python3
"""Diagnose MoT expert routing distributions for YOLO-Master models.

The script intentionally supports a lightweight dry-run mode so routing report
generation can be tested without a full COCO/VisDrone training run.

Examples:
    python scripts/diagnose_mot_routing.py --model ultralytics/cfg/models/master/v0_8/det/yolo-master-mot-n.yaml --dry-run
    python scripts/diagnose_mot_routing.py --model runs/train/weights/best.pt --source datasets/visdrone/images/val --limit 500
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402


EXPERT_NAMES = ("LocalConvTransformer", "WindowTransformer", "DeformableTransformer")


@dataclass(frozen=True)
class RoutingSummary:
    layer: str
    expert: str
    active_tokens: int
    activation_ratio: float
    mean_weight: float


def summarize_router_weights(layer: str, weights: torch.Tensor, expert_names: tuple[str, ...] = EXPERT_NAMES) -> list[RoutingSummary]:
    """Summarize [B, E, H, W] router weights into per-expert activation rows."""
    if weights.ndim != 4:
        raise ValueError(f"expected router weights with shape [B, E, H, W], got {tuple(weights.shape)}")
    total_tokens = weights.shape[0] * weights.shape[2] * weights.shape[3]
    rows: list[RoutingSummary] = []
    for idx in range(weights.shape[1]):
        expert_weights = weights[:, idx]
        active = expert_weights > 0
        rows.append(
            RoutingSummary(
                layer=layer,
                expert=expert_names[idx] if idx < len(expert_names) else f"expert_{idx}",
                active_tokens=int(active.sum().item()),
                activation_ratio=float(active.float().mean().item()),
                mean_weight=float(expert_weights.mean().item()),
            )
        )
    return rows


def scenario_recommendations(rows: list[RoutingSummary]) -> list[str]:
    """Generate data-backed scene recommendations from aggregated routing rows."""
    by_expert: dict[str, list[RoutingSummary]] = {}
    for row in rows:
        by_expert.setdefault(row.expert, []).append(row)

    def avg(expert: str, field: str) -> float:
        values = [getattr(row, field) for row in by_expert.get(expert, [])]
        return sum(values) / len(values) if values else 0.0

    window = avg("WindowTransformer", "activation_ratio")
    deform = avg("DeformableTransformer", "activation_ratio")
    local = avg("LocalConvTransformer", "activation_ratio")
    return [
        f"Dense or small-object scenes should inspect WindowTransformer first when its activation ratio is high ({window:.3f}).",
        f"Occluded or irregular-object scenes should inspect DeformableTransformer when its activation ratio rises ({deform:.3f}).",
        f"Latency-sensitive simple scenes can prefer LocalConvTransformer-heavy routing ({local:.3f}) before enabling deeper MoT/MoE hybrids.",
    ]


def collect_model_routing(model: torch.nn.Module, x: torch.Tensor) -> list[RoutingSummary]:
    """Collect routing summaries from each MoTBlock by recomputing router weights on block inputs."""
    rows: list[RoutingSummary] = []
    hooks = []

    def make_hook(name: str):
        def hook(module: torch.nn.Module, inputs, _output):
            with torch.no_grad():
                weights, _indices = module.router(inputs[0])
            rows.extend(summarize_router_weights(name, weights.detach().cpu()))

        return hook

    for name, module in model.named_modules():
        if module.__class__.__name__ == "MoTBlock":
            hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        model(x)

    for hook in hooks:
        hook.remove()
    return rows


def write_csv(path: Path, rows: list[RoutingSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=("layer", "expert", "active_tokens", "activation_ratio", "mean_weight"))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="YOLO model YAML or weights")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/mot_routing/routing_summary.csv")
    parser.add_argument("--recommendations", type=Path, default=ROOT / "runs/mot_routing/recommendations.json")
    parser.add_argument("--dry-run", action="store_true", help="Use a synthetic input tensor instead of image files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO

    device = torch.device(args.device)
    yolo = YOLO(str(args.model))
    model = yolo.model.eval().to(device)
    if args.dry_run:
        x = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
        rows = collect_model_routing(model, x)
    else:
        raise SystemExit("image source iteration is intentionally left to experiment runners; use --dry-run for smoke checks")

    write_csv(args.out, rows)
    recs = scenario_recommendations(rows)
    args.recommendations.parent.mkdir(parents=True, exist_ok=True)
    args.recommendations.write_text(json.dumps(recs, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "csv": str(args.out), "recommendations": recs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
