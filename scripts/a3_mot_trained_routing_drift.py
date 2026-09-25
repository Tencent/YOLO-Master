#!/usr/bin/env python3
"""Stage 0: measure routing drift on a genuinely trained MoT checkpoint.

This is a hard precondition for the five-family precision harness. It never
injects or randomizes router parameters. A zero/non-finite terminal router
projection fails the gate before any image is evaluated.
"""

from __future__ import annotations

import argparse
import csv
import platform
import sys
from collections import defaultdict
from pathlib import Path

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.a3_precision.common import collect_images, file_set_digest, json_dump, letterbox_image, sha256_file
from scripts.a3_precision.routing import (
    RouteCapture,
    compare_route_tensors,
    simulated_precision,
    weighted_layer_summary,
)
from ultralytics import YOLO


def _trained_router_audit(layers: dict[str, torch.nn.Module]) -> dict:
    """Audit the last router projection, which is explicitly zero-initialized by MoT."""
    records = {}
    failures = []
    for name, module in layers.items():
        router = module if type(module).__name__ == "_MoTRouter" else getattr(module, "router")
        network = getattr(router, "router", None)
        terminal = network[-1] if isinstance(network, torch.nn.Sequential) and len(network) else None
        parameters = list(terminal.parameters()) if isinstance(terminal, torch.nn.Module) else list(router.parameters())
        flattened = torch.cat([parameter.detach().float().cpu().reshape(-1) for parameter in parameters])
        finite = bool(torch.isfinite(flattened).all())
        nonzero = int(torch.count_nonzero(flattened))
        stats = {
            "module_type": type(module).__name__,
            "router_type": type(router).__name__,
            "terminal_type": type(terminal).__name__ if terminal is not None else None,
            "parameters": int(flattened.numel()),
            "nonzero_parameters": nonzero,
            "finite": finite,
            "mean": float(flattened.mean()),
            "std": float(flattened.std(unbiased=False)),
            "l2_norm": float(torch.linalg.vector_norm(flattened)),
            "passed": finite and nonzero > 0 and float(torch.linalg.vector_norm(flattened)) > 1e-10,
        }
        records[name] = stats
        if not stats["passed"]:
            failures.append(name)
    return {"passed": not failures, "failed_layers": failures, "layers": records}


def _checkpoint_training_metadata(yolo: YOLO, checkpoint: Path) -> dict:
    ckpt = getattr(yolo, "ckpt", None)
    if not isinstance(ckpt, dict):
        return {"checkpoint_suffix": checkpoint.suffix, "metadata_available": False}
    train_args = ckpt.get("train_args") if isinstance(ckpt.get("train_args"), dict) else {}
    return {
        "checkpoint_suffix": checkpoint.suffix,
        "metadata_available": True,
        "epoch": ckpt.get("epoch"),
        "best_fitness": float(ckpt["best_fitness"]) if isinstance(ckpt.get("best_fitness"), (int, float)) else None,
        "train_epochs": train_args.get("epochs"),
        "train_data": train_args.get("data"),
        "train_imgsz": train_args.get("imgsz"),
        "train_seed": train_args.get("seed"),
    }


def run(args: argparse.Namespace) -> Path:
    checkpoint = args.model.expanduser().resolve()
    if checkpoint.suffix.lower() != ".pt" or not checkpoint.is_file():
        raise SystemExit("--model must point to an existing trained .pt checkpoint")
    images = collect_images(args.source, limit=args.limit, seed=args.seed)
    if not images:
        raise SystemExit("no validation images found")
    requested_device = str(args.device)
    if requested_device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable; pass --device cpu explicitly for a CPU probe")
    device = torch.device(requested_device)

    torch.manual_seed(args.seed)
    yolo = YOLO(str(checkpoint))
    training = _checkpoint_training_metadata(yolo, checkpoint)
    trained_checkpoint_gate = bool(
        training.get("metadata_available")
        and isinstance(training.get("train_epochs"), (int, float))
        and int(training["train_epochs"]) > 0
    )
    if not trained_checkpoint_gate:
        raise SystemExit(
            "trained-checkpoint gate failed: checkpoint must contain positive train_args.epochs metadata; "
            "random/untrained or metadata-free .pt files are not accepted"
        )
    model = yolo.model.eval().to(device).float()
    capture = RouteCapture(model, family="mot")
    audit = _trained_router_audit(capture.layers)
    if not audit["passed"]:
        failed = ", ".join(audit["failed_layers"])
        raise SystemExit(
            f"trained-router gate failed for {failed}; zero/random router injection is forbidden. "
            "Use a genuinely trained MoT checkpoint."
        )

    output_dir = args.out.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline: dict[str, dict[str, object]] = {}
    with RouteCapture(model, family="mot") as collector:
        with torch.inference_mode():
            for image_path in images:
                collector.clear()
                array = letterbox_image(image_path, args.imgsz)
                tensor = torch.from_numpy(array).to(device)
                model(tensor)
                missing = sorted(set(collector.layers) - set(collector.current))
                if missing:
                    raise RuntimeError(f"routers did not execute for {image_path.name}: {missing}")
                baseline[str(image_path)] = dict(collector.current)

    per_sample: list[dict] = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for precision in ("fp16", "int8"):
        with RouteCapture(model, family="mot") as collector:
            with simulated_precision(
                model,
                precision,
                scope=args.scope,
                routed=collector.layers,
                per_channel=args.per_channel,
            ):
                with torch.inference_mode():
                    for sample_index, image_path in enumerate(images):
                        collector.clear()
                        tensor = torch.from_numpy(letterbox_image(image_path, args.imgsz)).to(device)
                        model(tensor)
                        for layer_name, reference in baseline[str(image_path)].items():
                            metrics = compare_route_tensors(reference, collector.current[layer_name])
                            record = {
                                "sample_index": sample_index,
                                "image": str(image_path),
                                "layer": layer_name,
                                "precision": precision,
                                **metrics,
                            }
                            per_sample.append(record)
                            grouped[(precision, layer_name)].append(record)

    layers = {
        precision: {
            layer_name: weighted_layer_summary(grouped[(precision, layer_name)]) for layer_name in capture.layers
        }
        for precision in ("fp16", "int8")
    }
    summary = {
        "schema_version": 2,
        "stage": "trained_mot_routing_drift",
        "status": "passed",
        "methodology": {
            "checkpoint_required": "trained .pt; terminal router projections must be finite and non-zero",
            "random_router_injection": False,
            "precision_simulation": "weight quantize-dequantize followed by FP32 kernels",
            "scope": args.scope,
            "int8_granularity": "per-output-channel" if args.per_channel else "per-tensor",
            "preprocess": "aspect-ratio letterbox, RGB, [0,1], NCHW",
        },
        "model": {
            "path": str(checkpoint),
            "sha256": sha256_file(checkpoint),
            "size_bytes": checkpoint.stat().st_size,
            "training": training,
            "trained_checkpoint_gate": trained_checkpoint_gate,
        },
        "router_gate": audit,
        "dataset": {
            "source": str(args.source.expanduser().resolve()),
            "images": len(images),
            "imgsz": args.imgsz,
            "selection_digest": file_set_digest(images, args.source.expanduser().resolve()),
            "first_image": str(images[0]),
            "last_image": str(images[-1]),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda": torch.version.cuda,
        },
        "layers": layers,
    }
    summary_path = json_dump(output_dir / "trained_mot_routing_drift.json", summary)
    fields = list(per_sample[0]) if per_sample else []
    with (output_dir / "trained_mot_routing_per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(per_sample)
    json_dump(output_dir / "trained_mot_router_gate.json", audit)
    return summary_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="genuinely trained MoT .pt checkpoint")
    parser.add_argument("--source", type=Path, required=True, help="validation image directory, image, or txt list")
    parser.add_argument("--limit", type=int, default=548)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scope", choices=("router_only", "full_model_weight"), default="router_only")
    parser.add_argument("--per-channel", dest="per_channel", action="store_true")
    parser.add_argument("--no-per-channel", dest="per_channel", action="store_false")
    parser.set_defaults(per_channel=True)
    parser.add_argument("--out", type=Path, default=Path("runs/a3_precision/stage0_trained_mot_drift"))
    return parser.parse_args()


if __name__ == "__main__":
    print(run(parse_args()))
