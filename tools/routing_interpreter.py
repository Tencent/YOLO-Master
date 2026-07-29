"""Generate routing diagnostics and heatmaps from a YOLO-Master checkpoint.

Two modes are supported:

* **Single-image mode** (default) — pass a checkpoint and an image path.  Renders
  router confidence / assignment maps and per-expert output-feature heatmaps,
  and computes layer summaries and collapse checks.

* **Dataset mode** — pass ``--dataset VisDrone.yaml`` instead of an image.  Runs
  dataset-level sparse top‑k hit statistics, router differentiation metrics
  (KL divergence, weight spread), and collapse reports.  Results are written as
  ``dataset_routing_report.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="YOLO checkpoint or model YAML")
    parser.add_argument(
        "image", type=Path, nargs="?", default=None,
        help="input image (required unless --dataset is used)",
    )
    parser.add_argument(
        "--dataset",
        help="dataset YAML path for batch analysis (replaces single-image mode)",
    )
    parser.add_argument(
        "--layer",
        help="exact routed layer name; omit to capture all leaf routed layers",
    )
    parser.add_argument(
        "--expert", type=int,
        help="also run a forced-expert counterfactual for --layer",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640, help="square inference size",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="torch device, for example cpu, mps, or cuda:0",
    )
    parser.add_argument(
        "--half", action="store_true",
        help="use float16 inference (CUDA recommended)",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("runs/routing_interpreter"),
        help="output directory",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="max images to process in dataset mode (default: all)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="batch size for dataset mode dataloader",
    )
    return parser


def _load_batch(
    image_path: Path, imgsz: int, device: torch.device, dtype: torch.dtype,
) -> torch.Tensor:
    """Load one image using the same letterbox and channel conventions as prediction."""
    import numpy as np

    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.patches import imread

    image = imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"could not read image: {image_path}")
    resized = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(
        image=image
    )
    rgb = np.ascontiguousarray(resized[..., ::-1].transpose(2, 0, 1))
    return (
        torch.from_numpy(rgb).unsqueeze(0).to(device=device, dtype=dtype).div_(255.0)
    )


def _load_model(
    model_path: Path, device: torch.device, half: bool,
) -> torch.nn.Module:
    """Load detection YAMLs and checkpoints without importing unrelated model families."""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.patches import torch_load

    suffix = model_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        model = DetectionModel(str(model_path), ch=3, verbose=False)
    elif suffix in {".pt", ".pth"}:
        checkpoint = torch_load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict):
            model = checkpoint.get("ema") or checkpoint.get("model")
        else:
            model = checkpoint
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                f"checkpoint does not contain an nn.Module under 'ema' or "
                f"'model': {model_path}"
            )
    else:
        raise ValueError(
            f"model must be a .pt, .pth, .yaml, or .yml file, got: {model_path}"
        )
    model = model.to(device).eval()
    for module in model.modules():
        if hasattr(module, "inplace"):
            module.inplace = True
        elif isinstance(module, torch.nn.Upsample) and not hasattr(
            module, "recompute_scale_factor"
        ):
            module.recompute_scale_factor = None
    return model.half() if half else model.float()


def _run_single_image(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> int:
    """Single-image mode: expert feature heatmaps + collapse report."""
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    network = _load_model(args.model, device, args.half)
    batch = _load_batch(args.image, args.imgsz, device, dtype)

    interpreter = RoutingInterpreter(network)
    heatmaps, expert_features = interpreter.capture_routing_and_expert_features(
        batch, layer_name=args.layer,
    )
    visualizations = interpreter.save_routing_visualizations(
        heatmaps, args.output, input_image=batch,
        expert_features=expert_features,
    )
    summaries = interpreter.collect_layer_summaries(heatmaps=heatmaps)
    collapse = interpreter.detect_routing_collapse(heatmaps=heatmaps)
    causal = (
        interpreter.routing_causal_analysis(
            batch, args.layer, args.expert,
        ).to_dict()
        if args.expert is not None
        else None
    )

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "routing_report.json"
    payload = {
        "model": str(args.model),
        "image": str(args.image),
        "layer": args.layer,
        "heatmaps": {name: h.to_dict() for name, h in heatmaps.items()},
        "expert_features": {
            name: feat.to_dict()
            for name, feat in expert_features.items()
        },
        "visualizations": {
            name: {artifact: str(path) for artifact, path in artifacts.items()}
            for name, artifacts in visualizations.items()
        },
        "summaries": [s.to_dict() for s in summaries],
        "collapse": {n: r.to_dict() for n, r in collapse.items()},
        "causal": causal,
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8",
    )
    print(f"Routing report: {report_path}")
    print(f"Heatmaps: {len(heatmaps)}")
    return 0


def _run_dataset_analysis(
    args: argparse.Namespace,
    device: torch.device,
) -> int:
    """Dataset mode: sparse top-k stats + differentiation metrics."""
    from ultralytics.data.build import build_dataloader, build_yolo_dataset
    from ultralytics.data.utils import check_det_dataset
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    dtype = torch.float16 if args.half else torch.float32

    network = _load_model(args.model, device, args.half)

    # ── build dataset & dataloader ─────────────────────────────
    data_dict = check_det_dataset(str(args.dataset))
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG_DICT
    cfg = get_cfg(DEFAULT_CFG_DICT)
    cfg.imgsz = args.imgsz
    cfg.rect = False
    cfg.batch = args.batch_size
    cfg.fraction = 1.0
    dataset = build_yolo_dataset(
        cfg,
        str(data_dict["val"]),
        batch=args.batch_size,
        data=data_dict,
        mode="val",
        stride=32,
    )
    dataloader = build_dataloader(
        dataset,
        batch=args.batch_size,
        workers=0,
        shuffle=False,
        rank=-1,
    )

    def _forward_tensor(_model, _batch):
        """Pass the image tensor positionally, converting to float32."""
        img = _batch["img"]
        if img.dtype != torch.float32 and img.dtype != torch.float16:
            img = img.float() / 255.0
        return _model(img.to(next(_model.parameters()).device))

    # ── run analysis ───────────────────────────────────────────
    interpreter = RoutingInterpreter(network)

    sparse_topk, differentiation, collapse = interpreter.run_dataset_analysis(
        dataloader,
        layer_name=args.layer,
        max_samples=args.max_samples,
        forward_fn=_forward_tensor,
    )

    # ── write JSON report ──────────────────────────────────────
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "dataset_routing_report.json"
    num_samples = max(
        (s.num_samples for s in sparse_topk.values()), default=0,
    )
    payload = {
        "model": str(args.model),
        "dataset": str(args.dataset),
        "num_samples": num_samples,
        "sparse_topk": {
            name: s.to_dict() for name, s in sparse_topk.items()
        },
        "differentiation": {
            name: m.to_dict() for name, m in differentiation.items()
        },
        "collapse": {
            name: r.to_dict() for name, r in collapse.items()
        },
    }
    report_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8",
    )
    print(f"Dataset routing report: {report_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run routing capture, collapse checks, rendering, or dataset analysis."""
    args = build_parser().parse_args(argv)

    # ── validation ─────────────────────────────────────────────
    if args.dataset and args.image:
        raise SystemExit(
            "Cannot specify both --dataset and a positional image argument"
        )
    if not args.dataset and not args.image:
        raise SystemExit(
            "Either an image path or --dataset is required"
        )
    if args.expert is not None and not args.layer:
        raise SystemExit(
            "--expert requires --layer because counterfactual routing targets "
            "one exact layer"
        )
    if args.imgsz <= 0:
        raise SystemExit("--imgsz must be positive")

    device = torch.device(args.device)
    dtype = torch.float16 if args.half else torch.float32

    if args.dataset:
        return _run_dataset_analysis(args, device)

    return _run_single_image(args, device, dtype)


if __name__ == "__main__":
    raise SystemExit(main())
