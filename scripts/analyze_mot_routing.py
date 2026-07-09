#!/usr/bin/env python3
"""Analyze MoT expert routing patterns on image datasets.

The script runs a YOLO-Master model, hooks every MoTBlock router, and writes
per-image/per-block expert activation summaries. It is intended for the
scenario analysis in MoT ablations: dense vs sparse scenes, small vs large
objects, and optional VisDrone occlusion groups.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from ultralytics import YOLO  # noqa: E402
from ultralytics.nn.modules.mot import MoTBlock  # noqa: E402


EXPERT_NAMES = ("LocalConvTransformer", "WindowTransformer", "DeformableTransformer")
IMG_SUFFIXES = {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


def torch_device(device: str) -> str:
    device = str(device or "cpu").strip().lower()
    if device in {"cpu", "mps"} or device.startswith("cuda"):
        return device
    if device.isdigit():
        return f"cuda:{device}" if torch.cuda.is_available() else "cpu"
    if "," in device:
        first = next((x.strip() for x in device.split(",") if x.strip()), "0")
        return f"cuda:{first}" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dataset_path(data_yaml: Path, value: str | list[str]) -> list[Path]:
    with data_yaml.open() as f:
        data = yaml.safe_load(f)
    base = Path(data.get("path", data_yaml.parent))
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()
    values = value if isinstance(value, list) else [value]
    paths = []
    for item in values:
        p = Path(item)
        if not p.is_absolute():
            p = base / p
        paths.append(p)
    return paths


def image_paths_from_data(data_yaml: Path, split: str, limit: int) -> list[Path]:
    with data_yaml.open() as f:
        data = yaml.safe_load(f)
    if split not in data:
        raise SystemExit(f"{data_yaml} has no split '{split}'")
    images: list[Path] = []
    for p in resolve_dataset_path(data_yaml, data[split]):
        if p.is_file() and p.suffix == ".txt":
            root = p.parent
            for line in p.read_text().splitlines():
                if line.strip():
                    q = Path(line.strip())
                    images.append(q if q.is_absolute() else root / q)
        elif p.is_file() and p.suffix.lower() in IMG_SUFFIXES:
            images.append(p)
        elif p.is_dir():
            images.extend(x for x in sorted(p.rglob("*")) if x.suffix.lower() in IMG_SUFFIXES)
    return images[:limit] if limit > 0 else images


def image_paths_from_source(source: Path, limit: int) -> list[Path]:
    if source.is_file() and source.suffix == ".txt":
        root = source.parent
        images = [Path(x.strip()) for x in source.read_text().splitlines() if x.strip()]
        images = [x if x.is_absolute() else root / x for x in images]
    elif source.is_file():
        images = [source]
    else:
        images = [x for x in sorted(source.rglob("*")) if x.suffix.lower() in IMG_SUFFIXES]
    return images[:limit] if limit > 0 else images


def infer_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        idx = len(parts) - 1 - parts[::-1].index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_yolo_labels(image_path: Path) -> np.ndarray:
    label_path = infer_label_path(image_path)
    if not label_path.exists():
        return np.zeros((0, 5), dtype=np.float32)
    rows = []
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) >= 5:
            rows.append([float(x) for x in parts[:5]])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 5), dtype=np.float32)


def read_visdrone_occlusion(image_path: Path, ann_dir: Path | None) -> str:
    if ann_dir is None:
        return "unknown"
    ann = ann_dir / f"{image_path.stem}.txt"
    if not ann.exists():
        return "unknown"
    occ = []
    for line in ann.read_text().splitlines():
        parts = line.strip().split(",")
        if len(parts) >= 8 and parts[4] != "0":
            try:
                occ.append(int(parts[7]))
            except ValueError:
                pass
    if not occ:
        return "unknown"
    ratio = sum(x >= 1 for x in occ) / len(occ)
    return "occluded" if ratio >= 0.3 else "clear"


def scene_tags(labels: np.ndarray, dense_threshold: int, small_area: float, large_area: float) -> dict[str, str]:
    count = int(labels.shape[0])
    if count >= dense_threshold:
        density = "dense"
    elif count <= max(1, dense_threshold // 4):
        density = "sparse"
    else:
        density = "medium"

    if count == 0:
        scale = "empty"
        irregular = "unknown"
    else:
        areas = labels[:, 3] * labels[:, 4]
        median_area = float(np.median(areas))
        if median_area <= small_area:
            scale = "small"
        elif median_area >= large_area:
            scale = "large"
        else:
            scale = "mixed"
        ratios = labels[:, 3] / np.clip(labels[:, 4], 1e-6, None)
        irregular = "irregular" if float(np.mean((ratios > 3.0) | (ratios < 1.0 / 3.0))) >= 0.3 else "regular"
    return {"object_count": str(count), "density": density, "scale": scale, "shape": irregular}


def preprocess(image_path: Path, imgsz: int, device: str) -> torch.Tensor:
    im = cv2.imread(str(image_path))
    if im is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(im).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    return tensor.to(torch.device(device))


def summarize_weights(weights: torch.Tensor) -> dict[str, float]:
    w = weights[0].float().cpu()  # [E,H,W] or [E,1,1]
    means = w.flatten(1).mean(dim=1).numpy()
    winners = w.argmax(dim=0).flatten().numpy()
    total = max(1, winners.size)
    row: dict[str, float] = {}
    for idx, name in enumerate(EXPERT_NAMES):
        row[f"{name}_mean_weight"] = float(means[idx])
        row[f"{name}_top1_token_frac"] = float((winners == idx).sum() / total)
    row["top_expert"] = EXPERT_NAMES[int(np.argmax(means))]
    return row


def save_heatmap(weights: torch.Tensor, out_dir: Path, stem: str, module_name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    w = weights[0].float().cpu().numpy()
    safe_module = module_name.replace(".", "_")
    for idx, name in enumerate(EXPERT_NAMES):
        plt.figure(figsize=(4, 4))
        plt.imshow(w[idx], cmap="magma")
        plt.axis("off")
        plt.title(name)
        plt.tight_layout()
        plt.savefig(out_dir / f"{stem}_{safe_module}_{idx}_{name}.png", dpi=160)
        plt.close()


def write_csv(path: Path, rows: list[dict[str, str | float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    groups: dict[tuple[str, str, str, str], list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["density"]), str(row["scale"]), str(row["shape"]), str(row["occlusion"]))].append(row)
    out = []
    metric_keys = [f"{name}_mean_weight" for name in EXPERT_NAMES] + [f"{name}_top1_token_frac" for name in EXPERT_NAMES]
    for key, group in sorted(groups.items()):
        item: dict[str, str | float] = {
            "density": key[0],
            "scale": key[1],
            "shape": key[2],
            "occlusion": key[3],
            "samples": len(group),
        }
        for metric in metric_keys:
            vals = [float(row[metric]) for row in group if metric in row]
            item[metric] = float(np.mean(vals)) if vals else 0.0
        out.append(item)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to a trained .pt model or YAML config.")
    parser.add_argument("--data", type=Path, help="Dataset YAML. Use with --split.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--source", type=Path, help="Image, directory, or txt file. Overrides --data.")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/mot_routing")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--limit", type=int, default=256)
    parser.add_argument("--dense-threshold", type=int, default=20)
    parser.add_argument("--small-area", type=float, default=0.01)
    parser.add_argument("--large-area", type=float, default=0.08)
    parser.add_argument("--visdrone-ann-dir", type=Path)
    parser.add_argument("--save-heatmaps", type=int, default=24)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch_device(args.device)
    images = image_paths_from_source(args.source, args.limit) if args.source else image_paths_from_data(args.data, args.split, args.limit)
    if not images:
        raise SystemExit("no images found")

    yolo = YOLO(str(args.model))
    model = yolo.model.eval().to(torch.device(device))
    captures: list[tuple[str, torch.Tensor]] = []
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, MoTBlock):
            hooks.append(module.router.register_forward_hook(lambda _m, _i, out, n=name: captures.append((n, out[0].detach()))))
    if not hooks:
        raise SystemExit(f"no MoTBlock modules found in {args.model}")

    rows: list[dict[str, str | float]] = []
    heatmaps_left = args.save_heatmaps
    with torch.inference_mode():
        for idx, image_path in enumerate(images):
            captures.clear()
            x = preprocess(image_path, args.imgsz, device)
            _ = model(x)
            labels = read_yolo_labels(image_path)
            tags = scene_tags(labels, args.dense_threshold, args.small_area, args.large_area)
            tags["occlusion"] = read_visdrone_occlusion(image_path, args.visdrone_ann_dir)
            for module_name, weights in captures:
                row: dict[str, str | float] = {
                    "image": str(image_path),
                    "module": module_name,
                    **tags,
                    **summarize_weights(weights),
                }
                rows.append(row)
                if heatmaps_left > 0:
                    save_heatmap(weights, args.out / "heatmaps", image_path.stem, module_name)
            heatmaps_left -= 1
            if (idx + 1) % 25 == 0:
                print(f"[routing] processed {idx + 1}/{len(images)} images")

    for hook in hooks:
        hook.remove()

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "routing_records.csv", rows)
    write_csv(args.out / "routing_summary_by_scene.csv", aggregate(rows))
    (args.out / "routing_summary_by_scene.json").write_text(
        json.dumps(aggregate(rows), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[routing] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
