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
ROUTING_METRICS = tuple(
    f"{name}_{suffix}" for name in EXPERT_NAMES for suffix in ("mean_weight", "top1_token_frac")
)
SCENE_CONTRASTS = {
    "density": ("dense", "sparse"),
    "scale": ("small", "large"),
    "shape": ("irregular", "regular"),
    "occlusion": ("occluded", "clear"),
}


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


def derive_scene_thresholds(
    labels_by_image: list[np.ndarray], quantile_low: float = 0.25, quantile_high: float = 0.75
) -> dict[str, float]:
    """Derive balanced, dataset-specific density and scale thresholds before model inference."""
    if not 0 <= quantile_low < quantile_high <= 1:
        raise ValueError(f"scene quantiles must satisfy 0 <= low < high <= 1, got {quantile_low}, {quantile_high}")
    if not labels_by_image:
        raise ValueError("cannot derive scene thresholds without labels")
    counts = np.asarray([len(labels) for labels in labels_by_image], dtype=np.float64)
    median_areas = np.asarray(
        [float(np.median(labels[:, 3] * labels[:, 4])) for labels in labels_by_image if len(labels)], dtype=np.float64
    )
    if not len(median_areas):
        raise ValueError("cannot derive scale thresholds from empty labels")
    return {
        "sparse_threshold": float(np.quantile(counts, quantile_low)),
        "dense_threshold": float(np.quantile(counts, quantile_high)),
        "small_area": float(np.quantile(median_areas, quantile_low)),
        "large_area": float(np.quantile(median_areas, quantile_high)),
    }


def scene_tags(
    labels: np.ndarray,
    dense_threshold: float,
    small_area: float,
    large_area: float,
    sparse_threshold: float | None = None,
) -> dict[str, str]:
    count = int(labels.shape[0])
    sparse_threshold = max(1.0, dense_threshold / 4) if sparse_threshold is None else sparse_threshold
    if count >= dense_threshold:
        density = "dense"
    elif count <= sparse_threshold:
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


def letterbox_image(im: np.ndarray, imgsz: int, color: int = 114) -> np.ndarray:
    """Resize without aspect-ratio distortion and center-pad to a square inference canvas."""
    height, width = im.shape[:2]
    if height <= 0 or width <= 0 or imgsz <= 0:
        raise ValueError(f"invalid letterbox shape: image={im.shape}, imgsz={imgsz}")
    scale = min(imgsz / height, imgsz / width)
    new_width, new_height = int(round(width * scale)), int(round(height * scale))
    if (new_width, new_height) != (width, height):
        im = cv2.resize(im, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_w, pad_h = imgsz - new_width, imgsz - new_height
    left, right = int(round(pad_w / 2 - 0.1)), int(round(pad_w / 2 + 0.1))
    top, bottom = int(round(pad_h / 2 - 0.1)), int(round(pad_h / 2 + 0.1))
    return cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(color,) * 3)


def preprocess(image_path: Path, imgsz: int, device: str) -> torch.Tensor:
    im = cv2.imread(str(image_path))
    if im is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = letterbox_image(im, imgsz)
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
        image = plt.imshow(w[idx], cmap="magma", vmin=0.0, vmax=1.0)
        plt.colorbar(image, fraction=0.046, pad=0.04, label="routing weight")
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
    groups: dict[tuple[str, str, str, str, str], list[dict[str, str | float]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row["module"]),
                str(row["density"]),
                str(row["scale"]),
                str(row["shape"]),
                str(row["occlusion"]),
            )
        ].append(row)
    out = []
    for key, group in sorted(groups.items()):
        item: dict[str, str | float] = {
            "module": key[0],
            "density": key[1],
            "scale": key[2],
            "shape": key[3],
            "occlusion": key[4],
            "samples": len(group),
        }
        for metric in ROUTING_METRICS:
            vals = [float(row[metric]) for row in group if metric in row]
            item[metric] = float(np.mean(vals)) if vals else 0.0
        out.append(item)
    return out


def _image_level_values(
    rows: list[dict[str, str | float]], factor: str, metric: str, module: str | None = None
) -> dict[str, np.ndarray]:
    """Average repeated layer records per image so tests use images, not layers, as independent samples."""
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        if module is not None and str(row["module"]) != module:
            continue
        group = str(row[factor])
        if group and metric in row:
            grouped[(group, str(row["image"]))].append(float(row[metric]))
    values: dict[str, list[float]] = defaultdict(list)
    for (group, _image), samples in grouped.items():
        values[group].append(float(np.mean(samples)))
    return {group: np.asarray(samples, dtype=np.float64) for group, samples in values.items()}


def aggregate_marginal(rows: list[dict[str, str | float]]) -> list[dict[str, str | float]]:
    """Aggregate each scene factor separately, globally and per MoT block, at the image level."""
    modules = sorted({str(row["module"]) for row in rows})
    out: list[dict[str, str | float]] = []
    for scope, module in [("global", None), *(("module", name) for name in modules)]:
        for factor in SCENE_CONTRASTS:
            by_metric = {metric: _image_level_values(rows, factor, metric, module) for metric in ROUTING_METRICS}
            groups = sorted({group for values in by_metric.values() for group in values})
            for group in groups:
                item: dict[str, str | float] = {
                    "scope": scope,
                    "module": module or "all",
                    "factor": factor,
                    "group": group,
                    "images": max((len(values.get(group, ())) for values in by_metric.values()), default=0),
                }
                for metric, values in by_metric.items():
                    samples = values.get(group, np.empty(0))
                    item[metric] = float(samples.mean()) if len(samples) else 0.0
                out.append(item)
    return out


def _hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Bias-corrected standardized mean difference (positive means group A is higher)."""
    n_a, n_b = len(a), len(b)
    pooled_df = n_a + n_b - 2
    if n_a < 2 or n_b < 2 or pooled_df <= 0:
        return float("nan")
    pooled_var = ((n_a - 1) * a.var(ddof=1) + (n_b - 1) * b.var(ddof=1)) / pooled_df
    if pooled_var <= 0:
        return 0.0 if float(a.mean()) == float(b.mean()) else float("inf")
    correction = 1.0 - 3.0 / (4.0 * (n_a + n_b) - 9.0)
    return float(correction * (a.mean() - b.mean()) / np.sqrt(pooled_var))


def _bootstrap_difference_ci(
    a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n_resamples: int
) -> tuple[float, float]:
    """Return a percentile 95% bootstrap CI for the difference in image-level means."""
    a_idx = rng.integers(0, len(a), size=(n_resamples, len(a)))
    b_idx = rng.integers(0, len(b), size=(n_resamples, len(b)))
    differences = a[a_idx].mean(axis=1) - b[b_idx].mean(axis=1)
    low, high = np.quantile(differences, (0.025, 0.975))
    return float(low), float(high)


def _benjamini_hochberg(rows: list[dict[str, str | float]]) -> None:
    """Add FDR-adjusted q-values in place across every valid scene contrast."""
    valid = [(index, float(row["permutation_p"])) for index, row in enumerate(rows) if "permutation_p" in row]
    if not valid:
        return
    ordered = sorted(valid, key=lambda item: item[1])
    adjusted = [0.0] * len(ordered)
    running = 1.0
    for rank in range(len(ordered), 0, -1):
        value = min(running, ordered[rank - 1][1] * len(ordered) / rank)
        adjusted[rank - 1] = value
        running = value
    for (index, _), value in zip(ordered, adjusted):
        rows[index]["fdr_q"] = float(value)


def scene_contrasts(
    rows: list[dict[str, str | float]], n_resamples: int = 10_000, seed: int = 42
) -> list[dict[str, str | float]]:
    """Compute global and per-layer scene contrasts with robust uncertainty and multiplicity control."""
    from scipy.stats import permutation_test

    rng = np.random.default_rng(seed)
    modules = sorted({str(row["module"]) for row in rows})
    metrics = ("DeformableTransformer_mean_weight", "DeformableTransformer_top1_token_frac")
    out: list[dict[str, str | float]] = []

    def mean_difference(a, b, axis=0):
        return np.mean(a, axis=axis) - np.mean(b, axis=axis)

    for scope, module in [("global", None), *(("module", name) for name in modules)]:
        for factor, (group_a, group_b) in SCENE_CONTRASTS.items():
            for metric in metrics:
                grouped = _image_level_values(rows, factor, metric, module)
                a, b = grouped.get(group_a, np.empty(0)), grouped.get(group_b, np.empty(0))
                item: dict[str, str | float] = {
                    "scope": scope,
                    "module": module or "all",
                    "factor": factor,
                    "group_a": group_a,
                    "group_b": group_b,
                    "metric": metric,
                    "n_a": len(a),
                    "n_b": len(b),
                }
                if len(a) >= 2 and len(b) >= 2:
                    ci_low, ci_high = _bootstrap_difference_ci(a, b, rng, n_resamples)
                    result = permutation_test(
                        (a, b),
                        mean_difference,
                        vectorized=True,
                        n_resamples=n_resamples,
                        batch=256,
                        alternative="two-sided",
                        rng=rng,
                    )
                    item.update(
                        {
                            "mean_a": float(a.mean()),
                            "mean_b": float(b.mean()),
                            "mean_difference": float(a.mean() - b.mean()),
                            "ci95_low": ci_low,
                            "ci95_high": ci_high,
                            "hedges_g": _hedges_g(a, b),
                            "permutation_p": float(result.pvalue),
                        }
                    )
                out.append(item)
    _benjamini_hochberg(out)
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
    parser.add_argument("--scene-quantiles", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scene-quantile-low", type=float, default=0.25)
    parser.add_argument("--scene-quantile-high", type=float, default=0.75)
    parser.add_argument("--visdrone-ann-dir", type=Path)
    parser.add_argument("--save-heatmaps", type=int, default=24)
    parser.add_argument("--stats-resamples", type=int, default=10_000)
    parser.add_argument("--stats-seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch_device(args.device)
    images = image_paths_from_source(args.source, args.limit) if args.source else image_paths_from_data(args.data, args.split, args.limit)
    if not images:
        raise SystemExit("no images found")

    label_cache = {image_path: read_yolo_labels(image_path) for image_path in images}
    if args.scene_quantiles:
        thresholds = derive_scene_thresholds(
            list(label_cache.values()), args.scene_quantile_low, args.scene_quantile_high
        )
        threshold_mode = "dataset_quantiles"
    else:
        thresholds = {
            "sparse_threshold": float(max(1, args.dense_threshold // 4)),
            "dense_threshold": float(args.dense_threshold),
            "small_area": float(args.small_area),
            "large_area": float(args.large_area),
        }
        threshold_mode = "fixed"
    print(f"[routing] scene thresholds ({threshold_mode}): {thresholds}")

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
            labels = label_cache[image_path]
            tags = scene_tags(labels, **thresholds)
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
    (args.out / "scene_thresholds.json").write_text(
        json.dumps(
            {
                "mode": threshold_mode,
                "quantile_low": args.scene_quantile_low if args.scene_quantiles else None,
                "quantile_high": args.scene_quantile_high if args.scene_quantiles else None,
                **thresholds,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    write_csv(args.out / "routing_records.csv", rows)
    scene_summary = aggregate(rows)
    marginal_summary = aggregate_marginal(rows)
    contrasts = scene_contrasts(rows, n_resamples=args.stats_resamples, seed=args.stats_seed)
    write_csv(args.out / "routing_summary_by_scene.csv", scene_summary)
    write_csv(args.out / "routing_summary_marginal.csv", marginal_summary)
    write_csv(args.out / "routing_scene_contrasts.csv", contrasts)
    (args.out / "routing_summary_by_scene.json").write_text(
        json.dumps(scene_summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out / "routing_scene_contrasts.json").write_text(
        json.dumps(contrasts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[routing] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
