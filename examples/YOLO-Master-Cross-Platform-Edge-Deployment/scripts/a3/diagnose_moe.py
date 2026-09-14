#!/usr/bin/env python
"""Project03 deliverable 1: MoE expert-utilization diagnosis.

Runs in the conda yolo_master env (old-lineage ultralytics editable at
/data/YOLO-Master, matching the trained checkpoints). One validation pass per model
collects BOTH the expert usage statistics (via the upstream ExpertUsageTracker) and
the baseline mAP, then emits machine-readable stats, the upstream text report, the
upstream heatmap/bar plots, per-layer histograms with candidate threshold lines, and
an optional scene-dependent activation analysis (simple/medium/complex buckets by GT
object count).

Model-family notes baked in:
  - ES_MOE (EsMoE-N): sparse eval collapses mAP (documented repo issue), so
    use_sparse_inference is forced off for the val pass; its router emits dense
    softmax weights, making hit-shares degenerate by construction - avg_weight is
    the discriminative signal and is highlighted in the report.
  - UltraOptimizedMoE (UoMoE): image-level top-k routing; hits are per image, so the
    report also prints the absolute sample count behind every percentage.

Usage:
  python scripts/project03/diagnose_moe.py \
      --model /data/model_zoo/src/yolo-master-UoMoE-N_aitodv2_best.pt \
      --data AI-TOD-v2.yaml --max-images 120 --scene-split objects
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


CANDIDATE_THRESHOLDS = (0.05, 0.10, 0.15, 0.20, 0.30)


@contextlib.contextmanager
def _pushd(path: Path):
    prev = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def _resolve_dataset(data: str):
    """Return (yaml_path, dataset_root, names) for a dataset yaml name/path."""
    import yaml
    p = Path(data)
    if not p.exists():
        p = Path("/data/YOLO-Master/ultralytics/cfg/datasets") / data
    if not p.exists():
        raise FileNotFoundError(f"dataset yaml not found: {data}")
    cfg = yaml.safe_load(p.read_text())
    root = Path(cfg["path"])
    if not root.is_absolute():
        try:
            from ultralytics.utils import SETTINGS
            root = Path(SETTINGS["datasets_dir"]) / root
        except Exception:
            root = Path("/data/datasets") / root
    return p, root, cfg


def _subset_yaml(src_yaml: Path, root: Path, cfg: dict, split: str, n: int, workdir: Path) -> str:
    """Build a temp dataset rooted at a symlinked N-image subset of the split."""
    img_src = root / cfg[split]
    lbl_src = Path(str(img_src).replace("/images/", "/labels/").replace("images/", "labels/"))
    imgs = sorted(img_src.iterdir())[:n]
    sub = workdir / "subset"
    (sub / "images" / split).mkdir(parents=True, exist_ok=True)
    (sub / "labels" / split).mkdir(parents=True, exist_ok=True)
    for im in imgs:
        (sub / "images" / split / im.name).symlink_to(im.resolve())
        lb = lbl_src / (im.stem + ".txt")
        if lb.exists():
            (sub / "labels" / split / lb.name).symlink_to(lb.resolve())
    out = {"path": str(sub), "train": f"images/{split}", "val": f"images/{split}",
           "nc": cfg.get("nc", len(cfg.get("names", {}))), "names": cfg["names"]}
    import yaml
    y = workdir / "subset.yaml"
    y.write_text(yaml.safe_dump(out))
    return str(y)


def _strip_property_shadows(model) -> int:
    """Remove stale instance attributes that shadow class-level properties.

    Old checkpoints (Jan-2026 released COCO EsMoE-N) pickled plain instance values
    (e.g. aux_loss) for names the CURRENT class defines as read-only properties;
    upstream's _robust_deepcopy re-setattrs instance attrs and explodes on the
    missing setter. The property recomputes the value, so dropping the stale copy
    is lossless. Returns the number of attributes stripped.
    """
    n = 0
    for m in model.model.modules():
        cls = type(m)
        for k in list(m.__dict__.keys()):
            if isinstance(getattr(cls, k, None), property):
                m.__dict__.pop(k)
                n += 1
    return n


# attributes today's MoE / router classes read in eval forward that pre-date the
# released Feb-2026 pickles (upstream 8.4.x dropped the in-class compat shims).
# Absence is the signal: defaults = what the checkpoints were trained with.
_LEGACY_MOE_DEFAULTS = {
    "progressive_sparsity": False, "_training_step": 0, "warmup_steps": 5000,
    "expert_dropout_rate": 0.15, "dropout_interval": 100, "detach_routing": False,
    "last_routing_snapshot": None, "_moe_nonfinite_events": 0,
}
_LEGACY_ROUTER_DEFAULTS = {"capacity_factor": None, "noise_std": 0.0}


def _fix_legacy_attrs(model) -> dict:
    """Fill attributes missing from old pickles on MoE blocks and their routers.

    Returns {attr: count} of what was filled, so the log shows exactly which
    compat gaps the running ultralytics lineage has against the released weights.
    """
    filled: dict = {}
    for mod in model.model.modules():
        if hasattr(mod, "experts") and hasattr(mod, "routing"):
            for k, v in _LEGACY_MOE_DEFAULTS.items():
                if not hasattr(mod, k):
                    setattr(mod, k, {} if v is None and k == "last_routing_snapshot" else v)
                    filled[k] = filled.get(k, 0) + 1
            if not hasattr(mod, "_current_top_k"):
                mod._current_top_k = getattr(mod, "top_k", len(mod.experts))
                filled["_current_top_k"] = filled.get("_current_top_k", 0) + 1
            r = mod.routing
            for k, v in _LEGACY_ROUTER_DEFAULTS.items():
                if not hasattr(r, k):
                    setattr(r, k, v)
                    filled[f"routing.{k}"] = filled.get(f"routing.{k}", 0) + 1
    return filled


def _fix_add_residual(model) -> int:
    """Repair add_residual on OptimizedMOEImproved blocks restored from old checkpoints.

    The upstream compat shim defaults a MISSING add_residual to True, but checkpoints
    predating the attribute (the Jan-2026 released COCO v0.1-N) were trained WITHOUT the
    residual: the wrong default silently collapses classification to ~0.04 confidence
    while boxes stay sane. Absence of the attribute is the signal it must be False.
    Returns the number of blocks forced to False.
    """
    filled = _fix_legacy_attrs(model)
    if filled:
        print(f"[repair] legacy attrs filled: {filled}")
    n_false = 0
    for mod in model.model.modules():
        if type(mod).__name__ != "OptimizedMOEImproved":
            continue
        had = "add_residual" in vars(mod)
        if hasattr(mod, "_ensure_compat_attrs"):
            mod._ensure_compat_attrs()
        if not had:
            mod.add_residual = False
            n_false += 1
    return n_false


def _force_dense_esmoe(model) -> bool:
    """Set use_sparse_inference=False on ES_MOE modules; True if any were found."""
    from ultralytics.nn.modules.moe.modules import ES_MOE
    found = False
    for m in model.model.modules():
        if isinstance(m, ES_MOE):
            m.use_sparse_inference = False
            found = True
    return found


def _stats_to_dict(tracker, model) -> dict:
    """usage_stats -> plain-JSON dict, zero-hit experts filled from module metadata."""
    modules = dict(model.model.named_modules())
    layers = {}
    for layer_name, stats in tracker.usage_stats.items():
        parent = ".".join(layer_name.split(".")[:-1])
        num_experts = None
        if parent in modules and hasattr(modules[parent], "experts"):
            num_experts = len(modules[parent].experts)
        total_hits = float(sum(s.hits for s in stats.values()))
        ids = set(range(num_experts)) if num_experts else set(int(i) for i in stats)
        experts = {}
        for eid in sorted(ids):
            s = stats.get(eid, None) or stats.get(np.int64(eid), None)
            hits = float(getattr(s, "hits", 0.0)) if s else 0.0
            share = hits / total_hits if total_hits > 0 else 0.0
            experts[int(eid)] = {
                "hits": hits,
                "share": round(share, 6),
                "avg_weight": round(float(getattr(s, "avg_weight", 0.0)) if s else 0.0, 6),
            }
        shares = np.array([e["share"] for e in experts.values()])
        gini = 0.0
        if shares.sum() > 0:
            sorted_s = np.sort(shares)
            k = len(sorted_s)
            gini = float((2 * np.arange(1, k + 1) - k - 1).dot(sorted_s) / (k * sorted_s.sum()))
        degenerate = bool(len(shares) > 1 and (shares.max() - shares.min()) < 1e-6)
        layers[layer_name] = {
            "num_experts": len(experts),
            "total_hits": total_hits,
            "load_balance_std": round(float(shares.std()), 6),
            "gini": round(gini, 6),
            "hit_shares_degenerate": degenerate,
            "experts": experts,
        }
    return layers


def _plot_layer_histograms(layers: dict, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for layer_name, info in layers.items():
        ids = sorted(info["experts"])
        shares = [info["experts"][i]["share"] for i in ids]
        weights = [info["experts"][i]["avg_weight"] for i in ids]
        fig, ax = plt.subplots(figsize=(max(6, len(ids) * 0.7), 4))
        ax.bar([str(i) for i in ids], shares, color="#4c72b0", label="hit share")
        ax.plot([str(i) for i in ids], weights, "o-", color="#dd8452", label="avg weight")
        for t in CANDIDATE_THRESHOLDS:
            ax.axhline(t, ls="--", lw=0.8, color="grey")
            ax.text(len(ids) - 0.4, t, f"{t:.2f}", fontsize=7, va="bottom", color="grey")
        ax.set_title(f"{layer_name}  (hits={info['total_hits']:.0f}"
                     + (", DEGENERATE hit shares - use avg_weight" if info["hit_shares_degenerate"] else "")
                     + ")")
        ax.set_ylabel("share / weight")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"hist_{layer_name.replace('.', '_')}.png", dpi=120)
        plt.close(fig)


def _scene_buckets(root: Path, cfg: dict, split: str, max_images: int):
    """Bucket split images into simple/medium/complex by GT object-count terciles."""
    img_dir = root / cfg[split]
    lbl_dir = Path(str(img_dir).replace("/images/", "/labels/").replace("images/", "labels/"))
    counts = []
    imgs = sorted(img_dir.iterdir())
    if max_images:
        imgs = imgs[:max_images]
    for im in imgs:
        lb = lbl_dir / (im.stem + ".txt")
        n = sum(1 for _ in open(lb)) if lb.exists() else 0
        counts.append((im, n))
    ns = sorted(n for _, n in counts)
    t1, t2 = ns[len(ns) // 3], ns[2 * len(ns) // 3]
    buckets = {"simple": [], "medium": [], "complex": []}
    for im, n in counts:
        buckets["simple" if n <= t1 else ("medium" if n <= t2 else "complex")].append(str(im))
    return buckets, (t1, t2)


def diagnose_one(model_path: str, args, out_root: Path) -> dict:
    from ultralytics import YOLO
    from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker

    stem = Path(model_path).stem
    out = out_root / stem
    out.mkdir(parents=True, exist_ok=True)

    yaml_path, root, cfg = _resolve_dataset(args.data)
    model = YOLO(model_path)
    n_res = _fix_add_residual(model)
    n_shadow = _strip_property_shadows(model)
    if n_shadow:
        print(f"[{stem}] stripped {n_shadow} property-shadowing instance attr(s)")
    if n_res:
        print(f"[{stem}] add_residual compat repair applied to {n_res} legacy MoE block(s)")
    dense_forced = _force_dense_esmoe(model)
    if dense_forced:
        print(f"[{stem}] ES_MOE detected: use_sparse_inference=False forced for honest eval")

    with tempfile.TemporaryDirectory() as td:
        data = str(yaml_path)
        if args.max_images:
            data = _subset_yaml(yaml_path, root, cfg, args.split, args.max_images, Path(td))
            print(f"[{stem}] subset: first {args.max_images} images of {args.split}")

        with ExpertUsageTracker(model.model) as tracker:
            res = model.val(data=data, split="val", batch=args.batch, imgsz=args.imgsz,
                            device=args.device, workers=args.workers, verbose=False, plots=False)
            # capture upstream text report + plots (plots save to CWD -> redirect)
            buf = io.StringIO()
            from ultralytics.utils import LOGGER
            import logging
            h = logging.StreamHandler(buf)
            LOGGER.addHandler(h)
            with _pushd(out):
                with contextlib.redirect_stdout(buf):
                    tracker.print_report()
            LOGGER.removeHandler(h)
            (out / "report.txt").write_text(buf.getvalue())
            layers = _stats_to_dict(tracker, model)

    result = {
        "model": str(model_path),
        "dataset": str(yaml_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "max_images": args.max_images,
        "es_moe_dense_eval_forced": dense_forced,
        "baseline": {"mAP50": round(float(res.box.map50), 5),
                     "mAP50_95": round(float(res.box.map), 5)},
        "layers": layers,
    }
    (out / "usage.json").write_text(json.dumps(result, indent=2))
    _plot_layer_histograms(layers, out)
    print(f"[{stem}] baseline mAP50={result['baseline']['mAP50']} "
          f"mAP50-95={result['baseline']['mAP50_95']}  layers={len(layers)}")

    if args.scene_split == "objects":
        _scene_analysis(model, root, cfg, args, out, stem)
    return result


def _scene_analysis(model, root, cfg, args, out: Path, stem: str):
    from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker
    buckets, (t1, t2) = _scene_buckets(root, cfg, args.split, args.max_images)
    print(f"[{stem}] scene buckets (GT objects): simple<={t1} < medium<={t2} < complex "
          f"({', '.join(f'{k}:{len(v)}' for k, v in buckets.items())})")
    scene = {}
    for name, imgs in buckets.items():
        if not imgs:
            continue
        with ExpertUsageTracker(model.model) as tracker:
            for i in range(0, len(imgs), args.batch):
                model.predict(imgs[i:i + args.batch], imgsz=args.imgsz, device=args.device,
                              verbose=False, save=False, conf=0.25)
            scene[name] = {"n_images": len(imgs),
                           "layers": _stats_to_dict(tracker, model)}
    # divergence: per layer, L1 distance between bucket share vectors
    divergence = {}
    names = [n for n in ("simple", "medium", "complex") if n in scene]
    if len(names) >= 2:
        for layer in scene[names[0]]["layers"]:
            vecs = []
            for n in names:
                info = scene[n]["layers"].get(layer)
                if info:
                    ids = sorted(info["experts"])
                    vecs.append(np.array([info["experts"][i]["share"] for i in ids]))
            if len(vecs) >= 2:
                divergence[layer] = round(float(np.abs(vecs[0] - vecs[-1]).sum()), 6)
    payload = {"bucket_thresholds": {"simple_max": t1, "medium_max": t2},
               "buckets": scene, "l1_divergence_simple_vs_complex": divergence}
    (out / "scene_usage.json").write_text(json.dumps(payload, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    layers_all = sorted({l for s in scene.values() for l in s["layers"]})
    for layer in layers_all:
        fig, ax = plt.subplots(figsize=(7, 4))
        width = 0.8 / max(1, len(names))
        for bi, n in enumerate(names):
            info = scene[n]["layers"].get(layer)
            if not info:
                continue
            ids = sorted(info["experts"])
            ax.bar(np.arange(len(ids)) + bi * width,
                   [info["experts"][i]["share"] for i in ids], width, label=n)
        ax.set_title(f"{layer} expert share by scene complexity "
                     f"(L1 s-vs-c: {divergence.get(layer, 0):.3f})")
        ax.set_xlabel("expert id")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"scene_{layer.replace('.', '_')}.png", dpi=120)
        plt.close(fig)
    print(f"[{stem}] scene analysis: {len(layers_all)} layers, "
          f"max L1 divergence {max(divergence.values()) if divergence else 0:.3f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", nargs="+", required=True, help="checkpoint path(s)")
    ap.add_argument("--data", default="AI-TOD-v2.yaml")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=0,
                    help="val dataloader workers (0 = safe for py3.14 local; use 8 on pods)")
    ap.add_argument("--max-images", type=int, default=0, help="0 = full split")
    ap.add_argument("--out", default="runs/project03/diagnosis")
    ap.add_argument("--scene-split", choices=["objects", "off"], default="objects")
    args = ap.parse_args()

    out_root = Path(args.out)
    rows = []
    for mp in args.model:
        r = diagnose_one(mp, args, out_root)
        rows.append({"model": Path(mp).stem,
                     "mAP50": r["baseline"]["mAP50"], "mAP50_95": r["baseline"]["mAP50_95"],
                     "layers": len(r["layers"]),
                     "dense_forced": r["es_moe_dense_eval_forced"],
                     "max_gini": max((l["gini"] for l in r["layers"].values()), default=0)})
    with open(out_root / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nsummary -> {out_root}/summary.csv")


if __name__ == "__main__":
    main()
