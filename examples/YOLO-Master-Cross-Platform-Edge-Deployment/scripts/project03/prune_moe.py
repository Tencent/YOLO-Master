#!/usr/bin/env python
"""Project03 deliverable 2: intelligent MoE expert pruning (progressive, batchable).

Builds on the upstream MoEPruner (ultralytics/nn/modules/moe/pruning.py) via
subclassing - the upstream tree is never modified. Fixes applied here:

  1. Planning goes through _expert_score, activating the otherwise-dead
     importance modes: "usage" (hit share) and "usage_weight" (hit share x mean
     routing weight - required for ES_MOE, whose dense-softmax router makes raw hit
     shares degenerate by construction). Auto-switches with a warning when
     degeneracy is detected.
  2. Robust router-projection surgery: the upstream finder only inspects the LAST
     layer of the router Sequential and silently skips weight pruning when it is a
     normalization layer (OptimizedMOEImproved / EfficientSpatialRouter ends in
     BatchNorm2d), leaving the router emitting logits for pruned experts - a silent
     accuracy corruption. Here the Sequential is scanned backward for the projection
     (Conv2d/Linear with out == num_experts), trailing num_experts-sized
     normalization layers are rebuilt with the kept channels' statistics, and if no
     projection is found the layer HARD-FAILS instead of corrupting.
  3. --min-keep floor (default: the layer's top_k) so routing stays meaningful;
     _current_top_k is synced where present (stale-attribute hazard).
  4. One diagnosis feeds the whole progressive sweep: pass --stats usage.json from
     diagnose_moe.py, or let this script run the val pass once and reuse it for
     every threshold.

Outputs per model under --out: <stem>_pruned_t{thr}.pt per threshold, sweep.csv
(threshold, kept experts, params, dParams%, GFLOPs, mAP50, mAP50-95, dmAP), sweep.png,
plan_t{thr}.json, and final.json when --final-eval aitod-official is requested.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def _load_stats_json(path: str):
    """Rehydrate diagnose_moe.py usage.json into {layer: {id: StatsLike}}."""
    class _S:
        __slots__ = ("hits", "avg_weight")

        def __init__(self, hits, avg_weight):
            self.hits = hits
            self.avg_weight = avg_weight

    d = json.loads(Path(path).read_text())
    out = {}
    for layer, info in d["layers"].items():
        out[layer] = {int(i): _S(e["hits"], e["avg_weight"]) for i, e in info["experts"].items()}
    return out, d


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


def _fix_add_residual(model) -> int:
    """Repair add_residual on OptimizedMOEImproved blocks restored from old checkpoints.

    The upstream compat shim defaults a MISSING add_residual to True, but checkpoints
    predating the attribute (the Jan-2026 released COCO v0.1-N) were trained WITHOUT the
    residual: the wrong default silently collapses classification to ~0.04 confidence
    while boxes stay sane. Absence of the attribute is the signal it must be False.
    Returns the number of blocks forced to False.
    """
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


class Project03Pruner:
    """Diagnosis-once, surgery-per-threshold pruner wrapping upstream MoEPruner logic."""

    def __init__(self, model_path: str, dataset: str, device: str = "cpu",
                 imgsz: int = 640, batch: int = 8, importance: str = "usage",
                 min_keep: int = 0, keep_top_m: int = 0, workers: int = 0):
        from ultralytics import YOLO
        self.model_path = model_path
        self.dataset = dataset
        self.device = device
        self.imgsz = imgsz
        self.batch = batch
        self.importance = importance
        self.min_keep = min_keep
        self.keep_top_m = keep_top_m
        self.workers = workers
        self.model = YOLO(model_path)
        n_res = _fix_add_residual(self.model)
        n_shadow = _strip_property_shadows(self.model)
        if n_shadow:
            print(f"[compat] stripped {n_shadow} property-shadowing instance attr(s)")
        if n_res:
            print(f"[compat] add_residual repair applied to {n_res} legacy MoE block(s)")
        self._force_dense_esmoe()
        self.usage_stats = {}

    def _force_dense_esmoe(self):
        from ultralytics.nn.modules.moe.modules import ES_MOE
        self.has_esmoe = False
        for m in self.model.model.modules():
            if isinstance(m, ES_MOE):
                m.use_sparse_inference = False
                self.has_esmoe = True

    # ---------------- diagnosis ----------------

    def diagnose(self, stats_json: str | None, data_for_val: str):
        if stats_json:
            self.usage_stats, meta = _load_stats_json(stats_json)
            print(f"[stats] loaded from {stats_json} ({len(self.usage_stats)} layers)")
            return
        from ultralytics.nn.modules.moe.analysis import ExpertUsageTracker
        print("[stats] running diagnosis val pass...")
        with ExpertUsageTracker(self.model.model) as tracker:
            self.model.val(data=data_for_val, split="val", batch=self.batch,
                           imgsz=self.imgsz, device=self.device, workers=self.workers,
                           verbose=False, plots=False)
            self.usage_stats = {k: dict(v) for k, v in tracker.usage_stats.items()}

    # ---------------- planning ----------------

    def _scores(self, layer_stats: dict, num_experts: int) -> dict[int, float]:
        total_hits = float(sum(getattr(s, "hits", 0.0) for s in layer_stats.values()))
        scores = {}
        for eid in range(num_experts):
            s = layer_stats.get(eid)
            hits = float(getattr(s, "hits", 0.0)) if s else 0.0
            base = hits / total_hits if total_hits > 0 else 0.0
            if self.importance == "usage_weight":
                base *= float(getattr(s, "avg_weight", 0.0)) if s else 0.0
            scores[eid] = base
        return scores

    def plan(self, threshold: float) -> dict[str, list[int]]:
        modules = dict(self.model.model.named_modules())
        plan = {}
        for layer_name, stats in self.usage_stats.items():
            parent = ".".join(layer_name.split(".")[:-1])
            moe = modules.get(parent)
            if moe is None or not hasattr(moe, "experts"):
                continue
            E = len(moe.experts)
            top_k = int(getattr(moe, "top_k", 1) or 1)
            raw_shares = self._scores_raw_shares(stats, E)
            degenerate = E > 1 and (max(raw_shares.values()) - min(raw_shares.values())) < 1e-6
            if degenerate and self.importance != "usage_weight":
                print(f"  [warn] {layer_name}: hit shares degenerate (dense router) - "
                      f"switching importance to usage_weight for this layer")
            importance = "usage_weight" if degenerate else self.importance
            saved = self.importance
            self.importance = importance
            scores = self._scores(stats, E)
            self.importance = saved
            # normalize scores to shares for thresholding comparability
            tot = sum(scores.values()) or 1.0
            shares = {i: v / tot for i, v in scores.items()}
            keep = [i for i, v in sorted(shares.items()) if v >= threshold]
            floor = max(self.min_keep or top_k, 1)
            if self.keep_top_m:
                floor = max(floor, self.keep_top_m)
            if len(keep) < floor:
                order = sorted(shares.items(), key=lambda kv: -kv[1])
                keep = sorted(int(i) for i, _ in order[:floor])
            plan[parent] = sorted(int(i) for i in keep)
        return plan

    def _scores_raw_shares(self, layer_stats, E):
        total = float(sum(getattr(s, "hits", 0.0) for s in layer_stats.values())) or 1.0
        return {i: float(getattr(layer_stats.get(i), "hits", 0.0) or 0.0) / total for i in range(E)}

    # ---------------- surgery ----------------

    @staticmethod
    def _find_projection(seq: nn.Sequential, num_experts: int):
        """Backward-scan for the projection layer; return (idx, trailing_norm_idxs)."""
        proj_idx = None
        for i in range(len(seq) - 1, -1, -1):
            m = seq[i]
            if isinstance(m, nn.Conv2d) and m.out_channels == num_experts:
                proj_idx = i
                break
            if isinstance(m, nn.Linear) and m.out_features == num_experts:
                proj_idx = i
                break
        if proj_idx is None:
            return None, []
        norm_idxs = [i for i in range(proj_idx + 1, len(seq))
                     if isinstance(seq[i], (nn.BatchNorm2d, nn.GroupNorm))
                     and getattr(seq[i], "num_features", getattr(seq[i], "num_channels", -1)) == num_experts]
        return proj_idx, norm_idxs

    def _prune_router(self, router: nn.Module, keep: list[int], E_old: int, layer: str):
        seq = None
        for attr in ("router", "routing_network"):
            cand = getattr(router, attr, None)
            if isinstance(cand, nn.Sequential):
                seq = cand
                break
        if seq is None:
            raise RuntimeError(f"{layer}: router has no 'router'/'routing_network' Sequential")
        proj_idx, norm_idxs = self._find_projection(seq, E_old)
        if proj_idx is None:
            raise RuntimeError(f"{layer}: projection layer with out=={E_old} not found - "
                               f"refusing to shrink experts without shrinking the router "
                               f"(upstream would silently corrupt here)")
        proj = seq[proj_idx]
        with torch.no_grad():
            if isinstance(proj, nn.Conv2d):
                new = nn.Conv2d(proj.in_channels, len(keep), proj.kernel_size, proj.stride,
                                proj.padding, bias=proj.bias is not None)
            else:
                new = nn.Linear(proj.in_features, len(keep), bias=proj.bias is not None)
            new.weight.data = proj.weight.data[keep].clone()
            if proj.bias is not None:
                new.bias.data = proj.bias.data[keep].clone()
            seq[proj_idx] = new
            for ni in norm_idxs:
                old_n = seq[ni]
                if isinstance(old_n, nn.BatchNorm2d):
                    nn_new = nn.BatchNorm2d(len(keep), eps=old_n.eps, momentum=old_n.momentum,
                                            affine=old_n.affine,
                                            track_running_stats=old_n.track_running_stats)
                    if old_n.affine:
                        nn_new.weight.data = old_n.weight.data[keep].clone()
                        nn_new.bias.data = old_n.bias.data[keep].clone()
                    if old_n.track_running_stats:
                        nn_new.running_mean.data = old_n.running_mean.data[keep].clone()
                        nn_new.running_var.data = old_n.running_var.data[keep].clone()
                        nn_new.num_batches_tracked.data = old_n.num_batches_tracked.data.clone()
                else:  # GroupNorm sized num_experts
                    g = min(old_n.num_groups, len(keep))
                    while g > 1 and len(keep) % g:
                        g -= 1
                    nn_new = nn.GroupNorm(g, len(keep), eps=old_n.eps, affine=old_n.affine)
                    if old_n.affine:
                        nn_new.weight.data = old_n.weight.data[keep].clone()
                        nn_new.bias.data = old_n.bias.data[keep].clone()
                seq[ni] = nn_new
        router.num_experts = len(keep)
        if hasattr(router, "top_k"):
            router.top_k = min(int(router.top_k), len(keep))

    def surgery(self, plan: dict[str, list[int]]):
        new_model = copy.deepcopy(self.model.model)
        modules = dict(new_model.named_modules())
        for parent, keep in plan.items():
            moe = modules[parent]
            E_old = len(moe.experts)
            if len(keep) == E_old:
                continue
            print(f"  {parent}: experts {E_old} -> {len(keep)} (keeping {keep})")
            moe.experts = nn.ModuleList([moe.experts[i] for i in keep])
            moe.num_experts = len(keep)
            if hasattr(moe, "top_k") and int(moe.top_k) > len(keep):
                moe.top_k = len(keep)
            if hasattr(moe, "_current_top_k") and moe._current_top_k is not None:
                try:
                    moe._current_top_k = min(int(moe._current_top_k), len(keep))
                except Exception:
                    pass
            self._prune_router(moe.routing, keep, E_old, parent)
            # per-expert state tensors (e.g. ES_MOE's expert_usage_counts buffer,
            # updated even at eval): shrink anything expert-indexed along dim 0
            idx = torch.tensor(keep, dtype=torch.long)
            for owner in (moe, moe.routing):
                for bname, buf in list(owner._buffers.items()):
                    if buf is not None and buf.dim() >= 1 and buf.shape[0] == E_old:
                        owner._buffers[bname] = buf[idx].clone()
            # post-surgery invariant
            pi, _ = self._find_projection(
                getattr(moe.routing, "router", None) or getattr(moe.routing, "routing_network"),
                len(keep))
            assert pi is not None, f"{parent}: router width != kept experts after surgery"
        return new_model

    def save(self, pruned, out_path: Path, threshold: float, plan: dict):
        src = torch.load(self.model_path, map_location="cpu", weights_only=False)
        ckpt = {"model": pruned, "updates": None,
                "train_args": src.get("train_args") if isinstance(src, dict) else None,
                "pruning_info": {"threshold": threshold, "importance": self.importance,
                                 "plan": {k: [int(i) for i in v] for k, v in plan.items()},
                                 "source": str(self.model_path)}}
        torch.save(ckpt, out_path)


def _params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def _gflops(model, imgsz: int) -> float:
    try:
        from ultralytics.utils.torch_utils import get_flops
        return round(float(get_flops(model, imgsz)), 3)
    except Exception:
        return float("nan")


def _fast_val(pt_path: str, data: str, args, dense_esmoe: bool) -> tuple[float, float]:
    from ultralytics import YOLO
    m = YOLO(pt_path)
    _fix_add_residual(m)             # fresh loads need the same legacy-checkpoint
    _strip_property_shadows(m)       # repairs as the pruner's own model
    if dense_esmoe:
        from ultralytics.nn.modules.moe.modules import ES_MOE
        for mod in m.model.modules():
            if isinstance(mod, ES_MOE):
                mod.use_sparse_inference = False
    r = m.val(data=data, split="val", batch=args.batch, imgsz=args.imgsz,
              device=args.device, workers=args.workers, verbose=False, plots=False)
    return float(r.box.map50), float(r.box.map)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="AI-TOD-v2.yaml")
    ap.add_argument("--thresholds", default="0.05,0.10,0.15,0.20,0.30")
    ap.add_argument("--importance", choices=["usage", "usage_weight"], default="usage")
    ap.add_argument("--keep-top-m", type=int, default=0)
    ap.add_argument("--min-keep", type=int, default=0, help="0 = layer top_k")
    ap.add_argument("--stats", default="", help="usage.json from diagnose_moe.py")
    ap.add_argument("--final-eval", choices=["fast", "aitod-official"], default="fast")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--max-images", type=int, default=0, help="subset for fast sweep deltas")
    ap.add_argument("--workers", type=int, default=0,
                    help="val dataloader workers (0 = safe for py3.14 local; use 8 on pods)")
    ap.add_argument("--out", default="")
    ap.add_argument("--map-budget", type=float, default=0.005,
                    help="allowed mAP50-95 drop when picking the best variant")
    args = ap.parse_args()

    stem = Path(args.model).stem
    out = Path(args.out or f"runs/project03/prune/{stem}")
    out.mkdir(parents=True, exist_ok=True)

    # resolve dataset yaml (name -> repo cfg path), optional subset for fast deltas
    sys.path.insert(0, str(Path(__file__).parent))
    from diagnose_moe import _resolve_dataset, _subset_yaml
    yaml_path, root, cfg = _resolve_dataset(args.data)
    tmp = tempfile.TemporaryDirectory()
    data_val = str(yaml_path)
    if args.max_images:
        data_val = _subset_yaml(yaml_path, root, cfg, "val", args.max_images, Path(tmp.name))
        print(f"[data] fast deltas on first {args.max_images} val images")

    pruner = Project03Pruner(args.model, data_val, args.device, args.imgsz, args.batch,
                             args.importance, args.min_keep, args.keep_top_m, args.workers)
    pruner.diagnose(args.stats or None, data_val)

    base_params = _params(pruner.model.model)
    base_flops = _gflops(pruner.model.model, args.imgsz)
    print(f"[base] params {base_params/1e6:.3f}M  GFLOPs {base_flops}")
    base_map50, base_map = _fast_val(args.model, data_val, args, pruner.has_esmoe)
    print(f"[base] mAP50 {base_map50:.4f}  mAP50-95 {base_map:.4f}")

    rows = []
    plan_cache: dict[str, dict] = {}
    for thr in [float(t) for t in args.thresholds.split(",") if t]:
        print(f"\n=== threshold {thr:.2f} ===")
        plan = pruner.plan(thr)
        (out / f"plan_t{thr:.2f}.json").write_text(json.dumps(plan, indent=2))
        plan_key = json.dumps(plan, sort_keys=True)
        if plan_key in plan_cache:
            cached = dict(plan_cache[plan_key])
            cached["threshold"] = thr
            rows.append(cached)
            print(f"  identical plan to t={plan_cache[plan_key]['threshold']} -> metrics reused (dedup)")
            continue
        pruned = pruner.surgery(plan)
        pt = out / f"{stem}_pruned_t{thr:.2f}.pt"
        pruner.save(pruned, pt, thr, plan)
        # reload check + 2-image predict smoke
        from ultralytics import YOLO
        chk = YOLO(str(pt))
        imgs = sorted((root / cfg["val"]).iterdir())[:2]
        chk.predict([str(i) for i in imgs], imgsz=args.imgsz, device=args.device,
                    verbose=False, save=False)
        p = _params(pruned)
        fl = _gflops(pruned, args.imgsz)
        m50, m = _fast_val(str(pt), data_val, args, pruner.has_esmoe)
        kept = {k.split(".")[-1]: len(v) for k, v in plan.items()}
        row = {"threshold": thr, "kept_experts": json.dumps(kept),
               "params_M": round(p / 1e6, 4),
               "dParams_pct": round(100 * (base_params - p) / base_params, 2),
               "GFLOPs": fl,
               "mAP50": round(m50, 5), "mAP50_95": round(m, 5),
               "dmAP50_95": round(m - base_map, 5)}
        rows.append(row)
        plan_cache[plan_key] = row
        print(f"  params {row['params_M']}M ({row['dParams_pct']}% cut)  "
              f"GFLOPs {fl}  mAP50-95 {m:.4f} (d {row['dmAP50_95']:+.4f})")

    with open(out / "sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["threshold", "kept_experts", "params_M",
                                          "dParams_pct", "GFLOPs", "mAP50", "mAP50_95", "dmAP50_95"])
        w.writeheader()
        w.writerows([{**r} for r in rows])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([r["dParams_pct"] for r in rows], [r["dmAP50_95"] for r in rows], "o-")
    for r in rows:
        ax.annotate(f"t={r['threshold']}", (r["dParams_pct"], r["dmAP50_95"]), fontsize=8)
    ax.axhline(-args.map_budget, ls="--", color="red", lw=0.8)
    ax.set_xlabel("params reduction %")
    ax.set_ylabel("dmAP50-95")
    ax.set_title(f"{stem}: pruning sweep (base {base_map:.4f})")
    fig.tight_layout()
    fig.savefig(out / "sweep.png", dpi=130)

    ok = [r for r in rows if r["dmAP50_95"] >= -args.map_budget]
    best = max(ok, key=lambda r: r["dParams_pct"]) if ok else None
    summary = {"model": args.model, "base": {"params_M": round(base_params / 1e6, 4),
               "GFLOPs": base_flops, "mAP50": round(base_map50, 5), "mAP50_95": round(base_map, 5)},
               "rows": rows, "best_within_budget": best}
    print(f"\n[best within {args.map_budget} budget] {best}")

    if args.final_eval == "aitod-official" and best is not None:
        pt = out / f"{stem}_pruned_t{best['threshold']:.2f}.pt"
        print("[final] official AI-TOD eval on best variant (this is slow on CPU)...")
        r = subprocess.run([sys.executable, "/data/YOLO-Master/scripts/reproduce/eval_aitod.py",
                            "--weights", str(pt), "--split", "val", "--device", args.device],
                           capture_output=True, text=True)
        (out / "final_official_eval.log").write_text(r.stdout + r.stderr)
        summary["official_eval_log"] = "final_official_eval.log"

    (out / "final.json").write_text(json.dumps(summary, indent=2))
    print(f"\nartifacts -> {out}")


if __name__ == "__main__":
    main()
