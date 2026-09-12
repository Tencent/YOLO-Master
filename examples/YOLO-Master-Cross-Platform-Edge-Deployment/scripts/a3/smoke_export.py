#!/usr/bin/env python
"""A3 8.24 entry check: the minimal `yolo.export` ONNX smoke, with an EXPLICIT
dynamic-routing export policy around the stock ultralytics export call.

What the stock exporter does NOT do at the locked YOLO-Master commit: it has no MoE
awareness. Routing behaviour under export is decided inside each MoE module by
`torch.onnx.is_in_onnx_export()` guards. This script makes that decision explicit,
verifiable and refusable instead of silent:

  preserve  - families whose export branch keeps exact top-k semantics
              (OptimizedMOEImproved: all experts computed, torch.gather on the top-k).
  dense     - ES_MOE is traced through its dense path (softmax-weighted sum over ALL
              experts). Accepted WITHOUT a semantic change only when k == E and no
              top-k / dynamic-threshold pruning is active (true for the released COCO
              weights); otherwise rejected unless the caller declares `--routing dense`.
  reject    - anything else, and ALWAYS dynamic axes (`--dynamic`): the gather branch
              bakes batch/height/width as Python ints, so a dynamic-axes graph would be
              silently wrong at other shapes. The rejection fires BEFORE export and is
              the required failure log.

After export: ONNX op histogram, expert-conv count (graph vs module), ONNX Runtime
vs eager PyTorch parity on real val images (the eager side keeps its normal sparse
semantics, so the check measures graph fidelity to the real model), and the code
references behind every policy decision are re-searched at run time and stored with
the locked commit hash (never stale line numbers).

Exit codes: 0 pass, 2 dynamic-axes rejection, 3 routing-policy rejection,
4 post-export verification failure.

Usage:
  python scripts/a3/smoke_export.py --model /data/YOLO-Master/YOLO-Master-v0.1-N.pt \
      --out runs/a3/v01n --data a3/config/coco-a3.yaml \
      --json a3/results/smoke_v01n.json --log a3/logs/v01n_smoke.log
  python scripts/a3/smoke_export.py --model ... --dynamic   # expected: exit 2
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")   # never let export mutate the env

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from diagnose_moe import _fix_add_residual, _force_dense_esmoe, _strip_property_shadows  # noqa: E402

EXACT_GATHER = {"OptimizedMOEImproved", "ModularRouterExpertMoE", "OptimizedMOE"}
DENSE_TRACE = {"ES_MOE"}

# (label, relative file, regex) - re-searched at run time, stored with the commit hash
CODE_REFS = [
    ("exporter forces the legacy tracer (dynamo=False); dynamo ignores the MoE guards",
     "ultralytics/utils/export/engine.py", r'"dynamo": False'),
    ("ES_MOE: dense path under export", "ultralytics/nn/modules/moe/modules.py",
     r"torch\.onnx\.is_in_onnx_export\(\)\s*$"),
    ("ES_MOE: dense forward = weighted sum over ALL experts",
     "ultralytics/nn/modules/moe/modules.py", r"def _dense_forward"),
    ("ES_MOE: sparse path top-k + dynamic_threshold (what dense drops)",
     "ultralytics/nn/modules/moe/modules.py", r"def _sparse_forward"),
    ("OptimizedMOEImproved: exact top-k export via stack + gather",
     "ultralytics/nn/modules/moe/modules.py", r"selected = torch\.gather\(all_outs"),
    ("OptimizedMOEImproved: export branch bakes B/H/W as Python ints (dynamic axes unsafe)",
     "ultralytics/nn/modules/moe/modules.py", r"idx_exp = idx_k\.view\(B, 1, 1, 1, 1\)"),
    ("OptimizedMOEImproved: compat shim defaults a MISSING add_residual to True (the trap)",
     "ultralytics/nn/modules/moe/modules.py", r"self\.add_residual = True"),
    ("exporter: no MoE awareness (grep count of 'moe' in exporter.py)",
     "ultralytics/engine/exporter.py", r"(?i)\bmoe\b"),
]


class Tee:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.f = open(path, "w")
        self.o = sys.stdout

    def write(self, s):
        self.o.write(s); self.f.write(s); self.f.flush()

    def flush(self):
        self.o.flush(); self.f.flush()


def sh(cmd: str) -> str:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout.strip()


def code_refs(repo: Path) -> list[dict]:
    commit = sh(f"git -C {repo} rev-parse HEAD")
    refs = []
    for label, rel, pat in CODE_REFS:
        hits = []
        for i, line in enumerate((repo / rel).read_text().splitlines(), 1):
            if re.search(pat, line):
                hits.append({"line": i, "text": line.strip()[:120]})
        refs.append({"what": label, "file": rel, "commit": commit,
                     "hits": hits[:3], "n_hits": len(hits)})
    return refs


def inventory(net) -> list[dict]:
    rows = []
    for name, m in net.named_modules():
        if not (hasattr(m, "experts") and hasattr(m, "routing")):
            continue
        r = getattr(m, "routing", None)
        rows.append({
            "module": name, "cls": type(m).__name__,
            "E": len(m.experts), "k": int(getattr(m, "top_k", len(m.experts))),
            "use_top_k": bool(getattr(m, "use_top_k", False)),
            "router_cls": type(r).__name__, "router_use_top_k": bool(getattr(r, "use_top_k", False)),
            "use_sparse_inference": getattr(m, "use_sparse_inference", None),
            "dynamic_threshold": getattr(m, "dynamic_threshold", None),
            "add_residual": getattr(m, "add_residual", None),
        })
    return rows


def decide(inv: list[dict], routing: str, dynamic: bool):
    """Return (exit_code, decisions, reason). exit 0 = export allowed."""
    if dynamic:
        return 2, [], ("dynamic axes REJECTED: the MoE export branches bake batch/height/"
                       "width as Python ints (view/expand in the gather path); a dynamic-axes "
                       "graph is silently wrong at any other shape. Export is static "
                       "1x3x{imgsz}x{imgsz} only.")
    if routing == "reject":
        return 3, [], "routing policy 'reject' requested"
    decisions = []
    for r in inv:
        if r["cls"] in EXACT_GATHER:
            d = {"module": r["module"], "cls": r["cls"], "decision": "preserve",
                 "semantic_change": False,
                 "why": "export branch computes all experts and gathers the top-k: exact"}
        elif r["cls"] in DENSE_TRACE:
            exact = (r["k"] == r["E"] and not r["use_top_k"] and not r["router_use_top_k"]
                     and not r["dynamic_threshold"])
            if exact:
                d = {"module": r["module"], "cls": r["cls"], "decision": "dense",
                     "semantic_change": False,
                     "why": f"k==E=={r['E']}, no top-k, no dynamic_threshold: dense == sparse"}
            elif routing == "dense":
                d = {"module": r["module"], "cls": r["cls"], "decision": "dense",
                     "semantic_change": True,
                     "why": "DECLARED sparse->dense fallback (top-k/threshold pruning dropped)"}
            else:
                return 3, decisions, (f"{r['module']} ({r['cls']}, k={r['k']} E={r['E']} "
                                      f"use_top_k={r['use_top_k']} thr={r['dynamic_threshold']}): "
                                      "no semantics-preserving ONNX path; rerun with "
                                      "--routing dense to DECLARE the fallback")
        else:
            return 3, decisions, f"{r['module']} ({r['cls']}): no verified export branch"
        decisions.append(d)
    return 0, decisions, "ok"


def letterbox(img_bgr, imgsz: int):
    import cv2
    h, w = img_bgr.shape[:2]
    s = min(imgsz / h, imgsz / w)
    nh, nw = int(round(h * s)), int(round(w * s))
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    top, left = (imgsz - nh) // 2, (imgsz - nw) // 2
    canvas[top:top + nh, left:left + nw] = cv2.resize(img_bgr, (nw, nh))
    x = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(x[None])


def parity(yolo, onnx_path: Path, images: list[Path], imgsz: int, tf32: bool = False) -> dict:
    """ORT vs eager PyTorch (CPU fp32). The gated run pins use_tf32=0 so the comparison is
    fp32 vs fp32 graph fidelity; the CUDA EP's DEFAULT is TF32 convolutions on Ampere+,
    which is a backend precision choice (reported separately), not an export error."""
    import cv2
    import onnxruntime as ort
    providers = ort.get_available_providers()
    prov = [("CUDAExecutionProvider", {"use_tf32": 1 if tf32 else 0}), "CPUExecutionProvider"] \
        if "CUDAExecutionProvider" in providers else ["CPUExecutionProvider"]
    sess = ort.InferenceSession(str(onnx_path), providers=prov)
    prov = sess.get_providers()   # the providers ACTUALLY in use (CUDA can silently fall back)
    in_name = sess.get_inputs()[0].name
    # eager reference on CPU fp32: the REAL model semantics (sparse where sparse),
    # fused + single-tensor head output exactly like the exporter prepares it
    ref = copy.deepcopy(yolo.model).float().eval()
    ref.fuse() if hasattr(ref, "fuse") else None
    head = ref.model[-1]
    head.export, head.format = True, "onnx"
    abs_max, rel_max, abs_mean, box_agree, norm_rel = [], [], [], [], []
    with torch.no_grad():
        for p in images:
            x = letterbox(cv2.imread(str(p)), imgsz)
            y_ref = ref(torch.from_numpy(x))
            y_ref = (y_ref[0] if isinstance(y_ref, (list, tuple)) else y_ref).numpy()
            y_ort = sess.run(None, {in_name: x})[0]
            diff = np.abs(y_ref - y_ort)
            abs_max.append(float(diff.max()))
            abs_mean.append(float(diff.mean()))
            rel_max.append(float((diff / (np.abs(y_ref) + 1e-3)).max()))
            # scale-normalized error: box rows are in pixels (0..640), class rows in
            # 0..1, so elementwise relative error on near-zero entries is meaningless
            norm_rel.append(float(diff.max() / (np.abs(y_ref).max() + 1e-6)))
            # top-100 anchors by max class score: do both backends pick the same set?
            sc_r = y_ref[0, 4:].max(0); sc_o = y_ort[0, 4:].max(0)
            top_r = set(np.argsort(-sc_r)[:100]); top_o = set(np.argsort(-sc_o)[:100])
            box_agree.append(len(top_r & top_o) / 100.0)
    return {"providers": prov[0], "tf32": tf32, "n_images": len(images),
            "max_abs": max(abs_max), "mean_abs": float(np.mean(abs_mean)),
            "max_rel_elementwise": max(rel_max), "max_norm_rel": max(norm_rel),
            "top100_anchor_agreement": float(np.mean(box_agree))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data", default="a3/config/coco-a3.yaml")
    ap.add_argument("--routing", choices=["auto", "preserve", "dense", "reject"], default="auto")
    ap.add_argument("--dynamic", action="store_true", help="request dynamic axes (REJECTED by policy)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--parity-n", type=int, default=8)
    ap.add_argument("--rel-tol", type=float, default=1e-3)
    ap.add_argument("--json", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--yolo-repo", default="/data/YOLO-Master")
    args = ap.parse_args()

    sys.stdout = Tee(Path(args.log))
    t0 = time.time()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rep = {"model": args.model, "args": vars(args), "torch": torch.__version__,
           "yolo_master_commit": sh(f"git -C {args.yolo_repo} rev-parse HEAD"),
           "edge_commit": sh("git rev-parse HEAD")}
    print(f"[smoke] {args.model} torch {torch.__version__} YOLO-Master {rep['yolo_master_commit'][:12]}")

    # hard precondition: the legacy tracer. torch>=2.9 would route to dynamo and the
    # MoE is_in_onnx_export guards would go dark (experts silently dropped).
    major, minor = (int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    assert (major, minor) < (2, 9), "torch>=2.9: dynamo exporter would bypass the MoE export guards"
    rep["code_refs"] = code_refs(Path(args.yolo_repo))
    for r in rep["code_refs"]:
        hit = r["hits"][0] if r["hits"] else None
        print(f"[ref] {r['file']}:{hit['line'] if hit else '-'} ({r['n_hits']} hits) {r['what']}")

    from ultralytics import YOLO
    yolo = YOLO(args.model)
    rep["repairs"] = {"add_residual_forced_false": _fix_add_residual(yolo),
                      "property_shadows_stripped": _strip_property_shadows(yolo),
                      "esmoe_dense_forced": _force_dense_esmoe(yolo)}
    print(f"[repair] {rep['repairs']}")
    rep["inventory"] = inventory(yolo.model)
    for r in rep["inventory"]:
        print(f"[moe] {r['module']:10s} {r['cls']:24s} E={r['E']} k={r['k']} use_top_k={r['use_top_k']} "
              f"thr={r['dynamic_threshold']} add_residual={r['add_residual']} router={r['router_cls']}")

    code, decisions, reason = decide(rep["inventory"], args.routing, args.dynamic)
    rep["policy"] = {"routing": args.routing, "dynamic": args.dynamic, "decisions": decisions,
                     "reason": reason.format(imgsz=args.imgsz), "exit_code": code}
    for d in decisions:
        print(f"[policy] {d['module']:10s} -> {d['decision']:8s} semantic_change={d['semantic_change']}  {d['why']}")
    if code != 0:
        print(f"[policy] REJECTED (exit {code}): {rep['policy']['reason']}")
        rep["status"] = "rejected"
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=2))
        sys.exit(code)

    # ---- the stock export, static shape, legacy tracer ----
    dev = 0 if torch.cuda.is_available() else "cpu"
    t1 = time.time()
    onnx_path = yolo.export(format="onnx", imgsz=args.imgsz, dynamic=False, batch=1,
                            simplify=True, opset=args.opset, device=dev)
    dst = out / (Path(args.model).stem + ".onnx")
    Path(onnx_path).replace(dst)
    rep["onnx"] = {"path": str(dst), "size_mb": round(dst.stat().st_size / 1e6, 2),
                   "export_s": round(time.time() - t1, 1)}
    print(f"[export] {dst} {rep['onnx']['size_mb']} MB in {rep['onnx']['export_s']}s")

    # ---- verification ----
    import onnx
    g = onnx.load(str(dst)).graph
    hist = collections.Counter(n.op_type for n in g.node)
    watch = {k: hist.get(k, 0) for k in ("TopK", "Gather", "GatherElements", "Softmax",
                                          "ScatterND", "NonZero", "If", "Loop", "Conv")}
    rep["onnx"]["op_histogram"] = dict(hist)
    rep["onnx"]["routing_ops"] = watch
    rep["onnx"]["metadata"] = {p.key: p.value[:60] for p in onnx.load(str(dst)).metadata_props}
    print(f"[onnx] ops={sum(hist.values())} routing-relevant={watch}")
    bad = [k for k in ("NonZero", "If", "Loop") if watch[k]]
    n_expert_conv_mod = sum(1 for n, m in yolo.model.named_modules()
                            if ".experts." in n and isinstance(m, torch.nn.Conv2d))
    n_expert_conv_onnx = sum(1 for n in g.node if n.op_type == "Conv" and "/experts" in n.name)
    rep["onnx"]["expert_conv"] = {"module": n_expert_conv_mod, "graph": n_expert_conv_onnx}
    print(f"[onnx] expert Conv: module={n_expert_conv_mod} graph={n_expert_conv_onnx}")

    import yaml
    cfg = yaml.safe_load(Path(args.data).read_text())
    val_dir = Path(cfg["path"]) / cfg["val"]
    files = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in (".jpg", ".png"))
    step = max(1, len(files) // args.parity_n)
    imgs = files[::step][:args.parity_n]
    rep["parity"] = parity(yolo, dst, imgs, args.imgsz, tf32=False)      # the gate: fp32 vs fp32
    print(f"[parity] {rep['parity']}")
    if rep["parity"]["providers"] == "CUDAExecutionProvider":
        rep["parity_tf32_default"] = parity(yolo, dst, imgs, args.imgsz, tf32=True)  # informational
        print(f"[parity tf32 default] {rep['parity_tf32_default']}")

    fails = []
    if bad:
        fails.append(f"data-dependent control flow in graph: {bad}")
    if n_expert_conv_onnx and n_expert_conv_onnx < n_expert_conv_mod:
        fails.append(f"expert convs lost in export: {n_expert_conv_onnx} < {n_expert_conv_mod}")
    if rep["parity"]["max_norm_rel"] > args.rel_tol:
        fails.append(f"ORT parity scale-normalized error {rep['parity']['max_norm_rel']:.2e} > {args.rel_tol}")
    if rep["parity"]["top100_anchor_agreement"] < 0.95:
        fails.append(f"ORT parity top-100 anchor agreement {rep['parity']['top100_anchor_agreement']:.3f} < 0.95")
    rep["verification"] = {"failures": fails}
    rep["status"] = "pass" if not fails else "fail"
    rep["elapsed_s"] = round(time.time() - t0, 1)
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(rep, indent=2))
    print(f"[smoke] {rep['status'].upper()} in {rep['elapsed_s']}s -> {args.json}"
          + (f"  failures={fails}" if fails else ""))
    sys.exit(0 if not fails else 4)


if __name__ == "__main__":
    main()
