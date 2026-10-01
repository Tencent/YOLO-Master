#!/usr/bin/env python3
"""Export the v26.08 mixture families (MoA / MoT / MoLoRA) to ONNX for edge validation.

Runs inside a venv that has torch/onnx/onnxruntime but NO ultralytics installed;
--repo points at a YOLO-Master v26.08+ checkout and is prepended to sys.path, so the
research environment's editable install is never touched.

Produces, under --out:
  moa-n.onnx / mot-n.onnx / moa-mot-n.onnx      v0_10 lineage, [1, 4+nc, anchors]
  yolo26-moa-n-e2e.onnx                          yolo26 lineage, end2end [1, max_det, 6]
  yolo26-moa-n-anchors.onnx                      same cfg exported with end2end=False
  molora-merged.onnx                             MoLoRA(uniform-merged) on the MoA base
  molora-routed.onnx                             MoLoRA molora_export_mode=routing_preserved
plus, per model: <stem>.metadata.yaml (names/imgsz/end2end sidecar for MNN/ncnn) and
<stem>.ref.npz (eager predict reference on --ref-images: xyxy/conf/cls per image, the
ground truth for tests/validate_mixture.py).

Weights are random-init (upstream ships none for these families) - detections are
arbitrary but deterministic (seeded), which is exactly what pipeline/parity validation
needs. Pass --base <pt> to try wrapping real weights for the MoLoRA pair instead.
"""
import argparse
import sys
from pathlib import Path


def lower_isinf_isnan(onnx_path):
    """Rewrite IsInf/IsNaN (v26.08 router NaN-recovery guards) into universally supported
    ops - MNN has converters for neither. Exact lowerings, no semantics change:
      IsNaN(x) -> Not(Equal(x, x))          (NaN is the only value != itself)
      IsInf(x) -> Greater(Abs(x), 3.0e38)   (beyond float32 finite range)
    Applied in place; ORT/ncnn consume the rewritten graph identically.
    """
    import onnx
    from onnx import helper, numpy_helper
    import numpy as np
    m = onnx.load(onnx_path)
    g = m.graph
    new_nodes, n_low = [], 0
    big = None
    for node in g.node:
        if node.op_type == "IsNaN":
            eq = node.output[0] + "_eq"
            new_nodes.append(helper.make_node("Equal", [node.input[0], node.input[0]], [eq]))
            new_nodes.append(helper.make_node("Not", [eq], [node.output[0]]))
            n_low += 1
        elif node.op_type == "IsInf":
            if big is None:
                big = "edge_isinf_threshold"
                g.initializer.append(numpy_helper.from_array(
                    np.array(3.0e38, np.float32), big))
            ab = node.output[0] + "_abs"
            cf = node.output[0] + "_absf"
            new_nodes.append(helper.make_node("Abs", [node.input[0]], [ab]))
            # guards can run on double tensors; float cast keeps Greater single-typed
            # (double inf casts to float inf, so the comparison is unaffected)
            new_nodes.append(helper.make_node("Cast", [ab], [cf], to=1))  # 1 = FLOAT
            new_nodes.append(helper.make_node("Greater", [cf, big], [node.output[0]]))
            n_low += 1
        else:
            new_nodes.append(node)
    if n_low:
        del g.node[:]
        g.node.extend(new_nodes)
        onnx.save(m, onnx_path)
    return n_low


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="YOLO-Master v26.08 checkout (prepended to sys.path)")
    ap.add_argument("--out", default="models/mixture")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--ref-images", default="", help="dir with images for the predict reference (first --ref-n)")
    ap.add_argument("--ref-n", type=int, default=3)
    ap.add_argument("--conf", type=float, default=0.01, help="reference predict conf (low: random weights)")
    ap.add_argument("--base", default="", help="optional .pt to use as the MoLoRA base model")
    ap.add_argument("--only", default="", help="comma-separated subset of target names")
    ap.add_argument("--save-pt", action="store_true",
                    help="also dump each target's state_dict to <out>/<name>.pt so other "
                         "envs (e.g. the CoreML py3.11/torch2.5.1 venv) can rebuild the "
                         "architecture from cfg and load bit-identical weights instead of "
                         "trusting cross-torch-version RNG reproducibility")
    return ap.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.repo)
    import numpy as np
    import torch
    from ultralytics import YOLO, __version__

    print(f"[env] ultralytics {__version__} from {args.repo}")
    assert not __version__.startswith("8.3."), "old-lineage ultralytics resolved - wrong sys.path"
    torch.manual_seed(0)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    ref_imgs = []
    if args.ref_images:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        ref_imgs = sorted(p for p in Path(args.ref_images).iterdir() if p.suffix.lower() in exts)[: args.ref_n]
        print(f"[ref] {len(ref_imgs)} reference images")

    V010 = "ultralytics/cfg/models/master/v0_10/det"
    Y26 = "ultralytics/cfg/models/26"

    def spice_head(y):
        """Make the untrained model emit real, WELL-SEPARATED detections.
        Two failure modes of raw random init, both of which make validation vacuous or
        meaningless: (1) class biases start near log(1e-5), so nothing ever crosses a
        confidence threshold (empty-set comparisons pass trivially); (2) deep random
        nets attenuate input signal, so logits are bias-dominated CONSTANTS per class -
        thousands of exact score ties make topk/NMS ordering arbitrary per
        implementation and box-level parity impossible. Fix: spread the biases AND
        amplify the final conv weights of the class and box branches so scores/boxes
        vary spatially. Deterministic (fixed generator), both one2many and one2one."""
        import torch.nn as nn
        det = y.model.model[-1]
        g = torch.Generator().manual_seed(42)
        # class branch: strong amplification for score separation; box branch: mild
        # (x30 produced frame-sized boxes whose NMS survivor sets diverge chaotically)
        for name, brange, wmul in (("cv3", (-4.0, 0.5), 30.0), ("one2one_cv3", (-4.0, 0.5), 30.0),
                                   ("cv2", (0.0, 1.0), 3.0), ("one2one_cv2", (0.0, 1.0), 3.0)):
            branch = getattr(det, name, None)
            if branch is None:
                continue
            for seq in branch:
                last = [m for m in seq.modules() if isinstance(m, nn.Conv2d)][-1]
                last.weight.data.mul_(wmul)
                if last.bias is not None:
                    last.bias.data = torch.empty_like(last.bias).uniform_(*brange, generator=g)
        return y

    def build(cfg):
        torch.manual_seed(0)                      # deterministic random init per target
        return spice_head(YOLO(str(Path(args.repo) / cfg)))

    def build_molora(merged):
        from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
        y = build(f"{V010}/yolo-master-moa-n.yaml") if not args.base else YOLO(args.base)
        torch.manual_seed(1)
        mol = MoLoRAModel(y.model, MoLoRAConfig(r=8, alpha=16, num_experts=4, top_k=2))
        if merged:
            mol.merge(mode="uniform")             # calibration-free; works untrained
        y.model = mol.model
        return y

    TARGETS = {
        "moa-n":                (lambda: build(f"{V010}/yolo-master-moa-n.yaml"), {}),
        "mot-n":                (lambda: build(f"{V010}/yolo-master-mot-n.yaml"), {}),
        "moa-mot-n":            (lambda: build(f"{V010}/yolo-master-moa-mot-n.yaml"), {}),
        "yolo26-moa-n-e2e":     (lambda: build(f"{Y26}/yolo26-master-moa-n.yaml"), {}),
        "yolo26-moa-n-anchors": (lambda: build(f"{Y26}/yolo26-master-moa-n.yaml"), {"end2end": False}),
        "molora-merged":        (lambda: build_molora(True), {}),
        "molora-routed":        (lambda: build_molora(False), {"molora_export_mode": "routing_preserved"}),
    }
    only = {s for s in args.only.split(",") if s} or set(TARGETS)

    results = {}
    for name, (factory, kw) in TARGETS.items():
        if name not in only:
            continue
        print(f"\n=== {name} ===")
        try:
            y = factory()
            if args.save_pt:
                # post-factory state: spiced head, and for molora targets the wrapped
                # (and for molora-merged, merged) module tree - the loader must recreate
                # the same structure before load_state_dict(strict=True)
                torch.save(y.model.state_dict(), out / f"{name}.pt")
                print(f"  state_dict -> {name}.pt")
            onnx_path = y.export(format="onnx", imgsz=args.imgsz, device="cpu",
                                 simplify=False, dynamic=False, **kw)
            dst = out / f"{name}.onnx"
            Path(onnx_path).replace(dst)
            n_low = lower_isinf_isnan(dst)
            if n_low:
                print(f"  lowered {n_low} IsInf/IsNaN guard(s) for MNN compatibility")

            # sidecar for MNN/ncnn conversions (parsed by the edge runtime's read_ncnn_yaml)
            names = y.model.names if isinstance(y.model.names, dict) else dict(enumerate(y.model.names))
            e2e = bool(getattr(y.model.model[-1], "end2end", False)) and kw.get("end2end", True) is not False
            with open(out / f"{name}.metadata.yaml", "w") as f:
                f.write(f"imgsz:\n- {args.imgsz}\n- {args.imgsz}\n")
                f.write(f"end2end: {str(e2e).lower()}\n")
                f.write("names:\n")
                for k in sorted(names):
                    f.write(f"  {k}: {names[k]}\n")

            # eager predict reference (the exported-vs-eager ground truth for the harness).
            # If the export disabled end2end, the eager reference must too - otherwise
            # predict uses the one2one branch while the export carries one2many.
            if kw.get("end2end", True) is False:
                y.model.model[-1].end2end = False
            if ref_imgs:
                # Torch-model predict letterboxes to the MINIMAL stride rectangle (rect
                # inference, e.g. 640x384) while the edge CLI always runs the square
                # 640x640 canvas - a different network input, so boxes can never match.
                # Feed predict a pre-letterboxed square canvas (the CLI's exact math)
                # and map its boxes back to original-image coordinates.
                import cv2
                ref = {}
                for img in ref_imgs:
                    bgr = cv2.imread(str(img))
                    h, w = bgr.shape[:2]
                    r = min(args.imgsz / w, args.imgsz / h)
                    nw, nh = round(w * r), round(h * r)
                    px, py = (args.imgsz - nw) // 2, (args.imgsz - nh) // 2
                    canvas = np.full((args.imgsz, args.imgsz, 3), 114, np.uint8)
                    canvas[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh))
                    res = y.predict(canvas, imgsz=args.imgsz, conf=args.conf, iou=0.7,
                                    max_det=300, device="cpu", verbose=False)[0]
                    b = res.boxes
                    xyxy = b.xyxy.cpu().numpy()
                    xyxy = (xyxy - [px, py, px, py]) / r          # canvas -> original px
                    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w)
                    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h)
                    ref[img.stem + "_xyxy"] = xyxy
                    ref[img.stem + "_conf"] = b.conf.cpu().numpy()
                    ref[img.stem + "_cls"] = b.cls.cpu().numpy()
                    # RAW eager output on the same canvas: the tie-immune ground truth.
                    # Random-init mixture trunks can collapse activations to constants,
                    # making box-set comparisons ill-defined at topk/NMS boundaries;
                    # raw-tensor parity is stronger and unaffected.
                    blob = torch.from_numpy(
                        canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
                    ).contiguous()
                    # Compute the raw reference under EXPORT semantics: v26.08 declares
                    # dense fallback for routed modules (MoT eager top-k differs from its
                    # exported full-softmax blend BY DESIGN), so patch the export flag
                    # for this forward to reproduce exactly what the ONNX graph computes.
                    y.model.eval()
                    _orig = torch.onnx.is_in_onnx_export
                    torch.onnx.is_in_onnx_export = lambda: True
                    try:
                        with torch.no_grad():
                            raw = y.model(blob)
                    finally:
                        torch.onnx.is_in_onnx_export = _orig
                    while isinstance(raw, (list, tuple)):
                        raw = raw[0]
                    ref[img.stem + "_raw"] = raw.cpu().numpy()
                np.savez(out / f"{name}.ref.npz", **ref)

            results[name] = f"OK ({dst.stat().st_size // 1024} KB)"
        except Exception as e:  # noqa: BLE001 - a per-target failure must not kill the batch
            results[name] = f"FAILED: {type(e).__name__}: {e}"
            import traceback
            traceback.print_exc()

    print("\n===== export summary =====")
    for name, res in results.items():
        print(f"  {name:24s} {res}")
    return 0 if all(r.startswith("OK") for r in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
