#!/usr/bin/env python
"""Export YOLO-Master DENSE checkpoints to static ONNX for ONNX Runtime's QNN (Hexagon HTP) provider.

Runs in /data/tmp/venv-v2608 (torch 2.13, onnx 1.22, onnxruntime 1.28, onnxslim) with
PYTHONPATH=/data/YOLO-Master (the fork, ultralytics 8.4.101), from the repo root.

The QNN EP wants a batch-1, static-shape fp32 graph (it runs it as fp16 on the HTP, or as A8W8 /
A16W8 QDQ after scripts/quantize_onnx_qnn.py) whose ops all have HTP builders: anything else is
partitioned to the CPU EP with a tensor transfer at every seam. `Erf`, `GroupNormalization`,
`NonMaxSuppression`, `Loop` and `If` have no builder at all (plan "Confirmed facts"). The stock
`format=onnx` export of these checkpoints is already static and free of those, but it carries the
same glue the ncnn export did: the MoE router's `[B,E,1,1].repeat(1,1,H,W)` lands as `Tile` +
full-map `Mul`s, and the area attention is spelled as ~7 Transpose/Reshape per block around the
MatMul/Softmax pair. This driver traces with the export-time forward rewrites of
scripts/export_ncnn_dense.py active (R1 scale folded into the qkv conv, R2 attention as one conv
-> split -> `F.scaled_dot_product_attention` -> NCHW, R3 router weights left unexpanded) and lets
the fork's own ONNX exporter (legacy tracer; `torch2onnx` forces `dynamo=False`, which is what
makes the `torch.jit.is_tracing()` guards of R2/R3 fire) plus onnxslim do the rest. At opset 17
SDPA decomposes back to MatMul + Softmax, which is exactly the form the HTP builders take, so the
ONNX-relevant gain is R3 (no Tile, gate broadcast natively) and the leaner attention chain.

The released v0.1-N COCO checkpoint needs the two shims documented in export_ncnn_dense.py:
`repair_legacy_v01_det` (missing `add_residual` / `capacity_factor`) and `det_dispatch_shim`
(top-k expert SET frozen for the trace input, expert WEIGHTS dynamic). Both are re-applied here
unchanged; the baked-vs-probe routing is printed, not hidden.

Every export is validated in place: `onnx.checker`, static I/O shapes, op census with a hard fail
on the QNN-unsupported set (and on `Tile` when the rewrites ran - R3 must have fired), then an ORT
CPU-EP run on the bench probe (results/int8_bench/probe_coco.jpg letterboxed exactly like
cpp/src/common.cpp) against the fp32 PyTorch model: scaled maxdiff (max of box/imgsz, class, rest)
within 1e-4 as the ncnn exporter gates, boxes of scoring anchors within 2e-2 px, class scores within
1e-4, and the post-NMS count at conf 0.25 / IoU 0.5 (the C++ defaults) within +-1 of the ncnn
CLI's. The PyTorch probe outputs are saved next to the graph as `model.ref.npz` for
tests/validate_mixture.py-style checks against other providers.

`--onnx` mode (no checkpoint on this pod, e.g. EsMoE-N VisDrone): take an existing static ONNX,
lift it to `--opset` with onnx.version_converter when that is numerically clean (ORT CPU, same
probe, same gates against the original), otherwise keep it as is and say so; then run the same
post-checks and write the same metadata.

Usage:
    PYTHONPATH=/data/YOLO-Master /data/tmp/venv-v2608/bin/python scripts/export_onnx_dense.py \\
        --pt /data/coreml_export/YOLO-Master-v0.1-seg-N.pt --out models/v0.1-seg-n_ncnn \\
        --compare-onnx models/archive/v0.1-seg-n-stock.onnx --expect-dets 15
    ... --pt /data/YOLO-Master/YOLO-Master-v0.1-N.pt --out models/v0.1-n_ncnn --expect-dets 14
    ... --onnx models/esmoe_n_visdrone_sim.onnx --out models/esmoe_n_visdrone_ncnn \\
        --probe results/int8_bench/probe_visdrone.jpg
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_ncnn_dense import (
    PROBE,
    _Outputs,
    as_tuple,
    bake_det_routing,
    dense_rewrites,
    det_count,
    det_dispatch_shim,
    install_aattn_convs,
    load_probe,
    out_diff,
    prepare_model,
)

# No HTP builder exists for these (plan "Confirmed facts", op_builder_factory.cc): a graph that
# contains one is not a QNN graph but a CPU graph with a QNN island.
QNN_UNSUPPORTED = ("Erf", "GroupNormalization", "NonMaxSuppression", "Loop", "If")
WATCH = ("Tile", "MatMul", "Softmax", "Transpose", "Reshape", "Mul", "Conv", "Resize", "Gather", "TopK", "Expand")
# C++ runtime defaults (cpp/include/yolomaster.hpp Config): argmax class, per-class greedy NMS.
CONF, IOU, MAX_DET = 0.25, 0.50, 300
# Box gate: 2e-2 px on the anchors the runtime emits (torch score >= CONF). Off-detection anchors show
# ORT-vs-torch conv accumulation noise up to ~2.3e-2 px on the STOCK export too (measured, seg-N anchor 7857,
# conf 0.000), so they are judged by the ncnn exporter's scaled gate (max(box/imgsz, cls, rest) <= 1e-4).
BOX_TOL_PX, CLS_TOL, REST_TOL, SCALED_TOL = 2e-2, 1e-4, 1e-3, 1e-4


# ---------------------------------------------------------------------------
# probe + numpy post-processing (mirrors cpp/src/common.cpp decode_candidates + nms_and_cap)
# ---------------------------------------------------------------------------


def probe_blob(probe: Path, imgsz: int) -> np.ndarray:
    """Letterboxed NCHW RGB/255 blob of `probe`; bit-checked against its `*_<imgsz>.f32` sibling when present."""
    if probe.resolve() == PROBE.resolve():
        return load_probe(imgsz)  # the ncnn exporter's path, incl. the probe_640.f32 equality assert
    from make_probe_f32 import preprocess

    blob, _ = preprocess(probe, imgsz)
    f32 = probe.with_name(f"{probe.stem}_{imgsz}.f32")
    if f32.exists():
        ref = np.fromfile(f32, dtype=np.float32).reshape(1, 3, imgsz, imgsz)
        assert np.array_equal(blob, ref), f"letterbox drifted from {f32}"
    return blob


def nms_count(out0: np.ndarray, nc: int, conf: float = CONF, iou: float = IOU, max_det: int = MAX_DET) -> int:
    """Post-NMS detection count the C++ core would report (argmax class, per-class greedy NMS, max_det cap).

    Works in letterbox pixels: the runtime un-letterboxes first, but IoU is invariant under the uniform
    scale + translation of the inverse letterbox, so the kept set is identical.
    """
    cls = out0[0, 4 : 4 + nc]
    best, score = cls.argmax(axis=0), cls.max(axis=0)
    sel = np.nonzero(score >= conf)[0]
    if sel.size == 0:
        return 0
    cx, cy, w, h = (out0[0, i, sel].astype(np.float64) for i in range(4))
    off = best[sel] * (2.0 * out0.shape[2] + 8192.0)  # class stratification, as nms_and_cap does
    x1, y1, x2, y2 = cx - w / 2 + off, cy - h / 2 + off, cx + w / 2 + off, cy + h / 2 + off
    order = np.argsort(-score[sel], kind="stable")
    dead = np.zeros(sel.size, bool)
    keep = 0
    for m, i in enumerate(order):
        if dead[i]:
            continue
        keep += 1
        j = order[m + 1 :]
        j = j[~dead[j]]
        xx1, yy1 = np.maximum(x1[i], x1[j]), np.maximum(y1[i], y1[j])
        xx2, yy2 = np.minimum(x2[i], x2[j]), np.minimum(y2[i], y2[j])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        uni = (x2[i] - x1[i]) * (y2[i] - y1[i]) + (x2[j] - x1[j]) * (y2[j] - y1[j]) - inter
        dead[j[np.where(uni > 0, inter / np.where(uni > 0, uni, 1), 0) > iou]] = True
    return min(keep, max_det)


# ---------------------------------------------------------------------------
# ONNX inspection
# ---------------------------------------------------------------------------


def op_census(model) -> Counter:
    return Counter(n.op_type for n in model.graph.node)


def census_line(c: Counter) -> str:
    return f"{sum(c.values())} nodes, {len(c)} types | " + " ".join(f"{t}={c.get(t, 0)}" for t in WATCH)


def io_shapes(model) -> list[tuple[str, list]]:
    """(name, dims) of every graph input/output; a dim is an int or the offending dim_param string."""
    out = []
    for v in list(model.graph.input) + list(model.graph.output):
        dims = [d.dim_value if d.dim_value > 0 else (d.dim_param or "?") for d in v.type.tensor_type.shape.dim]
        out.append((v.name, dims))
    return out


def check_graph(model, rewrite_ran: bool) -> Counter:
    """checker + static I/O + census; raises on QNN-unsupported ops (and on Tile when R3 should have fired)."""
    import onnx

    onnx.checker.check_model(model, full_check=True)
    shapes = io_shapes(model)
    bad = [(n, d) for n, d in shapes if any(not isinstance(x, int) for x in d)]
    if bad:
        raise RuntimeError(f"non-static graph I/O (QNN needs fixed shapes): {bad}")
    print("[graph] I/O " + ", ".join(f"{n}{tuple(d)}" for n, d in shapes))
    types = op_census(model)
    print(f"[graph] {census_line(types)}")
    hit = {t: types[t] for t in QNN_UNSUPPORTED if types.get(t)}
    if hit:
        raise RuntimeError(f"QNN-unsupported ops in graph: {hit}")
    if types.get("Tile"):
        msg = f"Tile x{types['Tile']} in graph"
        if rewrite_ran:
            # R3 replaces the router's .repeat under torch.jit.is_tracing(); a surviving Tile means the trace
            # did not go through the legacy tracer (dynamo) or the router class was not the patched one.
            raise RuntimeError(f"{msg}: R3 did not fire (dynamo exporter, or a router class dense_rewrites misses)")
        print(f"[graph] WARNING {msg} (stock trace, no rewrite: the router .repeat is kept; QNN has a Tile builder)")
    return types


def set_metadata(model, extra: dict[str, str]):
    """Add/overwrite metadata_props; the ultralytics keys (task, names, imgsz, stride, ...) are kept."""
    have = {p.key: p for p in model.metadata_props}
    for k, v in extra.items():
        if k in have:
            have[k].value = v
        else:
            p = model.metadata_props.add()
            p.key, p.value = k, v


def meta_dict(model) -> dict[str, str]:
    return {p.key: p.value for p in model.metadata_props}


def run_ort(path: Path, blob: np.ndarray) -> list[np.ndarray]:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    return sess.run(None, {sess.get_inputs()[0].name: blob})


def gate_outputs(got: list[np.ndarray], ref: list[np.ndarray], nc: int, imgsz: int, tag: str) -> None:
    """Fail unless: scaled maxdiff <= 1e-4 (all anchors), boxes of scoring anchors <= 2e-2 px, class scores
    <= 1e-4, mask coeffs / proto <= 1e-3 (abs)."""
    if len(got) != len(ref):
        raise RuntimeError(f"{tag}: {len(got)} outputs, reference has {len(ref)}")
    worst = []
    for i, (g, r) in enumerate(zip(got, ref)):
        if tuple(g.shape) != tuple(r.shape):
            raise RuntimeError(f"{tag}: out{i} shape {tuple(g.shape)} != reference {tuple(r.shape)}")
        txt, scaled = out_diff(g, r, nc, imgsz, i == 0)
        d = np.abs(g.astype(np.float64) - r.astype(np.float64))
        if i > 0:
            # out_diff leaves non-out0 outputs unscaled; the seg proto spans ~[-0.3, 5.8], so judge it against
            # its own magnitude the way boxes are judged against imgsz (1.3e-4 abs on a 5.8 proto is fp32 noise)
            scale = max(1.0, float(np.abs(r).max()))
            scaled = float(d.max()) / scale
            txt += f" (ref max |.| {scale:.2f})"
        worst.append((scaled, SCALED_TOL, f"out{i} scaled"))
        if i == 0:
            hot = r[0, 4 : 4 + nc].max(axis=0) >= CONF
            box_hot = float(d[0, :4][:, hot].max(initial=0.0))
            txt += f" | box on {int(hot.sum())} anchors >= {CONF}: {box_hot:.2e} (px)"
            worst += [
                (box_hot, BOX_TOL_PX, "box px (scoring anchors)"),
                (float(d[0, 4 : 4 + nc].max()), CLS_TOL, "cls"),
                (float(d[0, 4 + nc :].max(initial=0.0)), REST_TOL, "mask coeffs"),
            ]
        else:
            worst.append((float(d.max()), REST_TOL, f"out{i}"))
        print(f"[{tag}] out{i}{tuple(g.shape)}: {txt} | scaled {scaled:.2e}")
    fails = [f"{what} {v:.2e} > {tol:.0e}" for v, tol, what in worst if v > tol]
    if fails:
        raise RuntimeError(f"{tag}: numerics gate failed: " + "; ".join(fails))


def report_dets(got: list[np.ndarray], ref: list[np.ndarray], nc: int, expect: int | None, tag: str) -> None:
    n_ort, n_ref = nms_count(got[0], nc), nms_count(ref[0], nc)
    print(
        f"[{tag}] dets@{CONF} after NMS(iou {IOU}): {n_ort} (torch {n_ref}"
        + (f", ncnn CLI {expect}" if expect is not None else "")
        + f") | pre-NMS anchors {det_count(got[0], nc)} (torch {det_count(ref[0], nc)})"
    )
    target = expect if expect is not None else n_ref
    if abs(n_ort - target) > 1:
        raise RuntimeError(f"{tag}: detection count {n_ort} not within +-1 of {target}")


# ---------------------------------------------------------------------------
# checkpoint path
# ---------------------------------------------------------------------------


def export_from_pt(args, dst: Path, probe: torch.Tensor) -> tuple[Path, list[np.ndarray], int, str, dict]:
    """Trace `--pt` with the rewrites into `<dst>/model.onnx`; returns (path, torch ref outputs, nc, task, names)."""
    from ultralytics import YOLO, __version__
    from ultralytics.nn.modules import Detect

    print(f"[env] ultralytics {__version__} torch {torch.__version__}")
    assert not __version__.startswith("8.3."), "old-lineage ultralytics resolved - wrong PYTHONPATH"

    # prepare_model = load + legacy repair + fuse + Detect export flags (ncnn exporter parity; the Exporter
    # re-applies its own flags on its deepcopy, fuse() is a no-op on an already fused model).
    model, names, task = prepare_model(args.pt, args.imgsz)
    if args.task != "auto" and task != args.task:
        raise RuntimeError(f"--task {args.task} but checkpoint task is {task}")
    nc = len(names)
    for m in model.modules():
        if isinstance(m, Detect):
            m.format = "onnx"
    wrapped = _Outputs(model)
    ex = torch.zeros(1, 3, args.imgsz, args.imgsz)  # the Exporter traces on zeros; bake the routing for that
    routed = any(type(m).__name__ == "OptimizedMOEImproved" for m in model.modules())
    if routed:
        for note in bake_det_routing(model, ex, probe):
            print(f"  [bake] {note}")

    with torch.no_grad():
        ref = [t.numpy() for t in as_tuple(wrapped(probe))]
    print(f"[ref] {task} outputs: " + ", ".join(f"out{i}{t.shape}" for i, t in enumerate(ref)))
    maxconf = float(ref[0][0, 4 : 4 + nc].max())
    print(f"[ref] probe max class conf {maxconf:.3f}, pre-NMS dets@{CONF} = {det_count(ref[0], nc)}")
    if maxconf < 0.5:
        raise RuntimeError("reference max confidence < 0.5: checkpoint compat repair is wrong (see ncnn exporter)")

    rewrite = not args.no_rewrite
    if rewrite:
        n = install_aattn_convs(model)
        print(f"[rewrite] {n} AAttn qkv convs re-registered (rows permuted, q rows scaled by hd**-0.5)")
        with torch.no_grad(), dense_rewrites():
            got = [t.numpy() for t in as_tuple(wrapped(probe))]
        gate_outputs(got, ref, nc, args.imgsz, "selftest patched-vs-original eager")

    # The Exporter writes next to `pt_path`; point it into a scratch dir under `dst` (a symlink keeps
    # file_size() honest) so nothing lands beside the checkpoint in the research repo.
    tmp = dst / "_export_tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    link = tmp / Path(args.pt).name
    link.symlink_to(Path(args.pt).resolve())
    model.pt_path = str(link)
    y = YOLO(args.pt)
    y.model = model  # the repaired / fused / rewritten instance, not a fresh load

    ctx = contextlib.ExitStack()
    if routed:
        ctx.enter_context(det_dispatch_shim())
    if rewrite:
        ctx.enter_context(dense_rewrites())
    with ctx:
        produced = y.export(
            format="onnx",
            imgsz=args.imgsz,
            opset=args.opset,
            simplify=True,
            dynamic=False,
            half=False,
            batch=1,
            device="cpu",
            verbose=False,
        )
    out = dst / "model.onnx"
    shutil.move(str(produced), out)
    shutil.rmtree(tmp, ignore_errors=True)
    print(f"[export] {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return out, ref, nc, task, names


# ---------------------------------------------------------------------------
# existing-ONNX path (no checkpoint on this machine)
# ---------------------------------------------------------------------------


def export_from_onnx(args, dst: Path, blob: np.ndarray) -> tuple[Path, list[np.ndarray], int, str, dict]:
    """Copy `--onnx` to `<dst>/model.onnx`, lifted to `--opset` when the version converter is numerically clean."""
    import ast

    import onnx
    from onnx import version_converter

    src = Path(args.onnx)
    m = onnx.load(str(src))
    # The 8.3.240-era exports carry stale intermediate value_info (rank-0 shapes on the attention Transposes)
    # that ORT ignores but onnx.checker(full_check) and the version converter reject; value_info is optional
    # metadata, so drop it (the checker / converter re-infer shapes from scratch).
    del m.graph.value_info[:]
    meta = meta_dict(m)
    task = meta.get("task") or (args.task if args.task != "auto" else "detect")
    if args.task != "auto" and task != args.task:
        raise RuntimeError(f"--task {args.task} but ONNX metadata says {task}")
    names = ast.literal_eval(meta["names"]) if "names" in meta else {}
    nc = len(names) or (m.graph.output[0].type.tensor_type.shape.dim[1].dim_value - 4)
    have = next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))
    print(f"[onnx] {src} opset {have}, {census_line(op_census(m))}")
    onnx.checker.check_model(m, full_check=True)
    ref = run_ort(src, blob)  # the original graph IS the reference here (no torch model to compare against)

    out = dst / "model.onnx"
    converted = None
    if have < args.opset:
        try:
            cand = version_converter.convert_version(m, args.opset)
            onnx.checker.check_model(cand, full_check=True)
            tmp = dst / "_convert_tmp.onnx"
            onnx.save(cand, str(tmp))
            got = run_ort(tmp, blob)
            gate_outputs(got, ref, nc, args.imgsz, f"opset{args.opset}-vs-opset{have}")
            tmp.unlink()
            converted = cand
            print(f"[onnx] version_converter {have} -> {args.opset}: clean")
        except Exception as e:  # noqa: BLE001 - any failure means "ship the original opset", reported
            print(f"[onnx] version_converter {have} -> {args.opset} REJECTED, keeping opset {have}: {e}")
            (dst / "_convert_tmp.onnx").unlink(missing_ok=True)
    elif have > args.opset:
        print(f"[onnx] already opset {have} >= {args.opset}, kept")
    onnx.save(converted if converted is not None else m, str(out))
    print(f"[export] {out} ({out.stat().st_size / 1e6:.2f} MB)")
    return out, ref, nc, task, names


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--pt", help="checkpoint (.pt) to trace")
    src.add_argument("--onnx", help="existing static ONNX to lift/copy instead (no checkpoint on this machine)")
    ap.add_argument("--out", required=True, help="model dir, e.g. models/v0.1-seg-n_ncnn (writes <out>/model.onnx)")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-rewrite", action="store_true", help="stock graph (Tile stays; baseline for comparison)")
    ap.add_argument("--task", choices=("auto", "detect", "segment"), default="auto")
    ap.add_argument("--probe", default=str(PROBE), help="probe image for the ORT-vs-torch check")
    ap.add_argument("--expect-dets", type=int, default=None, help="post-NMS count the ncnn CLI reports (+-1 gate)")
    ap.add_argument("--compare-onnx", default=None, help="another ONNX whose op census is printed against this one")
    args = ap.parse_args()

    import onnx

    dst = Path(args.out)
    dst.mkdir(parents=True, exist_ok=True)
    blob = probe_blob(Path(args.probe), args.imgsz)
    if args.pt:
        out, ref, nc, task, names = export_from_pt(args, dst, torch.from_numpy(blob))
    else:
        out, ref, nc, task, names = export_from_onnx(args, dst, blob)

    # metadata: ultralytics' keys stay, ours are added (ym_quant="" = fp32 graph; quantize_onnx_qnn.py fills it)
    m = onnx.load(str(out))
    set_metadata(
        m,
        {
            "ym_quant": "",
            "ym_export": "export_onnx_dense",
            "ym_rewrite": "none" if (args.no_rewrite or args.onnx) else "R1R2R3",
            "task": task,
            "names": str(names),
            "imgsz": str([args.imgsz, args.imgsz]),
        },
    )
    onnx.save(m, str(out))
    types = check_graph(m, rewrite_ran=bool(args.pt) and not args.no_rewrite)
    print(f"[graph] opset {[(o.domain or 'ai.onnx', o.version) for o in m.opset_import]} ir {m.ir_version}")
    if args.compare_onnx:
        other = op_census(onnx.load(str(args.compare_onnx)))
        print(f"[census] {args.compare_onnx}: {census_line(other)}")
        delta = {t: types.get(t, 0) - other.get(t, 0) for t in set(types) | set(other)}
        print("[census] delta (this - other): " + " ".join(f"{t}{d:+d}" for t, d in sorted(delta.items()) if d))

    # reference first, gates second: a failed gate leaves the .npz behind for diagnosis
    np.savez(dst / "model.ref.npz", **{f"out{i}": r for i, r in enumerate(ref)})
    got = run_ort(out, blob)
    gate_outputs(got, ref, nc, args.imgsz, "ort-vs-torch" if args.pt else "ort-vs-source")
    report_dets(got, ref, nc, args.expect_dets, "ort")
    print(f"[done] {out} + model.ref.npz")


if __name__ == "__main__":
    main()
