#!/usr/bin/env python
"""Static QDQ quantization of a `model.onnx` for ONNX Runtime's QNN (Hexagon HTP) provider.

Runs in /data/tmp/venv-v2608 (onnx 1.22, onnxruntime 1.28 with `onnxruntime.quantization`), from
the repo root, on x86: the QNN EP only CONSUMES QDQ graphs, the quantization itself is a desktop
step. The float graph is the opset-17 static export of scripts/export_onnx_dense.py.

Pipeline (the ORT-documented QNN recipe, `onnxruntime.quantization.execution_providers.qnn`):
  1. `qnn_preprocess_model`: fusions the HTP likes (LayerNorm/Gelu style, none fire on these
     graphs) and shape inference; I/O layout untouched (the C++ core feeds NCHW float).
  2. a `CalibrationDataReader` over the letterboxed 640x640 PNGs of `calib/ncnn_*_1024_640`
     (the same pre-rendered sets `quantize_ncnn_int8.py` feeds ncnn2table): BGR png -> RGB /255
     -> NCHW float32, i.e. exactly what cpp/src/common.cpp does to a camera frame (gray-114
     letterbox already baked into the PNG). `--n` images, evenly spread over the set.
  3. `get_qnn_qdq_config(activation QUInt16 + weight QInt8 per-channel)` = "a16w8" (the HTP's
     accuracy-first mode; u16 activations keep the sigmoid/softmax chains and the MoE gates
     honest) or `QUInt8` activations = "a8w8" (the fast mode: expected to lose accuracy on these
     graphs, produced as the documented attempt). Calibration `--method minmax` (default; the
     method QNN recommends for u16), `percentile` / `entropy` as fallbacks when MinMax kills
     accuracy. Everything else stays at the helper's QNN defaults (`add_qtype_converts`,
     symmetric int8 weights, per-op 16-bit overrides).
  4. `quantize` -> `<dir>/model-<scheme>.onnx` with `metadata_props` `ym_quant=<scheme>` (what
     cpp/src/ort_backend.cpp reads to label the backend "ort-QNN-htp-a16w8") plus the calibration
     provenance in `ym_calib`. The ultralytics keys (task, names, imgsz) are carried over.

Post-check (ORT CPU EP, slow for u16 but exact): the bench probe (`results/int8_bench/
probe_coco.jpg`, `probe_visdrone.jpg` for VisDrone models) through the float `model.onnx` and the
quantized graph; reports out0 max-abs diff (raw and scaled the way export_onnx_dense.py gates),
the post-NMS detection count at the C++ defaults (conf 0.25 / IoU 0.5) of both, and fails when
the quantized count is not within `--det-tol` of the float count (a dead graph, not a
certification: that is the smoke-set mAP of the device dumps, see tests/README.md).

`--dump-preds` mode (no quantization): run ANY of these ONNX files on the CPU EP over a dir of
jpgs at val settings (conf 0.001 / IoU 0.7 / max 300, argmax class + per-class greedy NMS, the
core's rule) and write `class conf x1 y1 x2 y2` txts in original pixels, the format
scripts/eval_map.py scores and the device OrtDumpTest.kt writes - so float-vs-a16w8 gets a
desktop mAP delta on the 200-image smoke set before the phone is involved.

Measured on the 200-image smoke sets (CPU EP, box mAP50-95, 2026-09-08): plain a16w8 MinMax loses
1.23 pt on seg-N (0.4613 -> 0.4490), 0.56 on v0.1-N (0.4700 -> 0.4644), 0.35 on EsMoE-N VisDrone
(0.1688 -> 0.1653). The seg-N loss is int8 WEIGHT error, not u16 activations: a16w16 is lossless
(0.4615), keeping BatchNorm float or int16-ing the attention/router convs changes nothing, and
percentile calibration (n=16/32; n=256 is OOM-killed even with --stride, the histogram calibrator
keeps every activation) is no better. `--w16-match model.25.` (int16 weights on the 37 seg-head
convs, +0.6 MB) brings seg-N to 0.4527 (-0.86 pt) and is the shipped recipe. a8w8 with MinMax is
dead on all three (0 detections on the probes; the files are parked as `model-a8w8.rejected.onnx`
so stage_models.sh never ships them).

Usage:
    /data/tmp/venv-v2608/bin/python scripts/quantize_onnx_qnn.py \\
        --onnx models/v0.1-seg-n_ncnn/model.onnx --calib-dir calib/ncnn_coco_1024_640 --n 256 \\
        --scheme a16w8 --w16-match model.25. --expect-dets 15
    ... --onnx models/v0.1-n_ncnn/model.onnx --scheme a16w8 --expect-dets 14
    ... --onnx models/esmoe_n_visdrone_ncnn/model.onnx --calib-dir calib/ncnn_visdrone_1024_640 \\
        --probe results/int8_bench/probe_visdrone.jpg --scheme a8w8 --expect-dets 43
    ... --onnx models/v0.1-seg-n_ncnn/model-a16w8.onnx --dump-preds /tmp/pred_a16w8 \\
        --images <smoke jpg dir> --limit 200
    PYTHONPATH=/data/YOLO-Master ... scripts/eval_map.py --preds /tmp/pred_a16w8 --images <dir> \\
        --labels /data/datasets/coco/labels/val2017 --names-yaml models/v0.1-seg-n_ncnn/metadata.yaml
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_probe_f32 import preprocess  # the runtime letterbox (numpy + cv2)

SCHEMES = {
    # scheme -> (activation QuantType name, weight QuantType name)
    "a16w8": ("QUInt16", "QInt8"),
    "a8w8": ("QUInt8", "QInt8"),
    # diagnostic only (the runtime never picks it): tells weight error from activation error
    "a16w16": ("QUInt16", "QInt16"),
}
METHODS = {"minmax": "MinMax", "percentile": "Percentile", "entropy": "Entropy"}
PROBE_COCO = Path("results/int8_bench/probe_coco.jpg")
PROBE_VISDRONE = Path("results/int8_bench/probe_visdrone.jpg")
# C++ runtime defaults for the probe count (cpp/include/yolomaster.hpp Config) and ultralytics val
# settings for the prediction dumps (tests/certify_ncnn_int8.py VAL_ARGS).
CONF, IOU, MAX_DET = 0.25, 0.50, 300
VAL_CONF, VAL_IOU, VAL_MAX_DET = 0.001, 0.70, 300


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def meta_dict(model) -> dict[str, str]:
    return {p.key: p.value for p in model.metadata_props}


def set_metadata(model, extra: dict[str, str]) -> None:
    have = {p.key: p for p in model.metadata_props}
    for k, v in extra.items():
        if k in have:
            have[k].value = v
        else:
            p = model.metadata_props.add()
            p.key, p.value = k, v


def model_info(path: Path) -> tuple[int, int, str, dict]:
    """(nc, imgsz, task, names) from the export's metadata (names) / graph shapes (fallback)."""
    import onnx

    m = onnx.load(str(path), load_external_data=False)
    meta = meta_dict(m)
    names = ast.literal_eval(meta["names"]) if meta.get("names") else {}
    dims = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    imgsz = int(dims[-1])
    nc = len(names) or int(m.graph.output[0].type.tensor_type.shape.dim[1].dim_value) - 4
    task = meta.get("task") or ("segment" if len(m.graph.output) > 1 else "detect")
    if meta.get("imgsz"):
        imgsz = int(ast.literal_eval(meta["imgsz"])[0])
    return nc, imgsz, task, names


def cpu_threads() -> int:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # non-Linux
        return os.cpu_count() or 1


def pin_ort_threads() -> None:
    """Make every ORT session (ours and the calibrator's) use the cgroup's CPUs, not the host's.

    ORT sizes its default pool from the host core count and then fails `pthread_setaffinity_np` for
    every CPU outside the container's cpuset (one error line per thread, and idle threads); the
    quantization calibrator builds its own sessions with a bare `SessionOptions()`, so the fix is
    applied at the class.
    """
    import onnxruntime as ort

    n = cpu_threads()
    if getattr(ort.SessionOptions, "_ym_pinned", False):
        return
    orig = ort.SessionOptions.__init__

    def init(self, *a, **k):
        orig(self, *a, **k)
        self.intra_op_num_threads = n
        self.log_severity_level = 3

    ort.SessionOptions.__init__ = init
    ort.SessionOptions._ym_pinned = True


def make_session(path: Path, threads: int = 0):
    import onnxruntime as ort

    pin_ort_threads()
    so = ort.SessionOptions()
    if threads > 0:
        so.intra_op_num_threads = threads
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(sess, blob: np.ndarray) -> list[np.ndarray]:
    return sess.run(None, {sess.get_inputs()[0].name: blob})


def nms_keep(out0: np.ndarray, nc: int, conf: float, iou: float, max_det: int) -> np.ndarray:
    """Indices (into the anchor axis) the C++ core keeps: argmax class, per-class greedy NMS, cap.

    Letterbox pixels are fine here: IoU is invariant under the uniform scale + shift of the inverse letterbox.
    Returns anchors in descending score order.
    """
    cls = out0[0, 4 : 4 + nc]
    best, score = cls.argmax(axis=0), cls.max(axis=0)
    sel = np.nonzero(score >= conf)[0]
    if sel.size == 0:
        return sel
    cx, cy, w, h = (out0[0, i, sel].astype(np.float64) for i in range(4))
    off = best[sel] * (2.0 * out0.shape[2] + 8192.0)  # class stratification, as nms_and_cap does
    x1, y1, x2, y2 = cx - w / 2 + off, cy - h / 2 + off, cx + w / 2 + off, cy + h / 2 + off
    area = (x2 - x1) * (y2 - y1)
    order = np.argsort(-score[sel], kind="stable")
    dead = np.zeros(sel.size, bool)
    keep = []
    for m, i in enumerate(order):
        if dead[i]:
            continue
        keep.append(i)
        if len(keep) >= max_det:
            break
        j = order[m + 1 :]
        j = j[~dead[j]]
        if j.size == 0:
            continue
        xx1, yy1 = np.maximum(x1[i], x1[j]), np.maximum(y1[i], y1[j])
        xx2, yy2 = np.minimum(x2[i], x2[j]), np.minimum(y2[i], y2[j])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        uni = area[i] + area[j] - inter
        dead[j[np.where(uni > 0, inter / np.where(uni > 0, uni, 1), 0) > iou]] = True
    return sel[np.array(keep, dtype=np.int64)]


def det_count(out0: np.ndarray, nc: int) -> int:
    return int(nms_keep(out0, nc, CONF, IOU, MAX_DET).size)


def out0_diff(a: np.ndarray, b: np.ndarray, nc: int, imgsz: int) -> tuple[float, float, float, float]:
    """(max-abs overall, box max-abs px, class max-abs, scaled max = max(box/imgsz, cls, rest))."""
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    box, cls, rest = float(d[0, :4].max()), float(d[0, 4 : 4 + nc].max()), float(d[0, 4 + nc :].max(initial=0.0))
    return float(d.max()), box, cls, max(box / imgsz, cls, rest)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------


def pick_calib(calib_dir: Path, n: int) -> list[Path]:
    pngs = sorted(calib_dir.glob("*.png"))
    if not pngs:
        raise SystemExit(f"no *.png under {calib_dir} (expected the letterboxed set of quantize_ncnn_int8.py)")
    if n <= 0 or n >= len(pngs):
        return pngs
    idx = np.linspace(0, len(pngs) - 1, n).round().astype(int)
    return [pngs[i] for i in idx]


class PngReader:
    """CalibrationDataReader over pre-letterboxed PNGs: BGR png -> RGB /255 -> [1,3,H,W] float32."""

    def __init__(self, files: list[Path], input_name: str, imgsz: int):
        import cv2

        self.cv2, self.files, self.name, self.imgsz = cv2, list(files), input_name, imgsz
        self.all_files = self.files
        self.i = 0

    def get_next(self):
        if self.i >= len(self.files):
            return None
        f = self.files[self.i]
        self.i += 1
        bgr = self.cv2.imread(str(f))
        if bgr is None or bgr.shape[:2] != (self.imgsz, self.imgsz):
            raise RuntimeError(f"{f}: not a {self.imgsz}x{self.imgsz} letterboxed PNG")
        blob = bgr[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        return {self.name: np.ascontiguousarray(blob)}

    def rewind(self):
        self.i = 0

    def __len__(self):  # quantize_static needs it for the strided (chunked) calibration path
        return len(self.all_files)

    def set_range(self, start_index: int, end_index: int):  # strided calibration: only [start, end)
        self.files = self.all_files[start_index:end_index]
        self.i = 0


def quantize_model(args, src: Path, out: Path, nc: int, imgsz: int) -> dict:
    import onnx
    from onnxruntime.quantization import CalibrationMethod, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model

    pin_ort_threads()
    act_name, w_name = SCHEMES[args.scheme]
    act_t, w_t = getattr(QuantType, act_name), getattr(QuantType, w_name)
    method = getattr(CalibrationMethod, METHODS[args.method])
    files = pick_calib(Path(args.calib_dir), args.n)
    print(
        f"[calib] {len(files)} PNGs from {args.calib_dir} (of {len(list(Path(args.calib_dir).glob('*.png')))}), "
        f"method {METHODS[args.method]}, activation {act_name} weight {w_name} per_channel={not args.per_tensor}"
    )

    work = out.parent / f"_quant_tmp_{args.scheme}"
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    pre = work / "preprocessed.onnx"
    # ORT 1.28's qnn_preprocess_model trips over a graph WITH metadata_props (`save_and_reload_optimize_model`
    # does `dict.update(model.metadata_props)` on the repeated proto): feed it a metadata-free copy, the
    # source's keys are re-applied on the quantized file below.
    bare = work / "float_nometa.onnx"
    src_model = onnx.load(str(src))
    src_meta = meta_dict(src_model)
    del src_model.metadata_props[:]
    onnx.save(src_model, str(bare))
    del src_model
    t0 = time.time()
    changed = qnn_preprocess_model(str(bare), str(pre), fuse_layernorm=False)
    print(f"[preprocess] qnn_preprocess_model: {'modified' if changed else 'unchanged'} ({time.time() - t0:.1f}s)")
    if not changed:
        shutil.copy(bare, pre)

    pre_model = onnx.load(str(pre), load_external_data=False)
    input_name = pre_model.graph.input[0].name
    reader = PngReader(files, input_name, imgsz)
    # --keep-float: nodes of these op types stay float (DQ -> op -> Q islands; the HTP runs them as fp16).
    # The standalone BatchNormalization nodes of the attention blocks are the known case: their rank-1
    # scale/bias would otherwise be int8 per-tensor "weights".
    keep_float = [t for t in (args.keep_float or "").split(",") if t]
    excluded = [n.name for n in pre_model.graph.node if n.op_type in keep_float]
    if keep_float:
        print(f"[config] keep float: {len(excluded)} nodes of {keep_float}")
    # --w16-match: Conv weights whose initializer name contains one of the substrings get symmetric per-channel
    # INT16 instead of INT8 (mixed weight precision; the HTP takes int16 weights). Bias stays int32.
    w16 = [t for t in (args.w16_match or "").split(",") if t]
    inits = {i.name for i in pre_model.graph.initializer}
    overrides: dict[str, list[dict]] = {}
    if w16:
        for n in pre_model.graph.node:
            if n.op_type != "Conv" or len(n.input) < 2 or n.input[1] not in inits:
                continue
            if any(k in n.input[1] for k in w16):
                overrides[n.input[1]] = [{"quant_type": QuantType.QInt16, "symmetric": True, "axis": 0}]
        print(f"[config] int16 weights: {len(overrides)} Conv weights matching {w16}")
    del pre_model
    cfg = get_qnn_qdq_config(
        str(pre),
        reader,
        calibrate_method=method,
        activation_type=act_t,
        weight_type=w_t,
        per_channel=not args.per_tensor,
        activation_symmetric=False,
        weight_symmetric=True,
        keep_removable_activations=False,
        stride=args.stride,
        nodes_to_exclude=excluded or None,
        init_overrides=overrides or None,
    )
    # ORT's default MinMax over a 256-image reader averages nothing: min/max over all samples.
    # Percentile/Entropy build histograms on every tensor (memory-hungry on 600-node graphs); the
    # `stride` splits the reader into chunks the calibrator merges.
    if args.method == "percentile":
        cfg.extra_options["CalibPercentile"] = args.percentile
    if args.method == "entropy":
        cfg.extra_options["CalibMovingAverage"] = False
    print(
        f"[config] op_types_to_quantize={len(cfg.op_types_to_quantize or [])} types, "
        f"extra_options keys={sorted(cfg.extra_options)}"
    )
    t0 = time.time()
    quantize(str(pre), str(out), cfg)
    dt = time.time() - t0
    print(f"[quantize] {out} ({out.stat().st_size / 1e6:.2f} MB) in {dt:.0f}s")

    m = onnx.load(str(out))
    keep = {k: v for k, v in src_meta.items() if k not in ("ym_quant",)}
    prov = {
        "scheme": args.scheme,
        "activation": act_name,
        "weight": w_name,
        "per_channel": not args.per_tensor,
        "method": METHODS[args.method],
        "percentile": args.percentile if args.method == "percentile" else None,
        "keep_float": keep_float,
        "w16_match": w16,
        "w16_count": len(overrides),
        "calib_dir": str(args.calib_dir),
        "n": len(files),
        "tool": "onnxruntime.quantization.execution_providers.qnn",
        "onnxruntime": __import__("onnxruntime").__version__,
        "source": str(src),
    }
    set_metadata(m, {**keep, "ym_quant": args.scheme, "ym_calib": json.dumps(prov, separators=(",", ":"))})
    onnx.save(m, str(out))
    shutil.rmtree(work, ignore_errors=True)
    types = {}
    for n in m.graph.node:
        types[n.op_type] = types.get(n.op_type, 0) + 1
    print(
        f"[graph] {len(m.graph.node)} nodes: QuantizeLinear={types.get('QuantizeLinear', 0)} "
        f"DequantizeLinear={types.get('DequantizeLinear', 0)} Conv={types.get('Conv', 0)} "
        f"MatMul={types.get('MatMul', 0)} Softmax={types.get('Softmax', 0)}"
    )
    return prov


# ---------------------------------------------------------------------------
# probe post-check
# ---------------------------------------------------------------------------


def probe_check(src: Path, out: Path, probe: Path, nc: int, imgsz: int, expect: int | None, tol: int) -> dict:
    blob, _ = preprocess(probe, imgsz)
    t0 = time.time()
    ref = run(make_session(src), blob)
    t_ref = time.time() - t0
    sess = make_session(out)
    t0 = time.time()
    got = run(sess, blob)
    t_q = time.time() - t0
    if len(got) != len(ref) or any(tuple(g.shape) != tuple(r.shape) for g, r in zip(got, ref)):
        raise RuntimeError(f"output mismatch: {[g.shape for g in got]} vs {[r.shape for r in ref]}")
    mx, box, cls, scaled = out0_diff(got[0], ref[0], nc, imgsz)
    n_ref, n_q = det_count(ref[0], nc), det_count(got[0], nc)
    hot = ref[0][0, 4 : 4 + nc].max(axis=0) >= CONF
    d = np.abs(got[0].astype(np.float64) - ref[0].astype(np.float64))
    box_hot = float(d[0, :4][:, hot].max(initial=0.0))
    print(f"[probe] {probe}: float {t_ref * 1e3:.0f} ms, quant {t_q * 1e3:.0f} ms on the CPU EP")
    print(
        f"[probe] out0 max-abs {mx:.4f} (box {box:.3f} px all anchors, {box_hot:.3f} px on the {int(hot.sum())} "
        f"anchors >= {CONF}; cls {cls:.4f}) | scaled {scaled:.2e}"
    )
    for i in range(1, len(got)):
        di = float(np.abs(got[i].astype(np.float64) - ref[i].astype(np.float64)).max())
        print(f"[probe] out{i}{tuple(got[i].shape)} max-abs {di:.4f} (ref max |.| {float(np.abs(ref[i]).max()):.2f})")
    line = f"[probe] dets@{CONF} after NMS(iou {IOU}): quant {n_q} vs float {n_ref}"
    if expect is not None:
        line += f" (ncnn CLI / export gate {expect})"
    print(line)
    res = {
        "float_dets": n_ref,
        "quant_dets": n_q,
        "out0_maxabs": mx,
        "box_px_hot": box_hot,
        "cls_maxabs": cls,
        "scaled": scaled,
        "float_ms": t_ref * 1e3,
        "quant_ms": t_q * 1e3,
    }
    if abs(n_q - n_ref) > tol:
        raise SystemExit(f"FAIL: quantized detection count {n_q} not within +-{tol} of float {n_ref}")
    return res


# ---------------------------------------------------------------------------
# prediction dumps (desktop mAP)
# ---------------------------------------------------------------------------


def dump_preds(onnx_path: Path, images: Path, out_dir: Path, limit: int, nc: int, imgsz: int, threads: int) -> None:
    files = sorted(glob.glob(str(images / "*.jpg")))
    if limit > 0:
        files = files[:limit]
    if not files:
        raise SystemExit(f"no *.jpg under {images}")
    out_dir.mkdir(parents=True, exist_ok=True)
    sess = make_session(onnx_path, threads)
    name = sess.get_inputs()[0].name
    times, total = [], 0
    t_start = time.time()
    for k, f in enumerate(files):
        blob, (r, px, py, w, h) = preprocess(f, imgsz)
        t0 = time.time()
        out0 = sess.run(None, {name: blob})[0]
        times.append(time.time() - t0)
        keep = nms_keep(out0, nc, VAL_CONF, VAL_IOU, VAL_MAX_DET)
        cls = out0[0, 4 : 4 + nc][:, keep]  # (nc, K); NOT out0[0, 4:4+nc, keep], which numpy makes (K, nc)
        best, score = cls.argmax(axis=0), cls.max(axis=0)
        cx, cy, bw, bh = (out0[0, i, keep] for i in range(4))
        x1 = np.clip((cx - bw / 2 - px) / r, 0, w)
        y1 = np.clip((cy - bh / 2 - py) / r, 0, h)
        x2 = np.clip((cx + bw / 2 - px) / r, 0, w)
        y2 = np.clip((cy + bh / 2 - py) / r, 0, h)
        with open(out_dir / (Path(f).stem + ".txt"), "w") as fh:
            fh.writelines(
                f"{int(best[j])} {score[j]:.6f} {x1[j]:.3f} {y1[j]:.3f} {x2[j]:.3f} {y2[j]:.3f}\n"
                for j in range(keep.size)
            )
        total += int(keep.size)
        if (k + 1) % 25 == 0 or k + 1 == len(files):
            el = time.time() - t_start
            print(
                f"[dump] {k + 1}/{len(files)} images, {el:.0f}s elapsed, median {np.median(times) * 1e3:.0f} ms/img, "
                f"{total} dets so far",
                flush=True,
            )
    print(
        f"[dump] {out_dir}: {len(files)} txts, {total} dets at conf {VAL_CONF} / iou {VAL_IOU} / max {VAL_MAX_DET}, "
        f"median {np.median(times) * 1e3:.0f} ms/img on the CPU EP"
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--onnx", required=True, help="float model.onnx (or, with --dump-preds, any of the siblings)")
    ap.add_argument("--calib-dir", default="calib/ncnn_coco_1024_640", help="letterboxed 640 PNG set")
    ap.add_argument("--n", type=int, default=256, help="calibration images, evenly spread over the set (0 = all)")
    ap.add_argument("--scheme", choices=sorted(SCHEMES), default="a16w8")
    ap.add_argument("--method", choices=sorted(METHODS), default="minmax")
    ap.add_argument("--percentile", type=float, default=99.999, help="for --method percentile")
    ap.add_argument("--stride", type=int, default=None, help="calibration chunk size (memory), default = all")
    ap.add_argument("--per-tensor", action="store_true", help="per-tensor weights instead of per-channel")
    ap.add_argument("--w16-match", default="", help="comma-separated substrings: Conv weights matching get int16")
    ap.add_argument(
        "--keep-float", default="", help="comma-separated op types left unquantized (e.g. BatchNormalization)"
    )
    ap.add_argument("--out", default=None, help="default <dir of --onnx>/model-<scheme>.onnx")
    ap.add_argument(
        "--probe",
        default=None,
        help="post-check image (default: COCO probe, VisDrone probe for a dir/metadata that says visdrone)",
    )
    ap.add_argument("--expect-dets", type=int, default=None, help="post-NMS count of the reference (printed)")
    ap.add_argument("--det-tol", type=int, default=3, help="quantized probe count must be within +-tol of float")
    ap.add_argument("--dump-preds", default=None, help="write val-setting prediction txts of --onnx here (no quant)")
    ap.add_argument("--images", default=None, help="jpg dir for --dump-preds")
    ap.add_argument("--limit", type=int, default=0, help="first N sorted jpgs for --dump-preds (0 = all)")
    ap.add_argument("--threads", type=int, default=0, help="ORT intra-op threads for --dump-preds (0 = default)")
    args = ap.parse_args()

    src = Path(args.onnx)
    nc, imgsz, task, names = model_info(src)
    print(
        f"[model] {src} task={task} nc={nc} imgsz={imgsz} ym_quant={meta_dict(__import__('onnx').load(str(src), load_external_data=False)).get('ym_quant', '?')!r}"
    )

    if args.dump_preds:
        if not args.images:
            raise SystemExit("--dump-preds needs --images")
        dump_preds(src, Path(args.images), Path(args.dump_preds), args.limit, nc, imgsz, args.threads)
        return

    out = Path(args.out) if args.out else src.with_name(f"model-{args.scheme}.onnx")
    visdrone = "visdrone" in str(src).lower() or (names and str(names.get(0, "")).lower() == "pedestrian")
    probe = Path(args.probe) if args.probe else (PROBE_VISDRONE if visdrone else PROBE_COCO)
    if visdrone and "visdrone" not in str(args.calib_dir).lower():
        print(f"[calib] NOTE: VisDrone-domain model calibrated on {args.calib_dir} (no VisDrone set given)")

    prov = quantize_model(args, src, out, nc, imgsz)
    res = probe_check(src, out, probe, nc, imgsz, args.expect_dets, args.det_tol)
    print(
        "[done] "
        + json.dumps({"out": str(out), "size_mb": round(out.stat().st_size / 1e6, 2), **prov, **res}, default=str)
    )


if __name__ == "__main__":
    main()
