#!/usr/bin/env python
"""Project03 phase 2: TensorRT FP16/INT8 with reconstructed calibration.

Design constraints (from the phase-2 review of the earlier INT8 failure):
  - The detect head concatenates sigmoided class scores (range [0,1]) with box
    coordinates (range [0,imgsz]) into ONE output tensor. Per-tensor INT8 scaling of
    that tensor is dominated by the box range and quantizes every cls channel to
    zero (the observed "max cls = 0.0000" collapse). The ENTIRE head subgraph is
    therefore pinned to FP16 before calibration - it never enters INT8.
  - Entropy calibration only (IInt8EntropyCalibrator2). MinMax is not used anywhere.
  - PER-LAYER sensitivity, not per-portion heuristics: an ORT fp32 pass over the
    calibration set records activation statistics for every conv/concat output
    (dynamic range, outlier ratio, channel-wise range disparity); layers are ranked
    by quantization-hostility and greedily pinned to FP16, rebuild+re-eval each
    round, until the accuracy budget holds. The ranking table is emitted as the
    per-layer sensitivity report.
  - Calibration images are real letterboxed target-dataset images with
    inference-identical preprocessing (BGR->RGB, /255, gray-114 letterbox).

Modes build a ladder: fp32, fp16, int8-naive (everything quantized, for the
collapse demonstration), int8 (reconstructed). Every engine is evaluated with
ultralytics .val() (AutoBackend loads .engine) so numbers are protocol-identical
to the pruning phase, and timed with a raw enqueue loop (median of 200).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# preprocessing (identical to the edge/validation letterbox)
# ---------------------------------------------------------------------------

def letterbox_blob(img_path: str, imgsz: int) -> np.ndarray:
    import cv2
    bgr = cv2.imread(str(img_path))
    h, w = bgr.shape[:2]
    r = min(imgsz / w, imgsz / h)
    nw, nh = round(w * r), round(h * r)
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(bgr, (nw, nh))
    blob = canvas[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None] / 255.0
    return np.ascontiguousarray(blob)


def calib_image_list(images_dir: str, n: int) -> list[str]:
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    imgs = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in exts)
    if n and len(imgs) > n:
        step = len(imgs) / n
        imgs = [imgs[int(i * step)] for i in range(n)]   # spread, not head-biased
    return [str(p) for p in imgs]


# ---------------------------------------------------------------------------
# stage 1: per-layer activation statistics via onnxruntime
# ---------------------------------------------------------------------------

def sensitivity_scan(onnx_path: str, images: list[str], imgsz: int, out_csv: Path,
                     max_imgs: int = 48) -> list[dict]:
    """fp32 pass recording per-node output stats; returns rows ranked most-hostile first."""
    import onnx
    import onnxruntime as ort

    model = onnx.load(onnx_path)
    graph = model.graph
    watch = []
    for node in graph.node:
        if node.op_type in ("Conv", "Concat", "Add", "Mul", "MatMul") and node.output:
            watch.append((node.name or node.output[0], node.op_type, node.output[0]))
    existing = {o.name for o in graph.output}
    for _, _, tname in watch:
        if tname not in existing:
            graph.output.append(onnx.helper.make_empty_tensor_value_info(tname))
    tmp = str(Path(out_csv).with_suffix(".instrumented.onnx"))
    onnx.save(model, tmp)

    sess = ort.InferenceSession(tmp, providers=["CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    stats = {t: {"amax": 0.0, "p999": [], "ch_ratio": 0.0} for _, _, t in watch}
    for ip in images[:max_imgs]:
        blob = letterbox_blob(ip, imgsz)
        outs = sess.run(out_names, {sess.get_inputs()[0].name: blob})
        for name, arr in zip(out_names, outs):
            if name not in stats:
                continue
            a = np.abs(arr.astype(np.float32))
            st = stats[name]
            st["amax"] = max(st["amax"], float(a.max()))
            st["p999"].append(float(np.percentile(a, 99.9)))
            if arr.ndim == 4 and arr.shape[1] > 1:            # channel-wise range disparity
                ch = a.reshape(arr.shape[1], -1).max(axis=1)
                lo = max(float(ch.min()), 1e-6)
                st["ch_ratio"] = max(st["ch_ratio"], float(ch.max()) / lo)

    rows = []
    for nname, op, tname in watch:
        st = stats[tname]
        p999 = float(np.mean(st["p999"])) if st["p999"] else 0.0
        outlier = st["amax"] / max(p999, 1e-6)
        # hostility: outliers blow up the entropy histogram; channel disparity means one
        # per-tensor scale starves small-range channels (the cls-vs-box failure mode)
        hostility = outlier * np.log1p(st["ch_ratio"]) + 0.05 * np.log1p(st["amax"])
        rows.append({"layer": nname, "op": op, "tensor": tname, "amax": round(st["amax"], 4),
                     "p999": round(p999, 4), "outlier_ratio": round(outlier, 3),
                     "channel_range_ratio": round(st["ch_ratio"], 1),
                     "hostility": round(float(hostility), 4)})
    rows.sort(key=lambda r: -r["hostility"])
    import csv
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    Path(tmp).unlink(missing_ok=True)
    return rows


# ---------------------------------------------------------------------------
# TensorRT build
# ---------------------------------------------------------------------------

def _make_calibrator(trt, images: list[str], imgsz: int, cache: Path):
    # device memory via torch, not cuda-python: the cuda.bindings runtime call
    # returns (err, None) on some pods while torch CUDA works in the same process
    import torch

    class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            super().__init__()
            self.images = images
            self.idx = 0
            self.cache = cache
            self.dev = torch.empty(1 * 3 * imgsz * imgsz, dtype=torch.float32,
                                   device="cuda")

        def get_batch_size(self):
            return 1

        def get_batch(self, names):
            if self.idx >= len(self.images):
                return None
            blob = letterbox_blob(self.images[self.idx], imgsz)
            self.idx += 1
            self.dev.copy_(torch.from_numpy(np.ascontiguousarray(blob).ravel()))
            torch.cuda.synchronize()
            return [int(self.dev.data_ptr())]

        def read_calibration_cache(self):
            return self.cache.read_bytes() if self.cache.exists() else None

        def write_calibration_cache(self, data):
            self.cache.write_bytes(data)

    return EntropyCalibrator()


def build_engine(onnx_path: str, engine_path: Path, mode: str, imgsz: int,
                 calib_images: list[str] | None = None, calib_cache: Path | None = None,
                 pin_fp16_prefixes: list[str] | None = None,
                 pin_fp16_layers: set[str] | None = None) -> dict:
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)   # TRT >=10: explicit batch is the only mode
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(Path(onnx_path).read_bytes()):
        raise RuntimeError("\n".join(str(parser.get_error(i)) for i in range(parser.num_errors)))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    import os as _os
    if _os.environ.get("TRT_SPARSE") == "1":
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)
        print("  [build] sparse tensor-core weights enabled (2:4)")
    if _os.environ.get("TRT_OPT_LEVEL"):
        config.builder_optimization_level = int(_os.environ["TRT_OPT_LEVEL"])
        print(f"  [build] optimization level {config.builder_optimization_level}")
    # without DETAILED the engine inspector returns bare layer-name strings and
    # the precision audit cannot see per-layer precisions
    config.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    if mode == "fp32" and not ALLOW_TF32:
        # TRT enables TF32 by default on Ampere+, which silently tensor-core-accelerates
        # the "fp32" baseline. The criterion's baseline is UNOPTIMIZED FP32.
        config.clear_flag(trt.BuilderFlag.TF32)
    pinned = []
    if mode in ("fp16", "int8", "int8-naive", "int8-qdq"):
        config.set_flag(trt.BuilderFlag.FP16)
    if mode == "int8-qdq":
        # explicit quantization: scales come from Q/DQ nodes in the ONNX (QAT
        # export); no calibrator, no pins - the graph IS the precision spec
        config.set_flag(trt.BuilderFlag.INT8)
    if mode in ("int8", "int8-naive"):
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = _make_calibrator(trt, calib_images, imgsz, calib_cache)
        if mode == "int8":
            # OBEY, not PREFER: with a calibrator supplying scales for every tensor,
            # PREFER lets TRT silently keep pinned layers INT8 whenever the INT8
            # kernel wins the autotuner - every pin becomes a no-op.
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            prefixes = tuple(pin_fp16_prefixes or [])
            extra = pin_fp16_layers or set()
            for i in range(network.num_layers):
                layer = network.get_layer(i)
                if layer.name.startswith(prefixes) or layer.name in extra:
                    out0 = layer.get_output(0) if layer.num_outputs else None
                    # only float-computing layers accept a precision; index/shape/bool
                    # layers (e.g. the head Gather computing indices) must be skipped
                    # Cast/Identity layers ARE type conversions: constraining their
                    # output type contradicts their cast target (castLayer assert)
                    if (layer.type not in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE,
                                           trt.LayerType.CAST, trt.LayerType.IDENTITY)
                            and out0 is not None and out0.dtype == trt.DataType.FLOAT):
                        layer.precision = trt.DataType.HALF
                        if not out0.is_network_output:
                            layer.set_output_type(0, trt.DataType.HALF)
                        pinned.append(layer.name)

    t0 = time.time()
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError(f"engine build failed ({mode})")
    # ultralytics engine container: 4-byte little-endian metadata length + JSON + blob
    # (AutoBackend requires the header; raw engines eval-crash on metadata None)
    meta = json.dumps(ULTRA_META).encode()
    with open(engine_path, "wb") as f:
        f.write(len(meta).to_bytes(4, byteorder="little", signed=True))
        f.write(meta)
        f.write(blob)
    # ground-truth precision audit: ask the built engine what actually runs in
    # what precision (the autotuner, not the request, is the truth)
    prec_counts: dict = {}
    try:
        runtime = trt.Runtime(logger)
        eng = runtime.deserialize_cuda_engine(blob)
        insp = eng.create_engine_inspector()
        info = json.loads(insp.get_engine_information(trt.LayerInformationFormat.JSON))
        for lyr in info.get("Layers", []):
            if not isinstance(lyr, dict):
                p = "no-detail"
            else:
                # TRT 10 inspector: datatype lives per-output as "Format/Datatype"
                # (e.g. "Half", "Int8", "Float"), not in a top-level Precision key
                outs = lyr.get("Outputs") or []
                fmt = (outs[0].get("Format/Datatype", "?") if outs else "?")
                p = fmt.split()[0] if isinstance(fmt, str) else "?"
            prec_counts[p] = prec_counts.get(p, 0) + 1
        del insp, eng, runtime
    except Exception as e:
        prec_counts = {"inspector_failed": str(e)}
    print(f"  [audit {mode}] engine layer precisions: {prec_counts}")
    return {"mode": mode, "build_s": round(time.time() - t0, 1),
            "n_pinned_fp16": len(pinned), "pinned": pinned,
            "precisions": prec_counts,
            "size_MB": round(engine_path.stat().st_size / 1e6, 1)}


ULTRA_META: dict = {}   # filled by main() before any build
ALLOW_TF32 = True       # set from --tf32-baseline


def router_prefixes_from_onnx(onnx_path: str) -> list[str]:
    """MoE router subgraphs must NEVER be quantized: top-k routing flips expert
    selection on tiny logit perturbations (and even at k==E the softmax mixing
    weights scale whole expert branches). Statistical hostility ranking cannot see
    this - it is functional sensitivity, not representability."""
    import onnx
    model = onnx.load(onnx_path)
    prefixes = set()
    for node in model.graph.node:
        name = node.name or ""
        for token in ("/routing/", "/router/", "/routing_network/"):
            i = name.find(token)
            if i > 0:
                prefixes.add(name[:i + len(token)])
    return sorted(prefixes)


def attn_core_layers_from_onnx(onnx_path: str) -> list[str]:
    """Transformer surgery: the QKV / positional / output projection convs carry the
    attention FLOPs and quantize fine; the attention core (QK^T MatMul, score scale,
    softmax, PV MatMul) is precision-sensitive and stays FP16."""
    import onnx
    model = onnx.load(onnx_path)
    return [n.name for n in model.graph.node
            if "/attn/" in (n.name or "") and n.op_type in ("MatMul", "Mul", "Softmax")]


def head_prefix_from_onnx(onnx_path: str) -> str:
    """The detect head = the highest-numbered /model.N/ prefix in the graph."""
    import onnx
    model = onnx.load(onnx_path)
    idxs = set()
    for node in model.graph.node:
        for part in (node.name or "").split("/"):
            if part.startswith("model.") and part.split(".")[1].isdigit():
                idxs.add(int(part.split(".")[1]))
    return f"/model.{max(idxs)}/"


# ---------------------------------------------------------------------------
# latency + accuracy
# ---------------------------------------------------------------------------

def time_engine(engine_path: Path, imgsz: int, iters: int = 200, warmup: int = 50,
                batch: int = 1) -> dict:
    import tensorrt as trt
    import torch

    logger = trt.Logger(trt.Logger.ERROR)
    rt = trt.Runtime(logger)
    raw = engine_path.read_bytes()
    meta_len = int.from_bytes(raw[:4], byteorder="little", signed=True)
    engine = rt.deserialize_cuda_engine(raw[4 + meta_len:])   # skip ultralytics header
    ctx = engine.create_execution_context()
    bufs = []                       # torch tensors keep the device memory alive
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = ctx.get_tensor_shape(name)
        n = max(int(np.prod(shape)), 1)
        t = torch.empty(n, dtype=torch.float32, device="cuda")
        ctx.set_tensor_address(name, t.data_ptr())
        bufs.append(t)
    stream = torch.cuda.Stream()
    handle = stream.cuda_stream
    for _ in range(warmup):
        ctx.execute_async_v3(handle)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        ctx.execute_async_v3(handle)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    del bufs
    times.sort()
    return {"lat_ms_median": round(times[len(times) // 2], 3),
            "lat_ms_p90": round(times[int(len(times) * 0.9)], 3)}


def eval_engine(engine_path: Path, data: str, imgsz: int, batch: int, workers: int,
                max_images: int, split_dir: Path | None) -> tuple[float, float]:
    from ultralytics import YOLO
    m = YOLO(str(engine_path), task="detect")
    r = m.val(data=data, split="val", batch=1, imgsz=imgsz, device=0,
              workers=workers, verbose=False, plots=False)
    return float(r.box.map50), float(r.box.map)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def _write_ladder(ladder, out):
    import csv
    if not ladder:
        return
    fp32 = next((r for r in ladder if r["mode"] == "fp32"), None)
    for r in ladder:
        if fp32 and fp32["lat_ms_median"]:
            r["speedup_vs_fp32"] = round(fp32["lat_ms_median"] / r["lat_ms_median"], 2)
            r["lat_cut_pct"] = round(100 * (1 - r["lat_ms_median"] / fp32["lat_ms_median"]), 1)
    with open(out / "ladder.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=[k for k in ladder[0].keys() if k != "pinned"])
        w.writeheader()
        w.writerows(ladder)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", required=True, help=".pt (exported to onnx here) or .onnx")
    ap.add_argument("--data", required=True, help="dataset yaml for eval + calibration split")
    ap.add_argument("--calib-images", default="", help="dir for calibration images (default: val split of --data)")
    ap.add_argument("--calib-n", type=int, default=256)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=0.01, help="allowed mAP50-95 drop for int8 (task: <1%)")
    ap.add_argument("--pin-step", type=int, default=8, help="extra layers pinned per greedy round")
    ap.add_argument("--max-rounds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--eval-max-images", type=int, default=0, help="0 = full val for final; subset for greedy rounds")
    ap.add_argument("--modes", default="fp32,fp16,int8-naive,int8")
    ap.add_argument("--pin-modules", default="",
                    help="comma list, e.g. '0,1': whole /model.N/ blocks to always-pin FP16 "
                         "in int8 mode (the stem-pair golden recipe)")
    ap.add_argument("--ablate", default="",
                    help="e.g. '0-10': leave-one-out INT8 ablation - pin ALL modules except "
                         "one, per module in the range; per-module damage vs the all-pinned floor")
    ap.add_argument("--bisect", choices=["off", "on"], default="off",
                    help="on = adaptive whole-module bisection instead of greedy per-layer pinning")
    ap.add_argument("--pin-attn-core", choices=["off", "on"], default="off",
                    help="on = pin attention QK^T/softmax/PV FP16, keep qkv+proj convs INT8")
    ap.add_argument("--pins", choices=["all", "head", "routers", "none"], default="all",
                    help="always-pinned FP16 sets for implicit int8: head+routers (all), one of them, or none")
    ap.add_argument("--tf32-baseline", choices=["on", "off"], default="on",
                    help="off = criterion-faithful pure-FP32 baseline (TF32 disabled)")
    args = ap.parse_args()

    global ALLOW_TF32
    ALLOW_TF32 = args.tf32_baseline == "on"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # resolve dataset + calibration images
    import sys
    # diagnose_moe.py (dataset resolver + legacy repairs) ships alongside in scripts/a3
    sys.path.insert(0, str(Path(__file__).parent))
    from diagnose_moe import (_resolve_dataset, _fix_add_residual, _force_dense_esmoe,
                              _strip_property_shadows)
    yaml_path, root, cfg = _resolve_dataset(args.data)
    calib_dir = args.calib_images or str(root / cfg["val"])
    calib_imgs = calib_image_list(calib_dir, args.calib_n)
    print(f"[calib] {len(calib_imgs)} images from {calib_dir}")

    # onnx export if needed
    model_path = args.model
    if model_path.endswith(".pt"):
        from ultralytics import YOLO
        m = YOLO(model_path)
        # A3 rebase onto the RELEASED (non-pruned) weights: the Jan-2026 COCO v0.1-N
        # predates add_residual (compat shim defaults it True -> AP 0.7); ES_MOE must
        # run the same dense path the exporter traces. Same repairs as diagnose_moe.
        print(f"[repair] add_residual forced False on {_fix_add_residual(m)} blocks; "
              f"property shadows stripped {_strip_property_shadows(m)}; "
              f"ES_MOE dense={_force_dense_esmoe(m)}")
        onnx_path = m.export(format="onnx", imgsz=args.imgsz, device=0, simplify=True,
                             dynamic=False, batch=1)
        onnx_dst = out / (Path(model_path).stem + ".onnx")
        Path(onnx_path).replace(onnx_dst)
        model_path = str(onnx_dst)
    print(f"[onnx] {model_path}")

    ULTRA_META.update({
        "description": "project03 quantization ladder", "author": "project03",
        "version": "8.3.240", "task": "detect", "stride": 32, "batch": 1,
        "imgsz": [args.imgsz, args.imgsz],
        "names": {int(k): str(v) for k, v in cfg["names"].items()},
    })
    head_prefix = head_prefix_from_onnx(model_path)
    print(f"[head] pinned subgraph prefix: {head_prefix} (mixed-semantics concat lives here)")
    router_prefixes = router_prefixes_from_onnx(model_path)
    print(f"[router] always-pinned router subgraphs: {router_prefixes}")
    base_pins = {"all": [head_prefix] + router_prefixes, "head": [head_prefix],
                 "routers": list(router_prefixes), "none": []}[args.pins]
    print(f"[pins] --pins {args.pins}: {base_pins}")

    print("[scan] per-layer sensitivity (ORT fp32 activation statistics)...")
    rows = sensitivity_scan(model_path, calib_imgs, args.imgsz, out / "sensitivity_report.csv")
    ranked = [r["layer"] for r in rows if not r["layer"].startswith(head_prefix)
              and not any(r["layer"].startswith(p) for p in router_prefixes)]
    print(f"[scan] {len(rows)} layers profiled; top hostility outside head: "
          + ", ".join(f"{r['layer']}({r['hostility']})" for r in rows[:3]))

    # baseline for accuracy deltas: fp32 engine
    ladder = []
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    base_map = None
    pins_used: set[str] = set()
    if args.pin_attn_core == "on":
        attn_core = attn_core_layers_from_onnx(model_path)
        pins_used.update(attn_core)
        print(f"[attn] transformer surgery: {len(attn_core)} attention-core ops pinned FP16 "
              f"(qkv/pe/proj convs stay INT8)")
    for mode in modes:
        ep = out / f"model_{mode}.engine"
        cache = out / "calib_entropy.cache"
        if mode == "int8" and args.ablate:
            # leave-one-out ablation: damage_i = (all-pinned floor) - (all-pinned minus module i).
            # Additivity (verified by the bisect arm) makes this a per-module damage profile.
            import re as _re
            a_lo, a_hi = (int(x) for x in args.ablate.split("-"))
            mod_ids = sorted({int(m_.group(1)) for r in rows
                              if (m_ := _re.match(r"/model\.(\d+)/", r["layer"]))})
            head_mod = int(_re.match(r"/model\.(\d+)/", head_prefix).group(1))
            mod_ids = [i for i in mod_ids if i != head_mod]
            targets = [i for i in mod_ids if a_lo <= i <= a_hi]
            always = [head_prefix] + router_prefixes
            abl_rows = []

            def _be(mods, tag):
                prefs = always + [f"/model.{i}/" for i in mods]
                inf = build_engine(model_path, ep, "int8", args.imgsz, calib_imgs, cache,
                                   pin_fp16_prefixes=prefs, pin_fp16_layers=pins_used)
                a50, amm = eval_engine(ep, str(yaml_path), args.imgsz, 1, args.workers,
                                       args.eval_max_images, None)
                return inf, a50, amm

            info, m50, mm = _be(mod_ids, "floor-all")
            floor = mm
            print(f"[ablate floor] all modules pinned: mAP50-95={floor:.4f} (d {floor - base_map:+.4f})")
            for i in targets:
                try:
                    _, s50, smm = _be([j for j in mod_ids if j != i], f"minus-{i}")
                except RuntimeError as e:
                    print(f"[ablate minus-{i}] build infeasible ({e}); skipping")
                    abl_rows.append({"module": i, "mAP50": "", "mAP50_95": "",
                                     "damage": "", "note": "infeasible"})
                    continue
                dmg = floor - smm
                print(f"[ablate minus-{i}] module {i} INT8: mAP50-95={smm:.4f} damage={dmg:+.4f}")
                abl_rows.append({"module": i, "mAP50": round(s50, 5), "mAP50_95": round(smm, 5),
                                 "damage": round(dmg, 5), "note": ""})
            import csv as _csv
            with open(out / "ablate.csv", "w", newline="") as af:
                aw = _csv.DictWriter(af, fieldnames=["module", "mAP50", "mAP50_95", "damage", "note"])
                aw.writeheader(); aw.writerows(abl_rows)
            # ladder row should describe a well-defined engine: rebuild the all-pinned floor
            info, m50, mm = _be(mod_ids, "floor-final")
        elif mode == "int8" and args.bisect == "on":
            # adaptive module bisection: pin whole /model.N/ blocks to localize the
            # residual int8 damage that neither statistical ranking nor targeted
            # surgery (head/router/attn-core) explains. Sanity rung first: pinning
            # EVERY module must reproduce ~fp16 accuracy or the damage is outside
            # all module prefixes.
            import re as _re
            mod_ids = sorted({int(m_.group(1)) for r in rows
                              if (m_ := _re.match(r"/model\.(\d+)/", r["layer"]))})
            head_mod = int(_re.match(r"/model\.(\d+)/", head_prefix).group(1))
            mod_ids = [i for i in mod_ids if i != head_mod]
            always = [head_prefix] + router_prefixes
            bise_rows = []

            def _build_eval(mods, tag):
                prefs = always + [f"/model.{i}/" for i in mods]
                try:
                    inf = build_engine(model_path, ep, "int8", args.imgsz, calib_imgs, cache,
                                       pin_fp16_prefixes=prefs, pin_fp16_layers=pins_used)
                except RuntimeError as e:
                    print(f"[bisect {tag}] build infeasible ({e})")
                    return None, None, None
                a50, amm = eval_engine(ep, str(yaml_path), args.imgsz, 1, args.workers,
                                       args.eval_max_images, None)
                print(f"[bisect {tag}] pinned modules {mods}: mAP50-95={amm:.4f} "
                      f"(d {amm - base_map:+.4f})")
                bise_rows.append({"tag": tag, "modules": str(mods), "mAP50": round(a50, 5),
                                  "mAP50_95": round(amm, 5), "d": round(amm - base_map, 5)})
                return inf, a50, amm

            info, m50, mm = _build_eval(mod_ids, "sanity-all")
            cand = list(mod_ids)
            while info is not None and len(cand) > 1 and mm is not None                     and mm - base_map >= -args.budget:
                half = len(cand) // 2
                lo, hi = cand[:half], cand[half:]
                i_lo, s_lo, m_lo = _build_eval(lo, f"lo:{lo[0]}-{lo[-1]}")
                i_hi, s_hi, m_hi = _build_eval(hi, f"hi:{hi[0]}-{hi[-1]}")
                if m_lo is None and m_hi is None:
                    break
                if (m_lo or -1) >= (m_hi or -1):
                    cand, info, m50, mm = lo, i_lo, s_lo, m_lo
                else:
                    cand, info, m50, mm = hi, i_hi, s_hi, m_hi
                if mm - base_map < -args.budget:
                    print(f"[bisect] neither half alone recovers within budget at {cand}: "
                          f"damage is distributed; stopping at this granularity")
                    break
            print(f"[bisect] final module set: {cand}")
            import csv as _csv
            with open(out / "bisect.csv", "w", newline="") as bf:
                bw = _csv.DictWriter(bf, fieldnames=["tag", "modules", "mAP50", "mAP50_95", "d"])
                bw.writeheader(); bw.writerows(bise_rows)
            if info is None:
                raise RuntimeError("bisect: no feasible build")
        elif mode == "int8":
            # greedy per-layer pinning loop on top of the head pin
            recipe = [f"/model.{i.strip()}/" for i in args.pin_modules.split(",") if i.strip()]
            if recipe:
                print(f"[recipe] always-pinned modules: {recipe}")
            round_i = 0
            nxt: list[str] = []
            while True:
                try:
                    info = build_engine(model_path, ep, "int8", args.imgsz, calib_imgs, cache,
                                        pin_fp16_prefixes=base_pins + recipe,
                                        pin_fp16_layers=pins_used)
                except RuntimeError as e:
                    # OBEY hazard: a pin that splits a fusion group (e.g. only the Mul
                    # of a Conv+Sigmoid+Mul SiLU) can leave no implementable kernel.
                    # Drop the offending batch and rebuild the last feasible engine.
                    print(f"[int8 round {round_i}] build infeasible with batch {nxt[:3]}...; "
                          f"reverting batch and stopping greedy ({e})")
                    pins_used.difference_update(nxt)
                    try:
                        info = build_engine(model_path, ep, "int8", args.imgsz, calib_imgs, cache,
                                            pin_fp16_prefixes=base_pins + recipe,
                                            pin_fp16_layers=pins_used)
                    except RuntimeError as e2:
                        # infeasible even without greedy pins: the always-pin set itself
                        # splits a fusion group on this graph. That is a finding; record
                        # it and keep the ladder going.
                        print(f"[int8] FAILED (base pins {args.pins}): {e2}")
                        (out / "failed_int8.txt").write_text(f"pins={args.pins} {e2}\n")
                        info = None
                        break
                    m50, mm = eval_engine(ep, str(yaml_path), args.imgsz, 1, args.workers,
                                          args.eval_max_images, None)
                    break
                m50, mm = eval_engine(ep, str(yaml_path), args.imgsz, 1, args.workers,
                                      args.eval_max_images, None)
                d = mm - base_map if base_map is not None else 0.0
                print(f"[int8 round {round_i}] pins={len(pins_used)}+head "
                      f"mAP50-95={mm:.4f} (d {d:+.4f})")
                if base_map is None or d >= -args.budget or round_i >= args.max_rounds:
                    break
                nxt = [l for l in ranked if l not in pins_used][:args.pin_step]
                if not nxt:
                    break
                pins_used.update(nxt)
                print(f"          pinning next {len(nxt)} ranked layers: {nxt[:3]}...")
                round_i += 1
            if info is None:
                continue   # infeasible base-pin set recorded above; next mode
        else:
            info = build_engine(model_path, ep, mode, args.imgsz,
                                calib_imgs if mode == "int8-naive" else None,
                                cache if mode == "int8-naive" else None)
            m50, mm = eval_engine(ep, str(yaml_path), args.imgsz, 1, args.workers,
                                  args.eval_max_images, None)
        lat = time_engine(ep, args.imgsz)
        if mode == "fp32":
            base_map = mm
        row = {"mode": mode, **lat, "mAP50": round(m50, 5), "mAP50_95": round(mm, 5),
               "dmAP50_95": round(mm - (base_map if base_map is not None else mm), 5),
               "build_s": info["build_s"], "size_MB": info["size_MB"],
               "n_pinned_fp16": info.get("n_pinned_fp16", 0)}
        ladder.append(row)
        print(f"[{mode}] {lat['lat_ms_median']}ms  mAP50-95 {mm:.4f}  size {info['size_MB']}MB")
        _write_ladder(ladder, out)   # rows survive a later infeasible rung

    _write_ladder(ladder, out)
    (out / "pins.json").write_text(json.dumps({"head_prefix": head_prefix,
                                               "greedy_pins": sorted(pins_used)}, indent=2))
    print("\n===== ladder =====")
    for r in ladder:
        print(f"  {r['mode']:12s} {r['lat_ms_median']:>8}ms  cut {r.get('lat_cut_pct', 0):>5}%  "
              f"mAP50-95 {r['mAP50_95']}  d {r['dmAP50_95']:+}")
    print(f"artifacts -> {out}")


if __name__ == "__main__":
    main()
