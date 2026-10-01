#!/usr/bin/env python
"""Prepare the API-server model set: fp16 ONNX siblings (+ ORT-CPU parity vs fp32) and a
uniform per-model directory layout.

    python scripts/server/prepare_models.py --src v01n=runs/a3/v01n/YOLO-Master-v0.1-N.onnx \
        --src v01n-pruned=runs/project03/trt/v01-n-coco-pruned-surgery/YOLO-Master-v0.1-N_pruned_t0.10.onnx \
        --src esmoen=runs/a3/esmoen/YOLO-Master-EsMoE-N.onnx --out models/api --probe visdrone50/images/val/x.jpg

Writes models/api/<id>/model.onnx (copy of the fp32 export), model-fp16.onnx (graph in fp16,
fp32 I/O so every runtime binds float tensors), metadata.yaml (names/imgsz from the ONNX
metadata), and prepare_report.json with the fp32-vs-fp16 drift on the probe image.
"""
import argparse, sys, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import onnx
from onnxconverter_common import float16
import onnxruntime as ort


def probe_blob(path, imgsz):
    import cv2
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    r = min(imgsz / w, imgsz / h)
    nw, nh = int(round(w * r)), int(round(h * r))
    canvas = np.full((imgsz, imgsz, 3), 114, np.uint8)
    px, py = (imgsz - nw) // 2, (imgsz - nh) // 2
    canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh))
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


def run(onnx_path, blob):
    s = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = s.get_inputs()[0].name
    t0 = time.perf_counter(); out = s.run(None, {name: blob}); ms = (time.perf_counter() - t0) * 1000
    return out, ms


def write_metadata(onnx_path, dst):
    m = onnx.load(str(onnx_path), load_external_data=False)
    props = {p.key: p.value for p in m.metadata_props}
    shape = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    imgsz = shape[2] if len(shape) == 4 else 640
    names = None
    if "names" in props:
        try:
            import ast
            names = ast.literal_eval(props["names"])
        except Exception:
            names = None
    lines = ["description: YOLO-Master API model (scripts/server/prepare_models.py)", f"task: {props.get('task', 'detect')}",
             "batch: 1", "imgsz:", f"- {imgsz}", f"- {imgsz}", "end2end: false"]
    if names:
        lines.append("names:")
        for k in sorted(names, key=int):
            lines.append(f"  {k}: {names[k]}")
    (dst / "metadata.yaml").write_text("\n".join(lines) + "\n")
    return imgsz, names


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", action="append", required=True, help="id=path/to/fp32.onnx (repeatable)")
    ap.add_argument("--out", default="models/api")
    ap.add_argument("--probe", required=True)
    ap.add_argument("--keep-io-types", type=int, default=1)
    a = ap.parse_args()
    report = {}
    for spec in a.src:
        mid, src = spec.split("=", 1)
        dst = Path(a.out) / mid; dst.mkdir(parents=True, exist_ok=True)
        fp32 = dst / "model.onnx"
        if not fp32.exists() or fp32.stat().st_size != Path(src).stat().st_size:
            shutil.copyfile(src, fp32)
        imgsz, names = write_metadata(fp32, dst)
        # routing ops (TopK/GatherElements/Where) stay fp32: the converter mistypes TopK's value output,
        # and the expert-selection indices must not depend on fp16 rounding anyway. Nodes the converter
        # still mistypes (reported by ORT at load) are added to node_block_list and the conversion retried.
        block = list(float16.DEFAULT_OP_BLOCK_LIST) + ["TopK", "GatherElements", "Where", "ScatterElements", "ScatterND"]
        fp16 = dst / "model-fp16.onnx"
        node_block = []
        for attempt in range(12):
            m = onnx.load(str(fp32))
            del m.graph.value_info[:]   # stale float value_info makes the converter emit mistyped casts
            m = onnx.shape_inference.infer_shapes(m)
            if attempt == 0:            # the whole MoE routing subgraph stays fp32 (expert selection must not
                node_block = [n.name for n in m.graph.node if "/routing/" in n.name]   # depend on fp16 rounding)
            m16 = float16.convert_float_to_float16(m, keep_io_types=bool(a.keep_io_types),
                                                   op_block_list=block, node_block_list=node_block or None)
            onnx.save(m16, str(fp16))
            try:
                ort.InferenceSession(str(fp16), providers=["CPUExecutionProvider"]); break
            except Exception as e:  # noqa
                import re
                mm = re.search(r"node \(([^)]+)\)", str(e))
                name = mm.group(1) if mm else None
                if not name or name in node_block:
                    raise
                node_block.append(name)
                print(f"  [{mid}] fp16 retry {attempt + 1}: keeping node {name} in fp32", flush=True)
        report_nodes = list(node_block)
        # The routing subgraph cannot run in fp16 anywhere (1e30-class masks, TopK on rounded gates):
        # stamp the sidecar so the MNN / ncnn backends refuse whole-model fp16 and use their mixed
        # modes instead (MNN: the paths sidecar written by mnn_router_paths.py; ncnn: Layer::featmask).
        routed = any("/routing/" in n.name for n in onnx.load(str(fp32)).graph.node)
        meta_path = dst / "metadata.yaml"
        if meta_path.exists():
            lines = [l for l in meta_path.read_text().splitlines() if not l.startswith("fp16_safe:")]
            lines.append(f"fp16_safe: {'false' if routed else 'true'}")
            meta_path.write_text("\n".join(lines) + "\n")
        if routed:
            import subprocess
            subprocess.run([sys.executable, str(Path(__file__).with_name("mnn_router_paths.py")), str(fp32),
                            "--out", str(dst / "model.paths.json")], check=True)
        blob = probe_blob(a.probe, imgsz)
        o32, ms32 = run(fp32, blob)
        o16, ms16 = run(fp16, blob)
        drift = float(np.abs(o32[0].astype(np.float32) - o16[0].astype(np.float32)).max())
        rel = drift / (float(np.abs(o32[0]).max()) + 1e-9)
        # detections above conf 0.25 (argmax over classes; rows [4+nc, anchors])
        def count(o):
            arr = o[0][0]; return int((arr[4:].max(axis=0) > 0.25).sum())
        report[mid] = {"src": src, "imgsz": imgsz, "nc": len(names) if names else None,
                       "fp32_bytes": fp32.stat().st_size, "fp16_bytes": fp16.stat().st_size,
                       "ort_cpu_ms_fp32": round(ms32, 1), "ort_cpu_ms_fp16": round(ms16, 1),
                       "max_abs_drift": drift, "max_rel_drift": rel, "cands_fp32": count(o32), "cands_fp16": count(o16),
                       "fp32_pinned_nodes": report_nodes}
        print(f"{mid}: fp32 {fp32.stat().st_size/1e6:.1f} MB -> fp16 {fp16.stat().st_size/1e6:.1f} MB  drift max|d|={drift:.4f} "
              f"(rel {rel:.2e})  cands@0.25 {count(o32)} vs {count(o16)}", flush=True)
    (Path(a.out) / "prepare_report.json").write_text(json.dumps(report, indent=2))
    print("wrote", Path(a.out) / "prepare_report.json")


if __name__ == "__main__":
    main()
