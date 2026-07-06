#!/usr/bin/env python3
"""Export a fine-tuned YOLO-Master-EsMoE-N to ONNX + NCNN + MNN for edge deployment.

Pipeline (issue #51 requirements):
  * ONNX  : ultralytics onnxslim simplify + an explicit onnxsim pass + onnx.checker opset verify.
  * NCNN  : pnnx conversion, param/bin existence + load check.
  * MNN   : FP32 convert + optional INT8 weight quant.
  * INT8  : (optional) MNN post-training quantization with a >=300-image calibration set.

Each format is wrapped independently so a single failure does not abort the rest.

Usage (from project root):
    python scripts/export_models.py --model runs/train/esmoe_n_visdrone/weights/best.pt --imgsz 640
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import traceback
from pathlib import Path

from ultralytics import YOLO


PROJ_ROOT = Path(__file__).resolve().parents[1]
TRAINED_BEST = PROJ_ROOT.parent / "runs" / "train" / "esmoe_n_visdrone" / "weights" / "best.pt"
COCO_PT = PROJ_ROOT.parent / "weights" / "YOLO-Master-EsMoE-N.pt"
EXPORT_DIR = PROJ_ROOT.parent / "exports"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export EsMoE-N to ONNX/NCNN/MNN")
    default_model = TRAINED_BEST if TRAINED_BEST.exists() else COCO_PT
    p.add_argument("--model", type=Path, default=default_model)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--half", action="store_true", help="FP16 export (NCNN/MNN)")
    p.add_argument("--int8", action="store_true", help="MNN INT8 weight quant")
    p.add_argument(
        "--calib",
        type=Path,
        default=PROJ_ROOT.parent / "datasets" / "VisDrone" / "images" / "train",
        help="calibration image dir for INT8 PTQ (need >=300 images)",
    )
    p.add_argument("--no-mnn", action="store_true")
    p.add_argument("--no-ncnn", action="store_true")
    return p.parse_args()


def _fmt_size(n_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f}{unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f}TB"


def _dir_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def export_onnx(model: YOLO, imgsz: int, opset: int) -> dict:
    """Export ONNX with ultralytics onnxslim, then run a standalone onnxsim + checker pass."""
    info = {"format": "ONNX", "ok": False}
    out = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=True, dynamic=False)
    onnx_path = Path(out)
    info["path"] = str(onnx_path)

    import onnx

    # Explicit onnxsim pass (issue asks for onnxsim specifically, in addition to onnxslim).
    try:
        import onnxsim

        simp_path = onnx_path.with_name(onnx_path.stem + "_sim.onnx")
        model_sim, check = onnxsim.simplify(str(onnx_path), overwrite_input_shapes=[["images", 1, 3, imgsz, imgsz]])
        onnx.save(model_sim, str(simp_path))
        info["onnxsim"] = str(simp_path)
        info["onnxsim_check"] = bool(check)
    except Exception as e:  # onnxsim optional, do not fail the whole export
        info["onnxsim_error"] = str(e)

    m = onnx.load(str(onnx_path))
    onnx.checker.check_model(m)
    info["ok"] = True
    info["opset"] = next((op.version for op in m.opset_import if op.domain in ("", "ai.onnx")), None)
    info["ir_version"] = m.ir_version
    info["size"] = _fmt_size(_dir_size(onnx_path))
    return info


def export_ncnn(model: YOLO, src_model: Path, imgsz: int, half: bool) -> dict:
    """Export NCNN via pnnx, tolerantly. pnnx emits a reference model_pnnx.py whose
    codegen currently breaks on the MoE `where` op (SyntaxError) — that file is only a
    python reference and does NOT affect the .param/.bin. We therefore:

      1. call model.export(format='ncnn'), catching the post-codegen SyntaxError,
      2. independently verify the .param/.bin pnnx already wrote,
      3. detect unsupported ops (e.g. the `topk` from MoE dynamic routing, which ncnn
         fails to register) by capturing load warnings.
    """
    import io
    import contextlib

    info = {"format": "NCNN", "ok": False}
    ncnn_dir = src_model.parent / (src_model.stem + "_ncnn_model")
    try:
        model.export(format="ncnn", imgsz=imgsz, half=half)
    except SyntaxError as e:
        # pnnx's generated model_pnnx.py has invalid Python (aten::where codegen) —
        # the ncnn param/bin are already written; record and continue to verification.
        info["pnnx_codegen_note"] = f"model_pnnx.py SyntaxError (non-fatal for param/bin): {e}"
    param = ncnn_dir / "model.ncnn.param"
    binf = ncnn_dir / "model.ncnn.bin"
    info["param"] = str(param)
    info["bin"] = str(binf)
    if not (param.exists() and binf.exists()):
        info["error"] = "ncnn param/bin missing after pnnx"
        return info
    text = param.read_text(errors="ignore").splitlines()
    info["layers"] = int(text[0].split()[1]) if text and len(text[0].split()) > 1 else None
    # Grep for pnnx no-op passthrough layers (e.g. 'torch.topk') that ncnn can't register.
    noop_layers = sorted({l.split()[0] for l in text[1:] if l.startswith("torch.")})
    info["noop_layers"] = noop_layers

    import ncnn

    net = ncnn.Net()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        net.load_param(str(param))
        net.load_model(str(binf))
        ex = net.create_extractor()
        in_name = [l.split()[2] for l in text[1:] if l.startswith("Input")]
        if in_name:
            import numpy as np

            ex.input(in_name[0], ncnn.Mat(np.zeros((1, 3, imgsz, imgsz), dtype=np.float32)))
    log = buf.getvalue()
    info["ncnn_log_tail"] = log.strip().splitlines()[-3:] if log.strip() else []
    missing = [ln for ln in log.splitlines() if "not exists or registered" in ln]
    if missing:
        info["error"] = "ncnn graph has unsupported ops: " + "; ".join(
            m.replace("layer ", "").replace(" not exists or registered", "") for m in missing
        )
        info["size"] = _fmt_size(_dir_size(ncnn_dir))
        return info
    info["ok"] = True
    info["size"] = _fmt_size(_dir_size(ncnn_dir))
    return info


def export_mnn(model: YOLO, imgsz: int, half: bool, int8: bool) -> dict:
    """Export MNN (FP32 and optional INT8 weight quant)."""
    info = {"format": "MNN", "ok": False}
    out = model.export(format="mnn", imgsz=imgsz, half=half, int8=int8)
    mnn_path = Path(out)
    info["path"] = str(mnn_path)
    assert mnn_path.exists(), "mnn file missing"
    import MNN

    info["mnn_version"] = MNN.version()
    info["ok"] = True
    info["size"] = _fmt_size(_dir_size(mnn_path))
    if int8:
        info["quant"] = "int8-weight-quant"
    return info


def mnn_ptq_int8(src_mnn: Path, calib_dir: Path, out_mnn: Path, imgsz: int, n_images: int = 300) -> dict:
    """Bonus: full MNN post-training INT8 quantization with a >=300-image calibration set.

    Uses the `mnnquant` CLI (`src dst config.json`) — the only stable quantize entrypoint
    in MNN>=3.x — driven by a config JSON pointing at a >=300-image calibration list.
    """
    info = {"format": "MNN-INT8-PTQ", "ok": False}
    try:
        import sys

        imgs = sorted(p for p in calib_dir.rglob("*") if p.suffix.lower() in (".jpg", ".png", ".jpeg"))
        if len(imgs) < n_images:
            info["error"] = f"only {len(imgs)} calibration images (need {n_images})"
            return info
        out_mnn.parent.mkdir(parents=True, exist_ok=True)
        calib_list = out_mnn.parent / "calib_images.txt"
        calib_list.write_text("\n".join(str(p) for p in imgs[:n_images]))
        config = {
            "image_path": str(calib_list),
            "mean": [0.0, 0.0, 0.0],
            "normal": [1 / 255.0, 1 / 255.0, 1 / 255.0],
            "shape": [1, 3, imgsz, imgsz],
            "quant_bit": 8,
            "batch_count": 1,
        }
        cfg_path = out_mnn.parent / "quant_config.json"
        cfg_path.write_text(json.dumps(config))
        cmd = [sys.executable, "-m", "MNN.tools.mnnquant", str(src_mnn), str(out_mnn), str(cfg_path)]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        info["path"] = str(out_mnn)
        info["n_calib"] = n_images
        info["log_tail"] = proc.stdout.strip().splitlines()[-2:] + proc.stderr.strip().splitlines()[-2:]
        if out_mnn.exists():
            info["ok"] = True
            info["size"] = _fmt_size(_dir_size(out_mnn))
        else:
            info["error"] = "mnnquant produced no output"
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def main() -> None:
    args = parse_args()
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[export] model={args.model} imgsz={args.imgsz} opset={args.opset}")
    model = YOLO(str(args.model))

    # NOTE: keep the model's native (use_top_k=True) routing for ONNX/MNN here, so that
    # the patched ES_MOE._dense_forward (see scripts/patch_dense_forward.py) applies its
    # topk+threshold export-time pruning and the exported graph matches eager sparse eval
    # (this is what reproduces the README's 6% mAP / <0.5% consistency).
    # NCNN is exported separately by export_ncnn_dense.py, which DOES set use_top_k=False
    # (full-softmax) because pnnx cannot lower topk/comparison ops to NCNN.

    results = []
    try:
        results.append(export_onnx(model, args.imgsz, args.opset))
    except Exception as e:
        results.append({"format": "ONNX", "ok": False, "error": f"{type(e).__name__}: {e}"})
        traceback.print_exc()

    if not args.no_ncnn:
        try:
            results.append(export_ncnn(model, args.model, args.imgsz, args.half))
        except Exception as e:
            results.append({"format": "NCNN", "ok": False, "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()

    if not args.no_mnn:
        try:
            results.append(export_mnn(model, args.imgsz, args.half, args.int8))
        except Exception as e:
            results.append({"format": "MNN", "ok": False, "error": f"{type(e).__name__}: {e}"})
            traceback.print_exc()

        # Bonus: true PTQ INT8 with calibration >=300 images (source = FP32 .mnn).
        mnn_fp32 = Path(args.model).with_suffix(".mnn")
        if mnn_fp32.exists() and args.calib.exists():
            ptq_out = EXPORT_DIR / (Path(args.model).stem + "_int8_ptq.mnn")
            results.append(mnn_ptq_int8(mnn_fp32, args.calib, ptq_out, args.imgsz))

    # Gather exported artifacts into exports/ for a clean deployment layout.
    import shutil

    for r in results:
        for key in ("path", "param", "onnxsim"):
            p = r.get(key)
            if p and Path(p).exists() and EXPORT_DIR not in Path(p).parents:
                dst = EXPORT_DIR / Path(p).name
                try:
                    shutil.copy2(p, dst)
                    r[key] = str(dst)
                except Exception:
                    pass
    ncnn_src_dir = Path(args.model).parent / (Path(args.model).stem + "_ncnn_model")
    if ncnn_src_dir.exists():
        shutil.copytree(ncnn_src_dir, EXPORT_DIR / ncnn_src_dir.name, dirs_exist_ok=True)

    summary_path = EXPORT_DIR / "export_summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("\n=== EXPORT SUMMARY ===")
    for r in results:
        status = "OK" if r.get("ok") else "FAIL"
        line = f"[{status}] {r['format']}"
        if r.get("ok"):
            line += f" size={r.get('size')}"
            if "opset" in r:
                line += f" opset={r['opset']} ir={r['ir_version']}"
            if "layers" in r:
                line += f" layers={r['layers']}"
            if "onnxsim" in r:
                line += f" onnxsim_check={r.get('onnxsim_check')}"
        else:
            line += f" err={r.get('error')}"
        print(line)
    print(f"\n[export] summary written to {summary_path}")


if __name__ == "__main__":
    main()
