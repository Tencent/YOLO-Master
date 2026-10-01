#!/usr/bin/env python3
"""Certify a mixed-INT8 ncnn export against its float source on a val set (accuracy gate).

Runs the edge CLI twice at ultralytics val settings (`--conf 0.001 --iou 0.7
--multi-label`, max_det 300), scores both `--save-txt` dumps with
`scripts/eval_map.py::evaluate` (ultralytics DetMetrics), then applies:

  * GATE (hard):  mAP50-95(int8) >= mAP50-95(fp32) - `--gate-ap` (default 0.010 = 1.0 AP pt)
  * match (hard): mean per-image greedy same-class IoU>=0.8 match rate of the fp32 boxes in
    the int8 output at `--conf-parity` on the first `--parity-images` images >= `--match-rate`
  * det ratio (warn only): total_dets(int8)/total_dets(fp32) at conf 0.001 outside [0.85, 1.15]

The int8 run's `ep=` line must contain `int8`, otherwise the wrong model was scored and
the run aborts. Any two explicit dirs are accepted (`--fp32 X_ncnn --int8 X-int8-<tag>_ncnn`);
the runtime's `<name>-int8_ncnn` sibling rule is deliberately not used.

Exit 0 iff gate and match-rate pass; exit 1 otherwise. The JSON report and the markdown
row are ALWAYS written. The x86 ms/img column is a load/parse sanity number only: x86
int8 (AVX-VNNI) says nothing about ARM sdot/i8mm speed - that verdict comes from the device.

Interpreter: /root/anaconda3/envs/yolo_master/bin/python (ultralytics for DetMetrics).

Example:
    tests/certify_ncnn_int8.py --fp32 models/p03_v01n_ncnn --int8 models/p03_v01n-int8_ncnn \\
        --images /data/datasets/coco/images/val2017 --labels /data/datasets/coco/labels/val2017 \\
        --limit 200 --report results/int8_cert/p03_smoke200.json
"""
import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_ROOT / "scripts"))

from validate_mixture import match_rate, read_savetxt  # noqa: E402  (greedy same-class IoU matcher)
import eval_map  # noqa: E402  (ultralytics DetMetrics scorer; imports torch lazily at module load)

VAL_ARGS = ["--conf", "0.001", "--iou", "0.7", "--multi-label"]
DET_RATIO_BAND = (0.85, 1.15)


def _md5(path):
    """Return the hex md5 of a file (empty string when missing)."""
    p = Path(path)
    if not p.is_file():
        return ""
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_model(spec):
    """Resolve a model dir or `.param` path to (model_arg, dir, param, bin, metadata_yaml).

    Args:
        spec: an ncnn export dir (`model.ncnn.param/.bin` inside) or a bare `.param` path.

    Returns:
        tuple: (`-m` argument as given, directory, param path, bin path, metadata.yaml path or None).
    """
    p = Path(spec)
    if p.is_dir():
        param = p / "model.ncnn.param"
        binp = p / "model.ncnn.bin"
        meta = p / "metadata.yaml"
    elif p.suffix == ".param":
        param = p
        binp = p.with_suffix(".bin")
        meta = p.parent / "metadata.yaml"
    else:
        raise SystemExit(f"model must be an ncnn dir or a .param file: {spec}")
    if not param.is_file() or not binp.is_file():
        raise SystemExit(f"ncnn model files missing under {spec}: {param.name} / {binp.name}")
    return str(spec), p if p.is_dir() else p.parent, param, binp, (meta if meta.is_file() else None)


def read_quant_block(meta_path):
    """Return the `quant:` mapping and `precision:` scalar from an int8 metadata.yaml.

    Args:
        meta_path: path to metadata.yaml (may be None).

    Returns:
        dict: {"precision": str|None, "quant": dict|str|None}. `quant` falls back to the raw
        text of the block when pyyaml is unavailable or the file does not parse.
    """
    out = {"precision": None, "quant": None}
    if meta_path is None:
        return out
    text = Path(meta_path).read_text(encoding="utf-8")
    try:
        import yaml
        doc = yaml.safe_load(text) or {}
        out["precision"] = doc.get("precision")
        out["quant"] = doc.get("quant")
        return out
    except Exception:  # pragma: no cover - pyyaml missing or odd scalars
        m = re.search(r"^precision:\s*(\S+)", text, re.M)
        out["precision"] = m.group(1) if m else None
        m = re.search(r"^quant:\n((?:[ \t]+.*\n?)+)", text, re.M)
        out["quant"] = m.group(1) if m else None
        return out


def parse_cli_log(log):
    """Extract the `[model]` and `[summary]` fields from a yolomaster_edge stdout capture.

    Args:
        log: combined stdout text of one CLI run.

    Returns:
        dict: ep, note, frames, total_dets, infer_ms, total_ms (None when a field is absent).
    """
    d = {"ep": None, "note": "", "frames": None, "total_dets": None, "infer_ms": None, "total_ms": None}
    m = re.search(r"\[model\].*?\bep=(\S+)", log)
    if m:
        d["ep"] = m.group(1)
    m = re.search(r"\[model\].*?\bnote=(.*)$", log, re.M)      # "  note=<free text>" ends the [model] line
    if m:
        d["note"] = m.group(1).strip()
    m = re.search(r"\[summary\]\s+frames=(\d+)\s+total_dets=(\d+).*?infer=([\d.]+).*?total=([\d.]+)ms", log)
    if m:
        d["frames"] = int(m.group(1))
        d["total_dets"] = int(m.group(2))
        d["infer_ms"] = float(m.group(3))
        d["total_ms"] = float(m.group(4))
    return d


def run_cli(bin_path, model_arg, images, save_dir, precision, threads, limit, conf_args):
    """Run the edge CLI once with `--save-txt` and parse its log.

    Args:
        bin_path: yolomaster_edge binary.
        model_arg: value for `-m` (dir or .param).
        images: image dir for `-s`.
        save_dir: `--save-txt` destination (created).
        precision: `--precision` value (`fp32` for the float run, `auto` for the int8 dir).
        threads: `--threads`.
        limit: `--limit` (0 = all).
        conf_args: threshold args, e.g. VAL_ARGS or ["--conf", "0.25", ...].

    Returns:
        dict: parsed log fields plus `cmd` (shell string), `rc`, `stderr_tail`.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    cmd = [str(bin_path), "-m", str(model_arg), "-s", str(images), *conf_args,
           "--threads", str(threads), "--quiet", "--no-save", "--save-txt", str(save_dir),
           "--precision", precision]
    if limit and limit > 0:
        cmd += ["--limit", str(limit)]
    print(f"[run] {shlex.join(cmd)}", flush=True)
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=str(_ROOT))
    info = parse_cli_log(p.stdout)
    info.update(cmd=shlex.join(cmd), rc=p.returncode, stderr_tail=p.stderr[-2000:])
    if p.returncode != 0 or info["frames"] is None:
        raise SystemExit(f"CLI failed (rc={p.returncode}) for {model_arg} [{precision}]:\n"
                         f"{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    print(f"       ep={info['ep']} frames={info['frames']} total_dets={info['total_dets']} "
          f"infer={info['infer_ms']:.2f}ms total={info['total_ms']:.2f}ms"
          + (f" note={info['note']}" if info["note"] else ""), flush=True)
    return info


def box_parity(fp32_dir, int8_dir, images, n):
    """Per-image greedy same-class IoU>=0.8 match of two save-txt dumps on the first `n` images.

    Args:
        fp32_dir: save-txt dir of the float run (reference boxes).
        int8_dir: save-txt dir of the int8 run.
        images: image dir (sorted `*.jpg` listing defines the first `n`).
        n: number of images.

    Returns:
        dict: match (mean fraction of fp32 boxes reproduced by int8), match_rev (mean fraction of
        int8 boxes present in fp32), per-image lists, and box totals.
    """
    stems = [p.stem for p in sorted(Path(images).glob("*.jpg"))[:n]]
    fwd, rev, n_ref, n_got = [], [], 0, 0
    for s in stems:
        rb, _, rc = read_savetxt(Path(fp32_dir) / f"{s}.txt")
        gb, _, gc = read_savetxt(Path(int8_dir) / f"{s}.txt")
        n_ref += len(rb)
        n_got += len(gb)
        fwd.append(match_rate(rb, rc, gb, gc, iou_thr=0.8))
        rev.append(match_rate(gb, gc, rb, rc, iou_thr=0.8))
    return {
        "images": len(stems),
        "match": float(np.mean(fwd)) if fwd else 0.0,
        "match_rev": float(np.mean(rev)) if rev else 0.0,
        "fp32_boxes": int(n_ref),
        "int8_boxes": int(n_got),
        "per_image_match": [float(x) for x in fwd],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fp32", required=True, help="float ncnn export dir (reference)")
    ap.add_argument("--int8", required=True, help="int8 ncnn export dir (candidate; any name)")
    ap.add_argument("--images", required=True, help="val image dir (*.jpg)")
    ap.add_argument("--labels", required=True, help="YOLO-format label dir for --images")
    ap.add_argument("--limit", type=int, default=0, help="first N images (sorted) for the mAP runs; 0 = all")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--bin", default=str(_ROOT / "cpp/build_ncnn_x86/yolomaster_edge"))
    ap.add_argument("--conf-parity", type=float, default=0.25, help="conf for the box-match check")
    ap.add_argument("--parity-images", type=int, default=50, help="images for the box-match check")
    ap.add_argument("--report", required=True, help="JSON report path, e.g. results/int8_cert/<name>.json")
    ap.add_argument("--gate-ap", type=float, default=0.010,
                    help="max allowed mAP50-95 drop, absolute (0.010 = 1.0 AP point)")
    ap.add_argument("--match-rate", type=float, default=0.90, help="min mean box match rate (hard)")
    ap.add_argument("--workdir", default="", help="where the save-txt dumps go (default: a temp dir)")
    ap.add_argument("--keep", action="store_true", help="keep the save-txt dumps")
    args = ap.parse_args()

    bin_path = Path(args.bin)
    if not bin_path.is_file():
        raise SystemExit(f"runner binary not found: {bin_path}")
    fp32_arg, fp32_dir, fp32_param, fp32_bin, fp32_meta = resolve_model(args.fp32)
    int8_arg, int8_dir, int8_param, int8_bin, int8_meta = resolve_model(args.int8)
    images, labels = Path(args.images), Path(args.labels)
    if not images.is_dir() or not labels.is_dir():
        raise SystemExit(f"images/labels dir missing: {images} / {labels}")

    # class names: fp32 metadata -> int8 metadata -> VisDrone default; nc must agree.
    names_src = None
    names = None
    for src in (fp32_meta, int8_meta):
        if src is not None:
            try:
                names, names_src = eval_map.load_names_yaml(src), str(src)
                break
            except SystemExit:
                continue
    if names is None:
        names, names_src = [eval_map.NAMES[i] for i in sorted(eval_map.NAMES)], "builtin VisDrone"
    if fp32_meta is not None and int8_meta is not None:
        try:
            n_a, n_b = len(eval_map.load_names_yaml(fp32_meta)), len(eval_map.load_names_yaml(int8_meta))
        except SystemExit:
            n_a = n_b = None
        if n_a is not None and n_a != n_b:
            raise SystemExit(f"class count mismatch: {fp32_meta} has {n_a} names, {int8_meta} has {n_b} "
                             f"- these are not the same model; refusing to certify")
    print(f"[names] nc={len(names)} from {names_src}")

    model_name = Path(fp32_dir).name
    val_set = images.name
    n_par = args.parity_images if not args.limit else min(args.parity_images, args.limit)

    if args.workdir:
        work = Path(args.workdir)
        work.mkdir(parents=True, exist_ok=True)
    else:
        work = Path(tempfile.mkdtemp(prefix="int8_cert_"))
    print(f"[workdir] {work}" + ("" if args.keep or args.workdir else "  (deleted on exit unless --keep)"))

    report = {
        "model": model_name, "val_set": val_set, "images_dir": str(images), "labels_dir": str(labels),
        "fp32_dir": str(fp32_dir), "int8_dir": str(int8_dir),
        "fp32_bin_md5": _md5(fp32_bin), "int8_bin_md5": _md5(int8_bin),
        "fp32_bin_mb": round(fp32_bin.stat().st_size / 1e6, 3), "int8_bin_mb": round(int8_bin.stat().st_size / 1e6, 3),
        "names_source": names_src, "nc": len(names),
        "val_settings": {"conf": 0.001, "iou": 0.7, "multi_label": True, "max_det": 300},
        "limit": args.limit, "threads": args.threads, "bin": str(bin_path),
        "gate_ap": args.gate_ap, "match_rate_min": args.match_rate, "conf_parity": args.conf_parity,
        "parity_images": n_par, "det_ratio_band": list(DET_RATIO_BAND),
        "host": os.uname().nodename, "arch": os.uname().machine,
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "quant": read_quant_block(int8_meta), "commands": [], "runs": {},
        "verdict": "ERROR", "exit_code": 2, "error": None,
    }

    def write_report():
        out = Path(args.report)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"[report] {out}")

    try:
        # (1) val-settings runs
        r32 = run_cli(bin_path, fp32_arg, images, work / "fp32_val", "fp32", args.threads, args.limit, VAL_ARGS)
        report["commands"].append(r32["cmd"])
        report["runs"]["fp32_val"] = r32
        r8 = run_cli(bin_path, int8_arg, images, work / "int8_val", "auto", args.threads, args.limit, VAL_ARGS)
        report["commands"].append(r8["cmd"])
        report["runs"]["int8_val"] = r8
        if not r8["ep"] or "int8" not in r8["ep"]:
            raise SystemExit(f"ABORT: the --int8 model reported ep={r8['ep']!r}, which does not contain 'int8' "
                             f"- the wrong model was scored ({int8_arg} is not an int8 ncnn export; "
                             f"no int8 verdict can come from a float graph)")
        if r32["ep"] and "int8" in r32["ep"]:
            raise SystemExit(f"ABORT: the --fp32 model reported ep={r32['ep']!r} (an int8 graph cannot be the "
                             f"float reference)")
        if r32["frames"] != r8["frames"]:
            raise SystemExit(f"frame count mismatch fp32={r32['frames']} int8={r8['frames']}")

        # (2) mAP
        map50_32, map5095_32, n_img = eval_map.evaluate(work / "fp32_val", images, labels, names, limit=args.limit)
        map50_8, map5095_8, n_img8 = eval_map.evaluate(work / "int8_val", images, labels, names, limit=args.limit)
        assert n_img == n_img8
        if n_img != r32["frames"]:
            print(f"[warn] scored {n_img} images but the CLI ran {r32['frames']} frames "
                  f"(non-jpg inputs in {images}?)")
        delta = map5095_8 - map5095_32

        # (3) det-count ratio at conf 0.001
        ratio = (r8["total_dets"] / r32["total_dets"]) if r32["total_dets"] else float("nan")
        ratio_ok = DET_RATIO_BAND[0] <= ratio <= DET_RATIO_BAND[1]
        if not ratio_ok:
            print(f"[warn] det-count ratio int8/fp32 = {ratio:.3f} outside {DET_RATIO_BAND}")

        # (4) box parity at --conf-parity on the first n_par images
        par_args = ["--conf", str(args.conf_parity), "--iou", "0.7", "--multi-label"]
        p32 = run_cli(bin_path, fp32_arg, images, work / "fp32_par", "fp32", args.threads, n_par, par_args)
        p8 = run_cli(bin_path, int8_arg, images, work / "int8_par", "auto", args.threads, n_par, par_args)
        report["commands"] += [p32["cmd"], p8["cmd"]]
        report["runs"]["fp32_parity"] = p32
        report["runs"]["int8_parity"] = p8
        parity = box_parity(work / "fp32_par", work / "int8_par", images, n_par)
        match_ok = parity["match"] >= args.match_rate

        # (5) gate
        gate_ok = map5095_8 >= map5095_32 - args.gate_ap
        passed = gate_ok and match_ok
        verdict = "PASS" if passed else "FAIL"

        report.update({
            "n_images": n_img,
            "fp32": {"map50": map50_32, "map5095": map5095_32, "total_dets": r32["total_dets"],
                     "ep": r32["ep"], "note": r32["note"], "x86_infer_ms": r32["infer_ms"],
                     "x86_total_ms": r32["total_ms"]},
            "int8": {"map50": map50_8, "map5095": map5095_8, "total_dets": r8["total_dets"],
                     "ep": r8["ep"], "note": r8["note"], "x86_infer_ms": r8["infer_ms"],
                     "x86_total_ms": r8["total_ms"]},
            "delta_map5095": delta, "delta_map5095_pts": delta * 100.0,
            "det_ratio": ratio, "det_ratio_ok": ratio_ok,
            "parity": parity, "match_ok": match_ok,
            "gate_ok": gate_ok, "verdict": verdict, "exit_code": 0 if passed else 1, "error": None,
            "speed_note": "x86 ms/img is a load/parse sanity number only, not a speed result "
                          "(x86 int8 kernels != ARM sdot/i8mm)",
        })

        header = ("| model | val set | n | fp32 mAP50/50-95 | int8 mAP50/50-95 | delta50-95 (pts) | "
                  "dets fp32/int8 | ratio | match | int8 ep | int8 bin MB | "
                  "x86 ms/img fp32/int8 (x86, not a speed result) | PASS/FAIL |")
        row = (f"| {model_name} | {val_set} | {n_img} | {map50_32:.4f}/{map5095_32:.4f} | "
               f"{map50_8:.4f}/{map5095_8:.4f} | {delta * 100:+.2f} | "
               f"{r32['total_dets']}/{r8['total_dets']} | {ratio:.3f}{'' if ratio_ok else ' (!)'} | "
               f"{parity['match']:.3f}{'' if match_ok else ' (!)'} | {r8['ep']} | {report['int8_bin_mb']:.2f} | "
               f"{r32['infer_ms']:.1f}/{r8['infer_ms']:.1f} | {verdict} |")
        report["markdown_header"] = header
        report["markdown_row"] = row
        print()
        print(header)
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        print(row)
        print(f"\n[gate] mAP50-95 int8 {map5095_8:.4f} >= fp32 {map5095_32:.4f} - {args.gate_ap:.3f} "
              f"-> {'ok' if gate_ok else 'FAIL'};  match {parity['match']:.3f} >= {args.match_rate:.2f} "
              f"-> {'ok' if match_ok else 'FAIL'};  det ratio {ratio:.3f} "
              f"-> {'ok' if ratio_ok else 'WARN'}  => {verdict}")
        write_report()
        return 0 if passed else 1
    except SystemExit as e:
        report["error"] = str(e)
        report["verdict"] = "ERROR"
        report["exit_code"] = 2
        write_report()
        print(f"[error] {e}", file=sys.stderr)
        return 2
    finally:
        if not (args.keep or args.workdir):
            import shutil
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
