#!/usr/bin/env python
"""Side-by-side of two A3 evidence sets (e.g. a3/results-3ea98305 vs a3/results) and a
flip verdict: did any conclusion change, or only the absolute numbers?

  python scripts/a3/compare_baselines.py --old a3/results-3ea98305 --new a3/results
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def ladder(d: Path, name: str) -> dict:
    p = d / name
    if not p.exists():
        return {}
    return {r["mode"]: r for r in csv.DictReader(open(p))}


def jload(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text()) if p.exists() else None


def f(x, nd=4):
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default="a3/results-3ea98305")
    ap.add_argument("--new", default="a3/results")
    ap.add_argument("--labels", default="3ea98305,acce839c")
    args = ap.parse_args()
    lo, ln = args.labels.split(",")
    old, new = Path(args.old), Path(args.new)
    verdicts = []

    for tag, title in (("v01n", "v0.1-N"), ("esmoen", "EsMoE-N")):
        print(f"\n=== {title} ===")
        print(f"{'rung':34s} {lo+' mAP':>14s} {ln+' mAP':>14s} {lo+' d':>9s} {ln+' d':>9s} {lo+' ms':>8s} {ln+' ms':>8s}")
        bo, bn = jload(old, f"backend_{tag}.json"), jload(new, f"backend_{tag}.json")
        po = bo["pytorch"]["mAP50_95"] if bo else None
        pn = bn["pytorch"]["mAP50_95"] if bn else None
        rows = [("PyTorch", po, pn, 0, 0, None, None)]
        if bo and bn:
            rows.append(("ONNX Runtime CUDA EP", bo["ort"]["mAP50_95"], bn["ort"]["mAP50_95"],
                         bo["ort"]["delta_mAP50_95"], bn["ort"]["delta_mAP50_95"], None, None))
        Lo, Ln = ladder(old, f"ladder_{tag}.csv"), ladder(new, f"ladder_{tag}.csv")
        for mode, label in (("fp32", "TRT fp32 (TF32 off)"), ("fp16", "TRT fp16"),
                            ("int8", "TRT implicit int8 (head+routers)")):
            ro, rn = Lo.get(mode), Ln.get(mode)
            rows.append((label, ro and ro["mAP50_95"], rn and rn["mAP50_95"],
                         ro and ro["dmAP50_95"], rn and rn["dmAP50_95"],
                         ro and ro["lat_ms_median"], rn and rn["lat_ms_median"]))
        So, Sn = ladder(old, f"ladder_{tag}_stempair.csv"), ladder(new, f"ladder_{tag}_stempair.csv")
        fo, fn = Lo.get("fp32"), Ln.get("fp32")
        so, sn = So.get("int8"), Sn.get("int8")
        rows.append(("TRT implicit int8 + stem pair",
                     so and so["mAP50_95"], sn and sn["mAP50_95"],
                     so and fo and float(so["mAP50_95"]) - float(fo["mAP50_95"]),
                     sn and fn and float(sn["mAP50_95"]) - float(fn["mAP50_95"]),
                     so and so["lat_ms_median"], sn and sn["lat_ms_median"]))
        qo, qn = jload(old, f"qdq_{tag}.json"), jload(new, f"qdq_{tag}.json")
        rows.append(("explicit Q/DQ calibrate-only",
                     qo and qo["model"]["mAP50_95"], qn and qn["model"]["mAP50_95"],
                     qo and fo and qo["model"]["mAP50_95"] - float(fo["mAP50_95"]),
                     qn and fn and qn["model"]["mAP50_95"] - float(fn["mAP50_95"]),
                     qo and qo["model"]["lat_ms_median"], qn and qn["model"]["lat_ms_median"]))
        for label, mo, mn, do, dn, to, tn in rows:
            print(f"{label:34s} {f(mo):>14s} {f(mn):>14s} {f(do):>9s} {f(dn):>9s} {f(to,3):>8s} {f(tn,3):>8s}")

        # conclusions that could flip
        def pick(Lx, mode, key):
            r = Lx.get(mode); return float(r[key]) if r else None
        for name, cond in (
            ("fp16 lossless (|d| <= 0.002)",
             (lambda L: pick(L, "fp16", "dmAP50_95") is not None and abs(pick(L, "fp16", "dmAP50_95")) <= 0.002)),
            ("implicit int8 NOT faster than fp16",
             (lambda L: pick(L, "int8", "lat_ms_median") is not None and pick(L, "fp16", "lat_ms_median") is not None
              and pick(L, "int8", "lat_ms_median") >= pick(L, "fp16", "lat_ms_median") * 0.97)),
        ):
            vo, vn = cond(Lo), cond(Ln)
            verdicts.append((title, name, vo, vn))
        if qo and qn and fo and fn:
            vo = qo["model"]["mAP50_95"] - float(fo["mAP50_95"]) >= -0.02
            vn = qn["model"]["mAP50_95"] - float(fn["mAP50_95"]) >= -0.02
            verdicts.append((title, "explicit Q/DQ within 2 AP of fp32", vo, vn))
        so_, sn_ = jload(old, f"smoke_{tag}.json"), jload(new, f"smoke_{tag}.json")
        if so_ and sn_:
            verdicts.append((title, "smoke passes", so_["status"] == "pass", sn_["status"] == "pass"))
            d_o = {d["module"]: d["decision"] for d in so_["policy"]["decisions"]}
            d_n = {d["module"]: d["decision"] for d in sn_["policy"]["decisions"]}
            verdicts.append((title, f"routing decisions {d_n}", d_o == d_n, True))
        if Lo.get("int8") is None or Ln.get("int8") is None:
            verdicts.append((title, "implicit int8 build feasible", Lo.get("int8") is not None, Ln.get("int8") is not None))

    print("\n=== verdicts (old -> new) ===")
    flips = 0
    for title, name, vo, vn in verdicts:
        flag = "" if vo == vn else "   <-- FLIP"
        flips += vo != vn
        print(f"{title:8s} {name:60s} {str(vo):5s} -> {str(vn):5s}{flag}")
    print(f"\n{flips} conclusion flip(s)")


if __name__ == "__main__":
    main()
