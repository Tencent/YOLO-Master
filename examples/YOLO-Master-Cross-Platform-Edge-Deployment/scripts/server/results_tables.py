#!/usr/bin/env python3
"""Render the API_SERVER_RESULTS.md tables from a comparison output dir (summary.json + cells).

    python scripts/server/results_tables.py results/api_bench [results/api_bench_gpu4] > tables.md
"""
import json, re, sys
from pathlib import Path

ORDER_M = ["v01n-fp32", "v01n-fp16", "v01n-pruned-fp32", "esmoen-fp32", "esmoen-fp16"]
ORDER_B = ["trt", "ort-cuda", "mnn-cuda", "ncnn-cpu", "mnn-cpu"]
NAME_M = {"v01n-fp32": "v0.1-N fp32", "v01n-fp16": "v0.1-N fp16", "v01n-pruned-fp32": "v0.1-N pruned fp32",
          "esmoen-fp32": "EsMoE-N fp32", "esmoen-fp16": "EsMoE-N fp16"}
NAME_B = {"trt": "TensorRT", "ort-cuda": "ORT CUDA", "mnn-cuda": "MNN CUDA", "ncnn-cpu": "ncnn CPU", "mnn-cpu": "MNN CPU"}


def f(v, n=1):
    return ("%.*f" % (n, v)) if isinstance(v, (int, float)) else "-"


def load(d):
    rows = json.loads((Path(d) / "summary.json").read_text())
    return {(r["model"], r["backend"]): r for r in rows}


def gpu_util(d, cell, c):
    p = Path(d) / cell / f"gpu_during_c{c}.txt"
    if not p.exists():
        return "-"
    m = re.match(r"\s*(\d+) %, (\d+) MiB, ([\d.]+) W", p.read_text())
    return f"{m.group(1)}% / {m.group(2)} MiB" if m else "-"


def main():
    main_dir = sys.argv[1]; gpu4_dir = sys.argv[2] if len(sys.argv) > 2 else None
    R = load(main_dir); G = load(gpu4_dir) if gpu4_dir and (Path(gpu4_dir) / "summary.json").exists() else {}
    keys = [(m, b) for b in ORDER_B for m in ORDER_M if (m, b) in R]

    out = []
    out.append("### Table 1: bare runtime vs API, latency (ms) and accuracy\n")
    out.append("| model | backend | execution provider | CLI model | API model | CLI end-to-end | API client p50 | API client p95 | API client p99 | HTTP overhead | mAP50-95 CLI | mAP50-95 API | parity |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for m, b in keys:
        r = R[(m, b)]; cli, c1, sc = r["cli"], r["api_c1"], r["score"]
        par = sc.get("parity", "-"); par = "OK" if "parity=OK" in par else ("FAIL" if par != "-" else "-")
        cm = c1.get("client_ms", {}); sm = c1.get("server_ms", {})
        invalid = sc.get("cli", {}).get("map5095", 1) == 0
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            NAME_M[m], NAME_B[b], cli.get("ep", "-"), f(cli.get("infer"), 2), f(sm.get("infer", {}).get("p50"), 2),
            f(cli.get("total"), 2), f(cm.get("p50"), 2), f(cm.get("p95"), 2), f(cm.get("p99"), 2), f(c1.get("api_overhead_ms_p50"), 2),
            f(sc.get("cli", {}).get("map5095"), 4), f(sc.get("api", {}).get("map5095"), 4),
            "invalid output (0 dets)" if invalid else par))
    out.append("\n### Table 2: throughput (images/s over the 5000-image run)\n")
    hdr = "| model | backend | CLI sequential | API c=1 | API c=8, default workers | GPU util / mem at c=8 |"
    if G:
        hdr += " API c=8, 4 workers |"
    out.append(hdr); out.append("|---|---|---|---|---|---|" + ("---|" if G else ""))
    for m, b in keys:
        r = R[(m, b)]; cli, c1, c8 = r["cli"], r["api_c1"], r["api_c8"]
        w = "1" if b in ("trt", "ort-cuda", "mnn-cuda") else "2"
        line = "| %s | %s | %s | %s | %s (%s) | %s |" % (NAME_M[m], NAME_B[b], f(cli.get("img_s")), f(c1.get("throughput_img_s")),
                                                     f(c8.get("throughput_img_s")), w, gpu_util(main_dir, f"{m}__{b}", 8))
        if G:
            g = G.get((m, b), {}).get("api_c8", {})
            line += " %s |" % f(g.get("throughput_img_s"))
        out.append(line)
    out.append("\n### Table 3: API server stage breakdown at c=1 (server p50, ms)\n")
    out.append("| model | backend | queue | decode | pre | infer | post | total | client p50 |")
    out.append("|---|---|---|---|---|---|---|---|---|")
    for m, b in keys:
        r = R[(m, b)]; sm = r["api_c1"].get("server_ms", {}); cm = r["api_c1"].get("client_ms", {})
        out.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (NAME_M[m], NAME_B[b], *[f(sm.get(k, {}).get("p50"), 2) for k in ("queue", "decode", "pre", "infer", "post", "total")], f(cm.get("p50"), 2)))
    print("\n".join(out))


if __name__ == "__main__":
    main()
