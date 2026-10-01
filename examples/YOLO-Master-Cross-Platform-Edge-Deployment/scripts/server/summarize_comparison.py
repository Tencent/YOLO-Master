#!/usr/bin/env python3
"""Aggregate scripts/server/run_comparison.sh cells into a Markdown table + summary.json.

    python scripts/server/summarize_comparison.py /data/results/api_bench
"""
import json, re, sys
from pathlib import Path


def parse_cli_summary(p: Path):
    d = {}
    if not p.exists():
        return d
    t = p.read_text()
    m = re.search(r"frames=(\d+).*?pre=([\d.]+) infer=([\d.]+) post=([\d.]+) total=([\d.]+)ms.*?model-FPS=([\d.]+)\s+wall=([\d.]+)s", t, re.S)
    if m:
        d.update(frames=int(m.group(1)), pre=float(m.group(2)), infer=float(m.group(3)), post=float(m.group(4)),
                 total=float(m.group(5)), model_fps=float(m.group(6)), wall_s=float(m.group(7)))
        d["img_s"] = d["frames"] / d["wall_s"] if d["wall_s"] else 0
    e = re.search(r"ep=(\S+)", t)
    if e:
        d["ep"] = e.group(1)
    w = re.search(r"wall_s=([\d.]+) rc=(\d+)", t)
    if w:
        d["wall_s_outer"] = float(w.group(1)); d["rc"] = int(w.group(2))
    return d


def parse_score(p: Path):
    d = {}
    if not p.exists():
        return d
    for line in p.read_text().splitlines():
        m = re.match(r"(cli|api): images=(\d+)\s+mAP50=([\d.]+)\s+mAP50-95=([\d.]+)", line)
        if m:
            d[m.group(1)] = {"images": int(m.group(2)), "map50": float(m.group(3)), "map5095": float(m.group(4))}
        m = re.match(r"parity: (.*)", line)
        if m:
            d["parity"] = m.group(1)
    return d


def main():
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/data/results/api_bench")
    rows = []
    for cell in sorted(out.glob("*__*")):
        model, backend = cell.name.split("__", 1)
        cli = parse_cli_summary(cell / "cli.summary")
        c1 = json.loads((cell / "api_c1.json").read_text()) if (cell / "api_c1.json").exists() else {}
        c8 = json.loads((cell / "api_c8.json").read_text()) if (cell / "api_c8.json").exists() else {}
        sc = parse_score(cell / "score.txt")
        rows.append({"model": model, "backend": backend, "cli": cli, "api_c1": c1, "api_c8": c8, "score": sc})
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    lines = ["| model | backend | ep | CLI img/s | CLI infer ms | API c=1 img/s | API c=1 client p50 / p95 ms | API c=1 infer ms | API overhead p50 ms | API c=8 img/s | mAP50-95 CLI | mAP50-95 API | parity |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        cli, c1, c8, sc = r["cli"], r["api_c1"], r["api_c8"], r["score"]
        f = lambda v, n=1: ("%.*f" % (n, v)) if isinstance(v, (int, float)) else "-"
        par = sc.get("parity", "-"); par = "OK" if "parity=OK" in par else ("FAIL" if par != "-" else "-")
        lines.append("| %s | %s | %s | %s | %s | %s | %s / %s | %s | %s | %s | %s | %s | %s |" % (
            r["model"], r["backend"], cli.get("ep", "-"), f(cli.get("img_s")), f(cli.get("infer"), 2),
            f(c1.get("throughput_img_s")), f(c1.get("client_ms", {}).get("p50"), 2), f(c1.get("client_ms", {}).get("p95"), 2),
            f(c1.get("server_ms", {}).get("infer", {}).get("p50"), 2), f(c1.get("api_overhead_ms_p50"), 2),
            f(c8.get("throughput_img_s")), f(sc.get("cli", {}).get("map5095"), 4), f(sc.get("api", {}).get("map5095"), 4), par))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
