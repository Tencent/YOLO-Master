#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage3 collect_evidence v2:
  - mAP 取 results.csv 的 **best**(max over epochs 的 val mAP), 而非末行
    (末行在后期过拟合/崩溃的单元严重低估; seed2024 单元即有 last=0.023 vs best=0.210)
  - GPU 峰值显存从 training.log 进度行解析(GpuMem 列 '<x>G'), 替代进程退出后的 gpu_after 采样
用法: python collect_evidence.py [--runs-root <dir>]
"""
import argparse
import csv
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def best_metrics(results_csv: Path):
    """返回 (best_ep, best_mAP50, best_mAP50_95); results.csv 每行为一个 epoch。"""
    if not results_csv.exists():
        return None
    try:
        rows = list(csv.DictReader(results_csv.open()))
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    best = None
    for r in rows:
        try:
            m50 = float(r["metrics/mAP50(B)"])
            m95 = float(r["metrics/mAP50-95(B)"])
        except (KeyError, ValueError):
            continue
        if best is None or m50 > best[1]:
            best = (int(float(r["epoch"])), m50, m95)
    return best


def peak_gpu_gb(training_log: Path):
    """training.log 进度行 GpuMem 列峰值: 格式 '  1/100      3.14G  ...'(loss 不带 G)。"""
    if not training_log.exists():
        return None
    peak = 0.0
    pat = re.compile(r"(\d+\.\d+)G")
    try:
        for line in training_log.open(encoding="utf-8", errors="replace"):
            m = pat.search(line)
            if m:
                peak = max(peak, float(m.group(1)))
    except Exception:  # noqa: BLE001
        return None
    return round(peak, 2) if peak else None


def collect_one(run_dir: Path, unit) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"id": unit["name"], "status": "no_summary"}
    s = json.loads(summary_path.read_text())
    strat = s.get("strategy")
    res = s.get("resolved_config", {})
    row = {
        "id": s.get("name"),
        "dataset": unit.get("dataset"),
        "seed": unit.get("seed"),
        "strategy": strat,
        "status": "done" if s.get("exit_code") == 0 else "failed",
        "exit_code": s.get("exit_code"),
        "elapsed_sec": s.get("elapsed_sec"),
    }
    bm = best_metrics(run_dir / "train" / strat / "results.csv")
    if bm:
        row["best_epoch"], row["mAP50_best"], row["mAP50_95_best"] = bm
    pg = peak_gpu_gb(run_dir / "training.log")
    row["peak_gpu_gb"] = pg
    row.update({
        "planner_enabled": res.get("lora_planner_enabled"),
        "planner_backend": res.get("lora_planner_backend"),
        "strict": res.get("lora_vpeft_strict"),
        "adapter_budget": res.get("lora_adapter_budget"),
        "lora_r": res.get("lora_r"),
        "freeze": res.get("freeze"),
        "exclude_modules": res.get("lora_exclude_modules"),
    })
    log = run_dir / "training.log"
    if log.exists():
        vp = [ln.strip() for ln in log.open(encoding="utf-8", errors="replace") if "[V-PEFT]" in ln]
        row["vpeft_decision_lines"] = vp[-6:]
        row["vpeft_n_decision_lines"] = len(vp)
    else:
        row["vpeft_decision_lines"] = None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default=str(SCRIPT_DIR / "runs"))
    ap.add_argument("--matrix", default=str(SCRIPT_DIR / "matrix.json"))
    args = ap.parse_args()
    runs_root = Path(args.runs_root)
    matrix = json.loads(Path(args.matrix).read_text())
    rows = []
    for u in matrix["units"]:
        run_dir = runs_root / u["name"]
        if run_dir.exists():
            rows.append(collect_one(run_dir, u))
    out = SCRIPT_DIR / "evidence_summary.json"
    out.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    keys = ["id", "dataset", "seed", "strategy", "status", "exit_code", "elapsed_sec",
            "best_epoch", "mAP50_best", "mAP50_95_best", "peak_gpu_gb",
            "planner_enabled", "planner_backend", "strict", "adapter_budget", "lora_r", "freeze"]
    with (SCRIPT_DIR / "evidence_summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_done = sum(1 for r in rows if r.get("status") == "done")
    print(f"[collect_evidence] {len(rows)} runs, done={n_done} -> evidence_summary.json/csv")


if __name__ == "__main__":
    main()
