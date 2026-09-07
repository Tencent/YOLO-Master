#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage3 collect_evidence: 扫描 runs/<id>/summary.json + training.log,
产出 evidence_summary.json(细) 与 evidence_summary.csv(四维扁平行)。

用法: python collect_evidence.py [--runs-root <dir>]
"""
import argparse
import csv
import json
import re
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent


def peak_gpu_mem(training_log: Path) -> int | None:
    """training.log 里 Ultralytics 的进度行 'GPU_mem <MiB>' 峰值(如有)。"""
    if not training_log.exists():
        return None
    peak = 0
    try:
        for line in training_log.open(encoding="utf-8", errors="replace"):
            m = re.search(r"([\d.]+)G\s*([\d.]+)G|([\d]+)MiB", line)
            # 保守:只识别显存专用模式避免误判(不同版本格式不同,缺失即 None)
            m2 = re.search(r"GPU_mem[:=]?\s*([\d.]+)([GM]i?B)?", line)
            if m2:
                v = float(m2.group(1))
                peak = max(peak, v)
    except Exception:  # noqa: BLE001
        return None
    return peak


def collect_one(run_dir: Path, unit) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return {"id": unit["name"], "status": "no_summary"}
    s = json.loads(summary_path.read_text())
    strat = s.get("strategy")
    res = s.get("resolved_config", {})
    row = {
        "id": s.get("name"),
        "status": "done" if s.get("exit_code") == 0 else "failed",
        "exit_code": s.get("exit_code"),
        "strategy": strat,
        "dataset": unit.get("dataset"),
        "seed": unit.get("seed"),
        "elapsed_sec": s.get("elapsed_sec"),
        # 显存: gpu_before/after 为训练进程存在时采样(近似占用), 优于无
        "gpu_mem_after_mb": (s.get("gpu_after") or {}).get("gpu_mem_used_mb"),
        # 精度
        "mAP50": s.get("last_epoch_metrics", {}).get("metrics/mAP50(B)"),
        "mAP50_95": s.get("last_epoch_metrics", {}).get("metrics/mAP50-95(B)"),
        "epochs_done": s.get("last_epoch_metrics", {}).get("epoch"),
        # planner/resolved
        "planner_enabled": res.get("lora_planner_enabled"),
        "planner_backend": res.get("lora_planner_backend"),
        "strict": res.get("lora_vpeft_strict"),
        "adapter_budget": res.get("lora_adapter_budget"),
        "lora_r": res.get("lora_r"),
        "freeze": res.get("freeze"),
        "exclude_modules": res.get("lora_exclude_modules"),
    }
    # vpeft 决策行(日志 [V-PEFT] 尾部若干行,截断)
    log = run_dir / "training.log"
    if log.exists():
        vp = [ln.strip() for ln in log.open(encoding="utf-8", errors="replace") if "[V-PEFT]" in ln]
        row["vpeft_decision_lines"] = vp[-6:]
        n_acc = sum(1 for ln in vp if "Accept" in ln or "accept" in ln)
        n_ref = sum(1 for ln in vp if "Refuse" in ln)
        row["vpeft_accept_count"], row["vpeft_refuse_count"] = n_acc, n_ref
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
    # csv 扁平
    csv_path = SCRIPT_DIR / "evidence_summary.csv"
    keys = ["id", "status", "strategy", "dataset", "seed", "exit_code", "elapsed_sec",
            "gpu_mem_after_mb", "mAP50", "mAP50_95", "epochs_done", "planner_enabled",
            "planner_backend", "strict", "adapter_budget", "lora_r", "freeze"]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    n_done = sum(1 for r in rows if r.get("status") == "done")
    print(f"[collect_evidence] {len(rows)} runs, done={n_done} -> {out} / {csv_path}")


if __name__ == "__main__":
    main()
