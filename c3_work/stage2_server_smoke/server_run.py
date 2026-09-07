#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 stage2 · run_smoke.py 的服务器化运行器(单训练单元)

相对 smoke/c3/run_smoke.py 的改动:
  - 路径全部参数化/推断,不硬编码(REPO_ROOT 由脚本位置推断)
  - 运行目录默认 <script_dir>/runs/<name>,可通过 --runs-root 重定向
  - 增加 --tag 记录批次;保留"拒绝覆盖同名目录"纪律
  - 命令前/后采集 nvidia-smi;输出 command.sh / summary.json(resolved_config 子集)
  - --epochs 段式训练由外部(queue_runner/调度)用新 --name 的 resume 实现

用法(示例,从仓库根调用):
  python c3_work/stage2_server_smoke/server_run.py \
      --strategy vpeft \
      --data <abs neu_det.yaml> \
      --name neuk10_vpeft_824 \
      --epochs 50 --batch 16 --imgsz 640 --device 0 --seed 824

三策略定义与 smoke/c3/run_smoke.py 保持一致(勿重复定义):
  1) vpeft            : planner on + backend vpeft + budget 2.1M + exclude 已知缺陷层
  2) full_sft         : lora_r=0 (等价全参微调)
  3) frozen_backbone  : lora_r=0 + freeze 11 (冻结主干只训头)
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
# 关键:必须用当前解释器所在 env 的 yolo,而不是 PATH 上裸 yolo
# (本机 PATH 的 yolo 命中 base miniconda/他人项目,会加载错误的 ultralytics)
YOLO_EXE = str(Path(sys.executable).resolve().parent / "yolo")

STRATEGIES = {
    "vpeft": {
        "desc": "V-PEFT planner 自动放置(排除已知缺陷层 cap<8)",
        "lora_planner_enabled": True,
        "lora_planner_backend": "vpeft",
        "lora_adapter_budget": 2_100_000,
        "lora_vpeft_strict": True,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_exclude_modules": "routing_network.2, dfl.conv, 0.conv",
    },
    "full_sft": {
        "desc": "全参微调(等价 lora_r=0)",
        "lora_r": 0,
        "freeze": 0,
    },
    "frozen_backbone": {
        "desc": "冻结主干(前 11 层),只训头",
        "lora_r": 0,
        "freeze": 11,
    },
}


def gpu_snapshot(device: str) -> dict:
    """nvidia-smi 采样当前卡的显存/利用率。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().split("\n")
        rows = [x.split(",") for x in out if x]
        idx = int(device) if device.isdigit() else 0
        row = rows[idx] if idx < len(rows) else rows[0]
        return {"gpu_mem_used_mb": int(row[0].strip()), "gpu_mem_total_mb": int(row[1].strip()),
                "gpu_util": int(row[2].strip())}
    except Exception as e:  # noqa: BLE001
        return {"gpu_snapshot_error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="C3 服务器化三策略运行器(单单元)")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ap.add_argument("--data", required=True, help="数据集 yaml 绝对路径")
    ap.add_argument("--name", required=True, help="运行名(唯一,不允许覆盖)")
    ap.add_argument("--tag", default="", help="批次标签(如 neuk10_vpeft)")
    ap.add_argument("--model", default=str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=824)
    ap.add_argument("--amp", default=False)
    ap.add_argument("--runs-root", default=str(SCRIPT_DIR / "runs"))
    args = ap.parse_args()

    runs_root = Path(args.runs_root)
    run_dir = runs_root / args.name
    if run_dir.exists():
        raise SystemExit(f"[拒绝覆盖] 运行目录已存在: {run_dir}\n请使用新的 --name")
    run_dir.mkdir(parents=True)

    cfg = STRATEGIES[args.strategy]
    cmd = [
        YOLO_EXE, "detect", "train",
        "model=" + args.model,
        "data=" + args.data,
        f"epochs={args.epochs}", f"batch={args.batch}", f"imgsz={args.imgsz}",
        f"device={args.device}", f"seed={args.seed}", f"amp={str(args.amp).lower()}",
        f"project={run_dir / 'train'}", f"name={args.strategy}",
        "lora_save_adapters=True",
    ]
    for k, v in cfg.items():
        if k != "desc":
            cmd.append(f"{k}={v}")
    (run_dir / "command.sh").write_text(" ".join(cmd) + "\n")

    print(f"[C3-server] tag={args.tag or '-'} strategy={args.strategy} | {cfg['desc']}")
    print(f"  命令: {' '.join(cmd)}")
    # 训练完整输出落盘(证据/可复现): runs/<name>/training.log
    log_path = run_dir / "training.log"
    logf = open(log_path, "w", encoding="utf-8", errors="replace")

    gpu_before = gpu_snapshot(args.device)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=logf, stderr=subprocess.STDOUT, text=True)
    elapsed = time.time() - t0
    gpu_after = gpu_snapshot(args.device)
    logf.close()

    results_dir = run_dir / "train" / args.strategy
    summary = {
        "strategy": args.strategy,
        "desc": cfg["desc"],
        "tag": args.tag,
        "name": args.name,
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "timestamp": datetime.now().isoformat(),
    }
    rc = results_dir / "args.yaml"
    if rc.exists():
        resolved = yaml.safe_load(rc.read_text()) or {}
        keys = ["lora_planner_enabled", "lora_planner_backend", "lora_adapter_budget", "lora_vpeft_strict",
                "lora_r", "lora_alpha", "lora_exclude_modules", "lora_skip_stem", "epochs", "batch",
                "imgsz", "device", "amp", "freeze", "seed"]
        summary["resolved_config"] = {k: resolved[k] for k in keys if k in resolved}
    # 训练指标末行(results.csv 的 best 行)
    csvp = results_dir / "results.csv"
    if csvp.exists():
        lines = csvp.read_text().strip().splitlines()
        if lines:
            cols = lines[0].split(",")
            vals = lines[-1].split(",")
            summary["last_epoch_metrics"] = {c.strip(): v.strip() for c, v in zip(cols, vals)}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[C3-server] 完成 exit={proc.returncode} 耗时={elapsed:.1f}s -> {run_dir / 'summary.json'}")
    if proc.returncode != 0:
        print("[C3-server] WARNING: 非零退出码,请检查日志")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
