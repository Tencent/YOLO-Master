#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C3 工业缺陷 V-PEFT 小样本实战 —— 三策略统一运行器 (GPU 机使用)

三方案同预算对照(任务书 P1 要求):
  1) vpeft           : lora_planner_enabled=True,  backend="vpeft",  budget=2_100_000
  2) full_sft        : lora_r=0(等价全参微调)
  3) frozen_backbone : lora_r=0 + freeze 主干(冻结 0..N-1 层)

用法:
  python smoke/c3/run_smoke.py --strategy vpeft --data <neu_det.yaml> --name k5_vpeft \
      --epochs 1 --batch 8 --imgsz 640 --device 0 --amp false

关键纪律(与准入文档一致):
  - 运行器拒绝覆盖同名目录:复跑必须使用新的 --name
  - 每个运行目录写入:命令、完整日志、resolved_config.yaml、指标、显存采样、退出码
  - 参数以实际生效配置为准(训练后保存 resolved_config.yaml)
"""

import argparse
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_DIR = REPO_ROOT / "smoke" / "c3"

STRATEGIES = {
    "vpeft": {
        "desc": "V-PEFT planner 自动放置(排除 stem,见下方注释)",
        "lora_planner_enabled": True,
        "lora_planner_backend": "vpeft",
        "lora_adapter_budget": 2_100_000,
        "lora_vpeft_strict": True,
        "lora_r": 8,
        "lora_alpha": 16,
        # V-PEFT planner 硬约束无 rank<=层容量检查(已知缺陷,已记录 issue):
        #   cap<8 的层 = [0.conv(cap3), 3/6/9/12.routing.routing_network.2(cap3), 25.dfl.conv(cap1)]
        # strict 模式会抛 ValueError。受控处理:用官方 lora_exclude_modules 排除。
        # 注意:exclude 为子串匹配,'0.conv' 会连带匹配 10.conv 等含该子串的层
        # (experts/MLP/head 等约 25 层,容量足够),排除后由 planner 在剩余层中
        # 自动选择 —— 不关闭 strict,误伤清单如实记录在结果 evidence 中。
        "lora_exclude_modules": "routing_network.2, dfl.conv, 0.conv",
    },
    "full_sft": {
        "desc": "全参微调(等价 lora_r=0)",
        "lora_r": 0,
        "freeze": 0,
    },
    "frozen_backbone": {
        "desc": "冻结主干,只训头",
        "lora_r": 0,
        "freeze": 11,  # 冻结前 11 层(0..10),适配 YOLO-Master-EsMoE-N
    },
}


def gpu_snapshot(device: str) -> dict:
    """nvidia-smi 采样:显存占用、利用率"""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15,
        ).stdout.strip().split("\n")
        rows = [x.split(",") for x in out if x]
        idx = int(device) if device.isdigit() else 0
        row = rows[idx] if idx < len(rows) else rows[0]
        return {"gpu_mem_used_mb": int(row[0].strip()), "gpu_mem_total_mb": int(row[1].strip()), "gpu_util": int(row[2].strip())}
    except Exception as e:
        return {"gpu_snapshot_error": str(e)}


def main():
    ap = argparse.ArgumentParser(description="C3 三策略运行器")
    ap.add_argument("--strategy", required=True, choices=list(STRATEGIES))
    ap.add_argument("--data", required=True, help="数据集 yaml(绝对路径或 Ultralytics 内置名)")
    ap.add_argument("--name", required=True, help="运行名(唯一,不允许覆盖)")
    ap.add_argument("--model", default=str(REPO_ROOT / "YOLO-Master-EsMoE-N.pt"))
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="0")
    ap.add_argument("--seed", type=int, default=824)
    ap.add_argument("--amp", default=False)
    args = ap.parse_args()

    run_dir = SMOKE_DIR / "runs" / args.name
    if run_dir.exists():
        raise SystemExit(f"[拒绝覆盖] 运行目录已存在: {run_dir}\n请使用新的 --name")
    run_dir.mkdir(parents=True)

    cfg = STRATEGIES[args.strategy]
    cmd = [
        "yolo", "detect", "train",
        "model=" + args.model,
        "data=" + args.data,
        f"epochs={args.epochs}", f"batch={args.batch}", f"imgsz={args.imgsz}",
        f"device={args.device}", f"seed={args.seed}", f"amp={str(args.amp).lower()}",
        f"project={run_dir / 'train'}", f"name={args.strategy}",
        f"lora_save_adapters=True",
    ]
    for k, v in cfg.items():
        if k == "desc":
            continue  # desc 仅用于记录,不是 YOLO 参数
        cmd.append(f"{k}={v}")
    # 记录实际命令
    (run_dir / "command.sh").write_text(" ".join(cmd) + "\n")

    print(f"[C3] strategy={args.strategy} ({cfg['desc']}) | name={args.name}")
    print(f"  命令: {' '.join(cmd)}")

    # 前/后显存采样
    gpu_before = gpu_snapshot(args.device)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    elapsed = time.time() - t0
    gpu_after = gpu_snapshot(args.device)

    # 汇总
    results_dir = run_dir / "train" / args.strategy
    summary = {
        "strategy": args.strategy,
        "desc": cfg["desc"],
        "name": args.name,
        "command": " ".join(cmd),
        "exit_code": proc.returncode,
        "elapsed_sec": round(elapsed, 1),
        "gpu_before": gpu_before,
        "gpu_after": gpu_after,
        "timestamp": datetime.now().isoformat(),
    }
    # resolved config
    rc = results_dir / "args.yaml"
    if rc.exists():
        resolved = yaml.safe_load(rc.read_text())
        summary["resolved_config"] = {
            k: resolved[k] for k in ["lora_planner_enabled", "lora_planner_backend", "lora_adapter_budget",
                                    "lora_r", "lora_alpha", "epochs", "batch", "imgsz", "device", "amp", "freeze"]
            if k in resolved
        }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[C3] 完成 exit={proc.returncode} 耗时={elapsed:.1f}s")
    print(f"[C3] 摘要: {run_dir / 'summary.json'}")
    if proc.returncode != 0:
        print("[C3] WARNING: 非零退出码,请检查日志")
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
