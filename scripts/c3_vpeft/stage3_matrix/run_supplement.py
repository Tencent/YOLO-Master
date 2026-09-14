#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage3 补充单元(robustness 验证):待 GPU 空闲逐卡启动,断点式直到全部结束。

背景: 主 18 单元中 4 个 outlier(见 stage4/outlier) 需验证是确定性结果还是偶发:
  - neu_vpeft_s2025        : NEU 用新 seed 2025 重试 vpeft(原 s2024 best 0.369 停滞)
  - neu_frozen_s2025       : NEU 用 seed 2025 重试 frozen(原 s2024 best 0.210 卡死)
  - pcb_frozen_s824b       : DeepPCB frozen 用同 seed 824 重跑(原曲线 0.03~0.5 剧烈震荡,验证确定性)
"""
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
RUNS = SCRIPT_DIR / "runs"
STAGE2_RUN = SCRIPT_DIR.parent / "stage2_server_smoke" / "server_run.py"
ENV_PY = os.environ.get("ENV_PY", "/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/python")
CLEAN_ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
POLL = 40
MODEL = "/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master/YOLO-Master-EsMoE-N.pt"
DATA = {
    "neu": "/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/neu_det.yaml",
    "pcb": "/mnt/pfs/zitao_team/baiyouheng/datasets/DeepPCB-yolo/deeppcb_v1/deeppcb.yaml",
}
UNITS = [
    {"name": "neu_vpeft_s2025", "ds": "neu", "strat": "vpeft", "seed": 2025, "budget": 2_100_000},
    {"name": "neu_frozen_s2025", "ds": "neu", "strat": "frozen_backbone", "seed": 2025, "budget": None},
    {"name": "pcb_frozen_s824b", "ds": "pcb", "strat": "frozen_backbone", "seed": 824, "budget": None},
]
EPOCHS, BATCH, IMGSZ = 100, 8, 640


def idle_cards():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15).stdout.splitlines()
    return {int(p[0]) for p in (ln.split(",") for ln in out) if int(p[1]) < 1500 and float(p[2]) < 30.0}


def done(u):
    return (RUNS / u["name"] / "summary.json").exists()


def main():
    os.makedirs(LOG := SCRIPT_DIR / "logs", exist_ok=True)
    procs = {}
    print("[supplement] waiting for free GPU to launch", [u["name"] for u in UNITS], flush=True)
    while True:
        idle = idle_cards()
        for u in procs:
            if u in idle:  # 释放已占卡(避免自占判断错误,这里仅标记)
                pass
        finished = all(done(u) for u in UNITS) and not procs
        for name in list(procs):
            p = procs[name]
            if p.poll() is not None:
                ec = 1 if not done({"name": name}) else 0
                print(f"[supplement] {name} exit={p.returncode} ec={ec}", flush=True)
                procs.pop(name)
        # 启动
        for u in UNITS:
            if u["name"] in procs or done(u):
                continue
            if not idle:
                break
            card = min(idle)
            idle.discard(card)
            cmd = [ENV_PY, str(STAGE2_RUN), "--strategy", u["strat"], "--data", DATA[u["ds"]],
                   "--name", u["name"], "--tag", "supplement", "--model", MODEL,
                   "--epochs", str(EPOCHS), "--batch", str(BATCH), "--imgsz", str(IMGSZ),
                   "--device", str(card), "--seed", str(u["seed"]), "--runs-root", str(RUNS)]
            if u["budget"]:
                cmd += ["--lora-budget", str(u["budget"])]
            logf = open(LOG / f"{u['name']}.log", "a")
            procs[u["name"]] = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT,
                                                env=CLEAN_ENV)
            print(f"[supplement] LAUNCH {u['name']} card={card} pid={procs[u['name']].pid}", flush=True)
        if not procs and all(done(u) for u in UNITS):
            print("[supplement] ALL SUPPLEMENT DONE", flush=True)
            break
        time.sleep(POLL)


if __name__ == "__main__":
    main()
