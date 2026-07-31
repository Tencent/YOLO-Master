# -*- coding: utf-8 -*-
"""
issue #49 双5轮方案自动切换监视器
逻辑：
  阶段A: 监控 v01 训练 results.csv，满 5 个 epoch 后 kill v01 训练进程，启动 esmoe (5 epochs)
  阶段B: 监控 esmoe，若进程死且未满 5 epoch 则自动重启(resume)；满 5 epoch 后记录 DONE 退出
"""
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

REPO = r"D:\YOLO-Master"
PY = r"C:\Users\Administrator\AppData\Local\Microsoft\WindowsApps\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\python3.11.exe"
V01_CSV = os.path.join(REPO, r"runs\issue49\VisDrone_local_v01\results.csv")
ESMOE_CSV = os.path.join(REPO, r"runs\issue49\VisDrone_local_esmoe\results.csv")
ESMOE_LOG = os.path.join(REPO, r"runs\issue49\VisDrone_local_esmoe_train.log")
WATCH_LOG = os.path.join(REPO, r"runs\issue49\watch_switch.log")
TARGET_EPOCHS = 5
POLL_SEC = 120

DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP


def log(msg):
    line = f"[{datetime.now():%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    try:
        with open(WATCH_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass


def epoch_rows(csv_path):
    if not os.path.exists(csv_path):
        return 0
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
        return max(0, len(rows) - 1)
    except OSError:
        return 0


def find_train_pids(keyword):
    """返回命令行含 reproduce.py 且含 keyword 的 python3.11 进程 PID 列表（PowerShell CIM 查询）"""
    pids = []
    ps_cmd = ("Get-CimInstance Win32_Process -Filter \"Name='python3.11.exe'\" | "
              "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }")
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", ps_cmd],
            stderr=subprocess.DEVNULL, text=True, errors="ignore", timeout=60)
    except Exception as e:
        log(f"find_train_pids error: {e}")
        return pids
    for line in out.splitlines():
        line = line.strip()
        if not line or "\t" not in line:
            continue
        pid, cmd = line.split("\t", 1)
        if "reproduce.py" in cmd and keyword in cmd and "watch_switch" not in cmd:
            pids.append(pid.strip())
    return pids


def kill_pids(pids):
    for pid in pids:
        subprocess.call(["taskkill", "/F", "/PID", str(pid)],
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        log(f"killed PID {pid}")


def launch_esmoe():
    env = dict(os.environ)
    env["PYTHONPATH"] = REPO
    env["WANDB_MODE"] = "disabled"
    cmd = [PY, os.path.join(REPO, r"scripts\reproduce\reproduce.py"),
           "--data", r"scripts/reproduce/VisDrone_local.yaml",
           "--models", "esmoe",
           "--epochs", str(TARGET_EPOCHS),
           "--batch", "8", "--workers", "0", "--save-period", "1"]
    lf = open(ESMOE_LOG, "a", encoding="utf-8", errors="ignore")
    p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT,
                         creationflags=DETACHED)
    log(f"esmoe launched, PID={p.pid}")


def main():
    log(f"=== watcher start: v01满{TARGET_EPOCHS}ep后切换esmoe({TARGET_EPOCHS}ep) ===")
    phase = "A"
    # 若 esmoe 已有进度，直接进阶段B
    if epoch_rows(ESMOE_CSV) > 0 or find_train_pids("esmoe"):
        phase = "B"
        log("esmoe 已有进度/进程，直接进入阶段B")

    while True:
        if phase == "A":
            n = epoch_rows(V01_CSV)
            v01_pids = find_train_pids("v01")
            log(f"[A] v01 epochs={n}/{TARGET_EPOCHS} pids={v01_pids}")
            if n >= TARGET_EPOCHS:
                log(f"[A] v01 已满 {TARGET_EPOCHS} epochs，停止 v01，切换 esmoe")
                kill_pids(v01_pids)
                time.sleep(10)
                launch_esmoe()
                phase = "B"
            elif not v01_pids:
                log("[A] v01 进程不在但未满5ep —— 交给 hourly automation 重启 v01，本监视器继续等待")
        else:
            n = epoch_rows(ESMOE_CSV)
            es_pids = find_train_pids("esmoe")
            log(f"[B] esmoe epochs={n}/{TARGET_EPOCHS} pids={es_pids}")
            if n >= TARGET_EPOCHS:
                log("[B] esmoe 已满 5 epochs，全部完成！DONE")
                break
            if not es_pids:
                log("[B] esmoe 进程不在且未完成 → 自动重启(resume)")
                launch_esmoe()
        time.sleep(POLL_SEC)

    log("=== watcher exit (ALL DONE) ===")


if __name__ == "__main__":
    sys.exit(main())
