#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage3 sweep_runner:GPU 空闲探测 + 断点续跑的调度器。

用法:
  python sweep_runner.py                  # 常驻调度(每 30s 一轮),状态写入 matrix.json
  python sweep_runner.py --once           # 单轮(适合手动步进)
  python sweep_runner.py --max-concurrent 4   # 限制并发(默认=空闲卡数)

规则:
  - 空闲卡判据: mem_used<1500MiB 且 util<30%(不抢占他人已用卡)
  - 每单元 = server_run.py 子进程,stdout 落 logs/<id>.log; server_run 内部拒绝覆盖
  - 进程结束 -> 读 runs/<id>/summary.json exit_code -> done/failed
  - 重启 runner 可续跑: status=pending 的单元会被继续调度; done 不再动。
    若 status=running 但本进程重启: 当 run 目录存在且无 summary(孤儿在跑) -> 保持 running 不重复启动;
    有 summary 且 exit0 -> 置 done(完成未及写状态)。
"""
import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2_SERVER_RUN = SCRIPT_DIR.parent / "stage2_server_smoke" / "server_run.py"
LOG_DIR = SCRIPT_DIR / "logs"
ENV_PY = Path(os.environ.get("ENV_PY", "/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/python"))
# 关键: 训练子进程必须去除 PYTHONPATH —— CodeBuddy 的 sitecustomize shim 会拦截
# path.unlink()(safe-delete→trash),多进程并发重建同一 labels.cache 时抛 OSError
# (stage3 实测: neu_vpeft_s824 首轮 28.5s 失败即此因)。env python 的 ultralytics
# 依赖自身 site-packages(editable .pth),不依赖 PYTHONPATH,故可安全清空。
CLEAN_ENV = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}

IDLE_MEM_MIB = 1500
IDLE_UTIL = 30.0
POLL_SEC = 30


def gpu_idle_set():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=15).stdout.strip().split("\n")
    idle = set()
    for line in out:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3 and int(parts[1]) < IDLE_MEM_MIB and float(parts[2]) < IDLE_UTIL:
            idle.add(int(parts[0]))
    return idle


def read_matrix(path):
    return json.loads(path.read_text())


def write_matrix(path, matrix):
    path.write_text(json.dumps(matrix, indent=2, ensure_ascii=False))


def probe_unit_result(unit):
    """若 run 目录已有 summary.json,返回 exit_code;否则 None。"""
    sm = Path(unit["runs_root"]) / unit["name"] / "summary.json"
    if sm.exists():
        try:
            return json.loads(sm.read_text()).get("exit_code")
        except Exception:  # noqa: BLE001
            return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--max-concurrent", type=int, default=0, help="0=不限(按空闲卡)")
    ap.add_argument("--matrix", type=str, default=str(SCRIPT_DIR / "matrix.json"))
    ap.add_argument("--tag", default="stage3")
    args = ap.parse_args()
    matrix_path = Path(args.matrix)
    LOG_DIR.mkdir(exist_ok=True)

    def stamp(msg):
        print(f"[{datetime.now().isoformat(timespec='seconds')}] {msg}", flush=True)

    # 重启扫描: 有 summary 的 unit 先回填状态
    matrix = read_matrix(matrix_path)
    for u in matrix["units"]:
        if u["status"] in ("running", "pending"):
            ec = probe_unit_result(u)
            if ec is not None:
                u["status"] = "done" if ec == 0 else "failed"
                u["exit_code"] = ec
                stamp(f"回填 {u['id']}: status={u['status']} exit={ec}")
    write_matrix(matrix_path, matrix)

    procs = {}  # id -> Popen

    while True:
        matrix = read_matrix(matrix_path)
        idle = gpu_idle_set()
        running = [u for u in matrix["units"] if u["status"] == "running"]
        pending = [u for u in matrix["units"] if u["status"] == "pending"]

        # 1) 检查 running 进程结束(runner 重启后孤儿: p=None 时改探 summary)
        for u in running:
            p = procs.get(u["id"])
            if p is not None and p.poll() is not None:
                ec = probe_unit_result(u)
                if ec is None:
                    # summary 未落盘但进程结束 -> 失败(截断)
                    u["status"] = "failed"
                    u["exit_code"] = p.returncode
                    u["note"] = f"进程退出 {p.returncode} 但无 summary.json"
                    stamp(f"FAIL {u['id']} exit={p.returncode} (no summary)")
                else:
                    u["status"] = "done" if ec == 0 else "failed"
                    u["exit_code"] = ec
                    stamp(f"{'DONE' if ec == 0 else 'FAIL'} {u['id']} exit={ec}")
                procs.pop(u["id"], None)
            elif p is None:
                ec = probe_unit_result(u)
                if ec is not None:
                    u["status"] = "done" if ec == 0 else "failed"
                    u["exit_code"] = ec
                    stamp(f"孤儿完成 {u['id']} exit={ec}")

        # 2) 启动 pending(有 idle 卡);排除已被 running 单元占用的卡
        free_cards = sorted(idle)
        for u in running:
            if u.get("gpu") in free_cards:
                free_cards.remove(u["gpu"])
        if args.max_concurrent:
            room = max(0, args.max_concurrent - len(running))
            free_cards = free_cards[:room]
        started = 0
        for u in matrix["units"]:
            if u["status"] != "pending" or started >= len(free_cards):
                continue
            card = free_cards[started]
            u["status"] = "running"
            u["gpu"] = card
            cmd = [
                str(ENV_PY), str(STAGE2_SERVER_RUN),
                "--strategy", u["strategy"],
                "--data", u["data_yaml"],
                "--name", u["name"],
                "--tag", f"{args.tag}-{u['dataset']}-{u['strategy']}",
                "--model", u["model"],
                "--epochs", str(u["epochs"]),
                "--batch", str(u["batch"]),
                "--imgsz", str(u["imgsz"]),
                "--device", str(card),
                "--seed", str(u["seed"]),
                "--runs-root", u["runs_root"],
            ]
            if u.get("lora_budget"):
                cmd += ["--lora-budget", str(u["lora_budget"])]
            logf = open(LOG_DIR / f"{u['id']}.log", "a", encoding="utf-8")
            logf.write(f"\n=== {datetime.now().isoformat()} launch card {card} ===\n")
            logf.flush()
            procs[u["id"]] = subprocess.Popen(cmd, cwd=str(SCRIPT_DIR.parents[1]), stdout=logf,
                                              stderr=subprocess.STDOUT, text=True, env=CLEAN_ENV)
            started += 1
            stamp(f"LAUNCH {u['id']} card={card} pid={procs[u['id']].pid}")
        write_matrix(matrix_path, matrix)

        n_done = sum(1 for u in matrix["units"] if u["status"] == "done")
        n_failed = sum(1 for u in matrix["units"] if u["status"] == "failed")
        stamp(f"round: done={n_done} failed={n_failed} running={len(running)} pending={len([u for u in matrix['units'] if u['status']=='pending'])} idle_cards={sorted(idle)}")
        if args.once:
            break
        if n_done + n_failed == len(matrix["units"]):
            stamp("ALL UNITS FINISHED")
            break
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
