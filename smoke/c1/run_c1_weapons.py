#!/usr/bin/env python3
"""C1 Stage2 武器横评：跑 P2 武器族 (VARIANTS)，与 v0.1-N 基线(0.1959)公平同预算对比。

用法（在 D:/YOLO-Master 目录执行）：
    set PYTHONPATH=D:/YOLO-Master;D:/torch_cuda_pkg
    python D:/smoke/run_c1_weapons.py --epochs 50 --batch 8 --device 0 --workers 0 --seed 42 --no-sparse-eval --no-wandb --no-amp --project runs/c1_weapons
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/YOLO-Master")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, "D:/torch_cuda_pkg")

import os
os.chdir(str(ROOT))

from scripts.reproduce._reproduce_common import VARIANTS, DatasetSpec, run_dataset

# ---- 异常安全补丁：best.pt 文件锁(火绒HipsDaemon实时扫描)会导致 PermissionError ----
# 把 best.pt 写入包进 try/except：锁了就跳过(只丢 best.pt 不丢训练)，last.pt/results.csv 照常。
import ultralytics.engine.trainer as _trainer_mod


def _safe_save_model(self):
    import time as _t
    serialized_ckpt = self._serialize_checkpoint()
    self.wdir.mkdir(parents=True, exist_ok=True)
    # last.pt 是 resume 命根子：锁了重试几次再放弃(仍失败仅丢当前 epoch)
    for attempt in range(5):
        try:
            self.last.write_bytes(serialized_ckpt)
            break
        except (PermissionError, OSError) as e:
            if attempt == 4:
                print(f"[WARN] last.pt 5次重试仍被锁，放弃本次保存(仅丢当前epoch): {e}")
            else:
                _t.sleep(0.5)
    if self.best_fitness == self.fitness:
        try:
            self.best.write_bytes(serialized_ckpt)
        except (PermissionError, OSError) as e:
            print(f"[WARN] best.pt 被杀毒软件锁住，跳过本次 best 保存(训练继续): {e}")
    if (self.save_period > 0) and (self.epoch % self.save_period == 0):
        try:
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(serialized_ckpt)
        except Exception:
            pass
    try:
        self._refresh_healthy_checkpoint()
    except Exception as e:
        print(f"[WARN] last_healthy.pt 写入失败(非致命，训练继续): {e}")
    return True


_trainer_mod.BaseTrainer.save_model = _safe_save_model

# ---- 异常安全补丁(二)：args.yaml 启动期写入也会被火绒锁住(同因) ----
# 原 _save_run_args 仅在启动写 args.yaml(纯记录用)，锁了就跳过，不影响训练数值。
_orig_save_run_args = _trainer_mod.BaseTrainer._save_run_args


def _safe_save_run_args(self):
    try:
        _orig_save_run_args(self)
    except (PermissionError, OSError) as e:
        print(f"[WARN] args.yaml 被杀毒软件锁住，跳过本次写入(训练继续): {e}")


_trainer_mod.BaseTrainer._save_run_args = _safe_save_run_args
print("[PATCH] BaseTrainer._save_run_args 已打异常安全补丁(args.yaml 锁自动跳过)")
print("[PATCH] BaseTrainer.save_model 已打异常安全补丁(best.pt 锁自动跳过)")
# ---------------------------------------------------------------------------------

DATASET = DatasetSpec(
    name="VisDrone",
    data="VisDrone.yaml",          # 指向 D:/datasets/VisDrone
    project="runs/c1_weapons",
)

if __name__ == "__main__":
    # VARIANTS = EsMoE-P2-N / v0.1-P2-N / UoMoE-N / UoMoE-P2-N
    raise SystemExit(run_dataset(DATASET, models=VARIANTS))
