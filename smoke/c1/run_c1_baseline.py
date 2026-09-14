#!/usr/bin/env python3
"""C1 最小 on/off 对照：只跑 v0.1-N vs EsMoE-N 两个 baseline。

用法（在 D:/YOLO-Master 目录执行）：
    set PYTHONPATH=D:/YOLO-Master;D:/torch_cuda_pkg
    python D:/smoke/run_c1_baseline.py --epochs 50 --batch 8 --device 0 --workers 0 --seed 42 --no-sparse-eval --no-wandb --no-amp --project runs/c1_onoff
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path("D:/YOLO-Master")
sys.path.insert(0, str(ROOT))          # 让仓库内 ultralytics 包可被 import
sys.path.insert(0, "D:/torch_cuda_pkg") # torch cu118

# 切换 cwd 到仓库根，让 VisDrone.yaml / 相对配置路径正确解析
import os
os.chdir(str(ROOT))

from scripts.reproduce._reproduce_common import MODELS, DatasetSpec, run_dataset

DATASET = DatasetSpec(
    name="VisDrone",
    data="VisDrone.yaml",          # 已复制到仓库根，指向 D:/datasets/VisDrone
    project="runs/c1_onoff",
)

if __name__ == "__main__":
    # MODELS 里只有 v0.1-N 和 EsMoE-N
    raise SystemExit(run_dataset(DATASET, models=MODELS))
