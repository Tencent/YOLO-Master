#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage3 生成 18 单元 matrix.json = {NEU, DeepPCB} x {vpeft, full_sft, frozen_backbone} x seed{824,2024,777}.

用法: python make_matrix.py [--budget 2100000] [--epochs 100]
覆盖已有 matrix.json 前会打印 diff 提示;幂等(状态字段在 runner 中推进)。
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path(__file__).resolve().parent / "runs"

DATASETS = {
    "neu": "/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/neu_det.yaml",
    "pcb": "/mnt/pfs/zitao_team/baiyouheng/datasets/DeepPCB-yolo/deeppcb_v1/deeppcb.yaml",
}
STRATEGIES = ["vpeft", "full_sft", "frozen_backbone"]
SEEDS = [824, 2024, 777]
MODEL = str(ROOT / "YOLO-Master-EsMoE-N.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2_100_000, help="vpeft adapter budget(定案见 README/pilot)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=str, default=str(Path(__file__).parent / "matrix.json"))
    args = ap.parse_args()

    units = []
    for ds, yaml_path in DATASETS.items():
        for strat in STRATEGIES:
            for seed in SEEDS:
                name = f"{ds}_{strat}_s{seed}"
                units.append({
                    "id": name,
                    "dataset": ds,
                    "strategy": strat,
                    "seed": seed,
                    "name": name,
                    "model": MODEL,
                    "data_yaml": yaml_path,
                    "epochs": args.epochs,
                    "batch": args.batch,
                    "imgsz": args.imgsz,
                    "lora_budget": args.budget if strat == "vpeft" else None,
                    "runs_root": str(RUNS_ROOT),
                    "status": "pending",  # pending/running/done/failed
                    "exit_code": None,
                    "note": "",
                })
    matrix = {
        "meta": {
            "epochs": args.epochs, "batch": args.batch, "imgsz": args.imgsz,
            "vpeft_budget": args.budget, "model": MODEL,
            "decision": "EsMoE-N; epochs=100(≫warmup 3.0); budget 见 pilot 注记",
        },
        "units": units,
    }
    out = Path(args.out)
    if out.exists():
        old = json.loads(out.read_text())
        print(f"[make_matrix] 已存在 {out}({len(old['units'])} units); 将覆盖。旧 budget={old['meta'].get('vpeft_budget')} -> 新 {args.budget}")
    out.write_text(json.dumps(matrix, indent=2, ensure_ascii=False))
    print(f"[make_matrix] 写入 {out}: {len(units)} units")


if __name__ == "__main__":
    main()
