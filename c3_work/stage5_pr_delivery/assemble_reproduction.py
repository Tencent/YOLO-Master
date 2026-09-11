#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""stage5: 组装 reproduction/ 复现包(从各 stage 收集小文件, 大产物 runs/*.pt 不入)。
用法: python assemble_reproduction.py
"""
import json
import shutil
from pathlib import Path

C3 = Path(__file__).resolve().parent.parent
S1, S3, S4 = C3 / "stage1_env_data", C3 / "stage3_matrix", C3 / "stage4_analysis"
S5 = Path(__file__).resolve().parent
REP = S5 / "reproduction"


def cp(src: Path, dst: Path):
    if src.exists():
        shutil.copy2(src, dst)
        print(f"  cp {src.name} -> {dst}")
    else:
        print(f"  !! missing {src}")


def main():
    for sub in ("configs", "results", "datasets"):
        (REP / sub).mkdir(parents=True, exist_ok=True)
    print("== copying ==")
    cp(S1 / "README.md", REP / "data_and_env_from_stage1.md")
    cp(S1 / "prepare_neu_det.py", REP / "datasets" / "prepare_neu_det.py")
    cp(S1 / "prepare_deeppcb.py", REP / "datasets" / "prepare_deeppcb.py")
    cp(S1 / "datasets" / "manifest.md", REP / "datasets" / "manifest.md")
    cp(S1 / "env_setup.sh", REP / "configs" / "env_setup.sh")
    cp(S3 / "matrix.json", REP / "results" / "matrix.json")
    cp(S3 / "collect_evidence.py", REP / "results" / "collect_evidence.py")
    cp(S3 / "evidence_summary.csv", REP / "results" / "evidence_summary.csv")
    cp(S3 / "evidence_summary.json", REP / "results" / "evidence_summary.json")
    cp(S4 / "comparison_tables.md", REP / "results" / "comparison_tables.md")
    cp(S4 / "planner_audit_summary.md", REP / "results" / "planner_audit_summary.md")
    cp(S4 / "analysis_stats.py", REP / "results" / "analysis_stats.py")
    tpl = REP / "configs" / "paths.env"
    tpl.write_text("# 复制为部署路径模板(替换真实绝对路径)\n"
                   "export C3_DATASETS=/path/to/datasets\n"
                   "export YOLO_MODEL=/path/to/YOLO-Master-EsMoE-N.pt\n"
                   "export ENV_PY=/path/to/envs/yolo_master/bin/python\n"
                   "export RUNS_ROOT=/path/to/runs\n")
    print("== sample commands ==")
    cmds = {}
    for d in sorted((S3 / "runs").iterdir()):
        sj = d / "summary.json"
        if not sj.exists():
            continue
        data = d.name.split("_")[0]
        strat = json.loads(sj.read_text())["strategy"]
        cmds.setdefault(data, {})
        if strat not in cmds[data]:
            cmds[data][strat] = json.loads(sj.read_text())["command"]
    sample = REP / "configs" / "train_commands_example.sh"
    with sample.open("w") as f:
        f.write("#!/bin/bash\n# 同预算三策略命令样例(从正式单元 resolved command 提取, seed/device 可改)\n\n")
        for ds, d in cmds.items():
            f.write(f"## {ds}\n")
            for st, c in d.items():
                f.write(f"# {st}\n{c}\n\n")
    print(f"  -> {sample}")

    rep_readme = REP / "README.md"
    rep_readme.write_text(
        "# reproduction 复现包\n\n"
        "由 `../assemble_reproduction.py` 生成(stage1-4 可复现小文件收集, 大产物 runs/权重不入库)。\n"
        "- `data_and_env_from_stage1.md`: 环境重建与数据集准备流程(stage1 README)\n"
        "- `datasets/`: 数据转换脚本 + `manifest.md`(来源/许可/统计/SHA 留档位置)\n"
        "- `configs/`: env 模板 + 三策略训练命令样例 + 环境脚本\n"
        "- `results/`: matrix + evidence + 四维对照表 + planner 审计\n"
        "- `limitations.md`: 已知局限/许可风险/seed 稳定性(手写, 不随脚本覆盖)\n"
    )
    print("== reproduction/ 组装完成 ==")


if __name__ == "__main__":
    main()
