#!/usr/bin/env bash
# stage2 冒烟命令模板(env 装完后手动或由 AI 后台执行)
# 纪律:runs 目录唯一,拒绝覆盖;冒烟=快(1-2 epoch),不作数
set -euo pipefail

ENV_PY=/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master/bin/python
REPO=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master
SR=$REPO/c3_work/stage2_server_smoke/server_run.py
NEU=/mnt/pfs/zitao_team/baiyouheng/datasets/NEU-DET-yolo/neu_det_yolo_v3/neu_det.yaml
MODEL=$REPO/YOLO-Master-EsMoE-N.pt

echo "== 0. env 验证 =="
$ENV_PY -c "import torch,ultralytics;print('torch',torch.__version__,'cuda',torch.cuda.is_available());print('yolo',ultralytics.__version__)"
nvidia-smi --query-gpu=index,memory.total,memory.used --format=csv,noheader | head -8

echo "== 1. vpeft 冒烟(k10, 2 epoch) =="
$ENV_PY $SR --strategy vpeft --data $NEU --name smoke_k10_vpeft --tag smoke-neu-k10-vpeft \
    --model $MODEL --epochs 2 --batch 8 --imgsz 640 --device 0 --seed 824

echo "== 2. 三策略可达性冒烟(各 1 epoch) =="
for s in full_sft frozen_backbone; do
    $ENV_PY $SR --strategy $s --data $NEU --name smoke_k10_${s} --tag smoke-neu-k10-${s} \
        --model $MODEL --epochs 1 --batch 8 --imgsz 640 --device 0 --seed 824
done

echo "== 3. 证据提取提示 =="
echo "grep '[V-PEFT]' c3_work/stage2_server_smoke/runs/*/command.sh 运行日志; "
echo "cat c3_work/stage2_server_smoke/runs/<name>/train/<strategy>/args.yaml"
