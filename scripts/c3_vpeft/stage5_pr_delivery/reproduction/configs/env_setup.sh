#!/bin/bash
# ============================================================
# C3 stage1_env_data · 服务器环境搭建脚本 (A800, 共享排队)
# 目标：在用户工作区建立独立 conda env (python3.11) + editable 安装 YOLO-Master
# 用法：bash env_setup.sh > env_setup.log 2>&1   （建议 nohup 后台执行）
# ============================================================
set -e

CONDA=/mnt/pfs/zitao_team/miniconda3/bin/conda
PREFIX=/mnt/pfs/zitao_team/baiyouheng/conda_envs/yolo_master
REPO=/mnt/pfs/zitao_team/baiyouheng/research/TX_yolo/YOLO-Master
PROXY="http://10.198.7.60:7890"
export https_proxy=$PROXY http_proxy=$PROXY

echo "[1/6] conda create python=3.11 env (offline-first)..."
$CONDA create -p "$PREFIX" python=3.11 -y --offline 2>/dev/null \
  || $CONDA create -p "$PREFIX" python=3.11 -y

echo "[2/6] pip install torch/torchvision (pypi 清华内网镜像)..."
"$PREFIX/bin/pip" install torch==2.6.0 torchvision==0.21.0

echo "[3/6] pip install -r requirements.txt (仓库依赖)..."
"$PREFIX/bin/pip" install -r "$REPO/requirements.txt"

echo "[4/6] pip install -e . (YOLO-Master editable)..."
"$PREFIX/bin/pip" install -e "$REPO"

echo "[5/6] 版本验证 yolo version..."
"$PREFIX/bin/yolo" version 2>&1 | tail -3

echo "[6/6] torch CUDA 验证..."
"$PREFIX/bin/python" -c "import torch; print('torch', torch.__version__, '| cuda_build', torch.version.cuda, '| cuda_avail', torch.cuda.is_available(), '| ndev', torch.cuda.device_count())"

echo "ENV_SETUP_DONE"
