#!/bin/bash
# ============================================================
# C3 stage1_env_data · 数据集下载脚本 (走代理)
#   1) NEU-DET  : GitHub 镜像 Marfbin/NEU-DET-with-yolov8 (YOLOv8 格式, 无显式 LICENSE)
#   2) DeepPCB  : tangsanli5201/DeepPCB (官方镜像仓库)
# 用法: bash download_datasets.sh > download.log 2>&1  (nohup 后台)
# ============================================================
set -e

DATASETS=/mnt/pfs/zitao_team/baiyouheng/datasets
RAW=$DATASETS/raw
export https_proxy=http://10.198.7.60:7890 http_proxy=http://10.198.7.60:7890

mkdir -p "$RAW"

echo "[1/2] NEU-DET (Marfbin/NEU-DET-with-yolov8) ..."
if [ -d "$RAW/NEU-DET-with-yolov8" ]; then
  echo "已存在,跳过"
else
  git clone --depth 1 https://github.com/Marfbin/NEU-DET-with-yolov8 "$RAW/NEU-DET-with-yolov8"
fi

echo "[2/2] DeepPCB (tangsanli5201/DeepPCB) ..."
if [ -d "$RAW/DeepPCB" ]; then
  echo "已存在,跳过"
else
  git clone --depth 1 https://github.com/tangsanli5201/DeepPCB "$RAW/DeepPCB"
fi

echo "DOWNLOAD_DONE"
echo "--- NEU-DET 结构 ---"; find "$RAW/NEU-DET-with-yolov8" -maxdepth 2 -type d | head -20
echo "--- DeepPCB 结构 ---"; find "$RAW/DeepPCB" -maxdepth 2 -type d | head -20
