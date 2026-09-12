#!/bin/bash
# =============================================================================
# A1 实验产物上传脚本：训练完成后将本地 run 目录 + 终端日志归档到共享盘
#
# 用法：
#   ./upload_run_to_share.sh <本地run目录> [训练日志文件] [--with-wandb]
#
# 行为：
#   1. 校验 run 目录与 results.csv（训练完成标志）
#   2. 从 args.yaml 读取 run 名
#   3. 按 机器名 + 时间戳 创建归档目录：
#      /root/gpufree-share/experiments/runs/<日期>/<run名>_<机器名>_<时间戳>/
#   4. rsync run 目录全部产物（默认排除 wandb 本地缓存，线上已有；--with-wandb 保留）
#   5. 若提供训练日志文件（tee 落盘的 train.log），一并复制进归档目录
#   6. 打印归档路径，用于回填任务书"执行记录"
#
# 示例：
#   ./upload_run_to_share.sh /root/workspace/docs/experiments/2026-08-31/runs/a1-s1-c-coco-val2017 \
#       /root/workspace/docs/experiments/2026-08-31/logs/a1-s1-c-coco-val2017-train.log
# =============================================================================

set -euo pipefail

SHARE_ROOT="/root/gpufree-share/experiments/runs"
INCLUDE_WANDB=false
TRAIN_LOG=""

if [ $# -lt 1 ]; then
    echo "用法: $0 <本地run目录> [训练日志文件] [--with-wandb]" >&2
    exit 1
fi
RUN_DIR="$(realpath "$1")"
shift
for arg in "$@"; do
    if [ "$arg" = "--with-wandb" ]; then
        INCLUDE_WANDB=true
    else
        TRAIN_LOG="$arg"
    fi
done

# 1. 校验
if [ ! -d "$RUN_DIR" ]; then
    echo "错误: run 目录不存在: $RUN_DIR" >&2
    exit 1
fi
if [ ! -f "$RUN_DIR/results.csv" ]; then
    echo "错误: $RUN_DIR 下没有 results.csv，训练可能未完成" >&2
    exit 1
fi
if [ -n "$TRAIN_LOG" ] && [ ! -f "$TRAIN_LOG" ]; then
    echo "错误: 训练日志文件不存在: $TRAIN_LOG" >&2
    exit 1
fi

# 2. run 名（args.yaml 中 name 字段）
RUN_NAME="$(grep -E '^name:' "$RUN_DIR/args.yaml" | sed 's/^name: *//')"
if [ -z "$RUN_NAME" ]; then
    echo "错误: args.yaml 中未找到 name" >&2
    exit 1
fi

# 3. 机器名 + 时间戳 + 日期
MACHINE="$(hostname)"
STAMP="$(date +%Y%m%d_%H%M%S)"
TODAY="$(date +%Y-%m-%d)"
DEST="$SHARE_ROOT/$TODAY/${RUN_NAME}_${MACHINE}_${STAMP}"

# 4. 防覆盖
if [ -d "$DEST" ]; then
    echo "错误: 归档目标已存在: $DEST" >&2
    exit 1
fi

# 5. 上传 run 目录
mkdir -p "$DEST"
RSYNC_ARGS=(-a --info=progress2)
if [ "$INCLUDE_WANDB" = false ]; then
    RSYNC_ARGS+=(--exclude wandb)
    echo "说明: 默认排除 wandb/ 本地缓存（线上已同步），保留请加 --with-wandb"
fi
rsync "${RSYNC_ARGS[@]}" "$RUN_DIR/" "$DEST/"

# 6. 训练日志（tee 落盘文件，不在 run 目录内，需单独归档）
if [ -n "$TRAIN_LOG" ]; then
    cp "$TRAIN_LOG" "$DEST/$(basename "$TRAIN_LOG")"
    echo "训练日志已归档: $(basename "$TRAIN_LOG")"
else
    echo "提示: 未提供训练日志文件（tee 落盘的 train.log），如需归档请作为第二参数传入"
fi

# 7. 生成 README.md：wandb 出处 + 训练命令 + 模型参数（防产物混淆）
wandb_project="$(grep -E '^project:' "$RUN_DIR/args.yaml" | sed 's/^project: *//; s|/|-|g')"
cat > "$DEST/README.md" <<EOF
# $RUN_NAME 归档说明

- 归档时间：$TODAY $STAMP
- 执行机器：$MACHINE
- 归档来源：$RUN_DIR

## wandb 线上

- 项目：https://wandb.ai/${WANDB_ENTITY:-delei-kong-szu}/$wandb_project
- run 名：$RUN_NAME（在项目页按名称筛选；wandb/ 本地缓存未归档，线上已同步）

## 训练配置（摘录自 args.yaml）

| 参数 | 值 |
|---|---|
| model | $(grep -E '^model:' "$RUN_DIR/args.yaml" | sed 's/^model: *//') |
| data | $(grep -E '^data:' "$RUN_DIR/args.yaml" | sed 's/^data: *//') |
| epochs | $(grep -E '^epochs:' "$RUN_DIR/args.yaml" | sed 's/^epochs: *//') |
| imgsz | $(grep -E '^imgsz:' "$RUN_DIR/args.yaml" | sed 's/^imgsz: *//') |
| batch | $(grep -E '^batch:' "$RUN_DIR/args.yaml" | sed 's/^batch: *//') |
| seed | $(grep -E '^seed:' "$RUN_DIR/args.yaml" | sed 's/^seed: *//') |
| device | $(grep -E '^device:' "$RUN_DIR/args.yaml" | sed 's/^device: *//') |
| save_period | $(grep -E '^save_period:' "$RUN_DIR/args.yaml" | sed 's/^save_period: *//') |

完整参数见同目录 \`args.yaml\`；复现命令见任务书（gpufree-share/experiments/tasks/）。
EOF
echo "README.md 已生成（wandb 出处 + 训练配置）"

# 8. 结果
echo "=================================================="
echo "归档完成 ✅"
echo "  归档路径: $DEST"
echo "  内容: $(ls "$DEST" | tr '\n' ' ')"
echo "  回填任务书执行记录: 机器=$MACHINE 时间=$STAMP"
echo "=================================================="
