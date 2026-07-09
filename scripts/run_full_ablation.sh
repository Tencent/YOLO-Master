#!/bin/bash
# ============================================================
# MoE-aware PEFT 全量 COCO2017 消融实验启动脚本
# ============================================================
# Usage: cd /Users/gatilin/PycharmProjects/YOLO-Master-v0708/scripts && bash run_full_ablation.sh
#
# 说明:
#   - 使用全量 COCO2017 (118,287 train / 5,000 val)
#   - YOLO-Master-EsMoE-N.pt 作为预训练权重
#   - 每个实验独立运行，顺序执行
#   - 所有结果保存到 scripts/runs_* 目录
# ============================================================

set -euo pipefail

REPO_ROOT="/Users/gatilin/PycharmProjects/YOLO-Master-v0708"
SCRIPT_DIR="$REPO_ROOT/scripts"
DEVICE="mps"
MODEL="YOLO-Master-EsMoE-N.pt"
DATA="coco2017.yaml"

cd "$SCRIPT_DIR"

# 环境变量
export WANDB_MODE=disabled
export WANDB_SILENT=true
export KMP_DUPLICATE_LIB_OK=TRUE
export YOLO_AUTOINSTALL=false
export YOLO_VERBOSE=false

echo "============================================================"
echo "MoE-aware PEFT 全量 COCO2017 消融实验"
echo "============================================================"
echo "Model: $MODEL"
echo "Dataset: $DATA (118,287 train + 5,000 val)"
echo "Device: $DEVICE"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"

# ============================================================
# E1: MoLoRA per-expert rank vs uniform rank
# ============================================================
run_e1() {
    echo ""
    echo "[E1] Running: per-expert rank (uniform vs frequency)"
    echo "------------------------------------------------------"
    python3 ablation_moe_peft_e1_molora_rank.py
    echo "[E1] Done. Results: $SCRIPT_DIR/e1_molora_rank_results.json"
}

# ============================================================
# E2: Router calibration ablation
# ============================================================
run_e2() {
    echo ""
    echo "[E2] Running: router calibration (baseline / r=4 / r=8)"
    echo "------------------------------------------------------"
    python3 ablation_moe_peft_e2_router_calibration.py
    echo "[E2] Done. Results: $SCRIPT_DIR/e2_router_calibration_results.json"
}

# ============================================================
# E3: Expert load visualization
# ============================================================
run_e3() {
    echo ""
    echo "[E3] Running: expert load visualization"
    echo "------------------------------------------------------"
    python3 ablation_moe_peft_e3_expert_load_viz.py
    echo "[E3] Done. Results: $SCRIPT_DIR/e3_viz_outputs/"
}

# ============================================================
# EVAL: Unified evaluation (3 configs)
# ============================================================
run_eval() {
    echo ""
    echo "[EVAL] Running: unified evaluation (baseline + calib + freq)"
    echo "------------------------------------------------------"
    python3 eval_moe_peft.py --seeds 3
    echo "[EVAL] Done. Results: $SCRIPT_DIR/eval_moe_peft_results.json"
}

# ============================================================
# 主流程：顺序执行所有实验
# ============================================================
run_e1
run_e2
run_e3
run_eval

echo ""
echo "============================================================"
echo "所有实验完成！"
echo "End time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "============================================================"
echo ""
echo "结果文件:"
echo "  - $SCRIPT_DIR/e1_molora_rank_results.json"
echo "  - $SCRIPT_DIR/e2_router_calibration_results.json"
echo "  - $SCRIPT_DIR/e3_viz_outputs/e3_summary.json"
echo "  - $SCRIPT_DIR/e3_viz_outputs/e3_uniform.png"
echo "  - $SCRIPT_DIR/e3_viz_outputs/e3_frequency.png"
echo "  - $SCRIPT_DIR/eval_moe_peft_results.json"
echo ""
echo "训练日志目录:"
echo "  - $SCRIPT_DIR/runs_e1/"
echo "  - $SCRIPT_DIR/runs_e2/"
echo "  - $SCRIPT_DIR/runs_eval/"
echo "============================================================"
