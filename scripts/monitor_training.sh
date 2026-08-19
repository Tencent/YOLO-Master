#!/bin/bash
# Monitor training progress

PROJECT_DIR="/root/autodl-tmp/runs/visdrone_mot_ablation"

echo "=== Training Progress Monitor ==="
echo "Project: $PROJECT_DIR"
echo ""

for model in v10 v10_mot v10_moa; do
    echo "--- Model: $model ---"

    results_csv="$PROJECT_DIR/$model/results.csv"

    if [ -f "$results_csv" ]; then
        # Get last epoch
        last_line=$(tail -1 "$results_csv")
        epoch=$(echo "$last_line" | cut -d',' -f1)

        # Extract key metrics from last line
        mAP50=$(echo "$last_line" | awk -F',' '{for(i=1;i<=NF;i++){if($i~/metrics\/mAP50\(B\)/){print $(i+1)}}}' | head -1)
        mAP50_95=$(echo "$last_line" | awk -F',' '{for(i=1;i<=NF;i++){if($i~/metrics\/mAP50-95\(B\)/){print $(i+1)}}}' | head -1)

        echo "  Status: Training in progress"
        echo "  Current Epoch: $epoch"
        echo "  Latest mAP50: $mAP50"
        echo "  Latest mAP50-95: $mAP50_95"

        # Check if weights exist
        if [ -f "$PROJECT_DIR/$model/weights/best.pt" ]; then
            echo "  Best weights: ✓"
        fi
    else
        if [ -d "$PROJECT_DIR/$model" ]; then
            echo "  Status: Started (no results yet)"
        else
            echo "  Status: Not started"
        fi
    fi
    echo ""
done

echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
