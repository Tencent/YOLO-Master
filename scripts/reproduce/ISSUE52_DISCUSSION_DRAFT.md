# Issue #52 Technical Summary Draft

## Summary

This experiment evaluates MoE expert pruning and a Gini-driven dynamic MoE balance schedule on YOLO-Master-EsMoE-N using VisDrone. The base checkpoint is `runs/reproduce/visdrone/visdrone_100e_denseeval_esmoe/weights/best.pt`.

The pruning sweep completed five thresholds: `0.05, 0.10, 0.15, 0.20, 0.30`. Each threshold was evaluated with direct inference and with LoRA 10-epoch recovery. The dynamic schedule comparison completed three 100-epoch runs: fixed baseline, Gini-driven schedule, and fixed low-balance ablation.

## Reproduction Scripts

- `scripts/reproduce/reproduce_issue52_moe_pruning.py`
- `scripts/reproduce/reproduce_issue52_dynamic_schedule.py`

## Dynamic Schedule

The implemented strategy uses expert-usage Gini to adjust `balance_loss_coeff`.

```text
gini_ema_t = beta * gini_ema_{t-1} + (1 - beta) * gini_t
balance_loss_coeff_t = clip(base * exp(alpha * (gini_ema_t - target)), min_coeff, max_coeff)
```

Default values:

```text
base=1.0, target=0.25, alpha=1.0, beta=0.8, min_coeff=0.5, max_coeff=2.0
```

The feature is backward compatible because `moe_dynamic_schedule` defaults to `none`.

## Pruning Results

FLOPs are MoE-layer GFLOPs collected with `get_gflops()` hooks, not total detector GFLOPs.

| Threshold | Stage | mAP50 | mAP50-95 | MoE GFLOPs | Latency mean ms | Params M | Gini mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | direct | 0.30747 | 0.17719 | 2.04288 | 16.81 | 2.653 | 0.00000 |
| 0.05 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 31.67 | 2.883 | 0.24496 |
| 0.10 | direct | 0.30747 | 0.17719 | 2.04288 | 17.56 | 2.653 | 0.00000 |
| 0.10 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 33.26 | 2.883 | 0.24496 |
| 0.15 | direct | 0.30747 | 0.17719 | 2.04288 | 16.75 | 2.653 | 0.00000 |
| 0.15 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 32.28 | 2.883 | 0.24496 |
| 0.20 | direct | 0.30747 | 0.17719 | 2.04288 | 17.73 | 2.653 | 0.00000 |
| 0.20 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 31.89 | 2.883 | 0.24496 |
| 0.30 | direct | 0.30747 | 0.17719 | 2.04288 | 16.90 | 2.653 | 0.00000 |
| 0.30 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 31.03 | 2.883 | 0.24496 |

## Dynamic Schedule Results

| Variant | Final mAP50 | Final mAP50-95 | Best mAP50-95 | Epoch to 95% baseline | Convergence ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed baseline | 0.30612 | 0.17615 | 0.17664 | 65 | 1.000 |
| gini_balance | 0.30571 | 0.17631 | 0.17755 | 58 | 0.892 |
| low-balance ablation | 0.30677 | 0.17577 | 0.17623 | 63 | 0.969 |

## W&B Links

Dynamic schedule runs:

| Run | W&B |
| --- | --- |
| fixed baseline | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_dynamic_schedule_full/runs/y7eh0ibu> |
| gini_balance | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_dynamic_schedule_full/runs/mdw43bgw> |
| low-balance ablation | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_dynamic_schedule_full/runs/0cqpz8t1> |

LoRA10 pruning recovery runs:

| Threshold | W&B |
| --- | --- |
| 0.05 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/3auquiwj> |
| 0.10 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/rjgqp23e> |
| 0.15 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/2aj9l005> |
| 0.20 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/8vp5z0fu> |
| 0.30 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/z37qrbxs> |

Direct pruning validations are recorded in `summary.csv`; they do not create separate W&B training runs.

## Interpretation

The pruning sweep is mainly a negative result: expert utilization is almost perfectly uniform in this checkpoint, so thresholds up to 0.30 do not create meaningful expert removal. Direct inference is therefore the best choice among tested pruning recovery strategies.

LoRA 10-epoch recovery underperformed direct inference. It also increased parameter count and latency because adapter parameters remain active. This suggests that LoRA recovery is not automatically beneficial for balanced ES-MoE checkpoints, at least with this short schedule.

The Gini dynamic schedule gives a modest convergence benefit. It reaches 95% of baseline final mAP50-95 in 58 epochs, compared with 65 epochs for the fixed baseline. Final accuracy is similar across variants, so the main benefit is convergence speed rather than final mAP.

## Scenario Recommendation

For server-side inference, use direct inference with a conservative threshold such as `0.05` or `0.15`. These points preserve the best measured mAP50-95 and have the lowest latency among the completed direct runs.

For edge-side inference, the current threshold range is not sufficient to justify aggressive pruning. I recommend testing higher thresholds or a checkpoint with more skewed expert utilization before claiming an edge deployment sweet spot.

## Known Limitations

- FLOPs are MoE-layer GFLOPs, not total model GFLOPs.
- The current checkpoint has very uniform expert usage, so the threshold sweep does not strongly exercise MoEPruner.
- LoRA10 uses adapter parameters and may not be a fair latency-improvement strategy unless adapters are fused or exported carefully.
- Results are from VisDrone only. COCO can be added as a follow-up command template rather than the first-line result.

## Artifacts

- `runs/reproduce/issue52_moe_pruning_full/summary.csv`
- `runs/reproduce/issue52_moe_pruning_full/per_layer_experts.csv`
- `runs/reproduce/issue52_moe_pruning_full/expert_usage_gini.csv`
- `runs/reproduce/issue52_moe_pruning_full/plots/threshold_map_latency_3d.png`
- `runs/reproduce/issue52_moe_pruning_full/plots/pareto_accuracy_latency.png`
- `runs/reproduce/issue52_dynamic_schedule_full/dynamic_schedule_summary.csv`
