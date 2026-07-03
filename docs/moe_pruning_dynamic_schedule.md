# MoE Pruning and Dynamic Schedule

This guide documents the Tencent Rhino-Bird issue #52 workflow for MoE expert pruning and dynamic MoE hyperparameter scheduling. The experiments use VisDrone and a trained YOLO-Master-EsMoE-N checkpoint.

## Review Positioning

This PR provides the reusable issue #52 tooling and reports the current experimental outcome honestly. The pruning sweep, metric collection, plotting, LoRA recovery path, and dynamic schedule ablation are implemented and reproducible. However, the VisDrone checkpoint reproduced from the merged issue #49 workflow has nearly uniform expert utilization, so usage-threshold pruning does not remove experts for thresholds `0.05` to `0.30`.

This should be read as a negative pruning finding for the current checkpoint and threshold criterion, not as evidence that MoE pruning is generally ineffective. Effective structural pruning likely needs either a checkpoint with more skewed routing or a stronger expert-importance metric than hit-count usage alone.

## Files

| file | purpose |
| --- | --- |
| `scripts/moe_pruning_sweep.py` | Runs MoEPruner threshold sweeps, direct validation, LoRA 10-epoch recovery, metrics, and plots. |
| `scripts/run_moe_dynamic_schedule_ablation.py` | Runs baseline, dynamic schedule, and low-balance ablation training; summarizes convergence speed. |
| `scripts/issue52_pruning_results.csv` | Compact pruning-sweep result table for review. |
| `scripts/issue52_per_layer_experts.csv` | Compact per-layer retained-expert table for the direct pruning sweep. |
| `scripts/issue52_expert_usage_gini.csv` | Compact per-layer expert-usage Gini table for the direct pruning sweep. |
| `scripts/issue52_alternative_pruning_results.csv` | Compact result for the optional weighted Top-2 pruning probe. |
| `scripts/issue52_dynamic_schedule_results.csv` | Compact dynamic-schedule result table for review. |
| `ultralytics/nn/modules/moe/schedule.py` | Implements expert-usage Gini, EMA scheduling, and balance-loss coefficient application. |
| `tests/test_moe_dynamic_schedule.py` | Focused regression tests for Gini, dynamic scheduling defaults, and ES-MoE FLOPs. |

## Commands

Pruning sweep:

```bash
python scripts/moe_pruning_sweep.py \
  --project runs/reproduce/issue52_moe_pruning_full \
  --thresholds 0.05 0.10 0.15 0.20 0.30 \
  --lora-epochs 10 \
  --lora-r 8 \
  --lora-alpha 16 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --latency-warmup 20 \
  --latency-reps 100 \
  --wandb offline \
  --plots \
  --exist-ok
```

Dynamic schedule comparison:

```bash
python scripts/run_moe_dynamic_schedule_ablation.py \
  --variant all \
  --project runs/reproduce/issue52_dynamic_schedule_full \
  --epochs 100 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --patience 0 \
  --wandb offline \
  --plots \
  --exist-ok
```

Optional weighted fixed-budget pruning probe:

```bash
python scripts/moe_pruning_sweep.py \
  --project runs/reproduce/issue52_weighted_top2_direct \
  --thresholds 0.10 \
  --direct-only \
  --importance-mode usage_weight \
  --keep-top-m 2 \
  --baseline-map 0.1771901268670128 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --latency-warmup 20 \
  --latency-reps 100 \
  --wandb disabled \
  --exist-ok
```

Optional LoRA10 recovery check for the weighted Top-2 probe:

```bash
python scripts/moe_pruning_sweep.py \
  --project runs/reproduce/issue52_weighted_top2_direct \
  --thresholds 0.10 \
  --skip-existing \
  --importance-mode usage_weight \
  --keep-top-m 2 \
  --baseline-map 0.1771901268670128 \
  --lora-epochs 10 \
  --lora-r 8 \
  --lora-alpha 16 \
  --imgsz 640 \
  --batch 8 \
  --device 0 \
  --workers 8 \
  --latency-warmup 20 \
  --latency-reps 100 \
  --wandb disabled \
  --exist-ok
```

## Dynamic Schedule

The schedule is disabled by default with `moe_dynamic_schedule: "none"`. Enable it with `moe_dynamic_schedule=gini_balance`.

```text
gini_ema_t = beta * gini_ema_{t-1} + (1 - beta) * gini_t
balance_loss_coeff_t = clip(base * exp(alpha * (gini_ema_t - target)), min_coeff, max_coeff)
```

Default values:

```text
base=1.0, target=0.25, alpha=1.0, beta=0.8, min_coeff=0.5, max_coeff=2.0
```

## Results

Settings: VisDrone, `imgsz=640`, base checkpoint from the merged issue #49 workflow, preferably `runs/reproduce/visdrone/VisDrone_EsMoE-N/weights/best.pt` after training with `--no-sparse-eval`. The direct pruning rows were rechecked with that merged workflow checkpoint path. FLOPs below are MoE-layer GFLOPs collected through module `get_gflops()` hooks, not total model GFLOPs. Machine-readable result tables are provided in `scripts/issue52_pruning_results.csv`, `scripts/issue52_per_layer_experts.csv`, `scripts/issue52_expert_usage_gini.csv`, `scripts/issue52_alternative_pruning_results.csv`, and `scripts/issue52_dynamic_schedule_results.csv`.

| Threshold | Stage | mAP50 | mAP50-95 | MoE GFLOPs | Latency mean ms | Params M | Gini mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.05 | direct | 0.30747 | 0.17719 | 2.04288 | 16.53 | 2.653 | 0.00000 |
| 0.05 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 31.67 | 2.883 | 0.24496 |
| 0.10 | direct | 0.30747 | 0.17719 | 2.04288 | 16.37 | 2.653 | 0.00000 |
| 0.10 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 33.26 | 2.883 | 0.24496 |
| 0.15 | direct | 0.30747 | 0.17719 | 2.04288 | 17.26 | 2.653 | 0.00000 |
| 0.15 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 32.28 | 2.883 | 0.24496 |
| 0.20 | direct | 0.30747 | 0.17719 | 2.04288 | 16.77 | 2.653 | 0.00000 |
| 0.20 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 31.89 | 2.883 | 0.24496 |
| 0.30 | direct | 0.30747 | 0.17719 | 2.04288 | 16.02 | 2.653 | 0.00000 |
| 0.30 | LoRA10 | 0.28568 | 0.16146 | 2.30830 | 31.03 | 2.883 | 0.24496 |

Dynamic schedule comparison:

| Variant | Final mAP50 | Final mAP50-95 | Best mAP50-95 | Epoch to 95% baseline | Convergence ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| fixed baseline | 0.30612 | 0.17615 | 0.17664 | 65 | 1.000 |
| gini_balance | 0.30571 | 0.17631 | 0.17755 | 58 | 0.892 |
| low-balance ablation | 0.30677 | 0.17577 | 0.17623 | 63 | 0.969 |

Experimental weighted Top-2 pruning probe:

| Variant | Stage | mAP50 | mAP50-95 | MoE GFLOPs | Latency mean ms | Params M | Retained experts |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| usage-threshold baseline | direct | 0.30747 | 0.17719 | 2.04288 | 16.37 | 2.653 | 3/3 |
| usage_weight + keep_top_m=2 | direct | 0.20923 | 0.11402 | 1.29516 | 14.77 | 2.535 | 2/2 |
| usage_weight + keep_top_m=2 | LoRA10 | 0.26163 | 0.14550 | 2.30830 | 40.67 | 2.883 | 3/3 |

## Findings

- The five pruning thresholds produced identical direct accuracy, Params, and MoE GFLOPs. Expert usage is highly uniform in this checkpoint, so thresholds from 0.05 to 0.30 did not remove experts after the issue #49 merged-checkpoint recheck.
- The compact per-layer tables confirm that all four ES-MoE layers keep `3/3` experts and that direct expert-usage Gini stays at or near zero for every tested threshold.
- The sweep now records `structure_status` and `structure_note` in `summary.csv`, so LoRA recovery runs that rebuild the original expert structure are reported instead of silently treated as structurally pruned results.
- The optional `usage_weight + keep_top_m=2` probe confirms that structural pruning is possible: all four ES-MoE layers become `2/2`, MoE-layer GFLOPs drop from `2.04288` to `1.29516`, and Params drop from `2.653M` to `2.535M`. Direct accuracy also drops sharply, so this is a diagnostic probe rather than a deployment recommendation.
- LoRA10 recovery on the weighted Top-2 checkpoint improves mAP50-95 from `0.11402` to `0.14550`, but the saved LoRA model reloads as `3/3` experts with higher Params, MoE GFLOPs, and latency. It is therefore useful as a recovery-path diagnostic, not as a valid structurally pruned deployment point.
- Direct inference is stronger than LoRA10 recovery in this run. The LoRA adapter adds parameters and latency, while 10 epochs did not recover or improve accuracy.
- The dynamic Gini schedule reaches 95% of the fixed baseline final mAP50-95 faster: 58 epochs vs 65 epochs, a convergence ratio of 0.892.
- For server inference, use unpruned/direct EsMoE or a conservative direct threshold only as a safety check. For edge pruning, a checkpoint with less uniform expert utilization or a stronger importance metric is needed before claiming a clear Pareto gain.

## Recommended Follow-ups

- Extend the experimental expert-importance score beyond `usage_weight` by adding activation magnitude and optional loss-sensitivity probes.
- Test a less balance-regularized checkpoint, because the current balanced routing leaves all experts above the pruning thresholds.
- Use `--keep-top-m` for research-only fixed-budget ablations, while keeping the default MoEPruner path conservative and backward compatible.
- Re-run the Pareto sweep after one of the above changes, then update the server/edge recommendation from measured accuracy-latency trade-offs instead of threshold labels alone.

## Plot Artifacts

The pruning script writes the following plots when `--plots` is enabled:

| artifact | purpose |
| --- | --- |
| `plots/threshold_map_latency_3d.png` | Threshold to mAP/FLOPs/latency 3D view. |
| `plots/threshold_curves.png` | Threshold curves for mAP, MoE GFLOPs, and latency. |
| `plots/pareto_accuracy_latency.png` | Accuracy-latency Pareto front with the selected sweet spot. |

## W&B Links

| Run | W&B |
| --- | --- |
| dynamic fixed baseline | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_dynamic_schedule_full/runs/y7eh0ibu> |
| dynamic gini_balance | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_dynamic_schedule_full/runs/mdw43bgw> |
| dynamic low-balance ablation | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_dynamic_schedule_full/runs/0cqpz8t1> |
| pruning 0.05 LoRA10 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/3auquiwj> |
| pruning 0.10 LoRA10 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/rjgqp23e> |
| pruning 0.15 LoRA10 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/2aj9l005> |
| pruning 0.20 LoRA10 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/8vp5z0fu> |
| pruning 0.30 LoRA10 | <https://wandb.ai/2063055270-harbin-institute-of-technology/-shared-node-ljs25S003031-tencent_rhino_bird_yolo_master_esmoe-YOLO-Master-runs-reproduce-issue52_moe_pruning_full/runs/z37qrbxs> |

Direct pruning validations are summarized in `summary.csv`; W&B links above correspond to training runs that create W&B records.

## Known Issues

- The pruning script reports MoE-layer GFLOPs gathered through `get_gflops()` hooks. It does not report total detector GFLOPs.
- LoRA10 results include adapter parameters, so Params and latency can increase compared with direct pruned inference.
- Since expert utilization is nearly uniform, the current threshold sweep is a negative result for pruning effectiveness rather than a strong pruning win.
- The current sweet spot is selected from a flat Pareto surface because no experts are removed; it should not be interpreted as a real deployment recommendation until structural pruning changes Params, FLOPs, or latency.
- The weighted Top-2 probe shows structural savings but a large direct accuracy drop; current LoRA10 recovery does not preserve the `2/2` expert structure, so a pruning-aware LoRA resume path, layer-wise budgets, or less aggressive pruning is needed before it is useful for deployment.
- This checkout imports `ultralytics.utils.lora.sensitivity`, but the file is absent in `origin/main`; a no-op compatibility shim is included so YOLO and LoRA paths import correctly.
