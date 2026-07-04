# MoE Pruning And Dynamic Scheduling

This note describes the issue-52 experiment workflow and the dynamic
hyperparameter scheduler added for MoE training.

## Dynamic Balance Scheduling

`MoEDynamicScheduler` adjusts the MoE balance-loss coefficient from the current
expert usage distribution. It computes the Gini coefficient of expert usage and
updates:

```text
coeff = clamp(base_coeff * (1 + gain * (ema_gini - target_gini)),
              min_balance_coeff,
              max_balance_coeff)
```

When routing collapses to a few experts, Gini rises and the balance term becomes
stronger. When routing is already balanced, the coefficient relaxes so experts
can specialize. The scheduler is opt-in and backward compatible:

```python
from ultralytics.nn.modules.moe.scheduler import MoEDynamicSchedulerConfig
from ultralytics.nn.modules.moe.loss import MoELoss

loss_fn = MoELoss(
    num_experts=8,
    top_k=2,
    dynamic_scheduler_config=MoEDynamicSchedulerConfig(target_gini=0.25, gain=1.5),
)
```

`AdaptiveBalanceController` accepts the same config for MoE blocks that use the
controller path.

## Pruning Sweep

Generate the required five-threshold experiment plan:

```bash
python scripts/moe_pruning_sweep.py \
  --model runs/train/esmoe_n/weights/best.pt \
  --dataset coco.yaml \
  --thresholds 0.05 0.10 0.15 0.20 0.30 \
  --out-dir runs/moe_pruning_sweep
```

The script writes:

- `moe_pruning_sweep_manifest.json` with exact commands for each threshold.
- `moe_pruning_sweep.csv` with rows for direct inference and LoRA 10-epoch
  recovery.

Fill the CSV with measured `mAP50-95`, `mAP50`, `gflops`, `latency_ms`,
`params_m`, `experts_per_layer`, and `expert_usage_gini` after running the
commands on the target GPU machine.

## Plotting

After metrics are filled:

```bash
python scripts/plot_moe_pruning_sweep.py runs/moe_pruning_sweep/moe_pruning_sweep.csv
```

This emits threshold curves for mAP/FLOPs/latency and a latency-accuracy Pareto
front. The sweet spot is selected by best `mAP50-95 / latency_ms` among Pareto
points.

## Suggested Ablations

- Fixed baseline: no dynamic scheduler.
- Dynamic schedule: Gini-driven balance coefficient.
- Ablation: same scheduler with lower gain or disabled EMA.

Convergence acceleration can be reported as:

```text
speedup = baseline_epoch_to_95pct_final_map / experiment_epoch_to_95pct_final_map
```

Check for side effects such as late-stage over-balancing, oscillating Gini, or
final mAP loss. If those appear, reduce `gain`, increase `ema_momentum`, or raise
`target_gini`.
