# Mixture Optimization Execution Checklist

This checklist keeps the report's proposed optimizations evidence-driven. It
does not treat random-input latency as mAP evidence and does not change DDP
communication or V-PEFT candidate ordering without a measured gate.

## 1. Regional Head: 4096 vs 8192 vs unlimited

```bash
python benchmarks/benchmark_moa_regional.py \
  --device cpu --height 80 --width 80 \
  --kv-budget 4096 8192 none --output output/regional_head.json
```

Record `latency_ms_mean`, `peak_memory_bytes` (CUDA only), `pooled_tokens`,
and, with matched trained artifacts, `mAP.map50_95`:

```bash
python benchmarks/benchmark_moa_regional.py \
  --weights /path/to/checkpoint.pt --data /path/to/data.yaml \
  --device cuda --imgsz 640 --kv-budget 4096 8192 none
```

For a full model validation run, the same switch is available as
`moa_regional_max_kv_tokens=4096|8192|0` (`0` means unlimited):

```bash
yolo val model=/path/to/checkpoint.pt data=/path/to/data.yaml \
  imgsz=640 device=cuda moa_regional_max_kv_tokens=4096
```

Gate: choose the smallest budget whose mAP drop is within the project
tolerance and whose latency/memory measurement is reproducibly better.

## 2. MoT LocalConv P3/P4 ablation

`0` preserves the existing global attention path; positive values enable local
window attention only when the feature map exceeds the window area.

```bash
python benchmarks/benchmark_mot_local_window.py --stage P3 --height 80 --width 80 --window 0 7 14 \
  --output output/mot_p3_local_window.json
python benchmarks/benchmark_mot_local_window.py --stage P4 --height 40 --width 40 --window 0 7 14 \
  --output output/mot_p4_local_window.json
```

For a trained model, pass `--weights` and `--data` to collect matched mAP.
Use `mot_local_attn_window: 0` in the default config and override it only in
an experiment YAML/CLI invocation until the ablation gate passes.

## 3. MoE DDP communication evidence

Single-process output is a compute baseline and cannot prove a communication
bottleneck. Run a real multi-rank job:

```bash
torchrun --nproc_per_node=2 benchmarks/profile_moe_ddp_collectives.py \
  --steps 50 --tokens 4096 --output output/moe_ddp_collectives.json
```

Inspect `all_reduce_event_count` and
`all_reduce_fraction_of_step_cpu`. Only if the multi-rank fraction is material
should a batched collective protocol be designed and benchmarked against the
current mathematical contract. The packed-statistics implementation is now
available in `MoELoss`; rerun the same command before/after on the target NCCL
hardware and require both lower event count and unchanged loss/gradient tests.

## 4. Latent Router recommendation

Keep `LatentRouter.router_init_std=0.0` as the compatibility default. Use the
existing recommended model configs with `router_init_std=0.02`:

```text
ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020.yaml
ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020-temp025.yaml
ultralytics/cfg/models/26/yolo26-master-latent-n-initperturb020-temp05.yaml
```

## 5. V-PEFT candidate diagnostics

```bash
python benchmarks/benchmark_vpeft_diagnostics.py --solver dco --max-iter 40 \
  --output output/vpeft_fixed_variant.json
python benchmarks/benchmark_vpeft_diagnostics.py --solver dco --max-iter 40 --optimize-variant \
  --output output/vpeft_variant_search.json
```

Compare `n_variant_candidates`, `elapsed_seconds`, `final_utility`, budget
feasibility, and target count over representative graphs. Planner pre-sorting
remains deferred until those measurements demonstrate a repeatable benefit.
