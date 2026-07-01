# YOLO-Master Domain-Specific LoRA Tuning

This document tracks the domain-specific LoRA tuning experiments for YOLO-Master
models. It is intended to make the Brain Tumor and VisDrone runs reproducible,
document the LoRA target policy for MoE-based YOLO-Master variants, and provide a
single place for result tables, logs, and follow-up analysis.

## Scope

- Model family: YOLO-Master detection models
- Primary model config: `ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml`
- LoRA configs:
  - `examples/lora_examples/yolo_master_brain_tumor_lora.yaml`
  - `examples/lora_examples/yolo_master_visdrone_lora.yaml`
- Completed experiment set:
  - Brain Tumor LoRA rank sweep: `r=4`, `r=8`, `r=16`
- Planned experiment set:
  - VisDrone LoRA rank sweep
  - Optional routing-layer ablation

## Repository Layout

```text
examples/lora_examples/
  yolo_master_brain_tumor_lora.yaml
  yolo_master_visdrone_lora.yaml
  yolo_master_lora_README.md

logs/
  brain_tumor_r4.log
  brain_tumor_r8.log
  brain_tumor_r16.log

runs/lora_examples/
  brain_tumor_r4/
    args.yaml
    results.csv
    results.png
    weights/
      best.pt
      last.pt
      lora_adapter_best/
  brain_tumor_r8/
  brain_tumor_r16/
```

## Experimental Setup

| Item | Value |
| --- | --- |
| Task | Detection |
| Model | `ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml` |
| Dataset | `ultralytics/cfg/datasets/brain-tumor.yaml` |
| Image size | `640` |
| Epochs | `40` |
| Batch size | `16` |
| Optimizer | `auto` resolved to AdamW in the recorded runs |
| AMP | Enabled |
| Seed | `0` |
| Project dir | `runs/lora_examples` |

## LoRA Target Policy

The YOLO-Master v0.10 model uses `VisualEnhancedAdaptiveGateMoE` blocks instead
of the older `ES_MOE` blocks. Therefore, the LoRA targets are selected from the
actual v0.10 module names.

Main tuning targets:

```yaml
lora_target_modules: [
  "conv", "fused_conv", "bottleneck.0", "shared_feature.0", "static_net.3", "proj",
  "expert_projections.0.0", "expert_projections.1.0", "expert_projections.2.0", "expert_projections.3.0",
  "expert_projections.4.0", "expert_projections.5.0", "expert_projections.6.0", "expert_projections.7.0",
  "expert_projections.8.0", "expert_projections.9.0", "expert_projections.10.0", "expert_projections.11.0",
  "expert_projections.12.0", "expert_projections.13.0", "expert_projections.14.0", "expert_projections.15.0"
]
```

Routing and gate layers are excluded from the main LoRA target set:

```yaml
lora_exclude_modules: ["router", "routing", "gate", "gating"]
```

Rationale: the main experiment adapts visual, expert, and projection layers while
keeping expert-assignment dynamics controlled. Routing-layer LoRA should be
reported separately as an ablation if tested.

## Brain Tumor Runs

### Commands

```bash
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml \
  model=ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml \
  data=ultralytics/cfg/datasets/brain-tumor.yaml \
  lora_r=4 lora_alpha=8 name=brain_tumor_r4

yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml \
  model=ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml \
  data=ultralytics/cfg/datasets/brain-tumor.yaml \
  lora_r=8 lora_alpha=16 name=brain_tumor_r8

yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml \
  model=ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml \
  data=ultralytics/cfg/datasets/brain-tumor.yaml \
  lora_r=16 lora_alpha=32 name=brain_tumor_r16
```

### Result Summary

| Run | Rank | Trainable params | Adapter params | Best epoch | mAP50 | mAP50-95 | Train time | Peak GPU mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `brain_tumor_r4` | 4 | 468,290 | 123,116 | 30 | 0.43492 | 0.28312 | 0.664 h | 3.95G |
| `brain_tumor_r8` | 8 | 596,782 | 251,608 | 35 | 0.46004 | 0.31215 | 0.664 h | 3.99G |
| `brain_tumor_r16` | 16 | 848,390 | 503,216 | 37 | 0.48212 | 0.34044 | 0.664 h | 4.03G |

Notes:

- `Trainable params` and `Adapter params` are taken from the `[LoRA] Stats` log line.
- `Train time` is taken from the `epochs completed in` log line.
- `Peak GPU mem` is the maximum `GPU_mem` value observed in the epoch progress lines.

## Observations

- Higher LoRA rank improved the Brain Tumor validation metrics in the completed sweep.
- `r16` achieved the best recorded mAP50-95 among the three runs.
- MoE routing-collapse warnings appeared during training and were handled by the existing recovery/noise adjustment logic.
- The recorded runs used `optimizer=auto`; the trainer resolved the effective optimizer and learning rate in the run logs.



## VisDrone Runs

VisDrone experiments should use the same result-reporting format once completed.

Recommended smoke run:

```bash
yolo train cfg=examples/lora_examples/yolo_master_visdrone_lora.yaml \
  model=ultralytics/cfg/models/master/v0_10/det/yolo-master-n.yaml \
  data=ultralytics/cfg/datasets/VisDrone.yaml \
  epochs=3 batch=4 imgsz=640 \
  lora_r=8 lora_alpha=16 \
  name=visdrone_v0_10_lora_smoke
```

Planned result table:

| Run | Rank | Trainable params | Adapter params | Best epoch | mAP50 | mAP50-95 | Train time | Peak GPU mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `visdrone_lora_r4` | 4 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `visdrone_lora_r8` | 8 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| `visdrone_lora_r16` | 16 | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

## Reproducing Result Tables

Use `results.csv` to identify the best epoch by `metrics/mAP50-95(B)`. Use the
matching log file to extract trainable parameters, adapter parameters, total
training time, and peak `GPU_mem`.

```bash
python - <<'PY'
import csv
import re
from pathlib import Path

ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
stats_re = re.compile(r"Trainable: ([\d,]+).*Adapter Params: ([\d,]+)")
time_re = re.compile(r"epochs completed in ([0-9.]+) hours")
mem_re = re.compile(r"\b\d+\s*/\s*\d+\s+([0-9.]+)G\b")

for run_dir in sorted(Path("runs/lora_examples").glob("brain_tumor_r*")):
    results = run_dir / "results.csv"
    if not results.exists():
        continue
    rows = list(csv.DictReader(results.open()))
    best = max(rows, key=lambda r: float(r["metrics/mAP50-95(B)"]))

    log_path = Path("logs") / f"{run_dir.name}.log"
    trainable = adapter = train_time = peak_mem = "NA"
    if log_path.exists():
        text = ansi.sub("", log_path.read_text(errors="ignore")).replace("\r", "\n")
        if m := stats_re.search(text):
            trainable, adapter = m.groups()
        if m := time_re.search(text):
            train_time = f"{m.group(1)} h"
        mem_values = [float(x) for x in mem_re.findall(text)]
        if mem_values:
            peak_mem = f"{max(mem_values):.2f}G"

    print(
        run_dir.name,
        "epoch=", best["epoch"],
        "P=", best["metrics/precision(B)"],
        "R=", best["metrics/recall(B)"],
        "mAP50=", best["metrics/mAP50(B)"],
        "mAP50-95=", best["metrics/mAP50-95(B)"],
        "trainable=", trainable,
        "adapter=", adapter,
        "train_time=", train_time,
        "peak_mem=", peak_mem,
    )
PY
```

## Reporting Checklist

- Record the exact command used for each run.
- Keep `args.yaml`, `results.csv`, `results.png`, `best.pt`, and adapter files.
- Record selected LoRA module count from the log line:
  `Final Targets Passed to PEFT`.
- Record trainable parameter count from the log line:
  `[LoRA] Stats: Trainable`.
- Record total training time from the log line:
  `epochs completed in`.
- Record numerical warnings such as routing collapse, NaN recovery, or fitness
  collapse.
- Do not compare wall-clock time for runs launched concurrently on the same GPU.

## Troubleshooting Notes

### Fitness Collapse or NaN Weights

If a run fails with corrupted `last.pt` or NaN/Inf weights, start a new run name
instead of resuming from the corrupted checkpoint. Conservative settings:

```bash
yolo train cfg=examples/lora_examples/yolo_master_brain_tumor_lora.yaml \
  lora_r=16 lora_alpha=16 \
  lr0=0.0003 lora_lr_mult=0.5 lora_alpha_warmup=5 \
  amp=False \
  name=brain_tumor_r16_stable_debug
```

### Target Modules Do Not Match Expectations

`lora_target_modules` entries are matching rules. The trainer expands them into
full module names after structural filtering. Validate the final set through the
log line:

```text
[LoRA] Final Targets Passed to PEFT (List Length: ...)
```

For YOLO-Master v0.10, `expert_projections.0.0` through
`expert_projections.15.0` are valid in layer 11. There is no
`expert_projections.16.0` for the current `yolo-master-n.yaml`.

## Next Steps

- Add a short analysis paragraph comparing `r4`, `r8`, and `r16`.
- Add GPU memory summaries from logs or system-level profiling.
- Add VisDrone smoke and formal experiment results.
- Add an optional routing-LoRA ablation table if routing targets are tested.
