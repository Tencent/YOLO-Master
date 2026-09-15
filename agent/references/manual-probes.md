# Manual Probes

Use these for stronger confidence than default smoke suites, without pulling slow jobs into routine validation.

## Environment Doctor

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.system","action":"doctor","params":{"ensure_cli":true}}' --pretty
```

## Extended CLI Suite

```bash
python agent/scripts/validate_yolo_master_skill.py --suite extended --pretty --summary-only
```

## Direct CLI Train/Val

```bash
yolo train model=scripts/peft_validation/yolo11n.pt data=agent/assets/mini-detect/mini_detect.yaml imgsz=64 epochs=1 batch=1 device=mps workers=0 plots=False verbose=False patience=1 project=runs/agent name=train-mini-mps-manual
```

```bash
yolo val model=scripts/peft_validation/yolo11n.pt data=agent/assets/mini-detect/mini_detect.yaml imgsz=16 batch=1 device=mps workers=0 plots=False verbose=False project=runs/agent name=val-mini-mps-manual
```

## Structured Dispatcher with MPS

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.train","runtime":{"prefer_cli":true,"prefer_mps":true},"inputs":{"model":"scripts/peft_validation/yolo11n.pt","data":"agent/assets/mini-detect/mini_detect.yaml"},"params":{"epochs":1,"imgsz":64,"batch":1,"workers":0,"plots":false,"verbose":false,"patience":1},"artifacts":{},"policy":{"dry_run":false}}' --pretty
```

## Regenerate Open-World Report

```bash
python agent/scripts/regenerate_open_world_report.py \
  --input agent/logs/qwen-open-world-small-batch.json \
  --json-out agent/logs/qwen-open-world-small-batch-report.json \
  --md-out agent/logs/qwen-open-world-small-batch-report.md
```
