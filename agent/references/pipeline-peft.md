# Pipeline & PEFT Tools

## Pipeline

`yolo.pipeline.experiment` for end-to-end train/val/export/benchmark flows. Accepts stage keys (`train`, `val`, `export`, `benchmark`) or explicit `params.stages` list, plus `inspect`, `lora_diagnose`, `moe_diagnose`, `peft_compare`. Real runs write `progress.jsonl` next to the manifest.

For packaged MoE/MoA/MoT/Latent MoE configurations, `inputs.profile` accepts the stable identifier printed by `yolo mixtures`. Do not also pass `inputs.model`.

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.pipeline.experiment","inputs":{"profile":"26/yolo26-master-mot-n","data":"coco8.yaml"},"params":{"train":{"epochs":1,"imgsz":32},"val":{"imgsz":32},"export":{"format":"onnx"}},"policy":{"dry_run":true}}' --pretty
```

Every `skill_manifest.json` is schema-versioned. Profile-driven manifests record catalog metadata and model YAML SHA-256. Credentials are written as `<redacted>`.

## Release Audit

`yolo.release.audit` builds a read-only release bundle from a completed manifest:

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.release.audit","inputs":{"manifest":"runs/agent/experiment/skill_manifest.json"},"params":{"output":"runs/agent/experiment/release_bundle.json"},"policy":{"dry_run":false}}' --pretty
```

Returns `publishable`, `experimental`, or `refused`. Inspect `decision.missing` and `decision.hard_failures` for details. Legacy manifests without `schema_version` are always `refused`.

Local gate:

```bash
python scripts/audit_release_manifest.py runs/agent/experiment/skill_manifest.json \
  --output runs/agent/experiment/release_bundle.json --fail-on experimental
```

## LoRA Diagnose

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.lora.diagnose","inputs":{"model":"yolo11n.pt"},"params":{"path":"runs/train/exp/weights/lora_adapter_best","svd_max_layers":20,"spectrum_max_layers":12},"policy":{"dry_run":true}}' --pretty
```

## PEFT Compare

```bash
python agent/scripts/run_yolo_master_skill.py --json '{"skill":"yolo.eval.peft_compare","inputs":{"model":"yolo11n.pt","data":"coco8.yaml"},"params":{"train":{"epochs":1,"imgsz":32,"batch":1},"variants":[{"name":"full_sft","train":{"lora_r":0}},{"name":"lora_r8","train":{"lora_type":"lora","lora_r":8,"lora_alpha":16}}]},"policy":{"dry_run":true}}' --pretty
```
