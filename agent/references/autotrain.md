# AutoTrain Loop

The bundled validator and case pack keep the skill surface honest.

## Quick Start

```bash
python agent/scripts/validate_yolo_master_skill.py --suite quick --pretty --summary-only
```

`quick` is the default iteration suite: `fast-smoke` + `dry-run` + `contract`. Use `all` only for the full non-manual regression pass. Sub-suites (`fast-smoke`, `cli-smoke`, `deep-smoke`, `dry-run`, `contract`, `extended`) can be run individually via `--suite <name>`.

## Key Details

- Case pack: `assets/autotrain_cases/` grouped by skill, with `assets/autotrain_cases.json` for compatibility
- Report: `logs/autotrain-report.json`
- `policy.dry_run=true` for cheap coverage before real runs
- Validator enables short-lived runtime cache for Torch/MPS detection
- `fast-smoke` — bootstrap and planning with tight timing budgets
- `cli-smoke` — real `yolo` CLI cold-start execution
- `deep-smoke` — heavyweight real-model inspection and local `.pt` inference
- `extended-cli` — slower real CLI probes (mini-dataset train/val on MPS), marked `manual_only`
- `contract` — failure-path behavior, manifest emission, in-process recovery probes, multimodal stub probes
- CLI failures carry categorized hints for agent recovery
- Built-in dataset YAML names (e.g. `coco128.yaml`) are auto-resolved against the local repository
- CLI responses carry environment metadata; auto-selected runs include recovery trail on device fallback
