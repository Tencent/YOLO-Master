#!/usr/bin/env python3
"""Run and verify live E3 routing telemetry with the real TensorBoard callback."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "moe": "ultralytics/cfg/models/26/yolo26-master-n.yaml",
    "mot": "ultralytics/cfg/models/26/yolo26-master-mot-n.yaml",
    "latent": "ultralytics/cfg/models/26/yolo26-master-latent-n.yaml",
}


def build_parser() -> argparse.ArgumentParser:
    """Build the live-training validation command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="coco8.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--imgsz", type=int, default=64)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--routing-interval", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("runs/e3_routing_validation"))
    parser.add_argument("--skip-disabled", action="store_true", help="skip the telemetry-off control run")
    return parser


def _run(command: list[str], *, env: dict[str, str]) -> None:
    """Run one subprocess from the repository root."""
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def _git_metadata() -> dict[str, object]:
    """Record the exact source revision used by the validation."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout
    return {"commit": commit, "dirty": bool(dirty.strip())}


def _training_signature(run_dir: Path) -> list[dict[str, str]]:
    """Return deterministic training values while excluding elapsed wall time."""
    with (run_dir / "results.csv").open(newline="", encoding="utf-8") as file:
        return [
            {key.strip(): value for key, value in row.items() if key.strip() not in {"epoch", "time"}}
            for row in csv.DictReader(file)
        ]


def _event_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Read scalar events from one TensorBoard run."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError(f"TensorBoard event was not created in {run_dir}")
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return {
        tag: [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags().get("scalars", [])
    }


def _validate_enabled(run_dir: Path, *, epochs: int) -> dict[str, object]:
    """Require live JSON and TensorBoard routing evidence to agree."""
    from ultralytics.utils.routing_telemetry import routing_snapshot_scalars

    telemetry_path = run_dir / "telemetry.json"
    if not telemetry_path.is_file():
        raise RuntimeError(f"telemetry JSON was not created in {run_dir}")
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    last = telemetry["ranks"][0]["routing"]["last"]
    expected = routing_snapshot_scalars(last)
    if not expected or int(last.get("routed_layers", 0)) <= 0:
        raise RuntimeError(f"no routed layers were captured in {run_dir}")
    if int(last.get("invalid_layers", 0)) or int(last.get("unsupported_layers", 0)):
        raise RuntimeError(f"invalid or unsupported routed layers were captured in {run_dir}")

    scalars = _event_scalars(run_dir)
    routing = {tag: values for tag, values in scalars.items() if tag.startswith("routing/")}
    expected_steps = set(range(1, epochs + 1))
    if not routing or any(expected_steps - {step for step, _ in values} for values in routing.values()):
        raise RuntimeError(f"routing scalars do not cover every epoch in {run_dir}")
    for tag, value in expected.items():
        if tag not in routing or abs(routing[tag][-1][1] - float(value)) > 1e-5:
            raise RuntimeError(f"TensorBoard/JSON mismatch for {tag} in {run_dir}")
    return {
        "routed_layers": int(last["routed_layers"]),
        "routing_observations": int(telemetry["ranks"][0]["routing"]["observations"]),
        "routing_scalar_tags": len(routing),
        "steps": sorted(expected_steps),
    }


def _validate_disabled(run_dir: Path) -> dict[str, object]:
    """Require the control run to emit no routing telemetry."""
    if (run_dir / "telemetry.json").exists() or (run_dir / "telemetry_rank_0.json").exists():
        raise RuntimeError("telemetry-off control unexpectedly wrote telemetry JSON")
    routing = [tag for tag in _event_scalars(run_dir) if tag.startswith("routing/")]
    if routing:
        raise RuntimeError(f"telemetry-off control unexpectedly wrote routing tags: {routing[:5]}")
    return {"telemetry_json": False, "routing_scalar_tags": 0}


def main(argv: list[str] | None = None) -> int:
    """Run the live three-family validation and its telemetry-off control."""
    args = build_parser().parse_args(argv)
    if args.epochs < 2:
        raise ValueError("--epochs must be at least 2 to prove live TensorBoard history")
    output = (ROOT / args.output).resolve() if not args.output.is_absolute() else args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_root = output / "ultralytics_config"
    config_root.mkdir(parents=True, exist_ok=True)

    # This environment is complete before any subprocess imports Ultralytics. The isolated
    # config directory prevents the validation from changing the user's global integration settings.
    env = {
        **os.environ,
        "YOLO_CONFIG_DIR": str(config_root),
        "YOLO_TRAIN_TELEMETRY": "1",
        "YOLO_TRAIN_TELEMETRY_ROUTING_ENABLED": "1",
        "YOLO_TRAIN_TELEMETRY_ROUTING_INTERVAL": str(max(args.routing_interval, 1)),
    }
    yolo = str(Path(sys.executable).with_name("yolo"))
    _run([yolo, "settings", "tensorboard=True", "wandb=False"], env=env)

    summary: dict[str, object] = {
        "schema_version": "e3.training_smoke.v1",
        "data": args.data,
        "device": args.device,
        "epochs": args.epochs,
        "git": _git_metadata(),
        "routing_interval": max(args.routing_interval, 1),
        "families": {},
    }
    common = [
        "train",
        f"data={args.data}",
        f"epochs={args.epochs}",
        f"imgsz={args.imgsz}",
        f"batch={args.batch}",
        f"device={args.device}",
        "workers=0",
        "plots=False",
        "val=False",
        "amp=False",
        "seed=0",
        "deterministic=True",
        f"project={output}",
        "exist_ok=True",
    ]
    for family, model in MODELS.items():
        run_dir = output / family
        _run([yolo, *common, f"model={model}", f"name={family}"], env=env)
        summary["families"][family] = _validate_enabled(run_dir, epochs=args.epochs)

    if not args.skip_disabled:
        disabled_env = {**env, "YOLO_TRAIN_TELEMETRY": "0"}
        run_dir = output / "disabled"
        _run([yolo, *common, f"model={MODELS['moe']}", "name=disabled"], env=disabled_env)
        summary["disabled_control"] = _validate_disabled(run_dir)
        if _training_signature(run_dir) != _training_signature(output / "moe"):
            raise RuntimeError("telemetry on/off runs produced different deterministic training values")
        summary["disabled_control"]["training_values_match_moe"] = True

    summary_path = output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"E3 live training validation passed: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
