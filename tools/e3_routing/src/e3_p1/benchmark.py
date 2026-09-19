"""Run the P1 paired training-path observer benchmark and dashboard smoke test."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import random
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from e3_p0.collector import RoutingCollector
from e3_p0.io_utils import environment, sha256_file, write_json, write_jsonl, write_manifest
from e3_p0.runner import _load_tensor, _resolve_images, _validate_events

from .dashboard import build_snapshot, load_jsonl, render_dashboard_html, smoke_test_server
from .plotting import save_benchmark_plot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


def _load_config(config_path: Path, run_id_override: str | None = None) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must contain a mapping: {config_path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be path-safe and contain at most 128 letters, digits, dots, underscores, or hyphens")
    config["run_id"] = run_id
    for name in ("batch_size", "warmup_pairs", "measured_pairs", "bootstrap_resamples"):
        config[name] = int(config[name])
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["measured_pairs"] < 5:
        raise ValueError("measured_pairs must be at least 5")
    config["sample_indices"] = [int(value) for value in config["sample_indices"]]
    if len(config["sample_indices"]) != config["batch_size"]:
        raise ValueError("P1 sample_indices count must equal batch_size for a fixed paired input")
    return config


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("e3_p1")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    console = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(path, encoding="utf-8", mode="w")
    console.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console)
    logger.addHandler(file_handler)
    return logger


def _tensor_leaves(value: Any, seen: set[int] | None = None) -> list[Any]:
    seen = seen or set()
    leaves = []
    if hasattr(value, "requires_grad") and hasattr(value, "dtype"):
        if id(value) not in seen and bool(value.requires_grad) and bool(value.dtype.is_floating_point):
            seen.add(id(value))
            leaves.append(value)
    elif isinstance(value, dict):
        for child in value.values():
            leaves.extend(_tensor_leaves(child, seen))
    elif isinstance(value, (list, tuple)):
        for child in value:
            leaves.extend(_tensor_leaves(child, seen))
    return leaves


def _surrogate_training_loss(output: Any) -> Any:
    tensors = _tensor_leaves(output)
    if not tensors:
        raise RuntimeError("Model train forward exposed no differentiable floating tensors")
    return sum(tensor.float().square().mean() for tensor in tensors)


def _state_sha256(model: Any) -> str:
    digest = hashlib.sha256()

    def update(value: Any) -> None:
        if hasattr(value, "detach") and hasattr(value, "shape"):
            tensor = value.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(value, dict):
            digest.update(b"dict")
            for key, child in sorted(value.items(), key=lambda item: str(item[0])):
                digest.update(str(key).encode("utf-8"))
                update(child)
        elif isinstance(value, (list, tuple)):
            digest.update(type(value).__name__.encode("ascii"))
            for child in value:
                update(child)
        else:
            digest.update(json.dumps(value, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8"))

    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        update(value)
    return digest.hexdigest()


def _measure_step(model: Any, sample: Any, *, seed: int, torch_module: Any) -> tuple[float, float]:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    model.zero_grad(set_to_none=True)
    if sample.device.type == "cuda":
        torch_module.cuda.synchronize(sample.device)
    started = time.perf_counter()
    output = model(sample)
    loss = _surrogate_training_loss(output)
    loss.backward()
    if sample.device.type == "cuda":
        torch_module.cuda.synchronize(sample.device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return elapsed_ms, float(loss.detach().cpu())


def _stats(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def _bootstrap_median_ci(values: list[float], *, resamples: int, seed: int) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = rng.choice(array, size=(resamples, len(array)), replace=True)
    medians = np.median(sampled, axis=1)
    return [float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))]


def _benchmark_family(
    *,
    family: str,
    profile: dict[str, Any],
    config: dict[str, Any],
    source_root: Path,
    sample: Any,
    logger: logging.Logger,
    torch_module: Any,
    yolo_class: Any,
    predicate: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    seed = int(config["seed"])
    model_config = source_root / profile["model_config"]
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    model_off = yolo_class(model_config).model.to(config["device"]).train()
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    model_on = yolo_class(model_config).model.to(config["device"]).train()
    off_hash = _state_sha256(model_off)
    on_hash = _state_sha256(model_on)
    if off_hash != on_hash:
        raise RuntimeError(f"Initial model state differs between paired conditions for family={family}")

    collector = RoutingCollector(family=family, run_id=config["run_id"], max_events=256)
    registered = collector.register(model_on, predicate=predicate)
    if not registered:
        raise RuntimeError(f"No routed modules discovered for P1 family={family}")

    loss_rtol = float(config["loss_match_rtol"])
    loss_atol = float(config["loss_match_atol"])
    rows = []
    try:
        for phase, count in (("warmup", config["warmup_pairs"]), ("measured", config["measured_pairs"])):
            for pair_index in range(count):
                absolute_index = pair_index if phase == "warmup" else config["warmup_pairs"] + pair_index
                pair_seed = seed + 1000 + absolute_index
                order = ["off", "on"] if absolute_index % 2 == 0 else ["on", "off"]
                timings = {}
                losses = {}
                events_before = len(collector.events)
                for condition in order:
                    if condition == "on":
                        collector.set_context(
                            phase=phase,
                            pair_index=pair_index,
                            batch_size=config["batch_size"],
                            sample_indices=config["sample_indices"],
                            mode="train_forward_backward",
                        )
                        timings[condition], losses[condition] = _measure_step(
                            model_on, sample, seed=pair_seed, torch_module=torch_module
                        )
                    else:
                        timings[condition], losses[condition] = _measure_step(
                            model_off, sample, seed=pair_seed, torch_module=torch_module
                        )
                new_events = collector.events[events_before:]
                if len(new_events) != len(registered):
                    raise RuntimeError(
                        f"Expected {len(registered)} observer events, got {len(new_events)} for family={family}"
                    )
                if phase == "measured" and pair_index == 0:
                    _validate_events(new_events)
                loss_match = bool(np.isclose(losses["off"], losses["on"], rtol=loss_rtol, atol=loss_atol))
                if not loss_match:
                    raise RuntimeError(
                        f"Observer changed surrogate loss for family={family}: off={losses['off']} on={losses['on']}"
                    )
                if phase == "measured":
                    slowdown = (timings["on"] - timings["off"]) / timings["off"] * 100.0
                    rows.append(
                        {
                            "family": family,
                            "pair_index": pair_index,
                            "execution_order": order,
                            "pair_seed": pair_seed,
                            "off_ms": timings["off"],
                            "on_ms": timings["on"],
                            "paired_slowdown_percent": slowdown,
                            "off_loss": losses["off"],
                            "on_loss": losses["on"],
                            "loss_match": loss_match,
                            "captured_events": len(new_events),
                        }
                    )
                collector.events.clear()
    finally:
        collector.remove()

    off_values = [row["off_ms"] for row in rows]
    on_values = [row["on_ms"] for row in rows]
    slowdowns = [row["paired_slowdown_percent"] for row in rows]
    target = float(config["target_slowdown_percent"])
    slowdown_stats = _stats(slowdowns)
    slowdown_stats["bootstrap_95_ci"] = _bootstrap_median_ci(
        slowdowns, resamples=config["bootstrap_resamples"], seed=seed + len(family)
    )
    status = "PASS" if slowdown_stats["median"] < target else "FAIL"
    summary = {
        "status": status,
        "model_config": profile["model_config"],
        "model_parameters": sum(parameter.numel() for parameter in model_off.parameters()),
        "paired_model_state_sha256": off_hash,
        "registered_modules": registered,
        "measured_pairs": len(rows),
        "order_balance": {
            "off_then_on": sum(row["execution_order"] == ["off", "on"] for row in rows),
            "on_then_off": sum(row["execution_order"] == ["on", "off"] for row in rows),
        },
        "off_ms": _stats(off_values),
        "on_ms": _stats(on_values),
        "paired_slowdown_percent": slowdown_stats,
        "target_slowdown_percent": target,
        "target_rule": "PASS when the preregistered median paired slowdown is below the target",
        "loss_equivalence": "PASS",
        "hooks_removed": not collector.handles,
        "benchmark_scope": "real train-mode model forward + differentiable surrogate backward; optimizer step excluded",
    }
    logger.info(
        "family=%s status=%s median_slowdown=%.3f%% ci95=[%.3f, %.3f] pairs=%d",
        family,
        status,
        slowdown_stats["median"],
        slowdown_stats["bootstrap_95_ci"][0],
        slowdown_stats["bootstrap_95_ci"][1],
        len(rows),
    )
    del model_off, model_on
    return summary, rows


def run(config_path: Path, *, run_id: str | None = None, update_latest: bool = True) -> Path:
    config = _load_config(config_path, run_id)
    run_dir = PROJECT_ROOT / "artifacts" / "p1" / config["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty P1 evidence directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir / "full.log")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (run_dir / "command.txt").write_text("run_p1_benchmark.cmd\n", encoding="utf-8")

    source_root = (PROJECT_ROOT / config["runtime_root"]).resolve()
    p0_repo_root = (PROJECT_ROOT / config["p0_repo_root"]).resolve()
    if not (p0_repo_root / "artifacts" / "p0" / "LATEST.txt").is_file():
        raise FileNotFoundError(f"P0 evidence repository is missing or incomplete: {p0_repo_root}")
    sys.path.insert(0, str(source_root))
    os.chdir(PROJECT_ROOT)
    import torch
    from ultralytics import YOLO
    from ultralytics.nn.modules.routing_protocol import is_routed_module

    paths, dataset_meta = _resolve_images(config["dataset"], config["dataset_split"], config["sample_indices"])
    tensors = []
    input_records = []
    for sample_index, path in zip(config["sample_indices"], paths):
        tensor, metadata = _load_tensor(path, int(config["image_size"]), torch)
        metadata["sample_index"] = sample_index
        tensors.append(tensor)
        input_records.append(metadata)
    sample = torch.stack(tensors, dim=0).to(config["device"])
    write_json(
        run_dir / "input.json",
        {
            **dataset_meta,
            "dataset": config["dataset"],
            "split": config["dataset_split"],
            "batch_size": config["batch_size"],
            "images": input_records,
        },
    )

    logger.info(
        "scope=E3 P1 paired train-path observer benchmark device=%s batch=%d imgsz=%d warmup=%d measured=%d",
        config["device"],
        config["batch_size"],
        config["image_size"],
        config["warmup_pairs"],
        config["measured_pairs"],
    )
    family_summaries = {}
    all_rows = []
    for family, profile in config["profiles"].items():
        summary, rows = _benchmark_family(
            family=family,
            profile=profile,
            config=config,
            source_root=source_root,
            sample=sample,
            logger=logger,
            torch_module=torch,
            yolo_class=YOLO,
            predicate=is_routed_module,
        )
        family_summaries[family] = summary
        all_rows.extend(rows)
    write_jsonl(run_dir / "benchmark-pairs.jsonl", all_rows)
    save_benchmark_plot(family_summaries, run_dir / "observer-overhead.png", float(config["target_slowdown_percent"]))

    p0_run_id = (p0_repo_root / "artifacts" / "p0" / "LATEST.txt").read_text(encoding="utf-8").strip()
    p0_events = p0_repo_root / "artifacts" / "p0" / p0_run_id / "routing-all.jsonl"
    dashboard_events = run_dir / "dashboard-events.jsonl"
    shutil.copy2(p0_events, dashboard_events)
    dashboard_snapshot = build_snapshot(load_jsonl(dashboard_events))
    write_json(run_dir / "dashboard-snapshot.json", dashboard_snapshot)
    (run_dir / "dashboard.html").write_text(render_dashboard_html(), encoding="utf-8")
    dashboard_smoke = smoke_test_server(dashboard_events)
    write_json(run_dir / "dashboard-smoke.json", dashboard_smoke)
    if dashboard_smoke["status"] != "PASS":
        raise RuntimeError(f"Dashboard HTTP smoke failed: {dashboard_smoke}")
    write_json(
        run_dir / "dashboard-source.json",
        {
            "source_p0_run_id": p0_run_id,
            "source_repository": "Ricky-7-Yan/YOLO-Master-E3-P0",
            "source_relative_path_from_p1": Path(os.path.relpath(p0_events, PROJECT_ROOT)).as_posix(),
            "source_sha256": sha256_file(p0_events),
            "copied_event_count": dashboard_snapshot["event_count"],
            "families": sorted(dashboard_snapshot["families"]),
        },
    )
    write_json(
        run_dir / "environment.json",
        environment(torch, source_root, PROJECT_ROOT, config["official_runtime_ref"], config["official_runtime_tree"]),
    )
    overall_status = "PASS" if all(item["status"] == "PASS" for item in family_summaries.values()) else "FAIL"
    summary = {
        "status": overall_status,
        "scope": "E3 P1 realtime dashboard + paired train-path observer-overhead benchmark",
        "run_id": config["run_id"],
        "official_runtime_ref": config["official_runtime_ref"],
        "official_runtime_tree": config["official_runtime_tree"],
        "families": family_summaries,
        "dashboard": dashboard_smoke,
        "target_slowdown_percent": float(config["target_slowdown_percent"]),
        "methodology": {
            "paired_conditions": "identical independently instantiated model states; hook off vs hook on",
            "order": "alternating AB/BA to reduce order bias",
            "warmup_pairs": config["warmup_pairs"],
            "measured_pairs": config["measured_pairs"],
            "timed_region": "real model train-mode forward plus differentiable surrogate backward",
            "excluded": "model construction, hook registration/removal, optimizer step, disk serialization, dashboard HTTP",
            "primary_statistic": "median of per-pair slowdown percentages",
            "uncertainty": "deterministic-seed bootstrap 95% interval for the paired median",
        },
        "limitations": [
            "This is a training-path microbenchmark, not a full detector epoch benchmark with labels and optimizer updates.",
            "CPU scheduling noise remains visible; paired AB/BA ordering and bootstrap intervals reduce but do not eliminate it.",
            "The dashboard is local and read-only; authentication and remote multi-user deployment are outside P1 scope.",
        ],
    }
    write_json(run_dir / "summary.json", summary)
    write_manifest(run_dir)
    if update_latest:
        latest = PROJECT_ROOT / "artifacts" / "p1" / "LATEST.txt"
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(config["run_id"] + "\n", encoding="utf-8")
    logger.info("result=%s dashboard=%s artifacts=%s", overall_status, dashboard_smoke["status"], run_dir)
    write_manifest(run_dir)
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "p1" / "p1.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args.config.resolve(), run_id=args.run_id, update_latest=not args.no_update_latest)
