"""Run the strengthened P1 full detection-step paired benchmark."""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TextIO

import numpy as np
import yaml

from e3_p0.collector import RoutingCollector
from e3_p0.io_utils import environment, sha256_file, write_json, write_jsonl, write_manifest
from e3_p0.runner import _load_tensor, _resolve_images, _validate_events

from .benchmark import _bootstrap_median_ci, _state_sha256, _stats
from .plotting import save_strengthened_plot

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
CONDITIONS = ("off", "capture", "capture_jsonl")
CONDITION_ORDERS = (
    ("off", "capture", "capture_jsonl"),
    ("off", "capture_jsonl", "capture"),
    ("capture", "off", "capture_jsonl"),
    ("capture", "capture_jsonl", "off"),
    ("capture_jsonl", "off", "capture"),
    ("capture_jsonl", "capture", "off"),
)


def _load_strengthened_config(config_path: Path, run_id_override: str | None = None) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must contain a mapping: {config_path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be path-safe and contain at most 128 letters, digits, dots, underscores, or hyphens")
    config["run_id"] = run_id
    for name in ("batch_size", "warmup_pairs", "bootstrap_resamples"):
        config[name] = int(config[name])
        if config[name] <= 0:
            raise ValueError(f"{name} must be positive")
    if config["warmup_pairs"] % len(CONDITION_ORDERS) != 0:
        raise ValueError("warmup_pairs must be divisible by 6 for balanced three-condition ordering")
    batches = [[int(value) for value in batch] for batch in config["sample_batches"]]
    if not batches or any(len(batch) != config["batch_size"] for batch in batches):
        raise ValueError("every sample_batches entry must contain exactly batch_size indices")
    if len({value for batch in batches for value in batch}) != sum(len(batch) for batch in batches):
        raise ValueError("sample_batches must not repeat image indices")
    config["sample_batches"] = batches
    if not isinstance(config.get("profiles"), dict) or not config["profiles"]:
        raise ValueError("profiles must contain at least one routing family")
    for family, profile in config["profiles"].items():
        profile["measured_pairs"] = int(profile["measured_pairs"])
        if profile["measured_pairs"] < 30 or profile["measured_pairs"] % len(CONDITION_ORDERS) != 0:
            raise ValueError(f"profiles.{family}.measured_pairs must be at least 30 and divisible by 6")
    return config


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("e3_p1_strengthened")
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


def _condition_order(pair_index: int) -> list[str]:
    return list(CONDITION_ORDERS[pair_index % len(CONDITION_ORDERS)])


def _label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)
    positions = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not positions:
        raise ValueError(f"Image path has no 'images' component: {image_path}")
    parts[positions[-1]] = "labels"
    return Path(*parts).with_suffix(".txt")


def _load_yolo_labels(label_path: Path, *, num_classes: int) -> np.ndarray:
    if not label_path.is_file():
        raise FileNotFoundError(f"YOLO label file not found: {label_path}")
    rows = []
    for line_number, raw in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        values = [float(value) for value in raw.split()]
        if len(values) != 5:
            raise ValueError(f"Expected 5 YOLO label fields at {label_path}:{line_number}")
        class_id, *box = values
        if class_id != int(class_id) or not 0 <= int(class_id) < num_classes:
            raise ValueError(f"Invalid class id at {label_path}:{line_number}: {class_id}")
        if any(not 0.0 <= value <= 1.0 for value in box):
            raise ValueError(f"YOLO box must be normalized to [0, 1] at {label_path}:{line_number}")
        rows.append(values)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def _build_detection_batches(
    config: dict[str, Any], *, torch_module: Any, num_classes: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    flat_indices = [index for batch in config["sample_batches"] for index in batch]
    paths, dataset_meta = _resolve_images(config["dataset"], config["dataset_split"], flat_indices)
    by_index = dict(zip(flat_indices, paths))
    batches = []
    input_records = []
    for batch_number, indices in enumerate(config["sample_batches"]):
        images = []
        classes = []
        boxes = []
        batch_indices = []
        for within_batch, sample_index in enumerate(indices):
            image_path = by_index[sample_index]
            image, metadata = _load_tensor(image_path, int(config["image_size"]), torch_module)
            label_path = _label_path(image_path)
            labels = _load_yolo_labels(label_path, num_classes=num_classes)
            metadata.update(
                {
                    "sample_index": sample_index,
                    "batch_number": batch_number,
                    "label_path": str(label_path),
                    "label_sha256": sha256_file(label_path),
                    "target_count": len(labels),
                }
            )
            input_records.append(metadata)
            images.append(image)
            if len(labels):
                classes.append(torch_module.from_numpy(labels[:, :1]))
                boxes.append(torch_module.from_numpy(labels[:, 1:]))
                batch_indices.append(torch_module.full((len(labels),), within_batch, dtype=torch_module.long))
        device = config["device"]
        batch = {
            "img": torch_module.stack(images, dim=0).to(device),
            "cls": torch_module.cat(classes, dim=0).to(device) if classes else torch_module.empty((0, 1), device=device),
            "bboxes": torch_module.cat(boxes, dim=0).to(device) if boxes else torch_module.empty((0, 4), device=device),
            "batch_idx": (
                torch_module.cat(batch_indices, dim=0).to(device)
                if batch_indices
                else torch_module.empty((0,), dtype=torch_module.long, device=device)
            ),
        }
        batches.append(batch)
    return batches, {
        **dataset_meta,
        "dataset": config["dataset"],
        "split": config["dataset_split"],
        "sample_batches": config["sample_batches"],
        "batch_size": config["batch_size"],
        "images": input_records,
    }


def _official_optimizer(model: Any, config: dict[str, Any], base_trainer_class: Any) -> Any:
    optimizer_config = config["optimizer"]
    proxy = SimpleNamespace(
        args=SimpleNamespace(
            moe_router_lr_scale=float(optimizer_config["router_lr_scale"]),
            lora_lr_mult=1.0,
            warmup_bias_lr=0.0,
        ),
        data={"nc": int(model.yaml["nc"])},
        adapter_controller=None,
    )
    return base_trainer_class.build_optimizer(
        proxy,
        model,
        name=str(optimizer_config["name"]),
        lr=float(optimizer_config["learning_rate"]),
        momentum=float(optimizer_config["momentum"]),
        decay=float(optimizer_config["weight_decay"]),
        iterations=1000,
    )


def _finite_scalar(value: float, *, name: str) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"Non-finite {name}: {value}")
    return value


def _full_detection_step(
    model: Any,
    optimizer: Any,
    batch: dict[str, Any],
    *,
    seed: int,
    torch_module: Any,
    gradient_clip_norm: float,
    post_step: Any | None = None,
) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch_module.manual_seed(seed)
    if batch["img"].device.type == "cuda":
        torch_module.cuda.synchronize(batch["img"].device)
    started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    losses, loss_items = model.loss(batch)
    total_loss = losses.sum()
    total_loss.backward()
    gradient_norm = torch_module.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)
    optimizer.step()
    if post_step is not None:
        post_step()
    if batch["img"].device.type == "cuda":
        torch_module.cuda.synchronize(batch["img"].device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    result = {
        "elapsed_ms": _finite_scalar(float(elapsed_ms), name="elapsed_ms"),
        "loss_total": _finite_scalar(float(total_loss.detach().cpu()), name="loss_total"),
        "loss_components": [_finite_scalar(float(value), name="loss_component") for value in loss_items.cpu().view(-1)],
        "gradient_norm_preclip": _finite_scalar(float(gradient_norm.detach().cpu()), name="gradient_norm"),
    }
    return result


def _assert_step_equivalence(
    reference: list[dict[str, Any]],
    candidate: list[dict[str, Any]],
    *,
    family: str,
    condition: str,
    config: dict[str, Any],
) -> None:
    if len(reference) != len(candidate):
        raise RuntimeError(f"Step count differs for family={family} condition={condition}")
    for step_index, (off, observed) in enumerate(zip(reference, candidate)):
        if not np.isclose(
            off["loss_total"],
            observed["loss_total"],
            rtol=float(config["loss_match_rtol"]),
            atol=float(config["loss_match_atol"]),
        ):
            raise RuntimeError(
                f"Detection loss changed for family={family} condition={condition} step={step_index}: "
                f"off={off['loss_total']} observed={observed['loss_total']}"
            )
        if not np.allclose(
            off["loss_components"],
            observed["loss_components"],
            rtol=float(config["loss_match_rtol"]),
            atol=float(config["loss_match_atol"]),
        ):
            raise RuntimeError(f"Detection loss components changed for family={family} condition={condition}")
        if not np.isclose(
            off["gradient_norm_preclip"],
            observed["gradient_norm_preclip"],
            rtol=float(config["gradient_match_rtol"]),
            atol=float(config["gradient_match_atol"]),
        ):
            raise RuntimeError(f"Gradient norm changed for family={family} condition={condition}")


def _comparison_summary(values: list[float], *, config: dict[str, Any], seed: int) -> dict[str, Any]:
    result = _stats(values)
    result["bootstrap_95_ci"] = _bootstrap_median_ci(
        values, resamples=int(config["bootstrap_resamples"]), seed=seed
    )
    result["task_status"] = "PASS" if result["median"] < float(config["target_slowdown_percent"]) else "FAIL"
    result["evidence_strength"] = (
        "STRONG" if result["bootstrap_95_ci"][1] < float(config["target_slowdown_percent"]) else "INCONCLUSIVE"
    )
    return result


def _writer_callback(handle: TextIO) -> Any:
    def write_event(event: dict[str, Any]) -> None:
        handle.write(json.dumps(event, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n")

    return write_event


def _validate_event_stream(events: list[dict[str, Any]], *, expected_count: int, label: str) -> dict[str, Any]:
    sequences = [int(event["sequence"]) for event in events]
    expected_sequences = list(range(expected_count))
    if len(events) != expected_count:
        raise RuntimeError(f"Event count differs for {label}: expected={expected_count} actual={len(events)}")
    if sequences != expected_sequences:
        raise RuntimeError(f"Event sequence is not contiguous for {label}")
    return {
        "expected_count": expected_count,
        "actual_count": len(events),
        "first_sequence": sequences[0] if sequences else None,
        "last_sequence": sequences[-1] if sequences else None,
        "contiguous_from_zero": True,
    }


def _benchmark_family(
    *,
    family: str,
    profile: dict[str, Any],
    config: dict[str, Any],
    source_root: Path,
    batches: list[dict[str, Any]],
    run_dir: Path,
    logger: logging.Logger,
    torch_module: Any,
    yolo_class: Any,
    predicate: Any,
    base_trainer_class: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    seed = int(config["seed"])
    model_config = source_root / profile["model_config"]
    models = {}
    optimizers = {}
    collectors: dict[str, RoutingCollector] = {}
    writer_path = run_dir / f"writer-events-{family}.jsonl"
    writer_handle = writer_path.open("w", encoding="utf-8", newline="\n")
    try:
        for condition in CONDITIONS:
            random.seed(seed)
            np.random.seed(seed)
            torch_module.manual_seed(seed)
            model = yolo_class(model_config).model.to(config["device"]).train()
            if isinstance(model.args, dict):
                from ultralytics.cfg import get_cfg

                model.args = get_cfg(model.args)
            models[condition] = model
            optimizers[condition] = _official_optimizer(model, config, base_trainer_class)
        initial_model_hashes = {condition: _state_sha256(model) for condition, model in models.items()}
        initial_optimizer_hashes = {
            condition: _state_sha256(optimizer) for condition, optimizer in optimizers.items()
        }
        if len(set(initial_model_hashes.values())) != 1 or len(set(initial_optimizer_hashes.values())) != 1:
            raise RuntimeError(f"Initial model/optimizer states differ for family={family}")

        for condition in ("capture", "capture_jsonl"):
            collector = RoutingCollector(family=family, run_id=config["run_id"], max_events=4096)
            if condition == "capture_jsonl":
                collector.event_callback = _writer_callback(writer_handle)
            registered = collector.register(models[condition], predicate=predicate)
            if not registered:
                raise RuntimeError(f"No routed modules discovered for family={family} condition={condition}")
            collectors[condition] = collector
        if collectors["capture"].registered_names != collectors["capture_jsonl"].registered_names:
            raise RuntimeError(f"Collector module sets differ for family={family}")
        registered = collectors["capture"].registered_names

        rows = []
        capture_samples = []
        total_pairs = int(config["warmup_pairs"]) + int(profile["measured_pairs"])
        for absolute_index in range(total_pairs):
            phase = "warmup" if absolute_index < int(config["warmup_pairs"]) else "measured"
            pair_index = absolute_index if phase == "warmup" else absolute_index - int(config["warmup_pairs"])
            order = _condition_order(absolute_index)
            condition_results: dict[str, dict[str, Any]] = {}
            for condition in order:
                step_results = []
                captured_total = 0
                for batch_number, batch in enumerate(batches):
                    step_seed = seed + 10000 + absolute_index * len(batches) + batch_number
                    collector = collectors.get(condition)
                    events_before = len(collector.events) if collector is not None else 0
                    if collector is not None:
                        collector.set_context(
                            phase=phase,
                            pair_index=pair_index,
                            batch_number=batch_number,
                            batch_size=int(config["batch_size"]),
                            sample_indices=config["sample_batches"][batch_number],
                            mode="detection_loss_backward_optimizer",
                            condition=condition,
                        )
                    post_step = None
                    if condition == "capture_jsonl" and bool(config["writer_flush_each_step"]):
                        post_step = writer_handle.flush
                    result = _full_detection_step(
                        models[condition],
                        optimizers[condition],
                        batch,
                        seed=step_seed,
                        torch_module=torch_module,
                        gradient_clip_norm=float(config["gradient_clip_norm"]),
                        post_step=post_step,
                    )
                    if collector is not None:
                        new_events = collector.events[events_before:]
                        if len(new_events) != len(registered):
                            raise RuntimeError(
                                f"Expected {len(registered)} events, got {len(new_events)} for "
                                f"family={family} condition={condition} phase={phase} pair={pair_index} batch={batch_number}"
                            )
                        if phase == "measured" and pair_index == 0:
                            _validate_events(new_events)
                            if condition == "capture":
                                capture_samples.extend(new_events)
                        captured_total += len(new_events)
                    step_results.append(result)
                condition_results[condition] = {
                    "elapsed_ms": sum(step["elapsed_ms"] for step in step_results),
                    "steps": step_results,
                    "captured_events": captured_total,
                }

            for condition in ("capture", "capture_jsonl"):
                _assert_step_equivalence(
                    condition_results["off"]["steps"],
                    condition_results[condition]["steps"],
                    family=family,
                    condition=condition,
                    config=config,
                )
            model_hashes = {condition: _state_sha256(model) for condition, model in models.items()}
            optimizer_hashes = {
                condition: _state_sha256(optimizer) for condition, optimizer in optimizers.items()
            }
            if len(set(model_hashes.values())) != 1:
                raise RuntimeError(f"Model trajectories diverged for family={family} phase={phase} pair={pair_index}")
            if len(set(optimizer_hashes.values())) != 1:
                raise RuntimeError(f"Optimizer trajectories diverged for family={family} phase={phase} pair={pair_index}")
            if phase == "measured":
                off_ms = condition_results["off"]["elapsed_ms"]
                capture_ms = condition_results["capture"]["elapsed_ms"]
                writer_ms = condition_results["capture_jsonl"]["elapsed_ms"]
                rows.append(
                    {
                        "family": family,
                        "pair_index": pair_index,
                        "execution_order": order,
                        "step_count": len(batches),
                        "conditions": condition_results,
                        "capture_vs_off_percent": (capture_ms - off_ms) / off_ms * 100.0,
                        "capture_jsonl_vs_off_percent": (writer_ms - off_ms) / off_ms * 100.0,
                        "writer_increment_vs_capture_percent": (writer_ms - capture_ms) / capture_ms * 100.0,
                        "model_state_sha256": model_hashes,
                        "optimizer_state_sha256": optimizer_hashes,
                        "state_equivalence": True,
                    }
                )

        expected_event_count = total_pairs * len(batches) * len(registered)
        event_stream_validation = {
            condition: _validate_event_stream(
                collector.events,
                expected_count=expected_event_count,
                label=f"family={family} condition={condition}",
            )
            for condition, collector in collectors.items()
        }

        capture_values = [row["capture_vs_off_percent"] for row in rows]
        full_values = [row["capture_jsonl_vs_off_percent"] for row in rows]
        writer_values = [row["writer_increment_vs_capture_percent"] for row in rows]
        comparisons = {
            "capture_vs_off_percent": _comparison_summary(capture_values, config=config, seed=seed + 101),
            "capture_jsonl_vs_off_percent": _comparison_summary(full_values, config=config, seed=seed + 202),
            "writer_increment_vs_capture_percent": _comparison_summary(writer_values, config=config, seed=seed + 303),
        }
        primary = comparisons["capture_jsonl_vs_off_percent"]
        summary = {
            "status": primary["task_status"],
            "evidence_strength": primary["evidence_strength"],
            "model_config": profile["model_config"],
            "model_parameters": sum(parameter.numel() for parameter in models["off"].parameters()),
            "registered_modules": registered,
            "measured_pairs": len(rows),
            "steps_per_condition": len(rows) * len(batches),
            "order_position_balance": {
                condition: [sum(row["execution_order"][position] == condition for row in rows) for position in range(3)]
                for condition in CONDITIONS
            },
            "initial_model_state_sha256": initial_model_hashes,
            "initial_optimizer_state_sha256": initial_optimizer_hashes,
            "loss_gradient_and_state_equivalence": "PASS",
            "event_stream_validation": event_stream_validation,
            "hooks_removed": False,
            "condition_ms": {
                condition: _stats([row["conditions"][condition]["elapsed_ms"] for row in rows])
                for condition in CONDITIONS
            },
            "comparisons": comparisons,
            "primary_rule": "capture_jsonl vs off paired median slowdown must be below target",
            "target_slowdown_percent": float(config["target_slowdown_percent"]),
        }
        logger.info(
            "family=%s status=%s strength=%s full_median=%.3f%% ci95=[%.3f, %.3f] pairs=%d",
            family,
            summary["status"],
            summary["evidence_strength"],
            primary["median"],
            primary["bootstrap_95_ci"][0],
            primary["bootstrap_95_ci"][1],
            len(rows),
        )
        return summary, rows, capture_samples
    finally:
        writer_handle.close()
        for collector in collectors.values():
            collector.remove()


def run(
    config_path: Path,
    *,
    run_id: str | None = None,
    update_latest: bool = True,
    family: str | None = None,
    preflight: bool = False,
) -> Path:
    config = _load_strengthened_config(config_path, run_id)
    if family is not None:
        if family not in config["profiles"]:
            raise ValueError(f"Unknown strengthened profile: {family}")
        config["profiles"] = {family: config["profiles"][family]}
    if preflight:
        config["warmup_pairs"] = 0
        for profile in config["profiles"].values():
            profile["measured_pairs"] = len(CONDITION_ORDERS)
        config["execution_mode"] = "preflight"
    else:
        config["execution_mode"] = "formal"
    run_dir = PROJECT_ROOT / "artifacts" / "p1-strengthened" / config["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty strengthened evidence directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = _logger(run_dir / "full.log")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )
    command = "python -m e3_p1.strengthened --config configs/p1_strengthened.yaml"
    if run_id:
        command += f" --run-id {config['run_id']}"
    if family:
        command += f" --family {family}"
    if preflight:
        command += " --preflight"
    if not update_latest:
        command += " --no-update-latest"
    (run_dir / "command.txt").write_text(command + "\n", encoding="utf-8", newline="\n")

    source_root = (PROJECT_ROOT / config["runtime_root"]).resolve()
    p0_repo_root = (PROJECT_ROOT / config["p0_repo_root"]).resolve()
    try:
        if not (p0_repo_root / "src" / "e3_p0").is_dir():
            raise FileNotFoundError(f"P0 contract repository is missing: {p0_repo_root}")
        sys.path.insert(0, str(source_root))
        os.chdir(PROJECT_ROOT)
        import torch
        from ultralytics import YOLO
        from ultralytics.engine.trainer import BaseTrainer
        from ultralytics.nn.modules.routing_protocol import is_routed_module

        first_profile = next(iter(config["profiles"].values()))
        probe = YOLO(source_root / first_profile["model_config"]).model
        num_classes = int(probe.yaml["nc"])
        del probe
        batches, input_metadata = _build_detection_batches(config, torch_module=torch, num_classes=num_classes)
        write_json(run_dir / "input.json", input_metadata)
        write_json(
            run_dir / "environment.json",
            environment(torch, source_root, PROJECT_ROOT, config["official_runtime_ref"], config["official_runtime_tree"]),
        )
        logger.info(
            "scope=E3 P1 strengthened full detection step device=%s batches=%d batch=%d imgsz=%d",
            config["device"],
            len(batches),
            config["batch_size"],
            config["image_size"],
        )
        family_summaries = {}
        all_rows = []
        capture_samples = []
        for family_name, profile in config["profiles"].items():
            summary, rows, samples = _benchmark_family(
                family=family_name,
                profile=profile,
                config=config,
                source_root=source_root,
                batches=batches,
                run_dir=run_dir,
                logger=logger,
                torch_module=torch,
                yolo_class=YOLO,
                predicate=is_routed_module,
                base_trainer_class=BaseTrainer,
            )
            summary["hooks_removed"] = True
            family_summaries[family_name] = summary
            all_rows.extend(rows)
            capture_samples.extend(samples)
        write_jsonl(run_dir / "fullstep-pairs.jsonl", all_rows)
        write_jsonl(run_dir / "capture-schema-samples.jsonl", capture_samples)
        save_strengthened_plot(
            family_summaries,
            run_dir / "fullstep-layered-overhead.png",
            float(config["target_slowdown_percent"]),
        )
        overall_status = "PASS" if all(item["status"] == "PASS" for item in family_summaries.values()) else "FAIL"
        overall_strength = (
            "STRONG"
            if all(item["evidence_strength"] == "STRONG" for item in family_summaries.values())
            else "INCONCLUSIVE"
        )
        summary = {
            "status": overall_status if not preflight else "PREFLIGHT_COMPLETE",
            "criterion_preview": overall_status if preflight else None,
            "formal_verdict_eligible": not preflight,
            "execution_mode": config["execution_mode"],
            "evidence_strength": overall_strength,
            "scope": "E3 P1 strengthened real detection-loss/backward/optimizer paired benchmark",
            "run_id": config["run_id"],
            "official_runtime_ref": config["official_runtime_ref"],
            "official_runtime_tree": config["official_runtime_tree"],
            "families": family_summaries,
            "target_slowdown_percent": float(config["target_slowdown_percent"]),
            "methodology": {
                "conditions": list(CONDITIONS),
                "order": "six balanced permutations across three conditions",
                "warmup_pairs": config["warmup_pairs"],
                "sample_batches": config["sample_batches"],
                "timed_region": "zero_grad + real detection loss + backward + gradient clip + SGD step + optional JSONL write/flush",
                "excluded": "model construction, hook registration/removal, state hashing, schema validation, plotting, dashboard HTTP",
                "primary_statistic": "median paired capture_jsonl-vs-off slowdown",
                "uncertainty": "deterministic-seed bootstrap 95% interval for paired median",
                "task_rule": "each family primary median must be below 10 percent",
                "strength_rule": "STRONG only when each family bootstrap upper bound is also below 10 percent",
            },
            "limitations": [
                "CPU-only random-initialized models; no NVIDIA CUDA, AMP or multi-epoch convergence claim.",
                "Images and labels are deterministically preloaded so unrelated file I/O cannot dilute observer slowdown.",
                "The JSONL condition flushes every step but does not call fsync; dashboard HTTP remains an independent consumer.",
            ],
        }
        write_json(run_dir / "summary.json", summary)
        write_manifest(run_dir)
        if overall_status != "PASS" and not preflight:
            raise RuntimeError("Strengthened formal task criterion failed; see summary.json")
        if update_latest:
            latest = PROJECT_ROOT / "artifacts" / "p1-strengthened" / "LATEST.txt"
            latest.parent.mkdir(parents=True, exist_ok=True)
            latest.write_text(config["run_id"] + "\n", encoding="utf-8", newline="\n")
        logger.info("result=%s evidence_strength=%s artifacts=%s", overall_status, overall_strength, run_dir)
        write_manifest(run_dir)
        return run_dir
    except Exception as exc:
        failure = {
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "failure.json", failure)
        logger.exception("strengthened benchmark failed")
        write_manifest(run_dir)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "p1_strengthened.yaml")
    parser.add_argument("--run-id")
    parser.add_argument("--family")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--no-update-latest", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        args.config.resolve(),
        run_id=args.run_id,
        update_latest=not args.no_update_latest,
        family=args.family,
        preflight=args.preflight,
    )
