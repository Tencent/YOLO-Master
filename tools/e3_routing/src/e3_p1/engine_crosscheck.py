"""Cross-check the P1 observer inside the real Ultralytics DetectionTrainer lifecycle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

import numpy as np
import yaml

from e3_p0.collector import RoutingCollector
from e3_p0.io_utils import environment, git_identity, sha256_file, write_json, write_jsonl, write_manifest

from .dashboard import build_snapshot, load_jsonl, render_dashboard_html, smoke_test_server
from .plotting import save_engine_crosscheck_plot
from .strengthened import _validate_event_stream, _writer_callback

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
CONDITIONS = ("off", "capture_jsonl")
LATENT_TRANSIENT_FIELDS = ("_last_routing_logits", "_last_routing_probs", "_last_routing_summary")


def _load_engine_config(config_path: Path, run_id_override: str | None = None) -> dict[str, Any]:
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise TypeError(f"Config must contain a mapping: {config_path}")
    config = dict(loaded)
    run_id = str(run_id_override if run_id_override is not None else config.get("run_id", "")).strip()
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError("run_id must be path-safe and contain at most 128 letters, digits, dots, underscores, or hyphens")
    config["run_id"] = run_id
    for name in ("batch_size", "image_size", "epochs", "warmup_pairs", "measured_pairs"):
        config[name] = int(config[name])
        if config[name] < 0 or name in {"batch_size", "image_size", "measured_pairs"} and config[name] == 0:
            raise ValueError(f"{name} must be positive (warmup_pairs may be zero)")
    if config["measured_pairs"] % 2:
        raise ValueError("measured_pairs must be even for balanced AB/BA ordering")
    if not isinstance(config.get("profiles"), dict) or len(config["profiles"]) < 1:
        raise ValueError("profiles must contain at least one routing family")
    return config


def _condition_order(pair_index: int) -> list[str]:
    return list(CONDITIONS if pair_index % 2 == 0 else reversed(CONDITIONS))


def _hash_value(value: Any) -> str:
    """Return a stable recursive SHA-256 for tensors and JSON-like containers."""

    digest = hashlib.sha256()

    def update(child: Any) -> None:
        if hasattr(child, "detach") and hasattr(child, "shape"):
            tensor = child.detach().cpu().contiguous()
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        elif isinstance(child, dict):
            digest.update(b"dict")
            for key, item in sorted(child.items(), key=lambda pair: str(pair[0])):
                digest.update(str(key).encode("utf-8"))
                update(item)
        elif isinstance(child, (list, tuple)):
            digest.update(type(child).__name__.encode("ascii"))
            for item in child:
                update(item)
        else:
            digest.update(json.dumps(child, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8"))

    update(value)
    return digest.hexdigest()


def _state_sha256(stateful: Any) -> str:
    return _hash_value(stateful.state_dict())


def _clear_nonleaf_latent_transients(model: Any) -> list[dict[str, Any]]:
    """Clear constructor-time Latent snapshots that prevent ModelEMA deepcopy."""

    cleared = []
    for module_name, module in model.named_modules():
        for field in LATENT_TRANSIENT_FIELDS:
            value = getattr(module, field, None)
            if hasattr(value, "is_leaf") and not bool(value.is_leaf):
                cleared.append(
                    {
                        "module": module_name,
                        "type": type(module).__name__,
                        "field": field,
                        "shape": list(value.shape),
                        "state_dict_member": field in module.state_dict(),
                    }
                )
                setattr(module, field, None)
    return cleared


def _logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("e3_p1_engine_crosscheck")
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


def _read_metrics_csv(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    return {str(key).strip(): str(value).strip() for key, value in rows[-1].items()} if rows else {}


def _verify_runtime_inputs(source_root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Verify a Git checkout by identity or an extracted snapshot by preregistered file hashes."""

    identity = git_identity(source_root)
    expected_files = config.get("runtime_input_sha256")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("runtime_input_sha256 must preregister the Trainer and routed-model source files")
    records = []
    for relative_path, expected_hash in expected_files.items():
        path = source_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Preregistered runtime input is missing: {relative_path}")
        actual_hash = sha256_file(path)
        if actual_hash != expected_hash:
            raise RuntimeError(f"Runtime input hash differs for {relative_path}")
        records.append({"path": relative_path, "sha256": actual_hash, "status": "MATCH"})
    identity_policy = config.get("runtime_identity_policy", "exact")
    if identity_policy not in {"exact", "base_ancestor"}:
        raise ValueError("runtime_identity_policy must be exact or base_ancestor")
    if identity["head"] is not None and identity_policy == "exact":
        if identity["head"] != config["official_runtime_ref"]:
            raise RuntimeError("Official runtime commit differs from the preregistered ref")
        if identity["tree"] != config["official_runtime_tree"]:
            raise RuntimeError("Official runtime tree differs from the preregistered tree")
    elif identity["head"] is not None:
        ancestry = subprocess.run(
            ["git", "-C", str(source_root), "merge-base", "--is-ancestor", config["official_runtime_ref"], "HEAD"],
            capture_output=True,
            check=False,
            text=True,
            encoding="utf-8",
        )
        if ancestry.returncode != 0:
            raise RuntimeError("Configured runtime base is not an ancestor of the integration checkout")
    mode = "extracted_snapshot_file_hashes"
    if identity["head"] is not None:
        mode = "git_identity_and_file_hashes" if identity_policy == "exact" else "git_base_ancestor_and_file_hashes"
    return {
        "status": "PASS",
        "mode": mode,
        "identity_policy": identity_policy,
        "configured_ref": config["official_runtime_ref"],
        "configured_tree": config["official_runtime_tree"],
        "git_identity": identity,
        "files": records,
    }


class _TrainerObserver:
    """Own callback state for exactly one trainer execution."""

    def __init__(
        self,
        *,
        family: str,
        condition: str,
        pair_index: int,
        phase: str,
        run_id: str,
        writer_path: Path | None,
        predicate: Any,
    ) -> None:
        self.family = family
        self.condition = condition
        self.pair_index = pair_index
        self.phase = phase
        self.run_id = run_id
        self.writer_path = writer_path
        self.predicate = predicate
        self.collector: RoutingCollector | None = None
        self.writer: TextIO | None = None
        self.registered_modules: list[str] = []
        self.batch_records: list[dict[str, Any]] = []
        self.epoch_started: float | None = None
        self.epoch_windows_ms: list[float] = []
        self.initial_model_sha256: str | None = None
        self.initial_optimizer_sha256: str | None = None
        self.final_model_sha256: str | None = None
        self.final_optimizer_sha256: str | None = None
        self.final_ema_sha256: str | None = None
        self.loss_items: list[float] = []

    def install(self, trainer: Any) -> None:
        trainer.add_callback("on_train_start", self.on_train_start)
        trainer.add_callback("on_train_epoch_start", self.on_train_epoch_start)
        trainer.add_callback("on_train_batch_start", self.on_train_batch_start)
        trainer.add_callback("on_train_batch_end", self.on_train_batch_end)
        trainer.add_callback("on_train_epoch_end", self.on_train_epoch_end)
        trainer.add_callback("on_train_end", self.on_train_end)
        trainer.add_callback("teardown", self.teardown)

    def on_train_start(self, trainer: Any) -> None:
        self.initial_model_sha256 = _state_sha256(trainer.model)
        self.initial_optimizer_sha256 = _state_sha256(trainer.optimizer)
        if self.condition == "capture_jsonl":
            if self.writer_path is None:
                raise RuntimeError("capture_jsonl requires a writer path")
            self.writer = self.writer_path.open("w", encoding="utf-8", newline="\n")
            self.collector = RoutingCollector(
                family=self.family,
                run_id=self.run_id,
                max_events=4096,
                event_callback=_writer_callback(self.writer),
            )
            self.registered_modules = self.collector.register(trainer.model, predicate=self.predicate)
            if not self.registered_modules:
                raise RuntimeError(f"No routed modules discovered for family={self.family}")

    def on_train_epoch_start(self, trainer: Any) -> None:
        del trainer
        self.epoch_started = time.perf_counter()

    def on_train_batch_start(self, trainer: Any) -> None:
        batch = trainer.batch
        batch_index = len(self.batch_records)
        files = [str(value) for value in batch.get("im_file", [])]
        payload = {
            "img": batch.get("img"),
            "cls": batch.get("cls"),
            "bboxes": batch.get("bboxes"),
            "batch_idx": batch.get("batch_idx"),
            "im_file": files,
        }
        self.batch_records.append(
            {
                "batch_index": batch_index,
                "im_file": files,
                "batch_size": int(batch["img"].shape[0]),
                "target_count": int(batch.get("cls", []).shape[0]) if hasattr(batch.get("cls"), "shape") else 0,
                "raw_batch_sha256": _hash_value(payload),
            }
        )
        if self.collector is not None:
            self.collector.set_context(
                phase=self.phase,
                engine="ultralytics.models.yolo.detect.DetectionTrainer",
                condition=self.condition,
                pair_index=self.pair_index,
                epoch=int(trainer.epoch),
                batch_index=batch_index,
                batch_size=int(batch["img"].shape[0]),
                sample_paths=files,
            )

    def on_train_batch_end(self, trainer: Any) -> None:
        del trainer
        if self.writer is not None:
            self.writer.flush()

    def on_train_epoch_end(self, trainer: Any) -> None:
        del trainer
        if self.epoch_started is None:
            raise RuntimeError("epoch timer was not started")
        self.epoch_windows_ms.append((time.perf_counter() - self.epoch_started) * 1000.0)

    def on_train_end(self, trainer: Any) -> None:
        self.final_model_sha256 = _state_sha256(trainer.model)
        self.final_optimizer_sha256 = _state_sha256(trainer.optimizer)
        self.final_ema_sha256 = _state_sha256(trainer.ema.ema) if trainer.ema is not None else None
        self.loss_items = [float(value) for value in trainer.tloss.detach().cpu().view(-1)]

    def teardown(self, trainer: Any) -> None:
        del trainer
        self.close()

    def close(self) -> None:
        if self.collector is not None:
            self.collector.remove()
        if self.writer is not None and not self.writer.closed:
            self.writer.flush()
            self.writer.close()


def _assert_pair_equivalence(off: dict[str, Any], observed: dict[str, Any], *, family: str, pair_index: int) -> None:
    exact_fields = (
        "initial_model_sha256",
        "initial_optimizer_sha256",
        "final_model_sha256",
        "final_optimizer_sha256",
        "final_ema_sha256",
        "optimizer_steps",
        "batch_count",
        "epochs_completed",
        "sanitized_transients",
    )
    for field in exact_fields:
        if off[field] != observed[field]:
            raise RuntimeError(f"Pair mismatch family={family} pair={pair_index} field={field}")
    off_batches = [item["raw_batch_sha256"] for item in off["batches"]]
    observed_batches = [item["raw_batch_sha256"] for item in observed["batches"]]
    if off_batches != observed_batches:
        raise RuntimeError(f"Trainer batch stream changed family={family} pair={pair_index}")
    if not np.allclose(off["loss_items"], observed["loss_items"], rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"Trainer loss items changed family={family} pair={pair_index}")


def _run_condition(
    *,
    trainer_class: Any,
    predicate: Any,
    source_root: Path,
    scratch_root: Path,
    run_dir: Path,
    config: dict[str, Any],
    family: str,
    profile: dict[str, Any],
    condition: str,
    pair_index: int,
    phase: str,
    seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    label = f"{family}-{phase}-p{pair_index:02d}-{condition}"
    writer_path = run_dir / f"events-{label}.jsonl" if condition == "capture_jsonl" else None
    overrides = {
        "model": str(source_root / profile["model_config"]),
        "data": config["dataset"],
        "epochs": int(config["epochs"]),
        "batch": int(config["batch_size"]),
        "imgsz": int(config["image_size"]),
        "device": config["device"],
        "workers": 0,
        "amp": False,
        "cache": False,
        "val": False,
        "plots": False,
        "save": False,
        "pretrained": False,
        "optimizer": config["optimizer"]["name"],
        "lr0": float(config["optimizer"]["learning_rate"]),
        "momentum": float(config["optimizer"]["momentum"]),
        "weight_decay": float(config["optimizer"]["weight_decay"]),
        "warmup_epochs": 0.0,
        "close_mosaic": 0,
        "nbs": int(config["batch_size"]),
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "fliplr": 0.0,
        "flipud": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "multi_scale": 0.0,
        "project": str(scratch_root),
        "name": label,
        "exist_ok": False,
        "verbose": False,
        "seed": seed,
        "deterministic": True,
    }
    random.seed(seed)
    np.random.seed(seed)
    lifecycle_started = time.perf_counter()
    trainer = trainer_class(overrides=overrides)
    observer = _TrainerObserver(
        family=family,
        condition=condition,
        pair_index=pair_index,
        phase=phase,
        run_id=f"{config['run_id']}-{family}-{phase}-p{pair_index:02d}",
        writer_path=writer_path,
        predicate=predicate,
    )
    observer.install(trainer)
    try:
        trainer.train()
        full_lifecycle_ms = (time.perf_counter() - lifecycle_started) * 1000.0
        if len(observer.epoch_windows_ms) != int(config["epochs"]) or observer.final_model_sha256 is None:
            raise RuntimeError(f"Trainer callbacks did not complete for {label}")
        events = load_jsonl(writer_path, limit=10000) if writer_path is not None else []
        expected_events = len(observer.registered_modules) * len(observer.batch_records)
        event_validation = (
            _validate_event_stream(events, expected_count=expected_events, label=label)
            if condition == "capture_jsonl"
            else {"expected_count": 0, "actual_count": 0, "contiguous_from_zero": True}
        )
        row = {
            "family": family,
            "phase": phase,
            "pair_index": pair_index,
            "condition": condition,
            "seed": seed,
            "epoch_window_ms": float(sum(observer.epoch_windows_ms)),
            "epoch_windows_ms": observer.epoch_windows_ms,
            "full_lifecycle_ms": full_lifecycle_ms,
            "batch_count": len(observer.batch_records),
            "epochs_completed": len(observer.epoch_windows_ms),
            "optimizer_steps": int(trainer.optimizer_steps),
            "amp_enabled": bool(trainer.amp),
            "device": str(trainer.device),
            "registered_modules": observer.registered_modules,
            "sanitized_transients": list(getattr(trainer, "evidence_sanitized_transients", [])),
            "event_validation": event_validation,
            "initial_model_sha256": observer.initial_model_sha256,
            "initial_optimizer_sha256": observer.initial_optimizer_sha256,
            "final_model_sha256": observer.final_model_sha256,
            "final_optimizer_sha256": observer.final_optimizer_sha256,
            "final_ema_sha256": observer.final_ema_sha256,
            "loss_items": observer.loss_items,
            "batches": observer.batch_records,
            "metrics": _read_metrics_csv(Path(trainer.csv)),
        }
        return row, events
    finally:
        observer.close()


def _family_summary(rows: list[dict[str, Any]], *, measured_pairs: int) -> dict[str, Any]:
    measured = [row for row in rows if row["phase"] == "measured"]
    by_pair: dict[int, dict[str, dict[str, Any]]] = {}
    for row in measured:
        by_pair.setdefault(int(row["pair_index"]), {})[str(row["condition"])] = row
    slowdowns = []
    lifecycle_slowdowns = []
    for conditions in by_pair.values():
        off = conditions["off"]
        observed = conditions["capture_jsonl"]
        slowdowns.append((observed["epoch_window_ms"] / off["epoch_window_ms"] - 1.0) * 100.0)
        lifecycle_slowdowns.append((observed["full_lifecycle_ms"] / off["full_lifecycle_ms"] - 1.0) * 100.0)
    observed_rows = [row for row in measured if row["condition"] == "capture_jsonl"]
    return {
        "measured_pairs": measured_pairs,
        "ab_pairs": sum(_condition_order(index)[0] == "off" for index in by_pair),
        "ba_pairs": sum(_condition_order(index)[0] == "capture_jsonl" for index in by_pair),
        "epoch_window_slowdown_percent": {
            "values": slowdowns,
            "median": float(np.median(slowdowns)),
            "min": float(np.min(slowdowns)),
            "max": float(np.max(slowdowns)),
        },
        "full_lifecycle_slowdown_percent": {
            "values": lifecycle_slowdowns,
            "median": float(np.median(lifecycle_slowdowns)),
            "min": float(np.min(lifecycle_slowdowns)),
            "max": float(np.max(lifecycle_slowdowns)),
        },
        "registered_modules": observed_rows[0]["registered_modules"],
        "batch_count_per_run": observed_rows[0]["batch_count"],
        "epochs_per_run": observed_rows[0]["epochs_completed"],
        "optimizer_steps_per_run": observed_rows[0]["optimizer_steps"],
        "writer_event_count": sum(row["event_validation"]["actual_count"] for row in observed_rows),
        "setup_transient_sanitization_count_per_run": len(observed_rows[0]["sanitized_transients"]),
        "pair_equivalence": "PASS",
        "purpose": "integration cross-check; not a replacement formal overhead verdict",
    }


def run(
    config_path: Path,
    *,
    run_id: str | None = None,
    update_latest: bool = True,
    family: str | None = None,
    preflight: bool = False,
) -> Path:
    config = _load_engine_config(config_path, run_id)
    if family is not None:
        if family not in config["profiles"]:
            raise ValueError(f"Unknown engine profile: {family}")
        config["profiles"] = {family: config["profiles"][family]}
    if preflight:
        config["warmup_pairs"] = 0
        config["measured_pairs"] = 2
    config["execution_mode"] = "preflight" if preflight else "integration_crosscheck"
    run_dir = PROJECT_ROOT / "artifacts" / "p1-engine" / config["run_id"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty engine evidence directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    scratch_root = run_dir / "_scratch"
    logger = _logger(run_dir / "full.log")
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )
    command = "python -m e3_p1.engine_crosscheck --config configs/p1_engine_crosscheck.yaml"
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
    try:
        runtime_verification = _verify_runtime_inputs(source_root, config)
        write_json(run_dir / "runtime-inputs.json", runtime_verification)
        sys.path.insert(0, str(source_root))
        os.chdir(PROJECT_ROOT)
        import torch
        from ultralytics.models.yolo.detect import DetectionTrainer
        from ultralytics.nn.modules.routing_protocol import is_routed_module

        class EvidenceDetectionTrainer(DetectionTrainer):
            """Use the official train loop while suppressing checkpoint I/O in this evidence harness."""

            def _bootstrap_healthy_checkpoint(self) -> None:
                return None

            def _refresh_healthy_checkpoint(self) -> bool:
                return False

            def save_model(self) -> bool:
                return False

            def get_model(self, cfg: str | None = None, weights: str | None = None, verbose: bool = True) -> Any:
                model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
                self.evidence_sanitized_transients = _clear_nonleaf_latent_transients(model)
                return model

        write_json(
            run_dir / "environment.json",
            environment(torch, source_root, PROJECT_ROOT, config["official_runtime_ref"], config["official_runtime_tree"]),
        )
        all_rows = []
        all_events = []
        family_summaries = {}
        total_pairs = int(config["warmup_pairs"]) + int(config["measured_pairs"])
        for family_name, profile in config["profiles"].items():
            family_rows = []
            for absolute_index in range(total_pairs):
                phase = "warmup" if absolute_index < config["warmup_pairs"] else "measured"
                pair_index = absolute_index if phase == "warmup" else absolute_index - config["warmup_pairs"]
                seed = int(config["seed"]) + 1000 * list(config["profiles"]).index(family_name) + pair_index
                order = _condition_order(absolute_index)
                pair_rows = {}
                for position, condition in enumerate(order):
                    row, events = _run_condition(
                        trainer_class=EvidenceDetectionTrainer,
                        predicate=is_routed_module,
                        source_root=source_root,
                        scratch_root=scratch_root,
                        run_dir=run_dir,
                        config=config,
                        family=family_name,
                        profile=profile,
                        condition=condition,
                        pair_index=pair_index,
                        phase=phase,
                        seed=seed,
                    )
                    row["order"] = order
                    row["position"] = position
                    pair_rows[condition] = row
                    family_rows.append(row)
                    all_rows.append(row)
                    all_events.extend(events)
                _assert_pair_equivalence(pair_rows["off"], pair_rows["capture_jsonl"], family=family_name, pair_index=pair_index)
                logger.info(
                    "family=%s phase=%s pair=%d order=%s epoch_off_ms=%.3f epoch_on_ms=%.3f equivalent=PASS",
                    family_name,
                    phase,
                    pair_index,
                    "/".join(order),
                    pair_rows["off"]["epoch_window_ms"],
                    pair_rows["capture_jsonl"]["epoch_window_ms"],
                )
            family_summaries[family_name] = _family_summary(
                family_rows, measured_pairs=int(config["measured_pairs"])
            )
        write_jsonl(run_dir / "trainer-runs.jsonl", all_rows)
        write_jsonl(run_dir / "routing-engine-all.jsonl", all_events)
        snapshot = build_snapshot(all_events)
        write_json(run_dir / "dashboard-snapshot.json", snapshot)
        (run_dir / "dashboard.html").write_text(render_dashboard_html(), encoding="utf-8", newline="\n")
        dashboard_smoke = smoke_test_server(run_dir / "routing-engine-all.jsonl")
        write_json(run_dir / "dashboard-smoke.json", dashboard_smoke)
        if dashboard_smoke["status"] != "PASS" or set(dashboard_smoke["api_families"]) != set(config["profiles"]):
            raise RuntimeError("Dashboard failed to consume the Trainer-generated event stream")
        save_engine_crosscheck_plot(
            family_summaries,
            run_dir / "trainer-epoch-crosscheck.png",
            float(config["target_slowdown_percent"]),
        )
        summary = {
            "status": "PREFLIGHT_COMPLETE" if preflight else "PASS",
            "execution_mode": config["execution_mode"],
            "formal_verdict_eligible": False,
            "scope": "real Ultralytics DetectionTrainer and coco8 dataloader integration cross-check",
            "run_id": config["run_id"],
            "official_runtime_ref": config["official_runtime_ref"],
            "official_runtime_tree": config["official_runtime_tree"],
            "runtime_verification": runtime_verification["mode"],
            "families": family_summaries,
            "dashboard": dashboard_smoke,
            "event_count": len(all_events),
            "methodology": {
                "conditions": list(CONDITIONS),
                "order": "balanced AB/BA by pair",
                "warmup_pairs": int(config["warmup_pairs"]),
                "measured_pairs": int(config["measured_pairs"]),
                "timed_epoch_window": "sum of each on_train_epoch_start through on_train_epoch_end window, including dataloader iteration, preprocess, detection loss, backward, optimizer steps, event JSON encoding/write/flush",
                "full_lifecycle_window": "trainer construction through train completion",
                "determinism": "same pair seed; stochastic geometric/color augmentation disabled; raw collated batch hashes must match",
                "equivalence": "initial/final model, optimizer and EMA hashes; raw batch hashes; loss items; optimizer step count",
                "checkpoint_policy": "checkpoint serialization suppressed in the harness; official dataloader, preprocess, loss, backward, optimizer and callback loop retained",
                "latent_setup_guard": "clear only non-leaf _last_routing_logits/probs/summary snapshots after model construction so ModelEMA deepcopy can start; records are stored per run",
            },
            "interpretation": "Integration evidence only. The preregistered 108-pair strengthened run remains the formal <10% overhead verdict.",
            "limitations": [
                "CPU-only, five coco8 train epochs per execution, batch=2, imgsz=64.",
                "The configured measured pairs are descriptive and intentionally not used for a new confidence-bound verdict.",
                "Validation, checkpoint serialization, CUDA, AMP and multi-epoch convergence are outside this cross-check.",
                "The frozen Latent model cannot enter ModelEMA deepcopy without clearing nine constructor-time non-leaf routing snapshot fields; this harness records and applies that setup guard without changing state_dict parameters.",
            ],
        }
        write_json(run_dir / "summary.json", summary)
        if scratch_root.exists():
            resolved_scratch = scratch_root.resolve()
            if run_dir.resolve() not in resolved_scratch.parents:
                raise RuntimeError("Refusing to remove scratch directory outside the evidence run")
            shutil.rmtree(resolved_scratch)
        write_manifest(run_dir)
        if update_latest:
            latest = PROJECT_ROOT / "artifacts" / "p1-engine" / "LATEST.txt"
            latest.parent.mkdir(parents=True, exist_ok=True)
            latest.write_text(config["run_id"] + "\n", encoding="utf-8", newline="\n")
        logger.info("result=PASS trainer_runs=%d events=%d artifacts=%s", len(all_rows), len(all_events), run_dir)
        write_manifest(run_dir)
        return run_dir
    except Exception as exc:
        write_json(
            run_dir / "failure.json",
            {
                "status": "FAIL",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        logger.exception("engine cross-check failed")
        write_manifest(run_dir)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "p1_engine_crosscheck.yaml")
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
