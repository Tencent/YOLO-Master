from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.cli.async_jobs import async_requested, submit_async_skill
from runtime.cli.contract import json_safe, plan_response, response, write_manifest
from runtime.cli.device import (
    apply_runtime_defaults,
    collect_environment_report,
    resolve_device_selection,
)
from runtime.cli.executor import (
    build_cli_key_values,
    build_evaluation_summary,
    cli_args_from_values,
    cli_logs,
    cli_plan,
    cli_save_dir,
    ensure_cli_success,
    parse_cli_speed,
    parse_detection_cli_metrics,
    parse_predict_cli_output,
    read_results_csv_metrics,
)
from runtime.cli.normalize import is_dry_run, prefer_cli


@dataclass(frozen=True)
class CoreDeps:
    build_model: Callable[[dict[str, Any]], Any]
    run_cli: Callable[..., dict[str, Any]]
    run_cli_with_recovery: Callable[..., dict[str, Any]]


def metrics_payload(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    if hasattr(metrics, "results_dict"):
        return json_safe(metrics.results_dict)
    if isinstance(metrics, dict):
        return json_safe(metrics)
    return {"value": json_safe(metrics)}


def summarize_results(results: Any, max_items: int = 10) -> list[dict[str, Any]]:
    summary = []
    iterable = list(results)
    for result in iterable[:max_items]:
        item: dict[str, Any] = {
            "path": str(getattr(result, "path", "")),
            "speed": json_safe(getattr(result, "speed", {})),
        }
        boxes = getattr(result, "boxes", None)
        masks = getattr(result, "masks", None)
        probs = getattr(result, "probs", None)
        obb = getattr(result, "obb", None)
        if boxes is not None:
            try:
                item["boxes"] = len(boxes)
            except Exception:
                item["boxes"] = 0
        if masks is not None:
            try:
                item["masks"] = len(masks)
            except Exception:
                item["masks"] = 0
        if obb is not None:
            try:
                item["obb"] = len(obb)
            except Exception:
                item["obb"] = 0
        if probs is not None:
            item["classification"] = {
                "top1": json_safe(getattr(probs, "top1", None)),
                "top1conf": json_safe(getattr(probs, "top1conf", None)),
            }
        summary.append(item)
    return summary


def run_train_like(request: dict[str, Any], skill_name: str, deps: CoreDeps) -> dict[str, Any]:
    params = dict(request["params"])
    if request["inputs"].get("data") and "data" not in params:
        params["data"] = request["inputs"]["data"]
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose="train")
    effective_request = deepcopy(request)
    effective_request["params"] = params
    if async_requested(request) and not is_dry_run(request):
        effective_request["skill"] = skill_name
        return submit_async_skill(effective_request)
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
            return cli_plan(
                request,
                cli_args_from_values("train", values),
                extra={
                    "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                    "auto_completed": auto_completed,
                },
            )
        return plan_response(
            request,
            "training dry run prepared",
            "python_api",
            "YOLO(...).train",
            params=params,
            next_actions=["yolo.val", "yolo.export"],
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
            },
        )

    if prefer_cli(request):
        values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
        cli_execution = deps.run_cli_with_recovery(
            request,
            "train",
            values,
            failure_summary="training failed",
            selected_device=chosen_device,
            selection_source=device_selection["source"],
        )
        failed = cli_execution["failed"]
        if failed:
            return failed
        cli_result = cli_execution["cli_result"]
        values = cli_execution["values"]
        final_device = cli_execution["device"]
        recovery = cli_execution["recovery"]
        save_dir = cli_save_dir(request, values)
        artifacts = []
        metrics = read_results_csv_metrics(save_dir)
        parsed_metrics, _ = parse_detection_cli_metrics(cli_result["stdout"])
        for key, value in parsed_metrics.items():
            metrics.setdefault(key, value)
        evaluation = build_evaluation_summary(metrics, cli_result["stdout"])
        environment = collect_environment_report(
            effective_request,
            selected_device=final_device,
            requested_device=chosen_device,
            selection_source="recovery" if recovery and recovery.get("recovered") else device_selection["source"],
            cli_info=cli_result["install"],
        )
        if save_dir and save_dir.exists():
            best = save_dir / "weights" / "best.pt"
            last = save_dir / "weights" / "last.pt"
            if best.exists():
                artifacts.append({"kind": "checkpoint", "label": "best", "path": str(best.resolve())})
            if last.exists():
                artifacts.append({"kind": "checkpoint", "label": "last", "path": str(last.resolve())})
            if (save_dir / "results.csv").exists():
                artifacts.append({"kind": "csv", "path": str((save_dir / "results.csv").resolve())})
            if (save_dir / "args.yaml").exists():
                artifacts.append({"kind": "config", "path": str((save_dir / "args.yaml").resolve())})
        return response(
            skill_name,
            "ok",
            "training finished after automatic cpu fallback" if recovery and recovery.get("recovered") else "training finished",
            job={
                "mode": "sync",
                "save_dir": json_safe(save_dir),
                "resume_supported": True,
                "executor": "cli",
                "device": final_device,
            },
            metrics=metrics,
            evaluation=evaluation,
            environment=environment,
            auto_completed=auto_completed,
            artifacts=artifacts,
            logs=cli_logs(cli_result),
            attempts=cli_execution["attempts"] if recovery else [],
            recovery=recovery or {},
            next_actions=["yolo.val", "yolo.export"],
        )

    model = deps.build_model(request)
    metrics = model.train(**params)
    environment = collect_environment_report(effective_request, selected_device=chosen_device)
    artifacts = []
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None)
    if save_dir:
        save_dir = Path(save_dir)
        best = getattr(trainer, "best", None)
        last = getattr(trainer, "last", None)
        if best and Path(best).exists():
            artifacts.append({"kind": "checkpoint", "label": "best", "path": str(Path(best).resolve())})
        if last and Path(last).exists():
            artifacts.append({"kind": "checkpoint", "label": "last", "path": str(Path(last).resolve())})
        if (save_dir / "results.csv").exists():
            artifacts.append({"kind": "csv", "path": str((save_dir / "results.csv").resolve())})
        if (save_dir / "args.yaml").exists():
            artifacts.append({"kind": "config", "path": str((save_dir / "args.yaml").resolve())})
    payload = response(
        skill_name,
        "ok",
        "training finished",
        job={"mode": "sync", "save_dir": json_safe(save_dir), "resume_supported": True, "device": chosen_device},
        metrics=metrics_payload(metrics or getattr(model, "metrics", None)),
        environment=environment,
        auto_completed=auto_completed,
        artifacts=artifacts,
        next_actions=["yolo.val", "yolo.export"],
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_val(request: dict[str, Any], deps: CoreDeps) -> dict[str, Any]:
    params = dict(request["params"])
    if request["inputs"].get("data") and "data" not in params:
        params["data"] = request["inputs"]["data"]
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose="val")
    effective_request = deepcopy(request)
    effective_request["params"] = params
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
            return cli_plan(
                request,
                cli_args_from_values("val", values),
                extra={
                    "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                    "auto_completed": auto_completed,
                },
            )
        return plan_response(
            request,
            "validation dry run prepared",
            "python_api",
            "YOLO(...).val",
            params=params,
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
            },
        )

    if prefer_cli(request):
        values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
        cli_execution = deps.run_cli_with_recovery(
            request,
            "val",
            values,
            failure_summary="validation failed",
            selected_device=chosen_device,
            selection_source=device_selection["source"],
        )
        failed = cli_execution["failed"]
        if failed:
            return failed
        cli_result = cli_execution["cli_result"]
        values = cli_execution["values"]
        final_device = cli_execution["device"]
        recovery = cli_execution["recovery"]
        save_dir = cli_save_dir(request, values)
        artifacts = []
        metrics, evaluation = parse_detection_cli_metrics(cli_result["stdout"])
        speed = parse_cli_speed(cli_result["stdout"])
        evaluation = build_evaluation_summary(metrics, cli_result["stdout"]) if metrics else ({"speed_ms": speed} if speed else {})
        environment = collect_environment_report(
            effective_request,
            selected_device=final_device,
            requested_device=chosen_device,
            selection_source="recovery" if recovery and recovery.get("recovered") else device_selection["source"],
            cli_info=cli_result["install"],
        )
        if save_dir and (save_dir / "predictions.json").exists():
            artifacts.append({"kind": "json", "path": str((save_dir / "predictions.json").resolve())})
        return response(
            request["skill"],
            "ok",
            "validation finished after automatic cpu fallback" if recovery and recovery.get("recovered") else "validation finished",
            metrics=metrics,
            evaluation=evaluation,
            environment=environment,
            auto_completed=auto_completed,
            job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "cli", "device": final_device},
            artifacts=artifacts,
            logs=cli_logs(cli_result),
            attempts=cli_execution["attempts"] if recovery else [],
            recovery=recovery or {},
        )

    model = deps.build_model(request)
    metrics = model.val(**params)
    environment = collect_environment_report(effective_request, selected_device=chosen_device)
    artifacts = []
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir:
        save_dir = Path(save_dir)
        if (save_dir / "predictions.json").exists():
            artifacts.append({"kind": "json", "path": str((save_dir / "predictions.json").resolve())})
    payload = response(
        request["skill"],
        "ok",
        "validation finished",
        metrics=metrics_payload(metrics),
        environment=environment,
        auto_completed=auto_completed,
        job={"mode": "sync", "save_dir": json_safe(save_dir), "device": chosen_device},
        artifacts=artifacts,
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_predict_like(request: dict[str, Any], mode: str, deps: CoreDeps) -> dict[str, Any]:
    params = dict(request["params"])
    source = request["inputs"].get("source") or params.pop("source", None)
    if source is None:
        raise ValueError("`inputs.source` is required for predict/track.")
    max_items = int(params.pop("max_items", 10))
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose=mode)
    effective_request = deepcopy(request)
    effective_request["params"] = params
    effective_request["inputs"]["source"] = source
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params={"max_items"}, inject_save_dir=True)
            return cli_plan(
                effective_request,
                cli_args_from_values(mode, values),
                extra={
                    "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                    "auto_completed": auto_completed,
                },
            )
        target = "YOLO(...).predict" if mode == "predict" else "YOLO(...).track"
        plan_params = {"source": source, **params}
        return plan_response(
            effective_request,
            f"{mode} dry run prepared",
            "python_api",
            target,
            params=plan_params,
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
            },
        )

    if prefer_cli(request):
        values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params={"max_items"}, inject_save_dir=True)
        cli_execution = deps.run_cli_with_recovery(
            request,
            mode,
            values,
            failure_summary=f"{mode} failed",
            selected_device=chosen_device,
            selection_source=device_selection["source"],
        )
        failed = cli_execution["failed"]
        if failed:
            return failed
        cli_result = cli_execution["cli_result"]
        values = cli_execution["values"]
        final_device = cli_execution["device"]
        recovery = cli_execution["recovery"]
        save_dir = cli_save_dir(request, values)
        results, speed = parse_predict_cli_output(cli_result["stdout"])
        environment = collect_environment_report(
            effective_request,
            selected_device=final_device,
            requested_device=chosen_device,
            selection_source="recovery" if recovery and recovery.get("recovered") else device_selection["source"],
            cli_info=cli_result["install"],
        )
        payload = response(
            request["skill"],
            "ok",
            f"{mode} finished after automatic cpu fallback" if recovery and recovery.get("recovered") else f"{mode} finished",
            job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "cli", "device": final_device},
            results=results[:max_items],
            environment=environment,
            auto_completed=auto_completed,
            logs=cli_logs(cli_result),
            attempts=cli_execution["attempts"] if recovery else [],
            recovery=recovery or {},
        )
        if speed and payload["results"]:
            payload["results"][0]["speed"] = speed
        if save_dir and save_dir.exists():
            payload["artifacts"] = [{"kind": "directory", "path": str(save_dir.resolve())}]
        return payload

    model = deps.build_model(request)
    if mode == "predict":
        results = model.predict(source=source, **params)
    else:
        results = model.track(source=source, **params)
    save_dir = getattr(model.predictor, "save_dir", None)
    payload = response(
        request["skill"],
        "ok",
        f"{mode} finished",
        job={"mode": "sync", "save_dir": json_safe(save_dir), "device": chosen_device},
        environment=collect_environment_report(effective_request, selected_device=chosen_device),
        auto_completed=auto_completed,
        results=summarize_results(results, max_items=max_items),
    )
    if save_dir:
        payload["artifacts"] = [{"kind": "directory", "path": str(Path(save_dir).resolve())}]
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_export(request: dict[str, Any], deps: CoreDeps) -> dict[str, Any]:
    params = dict(request["params"])
    device_selection = resolve_device_selection(request, params)
    chosen_device = device_selection["device"]
    environment = collect_environment_report(
        request,
        selected_device=chosen_device,
        requested_device=chosen_device,
        selection_source=device_selection["source"],
    )
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
            return cli_plan(
                request,
                cli_args_from_values("export", values),
                extra={"environment": environment, "auto_completed": {}},
            )
        return plan_response(
            request,
            "export dry run prepared",
            "python_api",
            "YOLO(...).export",
            params=request["params"],
            extra={"environment": environment, "auto_completed": {}},
        )

    if prefer_cli(request):
        values = build_cli_key_values(request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
        cli_result = deps.run_cli(cli_args_from_values("export", values))
        failed = ensure_cli_success(request, cli_result, "export failed")
        if failed:
            return failed
        save_dir = cli_save_dir(request, values)
        artifacts = []
        if save_dir and save_dir.exists():
            for candidate in sorted(save_dir.rglob("*")):
                if candidate.is_file() and candidate.suffix not in {".csv", ".yaml", ".json", ".txt", ".jpg", ".png"}:
                    artifacts.append({"kind": "exported_model", "path": str(candidate.resolve())})
        return response(
            request["skill"],
            "ok",
            "export finished",
            artifacts=artifacts,
            environment=collect_environment_report(
                request,
                selected_device=chosen_device,
                requested_device=chosen_device,
                selection_source=device_selection["source"],
                cli_info=cli_result["install"],
            ),
            auto_completed={},
            job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "cli"},
            logs=cli_logs(cli_result),
        )

    model = deps.build_model(request)
    exported = model.export(**request["params"])
    payload = response(
        request["skill"],
        "ok",
        "export finished",
        artifacts=[{"kind": "exported_model", "path": str(Path(exported).resolve())}],
        environment=environment,
        auto_completed={},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_benchmark(request: dict[str, Any], deps: CoreDeps) -> dict[str, Any]:
    params = dict(request["params"])
    data = request["inputs"].get("data") or params.pop("data", None)
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose="benchmark")
    effective_request = deepcopy(request)
    effective_request["params"] = params
    if is_dry_run(request):
        if prefer_cli(request):
            values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
            return cli_plan(
                request,
                cli_args_from_values("benchmark", values),
                extra={
                    "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                    "auto_completed": auto_completed,
                },
            )
        return plan_response(
            request,
            "benchmark dry run prepared",
            "python_api",
            "YOLO(...).benchmark",
            params={"data": data, **params},
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
            },
        )

    if prefer_cli(request):
        values = build_cli_key_values(effective_request, skip_inputs={"task"}, skip_params=set(), inject_save_dir=True)
        cli_execution = deps.run_cli_with_recovery(
            request,
            "benchmark",
            values,
            failure_summary="benchmark failed",
            selected_device=chosen_device,
            selection_source=device_selection["source"],
        )
        failed = cli_execution["failed"]
        if failed:
            return failed
        cli_result = cli_execution["cli_result"]
        values = cli_execution["values"]
        final_device = cli_execution["device"]
        recovery = cli_execution["recovery"]
        save_dir = cli_save_dir(request, values)
        return response(
            request["skill"],
            "ok",
            "benchmark finished after automatic cpu fallback" if recovery and recovery.get("recovered") else "benchmark finished",
            data={"benchmark": {}},
            environment=collect_environment_report(
                effective_request,
                selected_device=final_device,
                requested_device=chosen_device,
                selection_source="recovery" if recovery and recovery.get("recovered") else device_selection["source"],
                cli_info=cli_result["install"],
            ),
            auto_completed=auto_completed,
            job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "cli", "device": final_device},
            logs=cli_logs(cli_result),
            attempts=cli_execution["attempts"] if recovery else [],
            recovery=recovery or {},
        )

    model = deps.build_model(request)
    benchmark_result = model.benchmark(data=data, **params)
    payload = response(
        request["skill"],
        "ok",
        "benchmark finished",
        data={"benchmark": json_safe(benchmark_result)},
        environment=collect_environment_report(effective_request, selected_device=chosen_device),
        auto_completed=auto_completed,
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_tune(request: dict[str, Any], deps: CoreDeps) -> dict[str, Any]:
    params = dict(request["params"])
    use_ray = bool(params.pop("use_ray", False))
    iterations = int(params.pop("iterations", 10))
    if request["inputs"].get("data") and "data" not in params:
        params["data"] = request["inputs"]["data"]
    if async_requested(request) and not is_dry_run(request):
        async_request = deepcopy(request)
        async_request["params"] = {"use_ray": use_ray, "iterations": iterations, **params}
        return submit_async_skill(async_request)
    if is_dry_run(request):
        return plan_response(
            request,
            "tune dry run prepared",
            "python_api",
            "YOLO(...).tune",
            params={"use_ray": use_ray, "iterations": iterations, **params},
        )

    model = deps.build_model(request)
    tuned = model.tune(use_ray=use_ray, iterations=iterations, **params)
    payload = response(request["skill"], "ok", "tuning finished", data={"tune": json_safe(tuned)})
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_lora_adapters(request: dict[str, Any], deps: CoreDeps) -> dict[str, Any]:
    action = request.get("action") or request["params"].get("action")
    if not action:
        raise ValueError("`action` is required for yolo.lora.adapters.")
    params = dict(request["params"])
    path = request["inputs"].get("path") or params.get("path")
    if is_dry_run(request):
        return plan_response(
            request,
            "LoRA adapter dry run prepared",
            "python_api",
            f"YOLO(...).lora::{action}",
            params={"path": path, **params},
        )

    model = deps.build_model(request)
    if action == "save":
        if not path:
            raise ValueError("`inputs.path` is required for adapter save.")
        ok = model.save_lora_only(path)
        payload = response(
            request["skill"],
            "ok" if ok else "failed",
            "adapter save finished" if ok else "adapter save skipped",
            artifacts=[{"kind": "adapter", "path": str(Path(path).resolve())}],
        )
    elif action == "load":
        if not path:
            raise ValueError("`inputs.path` is required for adapter load.")
        ok = model.load_lora(
            path,
            merge=bool(params.get("merge", False)),
            trainable=bool(params.get("trainable", False)),
        )
        payload = response(request["skill"], "ok" if ok else "failed", "adapter load finished")
    elif action == "merge":
        ok = model.merge_lora()
        payload = response(request["skill"], "ok" if ok else "failed", "adapter merge finished")
    else:
        raise ValueError(f"Unsupported adapter action: {action}")
    payload["manifest"] = str(write_manifest(request, payload))
    return payload
