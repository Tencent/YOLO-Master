from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from .contract import json_safe
from .progress import append_progress_event, progress_descriptor, progress_file_for_request


@dataclass(frozen=True)
class PipelineDeps:
    normalize_request: Callable[[dict[str, Any]], dict[str, Any]]
    is_dry_run: Callable[[dict[str, Any]], bool]
    response: Callable[..., dict[str, Any]]
    plan_response: Callable[..., dict[str, Any]]
    write_manifest: Callable[[dict[str, Any], dict[str, Any]], Any]
    best_checkpoint: Callable[[dict[str, Any]], str | None]
    run_system: Callable[[dict[str, Any]], dict[str, Any]]
    run_model_inspect: Callable[[dict[str, Any]], dict[str, Any]]
    run_train_like: Callable[[dict[str, Any], str], dict[str, Any]]
    run_val: Callable[[dict[str, Any]], dict[str, Any]]
    run_export: Callable[[dict[str, Any]], dict[str, Any]]
    run_benchmark: Callable[[dict[str, Any]], dict[str, Any]]
    run_lora_diagnose: Callable[[dict[str, Any]], dict[str, Any]]
    run_moe_diagnose: Callable[[dict[str, Any]], dict[str, Any]]
    run_peft_compare: Callable[[dict[str, Any]], dict[str, Any]]


DEFAULT_STAGE_ORDER = [
    "system",
    "inspect",
    "train",
    "val",
    "lora_diagnose",
    "moe_diagnose",
    "export",
    "benchmark",
    "peft_compare",
]

STAGE_SKILLS = {
    "system": "yolo.system",
    "inspect": "yolo.model.inspect",
    "train": "yolo.train",
    "val": "yolo.val",
    "lora_diagnose": "yolo.lora.diagnose",
    "moe_diagnose": "yolo.moe.diagnose",
    "export": "yolo.export",
    "benchmark": "yolo.benchmark",
    "peft_compare": "yolo.eval.peft_compare",
}


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def split_stage_config(config: Any) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], str | None]:
    if config is None:
        return {}, {}, {}, {}, None
    if not isinstance(config, dict):
        return {}, {}, {}, {}, None
    structural = {"inputs", "params", "policy", "artifacts", "skill"}
    stage_inputs = dict(config.get("inputs") or {})
    stage_params = dict(config.get("params") or {k: v for k, v in config.items() if k not in structural})
    stage_policy = dict(config.get("policy") or {})
    stage_artifacts = dict(config.get("artifacts") or {})
    return stage_inputs, stage_params, stage_policy, stage_artifacts, config.get("skill")


def requested_stages(params: dict[str, Any]) -> list[str]:
    explicit = params.get("stages")
    if isinstance(explicit, list) and explicit:
        stages = []
        for item in explicit:
            if isinstance(item, str):
                stages.append(item)
            elif isinstance(item, dict) and item.get("name"):
                stages.append(str(item["name"]))
        return stages
    return [stage for stage in DEFAULT_STAGE_ORDER if stage in params]


def stage_config(params: dict[str, Any], stage: str) -> dict[str, Any]:
    explicit = params.get("stages")
    if isinstance(explicit, list):
        for item in explicit:
            if isinstance(item, dict) and item.get("name") == stage:
                return dict(item)
    value = params.get(stage, {})
    return value if isinstance(value, dict) else {}


def pipeline_preview(request: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    preview = []
    for stage in requested_stages(params):
        _, stage_params, _, _, skill_override = split_stage_config(stage_config(params, stage))
        skill = skill_override or STAGE_SKILLS.get(stage, f"yolo.pipeline.{stage}")
        if stage == "train":
            skill = skill_override or stage_params.pop("skill", None) or ("yolo.lora.train" if _int_value(stage_params.get("lora_r")) > 0 else "yolo.train")
        preview.append({"stage": stage, "skill": skill, "params": json_safe(stage_params)})
    return preview


def stage_artifacts(request: dict[str, Any], stage: str, override: dict[str, Any]) -> dict[str, Any]:
    artifacts = dict(request.get("artifacts", {}))
    artifacts.update(override)
    base_name = artifacts.get("name")
    if base_name:
        artifacts["name"] = f"{base_name}-{stage}"
    elif artifacts.get("project"):
        artifacts["name"] = stage
    return artifacts


def build_stage_request(
    request: dict[str, Any],
    *,
    stage: str,
    skill: str,
    current_model: str | None,
    stage_inputs: dict[str, Any],
    stage_params: dict[str, Any],
    stage_policy: dict[str, Any],
    artifacts_override: dict[str, Any],
    deps: PipelineDeps,
) -> dict[str, Any]:
    inputs = {**deepcopy(request.get("inputs", {})), **stage_inputs}
    if stage not in {"system"} and current_model and not inputs.get("model"):
        inputs["model"] = current_model
    policy = {**deepcopy(request.get("policy", {})), **stage_policy}
    return deps.normalize_request(
        {
            "skill": skill,
            "runtime": deepcopy(request.get("runtime", {})),
            "inputs": inputs,
            "params": stage_params,
            "artifacts": stage_artifacts(request, stage, artifacts_override),
            "policy": policy,
            "request_id": f"{request.get('request_id')}-{stage}",
        }
    )


def execute_stage(stage: str, skill: str, stage_request: dict[str, Any], deps: PipelineDeps) -> dict[str, Any]:
    if stage == "system":
        return deps.run_system(stage_request)
    if stage == "inspect":
        return deps.run_model_inspect(stage_request)
    if stage == "train":
        return deps.run_train_like(stage_request, skill)
    if stage == "val":
        return deps.run_val(stage_request)
    if stage == "lora_diagnose":
        return deps.run_lora_diagnose(stage_request)
    if stage == "moe_diagnose":
        return deps.run_moe_diagnose(stage_request)
    if stage == "export":
        return deps.run_export(stage_request)
    if stage == "benchmark":
        return deps.run_benchmark(stage_request)
    if stage == "peft_compare":
        return deps.run_peft_compare(stage_request)
    raise ValueError(f"Unsupported pipeline stage: {stage}")


def run_experiment_pipeline(request: dict[str, Any], deps: PipelineDeps) -> dict[str, Any]:
    params = dict(request["params"])
    common_inputs = deepcopy(request["inputs"])
    current_model = common_inputs.get("model")
    stages = requested_stages(params)
    if not stages:
        raise ValueError("`params` must include at least one pipeline stage, e.g. train, val, export, or benchmark.")

    if deps.is_dry_run(request):
        preview = pipeline_preview(request, params)
        return deps.plan_response(
            request,
            "pipeline dry run prepared",
            "orchestrator",
            "yolo.pipeline.experiment",
            params={**params, "resolved_stages": preview},
            next_actions=[item["skill"] for item in preview],
            extra={"pipeline": {"stage_order": [item["stage"] for item in preview], "current_model": current_model}},
        )

    progress_path = progress_file_for_request(request)
    events = 0
    append_progress_event(progress_path, {"event": "pipeline_start", "stages": stages})
    events += 1

    stage_payloads: dict[str, Any] = {}
    artifacts: list[dict[str, Any]] = []
    failed_stage: str | None = None
    for stage in stages:
        raw_config = stage_config(params, stage)
        stage_inputs, stage_params, stage_policy, artifacts_override, skill_override = split_stage_config(raw_config)
        skill = skill_override or STAGE_SKILLS.get(stage)
        if stage == "train":
            skill = skill_override or stage_params.pop("skill", None) or ("yolo.lora.train" if _int_value(stage_params.get("lora_r")) > 0 else "yolo.train")
        if not skill:
            raise ValueError(f"Unsupported pipeline stage: {stage}")
        append_progress_event(progress_path, {"event": "stage_start", "stage": stage, "skill": skill})
        events += 1
        stage_request = build_stage_request(
            request,
            stage=stage,
            skill=skill,
            current_model=current_model,
            stage_inputs=stage_inputs,
            stage_params=stage_params,
            stage_policy=stage_policy,
            artifacts_override=artifacts_override,
            deps=deps,
        )
        payload = execute_stage(stage, skill, stage_request, deps)
        stage_payloads[stage] = payload
        artifacts.extend(payload.get("artifacts", []) or [])
        if payload.get("status") not in {"ok", "partial", "running"}:
            failed_stage = stage
            append_progress_event(progress_path, {"event": "stage_failed", "stage": stage, "status": payload.get("status")})
            events += 1
            break
        if stage == "train":
            current_model = deps.best_checkpoint(payload) or current_model
        append_progress_event(progress_path, {"event": "stage_end", "stage": stage, "status": payload.get("status")})
        events += 1

    status = "failed" if failed_stage else "ok"
    append_progress_event(progress_path, {"event": "pipeline_end", "status": status, "failed_stage": failed_stage})
    events += 1
    payload = deps.response(
        request["skill"],
        status,
        "pipeline failed" if failed_stage else "pipeline finished",
        stages=stage_payloads,
        pipeline={
            "stage_order": stages,
            "failed_stage": failed_stage,
            "current_model": current_model,
        },
        best_checkpoint=current_model,
        artifacts=artifacts,
        progress=progress_descriptor(progress_path, events),
    )
    payload["manifest"] = str(deps.write_manifest(request, payload))
    return payload
