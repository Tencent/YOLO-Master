from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from runtime.cli.contract import json_safe, plan_response, response, write_manifest
from runtime.cli.normalize import is_dry_run


@dataclass(frozen=True)
class ModelDeps:
    build_model: Callable[[dict[str, Any]], Any]


def run_model_inspect(request: dict[str, Any], deps: ModelDeps) -> dict[str, Any]:
    actions = request["params"].get("actions") or ["info", "names", "device", "task_map"]
    if is_dry_run(request):
        return plan_response(request, "inspect dry run prepared", "python_api", "YOLO(...).inspect", params={"actions": actions})

    model = deps.build_model(request)
    data: dict[str, Any] = {"task": model.task, "model_name": json_safe(getattr(model, "model_name", None))}
    for action in actions:
        if action == "info":
            data["info"] = json_safe(model.info(verbose=False))
        elif action == "names":
            data["names"] = json_safe(model.names)
        elif action == "device":
            data["device"] = str(model.device)
        elif action == "task_map":
            data["task_map"] = {k: list(v.keys()) for k, v in model.task_map.items()}
        elif action == "fuse":
            model.fuse()
            data["fused"] = True
        elif action == "reset_weights":
            model.reset_weights()
            data["reset_weights"] = True
        else:
            raise ValueError(f"Unsupported inspect action: {action}")
    try:
        model._check_is_pytorch_model()
        data["supports_pytorch_only"] = True
    except Exception:
        data["supports_pytorch_only"] = False
    payload = response(request["skill"], "ok", "model inspected", data=data)
    payload["manifest"] = str(write_manifest(request, payload))
    return payload
