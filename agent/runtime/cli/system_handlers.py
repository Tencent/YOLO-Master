from __future__ import annotations

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from runtime.cli.contract import ensure_manifest_dir, json_safe, plan_response, response
from runtime.cli.device import (
    collect_environment_report,
    doctor_recommendations,
    read_default_cfg,
    read_repo_version,
    resolve_default_device,
)
from runtime.cli.executor import (
    cli_install_command,
    cli_logs,
    cli_plan,
    ensure_cli_success,
    ensure_yolo_cli,
    find_yolo_cli,
    install_ultralytics_cli,
    kv_arg,
)
from runtime.cli.normalize import is_dry_run

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CFG_FILE = REPO_ROOT / "ultralytics" / "cfg" / "default.yaml"
MODULE_CACHE: dict[str, Any] = {}


@dataclass(frozen=True)
class SystemDeps:
    run_cli: Callable[..., dict[str, Any]]
    get_ultralytics_core: Callable[[], dict[str, Any]]
    default_cfg_file: Path = DEFAULT_CFG_FILE


def cached(name: str, loader: Callable[[], Any]) -> Any:
    if name not in MODULE_CACHE:
        MODULE_CACHE[name] = loader()
    return MODULE_CACHE[name]


def get_checks_helpers() -> dict[str, Any]:
    def _loader():
        checks = importlib.import_module("ultralytics.utils.checks")
        return {"collect_system_info": checks.collect_system_info}

    return cached("checks_helpers", _loader)


def run_system(request: dict[str, Any], deps: SystemDeps) -> dict[str, Any]:
    action = request.get("action") or request["params"].get("action") or "help"
    params = request["params"]

    if is_dry_run(request):
        if action == "install":
            return plan_response(
                request,
                "system install dry run prepared",
                "bootstrap",
                "pip install -e .",
                params={"cmd": cli_install_command()},
            )
        if action == "doctor":
            selected_device = resolve_default_device(request, params)
            environment = collect_environment_report(request, selected_device=selected_device)
            recommendations = doctor_recommendations(environment)
            return plan_response(
                request,
                "system doctor dry run prepared",
                "module",
                "yolo.system::doctor",
                params=params,
                extra={"environment": environment, "recommendations": recommendations},
            )
        cli_map = {
            "help": ["help"],
            "version": ["version"],
            "checks": ["checks"],
            "settings.get": ["settings"],
            "settings.update": [
                "settings",
                *[kv_arg(k, v) for k, v in (params.get("updates") or {k: v for k, v in params.items() if k != "action"}).items()],
            ],
            "settings.reset": ["settings", "reset"],
            "cfg.get": ["cfg"],
            "cfg.copy": ["copy-cfg"],
        }
        if action in cli_map:
            return cli_plan(request, cli_map[action])
        return plan_response(request, "system dry run prepared", "module", f"yolo.system::{action}", params=params)

    if action == "install":
        install = install_ultralytics_cli()
        return response(
            request["skill"],
            "ok" if install["returncode"] == 0 else "failed",
            "ultralytics CLI installed" if install["returncode"] == 0 else "ultralytics CLI install failed",
            data={"install": install, "yolo": find_yolo_cli()},
        )

    if action == "doctor":
        force_install = bool(params.get("force_install", False))
        ensure_cli = bool(params.get("ensure_cli", True))
        if ensure_cli:
            _, install = ensure_yolo_cli(force_install=force_install)
        else:
            cli_path = find_yolo_cli()
            install = {"status": "available" if cli_path else "missing", "path": cli_path}
        selected_device = resolve_default_device(request, params)
        environment = collect_environment_report(request, selected_device=selected_device, cli_info=install)
        recommendations = doctor_recommendations(environment)
        return response(
            request["skill"],
            "ok",
            "environment doctor collected",
            data={"environment": environment, "recommendations": recommendations},
            environment=environment,
            recommendations=recommendations,
        )

    if action in {"help", "version", "checks", "settings.get", "settings.update", "settings.reset", "cfg.get", "cfg.copy"}:
        cli_args = {
            "help": ["help"],
            "version": ["version"],
            "checks": ["checks"],
            "settings.get": ["settings"],
            "settings.update": [
                "settings",
                *[kv_arg(k, v) for k, v in (params.get("updates") or {k: v for k, v in params.items() if k != "action"}).items()],
            ],
            "settings.reset": ["settings", "reset"],
            "cfg.get": ["cfg"],
            "cfg.copy": ["copy-cfg"],
        }[action]
        cwd = ensure_manifest_dir(request) if action == "cfg.copy" else None
        cli_result = deps.run_cli(cli_args, cwd=cwd)
        failed = ensure_cli_success(request, cli_result, f"system action `{action}` failed")
        if failed:
            return failed
        if action == "help":
            return response(
                request["skill"],
                "ok",
                "available system actions",
                actions=[
                    "install",
                    "doctor",
                    "help",
                    "version",
                    "checks",
                    "settings.get",
                    "settings.update",
                    "settings.reset",
                    "cfg.get",
                    "cfg.copy",
                ],
                logs=cli_logs(cli_result),
            )
        if action == "version":
            match = re.search(r"\b\d+\.\d+\.\d+\b", f"{cli_result['stdout']}\n{cli_result['stderr']}")
            version = match.group(0) if match else read_repo_version()
            return response(request["skill"], "ok", "version collected", data={"version": version}, logs=cli_logs(cli_result))
        if action == "checks":
            return response(request["skill"], "ok", "system checks collected", logs=cli_logs(cli_result))
        if action == "settings.get":
            core = deps.get_ultralytics_core()
            return response(
                request["skill"],
                "ok",
                "settings collected",
                data={"settings": json_safe(dict(core["SETTINGS"]))},
                logs=cli_logs(cli_result),
            )
        if action == "settings.update":
            core = deps.get_ultralytics_core()
            updates = params.get("updates") or {k: v for k, v in params.items() if k != "action"}
            return response(
                request["skill"],
                "ok",
                "settings updated",
                data={"settings": json_safe(dict(core["SETTINGS"])), "updated": json_safe(updates)},
                logs=cli_logs(cli_result),
            )
        if action == "settings.reset":
            core = deps.get_ultralytics_core()
            return response(request["skill"], "ok", "settings reset", data={"settings": json_safe(dict(core["SETTINGS"]))}, logs=cli_logs(cli_result))
        if action == "cfg.get":
            return response(request["skill"], "ok", "default cfg loaded", data={"cfg": json_safe(read_default_cfg())}, logs=cli_logs(cli_result))
        if action == "cfg.copy":
            new_file = ensure_manifest_dir(request) / deps.default_cfg_file.name.replace(".yaml", "_copy.yaml")
            return response(
                request["skill"],
                "ok",
                "default cfg copied",
                artifacts=[{"kind": "config", "path": str(new_file.resolve())}],
                logs=cli_logs(cli_result),
            )
    raise ValueError(f"Unsupported yolo.system action: {action}")
