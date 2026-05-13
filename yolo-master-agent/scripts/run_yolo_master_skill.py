#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import contextlib
import importlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4


SKILL_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = SKILL_ROOT / "logs"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "runs" / "agent"
MODULE_CACHE: dict[str, Any] = {}
ULTRALYTICS_INIT = REPO_ROOT / "ultralytics" / "__init__.py"
DEFAULT_CFG_FILE = REPO_ROOT / "ultralytics" / "cfg" / "default.yaml"
DATASET_CFG_DIR = REPO_ROOT / "ultralytics" / "cfg" / "datasets"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUNTIME_CACHE_FILE = LOG_DIR / "runtime-cache.json"
RUNTIME_CACHE_TTL_SEC = 600

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.chdir(REPO_ROOT)


def cached(name: str, loader):
    if name not in MODULE_CACHE:
        MODULE_CACHE[name] = loader()
    return MODULE_CACHE[name]


def get_ultralytics_core() -> dict[str, Any]:
    def _loader():
        ultralytics = importlib.import_module("ultralytics")
        utils = importlib.import_module("ultralytics.utils")
        return {
            "YOLO": ultralytics.YOLO,
            "version": ultralytics.__version__,
            "SETTINGS": utils.SETTINGS,
            "SETTINGS_FILE": utils.SETTINGS_FILE,
            "YAML": utils.YAML,
        }

    return cached("ultralytics_core", _loader)


def get_cfg_helpers() -> dict[str, Any]:
    def _loader():
        cfg = importlib.import_module("ultralytics.cfg")
        return {
            "DEFAULT_CFG_PATH": cfg.DEFAULT_CFG_PATH,
            "copy_default_cfg": cfg.copy_default_cfg,
            "handle_yolo_solutions": cfg.handle_yolo_solutions,
        }

    return cached("cfg_helpers", _loader)


def get_checks_helpers() -> dict[str, Any]:
    def _loader():
        checks = importlib.import_module("ultralytics.utils.checks")
        return {"collect_system_info": checks.collect_system_info}

    return cached("checks_helpers", _loader)


def get_moe_helpers() -> dict[str, Any]:
    def _loader():
        analysis = importlib.import_module("ultralytics.nn.modules.moe.analysis")
        pruning = importlib.import_module("ultralytics.nn.modules.moe.pruning")
        return {
            "diagnose_model": analysis.diagnose_model,
            "prune_moe_model": pruning.prune_moe_model,
        }

    return cached("moe_helpers", _loader)


def get_ultralytics_module_info() -> dict[str, Any]:
    def _loader():
        spec = importlib.util.find_spec("ultralytics")
        module_path = Path(spec.origin).resolve() if spec and spec.origin else ULTRALYTICS_INIT.resolve()
        return {
            "path": str(module_path),
            "version": read_repo_version(),
            "local_repo_active": REPO_ROOT in module_path.parents or module_path == ULTRALYTICS_INIT.resolve(),
        }

    return cached("ultralytics_module_info", _loader)


def runtime_cache_enabled() -> bool:
    return os.environ.get("YOLO_MASTER_AGENT_RUNTIME_CACHE", "").lower() in {"1", "true", "yes"}


def read_torch_runtime_cache() -> dict[str, Any] | None:
    if not runtime_cache_enabled() or not RUNTIME_CACHE_FILE.exists():
        return None
    try:
        if time.time() - RUNTIME_CACHE_FILE.stat().st_mtime > RUNTIME_CACHE_TTL_SEC:
            return None
        payload = json.loads(RUNTIME_CACHE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if payload.get("python") != sys.executable or payload.get("platform") != sys.platform:
        return None
    data = payload.get("torch")
    return data if isinstance(data, dict) else None


def write_torch_runtime_cache(info: dict[str, Any]) -> None:
    if not runtime_cache_enabled():
        return
    try:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        RUNTIME_CACHE_FILE.write_text(
            json.dumps(
                {"python": sys.executable, "platform": sys.platform, "torch": json_safe(info)},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass


def get_torch_runtime() -> dict[str, Any]:
    def _loader():
        cached_info = read_torch_runtime_cache()
        if cached_info is not None:
            return cached_info
        info: dict[str, Any] = {
            "installed": False,
            "version": None,
            "cuda": {"available": False, "device_count": 0, "devices": []},
            "mps": {"built": False, "available": False},
        }
        try:
            torch = importlib.import_module("torch")
        except Exception:
            write_torch_runtime_cache(info)
            return info

        info["installed"] = True
        info["version"] = getattr(torch, "__version__", None)
        try:
            cuda_available = bool(torch.cuda.is_available())
        except Exception:
            cuda_available = False
        cuda_devices = []
        cuda_count = 0
        if cuda_available:
            try:
                cuda_count = int(torch.cuda.device_count())
                cuda_devices = [torch.cuda.get_device_name(i) for i in range(cuda_count)]
            except Exception:
                cuda_count = 0
                cuda_devices = []
        info["cuda"] = {"available": cuda_available, "device_count": cuda_count, "devices": cuda_devices}
        try:
            info["mps"] = {
                "built": bool(torch.backends.mps.is_built()),
                "available": bool(torch.backends.mps.is_available()),
            }
        except Exception:
            info["mps"] = {"built": False, "available": False}
        write_torch_runtime_cache(info)
        return info

    return cached("torch_runtime", _loader)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "results_dict"):
        return json_safe(value.results_dict)
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_repo_version() -> str:
    text = ULTRALYTICS_INIT.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError(f"Could not parse version from {ULTRALYTICS_INIT}")
    return match.group(1)


def read_default_cfg() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(DEFAULT_CFG_FILE.read_text(encoding="utf-8"))


def path_like(value: str) -> bool:
    return any(token in value for token in ("/", "\\", ".")) and not value.startswith(("http://", "https://"))


def normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: normalize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize_value(v) for v in value]
    if isinstance(value, str) and path_like(value):
        p = Path(value)
        if not p.is_absolute():
            candidate = (REPO_ROOT / p).resolve()
            if candidate.exists() or value.startswith((".", "/")) or "/" in value:
                return str(candidate)
    if isinstance(value, str):
        builtin_dataset = (DATASET_CFG_DIR / value).resolve()
        if builtin_dataset.exists():
            return str(builtin_dataset)
    return value


def normalize_request(request: dict[str, Any]) -> dict[str, Any]:
    request = deepcopy(request)
    request.setdefault("workspace_root", str(REPO_ROOT))
    request.setdefault("request_id", default_request_id(request.get("skill", "skill")))
    request.setdefault("runtime", {})
    request.setdefault("inputs", {})
    request.setdefault("params", {})
    request.setdefault("artifacts", {})
    request.setdefault("policy", {})
    request["inputs"] = normalize_value(request["inputs"])
    request["params"] = normalize_value(request["params"])
    request["artifacts"] = normalize_value(request["artifacts"])
    return request


def is_dry_run(request: dict[str, Any]) -> bool:
    return bool(request.get("policy", {}).get("dry_run", False))


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-").lower()
    return slug or "skill"


def default_request_id(skill: str) -> str:
    return f"{slugify(skill)}-{uuid4().hex[:8]}"


def prefer_cli(request: dict[str, Any]) -> bool:
    runtime = request.get("runtime", {})
    if runtime.get("prefer_python_api"):
        return False
    return runtime.get("prefer_cli", True)


def mps_available() -> bool:
    return bool(get_torch_runtime()["mps"]["available"])


def available_devices() -> list[str]:
    torch_info = get_torch_runtime()
    devices = ["cpu"]
    if torch_info["mps"]["available"]:
        devices.insert(0, "mps")
    if torch_info["cuda"]["available"] and torch_info["cuda"]["device_count"] > 0:
        devices.insert(0, "cuda:0")
    return devices


def default_auto_device(request: dict[str, Any]) -> str | None:
    runtime = request.get("runtime", {})
    if not runtime.get("auto_detect_device", True):
        return None
    devices = available_devices()
    if runtime.get("prefer_mps", True) and "mps" in devices and sys.platform == "darwin":
        return "mps"
    if runtime.get("prefer_cuda", True) and "cuda:0" in devices:
        return "cuda:0"
    if runtime.get("prefer_mps", True) and "mps" in devices:
        return "mps"
    return devices[0] if devices else None


def resolve_device_selection(request: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    explicit = params.get("device")
    if explicit not in (None, "", "auto"):
        return {"device": str(explicit), "source": "params"}
    runtime = request.get("runtime", {})
    runtime_device = runtime.get("device")
    if runtime_device not in (None, "", "auto"):
        return {"device": str(runtime_device), "source": "runtime"}
    auto_device = default_auto_device(request)
    return {"device": auto_device, "source": "auto" if auto_device else None}


def resolve_default_device(request: dict[str, Any], params: dict[str, Any]) -> str | None:
    return resolve_device_selection(request, params)["device"]


def reference_state(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {"requested": value, "resolved": value, "exists": False}
    resolved = normalize_value(value)
    state = {"requested": value, "resolved": resolved}
    if isinstance(resolved, str) and path_like(resolved):
        state["exists"] = Path(resolved).exists()
    else:
        state["exists"] = bool(resolved)
    return state


def collect_environment_report(
    request: dict[str, Any],
    *,
    selected_device: str | None = None,
    requested_device: str | None = None,
    selection_source: str | None = None,
    cli_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    runtime = request.get("runtime", {})
    torch_info = get_torch_runtime()
    module_info = get_ultralytics_module_info()
    cli_path = None
    cli_status = "missing"
    install = None
    if cli_info:
        cli_path = cli_info.get("path")
        cli_status = cli_info.get("status", cli_status)
        install = cli_info
    else:
        cli_path = find_yolo_cli()
        cli_status = "available" if cli_path else "missing"
    selection = resolve_device_selection(request, request.get("params", {}))
    requested = requested_device if requested_device is not None else selection["device"]
    selected = selected_device if selected_device is not None else requested
    source = selection_source if selection_source is not None else selection["source"]
    report = {
        "python": {"executable": sys.executable, "version": sys.version.split()[0]},
        "workspace": {"repo_root": str(REPO_ROOT), "cwd": str(Path.cwd())},
        "ultralytics": {
            "repo_version": read_repo_version(),
            "module_version": module_info["version"],
            "module_path": module_info["path"],
            "local_repo_active": module_info["local_repo_active"],
        },
        "cli": {
            "available": bool(cli_path),
            "path": cli_path,
            "status": cli_status,
            "install": install,
        },
        "devices": {
            "requested": requested,
            "selected": selected,
            "available": available_devices(),
            "selection_source": source,
            "torch": torch_info,
        },
        "runtime": json_safe(runtime),
        "references": {
            "model": reference_state(request.get("inputs", {}).get("model")),
            "data": reference_state(request.get("inputs", {}).get("data") or request.get("params", {}).get("data")),
            "source": reference_state(request.get("inputs", {}).get("source")),
        },
    }
    return report


def doctor_recommendations(environment: dict[str, Any]) -> list[str]:
    recommendations: list[str] = []
    cli = environment.get("cli", {})
    ultralytics = environment.get("ultralytics", {})
    devices = environment.get("devices", {})
    torch_info = devices.get("torch", {})

    if not cli.get("available"):
        recommendations.append("Run `python -m pip install -e .` to provision the local `yolo` CLI.")
    if not ultralytics.get("local_repo_active"):
        recommendations.append("Refresh the editable install with `python -m pip install -e .` so imports resolve to this repo.")
    if not torch_info.get("installed"):
        recommendations.append("Install PyTorch in the current environment before running train, val, benchmark, or predict.")
    if sys.platform == "darwin" and torch_info.get("mps", {}).get("available") and devices.get("selected") != "mps":
        recommendations.append("This host supports MPS; leave `device` unset or set `runtime.prefer_mps=true` for Apple Silicon acceleration.")
    if devices.get("selected") == "cpu" and sys.platform == "darwin" and not torch_info.get("mps", {}).get("available"):
        recommendations.append("MPS is unavailable, so heavy runs will stay on CPU until the PyTorch MPS runtime is available.")

    for label, state in (environment.get("references") or {}).items():
        if state.get("requested") not in (None, "") and not state.get("exists"):
            recommendations.append(f"Fix the `{label}` reference before launch: {state['requested']}")

    return recommendations


def apply_runtime_defaults(
    request: dict[str, Any],
    params: dict[str, Any],
    *,
    purpose: str,
) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    params = dict(params)
    auto_completed: dict[str, Any] = {}
    device_selection = resolve_device_selection(request, params)
    device = device_selection["device"]
    if device and "device" not in params:
        params["device"] = device
        auto_completed["device"] = device
        auto_completed["device_source"] = device_selection["source"]
    runtime = request.get("runtime", {})
    if purpose in {"train", "val", "benchmark"} and "workers" not in params and sys.platform == "darwin":
        params["workers"] = int(runtime.get("default_workers", 0))
        auto_completed["workers"] = params["workers"]
    return params, device, auto_completed


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text or "")


def cli_install_command() -> list[str]:
    return [sys.executable, "-m", "pip", "install", "-e", str(REPO_ROOT)]


def install_ultralytics_cli() -> dict[str, Any]:
    cmd = cli_install_command()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    return {
        "cmd": cmd,
        "returncode": proc.returncode,
        "stdout": strip_ansi(proc.stdout),
        "stderr": strip_ansi(proc.stderr),
    }


def find_yolo_cli() -> str | None:
    names = ("yolo", "yolo.exe", "yolo-script.py")
    candidates: list[Path] = []
    located = shutil.which("yolo")
    if located:
        candidates.append(Path(located))
    scripts_dir = sysconfig.get_path("scripts")
    if scripts_dir:
        base = Path(scripts_dir)
        candidates.extend(base / name for name in names)
    py_bin = Path(sys.executable).resolve().parent
    candidates.extend(py_bin / name for name in names)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists() and os.access(resolved, os.X_OK):
            return str(resolved)
    return None


def ensure_yolo_cli(force_install: bool = False) -> tuple[str, dict[str, Any]]:
    yolo_path = find_yolo_cli()
    if yolo_path and not force_install:
        return yolo_path, {"status": "available", "path": yolo_path}

    install = install_ultralytics_cli()
    yolo_path = find_yolo_cli()
    if install["returncode"] != 0 or not yolo_path:
        raise RuntimeError(
            "Failed to provision the `yolo` CLI via editable Ultralytics install.\n"
            f"cmd={install['cmd']}\nstdout={install['stdout']}\nstderr={install['stderr']}"
        )
    install["status"] = "installed"
    install["path"] = yolo_path
    return yolo_path, install


def cli_value(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, str):
        return value
    if value is None:
        return "None"
    return repr(value)


def kv_arg(key: str, value: Any) -> str:
    return f"{key}={cli_value(value)}"


def resolved_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def repo_cli_env() -> dict[str, str]:
    env = os.environ.copy()
    current = env.get("PYTHONPATH", "")
    prefix = str(REPO_ROOT)
    env["PYTHONPATH"] = prefix if not current else f"{prefix}{os.pathsep}{current}"
    return env


def cli_save_dir(request: dict[str, Any], params: dict[str, Any]) -> Path | None:
    project = params.get("project")
    name = params.get("name")
    if project and name:
        return resolved_path(str(project)) / str(name)
    if project:
        return resolved_path(str(project))
    return None


def inject_cli_artifact_location(request: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(params)
    if "project" not in enriched:
        enriched["project"] = str(DEFAULT_MANIFEST_DIR)
    if "name" not in enriched:
        enriched["name"] = request["request_id"]
    return enriched


def run_cli(args: list[str], cwd: Path | None = None, force_install: bool = False) -> dict[str, Any]:
    yolo_path, install = ensure_yolo_cli(force_install=force_install)
    cmd = [yolo_path, *args]
    proc = subprocess.run(cmd, cwd=cwd or REPO_ROOT, capture_output=True, text=True, env=repo_cli_env())
    return {
        "cmd": cmd,
        "cwd": str((cwd or REPO_ROOT).resolve()),
        "returncode": proc.returncode,
        "stdout": strip_ansi(proc.stdout),
        "stderr": strip_ansi(proc.stderr),
        "install": install,
    }


def cli_logs(cli_result: dict[str, Any]) -> dict[str, Any]:
    return {
        "cmd": cli_result["cmd"],
        "cwd": cli_result["cwd"],
        "stdout": cli_result["stdout"],
        "stderr": cli_result["stderr"],
        "install": cli_result["install"],
    }


def cli_plan(
    request: dict[str, Any],
    args: list[str],
    cwd: Path | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    merged_extra = {"bootstrap": {"install_if_missing": cli_install_command()}}
    if extra:
        merged_extra.update(json_safe(extra))
    return plan_response(
        request,
        f"{request['skill']} CLI dry run prepared",
        "cli",
        "yolo",
        params={"cmd": ["yolo", *args], "cwd": str((cwd or REPO_ROOT).resolve())},
        extra=merged_extra,
    )


def coerce_scalar(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return text
    if text.lower() in {"true", "false"}:
        return text.lower() == "true"
    if re.fullmatch(r"-?\d+", text):
        try:
            return int(text)
        except Exception:
            return text
    try:
        return float(text)
    except Exception:
        return text


def detect_cli_device(cli_result: dict[str, Any]) -> str | None:
    for arg in cli_result.get("cmd", [])[1:]:
        if isinstance(arg, str) and arg.startswith("device="):
            return arg.split("=", 1)[1]
    return None


def detect_missing_module(text: str) -> str | None:
    patterns = [
        re.compile(r"No module named ['\"]([^'\"]+)['\"]"),
        re.compile(r"No module named ([A-Za-z0-9_.-]+)"),
    ]
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            return match.group(1)
    return None


def classify_cli_failure(cli_result: dict[str, Any]) -> dict[str, Any]:
    text = f"{cli_result.get('stdout', '')}\n{cli_result.get('stderr', '')}"
    info: dict[str, Any] = {
        "type": "CalledProcessError",
        "returncode": cli_result["returncode"],
        "device": detect_cli_device(cli_result),
    }
    hints: list[str] = []
    category = "cli_runtime_error"

    if "Expected more than 1 value per channel when training" in text:
        category = "batchnorm_small_feature_map"
        hints = [
            "Increase `imgsz` so the deepest feature maps do not collapse to 1x1 during training.",
            "Keep `batch` above 1 when possible, or reduce stride pressure by using a larger image size.",
        ]
    elif "CUDA out of memory" in text or "CUDA error" in text or "not compiled with CUDA" in text:
        category = "cuda_runtime_error"
        hints = [
            "Retry with a smaller `batch` or `imgsz` on CUDA, or let the agent fall back to `cpu` when the device was auto-selected.",
            "Check that the selected CUDA runtime matches the current PyTorch build.",
        ]
    elif "not implemented for 'MPS'" in text or "MPS backend out of memory" in text or "MPS" in text and "not supported" in text:
        category = "mps_runtime_error"
        hints = [
            "Retry with a smaller `batch` or `imgsz` while keeping `device=mps`.",
            "If the operator is unsupported on MPS, override with `runtime.device=cpu` for confirmation.",
        ]
    elif "No module named" in text:
        category = "missing_dependency"
        missing_module = detect_missing_module(text)
        if missing_module:
            info["missing_module"] = missing_module
            hints = [
                f"Install the missing dependency in the current environment, for example `python -m pip install {missing_module}`.",
                "If the import should resolve from this repo, refresh the editable install with `python -m pip install -e .`.",
            ]
        else:
            hints = [
                "Install the missing dependency inside the current Python environment before retrying.",
            ]
    elif "Dataset" in text and "not found" in text:
        category = "dataset_not_found"
        hints = [
            "Verify that the dataset YAML resolves from the current workspace and that auto-download is allowed.",
        ]

    info["category"] = category
    if hints:
        info["hints"] = hints
    return info


def tail_text(text: str, max_lines: int = 20) -> str:
    lines = strip_ansi(text).splitlines()
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def cli_attempt_record(cli_result: dict[str, Any]) -> dict[str, Any]:
    record: dict[str, Any] = {
        "cmd": cli_result["cmd"],
        "cwd": cli_result["cwd"],
        "returncode": cli_result["returncode"],
        "device": detect_cli_device(cli_result),
    }
    if cli_result["returncode"] != 0:
        record["error"] = classify_cli_failure(cli_result)
        if cli_result.get("stdout"):
            record["stdout_tail"] = tail_text(cli_result["stdout"])
        if cli_result.get("stderr"):
            record["stderr_tail"] = tail_text(cli_result["stderr"])
    return json_safe(record)


def replace_cli_device(values: dict[str, Any], device: str) -> dict[str, Any]:
    updated = dict(values)
    updated["device"] = device
    return updated


def should_retry_with_cpu(
    request: dict[str, Any],
    cli_result: dict[str, Any],
    *,
    selected_device: str | None,
    selection_source: str | None,
) -> bool:
    if cli_result["returncode"] == 0:
        return False
    if selection_source != "auto" or selected_device in (None, "cpu"):
        return False
    if not request.get("runtime", {}).get("allow_device_fallback", True):
        return False
    error = classify_cli_failure(cli_result)
    return error.get("category") in {"mps_runtime_error", "cuda_runtime_error"}


def run_cli_with_recovery(
    request: dict[str, Any],
    mode: str,
    values: dict[str, Any],
    *,
    failure_summary: str,
    selected_device: str | None,
    selection_source: str | None,
) -> dict[str, Any]:
    cli_result = run_cli(cli_args_from_values(mode, values))
    attempts = [cli_attempt_record(cli_result)]
    recovery: dict[str, Any] | None = None
    final_values = dict(values)
    final_device = detect_cli_device(cli_result) or selected_device

    if should_retry_with_cpu(
        request,
        cli_result,
        selected_device=selected_device,
        selection_source=selection_source,
    ):
        first_error = classify_cli_failure(cli_result)
        recovery = {
            "attempted": True,
            "strategy": "device_fallback_to_cpu",
            "from_device": final_device,
            "to_device": "cpu",
            "trigger": first_error,
        }
        final_values = replace_cli_device(values, "cpu")
        cli_result = run_cli(cli_args_from_values(mode, final_values))
        attempts.append(cli_attempt_record(cli_result))
        final_device = "cpu"
        recovery["recovered"] = cli_result["returncode"] == 0
        recovery["status"] = "recovered" if recovery["recovered"] else "fallback_failed"

    failed = ensure_cli_success(
        request,
        cli_result,
        failure_summary,
        attempts=attempts,
        recovery=recovery,
    )
    return {
        "cli_result": cli_result,
        "values": final_values,
        "device": final_device,
        "attempts": attempts,
        "recovery": recovery,
        "failed": failed,
    }


def ensure_cli_success(
    request: dict[str, Any],
    cli_result: dict[str, Any],
    summary: str,
    *,
    attempts: list[dict[str, Any]] | None = None,
    recovery: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    if cli_result["returncode"] == 0:
        return None
    payload = response(
        request["skill"],
        "failed",
        summary,
        logs=cli_logs(cli_result),
        error=classify_cli_failure(cli_result),
    )
    if attempts:
        payload["attempts"] = attempts
    if recovery:
        payload["recovery"] = recovery
    return payload


def build_cli_key_values(
    request: dict[str, Any],
    *,
    skip_inputs: set[str] | None = None,
    skip_params: set[str] | None = None,
    inject_save_dir: bool = False,
) -> dict[str, Any]:
    skip_inputs = skip_inputs or set()
    skip_params = skip_params or set()
    values: dict[str, Any] = {}
    for key, value in request["inputs"].items():
        if key in skip_inputs or value is None:
            continue
        values[key] = value
    for key, value in request["params"].items():
        if key in skip_params or value is None:
            continue
        values[key] = value
    if inject_save_dir:
        values = inject_cli_artifact_location(request, values)
    return values


def cli_args_from_values(mode: str, values: dict[str, Any]) -> list[str]:
    args = [mode]
    for key, value in values.items():
        args.append(kv_arg(key, value))
    return args


def read_results_csv_metrics(save_dir: Path | None) -> dict[str, Any]:
    if not save_dir:
        return {}
    results_csv = Path(save_dir) / "results.csv"
    if not results_csv.exists():
        return {}
    try:
        with results_csv.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except Exception:
        return {}
    if not rows:
        return {}
    return {key: coerce_scalar(value) for key, value in rows[-1].items()}


def parse_cli_speed(stdout: str) -> dict[str, float]:
    speed: dict[str, float] = {}
    pattern = re.compile(
        r"^Speed:\s+([\d.]+)ms preprocess,\s+([\d.]+)ms inference(?:,\s+([\d.]+)ms loss)?,\s+([\d.]+)ms postprocess"
    )
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        match = pattern.match(line)
        if not match:
            continue
        speed = {
            "preprocess": float(match.group(1)),
            "inference": float(match.group(2)),
            "postprocess": float(match.group(4)),
        }
        if match.group(3) is not None:
            speed["loss"] = float(match.group(3))
    return speed


def parse_detection_cli_metrics(stdout: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pattern = re.compile(r"^all\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s*$")
    for raw_line in reversed(stdout.splitlines()):
        line = raw_line.strip()
        match = pattern.match(line)
        if not match:
            continue
        images = int(match.group(1))
        instances = int(match.group(2))
        precision = float(match.group(3))
        recall = float(match.group(4))
        map50 = float(match.group(5))
        map50_95 = float(match.group(6))
        raw_metrics = {
            "metrics/precision(B)": precision,
            "metrics/recall(B)": recall,
            "metrics/mAP50(B)": map50,
            "metrics/mAP50-95(B)": map50_95,
        }
        evaluation = {
            "images": images,
            "instances": instances,
            "precision": precision,
            "recall": recall,
            "map50": map50,
            "map50_95": map50_95,
        }
        return raw_metrics, evaluation
    return {}, {}


def build_evaluation_summary(metrics: dict[str, Any], stdout: str = "") -> dict[str, Any]:
    _, parsed = parse_detection_cli_metrics(stdout)
    evaluation = dict(parsed)
    mapping = {
        "metrics/precision(B)": "precision",
        "metrics/recall(B)": "recall",
        "metrics/mAP50(B)": "map50",
        "metrics/mAP50-95(B)": "map50_95",
        "train/box_loss": "train_box_loss",
        "train/cls_loss": "train_cls_loss",
        "train/dfl_loss": "train_dfl_loss",
        "train/moe_loss": "train_moe_loss",
        "val/box_loss": "val_box_loss",
        "val/cls_loss": "val_cls_loss",
        "val/dfl_loss": "val_dfl_loss",
        "val/moe_loss": "val_moe_loss",
        "epoch": "epoch",
        "time": "time_sec",
    }
    for source_key, target_key in mapping.items():
        if source_key in metrics:
            evaluation[target_key] = metrics[source_key]
    speed = parse_cli_speed(stdout)
    if speed:
        evaluation["speed_ms"] = speed
    return evaluation


def parse_predict_cli_output(stdout: str) -> tuple[list[dict[str, Any]], dict[str, float]]:
    results: list[dict[str, Any]] = []
    speed: dict[str, float] = parse_cli_speed(stdout)
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        match = re.match(r"^image\s+\d+/\d+\s+(.*?):\s+(.*)$", line)
        if match:
            item: dict[str, Any] = {"path": match.group(1), "raw": match.group(2)}
            if "no detections" in match.group(2).lower():
                item["boxes"] = 0
            results.append(item)
    return results, speed


def capture_output(func, *args, **kwargs) -> tuple[Any, str, str]:
    stdout_buffer, stderr_buffer = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = func(*args, **kwargs)
    return result, stdout_buffer.getvalue(), stderr_buffer.getvalue()


@contextlib.contextmanager
def pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def ensure_manifest_dir(request: dict[str, Any]) -> Path:
    project = request.get("artifacts", {}).get("project")
    name = request.get("artifacts", {}).get("name")
    base = (REPO_ROOT / project).resolve() if project else DEFAULT_MANIFEST_DIR / request["request_id"]
    target = base / name if name else base
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_manifest(request: dict[str, Any], payload: dict[str, Any]) -> Path:
    manifest_dir = ensure_manifest_dir(request)
    manifest_path = manifest_dir / "skill_manifest.json"
    manifest = {
        "skill": request.get("skill"),
        "request_id": request.get("request_id"),
        "status": payload.get("status"),
        "summary": payload.get("summary"),
        "artifacts": payload.get("artifacts", []),
        "metrics": payload.get("metrics", {}),
        "evaluation": payload.get("evaluation", {}),
        "environment": payload.get("environment", {}),
        "auto_completed": payload.get("auto_completed", {}),
        "attempts": payload.get("attempts", []),
        "recovery": payload.get("recovery", {}),
        "recommendations": payload.get("recommendations", []),
        "job": payload.get("job", {}),
        "dry_run": payload.get("dry_run", False),
    }
    manifest_path.write_text(json.dumps(json_safe(manifest), ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def finalize_payload(request: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    payload = dict(payload)
    payload.setdefault("request_id", request.get("request_id"))
    if "manifest" not in payload:
        payload["manifest"] = str(write_manifest(request, payload))
    return json_safe(payload)


def best_checkpoint(payload: dict[str, Any]) -> str | None:
    for artifact in payload.get("artifacts", []):
        if artifact.get("kind") == "checkpoint" and artifact.get("label") == "best":
            return artifact.get("path")
    for artifact in payload.get("artifacts", []):
        if artifact.get("kind") == "checkpoint":
            return artifact.get("path")
    return None


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


def response(skill: str, status: str, summary: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"skill": skill, "status": status, "summary": summary}
    payload.update(kwargs)
    return json_safe(payload)


def plan_response(
    request: dict[str, Any],
    summary: str,
    executor: str,
    target: str,
    params: dict[str, Any] | None = None,
    next_actions: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = response(
        request["skill"],
        "ok",
        summary,
        dry_run=True,
        plan={
            "executor": executor,
            "target": target,
            "inputs": json_safe(request.get("inputs", {})),
            "params": json_safe(params if params is not None else request.get("params", {})),
        },
        next_actions=next_actions or [],
    )
    if extra:
        payload.update(json_safe(extra))
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def build_model(request: dict[str, Any]) -> Any:
    inputs = request["inputs"]
    model_ref = inputs.get("model")
    if not model_ref:
        raise ValueError("`inputs.model` is required.")
    YOLO = get_ultralytics_core()["YOLO"]
    return YOLO(model_ref, task=inputs.get("task"))


def run_system(request: dict[str, Any]) -> dict[str, Any]:
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
            "settings.update": ["settings", *[kv_arg(k, v) for k, v in (params.get("updates") or {k: v for k, v in params.items() if k != "action"}).items()]],
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
            "settings.update": ["settings", *[kv_arg(k, v) for k, v in (params.get("updates") or {k: v for k, v in params.items() if k != "action"}).items()]],
            "settings.reset": ["settings", "reset"],
            "cfg.get": ["cfg"],
            "cfg.copy": ["copy-cfg"],
        }[action]
        cwd = ensure_manifest_dir(request) if action == "cfg.copy" else None
        cli_result = run_cli(cli_args, cwd=cwd)
        failed = ensure_cli_success(request, cli_result, f"system action `{action}` failed")
        if failed:
            return failed
        if action == "help":
            return response(
                request["skill"],
                "ok",
                "available system actions",
                actions=["install", "doctor", "help", "version", "checks", "settings.get", "settings.update", "settings.reset", "cfg.get", "cfg.copy"],
                logs=cli_logs(cli_result),
            )
        if action == "version":
            match = re.search(r"\b\d+\.\d+\.\d+\b", f"{cli_result['stdout']}\n{cli_result['stderr']}")
            version = match.group(0) if match else read_repo_version()
            return response(request["skill"], "ok", "version collected", data={"version": version}, logs=cli_logs(cli_result))
        if action == "checks":
            return response(request["skill"], "ok", "system checks collected", logs=cli_logs(cli_result))
        if action == "settings.get":
            core = get_ultralytics_core()
            return response(
                request["skill"],
                "ok",
                "settings collected",
                data={"settings": json_safe(dict(core["SETTINGS"]))},
                logs=cli_logs(cli_result),
            )
        if action == "settings.update":
            core = get_ultralytics_core()
            updates = params.get("updates") or {k: v for k, v in params.items() if k != "action"}
            return response(
                request["skill"],
                "ok",
                "settings updated",
                data={"settings": json_safe(dict(core["SETTINGS"])), "updated": json_safe(updates)},
                logs=cli_logs(cli_result),
            )
        if action == "settings.reset":
            core = get_ultralytics_core()
            return response(request["skill"], "ok", "settings reset", data={"settings": json_safe(dict(core["SETTINGS"]))}, logs=cli_logs(cli_result))
        if action == "cfg.get":
            return response(request["skill"], "ok", "default cfg loaded", data={"cfg": json_safe(read_default_cfg())}, logs=cli_logs(cli_result))
        if action == "cfg.copy":
            new_file = ensure_manifest_dir(request) / DEFAULT_CFG_FILE.name.replace(".yaml", "_copy.yaml")
            return response(
                request["skill"],
                "ok",
                "default cfg copied",
                artifacts=[{"kind": "config", "path": str(new_file.resolve())}],
                logs=cli_logs(cli_result),
            )
    raise ValueError(f"Unsupported yolo.system action: {action}")


def run_model_inspect(request: dict[str, Any]) -> dict[str, Any]:
    actions = request["params"].get("actions") or ["info", "names", "device", "task_map"]
    if is_dry_run(request):
        return plan_response(request, "inspect dry run prepared", "python_api", "YOLO(...).inspect", params={"actions": actions})

    model = build_model(request)
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


def run_train_like(request: dict[str, Any], skill_name: str) -> dict[str, Any]:
    params = dict(request["params"])
    if request["inputs"].get("data") and "data" not in params:
        params["data"] = request["inputs"]["data"]
    device_selection = resolve_device_selection(request, params)
    params, chosen_device, auto_completed = apply_runtime_defaults(request, params, purpose="train")
    effective_request = deepcopy(request)
    effective_request["params"] = params
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
        cli_execution = run_cli_with_recovery(
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

    model = build_model(request)
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


def run_val(request: dict[str, Any]) -> dict[str, Any]:
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
        cli_execution = run_cli_with_recovery(
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

    model = build_model(request)
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


def run_predict_like(request: dict[str, Any], mode: str) -> dict[str, Any]:
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
        cli_execution = run_cli_with_recovery(
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

    model = build_model(request)
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


def run_export(request: dict[str, Any]) -> dict[str, Any]:
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
        cli_result = run_cli(cli_args_from_values("export", values))
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

    model = build_model(request)
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


def run_benchmark(request: dict[str, Any]) -> dict[str, Any]:
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
        cli_execution = run_cli_with_recovery(
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

    model = build_model(request)
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


def run_tune(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    use_ray = bool(params.pop("use_ray", False))
    iterations = int(params.pop("iterations", 10))
    if request["inputs"].get("data") and "data" not in params:
        params["data"] = request["inputs"]["data"]
    if is_dry_run(request):
        return plan_response(
            request,
            "tune dry run prepared",
            "python_api",
            "YOLO(...).tune",
            params={"use_ray": use_ray, "iterations": iterations, **params},
        )

    model = build_model(request)
    tuned = model.tune(use_ray=use_ray, iterations=iterations, **params)
    payload = response(request["skill"], "ok", "tuning finished", data={"tune": json_safe(tuned)})
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_lora_adapters(request: dict[str, Any]) -> dict[str, Any]:
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

    model = build_model(request)
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
        ok = model.load_lora(path, merge=bool(params.get("merge", False)))
        payload = response(request["skill"], "ok" if ok else "failed", "adapter load finished")
    elif action == "merge":
        ok = model.merge_lora()
        payload = response(request["skill"], "ok" if ok else "failed", "adapter merge finished")
    else:
        raise ValueError(f"Unsupported adapter action: {action}")
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_moe_diagnose(request: dict[str, Any]) -> dict[str, Any]:
    inputs = request["inputs"]
    params = request["params"]
    model_path = inputs.get("model")
    dataset = inputs.get("data") or params.get("data", "coco8.yaml")
    batch_size = int(params.get("batch_size", 1))
    verbose = bool(params.get("verbose", False))
    output_dir = Path(params.get("output_dir") or ensure_manifest_dir(request) / "moe_diagnose")
    if is_dry_run(request):
        return plan_response(
            request,
            "MoE diagnose dry run prepared",
            "module",
            "diagnose_model",
            params={"model_path": model_path, "dataset": dataset, "batch_size": batch_size, "verbose": verbose, "output_dir": str(output_dir)},
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnose_model = get_moe_helpers()["diagnose_model"]
    with pushd(output_dir):
        _, stdout, stderr = capture_output(diagnose_model, model_path, dataset, batch_size, verbose)
    artifacts = []
    for name in ("expert_usage_heatmap.png", "expert_usage_bar.png"):
        file = output_dir / name
        if file.exists():
            artifacts.append({"kind": "image", "path": str(file.resolve())})
    payload = response(
        request["skill"],
        "ok",
        "moe diagnosis finished",
        artifacts=artifacts,
        logs={"stdout": stdout, "stderr": stderr},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_moe_prune(request: dict[str, Any]) -> dict[str, Any]:
    inputs = request["inputs"]
    params = request["params"]
    model_path = inputs.get("model")
    dataset = inputs.get("data") or params.get("data", "coco8.yaml")
    output_path = params.get("output_path") or str(ensure_manifest_dir(request) / "pruned_model.pt")
    threshold = float(params.get("threshold", 0.15))
    if is_dry_run(request):
        return plan_response(
            request,
            "MoE prune dry run prepared",
            "module",
            "prune_moe_model",
            params={"model_path": model_path, "output_path": output_path, "threshold": threshold, "dataset": dataset},
        )

    prune_moe_model = get_moe_helpers()["prune_moe_model"]
    ok, stdout, stderr = capture_output(prune_moe_model, model_path, output_path, threshold, dataset)
    payload = response(
        request["skill"],
        "ok" if ok else "failed",
        "moe prune finished" if ok else "moe prune failed",
        artifacts=[{"kind": "checkpoint", "path": str(Path(output_path).resolve())}] if Path(output_path).exists() else [],
        logs={"stdout": stdout, "stderr": stderr},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def format_solution_arg(key: str, value: Any) -> str:
    if isinstance(value, str):
        return f"{key}={value}"
    return f"{key}={repr(value)}"


def run_solutions(request: dict[str, Any]) -> dict[str, Any]:
    solution = request["inputs"].get("solution")
    if not solution:
        raise ValueError("`inputs.solution` is required for yolo.solutions.run.")
    if is_dry_run(request):
        if prefer_cli(request):
            args = ["solutions", solution]
            for key in ("model", "source"):
                if request["inputs"].get(key) is not None:
                    args.append(kv_arg(key, request["inputs"][key]))
            for key, value in request["params"].items():
                if key != "action":
                    args.append(kv_arg(key, value))
            return cli_plan(request, args)
        args = [solution]
        for key in ("model", "source"):
            if request["inputs"].get(key) is not None:
                args.append(format_solution_arg(key, request["inputs"][key]))
        for key, value in request["params"].items():
            if key != "action":
                args.append(format_solution_arg(key, value))
        return plan_response(request, "solutions dry run prepared", "module", "handle_yolo_solutions", params={"args": args})

    if prefer_cli(request):
        args = ["solutions", solution]
        for key in ("model", "source"):
            if request["inputs"].get(key) is not None:
                args.append(kv_arg(key, request["inputs"][key]))
        for key, value in request["params"].items():
            if key != "action":
                args.append(kv_arg(key, value))
        cli_result = run_cli(args)
        failed = ensure_cli_success(request, cli_result, "solutions run failed")
        if failed:
            return failed
        return response(
            request["skill"],
            "ok",
            "solution run finished",
            logs=cli_logs(cli_result),
            artifacts=[{"kind": "directory", "path": str((REPO_ROOT / "runs" / "solutions").resolve())}],
        )

    _, stdout, stderr = capture_output(get_cfg_helpers()["handle_yolo_solutions"], [solution, *[format_solution_arg(k, v) for k, v in request["inputs"].items() if k != "solution"], *[format_solution_arg(k, v) for k, v in request["params"].items() if k != "action"]])
    payload = response(
        request["skill"],
        "ok",
        "solution run finished",
        logs={"stdout": stdout, "stderr": stderr},
        artifacts=[{"kind": "directory", "path": str((REPO_ROOT / "runs" / "solutions").resolve())}],
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_ui_launch(request: dict[str, Any]) -> dict[str, Any]:
    mode = request["inputs"].get("mode") or request["params"].get("mode", "gradio")
    if is_dry_run(request):
        cmd_preview = [sys.executable, "app.py"] if mode == "gradio" else ["yolo", "solutions", "inference", f"model={request['inputs'].get('model', 'yolo11n.pt')}"]
        return plan_response(
            request,
            "UI launch dry run prepared",
            "cli" if mode == "streamlit" else "subprocess",
            mode,
            params={"cmd": cmd_preview},
        )

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout_path = LOG_DIR / f"{mode}.stdout.log"
    stderr_path = LOG_DIR / f"{mode}.stderr.log"
    stdout_handle = open(stdout_path, "ab")
    stderr_handle = open(stderr_path, "ab")
    if mode == "gradio":
        cmd = [sys.executable, "app.py"]
        url = request["params"].get("url", "http://127.0.0.1:7860")
    elif mode == "streamlit":
        model = request["inputs"].get("model") or "yolo11n.pt"
        yolo_path, _ = ensure_yolo_cli()
        cmd = [yolo_path, "solutions", "inference", f"model={model}"]
        url = request["params"].get("url", "http://127.0.0.1:8501")
    else:
        raise ValueError(f"Unsupported ui launch mode: {mode}")
    process = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=stdout_handle, stderr=stderr_handle, env=repo_cli_env())
    payload = response(
        request["skill"],
        "running",
        f"{mode} launcher started",
        job={"mode": "async", "pid": process.pid, "url": url},
        logs={"stdout_path": str(stdout_path.resolve()), "stderr_path": str(stderr_path.resolve())},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_pipeline(request: dict[str, Any]) -> dict[str, Any]:
    params = request["params"]
    common_inputs = deepcopy(request["inputs"])
    current_model = common_inputs.get("model")

    if is_dry_run(request):
        return plan_response(
            request,
            "pipeline dry run prepared",
            "orchestrator",
            "yolo.pipeline.experiment",
            params=params,
            next_actions=["yolo.train", "yolo.val", "yolo.export", "yolo.benchmark"],
        )

    stages = {}
    if "train" in params:
        train_request = normalize_request(
            {
                "skill": "yolo.train",
                "runtime": request.get("runtime", {}),
                "inputs": {**common_inputs, "model": current_model},
                "params": params["train"],
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": request.get("request_id"),
            }
        )
        stages["train"] = run_train_like(train_request, "yolo.train")
        current_model = best_checkpoint(stages["train"]) or current_model

    if "val" in params:
        val_request = normalize_request(
            {
                "skill": "yolo.val",
                "runtime": request.get("runtime", {}),
                "inputs": {**common_inputs, "model": current_model},
                "params": params["val"],
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": request.get("request_id"),
            }
        )
        stages["val"] = run_val(val_request)

    if "export" in params:
        export_request = normalize_request(
            {
                "skill": "yolo.export",
                "runtime": request.get("runtime", {}),
                "inputs": {**common_inputs, "model": current_model},
                "params": params["export"],
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": request.get("request_id"),
            }
        )
        stages["export"] = run_export(export_request)

    if "benchmark" in params:
        benchmark_request = normalize_request(
            {
                "skill": "yolo.benchmark",
                "runtime": request.get("runtime", {}),
                "inputs": {**common_inputs, "model": current_model},
                "params": params["benchmark"],
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": request.get("request_id"),
            }
        )
        stages["benchmark"] = run_benchmark(benchmark_request)

    payload = response(
        request["skill"],
        "ok",
        "pipeline finished",
        stages=stages,
        best_checkpoint=current_model,
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


HANDLERS = {
    "yolo.system": run_system,
    "yolo.model.inspect": run_model_inspect,
    "yolo.train": lambda request: run_train_like(request, "yolo.train"),
    "yolo.lora.train": lambda request: run_train_like(request, "yolo.lora.train"),
    "yolo.val": run_val,
    "yolo.predict": lambda request: run_predict_like(request, "predict"),
    "yolo.track": lambda request: run_predict_like(request, "track"),
    "yolo.export": run_export,
    "yolo.benchmark": run_benchmark,
    "yolo.tune": run_tune,
    "yolo.lora.adapters": run_lora_adapters,
    "yolo.moe.diagnose": run_moe_diagnose,
    "yolo.moe.prune": run_moe_prune,
    "yolo.solutions.run": run_solutions,
    "yolo.ui.launch": run_ui_launch,
    "yolo.pipeline.experiment": run_pipeline,
}


def load_request(args: argparse.Namespace) -> dict[str, Any]:
    if args.request:
        return json.loads(Path(args.request).read_text(encoding="utf-8"))
    if args.json:
        return json.loads(args.json)
    stdin = sys.stdin.read().strip()
    if stdin:
        return json.loads(stdin)
    raise ValueError("Provide --request, --json, or JSON on stdin.")


def main() -> int:
    parser = argparse.ArgumentParser(description="Structured dispatcher for the YOLO-Master agent skill.")
    parser.add_argument("--request", help="Path to a JSON request file.")
    parser.add_argument("--json", help="Inline JSON request.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    args = parser.parse_args()

    request: dict[str, Any] | None = None
    try:
        request = normalize_request(load_request(args))
        skill = request.get("skill")
        if skill not in HANDLERS:
            raise ValueError(f"Unsupported skill: {skill}")
        payload = HANDLERS[skill](request)
        payload = finalize_payload(request, payload)
    except Exception as exc:
        payload = response(
            request.get("skill", "unknown") if request else "unknown",
            "failed",
            str(exc),
            error={"type": type(exc).__name__, "traceback": traceback.format_exc()},
        )
        if request:
            payload = finalize_payload(request, payload)

    if args.pretty:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload.get("status") in {"ok", "running", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
