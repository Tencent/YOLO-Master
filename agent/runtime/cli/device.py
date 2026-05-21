from __future__ import annotations

import importlib
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from runtime.cli.contract import json_safe
from runtime.cli.executor import find_yolo_cli
from runtime.cli.normalize import reference_state

SKILL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = SKILL_ROOT / "logs"
ULTRALYTICS_INIT = REPO_ROOT / "ultralytics" / "__init__.py"
DEFAULT_CFG_FILE = REPO_ROOT / "ultralytics" / "cfg" / "default.yaml"
RUNTIME_CACHE_FILE = LOG_DIR / "runtime-cache.json"
RUNTIME_CACHE_TTL_SEC = 600
MODULE_CACHE: dict[str, Any] = {}


def cached(name: str, loader):
    if name not in MODULE_CACHE:
        MODULE_CACHE[name] = loader()
    return MODULE_CACHE[name]

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
def read_repo_version() -> str:
    text = ULTRALYTICS_INIT.read_text(encoding="utf-8")
    match = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    if not match:
        raise ValueError(f"Could not parse version from {ULTRALYTICS_INIT}")
    return match.group(1)
def read_default_cfg() -> dict[str, Any]:
    import yaml

    return yaml.safe_load(DEFAULT_CFG_FILE.read_text(encoding="utf-8"))
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
