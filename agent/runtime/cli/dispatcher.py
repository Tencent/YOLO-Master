#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import contextlib
import importlib
import io
import json
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import sysconfig
import time
import traceback
import urllib.error
import urllib.request
from copy import deepcopy
from pathlib import Path
from typing import Any
from uuid import uuid4

SKILL_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = SKILL_ROOT / "scripts"
for candidate in (SKILL_ROOT,):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from runtime.cli.contract import (
    ensure_manifest_dir,
    finalize_payload,
    json_safe,
    plan_response,
    response,
    write_manifest,
)
from runtime.cli.lora_tools import LoraDiagnoseDeps, run_lora_diagnose as run_lora_diagnose_impl
from runtime.cli.peft_compare import PeftCompareDeps, run_peft_compare as run_peft_compare_impl
from runtime.cli.pipeline import PipelineDeps, run_experiment_pipeline
from runtime.open_world.taxonomy import (
    aggregate_open_world_comparison,
    apply_open_world_assist_profile_defaults,
    build_open_world_comparison_entry,
    default_multimodal_max_output_tokens,
    effective_prompt_template_name,
    open_world_policy_enabled,
    open_world_template_enabled,
)
from runtime.multimodal.fusion import build_multimodal_fusion_preview as fusion_build_multimodal_fusion_preview
from runtime.evaluation.metrics import (
    aggregate_metric_preview,
    aggregate_multimodal_evaluation,
    build_item_metric_preview,
    build_metric_guardrail,
    classification_metric_delta,
    evaluate_classification_metric_preview,
    evaluate_detection_metric_preview,
    evaluate_segmentation_metric_preview,
    metric_delta,
    merge_counts,
    overall_multimodal_evaluation_status,
    preferred_verdict,
    prediction_records_to_coco,
    segmentation_metric_delta,
    yolo_coco_records_for_items,
)
from runtime.multimodal.runtime import (
    attach_multimodal_verdict,
    build_thinking_with_image_prompt,
    build_visual_search_crop_prompt,
    call_openai_compatible,
    default_llm_refine_developer_prompt,
    default_vlm_developer_prompt,
    extract_json_object,
    multimodal_overall_status,
    openai_config,
    run_visual_search_crop_passes as runtime_run_visual_search_crop_passes,
)
from runtime.multimodal.visual import (
    clamp_box_xyxy,
    encode_image_reference_for_openai as encode_image_reference_for_openai_raw,
    image_source_for_openai as visual_image_source_for_openai,
    load_pillow_image,
    normalize_detection_boxes,
    render_marked_image as visual_render_marked_image,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
LOG_DIR = SKILL_ROOT / "logs"
DEFAULT_MANIFEST_DIR = REPO_ROOT / "runs" / "agent"
MODULE_CACHE: dict[str, Any] = {}
ULTRALYTICS_INIT = REPO_ROOT / "ultralytics" / "__init__.py"
DEFAULT_CFG_FILE = REPO_ROOT / "ultralytics" / "cfg" / "default.yaml"
DATASET_CFG_DIR = REPO_ROOT / "ultralytics" / "cfg" / "datasets"
PROMPT_DIR = SKILL_ROOT / "assets" / "prompts"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
RUNTIME_CACHE_FILE = LOG_DIR / "runtime-cache.json"
RUNTIME_CACHE_TTL_SEC = 600
IMAGE_EXTENSIONS = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}
DEFAULT_OPEN_WORLD_LABEL_ALIASES = {
    "meal box": "bento box",
    "lunch box": "bento box",
    "lunchbox": "bento box",
    "bento meal": "bento box",
    "bento box meal": "bento box",
    "dried apricot": "dried fruit",
    "dried mango": "dried fruit",
    "apricot": "dried fruit",
    "pineapple chunks": "pineapple",
    "pineapple piece": "pineapple",
    "meat ball": "meatball",
    "meatball in sauce": "meatball",
    "bouquet": "flower arrangement",
    "flower bouquet": "flower arrangement",
    "flowers": "flower arrangement",
    "floral arrangement": "flower arrangement",
    "tree trunk": "log",
    "fallen tree": "log",
    "grass patch": "grass",
}
OPEN_WORLD_GENERIC_LABELS = {
    "food",
    "meal",
    "dish",
    "object",
    "objects",
    "container",
    "scene",
    "outdoor scene",
    "indoor scene",
}
DEFAULT_OPEN_WORLD_ALIASES_FILE = SKILL_ROOT / "assets" / "open_world_label_aliases.json"
OPEN_WORLD_TAXONOMY_DIR = SKILL_ROOT / "assets" / "open-world-taxonomy"
DEFAULT_OPEN_WORLD_TAXONOMY_FILES = {
    "lvis": OPEN_WORLD_TAXONOMY_DIR / "lvis_1203_classes.json",
    "v3det": OPEN_WORLD_TAXONOMY_DIR / "v3det_13204_classes.json",
}
MULTIMODAL_PARAM_KEYS = {
    "prompt",
    "question",
    "system_prompt",
    "developer_prompt",
    "thinking_with_image",
    "method",
    "prompt_template",
    "compact_open_world_profile",
    "open_world_profile",
    "open_world_assist_profile",
    "open_world_label_normalizer",
    "open_world_label_aliases",
    "open_world_label_aliases_path",
    "open_world_taxonomy_datasets",
    "open_world_taxonomy_topk",
    "open_world_taxonomy_min_score",
    "open_world_taxonomy_require_exact_for_generic",
    "open_world_taxonomy_hypernym_fallback",
    "open_world_filter_unmatched_taxonomy",
    "open_world_filter_generic_labels",
    "open_world_iou_relabel_enabled",
    "open_world_iou_relabel_threshold",
    "open_world_iou_relabel_max_yolo_confidence",
    "provider",
    "vlm_provider",
    "vlm_model",
    "llm_model",
    "openai_base_url",
    "openai_api_mode",
    "image_detail",
    "max_output_tokens",
    "temperature",
    "max_reasoning_items",
    "max_reasoning_boxes",
    "max_image_bytes",
    "structured_output",
    "enable_llm_refine",
    "skip_yolo",
    "detections",
    "use_marked_image",
    "visual_search_mode",
    "visual_search_max_regions",
    "visual_search_crop_margin",
    "visual_search_prompt",
    "fusion_mode",
    "fusion_policy",
    "fusion_enabled",
    "fusion_open_world_confidence_min",
    "fusion_add_confidence_min",
    "fusion_add_require_unlinked",
    "fusion_add_max_linked_yolo_confidence",
    "fusion_add_allowed_bbox_quality",
    "fusion_suppress_confidence_min",
    "fusion_adjust_confidence_min",
    "fusion_suppress_max_yolo_confidence",
    "fusion_relabel_max_yolo_confidence",
    "fusion_adjust_min_iou",
    "fusion_metric_guardrail",
    "fusion_guardrail_min_map50_95_delta",
    "fusion_guardrail_require_recall_nonnegative",
}
MULTIMODAL_EVALUATE_PARAM_KEYS = {
    "data",
    "split",
    "limit",
    "max_images",
    "offset",
    "stride",
    "shuffle",
    "seed",
    "include_ground_truth",
    "include_ground_truth_in_prompt",
    "run_yolo_val",
    "continue_on_error",
    "report_name",
}

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


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "on"}:
            return True
        if text in {"0", "false", "no", "off"}:
            return False
    return bool(value)


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


def image_source_for_openai(source: Any, results_summary: list[dict[str, Any]]) -> str | None:
    return visual_image_source_for_openai(source, results_summary, resolved_path=resolved_path)


def encode_image_reference_for_openai(image_ref: str, max_bytes: int = 20_000_000) -> dict[str, Any]:
    return globals()["encode_image_reference_for_openai_raw"](image_ref, resolved_path=resolved_path, max_bytes=max_bytes)


def render_marked_image(
    image_ref: str | Path,
    detections: list[dict[str, Any]],
    *,
    output_dir: Path,
    prefix: str,
    max_items: int = 24,
) -> dict[str, Any]:
    return visual_render_marked_image(
        image_ref,
        detections,
        resolved_path=resolved_path,
        output_dir=output_dir,
        prefix=prefix,
        max_items=max_items,
    )


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


def summarize_results_for_reasoning(results: Any, max_items: int = 5, max_boxes: int = 20) -> list[dict[str, Any]]:
    """Return compact, structured prediction evidence for multimodal reasoning."""
    summary = []
    for result in list(results)[:max_items]:
        item: dict[str, Any] = {
            "path": str(getattr(result, "path", "")),
            "speed": json_safe(getattr(result, "speed", {})),
            "detections": [],
        }
        names = getattr(result, "names", {}) or {}
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            try:
                xyxy = boxes.xyxy.detach().cpu().tolist()
                cls = boxes.cls.detach().cpu().tolist()
                conf = boxes.conf.detach().cpu().tolist()
                for idx, coords in enumerate(xyxy[:max_boxes]):
                    class_id = int(cls[idx]) if idx < len(cls) else None
                    item["detections"].append(
                        {
                            "index": idx,
                            "class_id": class_id,
                            "label": names.get(class_id, str(class_id)) if class_id is not None else None,
                            "confidence": round(float(conf[idx]), 4) if idx < len(conf) else None,
                            "xyxy": [round(float(v), 2) for v in coords],
                        }
                    )
            except Exception:
                try:
                    item["boxes"] = len(boxes)
                except Exception:
                    item["boxes"] = 0
        summary.append(item)
    return summary


def build_visual_search_crop_prompt(
    base_prompt: str,
    region: dict[str, Any],
    detections: list[dict[str, Any]],
) -> str:
    return (
        "You are inspecting a zoomed crop extracted from the original image.\n"
        "Focus on local object presence, object boundaries, and whether the YOLO box should be kept, adjusted, or suppressed.\n"
        "Return exactly one JSON object without markdown fences. Preserve any useful keys from the schema, especially answer, visual_evidence, "
        "yolo_cross_check, uncertainty, recommended_next_actions, visual_search, and fusion_hints.\n\n"
        f"Crop region:\n{json.dumps(json_safe(region), ensure_ascii=False, indent=2)}\n\n"
        f"Base task:\n{base_prompt}\n\n"
        f"YOLO detection summary:\n{json.dumps(json_safe(detections), ensure_ascii=False, indent=2)}\n"
    )


def default_visual_search_regions(detections: list[dict[str, Any]], max_regions: int = 2) -> list[dict[str, Any]]:
    normalized = normalize_detection_boxes(detections)
    ranked = sorted(
        normalized,
        key=lambda item: (
            float(item.get("confidence") or 0.0),
            (item["bbox_xyxy"][2] - item["bbox_xyxy"][0]) * (item["bbox_xyxy"][3] - item["bbox_xyxy"][1]),
        ),
    )
    regions: list[dict[str, Any]] = []
    for item in ranked[: max(0, max_regions)]:
        regions.append(
            {
                "region_id": f"auto-{item['index']}",
                "bbox_xyxy": item["bbox_xyxy"],
                "reason": f"low_confidence_{item.get('label')}",
                "priority": "high" if float(item.get("confidence") or 0.0) < 0.35 else "medium",
                "linked_yolo_indices": [item["index"]],
            }
        )
    return regions


def extract_visual_search_regions(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(verdict, dict):
        return []
    visual_search = verdict.get("visual_search", {})
    if not isinstance(visual_search, dict):
        return []
    regions = visual_search.get("search_regions", [])
    if not isinstance(regions, list):
        return []
    normalized: list[dict[str, Any]] = []
    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox_xyxy") or region.get("bbox") or region.get("xyxy")
        if not isinstance(bbox, (list, tuple)):
            continue
        normalized.append(
            {
                "region_id": str(region.get("region_id") or f"r{idx+1}"),
                "bbox_xyxy": [float(v) for v in bbox[:4]],
                "purpose": str(region.get("purpose") or region.get("reason") or "inspect uncertain region"),
                "priority": str(region.get("priority") or "medium"),
                "linked_yolo_indices": [int(v) for v in region.get("linked_yolo_indices", []) if isinstance(v, (int, float, str)) and str(v).isdigit()],
                "raw": region,
            }
        )
    return normalized


def run_visual_search_crop_passes(
    *,
    image_path: str | Path,
    base_prompt: str,
    detections: list[dict[str, Any]],
    initial_verdict: dict[str, Any],
    provider_cfg: dict[str, Any],
    multimodal_params: dict[str, Any],
    output_dir: Path,
    max_output_tokens: int,
    method: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode = str(multimodal_params.get("visual_search_mode") or "auto").lower()
    if mode in {"off", "none", "false", "0"}:
        return [], []
    if mode in {"auto", "smart"}:
        visual_search = initial_verdict.get("visual_search", {}) if isinstance(initial_verdict, dict) else {}
        if not (isinstance(visual_search, dict) and parse_bool(visual_search.get("needs_zoom"), False)):
            return [], []
    regions = extract_visual_search_regions(initial_verdict)
    if not regions and mode in {"always", "low-confidence", "low_confidence"}:
        regions = default_visual_search_regions(detections, max_regions=int(multimodal_params.get("visual_search_max_regions", 2)))
    if not regions:
        return [], []
    source_path = resolved_path(image_path)
    image = load_pillow_image(source_path).convert("RGB")
    width, height = image.size
    crop_margin = float(multimodal_params.get("visual_search_crop_margin", 0.18))
    max_regions = max(0, int(multimodal_params.get("visual_search_max_regions", 2)))
    region_results: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    crop_dir = output_dir / "visual-search"
    crop_dir.mkdir(parents=True, exist_ok=True)

    for idx, region in enumerate(regions[:max_regions]):
        bbox = clamp_box_xyxy(region.get("bbox_xyxy", []), width, height, margin=crop_margin)
        if bbox is None:
            continue
        crop = image.crop(tuple(bbox))
        crop_path = crop_dir / f"{Path(source_path).stem}-region-{idx + 1}.jpg"
        crop.save(crop_path, quality=92)
        artifacts.append({"kind": "crop_image", "path": str(crop_path.resolve()), "region_id": region.get("region_id")})
        crop_prompt = build_visual_search_crop_prompt(base_prompt, region, detections)
        crop_result = call_openai_compatible(
            model=str(provider_cfg["vlm_model"]),
            user_text=crop_prompt,
            developer_text=str(
                multimodal_params.get("developer_prompt")
                or multimodal_params.get("system_prompt")
                or "You are a careful visual search assistant. Focus on the crop and return concise structured JSON."
            ),
            image_url=encode_image_reference_for_openai(str(crop_path), max_bytes=int(multimodal_params.get("max_image_bytes", 20_000_000)))["image_url"],
            image_detail=str(multimodal_params.get("image_detail", "auto")),
            base_url=provider_cfg["base_url"],
            provider=str(provider_cfg.get("provider", "openai")),
            api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
            api_mode=str(provider_cfg["api_mode"]),
            max_output_tokens=max_output_tokens,
            temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
        )
        crop_result = attach_multimodal_verdict(crop_result)
        region_results.append(
            {
                "region": {**region, "bbox_xyxy": bbox},
                "crop": {"path": str(crop_path.resolve()), "bbox_xyxy": bbox},
                "vlm": crop_result,
            }
        )
    return region_results, artifacts


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, str):
            text = value.strip()
            if not re.fullmatch(r"[-+]?\d+(?:\.0+)?", text):
                return None
            return int(float(text))
        return int(value)
    except Exception:
        return None


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def extract_yolo_indices(value: Any) -> set[int]:
    indices: set[int] = set()
    if isinstance(value, dict):
        for key in ("index", "yolo_index", "yolo_idx", "yolo_box_index", "box_index"):
            idx = coerce_int(value.get(key))
            if idx is not None:
                indices.add(idx)
        for key in ("linked_yolo_indices", "indices", "yolo_indices"):
            for item in as_list(value.get(key)):
                idx = coerce_int(item)
                if idx is not None:
                    indices.add(idx)
        return indices
    if isinstance(value, list):
        for item in value:
            indices.update(extract_yolo_indices(item))
        return indices
    idx = coerce_int(value)
    if idx is not None:
        indices.add(idx)
    return indices


def valid_xyxy(box: Any) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    values: list[float] = []
    for item in box[:4]:
        value = coerce_float(item)
        if value is None:
            return None
        values.append(round(value, 3))
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def xyxy_to_xywh(box: list[float]) -> list[float]:
    return [round(box[0], 3), round(box[1], 3), round(box[2] - box[0], 3), round(box[3] - box[1], 3)]


def merge_verdicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_verdicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def proposal_key(proposal: dict[str, Any]) -> str:
    proposal_id = proposal.get("proposal_id")
    if proposal_id not in (None, ""):
        return f"id:{proposal_id}"
    bbox = valid_xyxy(proposal.get("bbox_xyxy") or proposal.get("bbox") or proposal.get("xyxy"))
    return f"bbox:{bbox}:{proposal.get('class_id')}:{proposal.get('label')}"


def resolve_proposal_reference(item: Any, proposals_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if isinstance(item, dict):
        proposal_id = item.get("proposal_id")
        if proposal_id not in (None, "") and str(proposal_id) in proposals_by_id:
            merged = dict(proposals_by_id[str(proposal_id)])
            merged.update({k: v for k, v in item.items() if v not in (None, "")})
            return merged
        return dict(item)
    key = str(item)
    if key in proposals_by_id:
        return dict(proposals_by_id[key])
    return None


def prompt_template_name(prompt_template: Any) -> str:
    if prompt_template in (None, ""):
        return ""
    value = str(prompt_template)
    return Path(value).stem if value.endswith(".md") else Path(value).name


def proposal_open_label(proposal: dict[str, Any]) -> str | None:
    for key in ("open_label", "object_label", "category_name", "label", "name"):
        value = proposal.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def proposal_coco_class_id(proposal: dict[str, Any]) -> int | None:
    for key in ("class_id", "category_id", "coco_class_id", "mapped_class_id", "canonical_class_id"):
        value = coerce_int(proposal.get(key))
        if value is not None:
            return value
    coco_candidate = proposal.get("coco_candidate")
    if isinstance(coco_candidate, dict):
        for key in ("class_id", "category_id"):
            value = coerce_int(coco_candidate.get(key))
            if value is not None:
                return value
    return None


def proposal_coco_label(proposal: dict[str, Any]) -> str | None:
    for key in ("coco_label", "canonical_label", "mapped_label", "label"):
        value = proposal.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    coco_candidate = proposal.get("coco_candidate")
    if isinstance(coco_candidate, dict):
        for key in ("label", "name"):
            value = coco_candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def build_multimodal_fusion_preview(
    *,
    detections: list[dict[str, Any]],
    verdict: dict[str, Any],
    multimodal_params: dict[str, Any],
    image_path: str | Path | None = None,
) -> dict[str, Any]:
    mode = str(multimodal_params.get("fusion_mode") or "preview").lower()
    policy = str(multimodal_params.get("fusion_policy") or "add_only").lower()
    enabled = parse_bool(multimodal_params.get("fusion_enabled"), mode not in {"off", "none", "false", "0"})
    if not enabled:
        return {"mode": "off", "strategy": "metric_safe_v1", "enabled": False}

    open_world_enabled = open_world_policy_enabled(multimodal_params) or open_world_template_enabled(multimodal_params)
    open_world_assist = policy in {"open_world_assist", "open-world-assist"}
    allow_add = policy in {"add_only", "balanced", "aggressive", "all"} or open_world_enabled
    allow_suppress = policy in {"balanced", "aggressive", "all"} or (open_world_enabled and not open_world_assist and policy in {"open_world", "open-world"})
    allow_adjust = policy in {"aggressive", "all"}
    allow_relabel = policy in {"aggressive", "all"}

    add_min = float(multimodal_params.get("fusion_add_confidence_min", 0.65))
    open_world_add_min = float(multimodal_params.get("fusion_open_world_confidence_min", 0.55 if open_world_enabled else add_min))
    add_require_unlinked = parse_bool(multimodal_params.get("fusion_add_require_unlinked"), True)
    add_max_linked_yolo_conf = float(multimodal_params.get("fusion_add_max_linked_yolo_confidence", 0.4))
    add_allowed_bbox_quality = {
        str(value).lower()
        for value in as_list(multimodal_params.get("fusion_add_allowed_bbox_quality") or ["exact", "estimated"])
        if value not in (None, "")
    }
    suppress_min = float(multimodal_params.get("fusion_suppress_confidence_min", 0.75))
    adjust_min = float(multimodal_params.get("fusion_adjust_confidence_min", 0.7))
    suppress_max_yolo_conf = float(multimodal_params.get("fusion_suppress_max_yolo_confidence", 0.45))
    relabel_max_yolo_conf = float(multimodal_params.get("fusion_relabel_max_yolo_confidence", 0.5))
    adjust_min_iou = float(multimodal_params.get("fusion_adjust_min_iou", 0.5))
    open_world_iou_relabel_enabled = parse_bool(multimodal_params.get("open_world_iou_relabel_enabled"), False)
    open_world_iou_relabel_threshold = float(multimodal_params.get("open_world_iou_relabel_threshold", 0.7) or 0.7)
    open_world_iou_relabel_max_yolo_conf = float(multimodal_params.get("open_world_iou_relabel_max_yolo_confidence", 0.8) or 0.8)
    yolo_boxes = normalize_detection_boxes(detections)
    yolo_by_index = {int(item["index"]): item for item in yolo_boxes}
    fusion_hints = verdict.get("fusion_hints", {}) if isinstance(verdict, dict) else {}
    if not isinstance(fusion_hints, dict):
        fusion_hints = {}
    cross_check = verdict.get("yolo_cross_check", {}) if isinstance(verdict, dict) else {}
    if not isinstance(cross_check, dict):
        cross_check = {}

    vlm_proposals = [item for item in as_list(verdict.get("vlm_detections")) if isinstance(item, dict)] if isinstance(verdict, dict) else []
    proposals_by_id = {
        str(item.get("proposal_id")): item
        for item in vlm_proposals
        if item.get("proposal_id") not in (None, "")
    }

    actions: list[dict[str, Any]] = []
    warnings: list[str] = []
    suppress_indices = set()
    open_world_predictions: list[dict[str, Any]] = []

    def yolo_confidence(idx: int) -> float | None:
        return coerce_float((yolo_by_index.get(idx) or {}).get("confidence"))

    def yolo_box(idx: int) -> list[float] | None:
        return valid_xyxy((yolo_by_index.get(idx) or {}).get("bbox_xyxy"))

    def can_suppress(idx: int, score: float | None) -> tuple[bool, str]:
        yolo_conf = yolo_confidence(idx)
        if score is not None and score < suppress_min:
            return False, "vlm_confidence_below_threshold"
        if yolo_conf is not None and yolo_conf > suppress_max_yolo_conf:
            return False, "yolo_confidence_too_high"
        return True, "vlm_false_positive_low_yolo_confidence"

    def can_add(proposal: dict[str, Any]) -> tuple[bool, str]:
        bbox_quality = str(proposal.get("bbox_quality") or "estimated").lower()
        if bbox_quality not in add_allowed_bbox_quality:
            return False, "bbox_quality_not_allowed"
        linked_indices = sorted(extract_yolo_indices(proposal))
        if add_require_unlinked and linked_indices:
            linked_confidences = [yolo_confidence(idx) for idx in linked_indices]
            if any(conf is not None and conf > add_max_linked_yolo_conf for conf in linked_confidences):
                return False, "linked_yolo_confidence_too_high"
        return True, "vlm_add_candidate"

    for item in as_list(fusion_hints.get("suppress_yolo_indices")) + as_list(cross_check.get("false_positives")):
        score = coerce_float(item.get("confidence")) if isinstance(item, dict) else None
        for idx in extract_yolo_indices(item):
            if not allow_suppress:
                actions.append(
                    {
                        "type": "reject_suppress",
                        "yolo_index": idx,
                        "confidence": score,
                        "yolo_confidence": yolo_confidence(idx),
                        "reason": "fusion_policy_disallows_suppress",
                        "policy": policy,
                    }
                )
                continue
            allowed, reason = can_suppress(idx, score)
            if allowed:
                suppress_indices.add(idx)
                actions.append(
                    {
                        "type": "suppress",
                        "yolo_index": idx,
                        "confidence": score,
                        "yolo_confidence": yolo_confidence(idx),
                        "reason": reason,
                    }
                )
            else:
                actions.append(
                    {
                        "type": "reject_suppress",
                        "yolo_index": idx,
                        "confidence": score,
                        "yolo_confidence": yolo_confidence(idx),
                        "reason": reason,
                        "threshold": suppress_min,
                        "max_yolo_confidence": suppress_max_yolo_conf,
                    }
                )

    adjustments: dict[int, dict[str, Any]] = {}
    for item in as_list(fusion_hints.get("adjust_boxes")):
        if not isinstance(item, dict):
            continue
        if not allow_adjust:
            idx_set = extract_yolo_indices(item)
            score = coerce_float(item.get("confidence"))
            for idx in idx_set:
                actions.append(
                    {
                        "type": "reject_adjust_box",
                        "yolo_index": idx,
                        "confidence": score,
                        "reason": "fusion_policy_disallows_adjust",
                        "policy": policy,
                    }
                )
            continue
        idx_set = extract_yolo_indices(item)
        bbox = valid_xyxy(item.get("bbox_xyxy") or item.get("bbox") or item.get("xyxy"))
        score = coerce_float(item.get("confidence"))
        if bbox is None:
            continue
        for idx in idx_set:
            original_box = yolo_box(idx)
            overlap = box_iou_xyxy(original_box, bbox) if original_box is not None else None
            if score is not None and score < adjust_min:
                actions.append({"type": "reject_adjust_box", "yolo_index": idx, "confidence": score, "threshold": adjust_min})
            elif original_box is None:
                actions.append({"type": "reject_adjust_box", "yolo_index": idx, "confidence": score, "reason": "missing_original_yolo_box"})
            elif overlap is None or overlap < adjust_min_iou:
                actions.append(
                    {
                        "type": "reject_adjust_box",
                        "yolo_index": idx,
                        "confidence": score,
                        "iou_with_original": overlap,
                        "min_iou": adjust_min_iou,
                        "reason": "adjustment_too_far_from_original",
                    }
                )
            else:
                adjustments[idx] = {"bbox_xyxy": bbox, "confidence": score, "raw": item}
                actions.append({"type": "adjust_box", "yolo_index": idx, "bbox_xyxy": bbox, "confidence": score, "iou_with_original": overlap})

    relabels: dict[int, dict[str, Any]] = {}
    for item in as_list(fusion_hints.get("relabel_yolo")):
        if not isinstance(item, dict):
            continue
        if not allow_relabel:
            idx_set = extract_yolo_indices(item)
            class_value = item.get("class_id") if item.get("class_id") is not None else item.get("to_class_id")
            class_value = class_value if class_value is not None else item.get("new_class_id")
            class_id = coerce_int(class_value)
            label = item.get("label") or item.get("to_label") or item.get("new_label")
            for idx in idx_set:
                actions.append(
                    {
                        "type": "reject_relabel",
                        "yolo_index": idx,
                        "class_id": class_id,
                        "label": label,
                        "reason": "fusion_policy_disallows_relabel",
                        "policy": policy,
                    }
                )
            continue
        idx_set = extract_yolo_indices(item)
        class_value = item.get("class_id") if item.get("class_id") is not None else item.get("to_class_id")
        class_value = class_value if class_value is not None else item.get("new_class_id")
        class_id = coerce_int(class_value)
        label = item.get("label") or item.get("to_label") or item.get("new_label")
        if class_id is None and label in (None, ""):
            continue
        for idx in idx_set:
            yolo_conf = yolo_confidence(idx)
            if yolo_conf is not None and yolo_conf > relabel_max_yolo_conf:
                actions.append(
                    {
                        "type": "reject_relabel",
                        "yolo_index": idx,
                        "class_id": class_id,
                        "label": label,
                        "yolo_confidence": yolo_conf,
                        "max_yolo_confidence": relabel_max_yolo_conf,
                        "reason": "yolo_confidence_too_high",
                    }
                )
                continue
            relabels[idx] = {"class_id": class_id, "label": label, "raw": item}
            actions.append({"type": "relabel", "yolo_index": idx, "class_id": class_id, "label": label, "yolo_confidence": yolo_conf})

    explicit_adds: list[dict[str, Any]] = []
    explicit_open_world_adds: list[dict[str, Any]] = []
    if allow_add:
        for item in as_list(fusion_hints.get("add_vlm_detections")):
            resolved = resolve_proposal_reference(item, proposals_by_id)
            if resolved is not None:
                explicit_adds.append(resolved)
        for item in as_list(fusion_hints.get("add_open_world_detections")):
            resolved = resolve_proposal_reference(item, proposals_by_id)
            if resolved is not None:
                explicit_open_world_adds.append(resolved)
    else:
        for item in as_list(fusion_hints.get("add_vlm_detections")):
            actions.append({"type": "reject_add", "proposal_id": item.get("proposal_id") if isinstance(item, dict) else str(item), "reason": "fusion_policy_disallows_add", "policy": policy})
    for proposal in vlm_proposals:
        action = str(proposal.get("coco_eval_action") or proposal.get("open_world_action") or "").lower()
        if action in {"add", "add_vlm", "add_vlm_detection", "new", "insert"}:
            if allow_add and (proposal_coco_class_id(proposal) is not None or not open_world_enabled):
                explicit_adds.append(proposal)
            elif allow_add and open_world_enabled:
                explicit_open_world_adds.append(proposal)
            else:
                actions.append({"type": "reject_add", "proposal_id": proposal.get("proposal_id"), "reason": "fusion_policy_disallows_add", "policy": policy})
        elif action in {"open_world_add", "open-world-add", "open_world", "discover", "novel"}:
            if allow_add and open_world_enabled:
                explicit_open_world_adds.append(proposal)
            else:
                actions.append({"type": "reject_open_world_add", "proposal_id": proposal.get("proposal_id"), "reason": "fusion_policy_disallows_open_world_add", "policy": policy})
        elif action in {"suppress", "drop", "remove"}:
            for idx in extract_yolo_indices(proposal):
                if not allow_suppress:
                    actions.append({"type": "reject_suppress", "yolo_index": idx, "proposal_id": proposal.get("proposal_id"), "reason": "fusion_policy_disallows_suppress", "policy": policy})
                    continue
                score = coerce_float(proposal.get("confidence"))
                allowed, reason = can_suppress(idx, score)
                if allowed:
                    suppress_indices.add(idx)
                    actions.append(
                        {
                            "type": "suppress",
                            "yolo_index": idx,
                            "proposal_id": proposal.get("proposal_id"),
                            "confidence": score,
                            "yolo_confidence": yolo_confidence(idx),
                            "reason": "vlm_proposal_action",
                        }
                    )
                else:
                    actions.append(
                        {
                            "type": "reject_suppress",
                            "yolo_index": idx,
                            "proposal_id": proposal.get("proposal_id"),
                            "confidence": score,
                            "yolo_confidence": yolo_confidence(idx),
                            "reason": reason,
                            "threshold": suppress_min,
                            "max_yolo_confidence": suppress_max_yolo_conf,
                        }
                    )

    fused_predictions: list[dict[str, Any]] = []
    for item in yolo_boxes:
        idx = int(item["index"])
        if idx in suppress_indices:
            continue
        prediction = {
            "source": "yolo",
            "index": idx,
            "class_id": item.get("class_id"),
            "label": item.get("label"),
            "confidence": item.get("confidence"),
            "bbox_xyxy": item.get("bbox_xyxy"),
            "coco_bbox_xywh": xyxy_to_xywh(item["bbox_xyxy"]),
            "action": "keep",
        }
        if idx in adjustments:
            prediction["bbox_xyxy"] = adjustments[idx]["bbox_xyxy"]
            prediction["coco_bbox_xywh"] = xyxy_to_xywh(adjustments[idx]["bbox_xyxy"])
            prediction["action"] = "adjusted"
        if idx in relabels:
            if relabels[idx].get("class_id") is not None:
                prediction["class_id"] = relabels[idx]["class_id"]
            if relabels[idx].get("label") not in (None, ""):
                prediction["label"] = relabels[idx]["label"]
            prediction["action"] = "relabelled" if prediction["action"] == "keep" else f"{prediction['action']}+relabelled"
        fused_predictions.append(prediction)

    seen_adds: set[str] = set()
    for proposal in explicit_adds:
        key = proposal_key(proposal)
        if key in seen_adds:
            continue
        seen_adds.add(key)
        bbox = valid_xyxy(proposal.get("bbox_xyxy") or proposal.get("bbox") or proposal.get("xyxy"))
        confidence = coerce_float(proposal.get("confidence"))
        class_value = proposal.get("class_id") if proposal.get("class_id") is not None else proposal.get("category_id")
        class_id = coerce_int(class_value)
        if bbox is None or class_id is None:
            warnings.append(f"Skipped VLM proposal without usable bbox/class_id: {proposal.get('proposal_id', key)}")
            continue
        allowed, reason = can_add(proposal)
        if not allowed:
            actions.append(
                {
                    "type": "reject_add",
                    "proposal_id": proposal.get("proposal_id"),
                    "confidence": confidence,
                    "bbox_quality": proposal.get("bbox_quality"),
                    "linked_yolo_indices": sorted(extract_yolo_indices(proposal)),
                    "reason": reason,
                    "policy": policy,
                }
            )
            continue
        if confidence is None or confidence < add_min:
            actions.append(
                {
                    "type": "reject_add",
                    "proposal_id": proposal.get("proposal_id"),
                    "confidence": confidence,
                    "threshold": add_min,
                }
            )
            continue
        fused_predictions.append(
            {
                "source": "vlm",
                "proposal_id": proposal.get("proposal_id"),
                "class_id": class_id,
                "label": proposal.get("label"),
                "confidence": round(confidence, 4),
                "bbox_xyxy": bbox,
                "coco_bbox_xywh": xyxy_to_xywh(bbox),
                "bbox_quality": proposal.get("bbox_quality", "estimated"),
                "action": "added",
                "linked_yolo_indices": sorted(extract_yolo_indices(proposal)),
            }
        )
        actions.append({"type": "add", "proposal_id": proposal.get("proposal_id"), "confidence": confidence})

    seen_open_world_adds: set[str] = set()
    for proposal in explicit_open_world_adds:
        key = proposal_key(proposal)
        if key in seen_open_world_adds:
            continue
        seen_open_world_adds.add(key)
        bbox = valid_xyxy(proposal.get("bbox_xyxy") or proposal.get("bbox") or proposal.get("xyxy"))
        confidence = coerce_float(proposal.get("confidence"))
        open_label = proposal_open_label(proposal)
        class_id = proposal_coco_class_id(proposal)
        coco_label = proposal_coco_label(proposal)
        if bbox is None or open_label in (None, ""):
            warnings.append(f"Skipped open-world VLM proposal without usable bbox/label: {proposal.get('proposal_id', key)}")
            continue
        relabel_hit = None
        if open_world_iou_relabel_enabled:
            for yolo_item in yolo_boxes:
                yolo_bbox = yolo_item.get("bbox_xyxy")
                iou = box_iou_xyxy(yolo_bbox, bbox) if isinstance(yolo_bbox, list) else 0.0
                if iou >= open_world_iou_relabel_threshold:
                    yolo_conf = coerce_float(yolo_item.get("confidence"))
                    if yolo_conf is not None and yolo_conf <= open_world_iou_relabel_max_yolo_conf:
                        relabel_hit = {
                            "yolo_index": yolo_item.get("index"),
                            "yolo_label": yolo_item.get("label"),
                            "yolo_confidence": yolo_conf,
                            "iou": iou,
                        }
                        break
        if relabel_hit is not None:
            record = {
                "source": "vlm_open_world",
                "proposal_id": proposal.get("proposal_id"),
                "open_label": open_label,
                "label": coco_label or open_label,
                "class_id": class_id,
                "confidence": round(confidence or 0.0, 4),
                "bbox_xyxy": bbox,
                "bbox_quality": proposal.get("bbox_quality", "estimated"),
                "action": "open_world_relabelled",
                "linked_yolo_indices": sorted(extract_yolo_indices(proposal)),
                "ontology_aliases": as_list(proposal.get("ontology_aliases")),
                "relabelled_from": relabel_hit,
            }
            open_world_predictions.append(record)
            actions.append(
                {
                    "type": "relabel_open_world",
                    "proposal_id": proposal.get("proposal_id"),
                    "open_label": open_label,
                    "yolo_index": relabel_hit["yolo_index"],
                    "yolo_label": relabel_hit["yolo_label"],
                    "iou": relabel_hit["iou"],
                    "confidence": confidence,
                }
            )
            continue
        allowed, reason = can_add(proposal)
        if not allowed:
            actions.append(
                {
                    "type": "reject_open_world_add",
                    "proposal_id": proposal.get("proposal_id"),
                    "confidence": confidence,
                    "bbox_quality": proposal.get("bbox_quality"),
                    "linked_yolo_indices": sorted(extract_yolo_indices(proposal)),
                    "reason": reason,
                    "policy": policy,
                }
            )
            continue
        if confidence is None or confidence < open_world_add_min:
            actions.append(
                {
                    "type": "reject_open_world_add",
                    "proposal_id": proposal.get("proposal_id"),
                    "confidence": confidence,
                    "threshold": open_world_add_min,
                    "reason": "open_world_confidence_below_threshold",
                }
            )
            continue
        record = {
            "source": "vlm_open_world",
            "proposal_id": proposal.get("proposal_id"),
            "open_label": open_label,
            "label": coco_label or open_label,
            "class_id": class_id,
            "confidence": round(confidence, 4),
            "bbox_xyxy": bbox,
            "bbox_quality": proposal.get("bbox_quality", "estimated"),
            "action": "open_world_added",
            "linked_yolo_indices": sorted(extract_yolo_indices(proposal)),
            "ontology_aliases": as_list(proposal.get("ontology_aliases")),
        }
        if class_id is not None:
            record["coco_bbox_xywh"] = xyxy_to_xywh(bbox)
            fused_predictions.append(record)
            actions.append(
                {
                    "type": "add_open_world_mapped",
                    "proposal_id": proposal.get("proposal_id"),
                    "confidence": confidence,
                    "class_id": class_id,
                    "open_label": open_label,
                }
            )
        else:
            open_world_predictions.append(record)
            actions.append(
                {
                    "type": "add_open_world",
                    "proposal_id": proposal.get("proposal_id"),
                    "confidence": confidence,
                    "open_label": open_label,
                }
            )

    image_id: int | str | None = None
    if image_path not in (None, ""):
        stem = Path(str(image_path)).stem
        image_id = int(stem) if stem.isdigit() else stem
    coco_records = []
    for prediction in fused_predictions:
        class_id = coerce_int(prediction.get("class_id"))
        score = coerce_float(prediction.get("confidence"))
        bbox = prediction.get("coco_bbox_xywh")
        if image_id is None or class_id is None or score is None or not isinstance(bbox, list):
            continue
        coco_records.append(
            {
                "image_id": image_id,
                "category_id": class_id,
                "bbox": bbox,
                "score": round(score, 6),
                "source": prediction.get("source"),
                "action": prediction.get("action"),
            }
        )

    summary = {
        "yolo_boxes": len(yolo_boxes),
        "vlm_proposals": len(vlm_proposals),
        "kept": sum(1 for item in fused_predictions if item.get("source") == "yolo"),
        "suppressed": len(suppress_indices),
        "added": sum(1 for item in fused_predictions if item.get("source") == "vlm"),
        "open_world_added": len(open_world_predictions),
        "open_world_mapped_to_coco": sum(1 for item in fused_predictions if item.get("source") == "vlm_open_world"),
        "adjusted": len(adjustments),
        "relabelled": len(relabels),
        "fused_boxes": len(fused_predictions),
        "coco_records": len(coco_records),
    }
    return {
        "mode": mode,
        "policy": policy,
        "enabled": True,
        "strategy": "metric_safe_v1",
        "thresholds": {
            "add_confidence_min": add_min,
            "open_world_confidence_min": open_world_add_min,
            "add_require_unlinked": add_require_unlinked,
            "add_max_linked_yolo_confidence": add_max_linked_yolo_conf,
            "add_allowed_bbox_quality": sorted(add_allowed_bbox_quality),
            "suppress_confidence_min": suppress_min,
            "adjust_confidence_min": adjust_min,
            "suppress_max_yolo_confidence": suppress_max_yolo_conf,
            "relabel_max_yolo_confidence": relabel_max_yolo_conf,
            "adjust_min_iou": adjust_min_iou,
        },
        "summary": summary,
        "actions": actions,
        "predictions": fused_predictions,
        "coco_predictions_preview": coco_records,
        "open_world_predictions_preview": open_world_predictions,
        "warnings": warnings,
    }


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


def split_yolo_and_multimodal_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    yolo_params = {}
    multimodal_params = {}
    for key, value in params.items():
        if key in MULTIMODAL_PARAM_KEYS:
            multimodal_params[key] = value
        else:
            yolo_params[key] = value
    return yolo_params, multimodal_params


def split_yolo_multimodal_evaluate_params(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    yolo_params: dict[str, Any] = {}
    multimodal_params: dict[str, Any] = {}
    evaluate_params: dict[str, Any] = {}
    for key, value in params.items():
        if key in MULTIMODAL_PARAM_KEYS:
            multimodal_params[key] = value
        elif key in MULTIMODAL_EVALUATE_PARAM_KEYS:
            evaluate_params[key] = value
        else:
            yolo_params[key] = value
    return yolo_params, multimodal_params, evaluate_params


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def read_image_list_file(path: Path, root: Path | None = None) -> list[Path]:
    images: list[Path] = []
    base = root or path.parent
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        candidate = Path(line)
        if not candidate.is_absolute():
            candidate = (base / candidate).resolve()
        if is_image_file(candidate):
            images.append(candidate)
    return images


def expand_image_reference(value: Any, root: Path | None = None) -> list[Path]:
    if value in (None, ""):
        return []
    if isinstance(value, (list, tuple, set)):
        images: list[Path] = []
        for item in value:
            images.extend(expand_image_reference(item, root=root))
        return images
    text = str(value)
    if text.startswith(("http://", "https://", "data:image/")):
        return []
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = ((root or REPO_ROOT) / candidate).resolve()
    if candidate.is_dir():
        return sorted(path.resolve() for path in candidate.rglob("*") if is_image_file(path))
    if candidate.is_file():
        if candidate.suffix.lower() == ".txt":
            return read_image_list_file(candidate, root=root)
        if is_image_file(candidate):
            return [candidate.resolve()]
    return []


def normalize_dataset_names(names: Any) -> dict[int, str]:
    if isinstance(names, dict):
        normalized = {}
        for key, value in names.items():
            try:
                normalized[int(key)] = str(value)
            except Exception:
                continue
        return normalized
    if isinstance(names, list):
        return {idx: str(value) for idx, value in enumerate(names)}
    return {}


def load_dataset_yaml(data_ref: Any) -> tuple[Path, dict[str, Any]]:
    import yaml

    if data_ref in (None, ""):
        raise ValueError("`inputs.data` or `params.data` is required when no image `source` is provided.")
    normalized = normalize_value(data_ref)
    data_path = Path(str(normalized))
    if not data_path.is_absolute():
        data_path = resolved_path(data_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML was not found: {data_ref}")
    loaded = yaml.safe_load(data_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Dataset YAML must contain a mapping: {data_path}")
    return data_path, loaded


def dataset_settings_dir() -> Path | None:
    try:
        settings = dict(get_ultralytics_core()["SETTINGS"])
    except Exception:
        return None
    value = settings.get("datasets_dir")
    return Path(str(value)).expanduser().resolve() if value else None


def resolve_dataset_root(data_path: Path, dataset_cfg: dict[str, Any]) -> Path:
    path_value = dataset_cfg.get("path")
    if path_value in (None, ""):
        return data_path.parent.resolve()
    candidate_path = Path(str(path_value)).expanduser()
    if candidate_path.is_absolute():
        return candidate_path.resolve()

    candidates = [
        (data_path.parent / candidate_path).resolve(),
        (REPO_ROOT / candidate_path).resolve(),
    ]
    settings_dir = dataset_settings_dir()
    if settings_dir is not None:
        candidates.append((settings_dir / candidate_path).resolve())
    candidates.append((REPO_ROOT.parent / "datasets" / candidate_path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def dataset_split_spec(dataset_cfg: dict[str, Any], requested_split: str) -> tuple[str, Any]:
    for split in (requested_split, "val", "train", "test"):
        value = dataset_cfg.get(split)
        if value not in (None, ""):
            return split, value
    raise ValueError(f"Dataset YAML has no usable split for `{requested_split}`.")


def collect_dataset_images(data_ref: Any, split: str) -> tuple[list[Path], dict[str, Any], dict[int, str]]:
    data_path, dataset_cfg = load_dataset_yaml(data_ref)
    root = resolve_dataset_root(data_path, dataset_cfg)
    actual_split, spec = dataset_split_spec(dataset_cfg, split)
    images = expand_image_reference(spec, root=root)
    names = normalize_dataset_names(dataset_cfg.get("names", {}))
    dataset_info = {
        "data": str(data_path),
        "root": str(root),
        "split": actual_split,
        "requested_split": split,
        "source": json_safe(spec),
        "names_count": len(names),
    }
    return images, dataset_info, names


def dedupe_images(images: list[Path]) -> list[Path]:
    seen: set[str] = set()
    unique: list[Path] = []
    for image in images:
        key = str(image.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(image.resolve())
    return unique


def select_image_sample(
    images: list[Path],
    *,
    limit: int | None,
    offset: int = 0,
    stride: int = 1,
    shuffle: bool = False,
    seed: int | None = None,
) -> list[Path]:
    selected = list(images)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)
    if offset > 0:
        selected = selected[offset:]
    if stride > 1:
        selected = selected[::stride]
    if limit is not None and limit > 0:
        selected = selected[:limit]
    return selected


def collect_multimodal_evaluation_images(
    request: dict[str, Any],
    evaluate_params: dict[str, Any],
    yolo_params: dict[str, Any],
    multimodal_params: dict[str, Any],
) -> tuple[list[Path], dict[str, Any], dict[int, str]]:
    split = str(evaluate_params.get("split", "val"))
    source_ref = request["inputs"].get("source") or yolo_params.pop("source", None) or multimodal_params.get("source")
    data_ref = request["inputs"].get("data") or evaluate_params.get("data") or yolo_params.pop("data", None)
    if source_ref not in (None, ""):
        images = expand_image_reference(source_ref)
        dataset_info = {"source": json_safe(source_ref), "split": None, "root": None, "names_count": 0}
        names: dict[int, str] = {}
    else:
        images, dataset_info, names = collect_dataset_images(data_ref, split)
    images = dedupe_images(images)
    limit_raw = evaluate_params.get("limit", evaluate_params.get("max_images", 5))
    limit = int(limit_raw) if limit_raw not in (None, "") else 5
    limit_value = None if limit <= 0 else limit
    seed_raw = evaluate_params.get("seed")
    seed = int(seed_raw) if seed_raw not in (None, "") else None
    sample = select_image_sample(
        images,
        limit=limit_value,
        offset=int(evaluate_params.get("offset", 0)),
        stride=max(1, int(evaluate_params.get("stride", 1))),
        shuffle=parse_bool(evaluate_params.get("shuffle"), False),
        seed=seed,
    )
    dataset_info["images_total"] = len(images)
    dataset_info["sample_count"] = len(sample)
    dataset_info["sample_limit"] = limit_value
    dataset_info["sample_offset"] = int(evaluate_params.get("offset", 0))
    dataset_info["sample_stride"] = max(1, int(evaluate_params.get("stride", 1)))
    if not sample:
        raise ValueError("No local images were found for multimodal evaluation.")
    return sample, dataset_info, names


def label_path_for_image(image_path: Path) -> Path:
    parts = image_path.resolve().parts
    if "images" in parts:
        idx = len(parts) - 1 - list(reversed(parts)).index("images")
        return Path(*parts[:idx], "labels", *parts[idx + 1 :]).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def read_ground_truth_summary(image_path: Path, names: dict[int, str], max_objects: int = 30) -> dict[str, Any]:
    label_path = label_path_for_image(image_path)
    summary: dict[str, Any] = {"path": str(label_path), "exists": label_path.exists(), "objects": 0, "labels": [], "label_counts": {}}
    if not label_path.exists():
        return summary
    labels: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_label_line(raw_line)
        if parsed is None:
            continue
        class_id, xywhn, segment = parsed
        label = names.get(class_id, str(class_id))
        counts[label] = counts.get(label, 0) + 1
        item = {"class_id": class_id, "label": label}
        if len(segment) >= 6:
            item["segment_points"] = len(segment) // 2
        else:
            item["xywhn"] = [round(float(value), 6) for value in xywhn[:4]]
        labels.append(item)
    summary["objects"] = len(labels)
    summary["labels"] = labels[:max_objects]
    summary["label_counts"] = counts
    if len(labels) > max_objects:
        summary["truncated"] = len(labels) - max_objects
    return summary


def image_size_for_metric(image_path: Path) -> tuple[int, int] | None:
    try:
        with load_pillow_image(image_path) as image:
            return image.size
    except Exception:
        return None


def xywhn_to_xyxy(xywhn: list[float], width: int, height: int) -> list[float] | None:
    if len(xywhn) < 4:
        return None
    x_center, y_center, box_w, box_h = [float(value) for value in xywhn[:4]]
    x1 = (x_center - box_w / 2.0) * width
    y1 = (y_center - box_h / 2.0) * height
    x2 = (x_center + box_w / 2.0) * width
    y2 = (y_center + box_h / 2.0) * height
    return valid_xyxy([x1, y1, x2, y2])


def ground_truth_records_for_metric(image_path: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    label_path = label_path_for_image(image_path)
    image_size = image_size_for_metric(image_path)
    if not label_path.exists() or image_size is None:
        return []
    width, height = image_size
    records: list[dict[str, Any]] = []
    for index, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parsed = parse_label_line(raw_line)
        if parsed is None:
            continue
        class_id, xywhn, segment = parsed
        bbox = None
        if len(segment) >= 6:
            polygon = polygon_points_from_segment(segment, width, height)
            bbox = polygon_bbox_xyxy(polygon)
        if bbox is None:
            bbox = xywhn_to_xyxy([float(value) for value in xywhn[:4]], width, height)
        if bbox is None:
            continue
        records.append(
            {
                "image_id": str(image_path.resolve()),
                "target_index": index,
                "class_id": class_id,
                "label": names.get(class_id, str(class_id)),
                "bbox_xyxy": bbox,
            }
        )
    return records


def parse_label_line(raw_line: str) -> tuple[int, list[float], list[float]] | None:
    parts = raw_line.strip().split()
    if len(parts) < 5:
        return None
    try:
        class_id = int(float(parts[0]))
        coords = [float(value) for value in parts[1:]]
    except Exception:
        return None
    return class_id, coords[:4], coords[4:]


def ground_truth_classification_records_for_metric(image_path: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    label_path = label_path_for_image(image_path)
    if not label_path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for raw_line in label_path.read_text(encoding="utf-8").splitlines():
        parsed = parse_label_line(raw_line)
        if parsed is None:
            continue
        class_id, _, _ = parsed
        key = (class_id, str(image_path.resolve()))
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "image_id": str(image_path.resolve()),
                "class_id": class_id,
                "label": names.get(class_id, str(class_id)),
            }
        )
    return records


def polygon_points_from_segment(segment: list[float], width: int, height: int) -> list[list[float]]:
    points: list[list[float]] = []
    usable = len(segment) - (len(segment) % 2)
    for idx in range(0, usable, 2):
        x = max(0.0, min(float(segment[idx]) * width, float(width)))
        y = max(0.0, min(float(segment[idx + 1]) * height, float(height)))
        points.append([round(x, 6), round(y, 6)])
    return points


def polygon_bbox_xyxy(points: list[list[float]]) -> list[float] | None:
    if len(points) < 3:
        return None
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return valid_xyxy([min(xs), min(ys), max(xs), max(ys)])


def polygon_area(points: list[list[float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for idx, point in enumerate(points):
        nxt = points[(idx + 1) % len(points)]
        area += point[0] * nxt[1] - nxt[0] * point[1]
    return abs(area) * 0.5


def point_in_polygon(point: tuple[float, float], polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False
    x, y = point
    inside = False
    j = len(polygon) - 1
    for i, current in enumerate(polygon):
        xi, yi = current
        xj, yj = polygon[j]
        intersects = ((yi > y) != (yj > y)) and (
            x < (xj - xi) * (y - yi) / ((yj - yi) or 1e-9) + xi
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def polygon_iou_approx(a: list[list[float]], b: list[list[float]], samples: int = 24) -> float:
    bbox_a = polygon_bbox_xyxy(a)
    bbox_b = polygon_bbox_xyxy(b)
    if bbox_a is None or bbox_b is None:
        return 0.0
    inter_x1 = max(bbox_a[0], bbox_b[0])
    inter_y1 = max(bbox_a[1], bbox_b[1])
    inter_x2 = min(bbox_a[2], bbox_b[2])
    inter_y2 = min(bbox_a[3], bbox_b[3])
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    inter_box_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    if inter_box_area <= 0:
        return 0.0
    inside_both = 0
    total = 0
    for xi in range(samples):
        for yi in range(samples):
            px = inter_x1 + (xi + 0.5) * (inter_x2 - inter_x1) / samples
            py = inter_y1 + (yi + 0.5) * (inter_y2 - inter_y1) / samples
            total += 1
            if point_in_polygon((px, py), a) and point_in_polygon((px, py), b):
                inside_both += 1
    intersection = inter_box_area * inside_both / total if total else 0.0
    area_a = polygon_area(a)
    area_b = polygon_area(b)
    union = area_a + area_b - intersection
    return round(intersection / union, 6) if union > 0 else 0.0


def ground_truth_segmentation_records_for_metric(image_path: Path, names: dict[int, str]) -> list[dict[str, Any]]:
    label_path = label_path_for_image(image_path)
    image_size = image_size_for_metric(image_path)
    if not label_path.exists() or image_size is None:
        return []
    width, height = image_size
    records: list[dict[str, Any]] = []
    for index, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines()):
        parsed = parse_label_line(raw_line)
        if parsed is None:
            continue
        class_id, _, segment = parsed
        if len(segment) < 6:
            continue
        polygon = polygon_points_from_segment(segment, width, height)
        bbox = polygon_bbox_xyxy(polygon)
        if bbox is None:
            continue
        records.append(
            {
                "image_id": str(image_path.resolve()),
                "target_index": index,
                "class_id": class_id,
                "label": names.get(class_id, str(class_id)),
                "polygon_xy": polygon,
                "bbox_xyxy": bbox,
            }
        )
    return records


def normalize_global_classification_items(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    items = verdict.get("global_classification", []) if isinstance(verdict, dict) else []
    normalized: list[dict[str, Any]] = []
    for item in as_list(items):
        if not isinstance(item, dict):
            continue
        class_id = coerce_int(item.get("class_id"))
        label = item.get("label")
        confidence = coerce_float(item.get("confidence"))
        if class_id is None and not label:
            continue
        normalized.append(
            {
                "class_id": class_id,
                "label": str(label) if label is not None else str(class_id),
                "confidence": round(confidence, 6) if confidence is not None else None,
            }
        )
    normalized.sort(key=lambda entry: float(entry.get("confidence") or 0.0), reverse=True)
    return normalized


def normalize_segmentation_proposals(verdict: dict[str, Any]) -> list[dict[str, Any]]:
    items = verdict.get("vlm_segmentation", []) if isinstance(verdict, dict) else []
    normalized: list[dict[str, Any]] = []
    for item in as_list(items):
        if not isinstance(item, dict):
            continue
        class_id = coerce_int(item.get("class_id"))
        label = item.get("label")
        bbox = valid_xyxy(item.get("bbox_xyxy"))
        polygon_raw = item.get("polygon_xy")
        polygon: list[list[float]] = []
        if isinstance(polygon_raw, list):
            for point in polygon_raw:
                if (
                    isinstance(point, (list, tuple))
                    and len(point) >= 2
                    and isinstance(point[0], (int, float))
                    and isinstance(point[1], (int, float))
                ):
                    polygon.append([round(float(point[0]), 6), round(float(point[1]), 6)])
        if bbox is None and polygon:
            bbox = polygon_bbox_xyxy(polygon)
        if class_id is None or bbox is None:
            continue
        normalized.append(
            {
                "proposal_id": item.get("proposal_id"),
                "class_id": class_id,
                "label": str(label) if label is not None else str(class_id),
                "bbox_xyxy": bbox,
                "polygon_xy": polygon,
                "mask_quality": item.get("mask_quality"),
            }
        )
    return normalized


def yolo_prediction_records_for_metric(image_path: Path, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in normalize_detection_boxes(detections):
        class_id = coerce_int(item.get("class_id"))
        confidence = coerce_float(item.get("confidence"))
        bbox = valid_xyxy(item.get("bbox_xyxy"))
        if class_id is None or confidence is None or bbox is None:
            continue
        records.append(
            {
                "image_id": str(image_path.resolve()),
                "source": "yolo",
                "index": item.get("index"),
                "class_id": class_id,
                "label": item.get("label"),
                "confidence": round(confidence, 6),
                "bbox_xyxy": bbox,
            }
        )
    return records


def fused_prediction_records_for_metric(image_path: Path, fusion_preview: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not isinstance(fusion_preview, dict):
        return records
    for item in fusion_preview.get("predictions", []) or []:
        if not isinstance(item, dict):
            continue
        class_id = coerce_int(item.get("class_id"))
        confidence = coerce_float(item.get("confidence"))
        bbox = valid_xyxy(item.get("bbox_xyxy"))
        if class_id is None or confidence is None or bbox is None:
            continue
        records.append(
            {
                "image_id": str(image_path.resolve()),
                "source": item.get("source", "fused"),
                "action": item.get("action"),
                "index": item.get("index"),
                "proposal_id": item.get("proposal_id"),
                "class_id": class_id,
                "label": item.get("label"),
                "confidence": round(confidence, 6),
                "bbox_xyxy": bbox,
            }
        )
    return records


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls or not precisions:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(len(mrec) - 1):
        if mrec[idx + 1] != mrec[idx]:
            ap += (mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]
    return round(ap, 6)


def match_class_predictions(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    class_id: int,
    iou_threshold: float,
) -> dict[str, Any]:
    class_predictions = sorted(
        [item for item in predictions if coerce_int(item.get("class_id")) == class_id],
        key=lambda item: float(item.get("confidence") or 0.0),
        reverse=True,
    )
    class_ground_truth = [item for item in ground_truth if coerce_int(item.get("class_id")) == class_id]
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    for item in class_ground_truth:
        gt_by_image.setdefault(str(item.get("image_id")), []).append(item)
    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for prediction in class_predictions:
        image_id = str(prediction.get("image_id"))
        best_iou = 0.0
        best_gt_index: int | None = None
        for gt_index, target in enumerate(gt_by_image.get(image_id, [])):
            if gt_index in matched.setdefault(image_id, set()):
                continue
            iou = box_iou_xyxy(prediction["bbox_xyxy"], target["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            matched[image_id].add(best_gt_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    tp_total = sum(tp_flags)
    fp_total = sum(fp_flags)
    fn_total = max(0, len(class_ground_truth) - tp_total)
    recalls: list[float] = []
    precisions: list[float] = []
    tp_cum = 0
    fp_cum = 0
    for tp, fp in zip(tp_flags, fp_flags):
        tp_cum += tp
        fp_cum += fp
        recalls.append(tp_cum / len(class_ground_truth) if class_ground_truth else 0.0)
        precisions.append(tp_cum / (tp_cum + fp_cum) if tp_cum + fp_cum else 0.0)
    return {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "ap": average_precision(recalls, precisions) if class_ground_truth else None,
    }


def evaluate_detection_metric_preview(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    iou_thresholds: list[float] | None = None,
) -> dict[str, Any]:
    thresholds = iou_thresholds or [round(0.5 + 0.05 * idx, 2) for idx in range(10)]
    if not ground_truth:
        return {
            "status": "skipped",
            "reason": "ground_truth_unavailable",
            "predictions": len(predictions),
            "ground_truth": 0,
        }
    gt_classes = sorted({int(item["class_id"]) for item in ground_truth if coerce_int(item.get("class_id")) is not None})
    pred_classes = sorted({int(item["class_id"]) for item in predictions if coerce_int(item.get("class_id")) is not None})
    all_classes = sorted(set(gt_classes) | set(pred_classes))
    per_threshold: dict[str, dict[str, Any]] = {}
    ap_by_threshold: dict[float, list[float]] = {threshold: [] for threshold in thresholds}
    counts_at_50 = {"tp": 0, "fp": 0, "fn": 0}
    per_class_50: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        threshold_counts = {"tp": 0, "fp": 0, "fn": 0}
        for class_id in all_classes:
            result = match_class_predictions(predictions, ground_truth, class_id=class_id, iou_threshold=threshold)
            threshold_counts["tp"] += int(result["tp"])
            threshold_counts["fp"] += int(result["fp"])
            threshold_counts["fn"] += int(result["fn"])
            if class_id in gt_classes:
                ap_by_threshold[threshold].append(float(result["ap"] or 0.0))
            if abs(threshold - 0.5) < 1e-9:
                per_class_50[str(class_id)] = {
                    "tp": int(result["tp"]),
                    "fp": int(result["fp"]),
                    "fn": int(result["fn"]),
                    "ap": result["ap"],
                }
        precision = threshold_counts["tp"] / (threshold_counts["tp"] + threshold_counts["fp"]) if threshold_counts["tp"] + threshold_counts["fp"] else 0.0
        recall = threshold_counts["tp"] / (threshold_counts["tp"] + threshold_counts["fn"]) if threshold_counts["tp"] + threshold_counts["fn"] else 0.0
        per_threshold[f"{threshold:.2f}"] = {
            **threshold_counts,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "map": round(sum(ap_by_threshold[threshold]) / len(ap_by_threshold[threshold]), 6) if ap_by_threshold[threshold] else 0.0,
        }
        if abs(threshold - 0.5) < 1e-9:
            counts_at_50 = threshold_counts
    precision_50 = counts_at_50["tp"] / (counts_at_50["tp"] + counts_at_50["fp"]) if counts_at_50["tp"] + counts_at_50["fp"] else 0.0
    recall_50 = counts_at_50["tp"] / (counts_at_50["tp"] + counts_at_50["fn"]) if counts_at_50["tp"] + counts_at_50["fn"] else 0.0
    map50 = per_threshold.get("0.50", {}).get("map", 0.0)
    maps = [value["map"] for value in per_threshold.values()]
    map50_95 = round(sum(maps) / len(maps), 6) if maps else 0.0
    f1 = (2 * precision_50 * recall_50 / (precision_50 + recall_50)) if precision_50 + recall_50 else 0.0
    return {
        "status": "ok",
        "basis": "yolo_label_metric_preview",
        "predictions": len(predictions),
        "ground_truth": len(ground_truth),
        "classes_with_ground_truth": len(gt_classes),
        "precision": round(precision_50, 6),
        "recall": round(recall_50, 6),
        "f1": round(f1, 6),
        "map50": round(float(map50), 6),
        "map50_95": map50_95,
        "counts_at_iou50": counts_at_50,
        "per_threshold": per_threshold,
        "per_class_iou50": per_class_50,
    }


def metric_delta(fused: dict[str, Any], yolo: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "map50", "map50_95"):
        if isinstance(fused.get(key), (int, float)) and isinstance(yolo.get(key), (int, float)):
            delta[key] = round(float(fused[key]) - float(yolo[key]), 6)
    if "map50_95" in delta:
        delta["direction"] = "improved" if delta["map50_95"] > 0 else ("regressed" if delta["map50_95"] < 0 else "unchanged")
    return delta


def evaluate_classification_metric_preview(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> dict[str, Any]:
    if not ground_truth:
        return {"status": "skipped", "reason": "ground_truth_unavailable", "ground_truth": 0, "predictions": len(predictions)}
    gt_labels: dict[str, set[int]] = {}
    for item in ground_truth:
        class_id = coerce_int(item.get("class_id"))
        if class_id is None:
            continue
        gt_labels.setdefault(str(item.get("image_id")), set()).add(int(class_id))
    pred_by_image: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        image_id = str(item.get("image_id"))
        pred_by_image.setdefault(image_id, []).append(item)
    total_images = len(gt_labels)
    exact_match = 0
    top1_correct = 0
    tp = 0
    fp = 0
    fn = 0
    for image_id, gt_classes in gt_labels.items():
        preds = sorted(pred_by_image.get(image_id, []), key=lambda entry: float(entry.get("confidence") or 0.0), reverse=True)
        pred_classes = {int(item["class_id"]) for item in preds if coerce_int(item.get("class_id")) is not None}
        if pred_classes == gt_classes:
            exact_match += 1
        if preds:
            top1 = coerce_int(preds[0].get("class_id"))
            if top1 is not None and top1 in gt_classes:
                top1_correct += 1
        tp += len(pred_classes & gt_classes)
        fp += len(pred_classes - gt_classes)
        fn += len(gt_classes - pred_classes)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "status": "ok",
        "basis": "label_presence_preview",
        "ground_truth": len(ground_truth),
        "predictions": len(predictions),
        "images_with_ground_truth": total_images,
        "top1_accuracy": round(top1_correct / total_images if total_images else 0.0, 6),
        "exact_set_accuracy": round(exact_match / total_images if total_images else 0.0, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "counts": {"tp": tp, "fp": fp, "fn": fn},
    }


def classification_metric_delta(fused: dict[str, Any], yolo: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("top1_accuracy", "exact_set_accuracy", "precision", "recall", "f1"):
        if isinstance(fused.get(key), (int, float)) and isinstance(yolo.get(key), (int, float)):
            delta[key] = round(float(fused[key]) - float(yolo[key]), 6)
    if "top1_accuracy" in delta:
        delta["direction"] = "improved" if delta["top1_accuracy"] > 0 else ("regressed" if delta["top1_accuracy"] < 0 else "unchanged")
    return delta


def evaluate_segmentation_metric_preview(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    if not ground_truth:
        return {"status": "skipped", "reason": "ground_truth_unavailable", "ground_truth": 0, "predictions": len(predictions)}
    predictions_sorted = sorted(
        predictions,
        key=lambda entry: float(entry.get("confidence") or 0.0),
        reverse=True,
    )
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    for item in ground_truth:
        gt_by_image.setdefault(str(item.get("image_id")), []).append(item)
    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    tp = 0
    fp = 0
    iou_sum = 0.0
    for prediction in predictions_sorted:
        class_id = coerce_int(prediction.get("class_id"))
        polygon = prediction.get("polygon_xy")
        image_id = str(prediction.get("image_id"))
        if class_id is None or not isinstance(polygon, list) or len(polygon) < 3:
            fp += 1
            continue
        best_iou = 0.0
        best_index: int | None = None
        for gt_index, target in enumerate(gt_by_image.get(image_id, [])):
            if gt_index in matched.setdefault(image_id, set()):
                continue
            if coerce_int(target.get("class_id")) != class_id:
                continue
            iou = polygon_iou_approx(polygon, target.get("polygon_xy", []))
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_index is not None and best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            tp += 1
            iou_sum += best_iou
        else:
            fp += 1
    fn = 0
    for image_id, targets in gt_by_image.items():
        fn += max(0, len(targets) - len(matched.get(image_id, set())))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    mean_iou = iou_sum / tp if tp else 0.0
    return {
        "status": "ok",
        "basis": "polygon_iou_preview",
        "ground_truth": len(ground_truth),
        "predictions": len(predictions),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "mean_iou": round(mean_iou, 6),
        "mask_ap50_proxy": round(precision, 6),
        "counts": {"tp": tp, "fp": fp, "fn": fn},
    }


def segmentation_metric_delta(fused: dict[str, Any], yolo: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "mean_iou", "mask_ap50_proxy"):
        if isinstance(fused.get(key), (int, float)) and isinstance(yolo.get(key), (int, float)):
            delta[key] = round(float(fused[key]) - float(yolo[key]), 6)
    if "mask_ap50_proxy" in delta:
        delta["direction"] = "improved" if delta["mask_ap50_proxy"] > 0 else ("regressed" if delta["mask_ap50_proxy"] < 0 else "unchanged")
    return delta


def yolo_classification_predictions_for_metric(image_path: Path, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    for item in normalize_detection_boxes(detections):
        class_id = coerce_int(item.get("class_id"))
        confidence = coerce_float(item.get("confidence"))
        if class_id is None:
            continue
        current = records.get(class_id)
        if current is None or float(confidence or 0.0) > float(current.get("confidence") or 0.0):
            records[class_id] = {
                "image_id": str(image_path.resolve()),
                "class_id": class_id,
                "label": item.get("label"),
                "confidence": round(confidence, 6) if confidence is not None else 0.0,
                "source": "yolo",
            }
    return list(records.values())


def fused_classification_predictions_for_metric(image_path: Path, verdict: dict[str, Any], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = normalize_global_classification_items(verdict)
    if records:
        return [
            {
                "image_id": str(image_path.resolve()),
                "class_id": item.get("class_id"),
                "label": item.get("label"),
                "confidence": item.get("confidence") if item.get("confidence") is not None else 0.0,
                "source": "vlm",
            }
            for item in records
        ]
    return yolo_classification_predictions_for_metric(image_path, detections)


def yolo_segmentation_predictions_for_metric(image_path: Path, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in normalize_detection_boxes(detections):
        class_id = coerce_int(item.get("class_id"))
        confidence = coerce_float(item.get("confidence"))
        bbox = valid_xyxy(item.get("bbox_xyxy"))
        if class_id is None or bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        records.append(
            {
                "image_id": str(image_path.resolve()),
                "class_id": class_id,
                "label": item.get("label"),
                "confidence": round(confidence, 6) if confidence is not None else 0.0,
                "polygon_xy": polygon,
                "bbox_xyxy": bbox,
                "source": "yolo_box_proxy",
            }
        )
    return records


def fused_segmentation_predictions_for_metric(image_path: Path, verdict: dict[str, Any], detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = normalize_segmentation_proposals(verdict)
    if records:
        return [
            {
                "image_id": str(image_path.resolve()),
                "class_id": item.get("class_id"),
                "label": item.get("label"),
                "confidence": 1.0,
                "polygon_xy": item.get("polygon_xy"),
                "bbox_xyxy": item.get("bbox_xyxy"),
                "source": "vlm_segmentation",
            }
            for item in records
        ]
    return yolo_segmentation_predictions_for_metric(image_path, detections)


def detection_label_counts(detections: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for item in detections:
        for detection in item.get("detections", []) or []:
            label = detection.get("label")
            if label is None:
                continue
            counts[str(label)] = counts.get(str(label), 0) + 1
    return counts


def build_multimodal_evaluation_prompt(
    base_prompt: str,
    image_path: Path,
    index: int,
    total: int,
    ground_truth: dict[str, Any],
    *,
    include_ground_truth: bool,
) -> str:
    prompt = (
        f"{base_prompt}\n\n"
        f"Evaluation image {index + 1}/{total}: {image_path.name}. "
        "Focus on detector agreement, obvious false positives, likely missed objects, duplicates, and uncertainty."
    )
    if include_ground_truth:
        prompt += "\n\nGround-truth labels for post-hoc comparison:\n" + json.dumps(ground_truth, ensure_ascii=False, indent=2)
    return prompt


def multimodal_prompt_from_request(request: dict[str, Any], params: dict[str, Any]) -> str:
    return (
        request.get("inputs", {}).get("prompt")
        or params.get("prompt")
        or params.get("question")
        or "Explain the scene, verify the YOLO detections, and identify important missed or uncertain visual evidence."
    )


def run_multimodal_infer(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    yolo_params, multimodal_params = split_yolo_and_multimodal_params(params)
    multimodal_params = apply_open_world_assist_profile_defaults(multimodal_params)
    source = request["inputs"].get("source") or yolo_params.pop("source", None) or multimodal_params.get("source")
    if source is None:
        raise ValueError("`inputs.source` is required for yolo.multimodal.infer.")
    prompt = multimodal_prompt_from_request(request, multimodal_params)
    provider_cfg = openai_config(multimodal_params)
    if provider_cfg.get("api_family") != "openai-compatible":
        raise ValueError(f"Unsupported multimodal provider: {provider_cfg['provider']}")

    max_items = int(yolo_params.pop("max_items", multimodal_params.get("max_reasoning_items", 3)))
    max_boxes = int(multimodal_params.get("max_reasoning_boxes", 20))
    thinking_with_image = parse_bool(multimodal_params.get("thinking_with_image"), True)
    structured_output = parse_bool(multimodal_params.get("structured_output"), True)
    prompt_template = multimodal_params.get("prompt_template")
    resolved_prompt_template = effective_prompt_template_name(provider_cfg, prompt_template, multimodal_params, prompt)
    default_max_output_tokens = default_multimodal_max_output_tokens(
        provider_cfg,
        prompt_template,
        multimodal_params,
        structured_output=structured_output,
        user_prompt=prompt,
    )
    max_output_tokens = int(multimodal_params.get("max_output_tokens", default_max_output_tokens))
    use_marked_image = parse_bool(multimodal_params.get("use_marked_image"), bool(resolved_prompt_template))
    visual_search_mode = str(multimodal_params.get("visual_search_mode") or "auto")
    fusion_mode = str(multimodal_params.get("fusion_mode") or "preview")
    device_selection = resolve_device_selection(request, yolo_params)
    yolo_params, chosen_device, auto_completed = apply_runtime_defaults(request, yolo_params, purpose="predict")
    effective_request = deepcopy(request)
    effective_request["params"] = yolo_params
    effective_request["inputs"]["source"] = source

    method = str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection"))
    if is_dry_run(request):
        return plan_response(
            effective_request,
            "multimodal inference dry run prepared",
            "orchestrator",
            "yolo.multimodal.infer",
            params={
                "stages": [
                    {"name": "yolo_predict", "executor": "python_api", "target": "YOLO(...).predict", "params": yolo_params},
                    {
                        "name": "vlm_reasoning",
                        "executor": "openai.compatible",
                        "provider": provider_cfg["provider"],
                        "model": provider_cfg["vlm_model"],
                        "api_mode": provider_cfg["api_mode"],
                        "image": "input_image" if thinking_with_image else "text_only",
                        "method": method,
                        "structured_output": structured_output,
                        "prompt_template": resolved_prompt_template,
                        "marked_image": use_marked_image,
                    },
                    {
                        "name": "visual_search_crop_pass",
                        "executor": "openai.compatible",
                        "mode": visual_search_mode,
                        "enabled": visual_search_mode.lower() not in {"off", "none", "false", "0"},
                    },
                    {
                        "name": "fusion_preview",
                        "executor": "python",
                        "strategy": "metric_safe_v1",
                        "mode": fusion_mode,
                    },
                    {
                        "name": "llm_refine",
                        "executor": "openai.compatible",
                        "provider": provider_cfg["provider"],
                        "model": provider_cfg["llm_model"],
                        "api_mode": provider_cfg["api_mode"],
                        "enabled": bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine")),
                    },
                ],
                "prompt": prompt,
            },
            extra={
                "environment": collect_environment_report(effective_request, selected_device=chosen_device),
                "auto_completed": auto_completed,
                "multimodal": {
                    "provider": provider_cfg["provider"],
                    "vlm_model": provider_cfg["vlm_model"],
                    "llm_model": provider_cfg["llm_model"],
                    "api_mode": provider_cfg["api_mode"],
                    "api_key_env": provider_cfg["api_key_env"],
                    "api_key_present": provider_cfg["api_key_present"],
                    "method": method,
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                    "prompt_template": resolved_prompt_template,
                    "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
                    "open_world_filters": {
                        "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                        "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                        "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                        "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
                    },
                    "use_marked_image": use_marked_image,
                    "visual_search_mode": visual_search_mode,
                    "fusion_mode": fusion_mode,
                },
            },
        )

    detections: list[dict[str, Any]]
    save_dir = None
    yolo_error = None
    if bool(multimodal_params.get("skip_yolo", False)):
        detections = json_safe(request["inputs"].get("detections") or multimodal_params.get("detections") or [])
    else:
        try:
            model = build_model(effective_request)
            results = model.predict(source=source, **yolo_params)
        except Exception as exc:
            if device_selection["source"] == "auto" and chosen_device not in (None, "cpu"):
                retry_params = replace_cli_device(yolo_params, "cpu")
                effective_request["params"] = retry_params
                try:
                    model = build_model(effective_request)
                    results = model.predict(source=source, **retry_params)
                    yolo_params = retry_params
                    chosen_device = "cpu"
                    auto_completed["device"] = "cpu"
                    auto_completed["device_source"] = "recovery"
                except Exception:
                    raise exc
            else:
                raise
        save_dir = getattr(model.predictor, "save_dir", None)
        detections = summarize_results_for_reasoning(results, max_items=max_items, max_boxes=max_boxes)

    image_ref = image_source_for_openai(source, detections)
    vlm_result: dict[str, Any]
    llm_result: dict[str, Any] | None = None
    image_meta: dict[str, Any] = {
        "requested": image_ref,
        "thinking_with_image": thinking_with_image,
        "attached": False,
    }
    visual_artifacts: list[dict[str, Any]] = []
    fusion_artifacts: list[dict[str, Any]] = []
    visual_search_passes: list[dict[str, Any]] = []
    marked_meta: dict[str, Any] | None = None
    marked_error: dict[str, Any] | None = None
    if thinking_with_image and image_ref is None:
        vlm_result = {
            "status": "blocked",
            "provider": "openai",
            "summary": "No image source was available for multimodal reasoning.",
        }
    elif not provider_cfg["api_key_present"]:
        vlm_result = {
            "status": "blocked",
            "provider": "openai",
            "summary": "OPENAI_API_KEY is not set; multimodal reasoning was skipped.",
            "api_key_env": provider_cfg["api_key_env"],
        }
    else:
        try:
            encoded_image = None
            if thinking_with_image and image_ref is not None:
                reasoning_image_ref = image_ref
                if use_marked_image and not str(image_ref).startswith(("http://", "https://", "data:image/")):
                    try:
                        marked = render_marked_image(
                            image_ref,
                            detections,
                            output_dir=ensure_manifest_dir(request) / "visual-search",
                            prefix=slugify(Path(str(image_ref)).stem),
                        )
                        reasoning_image_ref = marked["path"]
                        marked_meta = marked
                        visual_artifacts.append({"kind": "marked_image", "path": marked["path"]})
                    except Exception as mark_exc:
                        marked_error = {"type": type(mark_exc).__name__, "message": str(mark_exc)}
                encoded_image = encode_image_reference_for_openai(
                    reasoning_image_ref,
                    max_bytes=int(multimodal_params.get("max_image_bytes", 20_000_000)),
                )
                image_meta = {k: v for k, v in encoded_image.items() if k != "image_url"}
                image_meta["requested"] = image_ref
                image_meta["reasoning_input"] = reasoning_image_ref
                if marked_meta:
                    image_meta["marked"] = marked_meta
                if marked_error:
                    image_meta["marked_error"] = marked_error
                image_meta["thinking_with_image"] = True
                image_meta["attached"] = True
            user_text = build_thinking_with_image_prompt(
                prompt,
                detections,
                method=method,
                thinking_with_image=thinking_with_image,
                structured_output=structured_output,
                prompt_template=str(resolved_prompt_template) if resolved_prompt_template is not None else None,
                prompt_dir=PROMPT_DIR,
            )
            vlm_result = call_openai_compatible(
                model=str(provider_cfg["vlm_model"]),
                user_text=user_text,
                developer_text=str(
                    multimodal_params.get("developer_prompt")
                    or multimodal_params.get("system_prompt")
                    or default_vlm_developer_prompt(resolved_prompt_template)
                ),
                image_url=encoded_image["image_url"] if encoded_image else None,
                image_detail=str(multimodal_params.get("image_detail", "auto")),
                base_url=provider_cfg["base_url"],
                provider=str(provider_cfg.get("provider", "openai")),
                api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
                api_mode=str(provider_cfg["api_mode"]),
                max_output_tokens=max_output_tokens,
                temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
            )
            vlm_result = attach_multimodal_verdict(vlm_result)
            if vlm_result.get("status") == "ok" and image_ref is not None:
                visual_search_passes, search_artifacts = runtime_run_visual_search_crop_passes(
                    image_path=image_ref,
                    base_prompt=prompt,
                    detections=detections,
                    initial_verdict=vlm_result.get("verdict", {}),
                    provider_cfg=provider_cfg,
                    multimodal_params=multimodal_params,
                    output_dir=ensure_manifest_dir(request),
                    max_output_tokens=max_output_tokens,
                    method=method,
                    resolved_path_fn=resolved_path,
                    normalize_detection_boxes_fn=normalize_detection_boxes,
                    call_openai_compatible_fn=call_openai_compatible,
                    attach_multimodal_verdict_fn=attach_multimodal_verdict,
                )
                visual_artifacts.extend(search_artifacts)
        except Exception as exc:
            vlm_result = {
                "status": "failed",
                "provider": provider_cfg.get("provider", "openai"),
                "summary": "Failed to prepare or call VLM reasoning.",
                "error": {"type": type(exc).__name__, "message": str(exc)},
            }

    llm_enabled = bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine"))
    if llm_enabled and vlm_result.get("status") == "ok":
        llm_model = str(provider_cfg.get("llm_model") or provider_cfg["vlm_model"])
        refine_prompt = (
            "Refine this YOLO + VLM inference into a compact final answer. Do not add unsupported visual claims. "
            "Return exactly one JSON object without Markdown fences. Use these keys: answer, visual_evidence, "
            "yolo_cross_check, uncertainty, recommended_next_actions. In yolo_cross_check, include arrays named "
            "confirmed, false_positives, possible_misses, duplicate_or_fragmented, and notes when applicable. "
            "If the VLM answer includes caption, global_classification, vlm_detections, vlm_segmentation, "
            "visual_search, or fusion_hints, preserve those keys and refine them conservatively.\n\n"
            f"User task:\n{prompt}\n\n"
            f"YOLO detection summary:\n{json.dumps(json_safe(detections), ensure_ascii=False, indent=2)}\n\n"
            f"VLM answer:\n{vlm_result.get('text', '')}\n\n"
            f"Parsed VLM verdict:\n{json.dumps(json_safe(vlm_result.get('verdict', {})), ensure_ascii=False, indent=2)}"
            f"\n\nVisual search crop passes:\n{json.dumps(json_safe(visual_search_passes), ensure_ascii=False, indent=2)}"
        )
        llm_result = call_openai_compatible(
            model=llm_model,
            user_text=refine_prompt,
            developer_text=default_llm_refine_developer_prompt(resolved_prompt_template),
            base_url=provider_cfg["base_url"],
            provider=str(provider_cfg.get("provider", "openai")),
            api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
            api_mode=str(provider_cfg["api_mode"]),
            max_output_tokens=max_output_tokens,
            temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
        )
        llm_result = attach_multimodal_verdict(llm_result)

    vlm_verdict = vlm_result.get("verdict", {}) if isinstance(vlm_result.get("verdict"), dict) else {}
    llm_verdict = llm_result.get("verdict", {}) if isinstance(llm_result, dict) and isinstance(llm_result.get("verdict"), dict) else {}
    fusion_verdict = merge_verdicts(vlm_verdict, llm_verdict)
    fusion_preview = fusion_build_multimodal_fusion_preview(
        detections=detections,
        verdict=fusion_verdict,
        multimodal_params=multimodal_params,
        image_path=image_ref,
        normalize_detection_boxes_fn=normalize_detection_boxes,
    )
    open_world_comparison = build_open_world_comparison_entry(
        image_path=image_ref,
        detections=detections,
        fusion_preview=fusion_preview,
        verdict=fusion_verdict,
        multimodal_params=multimodal_params,
        effective_prompt_template=str(resolved_prompt_template) if resolved_prompt_template is not None else None,
    )
    if fusion_preview.get("enabled"):
        fusion_path = ensure_manifest_dir(request) / "fusion-preview.json"
        fusion_path.write_text(json.dumps(json_safe({"image": image_ref, "fusion": fusion_preview}), ensure_ascii=False, indent=2), encoding="utf-8")
        fusion_preview["artifact"] = str(fusion_path.resolve())
        fusion_artifacts.append({"kind": "fusion_preview", "path": str(fusion_path.resolve())})

    overall_status = multimodal_overall_status(vlm_result, llm_result)
    summary_map = {
        "ok": "multimodal inference finished",
        "blocked": "YOLO inference finished, but multimodal reasoning was blocked",
        "partial": "YOLO inference finished; multimodal reasoning was incomplete",
    }
    summary = summary_map[overall_status]
    environment = collect_environment_report(effective_request, selected_device=chosen_device)
    payload = response(
        request["skill"],
        overall_status,
        summary,
        job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "python_api+openai", "device": chosen_device},
        results=detections,
        environment=environment,
        auto_completed=auto_completed,
        multimodal={
            "method": method,
            "thinking_with_image": thinking_with_image,
            "structured_output": structured_output,
            "prompt_template": prompt_template,
            "effective_prompt_template": resolved_prompt_template,
            "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
            "open_world_filters": {
                "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
            },
            "use_marked_image": use_marked_image,
            "visual_search": {"mode": visual_search_mode, "passes": visual_search_passes, "artifacts": visual_artifacts},
            "provider": {
                "name": provider_cfg["provider"],
                "base_url": provider_cfg["base_url"],
                "api_mode": provider_cfg["api_mode"],
                "api_key_env": provider_cfg["api_key_env"],
                "api_key_present": provider_cfg["api_key_present"],
            },
            "image": image_meta,
            "prompt": prompt,
            "vlm": vlm_result,
            "llm_refine": llm_result or {"status": "skipped"},
            "fusion": fusion_preview,
            "open_world_comparison": open_world_comparison,
        },
        next_actions=["yolo.predict", "yolo.val"],
    )
    if yolo_error:
        payload["yolo_error"] = yolo_error
    if save_dir:
        payload["artifacts"] = [{"kind": "directory", "path": str(Path(save_dir).resolve())}]
    if visual_artifacts or fusion_artifacts:
        payload.setdefault("artifacts", [])
        payload["artifacts"].extend(visual_artifacts + fusion_artifacts)
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_multimodal_evaluate(request: dict[str, Any]) -> dict[str, Any]:
    params = dict(request["params"])
    yolo_params, multimodal_params, evaluate_params = split_yolo_multimodal_evaluate_params(params)
    multimodal_params = apply_open_world_assist_profile_defaults(multimodal_params)
    prompt = multimodal_prompt_from_request(request, multimodal_params)
    provider_cfg = openai_config(multimodal_params)
    if provider_cfg.get("api_family") != "openai-compatible":
        raise ValueError(f"Unsupported multimodal provider: {provider_cfg['provider']}")

    thinking_with_image = parse_bool(multimodal_params.get("thinking_with_image"), True)
    structured_output = parse_bool(multimodal_params.get("structured_output"), True)
    prompt_template = multimodal_params.get("prompt_template")
    resolved_prompt_template = effective_prompt_template_name(provider_cfg, prompt_template, multimodal_params, prompt)
    default_max_output_tokens = default_multimodal_max_output_tokens(
        provider_cfg,
        prompt_template,
        multimodal_params,
        structured_output=structured_output,
        user_prompt=prompt,
    )
    max_output_tokens = int(multimodal_params.get("max_output_tokens", default_max_output_tokens))
    use_marked_image = parse_bool(multimodal_params.get("use_marked_image"), bool(resolved_prompt_template))
    visual_search_mode = str(multimodal_params.get("visual_search_mode") or "auto")
    fusion_mode = str(multimodal_params.get("fusion_mode") or "preview")
    include_ground_truth = parse_bool(
        evaluate_params.get("include_ground_truth_in_prompt", evaluate_params.get("include_ground_truth")),
        False,
    )
    run_yolo_val = parse_bool(evaluate_params.get("run_yolo_val"), False)
    continue_on_error = parse_bool(evaluate_params.get("continue_on_error"), True)
    report_name = str(evaluate_params.get("report_name") or "multimodal-evaluation.json")
    data_ref_for_baseline = request["inputs"].get("data") or evaluate_params.get("data") or yolo_params.get("data")
    device_selection = resolve_device_selection(request, yolo_params)
    yolo_params, chosen_device, auto_completed = apply_runtime_defaults(request, yolo_params, purpose="predict")
    if "verbose" not in yolo_params:
        yolo_params["verbose"] = False
        auto_completed["verbose"] = False

    effective_request = deepcopy(request)
    effective_request["params"] = yolo_params
    if request["inputs"].get("data") is not None:
        effective_request["inputs"]["data"] = request["inputs"]["data"]
    if request["inputs"].get("source") is not None:
        effective_request["inputs"]["source"] = request["inputs"]["source"]

    images, dataset_info, names = collect_multimodal_evaluation_images(request, evaluate_params, yolo_params, multimodal_params)
    effective_request["inputs"]["source"] = str(images[0].parent)
    environment = collect_environment_report(effective_request, selected_device=chosen_device)

    if is_dry_run(request):
        return plan_response(
            effective_request,
            "multimodal evaluation dry run prepared",
            "orchestrator",
            "yolo.multimodal.evaluate",
            params={
                "stages": [
                    {"name": "collect_images", "executor": "dataset_or_source_resolver"},
                    {"name": "yolo_predict_batch", "executor": "python_api", "target": "YOLO(...).predict"},
                    {"name": "marked_image_batch", "executor": "pillow", "enabled": use_marked_image},
                    {"name": "vlm_reasoning_batch", "executor": "openai.compatible"},
                    {"name": "visual_search_crop_pass", "executor": "openai.compatible", "mode": visual_search_mode},
                    {"name": "fusion_preview", "executor": "python", "strategy": "metric_safe_v1", "mode": fusion_mode},
                    {"name": "aggregate_report", "executor": "python"},
                ],
                "sample_count": len(images),
                "sample_images": [str(path) for path in images[:10]],
                "split": dataset_info.get("split"),
                "run_yolo_val": run_yolo_val,
                "include_ground_truth_in_prompt": include_ground_truth,
                "prompt": prompt,
            },
            extra={
                "environment": environment,
                "auto_completed": auto_completed,
                "multimodal": {
                    "provider": provider_cfg["provider"],
                    "vlm_model": provider_cfg["vlm_model"],
                    "llm_model": provider_cfg["llm_model"],
                    "api_mode": provider_cfg["api_mode"],
                    "api_key_env": provider_cfg["api_key_env"],
                    "api_key_present": provider_cfg["api_key_present"],
                    "method": str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection")),
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                    "prompt_template": prompt_template,
                    "effective_prompt_template": resolved_prompt_template,
                    "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
                    "use_marked_image": use_marked_image,
                    "visual_search_mode": visual_search_mode,
                    "fusion_mode": fusion_mode,
                    "open_world_filters": {
                        "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                        "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                        "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                        "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
                    },
                    "dataset": dataset_info,
                },
            },
        )

    model = build_model(effective_request)
    selected_device = chosen_device
    items: list[dict[str, Any]] = []
    save_dir = None
    method = str(multimodal_params.get("method") or ("thinking-with-image" if thinking_with_image else "detector-text-reflection"))
    llm_enabled = bool(provider_cfg.get("llm_model") or multimodal_params.get("enable_llm_refine"))

    for index, image_path in enumerate(images):
        image_item: dict[str, Any] = {
            "path": str(image_path),
            "index": index,
            "ground_truth": read_ground_truth_summary(image_path, names),
        }
        image_prompt = build_multimodal_evaluation_prompt(
            prompt,
            image_path,
            index,
            len(images),
            image_item["ground_truth"],
            include_ground_truth=include_ground_truth,
        )
        yolo_prediction = None
        try:
            yolo_prediction = model.predict(source=str(image_path), **yolo_params)
        except Exception as exc:
            if device_selection["source"] == "auto" and selected_device not in (None, "cpu") and request.get("runtime", {}).get("allow_device_fallback", True):
                retry_params = replace_cli_device(yolo_params, "cpu")
                effective_request["params"] = retry_params
                try:
                    model = build_model(effective_request)
                    yolo_prediction = model.predict(source=str(image_path), **retry_params)
                    yolo_params = retry_params
                    selected_device = "cpu"
                    auto_completed["device"] = "cpu"
                    auto_completed["device_source"] = "recovery"
                except Exception as retry_exc:
                    if not continue_on_error:
                        raise retry_exc
                    image_item["status"] = "failed"
                    image_item["error"] = {"type": type(retry_exc).__name__, "message": str(retry_exc)}
                    items.append(image_item)
                    continue
            elif continue_on_error:
                image_item["status"] = "failed"
                image_item["error"] = {"type": type(exc).__name__, "message": str(exc)}
                items.append(image_item)
                continue
            else:
                raise

        detections = summarize_results_for_reasoning(yolo_prediction, max_items=1, max_boxes=int(multimodal_params.get("max_reasoning_boxes", 20)))
        detection_summary = detections[0] if detections else {"path": str(image_path), "speed": {}, "detections": []}
        image_ref = image_source_for_openai(str(image_path), detections)
        image_meta: dict[str, Any] = {
            "requested": image_ref,
            "thinking_with_image": thinking_with_image,
            "attached": False,
        }
        item_visual_artifacts: list[dict[str, Any]] = []
        visual_search_passes: list[dict[str, Any]] = []
        marked_meta: dict[str, Any] | None = None
        marked_error: dict[str, Any] | None = None
        vlm_result: dict[str, Any]
        llm_result: dict[str, Any] | None = None
        if thinking_with_image and image_ref is None:
            vlm_result = {
                "status": "blocked",
                "provider": provider_cfg.get("provider", "openai"),
                "summary": "No image source was available for multimodal reasoning.",
            }
        elif not provider_cfg["api_key_present"]:
            vlm_result = {
                "status": "blocked",
                "provider": provider_cfg.get("provider", "openai"),
                "summary": f"{provider_cfg['api_key_env']} is not set; multimodal reasoning was skipped.",
                "api_key_env": provider_cfg["api_key_env"],
            }
        else:
            try:
                encoded_image = None
                if thinking_with_image and image_ref is not None:
                    reasoning_image_ref = image_ref
                    if use_marked_image and not str(image_ref).startswith(("http://", "https://", "data:image/")):
                        try:
                            marked = render_marked_image(
                                image_ref,
                                detections,
                                output_dir=ensure_manifest_dir(request) / "visual-search",
                                prefix=f"{index:04d}-{slugify(Path(str(image_ref)).stem)}",
                            )
                            reasoning_image_ref = marked["path"]
                            marked_meta = marked
                            item_visual_artifacts.append({"kind": "marked_image", "path": marked["path"]})
                        except Exception as mark_exc:
                            marked_error = {"type": type(mark_exc).__name__, "message": str(mark_exc)}
                    encoded_image = encode_image_reference_for_openai(
                        reasoning_image_ref,
                        max_bytes=int(multimodal_params.get("max_image_bytes", 20_000_000)),
                    )
                    image_meta = {k: v for k, v in encoded_image.items() if k != "image_url"}
                    image_meta["requested"] = image_ref
                    image_meta["reasoning_input"] = reasoning_image_ref
                    if marked_meta:
                        image_meta["marked"] = marked_meta
                    if marked_error:
                        image_meta["marked_error"] = marked_error
                    image_meta["thinking_with_image"] = True
                    image_meta["attached"] = True
                user_text = build_thinking_with_image_prompt(
                    image_prompt,
                    detections,
                    method=method,
                    thinking_with_image=thinking_with_image,
                    structured_output=structured_output,
                    prompt_template=str(resolved_prompt_template) if resolved_prompt_template is not None else None,
                    prompt_dir=PROMPT_DIR,
                )
                vlm_result = call_openai_compatible(
                    model=str(provider_cfg["vlm_model"]),
                    user_text=user_text,
                    developer_text=str(
                        multimodal_params.get("developer_prompt")
                        or multimodal_params.get("system_prompt")
                        or default_vlm_developer_prompt(resolved_prompt_template)
                    ),
                    image_url=encoded_image["image_url"] if encoded_image else None,
                    image_detail=str(multimodal_params.get("image_detail", "auto")),
                    base_url=provider_cfg["base_url"],
                    provider=str(provider_cfg.get("provider", "openai")),
                    api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
                    api_mode=str(provider_cfg["api_mode"]),
                    max_output_tokens=max_output_tokens,
                    temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
                )
                vlm_result = attach_multimodal_verdict(vlm_result)
                if vlm_result.get("status") == "ok" and image_ref is not None:
                    visual_search_passes, search_artifacts = runtime_run_visual_search_crop_passes(
                        image_path=image_ref,
                        base_prompt=image_prompt,
                        detections=detections,
                        initial_verdict=vlm_result.get("verdict", {}),
                        provider_cfg=provider_cfg,
                        multimodal_params=multimodal_params,
                        output_dir=ensure_manifest_dir(request),
                        max_output_tokens=max_output_tokens,
                        method=method,
                        resolved_path_fn=resolved_path,
                        normalize_detection_boxes_fn=normalize_detection_boxes,
                        call_openai_compatible_fn=call_openai_compatible,
                        attach_multimodal_verdict_fn=attach_multimodal_verdict,
                    )
                    item_visual_artifacts.extend(search_artifacts)
            except Exception as exc:
                vlm_result = {
                    "status": "failed",
                    "provider": provider_cfg.get("provider", "openai"),
                    "summary": "Failed to prepare or call VLM reasoning.",
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }

        if llm_enabled and vlm_result.get("status") == "ok":
            llm_model = str(provider_cfg.get("llm_model") or provider_cfg["vlm_model"])
            refine_prompt = (
                "Refine this YOLO + VLM evaluation into a compact final answer. Do not add unsupported visual claims. "
                "Return exactly one JSON object without Markdown fences. Use these keys: answer, visual_evidence, "
                "yolo_cross_check, uncertainty, recommended_next_actions. In yolo_cross_check, include arrays named "
                "confirmed, false_positives, possible_misses, duplicate_or_fragmented, and notes when applicable. "
                "If the VLM answer includes caption, global_classification, vlm_detections, vlm_segmentation, "
                "visual_search, or fusion_hints, preserve those keys and refine them conservatively.\n\n"
                f"User task:\n{image_prompt}\n\n"
                f"YOLO detection summary:\n{json.dumps(json_safe(detections), ensure_ascii=False, indent=2)}\n\n"
                f"VLM answer:\n{vlm_result.get('text', '')}\n\n"
                f"Parsed VLM verdict:\n{json.dumps(json_safe(vlm_result.get('verdict', {})), ensure_ascii=False, indent=2)}"
                f"\n\nVisual search crop passes:\n{json.dumps(json_safe(visual_search_passes), ensure_ascii=False, indent=2)}"
            )
            llm_result = call_openai_compatible(
                model=llm_model,
                user_text=refine_prompt,
                developer_text=default_llm_refine_developer_prompt(resolved_prompt_template),
                base_url=provider_cfg["base_url"],
                provider=str(provider_cfg.get("provider", "openai")),
                api_key_env=str(provider_cfg.get("api_key_env", "OPENAI_API_KEY")),
                api_mode=str(provider_cfg["api_mode"]),
                max_output_tokens=max_output_tokens,
                temperature=float(multimodal_params["temperature"]) if "temperature" in multimodal_params else None,
            )
            llm_result = attach_multimodal_verdict(llm_result)

        vlm_verdict = vlm_result.get("verdict", {}) if isinstance(vlm_result.get("verdict"), dict) else {}
        llm_verdict = llm_result.get("verdict", {}) if isinstance(llm_result, dict) and isinstance(llm_result.get("verdict"), dict) else {}
        merged_verdict = merge_verdicts(vlm_verdict, llm_verdict)
        fusion_preview = fusion_build_multimodal_fusion_preview(
            detections=detections,
            verdict=merged_verdict,
            multimodal_params=multimodal_params,
            image_path=image_ref,
            normalize_detection_boxes_fn=normalize_detection_boxes,
        )
        metric_preview = build_item_metric_preview(
            image_path=image_path,
            names=names,
            detections=detections,
            fusion_preview=fusion_preview,
            verdict=merged_verdict,
            ground_truth_records_for_metric_fn=ground_truth_records_for_metric,
            ground_truth_classification_records_for_metric_fn=ground_truth_classification_records_for_metric,
            ground_truth_segmentation_records_for_metric_fn=ground_truth_segmentation_records_for_metric,
            yolo_prediction_records_for_metric_fn=yolo_prediction_records_for_metric,
            fused_prediction_records_for_metric_fn=fused_prediction_records_for_metric,
            yolo_classification_predictions_for_metric_fn=yolo_classification_predictions_for_metric,
            fused_classification_predictions_for_metric_fn=fused_classification_predictions_for_metric,
            yolo_segmentation_predictions_for_metric_fn=yolo_segmentation_predictions_for_metric,
            fused_segmentation_predictions_for_metric_fn=fused_segmentation_predictions_for_metric,
            polygon_iou_approx_fn=polygon_iou_approx,
        )
        open_world_comparison = build_open_world_comparison_entry(
            image_path=image_path,
            detections=detections,
            fusion_preview=fusion_preview,
            verdict=merged_verdict,
            multimodal_params=multimodal_params,
            effective_prompt_template=str(resolved_prompt_template) if resolved_prompt_template is not None else None,
        )

        status = multimodal_overall_status(vlm_result, llm_result)
        image_item.update(
            {
                "status": status,
                "detector": {
                    "boxes": len(detection_summary.get("detections", []) or []),
                    "summary": detection_summary,
                    "label_counts": detection_label_counts(detections),
                },
                "prompt": image_prompt,
                "multimodal": {
                    "method": method,
                    "thinking_with_image": thinking_with_image,
                    "structured_output": structured_output,
                    "prompt_template": prompt_template,
                    "effective_prompt_template": resolved_prompt_template,
                    "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
                    "open_world_filters": {
                        "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                        "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                        "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                        "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
                    },
                    "use_marked_image": use_marked_image,
                    "visual_search_mode": visual_search_mode,
                    "fusion_mode": fusion_mode,
                    "provider": {
                        "name": provider_cfg["provider"],
                        "base_url": provider_cfg["base_url"],
                        "api_mode": provider_cfg["api_mode"],
                        "api_key_env": provider_cfg["api_key_env"],
                        "api_key_present": provider_cfg["api_key_present"],
                    },
                    "image": image_meta,
                    "visual_search": {"mode": visual_search_mode, "passes": visual_search_passes, "artifacts": item_visual_artifacts},
                    "prompt": image_prompt,
                    "vlm": vlm_result,
                    "llm_refine": llm_result or {"status": "skipped"},
                    "fusion": fusion_preview,
                    "open_world_comparison": open_world_comparison,
                },
                "metric_preview": metric_preview,
            }
        )
        if item_visual_artifacts:
            image_item["artifacts"] = item_visual_artifacts
        if status in {"blocked", "partial", "failed"}:
            image_item["notes"] = ["See multimodal cross-check and detection summary for details."]
        items.append(image_item)
        save_dir = getattr(getattr(model, "predictor", None), "save_dir", save_dir)

    aggregate = aggregate_multimodal_evaluation(items)
    metric_preview = aggregate_metric_preview(
        items,
        names,
        ground_truth_records_for_metric_fn=ground_truth_records_for_metric,
        ground_truth_classification_records_for_metric_fn=ground_truth_classification_records_for_metric,
        ground_truth_segmentation_records_for_metric_fn=ground_truth_segmentation_records_for_metric,
        yolo_prediction_records_for_metric_fn=yolo_prediction_records_for_metric,
        fused_prediction_records_for_metric_fn=fused_prediction_records_for_metric,
        yolo_classification_predictions_for_metric_fn=yolo_classification_predictions_for_metric,
        fused_classification_predictions_for_metric_fn=fused_classification_predictions_for_metric,
        yolo_segmentation_predictions_for_metric_fn=yolo_segmentation_predictions_for_metric,
        fused_segmentation_predictions_for_metric_fn=fused_segmentation_predictions_for_metric,
        merge_verdicts_fn=merge_verdicts,
        polygon_iou_approx_fn=polygon_iou_approx,
    )
    aggregate["metric_preview"] = metric_preview
    overall_status = overall_multimodal_evaluation_status(aggregate)
    if run_yolo_val and data_ref_for_baseline not in (None, ""):
        baseline_request = normalize_request(
            {
                "skill": "yolo.val",
                "runtime": request.get("runtime", {}),
                "inputs": {"model": request["inputs"]["model"], "data": data_ref_for_baseline},
                "params": {k: v for k, v in yolo_params.items() if k not in {"source"}},
                "artifacts": request.get("artifacts", {}),
                "policy": request.get("policy", {}),
                "request_id": f"{request.get('request_id', default_request_id('yolo.multimodal.evaluate'))}-baseline",
            }
        )
        baseline = run_val(baseline_request)
    else:
        baseline = {"status": "skipped"}

    final_selection_source = "recovery" if selected_device != chosen_device and selected_device == "cpu" else device_selection["source"]
    environment = collect_environment_report(
        effective_request,
        selected_device=selected_device,
        requested_device=chosen_device,
        selection_source=final_selection_source,
    )

    report_dir = ensure_manifest_dir(request)
    report_path = report_dir / report_name
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "skill": request.get("skill"),
        "request_id": request.get("request_id"),
        "dataset": dataset_info,
        "aggregate": aggregate,
        "metric_preview": metric_preview,
        "baseline": baseline,
        "items": items,
        "environment": environment,
        "auto_completed": auto_completed,
        "multimodal": {
            "method": method,
            "thinking_with_image": thinking_with_image,
            "structured_output": structured_output,
            "prompt_template": prompt_template,
            "effective_prompt_template": resolved_prompt_template,
            "open_world_assist_profile": multimodal_params.get("open_world_assist_profile"),
            "open_world_filters": {
                "taxonomy_min_score": multimodal_params.get("open_world_taxonomy_min_score"),
                "taxonomy_require_exact_for_generic": multimodal_params.get("open_world_taxonomy_require_exact_for_generic"),
                "filter_unmatched_taxonomy": multimodal_params.get("open_world_filter_unmatched_taxonomy"),
                "filter_generic_labels": multimodal_params.get("open_world_filter_generic_labels"),
            },
            "use_marked_image": use_marked_image,
            "visual_search_mode": visual_search_mode,
            "fusion_mode": fusion_mode,
            "provider": {
                "name": provider_cfg["provider"],
                "base_url": provider_cfg["base_url"],
                "api_mode": provider_cfg["api_mode"],
                "api_key_env": provider_cfg["api_key_env"],
                "api_key_present": provider_cfg["api_key_present"],
            },
            "dataset": dataset_info,
            "prompt": prompt,
        },
    }
    open_world_report = {
        "items": [
            item.get("multimodal", {}).get("open_world_comparison", {})
            for item in items
            if isinstance(item.get("multimodal", {}).get("open_world_comparison", {}), dict)
        ]
    }
    open_world_report["aggregate"] = aggregate_open_world_comparison(open_world_report["items"])
    report["open_world_comparison_report"] = open_world_report
    fusion_coco_records = [
        record
        for item in items
        for record in (
            item.get("multimodal", {})
            .get("fusion", {})
            .get("coco_predictions_preview", [])
            if isinstance(item.get("multimodal", {}).get("fusion", {}), dict)
            else []
        )
    ]
    metric_guardrail = build_metric_guardrail(
        items=items,
        metric_preview=metric_preview,
        fused_coco_records=fusion_coco_records,
        multimodal_params=multimodal_params,
        yolo_prediction_records_for_metric_fn=yolo_prediction_records_for_metric,
    )
    if fusion_coco_records:
        report["fusion_preview"] = {
            "strategy": "metric_safe_v1",
            "coco_prediction_records": len(fusion_coco_records),
            "note": "Preview records are VLM-assisted candidates; run COCO evaluation before treating them as metric gains.",
        }
    report["metric_guardrail"] = {k: v for k, v in metric_guardrail.items() if k != "records"}
    report_path.write_text(json.dumps(json_safe(report), ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts = [{"kind": "json", "path": str(report_path.resolve())}]
    open_world_report_path = report_dir / "open-world-comparison-report.json"
    open_world_report_path.write_text(json.dumps(json_safe(open_world_report), ensure_ascii=False, indent=2), encoding="utf-8")
    artifacts.append({"kind": "open_world_comparison_report", "path": str(open_world_report_path.resolve())})
    if fusion_coco_records:
        fusion_path = report_dir / "fusion-preview-coco-predictions.json"
        fusion_path.write_text(json.dumps(json_safe(fusion_coco_records), ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append({"kind": "fusion_coco_predictions_preview", "path": str(fusion_path.resolve())})
    if metric_preview.get("status") == "ok":
        metric_path = report_dir / "fusion-metric-preview.json"
        metric_path.write_text(json.dumps(json_safe(metric_preview), ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append({"kind": "fusion_metric_preview", "path": str(metric_path.resolve())})
    if metric_guardrail.get("records"):
        guarded_path = report_dir / "metric-guarded-coco-predictions.json"
        guarded_path.write_text(json.dumps(json_safe(metric_guardrail["records"]), ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts.append({"kind": "metric_guarded_coco_predictions", "path": str(guarded_path.resolve())})
    if save_dir:
        artifacts.append({"kind": "directory", "path": str(Path(save_dir).resolve())})

    payload = response(
        request["skill"],
        overall_status,
        f"multimodal evaluation finished on {len(items)} images",
        job={"mode": "sync", "save_dir": json_safe(save_dir), "executor": "python_api+openai", "device": selected_device},
        results=items,
        evaluation=aggregate,
        environment=environment,
        auto_completed=auto_completed,
        artifacts=artifacts,
        multimodal=report["multimodal"],
        metric_guardrail={k: v for k, v in metric_guardrail.items() if k != "records"},
        baseline=baseline,
        next_actions=["yolo.val", "yolo.multimodal.infer", "yolo.predict"],
    )
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


def run_lora_diagnose(request: dict[str, Any]) -> dict[str, Any]:
    return run_lora_diagnose_impl(
        request,
        LoraDiagnoseDeps(
            build_model=build_model,
            is_dry_run=is_dry_run,
            response=response,
            plan_response=plan_response,
            write_manifest=write_manifest,
        ),
    )


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


def run_peft_compare(request: dict[str, Any]) -> dict[str, Any]:
    return run_peft_compare_impl(
        request,
        PeftCompareDeps(
            normalize_request=normalize_request,
            is_dry_run=is_dry_run,
            response=response,
            plan_response=plan_response,
            write_manifest=write_manifest,
            best_checkpoint=best_checkpoint,
            run_train_like=run_train_like,
            run_val=run_val,
        ),
    )


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
    return run_experiment_pipeline(
        request,
        PipelineDeps(
            normalize_request=normalize_request,
            is_dry_run=is_dry_run,
            response=response,
            plan_response=plan_response,
            write_manifest=write_manifest,
            best_checkpoint=best_checkpoint,
            run_system=run_system,
            run_model_inspect=run_model_inspect,
            run_train_like=run_train_like,
            run_val=run_val,
            run_export=run_export,
            run_benchmark=run_benchmark,
            run_lora_diagnose=run_lora_diagnose,
            run_moe_diagnose=run_moe_diagnose,
            run_peft_compare=run_peft_compare,
        ),
    )


HANDLERS = {
    "yolo.system": run_system,
    "yolo.model.inspect": run_model_inspect,
    "yolo.train": lambda request: run_train_like(request, "yolo.train"),
    "yolo.lora.train": lambda request: run_train_like(request, "yolo.lora.train"),
    "yolo.val": run_val,
    "yolo.predict": lambda request: run_predict_like(request, "predict"),
    "yolo.track": lambda request: run_predict_like(request, "track"),
    "yolo.multimodal.infer": run_multimodal_infer,
    "yolo.multimodal.evaluate": run_multimodal_evaluate,
    "yolo.export": run_export,
    "yolo.benchmark": run_benchmark,
    "yolo.tune": run_tune,
    "yolo.lora.adapters": run_lora_adapters,
    "yolo.lora.diagnose": run_lora_diagnose,
    "yolo.eval.peft_compare": run_peft_compare,
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
