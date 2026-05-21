from __future__ import annotations

import contextlib
import csv
import io
import os
import re
import shutil
import site
import subprocess
import sys
import sysconfig
from pathlib import Path
from typing import Any

from runtime.cli.contract import json_safe, plan_response, response
from runtime.cli.normalize import coerce_scalar, resolved_path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_MANIFEST_DIR = REPO_ROOT / "runs" / "agent"
ANSI_RE = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")

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
    try:
        user_base = site.getuserbase()
    except Exception:
        user_base = None
    if user_base:
        candidates.extend(Path(user_base) / "bin" / name for name in names)
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
