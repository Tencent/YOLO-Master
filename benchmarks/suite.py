"""Declarative, resumable inference benchmarks for YOLO-Master models."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import yaml


SCHEMA_VERSION = 1
SUPPORTED_TASKS = {"inference"}
SUPPORTED_DTYPES = {"fp32", "fp16", "bf16"}
SUPPORTED_SUFFIXES = {".yaml", ".yml", ".pt", ".pth"}
REPO_ROOT = Path(__file__).resolve().parents[1]


def _mps_available() -> bool:
    """Return MPS availability without assuming the backend exists on old PyTorch."""
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that refuses duplicate mapping keys."""


def _construct_unique_mapping(loader: _UniqueKeyLoader, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
    mapping: dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            line = key_node.start_mark.line + 1
            raise ValueError(f"duplicate benchmark catalog key {key!r} at line {line}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


@dataclass(frozen=True)
class BenchmarkSettings:
    """Resolved execution settings for one benchmark case."""

    task: str = "inference"
    device: str = "cpu"
    dtype: str = "fp32"
    batch_size: int = 1
    image_size: int = 64
    warmup: int = 1
    iterations: int = 5
    seed: int = 0
    collect_flops: bool = True
    collect_routing: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCase:
    """One model variant in a resolved suite."""

    case_id: str
    label: str
    model: Path
    settings: BenchmarkSettings

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "model": str(self.model),
            "settings": self.settings.to_dict(),
        }


@dataclass(frozen=True)
class ResolvedBenchmarkSuite:
    """Validated benchmark suite ready for execution."""

    name: str
    description: str
    cases: tuple[BenchmarkCase, ...]
    catalog_path: Path
    root: Path


@dataclass(frozen=True)
class BenchmarkCaseResult:
    """Serializable outcome of one benchmark case."""

    case_id: str
    label: str
    model: str
    fingerprint: str
    status: str
    error: str | None
    metrics: dict[str, Any]
    routing: dict[str, Any]
    duration_s: float
    settings: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkCaseResult":
        return cls(
            case_id=str(value["case_id"]),
            label=str(value.get("label", value["case_id"])),
            model=str(value.get("model", "")),
            fingerprint=str(value.get("fingerprint", "")),
            status=str(value.get("status", "failed")),
            error=value.get("error"),
            metrics=dict(value.get("metrics", {})),
            routing=dict(value.get("routing", {})),
            duration_s=float(value.get("duration_s", 0.0)),
            settings=dict(value.get("settings", {})),
        )


@dataclass(frozen=True)
class BenchmarkRunReport:
    """Canonical benchmark result document."""

    schema_version: int
    suite_name: str
    description: str
    settings: dict[str, Any]
    environment: dict[str, Any]
    started_at: str
    completed_at: str | None
    cases: list[BenchmarkCaseResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "suite_name": self.suite_name,
            "description": self.description,
            "settings": self.settings,
            "environment": self.environment,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BenchmarkRunReport":
        return cls(
            schema_version=int(value.get("schema_version", SCHEMA_VERSION)),
            suite_name=str(value.get("suite_name", "")),
            description=str(value.get("description", "")),
            settings=dict(value.get("settings", {})),
            environment=dict(value.get("environment", {})),
            started_at=str(value.get("started_at", "")),
            completed_at=value.get("completed_at"),
            cases=[BenchmarkCaseResult.from_dict(item) for item in value.get("cases", [])],
        )


def list_suites(catalog_path: str | Path) -> dict[str, str]:
    """Return suite names and descriptions from a catalog."""
    catalog = _load_catalog(Path(catalog_path))
    suites = catalog.get("suites", {})
    return {str(name): str(value.get("description", "")) for name, value in suites.items()}


def load_suite(
    catalog_path: str | Path,
    suite_name: str,
    *,
    root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
    case_ids: Sequence[str] | None = None,
) -> ResolvedBenchmarkSuite:
    """Load, merge, and validate one benchmark suite from YAML."""
    catalog_path = Path(catalog_path).resolve()
    catalog = _load_catalog(catalog_path)
    if int(catalog.get("schema_version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"unsupported benchmark schema_version: {catalog.get('schema_version')!r}")

    suites = catalog.get("suites")
    if not isinstance(suites, dict) or suite_name not in suites:
        raise ValueError(f"unknown benchmark suite {suite_name!r}")
    suite_raw = suites[suite_name]
    if not isinstance(suite_raw, dict):
        raise ValueError(f"suite {suite_name!r} must be a mapping")

    root_path = Path(root).resolve() if root is not None else catalog_path.parents[1]
    defaults = _mapping(catalog.get("defaults", {}), "defaults")
    suite_settings = _mapping(suite_raw.get("settings", {}), f"suite {suite_name} settings")
    global_settings = {**defaults, **suite_settings, **dict(overrides or {})}
    raw_cases = suite_raw.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"suite {suite_name!r} must contain at least one case")

    selected = set(case_ids or [])
    cases: list[BenchmarkCase] = []
    seen: set[str] = set()
    for index, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError(f"suite {suite_name!r} case {index} must be a mapping")
        case_id = str(raw_case.get("id", "")).strip()
        if not case_id:
            raise ValueError(f"suite {suite_name!r} case {index} is missing id")
        if case_id in seen:
            raise ValueError(f"duplicate benchmark case id: {case_id}")
        seen.add(case_id)
        if selected and case_id not in selected:
            continue

        model_value = raw_case.get("model")
        if not isinstance(model_value, str) or not model_value.strip():
            raise ValueError(f"benchmark case {case_id!r} is missing model")
        model_path = Path(model_value).expanduser()
        if not model_path.is_absolute():
            model_path = root_path / model_path
        model_path = model_path.resolve()
        if model_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"benchmark case {case_id!r} has unsupported model suffix: {model_path.suffix}")
        if not model_path.exists():
            raise ValueError(f"benchmark case {case_id!r} model does not exist: {model_path}")

        case_overrides = {
            key: value
            for key, value in raw_case.items()
            if key not in {"id", "label", "model", "description"}
        }
        settings = _settings_from_mapping({**global_settings, **case_overrides}, case_id)
        cases.append(
            BenchmarkCase(
                case_id=case_id,
                label=str(raw_case.get("label", case_id)),
                model=model_path,
                settings=settings,
            )
        )

    if selected:
        missing = sorted(selected - seen)
        if missing:
            raise ValueError(f"unknown benchmark case ids: {', '.join(missing)}")
    if not cases:
        raise ValueError(f"suite {suite_name!r} has no selected cases")
    return ResolvedBenchmarkSuite(
        name=suite_name,
        description=str(suite_raw.get("description", "")),
        cases=tuple(cases),
        catalog_path=catalog_path,
        root=root_path,
    )


def _load_catalog(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"benchmark catalog does not exist: {path}")
    try:
        value = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid benchmark catalog YAML: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError("benchmark catalog root must be a mapping")
    return value


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a mapping")
    return dict(value)


def _settings_from_mapping(value: Mapping[str, Any], case_id: str) -> BenchmarkSettings:
    known = set(BenchmarkSettings.__dataclass_fields__)
    unknown = sorted(set(value) - known)
    if unknown:
        raise ValueError(f"benchmark case {case_id!r} has unknown settings: {', '.join(unknown)}")
    settings = BenchmarkSettings(**value)
    if settings.task not in SUPPORTED_TASKS:
        raise ValueError(f"benchmark task must be one of {sorted(SUPPORTED_TASKS)}, got {settings.task!r}")
    if settings.dtype not in SUPPORTED_DTYPES:
        raise ValueError(f"benchmark dtype must be one of {sorted(SUPPORTED_DTYPES)}, got {settings.dtype!r}")
    if not isinstance(settings.device, str) or not settings.device.strip():
        raise ValueError("benchmark device must be a non-empty string")
    for name in ("batch_size", "image_size", "iterations"):
        if not isinstance(getattr(settings, name), int) or getattr(settings, name) <= 0:
            raise ValueError(f"benchmark {name} must be a positive integer")
    if not isinstance(settings.warmup, int) or settings.warmup < 0:
        raise ValueError("benchmark warmup must be a non-negative integer")
    if not isinstance(settings.seed, int):
        raise ValueError("benchmark seed must be an integer")
    if not isinstance(settings.collect_flops, bool) or not isinstance(settings.collect_routing, bool):
        raise ValueError("benchmark collect_flops and collect_routing must be booleans")
    return settings


def summarize_latency(samples_ms: Sequence[float], batch_size: int) -> dict[str, float]:
    """Summarize synchronized latency samples with interpolated percentiles."""
    values = sorted(float(value) for value in samples_ms)
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency samples must contain finite non-negative values")
    p50 = _percentile(values, 0.50)
    return {
        "latency_mean_ms": sum(values) / len(values),
        "latency_p50_ms": p50,
        "latency_p95_ms": _percentile(values, 0.95),
        "latency_p99_ms": _percentile(values, 0.99),
        "latency_min_ms": values[0],
        "latency_max_ms": values[-1],
        "throughput_items_s": batch_size * 1000.0 / max(p50, 1e-12),
    }


def _percentile(values: Sequence[float], quantile: float) -> float:
    if len(values) == 1:
        return float(values[0])
    position = (len(values) - 1) * quantile
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(values[lower])
    fraction = position - lower
    return float(values[lower] + fraction * (values[upper] - values[lower]))


class BenchmarkSuiteRunner:
    """Execute and persist a benchmark suite case by case."""

    def __init__(
        self,
        suite: ResolvedBenchmarkSuite,
        output_dir: str | Path,
        *,
        case_executor: Callable[[BenchmarkCase, str], BenchmarkCaseResult] | None = None,
    ) -> None:
        self.suite = suite
        self.output_dir = Path(output_dir)
        self.case_executor = case_executor or self._execute_case

    def run(self, *, resume: bool = True) -> BenchmarkRunReport:
        """Run all selected cases, reusing only matching successful results."""
        previous = self._load_previous() if resume else None
        previous_by_id = {case.case_id: case for case in previous.cases} if previous else {}
        environment = collect_environment()
        report = BenchmarkRunReport(
            schema_version=SCHEMA_VERSION,
            suite_name=self.suite.name,
            description=self.suite.description,
            settings={"cases": [case.to_dict() for case in self.suite.cases]},
            environment=environment,
            started_at=_utc_now(),
            completed_at=None,
            cases=[],
        )

        for case in self.suite.cases:
            fingerprint = case_fingerprint(case, environment=environment)
            previous_case = previous_by_id.get(case.case_id)
            if (
                resume
                and previous_case is not None
                and previous_case.fingerprint == fingerprint
                and previous_case.status in {"ok", "reused"}
            ):
                result = replace(previous_case, status="reused")
            else:
                started = time.perf_counter()
                try:
                    result = self.case_executor(case, fingerprint)
                    if result.fingerprint != fingerprint:
                        result = replace(result, fingerprint=fingerprint)
                except Exception as exc:
                    result = BenchmarkCaseResult(
                        case_id=case.case_id,
                        label=case.label,
                        model=str(case.model),
                        fingerprint=fingerprint,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                        metrics={},
                        routing={},
                        duration_s=time.perf_counter() - started,
                        settings=case.settings.to_dict(),
                    )
                finally:
                    _clear_device_cache(case.settings.device)

            report.cases.append(result)
            write_reports(replace(report, completed_at=_utc_now()), self.output_dir)

        final_report = replace(report, completed_at=_utc_now())
        write_reports(final_report, self.output_dir)
        return final_report

    def _load_previous(self) -> BenchmarkRunReport | None:
        path = self.output_dir / "results.json"
        if not path.exists():
            return None
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            report = BenchmarkRunReport.from_dict(value)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            return None
        return report if report.suite_name == self.suite.name else None

    def _execute_case(self, case: BenchmarkCase, fingerprint: str) -> BenchmarkCaseResult:
        started = time.perf_counter()
        settings = case.settings
        device = resolve_device(settings.device)
        dtype = resolve_dtype(settings.dtype)
        torch.manual_seed(settings.seed)
        model = load_benchmark_model(case.model, device=device, dtype=dtype)
        input_tensor = torch.randn(
            settings.batch_size,
            3,
            settings.image_size,
            settings.image_size,
            device=device,
            dtype=dtype,
        )
        metrics = collect_model_metrics(model, input_tensor, settings)
        routing = collect_routing_metrics(model, input_tensor) if settings.collect_routing else {}
        return BenchmarkCaseResult(
            case_id=case.case_id,
            label=case.label,
            model=str(case.model),
            fingerprint=fingerprint,
            status="ok",
            error=None,
            metrics=metrics,
            routing=routing,
            duration_s=time.perf_counter() - started,
            settings={**settings.to_dict(), "resolved_device": str(device), "resolved_dtype": str(dtype)},
        )


def resolve_device(value: str) -> torch.device:
    """Resolve `auto` and common numeric CUDA device forms."""
    normalized = value.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if _mps_available():
            return torch.device("mps")
        return torch.device("cpu")
    if normalized.isdigit():
        normalized = f"cuda:{normalized}"
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    if device.type == "mps" and not _mps_available():
        raise RuntimeError("MPS device requested but MPS is unavailable")
    return device


def resolve_dtype(value: str) -> torch.dtype:
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[value]


def load_benchmark_model(model_path: Path, *, device: torch.device, dtype: torch.dtype) -> nn.Module:
    """Load local model YAMLs and trusted local checkpoints."""
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.utils.patches import torch_load

    if model_path.suffix.lower() in {".yaml", ".yml"}:
        model = DetectionModel(str(model_path), ch=3, verbose=False)
    else:
        checkpoint = torch_load(model_path, map_location="cpu")
        model = checkpoint.get("ema") or checkpoint.get("model") if isinstance(checkpoint, dict) else checkpoint
        if not isinstance(model, nn.Module):
            raise TypeError(f"checkpoint does not contain an nn.Module under 'ema' or 'model': {model_path}")
    return model.to(device=device, dtype=dtype).eval()


def collect_model_metrics(
    model: nn.Module, input_tensor: torch.Tensor, settings: BenchmarkSettings
) -> dict[str, Any]:
    """Collect size, FLOPs, eager latency, and routed-family counts."""
    from ultralytics.utils.torch_utils import get_flops

    with torch.inference_mode():
        for _ in range(settings.warmup):
            model(input_tensor)
        _sync_device(input_tensor.device)
        samples: list[float] = []
        for _ in range(settings.iterations):
            _sync_device(input_tensor.device)
            started = time.perf_counter()
            model(input_tensor)
            _sync_device(input_tensor.device)
            samples.append((time.perf_counter() - started) * 1000.0)

    family_counts = _routing_family_counts(model)
    return {
        "params_total": sum(parameter.numel() for parameter in model.parameters()),
        "params_trainable": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
        "model_size_mb": sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 1e6,
        "gflops": float(get_flops(model, imgsz=settings.image_size)) if settings.collect_flops else None,
        **summarize_latency(samples, settings.batch_size),
        **family_counts,
    }


def collect_routing_metrics(model: nn.Module, input_tensor: torch.Tensor) -> dict[str, Any]:
    """Capture cross-family routing health for the benchmark input."""
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    interpreter = RoutingInterpreter(model)
    try:
        heatmaps = interpreter.capture_routing(input_tensor)
    except ValueError:
        return {
            "routed_layers": 0,
            "collapsed_layers": 0,
            "mean_normalized_gini": 0.0,
            "mean_normalized_entropy": 0.0,
            "mean_dominant_share": 0.0,
            "layers": {},
        }
    reports = interpreter.detect_routing_collapse(heatmaps=heatmaps)
    values = list(reports.values())
    return {
        "routed_layers": len(values),
        "collapsed_layers": sum(report.collapsed for report in values),
        "mean_normalized_gini": _mean([report.normalized_gini for report in values]),
        "mean_normalized_entropy": _mean([report.normalized_entropy for report in values]),
        "mean_dominant_share": _mean([report.dominant_share for report in values]),
        "layers": {name: report.to_dict() for name, report in reports.items()},
    }


def _routing_family_counts(model: nn.Module) -> dict[str, int]:
    from ultralytics.utils.routing_interpreter import RoutingInterpreter

    counts = {"moe_layers": 0, "moa_layers": 0, "mot_layers": 0, "molora_layers": 0}
    interpreter = RoutingInterpreter(model)
    for module in interpreter._routed_modules(leaf_only=True).values():
        module_name = type(module).__module__.lower()
        class_name = type(module).__name__.lower()
        if ".molora." in module_name or "molora" in class_name:
            counts["molora_layers"] += 1
        elif ".moa." in module_name or "moa" in class_name:
            counts["moa_layers"] += 1
        elif ".mot." in module_name or "mot" in class_name:
            counts["mot_layers"] += 1
        else:
            counts["moe_layers"] += 1
    return counts


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _clear_device_cache(device_value: str) -> None:
    try:
        device = resolve_device(device_value)
    except (RuntimeError, ValueError):
        return
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def case_fingerprint(case: BenchmarkCase, *, environment: Mapping[str, Any] | None = None) -> str:
    """Hash resolved settings, runtime environment, and local model contents."""
    digest = hashlib.sha256()
    digest.update(json.dumps(case.to_dict(), sort_keys=True, separators=(",", ":")).encode())
    if environment is not None:
        digest.update(json.dumps(dict(environment), sort_keys=True, separators=(",", ":"), default=str).encode())
    with case.model.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_environment() -> dict[str, Any]:
    """Capture the environment required to interpret benchmark numbers."""
    commit = _git_output("rev-parse", "HEAD")
    dirty = bool(_git_output("status", "--porcelain"))
    cuda_devices = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    mps_device = ""
    mps = getattr(torch.backends, "mps", None)
    if _mps_available() and hasattr(mps, "get_name"):
        try:
            mps_device = str(mps.get_name())
        except RuntimeError:
            mps_device = "available"
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cuda_devices": cuda_devices,
        "mps_available": _mps_available(),
        "mps_device": mps_device,
        "git_commit": commit,
        "git_dirty": dirty,
        "relevant_worktree_sha256": _relevant_worktree_digest(),
    }


def _git_output(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
            cwd=REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _relevant_worktree_digest() -> str:
    """Hash tracked diffs and untracked source files that can change benchmark behavior."""
    digest = hashlib.sha256()
    try:
        diff = subprocess.run(
            ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--", "benchmarks", "ultralytics"],
            check=False,
            capture_output=True,
            timeout=10,
            cwd=REPO_ROOT,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain=v1", "-z", "--", "benchmarks", "ultralytics"],
            check=False,
            capture_output=True,
            timeout=10,
            cwd=REPO_ROOT,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if diff.returncode != 0 or status.returncode != 0:
        return ""

    digest.update(diff.stdout)
    digest.update(status.stdout)
    for entry in status.stdout.split(b"\0"):
        if not entry.startswith(b"?? "):
            continue
        relative = entry[3:].decode(errors="surrogateescape")
        path = REPO_ROOT / relative
        if not path.is_file():
            continue
        digest.update(relative.encode(errors="surrogateescape"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_reports(report: BenchmarkRunReport, output_dir: str | Path) -> dict[str, Path]:
    """Atomically write canonical JSON plus deterministic CSV and Markdown views."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "results.json"
    csv_path = output_dir / "results.csv"
    markdown_path = output_dir / "report.md"
    _atomic_write(json_path, json.dumps(report.to_dict(), indent=2, ensure_ascii=True) + "\n")
    _atomic_write(csv_path, _render_csv(report))
    _atomic_write(markdown_path, _render_markdown(report))
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _flatten_case(case: BenchmarkCaseResult) -> dict[str, Any]:
    row: dict[str, Any] = {
        "case_id": case.case_id,
        "label": case.label,
        "status": case.status,
        "model": case.model,
        "fingerprint": case.fingerprint,
        "duration_s": case.duration_s,
        "error": case.error or "",
    }
    row.update(case.metrics)
    for key, value in case.routing.items():
        row[f"routing_{key}"] = value
    for key, value in list(row.items()):
        if isinstance(value, (dict, list, tuple)):
            row[key] = json.dumps(value, sort_keys=True, ensure_ascii=True)
        elif value is None:
            row[key] = ""
    return row


def _render_csv(report: BenchmarkRunReport) -> str:
    rows = [_flatten_case(case) for case in report.cases]
    fixed = ["case_id", "label", "status", "model", "fingerprint", "duration_s", "error"]
    extra = sorted({key for row in rows for key in row} - set(fixed))
    from io import StringIO

    handle = StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=[*fixed, *extra], lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return handle.getvalue()


def _render_markdown(report: BenchmarkRunReport) -> str:
    lines = [
        f"# Benchmark Report: {report.suite_name}",
        "",
        report.description or "No description.",
        "",
        f"- Started: `{report.started_at}`",
        f"- Completed: `{report.completed_at or ''}`",
        f"- Git commit: `{report.environment.get('git_commit', '')}`",
        f"- Dirty worktree: `{report.environment.get('git_dirty', False)}`",
        "",
        "| case_id | label | status | params | GFLOPs | P50 ms | P95 ms | P99 ms | items/s | collapsed | error |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for case in report.cases:
        metrics = case.metrics
        routing = case.routing
        lines.append(
            "| {case_id} | {label} | {status} | {params} | {gflops} | {p50} | {p95} | {p99} | "
            "{throughput} | {collapsed} | {error} |".format(
                case_id=_md(case.case_id),
                label=_md(case.label),
                status=_md(case.status),
                params=_format_metric(metrics.get("params_total"), 0),
                gflops=_format_metric(metrics.get("gflops"), 3),
                p50=_format_metric(metrics.get("latency_p50_ms"), 3),
                p95=_format_metric(metrics.get("latency_p95_ms"), 3),
                p99=_format_metric(metrics.get("latency_p99_ms"), 3),
                throughput=_format_metric(metrics.get("throughput_items_s"), 2),
                collapsed=_format_metric(routing.get("collapsed_layers"), 0),
                error=_md(case.error or ""),
            )
        )
    lines.extend(["", "## Environment", "", "```json", json.dumps(report.environment, indent=2), "```", ""])
    return "\n".join(lines)


def _format_metric(value: Any, digits: int) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return _md(str(value))
    return f"{number:.{digits}f}"


def _md(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


__all__ = [
    "BenchmarkCase",
    "BenchmarkCaseResult",
    "BenchmarkRunReport",
    "BenchmarkSettings",
    "BenchmarkSuiteRunner",
    "ResolvedBenchmarkSuite",
    "case_fingerprint",
    "collect_environment",
    "collect_model_metrics",
    "collect_routing_metrics",
    "list_suites",
    "load_benchmark_model",
    "load_suite",
    "resolve_device",
    "summarize_latency",
    "write_reports",
]
