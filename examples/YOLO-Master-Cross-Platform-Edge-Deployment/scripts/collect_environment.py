#!/usr/bin/env python3
"""Collect reproducible host and toolchain metadata for an Issue #51 run.

The collector is deliberately dependency-free.  Missing optional tools are
represented as ``available=false`` rather than guessed values, which makes the
result suitable for an evidence bundle on Linux, Windows, macOS or Jetson.
The output is metadata only; it does not run inference or claim a benchmark
result.

Example::

    python scripts/collect_environment.py \
        --repo-root . --backend onnx --execution-provider cpu \
        --threads 4 --warmup 2 --runs 20 \
        --output artifacts/environment.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence


SCHEMA_VERSION = "issue51-environment/v1"


def _first_line(value: str) -> Optional[str]:
    for line in value.splitlines():
        line = line.strip()
        if line:
            return line
    return None


def _display_command(argv: Sequence[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in argv)


def run_probe(argv: Sequence[str], timeout: float = 5.0) -> Dict[str, Any]:
    """Run a version/probe command without invoking a shell."""
    args = [str(part) for part in argv]
    result: Dict[str, Any] = {
        "command": _display_command(args),
        "available": False,
        "version": None,
        "returncode": None,
    }
    try:
        completed = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        result["error"] = str(exc)
        return result
    result["returncode"] = completed.returncode
    result["available"] = completed.returncode == 0
    result["version"] = _first_line(completed.stdout) or _first_line(completed.stderr)
    if completed.returncode != 0:
        result["error"] = _first_line(completed.stderr) or _first_line(completed.stdout)
    return result


def _command_name(value: Optional[str], fallbacks: Iterable[str]) -> Optional[str]:
    candidates: List[str] = []
    if value:
        candidates.append(value)
    candidates.extend(fallbacks)
    for candidate in candidates:
        try:
            probe = run_probe(_split_command(candidate) + ["--version"])
        except ValueError:
            continue
        if probe["available"]:
            return candidate
    return None


def _split_command(value: str) -> List[str]:
    """Split a command override while retaining paths containing spaces."""
    try:
        return shlex.split(value, posix=(os.name != "nt"))
    except ValueError as exc:
        raise ValueError("invalid command override {!r}: {}".format(value, exc)) from exc


def _probe_command(value: Optional[str], fallbacks: Iterable[str]) -> Dict[str, Any]:
    selected = _command_name(value, fallbacks)
    if selected is None:
        return {
            "command": value or next(iter(fallbacks), ""),
            "available": False,
            "version": None,
            "returncode": None,
        }
    return run_probe(_split_command(selected) + ["--version"])


def _linux_cpu_model() -> Optional[str]:
    path = Path("/proc/cpuinfo")
    if not path.is_file():
        return None
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if ":" not in raw:
                continue
            key, value = raw.split(":", 1)
            if key.strip().lower() in {"model name", "hardware", "processor"} and value.strip():
                return value.strip()
    except OSError:
        return None
    return None


def _memory_bytes() -> Optional[int]:
    """Return physical memory when the platform exposes it without a package."""
    if sys.platform.startswith("linux"):
        try:
            for raw in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if raw.startswith("MemTotal:"):
                    match = re.search(r"(\d+)", raw)
                    if match:
                        return int(match.group(1)) * 1024
        except (OSError, ValueError):
            pass
    if sys.platform == "darwin":
        probe = run_probe(["sysctl", "-n", "hw.memsize"])
        if probe.get("available") and probe.get("version"):
            try:
                return int(str(probe["version"]))
            except ValueError:
                pass
    try:
        # psutil is optional; use it only when already installed.
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _physical_cpus() -> Optional[int]:
    try:
        import psutil  # type: ignore

        value = psutil.cpu_count(logical=False)
        return int(value) if value else None
    except (ImportError, AttributeError, OSError, ValueError):
        pass
    return None


def _package_version(names: Iterable[str]) -> Optional[str]:
    for name in names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def _python_packages() -> Dict[str, Optional[str]]:
    return {
        "numpy": _package_version(("numpy",)),
        "opencv": _package_version(("opencv-python", "opencv-python-headless")),
        "onnx": _package_version(("onnx",)),
        "onnxruntime": _package_version(("onnxruntime", "onnxruntime-gpu")),
        "ultralytics": _package_version(("ultralytics",)),
    }


def _git_metadata(repo_root: Path) -> Dict[str, Any]:
    root = repo_root.expanduser().resolve()
    result: Dict[str, Any] = {"root": str(root), "commit": None, "branch": None, "dirty": None}
    commit = run_probe(["git", "-C", str(root), "rev-parse", "HEAD"])
    if commit.get("available") and commit.get("version"):
        result["commit"] = str(commit["version"])
    branch = run_probe(["git", "-C", str(root), "branch", "--show-current"])
    if branch.get("available"):
        result["branch"] = branch.get("version")
    try:
        completed = subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
        )
        if completed.returncode == 0:
            result["dirty"] = bool(completed.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return result


def _sdk_probe(root_value: Optional[str], kind: str) -> Optional[Dict[str, Any]]:
    if not root_value:
        return None
    root = Path(root_value).expanduser().resolve()
    headers = {
        "onnxruntime": ("include/onnxruntime_cxx_api.h",),
        "ncnn": ("include/ncnn/net.h",),
        "mnn": ("include/MNN/Interpreter.hpp",),
        "tensorrt": ("include/NvInfer.h",),
    }.get(kind, ())
    libraries = {
        "onnxruntime": ("lib/libonnxruntime.so", "lib/onnxruntime.lib", "bin/onnxruntime.dll"),
        "ncnn": ("lib/libncnn.so", "lib/libncnn.a", "lib/ncnn.lib"),
        "mnn": ("lib/libMNN.so", "lib/libMNN.a", "lib/MNN.lib"),
        "tensorrt": ("lib/libnvinfer.so", "lib/nvinfer.dll", "lib/nvinfer.lib"),
    }.get(kind, ())
    header_state = {item: (root / item).is_file() for item in headers}
    library_state = {item: (root / item).is_file() for item in libraries}
    # Versioned Unix libraries are common in release archives; record the
    # directory scan without treating a missing unversioned symlink as failure.
    library_dirs = [root / "lib", root / "lib64", root / "bin", root / "build/src", root / "build"]
    patterns = {
        "onnxruntime": ("libonnxruntime.so*", "onnxruntime*.dll", "onnxruntime*.lib"),
        "ncnn": ("libncnn.so*", "libncnn.a", "ncnn*.dll", "ncnn*.lib"),
        "mnn": ("libMNN.so*", "libMNN.a", "MNN*.dll", "MNN*.lib"),
        "tensorrt": ("libnvinfer.so*", "nvinfer*.dll", "nvinfer*.lib"),
    }.get(kind, ())
    discovered: List[str] = []
    for directory in library_dirs:
        if not directory.is_dir():
            continue
        for pattern in patterns:
            discovered.extend(str(path.relative_to(root).as_posix()) for path in directory.glob(pattern) if path.is_file())
    return {
        "root": str(root),
        "exists": root.is_dir(),
        "headers": header_state,
        "libraries": library_state,
        "discovered_libraries": sorted(set(discovered), key=str.casefold),
    }


def collect_environment(args: argparse.Namespace) -> Dict[str, Any]:
    """Build the JSON-serialisable environment record."""
    compiler_override = getattr(args, "compiler", None) or os.environ.get("CXX")
    cpu_model = _linux_cpu_model() or platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER")
    host: Dict[str, Any] = {
        "system": platform.system().lower() or None,
        "release": platform.release() or None,
        "version": platform.version() or None,
        "platform": platform.platform(aliased=True) or None,
        "machine": platform.machine() or None,
        "processor": platform.processor() or None,
        "cpu_model": cpu_model or None,
        "logical_cpus": os.cpu_count(),
        "physical_cpus": _physical_cpus(),
        "memory_bytes": _memory_bytes(),
    }
    tools: Dict[str, Any] = {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "executable": str(Path(sys.executable).resolve()),
        },
        "compiler": _probe_command(compiler_override, ("g++", "clang++", "cl")),
        "cmake": _probe_command(None, ("cmake",)),
        "pkg_config": _probe_command(None, ("pkg-config",)),
        "python_packages": _python_packages(),
    }
    # pkg-config's --version is useful, but OpenCV's module version is a
    # separate query and is intentionally left unavailable if pkg-config is
    # not installed.
    pkg = tools["pkg_config"]
    if pkg.get("available"):
        tools["opencv_pkg_config"] = run_probe(["pkg-config", "--modversion", "opencv4"])
    else:
        tools["opencv_pkg_config"] = {"available": False, "version": None, "returncode": None}

    sdk_roots = {
        name: _sdk_probe(getattr(args, name + "_root", None), name)
        for name in ("onnxruntime", "ncnn", "mnn", "tensorrt")
    }
    runtime = {
        "backend": getattr(args, "backend", None),
        "execution_provider": getattr(args, "execution_provider", None),
        "threads": getattr(args, "threads", None),
        "warmup": getattr(args, "warmup", None),
        "runs": getattr(args, "runs", None),
    }
    gpu_probe = run_probe(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "captured_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "host": host,
        "tools": tools,
        "sdk_roots": sdk_roots,
        "gpu": gpu_probe,
        "runtime_protocol": runtime,
        "repository": _git_metadata(Path(args.repo_root)),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd(), help="repository root for Git metadata")
    parser.add_argument("--output", type=Path, help="write JSON to this path instead of stdout")
    parser.add_argument("--backend", default=None, help="backend used by the planned run")
    parser.add_argument("--execution-provider", default=None, help="execution provider used by the planned run")
    parser.add_argument("--threads", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--runs", type=int, default=None)
    parser.add_argument("--compiler", help="compiler command override (default: CXX, g++, clang++, or cl)")
    parser.add_argument("--onnxruntime-root", dest="onnxruntime_root", help="ONNX Runtime SDK root")
    parser.add_argument("--ncnn-root", dest="ncnn_root", help="NCNN SDK root")
    parser.add_argument("--mnn-root", dest="mnn_root", help="MNN SDK root")
    parser.add_argument("--tensorrt-root", dest="tensorrt_root", help="TensorRT SDK root")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if (args.threads is not None and args.threads <= 0) or (
        args.warmup is not None and args.warmup < 0
    ) or (args.runs is not None and args.runs <= 0):
        print("threads and runs must be positive; warmup must be non-negative", file=sys.stderr)
        return 2
    payload = collect_environment(args)
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
