"""Evidence I/O helpers shared by the P2 runner."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_manifest(run_dir: Path) -> Path:
    files = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item.name != "manifest.sha256.json"):
        files.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    destination = run_dir / "manifest.sha256.json"
    write_json(destination, {"algorithm": "sha256", "file_count": len(files), "files": files})
    return destination


def environment(torch_module: Any, source_root: Path, project_root: Path) -> dict[str, Any]:
    def git_value(root: Path, *args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True, encoding="utf-8"
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "torch": torch_module.__version__,
        "cuda_available": bool(torch_module.cuda.is_available()),
        "cuda_version": torch_module.version.cuda,
        "device_requested": "cpu",
        "cpu_threads": int(torch_module.get_num_threads()),
        "project_commit": git_value(project_root, "rev-parse", "HEAD"),
        "project_tree": git_value(project_root, "rev-parse", "HEAD^{tree}"),
        "source_root": str(source_root),
        "source_git_commit": git_value(source_root, "rev-parse", "HEAD"),
        "cwd": os.getcwd(),
    }
