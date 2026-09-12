"""Reproducibility and artifact helpers."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def git_identity(root: Path) -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args], cwd=root, check=True, capture_output=True, text=True, encoding="utf-8"
            ).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    return {
        "head": run("rev-parse", "HEAD"),
        "tree": run("show", "-s", "--format=%T", "HEAD"),
        "status_short": run("status", "--short"),
    }


def environment(
    torch_module: Any,
    source_root: Path,
    tool_root: Path,
    configured_ref: str,
    configured_tree: str,
) -> dict[str, Any]:
    import matplotlib
    import PIL
    import ultralytics

    return {
        "captured_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch_module.__version__,
        "ultralytics": ultralytics.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__,
        "cuda_available": bool(torch_module.cuda.is_available()),
        "torch_num_threads": int(torch_module.get_num_threads()),
        "source_root": str(source_root),
        "configured_official_ref": configured_ref,
        "configured_official_tree": configured_tree,
        "source_git": git_identity(source_root),
        "tool_git": git_identity(tool_root),
    }


def write_manifest(run_dir: Path) -> list[dict[str, Any]]:
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.sha256.json":
            entries.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    write_json(run_dir / "manifest.sha256.json", entries)
    return entries
