from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .contract import ensure_manifest_dir, json_safe


def progress_file_for_request(request: dict[str, Any], filename: str = "progress.jsonl") -> Path:
    path = ensure_manifest_dir(request) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def append_progress_event(path: Path, event: dict[str, Any]) -> dict[str, Any]:
    record = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **json_safe(event)}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


def progress_descriptor(path: Path, events: int) -> dict[str, Any]:
    return {
        "mode": "file_tail",
        "path": str(path.resolve()),
        "events": events,
        "format": "jsonl",
    }
