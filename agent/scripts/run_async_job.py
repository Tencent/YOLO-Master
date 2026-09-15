"""Supervise one Agent dispatcher and persist its exit status independently of callers."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from runtime.cli.contract import _atomic_write_json


def run(job_dir: Path, dispatcher: Path) -> int:
    """Keep terminal results separate from submission metadata to avoid fast-exit races."""
    try:
        child = subprocess.Popen([sys.executable, str(dispatcher), "--request", str(job_dir / "request.json")])
        returncode = child.wait()
        result = {"status": "completed" if returncode == 0 else "failed", "returncode": returncode}
    except (OSError, ValueError) as exc:
        result = {"status": "failed", "returncode": None, "error": str(exc)}
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _atomic_write_json(job_dir / "result.json", result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), Path(sys.argv[2])))
