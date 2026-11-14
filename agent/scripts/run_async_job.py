"""Supervise one Agent dispatcher and persist its exit status independently of callers."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from contextlib import suppress
from pathlib import Path

for root in (Path(__file__).resolve().parents[1], Path(__file__).resolve().parents[2]):
    sys.path.insert(0, str(root))

from runtime.cli.contract import _atomic_write_json, redact_job_value, request_secret_values


def run(job_dir: Path, dispatcher: Path, execution_request: Path | None = None) -> int:
    """Consume private inputs once; publish only sanitized child output and final status."""
    secrets: set[str] = set()
    child = None
    readers = []

    def drain(stream, target):
        try:
            for line in stream:
                target.write(redact_job_value(line, secrets))
                target.flush()
        finally:
            stream.close()

    try:
        # The two-argument legacy entry point still accepts existing request.json files.
        request_path = execution_request if execution_request is not None else job_dir / "request.json"
        try:
            request = json.loads(request_path.read_text(encoding="utf-8"))
            secrets = request_secret_values(request)
        finally:
            if execution_request is not None:
                execution_request.unlink(missing_ok=True)
        child = subprocess.Popen(
            [sys.executable, str(dispatcher)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for stream, target in ((child.stdout, sys.stdout), (child.stderr, sys.stderr)):
            reader = threading.Thread(target=drain, args=(stream, target), daemon=True)
            reader.start()
            readers.append(reader)
        try:
            child.stdin.write(json.dumps(request, ensure_ascii=False))
            child.stdin.close()
        except BrokenPipeError:
            # A fast failure may exit before consuming stdin; its exit code still wins.
            pass
        returncode = child.wait()
        for reader in readers:
            reader.join()
        result = {"status": "completed" if returncode == 0 else "failed", "returncode": returncode}
    except (OSError, ValueError) as exc:
        result = {"status": "failed", "returncode": None, "error": redact_job_value(str(exc), secrets)}
    finally:
        if execution_request is not None:
            execution_request.unlink(missing_ok=True)
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            if child.stdin is not None:
                with suppress(BrokenPipeError):
                    child.stdin.close()
            for reader in readers:
                reader.join()
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    _atomic_write_json(job_dir / "result.json", result)
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]) if len(sys.argv) > 3 else None))
