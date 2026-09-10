"""Serve an immutable P2 evidence directory with the standard library."""

from __future__ import annotations

import functools
import http.server
import json
import threading
import webbrowser
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_run_dir(run_dir: Path | None = None) -> Path:
    if run_dir is not None:
        resolved = run_dir.resolve()
    else:
        latest = PROJECT_ROOT / "artifacts" / "p2" / "LATEST.txt"
        if not latest.is_file():
            raise FileNotFoundError("No P2 evidence found. Run run_p2.cmd first.")
        run_id = latest.read_text(encoding="utf-8").strip()
        resolved = (latest.parent / run_id).resolve()
    if not (resolved / "demo.html").is_file() or not (resolved / "demo-index.json").is_file():
        raise FileNotFoundError(f"Not a P2 demo evidence directory: {resolved}")
    return resolved


def serve(*, host: str = "127.0.0.1", port: int = 8766, run_dir: Path | None = None, open_browser: bool = True) -> None:
    resolved = resolve_run_dir(run_dir)
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(resolved))
    server = http.server.ThreadingHTTPServer((host, port), handler)
    url = f"http://{host}:{port}/demo.html"
    summary = json.loads((resolved / "summary.json").read_text(encoding="utf-8"))
    print(f"P2 demo: {url}")
    print(f"Run: {summary['run_id']} | status={summary['status']} | Ctrl+C to stop")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
