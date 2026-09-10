"""Dependency-free local realtime dashboard for E3 routing JSONL streams."""

from __future__ import annotations

import argparse
import json
import threading
import urllib.request
import webbrowser
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from e3_p0.aggregation import aggregate_events


def load_jsonl_window(path: Path, *, limit: int = 1000) -> tuple[list[dict[str, Any]], int]:
    """Read the newest JSON objects and return both the window and valid source count.

    A partial final line is expected while a writer is active and is ignored. Corruption in
    any earlier non-empty line is surfaced instead of silently removing evidence.
    """

    if not path.is_file():
        return [], 0
    if limit <= 0:
        raise ValueError("limit must be positive")
    lines = [(number, line) for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if line.strip()]
    records: list[dict[str, Any]] = []
    for index, (line_number, line) in enumerate(lines):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            if index == len(lines) - 1:
                break
            raise ValueError(f"Invalid JSONL before final line at {path}:{line_number}") from exc
        if isinstance(value, dict):
            records.append(value)
    return records[-limit:], len(records)


def load_jsonl(path: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    """Read the newest valid JSON objects without failing on a partial final line."""

    records, _ = load_jsonl_window(path, limit=limit)
    return records


def build_snapshot(
    events: list[dict[str, Any]], *, source_event_count: int | None = None, window_limit: int | None = None
) -> dict[str, Any]:
    """Build one compact dashboard API payload from routing events."""

    aggregates = aggregate_events(events) if events else []
    families = {}
    for family in sorted({event.get("family", "unknown") for event in events}):
        family_events = [event for event in events if event.get("family") == family]
        family_aggregates = [item for item in aggregates if item["family"] == family]
        families[family] = {
            "events": len(family_events),
            "modules": len(family_aggregates),
            "mean_entropy": (
                sum(item["aggregate_metrics"]["entropy_normalized"] for item in family_aggregates)
                / len(family_aggregates)
                if family_aggregates
                else None
            ),
            "mean_gini": (
                sum(item["aggregate_metrics"]["load_gini"] for item in family_aggregates) / len(family_aggregates)
                if family_aggregates
                else None
            ),
        }
    source_count = len(events) if source_event_count is None else source_event_count
    if source_count < len(events):
        raise ValueError("source_event_count cannot be smaller than the returned event window")
    return {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "event_count": len(events),
        "source_event_count": source_count,
        "window_limit": window_limit,
        "truncated": source_count > len(events),
        "families": families,
        "aggregates": aggregates,
        "recent_events": [
            {
                "family": event.get("family"),
                "module": event.get("module", {}).get("name"),
                "entropy": event.get("routing", {}).get("entropy_normalized"),
                "gini": event.get("routing", {}).get("load_gini"),
                "sample_indices": event.get("runtime", {}).get("sample_indices"),
            }
            for event in events[-12:]
        ],
    }


def render_dashboard_html(refresh_ms: int = 1000) -> str:
    """Return the self-contained dashboard shell served by the local HTTP endpoint."""

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>YOLO-Master E3 Routing Dashboard</title>
<style>
:root{{--bg:#08111f;--panel:#101d31;--line:#263a55;--text:#e8f0fa;--muted:#91a4bb;--blue:#4da3ff;--gold:#ffbd59}}
*{{box-sizing:border-box}} body{{margin:0;background:radial-gradient(circle at top,#152944,var(--bg) 48%);color:var(--text);font:14px Inter,Segoe UI,sans-serif}}
main{{max-width:1280px;margin:auto;padding:28px}} h1{{font-size:28px;margin:0 0 6px}} .sub{{color:var(--muted);margin-bottom:22px}}
.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin-bottom:18px}} .card,.panel{{background:rgba(16,29,49,.92);border:1px solid var(--line);border-radius:14px;padding:16px;box-shadow:0 12px 36px #0005}}
.value{{font-size:28px;font-weight:700;color:var(--blue)}} .label{{color:var(--muted);margin-top:4px}} .grid{{display:grid;grid-template-columns:1.25fr .75fr;gap:16px}}
table{{width:100%;border-collapse:collapse}} th,td{{padding:9px;border-bottom:1px solid var(--line);text-align:left}} th{{color:var(--muted);font-weight:600}} .bar{{height:8px;background:#223653;border-radius:8px;overflow:hidden;min-width:120px}} .fill{{height:100%;background:linear-gradient(90deg,var(--blue),#72e2ae)}}
.status{{display:inline-flex;align-items:center;gap:7px;color:#72e2ae}} .dot{{width:8px;height:8px;border-radius:50%;background:#72e2ae;box-shadow:0 0 10px #72e2ae}}
@media(max-width:850px){{.grid{{grid-template-columns:1fr}}}}
</style></head>
<body><main><h1>YOLO-Master E3 Routing Dashboard</h1><div class="sub"><span class="status"><i class="dot"></i>live</span> · refresh {refresh_ms} ms · local read-only observer</div>
<section id="cards" class="cards"></section><section class="grid"><div class="panel"><h2>Module routing aggregates</h2><table><thead><tr><th>Family</th><th>Module</th><th>Entropy</th><th>Gini</th><th>Dominant</th></tr></thead><tbody id="modules"></tbody></table></div>
<div class="panel"><h2>Recent events</h2><table><thead><tr><th>Family</th><th>Module</th><th>Samples</th></tr></thead><tbody id="recent"></tbody></table></div></section></main>
<script>
const fmt=v=>v==null?'—':Number(v).toFixed(3); const esc=s=>String(s??'').replace(/[&<>\"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;'}}[c]));
async function refresh(){{const r=await fetch('/api/snapshot',{{cache:'no-store'}});const d=await r.json();
const eventLabel=d.truncated?`Events in window · ${{d.source_event_count}} total`:'Events in stream';
document.querySelector('#cards').innerHTML=`<div class="card"><div class="value">${{d.event_count}}</div><div class="label">${{eventLabel}}</div></div>`+Object.entries(d.families).map(([n,v])=>`<div class="card"><div class="value">${{n.toUpperCase()}}</div><div class="label">${{v.modules}} modules · H ${{fmt(v.mean_entropy)}} · G ${{fmt(v.mean_gini)}}</div></div>`).join('');
document.querySelector('#modules').innerHTML=d.aggregates.map(x=>`<tr><td>${{esc(x.family.toUpperCase())}}</td><td>${{esc(x.module)}}</td><td><div class="bar"><div class="fill" style="width:${{100*x.aggregate_metrics.entropy_normalized}}%"></div></div>${{fmt(x.aggregate_metrics.entropy_normalized)}}</td><td>${{fmt(x.aggregate_metrics.load_gini)}}</td><td>${{fmt(x.aggregate_metrics.dominant_expert_share)}}</td></tr>`).join('');
document.querySelector('#recent').innerHTML=d.recent_events.slice().reverse().map(x=>`<tr><td>${{esc((x.family||'').toUpperCase())}}</td><td>${{esc(x.module)}}</td><td>${{esc((x.sample_indices||[]).join(','))}}</td></tr>`).join('');}}
refresh();setInterval(refresh,{refresh_ms});
</script></body></html>"""


class DashboardServer(ThreadingHTTPServer):
    """HTTP server carrying an explicit JSONL source path."""

    def __init__(self, address: tuple[str, int], source: Path, refresh_ms: int = 1000):
        self.source = source
        self.refresh_ms = refresh_ms
        super().__init__(address, DashboardHandler)


class DashboardHandler(BaseHTTPRequestHandler):
    """Serve dashboard, health, and live snapshot endpoints."""

    server: DashboardServer

    def _send(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/":
            self._send(render_dashboard_html(self.server.refresh_ms).encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/healthz":
            self._send(b'{"status":"ok"}\n', "application/json")
            return
        if self.path == "/api/snapshot":
            events, source_event_count = load_jsonl_window(self.server.source)
            snapshot = build_snapshot(events, source_event_count=source_event_count, window_limit=1000)
            body = json.dumps(snapshot, ensure_ascii=False).encode("utf-8")
            self._send(body, "application/json; charset=utf-8")
            return
        self._send(b"not found\n", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def log_message(self, format: str, *args: Any) -> None:
        del format, args


def smoke_test_server(source: Path) -> dict[str, Any]:
    """Start the real HTTP server on an ephemeral port and verify all endpoints."""

    server = DashboardServer(("127.0.0.1", 0), source)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urllib.request.urlopen(base + "/healthz", timeout=5) as response:
            health_status = response.status
        with urllib.request.urlopen(base + "/", timeout=5) as response:
            html = response.read().decode("utf-8")
        with urllib.request.urlopen(base + "/api/snapshot", timeout=5) as response:
            snapshot = json.loads(response.read().decode("utf-8"))
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    return {
        "status": "PASS" if health_status == 200 and "Routing Dashboard" in html else "FAIL",
        "health_status": health_status,
        "html_contains_dashboard": "YOLO-Master E3 Routing Dashboard" in html,
        "api_event_count": snapshot["event_count"],
        "api_source_event_count": snapshot["source_event_count"],
        "api_window_limit": snapshot["window_limit"],
        "api_truncated": snapshot["truncated"],
        "api_families": sorted(snapshot["families"]),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, required=True, help="Routing JSONL file read on every refresh.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--refresh-ms", type=int, default=1000)
    parser.add_argument("--open", action="store_true", help="Open the local dashboard in the default browser.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.events.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Routing event stream does not exist: {source}")
    server = DashboardServer((args.host, args.port), source, args.refresh_ms)
    url = f"http://{args.host}:{args.port}/"
    print(f"E3 dashboard source: {source}")
    print(f"E3 dashboard URL: {url}")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
