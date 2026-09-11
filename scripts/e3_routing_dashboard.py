#!/usr/bin/env python3
"""Serve the local, read-only E3 Routing Observatory."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mimetypes
import re
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = ROOT / "reports/e3_routing_observability/dashboard"
EXAMPLE_DIR = ROOT / "reports/e3_routing_observability/results"
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
SUPPORTED_SCHEMAS = {
    "e3.cuda_final.v1",
    "e3.routing_snapshot.v1",
    "e3.overhead_benchmark.v1",
    "e3.training_smoke.v1",
    1,  # TrainingTelemetry aggregate.
}


def _inside(path: Path, root: Path) -> bool:
    """Return whether a resolved path belongs to a resolved root on Python 3.8+."""
    return path == root or root in path.parents


def _identifier(value: str, length: int) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:length]


def _json_safe_number(value: Any) -> float | int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(value) else None


def _normalise_vector(value: Any) -> tuple[list[float | int | None], bool]:
    if not isinstance(value, list):
        return [], False
    vector = [_json_safe_number(item) for item in value]
    return vector, any(item is None for item in vector)


def _read_json(path: Path, root: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Read one bounded JSON object without following a symlink outside its root."""
    try:
        resolved = path.resolve(strict=True)
        if not _inside(resolved, root):
            return None, "outside_root"
        if resolved.stat().st_size > MAX_JSON_BYTES:
            return None, "file_too_large"
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, "invalid_json"
    if not isinstance(payload, dict):
        return None, None
    if payload.get("schema_version") not in SUPPORTED_SCHEMAS:
        return None, None
    if payload.get("schema_version") == 1 and not isinstance(payload.get("ranks"), list):
        return None, None
    return payload, None


def _layer_values(layers: Any) -> list[dict[str, Any]]:
    if isinstance(layers, dict):
        values = layers.values()
    elif isinstance(layers, list):
        values = layers
    else:
        return []
    return [dict(value) for value in values if isinstance(value, dict)]


def _normalise_layers(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the shared layer contract from a snapshot or telemetry aggregate."""
    if payload.get("schema_version") == "e3.routing_snapshot.v1":
        if isinstance(payload.get("families"), dict):
            layers = []
            for family_name, family in payload["families"].items():
                if not isinstance(family, dict):
                    continue
                for layer in _layer_values(family.get("layers")):
                    layer.setdefault("family", family_name)
                    layers.append(layer)
            return layers
        return _layer_values(payload.get("layers"))
    if payload.get("schema_version") == 1:
        ranks = payload.get("ranks")
        if not isinstance(ranks, list) or not ranks or not isinstance(ranks[0], dict):
            return []
        routing = ranks[0].get("routing", {})
        last = routing.get("last", {}) if isinstance(routing, dict) else {}
        return _layer_values(last.get("layers") if isinstance(last, dict) else None)
    return []


def _routing_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != 1:
        return {}
    ranks = payload.get("ranks")
    if not isinstance(ranks, list) or not ranks or not isinstance(ranks[0], dict):
        return {}
    routing = ranks[0].get("routing", {})
    return routing if isinstance(routing, dict) else {}


def _normalise_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    """Keep browser-facing benchmark data compact and intentionally bounded."""
    families = payload.get("families") if isinstance(payload.get("families"), dict) else {}
    runs = payload.get("runs") if isinstance(payload.get("runs"), list) else []
    run_fields = {
        "condition",
        "count",
        "family",
        "mean_milliseconds",
        "median_milliseconds",
        "memory",
        "p95_milliseconds",
        "pair",
        "samples",
        "samples_per_second",
    }
    return {
        "schema_version": payload.get("schema_version"),
        "passed": payload.get("passed"),
        "environment": payload.get("environment") if isinstance(payload.get("environment"), dict) else {},
        "git": payload.get("git") if isinstance(payload.get("git"), dict) else {},
        "families": families,
        "runs": [
            {key: value for key, value in run.items() if key in run_fields} for run in runs if isinstance(run, dict)
        ],
    }


def _normalise_training_smoke(payload: dict[str, Any]) -> dict[str, dict[str, Any]] | None:
    """Accept both the original run map and the compact live-smoke family summary."""
    runs = payload.get("runs")
    if isinstance(runs, dict):
        return runs
    families = payload.get("families")
    if not isinstance(families, dict):
        return None
    normalized = {}
    for family, record in families.items():
        if not isinstance(record, dict):
            continue
        normalized[str(family)] = {
            "routed_layers": record.get("routed_layers"),
            "routing_observations": record.get("routing_observations"),
            "tensorboard_routing_scalar_tags": record.get(
                "tensorboard_routing_scalar_tags", record.get("routing_scalar_tags")
            ),
            "invalid_layers": record.get("invalid_layers", 0),
            "unsupported_layers": record.get("unsupported_layers", 0),
            "aux_statuses": record.get("aux_statuses") if isinstance(record.get("aux_statuses"), list) else [],
        }
    return normalized or None


def _normalise_layer(layer: dict[str, Any], index: int) -> dict[str, Any]:
    usage, invalid_usage = _normalise_vector(layer.get("expert_usage"))
    probabilities, invalid_probabilities = _normalise_vector(layer.get("mean_router_probs"))
    family = str(layer.get("family") or "unknown")
    aux = layer.get("aux_loss") if isinstance(layer.get("aux_loss"), dict) else {}
    num_experts = layer.get("num_experts")
    if not isinstance(num_experts, int) or num_experts < 1:
        num_experts = max(len(usage), len(probabilities), 0)
    raw_entropy = _json_safe_number(layer.get("entropy"))
    entropy = _json_safe_number(layer.get("normalized_entropy"))
    dominant = max((float(item) for item in usage if item is not None), default=None)
    minimum = min((float(item) for item in usage if item is not None), default=None)
    imbalance = dominant - minimum if dominant is not None and minimum is not None else None
    issues = []
    if num_experts and (len(usage) != num_experts or len(probabilities) != num_experts):
        issues.append("expert_vector_length_mismatch")
    if invalid_usage or invalid_probabilities:
        issues.append("non_finite_expert_value")
    if entropy is None or not 0 <= float(entropy) <= 1:
        issues.append("invalid_normalized_entropy")
    return {
        "id": _identifier(f"{layer.get('layer_name', index)}:{index}", 12),
        "name": str(layer.get("layer_name") or f"layer.{index}"),
        "module_type": str(layer.get("module_type") or "unknown"),
        "family": family,
        "num_experts": num_experts,
        "top_k": layer.get("top_k"),
        "routing_axis": str(layer.get("routing_axis") or "unknown"),
        "dispatch_policy": str(layer.get("dispatch_policy") or "unknown"),
        "probability_shape": layer.get("probability_shape") if isinstance(layer.get("probability_shape"), list) else [],
        "expert_usage": usage,
        "mean_router_probs": probabilities,
        "entropy": raw_entropy,
        "normalized_entropy": entropy,
        "dominant_share": dominant,
        "load_spread": imbalance,
        "aux": {
            "status": str(aux.get("status") or "unavailable"),
            "observed": _json_safe_number(aux.get("observed")),
            "configured": bool(aux.get("configured", False)),
        },
        "issues": issues,
    }


def _normalise_routing_map(value: Any) -> dict[str, Any] | None:
    """Validate bounded native-rendering data from one representative routing layer."""
    if not isinstance(value, dict) or value.get("kind") not in {"global", "spatial"}:
        return None
    num_experts = value.get("num_experts")
    probabilities = value.get("probabilities")
    if not isinstance(num_experts, int) or not 1 <= num_experts <= 64 or not isinstance(probabilities, list):
        return None
    if value["kind"] == "global":
        values = [_json_safe_number(item) for item in probabilities]
        if len(values) != num_experts or any(item is None for item in values):
            return None
        return {
            "kind": "global",
            "layer_name": str(value.get("layer_name") or "routing"),
            "num_experts": num_experts,
            "probabilities": values,
        }

    height, width = value.get("height"), value.get("width")
    if not isinstance(height, int) or not isinstance(width, int) or not (1 <= height <= 256 and 1 <= width <= 256):
        return None
    if len(probabilities) != num_experts:
        return None
    normalized = []
    for expert in probabilities:
        if not isinstance(expert, list) or len(expert) != height:
            return None
        rows = []
        for row in expert:
            if not isinstance(row, list) or len(row) != width:
                return None
            values = [_json_safe_number(item) for item in row]
            if any(item is None for item in values):
                return None
            rows.append(values)
        normalized.append(rows)
    return {
        "kind": "spatial",
        "layer_name": str(value.get("layer_name") or "routing"),
        "num_experts": num_experts,
        "height": height,
        "width": width,
        "input_image": str(value.get("input_image") or ""),
        "probabilities": normalized,
    }


def _workspace_name(directory: Path) -> str:
    return directory.name.replace("_", " ").replace("-", " ") or "routing run"


@dataclass(frozen=True)
class EvidenceFile:
    id: str
    path: Path
    name: str
    size: int
    schema: str
    root: Path
    media_type: str = "application/json; charset=utf-8"


class WorkspaceIndex:
    """Discover supported evidence and expose normalised, path-safe workspaces."""

    def __init__(self, roots: list[Path]):
        resolved = []
        for root in roots:
            candidate = root.expanduser().resolve()
            if candidate.is_dir() and candidate not in resolved:
                resolved.append(candidate)
        self.roots = resolved
        self.workspaces: dict[str, dict[str, Any]] = {}
        self.files: dict[tuple[str, str], EvidenceFile] = {}
        self.scan_errors: list[dict[str, str]] = []
        self.refresh()

    def refresh(self) -> None:
        discovered: list[tuple[Path, dict[str, Any], Path]] = []
        errors = []
        for root in self.roots:
            for path in sorted(root.rglob("*.json")):
                payload, error = _read_json(path, root)
                if error:
                    errors.append({"path": str(path.relative_to(root)), "reason": error})
                    continue
                if payload is not None:
                    discovered.append((path.resolve(), payload, root))
        manifests = [
            path.parent for path, payload, _ in discovered if payload.get("schema_version") == "e3.cuda_final.v1"
        ]
        grouped: dict[Path, list[tuple[Path, dict[str, Any], Path]]] = {}
        for path, payload, root in discovered:
            package = next(
                (
                    directory
                    for directory in manifests
                    if path == directory / "manifest.json" or directory in path.parents
                ),
                None,
            )
            grouped.setdefault(package or path.parent.resolve(), []).append((path, payload, root))
        workspaces = {}
        files = {}
        for directory, items in grouped.items():
            workspace = self._normalise_workspace(directory, items)
            workspaces[workspace["id"]] = workspace
            for item in workspace.pop("_files"):
                files[(workspace["id"], item.id)] = item
        self.workspaces = workspaces
        self.files = files
        self.scan_errors = errors

    def _normalise_workspace(self, directory: Path, items: list[tuple[Path, dict[str, Any], Path]]) -> dict[str, Any]:
        root = items[0][2]
        relative = str(directory.relative_to(root)) if directory != root else "."
        workspace_id = _identifier(f"{root}:{directory}", 16)
        raw_layers = []
        limitations: list[str] = []
        schemas = []
        benchmark = None
        training_runs = None
        metadata: dict[str, Any] = {}
        snapshot_metadata: dict[str, Any] = {}
        training_environment: dict[str, Any] = {}
        models: dict[str, dict[str, str]] = {}
        git: dict[str, Any] = {}
        routing_summary: dict[str, Any] = {}
        routing_maps: dict[str, dict[str, Any]] = {}
        evidence_files = []

        for path, payload, _ in items:
            schema = payload.get("schema_version")
            schemas.append(str(schema))
            raw_layers.extend(_normalise_layers(payload))
            limitations.extend(item for item in payload.get("limitations", []) if isinstance(item, str))
            if schema == "e3.routing_snapshot.v1":
                snapshot_metadata = {
                    key: payload.get(key)
                    for key in ("dataset", "device", "imgsz", "seed", "scope", "weights")
                    if payload.get(key) is not None
                }
                metadata.update(snapshot_metadata)
                if isinstance(payload.get("sample"), dict):
                    snapshot_metadata["sample"] = payload["sample"]
                families = payload.get("families") if isinstance(payload.get("families"), dict) else {}
                for family_name, family in families.items():
                    if not isinstance(family, dict):
                        continue
                    models[str(family_name)] = {
                        key: str(family[key]) for key in ("model", "model_sha256") if family.get(key) is not None
                    }
                    routing_map = _normalise_routing_map(family.get("routing_map"))
                    if routing_map is not None:
                        routing_maps[str(family_name)] = routing_map
            elif schema == "e3.cuda_final.v1":
                environment = payload.get("environment") if isinstance(payload.get("environment"), dict) else {}
                config = payload.get("config") if isinstance(payload.get("config"), dict) else {}
                dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
                metadata.update(
                    {
                        "device": environment.get("gpu") or config.get("device"),
                        "dataset": dataset.get("name") or config.get("data"),
                        "imgsz": config.get("imgsz"),
                        "batch": payload.get("batch"),
                    }
                )
                if isinstance(payload.get("git"), dict):
                    git.update(payload["git"])
                training = payload.get("training") if isinstance(payload.get("training"), dict) else {}
                evaluation = payload.get("evaluation") if isinstance(payload.get("evaluation"), dict) else {}
                routing_families = (
                    payload.get("routing", {}).get("families", {}) if isinstance(payload.get("routing"), dict) else {}
                )
                training_runs = {}
                for family_name, record in training.items():
                    if not isinstance(record, dict):
                        continue
                    test_record = evaluation.get(family_name) if isinstance(evaluation.get(family_name), dict) else {}
                    route_record = (
                        routing_families.get(family_name) if isinstance(routing_families.get(family_name), dict) else {}
                    )
                    training_runs[str(family_name)] = {
                        "epochs": record.get("epochs"),
                        "routed_layers": route_record.get("captured_layers"),
                        "metrics": record.get("metrics") if isinstance(record.get("metrics"), dict) else {},
                        "test_metrics": (
                            test_record.get("metrics") if isinstance(test_record.get("metrics"), dict) else {}
                        ),
                    }
            elif schema == "e3.overhead_benchmark.v1":
                benchmark = _normalise_benchmark(payload)
                if isinstance(payload.get("environment"), dict):
                    metadata.update(payload["environment"])
                if isinstance(payload.get("git"), dict):
                    git.update(payload["git"])
            elif schema == "e3.training_smoke.v1":
                training_runs = _normalise_training_smoke(payload)
                if isinstance(payload.get("environment"), dict):
                    training_environment = payload["environment"]
                    metadata.update(payload["environment"])
            elif schema == 1:
                ranks = payload.get("ranks")
                if isinstance(ranks, list) and ranks and isinstance(ranks[0], dict):
                    rank = ranks[0]
                    if isinstance(rank.get("metadata"), dict):
                        metadata.update(rank["metadata"])
                    if isinstance(rank.get("steps"), dict):
                        metadata["steps"] = rank["steps"]
                routing_summary = _routing_summary(payload)

            file_id = _identifier(str(path), 12)
            evidence_files.append(
                EvidenceFile(file_id, path, path.name, path.stat().st_size, str(payload.get("schema_version")), root)
            )
            if schema == "e3.routing_snapshot.v1":
                referenced: list[tuple[str, str]] = []
                for family in (payload.get("families") or {}).values():
                    if not isinstance(family, dict):
                        continue
                    if isinstance(family.get("static_figure"), str):
                        referenced.append((family["static_figure"], "image/png"))
                    visualizations = family.get("routing_visualizations")
                    if isinstance(visualizations, dict):
                        referenced.extend(
                            (value, "image/png") for value in visualizations.values() if isinstance(value, str)
                        )
                    routing_map = family.get("routing_map")
                    if isinstance(routing_map, dict) and isinstance(routing_map.get("input_image"), str):
                        media_type = (
                            "image/png" if routing_map["input_image"].lower().endswith(".png") else "image/jpeg"
                        )
                        referenced.append((routing_map["input_image"], media_type))
                for relative_name, media_type in referenced:
                    try:
                        artifact = (path.parent / relative_name).resolve(strict=True)
                        if (
                            not _inside(artifact, root)
                            or artifact.suffix.lower() not in {".png", ".jpg", ".jpeg"}
                            or artifact.stat().st_size > MAX_ARTIFACT_BYTES
                        ):
                            continue
                    except OSError:
                        continue
                    if any(item.path == artifact for item in evidence_files):
                        continue
                    evidence_files.append(
                        EvidenceFile(
                            _identifier(str(artifact), 12),
                            artifact,
                            artifact.name,
                            artifact.stat().st_size,
                            "image",
                            root,
                            media_type,
                        )
                    )

        layers = []
        seen_layers = set()
        for index, raw_layer in enumerate(raw_layers):
            layer = _normalise_layer(raw_layer, index)
            identity = (layer["family"], layer["name"], layer["module_type"])
            if identity not in seen_layers:
                seen_layers.add(identity)
                layers.append(layer)
        family_counts: dict[str, int] = {}
        for layer in layers:
            family_counts[layer["family"]] = family_counts.get(layer["family"], 0) + 1
        invalid_count = sum(bool(layer["issues"]) for layer in layers)
        unsupported_count = int(routing_summary.get("unsupported_layers") or 0)
        collapsed_count = int(routing_summary.get("collapsed_layers_max") or 0)
        workspace_issues = []
        if not layers:
            workspace_issues.append("no_routing_layers")
        if invalid_count:
            workspace_issues.append("invalid_layers")
        if unsupported_count:
            workspace_issues.append("unsupported_layers")
        if collapsed_count:
            workspace_issues.append("collapsed_layers")
        status = (
            "healthy" if not workspace_issues else ("empty" if workspace_issues == ["no_routing_layers"] else "warning")
        )

        return {
            "api_version": "e3.dashboard.v1",
            "id": workspace_id,
            "name": _workspace_name(directory),
            "root": root.name,
            "relative_path": relative,
            "schemas": sorted(set(schemas)),
            "status": status,
            "metadata": metadata,
            "snapshot_metadata": snapshot_metadata,
            "training_environment": training_environment,
            "models": models,
            "routing_maps": routing_maps,
            "git": git,
            "summary": {
                "layers": len(layers),
                "families": family_counts,
                "has_benchmark": benchmark is not None,
                "invalid_layers": invalid_count,
                "unsupported_layers": unsupported_count,
                "collapsed_layers": collapsed_count,
                "routing_observations": routing_summary.get("observations"),
                "sampling_interval_steps": routing_summary.get("sampling_interval_steps"),
                "visual_artifacts": sum(item.media_type == "image/png" for item in evidence_files),
            },
            "issues": workspace_issues,
            "layers": layers,
            "benchmark": benchmark,
            "training_runs": training_runs,
            "limitations": list(dict.fromkeys(limitations)),
            "evidence": [
                {
                    "id": item.id,
                    "name": item.name,
                    "relative_path": str(item.path.relative_to(item.root)),
                    "size": item.size,
                    "schema": item.schema,
                    "media_type": item.media_type,
                }
                for item in evidence_files
            ],
            "_files": evidence_files,
        }

    def list_payload(self) -> dict[str, Any]:
        runs = []
        for workspace in sorted(
            self.workspaces.values(),
            key=lambda item: (
                not item["summary"]["has_benchmark"],
                item["status"] != "healthy",
                item["name"],
            ),
        ):
            runs.append(
                {
                    key: workspace[key]
                    for key in ("id", "name", "root", "relative_path", "schemas", "status", "summary", "metadata")
                }
            )
        return {
            "api_version": "e3.dashboard.v1",
            "roots": [root.name for root in self.roots],
            "runs": runs,
            "scan_errors": self.scan_errors,
        }


class ObservatoryHandler(BaseHTTPRequestHandler):
    """Serve fixed dashboard assets and read-only workspace APIs."""

    index: WorkspaceIndex

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[routing-observatory] {self.address_string()} {format % args}")

    def _send_bytes(self, content: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=False).encode(), "application/json; charset=utf-8", status)

    def _send_error_json(self, reason: str, status: HTTPStatus) -> None:
        self._send_json({"error": reason, "status": int(status)}, status)

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/api/v1/runs":
            self._send_json(self.index.list_payload())
            return
        match = re.fullmatch(r"/api/v1/runs/([0-9a-f]{16})", path)
        if match:
            workspace = self.index.workspaces.get(match.group(1))
            self._send_json(workspace) if workspace else self._send_error_json("run_not_found", HTTPStatus.NOT_FOUND)
            return
        file_match = re.fullmatch(r"/api/v1/runs/([0-9a-f]{16})/files/([0-9a-f]{12})", path)
        if file_match:
            item = self.index.files.get((file_match.group(1), file_match.group(2)))
            if not item:
                self._send_error_json("evidence_not_found", HTTPStatus.NOT_FOUND)
                return
            try:
                resolved = item.path.resolve(strict=True)
                if not _inside(resolved, item.root) or resolved.stat().st_size > MAX_ARTIFACT_BYTES:
                    raise OSError
                self._send_bytes(resolved.read_bytes(), item.media_type)
            except OSError:
                self._send_error_json("evidence_unavailable", HTTPStatus.NOT_FOUND)
            return
        assets = {"/": "index.html", "/index.html": "index.html", "/app.js": "app.js", "/styles.css": "styles.css"}
        filename = assets.get(path)
        if filename:
            file_path = DASHBOARD_DIR / filename
            content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
            self._send_bytes(file_path.read_bytes(), f"{content_type}; charset=utf-8")
            return
        self._send_error_json("not_found", HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path != "/api/v1/refresh":
            self._send_error_json("not_found", HTTPStatus.NOT_FOUND)
            return
        self.index.refresh()
        self._send_json(self.index.list_payload())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, action="append", default=[], help="run directory to scan; repeatable")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-example", action="store_true", help="do not include the committed E3 example")
    parser.add_argument("--allow-remote", action="store_true", help="allow binding to a non-loopback host")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.allow_remote:
        raise SystemExit("Refusing a non-loopback host without --allow-remote")
    roots = args.root or [ROOT / "runs"]
    if not args.no_example:
        roots.append(EXAMPLE_DIR)
    index = WorkspaceIndex(roots)
    handler = type("BoundObservatoryHandler", (ObservatoryHandler,), {"index": index})
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Routing Observatory: http://{args.host}:{args.port}/")
    print(f"Scanning: {', '.join(str(root) for root in index.roots)}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nRouting Observatory stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
