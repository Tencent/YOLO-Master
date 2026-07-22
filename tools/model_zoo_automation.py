#!/usr/bin/env python3
"""Validate model-zoo submissions and run reproducible, allowlisted benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml

SCHEMA_VERSION = 1
ALLOWED_HOSTS = {
    "github.com",
    "objects.githubusercontent.com",
    "github-releases.githubusercontent.com",
    "huggingface.co",
    "cdn-lfs.huggingface.co",
}
ALLOWED_DATASETS = {"coco8.yaml", "coco.yaml"}
MAX_WEIGHT_BYTES = 2 * 1024**3
REQUIRED = {"schema_version", "name", "description", "license", "weights", "benchmark"}


def _load(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("submission must be a YAML mapping")
    return data


def validate_submission(path: Path, changed_files: list[str] | None = None) -> dict:
    data = _load(path)
    errors: list[str] = []
    missing = sorted(REQUIRED - data.keys())
    if missing:
        errors.append(f"missing required fields: {', '.join(missing)}")
    if data.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("name", "description", "license"):
        if key in data and (not isinstance(data[key], str) or not data[key].strip()):
            errors.append(f"{key} must be a non-empty string")

    weights = data.get("weights", {})
    if not isinstance(weights, dict):
        errors.append("weights must be a mapping")
        weights = {}
    url = weights.get("url", "")
    parsed = urlparse(url) if isinstance(url, str) else None
    if not parsed or parsed.scheme != "https" or parsed.hostname not in ALLOWED_HOSTS:
        errors.append("weights.url must use HTTPS on an allowlisted host")
    if Path(parsed.path).suffix.lower() != ".onnx" if parsed else True:
        errors.append("only ONNX weights are accepted; repository .pt files and pickle formats are forbidden")
    digest = weights.get("sha256", "")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest.lower()):
        errors.append("weights.sha256 must be a 64-character hexadecimal digest")
    size = weights.get("size_bytes")
    if not isinstance(size, int) or not 0 < size <= MAX_WEIGHT_BYTES:
        errors.append(f"weights.size_bytes must be between 1 and {MAX_WEIGHT_BYTES}")

    benchmark = data.get("benchmark", {})
    if not isinstance(benchmark, dict):
        errors.append("benchmark must be a mapping")
        benchmark = {}
    if benchmark.get("task") != "detect":
        errors.append("benchmark.task must be detect")
    if benchmark.get("dataset") not in ALLOWED_DATASETS:
        errors.append(f"benchmark.dataset must be one of {sorted(ALLOWED_DATASETS)}")
    for key, low, high in (("imgsz", 32, 2048), ("batch", 1, 128)):
        value = benchmark.get(key)
        if not isinstance(value, int) or not low <= value <= high:
            errors.append(f"benchmark.{key} must be an integer in [{low}, {high}]")

    if changed_files is not None:
        submissions = [p for p in changed_files if p.startswith("model-zoo/submissions/") and p.endswith((".yaml", ".yml"))]
        forbidden = [p for p in changed_files if Path(p).suffix.lower() in {".pt", ".pth", ".ckpt", ".onnx", ".engine"}]
        if len(submissions) != 1:
            errors.append("PR must add or update exactly one model-zoo/submissions YAML file")
        if forbidden:
            errors.append(f"weight binaries must not be committed: {', '.join(forbidden)}")
    if errors:
        raise ValueError("\n".join(f"- {error}" for error in errors))
    return data


def download_weights(data: dict, destination: Path) -> Path:
    weights = data["weights"]
    request = urllib.request.Request(weights["url"], headers={"User-Agent": "YOLO-Master-model-zoo/1"})
    hasher, total = hashlib.sha256(), 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(request, timeout=60) as response, destination.open("wb") as output:
        final_host = urlparse(response.geturl()).hostname
        if final_host not in ALLOWED_HOSTS:
            raise ValueError(f"redirected to non-allowlisted host: {final_host}")
        content_length = response.headers.get("Content-Length")
        if content_length and int(content_length) != weights["size_bytes"]:
            raise ValueError("Content-Length does not match declared size_bytes")
        while chunk := response.read(1024 * 1024):
            total += len(chunk)
            if total > weights["size_bytes"] or total > MAX_WEIGHT_BYTES:
                raise ValueError("download exceeds declared or maximum size")
            hasher.update(chunk)
            output.write(chunk)
    if total != weights["size_bytes"]:
        raise ValueError(f"downloaded size mismatch: expected {weights['size_bytes']}, got {total}")
    if hasher.hexdigest().lower() != weights["sha256"].lower():
        raise ValueError("downloaded weight SHA-256 mismatch")
    return destination


def run_benchmark(data: dict, output_dir: Path) -> dict:
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    weight_path = download_weights(data, output_dir / "model.onnx")
    cfg = data["benchmark"]
    model = YOLO(str(weight_path), task="detect")
    metrics = model.val(
        data=cfg["dataset"], imgsz=cfg["imgsz"], batch=cfg["batch"], device="cpu",
        project=str(output_dir), name="validation", exist_ok=True, plots=False, verbose=False,
    )
    results = {str(k): float(v) for k, v in getattr(metrics, "results_dict", {}).items() if isinstance(v, (int, float))}
    speed = {str(k): float(v) for k, v in getattr(metrics, "speed", {}).items() if isinstance(v, (int, float))}
    report = {
        "schema_version": SCHEMA_VERSION,
        "model": data["name"],
        "weights_sha256": data["weights"]["sha256"],
        "benchmark": cfg,
        "metrics": results,
        "speed_ms": speed,
    }
    (output_dir / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [f"## Model Zoo Benchmark: {data['name']}", "", "| Metric | Value |", "|---|---:|"]
    lines += [f"| `{key}` | {value:.6g} |" for key, value in sorted(results.items())]
    lines += ["", f"Weights SHA-256: `{data['weights']['sha256']}`"]
    (output_dir / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("validate")
    check.add_argument("submission", type=Path)
    check.add_argument("--changed-files-json", type=Path)
    bench = sub.add_parser("benchmark")
    bench.add_argument("submission", type=Path)
    bench.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        changed = None
        if getattr(args, "changed_files_json", None):
            raw = json.loads(args.changed_files_json.read_text(encoding="utf-8"))
            changed = [item["filename"] if isinstance(item, dict) else item for item in raw]
        data = validate_submission(args.submission, changed)
        if args.command == "benchmark":
            run_benchmark(data, args.output)
        else:
            print(f"valid model-zoo submission: {args.submission}")
    except Exception as exc:
        print(f"model-zoo automation failed:\n{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
