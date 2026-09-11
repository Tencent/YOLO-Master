#!/usr/bin/env python3
"""Release-baseline environment preflight for YOLO-Master.

Diagnoses the three environment failure classes recorded in
``reports/yolo_master_deep_analysis_20260821.md`` §17.2 before rerunning the
network-dependent test files that were excluded from the 2026-08-21 full-suite
baseline (``test_cli`` / ``test_integrations`` / ``test_exports`` /
``test_export_roundtrip`` / ``test_export_capability_matrix`` /
``test_solutions`` / ``test_benchmark_suite`` plus the ``test_python`` training
and multitask groups):

1. Corrupted weight caches (truncated ``.pt`` zip archives left by interrupted
   downloads).
2. Missing dataset image directories referenced by dataset YAMLs used in
   parametrized ``test_python`` cases (e.g. ``coco-multitask.yaml``).
3. Network reachability for weight/asset downloads.

Usage:
    python scripts/release_baseline_preflight.py                # diagnose only
    python scripts/release_baseline_preflight.py --fix          # delete corrupted weight caches
    python scripts/release_baseline_preflight.py --run-blocked  # run blocked test files when green
    python scripts/release_baseline_preflight.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Files excluded from the 2026-08-21 baseline due to offline network / load.
BLOCKED_TEST_FILES = [
    "tests/test_cli.py",
    "tests/test_integrations.py",
    "tests/test_exports.py",
    "tests/test_export_roundtrip.py",
    "tests/test_export_capability_matrix.py",
    "tests/test_solutions.py",
    "tests/test_benchmark_suite.py",
]

# Hosts required for weight and dataset downloads.
NETWORK_PROBES = [
    ("github.com", 443),
    ("objects.githubusercontent.com", 443),
    ("raw.githubusercontent.com", 443),
]

# Dataset YAMLs exercised by parametrized test_python multitask cases.
DATASET_YAMLS = [
    "ultralytics/cfg/datasets/coco-multitask.yaml",
    "ultralytics/cfg/datasets/coco8.yaml",
    "ultralytics/cfg/datasets/coco128.yaml",
]


def find_corrupt_weights() -> list[dict]:
    """Scan weight directories for truncated or non-zip ``.pt`` archives."""
    weight_dirs = [ROOT / "weights", ROOT / "datasets"]
    try:
        from ultralytics.utils import SETTINGS

        weight_dirs.append(Path(SETTINGS.get("weights_dir", ROOT / "weights")))
    except Exception:
        pass  # settings unavailable; repo-local scan is sufficient
    corrupt, seen = [], set()
    for directory in weight_dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.pt")):
            if path in seen:
                continue
            seen.add(path)
            if path.stat().st_size == 0 or not zipfile.is_zipfile(path):
                corrupt.append({"path": str(path), "size_bytes": path.stat().st_size})
    return corrupt


def check_datasets() -> list[dict]:
    """Verify that image directories referenced by dataset YAMLs exist.

    Relative ``path`` entries resolve against the Ultralytics ``datasets_dir``
    setting (falling back to the repo root), mirroring trainer resolution.
    """
    import yaml

    try:
        from ultralytics.utils import SETTINGS

        datasets_dir = Path(SETTINGS.get("datasets_dir", ROOT / "datasets"))
    except Exception:
        datasets_dir = ROOT / "datasets"

    problems = []
    for rel in DATASET_YAMLS:
        yaml_path = ROOT / rel
        if not yaml_path.is_file():
            problems.append({"yaml": rel, "issue": "yaml_missing"})
            continue
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        base = Path(data.get("path") or yaml_path.parent)
        if not base.is_absolute():
            base = datasets_dir / base
        missing = []
        for key in ("train", "val"):
            entry = data.get(key)
            if not entry:
                continue
            candidates = entry if isinstance(entry, list) else [entry]
            for item in candidates:
                target = base / str(item)
                if not target.exists():
                    missing.append(str(target))
        if missing:
            problems.append({"yaml": rel, "issue": "images_missing", "missing": missing})
    return problems


def check_network(timeout: float = 3.0) -> list[dict]:
    """Probe TCP reachability of hosts required for downloads."""
    results = []
    for host, port in NETWORK_PROBES:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                results.append({"host": host, "reachable": True})
        except OSError as exc:
            results.append({"host": host, "reachable": False, "error": str(exc)})
    return results


def _resolved_install_paths(location: str, editable_source: str) -> list[Path]:
    """Resolve metadata location strings to absolute paths."""
    paths: list[Path] = []
    if location:
        path = Path(location).expanduser()
        paths.append(path.resolve() if str(path) not in {".", ""} else Path.cwd().resolve())
    if editable_source:
        url = editable_source[7:] if editable_source.startswith("file://") else editable_source
        if url:
            paths.append(Path(url).expanduser().resolve())
    return paths


def check_cli_installation() -> dict:
    """Verify the ``yolo`` CLI resolves to THIS workspace, not a stale editable install.

    CLI subprocess tests spawn the globally installed ``yolo`` entry point; if a
    different workspace's editable install shadows this repo, test_cli results
    validate the wrong codebase (observed 2026-08-22: CLI resolved to the
    v260720 workspace).
    """
    import importlib.metadata

    try:
        result = subprocess.run(["yolo", "version"], capture_output=True, text=True, timeout=60)
        version = result.stdout.strip() if result.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        version = None
    location, editable_source, import_path = "", "", ""
    try:
        dist = importlib.metadata.distribution("ultralytics")
        location = str(dist.locate_file(""))
        direct = dist.read_text("direct_url.json")
        if direct:
            editable_source = json.loads(direct).get("url", "")
    except Exception:
        pass
    try:
        import ultralytics

        import_path = str(Path(ultralytics.__file__).resolve())
    except Exception:
        pass
    root = ROOT.resolve()
    metadata_ok = any(_is_this_repo(path, root) for path in _resolved_install_paths(location, editable_source))
    import_ok = bool(import_path) and _is_this_repo(Path(import_path).resolve(), root)
    return {
        "yolo_version": version,
        "install_location": location,
        "editable_source": editable_source,
        "import_path": import_path,
        "import_points_to_this_repo": import_ok,
        "metadata_points_to_this_repo": metadata_ok,
        "points_to_this_repo": import_ok and (metadata_ok or not editable_source),
    }


def _is_this_repo(path: Path, root: Path) -> bool:
    """Return whether ``path`` is the repo root or a file under it."""
    try:
        path.resolve().relative_to(root)
        return True
    except ValueError:
        return root == path.resolve() or str(root) in str(path)


def run_blocked_tests(python: str) -> int:
    """Run the baseline-excluded test files; only call when network is green."""
    cmd = [python, "-m", "pytest", *BLOCKED_TEST_FILES, "-q", "-p", "no:cacheprovider", "-n", "4", "--timeout=300"]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd, cwd=ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fix", action="store_true", help="Delete corrupted weight caches so they re-download")
    parser.add_argument("--run-blocked", action="store_true", help="Run baseline-excluded test files when checks pass")
    parser.add_argument("--json", type=Path, default=None, help="Write the report to this JSON path")
    args = parser.parse_args()

    report = {
        "corrupt_weights": find_corrupt_weights(),
        "dataset_problems": check_datasets(),
        "network": check_network(),
        "cli_installation": check_cli_installation(),
    }
    network_ok = all(item["reachable"] for item in report["network"])
    report["network_ok"] = network_ok

    print("== Release baseline preflight ==")
    for item in report["corrupt_weights"]:
        print(f"  CORRUPT weight: {item['path']} ({item['size_bytes']} bytes)")
    for item in report["dataset_problems"]:
        print(f"  DATASET issue: {item['yaml']}: {item['issue']} {item.get('missing', '')}")
    for item in report["network"]:
        status = "ok" if item["reachable"] else f"unreachable ({item.get('error', '')})"
        print(f"  NET {item['host']}: {status}")
    cli = report["cli_installation"]
    if cli["points_to_this_repo"]:
        print(f"  CLI install: ok ({cli['yolo_version']})")
    else:
        print(
            "  CLI install: WARNING — global yolo CLI resolves outside this repo: "
            f"{cli.get('editable_source') or cli.get('import_path') or cli.get('install_location')}"
        )
        print("    test_cli subprocess results validate the installed package, not this workspace.")

    if args.fix and report["corrupt_weights"]:
        if not network_ok:
            print("Refusing --fix while network is unreachable: caches could not be re-downloaded.")
            report["fix_applied"] = False
        else:
            for item in report["corrupt_weights"]:
                Path(item["path"]).unlink()
                print(f"  deleted {item['path']} (will re-download on next use)")
            report["fix_applied"] = True
            report["corrupt_weights"] = find_corrupt_weights()

    blockers = bool(report["corrupt_weights"] or not network_ok)
    # Dataset gaps only affect parametrized test_python cases, not the blocked
    # test files gated by --run-blocked; report them as warnings instead.
    report["baseline_ready"] = not blockers and not report["dataset_problems"]
    report["run_blocked_ready"] = not blockers
    print(f"baseline_ready: {report['baseline_ready']}  run_blocked_ready: {report['run_blocked_ready']}")

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.json}")

    if args.run_blocked:
        if blockers:
            print("Blockers remain; not running blocked test files.")
            return 1
        return run_blocked_tests(sys.executable)
    return 0 if report["baseline_ready"] else 1


if __name__ == "__main__":
    sys.exit(main())
