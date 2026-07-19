"""Generate and verify the upstream v8.4.101 integrity contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_REF = "v8.4.101"
UPSTREAM_COMMIT = "579b389c87c04b7f6a9a247730dac04922be8007"
MANIFEST_PATH = ROOT / "docs/governance/upstream-v8.4.101-manifest.json"

EXACT_ROOTS = (
    "ultralytics/nn/backends",
    "ultralytics/utils/export",
    "ultralytics/cfg/models/26",
)
EXTENSION_POINTS = (
    "ultralytics/nn/tasks.py",
    "ultralytics/utils/loss.py",
    "ultralytics/engine/trainer.py",
    "ultralytics/engine/exporter.py",
    "ultralytics/nn/modules/__init__.py",
    "ultralytics/cfg/__init__.py",
    "ultralytics/cfg/default.yaml",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_lf(path: Path) -> str:
    """Hash text with Git's Windows checkout line endings normalized to LF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _tracked_files(items: Iterable[str]) -> list[str]:
    files: list[str] = []
    for item in items:
        root = ROOT / item
        if root.is_file():
            files.append(item)
        elif root.is_dir():
            files.extend(
                path.relative_to(ROOT).as_posix()
                for path in root.rglob("*")
                if path.is_file() and not any(part == "__pycache__" for part in path.parts)
            )
    return sorted(set(files))


def _git_commit(ref: str) -> str:
    return subprocess.run(
        ["git", "rev-parse", f"{ref}^{{commit}}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def build_manifest() -> dict:
    """Build a deterministic manifest from the checked-out upstream tag."""
    commit = _git_commit(UPSTREAM_REF)
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(f"{UPSTREAM_REF} resolves to {commit}, expected {UPSTREAM_COMMIT}")
    exact_files = _tracked_files(EXACT_ROOTS)
    extension_files = _tracked_files(EXTENSION_POINTS)
    return {
        "schema_version": 1,
        "upstream": {"ref": UPSTREAM_REF, "commit": commit},
        "policy": {
            "exact_paths_are_byte_identical": True,
            "extension_points_require_symbol_level_review": True,
            "official_yolo26_configs_are_not_overwritten": True,
        },
        "exact_files": {
            path: {"sha256": _sha256(ROOT / path), "bytes": (ROOT / path).stat().st_size}
            for path in exact_files
        },
        "extension_points": {
            path: {"sha256": _sha256(ROOT / path), "bytes": (ROOT / path).stat().st_size}
            for path in extension_files
        },
        "counts": {
            "backends": len([path for path in exact_files if path.startswith("ultralytics/nn/backends/")]),
            "export_modules": len([path for path in exact_files if path.startswith("ultralytics/utils/export/")]),
            "yolo26_configs": len([path for path in exact_files if path.startswith("ultralytics/cfg/models/26/")]),
        },
    }


def verify_manifest(manifest: dict) -> list[str]:
    """Return integrity violations without requiring the upstream tag in shallow clones."""
    errors: list[str] = []
    upstream = manifest.get("upstream", {})
    if upstream.get("commit") != UPSTREAM_COMMIT:
        errors.append(f"manifest commit mismatch: {upstream.get('commit')!r}")
    for path, expected in manifest.get("exact_files", {}).items():
        file_path = ROOT / path
        if not file_path.is_file():
            errors.append(f"missing protected upstream file: {path}")
            continue
        actual = _sha256(file_path)
        expected_hash = expected.get("sha256")
        text_suffix = file_path.suffix.lower() in {".json", ".md", ".py", ".toml", ".txt", ".yaml", ".yml"}
        if actual != expected_hash and (not text_suffix or _sha256_lf(file_path) != expected_hash):
            errors.append(f"modified protected upstream file: {path}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write a regenerated manifest")
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    args = parser.parse_args()
    manifest_path = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    if args.write or not manifest_path.is_file():
        manifest = build_manifest()
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"wrote {manifest_path}: exact_files={len(manifest['exact_files'])}")
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = verify_manifest(manifest)
    if errors:
        raise SystemExit("upstream integrity failed:\n" + "\n".join(f"- {error}" for error in errors))
    print(f"upstream integrity passed: exact_files={len(manifest.get('exact_files', {}))}")


if __name__ == "__main__":
    main()
