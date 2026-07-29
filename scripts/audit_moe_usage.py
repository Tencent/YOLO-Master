#!/usr/bin/env python3
"""Audit public MoE class usage without assuming YAML is the only API surface."""

from __future__ import annotations

import argparse
import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
MOE_ROOT = ROOT / "ultralytics/nn/modules/moe"
DEFAULT_HISTORY = ROOT / "docs/governance/moe-variant-usage.json"


@dataclass(frozen=True)
class Usage:
    name: str
    source: str
    yaml: bool
    exported: bool
    tested: bool
    referenced: bool

    @property
    def disposition(self) -> str:
        if self.yaml or self.tested:
            return "retain"
        if self.exported or self.referenced:
            return "freeze"
        return "archive-candidate"


def _read_searchable_files(paths: list[Path]) -> dict[Path, str]:
    contents = {}
    for path in paths:
        try:
            contents[path] = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
    return contents


def _contains_symbol(contents: dict[Path, str], name: str) -> bool:
    pattern = re.compile(rf"\b{re.escape(name)}\b")
    return any(pattern.search(text) is not None for text in contents.values())


def _exported_names() -> set[str]:
    tree = ast.parse((MOE_ROOT / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets
        ):
            return {
                item.value for item in node.value.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)
            }
    return set()


def _assigned_string_collection(path: Path, name: str) -> set[str]:
    """Return string constants from a top-level set/dict assignment without importing the package."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            continue
        value = node.value
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name) and value.func.id == "frozenset":
            if not value.args:
                return set()
            value = value.args[0]
        if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
            return {item.value for item in value.elts if isinstance(item, ast.Constant) and isinstance(item.value, str)}
        if isinstance(value, ast.Dict):
            return {key.value for key in value.keys if isinstance(key, ast.Constant) and isinstance(key.value, str)}
    return set()


def registered_mixture_modules(root: Path = ROOT) -> set[str]:
    """Return YAML-visible module names from the additive mixture registry."""
    return _assigned_string_collection(root / "ultralytics/nn/mixture_registry.py", "MIXTURE_MODULES")


def yaml_mixture_references(module_names: Iterable[str], root: Path = ROOT) -> set[str]:
    """Return registered mixture module names referenced by model YAML files."""
    yaml_root = root / "ultralytics/cfg/models"
    contents = _read_searchable_files(list(yaml_root.rglob("*.yaml")) + list(yaml_root.rglob("*.yml")))
    return {name for name in module_names if _contains_symbol(contents, name)}


def load_usage_history(path: Path = DEFAULT_HISTORY) -> dict:
    """Load the versioned YAML-usage ledger, returning an empty ledger when absent."""
    if not path.exists():
        return {"schema_version": 1, "snapshots": []}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(payload.get("snapshots"), list):
        raise ValueError(f"unsupported MoE usage ledger schema in {path}")
    return payload


def record_usage_snapshot(path: Path, version: str, yaml_references: Iterable[str]) -> dict:
    """Insert or replace one explicit release snapshot in the usage ledger."""
    version = str(version).strip()
    if not version:
        raise ValueError("version must be non-empty")
    payload = load_usage_history(path)
    snapshot = {"version": version, "yaml_references": sorted(set(yaml_references))}
    snapshots = [item for item in payload["snapshots"] if item.get("version") != version]
    payload["snapshots"] = [*snapshots, snapshot]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def deprecation_candidates(
    experimental_names: Iterable[str], snapshots: Iterable[dict], *, window: int = 2
) -> set[str]:
    """Return variants absent from YAML in the latest distinct version snapshots."""
    if window < 2:
        raise ValueError(f"deprecation window must be >= 2, got {window}")
    latest: list[dict] = []
    seen_versions: set[str] = set()
    for snapshot in reversed(list(snapshots)):
        version = str(snapshot.get("version", ""))
        if not version or version in seen_versions:
            continue
        seen_versions.add(version)
        latest.append(snapshot)
        if len(latest) == window:
            break
    if len(latest) < window:
        return set()
    referenced = [set(snapshot.get("yaml_references", ())) for snapshot in latest]
    return {name for name in experimental_names if all(name not in names for names in referenced)}


def audit_usage(root: Path = ROOT) -> list[Usage]:
    definitions: dict[str, str] = {}
    for path in sorted(MOE_ROOT.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
                definitions[node.name] = f"{path.name}:{node.lineno}"

    exported = _exported_names()
    yaml_files = _read_searchable_files(
        list((root / "ultralytics/cfg").rglob("*.yaml")) + list((root / "ultralytics/cfg").rglob("*.yml"))
    )
    test_files = _read_searchable_files(list((root / "tests").rglob("*.py")))
    reference_files = _read_searchable_files(
        [
            path
            for folder in ("ultralytics", "scripts", "examples", "docs", "wiki")
            if (root / folder).exists()
            for path in (root / folder).rglob("*")
            if path.is_file() and path.suffix in {".py", ".md", ".ipynb"} and MOE_ROOT not in path.parents
        ]
    )
    return [
        Usage(
            name=name,
            source=source,
            yaml=_contains_symbol(yaml_files, name),
            exported=name in exported,
            tested=_contains_symbol(test_files, name),
            referenced=_contains_symbol(reference_files, name),
        )
        for name, source in sorted(definitions.items())
    ]


def render_markdown(rows: list[Usage]) -> str:
    counts = {
        status: sum(row.disposition == status for row in rows) for status in ("retain", "freeze", "archive-candidate")
    }
    lines = [
        "# MoE Class Usage Audit",
        "",
        "> Generated by `python scripts/audit_moe_usage.py`; deletion requires manual checkpoint/API review.",
        "",
        f"- Canonical public class definitions: {len(rows)}",
        f"- Retain: {counts['retain']}",
        f"- Freeze: {counts['freeze']}",
        f"- Archive candidates: {counts['archive-candidate']}",
        "",
        "| Class | Source | YAML | Exported | Tests | Other refs | Disposition |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    lines.extend(
        f"| `{row.name}` | `{row.source}` | {'yes' if row.yaml else 'no'} | "
        f"{'yes' if row.exported else 'no'} | {'yes' if row.tested else 'no'} | "
        f"{'yes' if row.referenced else 'no'} | **{row.disposition}** |"
        for row in rows
    )
    return "\n".join(lines) + "\n"


def render_variant_lifecycle(root: Path = ROOT, history_path: Path = DEFAULT_HISTORY, window: int = 2) -> str:
    """Render current YAML usage and version-qualified deprecation candidates."""
    moe_init = root / "ultralytics/nn/modules/moe/__init__.py"
    registered = registered_mixture_modules(root)
    experimental = _assigned_string_collection(moe_init, "EXPERIMENTAL_MOE_CLASSES")
    deprecated = _assigned_string_collection(moe_init, "DEPRECATED_MOE_CLASSES")
    governed = registered & (experimental | deprecated)
    current = yaml_mixture_references(registered, root)
    history = load_usage_history(history_path)
    eligible = deprecation_candidates(governed, history["snapshots"], window=window)
    lines = [
        "# YAML-Visible MoE Variant Lifecycle",
        "",
        f"- Registered mixture modules: {len(registered)}",
        f"- Referenced by current model YAML: {len(current)}",
        f"- Experimental/deprecated MoE variants under governance: {len(governed)}",
        f"- Recorded version snapshots: {len(history['snapshots'])}",
        f"- Eligible after {window} absent versions: {', '.join(sorted(eligible)) or 'none'}",
        f"- Declared deprecated: {', '.join(sorted(deprecated)) or 'none'}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY, help="Versioned YAML-usage ledger")
    parser.add_argument("--record-version", help="Record current YAML usage for an explicit release version")
    parser.add_argument("--window", type=int, default=2, help="Consecutive absent versions required for eligibility")
    parser.add_argument("--check-deprecated", action="store_true", help="Fail if eligible and declared tiers differ")
    args = parser.parse_args()

    registered = registered_mixture_modules()
    if args.record_version:
        record_usage_snapshot(args.history, args.record_version, yaml_mixture_references(registered))

    print(render_markdown(audit_usage()), end="\n")
    print(render_variant_lifecycle(history_path=args.history, window=args.window), end="")
    if args.check_deprecated:
        moe_init = ROOT / "ultralytics/nn/modules/moe/__init__.py"
        experimental = _assigned_string_collection(moe_init, "EXPERIMENTAL_MOE_CLASSES")
        deprecated = _assigned_string_collection(moe_init, "DEPRECATED_MOE_CLASSES")
        governed = registered & (experimental | deprecated)
        history = load_usage_history(args.history)
        eligible = deprecation_candidates(governed, history["snapshots"], window=args.window)
        if eligible != deprecated:
            missing = sorted(eligible - deprecated)
            ineligible = sorted(deprecated - eligible)
            raise SystemExit(f"MoE deprecated tier drift: missing={missing}, ineligible={ineligible}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
