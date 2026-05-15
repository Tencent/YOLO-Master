#!/usr/bin/env python3
"""
Verify that the package-split LoRA module preserves the old public surface and basic runtime behavior.

This script is intended to run inside a real project environment where torch/cv2/pytest are available.
It does not force long training runs by default. Instead it:

1. validates the Python environment and repo import path
2. performs import smoke checks for key LoRA symbols
3. compares the new package public API against the previous single-file module from git history
4. prints the exact pytest and smoke-run commands recommended for full validation
"""

from __future__ import annotations

import ast
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_OLD_REF = "ce317c6^"


def run_git_show_old_lora(old_ref: str) -> str:
    result = subprocess.run(
        ["git", "show", f"{old_ref}:ultralytics/utils/lora.py"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def top_level_defs(text: str) -> list[str]:
    tree = ast.parse(text)
    names = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
    return names


def module_all(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    return [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
    return []


def check_environment() -> dict[str, bool]:
    return {
        "python": True,
        "pytest": shutil.which("pytest") is not None,
        "git": shutil.which("git") is not None,
    }


def import_smoke() -> dict[str, bool]:
    import ultralytics  # noqa: F401
    from ultralytics.utils.lora import (  # noqa: F401
        FewShotLoRAConv,
        LoRAConfig,
        LoRAConfigBuilder,
        LoraTrainingStrategy,
        ManualLoRAConv,
        apply_lora,
        get_lora_param_groups,
        get_lora_training_stats,
        load_lora_adapters,
        merge_lora_weights,
        resolve_adalora_total_step,
        save_lora_adapters,
        suggest_lora_config_for_dataset,
        _apply_rtdetr_lora_safety,
        _is_adapter_param,
        _merge_manual_lora_conv,
        _unfreeze_detection_head,
    )

    return {
        "ultralytics_import": True,
        "lora_symbol_imports": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-ref",
        default=DEFAULT_OLD_REF,
        help="Git ref containing the pre-package single-file ultralytics/utils/lora.py baseline.",
    )
    args = parser.parse_args()

    env = check_environment()
    old_lora = run_git_show_old_lora(args.old_ref)
    new_public = module_all(REPO_ROOT / "ultralytics/utils/lora/__init__.py")

    old_defs = set(top_level_defs(old_lora))
    new_defs = set()
    for file in (
        REPO_ROOT / "ultralytics/utils/lora/api.py",
        REPO_ROOT / "ultralytics/utils/lora/config.py",
        REPO_ROOT / "ultralytics/utils/lora/fallback.py",
        REPO_ROOT / "ultralytics/utils/lora/io.py",
        REPO_ROOT / "ultralytics/utils/lora/training.py",
    ):
        new_defs.update(top_level_defs(file.read_text()))

    report: dict[str, object] = {
        "repo_root": str(REPO_ROOT),
        "old_ref": args.old_ref,
        "environment": env,
        "old_only_top_level_defs": sorted(old_defs - new_defs),
        "new_only_top_level_defs": sorted(new_defs - old_defs),
        "old_public_api": module_all(Path("/tmp/does-not-exist")) if False else [],
        "new_public_api": new_public,
        "missing_public_from_old___all__": [],
        "extra_public_vs_old___all__": [],
    }

    try:
        old_tree = ast.parse(old_lora)
        old_all = []
        for node in old_tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        old_all = [elt.value for elt in node.value.elts if isinstance(elt, ast.Constant)]
        report["old_public_api"] = old_all
        report["missing_public_from_old___all__"] = sorted(set(old_all) - set(new_public))
        report["extra_public_vs_old___all__"] = sorted(set(new_public) - set(old_all))
    except Exception as exc:
        report["old_public_api_error"] = f"{type(exc).__name__}: {exc}"

    try:
        report["import_smoke"] = import_smoke()
    except Exception as exc:
        report["import_smoke_error"] = f"{type(exc).__name__}: {exc}"

    report["recommended_commands"] = [
        "pytest -q tests/test_engine.py -k \"lora or build_optimizer\"",
        "pytest -q tests/test_python.py -k \"lora or fallback\"",
        "python tests/lora_e2e_smoke.py",
        "python tests/lora_rankless_smoke.py",
    ]

    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
