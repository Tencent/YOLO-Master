#!/usr/bin/env python3
"""Freeze and verify YOLO-PEFT (arXiv 2608.07051) paper claims against in-repo evidence.

The published numbers are stored in ``reports/yolo-peft-2608.07051-anchor.json``.
This script does not retrain the paper protocol. It:

* ``--check-anchor`` (default): assert README / README_CN contain every claim number
  and that the default planner backend is still documented as opt-in ``vpeft``.
* ``--protocol``: print the in-repo entry points needed to rerun planner / audit
  jobs without inventing a new training recipe.

Usage:
    python scripts/reproduce_yolo_peft_paper.py
    python scripts/reproduce_yolo_peft_paper.py --protocol
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANCHOR_PATH = ROOT / "reports" / "yolo-peft-2608.07051-anchor.json"
README_PATHS = (ROOT / "README.md", ROOT / "README_CN.md")
DEFAULT_YAML = ROOT / "ultralytics" / "cfg" / "default.yaml"


def load_anchor() -> dict:
    """Load the frozen paper-claim JSON.

    Returns:
        (dict): Parsed anchor payload.

    Examples:
        >>> payload = load_anchor()
        >>> payload["arxiv"]
        '2608.07051'
    """
    return json.loads(ANCHOR_PATH.read_text(encoding="utf-8"))


def check_anchor() -> list[str]:
    """Return human-readable failures if README or default.yaml drift from the anchor."""
    anchor = load_anchor()
    failures = []
    claims = anchor.get("claims") or {}
    texts = {path: path.read_text(encoding="utf-8") for path in README_PATHS}
    for token in ("0.7138", "0.7307", "0.6428", "0.6662", "43.9", "1.72"):
        for path, text in texts.items():
            if token not in text:
                failures.append(f"{path.name} is missing paper claim token {token}")
    for path, text in texts.items():
        if "2608.07051" not in text:
            failures.append(f"{path.name} is missing arXiv id 2608.07051")
        if "yolo-peft-2608.07051-anchor.json" not in text:
            failures.append(f"{path.name} does not link reports/yolo-peft-2608.07051-anchor.json")
    default_yaml = DEFAULT_YAML.read_text(encoding="utf-8")
    backend = (anchor.get("in_repo_entrypoints") or {}).get("default_value", "legacy")
    paper_backend = (anchor.get("in_repo_entrypoints") or {}).get("paper_backend", "vpeft")
    if f'lora_planner_backend: "{backend}"' not in default_yaml:
        failures.append(f"default.yaml planner backend is not {backend}")
    if paper_backend not in default_yaml:
        failures.append(f"default.yaml does not mention paper backend {paper_backend}")
    if claims.get("rtdetr_l_evaluated_lora_configs") != 7:
        failures.append("anchor RT-DETR-L evaluated config count drifted from 7")
    return failures


def print_protocol() -> None:
    """Print in-repo reproduction entry points from the frozen anchor."""
    anchor = load_anchor()
    entry = anchor.get("in_repo_entrypoints") or {}
    print(f"YOLO-PEFT {anchor['arxiv']}: {anchor['title']}")
    print(f"Status: {anchor['status']}")
    print()
    print("Published claims (frozen, not retrained by this script):")
    for key, value in (anchor.get("claims") or {}).items():
        print(f"  {key}: {value}")
    print()
    print("In-repo entry points:")
    for key, value in entry.items():
        print(f"  {key}: {value}")
    print()
    print("Enable the paper planner on a training run with:")
    print("  yolo train model=yolo11s.pt data=coco.yaml lora_r=16 lora_use_rslora=True \\")
    print("    lora_planner_enabled=True lora_planner_backend=vpeft")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--protocol", action="store_true", help="Print in-repo reproduction entry points")
    parser.add_argument(
        "--check-anchor",
        action="store_true",
        help="Verify README and default.yaml against the frozen claims",
    )
    args = parser.parse_args()
    if args.protocol:
        print_protocol()
        return 0
    failures = check_anchor()
    if failures:
        print("YOLO-PEFT paper anchor check failed:", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print(f"YOLO-PEFT paper anchor OK ({ANCHOR_PATH.relative_to(ROOT)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
