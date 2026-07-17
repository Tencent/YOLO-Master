"""Generate governance documentation from the packaged export capability matrix."""

from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "docs/governance/export-capability-matrix.md"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail when the checked-in document is stale or missing.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown output path.")
    return parser.parse_args()


def main() -> int:
    """Generate or verify the deterministic Markdown document."""
    from ultralytics.utils.export_capabilities import load_export_capability_matrix, render_export_capability_markdown

    args = parse_args()
    output = args.output
    rendered = render_export_capability_markdown(load_export_capability_matrix())

    if args.check:
        existing = output.read_text(encoding="utf-8") if output.exists() else ""
        if existing == rendered:
            print(f"Export capability documentation is current: {output}")
            return 0
        diff = difflib.unified_diff(
            existing.splitlines(keepends=True),
            rendered.splitlines(keepends=True),
            fromfile=str(output),
            tofile="generated",
        )
        print("".join(diff), end="")
        return 1

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(f"Generated export capability documentation: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
