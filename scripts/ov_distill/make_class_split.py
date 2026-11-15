"""Write the shared seen/unseen class split used by OV training and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.class_split import write_class_split


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output class split JSON")
    parser.add_argument("--unseen-ratio", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    artifact = write_class_split(Path(args.out), args.unseen_ratio, args.seed)
    print(
        json.dumps(
            {
                "path": str(Path(args.out).resolve()),
                "seen_classes": len(artifact["seen"]),
                "unseen_classes": len(artifact["unseen"]),
                "unseen": artifact["unseen"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
