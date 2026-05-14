#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parents[1]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from runtime.cli.regenerate_open_world_report import main


if __name__ == "__main__":
    raise SystemExit(main())
