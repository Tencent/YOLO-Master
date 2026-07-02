#!/usr/bin/env python3
"""Reproduce YOLO-Master-v0.1-N vs YOLO-Master-EsMoE-N on SKU-110K."""

from __future__ import annotations

from dense_verticals_common import run_dataset


if __name__ == "__main__":
    raise SystemExit(run_dataset("sku110k"))
