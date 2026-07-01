#!/usr/bin/env python3
"""Run issue #49 reproduction training on SKU-110K."""

from __future__ import annotations

from reproduce_common import run_dataset


if __name__ == "__main__":
    raise SystemExit(run_dataset("sku110k"))
