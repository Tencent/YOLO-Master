#!/usr/bin/env python3
"""OV-Distill: driver for the 20K annotation campaign (one chunk per invocation).

Picks the next unfinished (slice, teacher) pair -- slices in order, teachers in
the fixed rotation [wevisone, rexomni, qwenvl, groundingpi] -- and runs one
annotate chunk as a subprocess (single teacher per process for memory safety,
resumable via the done-key index of the shared raw.jsonl).

Designed to be called repeatedly (manually or by a cron job); each call makes
~4 minutes of progress and exits. Prints a JSON status line at the end.

Usage:
  python scripts/ov_distill/drive_20k.py [--chunk-seconds 260]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.annotate import load_done_keys  # noqa: E402

POOL = ROOT / "runs/ov_s2_20k"
RAW = POOL / "raw.jsonl"
PYTHON = Path.home() / "miniconda3/envs/dev/bin/python"

TEACHERS = {
    "wevisone": "wevisone:/Users/linxu/models/models/models/isLinXu--WeVisOne-VL-2B/snapshots/master",
    "rexomni": "rexomni:/Users/linxu/models/models/IDEA-Research--Rex-Omni/snapshots/master",
    "qwenvl": "qwenvl:/Users/linxu/models/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master",
    "groundingpi": "groundingpi:/Users/linxu/models/models/models/GroundingPI--GroundingPI/snapshots/master",
}
TEACHER_ORDER = ["wevisone", "rexomni", "qwenvl", "groundingpi"]
# done-keys in raw.jsonl carry the full client-side teacher names, not the short keys
TEACHER_NAMES = {
    "wevisone": "wevisone-isLinXu--WeVisOne-VL-2B",
    "rexomni": "rexomni-master",
    "qwenvl": "qwenvl-master",
    "groundingpi": "groundingpi-GroundingPI--GroundingPI",
}


def pick_next(pool: Path = POOL) -> dict:
    """Return the first incomplete (slice, teacher) assignment with its progress counters."""
    raw = pool / "raw.jsonl"
    done = load_done_keys(raw) if raw.exists() else set()
    for s in range(1, 16):
        manifest = pool / f"manifest_s{s:02d}.jsonl"
        if not manifest.exists():
            continue
        entries = [json.loads(line) for line in manifest.open() if line.strip()]
        for teacher in TEACHER_ORDER:
            full_name = TEACHER_NAMES[teacher]
            total = sum(len(e["queries"]) for e in entries)
            finished = sum(
                1
                for e in entries
                for q in e["queries"]
                if (e["image"], full_name, q) in done
            )
            if finished < total:
                return {"slice": s, "teacher": teacher, "done": finished, "total": total, "manifest": str(manifest)}
    return {"slice": None, "teacher": None}


def main() -> None:
    """CLI entry point: run one annotation chunk for the next unfinished assignment."""
    parser = argparse.ArgumentParser(description="20K annotation campaign driver (one chunk per call)")
    parser.add_argument("--chunk-seconds", type=int, default=260, help="subprocess timeout for the annotate chunk")
    args = parser.parse_args()

    nxt = pick_next()
    if nxt["teacher"] is None:
        print(json.dumps({"status": "all_done"}, ensure_ascii=False))
        return
    print(json.dumps({"status": "running", **{k: v for k, v in nxt.items() if k != "manifest"}}, ensure_ascii=False), flush=True)

    cmd = [
        str(PYTHON),
        str(ROOT / "scripts/ov_distill/run_pipeline.py"),
        "annotate",
        "--manifest",
        nxt["manifest"],
        "--teachers",
        TEACHERS[nxt["teacher"]],
        "--out",
        str(RAW),
        "--timeout",
        "120",
    ]
    try:
        subprocess.run(cmd, cwd=ROOT, timeout=args.chunk_seconds, check=False)
    except subprocess.TimeoutExpired:
        pass  # chunk overran its window; done-keys make the next call resume seamlessly
    after = pick_next()
    print(json.dumps({"status": "chunk_finished", "next": after}, ensure_ascii=False))


if __name__ == "__main__":
    main()
