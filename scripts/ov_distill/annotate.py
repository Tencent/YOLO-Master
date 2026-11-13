#!/usr/bin/env python3
"""OV-Distill · Multi-teacher batch annotation with resumable execution.

Input manifest JSONL, one line per image:
  {"image": "relative-or-absolute-path", "queries": ["person", "red car", ...]}

Output JSONL, one record per (image, teacher, query):
  {"image": ..., "teacher": ..., "query": ..., "boxes": [[x1,y1,x2,y2]...],
   "absent": false, "ok": true, "error": ""}

Resume semantics: (image, teacher, query) tuples with ok=true in the existing
output are skipped; failed records (ok=false) are retried on the next run and
their successful counterparts are appended.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple

from scripts.ov_distill.teachers import TeacherClient, build_teacher


def load_manifest(manifest: Path) -> List[dict]:
    """Load and validate the manifest, rejecting lines with missing fields."""
    items = []
    with manifest.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "image" not in item or "queries" not in item:
                raise ValueError(f"manifest line {ln} is missing the image/queries field: {line[:120]}")
            items.append(item)
    return items


def load_done_keys(out: Path) -> Set[Tuple[str, str, str]]:
    """Read the set of successfully completed (image, teacher, query) keys from an existing output file."""
    done: Set[Tuple[str, str, str]] = set()
    if not out.exists():
        return done
    with out.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("ok"):
                done.add((rec["image"], rec["teacher"], rec["query"]))
    return done


def run_annotate(manifest: Path, teacher_specs: List[str], out: Path, timeout: int = 120) -> Dict[str, int]:
    """Run batch annotation and return statistics.

    Args:
        manifest: Input JSONL, one line per {image, queries[]}.
        teacher_specs: Teacher spec list, see teachers.build_teacher.
        out: Output JSONL path; successful records are skipped and failed ones retried when it exists.
        timeout: Per-request timeout in seconds.
    """
    items = load_manifest(manifest)
    teachers: List[TeacherClient] = [build_teacher(spec) for spec in teacher_specs]
    out.parent.mkdir(parents=True, exist_ok=True)

    done = load_done_keys(out)

    stats = {"images": 0, "records": 0, "skipped": 0, "failed": 0}
    # Incremental writes: flush after every (image, teacher) pair so an interrupted
    # process loses at most one image of work. Failed records stay in the file for
    # diagnostics (clean only consumes ok=true) and are superseded by the successful
    # record appended after a retry.
    with out.open("a", encoding="utf-8") as fo:
        for item in items:
            image_rel = item["image"]
            image_path = Path(image_rel)
            if not image_path.is_absolute():
                # Resolve relative paths against the manifest directory first, then the CWD
                candidate = (manifest.parent / image_rel).resolve()
                image_path = candidate if candidate.exists() else image_path.resolve()
            queries = [str(q) for q in item["queries"]]
            stats["images"] += 1
            for teacher in teachers:
                pending = [q for q in queries if (image_rel, teacher.name, q) not in done]
                stats["skipped"] += len(queries) - len(pending)
                if not pending:
                    continue
                batch: List[dict] = []
                try:
                    results = teacher.detect(image_path, pending, timeout=timeout)
                    for res in results:
                        batch.append(
                            {
                                "image": image_rel,
                                "image_path": str(image_path),
                                "teacher": teacher.name,
                                "query": res.query,
                                "boxes": res.boxes,
                                "absent": res.absent,
                                "ok": True,
                                "error": "",
                            }
                        )
                        stats["records"] += 1
                except Exception as exc:  # a failing teacher must not abort the whole batch
                    for q in pending:
                        batch.append(
                            {
                                "image": image_rel,
                                "image_path": str(image_path),
                                "teacher": teacher.name,
                                "query": q,
                                "boxes": [],
                                "absent": False,
                                "ok": False,
                                "error": f"{type(exc).__name__}: {exc}",
                            }
                        )
                        stats["failed"] += 1
                for rec in batch:
                    fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
    return stats
