#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

SKILL_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

from runtime.open_world.taxonomy import merge_verified_open_world_candidates


DISPATCHER = SKILL_ROOT / "scripts" / "run_yolo_master_skill.py"
DEFAULT_JSON_OUT = SKILL_ROOT / "logs" / "qwen-open-world-profiles-report.json"
DEFAULT_MD_OUT = SKILL_ROOT / "logs" / "qwen-open-world-profiles-report.md"
DEFAULT_PROMPT = "Discover visible objects outside COCO and preserve them for open-world reasoning."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real open-world profile comparisons on a small image batch.")
    parser.add_argument("--images", nargs="+", required=True, help="Image paths to evaluate.")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--profiles", nargs="+", default=["strict", "balanced", "exploratory"])
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--vlm-model", default="qwen-vl-plus")
    parser.add_argument("--openai-base-url", default="https://dashscope.aliyuncs.com/compatible-mode/v1")
    parser.add_argument("--openai-api-mode", default="chat.completions")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def run_case(*, image_path: Path, profile: str, args: argparse.Namespace, env: dict[str, str]) -> dict[str, Any]:
    request = {
        "skill": "yolo.multimodal.infer",
        "inputs": {
            "model": args.model,
            "source": str(image_path),
            "prompt": args.prompt,
        },
        "params": {
            "thinking_with_image": True,
            "structured_output": True,
            "prompt_template": "vlm_open_world_detection",
            "fusion_policy": "open_world_assist",
            "open_world_assist_profile": profile,
            "vlm_model": args.vlm_model,
            "openai_base_url": args.openai_base_url,
            "openai_api_mode": args.openai_api_mode,
        },
        "policy": {"dry_run": False},
    }
    cmd = [sys.executable, str(DISPATCHER), "--json", json.dumps(request, ensure_ascii=False)]
    start = time.perf_counter()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, env=env, text=True, capture_output=True)
    elapsed = time.perf_counter() - start
    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    payload = json.loads(lines[-1]) if lines else {"status": "failed", "summary": "no payload"}
    comparison = payload.get("multimodal", {}).get("open_world_comparison", {}) if isinstance(payload, dict) else {}
    return {
        "profile": profile,
        "image": image_path.name,
        "image_path": str(image_path),
        "elapsed_sec": round(elapsed, 3),
        "status": payload.get("status"),
        "effective_prompt_template": payload.get("multimodal", {}).get("effective_prompt_template"),
        "open_world_assist_profile": payload.get("multimodal", {}).get("open_world_assist_profile"),
        "open_world_filters": payload.get("multimodal", {}).get("open_world_filters"),
        "open_world_comparison": comparison,
        "stdout_tail": lines[-1] if lines else "",
        "stderr": proc.stderr.strip(),
        "returncode": proc.returncode,
    }


def aggregate_profile(items: list[dict[str, Any]]) -> dict[str, Any]:
    open_world_counts: dict[str, int] = {}
    reasoning_only_counts: dict[str, int] = {}
    filtered_reason_counts: dict[str, int] = {}
    for item in items:
        comparison = item.get("open_world_comparison", {}) or {}
        for ow in comparison.get("open_world_predictions", []) or []:
            label = ow.get("canonical_open_label") or ow.get("open_label") or ow.get("label")
            if ow.get("open_world_report_bucket") == "reasoning_only":
                if label:
                    reasoning_only_counts[str(label)] = reasoning_only_counts.get(str(label), 0) + 1
                for reason in ow.get("open_world_filter_reasons", []) or []:
                    filtered_reason_counts[str(reason)] = filtered_reason_counts.get(str(reason), 0) + 1
                continue
            if label:
                open_world_counts[str(label)] = open_world_counts.get(str(label), 0) + 1
    return {
        "images": len(items),
        "status_counts": dict((status, sum(1 for item in items if item.get("status") == status)) for status in sorted({item.get("status") for item in items})),
        "open_world_label_counts": open_world_counts,
        "reasoning_only_label_counts": reasoning_only_counts,
        "filtered_reason_counts": filtered_reason_counts,
    }


def build_report(rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("profile"))].append(row)
    profiles = {}
    for profile, items in grouped.items():
        profiles[profile] = {
            "filters": items[0].get("open_world_filters") if items else {},
            "aggregate": aggregate_profile(items),
            "items": items,
        }
    verified_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        comparison = row.get("open_world_comparison", {}) or {}
        verified_by_image[row.get("image") or "unknown"].append(
            {
                "profile": row.get("profile"),
                "open_world_predictions": comparison.get("open_world_predictions", []),
            }
        )
    verified_open_world_list = {
        image: merge_verified_open_world_candidates(entries)
        for image, entries in verified_by_image.items()
    }
    return {
        "name": "qwen-open-world-profiles",
        "prompt": args.prompt,
        "profiles": profiles,
        "verified_open_world_list": verified_open_world_list,
        "images": [str(Path(image)) for image in args.images],
        "vlm_model": args.vlm_model,
        "openai_base_url": args.openai_base_url,
    }


def render_item_labels(entries: list[dict[str, Any]]) -> str:
    parts = []
    for entry in entries:
        label = entry.get("canonical_open_label") or entry.get("open_label") or entry.get("label") or "-"
        bucket = entry.get("open_world_report_bucket") or "enhancement_stats"
        reasons = ",".join(entry.get("open_world_filter_reasons", []) or [])
        suffix = "stats" if bucket == "enhancement_stats" else f"reasoning-only:{reasons or 'filtered'}"
        parts.append(f"{label}[{suffix}]")
    return ", ".join(parts) if parts else "-"


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Qwen Open-World Profile Comparison", ""]
    for profile in ["strict", "balanced", "exploratory"]:
        payload = report.get("profiles", {}).get(profile)
        if not payload:
            continue
        aggregate = payload.get("aggregate", {})
        lines.extend(
            [
                f"## {profile}",
                "",
                f"- filters: {json.dumps(payload.get('filters', {}), ensure_ascii=False)}",
                f"- status counts: {json.dumps(aggregate.get('status_counts', {}), ensure_ascii=False)}",
                f"- open-world label counts: {json.dumps(aggregate.get('open_world_label_counts', {}), ensure_ascii=False)}",
                f"- reasoning-only label counts: {json.dumps(aggregate.get('reasoning_only_label_counts', {}), ensure_ascii=False)}",
                f"- filtered reason counts: {json.dumps(aggregate.get('filtered_reason_counts', {}), ensure_ascii=False)}",
                "",
            ]
        )
        for item in payload.get("items", []):
            comparison = item.get("open_world_comparison", {}) or {}
            labels = render_item_labels(comparison.get("open_world_predictions", []) or [])
            lines.append(f"- `{item.get('image')}`: {labels}")
        lines.append("")
    if report.get("verified_open_world_list"):
        lines.extend(["## Verified", ""])
        for image, items in report.get("verified_open_world_list", {}).items():
            labels = ", ".join((item.get("canonical_open_label") or item.get("open_label") or "-") for item in items) or "-"
            lines.append(f"- `{image}`: {labels}")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required")
    env = os.environ.copy()
    rows: list[dict[str, Any]] = []
    for profile in args.profiles:
        for raw_image in args.images:
            image_path = Path(raw_image).expanduser().resolve()
            rows.append(run_case(image_path=image_path, profile=profile, args=args, env=env))
    report = build_report(rows, args)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "ok", "json_out": str(args.json_out.resolve()), "md_out": str(args.md_out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
