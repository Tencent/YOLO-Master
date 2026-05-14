#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


SKILL_ROOT = Path(__file__).resolve().parents[2]
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))

DEFAULT_INPUT = SKILL_ROOT / "logs" / "qwen-open-world-small-batch.json"
DEFAULT_JSON_OUT = SKILL_ROOT / "logs" / "qwen-open-world-small-batch-report.json"
DEFAULT_MD_OUT = SKILL_ROOT / "logs" / "qwen-open-world-small-batch-report.md"


def load_dispatcher_module() -> Any:
    return importlib.import_module("runtime.open_world.taxonomy")


def load_records(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a list in {path}, got {type(payload).__name__}")
    return [item for item in payload if isinstance(item, dict)]


def best_taxonomy_text(item: dict[str, Any]) -> str:
    taxonomy = item.get("taxonomy")
    best = taxonomy.get("best") if isinstance(taxonomy, dict) else None
    if not isinstance(best, dict):
        return "-"
    dataset = best.get("dataset") or "unknown"
    label = best.get("normalized_name") or best.get("name") or "-"
    score = best.get("score")
    if score is None:
        return f"{dataset}:{label}"
    return f"{dataset}:{label}@{float(score):.2f}"


def report_bucket_text(item: dict[str, Any]) -> str:
    bucket = item.get("open_world_report_bucket") or "enhancement_stats"
    reasons = item.get("open_world_filter_reasons") or []
    if bucket == "enhancement_stats":
        return "stats"
    reason_text = ",".join(str(reason) for reason in reasons if reason) or "filtered"
    return f"reasoning-only:{reason_text}"


def enrich_record(module: Any, record: dict[str, Any], multimodal_params: dict[str, Any]) -> dict[str, Any]:
    open_world_predictions = module.normalize_open_world_prediction_items(
        record.get("open_world_predictions_preview", []) if isinstance(record.get("open_world_predictions_preview"), list) else [],
        multimodal_params,
    )
    possible_misses = module.normalize_possible_miss_items(record.get("possible_misses"), multimodal_params)
    false_positives = module.normalize_possible_miss_items(record.get("false_positives"), multimodal_params)
    fusion_summary = record.get("fusion_summary") if isinstance(record.get("fusion_summary"), dict) else {}
    return {
        "profile": record.get("profile"),
        "image": record.get("image"),
        "status": record.get("status"),
        "verdict_parse_status": record.get("verdict_parse_status"),
        "effective_prompt_template": record.get("effective_prompt_template"),
        "answer": record.get("answer"),
        "vlm_labels": record.get("vlm_labels") if isinstance(record.get("vlm_labels"), list) else [],
        "open_world_predictions": open_world_predictions,
        "possible_misses": possible_misses,
        "false_positives": false_positives,
        "perturbation": {
            "suppressed": int(fusion_summary.get("suppressed", 0) or 0),
            "adjusted": int(fusion_summary.get("adjusted", 0) or 0),
            "relabelled": int(fusion_summary.get("relabelled", 0) or 0),
        },
    }


def build_report(module: Any, records: list[dict[str, Any]]) -> dict[str, Any]:
    multimodal_params = {
        "open_world_label_normalizer": True,
        "open_world_taxonomy_datasets": ["lvis", "v3det"],
        "open_world_taxonomy_topk": 5,
    }
    enriched = [enrich_record(module, record, multimodal_params) for record in records]
    by_profile: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in enriched:
        by_profile[str(item.get("profile") or "unknown")].append(item)

    profiles: dict[str, Any] = {}
    for profile, items in sorted(by_profile.items()):
        profiles[profile] = {
            "items": items,
            "aggregate": module.aggregate_open_world_comparison(items),
            "parsed": sum(1 for item in items if item.get("verdict_parse_status") == "parsed"),
        }

    return {
        "name": "qwen-open-world-small-batch",
        "source": str(DEFAULT_INPUT.resolve()),
        "profiles": profiles,
        "aggregate": module.aggregate_open_world_comparison(enriched),
        "items": enriched,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Qwen Open-World Small Batch Report", ""]
    profiles = report.get("profiles", {})
    for profile, payload in profiles.items():
        items = payload.get("items", [])
        aggregate = payload.get("aggregate", {})
        open_world_total = sum(aggregate.get("open_world_label_counts", {}).values())
        lines.extend(
            [
                f"## {profile}",
                "",
                f"- images: {len(items)}",
                f"- parsed: {payload.get('parsed', 0)}/{len(items)}",
                f"- open-world additions: {open_world_total}",
                f"- perturbation total: suppressed={aggregate.get('perturbation_totals', {}).get('suppressed', 0)}, adjusted={aggregate.get('perturbation_totals', {}).get('adjusted', 0)}, relabelled={aggregate.get('perturbation_totals', {}).get('relabelled', 0)}",
                f"- taxonomy datasets: {json.dumps(aggregate.get('taxonomy_dataset_counts', {}).get('open_world_predictions', {}), ensure_ascii=False)}",
                "",
            ]
        )
        for item in items:
            image = item.get("image") or "-"
            open_world = item.get("open_world_predictions", [])
            misses = item.get("possible_misses", [])
            ow_labels = ", ".join(
                f"{entry.get('canonical_open_label') or entry.get('open_label') or entry.get('label')}[{best_taxonomy_text(entry)}|{report_bucket_text(entry)}]"
                for entry in open_world
            ) or "-"
            miss_labels = ", ".join(
                f"{entry.get('canonical_label') or entry.get('label')}[{best_taxonomy_text(entry)}|{report_bucket_text(entry)}]"
                for entry in misses
            ) or "-"
            lines.append(f"- `{image}`: open-world=`{ow_labels}`; possible-misses=`{miss_labels}`")
        lines.append("")

    aggregate = report.get("aggregate", {})
    lines.extend(
        [
            "## Aggregate",
            "",
            f"- images: {aggregate.get('images', 0)}",
            f"- open-world label counts: {json.dumps(aggregate.get('open_world_label_counts', {}), ensure_ascii=False)}",
            f"- possible-miss label counts: {json.dumps(aggregate.get('possible_miss_label_counts', {}), ensure_ascii=False)}",
            f"- reasoning-only counts: {json.dumps(aggregate.get('reasoning_only_counts', {}), ensure_ascii=False)}",
            f"- filtered reason counts: {json.dumps(aggregate.get('filtered_reason_counts', {}), ensure_ascii=False)}",
            f"- taxonomy datasets: {json.dumps(aggregate.get('taxonomy_dataset_counts', {}), ensure_ascii=False)}",
            f"- taxonomy labels: {json.dumps(aggregate.get('taxonomy_label_counts', {}), ensure_ascii=False)}",
            f"- taxonomy unmatched: {json.dumps(aggregate.get('taxonomy_unmatched', {}), ensure_ascii=False)}",
            f"- perturbation totals: {json.dumps(aggregate.get('perturbation_totals', {}), ensure_ascii=False)}",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Regenerate taxonomy-enriched open-world small-batch reports.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--md-out", type=Path, default=DEFAULT_MD_OUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    module = load_dispatcher_module()
    records = load_records(args.input)
    report = build_report(module, records)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.md_out.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"status": "ok", "json_out": str(args.json_out.resolve()), "md_out": str(args.md_out.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
