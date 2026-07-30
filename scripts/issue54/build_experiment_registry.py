#!/usr/bin/env python3
"""Build a validated Issue #54 registry from explicit experiment manifests."""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.issue54.schema import (  # noqa: E402
    FORMAL_STATUS,
    REGISTRY_SCHEMA_VERSION,
    SchemaValidationError,
    canonical_payload_sha256,
    cli_error_message,
    ensure_outputs_available,
    load_json,
    validate_experiment_manifest,
    with_manifest_checksum,
    write_json,
)


def inference_level(seed_count: int) -> str:
    """Return the allowed evidence label for an independent formal seed count."""
    if seed_count < 3:
        return "insufficient_for_cross_seed_inference"
    if seed_count < 5:
        return "exploratory_only"
    return "minimum_for_stronger_cross_seed_claims"


def build_registry(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    """Validate manifests and reject identities that could inflate independent seed counts."""
    normalized = []
    experiment_ids: dict[str, dict[str, Any]] = {}
    logical_runs: dict[tuple[str, str, str, str, int], str] = {}
    checkpoints: dict[str, tuple[str, int]] = {}

    for raw in manifests:
        manifest = validate_experiment_manifest(raw)
        experiment_id = manifest["experiment_id"]
        if experiment_id in experiment_ids:
            raise SchemaValidationError(f"duplicate experiment_id: {experiment_id}")

        logical_key = (
            manifest["model_variant"],
            manifest["dataset"],
            manifest["dataset_version"],
            manifest["split"],
            manifest["seed"],
        )
        if logical_key in logical_runs:
            raise SchemaValidationError(
                "the same model/dataset/version/split/seed is registered more than once: "
                f"{logical_runs[logical_key]!r} and {experiment_id!r}"
            )
        logical_runs[logical_key] = experiment_id

        checkpoint_hash = manifest["checkpoint_sha256"]
        if checkpoint_hash is not None:
            if checkpoint_hash in checkpoints:
                other_id, other_seed = checkpoints[checkpoint_hash]
                raise SchemaValidationError(
                    "checkpoint hash reuse cannot create another independent experiment: "
                    f"{other_id!r}/seed={other_seed} and {experiment_id!r}/seed={manifest['seed']}"
                )
            checkpoints[checkpoint_hash] = (experiment_id, manifest["seed"])

        if manifest.get("manifest_sha256") is None:
            manifest = with_manifest_checksum(manifest)
        experiment_ids[experiment_id] = manifest
        normalized.append(manifest)

    normalized.sort(key=lambda item: item["experiment_id"])
    status_counts = Counter(item["status"] for item in normalized)
    variants: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for manifest in normalized:
        variants[manifest["model_variant"]].append(manifest)

    variant_summary = {}
    for variant, items in sorted(variants.items()):
        formal = [item for item in items if item["status"] == FORMAL_STATUS]
        formal_seeds = sorted({item["seed"] for item in formal})
        variant_summary[variant] = {
            "registered_experiments": len(items),
            "formal_independent_runs": len(formal),
            "formal_seed_count": len(formal_seeds),
            "formal_seeds": formal_seeds,
            "inference_level": inference_level(len(formal_seeds)),
            "status_counts": dict(sorted(Counter(item["status"] for item in items).items())),
        }

    registry = {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "experiments": normalized,
        "status_counts": dict(sorted(status_counts.items())),
        "variant_summary": variant_summary,
        "counting_rule": (
            "Only status=passed experiments with distinct experiment_id, logical seed identity, and checkpoint_sha256 "
            "count as independent training runs. Images, layers, repeats, diagnostic exports, and not_executed plans do "
            "not increase the formal seed count."
        ),
    }
    registry["registry_sha256"] = canonical_payload_sha256(registry, exclude=("registry_sha256",))
    return registry


def validate_registry(payload: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a registry to verify its experiments and canonical checksum."""
    if not isinstance(payload, dict) or payload.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise SchemaValidationError("unsupported or missing registry schema_version")
    experiments = payload.get("experiments")
    if not isinstance(experiments, list):
        raise SchemaValidationError("registry experiments must be a list")
    rebuilt = build_registry(experiments)
    expected = payload.get("registry_sha256")
    if expected != rebuilt["registry_sha256"]:
        raise SchemaValidationError("registry_sha256 does not match the canonical registry")
    return rebuilt


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, action="append", required=True, help="Explicit manifest JSON; repeatable."
    )
    parser.add_argument("--output", type=Path, required=True, help="Validated registry JSON.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file explicitly.")
    return parser.parse_args()


def main() -> int:
    """Build and write the registry."""
    args = parse_args()
    registry = build_registry([load_json(path) for path in args.manifest])
    ensure_outputs_available([args.output], overwrite=args.overwrite)
    write_json(args.output, registry, overwrite=args.overwrite)
    formal_counts = {key: value["formal_seed_count"] for key, value in registry["variant_summary"].items()}
    print(
        f"[issue54-registry] wrote {ascii(args.output.name)} with {len(registry['experiments'])} experiments; "
        f"formal counts={formal_counts}"
    )
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except (SchemaValidationError, OSError) as error:
        print(f"[issue54-registry] error: {cli_error_message(error)}", file=sys.stderr)
        exit_code = 2
    raise SystemExit(exit_code)
