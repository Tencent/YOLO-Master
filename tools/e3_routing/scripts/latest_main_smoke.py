"""Run a small five-family compatibility gate against the containing YOLO-Master checkout."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

TOOL_ROOT = Path(__file__).resolve().parents[1]


def _git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, check=False, text=True, encoding="utf-8"
    )
    return result.stdout.strip() or None if result.returncode == 0 else None


def _tensor_shapes(records: list[Any]) -> list[list[int]]:
    return sorted({tuple(int(item) for item in record.weights.shape) for record in records})


def _source_identity(root: Path) -> tuple[str | None, str]:
    commit = _git_value(root, "rev-parse", "HEAD")
    if commit:
        return commit, "git"
    match = re.search(r"-([0-9a-f]{40})$", root.name)
    return (match.group(1), "codeload-directory") if match else (None, "unresolved")


def run(repo_root: Path, image_size: int, seed: int) -> dict[str, Any]:
    """Execute P0 telemetry plus the P2 spatial/non-spatial contract on one deterministic input."""

    repo_root = repo_root.resolve()
    if not (repo_root / "ultralytics").is_dir():
        raise FileNotFoundError(f"YOLO-Master checkout not found: {repo_root}")
    sys.path.insert(0, str(TOOL_ROOT / "src"))
    sys.path.insert(0, str(repo_root))

    import torch
    from ultralytics import YOLO

    from e3_p0.collector import RoutingCollector
    from e3_p2.capture import SpatialRouterCollector, max_output_delta
    from e3_p2.runner import _audit_latent, _audit_moe, _audit_molora

    torch.manual_seed(seed)
    torch.set_num_threads(1)
    tensor = torch.linspace(-1.0, 1.0, steps=3 * image_size * image_size).reshape(1, 3, image_size, image_size)
    model_configs = {
        "moe": "ultralytics/cfg/models/26/yolo26-master-n.yaml",
        "mot": "ultralytics/cfg/models/26/yolo26-master-mot-n.yaml",
        "latent": "ultralytics/cfg/models/26/yolo26-master-latent-n.yaml",
        "moa": "ultralytics/cfg/models/26/yolo26-master-moa-n.yaml",
    }

    p0: dict[str, Any] = {}
    for family in ("moe", "mot", "latent"):
        wrapper = YOLO(repo_root / model_configs[family])
        model = wrapper.model.eval()
        with torch.inference_mode():
            reference = model(tensor)
        collector = RoutingCollector(family=family, run_id="latest-main-smoke")
        collector.set_context(phase="compatibility", batch_size=1, sample_indices=[0])
        names = collector.register(model)
        try:
            with torch.inference_mode():
                observed = model(tensor)
        finally:
            collector.remove()
        delta = max_output_delta(reference, observed)
        if not names or not collector.events:
            raise RuntimeError(f"{family} produced no routed telemetry")
        if delta > 1e-8:
            raise RuntimeError(f"{family} observer changed model output: {delta}")
        p0[family] = {"modules": len(names), "events": len(collector.events), "hook_output_delta": delta}
        del model, wrapper

    spatial: dict[str, Any] = {}
    for family, router_class in (("mot", "_MoTRouter"), ("moa", "_MoARouter")):
        wrapper = YOLO(repo_root / model_configs[family])
        model = wrapper.model.eval()
        with torch.inference_mode():
            reference = model(tensor)
        collector = SpatialRouterCollector(family=family, router_class=router_class)
        names = collector.register(model)
        try:
            with torch.inference_mode():
                observed = model(tensor)
        finally:
            collector.remove()
        delta = max_output_delta(reference, observed)
        if not collector.records:
            raise RuntimeError(f"{family} produced no spatial routing records")
        if delta > 1e-8:
            raise RuntimeError(f"{family} spatial hook changed model output: {delta}")
        spatial[family] = {
            "modules": len(names),
            "records": len(collector.records),
            "weight_shapes": _tensor_shapes(collector.records),
            "hook_output_delta": delta,
        }
        del model, wrapper

    unsupported = {
        "moe": _audit_moe(repo_root, tensor, torch, YOLO),
        "latent": _audit_latent(repo_root, tensor, torch, YOLO),
        "molora": _audit_molora(torch),
    }
    if any(item["status"] != "UNSUPPORTED" for item in unsupported.values()):
        raise RuntimeError("non-spatial family contract changed")

    source_commit, source_identity_mode = _source_identity(repo_root)
    return {
        "status": "PASS",
        "repo_root": str(repo_root),
        "source_commit": source_commit,
        "source_tree": _git_value(repo_root, "rev-parse", "HEAD^{tree}"),
        "source_identity_mode": source_identity_mode,
        "image_size": image_size,
        "seed": seed,
        "p0": p0,
        "p2_spatial": spatial,
        "p2_explicit_unsupported": {
            family: {
                "status": item["status"],
                "evidence_kind": item["evidence_kind"],
                "spatial_shape_count": item["spatial_shape_count"],
            }
            for family, item in unsupported.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.image_size < 32:
        parser.error("--image-size must be >= 32")
    result = run(args.repo_root, args.image_size, args.seed)
    payload = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
