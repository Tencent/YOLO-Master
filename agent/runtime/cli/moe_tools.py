from __future__ import annotations

import importlib
import re
from pathlib import Path
from typing import Any

from runtime.cli.contract import ensure_manifest_dir, json_safe, plan_response, response, write_manifest
from runtime.cli.executor import capture_output, pushd
from runtime.cli.normalize import is_dry_run


_CACHE: dict[str, Any] = {}


def get_moe_helpers() -> dict[str, Any]:
    if "moe_helpers" not in _CACHE:
        analysis = importlib.import_module("ultralytics.nn.modules.moe.analysis")
        pruning = importlib.import_module("ultralytics.nn.modules.moe.pruning")
        _CACHE["moe_helpers"] = {
            "diagnose_model": analysis.diagnose_model,
            "prune_moe_model": pruning.prune_moe_model,
        }
    return _CACHE["moe_helpers"]


def parse_moe_diagnose_stdout(stdout: str) -> dict[str, Any]:
    expert_usage: dict[str, dict[str, Any]] = {}
    layer_summary: list[dict[str, Any]] = []
    current_layer: str | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        layer_match = re.search(r"Layer:\s*(.+)$", line)
        if layer_match:
            current_layer = layer_match.group(1).strip()
            expert_usage.setdefault(current_layer, {})
            continue
        row_match = re.match(r"^(\d+)\s+\|\s+([\d.]+)%\s+\|\s+([\d.]+)\s+\|\s+([\d,]+)\s+\|\s+(.+)$", line)
        if row_match and current_layer:
            expert_idx = int(row_match.group(1))
            expert_usage.setdefault(current_layer, {})[f"expert_{expert_idx}"] = {
                "usage_pct": float(row_match.group(2)),
                "avg_weight": float(row_match.group(3)),
                "hits": int(row_match.group(4).replace(",", "")),
                "status": row_match.group(5).strip(),
            }
    retained_experts: list[str] = []
    pruned_experts: list[str] = []
    collapse_warning = False
    for layer, experts in expert_usage.items():
        active = 0
        for expert_name, stats in experts.items():
            usage_pct = float(stats.get("usage_pct", 0.0) or 0.0)
            qualified_name = f"{layer}.{expert_name}"
            if usage_pct > 0:
                active += 1
            if "DEAD" in str(stats.get("status", "")) or usage_pct <= 0.0:
                pruned_experts.append(qualified_name)
            else:
                retained_experts.append(qualified_name)
            if "HOT" in str(stats.get("status", "")) or usage_pct >= 80.0:
                collapse_warning = True
        total = len(experts)
        layer_summary.append(
            {
                "layer": layer,
                "active_experts": active,
                "total_experts": total,
                "utilization": round(active / total, 6) if total else 0.0,
            }
        )
    return {
        "expert_usage": expert_usage,
        "retained_experts": retained_experts,
        "pruned_experts": pruned_experts,
        "layer_summary": layer_summary,
        "collapse_warning": collapse_warning,
    }


def parse_moe_prune_stdout(stdout: str, *, original_model: Any, pruned_model: Any, threshold: float) -> dict[str, Any]:
    layer_details: list[dict[str, Any]] = []
    current_layer: str | None = None
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        pruning_match = re.search(r"Pruning\s+(.+)$", line)
        if pruning_match:
            current_layer = pruning_match.group(1).strip()
            continue
        experts_match = re.search(r"Experts:\s*(\d+)\s*(?:\u2192|->)\s*(\d+).*keeping\s*\[([^\]]*)\]", line)
        if experts_match and current_layer:
            before = int(experts_match.group(1))
            after = int(experts_match.group(2))
            retained = [
                int(value.strip())
                for value in experts_match.group(3).split(",")
                if value.strip().lstrip("+-").isdigit()
            ]
            pruned = [f"expert_{idx}" for idx in range(before) if idx not in retained]
            layer_details.append(
                {
                    "layer": current_layer,
                    "before": before,
                    "after": after,
                    "retained": [f"expert_{idx}" for idx in retained],
                    "pruned": pruned,
                }
            )
    before_total = sum(item["before"] for item in layer_details)
    after_total = sum(item["after"] for item in layer_details)
    prune_ratio = round((before_total - after_total) / before_total, 6) if before_total else 0.0
    validation_after = {"status": "ok"} if "Validation check: OK" in stdout else {"status": "unknown"}
    if "Verification failed" in stdout or "Validation check: OK" not in stdout:
        validation_after = {"status": "not_confirmed"}
    return {
        "original_model": json_safe(original_model),
        "pruned_model": str(Path(str(pruned_model)).resolve()),
        "threshold": threshold,
        "prune_ratio": prune_ratio,
        "layer_details": layer_details,
        "validation_before": {"status": "not_run", "reason": "underlying pruner only validates the pruned checkpoint"},
        "validation_after": validation_after,
    }


def run_moe_diagnose(request: dict[str, Any]) -> dict[str, Any]:
    inputs = request["inputs"]
    params = request["params"]
    model_path = inputs.get("model")
    dataset = inputs.get("data") or params.get("data", "coco8.yaml")
    batch_size = int(params.get("batch_size", 1))
    verbose = bool(params.get("verbose", False))
    output_dir = Path(params.get("output_dir") or ensure_manifest_dir(request) / "moe_diagnose")
    if is_dry_run(request):
        return plan_response(
            request,
            "MoE diagnose dry run prepared",
            "module",
            "diagnose_model",
            params={"model_path": model_path, "dataset": dataset, "batch_size": batch_size, "verbose": verbose, "output_dir": str(output_dir)},
            extra={
                "moe_diagnose": {
                    "expert_usage": {},
                    "retained_experts": [],
                    "pruned_experts": [],
                    "layer_summary": [],
                    "collapse_warning": False,
                    "artifacts": ["expert_usage_heatmap.png", "expert_usage_bar.png"],
                }
            },
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    diagnose_model = get_moe_helpers()["diagnose_model"]
    with pushd(output_dir):
        _, stdout, stderr = capture_output(diagnose_model, model_path, dataset, batch_size, verbose)
    artifacts = []
    for name in ("expert_usage_heatmap.png", "expert_usage_bar.png"):
        file = output_dir / name
        if file.exists():
            artifacts.append({"kind": "image", "path": str(file.resolve())})
    moe_diagnose = parse_moe_diagnose_stdout(stdout)
    moe_diagnose["artifacts"] = [artifact["path"] for artifact in artifacts]
    payload = response(
        request["skill"],
        "ok",
        "moe diagnosis finished",
        moe_diagnose=moe_diagnose,
        artifacts=artifacts,
        logs={"stdout": stdout, "stderr": stderr},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload


def run_moe_prune(request: dict[str, Any]) -> dict[str, Any]:
    inputs = request["inputs"]
    params = request["params"]
    model_path = inputs.get("model")
    dataset = inputs.get("data") or params.get("data", "coco8.yaml")
    output_path = params.get("output_path") or str(ensure_manifest_dir(request) / "pruned_model.pt")
    threshold = float(params.get("threshold", 0.15))
    if is_dry_run(request):
        return plan_response(
            request,
            "MoE prune dry run prepared",
            "module",
            "prune_moe_model",
            params={"model_path": model_path, "output_path": output_path, "threshold": threshold, "dataset": dataset},
            extra={
                "moe_prune": {
                    "original_model": model_path,
                    "pruned_model": output_path,
                    "threshold": threshold,
                    "prune_ratio": None,
                    "layer_details": [],
                    "validation_before": {"status": "planned"},
                    "validation_after": {"status": "planned"},
                }
            },
        )

    prune_moe_model = get_moe_helpers()["prune_moe_model"]
    ok, stdout, stderr = capture_output(prune_moe_model, model_path, output_path, threshold, dataset)
    payload = response(
        request["skill"],
        "ok" if ok else "failed",
        "moe prune finished" if ok else "moe prune failed",
        moe_prune=parse_moe_prune_stdout(stdout, original_model=model_path, pruned_model=output_path, threshold=threshold),
        artifacts=[{"kind": "checkpoint", "path": str(Path(output_path).resolve())}] if Path(output_path).exists() else [],
        logs={"stdout": stdout, "stderr": stderr},
    )
    payload["manifest"] = str(write_manifest(request, payload))
    return payload
