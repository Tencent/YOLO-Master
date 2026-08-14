#!/usr/bin/env python3
"""Export full-probability MoT routing records for explicit experiment manifests."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from scripts.diagnose_mot_routing import EXPERT_NAMES  # noqa: E402
from scripts.issue54.schema import (  # noqa: E402
    EXPERIMENT_MANIFEST_SCHEMA_VERSION,
    ROUTING_RECORD_SCHEMA_VERSION,
    SchemaValidationError,
    canonical_payload_sha256,
    cli_error_message,
    ensure_outputs_available,
    load_json,
    sha256_file,
    validate_experiment_manifest,
    validate_routing_record,
    with_manifest_checksum,
    write_json,
    write_jsonl,
)
from ultralytics.data.augment import LetterBox  # noqa: E402
from ultralytics.nn.modules.mot import MoTBlock  # noqa: E402
from ultralytics.utils.checks import check_is_path_safe  # noqa: E402


def _safe_existing_child(root: Path, relative_path: str, *, field: str) -> Path:
    """Resolve an explicit relative path below a root and reject traversal."""
    candidate = root / Path(relative_path)
    if not check_is_path_safe(root, candidate):
        raise SchemaValidationError(f"{field} does not resolve to an existing file below its declared root")
    if not candidate.is_file():
        raise SchemaValidationError(f"{field} must resolve to a file")
    return candidate.resolve()


def _image_tensor(path: Path, imgsz: int, device: torch.device) -> torch.Tensor:
    """Load one image with the repository's detector-style letterbox transform."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise SchemaValidationError(f"failed to decode image: {path.name}")
    image = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)(image=image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(image).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor.to(device)


def _tensor_sha256(tensor: torch.Tensor) -> str:
    """Return a stable digest for a CPU-contiguous tensor and its shape."""
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _module_state_sha256(model: nn.Module) -> str:
    """Fingerprint an in-memory model state without writing a checkpoint."""
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _mot_layers(model: nn.Module) -> list[tuple[int, str, MoTBlock]]:
    """Return MoT blocks in stable named-module order."""
    return [
        (index, name, module)
        for index, (name, module) in enumerate(
            (item for item in model.named_modules() if isinstance(item[1], MoTBlock))
        )
    ]


def _dense_probabilities_from_router_output(
    router: nn.Module,
    output: Any,
    *,
    layer_name: str,
    expert_count: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate the upstream router contract and reconstruct dense probabilities from raw logits."""
    if (
        not isinstance(output, tuple)
        or len(output) < 3
        or not all(isinstance(value, torch.Tensor) for value in output[:3])
    ):
        raise SchemaValidationError(
            f"MoT router {layer_name!r} did not expose tensor (weights, indices, logits); "
            "the export contract must be reviewed against upstream"
        )
    router_weights, router_indices, raw_logits = (value.detach() for value in output[:3])
    logits = raw_logits.float()
    weights = router_weights.float()
    if logits.ndim != 4 or weights.shape != logits.shape or logits.shape[1] != expert_count:
        raise SchemaValidationError(f"MoT router {layer_name!r} returned incompatible weight/logit shapes")
    if not torch.isfinite(logits).all():
        raise SchemaValidationError(f"MoT router {layer_name!r} returned non-finite logits")
    if torch.all(logits >= 0.0) and torch.allclose(
        logits.sum(dim=1),
        torch.ones_like(logits[:, 0]),
        rtol=0.0,
        atol=1e-6,
    ):
        raise SchemaValidationError(
            f"MoT router {layer_name!r} third output appears to be normalized probabilities, not raw logits"
        )

    temperature = getattr(router, "temperature", None)
    if temperature is None:
        raise SchemaValidationError(f"MoT router {layer_name!r} does not expose temperature")
    temperature_value = torch.as_tensor(temperature, dtype=logits.dtype, device=logits.device)
    if temperature_value.numel() != 1 or not torch.isfinite(temperature_value).all() or temperature_value.item() <= 0:
        raise SchemaValidationError(f"MoT router {layer_name!r} temperature must be one finite positive scalar")
    probabilities = torch.softmax(logits / temperature_value, dim=1)
    if (
        not torch.isfinite(probabilities).all()
        or torch.any(probabilities < 0.0)
        or not torch.allclose(
            probabilities.sum(dim=1),
            torch.ones_like(probabilities[:, 0]),
            rtol=0.0,
            atol=1e-6,
        )
    ):
        raise SchemaValidationError(f"MoT router {layer_name!r} produced invalid dense probabilities")

    if (
        weights.ndim != 4
        or not torch.isfinite(weights).all()
        or torch.any(weights < 0.0)
        or not torch.allclose(weights.sum(dim=1), torch.ones_like(weights[:, 0]), rtol=0.0, atol=1e-5)
    ):
        raise SchemaValidationError(f"MoT router {layer_name!r} returned invalid sparse weights")
    if (
        router_indices.ndim != 4
        or router_indices.shape[0] != logits.shape[0]
        or router_indices.shape[1] != top_k
        or router_indices.shape[2:] != logits.shape[2:]
        or torch.is_floating_point(router_indices)
    ):
        raise SchemaValidationError(f"MoT router {layer_name!r} returned incompatible top-k indices")
    indices = router_indices.to(dtype=torch.long)
    if torch.any(indices < 0) or torch.any(indices >= expert_count):
        raise SchemaValidationError(f"MoT router {layer_name!r} returned out-of-range expert indices")
    if top_k > 1 and torch.any(indices.sort(dim=1).values.diff(dim=1) == 0):
        raise SchemaValidationError(f"MoT router {layer_name!r} returned duplicate top-k expert indices")

    if top_k < expert_count:
        selected = probabilities.gather(1, indices)
        expected_weights = torch.zeros_like(probabilities)
        expected_weights.scatter_(1, indices, selected / selected.sum(dim=1, keepdim=True).clamp_min(1e-12))
    else:
        expected_weights = probabilities
    if not torch.allclose(weights, expected_weights, rtol=1e-5, atol=1e-6):
        raise SchemaValidationError(
            f"MoT router {layer_name!r} weights/indices are inconsistent with the declared raw logits"
        )
    return probabilities.cpu(), probabilities.argmax(dim=1).cpu()


def capture_mot_routing(
    model: nn.Module,
    batch: torch.Tensor,
    image_entries: list[dict[str, Any]],
    manifest: dict[str, Any],
    *,
    inference_repeat: int,
    timestamp: str,
) -> list[dict[str, Any]]:
    """Capture dense MoT probabilities without changing persistent model behavior."""
    manifest = validate_experiment_manifest(manifest)
    if manifest["status"] not in {"passed", "diagnostic"}:
        raise SchemaValidationError("routing export requires a passed or diagnostic experiment manifest")
    if batch.ndim != 4 or batch.shape[0] != len(image_entries):
        raise SchemaValidationError("batch must be BCHW and match the number of image entries")
    if not image_entries:
        raise SchemaValidationError("routing export requires at least one image entry")
    image_ids = [entry.get("image_id") for entry in image_entries if isinstance(entry, dict)]
    if (
        len(image_ids) != len(image_entries)
        or any(not isinstance(image_id, str) or not image_id for image_id in image_ids)
        or len(set(image_ids)) != len(image_entries)
    ):
        raise SchemaValidationError("image entries must have unique image_id values")
    if inference_repeat < 0:
        raise SchemaValidationError("inference_repeat must be non-negative")

    layers = _mot_layers(model)
    if not layers:
        raise SchemaValidationError("model contains no MoTBlock layers")

    captures: dict[str, tuple[torch.Tensor, torch.Tensor, int, int]] = {}
    handles = []
    training_flags = {module: module.training for module in model.modules()}
    try:
        for layer_index, layer_name, block in layers:
            expert_names = (
                EXPERT_NAMES
                if block.NUM_EXPERTS == len(EXPERT_NAMES)
                else tuple(f"Expert{index}" for index in range(block.NUM_EXPERTS))
            )

            def capture_hook(
                router: nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                current_name: str = layer_name,
                current_index: int = layer_index,
                current_names: tuple[str, ...] = expert_names,
                current_top_k: int = block.top_k,
            ) -> None:
                if current_name in captures:
                    raise SchemaValidationError(f"MoT router {current_name!r} executed more than once in one forward")
                probabilities, assignments = _dense_probabilities_from_router_output(
                    router,
                    output,
                    layer_name=current_name,
                    expert_count=len(current_names),
                    top_k=current_top_k,
                )
                captures[current_name] = (probabilities, assignments, current_index, current_top_k)

            handles.append(block.router.register_forward_hook(capture_hook))

        model.eval()
        with torch.no_grad():
            _ = model(batch)
    finally:
        for handle in handles:
            handle.remove()
        for module, was_training in training_flags.items():
            module.training = was_training

    if set(captures) != {name for _, name, _ in layers}:
        missing = sorted({name for _, name, _ in layers} - captures.keys())
        raise SchemaValidationError(f"not all MoT layers produced routing output: {missing}")

    records = []
    for layer_index, layer_name, block in layers:
        probabilities, assignments, captured_index, top_k = captures[layer_name]
        if captured_index != layer_index or probabilities.shape[0] != len(image_entries):
            raise SchemaValidationError(f"captured routing shape/index mismatch for {layer_name!r}")
        expert_names = (
            list(EXPERT_NAMES)
            if block.NUM_EXPERTS == len(EXPERT_NAMES)
            else [f"Expert{index}" for index in range(block.NUM_EXPERTS)]
        )
        for batch_index, entry in enumerate(image_entries):
            mean_probabilities = probabilities[batch_index].flatten(1).mean(dim=1)
            mean_probabilities = mean_probabilities / mean_probabilities.sum().clamp_min(1e-12)
            selected_index = int(mean_probabilities.argmax().item())
            assignment = assignments[batch_index]
            record = {
                "schema_version": ROUTING_RECORD_SCHEMA_VERSION,
                "experiment_id": manifest["experiment_id"],
                "model_variant": manifest["model_variant"],
                "seed": manifest["seed"],
                "dataset": manifest["dataset"],
                "dataset_version": manifest["dataset_version"],
                "split": manifest["split"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "image_id": entry["image_id"],
                "image_path": entry["image_path"],
                "image_sha256": entry["image_sha256"],
                "scene_groups": entry.get("scene_groups", {}),
                "layer_name": layer_name,
                "layer_index": layer_index,
                "expert_names": expert_names,
                "expert_probabilities": [float(value) for value in mean_probabilities.tolist()],
                "selected_expert": expert_names[selected_index],
                "top_k": int(top_k),
                "token_top1_indices": [int(value) for value in assignment.reshape(-1).tolist()],
                "spatial_shape": list(assignment.shape),
                "inference_repeat": inference_repeat,
                "inference_batch_actual": int(batch.shape[0]),
                "timestamp": timestamp,
                "status": manifest["status"],
                "failure_reason": None,
            }
            records.append(validate_routing_record(record))
    return records


class _SyntheticMoTModel(nn.Module):
    """Small deterministic MoT model used only for Phase 1 infrastructure smoke."""

    def __init__(self) -> None:
        super().__init__()
        self.mot = MoTBlock(12, num_heads=3, top_k=2, window_size=2, n_points=2)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.mot(value)
        return output


def synthetic_evidence(
    *,
    seed: int = 42,
    repeats: int = 2,
    timestamp: str = "2026-07-30T00:00:00Z",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Create deterministic diagnostic manifest and routing rows without data or training."""
    if repeats < 1:
        raise SchemaValidationError("repeats must be >= 1")
    torch.manual_seed(seed)
    model = _SyntheticMoTModel().cpu()
    # Keep evaluation inputs identical across model seeds, matching the formal
    # cross-seed protocol where every checkpoint sees the same validation images.
    generator = torch.Generator().manual_seed(1054)
    batch = torch.rand(2, 12, 4, 4, generator=generator)
    image_hashes = [_tensor_sha256(batch[index]) for index in range(batch.shape[0])]
    checkpoint_hash = _module_state_sha256(model)
    config_hash = canonical_payload_sha256({"kind": "synthetic_mot_config", "experts": list(EXPERT_NAMES)})
    manifest = {
        "schema_version": EXPERIMENT_MANIFEST_SCHEMA_VERSION,
        "experiment_id": f"issue54-synthetic-seed-{seed}",
        "model_variant": "mot_synthetic",
        "seed": seed,
        "dataset": "synthetic",
        "dataset_version": "phase1-v1",
        "dataset_manifest_sha256": canonical_payload_sha256({"image_sha256": image_hashes, "shape": [12, 4, 4]}),
        "split": "diagnostic",
        "requested_epochs": 1,
        "epochs": 0,
        "requested_batch": 2,
        "batch": 2,
        "effective_batch": 2,
        "imgsz": 4,
        "optimizer": "not_applicable",
        "precision_mode": "fp32_cpu_synthetic",
        "checkpoint_path": "synthetic/random-initialized-state-no-checkpoint",
        "checkpoint_sha256": checkpoint_hash,
        "config_path": "synthetic/mot-block.json",
        "config_sha256": config_hash,
        "git_commit": "5c0db33af899b039f94bfdd6453857ff9795542c",
        "timestamp": timestamp,
        "status": "diagnostic",
        "failure_reason": None,
    }
    manifest = with_manifest_checksum(validate_experiment_manifest(manifest))
    image_entries = [
        {
            "image_id": f"synthetic-{index:03d}",
            "image_path": f"synthetic/image_{index:03d}.tensor",
            "image_sha256": image_hashes[index],
            "scene_groups": {"density": "dense" if index else "sparse"},
        }
        for index in range(batch.shape[0])
    ]
    records = []
    for repeat in range(repeats):
        records.extend(
            capture_mot_routing(
                model,
                batch,
                image_entries,
                manifest,
                inference_repeat=repeat,
                timestamp=timestamp,
            )
        )
    return manifest, records


def _load_real_inputs(
    args: argparse.Namespace,
) -> tuple[dict[str, Any], nn.Module, list[dict[str, Any]], list[torch.Tensor]]:
    """Load explicit real-mode inputs without inferring metadata from run directories."""
    manifest = validate_experiment_manifest(load_json(args.manifest))
    if manifest["status"] != "passed":
        raise SchemaValidationError("real export requires a status=passed manifest")
    artifact_root = args.artifact_root.resolve()
    checkpoint = _safe_existing_child(artifact_root, manifest["checkpoint_path"], field="checkpoint_path")
    if sha256_file(checkpoint) != manifest["checkpoint_sha256"]:
        raise SchemaValidationError("checkpoint_sha256 does not match checkpoint_path")
    config = _safe_existing_child(artifact_root, manifest["config_path"], field="config_path")
    if sha256_file(config) != manifest["config_sha256"]:
        raise SchemaValidationError("config_sha256 does not match config_path")
    if (
        manifest["dataset_manifest_sha256"] is not None
        and sha256_file(args.image_manifest) != manifest["dataset_manifest_sha256"]
    ):
        raise SchemaValidationError("dataset_manifest_sha256 does not match image_manifest")

    from ultralytics import YOLO

    device = torch.device(args.device)
    model = YOLO(str(checkpoint)).model.to(device)
    image_payload = load_json(args.image_manifest)
    entries = image_payload.get("images")
    if not isinstance(entries, list) or not entries:
        raise SchemaValidationError("image manifest must contain a non-empty images list")

    tensors = []
    normalized_entries = []
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"image_id", "image_path", "image_sha256", "scene_groups"}:
            raise SchemaValidationError(
                "each image entry must contain exactly image_id, image_path, image_sha256, and scene_groups"
            )
        path = _safe_existing_child(args.data_root.resolve(), entry["image_path"], field="image_path")
        if sha256_file(path) != entry["image_sha256"]:
            raise SchemaValidationError(f"image_sha256 mismatch for image_id={entry['image_id']!r}")
        tensors.append(_image_tensor(path, manifest["imgsz"], device))
        normalized_entries.append(dict(entry))
    return manifest, model, normalized_entries, tensors


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Routing JSONL output.")
    parser.add_argument("--synthetic", action="store_true", help="Run deterministic CPU-only synthetic diagnostic.")
    parser.add_argument("--manifest-output", type=Path, help="Synthetic manifest JSON output.")
    parser.add_argument("--manifest", type=Path, help="Explicit status=passed experiment manifest for real export.")
    parser.add_argument("--image-manifest", type=Path, help="Explicit sanitized image manifest for real export.")
    parser.add_argument(
        "--artifact-root",
        type=Path,
        help="Local root containing the manifest checkpoint_path and config_path.",
    )
    parser.add_argument("--data-root", type=Path, help="Local root containing image_manifest relative paths.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--inference-batch", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42, help="Synthetic diagnostic seed only.")
    parser.add_argument(
        "--timestamp", help="Explicit ISO-8601 timestamp; synthetic default is fixed for reproducibility."
    )
    parser.add_argument("--overwrite", action="store_true", help="Replace existing output files explicitly.")
    return parser.parse_args()


def main() -> int:
    """Export synthetic or explicit real-image MoT routing records."""
    args = parse_args()
    if args.synthetic:
        if args.manifest_output is None:
            raise SchemaValidationError("--manifest-output is required with --synthetic")
        ensure_outputs_available([args.manifest_output, args.output], overwrite=args.overwrite)
        timestamp = args.timestamp or "2026-07-30T00:00:00Z"
        manifest, records = synthetic_evidence(seed=args.seed, repeats=args.repeats, timestamp=timestamp)
        write_json(args.manifest_output, manifest, overwrite=args.overwrite)
        write_jsonl(args.output, records, overwrite=args.overwrite)
        print(f"[issue54-export] synthetic diagnostic wrote {len(records)} records to {ascii(args.output.name)}")
        return 0

    required = {
        "--manifest": args.manifest,
        "--image-manifest": args.image_manifest,
        "--artifact-root": args.artifact_root,
        "--data-root": args.data_root,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise SchemaValidationError(f"real export missing required arguments: {missing}")
    if args.inference_batch < 1 or args.repeats < 1:
        raise SchemaValidationError("--inference-batch and --repeats must be >= 1")
    ensure_outputs_available([args.output], overwrite=args.overwrite)

    manifest, model, entries, tensors = _load_real_inputs(args)
    timestamp = args.timestamp or manifest["timestamp"]
    records = []
    for repeat in range(args.repeats):
        for start in range(0, len(entries), args.inference_batch):
            batch_entries = entries[start : start + args.inference_batch]
            batch = torch.stack(tensors[start : start + args.inference_batch])
            records.extend(
                capture_mot_routing(
                    model,
                    batch,
                    batch_entries,
                    manifest,
                    inference_repeat=repeat,
                    timestamp=timestamp,
                )
            )
    write_jsonl(args.output, records, overwrite=args.overwrite)
    print(f"[issue54-export] wrote {len(records)} records to {ascii(args.output.name)}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except (SchemaValidationError, OSError) as error:
        print(f"[issue54-export] error: {cli_error_message(error)}", file=sys.stderr)
        exit_code = 2
    raise SystemExit(exit_code)
