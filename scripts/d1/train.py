"""Portable D1 model inspection, cached-feature training and checkpoint evaluation.

The research queues remain in the archived branch. This entry uses the shared
Trainer; it does not claim the research queue's exact per-rank resume semantics.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from copy import copy
from pathlib import Path
from types import SimpleNamespace

import torch

from scripts.d1.artifacts import write_json
from scripts.d1.ema import EMA_IMPLEMENTATIONS, configure_d1_ema, validate_ema_implementation
from scripts.d1.rgb import ExportMixin, ExportRGBValidator, ScratchTrainer, audit_model, build_model
from scripts.d1.runtime import TRAINING_PRECISION, VALIDATION_PRECISION, RunMixin
from ultralytics.models.yolo.detect.foundation_train import D1FoundationDetectionTrainer
from ultralytics.models.yolo.detect.foundation_val import D1FoundationDetectionValidator
from ultralytics.nn.foundation.cache import canonical_json_bytes, sha256_bytes, sha256_file
from ultralytics.nn.foundation.npy_cache import open_feature_cache
from ultralytics.nn.foundation_detection_model import D1_AUX_REPORT_NAMES, D1FoundationDetectionModel
from ultralytics.nn.mixture_loss import initialize_mixture_loss_ema_buffer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "ultralytics/cfg/experiments/d1/cached-detection.yaml"
VARIANTS = {
    "BASE": {"file": "yolo26-d1-dinov3-latent-n.yaml", "downstream_parameters": 3542567},
    "DW": {"file": "yolo26-d1-dinov3-latent-p5-dw-n.yaml", "downstream_parameters": 1195943},
    "BN64": {"file": "yolo26-d1-dinov3-latent-p5-bottleneck64-n.yaml", "downstream_parameters": 1404839},
}
MODEL_VARIANTS = (*VARIANTS, "SCRATCH")
P3_MODES = ("bilinear", "separable_bilinear2x")


def model_config(variant, p3_upsample_mode="bilinear", *, nc=80):
    """Select only registered architecture changes, independently of experiment queues."""
    if variant not in MODEL_VARIANTS or not isinstance(p3_upsample_mode, str) or p3_upsample_mode not in P3_MODES:
        raise ValueError("Unknown D1 architecture or P3 implementation")
    if type(nc) is not int or nc <= 0:
        raise ValueError("nc must be a positive integer")
    filename = "yolo26-d1-scratch-total-l.yaml" if variant == "SCRATCH" else VARIANTS[variant]["file"]
    value = YAML.load(ROOT / "ultralytics/cfg/models/26" / filename)
    if variant == "SCRATCH":
        value["nc"] = nc
        return value
    value["latent_mixture"].update(value_fusion_mode="weighted_sum", value_fusion_weights=[1.0, 1.0, 1.0])
    value["adapter"]["p3_upsample_mode"] = p3_upsample_mode
    value["detect"]["nc"] = nc
    return value


def construct_model(variant, p3_upsample_mode="bilinear", *, nc=80, seed=0):
    """Keep model initialization deterministic without consuming the caller's RNG."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        config = model_config(variant, p3_upsample_mode, nc=nc)
        model = build_model(config, nc=nc) if variant == "SCRATCH" else D1FoundationDetectionModel(config)
        initialize_mixture_loss_ema_buffer(model)
    return model


class CachedTrainer(RunMixin, D1FoundationDetectionTrainer):
    """Opt in to the previously verified EMA implementation after setup or resume."""

    def __init__(self, *args, ema_implementation="scalar-v1", **kwargs):
        self.ema_implementation = validate_ema_implementation(ema_implementation)
        super().__init__(*args, **kwargs)

    def _setup_train(self):
        super()._setup_train()
        self.ema = configure_d1_ema(self.ema, self.model, self.ema_implementation)


class RGBTrainer(RunMixin, ScratchTrainer):
    """Use the same measured run and recovery contract for the random RGB baseline."""


class ExportValidator(ExportMixin, D1FoundationDetectionValidator):
    """Use the same official export policy as the RGB control."""


def strict_checkpoint(path, *, allow_scratch=False):
    """Load only an explicitly supplied trusted D1 checkpoint, then check every state key."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema")
    if source is None:
        source = checkpoint.get("model")
    if isinstance(source, D1FoundationDetectionModel):
        model = D1FoundationDetectionModel(source.config_dict())
    elif allow_scratch and type(source) is DetectionModel:
        model = build_model(source.yaml, nc=source.model[-1].nc)
    else:
        raise TypeError("Checkpoint must contain a D1FoundationDetectionModel")
    if isinstance(model, D1FoundationDetectionModel) or "_mixture_loss_ema_buf" in source.state_dict():
        initialize_mixture_loss_ema_buffer(model)
    model.load_state_dict(source.float().state_dict(), strict=True)
    if type(model) is DetectionModel:
        audit_model(model)
    if any("teacher" in key.lower() for key in model.state_dict()):
        raise ValueError("Teacher state must not appear in a downstream checkpoint")
    if any(
        not bool(torch.isfinite(value).all())
        for value in model.state_dict().values()
        if isinstance(value, torch.Tensor)
    ):
        raise FloatingPointError("Checkpoint contains nonfinite state")
    return model, checkpoint.get("epoch")


def claim_output(output, identity, *, rank=0, timeout=30):
    """Claim a fresh run; peers must match the same torchrun invocation."""
    marker = output / "run.json"
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
        write_json(marker, identity)
        return
    deadline = time.monotonic() + timeout
    while not marker.exists() and time.monotonic() < deadline:
        time.sleep(0.1)
    if not marker.exists() or json.loads(marker.read_text()) != identity:
        raise RuntimeError("Run output was not claimed by this invocation")


def input_contract(args):
    """Validate paths and cache contracts before claiming output or constructing a trainer."""
    if args.command == "train" and not args.approved:
        raise ValueError("Training requires explicit --approved")
    if args.batch <= 0 or args.workers < 0 or args.epochs <= 0:
        raise ValueError("batch/epochs must be positive and workers nonnegative")
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if args.command == "evaluate" and world_size != 1:
        raise ValueError("Independent evaluation requires a single process")
    if "," in args.device and world_size == 1:
        raise ValueError("Use external torchrun for multi-GPU cached training")
    if world_size > 1 and len(args.device.split(",")) != world_size:
        raise ValueError("The explicit device count must match torchrun WORLD_SIZE")
    if args.batch % world_size:
        raise ValueError("Global batch must be divisible by WORLD_SIZE")
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        raise ValueError("Keep run artifacts outside the source repository")
    data_path = args.data.resolve()
    data = YAML.load(data_path)
    expected_nc = 80 if args.dataset == "coco" else 10
    if len(data.get("names", [])) != expected_nc:
        raise ValueError("Dataset names do not match the explicit dataset kind")
    scratch = args.variant == "SCRATCH"
    caches = {} if scratch else {"val": args.val_cache.resolve()}
    if args.command == "train":
        if not scratch and args.train_cache is None:
            raise ValueError("Training requires --train-cache")
        if not scratch:
            caches["train"] = args.train_cache.resolve()
    else:
        if args.checkpoint is None or not args.checkpoint.is_file():
            raise FileNotFoundError("Evaluation requires a local trusted --checkpoint")
        if not scratch:
            caches["train"] = caches["val"]
    readers = {key: open_feature_cache(path) for key, path in caches.items()}
    if readers and readers["train"].contract != readers["val"].contract:
        raise ValueError("Train and val feature contracts differ")
    if not all(reader.records for reader in readers.values()):
        raise ValueError("Feature cache must not be empty")
    value = YAML.load(args.recipe)
    if not isinstance(value, dict) or set(value) != {"train", "model", "runtime"}:
        raise ValueError("Recipe must contain train/model/runtime sections")
    if value["train"].get("resume") is not False or value["train"].get("pretrained") is not False:
        raise ValueError("Portable training starts fresh; use the archived policy for exact research resume")
    model_cfg = model_config(args.variant, args.p3_upsample, nc=expected_nc)
    if not scratch:
        model_cfg["latent_mixture"].update(value["model"])
    if args.command == "evaluate":
        checkpoint_model = strict_checkpoint(args.checkpoint, allow_scratch=scratch)[0]
        if scratch != (type(checkpoint_model) is DetectionModel):
            raise ValueError("Checkpoint type does not match the requested variant")
        model_cfg = checkpoint_model.yaml if scratch else checkpoint_model.config_dict()
    runtime = value["runtime"]
    lists = {}
    for split in ("train", "val"):
        paths = data.get(split, [])
        for path in paths if isinstance(paths, list) else [paths]:
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = Path(data.get("path", data_path.parent)) / candidate
            if candidate.suffix == ".txt":
                lists[split + ":" + candidate.name] = sha256_file(candidate)
    identity = {
        "schema_version": "d1-portable-run-v1",
        "code_commit": subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        "code_dirty": bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain"], text=True)),
        "recipe_sha256": sha256_file(args.recipe),
        "data_yaml_sha256": sha256_file(data_path),
        "data_lists_sha256": lists,
        "model_sha256": sha256_bytes(canonical_json_bytes(model_cfg)),
        "caches": {
            key: {
                "contract_sha256": reader.index["contract_sha256"],
                "sample_count": len(reader.records),
                "content_sha256": reader.index.get("content_sha256", reader.index.get("source_content_sha256")),
            }
            for key, reader in readers.items()
        },
        "dataset": args.dataset,
        "variant": args.variant,
        "epoch_loader_policy": "bounded-v1" if args.telemetry else "shared",
        "validation_precision": VALIDATION_PRECISION if args.telemetry or args.command == "evaluate" else "shared",
        "training_precision": TRAINING_PRECISION
        if args.telemetry and not args.fp32
        else "fp32"
        if args.fp32
        else "shared",
        "ddp_reducer_policy": "prebuild-restore-v1" if args.telemetry and world_size > 1 else "shared",
        "command": args.command,
        "seed": args.seed,
        "epochs": args.epochs,
        "global_batch": args.batch,
        "workers": args.workers,
        "device": args.device,
        "world_size": world_size,
        "ema_implementation": "scalar-v1" if scratch else args.ema,
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None,
        "rendezvous": os.environ.get("TORCHELASTIC_RUN_ID", "single-process"),
        "trusted_cache": args.trusted_cache,
    }
    overrides = {
        **value["train"],
        "model": model_cfg,
        "data": str(data_path),
        "device": args.device,
        "batch": args.batch,
        "nbs": args.batch,
        "epochs": args.epochs,
        "workers": args.workers,
        "seed": args.seed,
        "project": str(output.parent),
        "name": output.name,
        "exist_ok": True,
        "max_det": 300 if args.dataset == "coco" else 500,
    }
    if scratch:
        overrides["latent_aux_gain"] = 0.0
    if args.window is not None and not 1 <= args.window <= args.epochs:
        raise ValueError("window must be within the full training schedule")
    if args.fp32:
        overrides["amp"] = False
    return identity, overrides, runtime, caches


def stable_run_identity(identity):
    """Resume changes launcher identity, never the experiment contract."""
    return {key: value for key, value in identity.items() if key not in {"rendezvous", "command", "checkpoint_sha256"}}


def official_coco(annotations, predictions, image_ids):
    """Score the selected original-image detections with explicit maxDets=100."""
    from faster_coco_eval import COCO, COCOeval_faster

    gt = COCO(str(annotations))
    if not set(image_ids).issubset(set(gt.getImgIds())):
        raise ValueError("Evaluation image IDs are absent from the supplied annotations")
    if predictions:
        dt = gt.loadRes(predictions)
    else:
        dt = COCO()
        dt.dataset = {"images": gt.dataset["images"], "categories": gt.dataset["categories"], "annotations": []}
        dt.createIndex()
    evaluator = COCOeval_faster(gt, dt, iouType="bbox")
    evaluator.params.imgIds = sorted(image_ids)
    evaluator.params.maxDets = [1, 10, 100]
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    names = ("AP", "AP50", "AP75", "APs", "APm", "APl")
    return {
        "backend": "faster-coco-eval",
        "maxDets": [1, 10, 100],
        "image_count": len(image_ids),
        "annotations_sha256": sha256_file(annotations),
        "metrics": dict(zip(names, map(float, evaluator.stats[:6]))),
    }


def run(args):
    """Execute one explicitly requested run, never a background experiment queue."""
    identity, overrides, runtime, caches = input_contract(args)
    output = args.output.resolve()
    rank = max(0, int(os.environ.get("RANK", "0")))
    scratch = args.variant == "SCRATCH"
    if args.resume_snapshot:
        if args.command != "train" or not args.telemetry:
            raise ValueError("Resume requires measured training")
        previous = json.loads((output / "run.json").read_text())
        if stable_run_identity(previous) != stable_run_identity(identity):
            raise ValueError("Resume contract differs from the original run")
        if args.resume_snapshot.resolve().parent != output:
            raise ValueError("Resume snapshot must belong to this output directory")
    else:
        claim_output(output, identity, rank=rank)
    model_yaml = output / "model.yaml"
    if rank == 0:
        write_json(model_yaml, overrides["model"])
    else:
        deadline = time.monotonic() + 30
        while not model_yaml.exists() and time.monotonic() < deadline:
            time.sleep(0.1)
        if not model_yaml.exists():
            raise RuntimeError("Primary rank did not publish the model config")
    overrides["model"] = str(model_yaml)
    run_options = {
        "run_identity": stable_run_identity(identity) if args.telemetry else None,
        "run_output": output,
        "resume_snapshot": args.resume_snapshot,
        "stop_after_epoch": args.window,
    }
    if scratch:
        trainer = RGBTrainer(overrides=overrides, **run_options)
    else:
        trainer = CachedTrainer(
            overrides=overrides,
            feature_caches=caches,
            trusted_feature_cache=args.trusted_cache,
            max_open_feature_shards=runtime["max_open_feature_shards"],
            feature_prefetch_factor=runtime["feature_prefetch_factor"],
            amp_init_scale=runtime["amp_init_scale"],
            amp_growth_interval=runtime["amp_growth_interval"],
            ema_implementation=args.ema,
            **run_options,
        )
    if args.command == "train":
        trainer.train()
        if rank == 0:
            write_json(
                output / "completed.json",
                {
                    "status": "completed" if trainer.epoch + 1 >= args.epochs else "window_completed",
                    "epoch": trainer.epoch + 1,
                    "metrics": trainer.metrics,
                    "identity": identity,
                },
            )
        return
    model, epoch = strict_checkpoint(args.checkpoint, allow_scratch=scratch)
    if (model.model[-1].nc if scratch else model.detect.nc) != trainer.data["nc"]:
        raise ValueError("Checkpoint nc does not match the dataset")
    trainer.model = model.to(trainer.device)
    trainer.set_model_attributes()
    trainer.model.set_head_attr(max_det=overrides["max_det"], agnostic_nms=False)
    trainer.amp = False  # Match measured training validation; FP16 activations can overflow in eval mode.
    trainer.world_size, trainer.epoch, trainer.epochs = 1, 0, 1
    trainer.stopper = SimpleNamespace(possible_stop=True)
    trainer.ema = SimpleNamespace(ema=None)
    trainer.loss_names = (
        ("box_loss", "cls_loss", "dfl_loss")
        if scratch
        else ("box_loss", "cls_loss", "dfl_loss", *D1_AUX_REPORT_NAMES, "mixture_aux_loss")
    )
    trainer.loss_items = torch.zeros(len(trainer.loss_names), device=trainer.device)
    validator_type = ExportRGBValidator if scratch else ExportValidator
    cache_options = (
        {}
        if scratch
        else {
            "feature_cache": caches["val"],
            "trusted_cache": args.trusted_cache,
            "max_open_shards": runtime["max_open_feature_shards"],
            "prefetch_factor": runtime["feature_prefetch_factor"],
        }
    )
    validator = validator_type(
        save_dir=output,
        args=copy(trainer.args),
        _callbacks=trainer.callbacks,
        dataset_kind=args.dataset,
        **cache_options,
    )
    validator.data = trainer.data
    validator.dataloader = validator.get_dataloader(trainer.data["val"], args.batch)
    started = time.perf_counter()
    metrics = validator(trainer=trainer)
    expected = len(validator.dataloader.dataset)
    if validator.seen != expected:
        raise RuntimeError("Incomplete validation image coverage")
    predictions = validator.jdict
    write_json(output / "predictions.json", predictions)
    report = {
        "status": "completed",
        "identity": identity,
        "checkpoint_epoch_zero_based": epoch,
        "strict_reload": True,
        "validation_precision": VALIDATION_PRECISION,
        "images": expected,
        "seconds": time.perf_counter() - started,
        "internal_metrics": metrics,
        "degenerate_boxes_removed": validator.degenerate_boxes_removed,
        "official": None,
    }
    image_ids = [Path(path).stem for path in validator.dataloader.dataset.im_files]
    if args.dataset == "coco" and args.annotations:
        report["official"] = official_coco(args.annotations, predictions, [int(s) for s in image_ids])
    elif args.dataset == "visdrone":
        from scripts.d1.evaluate_visdrone import export_predictions

        report["visdrone_export"] = export_predictions(predictions, image_ids, output / "visdrone-txt")
    write_json(output / "evaluation.json", report)


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=("inspect", "train", "evaluate"))
    result.add_argument("--variant", choices=MODEL_VARIANTS, default="BN64")
    result.add_argument("--p3-upsample", choices=P3_MODES, default="separable_bilinear2x")
    result.add_argument("--ema", choices=EMA_IMPLEMENTATIONS, default="foreach-v1")
    result.add_argument("--recipe", type=Path, default=RECIPE)
    result.add_argument("--dataset", choices=("coco", "visdrone"), default="coco")
    result.add_argument("--data", type=Path)
    result.add_argument("--train-cache", type=Path)
    result.add_argument("--val-cache", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--checkpoint", type=Path)
    result.add_argument("--annotations", type=Path)
    result.add_argument("--device", default="0")
    result.add_argument("--batch", type=int, default=16)
    result.add_argument("--epochs", type=int, default=100)
    result.add_argument("--workers", type=int, default=4)
    result.add_argument("--seed", type=int, default=0)
    result.add_argument("--trusted-cache", action="store_true")
    result.add_argument("--fp32", action="store_true")
    result.add_argument("--approved", action="store_true")
    result.add_argument("--telemetry", action="store_true", help="Record epoch cost and exact recovery snapshots")
    result.add_argument(
        "--window", type=int, help="Stop after this epoch without shortening the learning-rate schedule"
    )
    result.add_argument("--resume-snapshot", type=Path, help="Resume this measured run's trusted local snapshot")
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "inspect":
        model = construct_model(args.variant, args.p3_upsample, nc=80 if args.dataset == "coco" else 10, seed=args.seed)
        print(
            json.dumps(
                {
                    "variant": args.variant,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "p3_upsample": args.p3_upsample,
                    "teacher_in_checkpoint": False,
                },
                indent=2,
            )
        )
        return
    required = ("data", "output") if args.variant == "SCRATCH" else ("data", "val_cache", "output")
    if any(getattr(args, name) is None for name in required):
        raise ValueError("--data, --val-cache and --output are required")
    run(args)


if __name__ == "__main__":
    main()
