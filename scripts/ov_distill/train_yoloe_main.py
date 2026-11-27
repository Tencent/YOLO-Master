#!/usr/bin/env python3
"""Train YOLOE-26-Master-N on the YOLOE main detection/grounding datasets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path

import torch
import yaml
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLOE
from ultralytics.data.converter import convert_coco
from ultralytics.models.yolo.yoloe.train import YOLOETrainerFromScratch
from ultralytics.nn.modules.moe.modules import C3k2MoE
from ultralytics.utils import LOGGER, DEFAULT_CFG
from ultralytics.utils.torch_utils import init_seeds, unwrap_model


class YOLOEMoEPreserveAdapterTrainer(YOLOETrainerFromScratch):
    """Train the MoE residual adapters while keeping the pretrained YOLOE core fixed."""

    def __init__(self, cfg=DEFAULT_CFG, overrides: dict | None = None, _callbacks: dict | None = None):
        overrides = dict(overrides or {})
        self.preserve_coeff = float(overrides.pop("moe_preserve_coeff", 50.0))
        super().__init__(cfg=cfg, overrides=overrides, _callbacks=_callbacks)

    def _freeze_model_parameters(self) -> None:
        trainable = []
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = ".moe_adapter." in name or name.endswith(".moe_scale")
            if parameter.requires_grad:
                trainable.append(name)
        for module in self.model.modules():
            if isinstance(module, C3k2MoE):
                module.preserve_coeff = self.preserve_coeff
        if not trainable:
            raise RuntimeError("No MoE adapter parameters found; expected '.moe_adapter.' and '.moe_scale'.")
        self.freeze_layer_names = []
        trainable_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_count = sum(p.numel() for p in self.model.parameters())
        print(
            f"[yoloe-main] trainable MoE adapter parameters: {trainable_count:,}/{total_count:,} "
            f"({100 * trainable_count / total_count:.2f}%) across {len(trainable)} tensors; "
            f"preserve coefficient={self.preserve_coeff:g}"
        )

    def _model_train(self) -> None:
        self.model.train()
        for module in self.model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                parameters = tuple(module.parameters(recurse=False))
                if parameters and all(not parameter.requires_grad for parameter in parameters):
                    module.eval()


def prepare_objects365(root: Path, source_annotation: Path, source_images: Path) -> Path:
    """Convert processed Objects365 COCO polygons and write its local dataset YAML."""
    objects_root = root / "objects365"
    label_dir = objects_root / "labels" / "objects365_train_segm"
    image_link = objects_root / "images" / "objects365_train_segm"
    annotation_dir = root / "annotations" / "objects365_source"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    annotation_link = annotation_dir / source_annotation.name
    if not annotation_link.exists():
        annotation_link.symlink_to(source_annotation.resolve())

    if not label_dir.exists():
        convert_coco(
            labels_dir=str(annotation_dir),
            save_dir=str(objects_root),
            use_segments=True,
            use_keypoints=False,
            cls91to80=False,
        )
    if not label_dir.is_dir():
        raise RuntimeError(f"Objects365 conversion did not create expected labels: {label_dir}")

    image_link.parent.mkdir(parents=True, exist_ok=True)
    if not image_link.exists():
        image_link.symlink_to(source_images.resolve(), target_is_directory=True)
    elif not image_link.is_symlink() or image_link.resolve() != source_images.resolve():
        raise RuntimeError(f"Unexpected existing Objects365 image path: {image_link}")

    # Keep the workspace's /images/ path in every entry. Resolving the directory
    # symlink here would make Ultralytics look for labels beside the read-only source.
    train_list = objects_root / "objects365_train_segm.txt"
    if not train_list.exists():
        with train_list.open("w", encoding="utf-8") as output:
            for label_path in sorted(label_dir.glob("*.txt")):
                image_path = image_link / f"{label_path.stem}.jpg"
                if image_path.is_file():
                    output.write(f"{image_path}\n")

    official_yaml = ROOT / "ultralytics" / "cfg" / "datasets" / "Objects365.yaml"
    official = yaml.safe_load(official_yaml.read_text(encoding="utf-8"))
    local_yaml = root / "objects365.yaml"
    local = {
        "path": str(objects_root.resolve()),
        "train": str(train_list.resolve()),
        # The LVIS-aware evaluator is run after training; this split supplies class metadata only.
        "val": str(train_list.resolve()),
        "names": official["names"],
    }
    payload = yaml.safe_dump(local, sort_keys=False, allow_unicode=True)
    tmp_yaml = local_yaml.with_name("%s.tmp.%d" % (local_yaml.name, os.getpid()))
    tmp_yaml.write_text(payload, encoding="utf-8")
    os.replace(tmp_yaml, local_yaml)
    return local_yaml


def publish_train_text_context(trainer) -> None:
    """Publish class text embeddings to TSMoE routers before the first train forward.

    ``YOLOEModel`` does not own ``txt_feats`` by default (only ``YOLOWorld`` does),
    so ``BaseModel.forward`` never calls ``set_route_context()`` during training and
    the semantic router silently degrades to vision-only. Opt-in via
    ``YOLO_TSMOE_TRAIN_TEXT_CONTEXT=1`` keeps every legacy checkpoint reproducible.
    """
    from ultralytics.nn.modules.moe.text_semantic import set_route_context

    m = unwrap_model(trainer.model)
    names = trainer.data.get("names") or {}
    if not names:
        raise RuntimeError("train text context requires class names in trainer.data")
    class_names = [names[i] for i in sorted(int(k) for k in names)]
    was_training = m.training
    m.eval()
    with torch.no_grad():
        embeddings = m.get_text_pe(class_names, batch=80, cache_clip_model=True)
    if was_training:
        m.train()
    m.txt_feats = embeddings.detach()
    set_route_context(m.txt_feats)
    print(
        f"[yoloe-main] published train text context: "
        f"{tuple(m.txt_feats.shape)} for {len(class_names)} classes",
        flush=True,
    )


def run(args: argparse.Namespace) -> Path:
    root = args.data_root.resolve()
    annotations = root / "annotations"
    sources = {
        "objects365": annotations / "objects365_train_segm.json",
        "gqa": annotations / "final_mixed_train_no_coco_segm.json",
        "flickr": annotations / "final_flickr_separateGT_train_segm.json",
    }
    required = [*sources.values(), args.init_from, args.model_cfg, args.gqa_images, args.flickr_images]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    if not args.objects365_images.is_dir():
        raise FileNotFoundError(args.objects365_images)

    if int(os.getenv("RANK", -1)) in {-1, 0}:
        objects_yaml = prepare_objects365(root, sources["objects365"], args.objects365_images)
    else:
        objects_yaml = root / "objects365.yaml"
        for _ in range(1200):
            if objects_yaml.exists():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError(f"Rank {os.getenv("RANK")} timed out waiting for {objects_yaml}")
    model = YOLOE(str(args.model_cfg))
    if args.scratch:
        # Random init: the only way to compare architectures without spending (and
        # then measuring) a pretrained open-vocabulary ability that both arms would
        # otherwise degrade at different rates.
        LOGGER.info("scratch init: skipping checkpoint load")
    else:
        model.load(str(args.init_from))
    run_dir = root / "runs" / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = run_dir / "initial_master_model.pt"
    baseline_saved = False

    def save_initialized_baseline(trainer) -> None:
        nonlocal baseline_saved
        if baseline_saved or int(os.getenv("RANK", -1)) not in {-1, 0}:
            return
        initialized = deepcopy(unwrap_model(trainer.model)).float().cpu()
        torch.save(
            {"model": initialized.half(), "ema": None, "epoch": -1, "train_args": vars(trainer.args)},
            baseline_path,
        )
        baseline_saved = True
        print(f"[yoloe-main] saved pre-training LVIS baseline: {baseline_path}")

    model.add_callback("on_pretrain_routine_end", save_initialized_baseline)
    if os.getenv("YOLO_TSMOE_TRAIN_TEXT_CONTEXT", "") == "1":
        model.add_callback("on_pretrain_routine_end", publish_train_text_context)
    train_config = {
        "yolo_data": [str(objects_yaml)],
        "grounding_data": [
            {
                "img_path": str(args.gqa_images.resolve()),
                "json_file": str(sources["gqa"].resolve()),
                "max_text_samples": 80,
                "confirmed_negatives_only": True,
            },
            {
                "img_path": str(args.flickr_images.resolve()),
                "json_file": str(sources["flickr"].resolve()),
                "max_text_samples": 80,
                "confirmed_negatives_only": True,
            },
        ],
    }
    data = {"train": train_config, "val": {"yolo_data": [str(objects_yaml)]}}
    trainer = YOLOEMoEPreserveAdapterTrainer if args.train_scope == "preserve50" else YOLOETrainerFromScratch
    train_overrides = {"moe_preserve_coeff": args.preserve_coeff} if args.train_scope == "preserve50" else {}
    init_seeds(args.seed, deterministic=True)
    model.train(
        data=data,
        trainer=trainer,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        nbs=128,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        device=args.device,
        amp=not args.no_amp,
        seed=args.seed,
        deterministic=True,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=0.025,
        warmup_bias_lr=0.0,
        momentum=0.9,
        close_mosaic=2,
        fraction=args.fraction,
        warmup_epochs=args.warmup_epochs,
        lrf=args.lrf,
        plots=False,
        val=False,
        save=True,
        save_period=1,
        project=str(root / "runs"),
        name=args.name,
        exist_ok=True,
        verbose=False,
        **train_overrides,
    )
    if not baseline_saved and int(os.getenv("RANK", -1)) in {-1, 0}:
        raise RuntimeError("Training ended before the initialized baseline checkpoint was saved")

    run_dir = Path(model.trainer.save_dir)
    metadata = {
        "recipe": "YOLOE main detection and grounding datasets, adapted to YOLOE-26-Master-N",
        "datasets": {
            "objects365": str(sources["objects365"].resolve()),
            "gqa": str(sources["gqa"].resolve()),
            "flickr30k": str(sources["flickr"].resolve()),
        },
        "model_cfg": str(args.model_cfg.resolve()),
        "initial_weights": str(args.init_from.resolve()),
        "train_scope": args.train_scope,
        "preserve_coeff": args.preserve_coeff if args.train_scope == "preserve50" else 0,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "physical_batch": args.batch,
        "effective_batch": 128,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "fraction": args.fraction,
        "weight_decay": 0.025,
        "scheduler": "default linear decay (cos_lr=False)",
        "validation_during_training": False,
        "lvis_evaluation": "run separately with the LVIS federated evaluator",
        "note": "YOLOE-26-Master-N is an architecture adaptation; the YOLOE paper reports YOLOE-v8/v11 models.",
    }
    (run_dir / "yoloe_main_run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    if args.train_scope == "preserve50" and int(os.getenv("RANK", -1)) in {-1, 0}:
        checkpoint_path = Path(model.trainer.last)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        trained = checkpoint.get("ema") or checkpoint.get("model")
        if trained is None:
            raise RuntimeError(f"No model weights found in {checkpoint_path}")
        adapter_state = {
            key: value.detach().cpu()
            for key, value in trained.state_dict().items()
            if ".moe_adapter." in key or key.endswith(".moe_scale")
        }
        adapter_path = run_dir / "moe_adapter_only.pt"
        torch.save(
            {
                "state_dict": adapter_state,
                "base_checkpoint": str(baseline_path),
                "model_cfg": str(args.model_cfg.resolve()),
                "preserve_coeff": 50,
            },
            adapter_path,
        )
        print(f"[yoloe-main] saved adapter-only weights: {adapter_path}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "yoloe_main")
    parser.add_argument("--objects365-images", type=Path, default=ROOT.parent.parent / "public_data" / "objects365_v1" / "objects365" / "train")
    parser.add_argument("--gqa-images", type=Path, default=ROOT / "data" / "yoloe_main" / "gqa" / "images")
    parser.add_argument("--flickr-images", type=Path, default=ROOT / "data" / "yoloe_main" / "flickr" / "images")
    parser.add_argument("--model-cfg", type=Path, default=ROOT / "ultralytics" / "cfg" / "models" / "26" / "yoloe-26-master-n-seg.yaml")
    parser.add_argument("--init-from", type=Path, default=ROOT / "weights" / "yoloe-26n-seg.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the training set to use. Short probes set this so one epoch costs ~1.5k steps.",
    )
    parser.add_argument(
        "--warmup-epochs",
        dest="warmup_epochs",
        type=float,
        default=3.0,
        help="LR warmup length. Must be 0 for single-epoch probes or the whole run stays in warmup.",
    )
    parser.add_argument(
        "--lrf",
        type=float,
        default=0.01,
        help="Final LR as a fraction of lr0. 1.0 keeps the LR constant, which is what short probes need.",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8, help="Physical batch per GPU; effective batch is 128 via accumulation.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr0", type=float, default=0.002, help="Peak optimizer LR (applied literally, not per-sample).")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="AdamW",
        choices=("AdamW", "SGD", "MuSGD", "Adam", "NAdam", "RAdam", "auto"),
        help="Optimizer passed to Ultralytics trainer. Default AdamW keeps existing runs reproducible; "
             "use MuSGD to match the official yoloe-26n-seg.pt recipe (lr0 0.00125, momentum 0.9).",
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable AMP and train in FP32. Diagnoses fp16 overflow in the MoE routing path.",
    )
    parser.add_argument(
        "--preserve-coeff",
        dest="preserve_coeff",
        type=float,
        default=50.0,
        help="L2 weight on the MoE residual magnitude; keeps the adapter from overwriting the frozen core.",
    )
    parser.add_argument(
        "--cache",
        default="false",
        choices=("false", "ram", "disk"),
        help="Ultralytics image cache. \"ram\" keeps decoded images in memory (very large for Objects365).",
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="yoloe26_master_n_preserve50_main_30ep")
    parser.add_argument("--train-scope", choices=("preserve50", "full"), default="preserve50")
    parser.add_argument("--scratch", action="store_true", help="skip checkpoint init, train from random weights")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
