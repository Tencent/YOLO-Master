#!/usr/bin/env python3
"""S2 pipeline smoke test: YOLOETrainerFromScratch x yoloe-26-master-n (MoE backbone) x grounding JSON.

Runs a few epochs on the coco128 pseudo-label grounding dataset (CPU, imgsz=64,
batch=4) to verify:
  1. the composite configuration (A2C2fMoE backbone + YOLOEDetect head) is built
     correctly by YOLOEModel;
  2. the GoldG-style JSON produced by convert.py is consumed by GroundingDataset;
  3. the full chain of text-embedding caching, RandomLoadText negatives, and loss
     backpropagation works end to end.

Two text-encoder modes:
  - default (real): mobileclip2:b; placing the weights file mobileclip2_b.ts under
    ``weights/`` lets attempt_download_asset pick it up; requires the ``clip``
    tokenization package (clip-anytorch + setuptools<81).
  - ``--stub-text``: a deterministic pseudo encoder (512-d) for offline CI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from ultralytics import YOLOE  # noqa: E402
from ultralytics.models.yolo.yoloe import YOLOETrainerFromScratch  # noqa: E402

SMOKE_DIR = ROOT / "runs/ov_smoke_train"


class _StubTextModel(torch.nn.Module):
    """Deterministic pseudo text encoder: same text -> same 512-d embedding; for offline smoke tests only."""

    def tokenize(self, texts, truncate: bool = True):
        rows = []
        for t in texts:
            g = torch.Generator().manual_seed(abs(hash(str(t))) % (2**31))
            rows.append(torch.randint(0, 1000, (8,), generator=g))
        return torch.stack(rows)

    def encode_text(self, tokens, dtype: torch.dtype = torch.float32):
        g = torch.Generator().manual_seed(0)
        proj = torch.randn(8, 512, generator=g)
        emb = tokens.float() @ proj
        return torch.nn.functional.normalize(emb, dim=-1).to(dtype)


def _install_stub_text_model() -> None:
    import types

    # text_model.py imports clip at module top level; when it is missing, it tries to install
    # from github online (which hangs on this machine without network access). In stub mode we
    # pre-inject a fake clip module to bypass that dependency.
    if "clip" not in sys.modules:
        fake_clip = types.ModuleType("clip")
        fake_inner = types.ModuleType("clip.clip")
        fake_inner.tokenize = lambda texts, truncate=True: None
        fake_clip.clip = fake_inner
        sys.modules["clip"] = fake_clip
        sys.modules["clip.clip"] = fake_inner

    import ultralytics.nn.text_model as text_model_mod

    text_model_mod.build_text_model = lambda variant, device=None: _StubTextModel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stub-text", action="store_true", help="use the deterministic pseudo text encoder (offline smoke)")
    parser.add_argument("--epochs", type=int, default=1, help="number of smoke-training epochs")
    parser.add_argument(
        "--json-file",
        default="runs/ov_distill_coco128/dataset_grounding4/grounding_train.json",
        help="grounding annotation JSON (default: the four-teacher voting product)",
    )
    args = parser.parse_args()

    SMOKE_DIR.mkdir(parents=True, exist_ok=True)
    if args.stub_text:
        _install_stub_text_model()
        print("[smoke] text encoder: STUB (deterministic 512-d)")
        # composite config -> smoke copy (stub aligned with the mobileclip:blt cache name)
        src_cfg = ROOT / "ultralytics/cfg/models/26/yoloe-26-master-n.yaml"
        cfg = yaml.safe_load(src_cfg.read_text(encoding="utf-8"))
        cfg["text_model"] = "mobileclip:blt"
        smoke_cfg = SMOKE_DIR / "yoloe-26-master-n-smoke.yaml"
        with smoke_cfg.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    else:
        smoke_cfg = ROOT / "ultralytics/cfg/models/26/yoloe-26-master-n.yaml"
        print("[smoke] text encoder: mobileclip2:b (real)")

    model = YOLOE(str(smoke_cfg))
    print(f"[smoke] model built: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params")

    # 2. grounding data (product of convert.py --format grounding) + coco128 closed-set val
    #    the official coco128.yaml resolves `path` relative to DATASETS_DIR; rewrite it to an
    #    absolute path inside the repository here
    val_cfg = yaml.safe_load((ROOT / "ultralytics/cfg/datasets/coco128.yaml").read_text(encoding="utf-8"))
    val_cfg["path"] = str(ROOT / "datasets/coco128")
    val_yaml = SMOKE_DIR / "coco128_local.yaml"
    with val_yaml.open("w", encoding="utf-8") as f:
        yaml.safe_dump(val_cfg, f, sort_keys=False, allow_unicode=True)

    data = {
        "train": {
            "yolo_data": [],
            "grounding_data": [
                {
                    "img_path": str(ROOT / "datasets/coco128/images/train2017"),
                    "json_file": str(ROOT / args.json_file),
                }
            ],
        },
        "val": {"yolo_data": [str(val_yaml)]},
    }

    # 3. smoke training (the real encoder enables val by default to exercise the YOLOEDetectValidator chain)
    model.train(
        data=data,
        trainer=YOLOETrainerFromScratch,
        epochs=args.epochs,
        imgsz=64,
        batch=4,
        device="cpu",
        workers=0,
        plots=False,
        save=True,
        val=not args.stub_text,
        project=str(SMOKE_DIR),
        name="train_real" if not args.stub_text else "train",
        exist_ok=True,
        verbose=False,
    )
    print(f"[smoke] {args.epochs}-epoch training finished OK")


if __name__ == "__main__":
    main()
