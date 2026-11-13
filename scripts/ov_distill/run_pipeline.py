#!/usr/bin/env python3
"""OV-Distill: VLM-teacher distillation data pipeline (ADR-0007 M1).

Runs VLM grounding teachers such as Rex-Omni / WeVisOne-VL / GroundingPI
(GAM protocol) for batch open-vocabulary annotation over unlabeled images,
then applies multi-teacher cross-validation cleaning to produce region-text
pair training sets (YOLO grounding format) and abstention negative sets.

Subcommands:
  concepts   concept discovery (image directory -> tagging -> manifest JSONL, with negative injection)
  annotate   multi-teacher batch annotation (JSONL manifest input, resumable)
  clean      multi-teacher IoU voting + confidence bucketing + negative extraction
  clip-verify CLIP re-verification: image-text alignment filtering of medium-bucket boxes (GT-calibrated threshold)
  convert    cleaned results -> YOLO grounding training format
  smoke      synthetic images + MockTeacher end-to-end smoke test (no GPU required)

Usage (smoke):
  python scripts/ov_distill/run_pipeline.py smoke --out runs/ov_distill_smoke

Usage (real teachers):
  python scripts/ov_distill/run_pipeline.py annotate \
      --manifest data/ov_pool/manifest.jsonl \
      --teachers gam:http://127.0.0.1:8000/v1,rexomni:IDEA-Research/Rex-Omni \
      --out runs/ov_distill/raw.jsonl

Doc version: v1.0 (2026-11-08) | reference: docs/adr/0007-yolo26-open-vocabulary-evolution.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.annotate import run_annotate  # noqa: E402
from scripts.ov_distill.clean import run_clean  # noqa: E402
from scripts.ov_distill.concepts import run_concepts  # noqa: E402
from scripts.ov_distill.convert import run_convert  # noqa: E402
from scripts.ov_distill.smoke import run_smoke  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="OV-Distill: VLM-teacher distillation data pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ann = sub.add_parser("annotate", help="multi-teacher batch annotation")
    p_ann.add_argument("--manifest", required=True, help="input JSONL: one {image, queries[]} per line")
    p_ann.add_argument(
        "--teachers",
        required=True,
        help="comma-separated teacher specs: mock:seed / gam:<base_url> / rexomni:<model_path> / wevisone:<model_path> / groundingpi:<model_path> / qwenvl:<model_path>",
    )
    p_ann.add_argument("--out", required=True, help="output prediction JSONL (resumable; completed entries are skipped)")
    p_ann.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds")

    p_clean = sub.add_parser("clean", help="multi-teacher cross-validation cleaning")
    p_clean.add_argument("--raw", required=True, help="raw prediction JSONL produced by annotate")
    p_clean.add_argument("--out", required=True, help="cleaned high-confidence annotation JSONL")
    p_clean.add_argument("--neg-out", default="", help="abstention-negative JSONL path (default: negatives.jsonl next to --out)")
    p_clean.add_argument("--iou", type=float, default=0.5, help="IoU threshold for cross-teacher box clustering")
    p_clean.add_argument("--min-votes", type=int, default=2, help="minimum teacher votes required to keep a box")
    p_clean.add_argument(
        "--neg-min-absent",
        type=int,
        default=0,
        help="minimum abstaining teachers to record a negative (0 = auto n_teachers-1, floor of 2)",
    )

    p_conv = sub.add_parser("convert", help="convert to YOLO grounding training format")
    p_conv.add_argument("--cleaned", required=True, help="annotation JSONL produced by clean")
    p_conv.add_argument("--out", required=True, help="output dataset directory")
    p_conv.add_argument("--copy-images", action="store_true", help="copy images into the output directory (default: labels only)")
    p_conv.add_argument(
        "--format",
        default="yolo",
        choices=["yolo", "grounding"],
        help="yolo: txt+categories.json; grounding: GoldG single JSON (read directly by GroundingDataset)",
    )
    p_conv.add_argument("--buckets", default="", help="keep only the given buckets, comma-separated (e.g. high or high,medium)")
    p_conv.add_argument(
        "--image-root", default="", help="relative root for file_name in grounding format (should match the training img_path)"
    )

    p_con = sub.add_parser("concepts", help="concept discovery: image directory -> manifest JSONL")
    p_con.add_argument("--images", required=True, help="image directory (recursively scans jpg/jpeg/png/webp/bmp)")
    p_con.add_argument(
        "--tagger", required=True, help="tagger spec: mock:seed / file:<path> / vlm:<url>[:<model>] / ram:<ckpt>"
    )
    p_con.add_argument("--out", required=True, help="output manifest JSONL path")
    p_con.add_argument("--global-vocab", default="", help="global vocabulary file (one phrase per line), used for negative sampling")
    p_con.add_argument("--neg-per-image", type=int, default=0, help="negative queries injected per image (0 disables)")
    p_con.add_argument("--max-queries", type=int, default=32, help="cap on positive queries per image")
    p_con.add_argument("--seed", type=int, default=0, help="random seed for negative sampling")
    p_con.add_argument("--timeout", type=int, default=120, help="per-request timeout in seconds (vlm backend)")

    p_smoke = sub.add_parser("smoke", help="synthetic images + MockTeacher end-to-end smoke test")
    p_smoke.add_argument("--out", default="runs/ov_distill_smoke", help="smoke output directory")
    p_smoke.add_argument("--num-images", type=int, default=4, help="number of synthetic images")

    p_cv = sub.add_parser("clip-verify", help="CLIP re-verification: filter medium-bucket pseudo-labels by image-text alignment")
    p_cv.add_argument("--cleaned", required=True, help="annotation JSONL produced by clean")
    p_cv.add_argument("--out", required=True, help="output JSONL (with clip_score; low-score boxes dropped)")
    p_cv.add_argument("--model-dir", default="", help="W-CLIP-Base directory (default: ~/models/clip/W-CLIP-Base)")
    p_cv.add_argument("--buckets", default="medium", help="buckets to verify, comma-separated (default: medium only)")
    p_cv.add_argument("--threshold", type=float, default=0.0, help="cos x 100 threshold (0 = auto-calibrate against GT)")
    p_cv.add_argument("--calibrate-gt", default="", help="GT label directory for calibration (YOLO txt)")
    p_cv.add_argument("--calibrate-manifest", default="", help="image-subset manifest for calibration")
    p_cv.add_argument("--calibrate-keep", type=float, default=0.95, help="fraction of true boxes retained by calibration")
    p_cv.add_argument("--batch", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    """Pipeline entry point: dispatch subcommands."""
    args = parse_args()
    if args.command == "annotate":
        stats = run_annotate(
            manifest=Path(args.manifest),
            teacher_specs=[s for s in args.teachers.split(",") if s],
            out=Path(args.out),
            timeout=args.timeout,
        )
    elif args.command == "clean":
        stats = run_clean(
            raw=Path(args.raw),
            out=Path(args.out),
            neg_out=Path(args.neg_out) if args.neg_out else None,
            iou_thr=args.iou,
            min_votes=args.min_votes,
            neg_min_absent=args.neg_min_absent or None,
        )
    elif args.command == "convert":
        stats = run_convert(
            cleaned=Path(args.cleaned),
            out=Path(args.out),
            copy_images=args.copy_images,
            fmt=args.format,
            buckets={b for b in args.buckets.split(",") if b} or None,
            image_root=Path(args.image_root).resolve() if args.image_root else None,
        )
    elif args.command == "concepts":
        stats = run_concepts(
            images=Path(args.images),
            tagger_spec=args.tagger,
            out=Path(args.out),
            global_vocab=Path(args.global_vocab) if args.global_vocab else None,
            neg_per_image=args.neg_per_image,
            max_queries=args.max_queries,
            seed=args.seed,
            timeout=args.timeout,
        )
    elif args.command == "clip-verify":
        from scripts.ov_distill.clip_verify import DEFAULT_WCLIP, run_clip_verify

        stats = run_clip_verify(
            cleaned=Path(args.cleaned),
            out=Path(args.out),
            model_dir=Path(args.model_dir) if args.model_dir else DEFAULT_WCLIP,
            buckets=tuple(s.strip() for s in args.buckets.split(",") if s.strip()),
            threshold=args.threshold or None,
            calibrate_gt=Path(args.calibrate_gt) if args.calibrate_gt else None,
            calibrate_manifest=Path(args.calibrate_manifest) if args.calibrate_manifest else None,
            calibrate_keep=args.calibrate_keep,
            batch=args.batch,
        )
    elif args.command == "smoke":
        stats = run_smoke(out=Path(args.out), num_images=args.num_images)
    else:  # pragma: no cover - argparse required=True already guards this
        raise ValueError(f"unknown subcommand: {args.command}")
    print(json.dumps({"command": args.command, "stats": stats}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
