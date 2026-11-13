#!/usr/bin/env python3
"""OV-Distill: CLIP re-verification (ADR-0007 data pipeline Stage 3.5: multi-teacher voting + CLIP filtering).

Scores each cleaned box for image-text alignment: crop the box region, encode it
with the CLIP image tower, and take the cosine similarity against the query text
embedding (x 100, CLIP logits scale):
  - boxes below the threshold are dropped (pseudo-label noise; typically the
    teacher localized a similar object but assigned the wrong class);
  - surviving boxes are written out with a ``clip_score`` field for quality
    weighting or post-hoc analysis on the training side.

Only the ``medium`` bucket is verified by default (``high`` boxes are full-vote
plus high-IoU fusions and already trustworthy enough); widen the scope with
``--buckets``.

Threshold calibration (``--calibrate-gt`` + ``--calibrate-manifest``): on a GT-
annotated image subset, collect the score distribution of boxes that match GT
(IoU >= 0.5) and take the ``--calibrate-keep`` quantile as the threshold
(retaining 95% of true boxes by default), avoiding a hand-picked threshold.

The encoder backend defaults to W-CLIP-Base (local ONNX, onnxruntime CPU
inference, ~/models/clip/W-CLIP-Base), whose interface matches the openai CLIP
logits scale (cos x 100). Chunked resumable runs: scores are cached to
``<out>.cache.jsonl`` (key = image + query + box coordinates); reruns skip
already-scored records automatically.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.ov_distill.clean import box_iou  # noqa: E402
from scripts.ov_distill.make_coco128_manifest import COCO_NAMES  # noqa: E402
from scripts.ov_distill.teachers import image_size  # noqa: E402

DEFAULT_WCLIP = Path.home() / "models/clip/W-CLIP-Base"


class WClipEncoder:
    """W-CLIP-Base ONNX dual-tower encoder (image 224x224 / text max_length 72, 1024-d)."""

    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)

    def __init__(self, model_dir: Path):
        import onnxruntime as ort  # deferred import: pure-logic tests need no onnxruntime
        from transformers import AutoTokenizer

        model_dir = Path(model_dir)
        self.img_sess = ort.InferenceSession(
            str(model_dir / "wclip-base-image-encoder.onnx"), providers=["CPUExecutionProvider"]
        )
        self.txt_sess = ort.InferenceSession(
            str(model_dir / "wclip-base-text-encoder.onnx"), providers=["CPUExecutionProvider"]
        )
        self.tok = AutoTokenizer.from_pretrained(str(model_dir / "wclip-base-tokenizer"))

    def encode_images(self, bgr_crops: List[np.ndarray]) -> np.ndarray:
        """List of BGR uint8 crops -> L2-normalized image embeddings (N, 1024)."""
        import cv2

        batch = []
        for c in bgr_crops:
            img = cv2.resize(c[..., ::-1], (224, 224)).astype(np.float32) / 255.0
            batch.append(((img - self.MEAN) / self.STD).transpose(2, 0, 1))
        t = np.stack(batch).astype(np.float32)
        emb = self.img_sess.run(["image_embedding_1024"], {"image_tensor": t})[0]
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """List of texts -> L2-normalized text embeddings (N, 1024)."""
        enc = self.tok(
            texts, padding="max_length", truncation=True, max_length=72, return_tensors="np"
        )
        emb = self.txt_sess.run(
            ["text_embedding_1024"],
            {
                "text_token_id_tensor": enc["input_ids"].astype(np.int64),
                "text_mask_tensor": enc["attention_mask"].astype(np.int64),
            },
        )[0]
        return emb / np.linalg.norm(emb, axis=-1, keepdims=True)


def _cache_key(rec: dict) -> str:
    """Box-level cache key: image + query + coordinates (1 decimal place, tolerant to float jitter)."""
    box = ",".join(f"{v:.1f}" for v in rec["box"])
    return f"{rec['image']}|{rec['query']}|{box}"


def _load_cache(path: Path) -> Dict[str, float]:
    """Load the score cache (tolerates truncated trailing lines)."""
    cache: Dict[str, float] = {}
    if not path.exists():
        return cache
    for line in path.open("r", encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "key" in rec and "score" in rec:
            cache[rec["key"]] = float(rec["score"])
    return cache


def score_records(
    records: List[dict],
    encoder: WClipEncoder,
    cache_path: Path,
    batch: int = 64,
) -> Dict[str, float]:
    """Score records in batches; returns {cache_key: score} (cache hits included) and appends to the cache incrementally."""
    scores = _load_cache(cache_path)
    todo = [r for r in records if _cache_key(r) not in scores]
    # text embeddings are deduplicated per query
    text_emb: Dict[str, np.ndarray] = {}

    import cv2

    with cache_path.open("a", encoding="utf-8") as fc:
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            crops, valid = [], []
            for rec in chunk:
                img_path = Path(rec.get("image_path") or rec["image"])
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                x1, y1, x2, y2 = (int(round(v)) for v in rec["box"])
                h, w = img.shape[:2]
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                crops.append(img[y1:y2, x1:x2])
                valid.append(rec)
            if not valid:
                continue
            queries = sorted({r["query"] for r in valid})
            for q in queries:
                if q not in text_emb:
                    text_emb[q] = encoder.encode_texts([q])[0]
            img_emb = encoder.encode_images(crops)
            for rec, emb in zip(valid, img_emb):
                score = float(np.dot(emb, text_emb[rec["query"]]) * 100.0)
                key = _cache_key(rec)
                scores[key] = score
                fc.write(json.dumps({"key": key, "score": round(score, 4)}) + "\n")
            fc.flush()
            print(f"[clip-verify] scored {min(i + batch, len(todo))}/{len(todo)}", flush=True)
    return scores


def calibrate_threshold(
    records: List[dict],
    scores: Dict[str, float],
    gt_dir: Path,
    keep: float = 0.95,
    iou_thr: float = 0.5,
) -> Tuple[float, dict]:
    """Calibrate the threshold against GT: boxes matching GT count as true boxes; take the (1 - keep) quantile of true-box scores.

    Returns (threshold, calibration statistics). Records must already carry scores.
    """
    gt_cache: Dict[str, Dict[str, List[List[float]]]] = {}

    def gt_for(image: str) -> Dict[str, List[List[float]]]:
        if image not in gt_cache:
            lab = gt_dir / f"{Path(image).stem}.txt"
            w, h = image_size(Path(image))
            gt: Dict[str, List[List[float]]] = {}
            if lab.exists():
                for line in lab.read_text().splitlines():
                    p = line.split()
                    if len(p) < 5:
                        continue
                    xc, yc, bw, bh = (float(v) for v in p[1:5])
                    gt.setdefault(COCO_NAMES[int(p[0])], []).append(
                        [(xc - bw / 2) * w, (yc - bh / 2) * h, (xc + bw / 2) * w, (yc + bh / 2) * h]
                    )
            gt_cache[image] = gt
        return gt_cache[image]

    true_scores: List[float] = []
    false_scores: List[float] = []
    for rec in records:
        key = _cache_key(rec)
        if key not in scores:
            continue
        gts = gt_for(rec["image"]).get(rec["query"], [])
        hit = any(box_iou(rec["box"], g) >= iou_thr for g in gts)
        (true_scores if hit else false_scores).append(scores[key])
    if not true_scores:
        raise ValueError("no GT-matching boxes in the calibration set; cannot determine a threshold")
    thr = float(np.quantile(np.array(true_scores), 1.0 - keep))
    stats = {
        "n_true": len(true_scores),
        "n_false": len(false_scores),
        "true_score_mean": round(float(np.mean(true_scores)), 3),
        "false_score_mean": round(float(np.mean(false_scores)), 3) if false_scores else None,
        "threshold": round(thr, 3),
        "keep_quantile": keep,
    }
    return thr, stats


def run_clip_verify(
    cleaned: Path,
    out: Path,
    model_dir: Path = DEFAULT_WCLIP,
    buckets: Iterable[str] = ("medium",),
    threshold: Optional[float] = None,
    calibrate_gt: Optional[Path] = None,
    calibrate_manifest: Optional[Path] = None,
    calibrate_keep: float = 0.95,
    batch: int = 64,
) -> dict:
    """Main flow: score -> (optional) GT calibration -> threshold filtering -> write out.

    Args:
        threshold: explicit threshold (cos x 100 scale); when None, calibrate_* must be
            provided for automatic calibration.
        calibrate_manifest: manifest limiting the calibration image subset (typically the
            GT-annotated val1k); when None and calibrate_gt is given, all records are used
            for calibration.
    """
    records = [
        json.loads(line) for line in cleaned.open("r", encoding="utf-8") if line.strip()
    ]
    scope = set(buckets)
    in_scope = [r for r in records if r.get("bucket", "medium") in scope]

    cache_path = out.parent / f"{out.name}.cache.jsonl"
    encoder = WClipEncoder(model_dir)
    scores = score_records(in_scope, encoder, cache_path, batch=batch)

    if threshold is None:
        if calibrate_gt is None:
            raise ValueError("--calibrate-gt is required for automatic threshold calibration when --threshold is not given")
        cal_records = in_scope
        if calibrate_manifest is not None:
            keep_images = {
                json.loads(line)["image"]
                for line in calibrate_manifest.open("r", encoding="utf-8")
                if line.strip()
            }
            cal_records = [r for r in in_scope if r["image"] in keep_images]
        threshold, cal_stats = calibrate_threshold(cal_records, scores, Path(calibrate_gt), calibrate_keep)
        print(f"[clip-verify] calibrated threshold={threshold:.3f} stats={cal_stats}")
    else:
        cal_stats = None

    stats = {"pairs": 0, "kept": 0, "dropped_low_score": 0, "out_of_scope": 0}
    with out.open("w", encoding="utf-8") as fo:
        for rec in records:
            stats["pairs"] += 1
            if rec.get("bucket", "medium") not in scope:
                stats["out_of_scope"] += 1
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                continue
            score = scores.get(_cache_key(rec))
            if score is None or score < threshold:
                stats["dropped_low_score"] += 1
                continue
            rec["clip_score"] = round(score, 4)
            stats["kept"] += 1
            fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats["threshold"] = round(float(threshold), 3)
    if cal_stats:
        stats["calibration"] = cal_stats
    return stats


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="CLIP re-verification: pseudo-label image-text alignment filtering")
    parser.add_argument("--cleaned", required=True, help="JSONL produced by the clean subcommand")
    parser.add_argument("--out", required=True, help="output JSONL (with clip_score; low-score boxes dropped)")
    parser.add_argument("--model-dir", default=str(DEFAULT_WCLIP), help="W-CLIP-Base directory")
    parser.add_argument("--buckets", default="medium", help="buckets to verify (comma-separated; default: medium only)")
    parser.add_argument("--threshold", type=float, default=None, help="cos x 100 threshold; auto-calibrated when omitted")
    parser.add_argument("--calibrate-gt", default=None, help="GT label directory for calibration (YOLO txt)")
    parser.add_argument("--calibrate-manifest", default=None, help="image-subset manifest for calibration")
    parser.add_argument("--calibrate-keep", type=float, default=0.95, help="fraction of true boxes retained by calibration")
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    stats = run_clip_verify(
        cleaned=Path(args.cleaned),
        out=Path(args.out),
        model_dir=Path(args.model_dir),
        buckets=tuple(s.strip() for s in args.buckets.split(",") if s.strip()),
        threshold=args.threshold,
        calibrate_gt=Path(args.calibrate_gt) if args.calibrate_gt else None,
        calibrate_manifest=Path(args.calibrate_manifest) if args.calibrate_manifest else None,
        calibrate_keep=args.calibrate_keep,
        batch=args.batch,
    )
    print(json.dumps({"command": "clip-verify", "stats": stats}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
