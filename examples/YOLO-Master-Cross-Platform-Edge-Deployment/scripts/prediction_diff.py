#!/usr/bin/env python3
"""Diagnose per-image differences between two YOLO prediction directories.

The tool is intentionally independent of torch, OpenCV, and runtime SDKs. It
consumes the pixel-coordinate text format emitted by the C++ runner
(``class confidence x1 y1 x2 y2``), matches detections by class and IoU, and
keeps enough detail to investigate an accuracy-gate failure without rerunning
either backend.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class Prediction:
    class_id: int
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def box(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class Match:
    reference_index: int
    candidate_index: int
    iou: float
    confidence_abs_delta: float
    box_max_abs_delta: float


def _finite(value: str, field: str, path: Path, line_no: int) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"{path}:{line_no}: invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{path}:{line_no}: {field} must be finite")
    return result


def read_predictions(path: Path) -> list[Prediction]:
    """Read one YOLO pixel-xyxy prediction file with strict validation."""
    if not path.is_file():
        raise FileNotFoundError(f"prediction file not found: {path}")
    result: list[Prediction] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 6:
            raise ValueError(
                f"{path}:{line_no}: expected exactly 6 columns (class conf x1 y1 x2 y2), got {len(fields)}"
            )
        try:
            class_id = int(fields[0])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: class id must be an integer") from exc
        if class_id < 0:
            raise ValueError(f"{path}:{line_no}: class id must be non-negative")
        confidence = _finite(fields[1], "confidence", path, line_no)
        x1 = _finite(fields[2], "x1", path, line_no)
        y1 = _finite(fields[3], "y1", path, line_no)
        x2 = _finite(fields[4], "x2", path, line_no)
        y2 = _finite(fields[5], "y2", path, line_no)
        if confidence < 0.0 or confidence > 1.0:
            raise ValueError(f"{path}:{line_no}: confidence must be in [0,1]")
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"{path}:{line_no}: box must have positive width and height")
        result.append(Prediction(class_id, confidence, x1, y1, x2, y2))
    return result


def image_stem(path: Path) -> str:
    return path.stem.casefold()


def prediction_files(directory: Path) -> dict[str, Path]:
    if not directory.is_dir():
        raise NotADirectoryError(f"prediction directory not found: {directory}")
    files = sorted(
        (path for path in directory.rglob("*") if path.is_file() and path.suffix.casefold() == ".txt"),
        key=lambda path: path.as_posix().casefold(),
    )
    result: dict[str, Path] = {}
    for path in files:
        stem = image_stem(path)
        if not stem:
            raise ValueError(f"prediction file has an empty stem: {path}")
        if stem in result:
            raise ValueError(f"prediction stems are not unique: {stem}")
        result[stem] = path
    return result


def image_files(source: Path, root: Path | None = None) -> dict[str, Path]:
    """Resolve an image directory or list below one canonical root.

    The list syntax deliberately matches the mAP evaluators and evidence
    manifest: UTF-8 BOMs, comments and quoted paths are accepted, while a
    caller-supplied root prevents a list from reaching outside the dataset
    tree.  Keeping this parser identical is important because this tool is
    normally used to explain a failed cross-backend comparison.
    """
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    source = source.expanduser()
    explicit_root = root.expanduser().resolve() if root is not None else None
    if explicit_root is not None and not explicit_root.is_dir():
        raise NotADirectoryError(f"image normalization root not found: {explicit_root}")

    def finish(default_root: Path, paths: list[Path]) -> list[Path]:
        image_root = (explicit_root or default_root).resolve()
        resolved = [path.resolve() for path in paths]
        for path in resolved:
            try:
                path.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(
                    f"image {path} is outside evaluation root {image_root}; "
                    "pass --image-root containing every listed image"
                ) from exc
        return resolved

    if source.is_dir():
        paths = sorted(
            (path for path in source.rglob("*") if path.is_file() and path.suffix.casefold() in extensions),
            key=lambda path: path.as_posix().casefold(),
        )
    elif source.is_file():
        paths = []
        base = source.resolve().parent
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError as exc:
            raise ValueError(f"{source}: image list must be UTF-8 text") from exc
        for line_no, raw in enumerate(lines, 1):
            line = raw.strip()
            if line_no == 1:
                line = line.lstrip("\ufeff")
            if not line or line.startswith("#"):
                continue
            if len(line) >= 2 and line[0] == line[-1] and line[0] in {'"', "'"}:
                line = line[1:-1].strip()
            path = Path(line).expanduser()
            resolved = (base / path if not path.is_absolute() else path).resolve()
            if resolved.suffix.casefold() not in extensions or not resolved.is_file():
                raise ValueError(f"{source}:{line_no}: image list entry is not a supported image: {line}")
            paths.append(resolved)
    else:
        raise FileNotFoundError(f"image source not found: {source}")
    paths = finish(source if source.is_dir() else source.parent, paths)
    result: dict[str, Path] = {}
    for path in paths:
        stem = image_stem(path)
        if stem in result:
            raise ValueError(f"validation image stems are not unique: {stem}")
        result[stem] = path
    if not result:
        raise ValueError(f"no images found under {source}")
    return result


def box_iou(a: Prediction, b: Prediction) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a.x2 - a.x1) * max(0.0, a.y2 - a.y1)
    area_b = max(0.0, b.x2 - b.x1) * max(0.0, b.y2 - b.y1)
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def match_predictions(
    reference: Iterable[Prediction], candidate: Iterable[Prediction], iou_threshold: float
) -> list[Match]:
    """Greedily match same-class detections by descending IoU.

    Candidate/reference ordering in text files must not affect the result: all
    eligible pairs are sorted by IoU, confidence, and indices before matching.
    """
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("IoU threshold must be finite and in [0,1]")
    refs = list(reference)
    cands = list(candidate)
    pairs: list[tuple[float, float, int, int]] = []
    for ri, ref in enumerate(refs):
        for ci, cand in enumerate(cands):
            if ref.class_id != cand.class_id:
                continue
            overlap = box_iou(ref, cand)
            if overlap >= iou_threshold:
                pairs.append((overlap, min(ref.confidence, cand.confidence), ri, ci))
    pairs.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    used_refs: set[int] = set()
    used_cands: set[int] = set()
    matches: list[Match] = []
    for overlap, _, ri, ci in pairs:
        if ri in used_refs or ci in used_cands:
            continue
        used_refs.add(ri)
        used_cands.add(ci)
        ref, cand = refs[ri], cands[ci]
        box_delta = max(abs(a - b) for a, b in zip(ref.box, cand.box))
        matches.append(Match(ri, ci, overlap, abs(ref.confidence - cand.confidence), box_delta))
    return matches


def _percentile(values: list[float], pct: float) -> float:
    if not math.isfinite(pct) or not 0.0 <= pct <= 100.0:
        raise ValueError("percentile must be finite and in [0,100]")
    if not values:
        return 0.0
    values = sorted(values)
    rank = max(1, math.ceil(pct * len(values) / 100.0))
    return values[min(len(values) - 1, rank - 1)]


def _iou_statistics(values: list[float]) -> dict[str, float | int]:
    """Return stable summary statistics for matched IoUs.

    Nearest-rank percentiles are used so that a report is deterministic for a
    small validation subset and does not depend on a numerical interpolation
    convention.  Empty matches are represented by zero-valued statistics and
    an explicit count, which makes a minimum-IoU gate fail closed.
    """
    return {
        "matched_iou_count": len(values),
        "mean_iou": sum(values) / len(values) if values else 0.0,
        "min_iou": min(values, default=0.0),
        "p05_iou": _percentile(values, 5.0),
        "p50_iou": _percentile(values, 50.0),
        "p95_iou": _percentile(values, 95.0),
        "p99_iou": _percentile(values, 99.0),
    }


def compare_image(reference: list[Prediction], candidate: list[Prediction], iou_threshold: float) -> dict[str, object]:
    matches = match_predictions(reference, candidate, iou_threshold)
    matched_refs = {match.reference_index for match in matches}
    matched_cands = {match.candidate_index for match in matches}
    conf = [match.confidence_abs_delta for match in matches]
    boxes = [match.box_max_abs_delta for match in matches]
    ious = [match.iou for match in matches]
    unmatched_ref = len(reference) - len(matched_refs)
    unmatched_candidate = len(candidate) - len(matched_cands)
    # Count mismatches dominate the ranking; confidence/coordinate deltas make
    # otherwise equal-count images useful in the Top-K diagnostic list.
    score = (
        unmatched_ref
        + unmatched_candidate
        + max(boxes, default=0.0)
        + max(conf, default=0.0)
    )
    return {
        "reference_count": len(reference),
        "candidate_count": len(candidate),
        "matched": len(matches),
        "unmatched_reference": unmatched_ref,
        "unmatched_candidate": unmatched_candidate,
        **_iou_statistics(ious),
        "mean_confidence_abs_delta": sum(conf) / len(conf) if conf else 0.0,
        "max_confidence_abs_delta": max(conf, default=0.0),
        "p95_confidence_abs_delta": _percentile(conf, 95.0),
        "mean_box_max_abs_delta": sum(boxes) / len(boxes) if boxes else 0.0,
        "max_box_max_abs_delta": max(boxes, default=0.0),
        "p95_box_max_abs_delta": _percentile(boxes, 95.0),
        "difference_score": score,
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "image", "reference_count", "candidate_count", "matched",
        "unmatched_reference", "unmatched_candidate", "mean_iou", "min_iou",
        "matched_iou_count", "p05_iou", "p50_iou", "p95_iou", "p99_iou",
        "mean_confidence_abs_delta", "max_confidence_abs_delta",
        "mean_box_max_abs_delta", "max_box_max_abs_delta", "difference_score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)


def _write_debug_images(
    debug_dir: Path, rows: list[dict[str, object]], image_map: dict[str, Path],
    refs: dict[str, list[Prediction]], cands: dict[str, list[Prediction]], top_k: int,
) -> str | None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return "Pillow is not installed; --debug-dir was skipped"
    debug_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(rows, key=lambda row: (-float(row["difference_score"]), str(row["image"])))[:top_k]
    for row in selected:
        stem = str(row["image"])
        source = image_map.get(stem)
        if source is None:
            continue
        image = Image.open(source).convert("RGB")
        draw = ImageDraw.Draw(image)
        for pred in refs.get(stem, []):
            draw.rectangle(pred.box, outline=(40, 190, 90), width=2)
        for pred in cands.get(stem, []):
            draw.rectangle(pred.box, outline=(235, 80, 70), width=2)
        image.save(debug_dir / f"{stem}.jpg", quality=92)
    return None


def compare_directories(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    image_source: Path | None = None,
    image_root: Path | None = None,
    iou_threshold: float = 0.5,
    allow_missing: bool = False,
    top_k: int = 20,
    min_iou: float | None = None,
) -> dict[str, object]:
    if not math.isfinite(iou_threshold) or not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("matching IoU threshold must be finite and in [0,1]")
    if min_iou is not None and (not math.isfinite(min_iou) or not 0.0 <= min_iou <= 1.0):
        raise ValueError("minimum IoU gate must be finite and in [0,1]")
    ref_files = prediction_files(reference_dir)
    cand_files = prediction_files(candidate_dir)
    image_map = image_files(image_source, image_root) if image_source is not None else {}
    expected = set(image_map) if image_map else set(ref_files) | set(cand_files)
    if not expected:
        raise ValueError("both prediction directories are empty")
    if not allow_missing:
        missing_ref = sorted(expected - set(ref_files))
        missing_cand = sorted(expected - set(cand_files))
        extra_ref = sorted(set(ref_files) - expected) if image_map else []
        extra_cand = sorted(set(cand_files) - expected) if image_map else []
        if missing_ref or missing_cand or extra_ref or extra_cand:
            details = []
            if missing_ref: details.append("reference missing: " + ", ".join(missing_ref[:5]))
            if missing_cand: details.append("candidate missing: " + ", ".join(missing_cand[:5]))
            if extra_ref: details.append("reference has unexpected stems: " + ", ".join(extra_ref[:5]))
            if extra_cand: details.append("candidate has unexpected stems: " + ", ".join(extra_cand[:5]))
            raise ValueError("prediction file sets differ; " + "; ".join(details))
    rows: list[dict[str, object]] = []
    references: dict[str, list[Prediction]] = {}
    candidates: dict[str, list[Prediction]] = {}
    for stem in sorted(expected):
        ref = read_predictions(ref_files[stem]) if stem in ref_files else []
        cand = read_predictions(cand_files[stem]) if stem in cand_files else []
        references[stem], candidates[stem] = ref, cand
        row = {"image": stem, **compare_image(ref, cand, iou_threshold)}
        rows.append(row)
    rows_by_difference = sorted(rows, key=lambda row: (-float(row["difference_score"]), str(row["image"])))
    total_ref = sum(int(row["reference_count"]) for row in rows)
    total_cand = sum(int(row["candidate_count"]) for row in rows)
    total_matched = sum(int(row["matched"]) for row in rows)
    all_ious = [
        float(match.iou)
        for stem in sorted(expected)
        for match in match_predictions(references[stem], candidates[stem], iou_threshold)
    ]
    iou_summary = _iou_statistics(all_ious)
    summary = {
        "images": len(rows),
        "reference_detections": total_ref,
        "candidate_detections": total_cand,
        "matched_detections": total_matched,
        "unmatched_reference": sum(int(row["unmatched_reference"]) for row in rows),
        "unmatched_candidate": sum(int(row["unmatched_candidate"]) for row in rows),
        "images_with_count_difference": sum(
            int(row["reference_count"]) != int(row["candidate_count"]) for row in rows
        ),
        "max_confidence_abs_delta": max(float(row["max_confidence_abs_delta"]) for row in rows),
        "max_box_max_abs_delta": max(float(row["max_box_max_abs_delta"]) for row in rows),
        **iou_summary,
    }
    summary["min_iou_gate_passed"] = (
        min_iou is None or float(iou_summary["min_iou"]) >= min_iou
    )
    protocol = {"match_iou": iou_threshold, "box_format": "pixel_xyxy", "class_aware": True}
    if min_iou is not None:
        protocol["min_match_iou"] = min_iou
    return {
        "schema_version": 1,
        "protocol": protocol,
        "summary": summary,
        "images": rows,
        "top_differences": rows_by_difference[: max(1, top_k)],
        "_image_map": image_map,
        "_references": references,
        "_candidates": candidates,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="reference YOLO TXT directory")
    parser.add_argument("--candidate", type=Path, required=True, help="candidate YOLO TXT directory")
    parser.add_argument("--images", type=Path, help="validation image directory or ordered image list")
    parser.add_argument(
        "--image-root", type=Path,
        help="explicit root containing every listed image (for portable path checks)",
    )
    parser.add_argument("--iou", type=float, default=0.5, help="minimum same-class IoU for matching")
    parser.add_argument("--top-k", type=int, default=20, help="number of largest per-image differences to retain")
    parser.add_argument("--allow-missing", action="store_true", help="treat a missing prediction file as empty")
    parser.add_argument("--json", type=Path, help="write full machine-readable report")
    parser.add_argument("--csv", type=Path, help="write per-image CSV report")
    parser.add_argument("--debug-dir", type=Path, help="optional PIL visualizations for top differences")
    parser.add_argument("--max-unmatched", type=int, default=None, help="fail if total unmatched detections exceed this")
    parser.add_argument("--max-box-delta", type=float, default=None, help="fail if any matched box coordinate delta exceeds this")
    parser.add_argument("--max-conf-delta", type=float, default=None, help="fail if any matched confidence delta exceeds this")
    parser.add_argument(
        "--min-iou", "--min-match-iou", dest="min_iou", type=float, default=None,
        help="fail when the minimum IoU among matched detections is below this value",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    for name in ("max_unmatched", "max_box_delta", "max_conf_delta"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(float(value)) or value < 0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")
    if args.min_iou is not None and (not math.isfinite(args.min_iou) or not 0.0 <= args.min_iou <= 1.0):
        raise ValueError("--min-iou must be finite and in [0,1]")
    report = compare_directories(
        args.reference, args.candidate, image_source=args.images,
        image_root=args.image_root,
        iou_threshold=args.iou, allow_missing=args.allow_missing, top_k=args.top_k,
        min_iou=args.min_iou,
    )
    image_map = report.pop("_image_map")
    references = report.pop("_references")
    candidates = report.pop("_candidates")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.csv:
        _write_csv(args.csv, report["images"])
    warning = None
    if args.debug_dir:
        warning = _write_debug_images(args.debug_dir, report["images"], image_map, references, candidates, args.top_k)
    print(json.dumps({"summary": report["summary"], "top_differences": report["top_differences"]}, indent=2))
    if warning:
        print(f"warning: {warning}")
    summary = report["summary"]
    unmatched = int(summary["unmatched_reference"]) + int(summary["unmatched_candidate"])
    if args.max_unmatched is not None and unmatched > args.max_unmatched:
        return 1
    if args.max_box_delta is not None and float(summary["max_box_max_abs_delta"]) > args.max_box_delta:
        return 1
    if args.max_conf_delta is not None and float(summary["max_confidence_abs_delta"]) > args.max_conf_delta:
        return 1
    if args.min_iou is not None and float(summary["min_iou"]) < args.min_iou:
        print(
            "minimum matched IoU gate failed: "
            f"observed {float(summary['min_iou']):.6g} < minimum {args.min_iou:.6g}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
