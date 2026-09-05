#!/usr/bin/env python3
"""Compute VisDrone mAP50-95 from per-image pixel-xyxy predictions.

The evaluator delegates matching and AP integration to Ultralytics' own
``DetMetrics`` so the result is comparable to ``model.val``. Predictions must
use the runner's ``class conf x1 y1 x2 y2`` format. Native VisDrone rows are
accepted for diagnostics; for the formal acceptance gate, use labels produced
by the official ``visdrone2yolo`` conversion so ignored-region matching is
defined by the dataset conversion.
The result exposes both delta conventions: ``delta_mAP50-95_pp`` is the
absolute difference in percentage points (``(candidate-reference)*100``),
while ``delta_mAP50-95_pct`` is the relative percentage difference.  Use
``--max-abs-delta-pp`` for the Issue #51 acceptance budget; the older
``--max-abs-delta-pct`` option remains available for relative comparisons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

NAMES = {
    0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
    5: "truck", 6: "tricycle", 7: "awning-tricycle", 8: "bus", 9: "motor",
}
CLASS_TABLES = {
    "visdrone": NAMES,
    "sku110k": {0: "object"},
}
PROFILE_PROTOCOLS = {
    "visdrone": {"imgsz": 640, "conf": 0.001, "iou": 0.70, "max_det": 300},
    "sku110k": {"imgsz": 1280, "conf": 0.25, "iou": 0.60, "max_det": 300},
}
ROUTING_SEMANTICS = ("native_sparse", "dense_fallback", "dense_native", "not_applicable")


def manifest_name(path: Path, root: Path) -> str:
    """Return a mount-independent, POSIX-formatted image path."""
    resolved_path = path.resolve()
    resolved_root = root.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        # Preserve a relative relationship for symlinked files outside root;
        # absolute mount paths would make equivalent runs hash differently.
        try:
            relative = Path(os.path.relpath(resolved_path, resolved_root))
        except ValueError:
            # Different Windows drives have no common relative spelling. Use
            # the basename rather than failing after metrics have been computed.
            relative = Path(resolved_path.name)
    return relative.as_posix()


def sha256_file(path: Path) -> str:
    """Hash a file incrementally so large validation images stay bounded in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_content_manifest(images: list[Path], root: Path) -> str:
    """Return a stable digest over ordered relative names and image contents."""
    payload = "\n".join(
        f"{manifest_name(path, root)} {sha256_file(path)}" for path in images
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()

EXAMPLE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLE_ROOT.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def load_gt(
    path: Path,
    width: int,
    height: int,
    torch,
    label_format: str = "auto",
    num_classes: Optional[int] = None,
    strict: bool = False,
):
    """Load one label file.

    Diagnostic runs may ignore malformed rows for compatibility with legacy
    exports.  Formal Issue #51 runs pass ``strict=True`` so every annotation
    is either parsed or reported with its source line; silently dropping a row
    would change the denominator of the reported metric.
    """
    if label_format not in ("auto", "yolo", "visdrone"):
        raise ValueError("label_format must be yolo, visdrone, or auto")
    if not np.isfinite([width, height]).all() or width <= 0 or height <= 0:
        raise ValueError("image dimensions must be finite and positive")
    boxes, classes = [], []
    if not path.is_file():
        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(classes, dtype=torch.int64)
    try:
        rows = [(line_no, line.strip()) for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ) if line.strip() and not line.lstrip().startswith("#")]
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: labels must be UTF-8 text") from exc
    if not rows:
        return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(classes, dtype=torch.int64)
    first_line = rows[0][1]
    use_visdrone = label_format == "visdrone" or (
        label_format == "auto" and ("," in first_line or len(first_line.replace(",", " ").split()) >= 8)
    )
    for line_no, line in rows:
        # Native VisDrone rows carry eight fields; converted YOLO rows carry
        # exactly five.  Keep the diagnostic path permissive, but never allow
        # a malformed row through a formal metric.
        if not use_visdrone and "," in line:
            if strict:
                raise ValueError(f"{path}:{line_no}: mixed label formats; expected whitespace-separated YOLO row")
            continue
        fields = [field.strip() for field in line.split(",")] if "," in line else line.split()
        expected = 8 if use_visdrone else 5
        if len(fields) != expected:
            if strict:
                raise ValueError(f"{path}:{line_no}: expected exactly {expected} columns, got {len(fields)}")
            continue
        try:
            values = [float(value) for value in fields]
        except (TypeError, ValueError, OverflowError) as exc:
            if strict:
                raise ValueError(f"{path}:{line_no}: label contains a non-numeric value") from exc
            continue
        if not np.isfinite(values).all():
            if strict:
                raise ValueError(f"{path}:{line_no}: label contains NaN or Inf")
            continue
        if use_visdrone:
            left, top, bw, bh, score = values[0:5]
            category_value = values[5]
            if category_value != np.floor(category_value):
                if strict:
                    raise ValueError(f"{path}:{line_no}: VisDrone category must be an integer")
                continue
            category = int(category_value)
            if bw <= 0 or bh <= 0 or left < 0 or top < 0 or not 0.0 <= score <= 1.0:
                if strict:
                    raise ValueError(f"{path}:{line_no}: invalid VisDrone box, score, or origin")
                continue
            # Category 0 (ignored regions), category 11 (others), and score 0
            # are intentionally excluded by the official conversion.
            if category < 0 or category > 11:
                if strict:
                    raise ValueError(f"{path}:{line_no}: VisDrone category outside 0..11")
                continue
            if score == 0 or category in (0, 11):
                continue
            cls = category - 1
            box = [left, top, left + bw, top + bh]
        else:
            cls_value = values[0]
            if cls_value != np.floor(cls_value):
                if strict:
                    raise ValueError(f"{path}:{line_no}: YOLO class id must be an integer")
                continue
            cls = int(cls_value)
            cx, cy, bw, bh = values[1:5]
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and 0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                if strict:
                    raise ValueError(f"{path}:{line_no}: YOLO coordinates must be normalized and positive")
                continue
            box = [(cx - bw / 2) * width, (cy - bh / 2) * height,
                   (cx + bw / 2) * width, (cy + bh / 2) * height]
        if num_classes is not None and not 0 <= cls < num_classes:
            if strict:
                raise ValueError(f"{path}:{line_no}: class {cls} outside [0, {num_classes})")
            continue
        if not np.isfinite([cls, *box]).all() or bw <= 0 or bh <= 0 or cls < 0:
            if strict:
                raise ValueError(f"{path}:{line_no}: label contains a degenerate box")
            continue
        classes.append(cls)
        boxes.append(box)
    return torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4), torch.tensor(classes, dtype=torch.int64)


def load_predictions(path: Path, torch, num_classes: Optional[int] = None, strict: bool = False):
    """Load one pixel-xyxy prediction file with optional strict validation."""
    boxes, scores, classes = [], [], []
    if not path.is_file():
        return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                torch.tensor(scores, dtype=torch.float32), torch.tensor(classes, dtype=torch.int64))
    try:
        rows = [(line_no, line.strip()) for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ) if line.strip() and not line.lstrip().startswith("#")]
    except UnicodeDecodeError as exc:
        raise ValueError(f"{path}: predictions must be UTF-8 text") from exc
    for line_no, line in rows:
        fields = line.split()
        if len(fields) != 6:
            if strict:
                raise ValueError(f"{path}:{line_no}: prediction expects exactly 6 columns, got {len(fields)}")
            continue
        try:
            cls_value = float(fields[0]); score = float(fields[1])
            box = [float(value) for value in fields[2:6]]
        except (TypeError, ValueError, OverflowError) as exc:
            if strict:
                raise ValueError(f"{path}:{line_no}: prediction contains a non-numeric value") from exc
            continue
        if not np.isfinite([cls_value, score, *box]).all():
            if strict:
                raise ValueError(f"{path}:{line_no}: prediction contains NaN or Inf")
            continue
        if cls_value != np.floor(cls_value):
            if strict:
                raise ValueError(f"{path}:{line_no}: prediction class id must be an integer")
            continue
        cls = int(cls_value)
        if cls < 0 or (num_classes is not None and cls >= num_classes):
            if strict:
                raise ValueError(f"{path}:{line_no}: prediction class {cls} outside [0, {num_classes})")
            # Diagnostic runs may inspect legacy output directories that carry
            # stale class IDs.  Ignore the offending row consistently with the
            # other permissive parsing branches; formal acceptance remains strict.
            continue
        if not 0.0 <= score <= 1.0:
            if strict:
                raise ValueError(f"{path}:{line_no}: prediction confidence must be in [0, 1]")
            continue
        # Keep the evaluator's candidate contract aligned with the C++ and
        # MNN decoders: degenerate pixel boxes are never scored.
        if box[2] <= box[0] or box[3] <= box[1]:
            if strict:
                raise ValueError(f"{path}:{line_no}: prediction box must have x2>x1 and y2>y1")
            continue
        classes.append(cls); scores.append(score); boxes.append(box)
    return (torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            torch.tensor(scores, dtype=torch.float32), torch.tensor(classes, dtype=torch.int64))


def _txt_stem_map(directory: Path, kind: str) -> dict[str, Path]:
    """Return a case-insensitive stem map and reject duplicate stems."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{kind} directory not found: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(
        (candidate for candidate in directory.rglob("*")
         if candidate.is_file() and candidate.suffix.lower() == ".txt"),
        key=lambda p: p.as_posix().casefold(),
    ):
        stem = path.stem.casefold()
        if stem in result:
            raise RuntimeError(f"{kind} stems are not unique: {stem}")
        result[stem] = path
    return result


def _resolve_images(source: Path, root: Optional[Path] = None) -> tuple[Path, list[Path]]:
    """Resolve images below one canonical root.

    An explicit root is useful when a frozen list is stored outside the
    dataset tree.  Requiring every image below that root prevents portable
    evidence digests from containing absolute paths or ``..`` components.
    """
    source = source.expanduser()
    explicit_root = root.expanduser().resolve() if root is not None else None
    if explicit_root is not None and not explicit_root.is_dir():
        raise NotADirectoryError(f"image normalization root not found: {explicit_root}")

    def finish(default_root: Path, paths: list[Path]) -> tuple[Path, list[Path]]:
        image_root = (explicit_root or default_root).resolve()
        resolved_paths = [path.resolve() for path in paths]
        for path in resolved_paths:
            try:
                path.relative_to(image_root)
            except ValueError as exc:
                raise ValueError(
                    f"image {path} is outside evaluation root {image_root}; "
                    "pass --image-root containing every listed image"
                ) from exc
        return image_root, resolved_paths

    if source.is_dir():
        paths = sorted(
            (
                candidate for candidate in source.rglob("*")
                if candidate.is_file() and candidate.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
            ),
            key=lambda path: path.as_posix().casefold(),
        )
        return finish(source, paths)
    if not source.is_file():
        raise FileNotFoundError(f"image source not found: {source}")
    paths: list[Path] = []
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"{source}: image list must be UTF-8 text") from exc
    for line_no, raw in enumerate(lines, 1):
        # Lists exported by Windows tools may carry a UTF-8 BOM.  Quoted
        # entries are accepted as well so paths containing spaces retain the
        # same interpretation in Python and in the C++ runner.
        line = raw.strip()
        if line_no == 1:
            line = line.lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if len(line) >= 2 and line[0] == line[-1] and line[0] in {'"', "'"}:
            line = line[1:-1].strip()
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        candidate = candidate.resolve()
        if not candidate.is_file() or candidate.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
            raise ValueError(f"{source}:{line_no}: unsupported or missing image: {line}")
        paths.append(candidate)
    if not paths:
        raise ValueError(f"image list is empty: {source}")
    return finish(source.parent, paths)


def match_predictions(pred_cls, true_cls, iou, torch, iou_thresholds):
    """Ultralytics-compatible greedy class-aware matching for 0.50:0.95 IoU."""
    correct = np.zeros((pred_cls.shape[0], len(iou_thresholds)), dtype=bool)
    correct_class = true_cls[:, None] == pred_cls
    iou_np = (iou * correct_class).cpu().numpy()
    # Convert Torch scalars to Python floats before comparing with the NumPy
    # IoU matrix.  This avoids version-dependent ndarray/Tensor dispatch while
    # preserving the 0.50:0.95 thresholds exactly.
    for column, threshold in enumerate(iou_thresholds.tolist()):
        matches = np.array(np.nonzero(iou_np >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), column] = True
    return torch.tensor(correct)


def nonnegative_finite_float(value: str) -> float:
    """Parse a finite, non-negative percentage budget for the CLI."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not np.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def small_conf_float(value: str) -> float:
    """Parse the optional small-object confidence floor (-1 disables it)."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite number in [-1, 1]") from exc
    if not np.isfinite(parsed) or parsed < -1.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError("must be a finite number in [-1, 1]")
    return parsed


def nonnegative_finite_area(value: str) -> float:
    """Parse an original-image area threshold for the small-object sweep."""
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("must be a finite non-negative number") from exc
    if not np.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def delta_gate_passes(abs_delta_pct: float, max_abs_delta_pct: Optional[float]) -> bool:
    """Return whether a relative mAP delta is within the requested budget.

    ``None`` means that no gate was requested.  Budgets are expressed as a
    relative percentage of the reference mAP (for example, ``0.5`` means
    0.5% of the reference value).
    The comparison is inclusive so a result exactly on the declared maximum
    is accepted.
    """
    if max_abs_delta_pct is None:
        return True
    return float(abs_delta_pct) <= float(max_abs_delta_pct)


def delta_gate_passes_pp(abs_delta_pp: float, max_abs_delta_pp: Optional[float]) -> bool:
    """Return whether an absolute mAP delta in percentage points passes.

    mAP is represented as a fraction in the range ``[0, 1]``.  Consequently,
    ``(candidate - reference) * 100`` is the absolute percentage-point delta;
    it is different from the relative percentage used by
    :func:`delta_gate_passes`.
    """
    if max_abs_delta_pp is None:
        return True
    return float(abs_delta_pp) <= float(max_abs_delta_pp)


def validate_delta_budget(
    max_abs_delta_pct: Optional[float], reference_json: Optional[Path],
    max_abs_delta_pp: Optional[float] = None,
) -> None:
    """Reject a requested gate that cannot be compared with a reference."""
    if max_abs_delta_pct is not None and (
        not np.isfinite(float(max_abs_delta_pct)) or float(max_abs_delta_pct) < 0
    ):
        raise ValueError("--max-abs-delta-pct must be a finite non-negative number")
    if max_abs_delta_pct is not None and reference_json is None:
        raise ValueError("--max-abs-delta-pct requires --reference-json")
    if max_abs_delta_pp is not None and (
        not np.isfinite(float(max_abs_delta_pp)) or float(max_abs_delta_pp) < 0
    ):
        raise ValueError("--max-abs-delta-pp must be a finite non-negative number")
    if max_abs_delta_pp is not None and reference_json is None:
        raise ValueError("--max-abs-delta-pp requires --reference-json")
    if max_abs_delta_pct is not None and max_abs_delta_pp is not None:
        raise ValueError("choose either --max-abs-delta-pct or --max-abs-delta-pp")


def validate_smoke_gate(
    smoke: bool,
    max_abs_delta_pct: Optional[float],
    max_abs_delta_pp: Optional[float] = None,
) -> None:
    """Keep an acceptance delta gate from being attached to a smoke subset."""
    if smoke and (max_abs_delta_pct is not None or max_abs_delta_pp is not None):
        raise ValueError(
            "an mAP delta gate cannot be used with --smoke; smoke runs are not acceptance evidence"
        )


def validate_acceptance_image_floor(args: argparse.Namespace) -> None:
    """Keep the 500-image acceptance floor from being weakened by a CLI flag."""
    if not args.smoke and args.min_images < 500:
        raise ValueError(
            "Issue #51 acceptance requires --min-images >= 500; use --smoke for a smaller diagnostic run"
        )


def apply_delta_gate(
    result,
    max_abs_delta_pct: Optional[float],
    max_abs_delta_pp: Optional[float] = None,
) -> int:
    """Annotate ``result`` and return the process code for the optional gate."""
    if max_abs_delta_pct is None and max_abs_delta_pp is None:
        return 0
    passed = True
    if max_abs_delta_pct is not None:
        observed = result.get("abs_delta_mAP50-95_pct")
        if observed is None:
            raise ValueError("--max-abs-delta-pct requires a computed reference delta")
        passed = delta_gate_passes(observed, max_abs_delta_pct)
        result["max_abs_delta_mAP50-95_pct"] = max_abs_delta_pct
        result["mAP50-95_relative_delta_gate_passed"] = passed
    if max_abs_delta_pp is not None:
        observed_pp = result.get("abs_delta_mAP50-95_pp")
        if observed_pp is None:
            raise ValueError("--max-abs-delta-pp requires a computed reference delta")
        passed_pp = delta_gate_passes_pp(observed_pp, max_abs_delta_pp)
        result["max_abs_delta_mAP50-95_pp"] = max_abs_delta_pp
        result["mAP50-95_absolute_delta_gate_passed"] = passed_pp
        passed = passed and passed_pp
    result["mAP50-95_delta_gate_passed"] = passed
    return 0 if passed else 2


def extract_reference_map(payload: dict) -> float:
    """Extract mAP50-95 from common Ultralytics result JSON layouts."""
    if not isinstance(payload, dict):
        raise ValueError("reference JSON must contain an object")
    direct_keys = (
        "mAP50-95",
        "map50-95",
        "metrics/mAP50-95(B)",
        "metrics/mAP50-95",
    )
    mappings = [payload]
    for key in ("metrics", "results", "results_dict", "metrics_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            mappings.append(value)
    for mapping in mappings:
        for key in direct_keys:
            if key in mapping:
                try:
                    return float(mapping[key])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"reference mAP50-95 is not numeric: {mapping[key]!r}") from exc
        # Be tolerant of case differences while retaining the metric suffix
        # used by Ultralytics for box results.
        for key, value in mapping.items():
            normalized = str(key).lower().replace(" ", "")
            if normalized in {"map50-95", "map50-95(b)", "metrics/map50-95(b)", "metrics/map50-95"}:
                try:
                    return float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"reference mAP50-95 is not numeric: {value!r}") from exc
    raise ValueError(
        "reference JSON must contain mAP50-95, map50-95, or metrics/mAP50-95(B)"
    )


def _protocol_mismatches(reference: object, current: dict) -> list[str]:
    """Return protocol/manifest mismatches between two metric reports."""
    if not isinstance(reference, dict):
        return ["reference JSON must contain an object"]
    errors: list[str] = []
    ref_manifest = reference.get("image_manifest_sha256")
    cur_manifest = current.get("image_manifest_sha256")
    if ref_manifest is None:
        errors.append("reference JSON is missing image_manifest_sha256")
    elif str(ref_manifest).lower() != str(cur_manifest).lower():
        errors.append("reference image_manifest_sha256 does not match the candidate image list")
    if "image_manifest" in reference and reference.get("image_manifest") != current.get("image_manifest"):
        errors.append("reference image_manifest order/content does not match the candidate")
    # New reports pin image bytes in addition to their ordered names.  Keep
    # legacy scalar/name-only reports usable for diagnostics, but require the
    # content digest whenever the candidate advertises one (strict acceptance).
    current_content = current.get("image_list_sha256") or current.get("image_content_manifest_sha256")
    reference_content = reference.get("image_list_sha256") or reference.get("image_content_manifest_sha256")
    if current_content is not None:
        if reference_content is None:
            errors.append("reference JSON is missing image_list_sha256")
        elif str(reference_content).lower() != str(current_content).lower():
            errors.append("reference image_list_sha256 does not match the candidate images")
    ref_protocol = reference.get("protocol")
    cur_protocol = current.get("protocol")
    if not isinstance(ref_protocol, dict):
        errors.append("reference JSON is missing protocol metadata")
    elif not isinstance(cur_protocol, dict):
        errors.append("candidate JSON is missing protocol metadata")
    else:
        for key in ("imgsz", "max_det", "multi_label", "letterbox", "color", "layout"):
            if key not in ref_protocol:
                errors.append(f"reference protocol is missing {key}")
            elif ref_protocol[key] != cur_protocol.get(key):
                errors.append(f"reference protocol.{key} does not match the candidate")
        for key in ("conf", "iou", "small_conf", "small_area"):
            if key not in ref_protocol:
                errors.append(f"reference protocol is missing {key}")
                continue
            try:
                if abs(float(ref_protocol[key]) - float(cur_protocol.get(key))) > 1e-9:
                    errors.append(f"reference protocol.{key} does not match the candidate")
            except (TypeError, ValueError):
                errors.append(f"reference protocol.{key} is not numeric")
    ref_profile = reference.get("class_profile")
    if ref_profile is None:
        errors.append("reference JSON is missing class_profile")
    elif ref_profile != current.get("class_profile"):
        errors.append("reference class_profile does not match the candidate")
    ref_classes = reference.get("classes")
    if ref_classes is None:
        errors.append("reference JSON is missing classes")
    else:
        try:
            if int(ref_classes) != int(current.get("classes")):
                errors.append("reference class count does not match the candidate")
        except (TypeError, ValueError):
            errors.append("reference classes is not an integer")
    try:
        if int(reference.get("images")) != int(current.get("images")):
            errors.append("reference image count does not match the candidate")
    except (TypeError, ValueError):
        errors.append("reference JSON is missing a valid images field")
    if reference.get("label_format") is not None and reference.get("label_format") != current.get("label_format"):
        errors.append("reference label_format does not match the candidate")
    # A dense export and a sparse eager run can produce different predictions
    # even when every image and threshold is identical.  Compare the optional
    # field whenever either report declares it; formal callers additionally
    # require a non-null value before applying a delta gate.
    ref_routing = ref_protocol.get("routing_semantics") if isinstance(ref_protocol, dict) else None
    cur_routing = cur_protocol.get("routing_semantics") if isinstance(cur_protocol, dict) else None
    if ref_routing is not None or cur_routing is not None:
        if ref_routing is None:
            errors.append("reference protocol is missing routing_semantics")
        elif cur_routing is None:
            errors.append("candidate protocol is missing routing_semantics")
        elif ref_routing != cur_routing:
            errors.append("reference protocol.routing_semantics does not match the candidate")
    return errors


def validate_reference_metadata(reference: dict, current: dict, *, strict: bool) -> None:
    """Validate metadata needed for an auditable cross-backend comparison.

    Legacy metric JSON files containing only a scalar mAP remain usable for a
    diagnostic comparison.  A requested acceptance gate, however, must carry
    the same ordered image manifest, class profile and post-processing
    protocol; otherwise a passing delta could compare different experiments.
    """
    errors = _protocol_mismatches(reference, current)
    if errors and strict:
        raise ValueError("; ".join(errors))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preds", type=Path, required=True)
    parser.add_argument(
        "--images", type=Path, default=Path("/data/datasets/VisDrone/images/val"),
        help="validation image directory or ordered UTF-8 image list",
    )
    parser.add_argument(
        "--image-root", type=Path,
        help="root used to normalize list entries and compute portable image digests",
    )
    parser.add_argument("--labels", type=Path, default=Path("/data/datasets/VisDrone/labels/val"))
    parser.add_argument("--classes", choices=tuple(CLASS_TABLES), default="visdrone")
    parser.add_argument(
        "--label-format", choices=("auto", "yolo", "visdrone"), default="auto",
        help=(
            "YOLO normalized labels or raw VisDrone x/y/w/h/score/category rows "
            "(native rows are diagnostic only)"
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--smoke", action="store_true",
        help="allow fewer than --min-images for a dependency/data smoke check (not an Issue #51 acceptance run)",
    )
    parser.add_argument("--min-images", type=int, default=500, help="minimum images outside --smoke (default: 500)")
    parser.add_argument("--json", type=Path, help="optional JSON result path")
    parser.add_argument(
        "--reference-json", type=Path,
        help="optional PyTorch/reference JSON containing mAP50-95 for a relative delta",
    )
    parser.add_argument(
        "--max-abs-delta-pct",
        type=nonnegative_finite_float,
        default=None,
        metavar="PERCENT",
        help=(
            "fail with a non-zero status when the absolute relative mAP50-95 "
            "delta exceeds PERCENT (for example, 0.5 for a 0.5%% budget); "
            "requires --reference-json"
        ),
    )
    parser.add_argument(
        "--max-abs-delta-pp",
        type=nonnegative_finite_float,
        default=None,
        metavar="POINTS",
        help=(
            "fail when the absolute mAP50-95 delta exceeds POINTS percentage points "
            "(for example, 0.5 for the Issue #51 FP32 budget); requires --reference-json"
        ),
    )
    # These values are recorded in the result so a prediction directory cannot
    # be detached from the post-processing protocol used to produce it.
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=nonnegative_finite_float, default=None)
    parser.add_argument("--iou", type=nonnegative_finite_float, default=None)
    parser.add_argument("--max-det", type=int, default=None)
    parser.add_argument(
        "--small-conf", type=small_conf_float, default=-1.0,
        help="optional lower confidence for boxes below --small-area (-1 disables)",
    )
    parser.add_argument(
        "--small-area", type=nonnegative_finite_area, default=32.0 * 32.0,
        help="original-image area threshold for --small-conf (default: 1024)",
    )
    parser.add_argument(
        "--multi-label", dest="multi_label", action="store_true", default=True,
        help="record one detection per class and anchor (the Issue #51 recipe)",
    )
    parser.add_argument(
        "--single-label", dest="multi_label", action="store_false",
        help="record argmax-per-anchor post-processing (diagnostic only)",
    )
    parser.add_argument(
        "--routing-semantics", choices=ROUTING_SEMANTICS, default=None,
        help=(
            "EsMoE inference path: native_sparse, dense_fallback, dense_native, "
            "or not_applicable; required for formal runs"
        ),
    )
    args = parser.parse_args()
    defaults = PROFILE_PROTOCOLS[args.classes]
    for key in ("imgsz", "conf", "iou", "max_det"):
        if getattr(args, key) is None:
            setattr(args, key, defaults[key])
    return args


def main() -> int:
    args = parse_args()
    validate_delta_budget(args.max_abs_delta_pct, args.reference_json, args.max_abs_delta_pp)
    validate_smoke_gate(args.smoke, args.max_abs_delta_pct, args.max_abs_delta_pp)
    if not args.images.exists():
        raise FileNotFoundError(f"image source not found: {args.images}")
    if not args.preds.is_dir():
        raise FileNotFoundError(f"prediction directory not found: {args.preds}")
    if args.limit < 0 or args.min_images < 1 or args.max_det < 1 or args.imgsz <= 0:
        raise ValueError("imgsz/min-images/max-det must be positive (limit may be zero)")
    if not np.isfinite([args.conf, args.iou, args.small_conf, args.small_area]).all():
        raise ValueError("conf/iou/small-conf/small-area must be finite")
    if (not 0.0 <= args.conf <= 1.0 or not 0.0 <= args.iou <= 1.0
            or not -1.0 <= args.small_conf <= 1.0 or args.small_area < 0.0):
        raise ValueError("conf/iou must be in [0, 1], small-conf in [-1, 1], and small-area non-negative")
    validate_acceptance_image_floor(args)
    if not args.smoke and args.routing_semantics is None:
        raise ValueError(
            "formal Issue #51 evaluation requires --routing-semantics; "
            "use dense_fallback for the static export path"
        )
    if not args.smoke and args.label_format != "yolo":
        raise ValueError(
            "formal Issue #51 evaluation requires converted YOLO labels; "
            "use --label-format yolo (native VisDrone rows are diagnostic only)"
        )

    import torch
    from PIL import Image
    from ultralytics.utils.metrics import DetMetrics, box_iou

    iou_thresholds = torch.linspace(0.5, 0.95, 10)
    # Match the portable C++ runner's stb decoder so every backend evaluates
    # the same ordered image set.
    image_root, images = _resolve_images(args.images, args.image_root)
    if args.limit > 0:
        images = images[: args.limit]
    if not images:
        raise RuntimeError(f"no validation images found under {args.images}")
    stems = [image_path.stem.casefold() for image_path in images]
    if len(stems) != len(set(stems)):
        raise RuntimeError("validation image stems are not unique; use a flattened/renamed validation split")
    if not args.smoke and len(images) < args.min_images:
        raise RuntimeError(
            f"Issue #51 acceptance requires at least {args.min_images} images; found {len(images)}. "
            "Use --smoke only for a non-acceptance check."
        )

    # A formal run must account for exactly one label and prediction file per
    # image.  Ignoring an extra file can hide a stale prediction from a prior
    # run and makes the reported image manifest ambiguous.  Smoke runs may use
    # a subset, so they only require unique stems.
    label_files = (
        {}
        if args.smoke and not args.labels.exists()
        else _txt_stem_map(args.labels, "label")
    )
    prediction_files = _txt_stem_map(args.preds, "prediction")
    expected_stems = set(stems)
    if not args.smoke:
        missing_labels = sorted(expected_stems - set(label_files))
        missing_predictions = sorted(expected_stems - set(prediction_files))
        extra_labels = sorted(set(label_files) - expected_stems)
        extra_predictions = sorted(set(prediction_files) - expected_stems)
        problems = []
        if missing_labels:
            problems.append("missing labels: " + ", ".join(missing_labels[:5]))
        if missing_predictions:
            problems.append("missing predictions: " + ", ".join(missing_predictions[:5]))
        if extra_labels:
            problems.append("unexpected label stems: " + ", ".join(extra_labels[:5]))
        if extra_predictions:
            problems.append("unexpected prediction stems: " + ", ".join(extra_predictions[:5]))
        if problems:
            raise RuntimeError("formal evaluation requires an exact image/file set; " + "; ".join(problems))

    names = CLASS_TABLES[args.classes]
    metrics = DetMetrics()
    metrics.names = names
    for image_index, image_path in enumerate(images):
        image_stem = image_path.stem.casefold()
        label_path = label_files.get(image_stem, args.labels / f"{image_path.stem}.txt")
        prediction_path = prediction_files.get(image_stem, args.preds / f"{image_path.stem}.txt")
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes, gt_classes = load_gt(
            label_path, width, height, torch, args.label_format, len(names), strict=not args.smoke
        )
        pred_boxes, pred_scores, pred_classes = load_predictions(
            prediction_path, torch, len(names), strict=not args.smoke
        )
        n_pred, n_true = pred_boxes.shape[0], gt_boxes.shape[0]
        if n_pred and n_true:
            true_positive = match_predictions(
                pred_classes, gt_classes, box_iou(gt_boxes, pred_boxes), torch, iou_thresholds
            ).cpu().numpy()
        else:
            true_positive = np.zeros((n_pred, len(iou_thresholds)), dtype=bool)
        metrics.update_stats({
            "tp": true_positive,
            "target_cls": gt_classes.numpy(),
            # One image index per GT instance is required by DetMetrics. Using
            # class IDs here under-counts/over-counts images with mixed classes.
            "target_img": np.full(n_true, image_index, dtype=np.int64),
            "conf": pred_scores.numpy(),
            "pred_cls": pred_classes.numpy(),
            # Newer Ultralytics versions use this key for per-image metrics;
            # older versions ignore unknown dictionary entries.
            "im_name": image_path.name,
        })

    metrics.process()
    manifest_names = [manifest_name(path, image_root) for path in images]
    image_manifest = "\n".join(manifest_names) + "\n"
    map50 = float(metrics.box.map50)
    map5095 = float(metrics.box.map)
    if not np.isfinite([map50, map5095]).all():
        raise RuntimeError("mAP computation returned NaN or Inf; check labels and predictions")
    image_list_sha256 = image_content_manifest(images, image_root)
    result = {
        "images": len(images),
        "classes": len(names),
        "class_profile": args.classes,
        "label_format": args.label_format,
        "protocol": {
            "imgsz": args.imgsz,
            "conf": args.conf,
            "iou": args.iou,
            "max_det": args.max_det,
            "multi_label": bool(args.multi_label),
            "letterbox": True,
            "small_conf": args.small_conf,
            "small_area": args.small_area,
            "color": "RGB",
            "layout": "NCHW",
            "routing_semantics": args.routing_semantics,
        },
        "image_manifest_sha256": hashlib.sha256(image_manifest.encode("utf-8")).hexdigest(),
        "image_manifest": manifest_names,
        # This is the same ordered ``relative-path + file-SHA256`` digest used
        # by evidence_manifest.py.  Retain the older key for report readers
        # created before the evidence schema was introduced.
        "image_list_sha256": image_list_sha256,
        "image_content_manifest_sha256": image_list_sha256,
        "mAP50": map50,
        "mAP50-95": map5095,
    }
    if args.reference_json:
        if not args.reference_json.is_file():
            raise FileNotFoundError(f"reference JSON not found: {args.reference_json}")
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
        # A delta gate is an acceptance claim; require the reference run to
        # identify the same ordered images, classes and post-processing.  A
        # legacy scalar-only JSON remains available for smoke diagnostics.
        reference_metadata_strict = not args.smoke and (
            args.max_abs_delta_pct is not None or args.max_abs_delta_pp is not None
        )
        reference_metadata_errors = _protocol_mismatches(reference, result)
        result["reference_metadata_match"] = not reference_metadata_errors
        if reference_metadata_errors and not reference_metadata_strict:
            result["reference_metadata_warnings"] = reference_metadata_errors
        validate_reference_metadata(reference, result, strict=reference_metadata_strict)
        reference_map = extract_reference_map(reference)
        if not np.isfinite(reference_map) or not 0.0 <= reference_map <= 1.0:
            raise ValueError("reference mAP50-95 must be finite and in [0, 1]")
        if args.max_abs_delta_pct is not None and reference_map <= 0:
            raise ValueError("reference mAP50-95 must be positive when applying a relative delta gate")
        result["reference_mAP50-95"] = reference_map
        delta_abs = float(result["mAP50-95"] - reference_map)
        result["delta_mAP50-95_abs"] = delta_abs
        # Absolute percentage points (the terminology used in the Issue #51
        # reports) and relative percent are both retained to prevent ambiguity.
        result["delta_mAP50-95_pp"] = delta_abs * 100.0
        result["abs_delta_mAP50-95_pp"] = abs(delta_abs) * 100.0
        if reference_map > 0.0:
            delta_pct = float(delta_abs / reference_map * 100.0)
            result["delta_mAP50-95_pct"] = delta_pct
            result["abs_delta_mAP50-95_pct"] = abs(delta_pct)
        else:
            # A zero reference is meaningful for an absolute percentage-point
            # comparison, but a relative percentage is undefined.
            result["delta_mAP50-95_pct"] = None
            result["abs_delta_mAP50-95_pct"] = None
    gate_exit_code = apply_delta_gate(result, args.max_abs_delta_pct, args.max_abs_delta_pp)
    print(json.dumps(result, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if gate_exit_code:
        if args.max_abs_delta_pp is not None:
            print(
                "mAP50-95 absolute delta gate failed: "
                f"observed {result['abs_delta_mAP50-95_pp']:.6g} pp > "
                f"maximum {args.max_abs_delta_pp:.6g} pp",
                file=sys.stderr,
            )
        else:
            print(
                "mAP50-95 relative delta gate failed: "
                f"observed {result['abs_delta_mAP50-95_pct']:.6g}% > "
                f"maximum {args.max_abs_delta_pct:.6g}%",
                file=sys.stderr,
            )
    return gate_exit_code


if __name__ == "__main__":
    raise SystemExit(main())
