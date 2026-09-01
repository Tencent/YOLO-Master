#!/usr/bin/env python3
"""Create and validate a reproducible Issue #51 evidence manifest.

The manifest is intentionally independent of Ultralytics and of optional edge
runtime SDKs.  It records the ordered image set, hashes of the inputs and model
artifacts, the exact post-processing protocol, and the acceptance gates.  This
makes a result bundle auditable without placing large models or datasets in Git.

Typical use::

    python scripts/evidence_manifest.py create \
        --dataset visdrone --split val --images /data/VisDrone/images/val \
        --labels /data/VisDrone/labels/val --predictions artifacts/onnx_txt \
        --model onnx=artifacts/model.onnx --checkpoint runs/best.pt \
        --training-metadata artifacts/training-provenance.json \
        --acceptance --output artifacts/onnx-evidence.json

    python scripts/evidence_manifest.py validate artifacts/onnx-evidence.json \
        --acceptance

The ``--template`` mode is useful before a target machine and its data are
available.  A template is explicitly marked ``status=template`` and cannot be
mistaken for an acceptance result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


# Keep the manifest image universe identical to the portable C++ runner and
# both mAP evaluators.  TIFF files are intentionally excluded because stb's
# decoder used by the runner does not support them.
IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp"})
SCHEMA_VERSION = "issue51-evidence/v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_MODEL_FORMAT_RE = re.compile(r"^(onnx|ncnn|mnn)(?:$|[_.-])", re.IGNORECASE)
ROUTING_SEMANTICS = ("native_sparse", "dense_fallback", "dense_native", "not_applicable")


def sha256_file(path: Path) -> str:
    """Return the SHA256 digest of *path* without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def _normalise_path(path: Path, base: Path) -> str:
    """Return a stable, mount-independent path relative to *base*.

    Evidence records are later joined to a caller-provided verification root.
    Refuse files outside that root instead of emitting ``../`` paths that would
    be non-portable and could escape the verifier's sandbox.
    """
    resolved_path = path.resolve()
    resolved_base = base.resolve()
    try:
        relative = resolved_path.relative_to(resolved_base)
    except ValueError as exc:
        raise ValueError(
            "{} is outside evidence root {}; pass an explicit --image-root or "
            "--calibration-root when using an image list".format(path, base)
        ) from exc
    value = relative.as_posix()
    if _safe_relative_path(value) is None:
        raise ValueError("unsafe relative evidence path: {}".format(value))
    return value


def resolve_image_list(source: Path, root: Optional[Path] = None) -> Tuple[Path, List[Path]]:
    """Resolve a directory, image, or newline-delimited image list.

    List entries are resolved relative to the list file, while directory walks
    are recursive and sorted by their POSIX spelling.  ``root`` can be used to
    select an explicit normalization root when a list file lives outside the
    dataset directory.  Every selected file must remain below that root.
    """
    source = source.expanduser()
    explicit_root = root.expanduser().resolve() if root is not None else None
    if explicit_root is not None and not explicit_root.is_dir():
        raise NotADirectoryError("image normalization root not found: {}".format(explicit_root))

    def finish(base: Path, paths: List[Path]) -> Tuple[Path, List[Path]]:
        base = (explicit_root or base).resolve()
        resolved_paths = [path.resolve() for path in paths]
        for path in resolved_paths:
            try:
                path.relative_to(base)
            except ValueError as exc:
                raise ValueError(
                    "image {} is outside evidence root {}; pass a root containing "
                    "every listed image".format(path, base)
                ) from exc
        return base, resolved_paths

    if source.is_dir():
        paths = sorted(
            (p for p in source.rglob("*") if _is_image(p)),
            key=lambda p: p.as_posix().casefold(),
        )
        return finish(source, paths)
    if _is_image(source):
        return finish(source.parent, [source])
    if not source.is_file():
        raise FileNotFoundError("image source not found: {}".format(source))
    paths: List[Path] = []
    for line_number, raw in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        # Keep list semantics identical across the manifest tool, evaluators
        # and C++ runner.  BOMs are common in lists written on Windows; a
        # surrounding quote pair permits paths containing spaces.
        line = raw.strip()
        if line_number == 1:
            line = line.lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if len(line) >= 2 and line[0] == line[-1] and line[0] in {'"', "'"}:
            line = line[1:-1].strip()
        candidate = Path(line).expanduser()
        if not candidate.is_absolute():
            candidate = source.parent / candidate
        if not _is_image(candidate):
            raise ValueError("image list entry is not a supported image: {}".format(line))
        paths.append(candidate)
    if not paths:
        raise ValueError("image list is empty: {}".format(source))
    return finish(source.parent, paths)


def file_record(path: Path, base: Path) -> Dict[str, object]:
    """Describe a file using only portable metadata and a content hash."""
    if not path.is_file():
        raise FileNotFoundError("file not found: {}".format(path))
    stat = path.stat()
    return {
        "path": _normalise_path(path, base),
        "bytes": stat.st_size,
        "sha256": sha256_file(path),
    }


def collect_records(
    source: Optional[Path], base: Optional[Path] = None,
    suffixes: Optional[Iterable[str]] = None,
) -> List[Dict[str, object]]:
    """Collect deterministic records for a file or a directory tree."""
    if source is None:
        return []
    source = source.expanduser()
    if base is None:
        base = source if source.is_dir() else source.parent
    allowed = {suffix.lower() for suffix in suffixes} if suffixes else None
    if source.is_dir():
        files = sorted(
            (p for p in source.rglob("*") if p.is_file() and (allowed is None or p.suffix.lower() in allowed)),
            key=lambda p: p.as_posix().casefold(),
        )
    elif source.is_file():
        if allowed is not None and source.suffix.lower() not in allowed:
            raise ValueError("artifact path has an unsupported suffix: {}".format(source))
        files = [source]
    else:
        raise FileNotFoundError("artifact path not found: {}".format(source))
    return [file_record(path, base) for path in files]


def _image_records(paths: Sequence[Path], base: Path) -> List[Dict[str, object]]:
    return [file_record(path, base) for path in paths]


def _list_digest(records: Sequence[Dict[str, object]]) -> str:
    payload = "\n".join("{} {}".format(item["path"], item["sha256"]) for item in records)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _valid_digest(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _safe_relative_path(value: object) -> Optional[str]:
    """Return a canonical relative path, or ``None`` for an unsafe value.

    Manifest paths are later joined to user-supplied verification roots.  An
    absolute path or a ``..`` component would let a crafted manifest escape
    that root and would also make the evidence non-portable across hosts.
    """
    if not isinstance(value, str) or not value or "\x00" in value:
        return None
    if "\\" in value:
        return None
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if posix.is_absolute() or windows.is_absolute() or windows.drive:
        return None
    if any(part in ("", ".", "..") for part in posix.parts):
        # ``.`` is not emitted by our writer and accepting it would create
        # multiple spellings for the same artifact.  Empty parts are likewise
        # rejected to keep list digests canonical.
        return None
    return posix.as_posix()


def _validate_list_digest(
    records: Sequence[Dict[str, object]], value: object, label: str, required: bool,
) -> List[str]:
    """Validate a digest over an ordered file-record list."""
    errors: List[str] = []
    if value is None or value == "":
        if required:
            errors.append("{} is required".format(label))
        return errors
    if not _valid_digest(value):
        errors.append("{} must be a 64-character hex digest".format(label))
        return errors
    if any(not isinstance(item, dict) or "path" not in item or "sha256" not in item for item in records):
        # The record validator reports the structural problem; avoid raising a
        # secondary KeyError while trying to calculate a digest for it.
        return errors
    if records and str(value).lower() != _list_digest(records):
        errors.append("{} does not match its file list".format(label))
    elif not records and str(value).lower() != _list_digest(records):
        # A non-null digest on an empty list is still auditable and must be exact.
        errors.append("{} does not match its file list".format(label))
    return errors


def _model_format(name: object, model: Optional[Dict[str, object]] = None) -> Optional[str]:
    """Resolve a model artifact's runtime format from its key or file suffix.

    Release manifests commonly use keys such as ``onnx_fp32`` or
    ``ncnn-int8``.  Requiring the literal key ``onnx`` made otherwise valid
    manifests fail the acceptance gate, so the prefix is treated as the
    canonical format and a suffix is used as a fallback for generic keys.
    """
    match = _MODEL_FORMAT_RE.match(str(name).strip().lower())
    if match:
        return match.group(1).lower()
    if isinstance(model, dict) and isinstance(model.get("files"), list):
        formats = set()
        for item in model["files"]:
            if not isinstance(item, dict):
                continue
            suffix = Path(str(item.get("path", ""))).suffix.lower()
            if suffix == ".onnx":
                formats.add("onnx")
            elif suffix in {".param", ".bin"}:
                formats.add("ncnn")
            elif suffix == ".mnn":
                formats.add("mnn")
        if len(formats) == 1:
            return next(iter(formats))
    return None


def _stem_set(records: Iterable[Dict[str, object]]) -> Tuple[set, List[str]]:
    stems: Dict[str, str] = {}
    duplicates: List[str] = []
    for item in records:
        if not isinstance(item, dict) or not str(item.get("path", "")):
            continue
        stem = Path(str(item["path"])).stem.casefold()
        if stem in stems:
            duplicates.append("{} ({}, {})".format(stem, stems[stem], item["path"]))
        else:
            stems[stem] = str(item["path"])
    return set(stems), duplicates


def _stems(records: Iterable[Dict[str, object]]) -> set:
    """Return case-folded file stems for correspondence checks."""
    return {
        Path(str(item.get("path", ""))).stem.casefold()
        for item in records
        if isinstance(item, dict) and str(item.get("path", ""))
    }


def _validate_records(records: object, label: str) -> List[str]:
    """Validate the portable shape of a file-record list."""
    errors: List[str] = []
    if not isinstance(records, list):
        return ["{}.files must be a list".format(label)]
    seen_paths: Dict[str, int] = {}
    for index, item in enumerate(records):
        if not isinstance(item, dict):
            errors.append("{}[{}] must be an object".format(label, index))
            continue
        raw_path = item.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append("{}[{}].path is required".format(label, index))
        else:
            safe_path = _safe_relative_path(raw_path)
            if safe_path is None:
                errors.append("{}[{}].path must be a relative POSIX path without '..'".format(label, index))
            else:
                key = safe_path.casefold()
                if key in seen_paths:
                    errors.append(
                        "{} contains duplicate path {} (records {} and {})".format(
                            label, raw_path, seen_paths[key], index
                        )
                    )
                else:
                    seen_paths[key] = index
        digest = str(item.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            errors.append("{}[{}].sha256 must be a 64-character hex digest".format(label, index))
        size = item.get("bytes", -1)
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            errors.append("{}[{}].bytes must be non-negative".format(label, index))
    return errors


def _verify_records(
    records: object, root: Optional[Path], label: str,
) -> List[str]:
    """Verify file sizes and SHA256 values when a local root is supplied."""
    if root is None:
        return []
    if not isinstance(records, list):
        return ["{}.files must be a list".format(label)]
    errors: List[str] = []
    root = root.expanduser().resolve()
    for item in records:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("path", "")
        safe_path = _safe_relative_path(raw_path)
        if safe_path is None:
            errors.append("{} has an unsafe relative path: {}".format(label, raw_path))
            continue
        path = (root / PurePosixPath(safe_path)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            errors.append("{} escapes verification root: {}".format(label, raw_path))
            continue
        if not path.is_file():
            errors.append("{} missing: {}".format(label, path))
            continue
        recorded_size = item.get("bytes", -1)
        if not isinstance(recorded_size, int) or isinstance(recorded_size, bool):
            continue
        if recorded_size != path.stat().st_size:
            errors.append("{} size mismatch: {}".format(label, path))
        if str(item.get("sha256", "")).lower() != sha256_file(path):
            errors.append("{} SHA256 mismatch: {}".format(label, path))
    return errors


def _actual_record_hashes(records: object, root: Optional[Path]) -> set:
    """Hash existing records under *root* for the calibration disjoint gate."""
    if root is None or not isinstance(records, list):
        return set()
    hashes = set()
    root = root.expanduser().resolve()
    for item in records:
        if not isinstance(item, dict):
            continue
        safe_path = _safe_relative_path(item.get("path", ""))
        if safe_path is None:
            continue
        path = (root / PurePosixPath(safe_path)).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            continue
        if path.is_file():
            try:
                hashes.add(sha256_file(path).lower())
            except OSError:
                # _verify_records emits the user-facing missing/read error.
                continue
    return hashes


def _git_commit(start: Path) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(start), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return None
    value = completed.stdout.strip()
    return value if completed.returncode == 0 and value else None


def _command_version(command: str) -> Optional[str]:
    """Return the first version line of an installed command, if available."""
    try:
        completed = subprocess.run(
            [command, "--version"], check=False, capture_output=True, text=True,
        )
    except (OSError, UnicodeError):
        return None
    output = (completed.stdout or completed.stderr).splitlines()
    return output[0].strip() if completed.returncode == 0 and output else None


def _parse_named_paths(values: Sequence[str], option: str) -> Dict[str, Path]:
    """Parse repeatable ``NAME=PATH`` arguments with stable diagnostics."""
    models: Dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("{} must use NAME=PATH (got {!r})".format(option, value))
        name, raw_path = value.split("=", 1)
        name = name.strip().lower()
        if not name or not raw_path.strip():
            raise ValueError("{} must use a non-empty NAME=PATH".format(option))
        if name in models:
            raise ValueError("duplicate {} name: {}".format(option.lstrip("-"), name))
        models[name] = Path(raw_path).expanduser()
    return models


def _parse_model_specs(values: Sequence[str]) -> Dict[str, Path]:
    """Parse repeatable exported-model specifications."""
    return _parse_named_paths(values, "--model")


def _environment(repo_root: Path) -> Dict[str, object]:
    return {
        "python": sys.version.split()[0],
        "python_executable": str(Path(sys.executable).resolve()),
        "platform": platform.platform(aliased=True),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "git_commit": _git_commit(repo_root),
        "cmake": _command_version("cmake"),
        "cxx": _command_version(os.environ.get("CXX", "g++")),
    }


def _load_training_metadata(path: Optional[Path]) -> Optional[Dict[str, object]]:
    """Load an optional, JSON-serialisable training provenance record."""
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError("training metadata not found: {}".format(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("unable to read training metadata: {}".format(exc)) from exc
    if not isinstance(payload, dict):
        raise ValueError("training metadata must contain a JSON object")
    return payload


def _default_protocol(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "imgsz": args.imgsz,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
        "multi_label": bool(args.multi_label),
        "letterbox": not args.stretch,
        # Keep the optional small-object sweep in the signed protocol.  A
        # disabled sweep is represented by -1 rather than by omission so a
        # reference and candidate cannot silently use different thresholds.
        "small_conf": getattr(args, "small_conf", -1.0),
        "small_area": getattr(args, "small_area", 32.0 * 32.0),
        "color": "RGB",
        "layout": "NCHW",
        "normalization": "float32 / 255.0",
        "routing_semantics": getattr(args, "routing_semantics", None),
    }


def _empty_template(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "template",
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "image_count": 0,
            "images": [],
            "image_list_sha256": None,
        },
        "protocol": _default_protocol(args),
        "training": None,
        # ``labels`` and ``predictions`` are nullable file collections in the
        # schema.  A bare list is ambiguous (and fails JSON-Schema validation),
        # so an unavailable artifact is represented consistently as null.
        "artifacts": {
            "checkpoint": None,
            "models": {},
            "reports": {},
            "labels": None,
            "predictions": None,
        },
        "calibration": {
            "enabled": bool(args.int8),
            "image_count": 0,
            "images": [],
            "image_list_sha256": None,
            "disjoint_from_validation": None,
        },
        "environment": _environment(args.repo_root),
        "run": {"command": args.command or None},
        "gates": {
            "accuracy_min_images": 500,
            "fp32_max_abs_delta_pp": 0.5,
            "int8_max_abs_delta_pp": 1.0,
            "calibration_min_images": 300,
        },
    }


def build_manifest(args: argparse.Namespace) -> Dict[str, object]:
    if args.template:
        if args.acceptance:
            raise ValueError("--template cannot be combined with --acceptance")
        if args.int8 or args.calibration_images:
            raise ValueError("--template cannot be combined with --int8 or --calibration-images")
        template = _empty_template(args)
        template_errors = validate_manifest(template)
        if template_errors:
            raise ValueError("; ".join(template_errors))
        return template
    if args.images is None:
        raise ValueError("--images is required unless --template is used")
    image_base, image_paths = resolve_image_list(args.images, getattr(args, "image_root", None))
    image_records = _image_records(image_paths, image_base)
    models = _parse_model_specs(args.model)
    model_records: Dict[str, List[Dict[str, object]]] = {}
    for name, path in models.items():
        model_records[name] = collect_records(path)
    reports = _parse_named_paths(getattr(args, "report", []), "--report")
    report_records: Dict[str, List[Dict[str, object]]] = {}
    for name, path in reports.items():
        report_records[name] = collect_records(path)
    checkpoint = file_record(args.checkpoint, args.checkpoint.parent) if args.checkpoint else None
    label_records = collect_records(args.labels, suffixes={".txt"})
    prediction_records = collect_records(args.predictions, suffixes={".txt"})
    calibration_records: List[Dict[str, object]] = []
    calibration_base = None
    if args.calibration_images:
        calibration_base, calibration_paths = resolve_image_list(
            args.calibration_images, getattr(args, "calibration_root", None)
        )
        calibration_records = _image_records(calibration_paths, calibration_base)
    training = _load_training_metadata(getattr(args, "training_metadata", None))

    manifest: Dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "status": "acceptance-candidate" if args.acceptance else "diagnostic",
        "dataset": {
            "name": args.dataset,
            "split": args.split,
            "image_root": image_base.as_posix(),
            "image_count": len(image_records),
            "images": image_records,
            "image_list_sha256": _list_digest(image_records),
        },
        "protocol": _default_protocol(args),
        "training": training,
        "artifacts": {
            "checkpoint": checkpoint,
            "models": {
                name: {"files": records, "sha256": _list_digest(records)}
                for name, records in model_records.items()
            },
            "reports": {
                name: {"files": records, "sha256": _list_digest(records)}
                for name, records in report_records.items()
            },
            "labels": {"files": label_records, "count": len(label_records)} if args.labels else None,
            "predictions": {
                "files": prediction_records,
                "count": len(prediction_records),
            }
            if args.predictions
            else None,
        },
        "calibration": {
            "enabled": bool(args.int8 or args.calibration_images),
            "image_root": calibration_base.as_posix() if calibration_base else None,
            "image_count": len(calibration_records),
            "images": calibration_records,
            "image_list_sha256": _list_digest(calibration_records) if calibration_records else None,
            "disjoint_from_validation": None,
        },
        "environment": _environment(args.repo_root),
        "run": {"command": args.command or None},
        "gates": {
            "accuracy_min_images": 500,
            "fp32_max_abs_delta_pp": 0.5,
            "int8_max_abs_delta_pp": 1.0,
            "calibration_min_images": 300,
        },
    }
    if calibration_records:
        val_hashes = {str(item["sha256"]) for item in image_records}
        overlap = bool(val_hashes.intersection(str(item["sha256"]) for item in calibration_records))
        manifest["calibration"]["disjoint_from_validation"] = not overlap  # type: ignore[index]
        if overlap:
            raise ValueError("calibration images overlap the validation set by SHA256")
    elif args.int8:
        raise ValueError("--int8 requires --calibration-images and at least 300 images")
    validation_errors = validate_manifest(manifest, acceptance=args.acceptance)
    if validation_errors:
        raise ValueError("; ".join(validation_errors))
    return manifest


def validate_manifest(manifest: Dict[str, object], acceptance: bool = False) -> List[str]:
    """Return human-readable schema/gate violations (empty means valid)."""
    errors: List[str] = []
    if not isinstance(manifest, dict):
        return ["manifest must be an object"]
    if manifest.get("schema_version") != SCHEMA_VERSION:
        errors.append("unsupported schema_version")
    status = manifest.get("status")
    if status is not None and status not in {"template", "diagnostic", "acceptance-candidate"}:
        errors.append("unsupported status")
    if acceptance and status != "acceptance-candidate":
        errors.append("acceptance validation requires status=acceptance-candidate")
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict):
        return ["dataset must be an object"]
    if "name" in dataset and dataset.get("name") not in {"visdrone", "sku110k"}:
        errors.append("dataset.name must be visdrone or sku110k")
    if "split" in dataset and (
        not isinstance(dataset.get("split"), str) or not dataset.get("split")
    ):
        errors.append("dataset.split must be a non-empty string")
    images = dataset.get("images")
    if not isinstance(images, list):
        errors.append("dataset.images must be a list")
        images = []
    image_count = dataset.get("image_count")
    if isinstance(image_count, bool) or not isinstance(image_count, int) or image_count < 0:
        errors.append("dataset.image_count must be a non-negative integer")
    if dataset.get("image_count") != len(images):
        errors.append("dataset.image_count does not match dataset.images")
    record_errors = _validate_records(images, "dataset.images")
    errors.extend(record_errors)
    listed_digest = dataset.get("image_list_sha256")
    # A non-empty image list must always carry its ordered-list digest.  Empty
    # templates remain valid with a null digest, while acceptance manifests
    # are required to provide the field even before the image-floor check.
    if isinstance(images, list):
        errors.extend(_validate_list_digest(
            images, listed_digest, "dataset.image_list_sha256", required=acceptance or bool(images)
        ))
    _, duplicate_stems = _stem_set(images)
    if duplicate_stems:
        errors.append("duplicate image stems: " + ", ".join(duplicate_stems[:3]))
    if acceptance and len(images) < 500:
        errors.append("Issue #51 acceptance requires at least 500 validation images")
    protocol = manifest.get("protocol")
    if isinstance(protocol, dict):
        required_protocol = (
            "imgsz", "conf", "iou", "max_det", "multi_label", "letterbox",
        )
        if acceptance:
            required_protocol = required_protocol + (
                "small_conf", "small_area", "routing_semantics",
            )
        if acceptance:
            for key in required_protocol:
                if key not in protocol:
                    errors.append("acceptance protocol is missing {}".format(key))
        for key in ("imgsz", "max_det"):
            try:
                if isinstance(protocol.get(key), bool):
                    raise ValueError
                if int(protocol.get(key, 0)) <= 0:
                    errors.append("protocol.{} must be positive".format(key))
            except (TypeError, ValueError):
                errors.append("protocol.{} must be numeric".format(key))
        for key in ("conf", "iou"):
            try:
                if isinstance(protocol.get(key), bool):
                    raise ValueError
                value = float(protocol.get(key, -1))
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    errors.append("protocol.{} must be in [0, 1]".format(key))
            except (TypeError, ValueError):
                errors.append("protocol.{} must be numeric".format(key))
        if "small_conf" in protocol:
            try:
                if isinstance(protocol.get("small_conf"), bool):
                    raise ValueError
                small_conf = float(protocol.get("small_conf"))
                if not math.isfinite(small_conf) or not -1.0 <= small_conf <= 1.0:
                    errors.append("protocol.small_conf must be in [-1, 1]")
            except (TypeError, ValueError):
                errors.append("protocol.small_conf must be numeric")
        if "small_area" in protocol:
            try:
                if isinstance(protocol.get("small_area"), bool):
                    raise ValueError
                small_area = float(protocol.get("small_area"))
                if not math.isfinite(small_area) or small_area < 0.0:
                    errors.append("protocol.small_area must be non-negative")
            except (TypeError, ValueError):
                errors.append("protocol.small_area must be numeric")
        for key in ("multi_label", "letterbox"):
            if key in protocol and not isinstance(protocol.get(key), bool):
                errors.append("protocol.{} must be boolean".format(key))
        if acceptance and protocol.get("letterbox") is not True:
            errors.append("Issue #51 acceptance requires aspect-preserving letterbox preprocessing")
        routing_semantics = protocol.get("routing_semantics")
        if routing_semantics is not None and routing_semantics not in ROUTING_SEMANTICS:
            errors.append(
                "protocol.routing_semantics must be one of {}".format(
                    ", ".join(ROUTING_SEMANTICS)
                )
            )
        if acceptance and routing_semantics is None:
            errors.append(
                "acceptance protocol.routing_semantics is required; "
                "use dense_fallback for static exports"
            )
    elif acceptance:
        errors.append("protocol must be an object")
    training = manifest.get("training")
    if acceptance and not isinstance(training, dict):
        errors.append("acceptance manifest requires training provenance")
    if training is not None:
        if not isinstance(training, dict):
            errors.append("training must be an object or null")
        else:
            for key in ("epochs", "batch_size"):
                if key in training:
                    value = training[key]
                    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                        errors.append("training.{} must be a positive integer".format(key))
            if "seed" in training:
                seed = training["seed"]
                if isinstance(seed, bool) or not isinstance(seed, int):
                    errors.append("training.seed must be an integer")
            for key in ("base_model", "dataset_version", "optimizer", "lr_schedule", "command"):
                if key in training and training[key] is not None and not isinstance(training[key], str):
                    errors.append("training.{} must be a string or null".format(key))
            if acceptance:
                for key in ("base_model", "dataset_version", "command"):
                    if not isinstance(training.get(key), str) or not training[key].strip():
                        errors.append("acceptance training.{} must be a non-empty string".format(key))
                if isinstance(training.get("epochs"), bool) or not isinstance(training.get("epochs"), int):
                    errors.append("acceptance training.epochs is required")
                if isinstance(training.get("seed"), bool) or not isinstance(training.get("seed"), int):
                    errors.append("acceptance training.seed is required")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        errors.append("artifacts must be an object")
        artifacts = {}
    checkpoint = artifacts.get("checkpoint")
    if acceptance and not checkpoint:
        errors.append("acceptance manifest requires a checkpoint record")
    elif checkpoint is not None:
        if isinstance(checkpoint, dict):
            errors.extend(_validate_records([checkpoint], "artifacts.checkpoint"))
        else:
            errors.append("artifacts.checkpoint must be an object or null")
    models = artifacts.get("models")
    if acceptance and (not isinstance(models, dict) or not models):
        errors.append("acceptance manifest requires at least one exported model")
    if acceptance and isinstance(models, dict):
        model_formats = {
            resolved for name, model in models.items()
            if (resolved := _model_format(name, model)) is not None
        }
        if "onnx" not in model_formats:
            errors.append("Issue #51 acceptance requires an ONNX export")
        if not model_formats.intersection({"ncnn", "mnn"}):
            errors.append("Issue #51 acceptance requires an NCNN or MNN export")
    elif models is not None and not isinstance(models, dict):
        errors.append("artifacts.models must be an object")
    if isinstance(models, dict):
        for name, model in models.items():
            if not isinstance(model, dict):
                errors.append("artifacts.models.{} must be an object".format(name))
                continue
            model_label = "artifacts.models.{}".format(name)
            files = model.get("files")
            model_record_errors = _validate_records(files, model_label + ".files")
            errors.extend(model_record_errors)
            if acceptance and isinstance(files, list) and not files:
                errors.append(model_label + ".files must contain at least one artifact")
            digest = model.get("sha256")
            if acceptance:
                if not _valid_digest(digest):
                    errors.append(model_label + ".sha256 must be a 64-character hex digest")
                elif isinstance(files, list) and not model_record_errors and digest.lower() != _list_digest(files):
                    errors.append(model_label + ".sha256 does not match files")
            if acceptance:
                model_format = _model_format(name, model)
                if model_format == "ncnn" and isinstance(files, list):
                    suffixes = {Path(str(item.get("path", ""))).suffix.lower()
                                for item in files if isinstance(item, dict)}
                    if not {".param", ".bin"}.issubset(suffixes):
                        errors.append(model_label + " NCNN export requires both .param and .bin files")
            elif digest is not None:
                if not _valid_digest(digest):
                    errors.append(model_label + ".sha256 must be a 64-character hex digest")
                elif isinstance(files, list) and not model_record_errors and digest.lower() != _list_digest(files):
                    errors.append(model_label + ".sha256 does not match files")
    reports = artifacts.get("reports", {})
    if reports is None:
        reports = {}
    if not isinstance(reports, dict):
        errors.append("artifacts.reports must be an object")
        reports = {}
    if acceptance and not reports:
        errors.append("acceptance manifest requires at least one metric or benchmark report")
    if isinstance(reports, dict):
        for name, report in reports.items():
            report_label = "artifacts.reports.{}".format(name)
            if not isinstance(name, str) or not name.strip():
                errors.append("artifacts.reports names must be non-empty strings")
            if not isinstance(report, dict):
                errors.append(report_label + " must be an object")
                continue
            files = report.get("files")
            report_record_errors = _validate_records(files, report_label + ".files")
            errors.extend(report_record_errors)
            if acceptance and isinstance(files, list) and not files:
                errors.append(report_label + ".files must contain at least one artifact")
            digest = report.get("sha256")
            if not _valid_digest(digest):
                errors.append(report_label + ".sha256 must be a 64-character hex digest")
            elif isinstance(files, list) and not report_record_errors and digest.lower() != _list_digest(files):
                errors.append(report_label + ".sha256 does not match files")
    for key in ("labels", "predictions"):
        value = artifacts.get(key)
        if isinstance(value, dict):
            errors.extend(_validate_records(value.get("files"), "artifacts.{}.files".format(key)))
            count = value.get("count")
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                errors.append("artifacts.{}.count must be a non-negative integer".format(key))
            if count != len(value.get("files", [])):
                errors.append("artifacts.{}.count does not match files".format(key))
        elif acceptance:
            errors.append("acceptance manifest requires artifacts.{}".format(key))
        elif value is not None:
            errors.append("artifacts.{} must be an object or null".format(key))
    if acceptance and isinstance(artifacts.get("labels"), dict):
        if artifacts["labels"].get("count") != len(images):
            errors.append("acceptance requires one label record per validation image")
        elif _stems(artifacts["labels"].get("files", [])) != _stems(images):
            errors.append("label records do not correspond one-to-one with validation images")
    if acceptance and isinstance(artifacts.get("predictions"), dict):
        if artifacts["predictions"].get("count") != len(images):
            errors.append("acceptance requires one prediction record per validation image")
        elif _stems(artifacts["predictions"].get("files", [])) != _stems(images):
            errors.append("prediction records do not correspond one-to-one with validation images")
    calibration = manifest.get("calibration")
    if not isinstance(calibration, dict):
        errors.append("calibration must be an object")
        calibration = {}
    enabled = calibration.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        errors.append("calibration.enabled must be boolean")
    if acceptance and enabled is None:
        errors.append("acceptance calibration.enabled is required")
    if enabled is True:
        cal_images = calibration.get("images")
        if not isinstance(cal_images, list):
            errors.append("calibration.images must be a list")
            cal_images = []
        if calibration.get("image_count") != len(cal_images):
            errors.append("calibration.image_count does not match calibration.images")
        if (isinstance(calibration.get("image_count"), bool)
                or not isinstance(calibration.get("image_count"), int)
                or calibration.get("image_count", -1) < 0):
            errors.append("calibration.image_count must be a non-negative integer")
        if len(cal_images) < 300:
            errors.append("INT8 calibration requires at least 300 images")
        _, cal_duplicates = _stem_set(cal_images)
        if cal_duplicates:
            errors.append("duplicate calibration stems: " + ", ".join(cal_duplicates[:3]))
        disjoint = calibration.get("disjoint_from_validation")
        if acceptance and disjoint is not True:
            errors.append("acceptance requires calibration.disjoint_from_validation=true")
        elif disjoint not in (None, True, False):
            errors.append("calibration.disjoint_from_validation must be boolean or null")
        cal_record_errors = _validate_records(cal_images, "calibration.images")
        errors.extend(cal_record_errors)
        if isinstance(images, list) and isinstance(cal_images, list) and not record_errors and not cal_record_errors:
            overlap = {str(item["sha256"]).lower() for item in images}.intersection(
                str(item["sha256"]).lower() for item in cal_images
            )
            if overlap:
                errors.append("calibration set overlaps validation set by SHA256")
        errors.extend(_validate_list_digest(
            cal_images, calibration.get("image_list_sha256"),
            "calibration.image_list_sha256", required=acceptance or bool(cal_images)
        ))
    elif enabled is False:
        if calibration.get("images") not in (None, []):
            errors.append("calibration.images must be empty when calibration is disabled")
        if calibration.get("image_count", 0) not in (0, None):
            errors.append("calibration.image_count must be zero when calibration is disabled")
        if calibration.get("image_list_sha256") not in (None, ""):
            errors.append("calibration.image_list_sha256 must be null when calibration is disabled")
    elif calibration.get("images") not in (None, []):
        errors.append("calibration.images must be empty when calibration is disabled")
    environment = manifest.get("environment")
    if acceptance:
        if not isinstance(environment, dict):
            errors.append("acceptance manifest requires environment metadata")
        else:
            for key in ("python", "platform", "machine", "git_commit"):
                if not isinstance(environment.get(key), str) or not environment[key].strip():
                    errors.append("acceptance environment.{} must be a non-empty string".format(key))
        run = manifest.get("run")
        if not isinstance(run, dict):
            errors.append("acceptance manifest requires run metadata")
        elif not isinstance(run.get("command"), str) or not run["command"].strip():
            errors.append("acceptance run.command must be a non-empty string")
    return errors


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    path.write_text(canonical, encoding="utf-8")


def _create_command(args: argparse.Namespace) -> int:
    manifest = build_manifest(args)
    manifest["generated_at_utc"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    _write_json(args.output, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "output": str(args.output),
        "images": manifest["dataset"]["image_count"],  # type: ignore[index]
        "manifest_sha256": sha256_file(args.output),
    }, indent=2))
    return 0


def _validate_command(args: argparse.Namespace) -> int:
    if not args.manifest.is_file():
        raise FileNotFoundError("manifest not found: {}".format(args.manifest))
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(payload, acceptance=args.acceptance)
    if errors:
        for error in errors:
            print("[invalid] " + error, file=sys.stderr)
        return 2
    print(json.dumps({"status": "valid", "manifest": str(args.manifest)}, indent=2))
    return 0


def _verify_command(args: argparse.Namespace) -> int:
    if not args.manifest.is_file():
        raise FileNotFoundError("manifest not found: {}".format(args.manifest))
    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    errors = validate_manifest(payload, acceptance=args.acceptance)
    dataset = payload.get("dataset", {})
    artifacts = payload.get("artifacts", {})
    calibration = payload.get("calibration", {})
    if isinstance(dataset, dict):
        errors.extend(_verify_records(dataset.get("images"), args.images_root, "dataset.images"))
    if isinstance(artifacts, dict):
        errors.extend(_verify_records([artifacts.get("checkpoint")], args.checkpoint_root, "checkpoint"))
        models = artifacts.get("models")
        if isinstance(models, dict):
            for name, model in models.items():
                if isinstance(model, dict):
                    errors.extend(_verify_records(model.get("files"), args.models_root, "model.{}".format(name)))
        reports = artifacts.get("reports")
        if isinstance(reports, dict):
            for name, report in reports.items():
                if isinstance(report, dict):
                    errors.extend(_verify_records(
                        report.get("files"), getattr(args, "reports_root", None),
                        "report.{}".format(name),
                    ))
        if isinstance(artifacts.get("labels"), dict):
            errors.extend(_verify_records(artifacts["labels"].get("files"), args.labels_root, "labels"))
        if isinstance(artifacts.get("predictions"), dict):
            errors.extend(_verify_records(artifacts["predictions"].get("files"), args.predictions_root, "predictions"))
    if isinstance(calibration, dict):
        errors.extend(_verify_records(
            calibration.get("images"), getattr(args, "calibration_root", None), "calibration.images"
        ))
        # Recompute the split intersection from the supplied roots.  Checking
        # only the boolean recorded in JSON would allow an edited manifest to
        # bypass the INT8 calibration/validation separation gate.
        images_root = getattr(args, "images_root", None)
        calibration_root = getattr(args, "calibration_root", None)
        if isinstance(dataset, dict) and images_root is not None and calibration_root is not None:
            validation_hashes = _actual_record_hashes(dataset.get("images"), images_root)
            calibration_hashes = _actual_record_hashes(calibration.get("images"), calibration_root)
            if validation_hashes.intersection(calibration_hashes):
                errors.append("calibration set overlaps validation set by SHA256 (verified roots)")
    if errors:
        for error in errors:
            print("[invalid] " + error, file=sys.stderr)
        return 2
    print(json.dumps({"status": "verified", "manifest": str(args.manifest)}, indent=2))
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    create = sub.add_parser("create", help="create a manifest")
    create.add_argument("--images", type=Path, help="validation image directory, image, or list file")
    create.add_argument(
        "--image-root", type=Path,
        help=(
            "explicit root used to normalise validation image paths; required when "
            "a list file is outside the image tree"
        ),
    )
    create.add_argument("--labels", type=Path, help="converted YOLO label directory or file")
    create.add_argument("--predictions", type=Path, help="prediction directory or file")
    create.add_argument(
        "--report", action="append", default=[], metavar="NAME=PATH",
        help="mAP/benchmark or other report artifact (repeatable)",
    )
    create.add_argument("--checkpoint", type=Path, help="PyTorch checkpoint")
    create.add_argument(
        "--training-metadata", type=Path,
        help="training provenance JSON (required with --acceptance)",
    )
    create.add_argument("--model", action="append", default=[], metavar="NAME=PATH", help="exported model (repeatable)")
    create.add_argument("--calibration-images", type=Path, help="training-only calibration image directory/list")
    create.add_argument(
        "--calibration-root", type=Path,
        help="explicit root used to normalise calibration image paths",
    )
    create.add_argument("--dataset", choices=("visdrone", "sku110k"), default="visdrone")
    create.add_argument("--split", default="val")
    create.add_argument("--imgsz", type=int, default=640)
    create.add_argument("--conf", type=float, default=0.001)
    create.add_argument("--iou", type=float, default=0.70)
    create.add_argument("--max-det", type=int, default=300)
    create.add_argument(
        "--small-conf", type=float, default=-1.0,
        help="optional lower confidence for boxes below --small-area (-1 disables)",
    )
    create.add_argument(
        "--small-area", type=float, default=32.0 * 32.0,
        help="original-image area threshold for --small-conf (default: 1024)",
    )
    label_group = create.add_mutually_exclusive_group()
    label_group.add_argument(
        "--multi-label", dest="multi_label", action="store_true", default=True,
        help="record one detection per class and anchor (default)",
    )
    label_group.add_argument(
        "--single-label", dest="multi_label", action="store_false",
        help="record argmax-per-anchor decoding (diagnostic runs only)",
    )
    create.add_argument("--stretch", action="store_true", help="record stretch preprocessing instead of letterbox")
    create.add_argument(
        "--routing-semantics", choices=ROUTING_SEMANTICS,
        help=(
            "EsMoE inference path (required with --acceptance): native_sparse, "
            "dense_fallback, dense_native, or not_applicable"
        ),
    )
    create.add_argument("--int8", action="store_true", help="enable the >=300-image calibration gate")
    create.add_argument("--acceptance", action="store_true", help="enforce the Issue #51 evidence floor")
    create.add_argument("--command", help="exact command used for the run")
    create.add_argument("--repo-root", type=Path, default=Path.cwd())
    create.add_argument("--template", action="store_true", help="write an explicitly non-acceptance template")
    create.add_argument("--output", type=Path, required=True)
    create.set_defaults(func=_create_command)
    validate = sub.add_parser("validate", help="validate an existing manifest")
    validate.add_argument("manifest", type=Path)
    validate.add_argument("--acceptance", action="store_true")
    validate.set_defaults(func=_validate_command)
    verify = sub.add_parser("verify", help="validate and optionally verify file hashes")
    verify.add_argument("manifest", type=Path)
    verify.add_argument("--acceptance", action="store_true")
    verify.add_argument("--images-root", type=Path)
    verify.add_argument("--labels-root", type=Path)
    verify.add_argument("--predictions-root", type=Path)
    verify.add_argument("--models-root", type=Path)
    verify.add_argument("--reports-root", type=Path)
    verify.add_argument("--checkpoint-root", type=Path)
    verify.add_argument("--calibration-root", type=Path)
    verify.set_defaults(func=_verify_command)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
