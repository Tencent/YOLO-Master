#!/usr/bin/env python3
"""Standalone mAP50 / mAP50-95 — numpy only (no ultralytics/torch/cv2/PIL required).

Scores per-image prediction txts ('class conf x1 y1 x2 y2', pixel xyxy) against
YOLO-format labels, replicating the matching + AP integration used by
``eval_map.py``.  The macro average uses a fixed class profile (VisDrone's ten
classes by default), including classes absent from a particular split.  Runs
on-device (e.g. Jetson) where preds+labels+images already live.

  python3 eval_map_standalone.py --preds preds_fp16 --images images/val --labels labels/val
  python3 eval_map_standalone.py --preds preds --images images/val --labels labels/val \
      --profile sku110k --nc 1 --smoke
"""
import argparse, hashlib, json, os, struct
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
CLASS_PROFILES = {"visdrone": 10, "sku110k": 1}
ROUTING_SEMANTICS = ("native_sparse", "dense_fallback", "dense_native", "not_applicable")
PROFILE_PROTOCOLS = {
    "visdrone": {
        "imgsz": 640,
        "conf": 0.001,
        "iou": 0.70,
        "max_det": 300,
        "multi_label": True,
        "letterbox": True,
        "small_conf": -1.0,
        "small_area": 32.0 * 32.0,
        "color": "RGB",
        "layout": "NCHW",
    },
    "sku110k": {
        "imgsz": 1280,
        "conf": 0.25,
        "iou": 0.60,
        "max_det": 300,
        "multi_label": True,
        "letterbox": True,
        "small_conf": -1.0,
        "small_area": 32.0 * 32.0,
        "color": "RGB",
        "layout": "NCHW",
    },
}
# Backward-compatible alias for callers that imported the VisDrone defaults.
PROTOCOL = PROFILE_PROTOCOLS["visdrone"]

_trapz = getattr(np, "trapezoid", None) or np.trapz   # numpy>=2.0 renamed trapz->trapezoid


def jpeg_size(path):
    """(w, h) from a JPEG header — pure python, no image libs."""
    with open(path, "rb") as f:
        f.read(2)  # SOI
        while True:
            b = f.read(1)
            while b and b != b"\xff":
                b = f.read(1)
            marker = f.read(1)
            while marker == b"\xff":
                marker = f.read(1)
            m = marker[0]
            if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
                f.read(3)                      # len(2)+precision(1)
                h, w = struct.unpack(">HH", f.read(4))
                return w, h
            else:
                (seg_len,) = struct.unpack(">H", f.read(2))
                f.seek(seg_len - 2, 1)


def image_size(path):
    """Read dimensions for the formats accepted by the portable runner."""
    suffix = os.path.splitext(path)[1].lower()
    if suffix in (".jpg", ".jpeg"):
        width, height = jpeg_size(path)
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive: {}".format(path))
        return width, height
    with open(path, "rb") as f:
        header = f.read(32)
    if suffix == ".png" and header[:8] == b"\x89PNG\r\n\x1a\n" and len(header) >= 24:
        w, h = struct.unpack(">II", header[16:24])
        if w <= 0 or h <= 0:
            raise ValueError("image dimensions must be positive: {}".format(path))
        return int(w), int(h)
    if suffix == ".bmp" and header[:2] == b"BM" and len(header) >= 26:
        w, h = struct.unpack("<ii", header[18:26])
        width, height = abs(int(w)), abs(int(h))
        if width <= 0 or height <= 0:
            raise ValueError("image dimensions must be positive: {}".format(path))
        return width, height
    raise ValueError("unsupported or corrupt image: {}".format(path))


def _empty_gt():
    return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=int)


def _empty_pred():
    return np.zeros((0, 4), dtype=float), np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)


def _read_label_lines(path):
    """Read non-empty, non-comment rows while preserving line numbers."""
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as handle:
            return [(line_no, line.strip()) for line_no, line in enumerate(handle, 1)
                    if line.strip() and not line.lstrip().startswith("#")]
    except UnicodeDecodeError as exc:
        raise ValueError("{}: labels must be UTF-8 text ({})".format(path, exc)) from exc
    except OSError as exc:
        raise ValueError("{}: unable to read labels ({})".format(path, exc)) from exc


def _parse_numeric(fields, path, line_no, expected, kind):
    if len(fields) != expected:
        raise ValueError("{}:{}: {} expects exactly {} columns, got {}".format(
            path, line_no, kind, expected, len(fields)))
    try:
        values = np.asarray([float(value) for value in fields], dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("{}:{}: {} contains a non-numeric value".format(
            path, line_no, kind)) from exc
    if not np.isfinite(values).all():
        raise ValueError("{}:{}: {} contains NaN or Inf".format(path, line_no, kind))
    return values


def _class_id(value, path, line_no, num_classes, kind):
    if value != np.floor(value):
        raise ValueError("{}:{}: {} class id must be an integer, got {!r}".format(
            path, line_no, kind, value))
    cls = int(value)
    if cls < 0 or cls >= num_classes:
        raise ValueError("{}:{}: {} class id {} outside [0, {})".format(
            path, line_no, kind, cls, num_classes))
    return cls


def _resolve_num_classes(num_classes=None, nc=None):
    if nc is not None:
        if num_classes is not None and int(num_classes) != int(nc):
            raise ValueError("num_classes and nc disagree")
        num_classes = nc
    if num_classes is None:
        num_classes = CLASS_PROFILES["visdrone"]
    try:
        numeric = float(num_classes)
        if not np.isfinite(numeric) or numeric != np.floor(numeric):
            raise ValueError
        num_classes = int(numeric)
    except (TypeError, ValueError) as exc:
        raise ValueError("num_classes must be a positive integer") from exc
    if num_classes < 1:
        raise ValueError("num_classes must be a positive integer")
    return num_classes


def _prediction_files(directory, kind):
    """Index per-image text files by case-folded stem and reject duplicates."""
    if not os.path.isdir(directory):
        raise ValueError("{} directory not found: {}".format(kind, directory))
    result = {}
    paths = sorted(
        (os.path.join(root, name)
         for root, _, names in os.walk(directory)
         for name in names if name.lower().endswith(".txt")),
        key=lambda path: path.replace(os.sep, "/").casefold(),
    )
    for path in paths:
        stem = os.path.splitext(os.path.basename(path))[0].casefold()
        if not stem:
            raise ValueError("{} file has an empty stem: {}".format(kind, path))
        if stem in result:
            raise ValueError("{} stems are not unique: {}".format(kind, stem))
        result[stem] = path
    return result


def _resolve_images(source, root=None):
    """Resolve images below one root used for portable evidence paths."""
    source = os.path.expanduser(os.fspath(source))
    explicit_root = None if root is None else os.path.realpath(os.path.expanduser(os.fspath(root)))
    if explicit_root is not None and not os.path.isdir(explicit_root):
        raise ValueError("image normalization root not found: {}".format(explicit_root))

    def finish(default_root, paths):
        image_root = explicit_root or os.path.realpath(default_root)
        resolved_paths = [os.path.realpath(path) for path in paths]
        for path in resolved_paths:
            try:
                inside = os.path.commonpath([path, image_root]) == image_root
            except ValueError:
                inside = False
            if not inside:
                raise ValueError(
                    "image {} is outside evaluation root {}; pass --image-root "
                    "containing every listed image".format(path, image_root)
                )
        return image_root, resolved_paths

    if os.path.isdir(source):
        paths = sorted(
            (
                os.path.join(root, name)
                for root, _, names in os.walk(source)
                for name in names if os.path.splitext(name)[1].lower() in IMAGE_EXTS
            ),
            key=lambda path: path.replace(os.sep, "/").casefold(),
        )
        return finish(source, paths)
    if not os.path.isfile(source):
        raise ValueError("image source not found: {}".format(source))
    base = os.path.dirname(os.path.abspath(source))
    paths = []
    try:
        with open(source, encoding="utf-8") as handle:
            rows = list(enumerate(handle, 1))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError("unable to read image list {}: {}".format(source, exc)) from exc
    for line_no, raw in rows:
        # Match the C++ runner and the Ultralytics evaluator: tolerate a BOM
        # from Windows text editors and preserve quoted paths with spaces.
        line = raw.strip()
        if line_no == 1:
            line = line.lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if len(line) >= 2 and line[0] == line[-1] and line[0] in {'"', "'"}:
            line = line[1:-1].strip()
        path = os.path.expanduser(line)
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        path = os.path.abspath(path)
        if (not os.path.isfile(path)
                or os.path.splitext(path)[1].lower() not in IMAGE_EXTS):
            raise ValueError("{}:{}: unsupported or missing image: {}".format(source, line_no, line))
        paths.append(path)
    if not paths:
        raise ValueError("image list is empty: {}".format(source))
    return finish(base, paths)


def _image_manifest(images, root):
    """Return the canonical ordered image names and their SHA256 digest."""
    root = os.path.realpath(root)
    names = [os.path.relpath(os.path.realpath(path), root).replace(os.sep, "/") for path in images]
    payload = "\n".join(names) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), names


def _sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _image_content_manifest(images, root, names=None):
    """Digest the ordered image names and bytes used by the metric run."""
    if names is None:
        names = _image_manifest(images, root)[1]
    payload = "\n".join(
        "{} {}".format(name, _sha256_file(path))
        for path, name in zip(images, names)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reference_map(payload):
    """Extract mAP50-95 from common Ultralytics result JSON layouts."""
    if not isinstance(payload, dict):
        raise ValueError("reference JSON must contain an object")
    mappings = [payload]
    for key in ("metrics", "results", "results_dict", "metrics_dict"):
        value = payload.get(key)
        if isinstance(value, dict):
            mappings.append(value)
    keys = {"map50-95", "map50-95(b)", "metrics/map50-95", "metrics/map50-95(b)"}
    for mapping in mappings:
        for key, value in mapping.items():
            if str(key).lower().replace(" ", "") in keys:
                try:
                    return float(value)
                except (TypeError, ValueError) as exc:
                    raise ValueError("reference mAP50-95 must be numeric") from exc
    raise ValueError("reference JSON must contain mAP50-95")


def _reference_metadata_errors(
    reference, image_manifest_sha256, image_content_manifest_sha256,
    image_count, profile, num_classes, protocol, label_format="yolo"
):
    """Return mismatches that would invalidate a cross-backend comparison."""
    if not isinstance(reference, dict):
        return ["reference JSON must contain an object"]
    errors = []
    if reference.get("image_manifest_sha256") is None:
        errors.append("reference JSON is missing image_manifest_sha256")
    elif str(reference["image_manifest_sha256"]).lower() != image_manifest_sha256.lower():
        errors.append("reference image_manifest_sha256 does not match the candidate image list")
    reference_content = reference.get("image_list_sha256") or reference.get("image_content_manifest_sha256")
    if reference_content is None:
        errors.append("reference JSON is missing image_list_sha256")
    elif str(reference_content).lower() != str(image_content_manifest_sha256).lower():
        errors.append("reference image_list_sha256 does not match the candidate images")
    if reference.get("class_profile") is None:
        errors.append("reference JSON is missing class_profile")
    elif reference.get("class_profile") != profile:
        errors.append("reference class_profile does not match the candidate")
    try:
        if int(reference.get("classes")) != int(num_classes):
            errors.append("reference class count does not match the candidate")
    except (TypeError, ValueError):
        errors.append("reference JSON is missing a valid classes field")
    try:
        if int(reference.get("images")) != int(image_count):
            errors.append("reference image count does not match the candidate")
    except (TypeError, ValueError):
        errors.append("reference JSON is missing a valid images field")
    if reference.get("label_format") is not None and reference.get("label_format") != label_format:
        errors.append("reference label_format does not match the candidate")
    ref_protocol = reference.get("protocol")
    if not isinstance(ref_protocol, dict):
        errors.append("reference JSON is missing protocol metadata")
    else:
        for key in ("imgsz", "max_det", "multi_label", "letterbox", "color", "layout"):
            if key not in ref_protocol:
                errors.append("reference protocol is missing {}".format(key))
            elif ref_protocol[key] != protocol.get(key):
                errors.append("reference protocol.{} does not match the candidate".format(key))
        for key in ("conf", "iou", "small_conf", "small_area"):
            if key not in ref_protocol:
                errors.append("reference protocol is missing {}".format(key))
                continue
            try:
                if abs(float(ref_protocol[key]) - float(protocol.get(key))) > 1e-9:
                    errors.append("reference protocol.{} does not match the candidate".format(key))
            except (TypeError, ValueError):
                errors.append("reference protocol.{} is not numeric".format(key))
        ref_routing = ref_protocol.get("routing_semantics")
        cur_routing = protocol.get("routing_semantics")
        if ref_routing is not None or cur_routing is not None:
            if ref_routing is None:
                errors.append("reference protocol is missing routing_semantics")
            elif cur_routing is None:
                errors.append("candidate protocol is missing routing_semantics")
            elif ref_routing != cur_routing:
                errors.append("reference protocol.routing_semantics does not match the candidate")
    return errors


def load_gt(path, w, h, label_format="yolo", num_classes=None, nc=None):
    """Load YOLO or native VisDrone labels with strict, contextual validation.

    Missing and empty files represent an image without annotations and return
    empty arrays.  Existing malformed rows raise ``ValueError`` naming the
    source path and one-based line number; silently dropping a bad annotation
    would make an acceptance metric irreproducible.
    """
    if not np.isfinite([w, h]).all() or float(w) <= 0 or float(h) <= 0:
        raise ValueError("image dimensions must be finite and positive")
    if label_format not in ("yolo", "visdrone", "auto"):
        raise ValueError("label_format must be yolo, visdrone, or auto")
    num_classes = _resolve_num_classes(num_classes, nc)
    rows = _read_label_lines(path)
    if not rows:
        return _empty_gt()

    first_line = rows[0][1]
    # Native VisDrone annotations are commonly comma-separated, but several
    # conversion pipelines emit the same eight fields separated by spaces.
    # In auto mode, distinguish that form from the five-column normalized YOLO
    # dialect without weakening the strict row validation below.
    first_fields = first_line.split()
    use_visdrone = label_format == "visdrone" or (
        label_format == "auto" and ("," in first_line or len(first_fields) == 8)
    )
    visdrone_comma = "," in first_line
    boxes, classes = [], []
    for line_no, line in rows:
        if use_visdrone:
            if visdrone_comma:
                if "," not in line:
                    raise ValueError(
                        "{}:{}: mixed label formats; expected comma-separated VisDrone row".format(
                            path, line_no))
                fields = [field.strip() for field in line.split(",")]
            else:
                if "," in line:
                    raise ValueError(
                        "{}:{}: mixed label formats; expected whitespace-separated VisDrone row".format(
                            path, line_no))
                fields = line.split()
            values = _parse_numeric(fields, path, line_no, 8, "VisDrone ground truth")
            x, y, bw, bh, score, category = values[:6]
            if bw <= 0 or bh <= 0:
                raise ValueError("{}:{}: VisDrone ground-truth width and height must be positive".format(
                    path, line_no))
            if x < 0 or y < 0:
                raise ValueError("{}:{}: VisDrone ground-truth x/y must be non-negative".format(
                    path, line_no))
            if not 0.0 <= score <= 1.0:
                raise ValueError("{}:{}: VisDrone score must be in [0, 1]".format(path, line_no))
            if category != np.floor(category):
                raise ValueError("{}:{}: VisDrone category must be an integer".format(path, line_no))
            category = int(category)
            if category < 0 or category > 11:
                raise ValueError("{}:{}: VisDrone category {} outside 0..11".format(
                    path, line_no, category))
            # Category 0 (ignored regions), category 11 ('others'), and score==0
            # are ignored by visdrone2yolo.
            if score == 0 or category in (0, 11):
                continue
            cls = category - 1
            if cls >= num_classes:
                raise ValueError("{}:{}: VisDrone class {} outside [0, {})".format(
                    path, line_no, cls, num_classes))
            boxes.append([x, y, x + bw, y + bh])
            classes.append(cls)
        else:
            if "," in line:
                raise ValueError("{}:{}: mixed label formats; expected whitespace-separated YOLO row".format(
                    path, line_no))
            values = _parse_numeric(line.split(), path, line_no, 5, "YOLO ground truth")
            cls = _class_id(values[0], path, line_no, num_classes, "YOLO ground-truth")
            cx, cy, bw, bh = values[1:5]
            if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
                raise ValueError("{}:{}: YOLO ground-truth center must be in [0, 1]".format(
                    path, line_no))
            if not (0.0 < bw <= 1.0 and 0.0 < bh <= 1.0):
                raise ValueError("{}:{}: YOLO ground-truth width and height must be in (0, 1]".format(
                    path, line_no))
            cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
            boxes.append([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])
            classes.append(cls)
    return np.asarray(boxes, dtype=float).reshape(-1, 4), np.asarray(classes, dtype=int)


def load_pred(path, num_classes=None, nc=None):
    """Load ``class conf x1 y1 x2 y2`` predictions with strict validation."""
    num_classes = _resolve_num_classes(num_classes, nc)
    rows = _read_label_lines(path)
    if not rows:
        return _empty_pred()
    boxes, scores, classes = [], [], []
    for line_no, line in rows:
        values = _parse_numeric(line.split(), path, line_no, 6, "prediction")
        cls = _class_id(values[0], path, line_no, num_classes, "prediction")
        score = values[1]
        if not 0.0 <= score <= 1.0:
            raise ValueError("{}:{}: prediction confidence must be in [0, 1]".format(path, line_no))
        x1, y1, x2, y2 = values[2:6]
        if x2 <= x1 or y2 <= y1:
            raise ValueError("{}:{}: prediction box must have x2>x1 and y2>y1".format(path, line_no))
        boxes.append([x1, y1, x2, y2]); scores.append(score); classes.append(cls)
    return (np.asarray(boxes, dtype=float).reshape(-1, 4),
            np.asarray(scores, dtype=float), np.asarray(classes, dtype=int))


def box_iou(a, b):                                    # (N,4),(M,4) -> (N,M)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


IOUV = np.linspace(0.5, 0.95, 10)


def match(pred_cls, true_cls, iou):                   # -> (N_pred, 10) correct matrix
    correct = np.zeros((pred_cls.shape[0], 10), bool)
    cc = true_cls[:, None] == pred_cls[None, :]        # (M_gt, N_pred)
    iou = iou.T * cc                                   # iou passed (N_pred,M_gt) -> (M,N)
    for k, thr in enumerate(IOUV):
        gt_i, pr_i = np.nonzero(iou >= thr)
        if gt_i.size:
            m = np.stack([gt_i, pr_i, iou[gt_i, pr_i]], 1)
            m = m[m[:, 2].argsort()[::-1]]
            m = m[np.unique(m[:, 1], return_index=True)[1]]
            m = m[np.unique(m[:, 0], return_index=True)[1]]
            correct[m[:, 1].astype(int), k] = True
    return correct


def compute_ap(recall, precision):
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    return _trapz(np.interp(x, mrec, mpre), x)


def ap_per_class(tp, conf, pred_cls, target_cls, num_classes=None, nc=None):
    """Compute AP for every class in the declared profile.

    ``np.unique(target_cls)`` is deliberately not used to size the result:
    classes that have no ground truth (or no predictions) still contribute a
    zero AP to the macro average.  This keeps the standalone evaluator's mAP
    definition fixed and comparable across splits.
    """
    num_classes = _resolve_num_classes(num_classes, nc)
    tp = np.asarray(tp, dtype=bool)
    conf = np.asarray(conf, dtype=float).reshape(-1)
    pred_cls = np.asarray(pred_cls).reshape(-1)
    target_cls = np.asarray(target_cls).reshape(-1)
    if tp.ndim != 2 or tp.shape[1] != len(IOUV):
        raise ValueError("tp must have shape (N, {})".format(len(IOUV)))
    if tp.shape[0] != conf.size or tp.shape[0] != pred_cls.size:
        raise ValueError("tp, conf, and pred_cls must contain the same number of predictions")
    if not np.isfinite(conf).all():
        raise ValueError("prediction confidences must be finite")
    for values, name in ((pred_cls, "prediction"), (target_cls, "target")):
        if values.size:
            try:
                numeric = values.astype(float)
            except (TypeError, ValueError) as exc:
                raise ValueError("{} class ids must be numeric".format(name)) from exc
            if not np.isfinite(numeric).all() or not np.equal(numeric, np.floor(numeric)).all():
                raise ValueError("{} class ids must be finite integers".format(name))
            if np.any((numeric < 0) | (numeric >= num_classes)):
                raise ValueError("{} class ids outside [0, {})".format(name, num_classes))
    pred_cls = pred_cls.astype(int, copy=False)
    target_cls = target_cls.astype(int, copy=False)
    order = np.argsort(-conf, kind="stable")
    tp, pred_cls = tp[order], pred_cls[order]
    ap = np.zeros((num_classes, tp.shape[1]), dtype=float)
    for c in range(num_classes):
        mask = pred_cls == c
        n_gt = int((target_cls == c).sum())
        # A class without GT has AP=0 by the fixed-profile macro definition.
        if not mask.any() or n_gt == 0:
            continue
        fpc = (1 - tp[mask]).cumsum(axis=0)
        tpc = tp[mask].cumsum(axis=0)
        recall = tpc / (n_gt + 1e-16)
        # Every selected prediction contributes either a TP or FP, so this
        # denominator is strictly positive.  Avoid an epsilon here: adding one
        # would bias a perfect one-prediction class to AP=0.995 instead of 1.0.
        precision = tpc / (tpc + fpc)
        for j in range(tp.shape[1]):
            ap[c, j] = compute_ap(recall[:, j], precision[:, j])
    return ap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--images", default="images/val")
    ap.add_argument("--image-root", help="root used to normalize list entries and compute portable image digests")
    ap.add_argument("--labels", default="labels/val")
    ap.add_argument("--profile", "--classes", dest="profile", choices=tuple(CLASS_PROFILES),
                    default="visdrone", help="class profile (default: visdrone)")
    ap.add_argument("--nc", type=int, default=None,
                    help="number of classes; overrides the selected profile")
    ap.add_argument("--label-format", choices=("yolo", "visdrone", "auto"), default="yolo")
    ap.add_argument("--imgsz", type=int, default=None,
                    help="square model input; defaults to the selected profile")
    ap.add_argument("--conf", type=float, default=None,
                    help="global confidence threshold recorded for the run")
    ap.add_argument("--iou", type=float, default=None,
                    help="NMS IoU threshold recorded for the run")
    ap.add_argument("--max-det", type=int, default=None,
                    help="maximum detections per image; defaults to the selected profile")
    ap.add_argument(
        "--multi-label", dest="multi_label", action="store_true", default=None,
        help="record multi-label decoding (the Issue #51 recipe)",
    )
    ap.add_argument(
        "--single-label", dest="multi_label", action="store_false",
        help="record argmax-per-anchor decoding (diagnostic only)",
    )
    ap.add_argument(
        "--small-conf", type=float, default=-1.0,
        help="optional lower confidence for boxes below --small-area (-1 disables)",
    )
    ap.add_argument(
        "--small-area", type=float, default=32.0 * 32.0,
        help="original-image area threshold for --small-conf (default: 1024)",
    )
    ap.add_argument("--min-images", type=int, default=500)
    ap.add_argument("--smoke", action="store_true", help="allow a smaller diagnostic subset")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--json", help="write a machine-readable result JSON")
    ap.add_argument("--reference-json", help="PyTorch JSON containing mAP50-95")
    ap.add_argument(
        "--max-abs-delta-pct", type=float, default=None,
        help="absolute relative mAP50-95 gate in percent (requires --reference-json)",
    )
    ap.add_argument("--max-abs-delta-pp", type=float, default=None,
                    help="absolute mAP50-95 gate in percentage points")
    ap.add_argument(
        "--routing-semantics", choices=ROUTING_SEMANTICS, default=None,
        help=(
            "EsMoE inference path; required for formal runs (use dense_fallback "
            "for static exports)"
        ),
    )
    a = ap.parse_args()
    if a.min_images < 1 or a.limit < 0:
        ap.error("--min-images must be positive and --limit must be non-negative")
    if not a.smoke and a.min_images < 500:
        ap.error("Issue #51 acceptance requires --min-images >= 500; use --smoke for a smaller diagnostic run")
    if a.nc is not None and a.nc < 1:
        ap.error("--nc must be a positive integer")
    profile_protocol = PROFILE_PROTOCOLS[a.profile]
    protocol = dict(profile_protocol)
    for key in ("imgsz", "conf", "iou", "max_det", "multi_label"):
        value = getattr(a, key)
        if value is not None:
            protocol[key] = value
    if (protocol["imgsz"] <= 0 or protocol["max_det"] <= 0
            or not np.isfinite([protocol["conf"], protocol["iou"]]).all()
            or not 0.0 <= protocol["conf"] <= 1.0
            or not 0.0 <= protocol["iou"] <= 1.0):
        ap.error("imgsz/max-det must be positive and conf/iou must be finite in [0, 1]")
    if (not np.isfinite([a.small_conf, a.small_area]).all()
            or not -1.0 <= a.small_conf <= 1.0 or a.small_area < 0.0):
        ap.error("--small-conf must be finite in [-1, 1] and --small-area must be finite and non-negative")
    if a.max_abs_delta_pct is not None and a.reference_json is None:
        ap.error("--max-abs-delta-pct requires --reference-json")
    if a.max_abs_delta_pct is not None and (
            not np.isfinite(a.max_abs_delta_pct) or a.max_abs_delta_pct < 0):
        ap.error("--max-abs-delta-pct must be a finite non-negative number")
    if a.max_abs_delta_pp is not None and a.reference_json is None:
        ap.error("--max-abs-delta-pp requires --reference-json")
    if a.max_abs_delta_pp is not None and (
            not np.isfinite(a.max_abs_delta_pp) or a.max_abs_delta_pp < 0):
        ap.error("--max-abs-delta-pp must be a finite non-negative number")
    if a.max_abs_delta_pct is not None and a.max_abs_delta_pp is not None:
        ap.error("choose either --max-abs-delta-pct or --max-abs-delta-pp")
    if a.smoke and (a.max_abs_delta_pct is not None or a.max_abs_delta_pp is not None):
        ap.error("mAP delta gates cannot be used with --smoke")
    if not a.smoke and a.label_format != "yolo":
        ap.error("formal acceptance requires converted YOLO labels; use --label-format yolo")
    if not a.smoke and a.routing_semantics is None:
        ap.error(
            "formal Issue #51 evaluation requires --routing-semantics; "
            "use dense_fallback for the static export path"
        )
    num_classes = a.nc if a.nc is not None else CLASS_PROFILES[a.profile]
    image_root, imgs = _resolve_images(a.images, a.image_root)
    if a.limit:
        imgs = imgs[:a.limit]
    if not imgs:
        ap.error("no validation images found")
    stems = [os.path.splitext(os.path.basename(p))[0].casefold() for p in imgs]
    if len(stems) != len(set(stems)):
        ap.error("validation image stems are not unique")
    if not a.smoke and len(imgs) < a.min_images:
        ap.error("Issue #51 acceptance requires at least {} images (found {})".format(a.min_images, len(imgs)))
    try:
        label_files = _prediction_files(a.labels, "label") if os.path.isdir(a.labels) else {}
        prediction_files = _prediction_files(a.preds, "prediction")
    except ValueError as exc:
        ap.error(str(exc))
    expected_stems = {os.path.splitext(os.path.basename(path))[0].casefold() for path in imgs}
    if not a.smoke:
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
            ap.error("formal evaluation requires an exact image/file set; " + "; ".join(problems))
    all_tp, all_conf, all_pcls, all_tcls = [], [], [], []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        stem_key = stem.casefold()
        if not a.smoke:
            for required in (label_files[stem_key], prediction_files[stem_key]):
                if not os.path.isfile(required):
                    ap.error("missing per-image file: {}".format(required))
        try:
            w, h = image_size(p)
            gtb, gtc = load_gt(
                label_files.get(stem_key, os.path.join(a.labels, stem + ".txt")), w, h, a.label_format,
                num_classes=num_classes,
            )
            pb, ps, pc = load_pred(
                prediction_files.get(stem_key, os.path.join(a.preds, stem + ".txt")), num_classes=num_classes,
            )
        except (ValueError, OSError, struct.error, IndexError) as exc:
            ap.error(str(exc))
        all_tcls.append(gtc)
        if pb.shape[0] == 0:
            continue
        tp = (match(pc, gtc, box_iou(pb, gtb)) if gtb.shape[0]
              else np.zeros((pb.shape[0], len(IOUV)), bool))
        all_tp.append(tp); all_conf.append(ps); all_pcls.append(pc)
    tcls = np.concatenate(all_tcls) if all_tcls and any(x.size for x in all_tcls) else np.zeros((0,), int)
    if all_tp:
        tp = np.concatenate(all_tp)
        conf = np.concatenate(all_conf)
        pcls = np.concatenate(all_pcls)
    else:
        tp = np.zeros((0, len(IOUV)), bool); conf = np.zeros((0,)); pcls = np.zeros((0,), int)
    APc = ap_per_class(tp, conf, pcls, tcls, num_classes=num_classes)
    map50, map5095 = float(APc[:, 0].mean()), float(APc.mean())
    if not np.isfinite([map50, map5095]).all():
        ap.error("mAP computation returned NaN or Inf; check labels and predictions")
    image_manifest_sha256, image_manifest_names = _image_manifest(imgs, image_root)
    image_content_manifest_sha256 = _image_content_manifest(
        imgs, image_root, image_manifest_names
    )
    result = {
        "images": len(imgs), "classes": num_classes, "class_profile": a.profile,
        "mAP50": map50, "mAP50-95": map5095,
        "label_format": a.label_format,
        "image_manifest_sha256": image_manifest_sha256,
        "image_manifest": image_manifest_names,
        "image_list_sha256": image_content_manifest_sha256,
        "image_content_manifest_sha256": image_content_manifest_sha256,
        "protocol": dict(
            protocol,
            classes=num_classes,
            small_conf=a.small_conf,
            small_area=a.small_area,
            routing_semantics=a.routing_semantics,
        ),
    }
    exit_code = 0
    if a.reference_json:
        try:
            with open(a.reference_json, encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            ap.error("unable to read reference JSON: {}".format(exc))
        metadata_errors = _reference_metadata_errors(
            payload, image_manifest_sha256, image_content_manifest_sha256,
            len(imgs), a.profile, num_classes, result["protocol"], a.label_format
        )
        result["reference_metadata_match"] = not metadata_errors
        if metadata_errors and (
                a.smoke or (a.max_abs_delta_pct is None and a.max_abs_delta_pp is None)):
            result["reference_metadata_warnings"] = metadata_errors
        if metadata_errors and not a.smoke and (
                a.max_abs_delta_pct is not None or a.max_abs_delta_pp is not None):
            ap.error("; ".join(metadata_errors))
        try:
            reference = _reference_map(payload)
        except (TypeError, ValueError) as exc:
            ap.error(str(exc))
        if not np.isfinite(reference) or not 0.0 <= reference <= 1.0:
            ap.error("reference mAP50-95 must be finite and in [0, 1]")
        if a.max_abs_delta_pct is not None and reference <= 0.0:
            ap.error("reference mAP50-95 must be positive when applying a relative delta gate")
        delta_pp = (map5095 - reference) * 100.0
        delta_abs = map5095 - reference
        result.update({
            "reference_mAP50-95": reference,
            "delta_mAP50-95_abs": delta_abs,
            "delta_mAP50-95_pp": delta_pp,
            "abs_delta_mAP50-95_pp": abs(delta_pp),
        })
        if reference > 0.0:
            delta_pct = delta_abs / reference * 100.0
            result.update({
                "delta_mAP50-95_pct": delta_pct,
                "abs_delta_mAP50-95_pct": abs(delta_pct),
            })
        else:
            result.update({"delta_mAP50-95_pct": None, "abs_delta_mAP50-95_pct": None})
        if a.max_abs_delta_pct is not None:
            result["max_abs_delta_mAP50-95_pct"] = a.max_abs_delta_pct
            result["mAP50-95_relative_delta_gate_passed"] = (
                abs(result["delta_mAP50-95_pct"]) <= a.max_abs_delta_pct
            )
            exit_code = 0 if result["mAP50-95_relative_delta_gate_passed"] else 2
        if a.max_abs_delta_pp is not None:
            result["max_abs_delta_mAP50-95_pp"] = a.max_abs_delta_pp
            result["mAP50-95_absolute_delta_gate_passed"] = abs(delta_pp) <= a.max_abs_delta_pp
            exit_code = 0 if result["mAP50-95_absolute_delta_gate_passed"] else 2
        if a.max_abs_delta_pct is not None or a.max_abs_delta_pp is not None:
            result["mAP50-95_delta_gate_passed"] = exit_code == 0
    print(json.dumps(result, indent=2))
    if a.json:
        with open(a.json, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
            handle.write("\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
