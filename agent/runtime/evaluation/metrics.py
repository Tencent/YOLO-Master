from __future__ import annotations

from pathlib import Path
from typing import Any


def parse_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        if isinstance(value, str):
            text = value.strip()
            if text and text.lstrip("+-").replace(".0", "").isdigit():
                return int(float(text))
            return None
        return int(value)
    except Exception:
        return None


def coerce_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def valid_xyxy(box: Any) -> list[float] | None:
    if not isinstance(box, (list, tuple)) or len(box) < 4:
        return None
    values: list[float] = []
    for item in box[:4]:
        value = coerce_float(item)
        if value is None:
            return None
        values.append(round(value, 3))
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def xyxy_to_xywh(box: list[float]) -> list[float]:
    return [round(box[0], 3), round(box[1], 3), round(box[2] - box[0], 3), round(box[3] - box[1], 3)]


def box_iou_xyxy(a: list[float], b: list[float]) -> float:
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter_area
    return float(inter_area / union) if union > 0 else 0.0


def average_precision(recalls: list[float], precisions: list[float]) -> float:
    if not recalls or not precisions:
        return 0.0
    mrec = [0.0, *recalls, 1.0]
    mpre = [0.0, *precisions, 0.0]
    for idx in range(len(mpre) - 2, -1, -1):
        mpre[idx] = max(mpre[idx], mpre[idx + 1])
    ap = 0.0
    for idx in range(len(mrec) - 1):
        if mrec[idx + 1] != mrec[idx]:
            ap += (mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]
    return round(ap, 6)


def match_class_predictions(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]], *, class_id: int, iou_threshold: float) -> dict[str, Any]:
    class_predictions = sorted([item for item in predictions if coerce_int(item.get("class_id")) == class_id], key=lambda item: float(item.get("confidence") or 0.0), reverse=True)
    class_ground_truth = [item for item in ground_truth if coerce_int(item.get("class_id")) == class_id]
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    for item in class_ground_truth:
        gt_by_image.setdefault(str(item.get("image_id")), []).append(item)
    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    tp_flags: list[int] = []
    fp_flags: list[int] = []
    for prediction in class_predictions:
        image_id = str(prediction.get("image_id"))
        best_iou = 0.0
        best_gt_index: int | None = None
        for gt_index, target in enumerate(gt_by_image.get(image_id, [])):
            if gt_index in matched.setdefault(image_id, set()):
                continue
            iou = box_iou_xyxy(prediction["bbox_xyxy"], target["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gt_index = gt_index
        if best_gt_index is not None and best_iou >= iou_threshold:
            matched[image_id].add(best_gt_index)
            tp_flags.append(1)
            fp_flags.append(0)
        else:
            tp_flags.append(0)
            fp_flags.append(1)
    tp_total = sum(tp_flags)
    fp_total = sum(fp_flags)
    fn_total = max(0, len(class_ground_truth) - tp_total)
    recalls: list[float] = []
    precisions: list[float] = []
    tp_cum = 0
    fp_cum = 0
    for tp, fp in zip(tp_flags, fp_flags):
        tp_cum += tp
        fp_cum += fp
        recalls.append(tp_cum / len(class_ground_truth) if class_ground_truth else 0.0)
        precisions.append(tp_cum / (tp_cum + fp_cum) if tp_cum + fp_cum else 0.0)
    return {"tp": tp_total, "fp": fp_total, "fn": fn_total, "ap": average_precision(recalls, precisions) if class_ground_truth else None}


def evaluate_detection_metric_preview(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]], *, iou_thresholds: list[float] | None = None) -> dict[str, Any]:
    thresholds = iou_thresholds or [round(0.5 + 0.05 * idx, 2) for idx in range(10)]
    if not ground_truth:
        return {"status": "skipped", "reason": "ground_truth_unavailable", "predictions": len(predictions), "ground_truth": 0}
    gt_classes = sorted({int(item["class_id"]) for item in ground_truth if coerce_int(item.get("class_id")) is not None})
    pred_classes = sorted({int(item["class_id"]) for item in predictions if coerce_int(item.get("class_id")) is not None})
    all_classes = sorted(set(gt_classes) | set(pred_classes))
    per_threshold: dict[str, dict[str, Any]] = {}
    ap_by_threshold: dict[float, list[float]] = {threshold: [] for threshold in thresholds}
    counts_at_50 = {"tp": 0, "fp": 0, "fn": 0}
    per_class_50: dict[str, dict[str, Any]] = {}
    for threshold in thresholds:
        threshold_counts = {"tp": 0, "fp": 0, "fn": 0}
        for class_id in all_classes:
            result = match_class_predictions(predictions, ground_truth, class_id=class_id, iou_threshold=threshold)
            threshold_counts["tp"] += int(result["tp"])
            threshold_counts["fp"] += int(result["fp"])
            threshold_counts["fn"] += int(result["fn"])
            if class_id in gt_classes:
                ap_by_threshold[threshold].append(float(result["ap"] or 0.0))
            if abs(threshold - 0.5) < 1e-9:
                per_class_50[str(class_id)] = {"tp": int(result["tp"]), "fp": int(result["fp"]), "fn": int(result["fn"]), "ap": result["ap"]}
        precision = threshold_counts["tp"] / (threshold_counts["tp"] + threshold_counts["fp"]) if threshold_counts["tp"] + threshold_counts["fp"] else 0.0
        recall = threshold_counts["tp"] / (threshold_counts["tp"] + threshold_counts["fn"]) if threshold_counts["tp"] + threshold_counts["fn"] else 0.0
        per_threshold[f"{threshold:.2f}"] = {**threshold_counts, "precision": round(precision, 6), "recall": round(recall, 6), "map": round(sum(ap_by_threshold[threshold]) / len(ap_by_threshold[threshold]), 6) if ap_by_threshold[threshold] else 0.0}
        if abs(threshold - 0.5) < 1e-9:
            counts_at_50 = threshold_counts
    precision_50 = counts_at_50["tp"] / (counts_at_50["tp"] + counts_at_50["fp"]) if counts_at_50["tp"] + counts_at_50["fp"] else 0.0
    recall_50 = counts_at_50["tp"] / (counts_at_50["tp"] + counts_at_50["fn"]) if counts_at_50["tp"] + counts_at_50["fn"] else 0.0
    map50 = per_threshold.get("0.50", {}).get("map", 0.0)
    maps = [value["map"] for value in per_threshold.values()]
    map50_95 = round(sum(maps) / len(maps), 6) if maps else 0.0
    f1 = (2 * precision_50 * recall_50 / (precision_50 + recall_50)) if precision_50 + recall_50 else 0.0
    return {
        "status": "ok",
        "basis": "yolo_label_metric_preview",
        "predictions": len(predictions),
        "ground_truth": len(ground_truth),
        "classes_with_ground_truth": len(gt_classes),
        "precision": round(precision_50, 6),
        "recall": round(recall_50, 6),
        "f1": round(f1, 6),
        "map50": round(float(map50), 6),
        "map50_95": map50_95,
        "counts_at_iou50": counts_at_50,
        "per_threshold": per_threshold,
        "per_class_iou50": per_class_50,
    }


def metric_delta(fused: dict[str, Any], yolo: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "map50", "map50_95"):
        if isinstance(fused.get(key), (int, float)) and isinstance(yolo.get(key), (int, float)):
            delta[key] = round(float(fused[key]) - float(yolo[key]), 6)
    if "map50_95" in delta:
        delta["direction"] = "improved" if delta["map50_95"] > 0 else ("regressed" if delta["map50_95"] < 0 else "unchanged")
    return delta


def evaluate_classification_metric_preview(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]]) -> dict[str, Any]:
    if not ground_truth:
        return {"status": "skipped", "reason": "ground_truth_unavailable", "ground_truth": 0, "predictions": len(predictions)}
    gt_labels: dict[str, set[int]] = {}
    for item in ground_truth:
        class_id = coerce_int(item.get("class_id"))
        if class_id is None:
            continue
        gt_labels.setdefault(str(item.get("image_id")), set()).add(int(class_id))
    pred_by_image: dict[str, list[dict[str, Any]]] = {}
    for item in predictions:
        pred_by_image.setdefault(str(item.get("image_id")), []).append(item)
    total_images = len(gt_labels)
    exact_match = 0
    top1_correct = 0
    tp = 0
    fp = 0
    fn = 0
    for image_id, gt_classes in gt_labels.items():
        preds = sorted(pred_by_image.get(image_id, []), key=lambda entry: float(entry.get("confidence") or 0.0), reverse=True)
        pred_classes = {int(item["class_id"]) for item in preds if coerce_int(item.get("class_id")) is not None}
        if pred_classes == gt_classes:
            exact_match += 1
        if preds:
            top1 = coerce_int(preds[0].get("class_id"))
            if top1 is not None and top1 in gt_classes:
                top1_correct += 1
        tp += len(pred_classes & gt_classes)
        fp += len(pred_classes - gt_classes)
        fn += len(gt_classes - pred_classes)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return {
        "status": "ok",
        "basis": "label_presence_preview",
        "ground_truth": len(ground_truth),
        "predictions": len(predictions),
        "images_with_ground_truth": total_images,
        "top1_accuracy": round(top1_correct / total_images if total_images else 0.0, 6),
        "exact_set_accuracy": round(exact_match / total_images if total_images else 0.0, 6),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "counts": {"tp": tp, "fp": fp, "fn": fn},
    }


def classification_metric_delta(fused: dict[str, Any], yolo: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("top1_accuracy", "exact_set_accuracy", "precision", "recall", "f1"):
        if isinstance(fused.get(key), (int, float)) and isinstance(yolo.get(key), (int, float)):
            delta[key] = round(float(fused[key]) - float(yolo[key]), 6)
    if "top1_accuracy" in delta:
        delta["direction"] = "improved" if delta["top1_accuracy"] > 0 else ("regressed" if delta["top1_accuracy"] < 0 else "unchanged")
    return delta


def evaluate_segmentation_metric_preview(predictions: list[dict[str, Any]], ground_truth: list[dict[str, Any]], *, iou_threshold: float = 0.5, polygon_iou_approx_fn=None) -> dict[str, Any]:
    if not ground_truth:
        return {"status": "skipped", "reason": "ground_truth_unavailable", "ground_truth": 0, "predictions": len(predictions)}
    predictions_sorted = sorted(predictions, key=lambda entry: float(entry.get("confidence") or 0.0), reverse=True)
    gt_by_image: dict[str, list[dict[str, Any]]] = {}
    for item in ground_truth:
        gt_by_image.setdefault(str(item.get("image_id")), []).append(item)
    matched: dict[str, set[int]] = {image_id: set() for image_id in gt_by_image}
    tp = 0
    fp = 0
    iou_sum = 0.0
    for prediction in predictions_sorted:
        class_id = coerce_int(prediction.get("class_id"))
        polygon = prediction.get("polygon_xy")
        image_id = str(prediction.get("image_id"))
        if class_id is None or not isinstance(polygon, list) or len(polygon) < 3:
            fp += 1
            continue
        best_iou = 0.0
        best_index: int | None = None
        for gt_index, target in enumerate(gt_by_image.get(image_id, [])):
            if gt_index in matched.setdefault(image_id, set()):
                continue
            if coerce_int(target.get("class_id")) != class_id:
                continue
            iou = polygon_iou_approx_fn(polygon, target.get("polygon_xy", [])) if polygon_iou_approx_fn else 0.0
            if iou > best_iou:
                best_iou = iou
                best_index = gt_index
        if best_index is not None and best_iou >= iou_threshold:
            matched[image_id].add(best_index)
            tp += 1
            iou_sum += best_iou
        else:
            fp += 1
    fn = sum(max(0, len(targets) - len(matched.get(image_id, set()))) for image_id, targets in gt_by_image.items())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    mean_iou = iou_sum / tp if tp else 0.0
    return {"status": "ok", "basis": "polygon_iou_preview", "ground_truth": len(ground_truth), "predictions": len(predictions), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "mean_iou": round(mean_iou, 6), "mask_ap50_proxy": round(precision, 6), "counts": {"tp": tp, "fp": fp, "fn": fn}}


def segmentation_metric_delta(fused: dict[str, Any], yolo: dict[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in ("precision", "recall", "f1", "mean_iou", "mask_ap50_proxy"):
        if isinstance(fused.get(key), (int, float)) and isinstance(yolo.get(key), (int, float)):
            delta[key] = round(float(fused[key]) - float(yolo[key]), 6)
    if "mask_ap50_proxy" in delta:
        delta["direction"] = "improved" if delta["mask_ap50_proxy"] > 0 else ("regressed" if delta["mask_ap50_proxy"] < 0 else "unchanged")
    return delta


def merge_counts(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        try:
            target[str(key)] = target.get(str(key), 0) + int(value)
        except Exception:
            continue


def preferred_verdict(item: dict[str, Any]) -> dict[str, Any]:
    multimodal = item.get("multimodal", {}) or {}
    llm = multimodal.get("llm_refine", {}) or {}
    if isinstance(llm, dict) and isinstance(llm.get("verdict"), dict):
        return llm["verdict"]
    vlm = multimodal.get("vlm", {}) or {}
    return vlm.get("verdict", {}) if isinstance(vlm, dict) and isinstance(vlm.get("verdict"), dict) else {}


def verdict_field_count(verdict: dict[str, Any], field: str) -> int:
    cross_check = verdict.get("yolo_cross_check", {}) if isinstance(verdict, dict) else {}
    if not isinstance(cross_check, dict):
        return 0
    value = cross_check.get(field)
    if isinstance(value, list):
        return len(value)
    return 1 if value else 0


def build_item_metric_preview(
    *,
    image_path: Path,
    names: dict[int, str],
    detections: list[dict[str, Any]],
    fusion_preview: dict[str, Any],
    verdict: dict[str, Any] | None = None,
    ground_truth_records_for_metric_fn,
    ground_truth_classification_records_for_metric_fn,
    ground_truth_segmentation_records_for_metric_fn,
    yolo_prediction_records_for_metric_fn,
    fused_prediction_records_for_metric_fn,
    yolo_classification_predictions_for_metric_fn,
    fused_classification_predictions_for_metric_fn,
    yolo_segmentation_predictions_for_metric_fn,
    fused_segmentation_predictions_for_metric_fn,
    polygon_iou_approx_fn,
) -> dict[str, Any]:
    ground_truth = ground_truth_records_for_metric_fn(image_path, names)
    ground_truth_classes = ground_truth_classification_records_for_metric_fn(image_path, names)
    ground_truth_segments = ground_truth_segmentation_records_for_metric_fn(image_path, names)
    yolo_predictions = yolo_prediction_records_for_metric_fn(image_path, detections)
    fused_predictions = fused_prediction_records_for_metric_fn(image_path, fusion_preview)
    yolo_metrics = evaluate_detection_metric_preview(yolo_predictions, ground_truth)
    fused_metrics = evaluate_detection_metric_preview(fused_predictions, ground_truth)
    yolo_class_predictions = yolo_classification_predictions_for_metric_fn(image_path, detections)
    fused_class_predictions = fused_classification_predictions_for_metric_fn(image_path, verdict or {}, detections)
    yolo_class_metrics = evaluate_classification_metric_preview(yolo_class_predictions, ground_truth_classes)
    fused_class_metrics = evaluate_classification_metric_preview(fused_class_predictions, ground_truth_classes)
    yolo_seg_predictions = yolo_segmentation_predictions_for_metric_fn(image_path, detections)
    fused_seg_predictions = fused_segmentation_predictions_for_metric_fn(image_path, verdict or {}, detections)
    yolo_seg_metrics = evaluate_segmentation_metric_preview(yolo_seg_predictions, ground_truth_segments, polygon_iou_approx_fn=polygon_iou_approx_fn)
    fused_seg_metrics = evaluate_segmentation_metric_preview(fused_seg_predictions, ground_truth_segments, polygon_iou_approx_fn=polygon_iou_approx_fn)
    status = "ok" if yolo_metrics.get("status") == "ok" or fused_metrics.get("status") == "ok" else "skipped"
    return {
        "status": status,
        "basis": "multitask_label_preview",
        "image_id": str(image_path.resolve()),
        "ground_truth_objects": len(ground_truth),
        "ground_truth_classes": len(ground_truth_classes),
        "ground_truth_segments": len(ground_truth_segments),
        "detection": {"yolo": yolo_metrics, "fused": fused_metrics, "delta": metric_delta(fused_metrics, yolo_metrics)},
        "classification": {"yolo": yolo_class_metrics, "fused": fused_class_metrics, "delta": classification_metric_delta(fused_class_metrics, yolo_class_metrics)},
        "segmentation": {"yolo": yolo_seg_metrics, "fused": fused_seg_metrics, "delta": segmentation_metric_delta(fused_seg_metrics, yolo_seg_metrics)},
        "yolo": yolo_metrics,
        "fused": fused_metrics,
        "delta": metric_delta(fused_metrics, yolo_metrics),
    }


def aggregate_metric_preview(
    items: list[dict[str, Any]],
    names: dict[int, str],
    *,
    ground_truth_records_for_metric_fn,
    ground_truth_classification_records_for_metric_fn,
    ground_truth_segmentation_records_for_metric_fn,
    yolo_prediction_records_for_metric_fn,
    fused_prediction_records_for_metric_fn,
    yolo_classification_predictions_for_metric_fn,
    fused_classification_predictions_for_metric_fn,
    yolo_segmentation_predictions_for_metric_fn,
    fused_segmentation_predictions_for_metric_fn,
    merge_verdicts_fn,
    polygon_iou_approx_fn,
) -> dict[str, Any]:
    ground_truth: list[dict[str, Any]] = []
    ground_truth_classes: list[dict[str, Any]] = []
    ground_truth_segments: list[dict[str, Any]] = []
    yolo_predictions: list[dict[str, Any]] = []
    fused_predictions: list[dict[str, Any]] = []
    yolo_class_predictions: list[dict[str, Any]] = []
    fused_class_predictions: list[dict[str, Any]] = []
    yolo_seg_predictions: list[dict[str, Any]] = []
    fused_seg_predictions: list[dict[str, Any]] = []
    images_with_ground_truth = 0
    for item in items:
        image_path_value = item.get("path")
        if not image_path_value:
            continue
        image_path = Path(str(image_path_value))
        if not image_path.exists():
            continue
        item_ground_truth = ground_truth_records_for_metric_fn(image_path, names)
        item_ground_truth_classes = ground_truth_classification_records_for_metric_fn(image_path, names)
        item_ground_truth_segments = ground_truth_segmentation_records_for_metric_fn(image_path, names)
        if item_ground_truth:
            images_with_ground_truth += 1
        ground_truth.extend(item_ground_truth)
        ground_truth_classes.extend(item_ground_truth_classes)
        ground_truth_segments.extend(item_ground_truth_segments)
        detector_summary = item.get("detector", {}).get("summary", {}) if isinstance(item.get("detector"), dict) else {}
        if isinstance(detector_summary, dict):
            yolo_predictions.extend(yolo_prediction_records_for_metric_fn(image_path, [detector_summary]))
            yolo_class_predictions.extend(yolo_classification_predictions_for_metric_fn(image_path, [detector_summary]))
            yolo_seg_predictions.extend(yolo_segmentation_predictions_for_metric_fn(image_path, [detector_summary]))
        fusion_preview = item.get("multimodal", {}).get("fusion", {}) if isinstance(item.get("multimodal"), dict) else {}
        fused_predictions.extend(fused_prediction_records_for_metric_fn(image_path, fusion_preview if isinstance(fusion_preview, dict) else {}))
        verdict = {}
        multimodal = item.get("multimodal", {}) if isinstance(item.get("multimodal"), dict) else {}
        vlm = multimodal.get("vlm", {}) if isinstance(multimodal, dict) else {}
        llm_refine = multimodal.get("llm_refine", {}) if isinstance(multimodal, dict) else {}
        if isinstance(vlm, dict) and isinstance(vlm.get("verdict"), dict):
            verdict = merge_verdicts_fn(verdict, vlm.get("verdict", {}))
        if isinstance(llm_refine, dict) and isinstance(llm_refine.get("verdict"), dict):
            verdict = merge_verdicts_fn(verdict, llm_refine.get("verdict", {}))
        if isinstance(detector_summary, dict):
            fused_class_predictions.extend(fused_classification_predictions_for_metric_fn(image_path, verdict, [detector_summary]))
            fused_seg_predictions.extend(fused_segmentation_predictions_for_metric_fn(image_path, verdict, [detector_summary]))
    yolo_metrics = evaluate_detection_metric_preview(yolo_predictions, ground_truth)
    fused_metrics = evaluate_detection_metric_preview(fused_predictions, ground_truth)
    yolo_class_metrics = evaluate_classification_metric_preview(yolo_class_predictions, ground_truth_classes)
    fused_class_metrics = evaluate_classification_metric_preview(fused_class_predictions, ground_truth_classes)
    yolo_seg_metrics = evaluate_segmentation_metric_preview(yolo_seg_predictions, ground_truth_segments, polygon_iou_approx_fn=polygon_iou_approx_fn)
    fused_seg_metrics = evaluate_segmentation_metric_preview(fused_seg_predictions, ground_truth_segments, polygon_iou_approx_fn=polygon_iou_approx_fn)
    status = "ok" if yolo_metrics.get("status") == "ok" or fused_metrics.get("status") == "ok" else "skipped"
    return {
        "status": status,
        "basis": "multitask_label_preview",
        "note": "Lightweight same-sample preview for detection, label-presence classification, and rough segmentation. Run official task-specific validation before claiming benchmark gains.",
        "images_with_ground_truth": images_with_ground_truth,
        "ground_truth_objects": len(ground_truth),
        "ground_truth_classes": len(ground_truth_classes),
        "ground_truth_segments": len(ground_truth_segments),
        "yolo_predictions": len(yolo_predictions),
        "fused_predictions": len(fused_predictions),
        "detection": {"yolo": yolo_metrics, "fused": fused_metrics, "delta": metric_delta(fused_metrics, yolo_metrics)},
        "classification": {"yolo": yolo_class_metrics, "fused": fused_class_metrics, "delta": classification_metric_delta(fused_class_metrics, yolo_class_metrics)},
        "segmentation": {"yolo": yolo_seg_metrics, "fused": fused_seg_metrics, "delta": segmentation_metric_delta(fused_seg_metrics, yolo_seg_metrics)},
        "yolo": yolo_metrics,
        "fused": fused_metrics,
        "delta": metric_delta(fused_metrics, yolo_metrics),
    }


def prediction_records_to_coco(records: list[dict[str, Any]], image_path: Path | None = None) -> list[dict[str, Any]]:
    coco_records: list[dict[str, Any]] = []
    for record in records:
        class_id = coerce_int(record.get("class_id"))
        score = coerce_float(record.get("confidence"))
        bbox = valid_xyxy(record.get("bbox_xyxy"))
        if class_id is None or score is None or bbox is None:
            continue
        source_path = image_path or Path(str(record.get("image_id", "")))
        stem = source_path.stem if str(source_path) else str(record.get("image_id", ""))
        image_id: int | str = int(stem) if stem.isdigit() else stem
        coco_records.append({"image_id": image_id, "category_id": class_id, "bbox": xyxy_to_xywh(bbox), "score": round(score, 6), "source": record.get("source"), "action": record.get("action")})
    return coco_records


def yolo_coco_records_for_items(items: list[dict[str, Any]], *, yolo_prediction_records_for_metric_fn) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for item in items:
        image_path_value = item.get("path")
        if not image_path_value:
            continue
        image_path = Path(str(image_path_value))
        detector_summary = item.get("detector", {}).get("summary", {}) if isinstance(item.get("detector"), dict) else {}
        if isinstance(detector_summary, dict):
            records.extend(prediction_records_to_coco(yolo_prediction_records_for_metric_fn(image_path, [detector_summary]), image_path=image_path))
    return records


def build_metric_guardrail(*, items: list[dict[str, Any]], metric_preview: dict[str, Any], fused_coco_records: list[dict[str, Any]], multimodal_params: dict[str, Any], yolo_prediction_records_for_metric_fn) -> dict[str, Any]:
    enabled = parse_bool(multimodal_params.get("fusion_metric_guardrail"), True)
    if not enabled:
        return {"enabled": False, "selected": "fused_preview", "reason": "guardrail_disabled", "records": fused_coco_records}
    if metric_preview.get("status") != "ok":
        return {"enabled": True, "selected": "fused_preview", "reason": "metric_preview_unavailable", "records": fused_coco_records}
    delta = metric_preview.get("delta", {}) if isinstance(metric_preview.get("delta"), dict) else {}
    min_map_delta = float(multimodal_params.get("fusion_guardrail_min_map50_95_delta", 1e-6))
    require_recall = parse_bool(multimodal_params.get("fusion_guardrail_require_recall_nonnegative"), True)
    map_delta = coerce_float(delta.get("map50_95"))
    recall_delta = coerce_float(delta.get("recall"))
    material_change = len(fused_coco_records) != len(yolo_coco_records_for_items(items, yolo_prediction_records_for_metric_fn=yolo_prediction_records_for_metric_fn))
    if not material_change:
        fusion_summary = metric_preview.get("fused", {}) if isinstance(metric_preview.get("fused"), dict) else {}
        yolo_summary = metric_preview.get("yolo", {}) if isinstance(metric_preview.get("yolo"), dict) else {}
        material_change = any((fusion_summary.get(key) != yolo_summary.get(key)) for key in ("precision", "recall", "map50", "map50_95"))
    reasons: list[str] = []
    if not material_change:
        reasons.append("no_material_metric_or_prediction_change")
    if map_delta is None or map_delta < min_map_delta:
        reasons.append("map50_95_regressed")
    if require_recall and (recall_delta is None or recall_delta < 0.0):
        reasons.append("recall_regressed")
    if reasons:
        return {
            "enabled": True,
            "selected": "yolo_only",
            "reason": ",".join(reasons),
            "thresholds": {"min_map50_95_delta": min_map_delta, "require_recall_nonnegative": require_recall},
            "delta": delta,
            "records": yolo_coco_records_for_items(items, yolo_prediction_records_for_metric_fn=yolo_prediction_records_for_metric_fn),
        }
    return {
        "enabled": True,
        "selected": "fused_preview",
        "reason": "metric_preview_non_regressing",
        "thresholds": {"min_map50_95_delta": min_map_delta, "require_recall_nonnegative": require_recall},
        "delta": delta,
        "records": fused_coco_records,
    }


def aggregate_multimodal_evaluation(items: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    gt_counts: dict[str, int] = {}
    detection_counts: dict[str, int] = {}
    flag_counts = {"confirmed": 0, "false_positives": 0, "possible_misses": 0, "duplicate_or_fragmented": 0}
    fusion_counts = {"kept": 0, "suppressed": 0, "added": 0, "adjusted": 0, "relabelled": 0, "fused_boxes": 0, "coco_records": 0}
    parsed = 0
    total_boxes = 0
    total_gt_objects = 0
    gt_available = 0
    for item in items:
        status = str(item.get("status", "unknown"))
        status_counts[status] = status_counts.get(status, 0) + 1
        detector = item.get("detector", {}) or {}
        total_boxes += int(detector.get("boxes", 0) or 0)
        merge_counts(detection_counts, detector.get("label_counts", {}) or {})
        ground_truth = item.get("ground_truth", {}) or {}
        if ground_truth.get("exists"):
            gt_available += 1
            total_gt_objects += int(ground_truth.get("objects", 0) or 0)
            merge_counts(gt_counts, ground_truth.get("label_counts", {}) or {})
        multimodal = item.get("multimodal", {}) or {}
        vlm = multimodal.get("vlm", {}) or {}
        llm = multimodal.get("llm_refine", {}) or {}
        if vlm.get("verdict_parse_status") == "parsed" or llm.get("verdict_parse_status") == "parsed":
            parsed += 1
        verdict = preferred_verdict(item)
        for field in flag_counts:
            flag_counts[field] += verdict_field_count(verdict, field)
        fusion_summary = multimodal.get("fusion", {}).get("summary", {}) if isinstance(multimodal.get("fusion"), dict) else {}
        for key in fusion_counts:
            try:
                fusion_counts[key] += int(fusion_summary.get(key, 0) or 0)
            except Exception:
                continue
    total = len(items)
    return {
        "images_processed": total,
        "status_counts": status_counts,
        "verdicts_parsed": parsed,
        "verdict_parse_rate": round(parsed / total, 4) if total else 0.0,
        "detections_total": total_boxes,
        "ground_truth_total": total_gt_objects,
        "ground_truth_images": gt_available,
        "avg_detected_boxes": round(total_boxes / total, 4) if total else 0.0,
        "avg_ground_truth_objects": round(total_gt_objects / gt_available, 4) if gt_available else None,
        "detection_label_counts": detection_counts,
        "ground_truth_label_counts": gt_counts,
        "cross_check_flag_counts": flag_counts,
        "fusion_summary": fusion_counts,
    }


def overall_multimodal_evaluation_status(aggregate: dict[str, Any]) -> str:
    counts = aggregate.get("status_counts", {}) or {}
    total = int(aggregate.get("images_processed", 0) or 0)
    if total == 0:
        return "failed"
    if counts.get("ok") == total:
        return "ok"
    if counts.get("blocked") == total:
        return "blocked"
    if counts.get("failed") == total:
        return "failed"
    return "partial"
