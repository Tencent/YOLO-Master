from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from ..open_world.taxonomy import open_world_policy_enabled, open_world_template_enabled


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


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


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


def extract_yolo_indices(value: Any) -> set[int]:
    indices: set[int] = set()
    if isinstance(value, dict):
        for key in ("index", "yolo_index", "yolo_idx", "yolo_box_index", "box_index"):
            idx = coerce_int(value.get(key))
            if idx is not None:
                indices.add(idx)
        for key in ("linked_yolo_indices", "indices", "yolo_indices"):
            for item in as_list(value.get(key)):
                idx = coerce_int(item)
                if idx is not None:
                    indices.add(idx)
        return indices
    if isinstance(value, list):
        for item in value:
            indices.update(extract_yolo_indices(item))
        return indices
    idx = coerce_int(value)
    if idx is not None:
        indices.add(idx)
    return indices


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


def merge_verdicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base) if isinstance(base, dict) else {}
    if not isinstance(override, dict):
        return merged
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_verdicts(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def proposal_key(proposal: dict[str, Any]) -> str:
    proposal_id = proposal.get("proposal_id")
    if proposal_id not in (None, ""):
        return f"id:{proposal_id}"
    bbox = valid_xyxy(proposal.get("bbox_xyxy") or proposal.get("bbox") or proposal.get("xyxy"))
    return f"bbox:{bbox}:{proposal.get('class_id')}:{proposal.get('label')}"


def resolve_proposal_reference(item: Any, proposals_by_id: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    if isinstance(item, dict):
        proposal_id = item.get("proposal_id")
        if proposal_id not in (None, "") and str(proposal_id) in proposals_by_id:
            merged = dict(proposals_by_id[str(proposal_id)])
            merged.update({k: v for k, v in item.items() if v not in (None, "")})
            return merged
        return dict(item)
    key = str(item)
    if key in proposals_by_id:
        return dict(proposals_by_id[key])
    return None


def proposal_open_label(proposal: dict[str, Any]) -> str | None:
    for key in ("open_label", "object_label", "category_name", "label", "name"):
        value = proposal.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def proposal_coco_class_id(proposal: dict[str, Any]) -> int | None:
    for key in ("class_id", "category_id", "coco_class_id", "mapped_class_id", "canonical_class_id"):
        value = coerce_int(proposal.get(key))
        if value is not None:
            return value
    coco_candidate = proposal.get("coco_candidate")
    if isinstance(coco_candidate, dict):
        for key in ("class_id", "category_id"):
            value = coerce_int(coco_candidate.get(key))
            if value is not None:
                return value
    return None


def proposal_coco_label(proposal: dict[str, Any]) -> str | None:
    for key in ("coco_label", "canonical_label", "mapped_label", "label"):
        value = proposal.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    coco_candidate = proposal.get("coco_candidate")
    if isinstance(coco_candidate, dict):
        for key in ("label", "name"):
            value = coco_candidate.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


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


def build_multimodal_fusion_preview(
    *,
    detections: list[dict[str, Any]],
    verdict: dict[str, Any],
    multimodal_params: dict[str, Any],
    image_path: str | Path | None = None,
    normalize_detection_boxes_fn,
) -> dict[str, Any]:
    mode = str(multimodal_params.get("fusion_mode") or "preview").lower()
    policy = str(multimodal_params.get("fusion_policy") or "add_only").lower()
    enabled = parse_bool(multimodal_params.get("fusion_enabled"), mode not in {"off", "none", "false", "0"})
    if not enabled:
        return {"mode": "off", "strategy": "metric_safe_v1", "enabled": False}
    open_world_enabled = open_world_policy_enabled(multimodal_params) or open_world_template_enabled(multimodal_params)
    open_world_assist = policy in {"open_world_assist", "open-world-assist"}
    allow_add = policy in {"add_only", "balanced", "aggressive", "all"} or open_world_enabled
    allow_suppress = policy in {"balanced", "aggressive", "all"} or (open_world_enabled and not open_world_assist and policy in {"open_world", "open-world"})
    allow_adjust = policy in {"aggressive", "all"}
    allow_relabel = policy in {"aggressive", "all"}
    add_min = float(multimodal_params.get("fusion_add_confidence_min", 0.65))
    open_world_add_min = float(multimodal_params.get("fusion_open_world_confidence_min", 0.55 if open_world_enabled else add_min))
    add_require_unlinked = parse_bool(multimodal_params.get("fusion_add_require_unlinked"), True)
    add_max_linked_yolo_conf = float(multimodal_params.get("fusion_add_max_linked_yolo_confidence", 0.4))
    add_allowed_bbox_quality = {str(v).lower() for v in as_list(multimodal_params.get("fusion_add_allowed_bbox_quality") or ["exact", "estimated"]) if v not in (None, "")}
    suppress_min = float(multimodal_params.get("fusion_suppress_confidence_min", 0.75))
    adjust_min = float(multimodal_params.get("fusion_adjust_confidence_min", 0.7))
    suppress_max_yolo_conf = float(multimodal_params.get("fusion_suppress_max_yolo_confidence", 0.45))
    relabel_max_yolo_conf = float(multimodal_params.get("fusion_relabel_max_yolo_confidence", 0.5))
    adjust_min_iou = float(multimodal_params.get("fusion_adjust_min_iou", 0.5))
    open_world_iou_relabel_enabled = parse_bool(multimodal_params.get("open_world_iou_relabel_enabled"), False)
    open_world_iou_relabel_threshold = float(multimodal_params.get("open_world_iou_relabel_threshold", 0.7) or 0.7)
    open_world_iou_relabel_max_yolo_conf = float(multimodal_params.get("open_world_iou_relabel_max_yolo_confidence", 0.8) or 0.8)
    yolo_boxes = normalize_detection_boxes_fn(detections)
    yolo_by_index = {int(item["index"]): item for item in yolo_boxes}
    fusion_hints = verdict.get("fusion_hints", {}) if isinstance(verdict, dict) else {}
    fusion_hints = fusion_hints if isinstance(fusion_hints, dict) else {}
    cross_check = verdict.get("yolo_cross_check", {}) if isinstance(verdict, dict) else {}
    cross_check = cross_check if isinstance(cross_check, dict) else {}
    vlm_proposals = [item for item in as_list(verdict.get("vlm_detections")) if isinstance(item, dict)] if isinstance(verdict, dict) else []
    proposals_by_id = {str(item.get("proposal_id")): item for item in vlm_proposals if item.get("proposal_id") not in (None, "")}
    actions: list[dict[str, Any]] = []
    warnings: list[str] = []
    suppress_indices = set()
    open_world_predictions: list[dict[str, Any]] = []

    def yolo_confidence(idx: int) -> float | None:
        return coerce_float((yolo_by_index.get(idx) or {}).get("confidence"))

    def yolo_box(idx: int) -> list[float] | None:
        return valid_xyxy((yolo_by_index.get(idx) or {}).get("bbox_xyxy"))

    def can_suppress(idx: int, score: float | None) -> tuple[bool, str]:
        yolo_conf = yolo_confidence(idx)
        if score is not None and score < suppress_min:
            return False, "vlm_confidence_below_threshold"
        if yolo_conf is not None and yolo_conf > suppress_max_yolo_conf:
            return False, "yolo_confidence_too_high"
        return True, "vlm_false_positive_low_yolo_confidence"

    def can_add(proposal: dict[str, Any]) -> tuple[bool, str]:
        bbox_quality = str(proposal.get("bbox_quality") or "estimated").lower()
        if bbox_quality not in add_allowed_bbox_quality:
            return False, "bbox_quality_not_allowed"
        linked_indices = sorted(extract_yolo_indices(proposal))
        if add_require_unlinked and linked_indices:
            linked_confidences = [yolo_confidence(idx) for idx in linked_indices]
            if any(conf is not None and conf > add_max_linked_yolo_conf for conf in linked_confidences):
                return False, "linked_yolo_confidence_too_high"
        return True, "vlm_add_candidate"

    for item in as_list(fusion_hints.get("suppress_yolo_indices")) + as_list(cross_check.get("false_positives")):
        score = coerce_float(item.get("confidence")) if isinstance(item, dict) else None
        for idx in extract_yolo_indices(item):
            if not allow_suppress:
                actions.append({"type": "reject_suppress", "yolo_index": idx, "confidence": score, "yolo_confidence": yolo_confidence(idx), "reason": "fusion_policy_disallows_suppress", "policy": policy})
                continue
            allowed, reason = can_suppress(idx, score)
            if allowed:
                suppress_indices.add(idx)
                actions.append({"type": "suppress", "yolo_index": idx, "confidence": score, "yolo_confidence": yolo_confidence(idx), "reason": reason})
            else:
                actions.append({"type": "reject_suppress", "yolo_index": idx, "confidence": score, "yolo_confidence": yolo_confidence(idx), "reason": reason, "threshold": suppress_min, "max_yolo_confidence": suppress_max_yolo_conf})

    adjustments: dict[int, dict[str, Any]] = {}
    for item in as_list(fusion_hints.get("adjust_boxes")):
        if not isinstance(item, dict):
            continue
        idx_set = extract_yolo_indices(item)
        score = coerce_float(item.get("confidence"))
        if not allow_adjust:
            for idx in idx_set:
                actions.append({"type": "reject_adjust_box", "yolo_index": idx, "confidence": score, "reason": "fusion_policy_disallows_adjust", "policy": policy})
            continue
        bbox = valid_xyxy(item.get("bbox_xyxy") or item.get("bbox") or item.get("xyxy"))
        if bbox is None:
            continue
        for idx in idx_set:
            original_box = yolo_box(idx)
            overlap = box_iou_xyxy(original_box, bbox) if original_box is not None else None
            if score is not None and score < adjust_min:
                actions.append({"type": "reject_adjust_box", "yolo_index": idx, "confidence": score, "threshold": adjust_min})
            elif original_box is None:
                actions.append({"type": "reject_adjust_box", "yolo_index": idx, "confidence": score, "reason": "missing_original_yolo_box"})
            elif overlap is None or overlap < adjust_min_iou:
                actions.append({"type": "reject_adjust_box", "yolo_index": idx, "confidence": score, "iou_with_original": overlap, "min_iou": adjust_min_iou, "reason": "adjustment_too_far_from_original"})
            else:
                adjustments[idx] = {"bbox_xyxy": bbox, "confidence": score, "raw": item}
                actions.append({"type": "adjust_box", "yolo_index": idx, "bbox_xyxy": bbox, "confidence": score, "iou_with_original": overlap})

    relabels: dict[int, dict[str, Any]] = {}
    for item in as_list(fusion_hints.get("relabel_yolo")):
        if not isinstance(item, dict):
            continue
        idx_set = extract_yolo_indices(item)
        class_value = item.get("class_id") if item.get("class_id") is not None else item.get("to_class_id")
        class_value = class_value if class_value is not None else item.get("new_class_id")
        class_id = coerce_int(class_value)
        label = item.get("label") or item.get("to_label") or item.get("new_label")
        if not allow_relabel:
            for idx in idx_set:
                actions.append({"type": "reject_relabel", "yolo_index": idx, "class_id": class_id, "label": label, "reason": "fusion_policy_disallows_relabel", "policy": policy})
            continue
        if class_id is None and label in (None, ""):
            continue
        for idx in idx_set:
            yolo_conf = yolo_confidence(idx)
            if yolo_conf is not None and yolo_conf > relabel_max_yolo_conf:
                actions.append({"type": "reject_relabel", "yolo_index": idx, "class_id": class_id, "label": label, "yolo_confidence": yolo_conf, "max_yolo_confidence": relabel_max_yolo_conf, "reason": "yolo_confidence_too_high"})
                continue
            relabels[idx] = {"class_id": class_id, "label": label, "raw": item}
            actions.append({"type": "relabel", "yolo_index": idx, "class_id": class_id, "label": label, "yolo_confidence": yolo_conf})

    explicit_adds: list[dict[str, Any]] = []
    explicit_open_world_adds: list[dict[str, Any]] = []
    if allow_add:
        for item in as_list(fusion_hints.get("add_vlm_detections")):
            resolved = resolve_proposal_reference(item, proposals_by_id)
            if resolved is not None:
                explicit_adds.append(resolved)
        for item in as_list(fusion_hints.get("add_open_world_detections")):
            resolved = resolve_proposal_reference(item, proposals_by_id)
            if resolved is not None:
                explicit_open_world_adds.append(resolved)
    for proposal in vlm_proposals:
        action = str(proposal.get("coco_eval_action") or proposal.get("open_world_action") or "").lower()
        if action in {"add", "add_vlm", "add_vlm_detection", "new", "insert"}:
            if allow_add and (proposal_coco_class_id(proposal) is not None or not open_world_enabled):
                explicit_adds.append(proposal)
            elif allow_add and open_world_enabled:
                explicit_open_world_adds.append(proposal)
        elif action in {"open_world_add", "open-world-add", "open_world", "discover", "novel"} and allow_add and open_world_enabled:
            explicit_open_world_adds.append(proposal)
        elif action in {"suppress", "drop", "remove"}:
            for idx in extract_yolo_indices(proposal):
                score = coerce_float(proposal.get("confidence"))
                allowed, reason = can_suppress(idx, score)
                if allow_suppress and allowed:
                    suppress_indices.add(idx)
                    actions.append({"type": "suppress", "yolo_index": idx, "proposal_id": proposal.get("proposal_id"), "confidence": score, "yolo_confidence": yolo_confidence(idx), "reason": "vlm_proposal_action"})
                else:
                    actions.append({"type": "reject_suppress", "yolo_index": idx, "proposal_id": proposal.get("proposal_id"), "confidence": score, "yolo_confidence": yolo_confidence(idx), "reason": reason if allow_suppress else "fusion_policy_disallows_suppress", "policy": policy if not allow_suppress else None})

    fused_predictions: list[dict[str, Any]] = []
    for item in yolo_boxes:
        idx = int(item["index"])
        if idx in suppress_indices:
            continue
        prediction = {
            "source": "yolo",
            "index": idx,
            "class_id": item.get("class_id"),
            "label": item.get("label"),
            "confidence": item.get("confidence"),
            "bbox_xyxy": item.get("bbox_xyxy"),
            "coco_bbox_xywh": xyxy_to_xywh(item["bbox_xyxy"]),
            "action": "keep",
        }
        if idx in adjustments:
            prediction["bbox_xyxy"] = adjustments[idx]["bbox_xyxy"]
            prediction["coco_bbox_xywh"] = xyxy_to_xywh(adjustments[idx]["bbox_xyxy"])
            prediction["action"] = "adjusted"
        if idx in relabels:
            if relabels[idx].get("class_id") is not None:
                prediction["class_id"] = relabels[idx]["class_id"]
            if relabels[idx].get("label") not in (None, ""):
                prediction["label"] = relabels[idx]["label"]
            prediction["action"] = "relabelled" if prediction["action"] == "keep" else f"{prediction['action']}+relabelled"
        fused_predictions.append(prediction)

    seen_adds: set[str] = set()
    for proposal in explicit_adds:
        key = proposal_key(proposal)
        if key in seen_adds:
            continue
        seen_adds.add(key)
        bbox = valid_xyxy(proposal.get("bbox_xyxy") or proposal.get("bbox") or proposal.get("xyxy"))
        confidence = coerce_float(proposal.get("confidence"))
        class_value = proposal.get("class_id") if proposal.get("class_id") is not None else proposal.get("category_id")
        class_id = coerce_int(class_value)
        if bbox is None or class_id is None:
            warnings.append(f"Skipped VLM proposal without usable bbox/class_id: {proposal.get('proposal_id', key)}")
            continue
        allowed, reason = can_add(proposal)
        if not allowed or confidence is None or confidence < add_min:
            actions.append({"type": "reject_add", "proposal_id": proposal.get("proposal_id"), "confidence": confidence, "reason": reason if not allowed else "confidence_below_threshold", "threshold": add_min if confidence is None or confidence < add_min else None})
            continue
        fused_predictions.append({"source": "vlm", "proposal_id": proposal.get("proposal_id"), "class_id": class_id, "label": proposal.get("label"), "confidence": round(confidence, 4), "bbox_xyxy": bbox, "coco_bbox_xywh": xyxy_to_xywh(bbox), "bbox_quality": proposal.get("bbox_quality", "estimated"), "action": "added", "linked_yolo_indices": sorted(extract_yolo_indices(proposal))})
        actions.append({"type": "add", "proposal_id": proposal.get("proposal_id"), "confidence": confidence})

    seen_open_world_adds: set[str] = set()
    for proposal in explicit_open_world_adds:
        key = proposal_key(proposal)
        if key in seen_open_world_adds:
            continue
        seen_open_world_adds.add(key)
        bbox = valid_xyxy(proposal.get("bbox_xyxy") or proposal.get("bbox") or proposal.get("xyxy"))
        confidence = coerce_float(proposal.get("confidence"))
        open_label = proposal_open_label(proposal)
        class_id = proposal_coco_class_id(proposal)
        coco_label = proposal_coco_label(proposal)
        if bbox is None or open_label in (None, ""):
            warnings.append(f"Skipped open-world VLM proposal without usable bbox/label: {proposal.get('proposal_id', key)}")
            continue
        relabel_hit = None
        if open_world_iou_relabel_enabled:
            for yolo_item in yolo_boxes:
                yolo_bbox = yolo_item.get("bbox_xyxy")
                iou = box_iou_xyxy(yolo_bbox, bbox) if isinstance(yolo_bbox, list) else 0.0
                if iou >= open_world_iou_relabel_threshold:
                    yolo_conf = coerce_float(yolo_item.get("confidence"))
                    if yolo_conf is not None and yolo_conf <= open_world_iou_relabel_max_yolo_conf:
                        relabel_hit = {"yolo_index": yolo_item.get("index"), "yolo_label": yolo_item.get("label"), "yolo_confidence": yolo_conf, "iou": iou}
                        break
        if relabel_hit is not None:
            open_world_predictions.append({"source": "vlm_open_world", "proposal_id": proposal.get("proposal_id"), "open_label": open_label, "label": coco_label or open_label, "class_id": class_id, "confidence": round(confidence or 0.0, 4), "bbox_xyxy": bbox, "bbox_quality": proposal.get("bbox_quality", "estimated"), "action": "open_world_relabelled", "linked_yolo_indices": sorted(extract_yolo_indices(proposal)), "ontology_aliases": as_list(proposal.get("ontology_aliases")), "relabelled_from": relabel_hit})
            actions.append({"type": "relabel_open_world", "proposal_id": proposal.get("proposal_id"), "open_label": open_label, "yolo_index": relabel_hit["yolo_index"], "yolo_label": relabel_hit["yolo_label"], "iou": relabel_hit["iou"], "confidence": confidence})
            continue
        allowed, reason = can_add(proposal)
        if not allowed or confidence is None or confidence < open_world_add_min:
            actions.append({"type": "reject_open_world_add", "proposal_id": proposal.get("proposal_id"), "confidence": confidence, "reason": reason if not allowed else "open_world_confidence_below_threshold", "threshold": open_world_add_min if confidence is None or confidence < open_world_add_min else None})
            continue
        record = {"source": "vlm_open_world", "proposal_id": proposal.get("proposal_id"), "open_label": open_label, "label": coco_label or open_label, "class_id": class_id, "confidence": round(confidence, 4), "bbox_xyxy": bbox, "bbox_quality": proposal.get("bbox_quality", "estimated"), "action": "open_world_added", "linked_yolo_indices": sorted(extract_yolo_indices(proposal)), "ontology_aliases": as_list(proposal.get("ontology_aliases"))}
        if class_id is not None:
            record["coco_bbox_xywh"] = xyxy_to_xywh(bbox)
            fused_predictions.append(record)
            actions.append({"type": "add_open_world_mapped", "proposal_id": proposal.get("proposal_id"), "confidence": confidence, "class_id": class_id, "open_label": open_label})
        else:
            open_world_predictions.append(record)
            actions.append({"type": "add_open_world", "proposal_id": proposal.get("proposal_id"), "confidence": confidence, "open_label": open_label})

    image_id: int | str | None = None
    if image_path not in (None, ""):
        stem = Path(str(image_path)).stem
        image_id = int(stem) if stem.isdigit() else stem
    coco_records = []
    for prediction in fused_predictions:
        class_id = coerce_int(prediction.get("class_id"))
        score = coerce_float(prediction.get("confidence"))
        bbox = prediction.get("coco_bbox_xywh")
        if image_id is None or class_id is None or score is None or not isinstance(bbox, list):
            continue
        coco_records.append({"image_id": image_id, "category_id": class_id, "bbox": bbox, "score": round(score, 6), "source": prediction.get("source"), "action": prediction.get("action")})

    summary = {
        "yolo_boxes": len(yolo_boxes),
        "vlm_proposals": len(vlm_proposals),
        "kept": sum(1 for item in fused_predictions if item.get("source") == "yolo"),
        "suppressed": len(suppress_indices),
        "added": sum(1 for item in fused_predictions if item.get("source") == "vlm"),
        "open_world_added": len(open_world_predictions),
        "open_world_mapped_to_coco": sum(1 for item in fused_predictions if item.get("source") == "vlm_open_world"),
        "adjusted": len(adjustments),
        "relabelled": len(relabels),
        "fused_boxes": len(fused_predictions),
        "coco_records": len(coco_records),
    }
    return {
        "mode": mode,
        "policy": policy,
        "enabled": True,
        "strategy": "metric_safe_v1",
        "thresholds": {
            "add_confidence_min": add_min,
            "open_world_confidence_min": open_world_add_min,
            "add_require_unlinked": add_require_unlinked,
            "add_max_linked_yolo_confidence": add_max_linked_yolo_conf,
            "add_allowed_bbox_quality": sorted(add_allowed_bbox_quality),
            "suppress_confidence_min": suppress_min,
            "adjust_confidence_min": adjust_min,
            "suppress_max_yolo_confidence": suppress_max_yolo_conf,
            "relabel_max_yolo_confidence": relabel_max_yolo_conf,
            "adjust_min_iou": adjust_min_iou,
        },
        "summary": summary,
        "actions": actions,
        "predictions": fused_predictions,
        "coco_predictions_preview": coco_records,
        "open_world_predictions_preview": open_world_predictions,
        "warnings": warnings,
    }
