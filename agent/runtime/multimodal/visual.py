from __future__ import annotations

import base64
import mimetypes
from pathlib import Path
from typing import Any


def image_source_for_openai(source: Any, results_summary: list[dict[str, Any]], *, resolved_path) -> str | None:
    candidates: list[str] = []
    if source is not None:
        candidates.append(str(source))
    if results_summary and results_summary[0].get("path"):
        candidate = str(results_summary[0]["path"])
        if candidate not in candidates:
            candidates.append(candidate)
    for candidate in candidates:
        if candidate.startswith(("http://", "https://", "data:image/")):
            return candidate
        path = resolved_path(candidate)
        if path.exists() and path.is_file():
            return str(path)
    return None


def encode_image_reference_for_openai(image_ref: str, *, resolved_path, max_bytes: int = 20_000_000) -> dict[str, Any]:
    if image_ref.startswith(("http://", "https://", "data:image/")):
        return {"image_url": image_ref, "kind": "url" if image_ref.startswith(("http://", "https://")) else "data_url"}
    path = resolved_path(image_ref)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Image source for VLM is not a file or URL: {image_ref}")
    size = path.stat().st_size
    if size > max_bytes:
        raise ValueError(f"Image source is too large for inline VLM upload: {path} ({size} bytes)")
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"image_url": f"data:{mime_type};base64,{data}", "kind": "local_file", "path": str(path.resolve())}


def load_pillow_image(path: Path):
    from PIL import Image

    return Image.open(path)


def clamp_box_xyxy(box: list[float] | tuple[float, ...], width: int, height: int, margin: float = 0.0) -> list[int] | None:
    if len(box) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
    except Exception:
        return None
    dx = max(0.0, (x2 - x1) * float(margin))
    dy = max(0.0, (y2 - y1) * float(margin))
    x1 = max(0.0, x1 - dx)
    y1 = max(0.0, y1 - dy)
    x2 = min(float(width), x2 + dx)
    y2 = min(float(height), y2 + dy)
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))]


def normalize_detection_boxes(detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for item in detections:
        image_path = item.get("path")
        for idx, detection in enumerate(item.get("detections", []) or []):
            bbox = detection.get("xyxy") or detection.get("bbox_xyxy") or detection.get("bbox")
            if bbox is None:
                continue
            normalized.append(
                {
                    "image_path": image_path,
                    "index": int(detection.get("index", idx)),
                    "class_id": detection.get("class_id"),
                    "label": detection.get("label"),
                    "confidence": detection.get("confidence"),
                    "bbox_xyxy": [float(v) for v in bbox[:4]],
                }
            )
    return normalized


def render_marked_image(
    image_ref: str | Path,
    detections: list[dict[str, Any]],
    *,
    resolved_path,
    output_dir: Path,
    prefix: str,
    max_items: int = 24,
) -> dict[str, Any]:
    source_path = resolved_path(image_ref)
    output_dir.mkdir(parents=True, exist_ok=True)
    marked_path = output_dir / f"{prefix}-marked.jpg"
    image = load_pillow_image(source_path).convert("RGB")
    width, height = image.size
    draw = None
    try:
        from PIL import ImageDraw, ImageFont

        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
    except Exception:
        font = None
    normalized = normalize_detection_boxes(detections)[:max_items]
    if draw is not None:
        palette = ["#ff5f5f", "#3db7ff", "#7cff72", "#ffce3a", "#bf7cff", "#ff8d3a"]
        for item in normalized:
            bbox = clamp_box_xyxy(item["bbox_xyxy"], width, height, margin=0.01)
            if bbox is None:
                continue
            idx = int(item.get("index", 0))
            color = palette[idx % len(palette)]
            draw.rectangle(bbox, outline=color, width=3)
            label = f"#{idx} {item.get('label', 'obj')}"
            if item.get("confidence") is not None:
                label += f" {float(item['confidence']):.2f}"
            if font is not None:
                try:
                    tb = draw.textbbox((0, 0), label, font=font)
                    text_w = tb[2] - tb[0]
                    text_h = tb[3] - tb[1]
                except Exception:
                    text_w = len(label) * 6
                    text_h = 11
                text_x = bbox[0]
                text_y = max(0, bbox[1] - text_h - 4)
                draw.rectangle([text_x, text_y, min(width, text_x + text_w + 6), min(height, text_y + text_h + 4)], fill=color)
                draw.text((text_x + 3, text_y + 2), label, fill="black", font=font)
    image.save(marked_path, quality=92)
    return {"path": str(marked_path.resolve()), "source": str(source_path.resolve()), "boxes": len(normalized), "kind": "marked_image"}
