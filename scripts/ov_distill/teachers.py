#!/usr/bin/env python3
"""OV-Distill · Teacher client abstraction and implementations.

Unified interface: TeacherClient.detect(image_path, queries) -> List[TeacherResult].
Coordinates are always absolute image pixels [x1, y1, x2, y2]. absent=True when
the query is not present in the image.

Teacher specs (comma-separated items of annotate --teachers):
  mock:<seed>                Deterministic pseudo-annotation teacher (smoke/CI, no GPU)
  gam:<base_url>[:<model>]   GAM-protocol OpenAI-compatible service (GroundingPI / GroundAnything)
  rexomni:<model_path>       Rex-Omni (requires pip-installed rex_omni, local weights)
  wevisone:<model_path>      WeVisOne-VL (direct transformers load of Qwen3VL weights, requires transformers>=4.57)
  groundingpi:<model_path>   GroundingPI (trust_remote_code direct load, GAM protocol, requires transformers>=5.7)
  qwenvl:<model_path>        Qwen2.5-VL-Instruct generic VLM grounding (local weights)
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

from PIL import Image


@dataclass
class TeacherResult:
    """Teacher output for one image and one query."""

    query: str
    boxes: List[List[float]] = field(default_factory=list)  # absolute pixels [x1, y1, x2, y2]
    absent: bool = False
    raw: str = ""

    def to_dict(self) -> dict:
        """Serialize to a JSONL-writable structure."""
        return {"query": self.query, "boxes": self.boxes, "absent": self.absent}


class TeacherClient:
    """Teacher client base class. `name` feeds the teacher field of output records and must be unique."""

    name = "base"

    def detect(self, image_path: Path, queries: Sequence[str], timeout: int = 120) -> List[TeacherResult]:
        """Run detection for a set of queries on one image, returning results aligned with `queries`."""
        raise NotImplementedError


class MockTeacher(TeacherClient):
    """Deterministic pseudo-annotation teacher: box placement is fixed by the (image, query) hash
    (consistent across teachers); the seed only introduces slight jitter."""

    def __init__(self, seed: int = 0, absent_prefix: str = "zz_absent"):
        self.name = f"mock-{seed}"
        self.seed = seed
        self.absent_prefix = absent_prefix

    def detect(self, image_path: Path, queries: Sequence[str], timeout: int = 120) -> List[TeacherResult]:
        """Generate pseudo-boxes consistent across teachers (jittered only by seed);
        queries starting with absent_prefix simulate abstention."""
        with Image.open(image_path) as im:
            w, h = im.size
        jitter = (self.seed % 5) * 0.5  # systematic per-teacher jitter (pixel-level, IoU still > 0.8)
        results = []
        for q in queries:
            if q.startswith(self.absent_prefix):
                results.append(TeacherResult(query=q, boxes=[], absent=True, raw="mock-none"))
                continue
            # Box geometry and count are seed-free so different mock teachers can vote-align on the same (image, query)
            digest = hashlib.md5(f"{image_path.name}|{q}".encode("utf-8")).digest()
            n = 1 + digest[0] % 2
            boxes = []
            for i in range(n):
                off = digest[1 + i * 4 : 5 + i * 4]
                cx = 0.2 + 0.6 * (off[0] / 255.0)
                cy = 0.2 + 0.6 * (off[1] / 255.0)
                bw = 0.1 + 0.2 * (off[2] / 255.0)
                bh = 0.1 + 0.2 * (off[3] / 255.0)
                x1 = max(0.0, (cx - bw / 2) * w) + jitter
                y1 = max(0.0, (cy - bh / 2) * h) + jitter
                x2 = min(float(w), (cx + bw / 2) * w) + jitter
                y2 = min(float(h), (cy + bh / 2) * h) + jitter
                boxes.append([x1, y1, x2, y2])
            results.append(TeacherResult(query=q, boxes=boxes, raw="mock"))
        return results


_GAM_BOX_RE = re.compile(
    r"<\|object_ref_start\|>(?P<label>.*?)<\|object_ref_end\|>"
    r"<\|box_start\|>(?P<body>.*?)<\|box_end\|>",
    re.DOTALL,
)
_GAM_COORD_RE = re.compile(r"<(\d{1,3})>")


def parse_gam_output(text: str, query: str, width: int, height: int) -> TeacherResult:
    """Parse GAM spatial-token protocol output (0-999 grid) into absolute-pixel boxes.

    Example output:
      <|object_ref_start|>car<|object_ref_end|><|box_start|><100><200><500><650><|box_end|>
      <|object_ref_start|>absent<|object_ref_end|><|box_start|>None<|box_end|>
    """
    boxes: List[List[float]] = []
    for match in _GAM_BOX_RE.finditer(text):
        body = match.group("body")
        if "None" in body:
            continue
        coords = [int(v) for v in _GAM_COORD_RE.findall(body)]
        if len(coords) == 4:
            x1, y1, x2, y2 = coords
            boxes.append([x1 / 999 * width, y1 / 999 * height, x2 / 999 * width, y2 / 999 * height])
    absent = not boxes and "None" in text
    return TeacherResult(query=query, boxes=boxes, absent=absent, raw=text)


def parse_gam_multi(text: str, width: int, height: int) -> Dict[str, List[List[float]]]:
    """Multi-label GAM parsing: label (lowercased) -> absolute-pixel box list; a None payload maps to an empty list.

    A single box_start/end block may contain multiple comma-separated 4-tuples (protocol: payload := tuple ("," tuple)*).
    """
    out: Dict[str, List[List[float]]] = {}
    for match in _GAM_BOX_RE.finditer(text):
        label = match.group("label").strip().lower()
        body = match.group("body")
        bucket = out.setdefault(label, [])
        if "None" in body:
            continue
        coords = [int(v) for v in _GAM_COORD_RE.findall(body)]
        for off in range(0, len(coords) - 3, 4):
            x1, y1, x2, y2 = coords[off : off + 4]
            if x2 <= x1 or y2 <= y1:
                continue
            bucket.append([x1 / 999 * width, y1 / 999 * height, x2 / 999 * width, y2 / 999 * height])
    return out


class GAMClient(TeacherClient):
    """GAM-protocol OpenAI-compatible service client (GroundingPI / GroundAnything, vLLM/SGLang serving)."""

    def __init__(self, base_url: str, model: str = "groundingpi"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"gam-{model}"

    @staticmethod
    def build_prompt(query: str) -> str:
        """Build the object-detection prompt (GAM protocol)."""
        return f"Locate all the instances that match the following categories: {query}."

    def _chat(self, image_path: Path, prompt: str, timeout: int) -> str:
        """Send a single-image request to the OpenAI-compatible endpoint and return the text output."""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64," + base64.b64encode(image_path.read_bytes()).decode()
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.0,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["choices"][0]["message"]["content"]

    def detect(self, image_path: Path, queries: Sequence[str], timeout: int = 120) -> List[TeacherResult]:
        """Query the GAM service per query and parse the spatial-token output."""
        with Image.open(image_path) as im:
            w, h = im.size
        return [parse_gam_output(self._chat(image_path, self.build_prompt(q), timeout), q, w, h) for q in queries]


class RexOmniClient(TeacherClient):
    """Rex-Omni local inference client (lazily loads the rex_omni package).

    Device fallback order: MPS (Apple Silicon), then CUDA, then CPU; attention
    implementation defaults to sdpa (flash_attention_2 is only available on NVIDIA).
    """

    def __init__(self, model_path: str = "IDEA-Research/Rex-Omni", backend: str = "transformers"):
        import torch
        from rex_omni import RexOmniWrapper  # lazy import; CI does not need it installed

        if torch.backends.mps.is_available():
            device_map = "mps"
        elif torch.cuda.is_available():
            device_map = "auto"
        else:
            device_map = "cpu"
        self._rex = RexOmniWrapper(
            model_path=model_path,
            backend=backend,
            max_tokens=2048,
            temperature=0.0,
            top_p=0.05,
            top_k=1,
            torch_dtype=torch.bfloat16 if device_map != "cpu" else torch.float32,
            attn_implementation="sdpa",
            device_map=device_map,
        )
        self.name = f"rexomni-{Path(model_path).name}"

    def detect(self, image_path: Path, queries: Sequence[str], timeout: int = 120) -> List[TeacherResult]:
        """Call the Rex-Omni detection task and flatten results into the unified structure."""
        image = Image.open(image_path).convert("RGB")
        outputs = self._rex.inference(images=image, task="detection", categories=list(queries))
        preds = outputs[0]["extracted_predictions"]
        results = []
        for q in queries:
            items = preds.get(q, []) if isinstance(preds, dict) else []
            boxes = [list(map(float, it["coords"])) for it in items if it.get("type") == "box"]
            results.append(TeacherResult(query=q, boxes=boxes, absent=not boxes))
        return results


class WeVisOneClient(TeacherClient):
    """WeVisOne-VL local inference client (direct transformers load of Qwen3VL weights, no official package).

    Protocol identical to the official ``wevisone_vl`` (Tencent/WeVisOne-VL @ main):
    the prompt uses the detection task template; output is ``<ref>label</ref><box><x1><y1><x2><y2></box>``
    with 0-1000 normalized integer coordinate tokens, and the entire reply is ``None`` when the target is absent.

    Requires transformers >= 4.57 (qwen3_vl); when this conflicts with Rex-Omni's
    4.51.x requirement, run annotate in separate environments and passes
    (resume keys on image+teacher+query, so appending is safe).
    """

    _REF_RE = re.compile(
        r"<ref>(.*?)</ref>\s*<(box|point|polygon)>(.*?)(?:</(?:box|point|polygon)>|<ref>|$)",
        re.DOTALL | re.IGNORECASE,
    )
    _COORD_RE = re.compile(r"<(\d{1,4})>")

    def __init__(self, model_path: str = "tencent/WeVisOne-VL-2B"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # lazy import

        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        elif torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        else:
            device, dtype = "cpu", torch.float32
        # Load on CPU first, then move to the target device: direct device_map="mps"
        # loading segfaults on some torch/transformers combinations
        # (reproduced with torch 2.14 + transformers 5.21 @ macOS)
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        self._processor = AutoProcessor.from_pretrained(model_path)
        # For paths like .../isLinXu--WeVisOne-VL-2B/snapshots/master, take the meaningful directory name
        parts = [p for p in Path(model_path).parts if p not in {"", "snapshots", "master"}]
        self.name = f"wevisone-{parts[-1] if parts else Path(model_path).name}"

    @staticmethod
    def build_prompt(queries: Sequence[str]) -> str:
        """Official detection task template (prompts.py: build_prompt("detection", labels))."""
        labels = [str(q).strip() for q in queries if str(q).strip()]
        return "Locate all the instances that match the following description: " + ". ".join(labels) + "."

    @classmethod
    def parse_output(cls, text: str, width: int, height: int) -> Dict[str, List[List[float]]]:
        """Parse the <ref>/<box> structure, mapping 0-1000 normalized coordinates back to absolute pixels.

        Matches the behavior of the official visualization.parse_predictions;
        an entire reply of None returns an empty table.
        """
        text = (text or "").split("<|im_end|>", 1)[0].split("<|endoftext|>", 1)[0]
        if text.strip().lower() in {"", "none"}:
            return {}
        predictions: Dict[str, List[List[float]]] = {}
        for label, shape, body in cls._REF_RE.findall(text):
            label = label.strip().lower() or "object"
            if shape.lower() != "box":
                continue
            bucket = predictions.setdefault(label, [])
            for segment in body.split(","):
                values = [int(v) for v in cls._COORD_RE.findall(segment)]
                for offset in range(0, len(values) - 3, 4):
                    x1, y1, x2, y2 = values[offset : offset + 4]
                    if x2 <= x1 or y2 <= y1:
                        continue
                    bucket.append([x1 / 1000 * width, y1 / 1000 * height, x2 / 1000 * width, y2 / 1000 * height])
            if not bucket:
                predictions.pop(label, None)
        return predictions

    def detect(self, image_path: Path, queries: Sequence[str], timeout: int = 120) -> List[TeacherResult]:
        """Detect all queries on one image in a single forward pass."""
        image = Image.open(image_path).convert("RGB")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.build_prompt(queries)},
                ],
            },
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
        output_ids = generated[:, inputs["input_ids"].shape[1] :]
        raw = self._processor.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        parsed = self.parse_output(raw, image.width, image.height)
        results = []
        for q in queries:
            boxes = parsed.get(q.strip().lower(), [])
            results.append(TeacherResult(query=q, boxes=boxes, absent=not boxes, raw=raw))
        return results


class GroundingPIClient(TeacherClient):
    """GroundingPI local inference client (trust_remote_code direct weight load, GAM-protocol dense bbox).

    Protocol identical to the official grounding_pi.protocol: the prompt uses the dense
    bbox template (labels separated by ``</c>``), output is ``<|object_ref_start|>phrase<|object_ref_end|>``
    ``<|box_start|><x1><y1><x2><y2><|box_end|>`` on a 0-999 normalized grid, with ``None`` for abstention.

    The weight snapshot ships its own modeling/configuration/processing remote code,
    so the official repo's bundled transformers fork is not needed;
    requires transformers >= 5.7 (verified on 5.21 in the dev environment).
    """

    def __init__(self, model_path: str = "GroundingPI/GroundingPI"):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor  # lazy import

        if torch.backends.mps.is_available():
            device, dtype = "mps", torch.bfloat16
        elif torch.cuda.is_available():
            device, dtype = "cuda", torch.bfloat16
        else:
            device, dtype = "cpu", torch.float32
        # Load on CPU first, then move to the target device (same segfault workaround as WeVisOneClient)
        self._model = AutoModelForImageTextToText.from_pretrained(
            model_path, trust_remote_code=True, torch_dtype=dtype, attn_implementation="sdpa"
        ).to(device)
        self._processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        parts = [p for p in Path(model_path).parts if p not in {"", "snapshots", "master"}]
        self.name = f"groundingpi-{parts[-1] if parts else Path(model_path).name}"

    @staticmethod
    def build_prompt(queries: Sequence[str]) -> str:
        """Official dense bbox template (protocol.build_dense_bbox_prompt)."""
        labels = [str(q).strip() for q in queries if str(q).strip()]
        return "Locate all the instances that match the following categories: " + "</c>".join(labels) + "."

    def detect(self, image_path: Path, queries: Sequence[str], timeout: int = 120) -> List[TeacherResult]:
        """Detect all queries on one image in a single forward pass."""
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.build_prompt(queries)},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[text], images=[image], padding=True, return_tensors="pt").to(self._model.device)
        generated = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True)
        output_ids = generated[:, inputs["input_ids"].shape[1] :]
        raw = self._processor.batch_decode(output_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]

        parsed = parse_gam_multi(raw, image.width, image.height)
        results = []
        for q in queries:
            boxes = parsed.get(q.strip().lower(), [])
            results.append(TeacherResult(query=q, boxes=boxes, absent=not boxes, raw=raw))
        return results


class QwenVLGroundingClient(TeacherClient):
    """Qwen2.5-VL-Instruct open-vocabulary grounding client (lazily loads transformers).

    The model outputs bbox_2d coordinates in JSON (in the processor-rescaled image
    coordinate system); this client derives the rescale factor from image_grid_thw
    and maps coordinates back to absolute pixels of the original image.
    """

    def __init__(self, model_path: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration  # lazy import

        if torch.backends.mps.is_available():
            device_map, dtype = "mps", torch.bfloat16
        elif torch.cuda.is_available():
            device_map, dtype = "auto", torch.bfloat16
        else:
            device_map, dtype = "cpu", torch.float32
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=dtype, attn_implementation="sdpa", device_map=device_map
        )
        self._processor = AutoProcessor.from_pretrained(model_path, use_fast=False)
        self.name = f"qwenvl-{Path(model_path).name}"

    @staticmethod
    def build_prompt(queries: Sequence[str]) -> str:
        """Build the multi-category grounding prompt."""
        cats = ", ".join(f'"{q}"' for q in queries)
        return (
            f"Locate every instance that belongs to the following categories: {cats}. "
            'Report bbox coordinates in JSON format: [{"bbox_2d": [x1, y1, x2, y2], "label": "category"}]. '
            "If a category is not present in the image, do not output anything for it."
        )

    @staticmethod
    def parse_output(text: str) -> Dict[str, List[List[float]]]:
        """Extract boxes from the model output and aggregate by label (rescaled coordinate system).

        Tries full JSON-array parsing first; on failure (e.g. output truncated by
        max_new_tokens) falls back to extracting complete box objects one by one with
        a regex, so truncation only drops trailing objects, not the whole image.
        """
        out: Dict[str, List[List[float]]] = {}
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                items = json.loads(match.group(0))
                for it in items if isinstance(items, list) else []:
                    label = str(it.get("label", "")).strip().lower()
                    box = it.get("bbox_2d")
                    if label and isinstance(box, list) and len(box) == 4:
                        out.setdefault(label, []).append([float(v) for v in box])
                return out
            except json.JSONDecodeError:
                pass
        # Truncation tolerance: extract complete objects one by one
        obj_re = re.compile(
            r'\{\s*"bbox_2d"\s*:\s*\[\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,'
            r'\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
        )
        for m in obj_re.finditer(text):
            label = m.group(5).strip().lower()
            box = [float(m.group(i)) for i in range(1, 5)]
            out.setdefault(label, []).append(box)
        return out

    def detect(self, image_path: Path, queries: Sequence[str], timeout: int = 120) -> List[TeacherResult]:
        """Detect all queries on one image in a single forward pass, mapping coordinates back to the original image."""
        from qwen_vl_utils import process_vision_info  # lazy import

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": self.build_prompt(queries)},
                ],
            }
        ]
        text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self._processor(
            text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
        ).to(self._model.device)
        with Image.open(image_path) as im:
            orig_w, orig_h = im.size
        grid = inputs["image_grid_thw"][0]  # (t, h_patches, w_patches), patch=14
        scale_x = orig_w / (int(grid[2]) * 14)
        scale_y = orig_h / (int(grid[1]) * 14)

        generated = self._model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        output_ids = generated[:, inputs.input_ids.shape[1] :]
        raw = self._processor.batch_decode(output_ids, skip_special_tokens=True)[0]

        parsed = self.parse_output(raw)
        results = []
        for q in queries:
            boxes = [
                [b[0] * scale_x, b[1] * scale_y, b[2] * scale_x, b[3] * scale_y] for b in parsed.get(q.lower(), [])
            ]
            results.append(TeacherResult(query=q, boxes=boxes, absent=not boxes, raw=raw))
        return results


def build_teacher(spec: str) -> TeacherClient:
    """Build a client from a teacher spec string.

    Specs look like ``mock:0`` / ``gam:http://127.0.0.1:8000/v1[:model]`` / ``rexomni:<path>`` / ``wevisone:<path>``.
    """
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "mock":
        return MockTeacher(seed=int(rest or 0))
    if kind == "gam":
        base_url, _, model = rest.rpartition(":")
        if base_url.startswith(("http://", "https://")) and model and "/" not in model:
            return GAMClient(base_url=base_url, model=model)
        return GAMClient(base_url=rest)
    if kind == "rexomni":
        return RexOmniClient(model_path=rest)
    if kind == "wevisone":
        return WeVisOneClient(model_path=rest)
    if kind == "groundingpi":
        return GroundingPIClient(model_path=rest)
    if kind == "qwenvl":
        return QwenVLGroundingClient(model_path=rest)
    raise ValueError(f"unknown teacher spec: {spec!r} (supported: mock/gam/rexomni/wevisone/groundingpi/qwenvl)")


def image_size(image_path: Path) -> tuple:
    """Read the image (width, height)."""
    with Image.open(image_path) as im:
        return im.size
