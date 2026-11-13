#!/usr/bin/env python3
"""OV-Distill · Concept discovery (Stage 1): derive query vocabularies from unlabeled images and build a manifest.

Tagging backends (--tagger):
  mock:<seed>          Deterministic pseudo-tagging (CI/smoke, no GPU required)
  file:<vocab.txt>     Fixed vocabulary (one phrase per line) shared by all images
  vlm:<base_url>[:<model>]  OpenAI-compatible endpoint; the VLM lists visible objects (comma-separated phrases)
  ram:<checkpoint>     Local RAM/Tag2Text tagging (requires the recognize-anything env, lazy import)

Manifest generation: scan an image directory → per-image query list → JSONL consumable by annotate.
Supports injecting negative queries per image (entries sampled from a global vocabulary that the
image does not contain) for rejection training.

Usage:
  python scripts/ov_distill/run_pipeline.py concepts \
      --images data/ov_pool/images --tagger file:data/ov_pool/vocab.txt \
      --neg-per-image 2 --out data/ov_pool/manifest.jsonl
"""

from __future__ import annotations

import base64
import hashlib
import json
import random
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

_MOCK_POOL = [
    "person",
    "car",
    "dog",
    "bicycle",
    "traffic light",
    "backpack",
    "chair",
    "cup",
    "bird",
    "truck",
]


class ImageTagger:
    """Base image tagger: image path in, list of visible concept phrases out."""

    name = "base"

    def tag(self, image_path: Path, timeout: int = 120) -> List[str]:
        """Return the list of concept phrases visible in the image (lowercased, deduplicated, order-preserving)."""
        raise NotImplementedError


def _dedup(phrases: Sequence[str]) -> List[str]:
    """Lowercase-normalize, strip whitespace, and deduplicate while preserving order."""
    seen: Set[str] = set()
    out: List[str] = []
    for p in phrases:
        p = p.strip().lower()
        if p and p not in seen:
            seen.add(p)
            out.append(p)
    return out


class MockTagger(ImageTagger):
    """Deterministic pseudo-tagger: draws 2-4 concepts from a fixed pool by image-name hash."""

    def __init__(self, seed: int = 0):
        self.name = f"mock-tagger-{seed}"
        self.seed = seed

    def tag(self, image_path: Path, timeout: int = 120) -> List[str]:
        """Draw phrases deterministically from the (seed, image-name) hash."""
        digest = hashlib.md5(f"{self.seed}|{image_path.name}".encode()).digest()
        n = 2 + digest[0] % 3
        start = digest[1] % len(_MOCK_POOL)
        return [_MOCK_POOL[(start + i) % len(_MOCK_POOL)] for i in range(n)]


class FileTagger(ImageTagger):
    """Fixed-vocabulary tagger: every image returns the same vocabulary (for known target domains)."""

    def __init__(self, vocab_path: Path):
        self.name = f"file-{vocab_path.name}"
        self.vocab = _dedup(vocab_path.read_text(encoding="utf-8").splitlines())
        if not self.vocab:
            raise ValueError(f"empty vocabulary file: {vocab_path}")

    def tag(self, image_path: Path, timeout: int = 120) -> List[str]:
        """Return the full fixed vocabulary."""
        return list(self.vocab)


class VLMTagger(ImageTagger):
    """VLM tagger: OpenAI-compatible endpoint prompted to list visible objects as comma-separated phrases."""

    PROMPT = (
        "List all distinct object categories visible in this image. "
        "Answer with lowercase comma-separated short noun phrases only, no numbering, no explanations."
    )

    def __init__(self, base_url: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.name = f"vlm-{model}"

    def tag(self, image_path: Path, timeout: int = 120) -> List[str]:
        """Query the VLM and parse the comma-separated phrase list."""
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
                        {"type": "text", "text": self.PROMPT},
                    ],
                }
            ],
            "max_tokens": 512,
            "temperature": 0.0,
        }
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode())
        text = body["choices"][0]["message"]["content"]
        return _dedup(text.replace("\n", ",").split(","))


class RAMTagger(ImageTagger):
    """Local RAM/Tag2Text tagger (lazy import, requires a GPU environment)."""

    def __init__(self, checkpoint: str):
        try:
            import torch
            from ram.models import ram
            from ram import inference_ram
        except ImportError as exc:  # pragma: no cover - heavyweight dependency
            raise ImportError("RAMTagger requires the recognize-anything environment (ram package + torch)") from exc
        self._torch = torch
        self._inference = inference_ram
        self._model = ram(pretrained=checkpoint, image_size=384, vit="swin_l")
        self._model.eval()
        self._model = self._model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.name = f"ram-{Path(checkpoint).stem}"

    def tag(self, image_path: Path, timeout: int = 120) -> List[str]:  # pragma: no cover - requires GPU
        """Run RAM inference and split the tag string."""
        from torchvision import transforms

        transform = transforms.Compose(
            [
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        from PIL import Image

        image = transform(Image.open(image_path).convert("RGB")).unsqueeze(0).to(next(self._model.parameters()).device)
        with self._torch.no_grad():
            tags, _ = self._inference(image, self._model)
        return _dedup(tags.split("|"))


def build_tagger(spec: str) -> ImageTagger:
    """Build a tagger from a spec string: mock:<seed> / file:<path> / vlm:<url>[:<model>] / ram:<ckpt>."""
    kind, _, rest = spec.partition(":")
    kind = kind.strip().lower()
    if kind == "mock":
        return MockTagger(seed=int(rest or 0))
    if kind == "file":
        return FileTagger(Path(rest))
    if kind == "vlm":
        base_url, _, model = rest.rpartition(":")
        if base_url.startswith(("http://", "https://")) and model and "/" not in model:
            return VLMTagger(base_url=base_url, model=model)
        return VLMTagger(base_url=rest, model="default")
    if kind == "ram":
        return RAMTagger(checkpoint=rest)
    raise ValueError(f"unknown tagger spec: {spec!r} (supported: mock/file/vlm/ram)")


def load_vocab(vocab_path: Optional[Path]) -> List[str]:
    """Load the global vocabulary for negative sampling; returns an empty list when absent."""
    if vocab_path is None or not vocab_path.exists():
        return []
    return _dedup(vocab_path.read_text(encoding="utf-8").splitlines())


def sample_negatives(
    positives: Sequence[str],
    global_vocab: Sequence[str],
    neg_per_image: int,
    rng: random.Random,
) -> List[str]:
    """Sample entries absent from the image's positives out of the global vocabulary as negative queries."""
    pos_set = {p.lower() for p in positives}
    pool = [v for v in global_vocab if v.lower() not in pos_set]
    if not pool or neg_per_image <= 0:
        return []
    k = min(neg_per_image, len(pool))
    return rng.sample(pool, k)


def run_concepts(
    images: Path,
    tagger_spec: str,
    out: Path,
    global_vocab: Optional[Path] = None,
    neg_per_image: int = 0,
    max_queries: int = 32,
    seed: int = 0,
    timeout: int = 120,
) -> Dict[str, int]:
    """Scan an image directory, tag each image, and emit an annotate-ready manifest JSONL.

    Args:
        images: Image directory (recursively scanned for common image extensions).
        tagger_spec: Tagger spec, see build_tagger.
        out: Output manifest JSONL path; the image field is written as an absolute path.
        global_vocab: Global vocabulary file (one phrase per line) for negative sampling.
        neg_per_image: Number of negative queries injected per image (0 disables).
        max_queries: Per-image query cap (negatives are appended after truncating positives).
        seed: Random seed for negative sampling.
        timeout: Per-request timeout in seconds (vlm backend).
    """
    tagger = build_tagger(tagger_spec)
    vocab = load_vocab(global_vocab)
    rng = random.Random(seed)

    image_files = sorted(p for p in images.rglob("*") if p.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        raise FileNotFoundError(f"image directory is empty or does not exist: {images}")
    out.parent.mkdir(parents=True, exist_ok=True)

    stats = {"images": 0, "positives": 0, "negatives": 0, "failed": 0}
    with out.open("w", encoding="utf-8") as f:
        for img in image_files:
            stats["images"] += 1
            try:
                positives = tagger.tag(img, timeout=timeout)[:max_queries]
            except Exception as exc:  # record and continue on per-image failure
                print(f"[concepts] tagging failed for {img}: {type(exc).__name__}: {exc}")
                stats["failed"] += 1
                continue
            negatives = sample_negatives(positives, vocab, neg_per_image, rng)
            queries = positives + negatives
            f.write(json.dumps({"image": str(img.resolve()), "queries": queries}, ensure_ascii=False) + "\n")
            stats["positives"] += len(positives)
            stats["negatives"] += len(negatives)
    return stats
