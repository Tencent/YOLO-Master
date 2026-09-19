from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file

from scripts.d1_build_feature_cache import build_cache, build_resource_report
from ultralytics.data.foundation_cache import CacheIdentity


class FakeExtractor:
    device = torch.device("cpu")

    def __call__(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        batch = images.shape[0]
        return {
            "layer_3": torch.ones(batch, 8, 4, 4),
            "layer_7": torch.ones(batch, 8, 4, 4) * 2,
            "layer_11": torch.ones(batch, 8, 4, 4) * 3,
        }


def test_builder_writes_three_fp16_levels(tmp_path: Path):
    image = tmp_path / "images" / "sample.jpg"
    image.parent.mkdir()
    Image.new("RGB", (32, 24), color="red").save(image)
    output = tmp_path / "cache"
    manifest = build_cache(
        image_paths=[image],
        dataset_root=tmp_path,
        output_dir=output,
        extractor=FakeExtractor(),
        identity=CacheIdentity("fake", "rev", (3, 7, 11), 32, "fp"),
        limit=1,
    )
    tensors = load_file(str(output / manifest["samples"][0]["cache_path"]))
    assert set(tensors) == {"layer_3", "layer_7", "layer_11"}
    assert all(value.dtype == torch.float16 for value in tensors.values())
    assert (output / "resource-report.json").is_file()
    assert (output / "interface-dimensions.json").is_file()


def test_resource_report_contains_admission_fields():
    report = build_resource_report(
        samples=[{"bytes": 1024, "extraction_seconds": 0.1}],
        peak_vram_bytes=2048,
        read_seconds=0.01,
    )
    assert report["sample_count"] == 1
    assert report["cache_total_bytes"] == 1024
    assert report["peak_vram_bytes"] == 2048
    assert report["sequential_read_method"] == "python-buffered-full-file-read-warm-cache"
    assert report["sequential_read_mb_s"] > 0
