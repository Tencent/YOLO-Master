"""Data, preprocessing, teacher identity, and scratch parameter contracts."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch
import yaml

from ultralytics.nn.foundation_detection_model import D1FoundationDetectionModel
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[1]


CONFIG = ROOT / "ultralytics/cfg/experiments/d1/dinov3-vits16-coco2017.yaml"


CONTRACT = ROOT / "experiments/d1/manifests/experiment-contract.json"


SCRIPT = ROOT / "scripts/d1/prepare_coco.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("prepare_coco", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _walk_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _walk_strings(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk_strings(nested)


def test_recipe_locks_full_coco_and_vits16_without_random_augmentation() -> None:
    recipe = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    assert recipe["data"] == "coco.yaml"
    assert recipe["imgsz"] == 640
    assert recipe["seed"] == 0
    assert recipe["deterministic"] is True
    assert recipe["foundation_enabled"] is False
    assert recipe["foundation_teacher"] == "dinov3"
    assert recipe["foundation_model"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert recipe["foundation_teacher_dtype"] == "fp16"
    assert recipe["foundation_target_levels"] == ["p3", "p4", "p5"]
    for key in (
        "hsv_h",
        "hsv_s",
        "hsv_v",
        "degrees",
        "translate",
        "scale",
        "shear",
        "perspective",
        "flipud",
        "fliplr",
        "bgr",
        "mosaic",
        "mixup",
        "cutmix",
        "copy_paste",
        "erasing",
    ):
        assert recipe[key] == 0.0

    tracked_text = CONFIG.read_text(encoding="utf-8").lower()
    assert "coco8" not in tracked_text
    assert "coco-mini" not in tracked_text
    assert "/data/" not in tracked_text


def test_experiment_contract_locks_preprocessing_blocks_and_cache_schema() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))

    assert contract["schema_version"] == "d1-p0-contract-v1"
    assert contract["dataset"]["splits"] == {"train2017": 118287, "val2017": 5000}
    assert contract["dataset"]["class_count"] == 80
    assert contract["teacher"]["model_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert contract["features"] == {
        "grid_size": [40, 40],
        "hidden_size": 384,
        "output_blocks": [
            {"implementation_index": 3, "name": "block4", "ordinal": 4},
            {"implementation_index": 7, "name": "block8", "ordinal": 8},
            {"implementation_index": 11, "name": "block12", "ordinal": 12},
        ],
        "patch_size": 16,
        "raw_stride": 16,
    }
    assert contract["input"]["letterbox"] == {
        "auto": False,
        "center": True,
        "interpolation": "INTER_LINEAR",
        "padding_value": 114,
        "scale_fill": False,
        "scaleup": True,
        "stride": 32,
    }
    assert contract["input"]["normalize"]["mean"] == [0.485, 0.456, 0.406]
    assert contract["input"]["normalize"]["std"] == [0.229, 0.224, 0.225]
    assert contract["input"]["teacher_extra_crop"] is False
    assert contract["input"]["teacher_extra_resize"] is False
    assert contract["cache"]["schema_version"] == "d1-cache-v1"
    assert contract["cache"]["dtype"] == "float16"
    assert contract["cache"]["target_shard_bytes"] == 2 * 1024**3


def test_tracked_contract_has_no_host_paths_or_credentials() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    forbidden = ("/data/", "/root/", "10.210.", "password", "token", "authorization")
    for text in _walk_strings(contract):
        lowered = text.lower()
        assert not any(value in lowered for value in forbidden)


def test_split_list_is_sorted_stable_and_exact(tmp_path: Path) -> None:
    module = _load_script()
    module.EXPECTED_SPLITS = {"train2017": 3}
    image_dir = tmp_path / "images/train2017"
    image_dir.mkdir(parents=True)
    for name in ("000000000003.jpg", "000000000001.jpg", "000000000002.jpg"):
        (image_dir / name).write_bytes(name.encode())

    first = module.build_split_list(tmp_path, "train2017")
    second = module.build_split_list(tmp_path, "train2017")

    assert first == [
        "images/train2017/000000000001.jpg",
        "images/train2017/000000000002.jpg",
        "images/train2017/000000000003.jpg",
    ]
    assert first == second


@pytest.mark.parametrize("corrupt", [False, True])
def test_materialize_splits_preserves_provenance_and_fails_closed(tmp_path, corrupt):
    module = _load_script()
    module.EXPECTED_SPLITS = {"train2017": 2, "val2017": 1}
    data = tmp_path / "data"
    repo = tmp_path / "repo"
    manifests = repo / "experiments/d1/manifests"
    manifests.mkdir(parents=True)
    expected = {}
    for split, ids in {"train2017": [1, 2], "val2017": [3]}.items():
        directory = data / "images" / split
        directory.mkdir(parents=True)
        lines = []
        for image_id in ids:
            filename = f"{image_id:012d}.jpg"
            (directory / filename).write_bytes(b"fixture")
            lines.append(f"images/{split}/{filename}")
        expected[split] = {
            "count": len(lines),
            "sha256": module.hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest(),
        }
    if corrupt:
        expected["val2017"]["sha256"] = "0" * 64
    source = manifests / "coco2017-splits.json"
    source.write_text(json.dumps({"splits": expected}))
    original = source.read_bytes()
    if corrupt:
        with pytest.raises(ValueError, match="published split contract"):
            module.materialize_splits(data, repo)
        assert not list(manifests.glob("*.txt"))
    else:
        module.materialize_splits(data, repo)
        first = {p.name: p.read_bytes() for p in manifests.glob("*.txt")}
        module.materialize_splits(data, repo)
        assert first == {p.name: p.read_bytes() for p in manifests.glob("*.txt")}
        assert len(first) == 2
    assert source.read_bytes() == original


def test_split_overlap_is_compared_by_image_filename() -> None:
    module = _load_script()
    train = ["images/train2017/000000000001.jpg"]
    val = ["images/val2017/000000000001.jpg"]
    try:
        module.assert_disjoint_splits(train, val)
    except ValueError as exc:
        assert "000000000001.jpg" in str(exc)
    else:
        raise AssertionError("same COCO image id must not appear in both splits")


def test_manifest_writers_are_deterministic(tmp_path: Path) -> None:
    from scripts.d1.artifacts import write_json

    module = _load_script()
    lines = ["images/train2017/000000000001.jpg", "images/train2017/000000000002.jpg"]
    list_path = tmp_path / "split.txt"
    first_hash = module.write_lines(list_path, lines)
    first_bytes = list_path.read_bytes()
    second_hash = module.write_lines(list_path, list(reversed(list(reversed(lines)))))
    assert list_path.read_bytes() == first_bytes
    assert second_hash == first_hash

    manifest_path = tmp_path / "manifest.json"
    write_json(manifest_path, {"z": 1, "a": {"value": True}})
    first_bytes = manifest_path.read_bytes()
    write_json(manifest_path, {"a": {"value": True}, "z": 1})
    assert manifest_path.read_bytes() == first_bytes


def test_modelscope_vits16_contract_matches_expected_architecture() -> None:
    module = _load_script()
    assert module.MODEL_ID == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert module.EXPECTED_MODEL_CONFIG == {
        "hidden_size": 384,
        "model_type": "dinov3_vit",
        "num_attention_heads": 6,
        "num_hidden_layers": 12,
        "num_register_tokens": 4,
        "patch_size": 16,
    }
    assert module.MODEL_REVISION == "2e601320d0545509ab03374e2f8707f303e1de7a"
    assert module.MODEL_FILES["model.safetensors"] == (
        86_406_384,
        "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d",
    )


def test_teacher_verification_rejects_same_size_corruption(tmp_path, monkeypatch):
    module = _load_script()
    payloads = {
        "config.json": json.dumps(module.EXPECTED_MODEL_CONFIG).encode(),
        "model.safetensors": b"test-weights",
        "LICENSE.md": b"license",
        "README.md": b"readme",
    }
    for name, data in payloads.items():
        (tmp_path / name).write_bytes(data)
    monkeypatch.setattr(
        module,
        "MODEL_FILES",
        {name: (len(data), module.hashlib.sha256(data).hexdigest()) for name, data in payloads.items()},
    )
    assert module.verify_model(tmp_path, load_model=False)["model_loaded"] is False
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"X" + path.read_bytes()[1:])
    with pytest.raises(ValueError, match="SHA256"):
        module.verify_model(tmp_path, load_model=False)


def test_label_count_and_membership_are_checked(tmp_path):
    module = _load_script()
    directory = tmp_path / "labels/train2017"
    directory.mkdir(parents=True)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "instances_train2017.json").write_text("{}")
    label = directory / "a.txt"
    label.write_text("")
    lists = {"train2017": ["images/train2017/a.jpg", "images/train2017/b.jpg"]}
    assert module.verify_labels(tmp_path, lists, {"train2017": 1}) == {"train2017": 1}
    with pytest.raises(ValueError, match="count"):
        module.verify_labels(tmp_path, lists, {"train2017": 2})
    label.rename(directory / "unknown.txt")
    with pytest.raises(ValueError, match="membership"):
        module.verify_labels(tmp_path, lists, {"train2017": 1})


def test_prepare_rejects_removed_download_mode():
    module = _load_script()
    with pytest.raises(SystemExit):
        module.main(["--download"])


SCRATCH_CONFIG = ROOT / "ultralytics/cfg/models/26/yolo26-d1-scratch-total-l.yaml"


TEACHER_PARAMS = 21_596_544


def test_standard_topology_and_explicit_scale():
    cfg = YAML.load(SCRATCH_CONFIG)
    standard = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26.yaml")
    assert cfg["backbone"] == standard["backbone"]
    assert cfg["head"] == standard["head"]
    assert cfg["scale"] == "l"
    assert cfg["scales"] == {"l": [1.0, 0.9375, 512]}
    assert cfg["end2end"] is True and cfg["reg_max"] == 1


@pytest.mark.parametrize(
    ("nc", "expected_downstream", "expected_scratch"),
    [(10, 1_340_259, 23_032_340), (80, 1_404_839, 23_133_560)],
)
def test_total_match_forward_and_strict_reload(nc, expected_downstream, expected_scratch):
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        frozen_cfg = YAML.load(ROOT / "ultralytics/cfg/models/26/yolo26-d1-dinov3-latent-p5-bottleneck64-n.yaml")
        frozen_cfg["detect"]["nc"] = nc
        downstream = D1FoundationDetectionModel(frozen_cfg)
        assert sum(p.numel() for p in downstream.parameters()) == expected_downstream
        del downstream
        cfg = YAML.load(SCRATCH_CONFIG)
        cfg["nc"] = nc
        model = DetectionModel(cfg, verbose=False)
        assert sum(p.numel() for p in model.parameters()) == expected_scratch
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) == expected_scratch
        assert abs(expected_scratch / (TEACHER_PARAMS + expected_downstream) - 1) < 0.01
        assert model.stride.tolist() == [8, 16, 32]
        export_limit = 500 if nc == 10 else 300
        model.model[-1].max_det = export_limit
        shapes = []
        handle = model.model[-1].register_forward_pre_hook(
            lambda module, args: shapes.extend(tuple(x.shape) for x in args[0])
        )
        model.eval()
        try:
            with torch.inference_mode():
                prediction = model(torch.zeros(1, 3, 640, 640))
        finally:
            handle.remove()
        assert shapes == [(1, 240, 80, 80), (1, 480, 40, 40), (1, 480, 20, 20)]
        prediction = prediction[0] if isinstance(prediction, tuple) else prediction
        assert prediction.shape == (1, export_limit, 6)
        assert torch.isfinite(prediction).all()
        restored = DetectionModel(cfg, verbose=False)
        result = restored.load_state_dict(model.state_dict(), strict=True)
        assert not result.missing_keys and not result.unexpected_keys
    finally:
        torch.set_num_threads(previous_threads)


def test_final_comparison_budget_and_parameter_contract():
    from pathlib import Path

    from ultralytics.utils import YAML

    root = Path(__file__).resolve().parents[1]
    contract = YAML.load(root / "ultralytics/cfg/experiments/d1/paired-comparison.yaml")
    assert contract["seeds"] == [0, 1, 2]
    assert contract["world_size"] == 6
    assert contract["models"] == ["BN64", "SCRATCH"]
    assert {key: value["epochs"] for key, value in contract["datasets"].items()} == {"coco": 50, "visdrone": 120}
    from scripts.d1.runtime import TRAINING_PRECISION, VALIDATION_PRECISION

    assert contract["validation_precision"] == VALIDATION_PRECISION == "fp32-v1"
    assert contract["training_precision"] == TRAINING_PRECISION == "bf16-mixed-fp32-loss-v1"
    for value in contract["datasets"].values():
        assert value["global_batch"] % contract["world_size"] == 0
        assert abs(value["scratch_parameters"] / value["frozen_total_parameters"] - 1) < 0.01


def test_final_comparison_checkpoint_state_gate():
    import torch

    from scripts.d1.compare import compare_states

    keys = ("model", "ema", "optimizer", "scaler", "scheduler", "criterion", "optimizer_steps", "ema_updates", "ranks")
    left = {key: {"value": torch.tensor([1.0, 2.0])} for key in keys}
    right = {key: {"value": torch.tensor([1.0, 2.0])} for key in keys}
    assert compare_states(left, right) == []
    right["model"]["value"][0] += 1
    assert compare_states(left, right) == ["model/value"]
    del right["scaler"]
    assert "scaler/missing" in compare_states(left, right)


def test_final_comparison_keeps_full_schedule_for_gate(tmp_path):
    from scripts.d1.compare import training_command

    plan = {
        "contract": {"datasets": {"coco": {"global_batch": 384, "epochs": 50}}},
        "data": {"coco": {"splits": {"train": {"cache": "train"}, "val": {"cache": "val"}}}},
        "device": "0,1,2,3,4,5",
    }
    command = training_command(plan, "coco", "SCRATCH", tmp_path, tmp_path / "data.yaml", window=2)
    assert command[command.index("--epochs") + 1] == "50"
    assert command[command.index("--window") + 1] == "2"
    assert "--train-cache" not in command
    assert "--approved" in command and "--telemetry" in command
