"""Cached datasets, trainer/validator integration, and distributed final evaluation."""

from __future__ import annotations

import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from ultralytics.data.d1_cache import (
    D1_FEATURE_NAMES,
    D1_FEATURE_SHAPE,
    D1FeatureBatch,
    D1FeatureCacheDataset,
)
from ultralytics.models.yolo.detect import (
    D1FoundationDetectionTrainer,
    D1FoundationDetectionValidator,
    foundation_train,
)
from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.foundation.cache import FeatureCacheReader, FeatureCacheWriter, sha256_bytes
from ultralytics.utils import DEFAULT_CFG_DICT, YAML
from ultralytics.utils.errors import MoERouterError


def cache_contract(**updates):
    contract = {
        "schema_version": "d1-cache-v1",
        "model_id": "local/test-dinov3-vits16",
        "teacher_weights_sha256": "a" * 64,
        "preprocessing_sha256": "b" * 64,
        "output_layers": [4, 8, 12],
        "feature_names": list(D1_FEATURE_NAMES),
        "dtype": "float16",
        "expected_shape": list(D1_FEATURE_SHAPE),
    }
    contract.update(updates)
    return contract


def sample_features(value: float) -> dict[str, torch.Tensor]:
    return {
        name: torch.full(D1_FEATURE_SHAPE, value + index, dtype=torch.float32)
        for index, name in enumerate(D1_FEATURE_NAMES)
    }


def data_config() -> dict:
    return {"names": {index: str(index) for index in range(80)}, "nc": 80, "channels": 3}


def hyp_config() -> SimpleNamespace:
    return SimpleNamespace(**DEFAULT_CFG_DICT)


def build_fixture(tmp_path: Path, count: int = 2) -> tuple[Path, Path, Path]:
    data_root = tmp_path / "coco"
    image_dir = data_root / "images" / "train2017"
    label_dir = data_root / "labels" / "train2017"
    cache_dir = tmp_path / "cache"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    paths = []
    with FeatureCacheWriter(
        cache_dir,
        split="train2017",
        contract=cache_contract(),
        target_shard_bytes=16 * 1024**2,
    ) as writer:
        for index in range(1, count + 1):
            image_id = f"{index:012d}"
            image_path = image_dir / f"{image_id}.jpg"
            height, width = (100, 200) if index % 2 else (200, 100)
            assert cv2.imwrite(str(image_path), np.full((height, width, 3), index, dtype=np.uint8))
            (label_dir / f"{image_id}.txt").write_text("0 0.5 0.5 0.5 0.4\n", encoding="utf-8")
            paths.append(str(image_path))
            writer.add(
                sample_id=f"train2017/{image_id}",
                split="train2017",
                image_path=f"images/train2017/{image_id}.jpg",
                image_sha256=sha256_bytes(f"image-{index}".encode()),
                features=sample_features(float(index)),
            )
    split_file = tmp_path / "train100.txt"
    split_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
    return data_root, split_file, cache_dir


def make_dataset(tmp_path: Path, **kwargs) -> D1FeatureCacheDataset:
    _data_root, split_file, cache_dir = build_fixture(tmp_path)
    return D1FeatureCacheDataset(
        img_path=str(split_file),
        cache_dir=cache_dir,
        data=data_config(),
        imgsz=640,
        batch_size=2,
        hyp=hyp_config(),
        **kwargs,
    )


def test_cache_dataset_maps_sample_id_without_loading_rgb(tmp_path, monkeypatch) -> None:
    dataset = make_dataset(tmp_path)
    monkeypatch.setattr(dataset, "load_image", lambda *_args, **_kwargs: pytest.fail("RGB image was loaded"))

    sample = dataset[0]

    assert sample["sample_id"] == "train2017/000000000001"
    assert tuple(sample["features"]) == D1_FEATURE_NAMES
    assert all(value.dtype == torch.float16 for value in sample["features"].values())
    assert all(tuple(value.shape) == D1_FEATURE_SHAPE for value in sample["features"].values())
    assert sample["ori_shape"] == (100, 200)
    assert sample["resized_shape"] == (640, 640)
    assert sample["ratio_pad"] == ((3.2, 3.2), (0, 160))
    assert torch.allclose(sample["bboxes"], torch.tensor([[0.5, 0.5, 0.5, 0.2]]))
    assert torch.equal(sample["cls"], torch.tensor([[0.0]]))


def test_trusted_cache_checks_each_shard_only_once_per_worker(tmp_path, monkeypatch) -> None:
    dataset = make_dataset(tmp_path, trusted_cache=True, max_open_shards=2, prefetch_factor=1)
    original = torch.isfinite
    calls = 0

    def counted_isfinite(value):
        nonlocal calls
        calls += 1
        return original(value)

    monkeypatch.setattr(torch, "isfinite", counted_isfinite)
    first = dataset[0]["features"]
    repeated = dataset[0]["features"]

    assert calls == len(D1_FEATURE_NAMES)
    assert all(torch.equal(first[name], repeated[name]) for name in D1_FEATURE_NAMES)
    assert dataset.feature_reader.max_open_shards == 2
    assert dataset.prefetch_factor == 1


def test_collate_exposes_virtual_640_shape_without_rgb_tensor(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batch = dataset.collate_fn([dataset[0], dataset[1]])

    assert isinstance(batch["features"], D1FeatureBatch)
    assert batch["img"] is batch["features"]
    assert batch["img"].shape == torch.Size((2, 3, 640, 640))
    assert all(tuple(value.shape) == (2, *D1_FEATURE_SHAPE) for value in batch["features"].values())
    assert torch.equal(batch["batch_idx"], torch.tensor([0.0, 1.0]))
    assert batch["sample_id"] == ("train2017/000000000001", "train2017/000000000002")


def test_d1_ddp_policy_uses_static_dense_graph():
    trainer = object.__new__(D1FoundationDetectionTrainer)
    assert trainer.resolve_ddp_policy() == (False, True)


def test_trainer_preprocess_moves_features_without_rgb_division(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    trainer = object.__new__(D1FoundationDetectionTrainer)
    trainer.device = torch.device("cpu")
    trainer.amp = False

    result = trainer.preprocess_batch(batch)

    assert result["features"] is result["img"]
    assert result["features"]["block4"].dtype == torch.float32
    assert torch.all(result["features"]["block4"][0] == 1.0)
    assert torch.all(result["features"]["block12"][0] == 3.0)
    assert result["cls"].device.type == "cpu"


def test_validator_uses_virtual_size_and_restores_original_coordinates(tmp_path) -> None:
    dataset = make_dataset(tmp_path)
    batch = dataset.collate_fn([dataset[0]])
    validator = object.__new__(D1FoundationDetectionValidator)
    validator.device = torch.device("cpu")
    validator.args = SimpleNamespace(quantize=None)
    prepared = validator.preprocess(batch)

    validator.device = torch.device("cpu")
    target = validator._prepare_batch(0, prepared)
    prediction = {
        "bboxes": torch.tensor([[160.0, 256.0, 480.0, 384.0]]),
        "conf": torch.tensor([0.9]),
        "cls": torch.tensor([0.0]),
    }
    scaled = validator.scale_preds(prediction, target)

    assert target["imgsz"] == torch.Size((640, 640))
    assert torch.allclose(target["bboxes"], prediction["bboxes"])
    assert torch.allclose(scaled["bboxes"], torch.tensor([[50.0, 30.0, 150.0, 70.0]]), atol=1e-4)


def test_online_mode_is_explicit_and_can_compare_with_cache(tmp_path) -> None:
    _data_root, split_file, cache_dir = build_fixture(tmp_path)
    reader = FeatureCacheReader(cache_dir)

    def provider(im_file: str):
        return reader.get(f"train2017/{Path(im_file).stem}")

    dataset = D1FeatureCacheDataset(
        img_path=str(split_file),
        cache_dir=cache_dir,
        data=data_config(),
        feature_mode="online",
        online_feature_provider=provider,
        imgsz=640,
        batch_size=2,
        hyp=hyp_config(),
    )

    assert dataset.feature_mode == "online"
    assert dataset.compare_online_with_cache(0) == {name: 0.0 for name in D1_FEATURE_NAMES}
    assert torch.equal(dataset[0]["features"]["block8"], reader.get(dataset.sample_ids[0])["block8"])


def test_online_mismatch_and_invalid_contract_fail_fast(tmp_path) -> None:
    _data_root, split_file, cache_dir = build_fixture(tmp_path)
    reader = FeatureCacheReader(cache_dir)

    def changed_provider(im_file: str):
        values = reader.get(f"train2017/{Path(im_file).stem}")
        values["block8"] = values["block8"] + 1
        return values

    dataset = D1FeatureCacheDataset(
        img_path=str(split_file),
        cache_dir=cache_dir,
        data=data_config(),
        feature_mode="online",
        online_feature_provider=changed_provider,
        imgsz=640,
        batch_size=2,
        hyp=hyp_config(),
    )
    with pytest.raises(ValueError, match="block8 differ"):
        dataset.compare_online_with_cache(0)

    bad_cache = tmp_path / "bad-cache"
    with FeatureCacheWriter(
        bad_cache,
        split="train2017",
        contract=cache_contract(expected_shape=[384, 20, 20]),
    ):
        pass
    with pytest.raises(ValueError, match="expected_shape"):
        D1FeatureCacheDataset(
            img_path=str(split_file),
            cache_dir=bad_cache,
            data=data_config(),
            imgsz=640,
            batch_size=2,
            hyp=hyp_config(),
        )


def test_missing_cache_sample_and_non_coco_parent_fail_fast(tmp_path) -> None:
    _data_root, split_file, cache_dir = build_fixture(tmp_path, count=1)
    extra = tmp_path / "coco" / "images" / "train2017" / "000000000002.jpg"
    extra.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(extra), np.zeros((32, 32, 3), dtype=np.uint8))
    label = tmp_path / "coco" / "labels" / "train2017" / "000000000002.txt"
    label.parent.mkdir(parents=True, exist_ok=True)
    label.write_text("", encoding="utf-8")
    split_file.write_text(split_file.read_text(encoding="utf-8") + f"{extra}\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="missing 1 dataset samples"):
        D1FeatureCacheDataset(
            img_path=str(split_file),
            cache_dir=cache_dir,
            data=data_config(),
            imgsz=640,
            batch_size=2,
            hyp=hyp_config(),
        )


def test_feature_batch_rejects_bad_shape_dtype_or_keys() -> None:
    good = {name: torch.zeros(1, *D1_FEATURE_SHAPE) for name in D1_FEATURE_NAMES}
    with pytest.raises(ValueError, match="keys"):
        D1FeatureBatch({"block4": good["block4"]})
    bad = dict(good)
    bad["block8"] = torch.zeros(1, 384, 20, 20)
    with pytest.raises(ValueError, match="shape"):
        D1FeatureBatch(bad)
    bad = dict(good)
    bad["block12"] = torch.zeros(1, *D1_FEATURE_SHAPE, dtype=torch.float16)
    with pytest.raises(ValueError, match="dtype"):
        D1FeatureBatch(bad)


def test_real_feature_cache_dataset_trainer_and_model(tmp_path) -> None:
    cache_value = os.environ.get("D1_WP2_CACHE")
    data_value = os.environ.get("D1_COCO_ROOT")
    if not cache_value or not data_value:
        pytest.skip("D1_WP2_CACHE and D1_COCO_ROOT are not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    cache_dir, data_root = Path(cache_value), Path(data_value)
    reader = FeatureCacheReader(cache_dir)
    records = [reader.records[sample_id] for sample_id in sorted(reader.records)[:2]]
    paths = [str(data_root / record["image_path"]) for record in records]
    split_file = tmp_path / "real-cache.txt"
    split_file.write_text("\n".join(paths) + "\n", encoding="utf-8")
    dataset = D1FeatureCacheDataset(
        img_path=str(split_file),
        cache_dir=cache_dir,
        data=data_config(),
        imgsz=640,
        batch_size=2,
        hyp=hyp_config(),
    )
    batch = dataset.collate_fn([dataset[0], dataset[1]])
    trainer = object.__new__(D1FoundationDetectionTrainer)
    trainer.device = torch.device("cuda")
    trainer.amp = True
    batch = trainer.preprocess_batch(batch)
    model = D1FoundationDetectionModel().cuda().eval()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        predictions, _raw = model(batch["img"])

    assert batch["features"]["block4"].dtype == torch.float16
    assert tuple(predictions.shape) == (2, 300, 6)
    assert torch.isfinite(predictions).all()


def test_real_feature_cache_one_batch_train_and_validate(tmp_path) -> None:
    cache_value = os.environ.get("D1_WP2_CACHE")
    data_value = os.environ.get("D1_COCO_ROOT")
    if not cache_value or not data_value:
        pytest.skip("D1_WP2_CACHE and D1_COCO_ROOT are not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    cache_dir, data_root = Path(cache_value), Path(data_value)
    reader = FeatureCacheReader(cache_dir)
    records = [reader.records[sample_id] for sample_id in sorted(reader.records)[:2]]
    split_file = tmp_path / "real-train-val.txt"
    split_file.write_text(
        "\n".join(str(data_root / record["image_path"]) for record in records) + "\n",
        encoding="utf-8",
    )
    data_yaml = tmp_path / "coco-two.yaml"
    YAML.save(
        data_yaml,
        {
            "path": str(data_root),
            "train": str(split_file),
            "val": str(split_file),
            "names": data_config()["names"],
            "channels": 3,
        },
    )
    repo_root = Path(__file__).resolve().parents[1]
    experiment = YAML.load(repo_root / "ultralytics/cfg/experiments/d1/dinov3-vits16-coco2017.yaml")
    trainer = D1FoundationDetectionTrainer(
        overrides={
            **experiment,
            "data": str(data_yaml),
            "epochs": 1,
            "batch": 2,
            "workers": 0,
            "device": 0,
            "amp": False,
            "project": str(tmp_path / "runs"),
            "name": "cached-one-batch",
            "exist_ok": True,
            "plots": False,
            "save": False,
            "val": True,
            "pretrained": False,
            "verbose": False,
            "close_mosaic": 0,
        },
        feature_caches={"train": cache_dir, "val": cache_dir},
    )

    trainer.train()

    assert trainer.optimizer_steps == 1
    assert trainer.loss_items.shape == torch.Size((7,))
    assert trainer.loss_names == (
        "box_loss",
        "cls_loss",
        "dfl_loss",
        "latent_balance_loss",
        "latent_z_loss",
        "latent_aux_loss",
        "mixture_aux_loss",
    )
    assert trainer.validator.seen == 2
    csv_header = trainer.csv.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert {
        "train/latent_balance_loss",
        "train/latent_z_loss",
        "train/latent_aux_loss",
        "train/mixture_aux_loss",
    }.issubset(csv_header)


class DummyModel(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def config_dict(self):
        return {}


def final_eval_fixture(tmp_path, monkeypatch, rank, result=None, failure=None):
    monkeypatch.setattr(foundation_train, "RANK", rank)
    monkeypatch.setattr(foundation_train, "LOCAL_RANK", rank)
    monkeypatch.setattr(foundation_train, "D1FoundationDetectionModel", DummyModel)
    monkeypatch.setattr(foundation_train, "torch_distributed_zero_first", lambda _: nullcontext())
    monkeypatch.setattr(foundation_train, "strip_optimizer", lambda *args, **kwargs: {})
    monkeypatch.setattr(foundation_train, "load_checkpoint", lambda *args, **kwargs: (DummyModel(), {}))
    trainer = object.__new__(foundation_train.D1FoundationDetectionTrainer)
    trainer.model = DummyModel()
    trainer.ema = SimpleNamespace(ema=DummyModel())
    trainer.args = SimpleNamespace(plots=False)
    trainer.device = torch.device("cpu")
    trainer.last = tmp_path / "last.pt"
    trainer.best = tmp_path / "best.pt"
    trainer.last.touch()
    trainer.best.touch()
    trainer._select_final_eval_checkpoints = lambda: ([trainer.best], [])
    trainer._reset_non_checkpoint_moe_runtime_state = lambda: None
    trainer.metrics = {"previous": 1.0}
    trainer._d1_final_eval_checkpoint = None
    calls = []
    callbacks = []

    class Validator:
        args = SimpleNamespace(plots=False)

        def __call__(self, *, trainer):
            calls.append(trainer._d1_final_eval_checkpoint)
            assert trainer.ema.ema is None
            if failure is not None:
                raise failure
            return result

    trainer.validator = Validator()
    trainer.run_callbacks = lambda event: callbacks.append((event, trainer._d1_final_eval_checkpoint))
    return trainer, calls, callbacks


@pytest.mark.parametrize("rank", [-1, 0, 1, 2, 3, 4, 5])
def test_final_eval_all_ranks_validate_only_main_consumes_metrics(tmp_path, monkeypatch, rank):
    result = {"fitness": 0.2, "metrics/mAP50-95(B)": 0.1} if rank <= 0 else None
    trainer, calls, callbacks = final_eval_fixture(tmp_path, monkeypatch, rank, result=result)
    original_model, original_ema = trainer.model, trainer.ema.ema
    trainer.final_eval()
    assert calls == [trainer.best]
    assert trainer.model is original_model
    assert trainer.ema.ema is original_ema
    assert trainer._d1_final_eval_checkpoint is None
    if rank <= 0:
        assert trainer.metrics == {"metrics/mAP50-95(B)": 0.1}
        assert callbacks == [("on_fit_epoch_end", trainer.best)]
    else:
        assert trainer.metrics == {"previous": 1.0}
        assert callbacks == []


@pytest.mark.parametrize("failure", [None, RuntimeError("validator failed")])
def test_final_eval_main_rank_fails_closed_and_restores_state(tmp_path, monkeypatch, failure):
    trainer, _, callbacks = final_eval_fixture(tmp_path, monkeypatch, 0, failure=failure)
    original_model, original_ema = trainer.model, trainer.ema.ema
    exception = TypeError if failure is None else RuntimeError
    with pytest.raises(exception):
        trainer.final_eval()
    assert trainer.model is original_model
    assert trainer.ema.ema is original_ema
    assert trainer._d1_final_eval_checkpoint is None
    assert callbacks == []


def test_final_eval_fallback_records_the_checkpoint_actually_evaluated(tmp_path, monkeypatch):
    trainer, _, callbacks = final_eval_fixture(tmp_path, monkeypatch, 0)
    trainer._select_final_eval_checkpoints = lambda: ([trainer.best, trainer.last], [])

    class Validator:
        args = SimpleNamespace(plots=False)

        def __call__(self, *, trainer):
            if trainer._d1_final_eval_checkpoint == trainer.best:
                raise MoERouterError("unhealthy best router")
            return {"fitness": 0.1}

    trainer.validator = Validator()
    trainer.final_eval()
    assert callbacks == [("on_fit_epoch_end", trainer.last)]
    assert trainer._d1_final_eval_checkpoint is None


@pytest.mark.parametrize("shape", [(100, 200), (200, 100), (333, 517), (721, 1281)])
def test_rgb_original_pixels_labels_and_inverse_geometry_match_cache(tmp_path, shape):
    from scripts.d1.rgb import ScratchValidator, build_dataset
    from ultralytics.data.augment import LetterBox

    data_root, split_file, cache_dir = build_fixture(tmp_path, count=1)
    image_path = data_root / "images/train2017/000000000001.jpg"
    pixels = np.random.default_rng(7).integers(0, 256, (*shape, 3), dtype=np.uint8)
    assert cv2.imwrite(str(image_path), pixels)
    args = hyp_config()
    rgb = build_dataset(args, data_config(), str(split_file), 1)
    cached = D1FeatureCacheDataset(
        img_path=str(split_file), cache_dir=cache_dir, data=data_config(), imgsz=640, batch_size=1, hyp=args
    )
    sample, reference = rgb[0], cached[0]
    assert sample["ori_shape"] == reference["ori_shape"] == shape
    assert sample["ratio_pad"] == reference["ratio_pad"]
    assert torch.equal(sample["cls"], reference["cls"])
    assert torch.allclose(sample["bboxes"], reference["bboxes"], atol=1e-7, rtol=0)
    assert sample["img"].shape == (3, 640, 640)
    original = cv2.imread(str(image_path))
    expected = LetterBox((640, 640), auto=False, scaleup=True)(image=original)
    expected = torch.from_numpy(np.ascontiguousarray(expected[..., ::-1].transpose(2, 0, 1)))
    assert torch.equal(sample["img"], expected)
    assert np.array_equal(rgb.load_image(0)[0], original)
    assert not rgb.rect and not rgb.augment and not rgb.cache

    validator = ScratchValidator(save_dir=tmp_path / "validation", args={"plots": False})
    validator.data = data_config()
    assert isinstance(validator.build_dataset(str(split_file)), type(rgb))
    batch = rgb.collate_fn([sample])
    target = validator._prepare_batch(0, batch)
    prediction = {"bboxes": target["bboxes"].clone(), "conf": torch.tensor([0.9]), "cls": sample["cls"].flatten()}
    restored = validator.scale_preds(prediction, target)
    height, width = shape
    expected_box = torch.tensor([[width * 0.25, height * 0.3, width * 0.75, height * 0.7]])
    assert torch.allclose(restored["bboxes"], expected_box, atol=1e-4, rtol=0)
    cached.feature_reader.close()


def test_rgb_empty_labels_and_fixed_size_guard(tmp_path):
    from scripts.d1.rgb import build_dataset

    data_root, split_file, _ = build_fixture(tmp_path, count=1)
    (data_root / "labels/train2017/000000000001.txt").write_text("")
    args = hyp_config()
    sample = build_dataset(args, data_config(), str(split_file), 1)[0]
    assert sample["bboxes"].shape == (0, 4) and sample["cls"].shape == (0, 1)
    args.imgsz = 320
    with pytest.raises(ValueError, match="imgsz=640"):
        build_dataset(args, data_config(), str(split_file), 1)
    args.imgsz, args.multi_scale = 640, 0.5
    with pytest.raises(ValueError, match="multi_scale=0"):
        build_dataset(args, data_config(), str(split_file), 1)


@pytest.mark.parametrize("nc,expected", [(10, 23_032_340), (80, 23_133_560)])
def test_rgb_total_model_forward_and_strict_resume(tmp_path, nc, expected):
    from scripts.d1.rgb import ScratchTrainer, audit_model

    trainer = object.__new__(ScratchTrainer)
    trainer.data = {"nc": nc, "names": dict(enumerate(map(str, range(nc)))), "channels": 3}
    trainer.args, trainer.resume = hyp_config(), False
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        model = trainer.get_model(verbose=False)
        assert audit_model(model)["total_parameters"] == expected
        assert model.names == trainer.data["names"]
        model.eval()
        model.set_head_attr(max_det=500 if nc == 10 else 300)
        with torch.inference_mode():
            prediction = model(torch.zeros(1, 3, 640, 640))[0]
        assert prediction.shape == (1, 500 if nc == 10 else 300, 6)
        assert torch.isfinite(prediction).all()
        with pytest.raises(ValueError, match="Pretrained weights"):
            trainer.get_model(weights=model, verbose=False)
        trainer.resume = True
        restored = trainer.get_model(cfg=model.yaml, weights=model, verbose=False)
        assert all(torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items())
        del restored
        model.register_buffer("unexpected_resume_key", torch.zeros(1))
        with pytest.raises(RuntimeError, match="Unexpected key"):
            trainer.get_model(cfg=model.yaml, weights=model, verbose=False)
        del model.unexpected_resume_key
        with torch.no_grad():
            next(model.parameters()).view(-1)[0] = float("nan")
        with pytest.raises(FloatingPointError, match="nonfinite state"):
            trainer.get_model(cfg=model.yaml, weights=model, verbose=False)
    finally:
        torch.set_num_threads(previous_threads)


def test_rgb_rejects_changed_architecture_and_uses_shared_lifecycle():
    from scripts.d1.rgb import MODEL_CFG, ScratchTrainer, scratch_config
    from ultralytics.models.yolo.detect.train import DetectionTrainer

    config = YAML.load(MODEL_CFG)
    config["scales"]["l"][1] = 1.0
    with pytest.raises(ValueError, match="exact registered"):
        scratch_config(config, nc=80)
    with pytest.raises(ValueError, match="nc=10 or nc=80"):
        scratch_config(nc=1)
    trainer = object.__new__(ScratchTrainer)
    assert trainer.resolve_ddp_policy() == (False, True)
    assert ScratchTrainer._setup_train is DetectionTrainer._setup_train
    assert ScratchTrainer.optimizer_step is DetectionTrainer.optimizer_step
    assert ScratchTrainer.get_dataloader is DetectionTrainer.get_dataloader
    trainer.device = torch.device("cpu")
    assert trainer.check_amp_compatibility() is False


@pytest.mark.parametrize("bad_amp", [False, True])
def test_rgb_amp_check_is_local_and_restores_training_mode(monkeypatch, bad_amp):
    from scripts.d1.rgb import ScratchTrainer
    from ultralytics.engine import trainer as trainer_module

    class LocalModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, image):
            self.calls += 1
            assert not self.training
            return torch.full((1, 300, 6), float("nan") if bad_amp and self.calls == 2 else 1.0), {}

    trainer = object.__new__(ScratchTrainer)
    trainer.device, trainer.model = torch.device("cuda"), LocalModel()
    linspace = torch.linspace
    monkeypatch.setattr(torch, "linspace", lambda *a, **kw: linspace(*a))
    monkeypatch.setattr(torch, "autocast", lambda *a, **kw: nullcontext())
    monkeypatch.setattr(trainer_module, "check_amp", lambda *a: pytest.fail("Unrelated AMP model requested"))
    assert trainer.check_amp_compatibility() is (not bad_amp)
    assert trainer.model.training and trainer.model.calls == 2


@pytest.mark.parametrize("dataset_kind", ["coco", "visdrone"])
def test_rgb_official_exports_match_cached_validator(tmp_path, dataset_kind):
    from scripts.d1.rgb import ExportRGBValidator
    from scripts.d1.train import ExportValidator
    from ultralytics.data.converter import coco80_to_coco91_class

    _, split_file, cache_dir = build_fixture(tmp_path, count=1)
    args = {"plots": False, "save_json": True, "conf": 0.001, "imgsz": 640}
    nc = 80 if dataset_kind == "coco" else 10
    model = SimpleNamespace(names=dict(enumerate(map(str, range(nc)))), end2end=True)
    rgb = ExportRGBValidator(save_dir=tmp_path / "rgb-export", args=args, dataset_kind=dataset_kind)
    cached = ExportValidator(
        save_dir=tmp_path / "cache-export", args=args, dataset_kind=dataset_kind, feature_cache=cache_dir
    )
    for validator in (rgb, cached):
        validator.data = {"val": str(split_file)}
        validator.init_metrics(model)
        assert validator.args.save_json and validator.eval_json({"sentinel": 1}) == {"sentinel": 1}
        assert validator.class_map == (coco80_to_coco91_class() if dataset_kind == "coco" else list(range(10)))
        prediction = {
            "bboxes": torch.tensor([[160.0, 256.0, 480.0, 384.0], [10.0, 0.0, 100.0, 150.0]]),
            "conf": torch.tensor([0.9, 0.8]),
            "cls": torch.tensor([nc - 1.0, 0.0]),
        }
        target = {
            "im_file": "000000000001.jpg",
            "imgsz": (640, 640),
            "ori_shape": (100, 200),
            "ratio_pad": ((3.2, 3.2), (0, 160)),
            "ignored_regions": [[0, 0, 200, 100]],
        }
        validator.pred_to_json(validator.scale_preds(prediction, target), target)
    assert rgb.jdict == cached.jdict
    assert rgb.jdict[0]["bbox"] == [50.0, 30.0, 100.0, 40.0]
    assert rgb.jdict[0]["category_id"] == (90 if dataset_kind == "coco" else 9)
    assert rgb.jdict[0]["image_id"] == (1 if dataset_kind == "coco" else "000000000001")
    assert rgb.degenerate_boxes_removed == cached.degenerate_boxes_removed == (dataset_kind == "visdrone")
    before = list(rgb.jdict)
    empty = {"bboxes": torch.empty(0, 4), "conf": torch.empty(0), "cls": torch.empty(0)}
    rgb.pred_to_json(empty, target)
    assert rgb.jdict == before


@pytest.mark.parametrize("score", [float("nan"), float("inf"), -0.1, 1.1])
def test_rgb_visdrone_rejects_nonfinite_export_and_invalid_dataset(tmp_path, score):
    from scripts.d1.rgb import ExportRGBValidator

    with pytest.raises(ValueError, match="dataset_kind"):
        ExportRGBValidator(dataset_kind="unknown")
    validator = ExportRGBValidator(save_dir=tmp_path / "export", args={"plots": False}, dataset_kind="visdrone")
    validator.data = {"val": "images"}
    validator.init_metrics(SimpleNamespace(names=dict(enumerate(map(str, range(10)))), end2end=True))
    bad = {
        "bboxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "conf": torch.tensor([score]),
        "cls": torch.tensor([0.0]),
    }
    with pytest.raises((FloatingPointError, ValueError), match="Nonfinite prediction|Invalid confidence"):
        validator.pred_to_json(bad, {"im_file": "000001.jpg"})


@pytest.mark.parametrize(
    "bad_box",
    [
        [493.523, 208.945, 492.187, 211.992],
        [498.984, 246.984, 507.468, 245.906],
        [10.0, 20.0, 10.0, 30.0],
        [10.0, 20.0, 20.0, 20.0],
        [10.0, 20.0, 10.0001, 30.0],
    ],
)
def test_visdrone_export_filters_nonpositive_sizes_for_both_models(tmp_path, bad_box):
    from scripts.d1.evaluate_visdrone import export_predictions
    from scripts.d1.rgb import ExportRGBValidator
    from scripts.d1.train import ExportValidator

    _, split_file, cache_dir = build_fixture(tmp_path, count=1)
    records = []
    for kind, validator_type in (("rgb", ExportRGBValidator), ("cached", ExportValidator)):
        options = {"feature_cache": cache_dir} if kind == "cached" else {}
        validator = validator_type(save_dir=tmp_path / kind, args={"plots": False}, dataset_kind="visdrone", **options)
        validator.data = {"val": str(split_file)}
        model = SimpleNamespace(names=dict(enumerate(map(str, range(10)))), end2end=True)
        validator.init_metrics(model)
        prediction = {
            "bboxes": torch.tensor([[1.0, 2.0, 11.0, 22.0], bad_box, [3.0, 4.0, 13.0, 24.0]]),
            "conf": torch.tensor([0.9, 0.02, 0.8]),
            "cls": torch.tensor([0.0, 9.0, 1.0]),
        }
        validator.pred_to_json(prediction, {"im_file": "000001.jpg"})
        assert validator.degenerate_boxes_removed == 1
        assert [row["bbox"] for row in validator.jdict] == [[1.0, 2.0, 10.0, 20.0], [3.0, 4.0, 10.0, 20.0]]
        report = export_predictions(validator.jdict, ["000001"], tmp_path / (kind + "-txt"))
        assert report["image_count"] == 1
        records.append(validator.jdict)
        validator.init_metrics(model)
        assert validator.degenerate_boxes_removed == 0
    assert records[0] == records[1]
