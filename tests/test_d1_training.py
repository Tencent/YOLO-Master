"""Portable training/evaluation entry points and exact optional EMA updates."""

from copy import deepcopy
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch

from scripts.d1 import prepare_coco, train
from scripts.d1 import train as p5
from scripts.d1.cache_features import cache_contract
from scripts.d1.ema import D1ModelEMA, configure_d1_ema, validate_ema_implementation
from ultralytics.models.yolo.detect.foundation_train import D1FoundationDetectionTrainer
from ultralytics.nn.foundation.cache import FeatureCacheWriter
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import ModelEMA


@pytest.fixture
def inputs(tmp_path):
    data = tmp_path / "data"
    caches = {}
    for split, value in (("train2017", 1), ("val2017", 2)):
        images, labels = data / "images" / split, data / "labels" / split
        images.mkdir(parents=True)
        labels.mkdir(parents=True)
        name = f"{value:012d}"
        image = images / f"{name}.jpg"
        assert cv2.imwrite(str(image), np.full((40, 60, 3), value, dtype=np.uint8))
        (labels / f"{name}.txt").write_text("0 0.5 0.5 0.25 0.25\n")
        cache = tmp_path / split
        with FeatureCacheWriter(cache, split=split, contract=cache_contract(train.ROOT)) as writer:
            writer.add(
                sample_id=f"{split}/{name}",
                split=split,
                image_path=f"images/{split}/{name}.jpg",
                image_sha256=train.sha256_file(image),
                features={
                    n: torch.full((384, 40, 40), value / 10, dtype=torch.float16)
                    for n in ("block4", "block8", "block12")
                },
            )
        caches[split] = cache
    yaml = tmp_path / "data.yaml"
    YAML.save(
        yaml,
        {
            "path": str(data),
            "train": "images/train2017",
            "val": "images/val2017",
            "names": {i: str(i) for i in range(80)},
        },
    )
    args = train.parser().parse_args(
        [
            "train",
            "--data",
            str(yaml),
            "--train-cache",
            str(caches["train2017"]),
            "--val-cache",
            str(caches["val2017"]),
            "--output",
            str(tmp_path / "run"),
            "--device",
            "cpu",
            "--batch",
            "1",
            "--epochs",
            "1",
            "--workers",
            "0",
            "--fp32",
            "--approved",
        ]
    )
    return args


def test_training_needs_approval_before_reading_data(inputs):
    inputs.approved = False
    inputs.data = None
    with pytest.raises(ValueError, match="approved"):
        train.input_contract(inputs)


@pytest.mark.parametrize("telemetry", [False, True])
def test_validation_precision_is_part_of_run_identity(inputs, telemetry):
    inputs.telemetry = telemetry
    identity, _, _, _ = train.input_contract(inputs)
    assert identity["validation_precision"] == ("fp32-v1" if telemetry else "shared")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("fails", [False, True])
def test_validation_fp32_restores_training_amp(enabled, fails):
    from scripts.d1.runtime import RunMixin

    class Parent:
        def validate(self):
            assert self.amp is (not enabled)
            assert torch.is_autocast_enabled("cpu") is (not enabled)
            if fails:
                raise RuntimeError("validation failed")
            return {"AP": 0.1}, 0.1

    class Measured(RunMixin, Parent):
        pass

    trainer = object.__new__(Measured)
    trainer._run_enabled, trainer.amp, trainer.device = enabled, True, torch.device("cpu")
    with torch.autocast("cpu"):
        if fails:
            with pytest.raises(RuntimeError, match="validation failed"):
                trainer.validate()
        else:
            assert trainer.validate() == ({"AP": 0.1}, 0.1)
        assert torch.is_autocast_enabled("cpu")
    assert trainer.amp is True


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_validation_failure_names_metric(tmp_path, value):
    trainer = _runtime_trainer(tmp_path / "validation-failure")
    trainer.metrics = {"metrics/mAP50(B)": 0.2, "val/cls_loss": value}
    with pytest.raises(RuntimeError, match="epoch 37.*val/cls_loss"):
        trainer._handle_nan_recovery(36)


def test_rejects_internal_ddp_launch(inputs, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    inputs.device = "0,1"
    with pytest.raises(ValueError, match="torchrun"):
        train.input_contract(inputs)


def test_output_claim_fails_closed(tmp_path):
    output = tmp_path / "run"
    train.claim_output(output, {"run": "one"})
    train.claim_output(output, {"run": "one"}, rank=1)
    with pytest.raises(FileExistsError):
        train.claim_output(output, {"run": "two"})
    with pytest.raises(RuntimeError):
        train.claim_output(output, {"run": "two"}, rank=1, timeout=0)
    with pytest.raises(RuntimeError):
        train.claim_output(tmp_path / "missing", {}, rank=1, timeout=0)


def test_input_contract_and_fresh_run_only(inputs):
    identity, overrides, _, caches = train.input_contract(inputs)
    assert identity["dataset"] == "coco"
    assert overrides["batch"] == overrides["nbs"] == 1
    assert overrides["resume"] is False
    assert not overrides["amp"]
    assert overrides["model"]["adapter"]["p5_mode"] == "bottleneck"
    assert set(caches) == {"train", "val"}
    inputs.dataset = "visdrone"
    with pytest.raises(ValueError, match="names"):
        train.input_contract(inputs)


@pytest.mark.parametrize("implementation", ("scalar-v1", "foreach-v1"))
def test_install_ema_after_setup(implementation, monkeypatch):
    model = train.construct_model("BN64")
    existing = ModelEMA(model, updates=27)
    trainer = object.__new__(train.CachedTrainer)
    trainer.ema_implementation = implementation

    def setup(self):
        self.model, self.ema = model, existing

    monkeypatch.setattr(D1FoundationDetectionTrainer, "_setup_train", setup)
    trainer._setup_train()
    assert trainer.ema.ema is existing.ema
    assert trainer.ema.updates == 27
    assert isinstance(trainer.ema, D1ModelEMA) == (implementation == "foreach-v1")


def test_checkpoint_roundtrip_and_rejection(tmp_path):
    model = train.construct_model("BN64")
    checkpoint = tmp_path / "model.pt"
    torch.save({"model": model, "epoch": 2}, checkpoint)
    restored, epoch = train.strict_checkpoint(checkpoint)
    assert epoch == 2
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)
    torch.save({"model": torch.nn.Linear(1, 1)}, checkpoint)
    with pytest.raises(TypeError, match="D1"):
        train.strict_checkpoint(checkpoint)


@pytest.mark.parametrize("dataset", ("coco", "visdrone"))
def test_independent_evaluation_uses_all_images(inputs, dataset):
    inputs.dataset = dataset
    nc = 80 if dataset == "coco" else 10
    data = YAML.load(inputs.data)
    data["names"] = {i: str(i) for i in range(nc)}
    YAML.save(inputs.data, data)
    model = train.construct_model("BN64", nc=nc)
    checkpoint = inputs.output.parent / "checkpoint.pt"
    torch.save({"model": model, "epoch": 3}, checkpoint)
    inputs.command = "evaluate"
    inputs.checkpoint = checkpoint
    train.run(inputs)
    import json

    report = json.loads((inputs.output / "evaluation.json").read_text())
    assert report["strict_reload"] and report["images"] == 1
    assert report["validation_precision"] == "fp32-v1"
    assert report["checkpoint_epoch_zero_based"] == 3
    assert report["official"] is None
    assert (inputs.output / "predictions.json").is_file()
    if dataset == "visdrone":
        assert report["visdrone_export"]["image_count"] == 1


def test_synthetic_one_epoch_completes_and_saves(inputs):
    train.run(inputs)
    assert (inputs.output / "completed.json").is_file()
    assert (inputs.output / "weights/last.pt").is_file()
    train.strict_checkpoint(inputs.output / "weights/last.pt")


def test_visdrone_export_preserves_ids_and_filters_only_zero_area():
    validator = object.__new__(train.ExportValidator)
    validator.dataset_kind = "visdrone"
    validator.degenerate_boxes_removed = 0
    validator.jdict = []
    validator.class_map = list(range(10))
    prediction = {
        "bboxes": torch.tensor([[1.0, 2.0, 4.0, 6.0], [3.0, 4.0, 3.0, 5.0]]),
        "conf": torch.tensor([0.8, 0.9]),
        "cls": torch.tensor([1.0, 2.0]),
    }
    validator.pred_to_json(prediction, {"im_file": "00012.jpg"})
    assert len(validator.jdict) == 1
    assert validator.jdict[0]["image_id"] == "00012"
    assert validator.jdict[0]["category_id"] == 1
    assert validator.degenerate_boxes_removed == 1
    prediction["conf"][0] = float("nan")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        validator.pred_to_json(prediction, {"im_file": "00012.jpg"})


def test_coco_npy_conversion_preserves_source_and_is_repeatable(inputs):
    from scripts.d1.convert_npy import convert_preserving_source
    from ultralytics.nn.foundation.npy_cache import NpyFeatureCacheReader

    source = inputs.train_cache
    before = {p.name: train.sha256_file(p) for p in source.iterdir() if p.is_file()}
    output = inputs.output.parent / "npy"
    first = convert_preserving_source(source, output)
    assert first == convert_preserving_source(source, output)
    assert before == {p.name: train.sha256_file(p) for p in source.iterdir() if p.is_file()}
    reader = NpyFeatureCacheReader(output / "train2017")
    for sample in reader.records:
        reader.verify_sample(sample)


def test_ddp_batch_must_be_divisible(inputs, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "2")
    inputs.device = "0,1"
    inputs.batch = 3
    with pytest.raises(ValueError, match="divisible"):
        train.input_contract(inputs)


def test_preparation_keeps_published_manifest_unchanged(tmp_path, monkeypatch):
    repo, workspace = tmp_path / "repo", tmp_path / "work"
    manifest = repo / "experiments/d1/manifests"
    manifest.mkdir(parents=True)
    published = {"splits": {}, "labels": {"train2017": 1, "val2017": 1}}
    original = {"coco2017-splits.json": train.canonical_json_bytes(published)}
    for name, data in original.items():
        (manifest / name).write_bytes(data)
    monkeypatch.setattr(prepare_coco, "verify_contract", lambda *a: None)
    monkeypatch.setattr(
        prepare_coco,
        "validated_splits",
        lambda *a: {split: [f"images/{split}/{split}.jpg"] for split in ("train2017", "val2017")},
    )
    monkeypatch.setattr(prepare_coco, "verify_labels", lambda *a: published["labels"])
    monkeypatch.setattr(prepare_coco, "verify_model", lambda *a: {"files": {}})
    report = prepare_coco.verify_inputs(workspace / "data", workspace / "weights", workspace, repo=repo)
    assert {p.name: p.read_bytes() for p in manifest.iterdir()} == original
    assert (workspace / "verification.json").is_file()
    assert report["source_archives_verified"] is False


def test_official_coco_handles_empty_predictions(tmp_path):
    pytest.importorskip("faster_coco_eval")
    import json

    annotations = tmp_path / "instances.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "height": 40, "width": 60}],
                "categories": [{"id": 1, "name": "test"}],
                "annotations": [
                    {"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0}
                ],
            }
        )
    )
    report = train.official_coco(annotations, [], [1])
    assert report["metrics"]["AP"] == 0
    with pytest.raises(ValueError, match="absent"):
        train.official_coco(annotations, [], [99])


def assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0, equal_nan=True)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_same(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_same(a, b)
    else:
        assert left == right


@pytest.mark.parametrize("variant", p5.VARIANTS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64, torch.float16, torch.bfloat16))
def test_exact_update_and_persistent_buffer_rules(variant, dtype):
    model = p5.construct_model(variant).to(dtype=dtype)
    model.register_buffer("test_float", torch.randn(7, dtype=dtype))
    model.register_buffer("test_int", torch.tensor(4))
    model.register_buffer("test_bool", torch.tensor(False))
    model.register_buffer("test_temporary", torch.tensor(1.0), persistent=False)
    # Include heterogeneous source dtypes and empty tensors in the same update.
    model.register_buffer("test_double", torch.ones(3, dtype=torch.float64))
    model.register_buffer("test_empty", torch.empty(0))
    original = ModelEMA(model, updates=100, decay=0.99, tau=30)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    for _ in range(3):
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(0.002)
            model.test_float.mul_(1.5)
            model.test_int.add_(1)
            model.test_bool.logical_not_()
            model.test_temporary.add_(1)
        original.update(model)
        fast.update(model)
        assert fast.updates == original.updates
        assert_same(original.ema.state_dict(), fast.ema.state_dict())
        assert fast.ema.test_temporary.item() == 1
        assert not any(p.requires_grad for p in fast.ema.parameters())


def test_conversion_preserves_restored_state_and_is_local():
    model = p5.construct_model("BASE")
    existing = ModelEMA(model, updates=87)
    existing.enabled = False
    fast = configure_d1_ema(existing, model, "foreach-v1")
    assert fast.ema is existing.ema and fast.decay is existing.decay
    assert fast.updates == 87 and not fast.enabled
    fast.update(torch.nn.Linear(2, 2))
    assert fast.updates == 87
    assert type(existing) is ModelEMA
    assert configure_d1_ema(existing, model, "scalar-v1") is existing
    assert configure_d1_ema(None, model, "foreach-v1") is None
    with pytest.raises(TypeError):
        configure_d1_ema(fast, model, "foreach-v1")
    with pytest.raises(TypeError):
        configure_d1_ema(ModelEMA(torch.nn.Linear(2, 2)), model, "foreach-v1")
    with pytest.raises(TypeError):
        configure_d1_ema(existing, torch.nn.Linear(2, 2), "foreach-v1")


def test_dynamic_buffers_and_no_per_step_state_serialization(monkeypatch):
    model = p5.construct_model("BASE")
    original = ModelEMA(model)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    for obj in (model, original.ema, fast.ema):
        obj.register_buffer("late_buffer", torch.tensor(3.0))
    model.late_buffer = torch.tensor(5.0)
    original.update(model)
    monkeypatch.setattr(model, "state_dict", lambda *a, **k: pytest.fail("Serialized source during fast update"))
    fast.update(model)
    assert_same(original.ema.state_dict(), fast.ema.state_dict())


def test_hook_and_shape_changes_fail_before_partial_update():
    model = p5.construct_model("BASE")
    fast = configure_d1_ema(ModelEMA(model), model, "foreach-v1")
    hook = model.register_state_dict_pre_hook(lambda *a: None)
    with pytest.raises(ValueError, match="hooks"):
        fast.update(model)
    hook.remove()
    model._mixture_loss_ema_buf = torch.ones(100)
    before = deepcopy(fast.ema.state_dict())
    with pytest.raises(ValueError, match="shape/device"):
        fast.update(model)
    assert fast.updates == 0
    assert_same(before, fast.ema.state_dict())


@pytest.mark.parametrize("value", [None, True, "foreach", "typo", [], 1])
def test_invalid_registration(value):
    with pytest.raises(ValueError):
        validate_ema_implementation(value)


@pytest.mark.parametrize("variant", p5.VARIANTS)
def test_real_loss_optimizer_and_ema_exact(variant):
    torch.manual_seed(0)
    model = p5.construct_model(variant).train()
    original = ModelEMA(model)
    fast = configure_d1_ema(deepcopy(original), model, "foreach-v1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    batch = {
        "features": {key: torch.randn(1, 384, 4, 4) for key in model.adapter.source_names},
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
    }
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model(batch)
        loss.sum().backward()
        optimizer.step()
        original.update(model)
        fast.update(model)
        assert_same(original.ema.state_dict(), fast.ema.state_dict())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA; no download")
@pytest.mark.parametrize("variant", p5.VARIANTS)
def test_cuda_amp_overflow_and_resume(variant):
    from ultralytics.engine.extensions.recovery import TrainingRecoveryController
    from ultralytics.engine.trainer import BaseTrainer

    model = p5.construct_model(variant).cuda().train()
    ema = configure_d1_ema(ModelEMA(model), model, "foreach-v1")
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = torch.amp.GradScaler("cuda", init_scale=16)
    trainer = SimpleNamespace(
        model=model, ema=ema, optimizer=optimizer, scaler=scaler, optimizer_steps=0, adapter_controller=None
    )
    trainer._recovery_controller = lambda: TrainingRecoveryController(trainer)
    trainer._sync_nonfinite_flag = lambda bad: bool(bad)
    before = deepcopy(ema.ema.state_dict())
    scaler.scale(next(model.parameters()).square().mean()).backward()
    next(model.parameters()).grad.reshape(-1)[0] = float("inf")
    assert not BaseTrainer.optimizer_step(trainer)
    assert trainer.optimizer_steps == ema.updates == 0 and scaler.get_scale() == 8
    assert_same(before, ema.ema.state_dict())
    reference = ModelEMA(model)
    reference.ema.load_state_dict(ema.ema.state_dict(), strict=True)
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(next(model.parameters()).square().mean()).backward()
        assert BaseTrainer.optimizer_step(trainer)
        reference.update(model)
        assert_same(reference.ema.state_dict(), ema.ema.state_dict())
    restored = ModelEMA(model, updates=ema.updates)
    restored.ema.load_state_dict(ema.ema.state_dict(), strict=True)
    restored = configure_d1_ema(restored, model, "foreach-v1")
    restored.update(model)
    ema.update(model)
    assert_same(restored.ema.state_dict(), ema.ema.state_dict())


class _RuntimeScalarModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(()))
        self.register_buffer("temperature", torch.tensor(1.0))
        self.register_buffer("running", torch.tensor(0.0))
        self.criterion = self.init_criterion()

    def init_criterion(self):
        return SimpleNamespace(updates=0, o2m=0.8, o2o=0.2)


class _RuntimeDataset(torch.utils.data.Dataset):
    def __len__(self):
        return 16

    def __getitem__(self, index):
        import random

        return {"im_file": str(index), "value": torch.tensor(index + random.random() + np.random.random())}


class _RuntimeBase:
    def __init__(self, output, *, no_ema=False):
        self.args = SimpleNamespace(seed=19, epochs=8, workers=0, amp=False, time=None, compile=False, close_mosaic=0)
        self.save_dir = output
        self.wdir = output / "weights"
        self.last = self.wdir / "last.pt"
        self.save_period = 5
        self.device = torch.device("cpu")
        self.callbacks = {}
        self.world_size = 1
        self.resume = self.amp = self.stop = False
        self.start_epoch = self.optimizer_steps = 0
        self.accumulate = 1
        self.no_ema = no_ema
        self.metrics, self.fitness = {}, None

    def add_callback(self, name, callback):
        self.callbacks.setdefault(name, []).append(callback)

    def get_model(self, **kwargs):
        return _RuntimeScalarModel()

    def build_dataset(self, *args):
        return _RuntimeDataset()

    def _setup_train(self):
        self.model = self.get_model()
        self.ema = None if self.no_ema else ModelEMA(self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.scaler = torch.amp.GradScaler("cpu", enabled=True, init_scale=16, growth_interval=1_000_000)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda epoch: 1 - epoch / self.args.epochs)
        self.scheduler.last_epoch = -1
        self.train_loader = self.get_dataloader("train", batch_size=4, rank=-1)
        self.test_loader = self.get_dataloader("val", batch_size=4, rank=-1, mode="val")
        self.mixture_controller = SimpleNamespace(anneal_temperature=lambda: self.model.temperature.mul_(0.97))

    def _recovery_controller(self):
        from ultralytics.engine.extensions.recovery import TrainingRecoveryController

        return TrainingRecoveryController(self)

    def _sync_nonfinite_flag(self, bad):
        return bad

    def save_model(self):
        self.wdir.mkdir(parents=True, exist_ok=True)
        torch.save({"epoch": self.epoch, "model": deepcopy(self.model).half()}, self.last)
        if self.save_period > 0 and self.epoch % self.save_period == 0:
            (self.wdir / f"epoch{self.epoch}.pt").write_bytes(self.last.read_bytes())
        return True

    def final_eval(self):
        return "ordinary-final-eval"


def _runtime_trainer(output, *, resume=None, window=None, no_ema=False):
    from scripts.d1.runtime import RunMixin

    class ScalarTrainer(RunMixin, _RuntimeBase):
        pass

    trainer = ScalarTrainer(
        output,
        run_identity={"seed": 19, "schedule": 8},
        run_output=output,
        resume_snapshot=resume,
        stop_after_epoch=window,
        no_ema=no_ema,
    )
    trainer._setup_train()
    return trainer


def _runtime_epoch(trainer, epoch):
    import random
    import warnings

    trainer.epoch = epoch
    if epoch > trainer.start_epoch:
        trainer.mixture_controller.anneal_temperature()
    trainer._run_on_train_epoch_start(trainer)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trainer.scheduler.step()
    trainer.train_loader.set_epoch(epoch)
    for batch in trainer.train_loader:
        trainer.batch = batch
        trainer._run_on_train_batch_start(trainer)
        trainer.model.running.add_(batch["value"].mean())
        target = batch["value"].mean() / 16 + random.random() + np.random.random() + torch.rand(())
        trainer.loss = (trainer.model.weight * trainer.model.temperature - target).square()
        trainer.loss_items = trainer.loss.detach().reshape(1)
        trainer.scaler.scale(trainer.loss).backward()
        trainer.optimizer_step()
        trainer._run_on_train_batch_end(trainer)
    criterion = trainer.model.criterion
    criterion.updates += 1
    criterion.o2m *= 0.95
    criterion.o2o = 1 - criterion.o2m
    trainer._run_on_train_epoch_end(trainer)
    list(trainer.test_loader)
    trainer._run_on_fit_epoch_end(trainer)


@pytest.mark.parametrize("no_ema", [False, True])
def test_runtime_scalar_epoch_resume_exact(tmp_path, no_ema):
    import json
    import random

    from scripts.d1.runtime import state_digest

    random.seed(47)
    np.random.seed(47)
    torch.manual_seed(47)
    continuous = _runtime_trainer(tmp_path / "continuous", no_ema=no_ema)
    _runtime_epoch(continuous, 0)
    source = continuous.run_output / "resume.pt"
    snapshot = tmp_path / "epoch-one.pt"
    snapshot.write_bytes(source.read_bytes())
    _runtime_epoch(continuous, 1)
    expected = torch.load(source, weights_only=False)
    resumed = _runtime_trainer(tmp_path / "resumed", resume=snapshot, window=2, no_ema=no_ema)
    assert resumed.start_epoch == 1 and not resumed.resume and resumed.args.epochs == 8
    assert resumed.model.criterion is not continuous.model.criterion
    _runtime_epoch(resumed, 1)
    actual = torch.load(resumed.run_output / "resume.pt", weights_only=False)
    for key in ("model", "ema", "optimizer", "scaler", "scheduler", "criterion", "ranks", "optimizer_steps"):
        assert state_digest(actual[key]) == state_digest(expected[key]), key
    assert resumed.stop and not resumed.run_complete and resumed.final_eval() is None
    audit = json.loads((resumed.run_output / "resume-restored-rank-0.json").read_text())
    assert audit["matched"] and audit["expected_state_digests"] == audit["restored_state_digests"]
    assert (resumed.run_output / "epochs/rank-0-epoch-002.json").is_file()
    timing = json.loads((resumed.run_output / "validation/epoch-002.json").read_text())
    assert timing["epoch_wall_seconds"] > 0
    before = state_digest(actual)
    resumed.final_eval()
    assert state_digest(torch.load(resumed.run_output / "resume.pt", weights_only=False)) == before


def test_runtime_bounded_workers_reproduce_epoch_and_rank(tmp_path):
    from scripts.d1.runtime import state_digest

    trainer = _runtime_trainer(tmp_path / "workers")
    trainer.args.workers = 4
    first = trainer.get_dataloader("train", batch_size=2, rank=-1)
    assert first.num_workers == 4 and not first.persistent_workers and not hasattr(first, "iterator")
    first.set_epoch(0)
    epoch_zero = list(first)
    first.set_epoch(1)
    expected = list(first)
    second = trainer.get_dataloader("train", batch_size=2, rank=-1)
    second.set_epoch(1)
    assert state_digest(list(second)) == state_digest(expected)
    assert state_digest(epoch_zero) != state_digest(expected)
    assert sorted(int(name) for batch in expected for name in batch["im_file"]) == list(range(16))


@pytest.mark.parametrize("failure", ["overflow", "skipped", "counter"])
def test_runtime_optimizer_fails_closed(tmp_path, monkeypatch, failure):
    trainer = _runtime_trainer(tmp_path / failure)
    trainer.loss = trainer.model.weight.square()
    trainer.scaler.scale(trainer.loss).backward()
    if failure == "overflow":
        trainer.model.weight.grad.fill_(float("inf"))
    elif failure == "skipped":
        monkeypatch.setattr(trainer.scaler, "step", lambda optimizer: None)
    elif failure == "counter":
        trainer.optimizer.register_step_post_hook(lambda *args: setattr(trainer, "optimizer_steps", 100))
    with pytest.raises(RuntimeError, match="update|overflow|Non-finite"):
        trainer.optimizer_step()
    assert not (trainer.run_output / "resume.pt").exists()


def test_runtime_optimizer_state_checked_at_boundary_only(tmp_path, monkeypatch):
    trainer = _runtime_trainer(tmp_path / "optimizer-state", no_ema=True)
    trainer.loss = trainer.model.weight.square()
    trainer.scaler.scale(trainer.loss).backward()
    with monkeypatch.context() as patch:
        patch.setattr(trainer.model, "state_dict", lambda: pytest.fail("Per-step state serialization"))
        assert trainer.optimizer_step()
    trainer.optimizer.state[trainer.model.weight]["exp_avg"].fill_(float("nan"))
    with pytest.raises(RuntimeError, match="Non-finite"):
        trainer._recover_before_validation(0)


def test_runtime_periodic_fp32_and_seed_isolation(tmp_path):
    trainer = _runtime_trainer(tmp_path / "periodic")
    before = torch.get_rng_state().clone()
    first, second = trainer.get_model(), trainer.get_model()
    assert torch.equal(first.weight, second.weight) and torch.equal(before, torch.get_rng_state())
    for epoch in range(10):
        trainer.epoch = epoch
        trainer.save_model()
    assert sorted(path.name for path in trainer.wdir.glob("epoch*")) == ["epoch-005.pt", "epoch-010.pt"]
    checkpoint = torch.load(trainer.wdir / "epoch-005.pt", weights_only=False)
    assert checkpoint["epoch"] == 4
    assert checkpoint["model"].weight.dtype == checkpoint["ema"].weight.dtype == torch.float32
    assert not any("teacher" in key for key in checkpoint["model"].state_dict())
    assert trainer.save_period == 5


@pytest.mark.parametrize("corruption", ["identity", "groups", "tensor"])
def test_runtime_resume_rejects_mismatched_state(tmp_path, corruption):
    trainer = _runtime_trainer(tmp_path / "source")
    _runtime_epoch(trainer, 0)
    path = trainer.run_output / "resume.pt"
    state = torch.load(path, weights_only=False)
    if corruption == "identity":
        state["identity"]["seed"] += 1
    elif corruption == "groups":
        state["optimizer_groups"][0][0] = "wrong-parameter"
    else:
        state["model"]["weight"].add_(1)
    torch.save(state, path)
    with pytest.raises(ValueError, match="identity|groups|digest"):
        _runtime_trainer(tmp_path / "rejected", resume=path)


def test_runtime_default_delegates_and_preserves_existing_output(tmp_path):
    from scripts.d1.runtime import RunMixin

    class Ordinary(RunMixin, _RuntimeBase):
        pass

    trainer = Ordinary(tmp_path)
    assert not trainer.callbacks and trainer.final_eval() == "ordinary-final-eval"
    trainer = _runtime_trainer(tmp_path / "claimed")
    _runtime_epoch(trainer, 0)
    with pytest.raises(FileExistsError, match="fresh"):
        _runtime_trainer(trainer.run_output)
    resumed = _runtime_trainer(trainer.run_output, resume=trainer.run_output / "resume.pt", window=2)
    _runtime_epoch(resumed, 1)
    assert resumed.stop and resumed.start_epoch == 1


def test_runtime_amp_policy_is_shared(tmp_path, monkeypatch):
    setup = _RuntimeBase._setup_train
    scaler = torch.amp.GradScaler

    def amp_setup(self):
        setup(self)
        self.args.amp = self.amp = True

    monkeypatch.setattr(_RuntimeBase, "_setup_train", amp_setup)
    monkeypatch.setattr(torch.amp, "GradScaler", lambda device, **kwargs: scaler("cpu", **kwargs))
    trainer = _runtime_trainer(tmp_path / "amp-policy")
    assert trainer.scaler.get_scale() == 1.0
    assert not trainer.scaler.is_enabled()
    assert trainer.scaler.state_dict() == {}


def test_bf16_training_context_and_fp32_loss_keep_gradients(tmp_path):
    from scripts.d1.runtime import RunMixin
    from ultralytics.nn.mixture_loss import CompositeCriterion, build_composite_criterion

    trainer = object.__new__(RunMixin)
    trainer._run_enabled, trainer.amp, trainer.device = True, True, torch.device("cpu")
    model = torch.nn.Linear(4, 2)
    model._d1_loss_fp32 = True
    observations = []

    def native(predictions, batch):
        observations.append((predictions["scores"].dtype, torch.is_autocast_enabled("cpu")))
        assert predictions["indices"].dtype == torch.int64
        loss = predictions["scores"].square().mean()
        return loss, loss.detach().reshape(1)

    criterion = build_composite_criterion(model, native)
    assert isinstance(criterion, CompositeCriterion) and not criterion.enabled
    with trainer.training_autocast():
        predictions = model(torch.ones(2, 4))
        assert predictions.dtype == torch.bfloat16
        loss, _ = criterion({"scores": predictions, "indices": torch.tensor([1])}, {})
        assert torch.is_autocast_enabled("cpu")
    loss.backward()
    assert observations == [(torch.float32, False)]
    assert loss.dtype == torch.float32
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad).all() and model.weight.grad.abs().sum() > 0
    assert model.weight.dtype == torch.float32


def test_fp32_loss_policy_default_preserves_native_input():
    from ultralytics.nn.mixture_loss import CompositeCriterion

    model = torch.nn.Linear(1, 1)
    predictions = torch.ones(1, dtype=torch.bfloat16)
    calls = []

    def native(value, batch):
        calls.append((value is predictions, value.dtype, torch.is_autocast_enabled("cpu")))
        return value.sum(), value

    with torch.autocast("cpu"):
        CompositeCriterion(model, native)(predictions, {})
    assert calls == [(True, torch.bfloat16, True)]


def test_measured_bf16_hardware_check_and_default_delegation(monkeypatch):
    from scripts.d1.runtime import RunMixin

    class Parent:
        def check_amp_compatibility(self):
            return "default"

        def training_autocast(self):
            return "default-context"

    class Trainer(RunMixin, Parent):
        pass

    trainer = object.__new__(Trainer)
    trainer._run_enabled = False
    assert trainer.check_amp_compatibility() == "default"
    assert trainer.training_autocast() == "default-context"
    trainer._run_enabled, trainer.device = True, torch.device("cuda")
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert not trainer.check_amp_compatibility()
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert trainer.check_amp_compatibility()


def test_runtime_restores_rank_local_buffers_without_rank_local_ema(tmp_path, monkeypatch):
    from scripts.d1 import runtime

    source = _runtime_trainer(tmp_path / "rank-source")
    _runtime_epoch(source, 0)
    state = torch.load(source.run_output / "resume.pt", weights_only=False)
    local = deepcopy(state["ranks"][0])
    local["model_buffers"]["running"].add_(17)
    local["ema_buffers"] = None
    state["ranks"].append(local)
    state["state_digests"]["ranks"] = runtime.state_digest(state["ranks"])
    target = _runtime_trainer(tmp_path / "rank-target", no_ema=True)
    target._run_rank = 1
    target._run_resume_state = state
    target.resume_snapshot = source.run_output / "resume.pt"
    monkeypatch.setattr(runtime.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(runtime.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(runtime.dist, "all_reduce", lambda *args, **kwargs: None)
    target._run_restore()
    assert torch.equal(target.model.running, local["model_buffers"]["running"])
    assert not torch.equal(target.model.running, state["model"]["running"])
    assert target.ema is None and target.start_epoch == 1


@pytest.mark.parametrize("mode", ["train", "val"])
def test_runtime_rejects_filtered_image_lists(tmp_path, mode):
    trainer = _runtime_trainer(tmp_path / "list-check")
    path = tmp_path / f"{mode}.txt"
    path.write_text("\n".join(f"{index}.jpg" for index in range(17)))
    with pytest.raises(ValueError, match="dataset size.*list size"):
        trainer.get_dataloader(str(path), batch_size=2, rank=-1, mode=mode)
    path.write_text("\n".join(f"{index}.jpg" for index in range(16)))
    trainer.args.workers = 4
    loader = trainer.get_dataloader(str(path), batch_size=2, rank=-1, mode=mode)
    assert len(loader.dataset) == loader.source_manifest["listed_samples"] == 16
    assert loader.num_workers == 4 and loader.prefetch_factor == 1


def test_runtime_setup_manifest(tmp_path):
    import json

    trainer = _runtime_trainer(tmp_path / "manifest")
    report = json.loads((trainer.run_output / "setup-rank-0.json").read_text())
    assert report["seed"] == report["args"]["seed"] == 19
    assert report["datasets"]["train"]["dataset_size"] == report["datasets"]["val"]["dataset_size"] == 16
    assert report["datasets"]["train"]["rank_batches"] == 4
    assert report["optimizer_groups"][0]["names"] == ["weight"]
    assert report["amp"]["training_precision"] == "fp32"
    assert report["amp"]["validation_precision"] == "fp32-v1"


def test_runtime_last_best_periodic_share_fp32_tensor_bits(tmp_path):
    trainer = _runtime_trainer(tmp_path / "checkpoint-parity")
    trainer.best = trainer.wdir / "best.pt"
    trainer.best_fitness = trainer.fitness = 1.0
    trainer.epoch = 4
    with torch.no_grad():
        trainer.model.weight.fill_(1.0003)
        trainer.ema.ema.weight.fill_(0.5001)
        trainer.model.running.fill_(-0.0)
    expected = {"model": deepcopy(trainer.model.state_dict()), "ema": deepcopy(trainer.ema.ema.state_dict())}
    assert not torch.equal(expected["model"]["weight"], expected["model"]["weight"].half().float())
    trainer.save_model()
    periodic = trainer.wdir / "epoch-005.pt"
    trainer.save_model()  # Replay after periodic publication but before the resume boundary commits.
    original_best, original_periodic = trainer.best.read_bytes(), periodic.read_bytes()
    with torch.no_grad():
        trainer.model.weight.add_(2)
        trainer.ema.ema.weight.add_(3)
    for path in (trainer.last, trainer.best, periodic):
        checkpoint = torch.load(path, weights_only=False)
        assert checkpoint["epoch"] == 4
        for key in ("model", "ema"):
            for name, tensor in checkpoint[key].state_dict().items():
                assert tensor.dtype == torch.float32
                assert torch.equal(
                    tensor.reshape(-1).view(torch.uint8), expected[key][name].reshape(-1).view(torch.uint8)
                ), (path.name, key, name)
    trainer.epoch, trainer.fitness = 5, 0.5
    trainer.save_model()
    checkpoint = torch.load(trainer.last, weights_only=False)
    assert checkpoint["epoch"] == 5 and checkpoint["model"].weight.dtype == torch.float32
    assert torch.equal(checkpoint["model"].weight, trainer.model.weight)
    assert torch.equal(checkpoint["ema"].weight, trainer.ema.ema.weight)
    assert trainer.best.read_bytes() == original_best and periodic.read_bytes() == original_periodic


def test_runtime_atomic_snapshot_ignores_stale_attempt(tmp_path):
    from scripts.d1.runtime import atomic_torch

    stale = tmp_path / "resume.pt.part"
    stale.write_bytes(b"interrupted")
    atomic_torch(tmp_path / "resume.pt", {"epoch": 1})
    atomic_torch(tmp_path / "resume.pt", {"epoch": 2})
    assert torch.load(tmp_path / "resume.pt", weights_only=True)["epoch"] == 2
    assert stale.read_bytes() == b"interrupted"


def test_attention_default_and_legacy_state_keep_original_math():
    from ultralytics.nn.modules.block import Attention

    module = Attention(8, num_heads=2).eval()
    image = torch.randn(2, 8, 3, 3)
    q, k, v = module.qkv(image).view(2, 2, 8, 9).split([2, 2, 4], dim=2)
    weights = ((q * module.scale).transpose(-2, -1) @ k).softmax(dim=-1)
    expected = module.proj((v @ weights.transpose(-2, -1)).view(2, 8, 3, 3) + module.pe(v.reshape(2, 8, 3, 3)))
    torch.testing.assert_close(module(image), expected, rtol=0, atol=0)
    del module.fp32_attention
    torch.testing.assert_close(module(image), expected, rtol=0, atol=0)


def test_fp32_attention_prevents_half_dot_product_overflow_and_preserves_gradients():
    from ultralytics.nn.modules.block import Attention

    module = Attention(8, num_heads=2).half()
    module.qkv = torch.nn.Conv2d(8, 16, 1, bias=False).half()
    module.pe = torch.nn.Conv2d(8, 8, 1, bias=False).half()
    module.proj = torch.nn.Identity()
    with torch.no_grad():
        module.qkv.weight.zero_()
        module.qkv.weight[:, 0] = 1000
        module.pe.weight.zero_()
    image = torch.ones(2, 8, 3, 3, dtype=torch.float16, requires_grad=True)
    assert not torch.isfinite(module(image)).all()
    reference = deepcopy(module).float()(image.detach().float())
    keys = tuple(module.state_dict())
    module.fp32_attention = True
    actual = module(image)
    assert actual.dtype == image.dtype and torch.isfinite(actual).all()
    assert tuple(module.state_dict()) == keys
    torch.testing.assert_close(actual.float(), reference, rtol=1e-3, atol=1e-3)
    (actual.float().mean() / 1000).backward()
    assert image.grad is not None and torch.isfinite(image.grad).all() and image.grad.abs().sum() > 0
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())


def test_scratch_attention_policy_is_explicit_and_audited():
    from scripts.d1.rgb import audit_model, scratch_config
    from ultralytics.nn.modules.block import Attention

    model = train.construct_model("SCRATCH", nc=10)
    attention = [module for module in model.modules() if isinstance(module, Attention)]
    assert len(attention) == 3 and all(module.fp32_attention is True for module in attention)
    assert audit_model(model)["fp32_attention"] is True
    attention[-1].fp32_attention = False
    with pytest.raises(ValueError, match="FP32 math"):
        audit_model(model)
    config = scratch_config(nc=10)
    config.pop("fp32_attention")
    with pytest.raises(ValueError, match="exact registered"):
        scratch_config(config, nc=10)


def test_native_scratch_checkpoint_strict_reload(tmp_path):
    from scripts.d1.rgb import ScratchTrainer, scratch_config

    instance = object.__new__(ScratchTrainer)
    instance.resume = False
    instance.args = SimpleNamespace(cls_remap=True)
    instance.data = {"nc": 10, "names": {i: str(i) for i in range(10)}}
    model = instance.get_model(scratch_config(nc=10), verbose=False)
    assert "_mixture_loss_ema_buf" not in model.state_dict()
    checkpoint = tmp_path / "native-scratch.pt"
    torch.save({"model": model, "epoch": 2}, checkpoint)
    restored, epoch = train.strict_checkpoint(checkpoint, allow_scratch=True)
    assert epoch == 2
    from ultralytics.nn.modules.block import Attention

    assert restored.yaml["fp32_attention"] is True
    assert all(module.fp32_attention for module in restored.modules() if isinstance(module, Attention))
    for name, value in model.state_dict().items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, restored.state_dict()[name], rtol=0, atol=0)


def test_reducer_warmup_preserves_state_rng_and_criterion_binding(tmp_path, monkeypatch):
    from scripts.d1 import runtime

    trainer = _runtime_trainer(tmp_path / "warmup")
    original = trainer.model
    criterion = original.criterion
    before = runtime.state_digest(original.state_dict())
    rng_before = runtime.state_digest(runtime.rng_state(trainer.device))
    generator_before = trainer.train_loader.generator.get_state()

    class FakeDDP(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, batch):
            self.module.running.add_(1)
            self.module.criterion.updates += 1
            loss = self.module.weight * batch["value"].sum()
            return loss, loss.detach()

    monkeypatch.setattr(torch.nn.parallel, "DistributedDataParallel", FakeDDP)
    trainer.model = FakeDDP(original)
    trainer.preprocess_batch = lambda batch: batch
    trainer._run_warm_reducer()
    assert original.criterion is criterion and criterion.updates == 0
    assert runtime.state_digest(original.state_dict()) == before
    assert runtime.state_digest(runtime.rng_state(trainer.device)) == rng_before
    assert torch.equal(trainer.train_loader.generator.get_state(), generator_before)
    assert trainer.optimizer_steps == 0 and original.weight.grad is None


def test_runtime_amp_recipe_matches_policy():
    from scripts.d1.runtime import AMP_GROWTH_INTERVAL, AMP_INIT_SCALE
    from scripts.d1.train import RECIPE
    from ultralytics.utils import YAML

    runtime = YAML.load(RECIPE)["runtime"]
    assert runtime["amp_init_scale"] == AMP_INIT_SCALE == 0.0625
    assert runtime["amp_growth_interval"] == AMP_GROWTH_INTERVAL == 1_000_000


def test_runtime_overflow_records_failure_without_skipping_silently(tmp_path):
    import json

    trainer = _runtime_trainer(tmp_path / "overflow")
    trainer.epoch = 0
    trainer.loss = trainer.model.weight.square()
    trainer.scaler.scale(trainer.loss).backward()
    trainer.model.weight.grad.fill_(float("inf"))
    with pytest.raises(RuntimeError, match="Missing/repeated optimizer update or AMP overflow"):
        trainer.optimizer_step()
    report = json.loads((trainer.run_output / "failed-update-rank-0.json").read_text())
    assert report["gradient_nonfinite"] and not report["updated"]
    assert report["optimizer_steps"] == report["actual_steps"] == report["ema_updates"] == [0, 0]
