# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import contextlib
import csv
import json
import urllib
from copy import copy
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from tests import CFG, MODEL, MODELS, SOURCE, SOURCES_LIST, TASK_MODEL_DATA
from ultralytics import RTDETR, YOLO
from ultralytics.cfg import TASK2DATA, TASKS
from ultralytics.data.build import load_inference_source
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import (
    ARM64,
    ASSETS,
    ASSETS_URL,
    DEFAULT_CFG,
    DEFAULT_CFG_PATH,
    IS_JETSON,
    IS_RASPBERRYPI,
    LINUX,
    LOGGER,
    ONLINE,
    ROOT,
    WEIGHTS_DIR,
    WINDOWS,
    YAML,
    checks,
    is_github_action_running,
)
from ultralytics.utils.downloads import download
from ultralytics.utils.torch_utils import TORCH_1_11, TORCH_1_13


def make_stub_yolo():
    """Create a lightweight YOLO instance without loading weights from disk or network."""
    model = YOLO.__new__(YOLO)
    torch.nn.Module.__init__(model)
    model._check_is_pytorch_model = lambda: None
    model.trainer = mock.Mock()
    model.model = mock.Mock()
    return model


def test_model_forward():
    """Test the forward pass of the YOLO model."""
    model = YOLO(CFG)
    model(source=None, imgsz=32, augment=True)  # also test no source and augment


def test_model_methods():
    """Test various methods and properties of the YOLO model to ensure correct functionality."""
    model = YOLO(MODEL)

    # Model methods
    model.info(verbose=True, detailed=True)
    model = model.reset_weights()
    model = model.load(MODEL)
    model.to("cpu")
    model.fuse()
    model.clear_callback("on_train_start")
    model.reset_callbacks()

    # Model properties
    _ = model.names
    _ = model.device
    _ = model.transforms
    _ = model.task_map


def test_save_lora_only_uses_live_trainer_model(tmp_path):
    """Test that LoRA adapter export uses the live trainer model when available."""
    model = make_stub_yolo()
    model.trainer.model = mock.Mock(lora_enabled=True)

    with mock.patch("ultralytics.utils.lora.save_lora_adapters", return_value=True) as save_mock:
        assert model.save_lora_only(tmp_path / "lora_adapter")
        save_mock.assert_called_once_with(model.trainer.model, tmp_path / "lora_adapter")


def test_load_lora_delegates_to_adapter_loader(tmp_path):
    """Test that LoRA adapter loading is delegated to the shared utility."""
    model = make_stub_yolo()

    with mock.patch("ultralytics.utils.lora.load_lora_adapters", return_value=True) as load_mock:
        assert model.load_lora(tmp_path / "lora_adapter")
        load_mock.assert_called_once_with(model.model, tmp_path / "lora_adapter", merge=False, trainable=False)


def test_load_lora_supports_trainable_reload(tmp_path):
    """Test that load_lora can request a trainable PEFT reload for continued fine-tuning."""
    model = make_stub_yolo()

    with mock.patch("ultralytics.utils.lora.load_lora_adapters", return_value=True) as load_mock:
        assert model.load_lora(tmp_path / "lora_adapter", trainable=True)
        load_mock.assert_called_once_with(model.model, tmp_path / "lora_adapter", merge=False, trainable=True)


def test_train_preserves_active_lora_model():
    """Test that training keeps an already-loaded LoRA model instead of rebuilding it."""
    model = make_stub_yolo()
    model.model.lora_enabled = True
    model.model.yaml = {}
    model.overrides = {"model": "yolo11n.pt"}
    model.ckpt = None
    model.task = "detect"
    model.session = None
    model.callbacks = {}
    model._has_active_lora_model = mock.Mock(return_value=True)

    trainer = mock.Mock()
    trainer.get_model = mock.Mock()
    trainer.model = None
    trainer.train.side_effect = RuntimeError("stop")

    with mock.patch.object(model, "_smart_load", return_value=lambda overrides, _callbacks: trainer), \
         mock.patch("ultralytics.utils.checks.check_pip_update_available"), \
         pytest.raises(RuntimeError, match="stop"):
        model.train(data="coco8.yaml", epochs=1)

    assert trainer.get_model.call_count == 0
    assert trainer.model is model.model


def test_save_lora_only_supports_fallback_backend(tmp_path):
    """Test that fallback-backed adapters still use the shared save utility."""
    model = make_stub_yolo()
    model.trainer.model = mock.Mock(lora_enabled=True, lora_backend="fallback")

    with mock.patch("ultralytics.utils.lora.save_lora_adapters", return_value=True) as save_mock:
        assert model.save_lora_only(tmp_path / "fallback_adapter")
        save_mock.assert_called_once_with(model.trainer.model, tmp_path / "fallback_adapter")


def test_merge_lora_weights_clears_peft_runtime_state():
    """Test that PEFT merges remove stale LoRA runtime metadata."""
    from ultralytics.utils import lora as lora_utils

    class DummyModel:
        pass

    merged_base = object()
    model = DummyModel()
    model.model = mock.Mock()
    model.model.merge_and_unload.return_value = merged_base
    model.lora_enabled = True
    model.lora_backend = "peft"
    model.lora_variant = "ia3"
    model.lora_include_head = False
    model.lora_freeze_bn = True
    model.lora_target_modules = ["0.conv"]
    model.lora_target_audit = {"selected_count": 1}
    model.lora_runtime_metadata = {"effective_backend": "peft"}
    model.use_gradient_checkpointing = True

    with mock.patch.object(lora_utils, "_find_original_model_class", return_value=DummyModel):
        assert lora_utils.merge_lora_weights(model)

    assert model.model is merged_base
    for attr in (
        "lora_enabled",
        "lora_backend",
        "lora_variant",
        "lora_include_head",
        "lora_freeze_bn",
        "lora_target_modules",
        "lora_target_audit",
        "lora_runtime_metadata",
        "use_gradient_checkpointing",
    ):
        assert not hasattr(model, attr)


def test_load_fallback_adapter_state_restores_runtime_metadata(tmp_path):
    """Test that fallback adapter loading restores saved top-level runtime metadata."""
    from ultralytics.utils import lora as lora_utils

    model = torch.nn.Module()
    model.conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1)

    wrapped = lora_utils.ManualLoRAConv(torch.nn.Conv2d(3, 4, kernel_size=3, padding=1), r=2, alpha=4, dropout=0.1)
    torch.save(
        {
            "modules": {"conv": {"r": 2, "alpha": 4, "dropout": 0.1}},
            "state": {
                "conv": {
                    "lora_A": wrapped.lora_A.detach().clone(),
                    "lora_B": wrapped.lora_B.detach().clone(),
                }
            },
        },
        tmp_path / "fallback_adapter.pt",
    )

    payload = {
        "backend": "fallback",
        "variant": "lora",
        "weight_file": "fallback_adapter.pt",
        "include_head": True,
        "freeze_bn": True,
        "target_modules": ["conv"],
        "target_audit": {"selected_count": 1},
        "runtime_metadata": {"effective_backend": "fallback"},
    }

    loaded = lora_utils._load_fallback_adapter_state(model, tmp_path, payload)

    assert getattr(loaded, "lora_enabled", False) is True
    assert loaded.lora_backend == "fallback"
    assert loaded.lora_variant == "lora"
    assert loaded.lora_include_head is True
    assert loaded.lora_freeze_bn is True
    assert loaded.lora_target_modules == ["conv"]
    assert loaded.lora_target_audit["selected_count"] == 1
    assert loaded.lora_runtime_metadata == {"effective_backend": "fallback"}


def test_load_lora_force_replace_clears_stale_fallback_runtime_state(tmp_path):
    """Test that force-replacing fallback adapters clears stale top-level LoRA markers before reload."""
    from ultralytics.utils import lora as lora_utils

    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir()
    (adapter_dir / "fallback_meta.json").write_text('{"backend":"fallback","weight_file":"fallback_adapter.pt"}')

    model = torch.nn.Module()
    model.model = torch.nn.Sequential()
    model.lora_enabled = True
    model.lora_backend = "fallback"
    model.lora_include_head = True
    model.lora_freeze_bn = True
    model.lora_target_modules = ["old.conv"]
    model.lora_target_audit = {"stale": True}
    model.lora_runtime_metadata = {"stale": True}
    model.use_gradient_checkpointing = True

    def _fake_loader(current_model, path, payload):
        assert not hasattr(current_model, "lora_include_head")
        assert not hasattr(current_model, "lora_freeze_bn")
        assert not hasattr(current_model, "lora_target_modules")
        assert not hasattr(current_model, "lora_target_audit")
        assert not hasattr(current_model, "lora_runtime_metadata")
        assert not hasattr(current_model, "use_gradient_checkpointing")
        return current_model

    with mock.patch.object(lora_utils, "_load_fallback_adapter_state", side_effect=_fake_loader):
        assert lora_utils.load_lora_adapters(model, adapter_dir, force_replace=True)


def test_load_lora_adapters_restores_peft_runtime_metadata(tmp_path):
    """Test that PEFT adapter loading restores saved runtime metadata onto the model."""
    from ultralytics.utils import lora as lora_utils

    runtime_payload = {
        "backend": "peft",
        "variant": "ia3",
        "include_head": True,
        "freeze_bn": True,
        "target_modules": ["0.conv"],
        "target_audit": {"selected_count": 1},
        "runtime_metadata": {"effective_backend": "peft"},
    }
    (tmp_path / "runtime_metadata.json").write_text(json.dumps(runtime_payload))

    model = torch.nn.Module()
    model.model = object()

    class _DummyProxy(torch.nn.Module):
        peft_config = {"default": object()}

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 1))
            self.lora_A = torch.nn.Parameter(torch.ones(1))

        @classmethod
        def from_pretrained(cls, base_model, path, is_trainable=False):
            return cls()

        def __len__(self):
            return len(self.layers)

        def __getitem__(self, idx):
            return self.layers[idx]

        def children(self):
            return self.layers.children()

    def _fake_wrap(current_model, config):
        current_model.lora_enabled = True
        return current_model

    with mock.patch.object(lora_utils, "PEFT_AVAILABLE", True), \
         mock.patch.object(lora_utils, "PeftModel", _DummyProxy), \
         mock.patch.object(lora_utils, "PeftProxy", _DummyProxy), \
         mock.patch.object(lora_utils, "_wrap_top_level_lora_model", side_effect=_fake_wrap):
        assert lora_utils.load_lora_adapters(model, tmp_path)

    assert getattr(model, "lora_enabled", False) is True
    assert model.lora_backend == "peft"
    assert model.lora_variant == "ia3"
    assert model.lora_include_head is True
    assert model.lora_freeze_bn is True
    assert model.lora_target_modules == ["0.conv"]
    assert model.lora_target_audit == {"selected_count": 1}
    assert model.lora_runtime_metadata == {"effective_backend": "peft"}


def test_load_lora_adapters_merge_true_calls_merge_for_peft(tmp_path):
    """Test that PEFT adapter loading delegates to merge_lora_weights when merge=True."""
    from ultralytics.utils import lora as lora_utils

    model = torch.nn.Module()
    model.model = object()

    class _DummyProxy(torch.nn.Module):
        peft_config = {"default": object()}

        def __init__(self):
            super().__init__()
            self.layers = torch.nn.Sequential(torch.nn.Linear(1, 1))
            self.lora_A = torch.nn.Parameter(torch.ones(1))

        @classmethod
        def from_pretrained(cls, base_model, path, is_trainable=False):
            return cls()

        def __len__(self):
            return len(self.layers)

        def __getitem__(self, idx):
            return self.layers[idx]

        def children(self):
            return self.layers.children()

    def _fake_wrap(current_model, config):
        current_model.lora_enabled = True
        return current_model

    with mock.patch.object(lora_utils, "PEFT_AVAILABLE", True), \
         mock.patch.object(lora_utils, "PeftModel", _DummyProxy), \
         mock.patch.object(lora_utils, "PeftProxy", _DummyProxy), \
         mock.patch.object(lora_utils, "_wrap_top_level_lora_model", side_effect=_fake_wrap), \
         mock.patch.object(lora_utils, "merge_lora_weights", return_value=True) as merge_mock:
        assert lora_utils.load_lora_adapters(model, tmp_path, merge=True)
        merge_mock.assert_called_once_with(model)


def test_load_lora_compatible_state_dict_restores_adapter_params():
    """Test that LoRA-aware checkpoint loading restores compatible adapter tensors."""
    from ultralytics.utils import lora as lora_utils

    model = torch.nn.Module()
    model.base = torch.nn.Linear(2, 2)
    model.lora_A = torch.nn.Parameter(torch.zeros(2, 2))

    source_state = {k: torch.ones_like(v) for k, v in model.state_dict().items()}
    stats = lora_utils.load_lora_compatible_state_dict(model, source_state, context="unit-test")

    assert stats["adapter_loaded"] == 1
    assert torch.equal(model.lora_A, torch.ones_like(model.lora_A))


def test_load_lora_compatible_state_dict_rejects_adapter_shape_mismatch():
    """Test that changed LoRA rank/targets fail with a clear compatibility error."""
    from ultralytics.utils import lora as lora_utils

    model = torch.nn.Module()
    model.lora_A = torch.nn.Parameter(torch.zeros(2, 2))
    source_state = {"lora_A": torch.zeros(3, 2)}

    with pytest.raises(RuntimeError, match="adapter topology"):
        lora_utils.load_lora_compatible_state_dict(model, source_state, context="unit-test")


def test_build_lora_target_audit_counts_filters():
    """Test compact target audit accounting for user filtering and grouped-conv exclusions."""
    from ultralytics.utils import lora as lora_utils

    audit = lora_utils.build_lora_target_audit(
        valid_targets=["0.conv", "1.conv", "2.cv1"],
        selected_targets=["1.conv"],
        requested_targets=["conv"],
        incompatible_layers=["3.cv2"],
        peft_type="lora",
        rank=8,
    )

    assert audit["valid_count"] == 3
    assert audit["selected_count"] == 1
    assert audit["filtered_by_user_count"] == 2
    assert audit["incompatible_grouped_conv_count"] == 1


def test_merge_manual_lora_conv_preserves_forward_output():
    """Test that merging a fallback LoRA conv reproduces the wrapped forward numerically."""
    from ultralytics.utils.lora import ManualLoRAConv, _merge_manual_lora_conv

    torch.manual_seed(0)
    base_conv = torch.nn.Conv2d(3, 4, kernel_size=3, padding=1, bias=False)
    wrapped = ManualLoRAConv(base_conv, r=2, alpha=4, dropout=0.0)
    wrapped.lora_A.data.normal_(mean=0.0, std=0.2)
    wrapped.lora_B.data.normal_(mean=0.0, std=0.2)

    x = torch.randn(2, 3, 8, 8)
    expected = wrapped(x)

    merged = _merge_manual_lora_conv(wrapped)
    actual = merged(x)

    assert isinstance(merged, torch.nn.Conv2d)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)


def test_unfreeze_detection_head_only_unfreezes_head_params():
    """Test that LoRA head unfreeze only touches the detection head, not frozen backbone params."""
    from ultralytics.nn.modules.head import Detect
    from ultralytics.utils.lora import _unfreeze_detection_head

    class _ToyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1)
            self.detect = Detect(nc=2, ch=(8,))

    model = _ToyModel()
    for _, param in model.named_parameters():
        param.requires_grad = False

    unfrozen = _unfreeze_detection_head(model)
    head_params = {
        name for name, _ in model.named_parameters()
        if name == "detect" or name.startswith("detect.")
    }
    trainable_params = {name for name, param in model.named_parameters() if param.requires_grad}

    assert unfrozen > 0
    assert trainable_params
    assert trainable_params == head_params
    assert all(not param.requires_grad for name, param in model.named_parameters() if name.startswith("stem."))


def test_model_profile():
    """Test profiling of the YOLO model with `profile=True` to assess performance and resource usage."""
    from ultralytics.nn.tasks import DetectionModel

    model = DetectionModel()  # build model
    im = torch.randn(1, 3, 64, 64)  # requires min imgsz=64
    _ = model.predict(im, profile=True)


def test_predict_txt(tmp_path):
    """Test YOLO predictions with file, directory, and pattern sources listed in a text file."""
    file = tmp_path / "sources_multi_row.txt"
    with open(file, "w") as f:
        for src in SOURCES_LIST:
            f.write(f"{src}\n")
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 images


@pytest.mark.skipif(True, reason="disabled for testing")
def test_predict_csv_multi_row(tmp_path):
    """Test YOLO predictions with sources listed in multiple rows of a CSV file."""
    file = tmp_path / "sources_multi_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source"])
        writer.writerows([[src] for src in SOURCES_LIST])
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 images


@pytest.mark.skipif(True, reason="disabled for testing")
def test_predict_csv_single_row(tmp_path):
    """Test YOLO predictions with sources listed in a single row of a CSV file."""
    file = tmp_path / "sources_single_row.csv"
    with open(file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SOURCES_LIST)
    results = YOLO(MODEL)(source=file, imgsz=32)
    assert len(results) == 7  # 1 + 2 + 2 + 2 = 7 images


@pytest.mark.parametrize("model_name", MODELS)
def test_predict_img(model_name):
    """Test YOLO model predictions on various image input types and sources, including online images."""
    channels = 1 if model_name == "yolo11n-grayscale.pt" else 3
    model = YOLO(WEIGHTS_DIR / model_name)
    im = cv2.imread(str(SOURCE), flags=cv2.IMREAD_GRAYSCALE if channels == 1 else cv2.IMREAD_COLOR)  # uint8 NumPy array
    assert len(model(source=Image.open(SOURCE), save=True, verbose=True, imgsz=32)) == 1  # PIL
    assert len(model(source=im, save=True, save_txt=True, imgsz=32)) == 1  # ndarray
    assert len(model(torch.rand((2, channels, 32, 32)), imgsz=32)) == 2  # batch-size 2 Tensor, FP32 0.0-1.0 RGB order
    assert len(model(source=[im, im], save=True, save_txt=True, imgsz=32)) == 2  # batch
    assert len(list(model(source=[im, im], save=True, stream=True, imgsz=32))) == 2  # stream
    assert len(model(torch.zeros(320, 640, channels).numpy().astype(np.uint8), imgsz=32)) == 1  # tensor to numpy
    batch = [
        str(SOURCE),  # filename
        Path(SOURCE),  # Path
        f"{ASSETS_URL}/zidane.jpg?token=123" if ONLINE else SOURCE,  # URI
        im,  # OpenCV
        Image.open(SOURCE),  # PIL
        np.zeros((320, 640, channels), dtype=np.uint8),  # numpy
    ]
    assert len(model(batch, imgsz=32, classes=0)) == len(batch)  # multiple sources in a batch


@pytest.mark.parametrize("model", MODELS)
def test_predict_visualize(model):
    """Test model prediction methods with 'visualize=True' to generate and display prediction visualizations."""
    YOLO(WEIGHTS_DIR / model)(SOURCE, imgsz=32, visualize=True)


def test_predict_gray_and_4ch(tmp_path):
    """Test YOLO prediction on SOURCE converted to grayscale and 4-channel images with various filenames."""
    im = Image.open(SOURCE)

    source_grayscale = tmp_path / "grayscale.jpg"
    source_rgba = tmp_path / "4ch.png"
    source_non_utf = tmp_path / "non_UTF_测试文件_tést_image.jpg"
    source_spaces = tmp_path / "image with spaces.jpg"

    im.convert("L").save(source_grayscale)  # grayscale
    im.convert("RGBA").save(source_rgba)  # 4-ch PNG with alpha
    im.save(source_non_utf)  # non-UTF characters in filename
    im.save(source_spaces)  # spaces in filename

    # Inference
    model = YOLO(MODEL)
    for f in source_rgba, source_grayscale, source_non_utf, source_spaces:
        for source in Image.open(f), cv2.imread(str(f)), f:
            results = model(source, save=True, verbose=True, imgsz=32)
            assert len(results) == 1  # verify that an image was run
        f.unlink()  # cleanup


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.skipif(is_github_action_running(), reason="No auth https://github.com/JuanBindez/pytubefix/issues/166")
def test_youtube():
    """Test YOLO model on a YouTube video stream, handling potential network-related errors."""
    model = YOLO(MODEL)
    try:
        model.predict("https://youtu.be/G17sBkb38XQ", imgsz=96, save=True)
    # Handle internet connection errors and 'urllib.error.HTTPError: HTTP Error 429: Too Many Requests'
    except (urllib.error.HTTPError, ConnectionError) as e:
        LOGGER.error(f"YouTube Test Error: {e}")


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
@pytest.mark.parametrize("model", MODELS)
def test_track_stream(model, tmp_path):
    """Test streaming tracking on a short 10 frame video using ByteTrack tracker and different GMC methods.

    Note imgsz=160 required for tracking for higher confidence and better matches.
    """
    if model == "yolo11n-cls.pt":  # classification model not supported for tracking
        return
    video_url = f"{ASSETS_URL}/decelera_portrait_min.mov"
    model = YOLO(model)
    model.track(video_url, imgsz=160, tracker="bytetrack.yaml")
    model.track(video_url, imgsz=160, tracker="botsort.yaml", save_frames=True)  # test frame saving also

    # Test Global Motion Compensation (GMC) methods and ReID
    for gmc, reidm in zip(["orb", "sift", "ecc"], ["auto", "auto", "yolo11n-cls.pt"]):
        default_args = YAML.load(ROOT / "cfg/trackers/botsort.yaml")
        custom_yaml = tmp_path / f"botsort-{gmc}.yaml"
        YAML.save(custom_yaml, {**default_args, "gmc_method": gmc, "with_reid": True, "model": reidm})
        model.track(video_url, imgsz=160, tracker=custom_yaml)


@pytest.mark.parametrize("task,weight,data", TASK_MODEL_DATA)
def test_val(task: str, weight: str, data: str) -> None:
    """Test the validation mode of the YOLO model."""
    model = YOLO(weight)
    for plots in {True, False}:  # Test both cases i.e. plots=True and plots=False
        metrics = model.val(data=data, imgsz=32, plots=plots)
        metrics.to_df()
        metrics.to_csv()
        metrics.to_json()
        # Tests for confusion matrix export
        metrics.confusion_matrix.to_df()
        metrics.confusion_matrix.to_csv()
        metrics.confusion_matrix.to_json()


@pytest.mark.skipif(IS_JETSON or IS_RASPBERRYPI, reason="Edge devices not intended for training")
def test_train_scratch():
    """Test training the YOLO model from scratch using the provided configuration."""
    model = YOLO(CFG)
    model.train(data="coco8.yaml", epochs=2, imgsz=32, cache="disk", batch=-1, close_mosaic=1, name="model")
    model(SOURCE)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_train_ndjson():
    """Test training the YOLO model using NDJSON format dataset."""
    model = YOLO(WEIGHTS_DIR / "yolo11n.pt")
    model.train(data=f"{ASSETS_URL}/coco8-ndjson.ndjson", epochs=1, imgsz=32)


@pytest.mark.parametrize("scls", [False, True])
def test_train_pretrained(scls):
    """Test training of the YOLO model starting from a pre-trained checkpoint."""
    model = YOLO(WEIGHTS_DIR / "yolo11n-seg.pt")
    model.train(
        data="coco8-seg.yaml", epochs=1, imgsz=32, cache="ram", copy_paste=0.5, mixup=0.5, name=0, single_cls=scls
    )
    model(SOURCE)


def test_all_model_yamls():
    """Test YOLO model creation for all available YAML configurations in the `cfg/models` directory."""
    for m in (ROOT / "cfg" / "models").rglob("*.yaml"):
        if "rtdetr" in m.name:
            if TORCH_1_11:
                _ = RTDETR(m.name)(SOURCE, imgsz=640)  # must be 640
        else:
            YOLO(m.name)


@pytest.mark.skipif(WINDOWS, reason="Windows slow CI export bug https://github.com/ultralytics/ultralytics/pull/16003")
def test_workflow():
    """Test the complete workflow including training, validation, prediction, and exporting."""
    model = YOLO(MODEL)
    model.train(data="coco8.yaml", epochs=1, imgsz=32, optimizer="SGD")
    model.val(imgsz=32)
    model.predict(SOURCE, imgsz=32)
    model.export(format="torchscript")  # WARNING: Windows slow CI export bug


def test_predict_callback_and_setup():
    """Test callback functionality during YOLO prediction setup and execution."""

    def on_predict_batch_end(predictor):
        """Callback function that handles operations at the end of a prediction batch."""
        path, im0s, _ = predictor.batch
        im0s = im0s if isinstance(im0s, list) else [im0s]
        bs = [predictor.dataset.bs for _ in range(len(path))]
        predictor.results = zip(predictor.results, im0s, bs)  # results is list[batch_size]

    model = YOLO(MODEL)
    model.add_callback("on_predict_batch_end", on_predict_batch_end)

    dataset = load_inference_source(source=SOURCE)
    bs = dataset.bs  # access predictor properties
    results = model.predict(dataset, stream=True, imgsz=160)  # source already setup
    for r, im0, bs in results:
        print("test_callback", im0.shape)
        print("test_callback", bs)
        boxes = r.boxes  # Boxes object for bbox outputs
        print(boxes)


@pytest.mark.parametrize("model", MODELS)
def test_results(model: str, tmp_path):
    """Test YOLO model results processing and output in various formats."""
    im = f"{ASSETS_URL}/boats.jpg" if model == "yolo11n-obb.pt" else SOURCE
    results = YOLO(WEIGHTS_DIR / model)([im, im], imgsz=160)
    for r in results:
        assert len(r), f"'{model}' results should not be empty!"
        r = r.cpu().numpy()
        print(r, len(r), r.path)  # print numpy attributes
        r = r.to(device="cpu", dtype=torch.float32)
        r.save_txt(txt_file=tmp_path / "runs/tests/label.txt", save_conf=True)
        r.save_crop(save_dir=tmp_path / "runs/tests/crops/")
        r.to_df(decimals=3)  # Align to_ methods: https://docs.ultralytics.com/modes/predict/#working-with-results
        r.to_csv()
        r.to_json(normalize=True)
        r.plot(pil=True, save=True, filename=tmp_path / "results_plot_save.jpg")
        r.plot(conf=True, boxes=True)
        print(r, len(r), r.path)  # print after methods


def test_labels_and_crops():
    """Test output from prediction args for saving YOLO detection labels and crops."""
    imgs = [SOURCE, ASSETS / "zidane.jpg"]
    results = YOLO(WEIGHTS_DIR / "yolo11n.pt")(imgs, imgsz=160, save_txt=True, save_crop=True)
    save_path = Path(results[0].save_dir)
    for r in results:
        im_name = Path(r.path).stem
        cls_idxs = r.boxes.cls.int().tolist()
        # Check correct detections
        assert cls_idxs == ([0, 7, 0, 0] if r.path.endswith("bus.jpg") else [0, 0, 0])  # bus.jpg and zidane.jpg classes
        # Check label path
        labels = save_path / f"labels/{im_name}.txt"
        assert labels.exists()
        # Check detections match label count
        assert len(r.boxes.data) == len([line for line in labels.read_text().splitlines() if line])
        # Check crops path and files
        crop_dirs = list((save_path / "crops").iterdir())
        crop_files = [f for p in crop_dirs for f in p.glob("*")]
        # Crop directories match detections
        assert all(r.names.get(c) in {d.name for d in crop_dirs} for c in cls_idxs)
        # Same number of crops as detections
        assert len([f for f in crop_files if im_name in f.name]) == len(r.boxes.data)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_data_utils(tmp_path):
    """Test utility functions in ultralytics/data/utils.py, including dataset stats and auto-splitting."""
    from ultralytics.data.split import autosplit
    from ultralytics.data.utils import HUBDatasetStats
    from ultralytics.utils.downloads import zip_directory

    # from ultralytics.utils.files import WorkingDirectory
    # with WorkingDirectory(ROOT.parent / 'tests'):

    for task in TASKS:
        file = Path(TASK2DATA[task]).with_suffix(".zip")  # i.e. coco8.zip
        download(f"https://github.com/ultralytics/hub/raw/main/example_datasets/{file}", unzip=False, dir=tmp_path)
        stats = HUBDatasetStats(tmp_path / file, task=task)
        stats.get_json(save=True)
        stats.process_images()

    autosplit(tmp_path / "coco8")
    zip_directory(tmp_path / "coco8/images/val")  # zip


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_data_converter(tmp_path):
    """Test dataset conversion functions from COCO to YOLO format and class mappings."""
    from ultralytics.data.converter import coco80_to_coco91_class, convert_coco

    download(f"{ASSETS_URL}/instances_val2017.json", dir=tmp_path)
    convert_coco(
        labels_dir=tmp_path, save_dir=tmp_path / "yolo_labels", use_segments=True, use_keypoints=False, cls91to80=True
    )
    coco80_to_coco91_class()


def test_data_annotator(tmp_path):
    """Test automatic annotation of data using detection and segmentation models."""
    from ultralytics.data.annotator import auto_annotate

    auto_annotate(
        ASSETS,
        det_model=WEIGHTS_DIR / "yolo11n.pt",
        sam_model=WEIGHTS_DIR / "mobile_sam.pt",
        output_dir=tmp_path / "auto_annotate_labels",
    )


def test_events():
    """Test event sending functionality."""
    from ultralytics.utils.events import Events

    events = Events()
    events.enabled = True
    cfg = copy(DEFAULT_CFG)  # does not require deepcopy
    cfg.mode = "test"
    events(cfg)


def test_cfg_init():
    """Test configuration initialization utilities from the 'ultralytics.cfg' module."""
    from ultralytics.cfg import check_dict_alignment, copy_default_cfg, smart_value

    with contextlib.suppress(SyntaxError):
        check_dict_alignment({"a": 1}, {"b": 2})
    copy_default_cfg()
    (Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")).unlink(missing_ok=False)

    # Test smart_value() with comprehensive cases
    # Test None conversion
    assert smart_value("none") is None
    assert smart_value("None") is None
    assert smart_value("NONE") is None

    # Test boolean conversion
    assert smart_value("true") is True
    assert smart_value("True") is True
    assert smart_value("TRUE") is True
    assert smart_value("false") is False
    assert smart_value("False") is False
    assert smart_value("FALSE") is False

    # Test numeric conversion (ast.literal_eval)
    assert smart_value("42") == 42
    assert smart_value("-42") == -42
    assert smart_value("3.14") == 3.14
    assert smart_value("-3.14") == -3.14
    assert smart_value("1e-3") == 0.001

    # Test list/tuple conversion (ast.literal_eval)
    assert smart_value("[1, 2, 3]") == [1, 2, 3]
    assert smart_value("(1, 2, 3)") == (1, 2, 3)
    assert smart_value("[640, 640]") == [640, 640]

    # Test dict conversion (ast.literal_eval)
    assert smart_value("{'a': 1, 'b': 2}") == {"a": 1, "b": 2}

    # Test string fallback (when ast.literal_eval fails)
    assert smart_value("some_string") == "some_string"
    assert smart_value("path/to/file") == "path/to/file"
    assert smart_value("hello world") == "hello world"

    # Test that code injection is prevented (ast.literal_eval safety)
    # These should return strings, not execute code
    assert smart_value("__import__('os').system('ls')") == "__import__('os').system('ls')"
    assert smart_value("eval('1+1')") == "eval('1+1')"
    assert smart_value("exec('x=1')") == "exec('x=1')"


def test_utils_init():
    """Test initialization utilities in the Ultralytics library."""
    from ultralytics.utils import get_ubuntu_version, is_github_action_running

    get_ubuntu_version()
    is_github_action_running()


def test_utils_checks():
    """Test various utility checks for filenames, git status, requirements, image sizes, and versions."""
    checks.check_yolov5u_filename("yolov5n.pt")
    checks.check_requirements("numpy")  # check requirements.txt
    checks.check_imgsz([600, 600], max_dim=1)
    checks.check_imshow(warn=True)
    checks.check_version("ultralytics", "8.0.0")
    checks.print_args()


@pytest.mark.skipif(WINDOWS, reason="Windows profiling is extremely slow (cause unknown)")
def test_utils_benchmarks():
    """Benchmark model performance using 'ProfileModels' from 'ultralytics.utils.benchmarks'."""
    from ultralytics.utils.benchmarks import ProfileModels

    ProfileModels(["yolo11n.yaml"], imgsz=32, min_time=1, num_timed_runs=3, num_warmup_runs=1).run()


def test_utils_torchutils():
    """Test Torch utility functions including profiling and FLOP calculations."""
    from ultralytics.nn.modules.conv import Conv
    from ultralytics.utils.torch_utils import get_flops_with_torch_profiler, profile_ops, time_sync

    x = torch.randn(1, 64, 20, 20)
    m = Conv(64, 64, k=1, s=2)

    profile_ops(x, [m], n=3)
    get_flops_with_torch_profiler(m)
    time_sync()


def test_utils_ops():
    """Test utility operations for coordinate transformations and normalizations."""
    from ultralytics.utils.ops import (
        ltwh2xywh,
        ltwh2xyxy,
        make_divisible,
        xywh2ltwh,
        xywh2xyxy,
        xywhn2xyxy,
        xywhr2xyxyxyxy,
        xyxy2ltwh,
        xyxy2xywh,
        xyxy2xywhn,
        xyxyxyxy2xywhr,
    )

    make_divisible(17, torch.tensor([8]))

    boxes = torch.rand(10, 4)  # xywh
    torch.allclose(boxes, xyxy2xywh(xywh2xyxy(boxes)))
    torch.allclose(boxes, xyxy2xywhn(xywhn2xyxy(boxes)))
    torch.allclose(boxes, ltwh2xywh(xywh2ltwh(boxes)))
    torch.allclose(boxes, xyxy2ltwh(ltwh2xyxy(boxes)))

    boxes = torch.rand(10, 5)  # xywhr for OBB
    boxes[:, 4] = torch.randn(10) * 30
    torch.allclose(boxes, xyxyxyxy2xywhr(xywhr2xyxyxyxy(boxes)), rtol=1e-3)


def test_utils_files(tmp_path):
    """Test file handling utilities including file age, date, and paths with spaces."""
    from ultralytics.utils.files import file_age, file_date, get_latest_run, spaces_in_path

    file_age(SOURCE)
    file_date(SOURCE)
    get_latest_run(ROOT / "runs")

    path = tmp_path / "path/with spaces"
    path.mkdir(parents=True, exist_ok=True)
    with spaces_in_path(path) as new_path:
        print(new_path)


@pytest.mark.slow
def test_utils_patches_torch_save(tmp_path):
    """Test torch_save backoff when _torch_save raises RuntimeError."""
    from unittest.mock import MagicMock, patch

    from ultralytics.utils.patches import torch_save

    mock = MagicMock(side_effect=RuntimeError)

    with patch("ultralytics.utils.patches._torch_save", new=mock):
        with pytest.raises(RuntimeError):
            torch_save(torch.zeros(1), tmp_path / "test.pt")

    assert mock.call_count == 4, "torch_save was not attempted the expected number of times"


def test_nn_modules_conv():
    """Test Convolutional Neural Network modules including CBAM, Conv2, and ConvTranspose."""
    from ultralytics.nn.modules.conv import CBAM, Conv2, ConvTranspose, DWConvTranspose2d, Focus

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # Run all modules not otherwise covered in tests
    DWConvTranspose2d(c1, c2)(x)
    ConvTranspose(c1, c2)(x)
    Focus(c1, c2)(x)
    CBAM(c1)(x)

    # Fuse ops
    m = Conv2(c1, c2)
    m.fuse_convs()
    m(x)


def test_nn_modules_block():
    """Test various neural network block modules."""
    from ultralytics.nn.modules.block import C1, C3TR, BottleneckCSP, C3Ghost, C3x

    c1, c2 = 8, 16  # input and output channels
    x = torch.zeros(4, c1, 10, 10)  # BCHW

    # Run all modules not otherwise covered in tests
    C1(c1, c2)(x)
    C3x(c1, c2)(x)
    C3TR(c1, c2)(x)
    C3Ghost(c1, c2)(x)
    BottleneckCSP(c1, c2)(x)


@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_hub():
    """Test Ultralytics HUB functionalities."""
    from ultralytics.hub import export_fmts_hub, logout
    from ultralytics.hub.utils import smart_request

    export_fmts_hub()
    logout()
    smart_request("GET", "https://github.com", progress=True)


@pytest.fixture
def image():
    """Load and return an image from a predefined source (OpenCV BGR)."""
    return cv2.imread(str(SOURCE))


@pytest.mark.parametrize(
    "auto_augment, erasing, force_color_jitter",
    [
        (None, 0.0, False),
        ("randaugment", 0.5, True),
        ("augmix", 0.2, False),
        ("autoaugment", 0.0, True),
    ],
)
def test_classify_transforms_train(image, auto_augment, erasing, force_color_jitter):
    """Test classification transforms during training with various augmentations."""
    from ultralytics.data.augment import classify_augmentations

    transform = classify_augmentations(
        size=224,
        mean=(0.5, 0.5, 0.5),
        std=(0.5, 0.5, 0.5),
        scale=(0.08, 1.0),
        ratio=(3.0 / 4.0, 4.0 / 3.0),
        hflip=0.5,
        vflip=0.5,
        auto_augment=auto_augment,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.4,
        force_color_jitter=force_color_jitter,
        erasing=erasing,
    )

    transformed_image = transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))

    assert transformed_image.shape == (3, 224, 224)
    assert torch.is_tensor(transformed_image)
    assert transformed_image.dtype == torch.float32


@pytest.mark.slow
@pytest.mark.skipif(not ONLINE, reason="environment is offline")
def test_model_tune():
    """Tune YOLO model for performance improvement."""
    YOLO("yolo11n-pose.pt").tune(data="coco8-pose.yaml", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")
    YOLO("yolo11n-cls.pt").tune(data="imagenet10", plots=False, imgsz=32, epochs=1, iterations=2, device="cpu")


def test_model_embeddings():
    """Test YOLO model embeddings extraction functionality."""
    model_detect = YOLO(MODEL)
    model_segment = YOLO(WEIGHTS_DIR / "yolo11n-seg.pt")

    for batch in [SOURCE], [SOURCE, SOURCE]:  # test batch size 1 and 2
        assert len(model_detect.embed(source=batch, imgsz=32)) == len(batch)
        assert len(model_segment.embed(source=batch, imgsz=32)) == len(batch)


@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="YOLOWorld with CLIP is not supported in Python 3.12")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="YOLOWorld with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
def test_yolo_world():
    """Test YOLO world models with CLIP support."""
    model = YOLO(WEIGHTS_DIR / "yolov8s-world.pt")  # no YOLO11n-world model yet
    model.set_classes(["tree", "window"])
    model(SOURCE, conf=0.01)

    model = YOLO(WEIGHTS_DIR / "yolov8s-worldv2.pt")  # no YOLO11n-world model yet
    # Training from a pretrained model. Eval is included at the final stage of training.
    # Use dota8.yaml which has fewer categories to reduce the inference time of CLIP model
    model.train(
        data="dota8.yaml",
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
    )

    # test WorWorldTrainerFromScratch
    from ultralytics.models.yolo.world.train_world import WorldTrainerFromScratch

    model = YOLO("yolov8s-worldv2.yaml")  # no YOLO11n-world model yet
    model.train(
        data={"train": {"yolo_data": ["dota8.yaml"]}, "val": {"yolo_data": ["dota8.yaml"]}},
        epochs=1,
        imgsz=32,
        cache="disk",
        close_mosaic=1,
        trainer=WorldTrainerFromScratch,
    )


@pytest.mark.skipif(not TORCH_1_13, reason="YOLOE with CLIP requires torch>=1.13")
@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="YOLOE with CLIP is not supported in Python 3.12")
@pytest.mark.skipif(
    checks.IS_PYTHON_3_8 and LINUX and ARM64,
    reason="YOLOE with CLIP is not supported in Python 3.8 and aarch64 Linux",
)
def test_yoloe():
    """Test YOLOE models with MobileClip support."""
    # Predict
    # text-prompts
    model = YOLO(WEIGHTS_DIR / "yoloe-11s-seg.pt")
    names = ["person", "bus"]
    model.set_classes(names, model.get_text_pe(names))
    model(SOURCE, conf=0.01)

    from ultralytics import YOLOE
    from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

    # visual-prompts
    visuals = dict(
        bboxes=np.array([[221.52, 405.8, 344.98, 857.54], [120, 425, 160, 445]]),
        cls=np.array([0, 1]),
    )
    model.predict(
        SOURCE,
        visual_prompts=visuals,
        predictor=YOLOEVPSegPredictor,
    )

    # Val
    model = YOLOE(WEIGHTS_DIR / "yoloe-11s-seg.pt")
    # text prompts
    model.val(data="coco128-seg.yaml", imgsz=32)
    # visual prompts
    model.val(data="coco128-seg.yaml", load_vp=True, imgsz=32)

    # Train, fine-tune
    from ultralytics.models.yolo.yoloe import YOLOEPESegTrainer, YOLOESegTrainerFromScratch

    model = YOLOE("yoloe-11s-seg.pt")
    model.train(
        data="coco128-seg.yaml",
        epochs=1,
        close_mosaic=1,
        trainer=YOLOEPESegTrainer,
        imgsz=32,
    )
    # Train, from scratch
    model = YOLOE("yoloe-11s-seg.yaml")
    model.train(
        data=dict(train=dict(yolo_data=["coco128-seg.yaml"]), val=dict(yolo_data=["coco128-seg.yaml"])),
        epochs=1,
        close_mosaic=1,
        trainer=YOLOESegTrainerFromScratch,
        imgsz=32,
    )

    # prompt-free
    # predict
    model = YOLOE(WEIGHTS_DIR / "yoloe-11s-seg-pf.pt")
    model.predict(SOURCE)
    # val
    model = YOLOE("yoloe-11s-seg.pt")  # or select yoloe-m/l-seg.pt for different sizes
    model.val(data="coco128-seg.yaml", imgsz=32)


def test_yolov10():
    """Test YOLOv10 model training, validation, and prediction functionality."""
    model = YOLO("yolov10n.yaml")
    # train/val/predict
    model.train(data="coco8.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8.yaml", imgsz=32)
    model.predict(imgsz=32, save_txt=True, save_crop=True, augment=True)
    model(SOURCE)


def test_multichannel():
    """Test YOLO model multi-channel training, validation, and prediction functionality."""
    model = YOLO("yolo11n.pt")
    model.train(data="coco8-multispectral.yaml", epochs=1, imgsz=32, close_mosaic=1, cache="disk")
    model.val(data="coco8-multispectral.yaml")
    im = np.zeros((32, 32, 10), dtype=np.uint8)
    model.predict(source=im, imgsz=32, save_txt=True, save_crop=True, augment=True)
    model.export(format="onnx")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_grayscale(task: str, model: str, data: str, tmp_path) -> None:
    """Test YOLO model grayscale training, validation, and prediction functionality."""
    if task == "classify":  # not support grayscale classification yet
        return
    grayscale_data = tmp_path / f"{Path(data).stem}-grayscale.yaml"
    data = check_det_dataset(data)
    data["channels"] = 1  # add additional channels key for grayscale
    YAML.save(data=data, file=grayscale_data)
    # remove npy files in train/val splits if exists, might be created by previous tests
    for split in {"train", "val"}:
        for npy_file in (Path(data["path"]) / data[split]).glob("*.npy"):
            npy_file.unlink()

    model = YOLO(model)
    model.train(data=grayscale_data, epochs=1, imgsz=32, close_mosaic=1)
    model.val(data=grayscale_data)
    im = np.zeros((32, 32, 1), dtype=np.uint8)
    model.predict(source=im, imgsz=32, save_txt=True, save_crop=True, augment=True)
    export_model = model.export(format="onnx")

    model = YOLO(export_model, task=task)
    model.predict(source=im, imgsz=32)
