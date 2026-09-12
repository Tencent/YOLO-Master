"""Cached detector topology, checkpoints, and scalar auxiliary-loss composition."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from ultralytics.nn import D1FoundationDetectionModel
from ultralytics.nn.foundation.cache import FeatureCacheReader
from ultralytics.nn.mixture_loss import CompositeCriterion
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.routing_protocol import get_aux_record, iter_aux_records

SOURCE_NAMES = ("block4", "block8", "block12")


def make_features(
    *,
    batch: int = 1,
    height: int = 4,
    width: int = 4,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(0)
    return {
        name: torch.randn(batch, 384, height, width, device=device, dtype=dtype, generator=generator)
        for name in SOURCE_NAMES
    }


def empty_batch(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    device = features["block4"].device
    return {
        "features": features,
        "batch_idx": torch.empty(0, device=device),
        "cls": torch.empty(0, 1, device=device),
        "bboxes": torch.empty(0, 4, device=device),
    }


def test_default_config_builds_named_downstream_graph() -> None:
    model = D1FoundationDetectionModel()

    assert model.source_names == SOURCE_NAMES
    assert tuple(model.adapter.out_channels) == (64, 128, 256)
    assert tuple(model.mixtures) == ("p3", "p4", "p5")
    assert isinstance(model.model[-1], Detect)
    assert model.detect is model.model[-1]
    assert model.end2end
    assert torch.equal(model.stride, torch.tensor([8.0, 16.0, 32.0]))
    assert model.teacher_reference["model_id"] == "facebook/dinov3-vits16-pretrain-lvd1689m"
    assert tuple(model.teacher_reference["output_layers"]) == (4, 8, 12)


def test_training_forward_connects_adapter_mixtures_and_detect() -> None:
    model = D1FoundationDetectionModel().train()
    predictions = model(make_features(batch=2))

    assert tuple(predictions) == ("one2many", "one2one")
    for branch in predictions.values():
        assert tuple(branch["boxes"].shape) == (2, 4, 84)
        assert tuple(branch["scores"].shape) == (2, 80, 84)
        assert [tuple(value.shape) for value in branch["feats"]] == [
            (2, 64, 8, 8),
            (2, 128, 4, 4),
            (2, 256, 2, 2),
        ]
        assert torch.isfinite(branch["boxes"]).all()
        assert torch.isfinite(branch["scores"]).all()
    assert all(tuple(model.mixtures[name].routing_probs.shape) == (2, 4) for name in model.mixtures)


def test_eval_forward_decodes_with_explicit_strides() -> None:
    model = D1FoundationDetectionModel().eval()
    with torch.no_grad():
        decoded, raw = model(make_features())

    assert tuple(decoded.shape) == (1, 84, 6)
    assert torch.isfinite(decoded).all()
    assert tuple(raw) == ("one2many", "one2one")
    assert torch.equal(model.detect.stride.cpu(), torch.tensor([8.0, 16.0, 32.0]))
    assert set(model.detect.strides.flatten().cpu().tolist()) == {8.0, 16.0, 32.0}


def test_native_e2e_loss_and_composite_aux_backward() -> None:
    model = D1FoundationDetectionModel().train()
    loss, items = model(empty_batch(make_features()))

    assert isinstance(model.criterion, CompositeCriterion)
    assert loss.ndim == 0
    assert tuple(items.shape) == (7,)
    assert torch.isfinite(loss).all()
    assert torch.isfinite(items).all()
    assert model._last_mixture_aux_loss > 0
    assert model._mixture_aux_diagnostics["counts_by_kind"]["latent"] == 3
    assert items[3].item() == pytest.approx(model.last_latent_aux_metrics["latent_balance_loss"])
    assert items[4].item() == pytest.approx(model.last_latent_aux_metrics["latent_z_loss"])
    assert items[5].item() == pytest.approx(model.last_latent_aux_metrics["latent_aux_loss"])
    assert items[6].item() == pytest.approx(model.last_latent_aux_metrics["mixture_aux_loss"])
    loss.sum().backward()

    adapter_weight = model.adapter.branches["p3"]["block4"][0][0].weight
    detect_weight = model.detect.cv3[0][-1].weight
    assert adapter_weight.grad is not None and adapter_weight.grad.abs().sum() > 0
    assert detect_weight.grad is not None and detect_weight.grad.abs().sum() > 0
    for mixture in model.mixtures.values():
        grad = mixture.router.expert_head.bias.grad
        assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0


def test_raw_rgb_tensor_and_batch_without_features_fail_fast() -> None:
    model = D1FoundationDetectionModel()
    with pytest.raises(TypeError, match="feature mapping"):
        model(torch.zeros(1, 3, 640, 640))
    with pytest.raises(KeyError, match="features"):
        model.loss({"batch_idx": torch.empty(0), "cls": torch.empty(0, 1), "bboxes": torch.empty(0, 4)})


@pytest.mark.parametrize(
    "mutation",
    [
        lambda cfg: cfg["detect"].update(strides=[4, 8, 16]),
        lambda cfg: cfg["detect"].update(end2end="true"),
        lambda cfg: cfg["input_features"].update(channels=0),
        lambda cfg: cfg["latent_mixture"].update(unknown_option=1),
        lambda cfg: cfg.update(model_type="rgb_detection"),
    ],
)
def test_invalid_model_contract_fails_fast(mutation) -> None:
    config = D1FoundationDetectionModel().config_dict()
    mutation(config)
    with pytest.raises((TypeError, ValueError)):
        D1FoundationDetectionModel(config)


def test_checkpoint_contains_only_downstream_state_and_strictly_round_trips() -> None:
    torch.manual_seed(1)
    source = D1FoundationDetectionModel().eval()
    features = make_features()
    with torch.no_grad():
        expected = source(features)[0]
    payload = source.checkpoint_payload()

    assert set(payload) == {"schema_version", "state_dict", "config", "teacher_reference"}
    assert payload["schema_version"] == "d1-downstream-v1"
    assert payload["teacher_reference"] == source.teacher_reference
    assert all(key.startswith("model.") for key in payload["state_dict"])
    assert not any("teacher" in key or "dinov3" in key for key in payload["state_dict"])

    restored = D1FoundationDetectionModel.from_checkpoint_payload(payload).eval()
    with torch.no_grad():
        actual = restored(features)[0]
    assert torch.equal(expected, actual)

    mismatched = deepcopy(payload)
    mismatched["teacher_reference"]["model_id"] = "different/teacher"
    with pytest.raises(ValueError, match="teacher_reference"):
        D1FoundationDetectionModel.from_checkpoint_payload(mismatched)


def test_real_feature_cache_cuda_forward() -> None:
    cache_value = os.environ.get("D1_WP2_CACHE")
    if not cache_value:
        pytest.skip("D1_WP2_CACHE is not configured")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    reader = FeatureCacheReader(Path(cache_value))
    sample_id = min(reader.records)
    cached = reader.get(sample_id)
    features = {name: value.unsqueeze(0).cuda(non_blocking=True) for name, value in cached.items()}
    model = D1FoundationDetectionModel().cuda().eval()

    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        decoded, raw = model(features)

    assert tuple(decoded.shape) == (1, 300, 6)
    assert torch.isfinite(decoded).all()
    assert [tuple(value.shape) for value in raw["one2many"]["feats"]] == [
        (1, 64, 80, 80),
        (1, 128, 40, 40),
        (1, 256, 20, 20),
    ]


PYRAMID_NAMES = ("p3", "p4", "p5")


def make_aux_features(*, batch: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(17)
    return {name: torch.randn(batch, 384, 4, 4, generator=generator) for name in SOURCE_NAMES}


def test_three_latent_publications_are_collected_once_and_reported() -> None:
    model = D1FoundationDetectionModel().train()
    for mixture in model.mixtures.values():
        with torch.no_grad():
            mixture.router.expert_head.bias.copy_(torch.tensor([0.3, -0.2, 0.1, -0.1]))

    loss, items = model(empty_batch(make_aux_features()))

    assert loss.ndim == 0
    assert tuple(items.shape) == (7,)
    diagnostics = model._mixture_aux_diagnostics
    assert diagnostics["counts_by_kind"]["latent"] == 3
    assert diagnostics["stale_skipped"] == 0
    assert diagnostics["eval_skipped"] == 0
    assert diagnostics["duplicate_skipped"] == 0
    assert diagnostics["modules"] == ["LatentMixture"] * 3
    assert len(diagnostics["values_by_kind"]["latent"]) == 3

    records = iter_aux_records(model)
    assert len(records) == 3
    assert {record.kind for _module, record in records} == {"latent"}
    assert {record.step for _module, record in records} == {diagnostics["step"]}
    assert all(record.training and record.value.requires_grad for _module, record in records)

    metrics = model.last_latent_aux_metrics
    assert metrics["latent_publications"] == 3
    assert metrics["aux_step"] == diagnostics["step"]
    assert items[3].item() == pytest.approx(metrics["latent_balance_loss"])
    assert items[4].item() == pytest.approx(metrics["latent_z_loss"])
    assert items[5].item() == pytest.approx(metrics["latent_aux_loss"])
    assert items[6].item() == pytest.approx(metrics["mixture_aux_loss"])
    assert metrics["latent_aux_loss"] == pytest.approx(sum(diagnostics["values_by_kind"]["latent"]))
    assert metrics["latent_balance_loss"] > 0.0

    for name in PYRAMID_NAMES:
        snapshot = model.mixtures[name].routing_snapshot()
        expected = (
            model.mixtures[name].balance_loss_coeff * snapshot["balance_loss"]
            + model.mixtures[name].router_z_loss_coeff * snapshot["z_loss"]
        )
        assert snapshot["aux_loss"] == pytest.approx(expected)
        assert metrics[f"{name}_aux_loss"] == pytest.approx(expected)


def test_new_forward_step_replaces_publications_without_accumulation() -> None:
    model = D1FoundationDetectionModel().train()
    model(empty_batch(make_aux_features()))
    first = deepcopy(model._mixture_aux_diagnostics)

    model(empty_batch(make_aux_features()))
    second = model._mixture_aux_diagnostics

    assert second["step"] > first["step"]
    assert first["counts_by_kind"]["latent"] == second["counts_by_kind"]["latent"] == 3
    assert len(iter_aux_records(model)) == 3
    assert all(get_aux_record(model.mixtures[name]).step == second["step"] for name in PYRAMID_NAMES)


def test_enabled_latent_aux_reaches_every_router() -> None:
    model = D1FoundationDetectionModel().train()

    loss, _items = model(empty_batch(make_aux_features(batch=2)))
    loss.sum().backward()

    assert model.last_latent_aux_metrics["latent_aux_loss"] > 0.0
    assert model.last_latent_aux_metrics["mixture_aux_loss"] > 0.0
    for name in PYRAMID_NAMES:
        gradient = model.mixtures[name].router.expert_head.bias.grad
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_zero_global_latent_gain_disables_only_the_applied_aux() -> None:
    config = D1FoundationDetectionModel().config_dict()
    config["loss"]["latent_aux_gain"] = 0.0
    model = D1FoundationDetectionModel(config).train()

    loss, items = model(empty_batch(make_aux_features(batch=2)))

    assert items[4] > 0.0
    assert items[5] > 0.0
    assert items[6] == 0.0
    assert model.last_latent_aux_metrics["mixture_aux_loss"] == 0.0
    loss.sum().backward()
    for name in PYRAMID_NAMES:
        gradient = model.mixtures[name].router.expert_head.bias.grad
        assert gradient is not None
        assert torch.equal(gradient, torch.zeros_like(gradient))


def test_disabled_latent_aux_is_exact_graph_connected_zero() -> None:
    config = D1FoundationDetectionModel().config_dict()
    config["latent_mixture"]["balance_loss_coeff"] = 0.0
    config["latent_mixture"]["router_z_loss_coeff"] = 0.0
    config["loss"]["latent_aux_gain"] = 0.0
    model = D1FoundationDetectionModel(config).train()

    loss, items = model(empty_batch(make_aux_features(batch=2)))

    assert torch.equal(items[3:], torch.zeros_like(items[3:]))
    assert model._mixture_aux_diagnostics["counts_by_kind"]["latent"] == 3
    assert all(
        value == 0.0
        for key, value in model.last_latent_aux_metrics.items()
        if key not in {"aux_step", "latent_publications"}
    )
    loss.sum().backward()
    for name in PYRAMID_NAMES:
        gradient = model.mixtures[name].router.expert_head.bias.grad
        assert gradient is not None
        assert torch.equal(gradient, torch.zeros_like(gradient))


@pytest.mark.parametrize("field,value", [("counts_by_kind", {"latent": 2}), ("duplicate_skipped", 1)])
def test_incomplete_or_duplicate_collection_fails_fast(field: str, value: object) -> None:
    model = D1FoundationDetectionModel().train()
    model(empty_batch(make_aux_features()))
    model._mixture_aux_diagnostics[field] = value

    with pytest.raises(RuntimeError, match="exactly 3|duplicate_skipped"):
        model.mixture_aux_report_items()
