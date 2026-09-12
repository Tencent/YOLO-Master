"""CPU contracts for the released YOLOE P1 P5 text-router adapter."""

from __future__ import annotations

import hashlib

import pytest
import torch

from ultralytics.nn.mixture_loss import CompositeCriterion
from ultralytics.nn.modules.moa import C2fMoA
from ultralytics.nn.modules.mot import TextConditionedMoT
from ultralytics.nn.modules.routing_protocol import (
    clear_aux_records,
    collect_aux_loss,
    current_aux_step,
    get_aux_record,
    reset_routing_runtime_state,
)
from ultralytics.nn.tasks import YOLOEModel, YOLOESegModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.loss import E2ELoss, v8SegmentationLoss


MODEL_CFG = "yoloe-26n.yaml"
NUM_CLASSES = 65
TEXT_DIM = 512


def _model() -> YOLOEModel:
    torch.manual_seed(0)
    model = YOLOEModel(MODEL_CFG, ch=3, nc=NUM_CLASSES, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    return model


def _leaves(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [leaf for item in value.values() for leaf in _leaves(item)]
    if isinstance(value, (list, tuple)):
        return [leaf for item in value for leaf in _leaves(item)]
    return []


def _state_digest(module) -> str:
    digest = hashlib.sha256()
    for key, value in module.state_dict().items():
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _configure_router(module) -> None:
    """Make two detached condition coordinates select different experts."""

    with torch.no_grad():
        module.condition_projection.weight.zero_()
        module.condition_projection.bias.zero_()
        module.condition_projection.weight[0, 0] = 1.0
        module.router.weight.zero_()
        module.router.bias.zero_()
        module.router.weight[0, module.hidden_dim] = 1.0
        module.router.weight[1, module.hidden_dim] = -1.0


def _batch(images, tpe, router_condition):
    return {
        "img": images,
        "txt_feats": tpe,
        "router_condition": router_condition,
        "batch_idx": torch.arange(images.shape[0], dtype=torch.long),
        "cls": torch.zeros(images.shape[0], 1),
        "bboxes": torch.full((images.shape[0], 4), 0.5),
    }


def test_disabled_adapter_is_equivalent_and_does_not_change_module_list():
    model = _model().eval()
    images = torch.rand(1, 3, 64, 64)
    tpe = torch.randn(1, NUM_CLASSES, TEXT_DIM)
    before = model.predict(images, tpe=tpe)
    keys_before = set(model.state_dict())

    assert model.p5_text_router_enabled is False
    assert model.p5_text_router is None
    after = model.predict(images, tpe=tpe, router_condition=torch.randn(TEXT_DIM))
    assert len(_leaves(before)) == len(_leaves(after))
    for left, right in zip(_leaves(before), _leaves(after)):
        assert torch.equal(left, right)
    assert set(model.state_dict()) == keys_before


def test_disabled_adapter_preserves_other_routed_aux_criterion():
    model = _model()
    model.other_router = C2fMoA(16, 16, n=1, num_heads=3)

    assert model.p5_text_router_enabled is False
    assert isinstance(model.init_criterion(), CompositeCriterion)


def test_disabled_adapter_preserves_segmentation_criterion():
    model = YOLOESegModel("yoloe-26n-seg.yaml", ch=3, nc=NUM_CLASSES, verbose=False)
    model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)

    assert model.p5_text_router_enabled is False
    criterion = model.init_criterion()
    if isinstance(criterion, E2ELoss):
        assert isinstance(criterion.one2many, v8SegmentationLoss)
        assert isinstance(criterion.one2one, v8SegmentationLoss)
    else:
        assert isinstance(criterion, v8SegmentationLoss)


def test_released_load_then_enable_keeps_unfused_core_keys_additive():
    source = _model()
    target = _model()
    keys_before = set(target.state_dict())
    module_keys_before = {key for key in keys_before if key.startswith("model.")}

    target.load(source, verbose=False)
    head = target.model[-1]
    assert head.is_fused is False
    assert hasattr(head, "reprta") and not isinstance(head.reprta, torch.nn.Identity)

    adapter = target.enable_p5_text_router()
    keys_after = set(target.state_dict())
    module_keys_after = {key for key in keys_after if key.startswith("model.")}
    assert keys_before <= keys_after
    assert module_keys_before == module_keys_after
    assert any(key.startswith("_p5_text_router.") for key in keys_after - keys_before)
    assert target.p5_text_router is adapter
    assert target.p5_text_router_enabled is True
    assert target.model[-1].is_fused is False
    assert hasattr(target.model[-1], "reprta") and not isinstance(target.model[-1].reprta, torch.nn.Identity)
    target.train()
    assert target.training is True and target.model.training is False
    assert all(not module.training for module in target.model.modules())
    assert all(not parameter.requires_grad for parameter in target.model.parameters())
    adapter_parameter_ids = {id(parameter) for parameter in target.p5_text_router_parameters()}
    assert adapter_parameter_ids == {id(parameter) for parameter in adapter.parameters()}
    assert adapter_parameter_ids.isdisjoint({id(parameter) for parameter in target.model.parameters()})


def test_yoloe_adapter_reachability_gradients_aux_and_condition_separation(monkeypatch):
    model = _model().train()
    adapter = model.enable_p5_text_router()
    _configure_router(adapter)
    core_digest_before = _state_digest(model.model)
    assert all(not parameter.requires_grad for parameter in model.model.parameters())
    images = torch.rand(2, 3, 64, 64)
    tpe = torch.randn(1, NUM_CLASSES, TEXT_DIM)
    tpe_before = tpe.detach().clone()
    router_condition = torch.zeros(2, TEXT_DIM)
    router_condition[0, 0] = 1.0
    router_condition[1, 0] = -1.0
    router_condition.requires_grad_()

    clear_aux_records(step=101)
    model(images, tpe=tpe, router_condition=router_condition)
    assert adapter.last_routing_snapshot["executed_expert"] == [0, 1]
    assert adapter.last_routing_snapshot["expert_usage"].tolist() == [0.5, 0.5]
    assert adapter.last_routing_snapshot["opportunity_count"] == 2
    assert adapter.last_routing_snapshot["nontrivial_action_count"] == 2
    assert adapter.last_routing_snapshot["mean_router_entropy"] >= 0.0
    assert adapter.last_routing_snapshot["mean_top1_margin"] >= 0.0
    assert adapter.last_routing_snapshot["dense_reference_calls"] == 4
    assert adapter.last_routing_snapshot["actual_forward_calls"] == 2
    assert adapter.last_routing_snapshot["actual_forward_calls_by_expert"] == [1, 1]
    assert adapter.last_routing_snapshot["actual_expert_module_invocations_by_expert"] == [1, 1]
    assert adapter.last_routing_snapshot["dispatch"]["nontrivial_action_count"] == 2
    assert all(not expert._forward_hooks for expert in adapter.experts)
    assert torch.equal(tpe, tpe_before)

    first_aux, first_diagnostics = collect_aux_loss(
        model, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True
    )
    second_aux, second_diagnostics = collect_aux_loss(
        model, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True
    )
    assert first_aux.requires_grad and first_diagnostics["consumed"] == 1
    assert not second_aux.requires_grad and second_diagnostics["consumed"] == 0

    batch = _batch(images, tpe, router_condition)
    import ultralytics.nn.modules.routing_protocol as routing_protocol

    consume_count = {"value": 0}
    original_collect = routing_protocol.collect_aux_loss

    def counted_collect(*args, **kwargs):
        consume_count["value"] += 1
        return original_collect(*args, **kwargs)

    monkeypatch.setattr(routing_protocol, "collect_aux_loss", counted_collect)
    loss, loss_items = model.loss(batch)
    assert consume_count["value"] == 1
    assert not isinstance(model.criterion, CompositeCriterion)
    assert model.criterion.__class__.__name__ == "_P1TextRouterCriterion"
    assert "_mixture_loss_ema_buf" not in model._buffers
    assert loss.requires_grad and torch.isfinite(loss).all()
    assert loss_items.ndim == 1
    assert torch.isfinite(model._last_mixture_aux_loss)
    assert model._last_mixture_aux_loss.detach().abs() > 0
    assert torch.allclose(model._last_mixture_aux_loss, adapter.aux_loss.detach())
    assert torch.allclose(loss_items[-1], model._last_mixture_aux_loss)

    # The YOLOE loss consumed the one current-step publication and the model
    # cleared the registry after consumption; no stale/eval record remains.
    aux, diagnostics = collect_aux_loss(model, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True)
    assert not aux.requires_grad
    assert diagnostics["counts_by_kind"]["mot"] == 0
    assert diagnostics["stale_skipped"] == 0
    assert diagnostics["eval_skipped"] == 0

    model.zero_grad(set_to_none=True)
    loss.sum().backward()
    for parameter in (adapter.condition_projection.weight, adapter.router.weight):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().max() > 0
    for expert in adapter.experts:
        gradients = [parameter.grad for parameter in expert.parameters() if parameter.grad is not None]
        assert gradients and max(float(gradient.abs().max()) for gradient in gradients) > 0
    assert router_condition.grad is None

    optimizer = torch.optim.SGD(model.p5_text_router_parameters(), lr=1e-3)
    before = [[parameter.detach().clone() for parameter in expert.parameters()] for expert in adapter.experts]
    optimizer.step()
    for initial, expert in zip(before, adapter.experts):
        deltas = [(parameter.detach() - old).abs().max() for old, parameter in zip(initial, expert.parameters())]
        assert max(float(delta) for delta in deltas) > 0
    assert _state_digest(model.model) == core_digest_before


def test_yoloe_eval_adapter_publication_is_skipped():
    model = _model().eval()
    model.enable_p5_text_router()
    images = torch.rand(1, 3, 64, 64)
    tpe = torch.randn(1, NUM_CLASSES, TEXT_DIM)
    clear_aux_records(step=111)
    model.predict(images, tpe=tpe, router_condition=torch.ones(TEXT_DIM))
    aux, diagnostics = collect_aux_loss(model, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True)
    assert not aux.requires_grad
    assert diagnostics["counts_by_kind"]["mot"] == 0
    assert diagnostics["eval_skipped"] == 0
    assert diagnostics["consumed"] == 0


def test_yoloe_task_and_aux_gradient_paths_are_separate():
    """Task gradients reach both selected experts; raw aux gradients reach routing layers only."""

    model = _model().train()
    adapter = model.enable_p5_text_router()
    _configure_router(adapter)
    images = torch.rand(2, 3, 64, 64)
    tpe = torch.randn(1, NUM_CLASSES, TEXT_DIM)
    condition = torch.zeros(2, TEXT_DIM)
    condition[0, 0] = 1.0
    condition[1, 0] = -1.0
    batch = _batch(images, tpe, condition)

    clear_aux_records(step=551)
    predictions = model.forward(images, tpe=tpe, router_condition=condition)
    criterion = model.init_criterion()
    native_loss, _ = criterion.native_criterion(predictions, batch)
    task_scalar = native_loss.sum()
    aux_scalar = adapter.aux_loss
    adapter_parameters = tuple(adapter.parameters())
    task_gradients = torch.autograd.grad(task_scalar, adapter_parameters, retain_graph=True, allow_unused=True)
    aux_gradients = torch.autograd.grad(aux_scalar, adapter_parameters, retain_graph=True, allow_unused=True)

    def has_nonzero_finite(gradient):
        return gradient is not None and torch.isfinite(gradient).all() and gradient.abs().max() > 0

    names = [name for name, _ in adapter.named_parameters()]
    task_by_name = dict(zip(names, task_gradients))
    aux_by_name = dict(zip(names, aux_gradients))
    for expert_index in range(adapter.num_experts):
        expert_task_gradients = [
            task_by_name[name] for name in task_by_name if name.startswith(f"experts.{expert_index}.")
        ]
        assert expert_task_gradients
        assert all(torch.isfinite(gradient).all() for gradient in expert_task_gradients if gradient is not None)
        assert any(has_nonzero_finite(gradient) for gradient in expert_task_gradients)
    assert has_nonzero_finite(task_by_name["router.weight"])
    assert has_nonzero_finite(task_by_name["condition_projection.weight"])
    assert has_nonzero_finite(aux_by_name["condition_projection.weight"])
    assert has_nonzero_finite(aux_by_name["router.weight"])
    expert_aux_gradients = [
        gradient for name, gradient in aux_by_name.items() if name.startswith("experts.") and gradient is not None
    ]
    assert all(torch.isfinite(gradient).all() and gradient.abs().max() == 0 for gradient in expert_aux_gradients)

    model.zero_grad(set_to_none=True)
    combined = task_scalar + aux_scalar
    combined.backward()
    optimizer = torch.optim.SGD(model.p5_text_router_parameters(), lr=1e-3)
    before = {name: parameter.detach().clone() for name, parameter in adapter.named_parameters()}
    optimizer.step()
    for expert_index in range(adapter.num_experts):
        expert_updates = [
            (parameter.detach() - before[name]).abs().max()
            for name, parameter in adapter.named_parameters()
            if name.startswith(f"experts.{expert_index}.")
        ]
        assert expert_updates
        assert all(torch.isfinite(delta) for delta in expert_updates)
        assert any(delta > 0 for delta in expert_updates)
    for parameter_name in ("router.weight", "condition_projection.weight"):
        update = adapter.get_parameter(parameter_name).detach() - before[parameter_name]
        assert torch.isfinite(update).all() and update.abs().max() > 0
    clear_aux_records()


def test_yoloe_controlled_condition_pair_changes_router_and_adapter_not_classifier_prompt(monkeypatch):
    """A same-image fixed-classifier pair tests router and adapter sensitivity."""

    model = _model().eval()
    adapter = model.enable_p5_text_router()
    _configure_router(adapter)
    images = torch.rand(1, 3, 64, 64)
    images_before = images.clone()
    # Keep one classifier tensor fixed at the full released 65-class width;
    # derive the two non-zero conditions from disjoint base/new prompt rows.
    classifier_tpe = torch.zeros(1, NUM_CLASSES, TEXT_DIM)
    classifier_tpe[:, :48, 0] = 1.0
    classifier_tpe[:, 48:, 0] = -1.0
    classifier_tpe_before = classifier_tpe.clone()
    base_prompt_condition = classifier_tpe[:, :48].mean(dim=1).detach().clone()
    new_prompt_condition = classifier_tpe[:, 48:].mean(dim=1).detach().clone()
    observed_classifier_inputs = []
    observed_adapter_outputs = []
    head = model.model[-1]
    original_get_tpe = head.get_tpe

    def capture_classifier_input(value):
        observed_classifier_inputs.append(value.detach().clone())
        return original_get_tpe(value)

    monkeypatch.setattr(head, "get_tpe", capture_classifier_input)
    adapter_hook = adapter.register_forward_hook(
        lambda _module, _inputs, output: observed_adapter_outputs.append(output.detach().clone())
    )
    clear_aux_records(step=601)
    try:
        model.predict(images, tpe=classifier_tpe, router_condition=base_prompt_condition)
        base_routing_logits = adapter.last_routing_logits.detach().clone()
        base_route = adapter.last_routing_snapshot["route"].detach().clone()
        model.predict(images, tpe=classifier_tpe, router_condition=new_prompt_condition)
        new_routing_logits = adapter.last_routing_logits.detach().clone()
        new_route = adapter.last_routing_snapshot["route"].detach().clone()
    finally:
        adapter_hook.remove()

    assert len(observed_adapter_outputs) == 2
    assert torch.equal(images, images_before)
    assert torch.equal(classifier_tpe, classifier_tpe_before)
    assert len(observed_classifier_inputs) == 2
    assert all(torch.equal(value, classifier_tpe) for value in observed_classifier_inputs)
    assert classifier_tpe.untyped_storage().data_ptr() != base_prompt_condition.untyped_storage().data_ptr()
    assert classifier_tpe.untyped_storage().data_ptr() != new_prompt_condition.untyped_storage().data_ptr()
    assert base_prompt_condition.untyped_storage().data_ptr() != new_prompt_condition.untyped_storage().data_ptr()
    assert base_prompt_condition.requires_grad is False and new_prompt_condition.requires_grad is False
    assert torch.isfinite(base_routing_logits).all() and torch.isfinite(new_routing_logits).all()
    assert float((base_routing_logits - new_routing_logits).abs().sum()) > 1e-6
    assert not torch.equal(base_route, new_route)
    assert (base_route != new_route).any()

    adapter_delta = observed_adapter_outputs[0] - observed_adapter_outputs[1]
    assert float(adapter_delta.norm(p=2)) >= 1e-7
    assert float(adapter_delta.abs().max()) >= 1e-8


def test_text_router_failed_expert_forward_is_not_counted(monkeypatch):
    """A hook counts only an expert that successfully returns a batched output."""

    module = TextConditionedMoT(4, 4, text_dim=8, hidden_dim=2).eval()
    features = torch.randn(2, 4, 3, 3)
    failing_expert = module.experts[0]

    def fail_forward(_features):
        raise RuntimeError("expected expert failure")

    monkeypatch.setattr(failing_expert, "forward", fail_forward)
    with module._measure_successful_expert_forwards() as measurements:
        with pytest.raises(RuntimeError, match="expected expert failure"):
            failing_expert(features)

    assert measurements == {
        "module_invocations": 0,
        "output_batch_samples": 0,
        "module_invocations_by_expert": [0, 0],
        "output_batch_samples_by_expert": [0, 0],
    }
    assert not failing_expert._forward_hooks


def test_yoloe_loss_exception_clears_published_aux_record():
    model = _model().train()
    adapter = model.enable_p5_text_router()
    _configure_router(adapter)

    class RaisingCriterion:
        def __call__(self, predictions, batch):
            raise RuntimeError("expected criterion failure")

    model.criterion = RaisingCriterion()
    images = torch.rand(1, 3, 64, 64)
    batch = _batch(images, torch.randn(1, NUM_CLASSES, TEXT_DIM), torch.ones(TEXT_DIM))
    try:
        model.loss(batch)
    except RuntimeError as error:
        assert str(error) == "expected criterion failure"
    else:
        raise AssertionError("raising criterion unexpectedly returned")

    _, diagnostics = collect_aux_loss(model, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True)
    assert diagnostics["consumed"] == 0
    assert diagnostics["counts_by_kind"]["mot"] == 0


def test_text_router_duplicate_publication_keeps_first_graph_and_no_second_weight():
    """A same-step retry is observable but cannot contribute a second aux term."""

    module = TextConditionedMoT(4, 4, text_dim=8, hidden_dim=2).train()
    _configure_router(module)
    condition = torch.ones(2, 8)
    features = torch.randn(2, 4, 3, 3)
    clear_aux_records(step=301)
    module(features, condition=condition)
    first_record = get_aux_record(module)
    assert first_record is not None
    first, first_diagnostics = collect_aux_loss(
        module, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True
    )

    # Re-running the publisher without advancing/clearing the canonical step
    # must not replace the graph that the first collector consumed.
    module(features, condition=condition)
    second, second_diagnostics = collect_aux_loss(
        module, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True
    )
    assert first.requires_grad and first_diagnostics["consumed"] == 1
    assert not second.requires_grad and second_diagnostics["consumed"] == 0
    assert second_diagnostics["duplicate_published"] == 1
    assert get_aux_record(module).value is first_record.value
    assert module._routing_duplicate_publication_count == 1


def test_text_router_hard_attempt_receipt_counts_are_measured_and_eval_is_empty():
    """CPU harness freezes hard attempt totals and keeps eval outside aux accounting."""

    module = TextConditionedMoT(4, 4, text_dim=8, hidden_dim=2).train()
    _configure_router(module)
    features = torch.randn(2, 4, 3, 3)
    condition = torch.zeros(2, 8)
    condition[0, 0] = 1.0
    condition[1, 0] = -1.0

    # Preflight is an explicit eval/no-publication phase, followed by a reset
    # before the hard-attempt receipt begins.
    module.eval()
    clear_aux_records(step=401)
    module(features, condition=condition)
    _, eval_diagnostics = collect_aux_loss(
        module, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True
    )
    assert eval_diagnostics["consumed"] == 0
    assert eval_diagnostics["counts_by_kind"]["mot"] == 0
    assert eval_diagnostics["eval_skipped"] == 0
    reset_routing_runtime_state(module, step=402)
    module.train()

    totals = {"opportunity": 0, "dense": 0, "actual": 0, "action": 0}
    for step in range(20):
        clear_aux_records(step=403 + step)
        module(features, condition=condition)
        snapshot = module.last_routing_snapshot
        totals["opportunity"] += snapshot["opportunity_count"]
        totals["dense"] += snapshot["dense_reference_calls"]
        totals["actual"] += snapshot["actual_forward_calls"]
        totals["action"] += snapshot["nontrivial_action_count"]
        assert snapshot["actual_forward_calls_by_expert"] == [1, 1]
        assert snapshot["actual_expert_module_invocations_by_expert"] == [1, 1]
        assert "event_count" not in snapshot
        collected, diagnostics = collect_aux_loss(
            module, step=current_aux_step(), include_kinds=("mot",), return_diagnostics=True
        )
        assert collected.requires_grad and diagnostics["consumed"] == 1

    assert totals == {"opportunity": 40, "dense": 80, "actual": 40, "action": 40}
    assert float(module.last_routing_snapshot["expert_usage"].min()) >= 0.05

    # A repair receipt is independent from the attempt receipt: ten hard
    # steps, two samples each, have their own 20/40/20 counters.
    repair = {"opportunity": 0, "dense": 0, "actual": 0, "action": 0}
    for step in range(10):
        clear_aux_records(step=503 + step)
        module(features, condition=condition)
        snapshot = module.last_routing_snapshot
        repair["opportunity"] += snapshot["opportunity_count"]
        repair["dense"] += snapshot["dense_reference_calls"]
        repair["actual"] += snapshot["actual_forward_calls"]
        repair["action"] += snapshot["nontrivial_action_count"]
        assert snapshot["actual_forward_calls_by_expert"] == [1, 1]
        assert snapshot["actual_expert_module_invocations_by_expert"] == [1, 1]
        collect_aux_loss(module, step=current_aux_step(), include_kinds=("mot",))
    assert repair == {"opportunity": 20, "dense": 40, "actual": 20, "action": 20}
