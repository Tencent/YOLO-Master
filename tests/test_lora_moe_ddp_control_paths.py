"""Regression: LoRA must not target MoE control paths that break DDP."""
import torch
import torch.nn as nn
from types import SimpleNamespace

from ultralytics.engine.extensions import AdapterRuntimeController
from ultralytics.nn.modules.moe.gated import AdaptiveGateMoE
from ultralytics.utils.lora.config import LoRAConfigBuilder


def test_lora_auto_detect_excludes_complexity_estimator_when_include_moe_false():
    moe = AdaptiveGateMoE(32, 32, num_experts=4, top_k=2)
    # Wrap like a YOLO sequential layer index
    model = nn.Sequential(nn.Identity(), moe)
    targets = LoRAConfigBuilder.auto_detect_targets(model, r=8, include_moe=False)
    assert not any("complexity_estimator" in t for t in targets)
    assert not any("se_gate" in t for t in targets)
    assert not any("expert" in t.lower() for t in targets)


def test_lora_auto_detect_always_excludes_complexity_even_if_include_moe_true():
    moe = AdaptiveGateMoE(32, 32, num_experts=4, top_k=2)
    model = nn.Sequential(moe)
    targets = LoRAConfigBuilder.auto_detect_targets(model, r=8, include_moe=True)
    assert not any("complexity_estimator" in t for t in targets)
    assert not any("se_gate" in t for t in targets)


def test_adaptive_gate_complexity_gate_detaches_from_estimator():
    """Discrete complexity gate must not leave half-used grads into estimator."""
    torch.manual_seed(0)
    m = AdaptiveGateMoE(32, 32, num_experts=4, top_k=2).train()
    x = torch.randn(2, 32, 8, 8, requires_grad=True)
    out = m(x)
    out.mean().backward()
    # Estimator params should have no grad (detached discrete gate)
    for p in m.complexity_estimator.parameters():
        assert p.grad is None or float(p.grad.abs().sum()) == 0.0


def test_is_under_moe_block_detects_nested_control_path():
    moe = AdaptiveGateMoE(16, 16, num_experts=4, top_k=2)
    model = nn.Sequential(moe)
    modules = dict(model.named_modules())
    assert LoRAConfigBuilder._is_under_moe_block("0.complexity_estimator.1", modules)
    assert LoRAConfigBuilder._is_under_moe_block("0.proj", modules)
    assert not LoRAConfigBuilder._is_under_moe_block("does.not.exist", modules)


def test_adalora_schedule_advances_after_optimizer_step():
    class AdapterModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))
            self.lora_enabled = True
            self.steps = []

        def update_and_allocate(self, step):
            self.steps.append(step)

    trainer = SimpleNamespace(model=AdapterModel())
    controller = AdapterRuntimeController(trainer)

    controller.after_optimizer_step()
    controller.after_optimizer_step()

    assert trainer.model.steps == [1, 2]


def test_adapter_controller_active_recognizes_molora():
    model = nn.Linear(4, 4)
    model.molora_enabled = True
    controller = AdapterRuntimeController(SimpleNamespace(model=model))

    assert controller.enabled is False
    assert controller.active is True


def test_distillation_helpers_preserve_gradient_to_student():
    model = nn.Linear(4, 4)
    trainer = SimpleNamespace(model=model)
    controller = AdapterRuntimeController(trainer)
    student = torch.randn(2, 8, 6, requires_grad=True)
    teacher = torch.randn(2, 8, 6)

    loss = controller.compute_distillation_loss(student, teacher)
    response = controller.compute_response_distillation_loss(student, teacher)
    (loss + response).backward()

    assert student.grad is not None
    assert torch.isfinite(student.grad).all()
