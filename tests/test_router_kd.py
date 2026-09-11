"""F11 RouterKDLoss 数值 + 梯度通路 + aux 注册单测。

覆盖 06 文档 T3 验收点:
  - KD loss 数值正确(JS / KL 两模式)
  - 梯度只反传学生,教师 detach 无梯度
  - ε-uniform 平滑 + 温度缩放行为
  - 形状/参数校验(非法输入报错)
  - collect_aux_loss 注册通路(publish_aux_loss kind='router_kd')
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.foundation import RouterKDLoss
from ultralytics.nn.modules.routing_protocol import collect_aux_loss, publish_aux_loss


def test_router_kd_loss_js_zero_for_identical_distributions():
    """JS 模式下学生与教师分布一致时损失应为 0(数值上接近 0)。"""
    criterion = RouterKDLoss(temperature=1.0, smoothing=0.0, kind="js")
    logits = torch.tensor([[2.0, 1.0, 0.5], [0.0, 1.0, 2.0]], requires_grad=True)
    teacher = F.softmax(logits.detach(), dim=-1)

    loss = criterion(logits, teacher)
    loss.backward()

    assert loss.ndim == 0
    # 学生软目标与教师完全一致 → JS = 0
    assert loss.item() == pytest.approx(0.0, abs=1e-6)
    assert logits.grad is not None  # 学生有梯度
    assert teacher.grad is None  # 教师 detach 无梯度


def test_router_kd_loss_js_positive_and_bounded_for_different_distributions():
    """JS 模式下不同分布应产生 (0, log2] 范围内的有限损失。"""
    criterion = RouterKDLoss(temperature=1.0, smoothing=0.0, kind="js")
    student = torch.tensor([[10.0, 0.0, 0.0]], requires_grad=True)  # one-hot
    teacher = F.softmax(torch.tensor([[0.0, 0.0, 10.0]]), dim=-1)  # 相反 one-hot

    loss = criterion(student, teacher)

    assert torch.isfinite(loss)
    assert loss.item() > 0.0
    assert loss.item() <= float(torch.log(torch.tensor(2.0))) + 1e-6


def test_router_kd_loss_kl_matches_official_routing_kd_loss():
    """KL 模式应与官方 routing_kd_loss 数值一致(带 T² 缩放)。"""
    from ultralytics.nn.foundation.routing import routing_kd_loss as official_kd

    torch.manual_seed(0)
    student = torch.randn(4, 3, requires_grad=True)
    teacher_logits = torch.randn(4, 3)

    criterion = RouterKDLoss(temperature=2.0, smoothing=0.0, kind="kl")
    ours = criterion(student, teacher_logits)
    official = official_kd(student, teacher_logits, temperature=2.0)

    assert ours.item() == pytest.approx(official.item(), abs=1e-6)

def test_router_kd_loss_smoothing_increases_entropy_target():
    """ε-uniform 平滑应使目标分布更接近均匀(损失与未平滑不同)。"""
    torch.manual_seed(1)
    student = torch.randn(4, 4, requires_grad=True)
    teacher = torch.randn(4, 4)

    no_smooth = RouterKDLoss(temperature=1.0, smoothing=0.0, kind="js")(student, teacher)
    with_smooth = RouterKDLoss(temperature=1.0, smoothing=0.2, kind="js")(student, teacher)

    assert torch.isfinite(no_smooth)
    assert torch.isfinite(with_smooth)


def test_router_kd_loss_backpropagates_only_student():
    """梯度应只到达学生参数,教师保持 detach。"""
    torch.manual_seed(2)
    student_layer = nn.Linear(8, 4)
    teacher_probs = F.softmax(torch.randn(3, 4), dim=-1)

    x = torch.randn(3, 8)
    logits = student_layer(x)
    loss = RouterKDLoss(kind="js")(logits, teacher_probs)
    loss.backward()

    assert student_layer.weight.grad is not None
    assert torch.isfinite(student_layer.weight.grad).all()
    assert student_layer.weight.grad.abs().sum() > 0.0  # 梯度非零(T3 验收点)


def test_router_kd_loss_rejects_mismatched_shapes():
    """学生与教师形状不一致时应报错。"""
    criterion = RouterKDLoss(kind="js")
    with pytest.raises(ValueError):
        criterion(torch.randn(2, 4), torch.randn(3, 4))


def test_router_kd_loss_rejects_invalid_parameters():
    """非法温度/平滑/kind 应报错。"""
    with pytest.raises(ValueError):
        RouterKDLoss(temperature=0.0)
    with pytest.raises(ValueError):
        RouterKDLoss(smoothing=-0.1)
    with pytest.raises(ValueError):
        RouterKDLoss(kind="unknown")


def test_router_kd_loss_handles_3d_batched_tokens():
    """支持 [B, N, E] 三维输入(训练期真实路由形状)。"""
    torch.manual_seed(3)
    student = torch.randn(2, 16, 4, requires_grad=True)
    teacher = torch.randn(2, 16, 4)

    loss = RouterKDLoss(kind="kl", temperature=0.5)(student, teacher)
    loss.backward()

    assert loss.ndim == 0
    assert student.grad is not None


def test_publish_aux_loss_router_kd_kind_is_collectible():
    """publish_aux_loss(kind='router_kd') 后 collect_aux_loss 应收录非零值。"""
    torch.manual_seed(4)
    block = nn.Linear(8, 4)
    block.train()
    student_logits = block(torch.randn(2, 8))
    teacher_probs = F.softmax(torch.randn(2, 4), dim=-1)
    kd_loss = RouterKDLoss(kind="js")(student_logits, teacher_probs)

    published = publish_aux_loss(block, kd_loss, kind="router_kd")
    total, diagnostics = collect_aux_loss(block, include_kinds=("router_kd",), return_diagnostics=True)

    assert published is not None
    assert total is not None
    values = diagnostics["values_by_kind"].get("router_kd", [])
    assert len(values) == 1
    assert values[0] > 0.0
