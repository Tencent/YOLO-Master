import torch

from ultralytics.nn.modules.moa import C2fMoA
from ultralytics.nn.modules.moe.modules import UltraOptimizedMoE
from ultralytics.nn.modules.mot import C2fMoT
from ultralytics.utils.loss import _collect_mixture_aux_loss, _get_mixture_loss_ema


def test_mixture_aux_loss_uses_ema_scales():
    torch.manual_seed(0)
    model = torch.nn.Sequential(
        C2fMoA(32, 32, n=1, num_heads=3),
        C2fMoT(32, 32, n=1, num_heads=3, top_k=2, window_size=4, n_points=2),
        UltraOptimizedMoE(32, 32, num_experts=4, top_k=2),
    ).train()
    x = torch.randn(2, 32, 8, 8)
    model[0](x)
    model[1](x)
    model[2](x)

    loss1 = _collect_mixture_aux_loss(model, torch.device("cpu"))
    assert loss1.requires_grad and torch.isfinite(loss1)
    # EMA state is now a persistent buffer (_mixture_loss_ema_buf, shape [3])
    # rather than a plain dict attribute, so it survives state_dict() round-trips.
    assert hasattr(model, "_mixture_loss_ema_buf")

    loss2 = _collect_mixture_aux_loss(model, torch.device("cpu"))
    assert torch.isfinite(loss2)
    # EMA scales should stay positive and stable across steps
    ema_buf = model._mixture_loss_ema_buf
    assert all(float(ema_buf[i]) >= 1e-4 for i in range(ema_buf.numel()))
    diagnostics = model._mixture_aux_diagnostics
    assert diagnostics["counts_by_kind"]["moa"] == 1
    assert diagnostics["counts_by_kind"]["mot"] == 1
    assert diagnostics["counts_by_kind"]["moe"] == 1


def test_mixture_loss_ema_initializes_for_parameterized_model_without_tensor_bool():
    model = torch.nn.Linear(2, 2).train()

    ema = _get_mixture_loss_ema(model)

    assert ema["moe"] == 1.0
    assert abs(ema["mot"] - 0.1) < 1e-6
    assert abs(ema["moa"] - 0.1) < 1e-6
    assert model._mixture_loss_ema_buf.device == model.weight.device


def test_mixture_loss_ema_initializes_for_parameter_free_model():
    model = torch.nn.ReLU().train()

    ema = _get_mixture_loss_ema(model)

    assert ema["moe"] == 1.0
    assert abs(ema["mot"] - 0.1) < 1e-6
    assert abs(ema["moa"] - 0.1) < 1e-6
    assert model._mixture_loss_ema_buf.device.type == "cpu"


def test_moa_linear_attn_fp16_stable():
    from ultralytics.nn.modules.moa.moa import _GlobalAttnHead

    torch.manual_seed(0)
    head = _GlobalAttnHead(64, num_heads=4, head_dim=16).half().train()
    q = torch.randn(1, 4, 6400, 16, dtype=torch.float16)
    k = torch.randn(1, 4, 6400, 16, dtype=torch.float16)
    v = torch.randn(1, 4, 6400, 16, dtype=torch.float16)
    out = head._linear_attn(q, k, v)
    assert out.dtype == torch.float16
    assert torch.isfinite(out).all()
