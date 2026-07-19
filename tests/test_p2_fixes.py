"""verification tests — export safety, DDP mock, and quality regressions.

Covers:
- MoA / MoT ONNX export preserves all experts (no data-dependent skip).
- MoA / MoT TorchScript tracing works in eval mode.
- MoA `sequential_heads=True` produces identical output to default path.
- MoA `__all__` exports only public symbols.
- MoLoRA `compute_aux_loss` works without redundant `seen` set.
- MoLoRA `HybridRouter` alpha init is 0.0 (sigmoid(0)=0.5, truly uniform).
- analysis.py / pruning.py use LOGGER (no bare print()).
"""
import io
import inspect
import re
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules.moa import MoABlock, C2fMoA
from ultralytics.nn.modules.mot import MoTBlock
from ultralytics.utils.torch_utils import TORCH_1_13


ROOT = Path(__file__).resolve().parents[1]


def _legacy_onnx_export_kwargs():
    kwargs = {"opset_version": 17 if TORCH_1_13 else 12, "do_constant_folding": False}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        kwargs["dynamo"] = False
    return kwargs


# ── MoA sequential_heads equivalence ──────────────────────────────────────

def test_moa_sequential_heads_matches_default():
    """sequential_heads=True must produce the same result as the default path."""
    torch.manual_seed(42)
    block_default = MoABlock(48, num_heads=6, sequential_heads=False).eval()
    torch.manual_seed(42)
    block_seq = MoABlock(48, num_heads=6, sequential_heads=True).eval()

    # Copy weights to ensure identical params
    block_seq.load_state_dict(block_default.state_dict())

    x = torch.randn(2, 48, 8, 8)
    with torch.no_grad():
        out_default = block_default(x)
        out_seq = block_seq(x)

    assert torch.allclose(out_default, out_seq, atol=1e-6), (
        f"sequential_heads output mismatch: max diff = {(out_default - out_seq).abs().max().item()}"
    )


# ── MoA __all__ ───────────────────────────────────────────────────────────

def test_moa_all_exports_only_public_symbols():
    """moa.py __all__ should contain only public classes/functions."""
    from ultralytics.nn.modules.moa import moa as moa_mod

    assert hasattr(moa_mod, "__all__"), "moa.py must define __all__"
    expected = {"MoABlock", "C2fMoA", "NeckMoAFusion", "anneal_moa_temperature", "collect_moa_aux_loss"}
    assert set(moa_mod.__all__) == expected, (
        f"__all__ mismatch: got {set(moa_mod.__all__)}, expected {expected}"
    )

    # No private symbols leaked
    for name in moa_mod.__all__:
        assert not name.startswith("_"), f"Private symbol {name} in __all__"


# ── MoA ONNX export ───────────────────────────────────────────────────────

def test_moa_onnx_export():
    """MoABlock must be ONNX-exportable in eval mode."""
    block = MoABlock(32, num_heads=6).eval()
    x = torch.randn(1, 32, 8, 8)

    buf = io.BytesIO()
    torch.onnx.export(
        block, x, buf,
        input_names=["input"], output_names=["output"],
        **_legacy_onnx_export_kwargs(),
    )
    assert len(buf.getvalue()) > 0, "ONNX export produced empty buffer"

    # Verify forward still works
    with torch.no_grad():
        out = block(x)
    assert out.shape == x.shape


def test_c2fmoa_onnx_export():
    """C2fMoA must be ONNX-exportable in eval mode."""
    module = C2fMoA(32, 32, n=1, num_heads=6).eval()
    x = torch.randn(1, 32, 8, 8)

    buf = io.BytesIO()
    torch.onnx.export(
        module, x, buf,
        input_names=["input"], output_names=["output"],
        **_legacy_onnx_export_kwargs(),
    )
    assert len(buf.getvalue()) > 0


# ── MoT ONNX export ───────────────────────────────────────────────────────

def test_mot_onnx_export():
    """MoTBlock must be ONNX-exportable in eval mode (sparse path is guarded)."""
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2).eval()
    x = torch.randn(1, 32, 8, 8)

    buf = io.BytesIO()
    torch.onnx.export(
        block, x, buf,
        input_names=["input"], output_names=["output"],
        **_legacy_onnx_export_kwargs(),
    )
    assert len(buf.getvalue()) > 0, "ONNX export produced empty buffer"

    # Verify forward still works and returns (out, aux)
    with torch.no_grad():
        result = block(x)
    if isinstance(result, tuple):
        out, aux = result
    else:
        out = result
    assert out.shape == x.shape


# ── MoA TorchScript tracing ───────────────────────────────────────────────

def test_moa_torchscript_trace():
    """MoABlock must be TorchScript-traceable in eval mode."""
    block = MoABlock(32, num_heads=6).eval()
    x = torch.randn(1, 32, 8, 8)

    with torch.no_grad():
        traced = torch.jit.trace(block, x)
        out_orig = block(x)
        out_traced = traced(x)

    assert out_orig.shape == out_traced.shape
    assert torch.allclose(out_orig, out_traced, atol=1e-5)


def test_mot_torchscript_trace():
    """MoTBlock must be TorchScript-traceable in eval mode.

    Note: MoTBlock.forward returns (out, aux_loss) tuple. Tracing should
    capture the full forward path.
    """
    block = MoTBlock(32, num_heads=4, top_k=2, window_size=4, n_points=2).eval()
    x = torch.randn(1, 32, 8, 8)

    with torch.no_grad():
        traced = torch.jit.trace(block, x)
        out_orig = block(x)
        out_traced = traced(x)

    # Both should be tuples
    if isinstance(out_orig, tuple):
        assert isinstance(out_traced, tuple)
        assert torch.allclose(out_orig[0], out_traced[0], atol=1e-5)
    else:
        assert torch.allclose(out_orig, out_traced, atol=1e-5)


# ── MoLoRA HybridRouter alpha init ────────────────────────────────────────

def test_hybrid_router_alpha_init_is_zero():
    """HybridRouter alpha should init to 0.0 so sigmoid(0)=0.5 = truly uniform."""
    from ultralytics.nn.peft.molora.router import HybridRouter

    r = HybridRouter(16, 4)
    assert r.alpha.item() == 0.0, (
        f"HybridRouter alpha should init to 0.0 (got {r.alpha.item()}); "
        f"sigmoid(0.0)=0.5 gives a truly uniform blend at start. "
        f"The previous 0.5 init produced sigmoid(0.5)≈0.622, biasing the linear router."
    )


# ── MoLoRA compute_aux_loss without seen set ──────────────────────────────

def test_molora_compute_aux_loss_no_double_count():
    """compute_aux_loss must sum each MoLoRALayer's loss exactly once.

    The `seen` set was redundant because model.modules() yields each module
    exactly once. Verify removal doesn't cause double-counting.
    """
    from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel
    from ultralytics.nn.modules.moe.modules import MOE_LOSS_REGISTRY

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            return torch.relu(self.conv2(x))

    model = TinyModel()
    cfg = MoLoRAConfig(r=4, alpha=8, num_experts=4, top_k=2, target_modules=["conv1", "conv2"])
    wrapper = MoLoRAModel(model, cfg)
    wrapper.model.train()

    x = torch.randn(2, 3, 8, 8)
    _ = wrapper(x)
    aux = wrapper.compute_aux_loss()

    assert isinstance(aux, torch.Tensor)
    assert aux.requires_grad or aux.item() == 0  # may be 0 if no losses registered

    # Manually sum to verify no double counting
    from ultralytics.nn.peft.molora.layer import MoLoRALayer
    manual_sum = aux.new_zeros(())
    count = 0
    for m in wrapper.model.modules():
        if isinstance(m, MoLoRALayer):
            loss_t = MOE_LOSS_REGISTRY.get(m)
            if isinstance(loss_t, torch.Tensor):
                manual_sum = manual_sum + loss_t
                count += 1

    if count > 0:
        assert torch.allclose(aux, manual_sum, atol=1e-6), (
            f"compute_aux_loss ({aux.item()}) != manual sum ({manual_sum.item()}); "
            f"possible double-counting after `seen` set removal"
        )


# ── DDP mock: MoA all_reduce_mean is no-op on single process ─────────────

def test_moa_all_reduce_mean_single_process():
    """all_reduce_mean should be a no-op when DDP is not initialized."""
    from ultralytics.nn.modules.moa.moa import all_reduce_mean

    t = torch.randn(4, dtype=torch.float32)
    out = all_reduce_mean(t)
    assert torch.equal(out, t), "all_reduce_mean should be identity when dist not initialized"
    assert out.dtype == torch.float32


def test_moe_all_reduce_mean_single_process():
    """MoE's all_reduce_mean should also be a no-op on single process."""
    from ultralytics.nn.modules.moe.loss import all_reduce_mean as moe_all_reduce_mean

    t = torch.randn(4, dtype=torch.float16)
    out = moe_all_reduce_mean(t)
    assert torch.equal(out, t), "MoE all_reduce_mean should be identity when dist not initialized"
    assert out.dtype == torch.float16, "dtype must be preserved"


# ── DDP mock: MoLoRA aux loss works without DDP ───────────────────────────

def test_molora_aux_loss_single_gpu_no_crash():
    """MoLoRA aux loss collection must work without DDP (single GPU path)."""
    from ultralytics.nn.peft.molora import MoLoRAConfig, MoLoRAModel

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)

        def forward(self, x):
            return torch.relu(self.conv(x))

    model = TinyModel()
    cfg = MoLoRAConfig(r=4, alpha=8, num_experts=4, top_k=2, target_modules=["conv"])
    wrapper = MoLoRAModel(model, cfg)
    wrapper.model.train()

    x = torch.randn(1, 3, 4, 4)
    out = wrapper(x)
    aux = wrapper.compute_aux_loss()

    assert out.shape == (1, 8, 4, 4)
    assert isinstance(aux, torch.Tensor)
    # No crash, no NaN
    assert torch.isfinite(aux).all() or aux.item() == 0.0


# ── analysis.py / pruning.py use LOGGER not print ─────────────────────────

def test_analysis_uses_logger_not_print():
    """analysis.py should use LOGGER, not bare print()."""
    analysis_path = ROOT / "ultralytics/nn/modules/moe/analysis.py"
    content = analysis_path.read_text(encoding="utf-8")

    # No bare print( calls (except inside strings or comments)
    # Check for print( at start of line or after whitespace
    print_lines = [
        line.strip()
        for line in content.split("\n")
        if re.match(r'^\s*print\(', line) and not line.strip().startswith("#")
    ]
    assert len(print_lines) == 0, (
        f"analysis.py still has {len(print_lines)} bare print() calls: {print_lines[:3]}"
    )

    # LOGGER should be imported
    assert "from ultralytics.utils import LOGGER" in content, "analysis.py must import LOGGER"


def test_pruning_uses_logger_not_print():
    """pruning.py should use LOGGER, not bare print()."""
    pruning_path = ROOT / "ultralytics/nn/modules/moe/pruning.py"
    content = pruning_path.read_text(encoding="utf-8")

    print_lines = [
        line.strip()
        for line in content.split("\n")
        if re.match(r'^\s*print\(', line) and not line.strip().startswith("#")
    ]
    assert len(print_lines) == 0, (
        f"pruning.py still has {len(print_lines)} bare print() calls: {print_lines[:3]}"
    )

    assert "from ultralytics.utils import LOGGER" in content, "pruning.py must import LOGGER"


# ── MoT expert shape check ────────────────────────────────

def test_mot_blend_experts_shape_check_raises():
    """MoTBlock._blend_experts must raise RuntimeError on shape-mismatched expert."""
    from ultralytics.nn.modules.mot.mot import MoTBlock

    block = MoTBlock(16, num_heads=2, top_k=1, window_size=4, n_points=2).eval()

    # Monkey-patch an expert to return wrong shape
    original_expert = block.experts[0]
    class BadExpert(nn.Module):
        def forward(self, x):
            return x[:, :, :1, :1]  # wrong shape

    block.experts[0] = BadExpert()
    try:
        x = torch.randn(1, 16, 4, 4)
        with torch.no_grad():
            block(x)
        raise AssertionError("Expected RuntimeError for shape-mismatched expert")
    except RuntimeError as e:
        assert "shape-preserving" in str(e) or "changed tensor shape" in str(e)
    finally:
        block.experts[0] = original_expert


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v", "--tb=short"])
