"""ONNX and TorchScript export tests for all mixture module types.

Covers modules NOT in test_onnx_export_fix.py:
  - MoA (MoABlock, C2fMoA, NeckMoAFusion)
  - MoT (MoTBlock, C2fMoT)
  - MoLoRA (MoLoRALayer on Conv2d and Linear)

Existing test_onnx_export_fix.py already covers MoE (ES_MOE, OptimizedMoE, etc.),
so this file focuses on the remaining mixture types.

Key test: verify that all experts/heads are present in the exported graph
(no expert dropping during tracing) and that outputs match eager mode.
"""
import io
import torch
import torch.nn as nn
import pytest

from ultralytics.nn.modules.moa.moa import MoABlock, C2fMoA, NeckMoAFusion
from ultralytics.nn.modules.mot.mot import MoTBlock, C2fMoT
from ultralytics.nn.peft.molora.layer import MoLoRALayer


# ── Helpers ─────────────────────────────────────────────────────────────

def _export_onnx(module, dummy, name="module"):
    """Export module to ONNX and return parsed model (or None if onnx not installed)."""
    module.eval()
    buf = io.BytesIO()
    torch.onnx.export(
        module, dummy, buf,
        input_names=["input"], output_names=["output"],
        opset_version=17,
        do_constant_folding=False,
        dynamo=False,
    )
    buf.seek(0)
    try:
        import onnx
        model = onnx.load_from_string(buf.getvalue())
        return model, buf.getvalue()
    except ImportError:
        return None, buf.getvalue()


def _count_conv_nodes(model_proto):
    """Count Conv nodes in ONNX model."""
    count = 0
    for node in model_proto.graph.node:
        if node.op_type == "Conv":
            count += 1
    return count


def _count_matmul_nodes(model_proto):
    """Count MatMul nodes in ONNX model (for Linear experts)."""
    count = 0
    for node in model_proto.graph.node:
        if node.op_type in ("MatMul", "Gemm"):
            count += 1
    return count


# ── MoA ONNX export tests ───────────────────────────────────────────────

class TestMoAOnnxExport:
    """Verify MoA modules export to ONNX without losing heads."""

    def test_moablock_onnx_export(self):
        m = MoABlock(64, num_heads=6)
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(m, dummy, "MoABlock")
        assert model is not None or len(raw) > 0, "ONNX export failed"
        if model is not None:
            conv_count = _count_conv_nodes(model)
            # MoABlock has: local_head (conv), regional_head (conv),
            # global_head (conv), fusion (conv), ffn (2x conv)
            assert conv_count >= 3, f"MoABlock only has {conv_count} Conv nodes, expected ≥ 3"

    def test_c2fmoa_onnx_export(self):
        m = C2fMoA(64, 64, n=1, num_heads=6)
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(m, dummy, "C2fMoA")
        assert len(raw) > 0, "ONNX export failed"

    def test_neck_moa_fusion_onnx_export(self):
        m = NeckMoAFusion(64, 64, 64, num_heads=4)
        dummy_hi = torch.randn(1, 64, 8, 8)
        dummy_lo = torch.randn(1, 64, 4, 4)
        m.eval()
        buf = io.BytesIO()
        torch.onnx.export(
            m, (dummy_hi, dummy_lo), buf,
            input_names=["hi", "lo"], output_names=["output"],
            opset_version=17, do_constant_folding=False, dynamo=False,
        )
        assert len(buf.getvalue()) > 0

    def test_moablock_onnx_output_consistency(self):
        """ONNX output should match eager output within tolerance."""
        m = MoABlock(64, num_heads=6)
        dummy = torch.randn(1, 64, 8, 8)
        m.eval()
        with torch.no_grad():
            eager_out = m(dummy)
        model, raw = _export_onnx(m, dummy, "MoABlock")
        if model is not None:
            import onnxruntime as ort
            import numpy as np
            sess = ort.InferenceSession(raw)
            onnx_out = sess.run(None, {"input": dummy.numpy()})
            assert np.allclose(eager_out.numpy(), onnx_out[0], atol=1e-4), \
                "ONNX output differs from eager"


# ── MoT ONNX export tests ───────────────────────────────────────────────

class TestMoTOnnxExport:
    """Verify MoT modules export to ONNX without losing experts."""

    def test_motblock_onnx_export(self):
        m = MoTBlock(64, num_heads=4, top_k=2)
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(m, dummy, "MoTBlock")
        assert len(raw) > 0, "MoTBlock ONNX export failed"
        if model is not None:
            conv_count = _count_conv_nodes(model)
            # MoTBlock has 3 experts with conv layers + out_proj
            assert conv_count >= 3, f"MoTBlock only has {conv_count} Conv nodes, expected ≥ 3"

    def test_c2fmot_onnx_export(self):
        m = C2fMoT(64, 64, n=1, num_heads=4, top_k=2)
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(m, dummy, "C2fMoT")
        assert len(raw) > 0, "C2fMoT ONNX export failed"

    def test_motblock_dense_eval_onnx(self):
        """MoTBlock in eval uses sparse path; ONNX export forces dense via guard."""
        m = MoTBlock(64, num_heads=4, top_k=2, sparse_train=False)
        m.eval()
        dummy = torch.randn(1, 64, 8, 8)
        # Should not raise even though eval uses sparse path
        model, raw = _export_onnx(m, dummy, "MoTBlock_eval")
        assert len(raw) > 0


# ── MoLoRA ONNX export tests ────────────────────────────────────────────

class TestMoLoRAOnnxExport:
    """Verify MoLoRA layers export to ONNX."""

    def test_molora_conv_onnx_export(self):
        base = nn.Conv2d(64, 64, 3, padding=1)
        layer = MoLoRALayer(base, r=4, alpha=8, num_experts=4, top_k=2)
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(layer, dummy, "MoLoRA_Conv")
        assert len(raw) > 0, "MoLoRA Conv ONNX export failed"

    def test_molora_linear_onnx_export(self):
        base = nn.Linear(64, 128)
        layer = MoLoRALayer(base, r=4, alpha=8, num_experts=4, top_k=2)
        dummy = torch.randn(1, 64)
        layer.eval()
        buf = io.BytesIO()
        torch.onnx.export(
            layer, dummy, buf,
            input_names=["input"], output_names=["output"],
            opset_version=17, do_constant_folding=False, dynamo=False,
        )
        assert len(buf.getvalue()) > 0

    def test_molora_single_expert_onnx(self):
        """Single-expert MoLoRA (no routing) should export cleanly."""
        base = nn.Conv2d(64, 64, 3, padding=1)
        layer = MoLoRALayer(base, r=4, alpha=8, num_experts=1, top_k=1)
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(layer, dummy, "MoLoRA_1expert")
        assert len(raw) > 0

    def test_molora_merged_onnx(self):
        """Merged MoLoRA (weights absorbed into base) should export."""
        base = nn.Conv2d(64, 64, 3, padding=1)
        layer = MoLoRALayer(base, r=4, alpha=8, num_experts=2, top_k=1)
        layer.merge_weights()
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(layer, dummy, "MoLoRA_merged")
        assert len(raw) > 0


# ── TorchScript export tests ────────────────────────────────────────────

class TestTorchScriptExport:
    """Verify mixture modules can be traced/scripted to TorchScript."""

    def test_moablock_trace(self):
        m = MoABlock(64, num_heads=6)
        m.eval()
        dummy = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            traced = torch.jit.trace(m, dummy)
        with torch.no_grad():
            eager_out = m(dummy)
            traced_out = traced(dummy)
        assert torch.allclose(eager_out, traced_out, atol=1e-4), \
            "MoABlock traced output differs from eager"

    def test_motblock_trace(self):
        m = MoTBlock(64, num_heads=4, top_k=2)
        m.eval()
        dummy = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            traced = torch.jit.trace(m, dummy)
        with torch.no_grad():
            eager_out = m(dummy)
            traced_out = traced(dummy)
        # MoT forward returns (out, aux) tuple
        if isinstance(eager_out, tuple):
            eager_out = eager_out[0]
        if isinstance(traced_out, tuple):
            traced_out = traced_out[0]
        assert torch.allclose(eager_out, traced_out, atol=1e-4), \
            "MoTBlock traced output differs from eager"

    def test_molora_trace(self):
        base = nn.Conv2d(64, 64, 3, padding=1)
        layer = MoLoRALayer(base, r=4, alpha=8, num_experts=2, top_k=1)
        layer.eval()
        dummy = torch.randn(1, 64, 8, 8)
        with torch.no_grad():
            traced = torch.jit.trace(layer, dummy)
        with torch.no_grad():
            eager_out = layer(dummy)
            traced_out = traced(dummy)
        assert torch.allclose(eager_out, traced_out, atol=1e-4), \
            "MoLoRA traced output differs from eager"


# ── Expert preservation tests ───────────────────────────────────────────

class TestExpertPreservation:
    """Verify that ONNX export preserves all experts (no expert dropping)."""

    def test_mot_all_experts_in_graph(self):
        """MoTBlock with 3 experts should have all 3 in the ONNX graph."""
        m = MoTBlock(64, num_heads=4, top_k=2)
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(m, dummy, "MoTBlock")
        if model is None:
            pytest.skip("onnx not installed")
        # Count Conv nodes — should reflect all 3 experts, not just top_k=2
        conv_count = _count_conv_nodes(model)
        # Each expert has at least 1 conv (attention proj), plus out_proj
        assert conv_count >= 3, \
            f"MoTBlock ONNX has only {conv_count} Conv nodes — experts may be dropped"

    def test_molora_all_experts_in_graph(self):
        """MoLoRA with 4 experts should have all 4 in the ONNX graph."""
        base = nn.Conv2d(64, 64, 3, padding=1)
        layer = MoLoRALayer(base, r=4, alpha=8, num_experts=4, top_k=2)
        dummy = torch.randn(1, 64, 8, 8)
        model, raw = _export_onnx(layer, dummy, "MoLoRA")
        if model is None:
            pytest.skip("onnx not installed")
        conv_count = _count_conv_nodes(model)
        # Base conv (1) + 4 expert convs (4) = at least 5 Conv nodes
        assert conv_count >= 5, \
            f"MoLoRA ONNX has only {conv_count} Conv nodes — experts may be dropped"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
