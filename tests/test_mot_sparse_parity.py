import torch

from ultralytics.nn.modules.mot import MoTBlock


def test_mot_sparse_and_dense_eval_are_close():
    torch.manual_seed(0)
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=False).eval()
    x = torch.randn(4, 24, 4, 4)
    with torch.no_grad():
        weights, indices, _ = block.router(x, return_logits=True)
        dense_mixture = sum((expert(x) * weights[:, i:i + 1] for i, expert in enumerate(block.experts)), torch.zeros_like(x))
        out_dense = block.out_norm(block.out_proj(dense_mixture)) + x
        out_sparse, _ = block(x)
    assert torch.isfinite(out_sparse).all()
    assert out_dense.shape == out_sparse.shape
    assert torch.allclose(out_dense, out_sparse, atol=1e-5, rtol=1e-4)
    assert block._last_dispatch_stats["mode"] == "sample_sparse"


def test_mot_dense_export_path_records_all_experts():
    block = MoTBlock(24, num_heads=3, top_k=1, sparse_train=False).train()
    block(torch.randn(2, 24, 4, 4))
    assert block._last_dispatch_stats["mode"] == "dense"
    assert block._last_dispatch_stats["expert_calls"] == len(block.experts)
