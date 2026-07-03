"""Patch ES_MOE._dense_forward: traceable export-path pruning so the exported graph
matches _sparse_forward (eager eval) numerically. Idempotent (replaces whole method).

  * ONNX/MNN (routing use_top_k=True): topk+threshold pruning, matches eager eval.
  * NCNN     (routing use_top_k=False): pnnx can't lower topk/comparison ops, so NO
    pruning is applied (all experts, dense-full). NCNN trades accuracy for portability.
"""
import sys
from pathlib import Path

NEW_BODY = (
    "    def _dense_forward(self, x, routing_weights):\n"
    '        """Dense forward: compute all experts (used during training).\n'
    "\n"
    "        During export (ONNX/TorchScript tracing) replicate _sparse_forward's\n"
    "        dynamic pruning so the exported graph matches eager eval numerically.\n"
    "        Without this the export keeps ALL experts (incl. weak ones below the\n"
    "        pruning threshold), corrupting output and dropping mAP to ~0.\n"
    "        - ONNX/MNN (routing use_top_k=True): topk+threshold pruning.\n"
    "        - NCNN     (routing use_top_k=False): no pruning (pnnx can't lower topk/\n"
    "          comparison ops) — all experts, dense-full. Training path unchanged.\n"
    '        """\n'
    "        import torch.nn.functional as F\n"
    "        if torch.onnx.is_in_onnx_export() or torch.jit.is_tracing():\n"
    "            B, E, H, W = routing_weights.shape\n"
    "            importance = routing_weights.view(B, E, -1).mean(dim=2)            # [B,E]\n"
    '            thr = getattr(self, "dynamic_threshold", 0.4)\n'
    '            use_topk = getattr(self.routing, "use_top_k", True)\n'
    "            if use_topk:\n"
    "                _, topk_indices = torch.topk(importance, self.top_k, dim=1)\n"
    "                in_topk = F.one_hot(topk_indices, E).sum(dim=1).bool()\n"
    "                rank0 = F.one_hot(importance.argmax(dim=1), E).bool()\n"
    "                keep = in_topk & (rank0 | (importance >= thr))\n"
    "                masked_w = routing_weights * keep.to(routing_weights.dtype).view(B, E, 1, 1)\n"
    "            else:\n"
    "                # NCNN: pnnx cannot lower topk OR comparison ops, so pruning is\n"
    "                # impossible — fall back to all-experts dense (valid graph).\n"
    "                masked_w = routing_weights\n"
    "            final_output = 0\n"
    "            for i, expert in enumerate(self.experts):\n"
    "                final_output = final_output + expert(x) * masked_w[:, i:i + 1, :, :]\n"
    "            return final_output\n"
    "        final_output = 0\n"
    "        for i, expert in enumerate(self.experts):\n"
    "            expert_out = expert(x)\n"
    "            weight = routing_weights[:, i:i + 1, :, :]\n"
    "            final_output = final_output + expert_out * weight\n"
    "        return final_output\n"
    "\n"
)

p = Path(sys.argv[1])
s = p.read_text()
a = s.index("    def _dense_forward(self, x, routing_weights):")
b = s.index("    def _compute_load_balancing_loss")
s = s[:a] + NEW_BODY + s[b:]
p.write_text(s)
print("PATCHED _dense_forward (topk + topk-free pruning)")
