"""V-PEFT Compiler: Graph Representation Learning for Architecture Encoding.

Provides a typed heterogeneous computation graph builder and a GATv2-based
architecture encoder that replaces the static 10-D ArchitectureFingerprint
with differentiable, topology-aware node and global embeddings.

Target: AAAI 2026, Section 4.1.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Vocabularies
# ---------------------------------------------------------------------------
_MODULE_TYPE_VOCAB = {
    "Conv2d": 0,
    "Linear": 1,
    "MultiheadAttention": 2,
    "BatchNorm2d": 3,
    "LayerNorm": 4,
    "DepthwiseConv2d": 5,
    "GroupConv2d": 6,
    "AttentionLinear": 7,
    "Other": 8,
}

_SEMANTIC_ROLE_VOCAB = {
    "stem": 0,
    "backbone": 1,
    "neck": 2,
    "head": 3,
    "DFL": 4,
    "text_fusion": 5,
    "MoE_router": 6,
    "MoE_expert": 7,
    "MSDeformAttn": 8,
    "AAttn": 9,
    "other": 10,
}

_EDGE_TYPES = ["sequential", "residual", "attention"]


# ---------------------------------------------------------------------------
# Public dataclasses
# ---------------------------------------------------------------------------

@dataclass
class NodeAttributes:
    """8-dimensional node attribute vector (from problem formulation).

    Attributes:
        tau_i: Operator-type index (see _MODULE_TYPE_VOCAB).
        c_in:  Input channels / features.
        c_out: Output channels / features.
        k_i:   Kernel size (Conv2d) or sequence length hint (1 for Attention, 0 otherwise).
        d_i:   Topological depth in the module tree (number of dots in the name).
        l_i:   Layer index inside the parent Sequential container (0 if none).
        rho_i: Residual-connection flag (0 or 1).
        sigma_i: Semantic-role index (see _SEMANTIC_ROLE_VOCAB).
    """
    tau_i: int
    c_in: int
    c_out: int
    k_i: int
    d_i: int
    l_i: int
    rho_i: int
    sigma_i: int


@dataclass
class GraphNode:
    """A node in the computation graph."""
    name: str
    module: Any  # nn.Module in new path, ModuleNode/dummy in legacy path
    attributes: NodeAttributes

    @property
    def semantic_role(self) -> str:
        """Semantic role string for compatibility with policy module."""
        inv_map = {v: k for k, v in _SEMANTIC_ROLE_VOCAB.items()}
        return inv_map.get(self.attributes.sigma_i, "other")

    def params_for_rank(self, rank: int, variant: str) -> float:
        """Adapter parameter count for a given rank and variant."""
        if variant in ("lora", "dora"):
            if self.attributes.tau_i in (
                _MODULE_TYPE_VOCAB["Conv2d"],
                _MODULE_TYPE_VOCAB["DepthwiseConv2d"],
                _MODULE_TYPE_VOCAB["GroupConv2d"],
            ):
                k = self.attributes.k_i
                return float(rank * (self.attributes.c_in + self.attributes.c_out) * k)
            else:
                return float(rank * (self.attributes.c_in + self.attributes.c_out))
        elif variant == "ia3":
            return float(self.attributes.c_in)
        elif variant in ("loha", "lokr"):
            return float((rank ** 2) * min(self.attributes.c_in, self.attributes.c_out))
        else:
            return float(rank * (self.attributes.c_in + self.attributes.c_out))


@dataclass
class GraphEdge:
    """A typed edge in the computation graph."""
    src: int
    dst: int
    edge_type: Literal["sequential", "residual", "attention"]


class ModuleNode:
    """Backward-compatible lightweight node used by solver / policy modules.

    This class is kept for compatibility with the solver, policy, and constraints
    modules that construct ``ComputationGraph`` via ``ModuleNode`` lists.
    """

    def __init__(
        self,
        name: str,
        op_type: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
    ):
        self.name = name
        self.op_type = op_type
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size
        self.groups = groups

    def __repr__(self) -> str:
        return (
            f"ModuleNode({self.name}, {self.op_type}, "
            f"in={self.in_channels}, out={self.out_channels})"
        )

    @property
    def semantic_role(self) -> str:
        """Semantic role string for compatibility with policy module."""
        lname = self.name.lower()
        if any(k in lname for k in ("head", "detect", "segment", "pose", "obb")):
            return "head"
        if any(k in lname for k in ("neck", "fpn", "pan")):
            return "neck"
        if any(k in lname for k in ("backbone", "layer", "stage", "res", "encoder", "body")):
            return "backbone"
        if "attn" in lname or "attention" in lname:
            return "attention"
        return "other"

    def params_for_rank(self, rank: int, variant: str) -> float:
        """Adapter parameter count for a given rank and variant."""
        if variant in ("lora", "dora"):
            if self.op_type in ("Conv2d", "DepthwiseConv2d", "GroupConv2d"):
                k = self.kernel_size[0] * self.kernel_size[1]
                return float(rank * (self.in_channels + self.out_channels) * k)
            else:
                return float(rank * (self.in_channels + self.out_channels))
        elif variant == "ia3":
            return float(self.in_channels)
        elif variant in ("loha", "lokr"):
            return float((rank ** 2) * min(self.in_channels, self.out_channels))
        else:
            return float(rank * (self.in_channels + self.out_channels))


class ComputationGraph:
    """Heterogeneous computation graph G = (V, E).

    Supports both the **legacy** construction path (list of ``ModuleNode``
    objects consumed by solver/policy) and the **new** path (``GraphNode`` +
    ``GraphEdge`` lists consumed by the GATv2 encoder).
    """

    def __init__(
        self,
        nodes: Optional[List[GraphNode]] = None,
        edges: Optional[List[GraphEdge]] = None,
        modules: Optional[List[ModuleNode]] = None,
        node_features: Optional[torch.Tensor] = None,
    ):
        self.nodes: List[GraphNode] = nodes if nodes is not None else []
        self.edges: List[GraphEdge] = edges if edges is not None else []
        self._legacy_modules: Optional[List[ModuleNode]] = modules
        self._node_features: Optional[torch.Tensor] = node_features

        # If legacy modules are provided but nodes are not, synthesise GraphNodes
        if not self.nodes and modules is not None:
            for m in modules:
                attr = NodeAttributes(
                    tau_i=_MODULE_TYPE_VOCAB.get(m.op_type, _MODULE_TYPE_VOCAB["Other"]),
                    c_in=m.in_channels,
                    c_out=m.out_channels,
                    k_i=m.kernel_size[0] if m.kernel_size[0] > 0 else 1,
                    d_i=0,
                    l_i=0,
                    rho_i=0,
                    sigma_i=_SEMANTIC_ROLE_VOCAB.get(m.semantic_role, _SEMANTIC_ROLE_VOCAB["other"]),
                )
                self.nodes.append(GraphNode(name=m.name, module=m, attributes=attr))

    # ------------------------------------------------------------------
    # Legacy compatibility (solver / policy / constraints)
    # ------------------------------------------------------------------
    @property
    def modules(self) -> List[ModuleNode]:
        if self._legacy_modules is not None:
            return self._legacy_modules
        if not self.nodes:
            return []
        # Best-effort conversion from GraphNode to ModuleNode
        out = []
        for n in self.nodes:
            attr = n.attributes
            op_name = {v: k for k, v in _MODULE_TYPE_VOCAB.items()}.get(attr.tau_i, "Other")
            out.append(
                ModuleNode(
                    name=n.name,
                    op_type=op_name,
                    in_channels=attr.c_in,
                    out_channels=attr.c_out,
                    kernel_size=attr.k_i if attr.k_i > 0 else 1,
                    groups=getattr(n.module, "groups", 1),
                )
            )
        return out

    @property
    def n_nodes(self) -> int:
        if self._legacy_modules is not None:
            return len(self._legacy_modules)
        return len(self.nodes)

    def get_node_importances(self) -> torch.Tensor:
        if self._node_features is not None:
            return self._node_features.mean(dim=-1)
        return torch.ones(self.n_nodes, dtype=torch.float32)

    def get_module_names(self) -> List[str]:
        return [m.name for m in self.modules]

    def get_module_types(self) -> List[str]:
        return [m.op_type for m in self.modules]

    def estimate_params(self, idx: int, rank, variant: str):
        """Estimate adapter parameters for module ``idx`` with ``rank`` and ``variant``.

        Supports both scalar floats and 0-D tensors for ``rank`` to preserve
        gradient flow in differentiable solvers.
        """
        m = self.modules[idx]
        if variant in ("lora", "dora"):
            if m.op_type in ("Conv2d", "DepthwiseConv2d", "GroupConv2d"):
                k = m.kernel_size[0] * m.kernel_size[1]
                val = rank * (m.in_channels + m.out_channels) * k
            else:
                val = rank * (m.in_channels + m.out_channels)
        elif variant == "ia3":
            val = m.in_channels
        elif variant in ("loha", "lokr"):
            val = (rank ** 2) * min(m.in_channels, m.out_channels)
        else:
            val = rank * (m.in_channels + m.out_channels)
        return float(val) if not isinstance(val, torch.Tensor) else val

    # ------------------------------------------------------------------
    # New path helpers
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return self.n_nodes


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class ComputationGraphBuilder:
    """Build a typed heterogeneous computation graph from a PyTorch model.

    Scans ``model.named_modules()`` and creates a ``GraphNode`` for each
    leaf/iconic module of interest (Conv2d, Linear, MultiheadAttention,
    BatchNorm2d, LayerNorm).  Edges are inferred from the module hierarchy,
    Sequential ordering, and lightweight name heuristics for residual/attention
    connectivity.

    Example::
        builder = ComputationGraphBuilder()
        graph = builder.build(model)
    """

    def __init__(self) -> None:
        self._node_map: Dict[str, int] = {}
        self._nodes: List[GraphNode] = []
        self._edges: List[GraphEdge] = []

    @staticmethod
    def _unwrap_model(model: nn.Module) -> nn.Module:
        """Unwrap DDP / DataParallel / torch.compile wrappers."""
        while hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        return model

    @staticmethod
    def _get_operator_type(module: nn.Module) -> int:
        """Map a PyTorch module to its operator-type index."""
        if isinstance(module, nn.Conv2d):
            if module.in_channels == module.out_channels == module.groups:
                return _MODULE_TYPE_VOCAB["DepthwiseConv2d"]
            elif module.groups > 1:
                return _MODULE_TYPE_VOCAB["GroupConv2d"]
            return _MODULE_TYPE_VOCAB["Conv2d"]
        if isinstance(module, nn.Linear):
            return _MODULE_TYPE_VOCAB["Linear"]
        if isinstance(module, nn.MultiheadAttention):
            return _MODULE_TYPE_VOCAB["MultiheadAttention"]
        if isinstance(module, nn.BatchNorm2d):
            return _MODULE_TYPE_VOCAB["BatchNorm2d"]
        if isinstance(module, nn.LayerNorm):
            return _MODULE_TYPE_VOCAB["LayerNorm"]
        return _MODULE_TYPE_VOCAB["Other"]

    @staticmethod
    def _has_residual_pattern(name: str) -> bool:
        """Heuristic residual-block detection from module/path names."""
        keywords = {
            "c2f", "c3k2", "c3", "c2psa", "a2c2f", "bottleneck",
            "residual", "sppf", "spp", "c2fcib", "repc3", "c3k",
            "shortcut", "add",
        }
        return any(kw in name.lower() for kw in keywords)

    @staticmethod
    def _infer_semantic_role(name: str) -> int:
        """Infer semantic role from the module's full name."""
        lname = name.lower()
        if "stem" in lname or "patch_embed" in lname or "conv0" in lname:
            return _SEMANTIC_ROLE_VOCAB["stem"]
        if any(k in lname for k in ("text_encoder", "clip", "text_fusion", "world_embed", "text_proj")):
            return _SEMANTIC_ROLE_VOCAB["text_fusion"]
        if any(k in lname for k in ("moe_router", "moe_gate", "router")) and "moe_expert" not in lname:
            return _SEMANTIC_ROLE_VOCAB["MoE_router"]
        if any(k in lname for k in ("moe_expert", "expert")):
            return _SEMANTIC_ROLE_VOCAB["MoE_expert"]
        if any(k in lname for k in ("msdeform", "deformable")) and "msdeform" in lname:
            return _SEMANTIC_ROLE_VOCAB["MSDeformAttn"]
        if "dfl" in lname:
            return _SEMANTIC_ROLE_VOCAB["DFL"]
        if any(k in lname for k in ("head", "detect", "segment", "pose", "obb", "yoloedetect", "v10detect")):
            return _SEMANTIC_ROLE_VOCAB["head"]
        if any(k in lname for k in ("neck", "fpn", "pan")):
            return _SEMANTIC_ROLE_VOCAB["neck"]
        if any(k in lname for k in ("backbone", "layer", "stage", "res", "encoder", "body")):
            return _SEMANTIC_ROLE_VOCAB["backbone"]
        if "aattn" in lname or any(k in lname for k in ("attention", "attn")):
            return _SEMANTIC_ROLE_VOCAB["AAttn"]
        return _SEMANTIC_ROLE_VOCAB["other"]

    @staticmethod
    def _get_sequential_index(name: str, root: nn.Module) -> int:
        """Return the position of this module inside its parent Sequential, or 0."""
        if "." not in name:
            return 0
        parent_name, child_name = name.rsplit(".", 1)
        try:
            parent = root.get_submodule(parent_name) if hasattr(root, "get_submodule") else None
            if parent is None:
                parts = parent_name.split(".")
                parent = root
                for p in parts:
                    parent = getattr(parent, p, None)
                    if parent is None:
                        return 0
            if isinstance(parent, nn.Sequential):
                for idx, (n, _) in enumerate(parent.named_children()):
                    if n == child_name:
                        return idx
        except Exception:
            pass
        return 0

    def _build_nodes(self, model: nn.Module) -> Tuple[List[GraphNode], Dict[str, int]]:
        """Phase 1: create GraphNodes for all target modules."""
        target_types = (nn.Conv2d, nn.Linear, nn.MultiheadAttention, nn.BatchNorm2d, nn.LayerNorm)
        nodes: List[GraphNode] = []
        node_map: Dict[str, int] = {}

        for name, module in model.named_modules():
            if not isinstance(module, target_types):
                continue

            tau_i = self._get_operator_type(module)

            if isinstance(module, nn.Conv2d):
                c_in = module.in_channels
                c_out = module.out_channels
                k_i = module.kernel_size[0] if isinstance(module.kernel_size, (tuple, list)) else module.kernel_size
            elif isinstance(module, nn.Linear):
                c_in = module.in_features
                c_out = module.out_features
                k_i = 0
            elif isinstance(module, nn.MultiheadAttention):
                c_in = module.embed_dim
                c_out = module.embed_dim
                k_i = 1
            elif isinstance(module, nn.BatchNorm2d):
                c_in = module.num_features
                c_out = module.num_features
                k_i = 0
            else:  # LayerNorm
                normalized_shape = module.normalized_shape
                c_in = normalized_shape[0] if isinstance(normalized_shape, (tuple, list)) else normalized_shape
                c_out = c_in
                k_i = 0

            d_i = name.count(".")
            l_i = self._get_sequential_index(name, model)
            rho_i = 1 if self._has_residual_pattern(name) else 0
            # Also set rho_i=1 if any parent module is a known residual block type.
            if not rho_i:
                try:
                    parts = name.split(".")
                    for depth in range(1, len(parts)):
                        prefix = ".".join(parts[:depth])
                        parent = model.get_submodule(prefix) if hasattr(model, "get_submodule") else None
                        if parent is None:
                            parent_parts = prefix.split(".")
                            parent = model
                            for p in parent_parts:
                                parent = getattr(parent, p)
                        cls_name = parent.__class__.__name__.lower()
                        if any(kw in cls_name for kw in {"residual", "bottleneck", "c2f", "c3", "c2psa", "a2c2f", "c3k2"}):
                            rho_i = 1
                            break
                except Exception:
                    pass
            sigma_i = self._infer_semantic_role(name)

            attr = NodeAttributes(
                tau_i=tau_i,
                c_in=c_in,
                c_out=c_out,
                k_i=k_i,
                d_i=d_i,
                l_i=l_i,
                rho_i=rho_i,
                sigma_i=sigma_i,
            )
            node = GraphNode(name=name, module=module, attributes=attr)
            node_map[name] = len(nodes)
            nodes.append(node)

        return nodes, node_map

    def _build_edges(self, model: nn.Module, nodes: List[GraphNode], node_map: Dict[str, int]) -> List[GraphEdge]:
        """Phase 2: infer sequential, residual, and attention edges."""
        edges: List[GraphEdge] = []
        seen: Set[Tuple[int, int, str]] = set()

        def _add(src: int, dst: int, etype: str) -> None:
            key = (src, dst, etype)
            if key not in seen:
                seen.add(key)
                edges.append(GraphEdge(src=src, dst=dst, edge_type=etype))  # type: ignore[arg-type]

        # --- Sequential edges ---
        for i, node in enumerate(nodes):
            name = node.name
            # Parent -> child (hierarchy)
            if "." in name:
                parent_name = name.rsplit(".", 1)[0]
                if parent_name in node_map:
                    _add(node_map[parent_name], i, "sequential")

            # Sequential sibling -> next sibling
            if "." in name:
                parent_name, child_name = name.rsplit(".", 1)
                try:
                    parent = model.get_submodule(parent_name) if hasattr(model, "get_submodule") else None
                    if parent is None:
                        parts = parent_name.split(".")
                        parent = model
                        for p in parts:
                            parent = getattr(parent, p)
                    if isinstance(parent, nn.Sequential):
                        children = list(parent.named_children())
                        for idx, (n, _) in enumerate(children):
                            if n == child_name and idx + 1 < len(children):
                                next_name = f"{parent_name}.{children[idx + 1][0]}"
                                if next_name in node_map:
                                    _add(i, node_map[next_name], "sequential")
                                break
                except Exception:
                    pass

        # --- Residual edges (name-heuristic + parent class check) ---
        block_buckets: Dict[str, List[int]] = defaultdict(list)
        for i, node in enumerate(nodes):
            parts = node.name.split(".")
            for depth in range(1, len(parts)):
                prefix = ".".join(parts[:depth])
                block_buckets[prefix].append(i)

        for prefix, indices in block_buckets.items():
            if len(indices) < 2:
                continue
            has_residual = any(self._has_residual_pattern(nodes[j].name) for j in indices)
            # Also check parent module class name for known residual block types.
            if not has_residual:
                try:
                    parent = model.get_submodule(prefix) if hasattr(model, "get_submodule") else None
                    if parent is None:
                        parts = prefix.split(".")
                        parent = model
                        for p in parts:
                            parent = getattr(parent, p)
                    cls_name = parent.__class__.__name__.lower()
                    if any(kw in cls_name for kw in {"residual", "bottleneck", "c2f", "c3", "c2psa", "a2c2f", "c3k2"}):
                        has_residual = True
                except Exception:
                    pass
            if not has_residual:
                continue

            convs = [j for j in indices if nodes[j].attributes.tau_i in (0, 5, 6)]  # Conv variants
            if len(convs) >= 2:
                _add(convs[0], convs[-1], "residual")

            shortcut_nodes = [j for j in indices if any(k in nodes[j].name.lower() for k in ("shortcut", "add", "skip"))]
            for j in shortcut_nodes:
                for k in indices:
                    if k != j:
                        _add(k, j, "residual")
                        _add(j, k, "residual")

        # --- Attention edges (Q/K/V -> Proj) ---
        for i, node in enumerate(nodes):
            if node.attributes.tau_i != 2:  # not MultiheadAttention
                continue
            qkv_idxs: List[int] = []
            proj_idxs: List[int] = []
            for child_name, _ in node.module.named_modules():
                if child_name == "":
                    continue
                full_name = f"{node.name}.{child_name}"
                if full_name not in node_map:
                    continue
                cname = child_name.lower()
                if any(k in cname for k in ("qkv", "query", "key", "value", "q_proj", "k_proj", "v_proj", "in_proj")):
                    qkv_idxs.append(node_map[full_name])
                elif any(k in cname for k in ("proj", "out_proj", "o_proj", "output")):
                    proj_idxs.append(node_map[full_name])
            for qkv in qkv_idxs:
                for proj in proj_idxs:
                    _add(qkv, proj, "attention")

        return edges

    def build(self, model: nn.Module) -> ComputationGraph:
        """Build the computation graph from a PyTorch model.

        Args:
            model: A PyTorch ``nn.Module`` (e.g., a YOLO/RT-DETR model).

        Returns:
            A ``ComputationGraph`` with typed nodes and edges.
        """
        model = self._unwrap_model(model)
        nodes, node_map = self._build_nodes(model)
        edges = self._build_edges(model, nodes, node_map)
        return ComputationGraph(nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Internal node-feature encoder
# ---------------------------------------------------------------------------

class _NodeAttributeEncoder(nn.Module):
    """Learnable encoder that maps raw node attributes to initial feature vectors.

    Follows the design-doc formula:
        x_i = W_proj · Concat[e_t; e_s; c; h; r] + b_proj
    """

    def __init__(self, hidden_dim: int = 64) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        d_t = 32
        d_s = 32
        d_r = 8
        c_dim = 5  # [log2(c_in+1), log2(c_out+1), log2(k_i+1), log2(g_i+1), d_i/D_max]
        h_dim = 2  # [log2(p_i+1), log2(f_i+1)]
        in_dim = d_t + d_s + c_dim + h_dim + d_r  # 79

        self.type_embed = nn.Embedding(len(_MODULE_TYPE_VOCAB), d_t)
        self.role_embed = nn.Embedding(len(_SEMANTIC_ROLE_VOCAB), d_s)
        self.continuous_norm = nn.LayerNorm(c_dim)
        self.hw_norm = nn.LayerNorm(h_dim)
        self.w_r = nn.Parameter(torch.randn(d_r))
        self.W_proj = nn.Linear(in_dim, hidden_dim, bias=False)
        self.b_proj = nn.Parameter(torch.zeros(hidden_dim))
        self.register_buffer("D_max", torch.tensor(50.0))
        self.register_buffer("L_max", torch.tensor(100.0))

    def forward(self, nodes: List[GraphNode]) -> torch.Tensor:
        """Encode a list of GraphNodes into initial feature vectors [N, hidden_dim]."""
        if not nodes:
            return torch.empty(0, self.hidden_dim, device=self.W_proj.weight.device, dtype=self.W_proj.weight.dtype)

        device = self.W_proj.weight.device
        dtype = self.W_proj.weight.dtype

        features = []
        for node in nodes:
            attr = node.attributes
            mod = node.module

            e_t = self.type_embed(torch.tensor(attr.tau_i, device=device, dtype=torch.long))
            e_s = self.role_embed(torch.tensor(attr.sigma_i, device=device, dtype=torch.long))

            g_i = getattr(mod, "groups", 1)
            p_i = sum(p.numel() for p in mod.parameters())
            if isinstance(mod, nn.Conv2d):
                k = attr.k_i
                f_i = 2 * attr.c_in * attr.c_out * k * k
            elif isinstance(mod, nn.Linear):
                f_i = 2 * attr.c_in * attr.c_out
            elif isinstance(mod, nn.MultiheadAttention):
                f_i = 2 * attr.c_in * attr.c_in
            else:
                f_i = 0.0

            c = torch.tensor(
                [
                    math.log2(max(attr.c_in, 0) + 1.0),
                    math.log2(max(attr.c_out, 0) + 1.0),
                    math.log2(max(attr.k_i, 0) + 1.0),
                    math.log2(max(g_i, 0) + 1.0),
                    min(attr.d_i / self.D_max.item(), 1.0),
                ],
                device=device,
                dtype=dtype,
            )
            c = self.continuous_norm(c)

            h = torch.tensor(
                [math.log2(p_i + 1.0), math.log2(f_i + 1.0)],
                device=device,
                dtype=dtype,
            )
            h = self.hw_norm(h)

            r = attr.rho_i * self.w_r

            x = torch.cat([e_t, e_s, c, h, r], dim=0)
            features.append(x)

        x = torch.stack(features)  # [N, 79]
        return self.W_proj(x) + self.b_proj  # [N, hidden_dim]


# ---------------------------------------------------------------------------
# GATv2 encoder
# ---------------------------------------------------------------------------

class _GATv2Layer(nn.Module):
    """Single GATv2 message-passing layer with edge-type-specific weights."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        edge_types: List[str],
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.edge_types = edge_types
        self.dropout = dropout

        self.W_self = nn.Linear(hidden_dim, hidden_dim, bias=False)

        for etype in edge_types:
            setattr(self, f"W_q_{etype}", nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, f"W_k_{etype}", nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, f"W_msg_{etype}", nn.Linear(hidden_dim, hidden_dim, bias=False))
            setattr(self, f"a_{etype}", nn.Parameter(torch.randn(num_heads, self.head_dim, 1)))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
        for etype in self.edge_types:
            nn.init.xavier_uniform_(getattr(self, f"a_{etype}"))

    @staticmethod
    def _scatter_softmax(scores: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Compute softmax over incoming edges per destination node and per head.

        Args:
            scores: [E, K] attention logits.
            dst:    [E] destination indices.
            num_nodes: Number of nodes.

        Returns:
            alpha: [E, K] normalised attention weights.
        """
        K = scores.shape[1]
        max_score = torch.full((num_nodes, K), -1e9, device=scores.device, dtype=scores.dtype)
        max_score.scatter_reduce_(0, dst.unsqueeze(1).expand(-1, K), scores, reduce="amax", include_self=False)
        max_per_edge = max_score[dst]  # [E, K]

        exp_scores = torch.exp(scores - max_per_edge)
        sum_exp = torch.zeros(num_nodes, K, device=scores.device, dtype=scores.dtype)
        sum_exp.scatter_add_(0, dst.unsqueeze(1).expand(-1, K), exp_scores)
        alpha = exp_scores / (sum_exp[dst] + 1e-8)
        return alpha

    def _message(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        etype: str,
        num_nodes: int,
    ) -> torch.Tensor:
        """Compute edge-type-specific messages for one edge type.

        Returns:
            Tensor of shape [N, hidden_dim].
        """
        E = src.shape[0]
        if E == 0:
            return torch.zeros(num_nodes, self.hidden_dim, device=h.device, dtype=h.dtype)

        K = self.num_heads
        D = self.head_dim

        W_q = getattr(self, f"W_q_{etype}")
        W_k = getattr(self, f"W_k_{etype}")
        W_msg = getattr(self, f"W_msg_{etype}")
        a = getattr(self, f"a_{etype}")

        h_dst = h[dst]  # [E, hidden_dim]
        h_src = h[src]  # [E, hidden_dim]

        q = W_q(h_dst).view(E, K, D)  # [E, K, D]
        k = W_k(h_src).view(E, K, D)  # [E, K, D]
        msg = W_msg(h_src).view(E, K, D)  # [E, K, D]

        attn_logits = F.leaky_relu(q + k, negative_slope=0.2)  # [E, K, D]
        scores = torch.einsum("ekd,kdh->ek", attn_logits, a)  # [E, K]

        alpha = self._scatter_softmax(scores, dst, num_nodes)  # [E, K]

        if self.training and self.dropout > 0.0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        weighted = alpha.unsqueeze(-1) * msg  # [E, K, D]

        out = torch.zeros(num_nodes, K, D, device=h.device, dtype=h.dtype)
        dst_exp = dst.unsqueeze(1).unsqueeze(2).expand(-1, K, D)  # [E, K, D]
        out.scatter_add_(0, dst_exp, weighted)

        out = out.reshape(num_nodes, -1)  # [N, hidden_dim]
        return out

    def forward(self, h: torch.Tensor, edge_index_dict: Dict[str, Tuple[torch.Tensor, torch.Tensor]], num_nodes: int) -> torch.Tensor:
        """Forward one GATv2 layer.

        Args:
            h: [N, hidden_dim] current node embeddings.
            edge_index_dict: Mapping edge_type -> (src, dst) tensors.
            num_nodes: Number of nodes.

        Returns:
            Updated embeddings [N, hidden_dim].
        """
        h_self = self.W_self(h)  # [N, hidden_dim]

        msgs = torch.zeros_like(h_self)
        for etype, (src, dst) in edge_index_dict.items():
            msgs = msgs + self._message(h, src, dst, etype, num_nodes)

        return F.gelu(msgs + h_self)


class GATv2ArchitectureEncoder(nn.Module):
    """GATv2-based Architecture Encoder.

    Encodes a typed heterogeneous computation graph into per-node embeddings
    and a global graph embedding via attention-weighted pooling.

    Args:
        hidden_dim: Dimension of hidden/node embeddings (default 64).
        num_layers: Number of GATv2 message-passing layers (default 3).
        num_heads: Number of attention heads (default 4).
        dropout: Dropout probability on attention weights (default 0.1).
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 3, num_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.node_encoder = _NodeAttributeEncoder(hidden_dim)
        self.gnn_layers = nn.ModuleList(
            [_GATv2Layer(hidden_dim, num_heads, _EDGE_TYPES, dropout) for _ in range(num_layers)]
        )

        # Global readout: gamma_i = softmax_i(w_g^T GELU(W_g h_i))
        self.W_g = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.w_g = nn.Parameter(torch.randn(hidden_dim))

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.w_g, std=0.01)

    @staticmethod
    def _build_edge_index_dict(edges: List[GraphEdge], num_nodes: int, device: torch.device) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        """Convert Python edge list to per-type (src, dst) index tensors."""
        buckets: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        for e in edges:
            buckets[e.edge_type].append((e.src, e.dst))
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for etype in _EDGE_TYPES:
            if etype in buckets and buckets[etype]:
                pairs = buckets[etype]
                src = torch.tensor([p[0] for p in pairs], device=device, dtype=torch.long)
                dst = torch.tensor([p[1] for p in pairs], device=device, dtype=torch.long)
                out[etype] = (src, dst)
            else:
                out[etype] = (
                    torch.empty(0, device=device, dtype=torch.long),
                    torch.empty(0, device=device, dtype=torch.long),
                )
        return out

    def forward(self, graph: ComputationGraph) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a computation graph.

        Args:
            graph: A ``ComputationGraph`` built by ``ComputationGraphBuilder``.

        Returns:
            A tuple of:
                - node_embeddings: [N, hidden_dim]
                - global_embedding: [hidden_dim]
        """
        num_nodes = len(graph.nodes)
        if num_nodes == 0:
            return (
                torch.empty(0, self.hidden_dim, device=self.W_g.weight.device, dtype=self.W_g.weight.dtype),
                torch.zeros(self.hidden_dim, device=self.W_g.weight.device, dtype=self.W_g.weight.dtype),
            )

        x = self.node_encoder(graph.nodes)  # [N, hidden_dim]
        device = x.device
        edge_index_dict = self._build_edge_index_dict(graph.edges, num_nodes, device)

        h = x
        for layer in self.gnn_layers:
            h = layer(h, edge_index_dict, num_nodes)
            if self.training and self.dropout > 0.0:
                h = F.dropout(h, p=self.dropout, training=self.training)

        # Global readout: attention-weighted pooling
        gate = F.gelu(self.W_g(h))  # [N, hidden_dim]
        scores = gate @ self.w_g  # [N]
        gamma = F.softmax(scores, dim=0)  # [N]
        h_G = (gamma.unsqueeze(1) * h).sum(dim=0)  # [hidden_dim]

        return h, h_G


__all__ = [
    "NodeAttributes",
    "GraphNode",
    "GraphEdge",
    "ModuleNode",
    "ComputationGraph",
    "ComputationGraphBuilder",
    "GATv2ArchitectureEncoder",
]
