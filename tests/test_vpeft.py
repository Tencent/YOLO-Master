"""Regression tests for the experimental V-PEFT solver package."""

import torch
import torch.nn as nn
from pathlib import Path

import pytest


def test_vpeft_core_package_imports_without_unimplemented_adapter():
    import ultralytics.vpeft as vpeft

    assert vpeft.PlacementPolicy is not None
    assert vpeft.AlternatingOptimizationSolver is not None
    assert "PlacementPolicy" in vpeft.__all__
    assert "DynamicAdapterModel" not in vpeft.__all__


def test_graph_node_info_preserves_grouped_convolution_metadata():
    from ultralytics.vpeft import ComputationGraphBuilder, NodeInfo

    model = nn.Sequential(nn.Conv2d(4, 4, 3, groups=4), nn.Conv2d(4, 8, (1, 3), groups=2))
    graph = ComputationGraphBuilder().build(model)
    depthwise = NodeInfo(graph.nodes[0])
    grouped = NodeInfo(graph.nodes[1])

    assert depthwise.operator_type == "DepthwiseConv2d"
    assert depthwise.groups == 4 and depthwise.is_depthwise
    assert depthwise.kernel_size == (3, 3)
    assert grouped.operator_type == "GroupConv2d"
    assert grouped.groups == 2 and grouped.kernel_size == (1, 3)


def test_hard_mask_accepts_supported_grouped_and_opt_in_depthwise_ranks():
    from ultralytics.vpeft import ComputationGraphBuilder, ConstraintRegistry

    model = nn.Sequential(nn.Conv2d(8, 8, 3, groups=8), nn.Conv2d(8, 16, 3, groups=4))
    graph = ComputationGraphBuilder().build(model)

    default_mask = ConstraintRegistry.default().get_hard_mask(graph, "lora", candidate_ranks=[4, 8])
    enabled_mask = ConstraintRegistry.default({"allow_depthwise": True}).get_hard_mask(
        graph, "lora", candidate_ranks=[4, 8]
    )

    assert default_mask.tolist() == [False, True]
    assert enabled_mask.tolist() == [True, True]


def test_vpeft_conv_budget_preserves_rectangular_kernel_and_group_layout():
    from ultralytics.vpeft import ComputationGraphBuilder, NodeInfo
    from ultralytics.utils.lora.fallback import ManualLoRAConv

    dense_conv = nn.Conv2d(8, 16, (1, 3))
    grouped_conv = nn.Conv2d(8, 16, (1, 3), groups=4)
    graph = ComputationGraphBuilder().build(nn.Sequential(dense_conv, grouped_conv))
    dense_adapter = ManualLoRAConv(dense_conv, r=4)
    grouped_adapter = ManualLoRAConv(grouped_conv, r=4)
    dense_trainable = sum(parameter.numel() for parameter in dense_adapter.parameters() if parameter.requires_grad)
    grouped_trainable = sum(parameter.numel() for parameter in grouped_adapter.parameters() if parameter.requires_grad)

    assert graph.modules[0].kernel_size == (1, 3)
    assert graph.estimate_params(0, 4, "lora") == dense_trainable == 160.0
    assert graph.estimate_params(1, 4, "lora") == grouped_trainable == 40.0
    assert graph.nodes[0].params_for_rank(4, "lora") == 160.0
    assert NodeInfo(graph.nodes[1]).kernel_size == (1, 3)


def test_semantic_protection_is_case_insensitive_for_reserved_roles_and_names():
    from ultralytics.vpeft import NodeInfo, SemanticProtectionConstraint

    constraint = SemanticProtectionConstraint(exclude_modules=["Custom.Block"])

    assert not constraint.is_feasible(NodeInfo(name="model.dfl", semantic_role="DFL"), "lora", 4)
    assert not constraint.is_feasible(
        NodeInfo(name="encoder.msdeform", semantic_role="MSDeformAttn"), "lora", 4
    )
    assert not constraint.is_feasible(NodeInfo(name="custom.block.proj", semantic_role="backbone"), "lora", 4)


def test_vpeft_alternating_solver_smoke():
    from ultralytics.vpeft import (
        AlternatingOptimizationSolver,
        ComputationGraph,
        ConstraintRegistry,
        ModuleNode,
    )

    graph = ComputationGraph(
        modules=[ModuleNode("backbone.fc", "Linear", 8, 8)],
        node_features=torch.ones(1, 8),
    )
    constraints = ConstraintRegistry.default({"max_params": 1_000})
    decision = AlternatingOptimizationSolver(max_iter=2, rank_min=4, rank_max=8, rank_step=4).solve(
        graph, budget=1_000, variant="lora", constraints=constraints
    )

    assert decision.status == "ACCEPT"
    assert decision.target_modules == ["backbone.fc"]
    assert 0 < decision.budget_used <= 1_000


def test_vpeft_registry_uses_canonical_soft_constraint_names():
    from ultralytics.vpeft import ComputationGraph, ConstraintRegistry, ModuleNode

    registry = ConstraintRegistry.default({"soft_constraints": ["budget", "deploy"]})
    assert registry.hard_constraint_names() == [
        "C_op", "C_sem", "C_budget", "C_deploy", "C_compat", "C_moe", "C_div"
    ]
    assert registry.soft_constraint_names() == ["C_budget", "C_deploy"]
    graph = ComputationGraph(modules=[ModuleNode("backbone.fc", "Linear", 8, 8)])
    values = registry.evaluate_soft(graph, torch.tensor([1.0]), torch.tensor([4]), ["lora"])
    assert set(values) == {"C_budget", "C_deploy"}

    soft_only = ConstraintRegistry.default({"hard_constraints": [], "soft_constraints": ["budget"]})
    assert soft_only.hard_constraint_names() == []
    assert soft_only.soft_constraint_names() == ["C_budget"]


def test_vpeft_ao_budget_accounts_for_per_node_variants():
    from ultralytics.vpeft import AlternatingOptimizationSolver, ComputationGraph, ConstraintRegistry, ModuleNode

    graph = ComputationGraph(
        modules=[
            ModuleNode("backbone.linear", "Linear", 8, 8),
            ModuleNode("backbone.conv", "Conv2d", 8, 8, kernel_size=3),
        ]
    )
    decision = AlternatingOptimizationSolver(
        max_iter=2, rank_min=4, rank_max=4, rank_step=4
    ).solve(graph, budget=10_000, variant="lora", constraints=ConstraintRegistry.default())

    assert len(decision.variants) == graph.n_nodes
    assert decision.variants == ["ia3", "lora"]
    assert decision.budget_used == sum(
        graph.estimate_params(index, int(decision.ranks[index]), decision.variants[index])
        for index in range(graph.n_nodes)
        if decision.placement[index] > 0.5
    )


def test_vpeft_solver_assigns_only_feasible_grouped_conv_ranks():
    from ultralytics.vpeft import AlternatingOptimizationSolver, ComputationGraphBuilder, ConstraintRegistry

    graph = ComputationGraphBuilder().build(nn.Sequential(nn.Conv2d(8, 8, 3, groups=8)))
    constraints = ConstraintRegistry.default({"allow_depthwise": True, "max_params": 10_000})
    decision = AlternatingOptimizationSolver(max_iter=1, rank_min=4, rank_max=8, rank_step=4).solve(
        graph, budget=10_000, variant="lora", constraints=constraints
    )

    assert decision.status == "ACCEPT"
    assert decision.ranks.tolist() == [8.0]
    assert constraints.is_rank_feasible(graph, 0, "lora", int(decision.ranks[0].item()))


def test_vpeft_ao_budget_dual_penalty_excludes_negative_density_candidates():
    from ultralytics.vpeft import AlternatingOptimizationSolver, ComputationGraph, ModuleNode

    graph = ComputationGraph(
        modules=[
            ModuleNode("backbone.small", "Linear", 8, 8),
            ModuleNode("backbone.large", "Linear", 64, 64),
        ],
        node_features=torch.ones(2, 8),
    )
    solver = AlternatingOptimizationSolver(max_iter=1, rank_min=4, rank_max=4, rank_step=4)
    class _SoftPenalty:
        def _node_info_from_graph(self, graph, index):
            return index

        def compute_penalty_breakdown(self, node_info, variant, rank):
            return {"latency": 1_000.0 if node_info == 1 else 0.0}

    hard_mask = torch.ones(2, dtype=torch.bool)
    ranks = torch.tensor([4, 4])
    utilities = torch.ones(2)
    placement = solver._optimize_pi(
        graph,
        ranks,
        ["lora", "lora"],
        2_000,
        utilities,
        hard_mask,
        _SoftPenalty(),
        {"budget": 0.0, "latency": 1.0, "memory": 0.0, "deploy": 0.0},
    )
    # The latency dual makes the large module's density negative.  AO should
    # therefore retain only the positive-density candidate, not fill budget
    # with a penalised module.
    assert placement.tolist() == [1.0, 0.0]


def test_vpeft_softplus_penalty_stays_finite_for_large_violations():
    from ultralytics.vpeft.solver import _softplus_penalty

    values = _softplus_penalty(torch.tensor([10.0, 100.0, 1_000.0]))

    assert torch.isfinite(values).all()
    assert torch.allclose(values, torch.tensor([10.0, 100.0, 1_000.0]))


def test_graph_builder_annotates_moe_expert_groups_without_manual_registration():
    from ultralytics.vpeft import ComputationGraphBuilder, ConstraintRegistry

    class TinyMoE(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_experts = 2
            self.experts = nn.ModuleList([nn.Conv2d(4, 4, 1), nn.Conv2d(4, 4, 1)])

    graph = ComputationGraphBuilder().build(TinyMoE())
    assert {node.annotations["moe_group"] for node in graph.nodes} == {"experts"}
    mask = ConstraintRegistry.default({"allow_depthwise": True}).get_hard_mask(
        graph, "lora", candidate_ranks=[4, 8]
    )
    assert mask.tolist() == [True, True]


def test_gradient_sensitivity_selector_ranks_real_gradients():
    from ultralytics.utils.lora.sensitivity import GradientSensitivitySelector

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    loader = [torch.randn(3, 4) for _ in range(2)]
    report = GradientSensitivitySelector(model, loader, num_batches=2, top_ratio=0.5).select_targets(["0", "2"])
    assert report.selected_targets == ["2"] or report.selected_targets == ["0"]
    assert all(layer.score >= 0 for layer in report.layers)


def test_vpeft_differentiable_solver_survives_large_budget_violation():
    from ultralytics.vpeft import (
        ComputationGraph,
        ConstraintRegistry,
        DifferentiableOptimizationSolver,
        ModuleNode,
    )

    graph = ComputationGraph(
        modules=[ModuleNode("backbone.fc", "Linear", 1_024, 1_024)],
        node_features=torch.ones(1, 8),
    )
    solver = DifferentiableOptimizationSolver(max_iter=3, rank_min=4, rank_max=8, rank_step=4)
    decision = solver.solve(
        graph, budget=1, variant="lora", constraints=ConstraintRegistry.default({"max_params": 1})
    )

    assert decision.status == "REFUSE" and decision.budget_used == 0
    assert all(
        torch.isfinite(parameter).all()
        for module in (solver._placement_mlp, solver._rank_mlp)
        for parameter in module.parameters()
    )


@pytest.mark.parametrize(
    ("model_class", "config", "expected_head"),
    [
        ("DetectionModel", "yolo26.yaml", "Detect"),
        ("SegmentationModel", "yolo26-seg.yaml", "Segment26"),
        ("PoseModel", "yolo26-pose.yaml", "Pose26"),
        ("OBBModel", "yolo26-obb.yaml", "OBB26"),
        ("SemanticSegmentationModel", "yolo26-sem.yaml", "SemanticSegment"),
        ("YOLOEModel", "yoloe-26.yaml", "YOLOEDetect"),
        ("YOLOESegModel", "yoloe-26-seg.yaml", "YOLOESegment26"),
    ],
)
def test_vpeft_marks_all_yolo26_specialized_heads(model_class, config, expected_head):
    from ultralytics.nn import tasks
    from ultralytics.vpeft import ComputationGraphBuilder, ConstraintRegistry

    root = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/26"
    model = getattr(tasks, model_class)(root / config, ch=3, nc=5, verbose=False)
    graph = ComputationGraphBuilder().build(model)
    head_index = len(model.model) - 1
    head_nodes = [node for node in graph.nodes if node.name.startswith(f"model.{head_index}.")]

    assert head_nodes
    assert {node.semantic_role for node in head_nodes} == {"head"}
    assert {node.annotations["head_family"] for node in head_nodes} == {expected_head}
    mask = ConstraintRegistry.default().get_hard_mask(graph, "lora", candidate_ranks=[4, 8])
    assert not any(
        bool(mask[index])
        for index, node in enumerate(graph.nodes)
        if node.name.startswith(f"model.{head_index}.")
    )


def test_vpeft_distinguishes_yolo26_one_to_one_and_shared_backbone():
    from ultralytics.nn.tasks import DetectionModel
    from ultralytics.vpeft import ComputationGraphBuilder

    root = Path(__file__).resolve().parents[1] / "ultralytics/cfg/models/26/yolo26.yaml"
    model = DetectionModel(root, ch=3, nc=5, verbose=False)
    graph = ComputationGraphBuilder().build(model)
    one2one = [node for node in graph.nodes if ".one2one_" in node.name]
    backbone = [node for node in graph.nodes if node.name.startswith("model.0.")]

    assert one2one and {node.annotations["head_branch"] for node in one2one} == {"one2one"}
    assert all(node.annotations["in_head"] and not node.annotations["shared_backbone"] for node in one2one)
    assert backbone and all(node.annotations["shared_backbone"] for node in backbone)
