"""Capacity-guard contracts for the V-PEFT planner and placement-plan validation.

Narrow layers (``min(in_channels, out_channels)`` below the solver's minimum
candidate rank) used to be projected as feasible targets and only fail later in
``PlacementPlan.validate_model``, which silently demoted the run to the legacy
planner. These tests pin the shared capacity semantics: the solver skips narrow
layers, the generated plan records them, and external plans are still rejected.
"""

import pytest
import torch.nn as nn

from ultralytics.utils.lora.api import _build_vpeft_placement_plan, apply_lora
from ultralytics.utils.lora.config import LoRAConfig
from ultralytics.vpeft import (
    AlternatingOptimizationSolver,
    ComputationGraph,
    ComputationGraphBuilder,
    ConstraintRegistry,
    ModuleNode,
    NodeInfo,
    PlacementPlan,
    PlacementTarget,
    RankCapacityConstraint,
)


def _narrow_model() -> nn.Sequential:
    """Model with narrow layers whose capacity is below the minimum rank (4)."""
    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),  # "0" stem: capacity 3
        nn.Conv2d(8, 16, 3, padding=1),  # "1": capacity 8
        nn.Conv2d(16, 4, 1),  # "2": capacity 4 (rank 4 is still a valid factor)
        nn.Conv2d(4, 3, 1),  # "3": capacity 3
    )


def test_capacity_constraint_mirrors_plan_validation():
    guard = RankCapacityConstraint()

    wide = NodeInfo(ModuleNode("backbone.conv", "Conv2d", 16, 4))
    assert guard.capacity(wide) == 4
    assert guard.is_feasible(wide, "lora", 4)  # rank == capacity is allowed
    assert not guard.is_feasible(wide, "lora", 8)

    narrow = NodeInfo(ModuleNode("routing.routing_network.2", "Conv2d", 8, 3))
    assert guard.capacity(narrow) == 3
    assert not guard.is_feasible(narrow, "lora", 4)
    assert guard.is_feasible(narrow, "lora", 2)


def test_capacity_constraint_ignores_unknown_dimensions_and_other_operators():
    guard = RankCapacityConstraint()

    # Synthetic graphs without channel metadata must not be dropped.
    assert guard.is_feasible(NodeInfo(ModuleNode("unknown", "Conv2d", 0, 0)), "lora", 8)
    # Operators that cannot host an adapter stay the responsibility of C_op.
    assert guard.is_feasible(NodeInfo(ModuleNode("bn", "BatchNorm2d", 3, 3)), "lora", 8)


def test_registry_registers_capacity_guard_as_hard_constraint():
    registry = ConstraintRegistry.default()
    assert registry.hard_constraint_names()[-1] == "C_cap"

    graph = ComputationGraph(modules=[ModuleNode("stem", "Conv2d", 4, 3)])
    assert graph.n_nodes == 1
    assert not registry.is_rank_feasible(graph, 0, "lora", 4)
    assert registry.is_rank_feasible(graph, 0, "lora", 2)

    # The legacy string form resolves to the same canonical constraint.
    legacy = ConstraintRegistry(hard_constraints=["cap"])
    assert legacy.hard_constraint_names() == ["C_cap"]
    assert not legacy.is_rank_feasible(graph, 0, "lora", 4)


def test_solver_skips_narrow_layers_instead_of_projecting_them():
    graph = ComputationGraphBuilder().build(_narrow_model())
    constraints = ConstraintRegistry.default({"max_params": 1_000_000})

    assert constraints.get_hard_mask(graph, "lora", candidate_ranks=[4, 8]).tolist() == [False, True, True, False]

    decision = AlternatingOptimizationSolver(max_iter=2, rank_min=4, rank_max=4, rank_step=4).solve(
        graph, budget=1_000_000, variant="lora", constraints=constraints
    )

    assert decision.status == "ACCEPT"
    assert set(decision.target_modules) == {"1", "2"}
    assert all(
        constraints.is_rank_feasible(graph, index, "lora", int(decision.ranks[index].item()))
        for index, name in enumerate(graph.get_module_names())
        if name in decision.target_modules
    )


def test_generated_plan_audits_capacity_excluded_layers():
    model = _narrow_model()
    plan = _build_vpeft_placement_plan(
        model,
        LoRAConfig(r=4, alpha=8, backend="fallback", planner_backend="vpeft", adapter_budget=1_000_000),
    )

    assert plan.status == "ACCEPT"
    assert {target.name for target in plan.targets} == {"1", "2"}
    assert plan.metadata["capacity_excluded"] == [
        {"name": "0", "rank": 4, "capacity": 3},
        {"name": "3", "rank": 4, "capacity": 3},
    ]
    # The generated plan is consumable: before the guard this raised for "0"/"3".
    plan.validate_model(model)


def test_apply_lora_with_vpeft_keeps_plan_instead_of_falling_back():
    wrapped = apply_lora(
        _narrow_model(),
        LoRAConfig(r=4, alpha=8, backend="fallback", planner_backend="vpeft", adapter_budget=1_000_000),
    )

    plan = wrapped.lora_placement_plan
    assert plan["status"] == "ACCEPT"
    assert set(wrapped.lora_target_modules) == {"1", "2"}
    assert [item["name"] for item in plan["metadata"]["capacity_excluded"]] == ["0", "3"]
    assert wrapped.lora_runtime_metadata.get("vpeft_fallback") is None


def test_validate_model_reports_every_capacity_violation():
    model = _narrow_model()
    plan = PlacementPlan(
        model_fingerprint="",
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 0},
        targets=(PlacementTarget("0", "lora", 4), PlacementTarget("3", "lora", 4)),
        status="ACCEPT",
    )

    with pytest.raises(ValueError) as excinfo:
        plan.validate_model(model)

    message = str(excinfo.value)
    assert "exceeds layer capacity" in message
    assert "'0' (rank=4 > capacity=3)" in message
    assert "'3' (rank=4 > capacity=3)" in message
    assert "capacity_excluded" in message


def test_capacity_boundary_is_shared_by_guard_and_plan_validation():
    model = nn.Sequential(nn.Conv2d(16, 4, 1))
    graph = ComputationGraphBuilder().build(model)

    assert ConstraintRegistry.default().is_rank_feasible(graph, 0, "lora", 4)

    plan = PlacementPlan(
        model_fingerprint="",
        planner_backend="vpeft",
        solver="ao",
        budget={"max_adapter_params": 0},
        targets=(PlacementTarget("0", "lora", 4),),
        status="ACCEPT",
    )
    plan.validate_model(model)
