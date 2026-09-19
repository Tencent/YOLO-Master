from pathlib import Path

import pytest

from scripts.plot_b1_controlled_interventions import (
    read_rows,
    render_controlled_svg,
    render_opportunity_svg,
    validate_conditions,
    validate_contrasts,
    validate_opportunities,
)


DATA_ROOT = Path("reports/text_router_analysis/controlled_interventions")


def test_controlled_intervention_tables_are_complete_and_consistent():
    conditions = validate_conditions(read_rows(DATA_ROOT / "conditions.csv"))
    contrasts = validate_contrasts(read_rows(DATA_ROOT / "paired_bootstrap.csv"), conditions)

    assert len(conditions) == 24
    assert len(contrasts) == 30
    assert [int(conditions[(seed, "T")]["expert0_count"]) for seed in (0, 1, 2)] == [4998, 7, 0]
    assert all(float(contrasts[(seed, "GW-T", "AP")]["point_delta"]) < 0 for seed in (0, 1, 2))
    assert all(float(contrasts[(seed, "E1-E0", "AP")]["point_delta"]) < 0 for seed in (0, 1, 2))


def test_controlled_intervention_svg_has_accessible_metadata():
    conditions = validate_conditions(read_rows(DATA_ROOT / "conditions.csv"))
    contrasts = validate_contrasts(read_rows(DATA_ROOT / "paired_bootstrap.csv"), conditions)

    svg = render_controlled_svg(conditions, contrasts)

    assert 'role="img"' in svg
    assert "Controlled routing interventions across three checkpoints" in svg
    assert "E0 4998 / E1 2" in svg


def test_opportunity_table_is_a_complete_partition_and_renders():
    opportunities = validate_opportunities(read_rows(DATA_ROOT / "opportunity_summary.csv"))

    assert [int(opportunities[seed]["event_count"]) for seed in (0, 1, 2)] == [5000, 5000, 5000]
    assert [int(opportunities[seed]["nontrivial_action_count"]) for seed in (0, 1, 2)] == [555, 570, 584]
    assert float(opportunities[1]["oracle_minus_best_fixed_new17_AP"]) < 0
    svg = render_opportunity_svg(opportunities)
    assert 'role="img"' in svg
    assert "Local expert opportunity and target-metric outcome" in svg


def test_validation_rejects_a_changed_point_delta():
    conditions = validate_conditions(read_rows(DATA_ROOT / "conditions.csv"))
    rows = read_rows(DATA_ROOT / "paired_bootstrap.csv")
    rows[0]["point_delta"] = "1"

    with pytest.raises(ValueError, match="point delta mismatch"):
        validate_contrasts(rows, conditions)
