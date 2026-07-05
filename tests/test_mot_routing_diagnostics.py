import torch

from scripts.diagnose_mot_routing import scenario_recommendations, summarize_router_weights


def test_summarize_router_weights_counts_sparse_tokens():
    weights = torch.zeros(1, 3, 2, 2)
    weights[:, 0, :, :] = 0.7
    weights[:, 1, 0, :] = 0.3
    rows = summarize_router_weights("mot.0", weights)

    assert [row.expert for row in rows] == ["LocalConvTransformer", "WindowTransformer", "DeformableTransformer"]
    assert rows[0].active_tokens == 4
    assert rows[1].active_tokens == 2
    assert rows[2].active_tokens == 0
    assert rows[0].activation_ratio == 1.0


def test_scenario_recommendations_are_data_backed():
    weights = torch.ones(1, 3, 2, 2) / 3
    rows = summarize_router_weights("mot.0", weights)
    recs = scenario_recommendations(rows)
    assert len(recs) == 3
    assert all(any(char.isdigit() for char in rec) for rec in recs)
