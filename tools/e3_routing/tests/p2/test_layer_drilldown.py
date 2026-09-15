import json
from pathlib import Path

import numpy as np
import pytest

from e3_p2.io_utils import sha256_file, write_manifest
from e3_p2.layer_drilldown import margin_bin_summary, summarize_layer_attribution, verify_parent_evidence


def _comparison(transform: str, module: str, seed: int, value: float) -> dict:
    return {
        "family": "moa",
        "comparison_type": f"appearance_{transform}",
        "module": module,
        "seed": seed,
        "metrics": {"probability_mae": value},
    }


def test_parent_verification_rejects_mutation_and_extra_files(tmp_path: Path):
    (tmp_path / "payload.txt").write_text("locked\n", encoding="utf-8")
    manifest = write_manifest(tmp_path)
    digest = sha256_file(manifest)
    result = verify_parent_evidence(tmp_path, digest)
    assert result["status"] == "PASS"
    assert result["verified_file_count"] == 1
    (tmp_path / "payload.txt").write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="integrity mismatch"):
        verify_parent_evidence(tmp_path, digest)
    (tmp_path / "payload.txt").write_text("locked\n", encoding="utf-8")
    (tmp_path / "extra.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="file set mismatch"):
        verify_parent_evidence(tmp_path, digest)


def test_layer_attribution_ranks_target_and_normalizes_shares():
    records = []
    for transform in ("dim", "bright"):
        for seed in (0, 1):
            records.extend(
                [
                    _comparison(transform, "target", seed, 4.0 + seed),
                    _comparison(transform, "other", seed, 1.0),
                ]
            )
    result = summarize_layer_attribution(records, "moa", ["dim", "bright"], "target")
    assert result["target_rank_one_transform_seed_count"] == 4
    assert result["transform_seed_group_count"] == 4
    for item in result["by_transform"].values():
        assert item["target_rank"] == 1
        assert sum(entry["share_of_layer_mean_mae_sum"] for entry in item["ranking"]) == pytest.approx(1.0)


def test_margin_bins_partition_tokens_once_and_localize_switches():
    margins = np.arange(100, dtype=np.float64)
    switched = margins < 10
    result = margin_bin_summary(margins, switched, 10)
    assert sum(item["token_comparison_count"] for item in result["bins"]) == 100
    assert sum(item["switch_count"] for item in result["bins"]) == 10
    assert result["bins"][0]["switch_fraction"] == 1.0
    assert all(item["switch_fraction"] == 0.0 for item in result["bins"][1:])


def test_parent_manifest_digest_is_bound(tmp_path: Path):
    (tmp_path / "payload.json").write_text(json.dumps({"value": 1}), encoding="utf-8")
    write_manifest(tmp_path)
    with pytest.raises(RuntimeError, match="manifest digest mismatch"):
        verify_parent_evidence(tmp_path, "0" * 64)
