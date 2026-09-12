from __future__ import annotations

from pathlib import Path

import pytest
import torch

from e3_p1.engine_crosscheck import (
    _assert_pair_equivalence,
    _clear_nonleaf_latent_transients,
    _condition_order,
    _hash_value,
    _load_engine_config,
    _verify_runtime_inputs,
)


def test_engine_order_is_balanced_ab_ba():
    assert _condition_order(0) == ["off", "capture_jsonl"]
    assert _condition_order(1) == ["capture_jsonl", "off"]


def test_engine_config_requires_even_measured_pairs(tmp_path: Path):
    path = tmp_path / "engine.yaml"
    path.write_text(
        "run_id: test\nbatch_size: 2\nimage_size: 64\nepochs: 5\nwarmup_pairs: 0\nmeasured_pairs: 3\nprofiles:\n  mot: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="even"):
        _load_engine_config(path)


def test_recursive_hash_is_order_stable_and_tensor_sensitive():
    first = {"b": [2, torch.tensor([1.0])], "a": 1}
    second = {"a": 1, "b": [2, torch.tensor([1.0])]}
    changed = {"a": 1, "b": [2, torch.tensor([2.0])]}
    assert _hash_value(first) == _hash_value(second)
    assert _hash_value(first) != _hash_value(changed)


def test_extracted_runtime_requires_matching_preregistered_hash(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_bytes(b"stable\n")
    config = {
        "official_runtime_ref": "ref",
        "official_runtime_tree": "tree",
        "runtime_input_sha256": {"source.py": "2b92ea252be0fbc26f70317cdaa7b6411ea634b50d55338cd8c495e4dbf25d1d"},
    }
    result = _verify_runtime_inputs(tmp_path, config)
    assert result["mode"] == "extracted_snapshot_file_hashes"
    source.write_bytes(b"changed\n")
    with pytest.raises(RuntimeError, match="hash differs"):
        _verify_runtime_inputs(tmp_path, config)


def test_runtime_identity_policy_is_explicit(tmp_path: Path):
    source = tmp_path / "source.py"
    source.write_bytes(b"stable\n")
    config = {
        "official_runtime_ref": "ref",
        "official_runtime_tree": "tree",
        "runtime_identity_policy": "guess",
        "runtime_input_sha256": {"source.py": "2b92ea252be0fbc26f70317cdaa7b6411ea634b50d55338cd8c495e4dbf25d1d"},
    }
    with pytest.raises(ValueError, match="runtime_identity_policy"):
        _verify_runtime_inputs(tmp_path, config)


def _row() -> dict:
    return {
        "initial_model_sha256": "a",
        "initial_optimizer_sha256": "b",
        "final_model_sha256": "c",
        "final_optimizer_sha256": "d",
        "final_ema_sha256": "e",
        "optimizer_steps": 2,
        "batch_count": 2,
        "epochs_completed": 1,
        "sanitized_transients": [],
        "batches": [{"raw_batch_sha256": "f"}, {"raw_batch_sha256": "g"}],
        "loss_items": [1.0, 2.0],
    }


def test_pair_equivalence_hard_fails_on_final_state_change():
    off = _row()
    observed = _row()
    _assert_pair_equivalence(off, observed, family="toy", pair_index=0)
    observed["final_model_sha256"] = "changed"
    with pytest.raises(RuntimeError, match="final_model_sha256"):
        _assert_pair_equivalence(off, observed, family="toy", pair_index=0)


class _TransientModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0]))
        self._last_routing_logits = self.weight * 2
        self._last_routing_probs = None
        self._last_routing_summary = None


def test_latent_setup_guard_only_clears_nonleaf_snapshot_fields():
    model = _TransientModule()
    before = dict(model.state_dict())
    records = _clear_nonleaf_latent_transients(model)
    assert records == [
        {
            "module": "",
            "type": "_TransientModule",
            "field": "_last_routing_logits",
            "shape": [1],
            "state_dict_member": False,
        }
    ]
    assert model._last_routing_logits is None
    assert model.state_dict().keys() == before.keys()
    assert torch.equal(model.state_dict()["weight"], before["weight"])
