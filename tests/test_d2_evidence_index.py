# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Offline integrity contracts for the compact D2 evidence index; no downloads or metric recomputation."""

import hashlib
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "experiments/rhino_d2/EVIDENCE_INDEX.json"
PUBLIC_COMMIT = "fedc360e73cdbfeaed930b17365458f632b1abc8"
REPOSITORY = "https://github.com/gao-666/YOLO-Master"
RESULT_DIR = "experiments/rhino_d2/results/p2_normalized_response_formal/"
EXPECTED_FILES = {
    "paired_results.csv": "paired_detection_csv",
    "paired_results.json": "paired_detection_json",
    "mechanism_summary.csv": "mechanism_summary",
    "robustness_summary.csv": "robustness_summary",
}


def read_index():
    """Read the index without accessing Git, a model, or the network."""
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))


def test_evidence_identity_contract():
    """Keep private freeze identities distinct from the independently published derivative."""
    index = read_index()
    assert index["schema_version"] == "d2-public-evidence-index-v1"
    assert index["integration_base"] == "d61cf5d87a4db8bd184c2241726778117ac7bddf"
    assert index["scientific_identity"] == {
        "scientific_anchor": "e3e11d0cfebcd0893f5cd5bae46cf7f48ba8429a",
        "packaging_anchor": "781a6f148a46d59a89ea40806de8e10eb7b98941",
        "published": False,
    }
    public = index["public_evidence"]
    assert public["type"] == "sanitized_derivative"
    assert public["commit"] == PUBLIC_COMMIT
    assert public["history_inherited"] is False
    assert public["repository"] == REPOSITORY
    assert public["commit_url"] == f"{REPOSITORY}/commit/{PUBLIC_COMMIT}"
    assert public["provenance_url"] == f"{REPOSITORY}/blob/{PUBLIC_COMMIT}/PROVENANCE.json"
    assert index["scientific_results_regenerated"] is False
    assert isinstance(index["hash_contract"], str) and index["hash_contract"]
    files = index["final_results"]["files"]
    assert len(files) == 4
    assert {entry["local_path"] for entry in files} == {RESULT_DIR + name for name in EXPECTED_FILES}


@pytest.mark.parametrize("name", sorted(EXPECTED_FILES))
def test_protected_table_bytes(name):
    """Verify exact bytes and recorded source/public hash equality, including original line endings."""
    entries = read_index()["final_results"]["files"]
    entry = next(item for item in entries if item["local_path"] == RESULT_DIR + name)
    assert entry["logical_role"] == EXPECTED_FILES[name]
    assert entry["byte_identical_to_frozen_source"] is True
    assert entry["byte_preservation_required"] is True
    assert re.fullmatch(r"[0-9a-f]{64}", entry["sha256"])
    assert entry["sha256"] == entry["source_sha256"] == entry["public_sha256"]
    content = (ROOT / entry["local_path"]).read_bytes()
    assert type(entry["bytes"]) is int and entry["bytes"] == len(content)
    assert hashlib.sha256(content).hexdigest() == entry["sha256"]
    assert entry["public_url"] == f"{REPOSITORY}/blob/{PUBLIC_COMMIT}/{RESULT_DIR}{name}"


def test_terminal_and_recovery_contract():
    """Check recorded decisions without recalculating scientific statistics."""
    index = read_index()
    result = json.loads((ROOT / RESULT_DIR / "paired_results.json").read_text(encoding="utf-8"))
    final = index["final_results"]
    assert final["primary_decision"] == result["status"] == "No detectable change"
    assert final["metric"] == result["metric"] == "late10_median_mAP50-95"
    assert final["route_terminal"] is result["route_terminal"] is True
    assert final["new_experiments_allowed"] is result["new_experiments_allowed"] is False
    recovery = index["recovery_disclosure"]
    assert recovery["b24_manual_interruption"] is True
    assert recovery["retained_analysis_uses_fresh_epoch0_restart"] is True
    path = "experiments/rhino_d2/P2_07_RECOVERY_DISCLOSURE.md"
    assert recovery["local_path"] == path and (ROOT / path).is_file()
    assert recovery["public_url"] == f"{REPOSITORY}/blob/{PUBLIC_COMMIT}/{path}"
    assert recovery["limitation"]
