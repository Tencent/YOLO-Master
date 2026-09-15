"""Contract tests for the YOLO-PEFT paper claim anchor."""

from scripts.reproduce_yolo_peft_paper import ANCHOR_PATH, check_anchor, load_anchor


def test_yolo_peft_anchor_file_exists_and_has_required_claims():
    payload = load_anchor()
    assert payload["arxiv"] == "2608.07051"
    claims = payload["claims"]
    assert claims["yolo11s_planner_rslora_map50_95"] == 0.7138
    assert claims["yolo12s_planner_rslora_map50_95"] == 0.7307
    assert claims["yolo11s_full_sft_map50_95"] == 0.6428
    assert claims["yolo12s_full_sft_map50_95"] == 0.6662
    assert claims["yolo11_peak_memory_reduction_pct"] == 43.9
    assert claims["yolo11_train_time_ratio"] == 1.72
    assert claims["rtdetr_l_lora_family_refuse"] is True
    assert claims["rtdetr_l_evaluated_lora_configs"] == 7
    assert payload["in_repo_entrypoints"]["paper_backend"] == "vpeft"
    assert payload["in_repo_entrypoints"]["default_value"] == "legacy"
    assert ANCHOR_PATH.is_file()


def test_yolo_peft_readme_and_default_yaml_match_anchor():
    failures = check_anchor()
    assert failures == [], failures
