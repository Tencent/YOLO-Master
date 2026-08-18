"""P0 isolated Foundation KD-loss ablation (B0/D0/D1/D2) runner contracts."""

import argparse
import json

import pytest

from scripts.foundation_p0_loss_ablation import (
    ARMS,
    BENCHMARK,
    _aggregate_records,
    _peak_memory_bytes,
    _validate_arms,
    build_run_plan,
    capture_scale_bucket_ap,
    main,
    parse_args,
    run_matrix,
)

DATASET = "/tmp/foundation-p0-coco.yaml"
MODEL = "yolo26-master-n.yaml"
TEACHER = "/tmp/fake-dinov3"


def _plan(**overrides):
    params = {
        "dataset": DATASET,
        "model": MODEL,
        "teacher_model": TEACHER,
        "project": "runs/detect/p0-test",
        "seeds": [7],
        "foundation_loss_weights": [0.01],
        "arms": ["baseline", "cosine", "relational", "hybrid"],
        "task": "detect",
        "epochs": 1,
        "fraction": 0.001,
        "imgsz": 128,
        "batch": 2,
        "device": "cpu",
        "workers": 0,
        "val": True,
        "scale_ap": False,
    }
    params.update(overrides)
    return build_run_plan(**params)


def test_plan_builds_baseline_and_isolated_loss_arms():
    plan = _plan()
    assert [spec["arm"] for spec in plan] == ["baseline", "cosine", "relational", "hybrid"]

    baseline = plan[0]
    assert baseline["foundation"] is False
    assert baseline["overrides"]["foundation_enabled"] is False
    assert baseline["overrides"]["foundation_loss_weight"] == 0.0
    assert baseline["overrides"]["foundation_teacher"] == "none"

    cosine = plan[1]
    assert cosine["overrides"]["foundation_enabled"] is True
    assert cosine["overrides"]["foundation_loss"] == "cosine"
    assert cosine["overrides"]["foundation_cosine_weight"] == 1.0
    assert cosine["overrides"]["foundation_relation_weight"] == 0.0

    relational = plan[2]
    assert relational["overrides"]["foundation_loss"] == "relational"
    assert relational["overrides"]["foundation_cosine_weight"] == 0.0
    assert relational["overrides"]["foundation_relation_weight"] == 1.0

    hybrid = plan[3]
    assert hybrid["overrides"]["foundation_loss"] == "hybrid"
    assert hybrid["overrides"]["foundation_cosine_weight"] == 1.0
    assert hybrid["overrides"]["foundation_relation_weight"] == 1.0

    for spec in plan[1:]:
        assert spec["overrides"]["foundation_teacher"] == "dinov3"
        assert spec["overrides"]["foundation_model"] == TEACHER
        assert spec["overrides"]["foundation_loss_weight"] == 0.01

    contracts = {spec["name"]: spec["initialization_contract"] for spec in plan}
    assert len(set(contracts)) == len(plan)
    for contract in contracts.values():
        assert contract == {
            "pretrained": False,
            "same_model_config": True,
            "same_seed": True,
            "same_dataset_split": True,
        }


def test_plan_multitask_adds_multitask_flags_only_for_foundation_arms():
    plan = _plan(task="multitask", arms=["baseline", "cosine"])
    assert plan[0]["overrides"]["foundation_multitask"] is False
    assert plan[1]["overrides"]["foundation_multitask"] is True
    assert plan[1]["overrides"]["foundation_multitask_tasks"] == ["detect", "segment", "pose"]


def test_validate_arms_requires_baseline_and_known_names():
    assert _validate_arms(["baseline", "cosine"]) == ["baseline", "cosine"]
    assert _validate_arms(["baseline"]) == ["baseline"]
    with pytest.raises(argparse.ArgumentTypeError):
        _validate_arms(["cosine"])
    with pytest.raises(argparse.ArgumentTypeError):
        _validate_arms(["baseline", "l2"])


def test_parse_args_boundaries():
    args = parse_args(["--arms", "baseline,cosine", "--seeds", "1,2", "--foundation-loss-weights", "0.01,0.05"])
    assert args.arms == ["baseline", "cosine"]
    assert args.seeds == [1, 2]
    assert args.foundation_loss_weights == [0.01, 0.05]
    with pytest.raises(SystemExit):
        parse_args(["--arms", "cosine"])
    with pytest.raises(SystemExit):
        parse_args(["--epochs", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--fraction", "0"])
    with pytest.raises(SystemExit):
        parse_args(["--scale-ap", "--task", "multitask"])


def _stub_runner(spec):
    return {
        "name": spec["name"],
        "arm": spec["arm"],
        "foundation": spec["foundation"],
        "seed": spec["seed"],
        "foundation_loss_weight": spec["foundation_loss_weight"],
        "validation_metrics": {"metrics/mAP50-95(B)": 0.5 if spec["arm"] == "baseline" else 0.51},
    }


def test_run_matrix_writes_progress_and_never_claims_accuracy(tmp_path):
    output = tmp_path / "ablation.json"
    plan = _plan()
    report = run_matrix(plan, output, runner=_stub_runner)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["benchmark"] == BENCHMARK
    assert payload["accuracy_claim"] is False
    assert payload["completed_runs"] == len(plan) == 4
    assert report["summary"]["paired_runs"] == 1
    pair = report["summary"]["pairs"][0]
    assert pair["baseline_complete"] is True
    assert set(pair["arms"]) == {"cosine", "relational", "hybrid"}
    for entry in pair["arms"].values():
        assert entry["validation_metric_deltas_vs_baseline"]["metrics/mAP50-95(B)"] == pytest.approx(0.01)


def test_run_matrix_resume_skips_completed_names(tmp_path):
    output = tmp_path / "ablation.json"
    plan = _plan(arms=["baseline", "cosine"])
    seen = []
    run_matrix(plan, output, runner=lambda spec: seen.append(spec["name"]) or _stub_runner(spec))
    assert seen == ["baseline-s7-w0.01", "cosine-s7-w0.01"]
    run_matrix(plan, output, runner=lambda spec: seen.append(spec["name"]) or _stub_runner(spec), resume=True)
    assert seen == ["baseline-s7-w0.01", "cosine-s7-w0.01"]


def test_run_matrix_resume_rejects_incompatible_report(tmp_path):
    output = tmp_path / "ablation.json"
    output.write_text(json.dumps({"benchmark": "other"}), encoding="utf-8")
    with pytest.raises(ValueError, match="incompatible"):
        run_matrix(_plan(), output, runner=_stub_runner, resume=True)


def test_aggregate_records_is_missing_safe():
    summary = _aggregate_records(
        [
            {
                "name": "baseline-s7-w0.01",
                "arm": "baseline",
                "seed": 7,
                "foundation_loss_weight": 0.01,
                "validation_metrics": {"metrics/mAP50(B)": 0.4},
            },
            {
                "name": "cosine-s7-w0.01",
                "arm": "cosine",
                "seed": 7,
                "foundation_loss_weight": 0.01,
                "validation_metrics": {},
            },
        ]
    )
    pair = summary["pairs"][0]
    assert pair["baseline_complete"] is True
    assert pair["arms"]["cosine"]["validation_metric_deltas_vs_baseline"] == {}
    assert summary["validation_pairs_with_metrics"] == 0


def test_peak_memory_bytes_cpu_is_none():
    assert _peak_memory_bytes("cpu") is None


def test_capture_scale_bucket_ap_missing_dataset_is_explicit_null(tmp_path):
    result = capture_scale_bucket_ap(
        tmp_path / "best.pt",
        str(tmp_path / "missing.yaml"),
        device="cpu",
        imgsz=128,
        batch=1,
        project=str(tmp_path),
        name="x",
    )
    assert result["available"] is False
    assert "dataset yaml not found" in result["reason"]
    assert "stats" not in result


def test_main_run_name_filter_and_dry_run(tmp_path, capsys):
    output = tmp_path / "ablation.json"
    exit_code = main(
        [
            "--dataset",
            DATASET,
            "--model",
            MODEL,
            "--teacher-model",
            TEACHER,
            "--arms",
            "baseline,cosine",
            "--seeds",
            "7",
            "--run-name",
            "cosine-s7-w0.01",
            "--dry-run",
            "--output",
            str(output),
        ]
    )
    assert exit_code == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["benchmark"] == BENCHMARK
    assert printed["total_runs"] == 1
    assert printed["plan"][0]["name"] == "cosine-s7-w0.01"
    assert not output.exists()


def test_main_run_name_rejects_unknown_name(tmp_path):
    with pytest.raises(SystemExit, match="not in plan"):
        main(
            [
                "--dataset",
                DATASET,
                "--model",
                MODEL,
                "--teacher-model",
                TEACHER,
                "--arms",
                "baseline",
                "--run-name",
                "cosine-s20260813-w0.01",
                "--dry-run",
                "--output",
                str(tmp_path / "ablation.json"),
            ]
        )


def test_run_matrix_run_name_executes_one_run_but_keeps_full_plan(tmp_path):
    output = tmp_path / "ablation.json"
    plan = _plan(arms=["baseline", "cosine"])
    seen = []
    report = run_matrix(
        plan, output, runner=lambda spec: seen.append(spec["name"]) or _stub_runner(spec), run_name="cosine-s7-w0.01"
    )
    assert seen == ["cosine-s7-w0.01"]
    assert report["completed_runs"] == 1
    assert report["total_runs"] == 2
    assert [spec["name"] for spec in report["plan"]] == ["baseline-s7-w0.01", "cosine-s7-w0.01"]
    with pytest.raises(ValueError, match="not in plan"):
        run_matrix(plan, output, runner=_stub_runner, run_name="hybrid-s7-w0.01")


def test_coco_eval_subset_restricts_to_predicted_images(tmp_path):
    pytest.importorskip("pycocotools")
    gt = {
        "info": {"description": "synthetic"},
        "images": [
            {"id": 1, "width": 8, "height": 8},
            {"id": 2, "width": 8, "height": 8},
            {"id": 3, "width": 8, "height": 8},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0},
            {"id": 3, "image_id": 3, "category_id": 1, "bbox": [0, 0, 4, 4], "area": 16, "iscrowd": 0},
        ],
        "categories": [{"id": 1, "name": "obj"}],
    }
    preds = [
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 4, 4], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [0, 0, 4, 4], "score": 0.8},
    ]
    gt_path = tmp_path / "gt.json"
    pred_path = tmp_path / "preds.json"
    gt_path.write_text(json.dumps(gt), encoding="utf-8")
    pred_path.write_text(json.dumps(preds), encoding="utf-8")

    from scripts.foundation_p0_loss_ablation import COCO_STATS, coco_eval_subset

    result = coco_eval_subset(gt_path, pred_path)
    assert result["available"] is True
    assert result["evaluated_images"] == 2
    assert result["gt_images"] == 3
    assert result["subset_eval"] is True
    assert set(result["stats"]) == set(COCO_STATS)
    # Perfect predictions on the evaluated subset must give perfect AP.
    assert result["stats"]["AP50_95"] == pytest.approx(1.0)
    assert "APs" in result["stats"] and "APm" in result["stats"] and "APl" in result["stats"]

    empty_path = tmp_path / "empty.json"
    empty_path.write_text("[]", encoding="utf-8")
    empty = coco_eval_subset(gt_path, empty_path)
    assert empty["available"] is False
    assert empty["evaluated_images"] == 0


def test_arm_names_match_plan_document():
    assert set(ARMS) == {"baseline", "cosine", "relational", "hybrid"}
