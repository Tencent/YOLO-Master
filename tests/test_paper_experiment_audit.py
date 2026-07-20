"""Tests for submission-evidence gates without launching training."""

from scripts.paper_experiment_audit import audit
from scripts.planner_mps_coco128_calibration import ModelSpec, _declared_family, build_matrix
from pathlib import Path
from types import SimpleNamespace


def _record(model, family, variant, seed, dataset="voc", targets=None, placement=None):
    record = {
        "experiment_id": f"{model}_{variant}_{seed}", "model_name": model, "architecture_family": family,
        "variant": variant, "seed": seed, "dataset": dataset, "epochs": 300, "imgsz": 640, "batch": 16,
        "optimizer": "AdamW", "lr0": 0.001, "lrf": 0.01, "weight_decay": 0.0005, "amp": True,
        "status": "success", "metrics": {"metrics/mAP50-95(B)": 0.5 if variant == "full" else 0.55},
        "lora_runtime_metadata": {"target_modules": targets or []}, "target_set": placement,
        "fingerprint": {"phi_attn": 0.45 if family == "yolo12" else 0.0}, "_source": "test",
    }
    return record


def test_smoke_only_evidence_is_rejected():
    report = audit([_record("yolo11n", "yolo_cnn", "full", 0), _record("yolo11n", "yolo_cnn", "lora", 0)])
    assert not report["gates"]["submission_grade"]
    assert not report["gates"]["three_seed_ready"]


def test_submission_gate_requires_explicit_placement_pairs():
    records = []
    for family, model in (("yolo_cnn", "yolo11n"), ("yolo12", "yolo12n"), ("rtdetr", "rtdetr")):
        for seed in range(3):
            records.append(_record(model, family, "full", seed, "voc"))
            records.append(_record(model, family, "full", seed, "coco"))
            records.append(_record(model, family, "lora", seed, "voc", ["manual"]))
            records.append(_record(model, family, "dora", seed, "coco", ["planner"]))
            records.append(_record(model, family, "loha", seed, "coco", ["planner"]))
    report = audit(records)
    # This intentionally incomplete fixture has no full variant/placement
    # cells for each architecture, so strict LOAO must remain closed.
    assert not report["gates"]["leave_one_architecture_ready"]
    assert report["gates"]["cross_dataset_ready"]
    assert not report["gates"]["controlled_placement_ready"]


def test_audit_rejects_placement_when_training_controls_differ():
    records = []
    for family, model in (("yolo_cnn", "yolo11n"), ("yolo12", "yolo12n"), ("rtdetr", "rtdetr")):
        for seed in range(3):
            for dataset in ("voc", "coco"):
                records.append(_record(model, family, "full", seed, dataset))
                records.append(_record(model, family, "lora", seed, dataset, ["a"], "manual"))
                changed = _record(model, family, "lora", seed, dataset, ["b"], "planner")
                changed["lora_dropout"] = 0.2
                records.append(changed)
    report = audit(records)
    assert not report["gates"]["controlled_placement_ready"]
    assert report["placement_mismatches"]


def test_audit_accepts_only_equal_controlled_placement_pairs():
    records = []
    for family, model in (("yolo_cnn", "yolo11n"), ("yolo12", "yolo12n"), ("rtdetr", "rtdetr")):
        for seed in range(3):
            for dataset in ("voc", "coco"):
                records.append(_record(model, family, "full", seed, dataset))
                for variant in ("lora", "dora", "loha"):
                    for placement, target in (("manual", "a"), ("planner", "b")):
                        records.append(_record(model, family, variant, seed, dataset, [target], placement))
    report = audit(records)
    assert report["gates"]["leave_one_architecture_ready"]
    assert report["gates"]["leave_one_variant_ready"]
    assert report["gates"]["cross_dataset_ready"]
    assert report["gates"]["three_seed_ready"]
    assert report["gates"]["controlled_placement_ready"]
    assert report["gates"]["submission_grade"]


def test_formal_matrix_has_two_datasets_three_seeds_and_paired_placements():
    args = SimpleNamespace(
        dataset_specs=[("voc", Path("VOC.yaml")), ("coco", Path("coco.yaml"))],
        data=None,
        seeds=[0, 1, 2],
        placements="planner,manual",
        ranks=[8],
        epochs=300,
        imgsz=640,
        batch=16,
        smoke=False,
    )
    model = ModelSpec("rtdetr-l", Path("rtdetr-l.pt"), "sha")
    matrix = build_matrix([model], args)
    assert {item.dataset_name for item in matrix} == {"voc", "coco"}
    assert {item.seed for item in matrix} == {0, 1, 2}
    assert {item.placement for item in matrix} == {"full", "planner", "manual"}
    assert {item.variant for item in matrix} >= {"full", "lora", "dora", "loha"}


def test_family_hint_does_not_call_cnn_models_other_families():
    assert _declared_family("rtdetr-l") == "rtdetr"
    assert _declared_family("yolo-world") == "yolo_world"
    assert _declared_family("yolo12s") == "yolo12"
