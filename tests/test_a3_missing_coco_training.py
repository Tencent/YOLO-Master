from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import yaml

from scripts.train_a3_missing_coco_weights import (
    MISSING_FAMILIES,
    load_contract,
    materialize_dataset_yaml,
    quick_verify_dataset,
    verify_dataset,
)


ROOT = Path(__file__).resolve().parents[1]
TRAINING_CONFIG = ROOT / "configs" / "a3_missing_coco_weights.yaml"
PRECISION_CONFIG = ROOT / "configs" / "a3_five_family_precision_coco.yaml"


def test_training_and_precision_configs_use_the_same_yolo26_family_matrix(monkeypatch, tmp_path):
    monkeypatch.setenv("A3_COCO_ROOT", str(tmp_path / "coco"))
    monkeypatch.setenv("A3_MOE_WEIGHT", str(tmp_path / "moe.pt"))
    monkeypatch.setenv("A3_INITIALIZER_COCO_WEIGHT", str(tmp_path / "yolo26n.pt"))
    monkeypatch.setenv("A3_TRAIN_OUTPUT_ROOT", str(tmp_path / "training"))
    monkeypatch.setenv("A3_WEIGHT_ROOT", str(tmp_path / "weights"))

    contract = load_contract(TRAINING_CONFIG)
    precision = yaml.safe_load(PRECISION_CONFIG.read_text(encoding="utf-8"))
    precision_configs = {row["family"]: row["config"] for row in precision["models"]}

    assert tuple(contract.families) == MISSING_FAMILIES
    assert precision_configs["moe"] == "${A3_MOE_CONFIG}"
    for family in ("moa", "mot", "latent"):
        expected = (ROOT / precision_configs[family].removeprefix("repo://")).resolve()
        assert contract.families[family].config == expected
    assert contract.families["molora"].trainer_overrides["molora_num_experts"] == 4
    assert contract.families["molora"].trainer_overrides["molora_top_k"] == 2
    assert contract.families["moa"].minimum_transfer_ratio == 0.70
    assert contract.families["mot"].minimum_transfer_ratio == 0.60
    assert contract.families["latent"].minimum_transfer_ratio == 0.25
    assert contract.training["epochs"] == 5
    assert contract.training["batch"] == 4
    assert contract.training["fraction"] == 0.10


def test_dataset_gate_and_materialized_yaml(monkeypatch, tmp_path):
    coco = tmp_path / "coco"
    for relative, count, suffix in (
        ("images/train2017", 3, ".jpg"),
        ("images/val2017", 2, ".jpg"),
        ("labels/train2017", 2, ".txt"),
        ("labels/val2017", 1, ".txt"),
    ):
        directory = coco / relative
        directory.mkdir(parents=True)
        for index in range(count):
            (directory / f"{index:012d}{suffix}").touch()

    monkeypatch.setenv("A3_COCO_ROOT", str(coco))
    monkeypatch.setenv("A3_MOE_WEIGHT", str(tmp_path / "moe.pt"))
    monkeypatch.setenv("A3_INITIALIZER_COCO_WEIGHT", str(tmp_path / "yolo26n.pt"))
    monkeypatch.setenv("A3_TRAIN_OUTPUT_ROOT", str(tmp_path / "training"))
    monkeypatch.setenv("A3_WEIGHT_ROOT", str(tmp_path / "weights"))
    contract = load_contract(TRAINING_CONFIG)
    contract = replace(
        contract,
        dataset={
            "expected_train_images": 3,
            "expected_val_images": 2,
            "minimum_train_labels": 2,
            "minimum_val_labels": 1,
        },
    )

    evidence = verify_dataset(contract)
    assert evidence["train_images"] == 3
    assert evidence["val_images"] == 2

    quick = quick_verify_dataset(contract)
    assert quick["mode"] == "quick"

    generated = yaml.safe_load(materialize_dataset_yaml(contract).read_text(encoding="utf-8"))
    assert generated["path"] == str(coco.resolve())
    assert generated["train"] == "images/train2017"
    assert generated["val"] == "images/val2017"
    assert len(generated["names"]) == 80
