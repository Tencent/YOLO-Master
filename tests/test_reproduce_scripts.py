from __future__ import annotations

import csv
import importlib.util
import subprocess
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMMON = ROOT / "scripts/reproduce/reproduce_common.py"


def load_common():
    spec = importlib.util.spec_from_file_location("reproduce_common", COMMON)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_dataset_specs_use_project_builtin_yamls() -> None:
    common = load_common()

    visdrone = common.dataset_spec("visdrone")
    sku110k = common.dataset_spec("sku110k")

    assert visdrone.data_yaml == ROOT / "ultralytics/cfg/datasets/VisDrone.yaml"
    assert sku110k.data_yaml == ROOT / "ultralytics/cfg/datasets/SKU-110K.yaml"
    assert visdrone.default_project == ROOT / "runs/issue49_visdrone"
    assert sku110k.default_project == ROOT / "runs/issue49_sku110k"


def test_issue_model_specs_resolve_expected_configs() -> None:
    common = load_common()

    specs = common.discover_model_specs(["all"])

    assert [spec.key for spec in specs] == ["esmoe_n", "v0_1_n"]
    assert specs[0].cfg == ROOT / "ultralytics/cfg/models/master/v0/det/yolo-master-n.yaml"
    assert specs[1].cfg == ROOT / "ultralytics/cfg/models/master/v0_1/det/yolo-master-n.yaml"


def test_collect_summary_reads_required_metrics(tmp_path: Path) -> None:
    common = load_common()
    specs = common.discover_model_specs(["esmoe_n"])
    run_dir = tmp_path / "esmoe_n_e3"
    run_dir.mkdir(parents=True)
    results_csv = run_dir / "results.csv"
    with results_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "train/box_loss",
                "train/cls_loss",
                "train/moe_loss",
                "val/box_loss",
                "val/cls_loss",
                "val/moe_loss",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "2",
                "metrics/mAP50(B)": "0.55",
                "metrics/mAP50-95(B)": "0.31",
                "train/box_loss": "1.2",
                "train/cls_loss": "0.7",
                "train/moe_loss": "0.05",
                "val/box_loss": "1.4",
                "val/cls_loss": "0.8",
                "val/moe_loss": "0.06",
            }
        )

    rows = common.collect_summary(
        dataset_name="VisDrone",
        data_yaml=ROOT / "ultralytics/cfg/datasets/VisDrone.yaml",
        project=tmp_path,
        specs=specs,
        run_suffix="e3",
    )

    expected_run_dir = str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir)
    assert rows[0]["dataset"] == "VisDrone"
    assert rows[0]["model"] == "esmoe_n"
    assert rows[0]["run_dir"] == expected_run_dir
    assert rows[0]["epoch"] == "2"
    assert rows[0]["metrics/mAP50(B)"] == "0.55"
    assert rows[0]["train/moe_loss"] == "0.05"


def test_write_summary_also_writes_markdown_table(tmp_path: Path) -> None:
    common = load_common()
    specs = common.discover_model_specs(["esmoe_n"])
    run_dir = tmp_path / "esmoe_n_e3"
    run_dir.mkdir(parents=True)
    with (run_dir / "results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "epoch",
                "metrics/mAP50(B)",
                "metrics/mAP50-95(B)",
                "train/box_loss",
                "train/cls_loss",
                "train/moe_loss",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "epoch": "3",
                "metrics/mAP50(B)": "0.44",
                "metrics/mAP50-95(B)": "0.22",
                "train/box_loss": "1.1",
                "train/cls_loss": "0.6",
                "train/moe_loss": "0.03",
            }
        )

    common.write_summary(
        dataset_name="VisDrone",
        data_yaml=ROOT / "ultralytics/cfg/datasets/VisDrone.yaml",
        project=tmp_path,
        specs=specs,
        run_suffix="e3",
        epochs_requested=3,
        imgsz=640,
        batch=8,
        seed=42,
    )

    table = (tmp_path / "summary.md").read_text(encoding="utf-8")
    assert "| dataset | model | model_name |" in table
    assert "| VisDrone | esmoe_n | YOLO-Master-EsMoE-N |" in table
    assert "0.44" in table
    assert "0.03" in table


def test_run_reproduce_dry_run_writes_raw_log(tmp_path: Path) -> None:
    log_file = tmp_path / "dry_run.log"

    result = subprocess.run(
        [
            "bash",
            "scripts/run_reproduce.sh",
            "--dry-run",
            "--dataset",
            "visdrone",
            "--model",
            "esmoe_n",
            "--epochs",
            "1",
            "--suffix",
            "pytest",
            "--wandb-mode",
            "disabled",
            "--log-file",
            str(log_file),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    log_text = log_file.read_text(encoding="utf-8")
    assert "[run_reproduce] command:" in log_text
    assert "[dataset] VisDrone" in log_text
    assert str(log_file) in result.stdout


def test_validator_imports_final_eval_helpers() -> None:
    from ultralytics.engine import validator

    assert hasattr(validator, "LOCAL_RANK")
    assert hasattr(validator, "convert_ndjson_to_yolo_if_needed")
    assert hasattr(validator, "torch_distributed_zero_first")


def test_convert_ndjson_helper_keeps_yaml_and_converts_ndjson(monkeypatch, tmp_path: Path) -> None:
    from ultralytics.data.utils import convert_ndjson_to_yolo_if_needed

    yaml_path = tmp_path / "data.yaml"
    ndjson_path = tmp_path / "data.ndjson"
    converted_yaml = tmp_path / "converted" / "data.yaml"
    calls = []

    async def fake_convert(path):
        calls.append(path)
        return converted_yaml

    fake_converter = types.SimpleNamespace(convert_ndjson_to_yolo=fake_convert)
    monkeypatch.setitem(sys.modules, "ultralytics.data.converter", fake_converter)

    assert convert_ndjson_to_yolo_if_needed(yaml_path) == yaml_path
    assert convert_ndjson_to_yolo_if_needed(ndjson_path) == str(converted_yaml)
    assert calls == [ndjson_path]
