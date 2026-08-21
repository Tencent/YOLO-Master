import csv

from scripts.compare_mot_ablation import LOSS_KEYS, row_total_loss, stability_from_results

HEADER = (
    "epoch,train/box_loss,train/cls_loss,train/dfl_loss,train/mixture_aux_loss,"
    "metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,val/mixture_aux_loss"
)


def _write_results(path, rows):
    path.write_text(HEADER + "\n" + "\n".join(rows) + "\n")
    return path


def test_loss_keys_track_trainer_mixture_aux_column():
    """The fused column the trainer actually emits must be summed, not the legacy per-mixture names."""
    assert "train/mixture_aux_loss" in LOSS_KEYS


def test_row_total_loss_includes_mixture_aux():
    row = {
        "train/box_loss": "1.0",
        "train/cls_loss": "2.0",
        "train/dfl_loss": "3.0",
        "train/mixture_aux_loss": "1.5",
    }
    assert row_total_loss(row) == 7.5


def test_row_total_loss_reads_legacy_per_mixture_columns():
    """Pre-fusion runs wrote train/moe_loss; those files must still total correctly."""
    row = {
        "train/box_loss": "1.0",
        "train/cls_loss": "2.0",
        "train/dfl_loss": "3.0",
        "train/moe_loss": "0.5",
    }
    assert row_total_loss(row) == 6.5


def test_stability_totals_account_for_aux_loss(tmp_path):
    results = _write_results(
        tmp_path / "results.csv",
        [
            "1,5.0,4.0,3.0,2.0,0.1,0.05,5.1,4.1,3.1,0.0",
            "2,1.0,1.0,1.0,2.0,0.3,0.15,1.1,1.1,1.1,0.0",
        ],
    )
    stability = stability_from_results(results)

    # 1+1+1+2 = 5.0, not 3.0 as it would be with the aux term dropped.
    assert stability["final_train_total_loss"] == "5.000000"
    assert stability["best_train_total_loss"] == "5.000000"
    assert stability["nan_detected"] == "False"
    assert stability["loss_diverged"] == "False"


def test_stability_flags_nonfinite_loss(tmp_path):
    results = _write_results(
        tmp_path / "results.csv",
        [
            "1,1.0,1.0,1.0,2.0,0.1,0.05,1.1,1.1,1.1,0.0",
            "2,nan,1.0,1.0,2.0,0.1,0.05,1.1,1.1,1.1,0.0",
        ],
    )
    stability = stability_from_results(results)

    assert stability["nan_detected"] == "True"
    assert stability["loss_diverged"] == "True"


def test_stability_detects_divergence(tmp_path):
    rows = ["1,1.0,1.0,1.0,0.0,0.1,0.05,1.1,1.1,1.1,0.0"]
    rows += [f"{i},20.0,20.0,20.0,0.0,0.1,0.05,1.1,1.1,1.1,0.0" for i in range(2, 8)]
    stability = stability_from_results(_write_results(tmp_path / "results.csv", rows))

    assert stability["loss_diverged"] == "True"
    assert stability["nan_detected"] == "False"


def test_summary_metric_keys_are_populated_from_results(tmp_path):
    """METRIC_KEYS must name real columns so summary.csv is not silently blank."""
    from scripts.compare_mot_ablation import METRIC_KEYS

    results = _write_results(
        tmp_path / "results.csv",
        ["1,1.0,1.0,1.0,2.0,0.3,0.15,1.1,1.1,1.1,0.0"],
    )
    with results.open(newline="") as f:
        row = next(csv.DictReader(f))

    for key in ("train/mixture_aux_loss", "val/mixture_aux_loss", "metrics/mAP50-95(B)"):
        assert key in METRIC_KEYS
        assert row.get(key) not in (None, "")
