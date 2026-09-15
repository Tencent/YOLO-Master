import pytest

from e3_p2.output_coupling_runner import (
    DETECTOR_TENSORS,
    analyze_output_coupling,
    build_output_coupling_records,
)


def _route(transform: str, sample: int, seed: int) -> dict:
    return {
        "transform": transform,
        "sample_index": sample,
        "seed": seed,
        "probability_mae": (sample + 1) * (seed + 1) / 100.0,
        "dominant_switch_fraction": (sample + 1) * (seed + 1) / 50.0,
    }


def _detector(transform: str, sample: int, seed: int) -> dict:
    tensors = {
        tensor: {"mean_absolute_change": (sample + 1) * (seed + 1) / (index + 2)}
        for index, tensor in enumerate(DETECTOR_TENSORS)
    }
    return {
        "transform": transform,
        "sample_index": sample,
        "seed": seed,
        "metrics": {"primary_tensor": "one2one_scores", "tensors": tensors},
    }


def test_build_and_analyze_output_coupling_matrix():
    transforms, seeds = ["dim", "blur"], [0, 1, 2]
    routes = [_route(t, image, seed) for t in transforms for image in range(6) for seed in seeds]
    detector = [_detector(t, image, seed) for t in transforms for image in range(6) for seed in seeds]
    records = build_output_coupling_records(routes, detector, transforms, 6, seeds)
    assert len(records) == 12
    assert records[2]["seed_count"] == 3
    result = analyze_output_coupling(records, transforms, 1000, 42)
    observed = result["by_transform"]["dim"]["detector_tensors"]["one2one_scores"][
        "route_endpoints"
    ]["probability_mae"]["bootstrap"]["observed_spearman_rho"]
    assert observed == pytest.approx(1.0)
    assert result == analyze_output_coupling(records, transforms, 1000, 42)


def test_build_output_coupling_rejects_incomplete_detector_matrix():
    transforms, seeds = ["dim"], [0, 1, 2]
    routes = [_route("dim", image, seed) for image in range(4) for seed in seeds]
    detector = [_detector("dim", image, seed) for image in range(4) for seed in seeds]
    with pytest.raises(ValueError, match="incomplete or duplicate detector seed matrix"):
        build_output_coupling_records(routes, detector[:-1], transforms, 4, seeds)


def test_constant_detector_endpoint_is_preserved_as_undefined():
    transforms, seeds = ["dim"], [0, 1, 2]
    routes = [_route("dim", image, seed) for image in range(6) for seed in seeds]
    detector = [_detector("dim", image, seed) for image in range(6) for seed in seeds]
    for item in detector:
        item["metrics"]["tensors"]["one2one_scores"]["mean_absolute_change"] = 0.0
    records = build_output_coupling_records(routes, detector, transforms, 6, seeds)
    result = analyze_output_coupling(records, transforms, 1000, 19)
    endpoint = result["by_transform"]["dim"]["detector_tensors"]["one2one_scores"][
        "route_endpoints"
    ]["probability_mae"]
    assert endpoint["status"] == "UNDEFINED_CONSTANT_VECTOR"
    assert endpoint["observed_spearman_rho"] is None
    assert endpoint["y_unique_value_count"] == 1
    assert endpoint["bootstrap"] is None
