# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import numpy as np
import pytest

from ultralytics import YOLO
from ultralytics.utils import ASSETS


@pytest.fixture(scope="module")
def model():
    """Load a lightweight detection model for Sparse SAHI tests."""
    return YOLO("yolov8n.pt")


def test_sparse_sahi_disabled_has_no_metadata(model):
    """Standard predict (sparse_sahi=False) must not attach sparse_sahi_metadata."""
    results = model.predict(source=ASSETS / "bus.jpg", imgsz=640, verbose=False, sparse_sahi=False)
    assert not hasattr(results[0], "sparse_sahi_metadata"), "metadata should not exist when sparse_sahi=False"


def test_sparse_sahi_enabled_attaches_metadata(model):
    """sparse_sahi=True must attach sparse_sahi_metadata with expected keys."""
    results = model.predict(
        source=ASSETS / "bus.jpg",
        imgsz=640,
        verbose=False,
        sparse_sahi=True,
        slice_size=640,
        overlap_ratio=0.2,
        objectness_threshold=0.15,
    )
    meta = getattr(results[0], "sparse_sahi_metadata", None)
    assert meta is not None, "sparse_sahi_metadata must be present when sparse_sahi=True"
    assert "objectness_map" in meta
    assert "slices" in meta
    assert "final_sources" in meta
    assert isinstance(meta["objectness_map"], np.ndarray)
    assert meta["objectness_map"].ndim == 2


def test_sparse_sahi_high_threshold_deactivates_slices(model):
    """With objectness_threshold=0.99, no slices should be activated (only global boxes)."""
    results = model.predict(
        source=ASSETS / "bus.jpg",
        imgsz=640,
        verbose=False,
        sparse_sahi=True,
        objectness_threshold=0.99,
    )
    meta = results[0].sparse_sahi_metadata
    assert len(meta["slices"]) == 0, "high threshold should deactivate all slices"
    assert all(s == 0 for s in meta["final_sources"]), "all boxes should come from global pass"


def test_sparse_sahi_low_threshold_activates_slices(model):
    """With low threshold and small slice_size, multiple slices should be activated."""
    results = model.predict(
        source=ASSETS / "bus.jpg",
        imgsz=640,
        verbose=False,
        sparse_sahi=True,
        slice_size=320,
        overlap_ratio=0.1,
        objectness_threshold=0.1,
    )
    meta = results[0].sparse_sahi_metadata
    assert len(meta["slices"]) > 0, "low threshold should activate slices"
    assert len(meta["final_sources"]) == len(results[0].boxes)


def test_sparse_sahi_box_count_differs_from_standard(model):
    """Sparse SAHI should typically produce different box count than standard inference."""
    r_std = model.predict(source=ASSETS / "bus.jpg", imgsz=640, verbose=False, sparse_sahi=False)
    r_sp = model.predict(
        source=ASSETS / "bus.jpg",
        imgsz=640,
        verbose=False,
        sparse_sahi=True,
        slice_size=320,
        objectness_threshold=0.1,
    )
    # Sparse SAHI with small slices + low threshold should find at least as many boxes
    assert len(r_sp[0].boxes) >= len(r_std[0].boxes)
