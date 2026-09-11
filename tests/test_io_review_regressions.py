"""Exercise cache failure cleanup and non-destructive dataset archives."""

import gc
import pickle
from zipfile import ZipFile

import numpy as np
import pytest

from ultralytics.data.utils import load_dataset_cache_file
from ultralytics.utils.downloads import delete_dsstore, zip_directory


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("kind", ["valid", "missing", "invalid"])
def test_cache_restores_gc_state(tmp_path, enabled, kind):
    """Restore the caller's GC state on successful, missing and corrupt cache loads."""
    original = gc.isenabled()
    path = tmp_path / "labels.cache"
    if kind == "valid":
        with path.open("wb") as handle:
            np.save(handle, {"version": "test"})
    elif kind == "invalid":
        path.write_bytes(b"corrupt")
    try:
        (gc.enable if enabled else gc.disable)()
        if kind == "valid":
            assert load_dataset_cache_file(path) == {"version": "test"}
        else:
            with pytest.raises((FileNotFoundError, ValueError, pickle.UnpicklingError)):
                load_dataset_cache_file(path)
        assert gc.isenabled() == enabled
    finally:
        (gc.enable if original else gc.disable)()


def test_zip_excludes_metadata_without_deleting_source(tmp_path):
    """Exclude complete metadata directories while leaving input files unchanged."""
    root = tmp_path / "images"
    (root / "nested" / "__MACOSX").mkdir(parents=True)
    for name in ["image.jpg", ".DS_Store", "nested/__MACOSX/resource", "nested/keep.jpg"]:
        (root / name).write_bytes(b"fixture")
    before = {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}
    archive = zip_directory(root, progress=False)
    with ZipFile(archive) as handle:
        assert set(handle.namelist()) == {"image.jpg", "nested/keep.jpg"}
    assert before == {str(p.relative_to(root)): p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_explicit_metadata_cleanup_handles_directories(tmp_path):
    """The standalone cleanup API must handle real directories as well as files."""
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "resource").touch()
    (tmp_path / ".DS_Store").touch()
    (tmp_path / "keep.jpg").touch()
    delete_dsstore(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["keep.jpg"]
