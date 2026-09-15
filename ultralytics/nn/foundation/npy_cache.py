"""Read losslessly converted D1 NPY caches without changing the feature contract."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from ultralytics.nn.foundation.cache import (
    FeatureCacheReader,
    canonical_json_bytes,
    normalize_cache_contract,
    sha256_bytes,
    sha256_file,
    sha256_tensor,
)

NPY_SCHEMA_VERSION = "d1-npy-cache-v1"
SPLITS = ("train2017", "val2017", "visdrone-train", "visdrone-val")


def npy_index_path(root: str | Path) -> Path | None:
    """NPY callers pass the split directory, not the combined cache directory."""
    root = Path(root)
    candidate = root.parent / f"{root.name}-index.json"
    return candidate if root.name in SPLITS and candidate.is_file() else None


class NpyFeatureCacheReader:
    """Expose the WP2 reader API for one-file-per-image, uncompressed FP16 arrays."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.index_path = npy_index_path(self.root)
        if self.index_path is None:
            raise FileNotFoundError(f"NPY split index is missing for {self.root}")
        self.index = json.loads(self.index_path.read_text())
        self.split = self.root.name
        if self.index.get("schema_version") != NPY_SCHEMA_VERSION or self.index.get("split") != self.split:
            raise ValueError("NPY index schema or split mismatch.")
        self.contract = normalize_cache_contract(self.index.get("contract", {}))
        if sha256_bytes(canonical_json_bytes(self.contract)) != self.index.get("contract_sha256"):
            raise ValueError("NPY contract checksum mismatch.")
        self.shape = (len(self.contract["feature_names"]), *self.contract["expected_shape"])
        if (
            self.index.get("shape") != list(self.shape)
            or self.index.get("dtype") != "float16"
            or self.index.get("feature_names") != self.contract["feature_names"]
        ):
            raise ValueError("NPY array metadata disagrees with the feature contract.")
        manifest = f"{self.split}-samples.jsonl"
        if self.index.get("samples_manifest") != manifest:
            raise ValueError("NPY samples manifest must use the canonical split filename.")
        data = (self.root.parent / manifest).read_bytes()
        if sha256_bytes(data) != self.index.get("samples_manifest_sha256"):
            raise ValueError("NPY samples manifest checksum mismatch.")
        self.records = {}
        previous = ""
        for line in data.splitlines():
            record = json.loads(line)
            sid = record.get("sample_id", "")
            id_pattern = r"[A-Za-z0-9][A-Za-z0-9_-]*" if self.split.startswith("visdrone-") else r"[0-9]{12}"
            if not re.fullmatch(rf"{self.split}/{id_pattern}", sid) or sid <= previous:
                raise ValueError("NPY sample IDs must be sorted, unique, and in the selected dataset split.")
            if record.get("split") != self.split or record.get("npy_path") != f"{sid}.npy":
                raise ValueError("NPY sample path or split mismatch.")
            if record.get("image_path") != f"images/{sid}.jpg":
                raise ValueError("NPY image identity mismatch.")
            if not re.fullmatch(r"[0-9a-f]{64}", record.get("npy_sha256", "")):
                raise ValueError("NPY file digest is invalid.")
            if type(record.get("npy_bytes")) is not int or record["npy_bytes"] <= 0:
                raise ValueError("NPY file size is invalid.")
            shard = record.get("source_shard", "")
            if not shard.startswith(self.split + "-") or Path(shard).name != shard:
                raise ValueError("NPY source shard identity is invalid.")
            if set(record.get("tensors", {})) != set(self.contract["feature_names"]):
                raise ValueError("NPY tensor metadata is incomplete.")
            self.records[sid] = record
            previous = sid
        if not self.records or len(self.records) != self.index.get("sample_count"):
            raise ValueError("NPY sample count mismatch.")
        if sum(r["npy_bytes"] for r in self.records.values()) != self.index.get("npy_bytes"):
            raise ValueError("NPY total byte count mismatch.")

    def get(self, sample_id: str, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        """Load owned contiguous memory; never keep mmap or file handles across batches."""
        record = self.records[sample_id]
        path = self.root.parent / record["npy_path"]
        if path.is_symlink():
            raise ValueError(f"NPY samples must not be symlinks: {sample_id}")
        with path.open("rb") as stream:
            if os.fstat(stream.fileno()).st_size != record["npy_bytes"]:
                raise ValueError(f"NPY file size changed: {sample_id}")
            array = np.load(stream, allow_pickle=False)
            if stream.tell() != record["npy_bytes"]:
                raise ValueError(f"NPY file contains trailing data: {sample_id}")
        if array.shape != self.shape or array.dtype != np.dtype("float16") or not array.flags.c_contiguous:
            raise ValueError(f"NPY shape, dtype, or layout mismatch: {sample_id}")
        return {
            name: torch.from_numpy(array[index]).to(device) for index, name in enumerate(self.contract["feature_names"])
        }

    def verify_sample(self, sample_id: str) -> None:
        """Check both the NPY envelope and the exact original per-layer bytes."""
        record = self.records[sample_id]
        if sha256_file(self.root.parent / record["npy_path"]) != record["npy_sha256"]:
            raise ValueError(f"NPY file checksum mismatch: {sample_id}")
        for name, tensor in self.get(sample_id).items():
            if not torch.isfinite(tensor).all() or sha256_tensor(tensor) != record["tensors"][name]["sha256"]:
                raise ValueError(f"NPY tensor checksum or finite-value mismatch: {sample_id}/{name}")

    def close(self) -> None:
        """No persistent descriptors or mappings are held."""


def open_feature_cache(root: str | Path, *, max_open_shards: int = 0):
    """Select a published format without falling back when an index is corrupt."""
    if npy_index_path(root) is not None:
        return NpyFeatureCacheReader(root)
    return FeatureCacheReader(root, max_open_shards=max_open_shards)


def validate_npy_evidence(root: Path, report_path: Path, split: str, expected_count: int) -> dict:
    """Check every conversion receipt and file size, plus spread-out tensor hashes.

    Full content SHA256 verification was performed during conversion and independent
    readback. This preflight revalidates that evidence; it is not a new full-byte scan.
    """
    reader = NpyFeatureCacheReader(root)
    index = reader.index
    summary = json.loads(report_path.read_text())
    if (
        report_path.resolve() != (root.parent / "summary.json").resolve()
        or summary.get("status") != "COMPLETED"
        or summary.get("schema_version") != NPY_SCHEMA_VERSION
        or summary.get("source_indices_sha256", {}).get(split) != index.get("source_index_sha256")
        or reader.split != split
        or len(reader.records) != expected_count
    ):
        raise ValueError("NPY conversion summary, index, or sample count disagree.")
    provenance = root.parent / "provenance" / split
    source = FeatureCacheReader(provenance)
    if (
        sha256_file(provenance / "index.json") != index["source_index_sha256"]
        or source.index["content_sha256"] != index["source_content_sha256"]
        or source.contract != reader.contract
        or set(source.records) != set(reader.records)
    ):
        raise ValueError("NPY source provenance mismatch.")
    for sid, record in reader.records.items():
        original = {k: v for k, v in record.items() if not k.startswith("npy_") and k != "source_shard"}
        original["shard"] = record["source_shard"]
        if original != source.records[sid]:
            raise ValueError(f"NPY original sample metadata changed: {sid}")
    verified = set()
    for shard in source.index["shards"]:
        receipt = json.loads((root.parent / "receipts" / f"{shard['filename']}.json").read_text())
        if (
            receipt.get("schema_version") != NPY_SCHEMA_VERSION
            or receipt.get("state") not in {"VERIFIED", "SOURCE_REMOVED"}
            or receipt.get("source_index_sha256") != index["source_index_sha256"]
            or receipt.get("source_shard") != shard
            or receipt.get("sample_count") != shard["sample_count"]
            or receipt.get("verified_tensor_count") != shard["sample_count"] * len(reader.contract["feature_names"])
            or len(receipt.get("files", [])) != shard["sample_count"]
        ):
            raise ValueError(f"NPY conversion receipt mismatch: {shard['filename']}")
        for file in receipt["files"]:
            sid = file["sample_id"]
            record = reader.records.get(sid, {})
            if (
                sid in verified
                or file
                != {
                    "sample_id": sid,
                    "path": record.get("npy_path"),
                    "bytes": record.get("npy_bytes"),
                    "sha256": record.get("npy_sha256"),
                }
                or record.get("source_shard") != shard["filename"]
            ):
                raise ValueError(f"NPY receipt coverage mismatch: {sid}")
            verified.add(sid)
    if verified != set(reader.records):
        raise ValueError("NPY conversion receipts omit samples.")
    entries = {entry.name: entry for entry in os.scandir(root)}
    expected_names = {Path(r["npy_path"]).name for r in reader.records.values()}
    if set(entries) != expected_names:
        raise ValueError("NPY file inventory differs from the index (missing, extra, or partial files).")
    for record in reader.records.values():
        entry = entries[Path(record["npy_path"]).name]
        if entry.is_symlink() or not entry.is_file() or entry.stat().st_size != record["npy_bytes"]:
            raise ValueError(f"NPY file size or type changed: {entry.name}")
    ids = sorted(reader.records)
    checked = sorted({ids[i] for i in np.linspace(0, len(ids) - 1, min(32, len(ids)), dtype=int)})
    for sid in checked:
        reader.verify_sample(sid)
    return {
        "split": split,
        "format": NPY_SCHEMA_VERSION,
        "report": report_path.name,
        "sample_count": len(ids),
        "shard_count": 0,
        "cache_bytes": index["npy_bytes"],
        "contract_sha256": index["contract_sha256"],
        "content_sha256": index["source_content_sha256"],
        "index_sha256": sha256_file(reader.index_path),
        "samples_manifest_sha256": index["samples_manifest_sha256"],
        "teacher_weights_sha256": reader.contract["teacher_weights_sha256"],
        "conversion_receipts": len(source.index["shards"]),
        "preflight_hashed_samples": checked,
        "verification_scope": "all conversion receipts and file sizes; up to 32 spread-out sample hashes",
    }
