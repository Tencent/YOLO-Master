"""Deterministic, sharded safetensors cache for frozen Foundation features."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

CACHE_SCHEMA_VERSION = "d1-cache-v1"
DEFAULT_TARGET_SHARD_BYTES = 2 * 1024**3
INDEX_FILENAME = "index.json"
SAMPLES_FILENAME = "samples.jsonl"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_DTYPES = {"float16": torch.float16}


def _safetensors_api():
    """Import the optional cache dependency only when cache I/O is requested."""
    try:
        from safetensors import safe_open
        from safetensors.torch import save_file
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(
            "Foundation feature caching requires safetensors. Install with: pip install -e '.[foundation]'"
        ) from exc
    return safe_open, save_file


def canonical_json_bytes(payload: Any) -> bytes:
    """Serialize JSON deterministically for hashes and manifests."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def sha256_bytes(data: bytes) -> str:
    """Return a lowercase SHA256 digest."""
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_bytes: int = 8 * 1024**2) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    """Hash the exact contiguous CPU tensor bytes."""
    value = tensor.detach().contiguous().cpu()
    return sha256_bytes(value.view(torch.uint8).numpy().tobytes())


def _validate_sha256(name: str, value: Any) -> str:
    value = str(value).lower()
    if not _SHA256_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a lowercase SHA256 digest, got {value!r}.")
    return value


def _normalize_layers(output_layers: Sequence[int]) -> tuple[int, ...]:
    if isinstance(output_layers, (str, bytes, bytearray, set, frozenset)) or not isinstance(output_layers, Sequence):
        raise TypeError("output_layers must be an ordered integer sequence.")
    layers = tuple(output_layers)
    if not layers or any(type(layer) is not int or layer <= 0 for layer in layers):
        raise ValueError("output_layers must contain positive integers.")
    if layers != tuple(sorted(set(layers))):
        raise ValueError("output_layers must be strictly increasing without duplicates.")
    return layers


def normalize_cache_contract(contract: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the fields that define cache compatibility."""
    if not isinstance(contract, Mapping):
        raise TypeError("cache contract must be a mapping.")
    dtype = str(contract.get("dtype", ""))
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported cache dtype {dtype!r}; WP2 requires float16.")
    layers = _normalize_layers(contract.get("output_layers", ()))
    feature_names = tuple(contract.get("feature_names", ()))
    if len(feature_names) != len(layers) or any(not isinstance(name, str) or not name for name in feature_names):
        raise ValueError("feature_names must provide one non-empty name for every output layer.")
    expected_shape = tuple(contract.get("expected_shape", ()))
    if len(expected_shape) != 3 or any(type(size) is not int or size <= 0 for size in expected_shape):
        raise ValueError("expected_shape must be a positive CHW shape.")
    normalized = {
        "schema_version": str(contract.get("schema_version", CACHE_SCHEMA_VERSION)),
        "model_id": str(contract.get("model_id", "")),
        "teacher_weights_sha256": _validate_sha256("teacher_weights_sha256", contract.get("teacher_weights_sha256")),
        "preprocessing_sha256": _validate_sha256("preprocessing_sha256", contract.get("preprocessing_sha256")),
        "output_layers": list(layers),
        "feature_names": list(feature_names),
        "dtype": dtype,
        "expected_shape": list(expected_shape),
    }
    if normalized["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(f"cache schema must be {CACHE_SCHEMA_VERSION!r}, got {normalized['schema_version']!r}.")
    if not normalized["model_id"]:
        raise ValueError("model_id must not be empty.")
    return normalized


def build_cache_key(
    *,
    image_sha256: str,
    preprocessing_sha256: str,
    teacher_weights_sha256: str,
    output_layers: Sequence[int],
    dtype: str,
    schema_version: str = CACHE_SCHEMA_VERSION,
) -> str:
    """Build the content key required by the D1 WP0 cache contract."""
    payload = {
        "dtype": str(dtype),
        "image_sha256": _validate_sha256("image_sha256", image_sha256),
        "output_blocks": list(_normalize_layers(output_layers)),
        "preprocessing_sha256": _validate_sha256("preprocessing_sha256", preprocessing_sha256),
        "schema_version": str(schema_version),
        "teacher_weights_sha256": _validate_sha256("teacher_weights_sha256", teacher_weights_sha256),
    }
    if payload["dtype"] not in _DTYPES:
        raise ValueError(f"unsupported cache dtype {payload['dtype']!r}.")
    if payload["schema_version"] != CACHE_SCHEMA_VERSION:
        raise ValueError(f"unsupported cache schema {payload['schema_version']!r}.")
    return sha256_bytes(canonical_json_bytes(payload))


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(data)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: Any) -> None:
    _atomic_write_bytes(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True).encode() + b"\n")


def _tensor_dtype_name(tensor: torch.Tensor) -> str:
    for name, dtype in _DTYPES.items():
        if tensor.dtype == dtype:
            return name
    return str(tensor.dtype).removeprefix("torch.")


class FeatureCacheWriter:
    """Write immutable safetensors shards and recover the index from committed shard headers."""

    def __init__(
        self,
        root: str | Path,
        *,
        split: str,
        contract: Mapping[str, Any],
        target_shard_bytes: int = DEFAULT_TARGET_SHARD_BYTES,
        shard_prefix: str | None = None,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.split = str(split)
        if not _SPLIT_PATTERN.fullmatch(self.split):
            raise ValueError(f"invalid cache split name {self.split!r}.")
        self.shard_prefix = self.split if shard_prefix is None else str(shard_prefix)
        if not _SPLIT_PATTERN.fullmatch(self.shard_prefix) or not (
            self.shard_prefix == self.split or self.shard_prefix.startswith(f"{self.split}-")
        ):
            raise ValueError(
                f"shard_prefix must equal {self.split!r} or start with {self.split + '-'!r}, got {self.shard_prefix!r}."
            )
        if type(target_shard_bytes) is not int or target_shard_bytes <= 0:
            raise ValueError("target_shard_bytes must be a positive integer.")
        self.target_shard_bytes = target_shard_bytes
        self.contract = normalize_cache_contract(contract)
        self.contract_sha256 = sha256_bytes(canonical_json_bytes(self.contract))
        self._records: dict[str, dict[str, Any]] = {}
        self._shards: dict[str, dict[str, Any]] = {}
        self._buffer_tensors: dict[str, torch.Tensor] = {}
        self._buffer_records: list[dict[str, Any]] = []
        self._buffer_bytes = 0
        self._next_shard = 0
        self._load_committed_shards()
        self._write_index()

    @property
    def sample_count(self) -> int:
        return len(self._records) + len(self._buffer_records)

    @property
    def committed_sample_count(self) -> int:
        return len(self._records)

    def record(self, sample_id: str) -> dict[str, Any] | None:
        record = self._records.get(str(sample_id))
        return dict(record) if record is not None else None

    def is_cached(self, sample_id: str, image_sha256: str) -> bool:
        """Return whether a committed sample matches the current image and contract."""
        record = self._records.get(str(sample_id))
        if record is None:
            return False
        image_sha256 = _validate_sha256("image_sha256", image_sha256)
        expected_key = self._cache_key(image_sha256)
        if record["image_sha256"] != image_sha256 or record["cache_key"] != expected_key:
            raise ValueError(f"cached sample {sample_id!r} does not match the current image or contract.")
        return True

    def add(
        self,
        *,
        sample_id: str,
        split: str,
        image_path: str,
        image_sha256: str,
        features: Mapping[str, torch.Tensor],
    ) -> bool:
        """Add one CHW feature set, returning False when an identical committed sample already exists."""
        sample_id = str(sample_id)
        split = str(split)
        if not sample_id:
            raise ValueError("sample_id must not be empty.")
        if split != self.split:
            raise ValueError(f"sample split {split!r} does not match writer split {self.split!r}.")
        if Path(image_path).is_absolute():
            raise ValueError("image_path must be relative so cache manifests remain portable.")
        image_sha256 = _validate_sha256("image_sha256", image_sha256)
        if self.is_cached(sample_id, image_sha256):
            return False
        if any(record["sample_id"] == sample_id for record in self._buffer_records):
            raise ValueError(f"duplicate uncommitted sample_id {sample_id!r}.")
        if not isinstance(features, Mapping) or tuple(features) != tuple(self.contract["feature_names"]):
            raise ValueError(
                f"features must be ordered as {tuple(self.contract['feature_names'])}, got {tuple(features)}."
            )

        cache_key = self._cache_key(image_sha256)
        dtype = _DTYPES[self.contract["dtype"]]
        expected_shape = tuple(self.contract["expected_shape"])
        tensors: dict[str, torch.Tensor] = {}
        tensor_records: dict[str, dict[str, Any]] = {}
        sample_bytes = 0
        for name, value in features.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"feature {name!r} must be a torch.Tensor.")
            if value.ndim != 3 or tuple(value.shape) != expected_shape:
                raise ValueError(f"feature {name!r} must have shape {expected_shape}, got {tuple(value.shape)}.")
            tensor = value.detach().to(device="cpu", dtype=dtype).contiguous().clone()
            if not torch.isfinite(tensor).all():
                raise ValueError(f"feature {name!r} contains NaN or Inf.")
            tensor_key = f"{cache_key}.{name}"
            nbytes = tensor.numel() * tensor.element_size()
            tensors[tensor_key] = tensor
            tensor_records[name] = {
                "key": tensor_key,
                "shape": list(tensor.shape),
                "dtype": self.contract["dtype"],
                "nbytes": nbytes,
                "sha256": sha256_tensor(tensor),
            }
            sample_bytes += nbytes

        if self._buffer_records and self._buffer_bytes + sample_bytes > self.target_shard_bytes:
            self.flush()
        self._buffer_tensors.update(tensors)
        self._buffer_records.append(
            {
                "sample_id": sample_id,
                "split": split,
                "image_path": Path(image_path).as_posix(),
                "image_sha256": image_sha256,
                "cache_key": cache_key,
                "shard": None,
                "tensors": tensor_records,
            }
        )
        self._buffer_bytes += sample_bytes
        return True

    def flush(self) -> None:
        """Atomically commit the current shard and refresh portable manifests."""
        if not self._buffer_records:
            return
        _safe_open, save_file = _safetensors_api()
        while True:
            shard_name = f"{self.shard_prefix}-{self._next_shard:05d}.safetensors"
            self._next_shard += 1
            shard_path = self.root / shard_name
            if not shard_path.exists():
                break
        records = sorted(self._buffer_records, key=lambda item: item["sample_id"])
        for record in records:
            record["shard"] = shard_name
        metadata = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "contract_sha256": self.contract_sha256,
            "contract_json": canonical_json_bytes(self.contract).decode(),
            "split": self.split,
            "records_json": canonical_json_bytes(records).decode(),
        }
        temporary = shard_path.with_name(f".{shard_path.name}.part")
        ordered_tensors = {key: self._buffer_tensors[key] for key in sorted(self._buffer_tensors)}
        save_file(ordered_tensors, temporary, metadata=metadata)
        self._validate_shard_header(temporary, records)
        os.replace(temporary, shard_path)
        shard_info = {
            "filename": shard_name,
            "split": self.split,
            "sample_count": len(records),
            "bytes": shard_path.stat().st_size,
            "sha256": sha256_file(shard_path),
        }
        for record in records:
            if record["sample_id"] in self._records:
                raise ValueError(f"duplicate sample_id recovered while committing: {record['sample_id']!r}.")
            self._records[record["sample_id"]] = record
        self._shards[shard_name] = shard_info
        self._buffer_tensors.clear()
        self._buffer_records.clear()
        self._buffer_bytes = 0
        self._write_index()

    def close(self) -> None:
        self.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        return False

    def _cache_key(self, image_sha256: str) -> str:
        return build_cache_key(
            image_sha256=image_sha256,
            preprocessing_sha256=self.contract["preprocessing_sha256"],
            teacher_weights_sha256=self.contract["teacher_weights_sha256"],
            output_layers=self.contract["output_layers"],
            dtype=self.contract["dtype"],
            schema_version=self.contract["schema_version"],
        )

    def _load_committed_shards(self) -> None:
        old_shards = {}
        index_path = self.root / INDEX_FILENAME
        if index_path.is_file():
            index = json.loads(index_path.read_text(encoding="utf-8"))
            if index.get("contract_sha256") not in (None, self.contract_sha256):
                raise ValueError("existing cache index uses a different contract.")
            if index.get("sample_count", 0) and index.get("target_shard_bytes") != self.target_shard_bytes:
                raise ValueError("existing cache index uses a different target_shard_bytes value.")
            old_shards = {entry["filename"]: entry for entry in index.get("shards", [])}

        for shard_path in sorted(self.root.glob("*.safetensors")):
            records, metadata = self._read_shard_metadata(shard_path)
            if metadata.get("contract_sha256") != self.contract_sha256:
                raise ValueError(f"shard {shard_path.name} uses a different cache contract.")
            for record in records:
                sample_id = record["sample_id"]
                if sample_id in self._records:
                    raise ValueError(f"duplicate sample_id {sample_id!r} in committed shards.")
                self._records[sample_id] = record
            previous = old_shards.get(shard_path.name, {})
            size = shard_path.stat().st_size
            shard_sha256 = previous.get("sha256") if previous.get("bytes") == size else None
            self._shards[shard_path.name] = {
                "filename": shard_path.name,
                "split": metadata["split"],
                "sample_count": len(records),
                "bytes": size,
                "sha256": shard_sha256 or sha256_file(shard_path),
            }
            match = re.fullmatch(rf"{re.escape(self.shard_prefix)}-(\d+)\.safetensors", shard_path.name)
            if match:
                self._next_shard = max(self._next_shard, int(match.group(1)) + 1)

    def _read_shard_metadata(
        self,
        path: Path,
        *,
        expected_shard_name: str | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, str]]:
        safe_open, _save_file = _safetensors_api()
        try:
            with safe_open(path, framework="pt", device="cpu") as handle:
                metadata = handle.metadata() or {}
                keys = set(handle.keys())
        except Exception as exc:
            raise ValueError(f"invalid safetensors shard {path.name}: {exc}") from exc
        if metadata.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"shard {path.name} has an unsupported schema.")
        try:
            records = json.loads(metadata["records_json"])
            embedded_contract = json.loads(metadata["contract_json"])
        except (KeyError, json.JSONDecodeError) as exc:
            raise ValueError(f"shard {path.name} has invalid cache metadata.") from exc
        logical_name = expected_shard_name or path.name
        if not isinstance(records, list) or any(
            record.get("split") != metadata.get("split") or record.get("shard") != logical_name for record in records
        ):
            raise ValueError(f"shard {path.name} has inconsistent sample records.")
        if normalize_cache_contract(embedded_contract) != self.contract:
            raise ValueError(f"shard {path.name} embeds a different cache contract.")
        expected_keys = {tensor["key"] for record in records for tensor in record.get("tensors", {}).values()}
        if keys != expected_keys:
            raise ValueError(f"shard {path.name} tensor keys do not match embedded records.")
        return records, metadata

    def _validate_shard_header(self, path: Path, records: list[dict[str, Any]]) -> None:
        recovered, metadata = self._read_shard_metadata(path, expected_shard_name=records[0]["shard"])
        if recovered != records or metadata.get("split") != self.split:
            raise ValueError(f"temporary shard {path.name} failed metadata validation.")

    def _write_index(self) -> None:
        records = sorted(self._records.values(), key=lambda item: (item["split"], item["sample_id"]))
        samples_path = self.root / SAMPLES_FILENAME
        temporary = samples_path.with_name(f".{samples_path.name}.part")
        digest = hashlib.sha256()
        with temporary.open("wb") as stream:
            for record in records:
                line = canonical_json_bytes(record) + b"\n"
                stream.write(line)
                digest.update(line)
        os.replace(temporary, samples_path)
        splits = Counter(record["split"] for record in records)
        content_records = [
            {
                "sample_id": record["sample_id"],
                "split": record["split"],
                "image_sha256": record["image_sha256"],
                "cache_key": record["cache_key"],
                "tensors": record["tensors"],
            }
            for record in records
        ]
        index = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "contract": self.contract,
            "contract_sha256": self.contract_sha256,
            "target_shard_bytes": self.target_shard_bytes,
            "sample_count": len(records),
            "split_counts": dict(sorted(splits.items())),
            "samples_manifest": SAMPLES_FILENAME,
            "samples_manifest_sha256": digest.hexdigest(),
            "content_sha256": sha256_bytes(canonical_json_bytes(content_records)),
            "shards": [self._shards[name] for name in sorted(self._shards)],
        }
        _atomic_write_json(self.root / INDEX_FILENAME, index)


class FeatureCacheReader:
    """Read samples from a verified portable feature-cache index."""

    def __init__(self, root: str | Path, *, max_open_shards: int = 0) -> None:
        if type(max_open_shards) is not int or max_open_shards < 0:
            raise ValueError("max_open_shards must be a non-negative integer.")
        self.root = Path(root)
        self.max_open_shards = max_open_shards
        self._handle_pid = os.getpid()
        self._open_shards: OrderedDict[tuple[str, str], tuple[Any, Any]] = OrderedDict()
        index_path = self.root / INDEX_FILENAME
        if not index_path.is_file():
            raise FileNotFoundError(index_path)
        self.index = json.loads(index_path.read_text(encoding="utf-8"))
        if self.index.get("schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError("feature cache index has an unsupported schema.")
        self.contract = normalize_cache_contract(self.index.get("contract", {}))
        expected_contract_sha = sha256_bytes(canonical_json_bytes(self.contract))
        if self.index.get("contract_sha256") != expected_contract_sha:
            raise ValueError("feature cache contract hash does not match index contents.")
        samples_path = self.root / self.index.get("samples_manifest", SAMPLES_FILENAME)
        data = samples_path.read_bytes()
        if sha256_bytes(data) != self.index.get("samples_manifest_sha256"):
            raise ValueError("feature cache samples manifest checksum mismatch.")
        self.records: dict[str, dict[str, Any]] = {}
        for line_number, line in enumerate(data.splitlines(), start=1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid cache sample record on line {line_number}.") from exc
            sample_id = record.get("sample_id")
            if not sample_id or sample_id in self.records:
                raise ValueError(f"duplicate or empty sample_id on line {line_number}.")
            self.records[sample_id] = record
        if len(self.records) != self.index.get("sample_count"):
            raise ValueError("feature cache sample count does not match index.")

    def _close_open_shards(self) -> None:
        while self._open_shards:
            _key, (context, _handle) = self._open_shards.popitem(last=False)
            context.__exit__(None, None, None)

    def _reset_after_fork(self) -> None:
        pid = os.getpid()
        if pid != self._handle_pid:
            # Handles opened by a parent process must never be reused by a DataLoader worker.
            self._open_shards = OrderedDict()
            self._handle_pid = pid

    def _open_cached_shard(self, path: Path, device: str) -> Any:
        self._reset_after_fork()
        key = (str(path), device)
        cached = self._open_shards.pop(key, None)
        if cached is not None:
            self._open_shards[key] = cached
            return cached[1]
        safe_open, _save_file = _safetensors_api()
        context = safe_open(path, framework="pt", device=device)
        handle = context.__enter__()
        self._open_shards[key] = (context, handle)
        while len(self._open_shards) > self.max_open_shards:
            _old_key, (old_context, _old_handle) = self._open_shards.popitem(last=False)
            old_context.__exit__(None, None, None)
        return handle

    def get(self, sample_id: str, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
        record = self.records.get(str(sample_id))
        if record is None:
            raise KeyError(sample_id)
        shard_path = self.root / record["shard"]
        device_name = str(device)
        features = {}
        if self.max_open_shards:
            handle = self._open_cached_shard(shard_path, device_name)
            for name in self.contract["feature_names"]:
                features[name] = handle.get_tensor(record["tensors"][name]["key"])
            return features
        safe_open, _save_file = _safetensors_api()
        with safe_open(shard_path, framework="pt", device=device_name) as handle:
            for name in self.contract["feature_names"]:
                features[name] = handle.get_tensor(record["tensors"][name]["key"])
        return features

    def close(self) -> None:
        """Close cached safetensors handles owned by the current process."""
        if os.getpid() == self._handle_pid:
            self._close_open_shards()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - Destructors must tolerate partial construction and interpreter shutdown.
            pass


def verify_feature_cache(root: str | Path, *, full_tensor_hash: bool = True) -> dict[str, Any]:
    """Verify index, shard hashes, tensor metadata, and optional full tensor checksums."""
    reader = FeatureCacheReader(root)
    root = reader.root
    indexed_shards = {entry["filename"]: entry for entry in reader.index.get("shards", [])}
    actual_shards = {path.name for path in root.glob("*.safetensors")}
    if actual_shards != set(indexed_shards):
        raise ValueError("feature cache shard set does not match index.")
    safe_open, _save_file = _safetensors_api()
    checked_tensors = 0
    total_tensor_bytes = 0
    records_by_shard: dict[str, list[dict[str, Any]]] = {}
    for record in reader.records.values():
        records_by_shard.setdefault(record["shard"], []).append(record)

    for shard_name in sorted(indexed_shards):
        shard_path = root / shard_name
        shard = indexed_shards[shard_name]
        if shard_path.stat().st_size != shard["bytes"]:
            raise ValueError(f"shard size mismatch: {shard_name}")
        if sha256_file(shard_path) != shard["sha256"]:
            raise ValueError(f"shard checksum mismatch: {shard_name}")
        records = records_by_shard.get(shard_name, [])
        if len(records) != shard["sample_count"]:
            raise ValueError(f"shard sample count mismatch: {shard_name}")
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for record in records:
                for name in reader.contract["feature_names"]:
                    tensor_info = record["tensors"][name]
                    tensor = handle.get_tensor(tensor_info["key"])
                    if list(tensor.shape) != tensor_info["shape"]:
                        raise ValueError(f"tensor shape mismatch: {record['sample_id']} {name}")
                    if _tensor_dtype_name(tensor) != tensor_info["dtype"]:
                        raise ValueError(f"tensor dtype mismatch: {record['sample_id']} {name}")
                    nbytes = tensor.numel() * tensor.element_size()
                    if nbytes != tensor_info["nbytes"]:
                        raise ValueError(f"tensor byte count mismatch: {record['sample_id']} {name}")
                    if full_tensor_hash and sha256_tensor(tensor) != tensor_info["sha256"]:
                        raise ValueError(f"tensor checksum mismatch: {record['sample_id']} {name}")
                    checked_tensors += 1
                    total_tensor_bytes += nbytes
    return {
        "schema_version": CACHE_SCHEMA_VERSION,
        "contract_sha256": reader.index["contract_sha256"],
        "content_sha256": reader.index["content_sha256"],
        "sample_count": len(reader.records),
        "shard_count": len(indexed_shards),
        "tensor_count": checked_tensors,
        "tensor_bytes": total_tensor_bytes,
        "cache_bytes": sum(entry["bytes"] for entry in indexed_shards.values()),
        "part_files": sorted(path.name for path in root.glob("*.part")),
    }


def compare_feature_caches(first: str | Path, second: str | Path) -> dict[str, Any]:
    """Require two independent cache builds to contain identical contracts and tensor records."""
    first_report = verify_feature_cache(first)
    second_report = verify_feature_cache(second)
    for field in ("contract_sha256", "content_sha256", "sample_count", "tensor_count", "tensor_bytes"):
        if first_report[field] != second_report[field]:
            raise ValueError(f"feature caches differ in {field}.")
    return {
        "identical": True,
        "contract_sha256": first_report["contract_sha256"],
        "content_sha256": first_report["content_sha256"],
        "sample_count": first_report["sample_count"],
        "tensor_count": first_report["tensor_count"],
        "tensor_bytes": first_report["tensor_bytes"],
        "first_shards": first_report["shard_count"],
        "second_shards": second_report["shard_count"],
    }


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "DEFAULT_TARGET_SHARD_BYTES",
    "FeatureCacheReader",
    "FeatureCacheWriter",
    "build_cache_key",
    "canonical_json_bytes",
    "compare_feature_caches",
    "normalize_cache_contract",
    "sha256_bytes",
    "sha256_file",
    "sha256_tensor",
    "verify_feature_cache",
]
