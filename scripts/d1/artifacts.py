"""Shared deterministic artifact I/O for D1 preparation, cache and evaluation tools."""

import json
import os
from pathlib import Path

from ultralytics.nn.foundation.cache import sha256_bytes as digest
from ultralytics.nn.foundation.cache import sha256_file as file_sha

__all__ = ("digest", "encoded", "file_sha", "immutable", "write_json")


def encoded(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def immutable(path, data, verify=False):
    """Validate an existing artifact or atomically publish new bytes without overwriting it."""
    path = Path(path)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Refusing symlink output: {path}")
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"Existing artifact differs: {path}")
    elif verify:
        raise FileNotFoundError(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".part")
        if tmp.is_symlink():
            raise ValueError("Unsafe temporary file")
        with tmp.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)


def write_json(path, payload):
    """Atomically replace a mutable progress/report JSON; not for immutable cache identity."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.part")
    temporary.write_bytes(encoded(payload))
    os.replace(temporary, path)
