# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Record the identity of a D2 experiment run: code version, dependencies, teacher weights, config hashes.

``metrics.csv`` says what a run produced and ``args.yaml`` says what it was asked to do, but neither says which
code produced it. Three weeks later that gap is unrecoverable: the commit has moved, the environment has been
reinstalled, and there is no way to tell whether an uncommitted edit was in the working tree at the time.

This writes that identity to JSON so a run can be audited later without relying on anyone's memory.

Two deliberate choices:

- **Dirty state is reported honestly, and twice.** A repository-wide ``dirty`` flag is usually true during active
  work (generated results, scratch files), which makes it useless on its own. A second flag covers only the paths
  that define the experiment, so a reviewer can tell "there were stray files" apart from "the experiment inputs
  were modified".
- **The teacher revision is read from the Hugging Face cache, not from config.** The cache stores snapshots under
  ``snapshots/<commit-sha>/``, so this records the revision that was actually loaded rather than one someone typed
  into a document. See ``limitations.md`` 2.3.

No token, password, or full environment dump is ever written; see ``limitations.md`` 4.

Usage
    python experiments/d2/scripts/record_environment.py
    python experiments/d2/scripts/record_environment.py --out results/environment_p1.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent  # experiments/d2, one level up from scripts/
ROOT = HERE.parents[1]

# Paths whose modification changes what the experiment *is*, as opposed to leaving stray files around.
EXPERIMENT_INPUT_PATHS = (
    "experiments/d2/configs/p1_voc",
    "experiments/d2/p1_voc_matrix.csv",
    "experiments/d2/scripts/validate_pair.py",
    "ultralytics/nn/foundation",
    "ultralytics/nn/foundation_distill_model.py",
    "ultralytics/cfg/default.yaml",
    "ultralytics/engine/trainer.py",
)

TEACHERS = (
    "facebook/dinov3-vits16-pretrain-lvd1689m",
    "google/siglip2-base-patch16-256",
)

# Weight files worth hashing; config/tokenizer files are covered by the snapshot revision itself.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pth")


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    """Return the SHA-256 of a file, read in chunks so multi-GB teacher weights do not land in memory.

    Args:
        path (Path): File to hash.
        chunk_size (int): Read size in bytes.

    Returns:
        (str): Lowercase hex digest.

    Examples:
        >>> import tempfile, pathlib
        >>> with tempfile.TemporaryDirectory() as directory:
        ...     sample = pathlib.Path(directory) / "sample.txt"
        ...     _ = sample.write_text("d2")
        ...     sha256_file(sample)[:16]
        'e788103ee15318fc'
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git(*args: str, strip: bool = True) -> str | None:
    """Run a git command in the repository root and return stdout, or None when git is unavailable.

    Args:
        *args (str): Arguments passed to ``git``.
        strip (bool): Strip surrounding whitespace. Must be False for ``status --porcelain``, whose status codes
            are column-aligned and whose first line therefore begins with a meaningful space.

    Returns:
        (str | None): Command stdout, or None when git is missing or the command failed.
    """
    try:
        result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() if strip else result.stdout


def parse_porcelain(porcelain: str) -> set[str]:
    """Extract changed paths from ``git status --porcelain`` output.

    Each line is ``XY PATH``, so the path starts at column 3 regardless of status code. Renames appear as
    ``old -> new``; the destination is what exists now, so that is what gets recorded.

    Args:
        porcelain (str): Raw, unstripped output of ``git status --porcelain``.

    Returns:
        (set[str]): Repository-relative paths.

    Examples:
        >>> sorted(parse_porcelain(" M a.py\\n?? b/c.yaml\\n"))
        ['a.py', 'b/c.yaml']
        >>> sorted(parse_porcelain("R  old.py -> new.py\\n"))
        ['new.py']
    """
    paths = set()
    for line in porcelain.splitlines():
        if len(line) <= 3:
            continue
        path = line[3:].strip().strip('"')
        if " -> " in path:
            path = path.split(" -> ", 1)[1].strip().strip('"')
        paths.add(path)
    return paths


def repository_state() -> dict:
    """Return commit identity plus two independent dirty flags: repository-wide and experiment-inputs-only."""
    porcelain = git("status", "--porcelain", strip=False) or ""
    changed = parse_porcelain(porcelain)
    dirty_inputs = sorted(path for path in changed if path.startswith(EXPERIMENT_INPUT_PATHS))
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "upstream_main": git("rev-parse", "upstream/main"),
        "repository_dirty": bool(porcelain),
        "experiment_inputs_dirty": bool(dirty_inputs),
        "experiment_inputs_changed_paths": dirty_inputs,
        "audited_input_paths": list(EXPERIMENT_INPUT_PATHS),
    }


def python_state() -> dict:
    """Return interpreter and platform identity."""
    return {"version": platform.python_version(), "executable": sys.executable, "platform": platform.platform()}


def torch_state() -> dict:
    """Return torch build and visible-device identity, degrading to a reason string when torch is absent."""
    try:
        import torch
    except ImportError as exc:
        return {"available": False, "reason": str(exc)}
    state = {
        "available": True,
        "version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
    }
    if state["cuda_available"]:
        state["device_name"] = torch.cuda.get_device_name(0)
        state["capability"] = list(torch.cuda.get_device_capability(0))
    return state


def package_versions() -> dict:
    """Return versions of the packages that can change Foundation behavior."""
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for name in ("transformers", "huggingface-hub", "ultralytics", "numpy", "opencv-python"):
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def teacher_state(model_id: str) -> dict:
    """Resolve a teacher from the local Hugging Face cache and record its revision and weight hashes.

    Nothing is downloaded: an absent teacher is reported as ``cached: False`` rather than fetched, so this script
    stays safe to run in any environment.

    Args:
        model_id (str): Hugging Face model id.

    Returns:
        (dict): Cache status, resolved revision, and per-file SHA-256 of the weight files.
    """
    record: dict = {"model_id": model_id, "cached": False}
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        record["reason"] = f"huggingface_hub unavailable: {exc}"
        return record
    try:
        snapshot = Path(snapshot_download(model_id, local_files_only=True))
    except Exception as exc:  # noqa: BLE001 - any resolution failure means "not usable from cache"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        return record

    record["cached"] = True
    # The Hugging Face cache lays snapshots out as .../snapshots/<commit-sha>/, so the directory name is the
    # revision that was actually resolved -- stronger evidence than a revision copied into a document by hand.
    record["revision"] = snapshot.name
    record["weights"] = {
        file.name: {"sha256": sha256_file(file), "bytes": file.stat().st_size}
        for file in sorted(snapshot.iterdir())
        if file.is_file() and file.suffix in WEIGHT_SUFFIXES
    }
    return record


def config_hashes() -> dict:
    """Return SHA-256 of every file that defines the experiment protocol."""
    targets = sorted((HERE / "configs" / "p1_voc").glob("p1_*.yaml"))
    targets += [HERE / "p1_voc_matrix.csv", HERE / "scripts" / "validate_pair.py"]
    return {str(path.relative_to(ROOT)): sha256_file(path) for path in targets if path.is_file()}


def main() -> int:
    """Collect the environment record and write it as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(HERE / "results" / "environment.json"))
    parser.add_argument(
        "--skip-teachers",
        action="store_true",
        help="skip teacher cache hashing, which reads several hundred MB",
    )
    args = parser.parse_args()

    record = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "repository": repository_state(),
        "python": python_state(),
        "torch": torch_state(),
        "packages": package_versions(),
        "teachers": {} if args.skip_teachers else {name: teacher_state(name) for name in TEACHERS},
        "experiment_inputs": config_hashes(),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, sort_keys=False) + "\n", encoding="utf-8")

    repo = record["repository"]
    print(f"commit          {repo['commit']}  ({repo['branch']})")
    print(f"repo dirty      {repo['repository_dirty']}")
    print(f"inputs dirty    {repo['experiment_inputs_dirty']}  {repo['experiment_inputs_changed_paths'] or ''}")
    print(f"python / torch  {record['python']['version']} / {record['torch'].get('version', 'absent')}")
    print(f"transformers    {record['packages']['transformers']}")
    for name, teacher in record["teachers"].items():
        status = teacher.get("revision") if teacher["cached"] else f"not cached ({teacher.get('reason', '')[:60]})"
        print(f"teacher         {name}  {status}")
    print(f"input files     {len(record['experiment_inputs'])} hashed")
    print(f"\nwritten to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
