"""Download and safely install official COCO 2017 Stuff and Panoptic annotation archives.

The installer keeps archives under ``<dataset-root>/.archives`` and extracts only missing members below the existing
COCO root. It validates the official archive MD5s before extraction and never replaces a label file unless ``--force``
is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


DEFAULT_ROOT = Path(os.environ.get("YOLO_MASTER_COCO_ROOT", Path.home() / "datasets" / "coco2017"))


@dataclass(frozen=True)
class AnnotationArchive:
    """Official COCO archive metadata and expected installed paths."""

    name: str
    url: str
    md5: str
    required_paths: tuple[str, ...]


ARCHIVES = {
    "panoptic": AnnotationArchive(
        name="panoptic_annotations_trainval2017.zip",
        url="https://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
        md5="4170db65fc022c9c296af880dbca6055",
        required_paths=(
            "annotations/panoptic_train2017.json",
            "annotations/panoptic_val2017.json",
            "annotations/panoptic_train2017",
            "annotations/panoptic_val2017",
        ),
    ),
    "stuff": AnnotationArchive(
        name="stuff_annotations_trainval2017.zip",
        url="https://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
        md5="2a27c15a2dfcbd2e1c9276dc23cac101",
        required_paths=(
            "annotations/stuff_train2017.json",
            "annotations/stuff_val2017.json",
            "annotations/stuff_train2017_pixelmaps",
            "annotations/stuff_val2017_pixelmaps",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse download and installation options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT, help="Existing COCO 2017 root.")
    parser.add_argument(
        "--tasks", nargs="+", choices=tuple(ARCHIVES), default=tuple(ARCHIVES), help="Archives to install."
    )
    parser.add_argument("--force", action="store_true", help="Replace existing extracted members.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Pass --insecure to curl only when the local TLS trust store rejects the official host certificate.",
    )
    return parser.parse_args()


def md5(path: Path) -> str:
    """Return the MD5 digest of an archive without loading it into memory."""
    digest = hashlib.md5()  # noqa: S324 - verifies COCO's published legacy archive checksum.
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_installed(root: Path, archive: AnnotationArchive) -> bool:
    """Return whether all required paths for one official archive are installed."""
    return all((root / path).exists() for path in archive.required_paths)


def download(root: Path, archive: AnnotationArchive, insecure: bool) -> Path:
    """Download one official archive, resuming a matching partial file when possible."""
    archive_dir = root / ".archives"
    archive_dir.mkdir(exist_ok=True)
    destination = archive_dir / archive.name
    if destination.is_file() and md5(destination) == archive.md5:
        return destination

    command = ["curl", "--fail", "--location", "--retry", "3", "--continue-at", "-", "--output", str(destination)]
    if insecure:
        command.append("--insecure")
    command.append(archive.url)
    subprocess.run(command, check=True)
    observed_md5 = md5(destination)
    if observed_md5 != archive.md5:
        raise RuntimeError(
            f"MD5 mismatch for {archive.name}: expected {archive.md5}, received {observed_md5}. "
            f"The partial archive was retained at {destination} so a later invocation can resume it."
        )
    return destination


def _safe_member_path(root: Path, member: str) -> Path:
    """Resolve a zip member below root, rejecting absolute paths and traversal."""
    relative = PurePosixPath(member)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe archive member: {member}")
    target = (root / Path(*relative.parts)).resolve()
    root_resolved = root.resolve()
    if os.path.commonpath((str(root_resolved), str(target))) != str(root_resolved):
        raise ValueError(f"Archive member escapes dataset root: {member}")
    return target


def extract_missing(archive_path: Path, root: Path, force: bool) -> int:
    """Safely extract missing files and return the number of written members."""
    written = 0
    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        for member in members:
            _safe_member_path(root, member.filename)
        for member in members:
            if member.is_dir():
                _safe_member_path(root, member.filename).mkdir(parents=True, exist_ok=True)
                continue
            target = _safe_member_path(root, member.filename)
            if target.exists() and not force:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1 << 20)
            written += 1
    return written


def installed_summary(root: Path, selected: list[str]) -> dict[str, object]:
    """Build a small installation report that records files available to the multi-task loader."""
    summary: dict[str, object] = {"dataset_root": str(root), "archives": {}}
    for task in selected:
        metadata = ARCHIVES[task]
        files = {
            path: sum(1 for _ in (root / path).glob("*.png")) if (root / path).is_dir() else (root / path).is_file()
            for path in metadata.required_paths
        }
        summary["archives"][task] = {"archive": metadata.name, "md5": metadata.md5, "paths": files}
    return summary


def main() -> None:
    """Install requested COCO dense annotation archives and write an integrity report."""
    args = parse_args()
    root = args.dataset_root.resolve()
    if not (root / "images" / "train2017").is_dir():
        raise FileNotFoundError(f"Expected an existing COCO image root at {root / 'images' / 'train2017'}")

    selected = list(dict.fromkeys(args.tasks))
    for task in selected:
        metadata = ARCHIVES[task]
        if is_installed(root, metadata) and not args.force:
            print(f"{task}: already installed")
            continue
        archive_path = download(root, metadata, insecure=args.insecure)
        written = extract_missing(archive_path, root, force=args.force)
        if not is_installed(root, metadata):
            raise RuntimeError(f"Extraction of {metadata.name} finished but required paths are incomplete")
        print(f"{task}: installed {written} files from {archive_path.name}")

    report = installed_summary(root, selected)
    report_path = root / "annotations" / "dense_annotations_install.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
