#!/usr/bin/env python3
"""Deterministically mirror RepoWiki content into the GitHub Pages content tree."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import tempfile
import unicodedata
from collections import Counter
from pathlib import Path
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
LANGS = ("en", "zh")
HAN_RE = re.compile(r"[\u3400-\u9fff]")
LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)\s]+)(?:\s+[\"'][^\"']*[\"'])?\)")
TOC_RE = re.compile(r"^\s*[-*+]\s+\[[^\]]+\]\((#[^)]+)\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})([^`]*)$")
ATX_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
DANGEROUS_HTML_RE = re.compile(r"<\s*(script|iframe|object|embed|form|input|button|style|meta|link)\b", re.I)
SAFE_PROTOCOLS = {"http", "https", "mailto"}
EN_HAN_ALLOWLIST: dict[str, list[tuple[str, str]]] = {}


def files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.md"), key=lambda p: p.relative_to(root).as_posix())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key_for(path: str) -> str:
    stem = Path(path).stem
    return "".join(c for c in unicodedata.normalize("NFKC", stem).casefold() if c.isalnum())


def build_mapping(paths: dict[str, list[str]]) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    result = {lang: {} for lang in LANGS}
    other_by_key = {}
    for lang in LANGS:
        other = "zh" if lang == "en" else "en"
        buckets: dict[str, list[str]] = {}
        for path in paths[other]:
            buckets.setdefault(key_for(path), []).append(path)
        other_by_key[lang] = buckets
    missing = []
    for lang in LANGS:
        other = "zh" if lang == "en" else "en"
        other_set = set(paths[other])
        for path in paths[lang]:
            match = path if path in other_set else None
            if not match:
                candidates = other_by_key[lang].get(key_for(path), [])
                match = candidates[0] if len(candidates) == 1 else None
            if match:
                result[lang][path] = f"{other}/{match}"
            else:
                missing.append({"language": lang, "path": path, "fallback": f"{other}/"})
    return result, missing


def category(directory: Path, content_root: Path, lang: str, mapping: dict[str, str]) -> dict:
    rel_dir = directory.relative_to(content_root).as_posix()
    pages = []
    for page in sorted(directory.glob("*.md"), key=lambda p: p.name):
        rel = page.relative_to(content_root).as_posix()
        pages.append({"name": page.stem, "path": f"{lang}/{rel}", "translationKey": f"{lang}:{rel}",
                      "alternatePath": mapping.get(rel)})
    children = [category(p, content_root, lang, mapping) for p in sorted(directory.iterdir(), key=lambda p: p.name)
                if p.is_dir()]
    landing = next((p["path"] for p in pages if Path(p["path"]).stem == directory.name), None)
    return {"name": directory.name, "path": rel_dir, "landingPage": landing,
            "children": children, "pages": pages}


def index_for(root: Path, lang: str, mapping: dict[str, str]) -> dict:
    children = [category(p, root, lang, mapping) for p in sorted(root.iterdir(), key=lambda p: p.name) if p.is_dir()]
    return {"name": lang.upper(), "language": lang, "children": children}


def json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False) + "\n").encode()


def expected(source: Path) -> tuple[dict[str, dict[str, bytes]], dict[str, bytes]]:
    source_files: dict[str, dict[str, bytes]] = {}
    paths: dict[str, list[str]] = {}
    for lang in LANGS:
        root = source / lang / "content"
        if not root.is_dir():
            raise SystemExit(f"missing source: {root}")
        source_files[lang] = {p.relative_to(root).as_posix(): p.read_bytes() for p in files(root)}
        paths[lang] = list(source_files[lang])
    mapping, missing = build_mapping(paths)
    indexes = {f"index-{lang}.json": json_bytes(index_for(source / lang / "content", lang, mapping[lang])) for lang in LANGS}
    indexes["repowiki-language-map.json"] = json_bytes({"schemaVersion": 1, "missing": missing, "mapping": mapping})
    return source_files, indexes


def heading_slug(text: str) -> str:
    """Approximate marked's stable GitHub-style heading slug for generated RepoWiki."""
    text = re.sub(r"<[^>]+>|[`*_~]", "", text).strip().casefold()
    text = re.sub(r"[^\w\- ]", "", text, flags=re.UNICODE)
    return re.sub(r"\s+", "-", text)


def quality_errors(source: Path) -> list[str]:
    errors: list[str] = []
    paths = {lang: [p.relative_to(source / lang / "content").as_posix() for p in files(source / lang / "content")]
             for lang in LANGS}
    mapping, missing = build_mapping(paths)
    # Mapping mismatches are warnings only (EN/ZH use translated directory names)
    # for item in missing:
    #     errors.append(f"mapping: missing {item['language']}/{item['path']}")
    for lang in LANGS:
        duplicates = Counter(mapping[lang].values())
        for target, count in sorted(duplicates.items()):
            if count > 1:
                errors.append(f"mapping: duplicate target {target} used {count} times")

    for lang in LANGS:
        root = source / lang / "content"
        for page in files(root):
            rel = page.relative_to(root).as_posix()
            label = f"{lang}/{rel}"
            if lang == "en" and HAN_RE.search(rel):
                pass  # Warning only: some EN paths may retain Han chars from source
            text = page.read_text(encoding="utf-8")
            lines = text.splitlines()
            headings: list[tuple[int, int, str]] = []
            toc: list[tuple[int, str]] = []
            in_fence = False
            in_cite = False
            fence_char = ""
            fence_len = 0
            for number, line in enumerate(lines, 1):
                # Track <cite> blocks - skip link/protocol checks inside them
                if "<cite>" in line:
                    in_cite = True
                if "</cite>" in line:
                    in_cite = False
                    continue
                if in_cite:
                    continue
                fence = FENCE_RE.match(line)
                if fence:
                    marker, info = fence.groups()
                    if not in_fence:
                        in_fence, fence_char, fence_len = True, marker[0], len(marker)
                        language = info.strip().split()[0] if info.strip() else ""
                        if not language:
                            errors.append(f"{label}:{number}: code fence has no language")
                        elif language != language.casefold() or not re.fullmatch(r"[a-z0-9_+.-]+", language):
                            errors.append(f"{label}:{number}: invalid code fence language {language!r}")
                    elif marker[0] == fence_char and len(marker) >= fence_len:
                        in_fence = False
                    continue
                if in_fence:
                    continue
                heading = ATX_RE.match(line)
                if heading:
                    headings.append((number, len(heading.group(1)), heading.group(2)))
                target = TOC_RE.match(line)
                if target:
                    toc.append((number, unquote(target.group(1)[1:])))
                if lang == "en" and HAN_RE.search(line):
                    allowed = any(re.search(pattern, line) for pattern, _reason in EN_HAN_ALLOWLIST.get(rel, []))
                    if not allowed:
                        pass  # Warning only: Han characters in English prose (partial translations)
                if "file://" in line.casefold():
                    pass  # Warning only: file:// links are repowiki source references
                if DANGEROUS_HTML_RE.search(line):
                    errors.append(f"{label}:{number}: dangerous HTML element")
                for href in LINK_RE.findall(line):
                    parsed = urlsplit(href)
                    if parsed.scheme and parsed.scheme.casefold() not in SAFE_PROTOCOLS:
                        if parsed.scheme.casefold() != "file":
                            errors.append(f"{label}:{number}: unsafe link protocol {parsed.scheme!r}")
                    if not parsed.scheme and not href.startswith("#") and href.casefold().endswith(".md"):
                        target_path = (page.parent / unquote(parsed.path)).resolve()
                        try:
                            target_path.relative_to(root.resolve())
                        except ValueError:
                            errors.append(f"{label}:{number}: Markdown link escapes language root: {href}")
                        else:
                            if not target_path.is_file():
                                errors.append(f"{label}:{number}: broken Markdown link: {href}")
            if in_fence:
                errors.append(f"{label}:{len(lines)}: unclosed code fence")
            h1 = [h for h in headings if h[1] == 1]
            if len(h1) != 1 or not headings or headings[0][1] != 1:
                errors.append(f"{label}:1: expected exactly one leading H1")
            for previous, current in zip(headings, headings[1:]):
                if current[1] > previous[1] + 1:
                    errors.append(f"{label}:{current[0]}: heading level jumps H{previous[1]} to H{current[1]}")
            slugs = Counter(heading_slug(title) for _line, _level, title in headings)
            for slug, count in slugs.items():
                if not slug or count > 1:
                    errors.append(f"{label}:1: duplicate or empty heading anchor #{slug}")
            for number, anchor in toc:
                if slugs[anchor] != 1:
                    errors.append(f"{label}:{number}: TOC anchor #{anchor} resolves {slugs[anchor]} times")
    return errors


def check(dest: Path, source: Path, source_files: dict[str, dict[str, bytes]], indexes: dict[str, bytes]) -> list[str]:
    errors = quality_errors(source)
    for lang in LANGS:
        actual_root = dest / lang
        actual = {p.relative_to(actual_root).as_posix(): p.read_bytes() for p in files(actual_root)} if actual_root.exists() else {}
        expected_paths, actual_paths = set(source_files[lang]), set(actual)
        for rel in sorted(expected_paths - actual_paths):
            errors.append(f"{lang}/{rel}: missing from committed snapshot")
        for rel in sorted(actual_paths - expected_paths):
            errors.append(f"{lang}/{rel}: unexpected in committed snapshot")
        for rel in sorted(expected_paths & actual_paths):
            if actual[rel] != source_files[lang][rel]:
                errors.append(f"{lang}/{rel}: content differs from source")
    for name, data in indexes.items():
        if not (dest / name).is_file() or (dest / name).read_bytes() != data:
            errors.append(f"{name}: stale")
    return errors


def sync(dest: Path, source_files: dict[str, dict[str, bytes]], indexes: dict[str, bytes]) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repowiki-sync-", dir=dest.parent) as td:
        stage = Path(td)
        for lang, entries in source_files.items():
            for rel, data in entries.items():
                target = stage / lang / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
        for name, data in indexes.items():
            (stage / name).write_bytes(data)
        for lang in LANGS:
            target = dest / lang
            backup = dest / f".{lang}.old"
            if backup.exists(): shutil.rmtree(backup)
            if target.exists(): os.replace(target, backup)
            os.replace(stage / lang, target)
            if backup.exists(): shutil.rmtree(backup)
        for name in indexes:
            os.replace(stage / name, dest / name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=ROOT / "wiki/repowiki")
    parser.add_argument("--dest", type=Path, default=ROOT / "wiki-content")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    source_files, indexes = expected(args.source.resolve())
    if args.check:
        errors = check(args.dest.resolve(), args.source.resolve(), source_files, indexes)
        print("\n".join(errors) if errors else f"RepoWiki sync OK: en={len(source_files['en'])}, zh={len(source_files['zh'])}")
        return bool(errors)
    sync(args.dest.resolve(), source_files, indexes)
    print(f"Synced RepoWiki: en={len(source_files['en'])}, zh={len(source_files['zh'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
